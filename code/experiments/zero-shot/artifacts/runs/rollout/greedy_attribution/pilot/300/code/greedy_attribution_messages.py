"""Selected prediction-row attention messages for full-prefill Qwen3-VL.

For a head, receiver i and sender j this records
    A[i,j] * dot(d objective / d attention_output[i], value[j]).
The attention output is before o_proj, so head values and gradients share the
same coordinates. Only prediction receivers in selected decoder layers are
measured. In particular, visual-to-visual and other indirect edges are not
covered, and scores from different layers must not be summed as total evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch


def _rotate_half(tensor):
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class AttentionMessages:
    """Capture a few signed source edges per head and prediction receiver.

    The class leaves the model's original attention forward/backward intact.
    It reconstructs only the requested attention rows under no_grad and checks
    their weighted-value output against the actual input to o_proj. Padding
    masks and causal masking are supported. Sequential decoding requires a
    differentiable SnapshotDynamicCache with immutable prior chunks.
    """

    def __init__(self, model, predictor_positions: Iterable[int], layers=(0, 17, 35), top_edges=2):
        self.model = model
        self.predictor_positions = [int(pos) for pos in predictor_positions]
        self.layers = tuple(int(layer) for layer in layers)
        self.top_edges = int(top_edges)
        if not self.predictor_positions or min(self.predictor_positions) < 0:
            raise ValueError("Expected nonnegative predictor positions")
        if len(set(self.predictor_positions)) != len(self.predictor_positions):
            raise ValueError("Predictor positions must be unique")
        if not self.layers or len(set(self.layers)) != len(self.layers) or self.top_edges < 1:
            raise ValueError("Expected unique layer indices and at least one edge per sign")
        self._handles = []
        self._tensor_handles = []
        self._records = {}
        self._pending_records = {}
        self._reconstructions = {}
        self._segment_offset = 0
        self._target = None
        self._results = {}
        self._hook_calls = {}
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("AttentionMessages is already entered")
        self._entered = True
        try:
            decoder_layers = self.model.model.language_model.layers
            if min(self.layers) < 0 or max(self.layers) >= len(decoder_layers):
                raise ValueError("Selected decoder layer does not exist")
            for index in self.layers:
                attention = decoder_layers[index].self_attn
                self._handles.append(
                    attention.register_forward_pre_hook(self._attention_pre_hook(index), with_kwargs=True)
                )
                self._handles.append(attention.o_proj.register_forward_pre_hook(self._output_pre_hook(index)))
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        for handle in self._handles + self._tensor_handles:
            handle.remove()
        self._handles.clear()
        self._tensor_handles.clear()
        self._records.clear()
        self._pending_records.clear()
        self._entered = False

    def begin_target(self, name=None):
        self._target = "unnamed" if name is None else str(name)
        self._results = {}
        self._hook_calls = {}

    @property
    def segment_offset(self):
        return self._segment_offset

    def begin_forward(self):
        """Release prior forward records before starting a new sequence graph."""
        for handle in self._tensor_handles:
            handle.remove()
        self._tensor_handles.clear()
        self._records.clear()
        self._pending_records.clear()
        self._reconstructions.clear()
        self._results.clear()
        self._hook_calls.clear()
        self._target = None
        self._segment_offset = 0

    def set_segment(self, offset):
        self._segment_offset = int(offset)
        if self._segment_offset < 0:
            raise ValueError('Segment offset must be nonnegative')

    @contextmanager
    def segment(self, offset):
        previous = self._segment_offset
        self.set_segment(offset)
        try:
            yield self
        finally:
            self._segment_offset = previous

    def _attention_pre_hook(self, index):
        def before(attention, args, kwargs):
            if not torch.is_grad_enabled():
                return
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("AttentionMessages expects batch-one full-prefill hidden states")
            cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
            if cache is not None and not hasattr(cache, 'base_key'):
                raise ValueError('Sequential message tracing requires an immutable-base SnapshotDynamicCache')
            if attention.training and attention.attention_dropout:
                raise ValueError("Attention-message reconstruction requires zero attention dropout")
            cache_position = kwargs.get('cache_position')
            offset = int(cache_position.reshape(-1)[0]) if cache_position is not None else self._segment_offset
            receivers = [pos for pos in self.predictor_positions if offset <= pos < offset + hidden.shape[1]]
            self._pending_records[index] = None
            if not receivers:
                return
            cos, sin = kwargs["position_embeddings"]
            with torch.no_grad():
                positions = torch.tensor([pos-offset for pos in receivers], device=hidden.device, dtype=torch.long)
                selected_hidden = hidden.index_select(1, positions)
                head_dim = attention.head_dim
                query = attention.q_norm(attention.q_proj(selected_hidden).view(1, len(positions), -1, head_dim)).transpose(1, 2)
                key = attention.k_norm(attention.k_proj(hidden).view(1, hidden.shape[1], -1, head_dim)).transpose(1, 2)
                value = attention.v_proj(hidden).view(1, hidden.shape[1], -1, head_dim).transpose(1, 2)
                qcos, qsin = cos.index_select(1, positions).unsqueeze(1), sin.index_select(1, positions).unsqueeze(1)
                query = query * qcos + _rotate_half(query) * qsin
                key = key * cos.unsqueeze(1) + _rotate_half(key) * sin.unsqueeze(1)
                if cache is not None:
                    prior_key, prior_value = cache.base_key(index), cache.base_value(index)
                    if prior_key is not None:
                        key = torch.cat((prior_key, key), dim=-2)
                        value = torch.cat((prior_value, value), dim=-2)
                sequence_length = key.shape[-2]
                if sequence_length != offset + hidden.shape[1]:
                    raise ValueError('Cache length does not match global segment positions')
                kv_heads = key.shape[1]
                head_groups = query.shape[1] // kv_heads
                grouped_query = query.float().reshape(1, kv_heads, head_groups, len(positions), head_dim)
                # [batch, kv_head, query_heads_per_kv, receiver, sender]
                scores = torch.einsum("bkgpd,bksd->bkgps", grouped_query, key.float())
                scores = scores.reshape(1, query.shape[1], len(positions), sequence_length) * attention.scaling
                global_positions = positions + offset
                causal = torch.arange(sequence_length, device=hidden.device)[None, :] <= global_positions[:, None]
                scores.masked_fill_(~causal[None, None], float("-inf"))
                mask = kwargs.get("attention_mask")
                if mask is not None:
                    if mask.ndim == 2:
                        scores.masked_fill_(~mask[:, None, None, :].bool(), float("-inf"))
                    elif mask.ndim == 4:
                        selected_mask = mask.index_select(-2, positions)
                        if selected_mask.dtype == torch.bool:
                            scores.masked_fill_(~selected_mask, float("-inf"))
                        else:
                            scores = scores + selected_mask.float()
                    else:
                        raise ValueError(f"Unsupported attention mask shape {mask.shape}")
                weights = scores.softmax(dim=-1)
                grouped_weights = weights.reshape(1, kv_heads, head_groups, len(positions), sequence_length)
                reconstructed = torch.einsum("bkgps,bksd->bkgpd", grouped_weights, value.float())
                reconstructed = reconstructed.reshape(1, query.shape[1], len(positions), head_dim).transpose(1, 2).reshape(1, len(positions), -1)
                record = {
                    "positions": positions,
                    "receiver_positions": receivers,
                    "offset": offset,
                    "weights": weights[0],
                    "value": value[0].detach(),
                    "reconstructed": reconstructed.detach(),
                    "heads": query.shape[1],
                    "kv_heads": kv_heads,
                    "head_groups": head_groups,
                    "head_dim": head_dim,
                    "sequence_length": sequence_length,
                }
                self._records[index, offset] = record
                self._pending_records[index] = record
        return before

    def _output_pre_hook(self, index):
        def before(module, args):
            if not torch.is_grad_enabled():
                return
            tensor = args[0]
            record = self._pending_records[index]
            if record is None:
                return
            record_key = (index, record['offset'])
            positions = record["positions"]
            with torch.no_grad():
                actual = tensor.detach().index_select(1, positions).float()
                delta = record["reconstructed"] - actual
                rms = actual.square().mean().sqrt()
                self._reconstructions[record_key] = {
                    "layer": index,
                    "maximum_absolute_difference": float(delta.abs().max()),
                    "rms_difference": float(delta.square().mean().sqrt()),
                    "reference_rms": float(rms),
                    "rms_relative_difference": float(delta.square().mean().sqrt() / rms.clamp_min(1e-12)),
                    "sequence_length": record["sequence_length"],
                    "receiver_count": len(record['receiver_positions']),
                    "squared_difference_sum": float(delta.square().sum()),
                    "squared_reference_sum": float(actual.square().sum()),
                    "element_count": actual.numel(),
                }
                # Verification is retained; this large temporary is not needed
                # during any target backward.
                record.pop("reconstructed", None)
            if not tensor.requires_grad:
                return

            def backward(gradient):
                if self._target is None:
                    return
                selected = gradient.index_select(1, positions).float()
                selected = selected.reshape(1, len(positions), record["heads"], record["head_dim"]).transpose(1, 2)
                grouped = selected.reshape(1, record["kv_heads"], record["head_groups"], len(positions), record["head_dim"])
                direction = torch.einsum("bkgpd,ksd->bkgps", grouped, record["value"].float())
                direction = direction.reshape(record["heads"], len(positions), record["sequence_length"])
                scores = record["weights"] * direction
                edge_count = min(self.top_edges, scores.shape[-1])
                positive_scores, positive_indices = scores.topk(edge_count, dim=-1)
                negative_scores, negative_indices = scores.topk(edge_count, dim=-1, largest=False)
                positive_weights = record["weights"].gather(-1, positive_indices)
                negative_weights = record["weights"].gather(-1, negative_indices)
                # CPU synchronization happens once per compact tensor, never
                # once per individual edge.
                arrays = [item.detach().cpu().tolist() for item in (
                    positive_scores, positive_indices, positive_weights,
                    negative_scores, negative_indices, negative_weights,
                )]
                pos_scores, pos_indices, pos_weights, neg_scores, neg_indices, neg_weights = arrays
                edges = []
                for head in range(record["heads"]):
                    for receiver, position in enumerate(record['receiver_positions']):
                        for values, indices, weights, sign in (
                            (pos_scores, pos_indices, pos_weights, "positive"),
                            (neg_scores, neg_indices, neg_weights, "negative"),
                        ):
                            for edge in range(edge_count):
                                score = values[head][receiver][edge]
                                if (sign == "positive" and score <= 0) or (sign == "negative" and score >= 0):
                                    continue
                                edges.append({
                                    "head": head,
                                    "receiver_position": position,
                                    "source_position": indices[head][receiver][edge],
                                    "score": score,
                                    "attention_weight": weights[head][receiver][edge],
                                    "sign": sign,
                                })
                self._results[record_key] = {
                    "layer": index,
                    "receiver_positions": record['receiver_positions'],
                    "edges": edges,
                    "head_score_sum": scores.sum(dim=-1).detach().cpu().tolist(),
                    "head_positive_score_sum": scores.clamp_min(0).sum(dim=-1).detach().cpu().tolist(),
                    "head_negative_score_sum": scores.clamp_max(0).sum(dim=-1).detach().cpu().tolist(),
                    "heads": record["heads"],
                    "sequence_length": record["sequence_length"],
                }
                self._hook_calls[record_key] = self._hook_calls.get(record_key, 0) + 1
            self._tensor_handles.append(tensor.register_hook(backward))
        return before

    def collect(self):
        if self._target is None:
            raise RuntimeError("Call begin_target before backward and collect")
        import math
        layers, reconstruction = [], []
        receiver_columns = {pos: column for column, pos in enumerate(self.predictor_positions)}
        for index in self.layers:
            chunks = [record for (layer, _), record in sorted(self._results.items()) if layer == index]
            if chunks:
                heads = chunks[0]['heads']
                combined = dict(layer=index, edges=[], heads=heads, sequence_length=max(r['sequence_length'] for r in chunks))
                fields = ('head_score_sum', 'head_positive_score_sum', 'head_negative_score_sum')
                for field in fields:
                    combined[field] = [[0.0] * len(self.predictor_positions) for _ in range(heads)]
                for chunk in chunks:
                    combined['edges'].extend(chunk['edges'])
                    for local_column, position in enumerate(chunk['receiver_positions']):
                        column = receiver_columns[position]
                        for field in fields:
                            for head in range(heads):
                                combined[field][head][column] = chunk[field][head][local_column]
                layers.append(combined)
            checks = [record for (layer, _), record in self._reconstructions.items() if layer == index]
            if checks:
                squared_delta = math.fsum(r['squared_difference_sum'] for r in checks)
                squared_reference = math.fsum(r['squared_reference_sum'] for r in checks)
                count = sum(r['element_count'] for r in checks)
                reconstruction.append(dict(layer=index,
                    maximum_absolute_difference=max(r['maximum_absolute_difference'] for r in checks),
                    rms_difference=math.sqrt(squared_delta/count),reference_rms=math.sqrt(squared_reference/count),
                    rms_relative_difference=math.sqrt(squared_delta/max(squared_reference,1e-24)),
                    sequence_length=max(r['sequence_length'] for r in checks),
                    receiver_count=sum(r['receiver_count'] for r in checks),
                    maximum_segment_rms_relative_difference=max(r['rms_relative_difference'] for r in checks)))
        return {
            "target": self._target,
            "selected_layers": list(self.layers),
            "predictor_positions": list(self.predictor_positions),
            "top_edges_per_head_receiver_sign": self.top_edges,
            "layers": layers,
            "reconstruction": reconstruction,
            "hook_calls": {f'{index}:{offset}': count for (index,offset), count in self._hook_calls.items()},
            "definition": "A_ij * dot(gradient of objective w.r.t. pre-o_proj head output at i, value_j); signed derivative under scaling that message with A and value fixed at the original forward",
            "coverage": "selected decoder layers and prediction receivers only; visual-to-visual and other indirect receivers are not measured; only top positive/negative source edges are retained",
            "interpretation": "local message-scale derivative, not causal evidence or an additive cross-layer decomposition; reconstruction differences must be checked",
        }
