"""Original-forward gradient summaries for Qwen3-VL greedy attribution.

The caller supplies a complete teacher-forcing sequence, including the original
video prefill, and pixel_values_videos with requires_grad=True. No inference KV
cache may replace that prefill. Predictor positions are N + step - 1, not the
positions of the tokens being predicted.

This module reports local sensitivity and activation scaling derivatives. Its
attention summaries describe whole attention-block outputs at receiving
positions; they are not source-to-destination attention-edge attributions.
"""

from __future__ import annotations

import contextlib
import math
import weakref
from typing import Any, Iterable, Mapping

import torch


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def _position_stats(activation: torch.Tensor, gradient: torch.Tensor) -> dict:
    """Reduce a batch-one sequence without retaining its full gradient."""
    if activation.shape != gradient.shape or activation.ndim != 3 or activation.shape[0] != 1:
        raise ValueError(f"Expected matching [1, positions, hidden] tensors: {activation.shape}, {gradient.shape}")
    dots, norms = [], []
    # Avoid simultaneous float32 copies of a long multimodal sequence.
    for begin in range(0, activation.shape[1], 1024):
        a = activation[0, begin:begin + 1024].float()
        g = gradient[0, begin:begin + 1024].float()
        dots.append((a * g).sum(dim=-1))
        norms.append(g.square().sum(dim=-1).sqrt())
    if not dots:
        empty = gradient.new_empty((0,), dtype=torch.float32)
        return {"activation_dot_gradient": empty, "gradient_norm_l2": empty}
    return {
        "activation_dot_gradient": torch.cat(dots).detach(),
        "gradient_norm_l2": torch.cat(norms).detach(),
    }


class AttributionHooks:
    """Collect per-position and decoder-block summaries on each backward.

    Usage::

        with AttributionHooks(model, N, length, positions) as capture:
            output = model(..., use_cache=False, logits_to_keep=positions)
            capture.begin_target("greedy_log_probability")
            objective.backward(retain_graph=True)
            summary = capture.collect()

    ``embedding_leaves`` contains the pre-scatter text embedding tensors for
    every segment of the current forward. ``embedding_offsets`` gives their
    global sequence offsets. ``embedding_leaf`` and ``merged_input`` retain
    the most recent segment for full-batch compatibility. None is detached
    from the active forward; the caller can use them with autograd.grad.

    For sequential replay call ``begin_forward()`` once, then
    ``set_segment(offset)`` before each full model call. Prefill starts at
    offset zero; a token appended after N prompt positions starts at N.
    A checkpoint wrapper must run its recomputation within
    ``with capture.segment(saved_offset)``. MLP output hooks cannot infer
    an earlier segment's offset from the process-wide current offset.

    Parameters are frozen while the context is entered and their original
    requires_grad flags are restored on exit. This class does not change
    train/eval mode, attention implementation, checkpointing, or pixel tensors.
    """

    def __init__(
        self,
        model,
        prompt_length: int,
        total_length: int,
        predictor_positions: Iterable[int],
        source_groups: Mapping[str, Iterable[int]] | None = None,
    ):
        self.model = model
        self.prompt_length = int(prompt_length)
        self.total_length = int(total_length)
        self.predictor_positions = [int(pos) for pos in predictor_positions]
        if not 0 < self.prompt_length <= self.total_length:
            raise ValueError("Expected 0 < prompt_length <= total_length")
        if not self.predictor_positions or any(
            pos < self.prompt_length - 1 or pos >= self.total_length
            for pos in self.predictor_positions
        ):
            raise ValueError("Predictor positions must be within the answer-predicting sequence")
        if len(set(self.predictor_positions)) != len(self.predictor_positions):
            raise ValueError("Predictor positions must be unique")
        self.source_groups = {
            str(name): [int(pos) for pos in positions]
            for name, positions in (source_groups or {}).items()
        }
        for name, positions in self.source_groups.items():
            if len(set(positions)) != len(positions) or any(pos < 0 or pos >= self.total_length for pos in positions):
                raise ValueError(f"Invalid source positions for {name}")
        self.embedding_leaf = None
        self.embedding_leaves = []
        self.embedding_offsets = []
        self.merged_input = None
        self._segment_offset = 0
        self._module_handles = []
        self._tensor_handles = []
        self._captured_tensors = {}
        self._previous_parameter_flags = []
        self._entered = False
        self._scale_indices = []
        self._scale_factor = 1.0
        self._target = None
        self._summaries = {}
        self._hook_calls = {}

    def __enter__(self):
        if self._entered:
            raise RuntimeError("AttributionHooks is already entered")
        self._entered = True
        try:
            self._previous_parameter_flags = [(p, p.requires_grad) for p in self.model.parameters()]
            for parameter, _ in self._previous_parameter_flags:
                parameter.requires_grad_(False)
            language = self.model.model.language_model
            self._module_handles.append(
                self.model.get_input_embeddings().register_forward_hook(self._embedding_hook)
            )
            self._module_handles.append(
                language.register_forward_pre_hook(self._merged_hook, with_kwargs=True)
            )
            for index, layer in enumerate(language.layers):
                self._module_handles.append(
                    layer.self_attn.register_forward_hook(self._block_hook("attention", index, tuple_output=True))
                )
                self._module_handles.append(
                    layer.mlp.register_forward_hook(self._block_hook("mlp", index))
                )
                self._module_handles.append(
                    layer.register_forward_pre_hook(self._residual_hook(index), with_kwargs=True)
                )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        for handle in self._module_handles + self._tensor_handles:
            handle.remove()
        self._module_handles.clear()
        self._tensor_handles.clear()
        self._captured_tensors.clear()
        for parameter, flag in self._previous_parameter_flags:
            parameter.requires_grad_(flag)
        self._previous_parameter_flags.clear()
        self.embedding_leaf = None
        self.embedding_leaves.clear()
        self.embedding_offsets.clear()
        self.merged_input = None
        self._entered = False

    @property
    def segment_offset(self):
        return self._segment_offset

    def set_segment(self, offset: int):
        """Select the global sequence offset for the next model segment."""
        offset = int(offset)
        if offset < 0 or offset >= self.total_length:
            raise ValueError(f"Segment offset outside total sequence: {offset}")
        self._segment_offset = offset

    @contextlib.contextmanager
    def segment(self, offset: int):
        """Restore the right source coordinates during checkpoint replay."""
        previous = self._segment_offset
        self.set_segment(offset)
        try:
            yield self
        finally:
            self._segment_offset = previous

    def begin_forward(self):
        """Release the preceding graph's hooks and begin a whole new replay.

        Do not call this between segments or between objectives sharing a
        retained graph. Embedding scale settings are intentionally retained.
        """
        for handle in self._tensor_handles:
            handle.remove()
        self._tensor_handles.clear()
        self._captured_tensors.clear()
        self.embedding_leaf = None
        self.embedding_leaves.clear()
        self.embedding_offsets.clear()
        self.merged_input = None
        self._segment_offset = 0
        self._target = None
        self._summaries = {}
        self._hook_calls = {}

    def _bounds(self, tensor):
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"Expected batch-one [1, segment positions, hidden], got {tuple(tensor.shape)}")
        begin = self._segment_offset
        end = begin + int(tensor.shape[1])
        if tensor.shape[1] < 1 or end > self.total_length:
            raise ValueError(f"Segment [{begin}, {end}) outside total sequence {self.total_length}")
        return begin, end

    def set_embedding_scale(self, indices: Iterable[int] | str, factor: float):
        """Set one source-scale direction for subsequent forwards.

        This is an internal embedding intervention, not a natural-language edit.
        The video placeholder embeddings will still be replaced by the original
        visual embeddings; scale pixel_values_videos separately when intended.
        """
        positions = self.source_groups[indices] if isinstance(indices, str) else list(indices)
        positions = [int(pos) for pos in positions]
        if not math.isfinite(factor) or len(set(positions)) != len(positions):
            raise ValueError("Expected a finite scale and unique positions")
        if any(pos < 0 or pos >= self.total_length for pos in positions):
            raise ValueError("Embedding scale position outside total_length")
        self._scale_indices, self._scale_factor = positions, float(factor)

    def clear_embedding_scale(self):
        self._scale_indices, self._scale_factor = [], 1.0

    def begin_target(self, name: str | None = None):
        """Clear summaries before backward on the current forward graph."""
        self._target = "unnamed" if name is None else str(name)
        self._summaries = {}
        self._hook_calls = {}

    def _capture(self, tensor: torch.Tensor, key: tuple, selected: bool):
        if not torch.is_grad_enabled() or not tensor.requires_grad:
            return
        begin, end = self._bounds(tensor)
        identity = (key, begin, end, id(tensor))
        previous = self._captured_tensors.get(identity)
        if previous is not None and previous() is tensor:
            # Non-reentrant checkpointing invokes a layer prehook again on the
            # same input tensor during recomputation. Its original hook already
            # receives the complete gradient; registering it twice is redundant.
            return
        self._captured_tensors[identity] = weakref.ref(tensor)
        if selected:
            selected_pairs = [(i, pos-begin) for i, pos in enumerate(self.predictor_positions) if begin <= pos < end]
            if not selected_pairs:
                return
            destinations = torch.tensor([i for i, _ in selected_pairs], device=tensor.device, dtype=torch.long)
            positions = torch.tensor([pos for _, pos in selected_pairs], device=tensor.device, dtype=torch.long)
            size = len(self.predictor_positions)
            # index_select copies the tiny prediction-position snapshot. This
            # also avoids later DeepStack in-place updates changing residuals.
            activation = tensor.detach().index_select(1, positions)
        else:
            positions = None
            activation = tensor.detach()
            destinations = torch.arange(begin, end, device=tensor.device, dtype=torch.long)
            size = self.total_length

        def backward(gradient):
            if self._target is None:
                return
            grad = gradient if positions is None else gradient.index_select(1, positions)
            local = _position_stats(activation, grad)
            if key not in self._summaries:
                self._summaries[key] = {name: value.new_zeros((size,)) for name, value in local.items()}
            # Sequence segments have disjoint global positions. An identical
            # checkpoint recomputation must replace its own slice, not add
            # duplicate norms or duplicate activation attribution.
            for name, values in local.items():
                self._summaries[key][name].index_copy_(0, destinations, values)
            self._hook_calls[key] = self._hook_calls.get(key, 0) + 1
            # Returning None leaves the propagated gradient unchanged.

        self._tensor_handles.append(tensor.register_hook(backward))

    def _embedding_hook(self, module, args, output):
        begin, end = self._bounds(output)
        if torch.is_grad_enabled():
            output.requires_grad_(True)
            self.embedding_leaf = output
            if not any(tensor is output for tensor in self.embedding_leaves):
                self.embedding_leaves.append(output)
                self.embedding_offsets.append(begin)
            self._capture(output, ("embedding",), selected=False)
        if self._scale_indices and self._scale_factor != 1.0:
            local = [pos-begin for pos in self._scale_indices if begin <= pos < end]
            if not local:
                return output
            scale = output.new_ones((1, output.shape[1], 1))
            scale[:, local] = self._scale_factor
            return output * scale
        return output

    def _merged_hook(self, module, args, kwargs):
        tensor = kwargs.get("inputs_embeds")
        if tensor is None:
            raise ValueError("Expected merged inputs_embeds at language_model entry")
        if torch.is_grad_enabled():
            self.merged_input = tensor
            self._capture(tensor, ("context",), selected=False)

    def _block_hook(self, kind: str, layer: int, tuple_output: bool = False):
        def forward(module, args, output):
            tensor = output[0] if tuple_output else output
            self._capture(tensor, (kind, layer), selected=True)
        return forward

    def _residual_hook(self, layer: int):
        def forward(module, args, kwargs):
            tensor = args[0] if args else kwargs.get("hidden_states")
            if tensor is None:
                raise ValueError("Missing decoder hidden_states")
            self._capture(tensor, ("residual_input", layer), selected=True)
        return forward

    def collect(self) -> dict:
        """Return CPU-only summaries. An absent key means unmeasured, not zero."""
        if self._target is None:
            raise RuntimeError("Call begin_target before backward and collect")
        result = {
            "target": self._target,
            "prompt_length": self.prompt_length,
            "total_length": self.total_length,
            "predictor_positions": list(self.predictor_positions),
            "layers": {kind: [] for kind in ("attention", "mlp", "residual_input")},
            "hook_calls": {":".join(map(str, key)): count for key, count in self._hook_calls.items()},
            "definitions": {
                "activation_dot_gradient": "signed derivative under a multiplicative scale of this activation at the original forward",
                "gradient_norm_l2": "L2 norm over hidden coordinates at each position; representation units are not cross-modality calibrated",
                "attention": "whole attention-block output at the receiving predictor position, not a per-source edge",
                "residual_input": "decoder-layer input at the predictor position; overlapping representations must not be summed across layers",
                "context": "language-model input after visual replacement, before DeepStack injections; pixel gradients additionally include DeepStack paths",
            },
        }
        for key, stats in self._summaries.items():
            if len(key) == 1:
                result[key[0]] = _cpu(stats)
            else:
                result["layers"][key[0]].append({"layer": key[1], **_cpu(stats)})
        for rows in result["layers"].values():
            rows.sort(key=lambda row: row["layer"])
        if "context" in result and self.source_groups:
            norms = result["context"]["gradient_norm_l2"]
            dots = result["context"]["activation_dot_gradient"]
            result["context_groups"] = {
                name: {
                    "positions": list(positions),
                    "position_count": len(positions),
                    "activation_dot_gradient": math.fsum(dots[pos] for pos in positions),
                    "gradient_norm_l2": math.sqrt(math.fsum(norms[pos] ** 2 for pos in positions)),
                }
                for name, positions in self.source_groups.items()
            }
        return result
