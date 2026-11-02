"""Generate actual answers and capture query/answer attention from the live KV-cache run."""
from __future__ import annotations
from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
import argparse
import hashlib
import json
from pathlib import Path
import time

from input_pipeline import ROOT, encode_sample, write_json, torch, np, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLVideoAdapter

PREVIOUS = ROOT


class GenerationCapture:
    def __init__(self, model):
        import transformers.models.qwen3_vl.modeling_qwen3_vl as module
        self.module = module
        self.original = module.eager_attention_forward
        self.layers = {id(x.self_attn): i for i, x in enumerate(model.model.language_model.layers)}
        module.eager_attention_forward = self.forward

    def configure(self, meta, device):
        self.meta = meta
        self.prompt_length = meta["sequence_length"]
        self.qpos = torch.tensor(meta["query_positions"], device=device)
        self.vpos = torch.tensor(meta["video_positions"], device=device)
        self.query = [None] * 36
        self.query_mass = [None] * 36
        self.steps = [[] for _ in range(36)]
        self.step_mass = [[] for _ in range(36)]
        self.positions = [[] for _ in range(36)]
        self.row_errors = []

    def forward(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if id(module) not in self.layers:
            return self.original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        layer = self.layers[id(module)]
        qlen, klen = query.shape[-2], key.shape[-2]
        prefill = qlen == self.prompt_length and klen == self.prompt_length
        assert prefill or qlen == 1
        if prefill:
            positions = torch.cat([self.qpos, self.qpos.new_tensor([qlen - 1])])
        else:
            positions = self.qpos.new_tensor([0])
        if not prefill or qlen <= 5500:
            out, weights = self.original(module, query, key, value, attention_mask, scaling, dropout=0.0)
            selected = weights[0].index_select(1, positions).float()
            del weights
        else:
            out = torch.empty((1, qlen, query.shape[1], query.shape[-1]), device=query.device, dtype=query.dtype)
            selected = torch.empty((query.shape[1], len(positions), klen), device=query.device, dtype=torch.float32)
            groups = module.num_key_value_groups
            assert groups == 4
            for start in range(0, query.shape[1], 4):
                end = start + 4
                block, weights = self.original(module, query[:, start:end], key[:, start//groups:end//groups],
                    value[:, start//groups:end//groups], attention_mask, scaling, dropout=0.0)
                out[:, :, start:end] = block
                selected[start:end] = weights[0].index_select(1, positions).float()
                del block, weights
        video = selected.index_select(2, self.vpos)
        t, hw = self.meta["video_grid_thw"][0], self.meta["spatial_tokens_per_bin"]
        temporal = video.reshape(32, len(positions), t, hw).sum(-1)
        video_mass = temporal.sum(-1)
        query_mass = selected.index_select(2, self.qpos).sum(-1)
        answer_mass = selected[:, :, self.prompt_length:].sum(-1)
        total = selected.sum(-1)
        mass = torch.stack([video_mass, query_mass, answer_mass, total-video_mass-query_mass-answer_mass], -1)
        self.row_errors.append((total-1).abs().max())
        if prefill:
            # Causal query rows must never see the assistant prefix after the query.
            mask = torch.arange(klen, device=query.device)[None, :] > self.qpos[:, None]
            assert not torch.any(selected[:, :-1].masked_select(mask[None]))
            self.query[layer] = temporal[:, :-1].detach()
            self.query_mass[layer] = mass[:, :-1].detach()
            temporal, mass = temporal[:, -1:], mass[:, -1:]
        self.steps[layer].append(temporal.detach())
        self.step_mass[layer].append(mass.detach())
        self.positions[layer].append(klen - 1)
        return out, None

    def arrays(self, response_length):
        assert all(x == self.positions[0] for x in self.positions)
        pos = self.positions[0]
        assert pos == list(range(self.prompt_length-1, self.prompt_length+response_length))
        steps = torch.stack([torch.cat(layer, dim=1) for layer in self.steps])
        mass = torch.stack([torch.cat(layer, dim=1) for layer in self.step_mass])
        # Row at P+j consumes response token j; row at P+j-1 predicts it.
        arrays = dict(query_temporal=torch.stack(self.query), query_key_mass=torch.stack(self.query_mass),
                      answer_temporal=steps[:, :, 1:], answer_key_mass=mass[:, :, 1:],
                      predictive_temporal=steps[:, :, :-1], predictive_key_mass=mass[:, :, :-1])
        max_error = float(torch.stack(self.row_errors).max())
        assert max_error < 0.0041, max_error
        arrays = {key: value.cpu().numpy() for key, value in arrays.items()}
        for key, value in arrays.items():
            assert np.isfinite(value).all(), key
            if key.endswith("temporal"):
                assert (value >= 0).all(), key
        for scope in ["query", "answer", "predictive"]:
            assert np.allclose(arrays[f"{scope}_temporal"].sum(-1), arrays[f"{scope}_key_mass"][..., 0], atol=2e-6)
        return arrays, max_error

    def restore(self):
        self.module.eager_attention_forward = self.original


def parse_answer(ids, tokenizer):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    prefixes = [tokenizer.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
    assert all(text.startswith(p) for p in prefixes), "Non-prefix UTF-8 token decoding requires explicit alignment"
    offsets = [(len(prefixes[i]), len(prefixes[i+1])) for i in range(len(ids))]
    adapter = Qwen3VLVideoAdapter()
    spans, span_offsets = adapter._parse_spans_with_offsets(text)
    per_span = []
    for span, (s, e) in zip(spans, span_offsets):
        indices = [i for i, (a, b) in enumerate(offsets) if a < e and b > s and any(c.isdigit() for c in text[a:b])]
        per_span.append(dict(span=span, token_indices=indices, first_token_index=indices[0] if indices else None))
    numeric = sorted({i for row in per_span for i in row["token_indices"]})
    return dict(text=text, token_ids=ids, tokens=[tokenizer.decode([i]) for i in ids], token_char_offsets=offsets,
                parsed_spans=spans, per_span_tokens=per_span, timestamp_token_indices=numeric,
                first_coordinate_indices=[row["first_token_index"] for row in per_span if row["first_token_index"] is not None])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20260929)
    results = GENERATION
    samples = json.loads((relocate_path(results / "samples.json")).read_text())
    assert samples == json.loads((relocate_path(QUERY / 'samples.json')).read_text())
    selected = [s for s in samples[:args.limit] if not args.only or s['sample_key'] in args.only]
    if all((relocate_path(results/s['dataset']/s['sample_key']/'validation.json')).exists() for s in selected):
        print('All selected generation samples complete; no model load required', flush=True)
        return
    write_json(relocate_path(results / "run_config.json"), dict(checkpoint="Qwen/Qwen3-VL-8B-Instruct",
        checkpoint_revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", dtype="bfloat16", layer_band=[8,21],
        generation=dict(do_sample=False, num_beams=1, max_new_tokens=128, repetition_penalty=1.0, use_cache=True),
        engine="Hugging Face eager (head chunks of 4 for prefill >5500 tokens), actual live generation",
        source_sample_manifest=str(relocate_path(QUERY / 'samples.json')),
        query_rows="input event query positions, captured during generation prefill",
        answer_rows="positions consuming generated answer tokens, matching MTLA response-token convention",
        predictive_rows="one position before each answer token, the forward that selected that token",
        answer_eos_excluded=True, answers_per_sample=1, gt_never_input=True))
    proc = AutoProcessor.from_pretrained(relocate_path(ROOT / "model"), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(relocate_path(ROOT / "model"), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0").eval()
    model.generation_config.top_k = 50  # neutral HF default; unused by greedy decoding
    capture = GenerationCapture(model)
    eos = set(model.generation_config.eos_token_id)
    try:
        for sample in samples[:args.limit]:
            key = sample["sample_key"]
            if args.only and key not in args.only:
                continue
            folder = relocate_path(results / sample["dataset"] / key)
            folder.mkdir(parents=True, exist_ok=True)
            if (relocate_path(folder / "validation.json")).exists():
                print("Already complete", key, flush=True)
                continue
            start = time.time()
            inputs, meta, text = encode_sample(sample, proc)
            assert text == (relocate_path(QUERY / sample["dataset"] / key / "expanded_input.txt")).read_text()
            previous_meta = json.loads((relocate_path(QUERY / sample["dataset"] / key / "metadata.json")).read_text())
            for field in ["query_positions", "video_positions", "sampled_timestamps", "video_grid_thw"]:
                assert meta[field] == previous_meta[field], field
            write_json(relocate_path(folder / "metadata.json"), meta)
            write_result_text((relocate_path(folder / "expanded_input.txt")), text)
            inputs = inputs.to("cuda:0")
            capture.configure(meta, "cuda:0")
            torch.cuda.reset_peak_memory_stats()
            print("Generating", key, "prompt tokens", meta["sequence_length"], flush=True)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=128,
                    temperature=1.0, top_p=1.0, repetition_penalty=1.0,
                    use_cache=True, return_dict_in_generate=True, output_scores=False, logits_to_keep=1)
            p = meta["sequence_length"]
            generated = output.sequences[0, p:].tolist()
            stopped_eos = bool(generated and generated[-1] in eos)
            answer_ids = generated[:-1] if stopped_eos else generated
            assert not any(i in eos for i in answer_ids)
            # At a length cap the final emitted token has not yet been consumed.
            # Flush it through the existing cache without adding it to the answer.
            if not stopped_eos and answer_ids:
                cache_position = p + len(answer_ids) - 1
                with torch.inference_mode():
                    flushed = model(input_ids=output.sequences[:, -1:],
                        past_key_values=output.past_key_values,
                        attention_mask=torch.ones(1, cache_position+1, device="cuda:0", dtype=torch.long),
                        cache_position=torch.tensor([cache_position], device="cuda:0"),
                        use_cache=True, logits_to_keep=1)
                del flushed
            answer = parse_answer(answer_ids, proc.tokenizer)
            answer.update(stopped_on_eos=stopped_eos, truncated=not stopped_eos,
                          generated_ids_including_eos=generated, generated_tokens_count=len(generated),
                          capture_positions=capture.positions[0])
            write_json(relocate_path(folder / "answer.json"), answer)
            arrays, row_error = capture.arrays(len(answer_ids))
            with np.load(relocate_path(QUERY / sample["dataset"] / key / "attention.npz")) as old:
                old_query = old["temporal_attention"]
            delta = np.abs(arrays["query_temporal"] - old_query)
            assert np.array_equal(arrays["query_temporal"], old_query), f"Changed query attention: {key} max diff {delta.max()}"
            np.savez_compressed(relocate_path(folder / "attention.npz"), **arrays,
                sampled_timestamps=np.array(meta["sampled_timestamps"]),
                mtla_uniform_bin_edges=np.array(meta["mtla_uniform_bin_edges"]))
            write_json(relocate_path(folder / "validation.json"), dict(query_matches_previous_bitwise=True,
                query_max_absolute_difference=float(delta.max()), attention_row_sum_max_error=row_error,
                answer_tokens=len(answer_ids), timestamp_tokens=len(answer["timestamp_token_indices"]),
                parsed_spans=len(answer["parsed_spans"]), capture_layers=36, capture_steps=len(capture.positions[0]),
                key_mass_order=["video", "input_query", "generated_answer", "other_prefix"],
                elapsed_seconds=time.time()-start, peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            del output, inputs, arrays
            capture.query = capture.query_mass = capture.steps = capture.step_mass = capture.row_errors = None
            torch.cuda.empty_cache()
            print("Saved", key, repr(answer["text"]), "seconds", round(time.time()-start,1), flush=True)
    finally:
        capture.restore()


if __name__ == "__main__":
    main()
