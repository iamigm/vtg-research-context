"""MTLA-backbone input-query -> preceding video attention, without GT in inputs."""
from __future__ import annotations
from result_layout import relocate_path, write_result_text, result_glob
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

sys.path.insert(0, str(relocate_path(ROOT / "MTLA")))
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from mtla.config import load_config
from mtla.data.charades import CharadesDataset
from mtla.data.qvhighlights import QVHighlightsDataset
from mtla.models.qwen3vl import Qwen3VLVideoAdapter


def write_json(path, obj):
    relocate_path(write_result_text(Path(path)), json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def select_samples(data_root, seed, count):
    """GT-independent shuffled queries, distinct videos, no duration/GT filters."""
    samples = []
    for name, split, config in [
        ("Charades-STA", "test", "charades"),
        ("QVHighlights", "val", "qvhighlights"),
        ("ActivityNet-Captions", "val_1", "qvhighlights"),
    ]:
        ann = relocate_path(data_root / name / "annotations_with_paths" / f"{split}.jsonl")
        ann_hash = hashlib.sha256(ann.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in ann.read_text().splitlines()]
        random.Random(seed).shuffle(rows)
        cfg = load_config(str(relocate_path(ROOT / "MTLA/configs" / f"{config}_qwen3vl.yaml")))
        used = set()
        for row in rows:
            if row["vid"] in used or not row.get("video_path"):
                continue
            video = data_root / name / row["video_path"]
            if not video.is_file() or not row.get("relevant_windows"):
                continue
            if name == "Charades-STA":
                query = row["query"].rstrip(".")
                prompt = CharadesDataset().prompt({"query": row["query"]})
            elif name == "QVHighlights":
                query = row["query"]
                prompt = QVHighlightsDataset().prompt(row)
            else:
                # Official sentence-level single-window task, Charades task wording;
                # longer-video QVHighlights FPS/pixel budget.
                query = row["query"].rstrip(".")
                prompt = CharadesDataset().prompt({"query": row["query"]})
            used.add(row["vid"])
            samples.append(dict(
                dataset=name, split=split, qid=str(row["qid"]), vid=row["vid"],
                query=query, original_query=row["query"], prompt=prompt,
                video_path=str(video), duration_annotation=row["duration"],
                gt_windows=row["relevant_windows"], preprocess=cfg.preprocess,
                annotation_file=str(ann), annotation_sha256=ann_hash,
                sample_key=f"{name}_{len(used):02d}",
            ))
            if len(used) == count:
                break
        assert len(used) == count
    return samples


def encode_sample(sample, proc):
    adapter = Qwen3VLVideoAdapter()
    messages = adapter._video_message(sample["video_path"], sample["prompt"], sample["preprocess"])
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    vids, metas = zip(*videos)
    # Preserve decoder metadata before the processor wraps/updates it.
    metadata = dict(metas[0])
    metadata["frames_indices"] = np.asarray(metadata["frames_indices"]).tolist()
    inputs = proc(text=text, images=images, videos=list(vids), video_metadata=list(metas),
                  do_resize=False, return_tensors="pt", **kwargs)
    ids = inputs["input_ids"][0].tolist()
    expanded = proc.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    enc = proc.tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == ids, "Decoded expanded input must round-trip exactly"
    query_start = expanded.rindex(sample["prompt"]) + len(sample["prompt"]) - len(sample["query"])
    query_end = query_start + len(sample["query"])
    assert expanded[query_start:query_end] == sample["query"]
    qpos = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < query_end and e > query_start]
    vpos = [i for i, tok in enumerate(ids) if tok == proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")]
    t, h, w = inputs["video_grid_thw"][0].tolist()
    hw = h * w // proc.video_processor.merge_size ** 2
    assert len(vpos) == t * hw and max(vpos) < min(qpos)
    timestamps = proc._calculate_timestamps(list(metadata["frames_indices"]), metadata["fps"], proc.video_processor.merge_size)
    assert len(timestamps) == t
    import av
    with av.open(sample["video_path"]) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration / av.time_base)
    assert all(0 <= s < e for s, e in sample["gt_windows"]), "Invalid GT interval"
    info = dict(
        sample, duration_video=duration, decoder_metadata=metadata,
        gt_outside_video=any(e > duration for s, e in sample["gt_windows"]),
        video_grid_thw=[t, h, w], spatial_tokens_per_bin=hw,
        sequence_length=len(ids), query_positions=qpos, video_positions=vpos,
        query_tokens=[proc.tokenizer.decode([ids[i]]) for i in qpos],
        sampled_timestamps=timestamps,
        mtla_uniform_bin_edges=np.linspace(0, duration, t + 1).tolist(),
        input_layout="native chat template: user [video timestamps+patches, MTLA instruction, Event: query], assistant prefix",
        literal_prefill_query_tags=False,
    )
    return inputs, info, expanded


class QueryCapture:
    """Unmodified stock eager forward, selecting query rows only after softmax.

    Queries/keys are the actual post-RoPE tensors passed by each decoder layer.
    Softmax includes *all* causally visible keys before selecting video columns.
    No approximation, top-k restriction, or video-only softmax is used in capture.
    """
    def __init__(self, model, full_eager_limit=5500, chunk_size=4):
        import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
        self.modeling = modeling
        self.original = modeling.eager_attention_forward
        self.full_eager_limit = full_eager_limit
        self.chunk_size = chunk_size
        self.layers = {id(layer.self_attn): i for i, layer in enumerate(model.model.language_model.layers)}
        modeling.eager_attention_forward = self.forward

    def configure(self, info, device):
        self.info = info
        self.qpos = torch.tensor(info["query_positions"], device=device)
        self.vpos = torch.tensor(info["video_positions"], device=device)
        self.temporal = [None] * len(self.layers)
        self.mass = [None] * len(self.layers)
        self.validation = []

    def forward(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if id(module) not in self.layers:
            return self.original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        assert query.shape[0] == 1 and query.shape[-2] == key.shape[-2]
        assert not module.training and dropout == 0
        n = key.shape[-2]
        # Absolute positions are required: a rectangular triangular mask would be wrong.
        causal = torch.arange(n, device=query.device)[None, :] <= self.qpos[:, None]
        mask = torch.zeros((1, 1, len(self.qpos), n), device=query.device, dtype=query.dtype)
        mask.masked_fill_(~causal[None, None], torch.finfo(query.dtype).min)
        if attention_mask is not None:
            actual = attention_mask.index_select(2, self.qpos)[:, :, :, :n]
            assert torch.equal(actual, mask), "Only unpadded pure-causal prompts supported"
        if n <= self.full_eager_limit:
            out, full_weights = self.original(module, query, key, value, attention_mask, scaling, dropout=0.0)
            weights = full_weights[0].index_select(1, self.qpos).float()
            del full_weights
            backend = "stock HF eager"
        else:
            # Split independent attention heads, retaining the complete Q x K
            # matrix for each head and the original GQA groups.
            assert attention_mask is not None
            out = torch.empty((1, n, query.shape[1], query.shape[-1]), device=query.device, dtype=query.dtype)
            weights = torch.empty((query.shape[1], len(self.qpos), n), device=query.device, dtype=torch.float32)
            groups = module.num_key_value_groups
            assert self.chunk_size % groups == 0
            for start in range(0, query.shape[1], self.chunk_size):
                end = min(start + self.chunk_size, query.shape[1])
                block, block_weights = self.original(module, query[:, start:end],
                    key[:, start // groups:end // groups], value[:, start // groups:end // groups],
                    attention_mask, scaling, dropout=0.0)
                out[:, :, start:end] = block
                weights[start:end] = block_weights[0].index_select(1, self.qpos).float()
                del block, block_weights
            backend = f"stock HF eager in head chunks of {self.chunk_size}"
        video = weights.index_select(2, self.vpos)
        t = self.info["video_grid_thw"][0]
        hw = self.info["spatial_tokens_per_bin"]
        layer = self.layers[id(module)]
        self.temporal[layer] = video.reshape(video.shape[0], len(self.qpos), t, hw).sum(-1).cpu().numpy()
        prefix = weights[:, :, :int(self.qpos[0])].sum(-1)
        video_mass = video.sum(-1)
        query_mass = weights.index_select(2, self.qpos).sum(-1)
        total = weights.sum(-1)
        self.mass[layer] = torch.stack([video_mass, prefix - video_mass, query_mass, total - prefix - query_mass], -1).cpu().numpy()
        # Stock eager casts FP32 softmax to BF16; allow its < 2^-8 relative rounding.
        assert float((total - 1).abs().max()) < 0.0041, f"Layer {layer}: invalid row sums {float(total.min())}..{float(total.max())}"
        assert float(weights.masked_select(~causal[None]).abs().max()) == 0
        self.validation.append(dict(layer=layer, backend=backend,
                                    attention_sum_max_error=float((total - 1).abs().max())))
        return out, None

    def restore(self):
        self.modeling.eager_attention_forward = self.original


def check_native_capture(device):
    """Independent full eager matrix vs selected absolute rows, including GQA/mask."""
    from types import SimpleNamespace
    from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
    torch.manual_seed(1701)
    module = SimpleNamespace(num_key_value_groups=4, training=False, is_causal=True)
    q = torch.randn(1, 8, 31, 32, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 31, 32, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    pos = torch.tensor([0, 8, 27, 30], device=device)
    mask = torch.full((1, 1, 31, 31), torch.finfo(q.dtype).min, device=device, dtype=q.dtype).triu(1)
    full, weights = eager_attention_forward(module, q, k, v, mask, 32 ** -0.5)
    sub, sw = eager_attention_forward(module, q.index_select(2, pos), k, v, mask.index_select(2, pos), 32 ** -0.5)
    assert torch.equal(sw, weights.index_select(2, pos))
    assert torch.equal(sub, full.index_select(1, pos))
    return dict(full_eager_vs_selected_rows="bitwise identical", absolute_causal_mask="passed", grouped_query_attention="passed")


def save_csv(folder, info, temporal):
    band = temporal[8:22].mean(axis=(0, 1, 2))
    all_layers = temporal.mean(axis=(0, 1, 2))
    norm = band / band.sum()
    edges = info["mtla_uniform_bin_edges"]
    with (relocate_path(folder / "temporal_attention.csv")).open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["bin", "sampled_time_seconds", "mtla_bin_start", "mtla_bin_end", "attention_L8_21", "attention_all_layers", "video_conditional_L8_21"])
        for i, ts in enumerate(info["sampled_timestamps"]):
            writer.writerow([i, ts, edges[i], edges[i + 1], band[i], all_layers[i], norm[i]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=relocate_path(Path("/data/video/MR")))
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--full-eager-limit", type=int, default=5500)
    parser.add_argument("--chunk-size", type=int, default=4, help="Heads per chunk, divisible by the GQA group size")
    parser.add_argument("--only", nargs="*", help="Run specified sample keys, keeping the full manifest")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    results = QUERY
    results.mkdir(exist_ok=True)
    samples = select_samples(args.data_root, args.seed, args.samples_per_dataset)
    if (relocate_path(results / "samples.json")).exists():
        previous = json.loads((relocate_path(results / "samples.json")).read_text())
        if previous != samples and list(result_glob(results, "*/*/attention.npz")):
            raise RuntimeError("Existing results use different samples. Preserve them and use a new experiment directory.")
    selected = [s for s in samples[:args.limit] if not args.only or s['sample_key'] in args.only]
    if not args.prepare_only and all((relocate_path(results/s['dataset']/s['sample_key']/'validation.json')).exists() for s in selected):
        print('All selected query samples complete; no model load required', flush=True)
        return
    write_json(relocate_path(results / "samples.json"), samples)
    write_json(relocate_path(results / "run_config.json"), dict(seed=args.seed, samples_per_dataset=args.samples_per_dataset,
               checkpoint="Qwen/Qwen3-VL-8B-Instruct", dtype="bfloat16", layer_band=[8, 21],
               checkpoint_revision=(relocate_path(ROOT / "model/.cache/huggingface/download/config.json.metadata")).read_text().splitlines()[0],
               selection="fixed-seed shuffled queries; distinct videos; no GT-position/duration/attention filter",
               attention="mean over event-query tokens, all heads, layers 8..21; sum spatial video tokens",
               torch=torch.__version__, transformers=transformers.__version__, attention_backend="eager (vision and text)",
               full_eager_limit=args.full_eager_limit, chunk_size=args.chunk_size,
               query_definition="input event sentence only; no response tokens or GT supplied",
               source_revision=json.loads((relocate_path(ROOT / "source_provenance.json")).read_text())["git_commit"]))
    proc = AutoProcessor.from_pretrained(relocate_path(ROOT / "model"), local_files_only=True)
    if args.prepare_only:
        for sample in samples:
            inputs, info, text = encode_sample(sample, proc)
            folder = relocate_path(results / sample["dataset"] / sample["sample_key"])
            folder.mkdir(parents=True, exist_ok=True)
            write_json(relocate_path(folder / "metadata.json"), info)
            write_result_text((relocate_path(folder / "expanded_input.txt")), text)
            print(sample["sample_key"], info["sequence_length"], sample["query"], flush=True)
        return
    device = "cuda:0"
    write_json(relocate_path(results / "capture_unit_validation.json"), check_native_capture(device))
    print("Loading Qwen3-VL-8B-Instruct", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        relocate_path(ROOT / "model"), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map=device).eval()
    capture = QueryCapture(model, args.full_eager_limit, args.chunk_size)
    try:
        for sample in samples[:args.limit]:
            if args.only and sample["sample_key"] not in args.only:
                continue
            folder = relocate_path(results / sample["dataset"] / sample["sample_key"])
            folder.mkdir(parents=True, exist_ok=True)
            if (relocate_path(folder / "attention.npz")).exists():
                print("Already complete:", sample["sample_key"], flush=True)
                continue
            start = time.time()
            inputs, info, text = encode_sample(sample, proc)
            write_json(relocate_path(folder / "metadata.json"), info)
            write_result_text((relocate_path(folder / "expanded_input.txt")), text)
            print("Running", sample["sample_key"], "sequence", info["sequence_length"], "query tokens", len(info["query_positions"]), flush=True)
            inputs = inputs.to(device)
            assert inputs["attention_mask"].all()
            capture.configure(info, device)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                output = model.model(**inputs, use_cache=False, return_dict=True)
            del output
            temporal = np.stack(capture.temporal)
            mass = np.stack(capture.mass)
            assert temporal.shape[:3] == (36, 32, len(info["query_positions"]))
            assert np.isfinite(temporal).all() and (temporal >= 0).all()
            assert np.allclose(temporal.sum(-1), mass[..., 0], atol=2e-6)
            np.savez_compressed(relocate_path(folder / "attention.npz"), temporal_attention=temporal, key_group_mass=mass,
                                sampled_timestamps=np.array(info["sampled_timestamps"]),
                                mtla_uniform_bin_edges=np.array(info["mtla_uniform_bin_edges"]))
            save_csv(folder, info, temporal)
            write_json(relocate_path(folder / "validation.json"), dict(layers=capture.validation,
                       shape=list(temporal.shape), key_group_order=["video", "other_prefill", "query", "post_query"],
                       elapsed_seconds=time.time() - start,
                       peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30))
            del inputs
            torch.cuda.empty_cache()
            print("Saved", sample["sample_key"], "seconds", round(time.time() - start, 1), flush=True)
    finally:
        capture.restore()


if __name__ == "__main__":
    main()
