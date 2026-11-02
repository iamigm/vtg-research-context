from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Descriptive attention/GT alignment analysis; no Pearson correlation."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = QUERY
DATASETS = ["Charades-STA", "QVHighlights", "ActivityNet-Captions"]
COLORS = ["#2171b5", "#e39024", "#239761"]


def merged_gt(windows, duration):
    intervals = sorted((max(0.0, float(s)), min(duration, float(e))) for s, e in windows if min(duration, e) > max(0, s))
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def overlap_fractions(edges, windows):
    overlap = np.zeros(len(edges) - 1)
    for s, e in windows:
        overlap += np.maximum(0, np.minimum(edges[1:], e) - np.maximum(edges[:-1], s))
    return overlap / np.diff(edges)


def analyze_sample(sample):
    folder = relocate_path(RESULTS / sample["dataset"] / sample["sample_key"])
    meta = json.loads((relocate_path(folder / "metadata.json")).read_text())
    validation = json.loads((relocate_path(folder / "validation.json")).read_text())
    with np.load(relocate_path(folder / "attention.npz")) as data:
        attn = data["temporal_attention"]
        mass = data["key_group_mass"]
    assert attn.shape == (36, 32, len(meta["query_positions"]), meta["video_grid_thw"][0])
    assert np.isfinite(attn).all() and attn.min() >= 0
    assert np.allclose(attn.sum(-1), mass[..., 0], atol=2e-6)
    assert np.max(np.abs(mass.sum(-1) - 1)) < 0.0041
    assert np.max(np.abs(mass[..., 3])) < 1e-6
    assert len(validation["layers"]) == 36
    raw = attn[8:22].mean(axis=(0, 1, 2)).astype(float)
    probability = raw / raw.sum()
    layer_raw = attn.mean(axis=(1, 2)).astype(float)
    layer_probability = layer_raw / layer_raw.sum(-1, keepdims=True)
    ts = np.array(meta["sampled_timestamps"])
    duration = meta["duration_video"]
    edges = np.r_[0, (ts[:-1] + ts[1:]) / 2, duration]
    assert np.all(np.diff(edges) > 0)
    windows = merged_gt(meta["gt_windows"], duration)
    fraction = overlap_fractions(edges, windows)
    coverage = sum(e - s for s, e in windows) / duration
    gt_mask = np.array([any(s <= t <= e for s, e in windows) for t in ts])
    peak_idx = int(raw.argmax())
    gt_mass = float(probability @ fraction)
    peak_after_first = 1 + int(raw[1:].argmax()) if len(raw) > 1 else 0
    row = dict(
        dataset=sample["dataset"], split=sample["split"], sample_key=sample["sample_key"],
        qid=sample["qid"], vid=sample["vid"], query=sample["query"],
        duration_video=duration, original_gt_windows=meta["gt_windows"], effective_gt_windows=windows,
        gt_outside_video=any(s < 0 or e > duration for s, e in meta["gt_windows"]),
        gt_overrun_seconds=max(0.0, max(e for s, e in meta["gt_windows"]) - duration),
        valid_gt_in_video=bool(windows), gt_time_coverage=coverage,
        gt_bin_coverage=float(gt_mask.mean()), total_video_attention=float(raw.sum()),
        attention_mass_in_gt=gt_mass, gt_mass_excess=gt_mass - coverage,
        gt_mass_enrichment=gt_mass / coverage if coverage > 0 else None,
        peak_time_seconds=float(ts[peak_idx]), peak_relative_time=float(ts[peak_idx] / duration),
        peak_in_gt=bool(gt_mask[peak_idx]), peak_is_first_bin=peak_idx == 0,
        first_bin_in_gt=bool(gt_mask[0]),
        peak_in_first_10_percent=bool(ts[peak_idx] / duration < 0.1),
        first_bin_attention_mass=float(probability[0]),
        peak_excluding_first_in_gt=bool(gt_mask[peak_after_first]),
        peak_excluding_first_seconds=float(ts[peak_after_first]),
        sequence_length=meta["sequence_length"], n_query_tokens=len(meta["query_positions"]),
        max_row_sum_error=max(v["attention_sum_max_error"] for v in validation["layers"]),
        peak_gpu_allocated_gib=validation["peak_gpu_allocated_gib"],
        backend=validation["layers"][0]["backend"],
    )
    # Absolute time remains in individual figures. This auxiliary resampling is
    # normalized by video duration and integrates piecewise-uniform bin mass.
    aligned_edges = np.linspace(0, duration, 101)
    aligned = np.zeros(100)
    density = probability / np.diff(edges)
    for i, (s, e) in enumerate(zip(aligned_edges[:-1], aligned_edges[1:])):
        aligned[i] = np.sum(density * np.maximum(0, np.minimum(edges[1:], e) - np.maximum(edges[:-1], s)))
    assert np.isclose(aligned.sum(), 1)
    gt_aligned = overlap_fractions(aligned_edges, windows)
    layer_hit = gt_mask[layer_raw.argmax(-1)].astype(float)
    layer_excess = layer_probability @ fraction - coverage
    return row, aligned, gt_aligned, layer_hit, layer_excess


def verify_provenance(samples):
    root = relocate_path(Path("/data/video/MR"))
    qv = {str(r["qid"]): r for r in map(json.loads, (relocate_path(root / "QVHighlights/highlight_val_release.jsonl")).read_text().splitlines())}
    ch = (relocate_path(root / "Charades-STA/charades_sta_test.txt")).read_text().splitlines()
    an = json.loads((relocate_path(root / "ActivityNet-Captions/annotations/val_1.json")).read_text())
    for s in samples:
        if s["dataset"] == "QVHighlights":
            r = qv[s["qid"]]
            assert (r["query"], r["vid"], r["relevant_windows"]) == (s["original_query"], s["vid"], s["gt_windows"])
        elif s["dataset"] == "Charades-STA":
            span, query = ch[int(s["qid"].split("_")[1])].split("##")
            vid, start, end = span.split()
            assert (vid, query, [[float(start), float(end)]]) == (s["vid"], s["original_query"], s["gt_windows"])
        else:
            r = an[s["vid"]]
            i = int(s["qid"].rsplit("_", 1)[1])
            assert (r["sentences"][i], [r["timestamps"][i]]) == (s["original_query"], s["gt_windows"])
    before = json.loads((relocate_path(ROOT / "source_sha256.json")).read_text())
    source = relocate_path(Path("/home/irteam/vtg_works/MTLA"))
    after = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    assert before == after
    assert all(hashlib.sha256((relocate_path(ROOT / "MTLA" / p)).read_bytes()).hexdigest() == h for p, h in before.items())
    return dict(official_annotations_verified=len(samples), original_source_files_unchanged=len(before), copied_source_files_unchanged=len(before))


def plots(rows, summary, aligned, gt_aligned, layer_hit, layer_excess):
    labels = ["Charades-STA", "QVHighlights", "ActivityNet"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    x = np.arange(3)
    axes[0, 0].bar(x - 0.24, [summary[d]["peak_in_gt_rate"] * 100 for d in DATASETS], 0.24, label="Observed peak in GT", color=COLORS)
    axes[0, 0].bar(x, [summary[d]["uniform_bin_peak_baseline"] * 100 for d in DATASETS], 0.24, label="Uniform-bin reference", color="#b8bdc7")
    axes[0, 0].bar(x + 0.24, [summary[d]["first_bin_baseline"] * 100 for d in DATASETS], 0.24, label="Always-first-bin reference", color="#64748b")
    axes[0, 0].set(xticks=x, xticklabels=labels, ylabel="Samples (%)", ylim=(0, 100), title="Does the highest attention peak fall inside GT?")
    axes[0, 0].legend(fontsize=8)
    for i, d in enumerate(DATASETS):
        subset = [r for r in rows if r["dataset"] == d and r["valid_gt_in_video"]]
        values = np.array([r["gt_mass_excess"] * 100 for r in subset])
        jitter = np.random.default_rng(13).uniform(-0.15, 0.15, len(values))
        axes[0, 1].scatter(i + jitter, values, s=14, alpha=0.6, color=COLORS[i])
        axes[0, 1].plot([i - 0.2, i + 0.2], [np.median(values)] * 2, lw=3, color="black")
    axes[0, 1].axhline(0, color="#666666", ls="--", lw=1)
    axes[0, 1].set(xticks=x, xticklabels=labels, ylabel="GT attention mass − GT time coverage (pp)", title="Attention inside GT, accounting for GT duration")
    axes[1, 0].bar(x - 0.16, [summary[d]["first_bin_peak_rate"] * 100 for d in DATASETS], 0.32, color=COLORS, label="Peak at first sampled bin")
    axes[1, 0].bar(x + 0.16, [summary[d]["first_10_percent_peak_rate"] * 100 for d in DATASETS], 0.32, color=COLORS, alpha=0.4, label="Peak in first 10% of video")
    axes[1, 0].set(xticks=x, xticklabels=labels, ylim=(0, 100), ylabel="Samples (%)", title="Early-video concentration")
    axes[1, 0].legend(fontsize=8)
    for i, d in enumerate(DATASETS):
        idx = [j for j, r in enumerate(rows) if r["dataset"] == d and r["valid_gt_in_video"]]
        axes[1, 1].plot(np.arange(36), layer_hit[idx].mean(0) * 100, label=labels[i], color=COLORS[i])
    axes[1, 1].axvspan(8, 21, color="#aaa", alpha=0.15, label="Default band 8–21")
    axes[1, 1].set(xlabel="Decoder layer (0-based)", ylabel="Samples with peak in GT (%)", ylim=(0, 100), title="Layer-specific peak alignment")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Input query → video attention · 100 samples per dataset\nLayers 8–21, all heads/query tokens · descriptive statistics; no Pearson correlation", fontsize=14)
    fig.savefig(relocate_path(RESULTS / "summary.png"), dpi=180)
    fig.savefig(relocate_path(RESULTS / "summary.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    t = (np.arange(100) + 0.5) / 100
    for i, d in enumerate(DATASETS):
        idx = [j for j, r in enumerate(rows) if r["dataset"] == d]
        y = aligned[idx] * 100  # density on normalized video time; integral = 1
        axes[i].plot(t, y.mean(0), color=COLORS[i], label="Mean video-conditional attention density")
        axes[i].plot(t, np.median(y, axis=0), color=COLORS[i], ls="--", alpha=0.7, label="Median density")
        axes[i].axhline(1, color="#777", ls=":", label="Uniform attention density")
        axes[i].set(title=d, ylabel="Attention density", xlabel="Relative video time (0 = start, 1 = end)", xlim=(0, 1))
        twin = axes[i].twinx()
        twin.fill_between(t, 0, gt_aligned[idx].mean(0), color="#2e9f64", alpha=0.13)
        twin.set(ylim=(0, 1), ylabel="GT occupancy across samples")
        axes[i].legend(fontsize=8, loc="upper right")
    fig.suptitle("Supplement: attention and GT location across relative video time\nIndividual sample figures retain seconds on the x-axis", fontsize=13)
    fig.savefig(relocate_path(RESULTS / "relative_time_summary.png"), dpi=180)
    fig.savefig(relocate_path(RESULTS / "relative_time_summary.pdf"))
    plt.close(fig)
    np.savez_compressed(relocate_path(RESULTS / "aggregate_arrays.npz"), relative_time_mass=aligned,
                        relative_time_gt_occupancy=gt_aligned, layer_peak_in_gt=layer_hit,
                        layer_gt_mass_excess=layer_excess)


def main():
    samples = json.loads((relocate_path(RESULTS / "samples.json")).read_text())
    assert len(samples) == 300
    records = [analyze_sample(s) for s in samples]
    rows = [r[0] for r in records]
    aligned, gt_aligned, layer_hit, layer_excess = [np.stack([r[k] for r in records]) for k in range(1, 5)]
    summary = {}
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset]
        assert len(subset) == len({r["vid"] for r in subset}) == 100
        valid = [r for r in subset if r["valid_gt_in_video"]]
        mean = lambda key: float(np.mean([r[key] for r in valid]))
        summary[dataset] = dict(samples=len(subset), valid_gt_samples=len(valid),
            gt_extends_beyond_video=sum(r["gt_outside_video"] for r in subset),
            gt_overrun_more_than_one_second=sum(r["gt_overrun_seconds"] > 1 for r in subset),
            peak_in_gt_count=sum(r["peak_in_gt"] for r in valid), peak_in_gt_rate=mean("peak_in_gt"),
            uniform_bin_peak_baseline=mean("gt_bin_coverage"),
            first_bin_baseline=mean("first_bin_in_gt"),
            mean_gt_time_coverage=mean("gt_time_coverage"), mean_gt_attention_mass=mean("attention_mass_in_gt"),
            mean_gt_mass_excess=mean("gt_mass_excess"),
            median_gt_mass_excess=float(np.median([r["gt_mass_excess"] for r in valid])),
            median_gt_mass_enrichment=float(np.median([r["gt_mass_enrichment"] for r in valid])),
            above_uniform_time_count=sum(r["gt_mass_excess"] > 0 for r in valid),
            first_bin_peak_rate=mean("peak_is_first_bin"), first_10_percent_peak_rate=mean("peak_in_first_10_percent"),
            peak_excluding_first_in_gt_rate=mean("peak_excluding_first_in_gt"),
            mean_first_bin_attention_mass=mean("first_bin_attention_mass"))
    write_result_text((relocate_path(RESULTS / "summary.json")), json.dumps(summary, indent=2) + "\n")
    write_result_text((relocate_path(RESULTS / "sample_metrics.json")), json.dumps(rows, indent=2) + "\n")
    with (relocate_path(RESULTS / "sample_metrics.csv")).open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = verify_provenance(samples)
    provenance.update(completed_samples=300, validated_layers=10800,
        max_row_sum_error=max(r["max_row_sum_error"] for r in rows),
        max_sequence_length=max(r["sequence_length"] for r in rows),
        peak_gpu_allocated_gib=max(r["peak_gpu_allocated_gib"] for r in rows))
    write_result_text((relocate_path(RESULTS / "final_validation.json")), json.dumps(provenance, indent=2) + "\n")
    plots(rows, summary, aligned, gt_aligned, layer_hit, layer_excess)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
