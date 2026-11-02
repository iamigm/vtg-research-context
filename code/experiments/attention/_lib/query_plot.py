from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Standalone temporal attention figures with separately labeled GT tracks."""
import csv
import html
import json
from pathlib import Path
import textwrap
import argparse
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np

RESULTS = QUERY
sys.path.insert(0, str(relocate_path(ROOT / "python_deps")))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 170, "font.family": "DejaVu Sans"})


def gt_track(ax, meta, labels=True):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    end_time = max(meta["duration_video"], max(e for s, e in meta["gt_windows"]))
    ax.set_xlim(0, end_time)
    if end_time > meta["duration_video"]:
        ax.axvspan(meta["duration_video"], end_time, facecolor="#dddddd", hatch="///", alpha=0.5)
    for start, end in meta["gt_windows"]:
        ax.axvspan(start, end, ymin=0.18, ymax=0.7, color="#2e9f64", alpha=0.85)
        if labels:
            ax.text((start + end) / 2, 0.85, f"[{start:g}, {end:g}]", ha="center", va="center", fontsize=8)
    ax.set_ylabel("GT", rotation=0, labelpad=14, va="center", color="#177746")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (seconds)")


def curve(ax, meta, attention, uniform=False, compact=False):
    band = attention[8:22].mean(axis=(0, 1, 2))
    overall = attention.mean(axis=(0, 1, 2))
    times = np.asarray(meta["sampled_timestamps"])
    if uniform:
        e = np.array(meta["mtla_uniform_bin_edges"])
        times = (e[1:] + e[:-1]) / 2
    ax.plot(times, band, color="#126cb6", lw=1.7, label="Layers 8–21")
    ax.plot(times, overall, color="#e38c32", lw=1.1, alpha=0.85, label="All 36 layers")
    ax.axhline(band.mean(), color="#777777", lw=0.8, ls=":", label="Uniform reference (same mass)")
    ax.set_ylim(bottom=0)
    end_time = max(meta["duration_video"], max(e for s, e in meta["gt_windows"]))
    ax.set_xlim(0, end_time)
    if end_time > meta["duration_video"]:
        ax.axvspan(meta["duration_video"], end_time, color="#aaaaaa", alpha=0.2, label="Beyond video duration")
    ax.set_ylabel("Attention score" if compact else "Attention score\n(probability mass / temporal bin)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper right", fontsize=7 if compact else 8, framealpha=0.85)


def detail(meta, attn):
    nq = len(meta["query_tokens"])
    height = 11.5 + max(0, nq - 20) * 0.12
    fig = plt.figure(figsize=(13.5, height))
    gs = fig.add_gridspec(8, 2, height_ratios=[3.0, 0.55, 0.3, max(2.4, nq * 0.105), 0.4, 0.3, 2.7, 0.4],
                          width_ratios=[1, 0.018], hspace=0.14, wspace=0.07)
    title = f"{meta['dataset']} · {meta['split']} · {meta['qid']} · video {meta['vid']}"
    fig.suptitle(title + "\n" + textwrap.fill(meta["query"].strip(), 115), fontsize=13, y=0.985)
    ax = fig.add_subplot(gs[0, 0])
    curve(ax, meta, attn)
    gt1 = fig.add_subplot(gs[1, 0], sharex=ax)
    gt_track(gt1, meta)
    gt1.set_xlabel("")
    ts = np.array(meta["sampled_timestamps"])
    edges = np.r_[0, (ts[1:] + ts[:-1]) / 2, meta["duration_video"]]
    assert np.all(np.diff(edges) > 0)
    by_token = attn[8:22].mean(axis=(0, 1))
    by_token = by_token / by_token.sum(-1, keepdims=True)
    token_ax = fig.add_subplot(gs[3, 0], sharex=ax)
    mesh = token_ax.pcolormesh(edges, np.arange(len(by_token) + 1), by_token, cmap="magma", shading="flat", rasterized=True)
    token_ax.set_ylim(len(by_token), 0)
    token_ax.set_yticks(np.arange(len(by_token)) + 0.5)
    token_ax.set_yticklabels([f"{i}: {token!r}" for i, token in enumerate(meta["query_tokens"])], fontsize=7)
    token_ax.set_ylabel("Query token")
    token_ax.set_title("Query-token detail · mean of heads and layers 8–21 · each row normalized over video", fontsize=10)
    token_ax.tick_params(labelbottom=False)
    fig.colorbar(mesh, cax=fig.add_subplot(gs[3, 1])).ax.set_ylabel("Video-conditional mass", fontsize=8)
    gt2 = fig.add_subplot(gs[4, 0], sharex=ax)
    gt_track(gt2, meta, labels=False)
    gt2.set_xlabel("")
    by_layer = attn.mean(axis=(1, 2))
    by_layer = by_layer / by_layer.sum(-1, keepdims=True)
    layer_ax = fig.add_subplot(gs[6, 0], sharex=ax)
    mesh = layer_ax.pcolormesh(edges, np.arange(len(by_layer) + 1), by_layer, cmap="magma", shading="flat", rasterized=True)
    layer_ax.set_ylim(len(by_layer), 0)
    layer_ax.set_yticks(np.arange(0, 36, 4) + 0.5, np.arange(0, 36, 4))
    for boundary in [8, 22]:
        layer_ax.axhline(boundary, color="#6ddace", lw=0.9, ls="--")
    layer_ax.set_ylabel("Decoder layer (0-based)")
    layer_ax.set_title("Layer detail · mean of heads and query tokens · each row normalized over video", fontsize=10)
    layer_ax.tick_params(labelbottom=False)
    fig.colorbar(mesh, cax=fig.add_subplot(gs[6, 1])).ax.set_ylabel("Video-conditional mass", fontsize=8)
    gt_track(fig.add_subplot(gs[7, 0], sharex=ax), meta, labels=False)
    fig.subplots_adjust(left=0.14, right=0.91, top=0.90, bottom=0.10)
    fig.text(0.14, 0.015, "Qwen3-VL-8B-Instruct · input event query → earlier video tokens · spatial-token sum, head/query mean\n"
             "Top: unnormalized attention from full causal softmax. Heatmaps: conditional on video. GT is never a model input.",
             fontsize=8, color="#444444")
    return fig


def load_sample(sample):
    folder = relocate_path(RESULTS / sample["dataset"] / sample["sample_key"])
    meta = json.loads((relocate_path(folder / "metadata.json")).read_text())
    with np.load(relocate_path(folder / "attention.npz")) as z:
        attn = z["temporal_attention"]
    return folder, meta, attn


def render_one(sample):
    folder, meta, attn = load_sample(sample)
    fig = detail(meta, attn)
    fig.savefig(relocate_path(folder / "attention.png"))
    fig.savefig(relocate_path(folder / "attention.pdf"))
    plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True,
                           gridspec_kw={"height_ratios": [4, 0.7], "hspace": 0.05})
    curve(axs[0], meta, attn, uniform=True)
    gt_track(axs[1], meta)
    axs[0].set_title(f"{meta['dataset']} · {meta['qid']} · MTLA uniform-duration temporal bins\n" + textwrap.fill(meta["query"].strip(), 105), fontsize=10)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.79, bottom=0.14)
    fig.savefig(relocate_path(folder / "attention_mtla_time.png"), bbox_inches="tight")
    plt.close(fig)
    write_result_text((relocate_path(folder / "PLOTTED")), "complete\n")


def overview_page(samples, title):
    nrows = (len(samples) + 2) // 3
    fig = plt.figure(figsize=(19, max(4.5, 4 * nrows + 0.5)))
    outer = fig.add_gridspec(nrows, 3, hspace=0.6, wspace=0.28)
    for index, sample in enumerate(samples):
        _, meta, attn = load_sample(sample)
        gs = outer[index // 3, index % 3].subgridspec(2, 1, height_ratios=[4, 0.8], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        curve(ax, meta, attn, compact=True)
        preview = textwrap.shorten(meta["query"].strip(), width=155, placeholder="…")
        ax.set_title(f"{meta['dataset']} · {meta['qid']}\n" + textwrap.fill(preview, 53), fontsize=9, pad=9)
        gt_track(fig.add_subplot(gs[1], sharex=ax), meta)
        ax.xaxis.set_major_locator(MaxNLocator(6))
    fig.suptitle(title, fontsize=18, y=0.99)
    fig.subplots_adjust(top=0.88 if nrows == 1 else 0.90, bottom=0.08 if nrows == 1 else 0.045,
                        left=0.05, right=0.985)
    return fig


def assemble(samples):
    from pypdf import PdfWriter
    all_writer = PdfWriter()
    links = []
    (relocate_path(RESULTS / "overview_pages")).mkdir(exist_ok=True)
    for name in dict.fromkeys(s["dataset"] for s in samples):
        subset = [s for s in samples if s["dataset"] == name]
        writer = PdfWriter()
        for s in subset:
            path = str(relocate_path(RESULTS / s["dataset"] / s["sample_key"] / "attention.pdf"))
            writer.append(path)
            all_writer.append(path)
        writer.write(str(relocate_path(RESULTS / f"{name}_detail.pdf")))
        writer.close()
        with PdfPages(relocate_path(RESULTS / f"{name}_overview.pdf")) as pdf:
            for start in range(0, len(subset), 9):
                fig = overview_page(subset[start:start+9], f"{name}: input query → video attention · samples {start+1}–{min(start+9,len(subset))}")
                fig.savefig(relocate_path(RESULTS / "overview_pages" / f"{name}_{start//9+1:02d}.png"))
                pdf.savefig(fig)
                plt.close(fig)
        links.append(f'<li>{name}: <a href="{name}_overview.pdf">100개 곡선 모음</a> · <a href="{name}_detail.pdf">100개 상세 그림</a></li>')
        print("Assembled", name, flush=True)
    all_writer.write(str(relocate_path(RESULTS / "attention_report.pdf")))
    all_writer.close()
    cards = []
    for sample in samples:
        key = sample["sample_key"]
        cards.append(f'<article data-dataset="{sample["dataset"]}"><h2>{sample["dataset"]} · {html.escape(sample["qid"])}</h2>'
                     f'<p>{html.escape(sample["query"])}</p><a href="{sample["dataset"]}/{key}/attention.png">'
                     f'<img loading="lazy" src="{sample["dataset"]}/{key}/attention.png"></a><p>'
                     f'<a href="{sample["dataset"]}/{key}/attention_mtla_time.png">MTLA 균등 시간축</a> · '
                     f'<a href="{sample["dataset"]}/{key}/temporal_attention.csv">CSV</a> · <a href="{sample["dataset"]}/{key}/attention.npz">배열</a> · '
                     f'<a href="{sample["dataset"]}/{key}/metadata.json">샘플 정보</a></p></article>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>MTLA attention · 300 samples</title>
<style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;background:#f5f7fa;color:#1e293b}img{width:100%}article{background:white;padding:24px;margin:30px 0;border-radius:12px}a{color:#126cb6}nav{position:sticky;top:0;background:#eef3f9;padding:14px;z-index:1}input,select{padding:8px;margin:0 12px}h1{line-height:1.2}</style>
<h1>MTLA 백본: query → video prefill attention</h1><p>각 데이터셋 100개, 총 300개 · Qwen3-VL-8B-Instruct · seed 20260929</p>
<p>파랑: 레이어 8–21, 주황: 전체 레이어. 시간(초)별 attention 곡선 아래 초록 막대로 원본 GT를 표시합니다.
GT가 영상 길이를 벗어나면 해당 범위를 회색으로 표시합니다. Pearson correlation은 계산하지 않았습니다.</p>
<p><a href="FINDINGS.md">분석 결과</a> · <a href="../../README.md">실험 설명</a> · <a href="summary.png">데이터셋 요약 그림</a> · <a href="attention_report.pdf">전체 상세 PDF</a></p>
<ul>''' + "".join(links) + '''</ul><img src="summary.png">
<nav><select id="dataset"><option value="">전체 데이터셋</option><option>Charades-STA</option><option>QVHighlights</option><option>ActivityNet-Captions</option></select><input id="search" placeholder="query / ID 검색"><span id="count"></span></nav>''' + "".join(cards) + '''
<script>const dataset=document.getElementById('dataset'),searchInput=document.getElementById('search'),count=document.getElementById('count');const filter=()=>{let n=0;document.querySelectorAll('article').forEach(a=>{const show=(!dataset.value||a.dataset.dataset===dataset.value)&&a.textContent.toLowerCase().includes(searchInput.value.toLowerCase());a.hidden=!show;if(show)n++;});count.textContent=n+' samples';};dataset.onchange=filter;searchInput.oninput=filter;filter();</script></html>'''
    write_result_text((relocate_path(RESULTS / "report.html")), page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    samples = json.loads((relocate_path(RESULTS / "samples.json")).read_text())
    while True:
        completed = 0
        for sample in samples:
            folder = relocate_path(RESULTS / sample["dataset"] / sample["sample_key"])
            if not (relocate_path(folder / "validation.json")).exists():
                continue
            if not (relocate_path(folder / "PLOTTED")).exists():
                render_one(sample)
                print("Plotted", sample["sample_key"], flush=True)
            completed += 1
        if completed == len(samples):
            break
        if not args.watch:
            raise RuntimeError(f"Only {completed}/{len(samples)} samples ready; use --watch during inference")
        time.sleep(10)
    assemble(samples)
    print("Complete:", len(samples), "samples", flush=True)


if __name__ == "__main__":
    main()
