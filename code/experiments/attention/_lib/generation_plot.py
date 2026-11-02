from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Plot actual query/answer attention with distinct GT and generated interval tracks."""
import argparse
import csv
import html
import json
import sys
import time
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from comparison import ROOT, RESULTS, DATASETS, LABELS, COLORS, load_sample, curves_for
sys.path.insert(0,str(relocate_path(ROOT/"python_deps")))

plt.rcParams.update({"font.size":9,"savefig.dpi":160,"axes.spines.top":False,"axes.spines.right":False})


def display_end(meta, answer):
    return max([meta["duration_video"]]+[e for s,e in meta["gt_windows"]]+[e for s,e in answer["parsed_spans"]])


def tracks(ax, meta, answer, labels=True):
    end=display_end(meta,answer)
    ax.set(xlim=(0,end),ylim=(0,2),yticks=[.45,1.4],yticklabels=["Pred","GT"],xlabel="Time (seconds)")
    ax.tick_params(axis="y",length=0)
    ax.spines["left"].set_visible(False)
    for windows,y,color in [(meta["gt_windows"],1.12,"#24965f"),(answer["parsed_spans"],.18,"#ca5256")]:
        for start,stop in windows:
            ax.broken_barh([(start,stop-start)],(y,.48),facecolors=color,alpha=.85)
            if labels:
                ax.text((start+stop)/2,y+.55,f"[{start:g}, {stop:g}]",ha="center",fontsize=7)
    if not answer["parsed_spans"]:
        ax.text(end*.02,.45,"No parsed interval",color="#ca5256",va="center",fontsize=8)
    if end>meta["duration_video"]:
        ax.axvspan(meta["duration_video"],end,color="#aaaaaa",alpha=.2,hatch="///")


def lines(ax,meta,answer,curves,normalized=False,predictive=False,compact=False):
    scopes=["query","predictive","predictive_coordinates"] if predictive else ["query","answer","coordinates"]
    for scope in scopes:
        if scope not in curves: continue
        values=curves[scope]
        if normalized: values=values/values.sum()
        ax.plot(meta["sampled_timestamps"],values,color=COLORS[scope],lw=1.5 if scope=="query" else 1.15,label=LABELS[scope],alpha=.95)
    ax.set_xlim(0,display_end(meta,answer)); ax.set_ylim(bottom=0)
    ax.set_ylabel("Conditional attention" if normalized else "Attention score")
    ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
    ax.tick_params(labelbottom=False)
    ax.grid(axis="y",alpha=.2)
    ax.legend(fontsize=6 if compact else 8,loc="upper right")


def detail(meta,answer,arrays):
    curves=curves_for(answer,arrays)
    nq=len(meta["query_tokens"]); na=len(answer["tokens"])
    hq=max(1.4,nq*.09); ha=max(1.6,na*.09)
    fig=plt.figure(figsize=(13.5,9+hq+ha))
    gs=fig.add_gridspec(9,2,height_ratios=[2,.8,.2,2,.8,.2,hq,ha,.8],width_ratios=[1,.017],hspace=.21,wspace=.08)
    title=f"{meta['dataset']} · {meta['qid']}\nQuery: "+textwrap.fill(meta["query"].strip(),110)
    response=textwrap.shorten(answer["text"].replace("\n"," "),width=230,placeholder="…")
    answer_label = "Answer (TRUNCATED at 128 tokens): " if answer["truncated"] else "Answer: "
    fig.suptitle(title+"\n"+answer_label+textwrap.fill(response,110),fontsize=11,y=.985)
    a=fig.add_subplot(gs[0,0]); lines(a,meta,answer,curves)
    a.set_title("Raw video attention · layers 8–21, all heads, mean over each token group",fontsize=10)
    track=fig.add_subplot(gs[1,0],sharex=a); tracks(track,meta,answer); track.set_xlabel("")
    b=fig.add_subplot(gs[3,0],sharex=a); lines(b,meta,answer,curves,normalized=True)
    b.set_title("Distribution shape · each curve normalized to total video attention = 1",fontsize=10)
    track=fig.add_subplot(gs[4,0],sharex=a); tracks(track,meta,answer,labels=False); track.set_xlabel("")
    t=np.array(meta["sampled_timestamps"]); edges=np.r_[0,(t[:-1]+t[1:])/2,meta["duration_video"]]
    for row,scope,tokens in [(6,"query",meta["query_tokens"]),(7,"answer",answer["tokens"])]:
        ax=fig.add_subplot(gs[row,0],sharex=a)
        value=arrays[f"{scope}_temporal"][8:22].mean((0,1))
        if len(value):
            value=value/value.sum(-1,keepdims=True)
            mesh=ax.pcolormesh(edges,np.arange(len(value)+1),value,cmap="magma",rasterized=True)
            ax.set_ylim(len(value),0)
            ax.set_yticks(np.arange(len(value))+.5)
            ax.set_yticklabels([f"{i}: {token!r}" for i,token in enumerate(tokens)],fontsize=6.5)
            fig.colorbar(mesh,cax=fig.add_subplot(gs[row,1])).ax.set_ylabel("Video-conditional mass",fontsize=7)
        ax.set_ylabel("Query token" if scope=="query" else "Answer token")
        ax.tick_params(labelbottom=False)
    tracks(fig.add_subplot(gs[8,0],sharex=a),meta,answer,labels=False)
    fig.subplots_adjust(left=.15,right=.91,top=.88,bottom=.085)
    fig.text(.15,.018,"Answer curves use rows consuming generated tokens (MTLA convention). EOS excluded. GT never enters the model.\n"
             "Pred is the model's parsed answer; heatmaps normalize each row over video. See predictive.png for rows producing tokens.",fontsize=8)
    return fig


def render_one(sample):
    folder,meta,answer,arrays=load_sample(sample)
    curves=curves_for(answer,arrays)
    fig=detail(meta,answer,arrays)
    fig.savefig(relocate_path(folder/"comparison.png")); fig.savefig(relocate_path(folder/"comparison.pdf")); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,gridspec_kw={"height_ratios":[3,3,1.2],"hspace":.1})
    lines(axes[0],meta,answer,curves,predictive=True)
    lines(axes[1],meta,answer,curves,normalized=True,predictive=True)
    tracks(axes[2],meta,answer)
    axes[0].set_title("Query vs attention used to PREDICT answer tokens\n"+textwrap.fill(meta["query"].strip(),105),fontsize=10)
    fig.subplots_adjust(left=.12,right=.98,top=.88,bottom=.1)
    fig.savefig(relocate_path(folder/"predictive.png"));plt.close(fig)
    with (relocate_path(folder/"temporal_comparison.csv")).open("w") as file:
        names=list(curves)
        writer=csv.writer(file);writer.writerow(["time_seconds"]+names+[f"{k}_video_conditional" for k in names])
        for i,t in enumerate(meta["sampled_timestamps"]):
            writer.writerow([t]+[curves[k][i] for k in names]+[curves[k][i]/curves[k].sum() for k in names])
    write_result_text((relocate_path(folder/"PLOTTED")), "complete\n")


def overview_page(samples,title):
    nrows=(len(samples)+1)//2
    fig=plt.figure(figsize=(17,4.2*nrows+1))
    outer=fig.add_gridspec(nrows,2,hspace=.65,wspace=.24)
    for i,sample in enumerate(samples):
        _,meta,answer,arrays=load_sample(sample);curves=curves_for(answer,arrays)
        gs=outer[i//2,i%2].subgridspec(2,1,height_ratios=[3.5,1.2],hspace=.07)
        ax=fig.add_subplot(gs[0]); lines(ax,meta,answer,curves,normalized=True,compact=True)
        q=textwrap.fill(textwrap.shorten(meta["query"].strip(),width=130,placeholder="…"),66)
        ax.set_title(f"{meta['qid']} · "+q,fontsize=8)
        tracks(fig.add_subplot(gs[1],sharex=ax),meta,answer)
    fig.suptitle(title+"\nBlue: query · orange: all answer tokens · purple: timestamp digits · green GT / red prediction",fontsize=12)
    fig.subplots_adjust(top=.86 if nrows==1 else .91,left=.06,right=.99,bottom=.07)
    return fig


def assemble(samples):
    from pypdf import PdfWriter
    all_pdf=PdfWriter();links=[]
    (relocate_path(RESULTS/"overview_pages")).mkdir(exist_ok=True)
    for name in DATASETS:
        subset=[s for s in samples if s["dataset"]==name];pdf=PdfWriter()
        for s in subset:
            path=str(relocate_path(RESULTS/ s["dataset"] / s["sample_key"]/"comparison.pdf"))
            pdf.append(path);all_pdf.append(path)
        pdf.write(str(relocate_path(RESULTS/f"{name}_detail.pdf")));pdf.close()
        with PdfPages(relocate_path(RESULTS/f"{name}_overview.pdf")) as overview:
            for start in range(0,100,6):
                fig=overview_page(subset[start:start+6],f"{name} · query vs generated answer · samples {start+1}–{min(start+6,100)}")
                overview.savefig(fig);fig.savefig(relocate_path(RESULTS/"overview_pages"/f"{name}_{start//6+1:02d}.png"));plt.close(fig)
        links.append(f'<li>{name}: <a href="{name}_overview.pdf">100개 비교 곡선</a> · <a href="{name}_detail.pdf">토큰별 상세 PDF</a></li>')
        print("Assembled",name,flush=True)
    all_pdf.write(str(relocate_path(RESULTS/"comparison_report.pdf")));all_pdf.close()
    cards=[]
    for s in samples:
        key=s["sample_key"];answer=json.loads((relocate_path(RESULTS/ s["dataset"] / key/"answer.json")).read_text())
        status = '<p><strong>128-token 제한으로 잘린 답변</strong></p>' if answer["truncated"] else ''
        cards.append(f'<article data-dataset="{s["dataset"]}"><h2>{s["dataset"]} · {html.escape(s["qid"])}</h2>'+status+
            f'<p>Query: {html.escape(s["query"])}</p><pre>Answer: {html.escape(answer["text"])}</pre>'
            f'<a href="{s["dataset"]}/{key}/comparison.png"><img loading="lazy" src="{s["dataset"]}/{key}/comparison.png"></a>'
            f'<p><a href="{s["dataset"]}/{key}/predictive.png">토큰 생성 직전 attention</a> · <a href="{s["dataset"]}/{key}/answer.json">답변·예측 구간</a> · '
            f'<a href="{s["dataset"]}/{key}/temporal_comparison.csv">CSV</a> · <a href="{s["dataset"]}/{key}/attention.npz">배열</a></p></article>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Query vs generated answer attention</title>
<style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;background:#f5f7fa;color:#1e293b}img{width:100%}article{background:white;padding:24px;margin:30px 0;border-radius:12px}a{color:#126cb6}pre{white-space:pre-wrap;background:#f1f5f9;padding:14px}nav{position:sticky;top:0;background:#eef3f9;padding:14px;z-index:1}input,select{padding:8px;margin:0 12px}</style>
<h1>입력 query vs 실제 생성 답변 attention</h1><p>동일한 300개 샘플 · Qwen3-VL-8B-Instruct · greedy 답변 1개씩 · GT 입력 없음</p>
<p>파랑: 입력 query, 주황: 답변 전체, 보라: 답변 timestamp 숫자. 초록: GT, 빨강: 모델 예측 구간.
상단은 원래 attention score, 그 아래는 영상 내 합계 1로 정규화한 분포입니다. Pearson correlation은 계산하지 않았습니다.</p>
<p><a href="FINDINGS.md">분석 결과</a> · <a href="../../README.md">실험 방법</a> · <a href="summary.png">요약 그림</a> · <a href="comparison_report.pdf">전체 상세 PDF</a></p><ul>'''+''.join(links)+'''</ul><img src="summary.png">
<nav><select id="dataset"><option value="">전체 데이터셋</option><option>Charades-STA</option><option>QVHighlights</option><option>ActivityNet-Captions</option></select><input id="search" placeholder="query / 답변 / ID 검색"><span id="count"></span></nav>'''+''.join(cards)+'''
<script>const d=document.getElementById('dataset'),s=document.getElementById('search'),c=document.getElementById('count');const filter=()=>{let n=0;document.querySelectorAll('article').forEach(a=>{const show=(!d.value||a.dataset.dataset===d.value)&&a.textContent.toLowerCase().includes(s.value.toLowerCase());a.hidden=!show;if(show)n++;});c.textContent=n+' samples';};d.onchange=filter;s.oninput=filter;filter();</script></html>'''
    write_result_text((relocate_path(RESULTS/"report.html")), page)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--watch",action="store_true");args=parser.parse_args()
    samples=json.loads((relocate_path(RESULTS/"samples.json")).read_text())
    while True:
        count=0
        for sample in samples:
            folder=relocate_path(RESULTS/ sample["dataset"] / sample["sample_key"])
            if not (relocate_path(folder/"validation.json")).exists(): continue
            if not (relocate_path(folder/"PLOTTED")).exists():
                render_one(sample);print("Plotted",sample["sample_key"],flush=True)
            count+=1
        if count==len(samples):break
        if not args.watch:raise RuntimeError(f"{count}/300 ready; use --watch")
        time.sleep(10)
    assemble(samples);print("Complete 300 comparison plots",flush=True)


if __name__=="__main__":main()
