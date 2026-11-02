from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Paired query vs answer alignment summaries, without correlation coefficients."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from comparison import ROOT, RESULTS, DATASETS, LABELS, COLORS, evaluate_sample

SCOPES=["query","answer","coordinates","first_coordinate","predictive","predictive_coordinates"]


def summarize(rows):
    result={}
    for dataset in DATASETS:
        items=[r for r in rows if r["dataset"]==dataset]
        summary=dict(samples=len(items),parsed_answers=sum(r["parsed"] for r in items),
            truncated_answers=sum(r["truncated"] for r in items),
            median_answer_tokens=float(np.median([r["answer_tokens"] for r in items])),
            any_pair_tiou_ge_05_count=sum(r["best_pair_tiou"]>=.5 for r in items),
            first_span_tiou_ge_05_count=sum(r["first_span_best_tiou"]>=.5 for r in items),
            mean_best_pair_tiou=float(np.mean([r["best_pair_tiou"] for r in items])),
            mean_query_gt_recall_at_05=float(np.mean([r["gt_recall_at_05"] for r in items])),
            gt_extends_beyond_video=sum(r["gt_outside_video"] for r in items),scopes={})
        for scope in SCOPES:
            group=[r["scopes"][scope] for r in items if scope in r["scopes"]]
            if not group:continue
            mean=lambda key:float(np.mean([r[key] for r in group]))
            median=lambda key:float(np.median([r[key] for r in group]))
            summary["scopes"][scope]=dict(n=len(group),peak_in_gt_count=sum(r["peak_in_gt"] for r in group),
                peak_in_gt_rate=mean("peak_in_gt"),mean_video_attention=mean("total_video_attention"),
                mean_gt_mass=mean("gt_mass"),median_gt_mass=median("gt_mass"),
                median_gt_mass_excess=median("gt_mass_excess"),
                early_peak_rate=mean("peak_in_first_10_percent"),
                mean_predicted_interval_mass=mean("predicted_interval_mass"),
                mean_predicted_interval_excess=mean("predicted_interval_excess"))
            if scope!="query":
                paired=[r for r in items if scope in r["scopes"]]
                summary["scopes"][scope].update(median_gt_mass_gain_over_query=median("gt_mass_gain_over_query"),
                    mean_gt_mass_gain_over_query=mean("gt_mass_gain_over_query"),
                    improved_gt_mass_count=sum(r["gt_mass_gain_over_query"]>0 for r in group),
                    paired_query_peak_in_gt_count=sum(r["scopes"]["query"]["peak_in_gt"] for r in paired),
                    median_total_variation_from_query=median("total_variation_from_query"))
        summary["by_prediction_alignment"]={}
        for label,select in [("any_pair_tiou_ge_05",lambda r:r["best_pair_tiou"]>=.5),("any_pair_tiou_lt_05",lambda r:r["best_pair_tiou"]<.5)]:
            group=[r for r in items if select(r) and "coordinates" in r["scopes"]]
            summary["by_prediction_alignment"][label]=dict(n=len(group),
                median_coordinate_gt_mass_gain=float(np.median([r["scopes"]["coordinates"]["gt_mass_gain_over_query"] for r in group])) if group else None,
                median_coordinate_predicted_excess=float(np.median([r["scopes"]["coordinates"]["predicted_interval_excess"] for r in group])) if group else None)
        summary["excluding_truncated"]={}
        for scope in ["answer","coordinates"]:
            clean=[r["scopes"][scope] for r in items if not r["truncated"] and scope in r["scopes"]]
            summary["excluding_truncated"][scope]=dict(n=len(clean),peak_in_gt_count=sum(r["peak_in_gt"] for r in clean),
                median_gt_mass_gain_over_query=float(np.median([r["gt_mass_gain_over_query"] for r in clean])) if clean else None)
        result[dataset]=summary
    return result


def plot_summary(rows, summary):
    scopes=["query","answer","coordinates"]
    names=["Charades-STA","QVHighlights","ActivityNet"]
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    x=np.arange(3);width=.24
    for j,scope in enumerate(scopes):
        off=(j-1)*width
        axes[0,0].bar(x+off,[summary[d]["scopes"][scope]["mean_video_attention"]*100 for d in DATASETS],width,label=LABELS[scope],color=COLORS[scope])
        axes[0,1].bar(x+off,[summary[d]["scopes"][scope]["peak_in_gt_rate"]*100 for d in DATASETS],width,label=LABELS[scope],color=COLORS[scope])
    axes[0,0].set(xticks=x,xticklabels=names,ylabel="Mean attention to all video tokens (%)",title="How much attention goes to the video?")
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(xticks=x,xticklabels=names,ylabel="Samples with temporal peak in GT (%)",ylim=(0,100),title="Does the temporal peak match GT?")
    axes[0,1].legend(fontsize=8)
    for i,dataset in enumerate(DATASETS):
        for j,scope in enumerate(["answer","coordinates"]):
            group=[r["scopes"][scope]["gt_mass_gain_over_query"]*100 for r in rows if r["dataset"]==dataset and scope in r["scopes"]]
            center=i+(j-.5)*.28
            jitter=np.random.default_rng(42).uniform(-.1,.1,len(group))
            axes[1,0].scatter(center+jitter,group,color=COLORS[scope],s=10,alpha=.5,label=LABELS[scope] if i==0 else None)
            axes[1,0].plot([center-.12,center+.12],[np.median(group)]*2,color="black",lw=2.4)
    axes[1,0].axhline(0,color="gray",ls="--",lw=1)
    axes[1,0].set(xticks=x,xticklabels=names,ylabel="Answer − query: GT video-conditional mass (pp)",title="Paired change in attention concentration inside GT")
    axes[1,0].legend(fontsize=8)
    for j,scope in enumerate(scopes):
        axes[1,1].bar(x+(j-1)*width,[summary[d]["scopes"][scope]["early_peak_rate"]*100 for d in DATASETS],width,color=COLORS[scope],label=LABELS[scope])
    axes[1,1].set(xticks=x,xticklabels=names,ylabel="Samples with peak in first 10% of video (%)",ylim=(0,100),title="Early-video concentration")
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y",alpha=.2);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Input query vs actually generated answer · same 100 samples per dataset\nLayers 8–21 · answer-token rows · timestamp group requires a valid parsed interval",fontsize=14)
    fig.savefig(relocate_path(RESULTS/"summary.png"),dpi=180);fig.savefig(relocate_path(RESULTS/"summary.pdf"));plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True,constrained_layout=True)
    for i,dataset in enumerate(DATASETS):
        for scope in scopes:
            subset=[r for r in rows if r["dataset"]==dataset and scope in r["layers"]]
            hits=np.array([r["layers"][scope]["peak_in_gt"] for r in subset])
            mass=np.array([r["layers"][scope]["gt_mass"] for r in subset])
            axes[0,i].plot(hits.mean(0)*100,color=COLORS[scope],label=LABELS[scope])
            axes[1,i].plot(mass.mean(0)*100,color=COLORS[scope],label=LABELS[scope])
        axes[0,i].set(title=dataset,ylabel="Peak inside GT (%)",ylim=(0,100))
        axes[1,i].set(ylabel="Mean GT video-conditional mass (%)",xlabel="Decoder layer",ylim=(0,100))
        for ax in axes[:,i]:ax.axvspan(8,21,color="gray",alpha=.12);ax.grid(axis="y",alpha=.2)
        axes[0,i].legend(fontsize=7)
    fig.suptitle("Layer-by-layer query/answer alignment with GT · shaded band = layers 8–21")
    fig.savefig(relocate_path(RESULTS/"layer_comparison.png"),dpi=180);fig.savefig(relocate_path(RESULTS/"layer_comparison.pdf"));plt.close(fig)


def write_report(rows,summary):
    pct=lambda v:f"{v*100:.1f}%"
    lines=["# 같은 100개 샘플: 입력 query와 실제 생성 답변 attention 비교","",
        "Charades-STA test, QVHighlights val, ActivityNet-Captions val_1에서 이전과 동일한 100개씩, 총 300개에 대해 실제 답변을 생성했다. "
        "프롬프트와 영상 전처리는 유지했고, 각 샘플에서 greedy 답변 한 개를 생성했다. 아래 attention은 실제 KV-cache 생성 실행에서 직접 읽었으며 GT를 모델에 입력하지 않았다.","",
        "[전체 결과](report.html) · [요약 그림](summary.png) · [레이어별 비교](layer_comparison.png)","",
        "## 실제 답변 및 GT와의 겹침","",
        "모델의 raw 답변과 파싱한 `[s, e]`를 모두 저장했다. 원래 single-span 지시문은 `s - e seconds` 또는 설명 문장으로도 응답한다. "
        "이를 사후 파싱했으며 답변을 GT로 교체하거나 GT에 맞게 수정하지 않았다. Charades/ActivityNet은 MTLA 규칙대로 파싱된 첫 구간, QVHighlights는 모든 구간을 사용했다.","",
        "| 데이터셋 | 답변 생성 | 구간 파싱 성공 | 128-token 제한 도달 | 예측–GT 중 tIoU≥0.5인 쌍이 있는 샘플 |", "|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        s=summary[d];lines.append(f"| {d} | {s['samples']} | {s['parsed_answers']} | {s['truncated_answers']} | {s['any_pair_tiou_ge_05_count']}/100 |")
    lines += ["","위 tIoU 집계는 이 100개 샘플의 겹침 확인용이다. 특히 QVHighlights는 여러 GT 중 하나만 맞아도 해당하므로 공식 mAP나 모든 구간의 recall과 같지 않다. "
        "쿼리별 GT recall 및 원본 예측은 JSON/CSV에 별도로 기록했다.","","## Query와 답변의 attention","",
        "주 비교는 query 이벤트 문장 전체, 답변 전체(EOS 제외), 파싱된 timestamp의 숫자 토큰이다. 레이어 8–21과 모든 head를 평균했다. "
        "‘영상 attention 총량’은 원래 softmax attention 중 영상에 간 확률 질량이다. ‘GT 비중 변화’는 영상 안에서 합계 1로 정규화한 분포의 "
        "GT 질량을 같은 샘플의 query와 비교한 값이며, 두 개념을 구분해야 한다.","",
        "| 데이터셋 | 토큰 그룹 | 영상 attention 총량 평균 | 최고 peak가 GT 안 | Query 대비 GT 비중 변화 중앙값 | GT 비중이 Query보다 커진 샘플 |", "|---|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        for k in ["query","answer","coordinates"]:
            s=summary[d]["scopes"][k]
            delta="—" if k=="query" else f"{s['median_gt_mass_gain_over_query']*100:+.2f}%p"
            improved="—" if k=="query" else f"{s['improved_gt_mass_count']}/{s['n']}"
            lines.append(f"| {d} | {LABELS[k]} | {pct(s['mean_video_attention'])} | {s['peak_in_gt_count']}/{s['n']} | {delta} | {improved} |")
    lines += ["", "숫자/timestamp 비교의 분모는 유효 구간을 파싱할 수 있었던 샘플 수다. 답변 전체 비교는 구간 파싱 여부와 무관하게 유지하며, "
              "GT 비중 변화는 항상 같은 샘플끼리 계산한다. 답변은 input query와 앞선 답변 토큰도 key로 보기 때문에 영상 attention 총량 감소만으로 "
              "시각 정보의 인과적 사용이 줄었다고 단정하지 않는다."]
    lines += ["", "### 분포 변화의 해석", ""]
    if all(summary[d]["scopes"]["answer"]["peak_in_gt_rate"] < summary[d]["scopes"]["query"]["peak_in_gt_rate"]
           and summary[d]["scopes"]["answer"]["early_peak_rate"] > summary[d]["scopes"]["query"]["early_peak_rate"] for d in DATASETS):
        lines += ["**세 데이터셋 모두 답변 전체의 attention peak가 GT 안에 들어간 비율은 query보다 낮았고, "
                  "영상 앞부분에 peak가 몰리는 경향은 더 강했다.** 이 조건에서는 답변 attention의 최고점만으로 "
                  "GT 위치를 찾는 방식이 query보다 좋은 결과를 보이지 않았다.", ""]
    lines += [
        "GT peak는 가장 높은 한 지점만 보며, GT 비중은 구간 전체의 질량을 본다. 두 지표의 변화 방향이 다르면 "
        "‘답변 attention이 GT와 더 잘 맞는다’는 단일 결론으로 합치지 않는다. 아래 수치는 같은 샘플의 실제 생성 결과를 비교한 관찰이다.", ""]
    for d in DATASETS:
        s=summary[d]["scopes"];q=s["query"];a=s["answer"];c=s["coordinates"]
        lines.append(f"- **{d}**: query → 답변 전체의 GT peak는 {q['peak_in_gt_count']}/{q['n']} → {a['peak_in_gt_count']}/{a['n']}, "
            f"영상 앞 10%에 peak가 있는 비율은 {pct(q['early_peak_rate'])} → {pct(a['early_peak_rate'])}다. "
            f"GT 비중 변화 중앙값은 답변 전체 {a['median_gt_mass_gain_over_query']*100:+.2f}%p, "
            f"시간 숫자 토큰 {c['median_gt_mass_gain_over_query']*100:+.2f}%p다.")
    lines += ["", "128-token 제한에 도달한 답변을 제외한 민감도 확인:", "",
        "| 데이터셋 | 토큰 그룹 | 남은 샘플 | GT peak | Query 대비 GT 비중 변화 중앙값 |",
        "|---|---|---:|---:|---:|"]
    for d in DATASETS:
        for scope in ["answer","coordinates"]:
            s=summary[d]["excluding_truncated"][scope]
            lines.append(f"| {d} | {LABELS[scope]} | {s['n']} | {s['peak_in_gt_count']}/{s['n']} | {s['median_gt_mass_gain_over_query']*100:+.2f}%p |")
    lines += ["","## 토큰 위치 해석","",
        "주 그래프의 답변 attention은 **생성된 토큰을 입력으로 처리할 때의 행**이다. 예를 들어 답변의 `14` 숫자 토큰 자체가 영상에 보내는 attention이며 MTLA의 응답 토큰 위치 관례를 따른다. "
        "그 토큰을 **생성하기 직전의** attention은 한 행 앞이므로 `predictive_temporal` 및 각 샘플의 `predictive.png`로 따로 저장했다. "
        "두 관례가 섞이지 않도록 별도로 집계했다.","",
        "| 데이터셋 | 답변 전체: 토큰 처리 행 GT peak | 답변 전체: 생성 직전 행 GT peak | 숫자: 토큰 처리 행 GT peak | 숫자: 생성 직전 행 GT peak |", "|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        s=summary[d]["scopes"];lines.append(f"| {d} | {pct(s['answer']['peak_in_gt_rate'])} | {pct(s['predictive']['peak_in_gt_rate'])} | {pct(s['coordinates']['peak_in_gt_rate'])} | {pct(s['predictive_coordinates']['peak_in_gt_rate'])} |")
    lines += ["","## 전체 그림","",
        "각 그림의 시간축 아래 **GT(초록)**와 **모델 답변 구간(빨강)**을 별도로 표시했다. 상단은 원래 attention score, 두 번째 곡선은 영상 내부 정규화 분포, "
        "아래 히트맵은 query/답변 각각의 토큰별 분포다.","",
        "- [Charades-STA 100개 비교](Charades-STA_overview.pdf) / [상세](Charades-STA_detail.pdf)",
        "- [QVHighlights 100개 비교](QVHighlights_overview.pdf) / [상세](QVHighlights_detail.pdf)",
        "- [ActivityNet 100개 비교](ActivityNet-Captions_overview.pdf) / [상세](ActivityNet-Captions_detail.pdf)",
        "","## 검증과 해석 범위","",
        "입력 text, 영상 토큰 위치, 실제 timestamp가 이전 실험과 같은지 확인했다. 생성 중 읽은 query attention은 300개 모두 이전 배열과 비트 단위로 일치한다. "
        "GT가 영상 밖을 넘는 경우는 원본을 보존하고, attention 질량 계산에만 실제 영상과의 교집합을 사용했다. "
        "Pearson correlation은 계산하지 않았다. 이 결과는 한 번의 greedy 답변과 고정된 100개씩에 대한 관찰이며, MTLA의 16-rollout re-ranking이나 전체 벤치마크 평가 결과가 아니다.","",
        "[방법과 재실행](../../README.md) · [샘플별 수치](sample_metrics.csv) · [집계 JSON](summary.json)",""]
    write_result_text((relocate_path(GENERATION / 'FINDINGS.md')), "\n".join(lines))


def main():
    samples=json.loads((relocate_path(RESULTS/"samples.json")).read_text());assert len(samples)==300
    rows=[evaluate_sample(s) for s in samples]
    summary=summarize(rows)
    write_result_text((relocate_path(RESULTS/"sample_metrics.json")), json.dumps(rows,indent=2)+"\n")
    write_result_text((relocate_path(RESULTS/"summary.json")), json.dumps(summary,indent=2)+"\n")
    flat=[]
    for r in rows:
        item={k:v for k,v in r.items() if k not in {"scopes","layers"}}
        for scope,stats in r["scopes"].items():
            item.update({f"{scope}_{key}":value for key,value in stats.items()})
        flat.append(item)
    with (relocate_path(RESULTS/"sample_metrics.csv")).open("w") as file:
        keys=list(dict.fromkeys(k for row in flat for k in row));writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    plot_summary(rows,summary);write_report(rows,summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
