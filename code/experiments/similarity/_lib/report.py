from result_layout import relocate_path, write_result_text
"""Generate self-contained interactive HTML and quantitative Markdown reports."""
from collections import Counter
import csv
import os
import hashlib
import html
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from prepare import ROOT, OUT, VIEW, video_key, write
from metrics import TEMPERATURES, aggregate, labels, metrics
from prepare_media import prepare_media
from adjacent_similarity import attach as attach_adjacent,markdown as adjacent_markdown

def fmt(x,digits=4): return '—' if x is None else f'{x:.{digits}f}'

def main():
    rows=json.loads((OUT/'samples.json').read_text())
    assert rows
    count=len(rows)
    cohort='pilot 300' if count==300 else f'full {count:,}'
    dataset_counts=dict(Counter(s['dataset'] for s in rows))
    media=prepare_media(rows)
    records=[]; payload=[]
    for s in rows:
        m=json.loads((OUT/'frames'/video_key(s['video_path'])/'metadata.json').read_text())
        item={k:s[k] for k in ['sample_key','dataset','query','duration','span']}
        item.update(video_key=video_key(s['video_path']),video_url=media[s['video_path']]['media'],timestamps=m['timestamps'],models={})
        for model in ['clip','qwen']:
            a=np.load(OUT/model/'scores'/f'{s["sample_key"]}.npz')
            np.testing.assert_array_equal(a['timestamps'],m['timestamps'])
            y=labels(a['timestamps'],s['span'])
            r=dict(model=model,sample_key=s['sample_key'],dataset=s['dataset'],**metrics(a['scores'],y))
            records.append(r)
            item['models'][model]=dict(scores=a['scores'].tolist(),metrics=r)
            adj,stats=attach_adjacent(OUT,model,item['video_key'],s['sample_key'],m['timestamps'])
            item['models'][model]['adjacent_scores']=adj
            r['adjacent_similarity']=stats
        payload.append(item)
    groups={}
    for ds in ['all']+sorted({r['dataset'] for r in records}):
        groups[ds]={model:aggregate([r for r in records if r['model']==model and (ds=='all' or r['dataset']==ds)]) for model in ['clip','qwen']}
    bins={}
    for name,lo,hi in [('GT<10%',0,.1),('10%<=GT<30%',.1,.3),('GT>=30%',.3,1.00001)]:
        bins[name]={model:aggregate([r for r in records if r['model']==model and lo<=r['gt_fraction']<hi]) for model in ['clip','qwen']}
    summary=dict(groups=groups,gt_fraction_bins=bins,temperatures=TEMPERATURES,primary_temperature=.07,
                 created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),complete_samples=len(rows))
    write(OUT/'summary.json',summary)
    write(OUT/'sample_metrics.json',records)
    with (OUT/'sample_metrics.csv').open('w') as f:
        columns=['model','dataset','sample_key','n_frames','n_gt','gt_fraction','ap','first_gt_rank','reciprocal_rank','first_gt_percentile']
        columns+=['adjacent_pairs','adjacent_mean_cosine','adjacent_min_cosine']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in records:
            row={k:r[k] for k in columns if not k.startswith('adjacent_')}
            row.update(adjacent_pairs=r['adjacent_similarity']['n_pairs'],adjacent_mean_cosine=r['adjacent_similarity']['mean_cosine'],adjacent_min_cosine=r['adjacent_similarity']['min_cosine'])
            writer.writerow(row)
    config={m:json.loads((OUT/m/'config.json').read_text()) for m in ['clip','qwen']}
    download=json.loads(__import__('pathlib').Path('/home/irteam/backbones/manifests/similarity_download.json').read_text())
    lines=[f'# Frame–query similarity: {cohort}','',f'완료: {summary["created_at"]}',
        '', '[샘플별 interactive HTML](report.html) · [전체 샘플 지표](artifacts/sample_metrics.json) · [CSV](artifacts/sample_metrics.csv) · [요약 JSON](artifacts/summary.json)',
        '', '## 설정', '',
        f'- 고정 manifest의 전체 {count:,}쌍을 사용했다. Dataset별 개수: {dataset_counts}. 재추출하지 않았다.',
        '- GT와 무관한 2fps sampling. 원본 frame index를 round(k × source_fps / 2)로 선택하고 실제 frame PTS 시작 시간을 저장했다. 중복 index는 제거했다.',
        '- 동일한 RGB JPEG 입력(최대 변 512px, quality=95, chroma subsampling 없음)을 두 모델에 제공했다. 각 모델 고유 resize/crop은 유지한다.',
        '- CLIP: openai/clip-vit-large-patch14, FP16 inference. 공식 projected image/text features, raw query, 최대 77 text tokens.',
        '- Qwen: Qwen/Qwen3-VL-Embedding-8B, BF16 inference, SDPA. 공식 chat template와 기본 instruction `Represent the user\'s input.` 사용. 마지막 유효 token의 final hidden state를 추출한다. 이미지 max_pixels=262144, min_pixels=4096, text truncation 비활성화.',
        '- Frame과 query는 독립적으로 인코딩한다. 전체 영상이나 frame+query 공동 입력, 생성, task fine-tuning은 사용하지 않는다.',
        '- 두 모델 모두 FP32로 변환해 L2 정규화하고 내적으로 cosine을 계산한다. CLIP learned logit scale은 적용하지 않는다.',
        '- 원본 frame, query/video embedding, FP32 raw score, timestamp, config, 모델 revision과 weight SHA256을 보관했다.',
        '', '| Backbone | Revision | Local path |','|---|---|---|']
    for model in download['models']:lines.append(f"| {model['repo_id']} | `{model['revision']}` | `{model['path']}` |")
    lines+=['','## 지표 정의','',
        '- Positive: frame PTS가 GT 구간 합집합 `[start,end)` 안에 있음. AP는 query 내부 frame 순위를 평가하고 sklearn average_precision_score로 계산한다. PR curve의 사다리꼴 면적과 다르다. 정확한 score tie는 AP에서 같은 threshold로 묶는다.',
        '- mAP: video–query 쌍별 AP의 동일 가중 평균. Micro pooling하지 않는다.',
        '- First GT rank: 내림차순 순위 중 첫 positive 위치(1부터). 낮을수록 좋다. 정확한 score tie에서는 timestamp 순서를 유지한다.',
        '- MRR (Mean Reciprocal Rank): 쿼리별 첫 GT frame 순위 r_q의 역수를 동일 가중 평균한 값, MRR = (1/Q) × Σ_q (1/r_q). Q는 GT sampled frame이 있는 유효 video–query 쌍 수다. 높을수록 좋고, 모든 쿼리의 1위 frame이 GT 안이면 1이다. 예를 들어 첫 GT 순위가 1·2·10위이면 MRR = (1 + 1/2 + 1/10)/3 ≈ 0.533이다. 상위 순위 차이에 민감하며, GT frame 전체의 순위를 평가하는 AP와 달리 GT frame 하나를 얼마나 빨리 찾는지 평가한다.',
        '- First GT percentile: (rank−1)/(N−1), 낮을수록 좋다. 서로 다른 frame 수의 영향을 확인하는 보조값이다.',
        '- p_i = softmax(cosine_i / τ). GT mass = Σ_GT p_i. Uniform baseline = N_GT/N. Lift = mass/baseline. 보고되는 mean lift는 샘플별 lift의 평균이며 평균 mass를 평균 baseline으로 나눈 값이 아니다.',
        '- GT sampled frame이 0이면 AP/rank/lift는 정의되지 않아 집계에서 제외하고 개수를 명시한다. GT 질량 자체는 0으로 저장한다. 전체 manifest를 확인한다.',
        '', '## 순위 성능','', '| Dataset | Model | N / valid | mAP ↑ | MRR ↑ | Median first GT rank ↓ | Mean first GT rank ↓ | Mean first GT percentile ↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds,models in groups.items():
        for model,g in models.items():lines.append(f"| {ds} | {model} | {g['n']} / {g['n_valid']} | {fmt(g['map'])} | {fmt(g['mrr'])} | {fmt(g['median_first_gt_rank'],1)} | {fmt(g['mean_first_gt_rank'],2)} | {fmt(g['mean_first_gt_percentile'])} |")
    lines+=['','## GT mass·uniform baseline·lift','', 'Temperature는 calibration된 확률이 아니다. 같은 τ라도 모델별 score scale 차이가 남는다. 주 지표는 순위 기반 mAP이며, 아래 τ=.07은 시각화 기본값으로 사전 고정했다.','',
        '| Dataset | Model | Mean GT mass | Mean uniform baseline | Mean lift |','|---|---|---:|---:|---:|']
    for ds,models in groups.items():
        for model,g in models.items():
            v=g['mass']['0.07'];lines.append(f"| {ds} | {model} | {fmt(v['gt_mass'])} | {fmt(v['uniform_baseline'])} | {fmt(v['lift'])} |")
    lines+=['',f'### Temperature sensitivity (all {count:,})','', '| τ | Model | Mean GT mass | Mean uniform baseline | Mean lift |','|---:|---|---:|---:|---:|']
    for tau in TEMPERATURES:
        for model,g in groups['all'].items():
            v=g['mass'][str(tau)];lines.append(f"| {tau} | {model} | {fmt(v['gt_mass'])} | {fmt(v['uniform_baseline'])} | {fmt(v['lift'])} |")
    lines+=['','## GT 비율별 순위 성능','', '| GT fraction | Model | N / valid | mAP | MRR |','|---|---|---:|---:|---:|']
    for name,models in bins.items():
        for model,g in models.items():lines.append(f"| {name} | {model} | {g['n']} / {g['n_valid']} | {fmt(g['map'])} | {fmt(g['mrr'])} |")
    lines+=['','## 검증 및 한계','',
        f'- [검증 결과](artifacts/validation.json): {count:,}쌍 × 두 모델의 timestamps·score·embedding norm·독립 metric 재계산과 mass 합계 확인.',
        '- 이미지 한 장씩의 retrieval 평가로, 동작 순서나 시간적 맥락을 이용하는 video encoder 성능은 측정하지 않는다.',
        '- 긴 GT는 첫 positive를 찾기 쉽고 AP baseline도 높다. Macro 평균이나 lift가 GT 길이 효과를 완전히 제거하지는 않는다.',
        '- CLIP과 Qwen은 모델 크기, 학습 데이터, 이미지 전처리와 text template가 다르다. 결과는 이 checkpoint와 입력 설정 사이의 비교다.',
        '- Softmax는 주어진 영상 안에서만 정규화되며 query relevance가 없는 영상에서도 합이 1이다. GT mass를 절대적인 정답 확률로 해석하지 않는다.',
        '', '## 재실행','', '공유 Python: `/home/irteam/experiments/_shared/timelens/.venv/bin/python`', '', '```bash',
        'cd /home/irteam/experiments/similarity',
        f'export SIMILARITY_RUN={VIEW.name}',
        f'export SIMILARITY_MANIFEST={__import__("prepare").MANIFEST}',
        'PY=/home/irteam/experiments/_shared/timelens/.venv/bin/python',
        '$PY _lib/download_models.py', '$PY _lib/prepare.py',
        '$PY _lib/infer.py --model clip --batch-size 256', '$PY _lib/infer.py --model qwen --batch-size 128',
        '$PY _lib/report.py', '$PY _lib/validate.py', '```','',
        'Embedding 캐시가 있으면 재사용한다. 입력·설정을 바꾸는 실험은 별도 output version으로 분리해야 한다.']
    document='\n'.join(lines)+'\n'
    document+='\n'+adjacent_markdown(records)
    write_result_text(VIEW/'REPORT.md', document)
    template=(ROOT/'_lib/viewer.html').read_text().replace('Pilot 300',cohort.title())
    template=template.replace('artifacts/',os.path.relpath(OUT,VIEW)+'/')
    blob=json.dumps(dict(samples=payload,summary=summary),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    write_result_text(VIEW/'report.html', template.replace('__DATA__',blob))
    print('REPORT COMPLETE',len(records),'records')

if __name__=='__main__':main()
