"""Validate and report per-query time/compute for all three 16fps methods."""
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from benchmark_16fps import ROOT,RUN,OUT,VIEW,MANIFEST,FPS,MODELS,TIMELENS,key,write,stamp
from metrics import aggregate,metrics,labels
from adjacent_similarity import attach as attach_adjacent,markdown as adjacent_markdown
sys.path.insert(0,str(TIMELENS))

def mean(rr,k):return float(np.mean([r[k] for r in rr]))
def stats(rr):
    return dict(n=len(rr),mean_gpu_seconds=mean(rr,'gpu_forward_wall_seconds'),
        median_gpu_seconds=float(np.median([r['gpu_forward_wall_seconds'] for r in rr])),
        p95_gpu_seconds=float(np.percentile([r['gpu_forward_wall_seconds'] for r in rr],95)),
        mean_cuda_event_seconds=mean(rr,'cuda_event_seconds'),
        mean_cached_wall_seconds=mean(rr,'cached_frame_wall_seconds'),
        mean_cpu_work_seconds=mean(rr,'cpu_input_work_seconds'),mean_h2d_seconds=mean(rr,'h2d_seconds'),
        mean_shared_source_prepare_seconds=mean(rr,'shared_source_preparation_seconds'),
        mean_tflops=float(np.mean([r['flops']['total_flops']/1e12 for r in rr])),
        mean_frames=mean(rr,'frame_count'),mean_input_tokens=mean(rr,'input_tokens'),
        mean_generated_tokens=mean(rr,'generated_tokens'),
        mean_peak_gpu_gib=float(np.mean([r['peak_gpu_allocated_bytes']/1024**3 for r in rr])))

def main():
    from timelens.utils import extract_time,iou
    rows=json.loads((OUT/'config/samples.json').read_text());assert rows==json.loads(MANIFEST.read_text()) and len(rows)==300
    records=[];lookup={};payload=[]
    for s in rows:
        meta=json.loads((OUT/'inputs/frames'/key(s['video_path'])/'metadata.json').read_text())
        source_fps=meta['source_fps'];indices=np.unique(np.rint(np.arange(0,meta['source_frames'],source_fps/16)).astype(int))
        indices=indices[indices<meta['source_frames']]
        np.testing.assert_array_equal(indices,meta['frame_indices'])
        item={k:s[k] for k in ['sample_key','dataset','query','duration','span']}
        item.update(video_key=key(s['video_path']),timestamps=meta['timestamps'],
            video_url=f'../../artifacts/runs/{RUN}/media/{key(s["video_path"])}.mp4',models={})
        for model in ['clip','qwen','instruct']:
            r=json.loads((OUT/'predictions'/model/f'{s["sample_key"]}.json').read_text())
            assert r['sample_key']==s['sample_key'] and r['query']==s['query'] and r['dataset']==s['dataset']
            assert r['frame_count']==meta['count'] and r['fps']==16
            for k in ['gpu_forward_wall_seconds','cuda_event_seconds','cached_frame_wall_seconds','cpu_input_work_seconds','h2d_seconds']:
                assert math.isfinite(r[k]) and r[k]>=0,(s['sample_key'],model,k)
            assert r['cached_frame_wall_seconds']>=r['gpu_forward_wall_seconds']
            assert r['flops']['total_flops']==sum(r['flops'][k] for k in ['vision_flops','text_flops','cosine_flops'])
            assert r['flops']['total_flops']>0 and not r['other_gpu_pids']
            if model!='instruct':
                with np.load(OUT/'scores'/model/f'{s["sample_key"]}.npz') as a:
                    np.testing.assert_array_equal(a['timestamps'],meta['timestamps'])
                    computed=metrics(a['scores'],labels(a['timestamps'],s['span']))
                    assert computed==r['similarity']
                    item['models'][model]=dict(scores=a['scores'].tolist(),metrics=dict(r['similarity'],model=model))
                    adj,stats_adj=attach_adjacent(OUT,model,item['video_key'],s['sample_key'],meta['timestamps'])
                    item['models'][model]['adjacent_scores']=adj
                    item['models'][model]['metrics']['adjacent_similarity']=stats_adj
                    r['adjacent_similarity']=stats_adj
            else:
                assert r['input']['frame_count']==meta['count'] and not r['input']['sampling_capped']
                assert r['input']['padded_frames']==2*math.ceil(meta['count']/2)
                assert r['input']['video_grid_thw'][0]*2==r['input']['padded_frames']
                assert r['input']['timestamps']==meta['timestamps']
                spans=extract_time(r['answer']) or []
                assert spans==r['predicted_spans'] or json.loads(json.dumps(spans))==r['predicted_spans']
                assert abs(r['iou']-(iou(s['span'][0],spans[0]) if spans else 0.))<1e-12
            records.append(r);lookup[model,s['sample_key']]=r
        payload.append(item)
    groups={}
    for ds in ['all']+sorted({s['dataset'] for s in rows}):
        groups[ds]={m:stats([r for r in records if r['model']==m and (ds=='all' or r['dataset']==ds)]) for m in ['clip','qwen','instruct']}
    similarity={ds:{m:aggregate([r['similarity'] for r in records if r['model']==m and (ds=='all' or r['dataset']==ds)]) for m in ['clip','qwen']} for ds in groups}
    generation={ds:dict(n=len(rr),miou=mean(rr,'iou'),r05=float(np.mean([r['iou']>=.5 for r in rr])),
                        parse_failures=sum(not r['predicted_spans'] for r in rr),truncated=sum(not r['eos_reached'] for r in rr))
                for ds in groups for rr in [[r for r in records if r['model']=='instruct' and (ds=='all' or r['dataset']==ds)]]}
    comparisons={}
    for m in ['clip','qwen']:
        pairs=[(lookup['instruct',s['sample_key']],lookup[m,s['sample_key']]) for s in rows]
        comparisons[m]=dict(mean_paired_gpu_ratio=float(np.mean([a['gpu_forward_wall_seconds']/b['gpu_forward_wall_seconds'] for a,b in pairs])),
            ratio_of_mean_gpu=groups['all']['instruct']['mean_gpu_seconds']/groups['all'][m]['mean_gpu_seconds'],
            mean_paired_gpu_difference=float(np.mean([a['gpu_forward_wall_seconds']-b['gpu_forward_wall_seconds'] for a,b in pairs])),
            mean_paired_wall_difference=float(np.mean([a['cached_frame_wall_seconds']-b['cached_frame_wall_seconds'] for a,b in pairs])),
            ratio_of_mean_flops=groups['all']['instruct']['mean_tflops']/groups['all'][m]['mean_tflops'])
    write(OUT/'measurements/summary.json',dict(groups=groups,similarity=similarity,generation=generation,comparisons=comparisons,completed_at=stamp()))
    write(OUT/'measurements/validation.json',dict(passed=True,samples=300,records=900,
        checks=['exact pilot membership','16fps source-index sampling, no frame cap',
            'matched frames across three methods; Instruct temporal padding only',
            'positive finite measured timings','no competing GPU processes',
            'component FLOPs sum','independent similarity metric and generation IoU recomputation']))
    with (OUT/'measurements/per_sample.csv').open('w') as f:
        columns=['model','dataset','sample_key','frame_count','input_tokens','generated_tokens',
            'gpu_forward_wall_seconds','cuda_event_seconds','cached_frame_wall_seconds','cpu_input_work_seconds',
            'h2d_seconds','shared_source_preparation_seconds','peak_gpu_allocated_bytes','total_flops']
        columns+=['adjacent_pairs','adjacent_mean_cosine','adjacent_min_cosine']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in records:
            row={k:r['flops']['total_flops'] if k=='total_flops' else r[k] for k in columns if not k.startswith('adjacent_')}
            a=r.get('adjacent_similarity',{})
            row.update(adjacent_pairs=a.get('n_pairs',''),adjacent_mean_cosine=a.get('mean_cosine',''),adjacent_min_cosine=a.get('min_cosine',''))
            w.writerow(row)
    VIEW.mkdir(parents=True,exist_ok=True)
    base=f'../../artifacts/runs/{RUN}'
    lines=['# Pilot 300 · 16fps 유사도와 Instruct 생성 비용 비교','',f'완료: {stamp()}',
        '',f'[샘플별 HTML](report.html) · [샘플별 비용 CSV]({base}/measurements/per_sample.csv) · [요약 JSON]({base}/measurements/summary.json) · [검증]({base}/measurements/validation.json)',
        '', '## 비교 조건','',
        '- 고정 pilot 300쌍: dataset별 100개, 고유 영상 292개. 세 모델은 같은 source frame index와 실제 PTS를 사용한다.',
        '- 16fps 목표로 nearest source index를 선택하며 중복 index는 제거한다. 원본 fps가 16보다 낮으면 실제 unique frame rate는 그보다 낮다. 영상 전체를 사용하며 768-frame 상한이나 GT 기반 sampling은 없다.',
        '- CLIP ViT-L/14 FP16, Qwen3-VL-Embedding-8B BF16, Qwen3-VL-8B-Instruct BF16. SDPA. 모델은 순차 실행하고 sample 간 작업을 겹치지 않는다. Similarity 내부 frame batch만 CPU prefetch한다. GPU 경쟁 작업이 감지되면 중단한다.',
        '- 유사도: 독립 frame/query embedding을 FP32 L2 정규화하고 cosine을 계산한다. CLIP batch256, Qwen batch128; 공통 JPEG 최대 변512, Qwen max_pixels262144.',
        '- Instruct: video→query 입력, TimeLens grounding prompt, greedy, KV cache, 최대512 output tokens, 마지막 위치의 vocabulary logits만 계산. 같은16fps frame을 video temporal patch2로 묶고 마지막 홀수 frame만 복제한다.',
        '- Instruct의 기존 min_tokens64·total_tokens14336 전처리 정책을 유지한다. total_tokens는 soft spatial budget이다. min_pixels가 우선하므로 긴 영상의 실제 visual token 수는14336을 넘을 수 있다. 실제 해상도·frame index·visual/input token 수를 샘플별 prediction에 보관했다.',
        '- 네이티브 전처리와 시간 결합 구조가 다르므로 같은 frame 수가 같은 model token 수나 공간 해상도를 뜻하지 않는다. 이 실험은 각 파이프라인의 비용 비교다.',
        '', '## 샘플당 시간과 연산량','',
        '| Dataset | Model | N | Mean GPU forward s | Median s | P95 s | Cached-frame wall s | Estimated TFLOPs/sample | Mean frames | Mean input tokens | Mean output tokens |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds,models in groups.items():
        for m,g in models.items():lines.append(f"| {ds} | {m} | {g['n']} | {g['mean_gpu_seconds']:.4f} | {g['median_gpu_seconds']:.4f} | {g['p95_gpu_seconds']:.4f} | {g['mean_cached_wall_seconds']:.4f} | {g['mean_tflops']:.3f} | {g['mean_frames']:.1f} | {g['mean_input_tokens']:.1f} | {g['mean_generated_tokens']:.1f} |")
    lines+=['','### 실제 실행 구간의 전체 경과 시간','',
        '아래에는 모델별 마지막 실행 구간의 파일 저장·진행 기록을 포함한 wall time도 제시한다. Loading과 warmup은 별도다. 재개한 경우 측정 sample 수를 명시하므로 전체 300개 측정 시간과 혼동하지 않는다.',
        '', '| Model | Measured samples | Stage wall seconds | Stage wall / sample | Model loading seconds |',
        '|---|---:|---:|---:|---:|']
    for model in ['clip','qwen','instruct']:
        stage=json.loads((OUT/'measurements'/f'{model}_stage.json').read_text())
        lines.append(f"| {model} | {stage['measured_samples']} | {stage['measured_stage_wall_seconds']:.3f} | {stage['measured_stage_wall_seconds']/stage['measured_samples']:.3f} | {stage['load_seconds']:.3f} |")
    lines+=['','## Instruct 대비 유사도 계산의 시간 차이','',
        '동일 query 쌍에서 `Instruct − similarity`를 계산했다. 양수면 생성이 더 오래 걸린다. 비율은 `Instruct / similarity`다.',
        '', '| Similarity model | Mean paired GPU Δ seconds | Mean paired wall Δ seconds | Mean paired GPU ratio | Ratio of mean GPU times | Ratio of mean FLOPs |',
        '|---|---:|---:|---:|---:|---:|']
    for m,c in comparisons.items():lines.append(f"| {m} | {c['mean_paired_gpu_difference']:.4f} | {c['mean_paired_wall_difference']:.4f} | {c['mean_paired_gpu_ratio']:.3f} | {c['ratio_of_mean_gpu']:.3f} | {c['ratio_of_mean_flops']:.3f} |")
    lines+=['','## 측정 정의','',
        '- GPU forward seconds: CUDA synchronize 전후 perf_counter로 재는 model 호출 wall time. Similarity는 query와 모든 frame batch 호출의 합, generation은 prefill+전체 autoregressive decode다. H2D·CPU 전처리·cosine·파일 저장은 제외한다. 모델 loading과 warmup도 제외한다.',
        '- CUDA event seconds도 원자료에 별도 기록했다. Generation은 첫 forward의 prefill과 이후 decode 시간을 구분했다.',
        '- Cached-frame wall: 공통 JPEG 읽기부터 전처리·H2D·모델·cosine 또는 답변 decode까지 실제 경과 시간. 결과 파일 쓰기는 제외한다. Similarity에서 CPU batch 준비와 GPU 계산이 겹치므로 CPU work와 GPU 시간을 단순 합산하면 안 된다.',
        '- Raw 영상 decode·resize·JPEG 저장은 공통 준비 단계에서 영상당 한 번 측정하고 각 query에 해당 video의 값을 연결했다. 병렬 준비에서 측정한 비용이며 cached-frame wall과 별개다. Literal raw-video end-to-end latency로 오인하지 않는다.',
        '- 샘플당 입력 비용은 캐시 적중을 가정하지 않는다. 같은 영상에 다른 query가 있어도 매 query에서 frame embedding을 새로 계산했다. embedding 파일만 재사용 가능하게 영상당 한 벌 저장했다.',
        '', '## FLOPs 정의와 계산식','',
        '- 추정 forward matmul/conv FLOPs다. Multiply-add=2 FLOPs, 1 TFLOP=10^12. 장치의 hardware counter로 측정한 명령 수가 아니다. Activation·normalization·softmax·embedding lookup·memory 이동·전처리 FLOPs는 제외했다.',
        '- Linear(in,out,T)=2×T×in×out. CLIP transformer는 attention projection과 2-layer MLP, Qwen은 GQA projection과 3-projection gated MLP를 실제 config로 계산했다.',
        '- Attention=4×hidden_size×query-key pair 수×layer 수. Language causal prefill은 S(S+1)/2 pair, cached decode j번째는 S+j pair다. Vision은 frame/temporal-patch별 full attention pair를 사용한다. 실제 fused kernel의 tile·padding 실행량은 다를 수 있다.',
        '- Vision patch convolution, spatial merger, Qwen deepstack merger 3개, CLIP projection, Instruct 실제 생성 token별 LM head, similarity cosine 내적을 포함했다. Generation은 S개 prefill token과 G−1개 cached decode token을 계산했다.',
        '- Similarity input tokens는 독립된 query/frame 시퀀스의 합이다. 하나의 긴 시퀀스 길이가 아니다. Generation input tokens는 단일 video–query 시퀀스 길이다.',
        '', '## 정량 성능','', '| Dataset | Model | mAP | MRR | Median first GT rank |','|---|---|---:|---:|---:|']
    for ds,models in similarity.items():
        for m,g in models.items():lines.append(f"| {ds} | {m} | {g['map']:.4f} | {g['mrr']:.4f} | {g['median_first_gt_rank']:.1f} |")
    lines+=['','AP는 GT frame을 positive로 한 query 내부 retrieval AP, mAP는 query별 동일 가중 평균이다. First GT rank는 첫 positive의 1-based 순위(낮을수록 좋음), MRR은 1/rank의 평균(높을수록 좋음)이다. GT는 PTS 기준 [start,end) 합집합이다. GT mass=Σ_GT softmax(cosine/τ), uniform=N_GT/N, lift=mass/uniform이며 τ=.01/.03/.07/.1/.3/1.0을 저장한다.',
        '', '| Dataset | Instruct mIoU | R@1 IoU≥.5 | Parse failures | Truncated |','|---|---:|---:|---:|---:|']
    for ds,g in generation.items():lines.append(f"| {ds} | {g['miou']:.4f} | {g['r05']:.4f} | {g['parse_failures']} | {g['truncated']} |")
    (VIEW/'REPORT.md').write_text('\n'.join(lines)+'\n\n'+adjacent_markdown([r for r in records if r['model']!='instruct']))
    # Reuse the validated interactive viewer, with this run's dense 16fps curves.
    media=OUT/'media';media.mkdir(exist_ok=True)
    pilot=ROOT/'artifacts/runs/pilot_300_v1/media'
    for item in payload:
        path=media/f'{item["video_key"]}.mp4'
        if not path.exists():os.link(pilot/path.name,path)
    template=(ROOT/'_lib/viewer.html').read_text().replace('Pilot 300','Pilot 300 · 16fps').replace('· 2fps','· 16fps')
    template=template.replace('artifacts/frames/',f'{base}/inputs/frames/').replace('artifacts/media/',f'{base}/media/')
    template=template.replace('artifacts/sample_metrics.csv',f'{base}/measurements/per_sample.csv').replace('artifacts/validation.json',f'{base}/measurements/validation.json')
    data=dict(samples=payload,summary=dict(groups=similarity,temperatures=[.01,.03,.07,.1,.3,1.0]))
    blob=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    (VIEW/'report.html').write_text(template.replace('__DATA__',blob))
    os.environ['SIMILARITY_RUN']=RUN
    import check_viewer
    check_viewer.VIEW=VIEW
    check_viewer.OUT=OUT/'measurements'
    check_viewer.main()
    print('16FPS REPORT AND VALIDATION COMPLETE: 300 samples × 3 models',flush=True)

if __name__=='__main__':main()
