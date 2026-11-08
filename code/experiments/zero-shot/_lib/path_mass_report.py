"""Reports for a three-query diagnostic; censored targets stay censored."""
import html,json,math,os
from path_mass_core import *

def iou(a,b):return max(0.,min(a[1],b[1])-max(a[0],b[0]))/max(1e-12,max(a[1],b[1])-min(a[0],b[0])) if a and b else 0.
def f(x,n=3):return '—' if x is None else f'{x:.{n}f}'
def selected(s):return max(s['intervals'],key=lambda x:x['native_mass'])['span'] if s['intervals'] else None
def metrics(values):return [sum(values)/len(values)]+[sum(v>=t for v in values)/len(values) for t in (.3,.5,.7)] if values else [None]*4

def report():
 from result_layout import relocate_path
 RESULT.mkdir(parents=True,exist_ok=True);rows=read(OUT/'metadata/cohort.json');data=[read(OUT/'queries'/f'{r["sample_key"]}.json') for r in rows];catalog=[]
 for d in data:
  row=d['sample'];key=row['sample_key'];source=read(SOURCE/'web/samples'/f'{key}.json');video=relocate_path((ROOT/'results/rollout/deterministic/pilot/900'/source['video']).resolve());catalog.append(dict(key=key,query=row['query'],dataset=row['dataset'],gt=row['span'][0],duration=d['input']['metadata_duration'],video=os.path.relpath(video,ROOT.parent)))
 write(OUT/'web/samples.json',catalog)
 lines=['# Path mass constraints · pilot3','','3개 데이터셋에서 고정 pilot300에 처음 등장하는 query를 하나씩 고른 진단이다. 일반 성능 향상을 결론내릴 표본이 아니다. 모델과 영상/query 입력은 기존 zero-shot과 동일한 Qwen3-VL-8B-Instruct·2 FPS이며 tensor hash를 대조했다. Sampling은 없다.','', '## 비교 조건과 종료 기준','','두 조건 모두 매 step의 native top-p=0.99 집합과 native 확률≥1%의 교집합을 만든다. 정수부 숫자·소수점 기호 위치에서만 대안을 추적하며 일반 문구·소수 자릿수는 greedy로 한 경로만 이어간다. 소수 자릿수에서는 branch entropy도 측정하지 않는다. 최종 출력의 소수점은 보존한다. 재정규화하지 않고 정책상 제외한 확률 질량을 따로 저장한다.','', '- **mass_only**: 출력 문법을 제한하지 않고 확률 질량으로 탐색한다.','- **format_and_mass**: 원래 prompt의 `The event happens in <start> - <end> seconds` 형식, 양수/0의 초 단위 숫자·소수점 한 자리만 허용한다. 마지막 마침표는 선택이다. 이미 start≥영상길이이거나 완성된 clipped start≥end인 prefix를 종료한다. 낮은 확률 token을 강제로 추가하지 않는다.','- 두 조건 모두 native 확률 질량이 큰 frontier prefix부터 선택하고, 그 안에서는 greedy spine을 완성한 뒤 다음 frontier를 선택한다. 공유 video/query KV를 재사용한다.','', 'V=완성된 유효 경로의 native 질량, U=미탐색 prefix의 native 질량 합+context limit 질량이다. `U/(V+U)≤ε`에서 ε=10%·5%·1% checkpoint를 기록한다. 한 번의 nested search에서 checkpoint를 얻으며 서로 독립 실행처럼 비용을 합산하지 않는다. 이 상한은 token 필터·숫자 분기 정책·해당 문법을 통과하는 최종 유효 분포에 대한 것이다. 원래 full model 분포 전체의99%를 보장하지 않는다.','', '**각 query·조건은 search wall 45초까지만 측정했다.** 시간 제한 전에 기준을 만족하지 못한 ε는 미도달이며, 완료된 ε 실험으로 계산하지 않는다. Model load·입력 준비·video prefill은45초 밖의 운영 비용이다. 미탐색 질량이 작아도 놓친 낮은 확률 후보에 높은 GT IoU가 있을 수 있다.','', '## 목표 질량까지의 비용','','아래 평균은 실제 목표에 도달한 query만 포함한다. 도달 수와 함께 읽어야 한다. Search forward는 prefill 이후 실제 모델 호출 수이며 KV rollback에 필요한 replay도 포함한다.','', '| 조건 | ε | 도달 query /3 | 평균 search forwards | 평균 search wall s | 평균 고유 구간 수 |','|---|---:|---:|---:|---:|---:|']
 for variant in CONFIG['variants']:
  for eps in CONFIG['epsilons']:
   ss=[d['variants'][variant]['snapshots'][str(eps)] for d in data if str(eps) in d['variants'][variant]['snapshots']];mean=lambda fun:sum(fun(s) for s in ss)/len(ss) if ss else None
   lines.append(f'| {variant} | {eps:.0%} | {len(ss)} /3 | {f(mean(lambda s:s["cost"]["native_forwards"]),1)} | {f(mean(lambda s:s["cost"]["wall_seconds"]))} | {f(mean(lambda s:len(s["intervals"])),1)} |')
 lines+=['','## ε 기준별 후보 품질','','해당 ε에 실제 도달한 query만의 결과다. Δpp는 그와 같은 query들의 greedy 대비 차이이며 n이 다른 행의 평균을 직접 비교하지 않는다. Oracle은 GT로 사후 선택한 후보 상한이다.','', '| 조건 | ε | 도달 n | 평가 | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |','|---|---:|---:|---|---:|---:|---:|---:|']
 for variant in CONFIG['variants']:
  for eps in CONFIG['epsilons']:
   pairs=[(d,d['variants'][variant]['snapshots'][str(eps)]) for d in data if str(eps) in d['variants'][variant]['snapshots']];base=metrics([iou(d['greedy']['span'],d['sample']['span'][0]) for d,x in pairs])
   for choice in ('native_mass','candidate_oracle'):
    values=[iou(selected(x),d['sample']['span'][0]) if choice=='native_mass' else max([iou(r['span'],d['sample']['span'][0]) for r in x['intervals']]+[0.]) for d,x in pairs];m=metrics(values)
    cells=[f'{v:.4f} ({100*(v-b):+.2f})' if v is not None else '—' for v,b in zip(m,base)];lines.append('| '+variant+f' | {eps:.0%} | {len(pairs)} | '+choice+' | '+' | '.join(cells)+' |')
 lines+=['','## 시간 제한 시점 또는1% 목표 도달 시점의 실제 탐색량','','U/(V+U)는 미탐색 유효 질량의 보수적 상한이다. 아래 interval entropy와 covariance는 탐색된 유효 구간에만 조건부인 통계이며 전체 분포의 정확한 값이 아니다. 서로 다른 sequence가 같은 clipped interval이면 질량을 합친다.','', '| Query | 조건 | 종료 사유 | Forwards | Search s | 구간 수 | 미탐색 상한 | V | 문법 제외 질량 | 정책 제외 질량 | 필터 제외 질량 |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for d in data:
  for variant in CONFIG['variants']:
   s=d['variants'][variant]['final'];lines.append(f'| {d["sample"]["sample_key"]} | {variant} | {s["reason"]} | {s["cost"]["native_forwards"]} | {f(s["cost"]["wall_seconds"])} | {len(s["intervals"])} | {f(s["unseen_valid_mass_upper_bound"])} | {f(s["valid_native_mass"])} | {f(s["grammar_rejected_mass"])} | {f(s["policy_rejected_mass"])} | {f(s["filtered_mass"])} |')
 lines+=['','## 후보 구간의 GT 평가','','IoU=예측/GT 교집합 길이÷합집합 길이,mIoU=query별 IoU 평균,R@τ=IoU≥τ 비율이다. 높을수록 좋다. 괄호 안 Δpp=100×(결과−같은 query의greedy). 최대 질량 후보는 native 경로 질량 합이 가장 큰 구간이며 verifier 선택 결과가 아니다. Oracle은 생성 후보 중 최고 GT IoU이며 실제 모델 성능이 아니다. GT는 탐색 이후 평가에만 사용한다.','', '| 조건 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |','|---|---:|---:|---:|---:|---:|']
 baseline=metrics([iou(d['greedy']['span'],d['sample']['span'][0]) for d in data])
 methods=[('greedy',None,None)]+[(v+' '+choice,v,choice) for v in CONFIG['variants'] for choice in ('native_mass','candidate_oracle')]
 for name,variant,choice in methods:
  values=[]
  for d in data:
   gt=d['sample']['span'][0]
   if variant is None:value=iou(d['greedy']['span'],gt)
   else:
    s=d['variants'][variant]['final'];value=iou(selected(s),gt) if choice=='native_mass' else max([iou(x['span'],gt) for x in s['intervals']]+[0.])
   values.append(value)
  m=metrics(values);lines.append('| '+name+' | 3 | '+' | '.join(f'{v:.4f} ({100*(v-b):+.2f})' for v,b in zip(m,baseline))+' |')
 lines+=['','## Query별 길이와 결과','','n=1은 모두 작은 표본이다. 아래 개별 행은 dataset·영상 길이·GT 비율을 함께 표시하며 세 query를 benchmark 평균으로 해석하지 않는다.','', '| Query | Dataset | 영상 s | GT/영상 길이 | Greedy IoU | Mass-only 최고 질량 / oracle IoU | Format 최고 질량 / oracle IoU |','|---|---|---:|---:|---:|---:|---:|']
 for d in data:
  r=d['sample'];gt=r['span'][0];cells=[]
  for v in CONFIG['variants']:
   s=d['variants'][v]['final'];cells.append(f'{iou(selected(s),gt):.4f} / {max([iou(x["span"],gt) for x in s["intervals"]]+[0.]):.4f}')
  lines.append(f'| {r["sample_key"]} | {r["dataset"]} | {d["input"]["metadata_duration"]:.2f} | {(gt[1]-gt[0])/r["duration"]:.4f} | {iou(d["greedy"]["span"],gt):.4f} | '+ ' | '.join(cells)+' |')
 lines+=['','## Greedy 대비 실측 연산 오버헤드','','Native decoder inputs는 실제 모델에 넣은 token position 수다. Prefill token과 decode token은 attention 비용이 다르므로 같은 FLOPs로 해석하지 않는다. CUDA stream seconds는 GPU kernel active time과 같지 않다. 공유 prefill은 각 방법의 단독 비교 비용에 한 번씩 귀속하고 합산하지 않는다.','', '| Query | 조건 | search forwards / greedy decode forwards | search decoder inputs / greedy decode inputs | prefill 포함 inputs 배수 | CUDA stream s |','|---|---|---:|---:|---:|---:|']
 for d in data:
  for variant in CONFIG['variants']:
   s=d['variants'][variant]['final'];a=s['cost'];b=d['greedy']['cost'];n=d['input']['input_tokens'];lines.append(f'| {d["sample"]["sample_key"]} | {variant} | {a["native_forwards"]/max(1,b["native_forwards"]):.2f} | {a["decoder_inputs"]/max(1,b["decoder_inputs"]):.2f} | {(n+a["decoder_inputs"])/(n+b["decoder_inputs"]):.3f} | {a["cuda_stream_seconds"]:.3f} |')
 lines+=['',f'운영 실측: model load {sum(d["model_load_wall_seconds"] for d in data):.3f}s; 입력 준비 {sum(d["input"]["input_prepare_wall_seconds"] for d in data):.3f}s; 공유 video/query prefill {sum(d["input"]["prefill_wall_seconds"] for d in data):.3f}s; peak allocated {max(d["peak_allocated_bytes"] for d in data)/2**30:.2f}GiB.','', '## 해석의 범위','','문법 제약은 허용 분포 자체를 바꾸며 분포 밖 질량을 따로 보고한다. ε checkpoint는 해당 허용 분포의 탐색 충실도를 통제한다.45초 결과끼리 비교한 차이에는 문법과 탐색 순서의 영향이 함께 있을 수 있다. 제한 시간 내 기준에 도달하지 못했다는 사실도 결과다. Semantic embedding·독립 재시청·final answer 재생성은 이번 짧은 비교의 측정 범위가 아니다.']
 lines+=['','## 영상 길이 × GT 비율 교차 진단','','각 칸은 n; greedy / mass_only 최대 질량 / format_and_mass 최대 질량 순서다. 조건별 진단 종료 시점의 값을 사용한다. n=1은 작은 표본이며 빈 칸은 —다. 모든 query의 개별 값은 위 표에 있다.']
 duration_edges=[0,30,60,120,180,math.inf];fraction_edges=[0,.05,.1,.2,.4,math.inf]
 def label(edges,i):return f'[{edges[i]:g}, {edges[i+1]:g})' if math.isfinite(edges[i+1]) else f'≥{edges[i]:g}'
 def bucket(v,edges):return next(i for i in range(5) if edges[i]<=v<edges[i+1])
 for k,name in enumerate(('mIoU','R@.3','R@.5','R@.7')):
  lines+=['',f'### {name}','','| 영상 길이 s | '+' | '.join(label(fraction_edges,i) for i in range(5))+' |','|---|'+'---|'*5]
  for dur in range(5):
   cells=[]
   for frac in range(5):
    subset=[d for d in data if bucket(d['sample']['duration'],duration_edges)==dur and bucket((d['sample']['span'][0][1]-d['sample']['span'][0][0])/d['sample']['duration'],fraction_edges)==frac]
    values=[]
    if subset:
     for method in ('greedy',*CONFIG['variants']):
      vals=[iou(d['greedy']['span'] if method=='greedy' else selected(d['variants'][method]['final']),d['sample']['span'][0]) for d in subset];values.append(metrics(vals)[k])
    cells.append(f'n={len(subset)}'+(' 작은 표본; '+' / '.join(f'{v:.3f}' for v in values) if subset else '; —'))
   lines.append('| '+label(duration_edges,dur)+' | '+' | '.join(cells)+' |')
 (RESULT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 raw=os.path.relpath(OUT,RESULT);root=os.path.relpath(ROOT.parent,RESULT);theme=os.path.relpath(ROOT.parent/'_lib/experiment_theme.css',RESULT);js=os.path.relpath(OUT/'web/viewer.js',RESULT)
 (RESULT/'detail.html').write_text(f'''<!doctype html><html lang="ko" data-family-default><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Path mass constraints · pilot3</title><link rel="stylesheet" href="{theme}"><style>body{{padding:16px;margin:0}}video{{width:100%;max-height:320px}}.box{{border:1px solid #cbd8e3;border-radius:8px;padding:12px;margin:12px 0}}.rows{{overflow:auto}}select{{max-width:100%}}</style></head><body><h1>Path mass constraints · pilot3</h1><p>각 데이터셋1 query · 조건별45초 진단 · 미도달 기준은 미완료로 표시</p><select id="sample" aria-label="query"></select><select id="variant" aria-label="조건"><option>mass_only</option><option>format_and_mass</option></select><select id="checkpoint" aria-label="checkpoint"><option value="final">진단 종료 시점</option><option value="0.1">미탐색10%</option><option value="0.05">미탐색5%</option><option value="0.01">미탐색1%</option></select><main id="content"></main><script>window.DIAG={{store:{json.dumps(raw)},root:{json.dumps(root)}}};</script><script src="{js}"></script></body></html>''')
 print('REPORT_READY',RESULT,flush=True)

if __name__=='__main__':report()
