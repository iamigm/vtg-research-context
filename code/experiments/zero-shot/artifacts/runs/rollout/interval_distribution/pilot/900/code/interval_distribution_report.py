"""Cumulative reference reports and query inspection, with no raw response copies."""
import json,math,os,sqlite3
from pathlib import Path
from interval_distribution_core import *

def iou(a,b):return max(0.,min(a[1],b[1])-max(a[0],b[0]))/max(1e-12,max(a[1],b[1])-min(a[0],b[0])) if a and b else 0.
def metric(values):return dict(miou=sum(values)/len(values) if values else None,**{f'r{t}':sum(v>=t/10 for v in values)/len(values) if values else None for t in (3,5,7)})
def cell(v):return '—' if v is None else f'{v:.4f}'
def method_iou(row,q,v,method):
 gt=row['span'][0]
 if method=='zero_shot':return iou(v['baseline'],gt)
 if method=='native_mass':return iou(max(q['intervals'],key=lambda x:x['log_native_mass'])['span'],gt) if q['intervals'] else 0.
 if method=='candidate_oracle':return max([iou(x['span'],gt) for x in q['intervals']]+[0.])
 if method=='verification_pool_oracle':return max([iou(x['span'],gt) for x in q['intervals']]+[iou(v['baseline'],gt)])
 return iou(v['finals'][method]['span'],gt)

def stratified(evaluated):
 methods=('zero_shot','native_mass','event_only','event_onset_offset','candidate_oracle')
 duration_edges=[0,30,60,120,180,math.inf];fraction_edges=[0,.05,.1,.2,.4,math.inf]
 def bucket(v,edges):return next(i for i in range(len(edges)-1) if edges[i]<=v<edges[i+1])
 def label(edges,i):return f'[{edges[i]:g}, {edges[i+1]:g})' if math.isfinite(edges[i+1]) else f'≥{edges[i]:g}'
 def duration(r):return bucket(r['duration'],duration_edges)
 def fraction(r):return bucket((r['span'][0][1]-r['span'][0][0])/r['duration'],fraction_edges)
 lines=['','## Dataset와 구간 길이별 정확도','','아래 모든 값은 해당 칸에 완료된 query만의 값이다. n<10은 작은 표본, n=0은 빈 칸이다. 고정 cohort는 층화 진단 표본이며 benchmark 전체 분포의 무작위 평균이 아니다. Δpp는 같은 칸의 zero-shot 대비 차이다.','']
 categories=[('dataset',sorted({r['dataset'] for r,q,v in evaluated}),lambda r:r['dataset'])]
 categories += [('영상 길이 s',[label(duration_edges,i) for i in range(5)],lambda r:label(duration_edges,duration(r))),('GT/영상 길이 비율',[label(fraction_edges,i) for i in range(5)],lambda r:label(fraction_edges,fraction(r)))]
 for title,labels,assign in categories:
  lines += [f'### {title}','','| 범위 | n | 방법 | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |','|---|---:|---|---:|---:|---:|---:|']
  for name in labels:
   records=[e for e in evaluated if assign(e[0])==name];base=metric([method_iou(*e,'zero_shot') for e in records])
   for method in methods:
    m=metric([method_iou(*e,method) for e in records]);values=[f'{m[k]:.4f} ({100*(m[k]-base[k]):+.2f})' if m[k] is not None else '—' for k in ('miou','r3','r5','r7')]
    lines.append('| '+ ' | '.join([name+(' · 작은 표본' if 0<len(records)<10 else ' · 빈 칸' if not records else ''),str(len(records)),method,*values])+' |')
 for key in ('miou','r3','r5','r7'):
  lines += ['',f'### 영상 길이 × GT 비율 · {key}','','각 칸은 n; zero-shot / native_mass / event_only / event_onset_offset / candidate_oracle 순서이며, 각 값 뒤 괄호는 zero-shot 대비 Δpp다. 비율 마지막 칸은[0.4,1]을 포함한다.','', '| 영상 길이 s | '+' | '.join(label(fraction_edges,i) for i in range(5))+' |','|---|'+'---|'*5]
  for d in range(5):
   cells=[]
   for f in range(5):
    records=[e for e in evaluated if duration(e[0])==d and fraction(e[0])==f];ms=[metric([method_iou(*e,m) for e in records])[key] for m in methods];base=ms[0]
    cells.append(f'n={len(records)}'+(' 작은 표본' if 0<len(records)<10 else ' 빈 칸' if not records else '')+'; '+(' / '.join(f'{v:.3f} ({100*(v-base):+.2f})' for v in ms) if records else '—'))
   lines.append('| '+label(duration_edges,d)+' | '+' | '.join(cells)+' |')
 return lines
def spec(count):return f'''# Interval distribution · pilot{count} · 실행 명세

Top-p=0.99는 원래 전체 vocabulary softmax에서 누적 임계값을 넘는 token까지 포함한다. 그 집합 중 원래 확률1% 미만 token은 제외한다. 1%는 포함하며 하한 때문에99%를 채우지 못해도 추가하지 않는다. Sampling·고정 node/rollout/token 예산·uncertainty pruning은 없다. EOS로 끝나는 허용 경로를 추적하며 실제 model context capacity에 도달하면 미완성 질량으로 남기고 전체 완료로 주장하지 않는다.

입력은 원래 Qwen3-VL-8B-Instruct,2 FPS,video_query,고정 H×W/frame IDs/PTS/tensor hash다. 무작위 반복 seed는 없다. 고정900 cohort의 정확한 prefix300을 decoding·두 모델 semantic·독립4B 검증·MD/HTML·검증까지 마친 뒤 나머지600으로 확장한다. 원자료는 canonical900에 한 번 저장하고300은 부분집합으로 참조한다.

경로 질량은 원래 token 확률의 곱이다. 분기별 raw token entropy와 다음 token entropy 변화,유효 완성 interval의 entropy·시작/종료 covariance·분기 내부/분기 사이 covariance를 저장한다. Interval entropy는 exact clipped (start,end) 쌍으로 계산하며 추가 binning은 없다. Clipping은[0,metadata duration],start<end,EOS 필수다. 조건부 최종 통계는 유효 잎 질량으로 정규화하고 분기 가중치는 p(token)×그 아래 유효 질량이다. 원래 확률 질량과 조건부 가중치를 분리한다. 필터 제외·무효·미완성 질량도 표시한다.

시간 가설은0.5초 grid에 probability-weighted density를 만들고 bandwidth1초로 smoothing한 뒤 watershed 봉우리·골짜기로 나눈다. Saddle가 낮은 쪽 peak의50%보다 높으면 합치며0.5/2초 bandwidth에서 partition 안정성을 검사한다. 이 grid는 clustering용이며 entropy를 다시 binning하지 않는다. 구간 자체를 잘라 비중첩으로 만들지 않는다. Group은 의미 사건 GT가 아니다.

Semantic은 원래 query 없이 각 clip을 독립 encoding한다. Instruct 한 forward에서 마지막 layer의 visual token 공간·시간 평균,4개 시간 bin 연결,공식 embedding과 같은 마지막 readout을 얻는다. Qwen3-VL-Embedding-8B는 같은 clip·고정 instruction·readout token의 normalized hidden state다. Temporal patch당 실제 시간 중심을 이용해 가중하고4개 bin이 비면 해당 조건은 결측이다. 선택된 프레임이 없으면 임의로 채우지 않고 semantic coverage 결측으로 남긴다. 원본 interval이 달라도 실제 frame/PTS/processor 입력이 같으면 재사용한다. 전체 hidden state는 보존하지 않고 normalized vector는 영상·모델 단위 임시 SQLite에서 통계 계산 후 삭제한다.

Unit vector의 가중 평균μ에서 dispersion=1−||μ||²,그룹 A/B의 가중 평균 cosine=μ_A·μ_B다. Self pair를 제외한 구간 평균 cosine도 따로 표시한다. Centroid cosine은두μ를 다시 정규화한 값이며 평균 pair cosine과 다르다. 모든 interval pair를 영구 저장하지 않고 group/branch 요약과 대표 interval IDs를 보존한다. 다른 clustering으로 재분석하려면 clip을 다시 encoding할 수 있다.

독립 검증 모델은 제안8B와 다른 checkpoint인 Qwen3-VL-4B-Instruct다. 각 유효 후보를 앞뒤1초 context와 함께 다시 보고,event/onset/offset의 A=supported,B=contradicted,C=uncertain 확률을 얻는다. 공통 video/query prefix KV를 한 번 계산한 뒤 세 짧은 criterion suffix를 평가한다. Score=P(A)+0.5P(C),선택지 원래 확률 합<1%이면0.5로 판단 보류다. Event-only와 세 criterion score의 기하평균 두 방식을 비교한다. Zero-shot baseline도 같은 방식으로 다시 보고,동점이면 baseline 우선·그다음 원래 경로 질량순이다. 최종 답은 기존 후보 또는 baseline이며 새 timestamp는 생성하지 않는다. GT는 이 모든 과정 이후 채점에만 쓴다. 모든 후보를 평가하되 동일 입력의 점수는 재사용한다. 성능 향상은 가설이며 결과로 검증한다.

[VideoMind 공식 구현](https://github.com/yeliudev/VideoMind/blob/6721d2564eeb9ea8ab6c0ab8eaa39bd231ee8927/videomind/dataset/wrappers/verifying.py)은 예측 구간 crop과 yes/no 검증을, [VideoHV-Agent 공식 구현](https://github.com/Haorane/VideoHV-Agent/blob/ddc160bd05e2d64922ca717d1834500fa7d8f492/VideoHV-Agent/video_hv/verifier.py)은 가설을 영상 근거로 검증하는 절차를 참고했다. 이번 pointwise4B 방법은 이 아이디어를 적용한 별도 조건이며 논문 전체의 재현이 아니다.

비용은 decoding/각 semantic 조건/독립 검증을 나눠 native forward 수,decoder inputs,CUDA stream seconds,wall seconds,peak allocated memory를 기록한다. 동일한 Instruct forward에서 얻는 세 pooling의 GPU encoding 비용은 공유 비용이지 세 번 수행한 것으로 합산하지 않는다. Event-only는 공통 prefill+event suffix,세 criterion은공통 prefill+세 suffix 비용이다. Native decoder inputs는 모델에 실제 넣은 token position 수이며 output token 수나 FLOPs가 아니다. CUDA stream seconds는 kernel active time과 같지 않다. Model load·입력 준비·report 시간도 운영 비용과 구분한다. 실측 share 비용을 방법별 독립 실행시간으로 주장하지 않는다.
'''

def report(final=False):
 rows=read(COHORT);state=read(OUT/'metadata/status.json') if (OUT/'metadata/status.json').exists() else {};active=state.get('active_queries',300);counts=[300]+([900] if active>=900 else [])
 for count in counts:
  result=ROOT/f'results/rollout/interval_distribution/pilot/{count}';artifact=ROOT/f'artifacts/runs/rollout/interval_distribution/pilot/{count}';result.mkdir(parents=True,exist_ok=True);(artifact/'web').mkdir(parents=True,exist_ok=True)
  cohort=rows[:count];catalog=[];evaluated=[];querysummaries=[]
  for row in cohort:
   key=row['sample_key'];qpath=OUT/'queries'/f'{key}.json';vpath=OUT/'verification'/f'{key}.json';item=dict(key=key,query=row['query'],dataset=row['dataset'],gt=row['span'][0],duration=row['duration'],complete=False)
   original=read(SOURCE/'web/samples'/f'{key}.json');video=(ROOT/'results/rollout/deterministic/pilot/900'/original['video']).resolve();from result_layout import relocate_path
   item['video']=os.path.relpath(relocate_path(video),ROOT.parent)
   if qpath.exists():
    q=read(qpath);item.update(traced=True,intervals=len(q['intervals']),entropy=q['root']['entropy']);querysummaries.append(q)
    if vpath.exists():
     v=read(vpath);item['complete']=v['complete']
     if v['complete']:evaluated.append((row,q,v))
   catalog.append(item)
  write(artifact/'web/samples.json',catalog);write(artifact/'metadata/status.json',dict(state,state='complete' if len(evaluated)==count else 'running',completed=len(evaluated),total=count)) if artifact!=OUT else None
  write(artifact/'metadata/source_records.json',dict(queries=count,cohort=str(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{count}/samples.json'),canonical_source=str(OUT),records=[dict(sample_key=r['sample_key'],tree=str(OUT/'trees'/f'{r["sample_key"]}.sqlite')) for r in cohort]))
  lines=[f'# Interval distribution · pilot{count}','',f'전체{count} query 중 decoding{len(querysummaries)}개,독립 검증{len(evaluated)}개 완료. Sampling과 고정 탐색 예산 없이 허용 경로를 추적한다. 완료하지 않은 query를 제외하고 진행 중 결과임을 명시한다.','', '## 정확도와 최종 답 변화','', 'IoU는 예측/GT 교집합 길이÷합집합 길이,mIoU는 query별 IoU 평균,R@τ는IoU≥τ 비율이다. 높을수록 좋다. Δpp=100×(같은 query의 결과−zero-shot). GT는 사후 채점에만 사용한다. Oracle은 실제 후보 중 사후 최고 IoU이며 모델 성능이 아니다. 확률 질량이 가장 큰 interval은 제안 분포 기준의 단일 답이다.','', '| 방법 | 공통 query | mIoU | Δpp vs zero-shot | R@.3 | R@.5 | R@.7 | 답 변경 |','|---|---:|---:|---:|---:|---:|---:|---:|']
  baseline=[iou(v['baseline'],r['span'][0]) for r,q,v in evaluated];base=metric(baseline)
  for method in ('zero_shot','native_mass','event_only','event_onset_offset','candidate_oracle','verification_pool_oracle'):
   values=[];changed=0
   for row,q,v in evaluated:
    gt=row['span'][0]
    if method=='zero_shot':value=iou(v['baseline'],gt)
    elif method=='native_mass':value=iou(max(q['intervals'],key=lambda x:x['log_native_mass'])['span'],gt) if q['intervals'] else 0.
    elif method=='candidate_oracle':value=max([iou(x['span'],gt) for x in q['intervals']]+[0.])
    elif method=='verification_pool_oracle':value=max([iou(x['span'],gt) for x in q['intervals']]+[iou(v['baseline'],gt)])
    else:value=iou(v['finals'][method]['span'],gt);changed+=v['finals'][method]['changed']
    values.append(value)
   m=metric(values);delta=100*(m['miou']-base['miou']) if values else None
   lines.append(f'| {method} | {len(values)} | {cell(m["miou"])} | {"—" if delta is None else f"{delta:+.2f}"} | {cell(m["r3"])} | {cell(m["r5"])} | {cell(m["r7"])} | {changed if method in ("event_only","event_onset_offset") else "—"} |')
  lines+=stratified(evaluated)
  lines+=['','## 다시 본 뒤 실제로 달라진 답','','구간 좌표가 baseline과 달라진 query를 답 변경으로 센다. 개선/악화는 같은 query의 IoU가 baseline보다 커짐/작아짐이며, 동일 IoU의 좌표 변경도 있을 수 있다.','', '| 검증 방식 | 답 변경 | IoU 개선 | IoU 악화 | IoU 동일 |','|---|---:|---:|---:|---:|']
  for name in ('event_only','event_onset_offset'):
   delta=[method_iou(r,q,v,name)-method_iou(r,q,v,'zero_shot') for r,q,v in evaluated]
   lines.append(f'| {name} | {sum(v["finals"][name]["changed"] for r,q,v in evaluated)} | {sum(d>1e-12 for d in delta)} | {sum(d< -1e-12 for d in delta)} | {sum(abs(d)<=1e-12 for d in delta)} |')
  lines+=['','## Decoding 분포','', 'Interval entropy는 exact clipped 구간의 조건부 분포에서−Σp log₂p,분산은 실제 시작·끝 시간의 covariance다. 필터 제외 질량과 무효·미완성 질량을 별도로 남긴다.','', '| Query | Nodes | 유효 interval | Entropy bits | 시작 variance s² | 끝 variance s² | 원래 유효 질량 |','|---|---:|---:|---:|---:|---:|---:|']
  for q in querysummaries:
   cov=q['root']['covariance'];mass=q['mass'].get('valid',0.);lines.append(f'| {q["sample_key"]} | {q["nodes"]} | {len(q["intervals"])} | {cell(q["root"]["entropy"])} | {cell(cov[0][0] if cov else None)} | {cell(cov[1][1] if cov else None)} | {mass:.6g} |')
  lines+=['','## 방법별 연산 오버헤드','', '아래는 현재 cohort의 실측 누적 비용이다. 공유 영상 입력의 재사용을 포함한 운영 비용이며 독립 실행시간으로 해석하지 않는다. Instruct 세 pooling은 한 번의 encoding을 공유한다.','', '| 단계 | Native forwards | Decoder inputs | CUDA stream s | Wall s |','|---|---:|---:|---:|---:|']
  lines.append(f'| decoding (prefill wall 별도) | {sum(q["cost"]["decoder_forwards"] for q in querysummaries)} | {sum(q["cost"]["decoder_inputs"] for q in querysummaries)} | {sum(q["cost"]["native_cuda_seconds"] for q in querysummaries):.3f} | {sum(q["cost"]["wall_seconds"] for q in querysummaries):.3f} |')
  lines.append(f'| decoding video/query prefill | {len(querysummaries)} | {sum(q["cost"].get("prefill_inputs",0) for q in querysummaries)} | — | {sum(q["cost"]["prefill_seconds"] for q in querysummaries):.3f} |')
  lines.append(f'| decoding input preparation | 0 | 0 | — | {sum(q["cost"].get("input_prepare_wall_seconds",0) for q in querysummaries):.3f} |')
  keys={r['sample_key'] for r in cohort}
  for phase in ('instruct','embedding'):
   costs=[]
   for p in (OUT/'semantic'/phase).glob('*.complete.json'):
    costs += [c for c in read(p)['costs'] if set(c.get('execution_queries',[]))&keys and c['status']=='complete']
   lines.append(f'| {phase} semantic encoding | {len(costs)} | {sum(c["input_tokens"] for c in costs)} | {sum(c["cuda_stream_seconds"] for c in costs):.3f} | {sum(c["wall_seconds"] for c in costs):.3f} |')
   lines.append(f'| {phase} input preparation | 0 | 0 | — | {sum(c.get("input_prepare_wall_seconds",0) for c in costs):.3f} |')
   for pool in CONFIG['semantics']:
    observed=[c['pooling_wall_seconds'][pool] for c in costs if pool in c.get('pooling_wall_seconds',{})]
    if observed:lines.append(f'| {pool} pooling (공유 encoding 후 추가) | 0 | 0 | 별도 미계측 | {sum(observed):.3f} |')
  for name,criteria in [('event_only',['event']),('event_onset_offset',['event','onset','offset'])]:
   costs=[]
   for row,q,v in evaluated:
    for candidate in v['input_cache'].values():costs.extend([candidate['prefill_cost']]+[candidate['assessments'][k]['cost'] for k in criteria])
   lines.append(f'| independent verifier {name} | {sum(c["forwards"] for c in costs)} | {sum(c["decoder_inputs"] for c in costs)} | {sum(c["cuda_stream_seconds"] for c in costs):.3f} | {sum(c["wall_seconds"] for c in costs):.3f} |')
  loads=read(OUT/'metadata/model_loads.json') if (OUT/'metadata/model_loads.json').exists() else []
  loads=[v for v in loads if v['purpose']=='production' and v.get('active_queries',300)<=count]
  lines.append(f'| model loading (native forward 아님) | 0 | 0 | — | {sum(v["wall_seconds"] for v in loads):.3f} |')
  baseline_inputs=sum(q.get('baseline_reference',{}).get('decoder_inputs',0) for q in querysummaries);baseline_forwards=sum(q.get('baseline_reference',{}).get('native_forwards',0) for q in querysummaries);search_inputs=sum(q['cost']['decoder_inputs']+q['cost'].get('prefill_inputs',0) for q in querysummaries);search_forwards=sum(q['cost']['decoder_forwards']+q['cost'].get('prefill_forwards',1) for q in querysummaries)
  lines+=['',f'동일한 decoding 완료 query {len(querysummaries)}개에서 greedy 기준은 {baseline_inputs:,} decoder inputs / {baseline_forwards:,} native forwards다. 경로 추적은 {search_inputs:,} inputs / {search_forwards:,} forwards로, 각각 {cell(search_inputs/baseline_inputs if baseline_inputs else None)}배 / {cell(search_forwards/baseline_forwards if baseline_forwards else None)}배다.','', 'Native greedy 기준의 독립 decoder inputs는 동일 영상/query prefill 입력 수 + 첫 token 이후 EOS 예측까지의 추가 greedy token forward 수다. 모델 간 decoder inputs는 FLOPs가 같다는 뜻이 아니다.','', '| 추가 검증 조건 | 대응 query | 추가 inputs / greedy inputs | 추가 forwards / greedy forwards | 입력 준비 wall s | peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
  for name,criteria in [('event_only',['event']),('event_onset_offset',['event','onset','offset'])]:
   extra=[];peaks=[];deni=sum(q['baseline_reference']['decoder_inputs'] for r,q,v in evaluated);denf=sum(q['baseline_reference']['native_forwards'] for r,q,v in evaluated)
   for r,q,v in evaluated:
    for candidate in v['input_cache'].values():extra.extend([candidate['prefill_cost']]+[candidate['assessments'][k]['cost'] for k in criteria]);peaks.append(candidate.get('peak_allocated_bytes',0))
   lines.append(f'| {name} | {len(evaluated)} | {cell(sum(c["decoder_inputs"] for c in extra)/deni if deni else None)} | {cell(sum(c["forwards"] for c in extra)/denf if denf else None)} | {sum(c.get("input_prepare_wall_seconds",0) for c in extra):.3f} | {max(peaks,default=0)/2**30:.3f} |')
  lines+=['','두 검증 조건은 공유 prefill을 각각 한 번 포함한 비교값이므로 서로 합산하지 않는다. Event-only 비용은 실제 실행한 공통 prefill+event suffix이고 독립 실행을 별도로 반복 측정한 wall time은 아니다. Semantic은 위 공유 encoding 비용과 pooling 추가 시간을 함께 비교한다.','']
  lines+=['','## Semantic 통계','', '각 조건의 unit vector 가중 평균μ에서 dispersion=1−||μ||²,그룹 사이 평균 cosine=μ_A·μ_B다. 원래 구간 확률로 가중하고 영상 입력이 없는 질량은 coverage로 따로 표시한다. 전체 vector는 영구 저장하지 않는다.','', '| 조건 | Query | 평균 semantic coverage | 평균 group dispersion |','|---|---:|---:|---:|']
  for pool,phase in [('instruct_mean','instruct'),('instruct_temporal_four','instruct'),('instruct_readout','instruct'),('embedding_readout','embedding')]:
   entries=[]
   for row in cohort:
    p=OUT/'semantic'/phase/f'{row["sample_key"]}.summary.json'
    if p.exists():
     e=read(p)['pools'].get(pool)
     if e:entries.append(e)
   cover=sum(e['embedding_coverage_mass'] for e in entries)/len(entries) if entries else None;valid=[g['semantic_dispersion'] for e in entries for g in e['groups'] if g.get('valid')];disp=sum(valid)/len(valid) if valid else None
   lines.append(f'| {pool} | {len(entries)} | {cell(cover)} | {cell(disp)} |')
  lines+=['','[실행 명세와 지표 정의](SPEC.md)','', '고정300 ⊂ 900 cohort이며 중복 query를 독립 합산하지 않는다. 첫300의 보고서·viewer·검증이 통과한 뒤에만 나머지600을 실행한다.']
  (result/'REPORT.md').write_text('\n'.join(lines)+'\n');(result/'SPEC.md').write_text(spec(count))
  raw=os.path.relpath(OUT,result);catalogurl=os.path.relpath(artifact/'web/samples.json',result);theme=os.path.relpath(ROOT.parent/'_lib/experiment_theme.css',result);asset=os.path.relpath(OUT/'web/viewer.js',result);base=os.path.relpath(ROOT.parent,result)
  html=f'''<!doctype html><html lang="ko" {'data-family-default' if count==300 else ''}><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Interval distribution · pilot{count}</title><link rel="stylesheet" href="{theme}"><style>body{{margin:0;padding:16px}}.layout{{display:grid;grid-template-columns:280px minmax(0,1fr);gap:16px}}#samples{{max-height:80vh;overflow:auto;display:grid;gap:5px}}button{{text-align:left}}video{{width:100%;max-height:340px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.box{{padding:14px;border:1px solid #ccd8e3;border-radius:9px;margin:12px 0}}.rows{{overflow:auto}}.track{{position:relative;height:18px;background:#edf1f5;margin:8px 0}}.track i{{position:absolute;height:100%;background:#4c85bb}}@media(max-width:800px){{.layout{{grid-template-columns:1fr}}#samples{{max-height:200px}}}}</style></head><body><h1>Interval distribution · pilot{count}</h1><p id="status">실행 상태 확인 중</p><main class="layout"><aside><input id="search" placeholder="query 검색" aria-label="query 검색"><div id="samples"></div></aside><article><h2 id="heading"></h2><div id="content">Query를 선택하세요.</div></article></main><script>window.TRACE={{store:{json.dumps(raw)},catalog:{json.dumps(catalogurl)},root:{json.dumps(base)}}};</script><script src="{asset}?sha={sha(OUT/'web/viewer.js')[:12]}"></script></body></html>'''
  (result/'detail.html').write_text(html)
 write(OUT/'web/summary.json',dict(active_queries=active,completed=state.get('completed',0),timestamp=stamp()))

if __name__=='__main__':report()
