"""Cumulative reports; MD owns numeric definitions, HTML owns interaction."""
import collections,html,math,os
from probability_budget_core import *
def metrics(vals):return [sum(vals)/len(vals)]+[sum(v>=t for v in vals)/len(vals) for t in (.3,.5,.7)] if vals else [None]*4
def f(v,n=3):return '—' if v is None else f'{v:.{n}f}'
def mean(vals):return sum(vals)/len(vals) if vals else None

def observation_report(rows,records):
 import datetime
 observations=[]
 for row in rows:
  path=OUT/'token_distributions'/f'{row["sample_key"]}.json'
  if path.exists():observations.append(read(path))
 execution=OUT/'metadata/distribution_execution.json';sessions=read(execution).get('sessions',[]) if execution.exists() else []
 overlap=0
 for row,q in records:
  if not q.get('timestamp'):continue
  end=datetime.datetime.fromisoformat(q['timestamp']).timestamp();start=end-q.get('query_search_wall_seconds',0)-q['input'].get('prefill_wall_seconds',0)-q['greedy']['cost'].get('wall_seconds',0)
  if any(start<=datetime.datetime.fromisoformat(s.get('finished_at') or stamp()).timestamp() and end>=datetime.datetime.fromisoformat(s['started_at']).timestamp() for s in sessions):overlap+=1
 costs=[r['cost'] for r in observations];counts=collections.Counter()
 for item in observations:counts.update(item['counts'])
 return ['','## Viewer용 token distribution 보충','','초기 원자료에 실제 numeric fork의 분포만 남아 있어, 나머지 계산 위치의 native top30을 동일 입력 hash·동일 token prefix로 보충한다. 이후 query는 탐색 시 모든 계산 위치의 분포와 정책 이유를 직접 기록해 별도 재계산을 줄인다. 탐색에서 직접 저장한 분포와 기존 deterministic trace를 먼저 재사용하고 누락 prefix만 single-token KV 실행으로 재계산한다. 새 후보를 생성하거나 기존 탐색의 node·확률·예산 배분·정답을 바꾸지 않는다. 소수 자릿수의 token distribution 표시는 fractional branching 또는 branch entropy 측정이 아니다.','',f'현재 범위에서 보충을 마친 query {len(observations)} / {len(rows)}개. 아직 분포가 없는 위치는 viewer에서 보충 대기로 표시한다. 보충 작업은 누적 원자료에만 저장하며 pilot별로 복제하지 않는다.','','| 항목 | 값 |','|---|---:|',f'| 탐색에서 직접 저장한 위치 | {counts["native_search"]} |',f'| 기존 reference 재사용 위치 | {counts["existing_reference"]} |',f'| 다른 방법에서 관측한 동일 prefix 재사용 위치 | {counts["other_method"]} |',f'| 새로 재계산한 고유 prefix | {counts["supplemental_replay"]} |',f'| 보충 prefill 입력 합 | {sum(c.get("prefill_inputs",0) for c in costs)} |',f'| 보충 decoder inputs 합 (replay 포함) | {sum(c.get("decoder_inputs",0) for c in costs)} |',f'| 보충 작업 wall 합 s | {sum(c.get("wall_seconds",0) for c in costs):.3f} |',f'| 보충 CUDA decode 합 s | {sum(c.get("decode_cuda_seconds",0) for c in costs):.3f} |',f'| 저장된 경로 확률과 최대 절대 차이 | {max([r.get("max_edge_probability_error",0) for r in observations]+[0]):.8f} |','', '보충 inputs와 wall은 위 search 예산·비용에 합산하지 않는 관측용 추가 비용이다. 고유 prefix 수는 같은 입력의 정확히 같은 token ID prefix를 한 번 센 값이다. 확률 차이는 각 prefix에서 원래 추적한 자식 token의 native 확률과 재계산 확률의 절대 차이이며 낮을수록 재현이 정확하다. Wall 합은 query별 실측 합이며 작업 전체 경과 시간이나 전용 GPU 성능을 뜻하지 않는다.','',f'공통 완료 query 중 보충 프로세스와 실행 시간대가 겹친 query는 {overlap} / {len(records)}개다. Search wall·CUDA event 시간은 실제 실행값이므로 같은 GPU에서 보충 작업이 실행된 동안의 자원 경합을 포함할 수 있다. 해당 wall 수치를 보충 작업이 없는 전용 GPU의 search 비용으로 해석하지 않는다. 실행 시간대는 metadata/distribution_execution.json, 보충 작업별 비용은 token_distributions와 logs/distribution_capture.log에 남긴다.','']

def spec(count):return f'''# Probability budget · pilot{count} · 실행 명세

고정 cohort는 기존 nested pilot900의 정확한 prefix다. Pilot300 전체의 추론·보고서·viewer·원자료 검증을 마친 뒤 누적900으로 진행한다. 이 실행은 candidate proposal 비교이며 independent verifier의 최종 답 선택 실험이 아니다. 기존 3-query 시간 제한 진단과 별도 계열이다.

모델은 Qwen3-VL-8B-Instruct, native temperature1, sampling 없음, seed 반복 없음이다. 입력은 기존 deterministic의2 FPS·video_query·frame IDs·PTS·H×W·tensor hash를 그대로 대조한다. 영상 FPS·frame 수·해상도를 가속 목적으로 줄이지 않는다.

매 step에서 native top-p=0.99 nucleus와 native 확률≥1%의 교집합을 만든다. 1% 미만은 nucleus를 채우기 위해 추가하지 않는다. 정수부 숫자와 소수점 기호에서만 대안을 만들고 일반 문구·소수 자릿수는 greedy다. 소수 자릿수의 branch entropy는 계산하지 않는다. 원자 token이 소수점을 가로지르면 정수부/점의 효과가 같은 fractional-only 대안은 최고 native 확률 token만 유지한다. 전체 vocabulary 확률을 sibling 사이에서 재정규화하지 않는다.

Depth는 이미 생성한 답변 token 수이며, branch 횟수나 미래 답변 길이가 아니다. Greedy 문구·소수 자릿수도 depth에는 포함한다. Video/query 입력 token은 제외한다. Prefix마다 native log probability를 누적한다.

- **levin**: 전역 frontier의 log(depth)−log(native path probability)가 최소인 node를 선택한다. Greedy 위치에서도 다음 prefix를 다른 frontier와 다시 비교한다. 원래 LevinTS의 첫 목표 종료를 다중 후보 열거로 바꾼 적용이다. 목표 tree는 이번 token/분기 정책을 통과한 tree다.
- **native_mass**: 첫 실제 numeric fork의 자식들로 서로 겹치지 않는 고정 subtree 그룹을 만든다. 그룹별 미탐색 native 질량 U에 비례해 실제 decoder 입력 연산을 배분한다. Credit은 소비한 실제 연산으로 갱신하고 quantum32에서 다음 그룹을 선택한다. 그룹 내에서는 최고 native prefix를 골라 greedy continuation을 완성한다. 초기 fork 이전의 비용도 전체 예산에 포함한다.
- **numeric_novelty**: 같은 scheduler에 w_i∝U_i(1+0.1q_i)를 적용한다. q_i는 그 그룹에서 완성한 유효 구간들의 native-mass 가중 평균 novelty다. 각 novelty=1−다른 그룹의 완성 구간과의 최대 temporal IoU다. 자기 그룹과 비교하지 않는다. 비교 근거가 없으면 보정0이며, unknown을 무효로 취급하지 않는다. 목표 weight의 상대 odds 보정은 최대1.1배다. 실제 연산 배분은 job·cache 단위여서 그 비율과 정확히 일치하지 않을 수 있다. 이 방법은 DORA의 배분 틀을 참고한 numeric variant이며 원래 PRM·embedding 기반 DORA의 재현이나 최적성 보장을 주장하지 않는다.

같은 interval에 도달한 서로 다른 sequence의 native 질량은 합친다. 후보 개수를 예산 가중치로 쓰거나 진짜 확률 질량을 중복 개수로 나누지 않는다. 다른 token prefix는 같은 interval이어도 모델 상태를 합치지 않는다. Semantic embedding·GT·신경망 reward는 배분에 사용하지 않는다. Temporal novelty는 사건 의미나 정답 확률이 아니다.

각 query×방법의 decoder 입력 상한은2,048이며512·1,024·2,048 checkpoint를 한 누적 search에서 기록한다. 샘플·방법 사이로 잔여 예산을 이월하지 않는다. Prefix replay도 상한에 포함하며 model forward 직전에 검사하므로 미완성 prefix는 pending으로 남긴다. 새 video/query prefill은 검색 상한 밖의 공유 비용으로 별도 기록한다. Greedy reference도 별도 연산으로 측정한다. 시간 제한으로 후보를 자르지 않는다.

V=완성된 유효 경로의 native 질량, U=아직 pending인 frontier 질량+context capacity 때문에 미완성인 질량이다. U/(V+U)≤0.01이면 조기 종료한다. 이 상한은 허용된 분기 정책 안의 최종 유효 분포에 관한 것이며 full model 분포 전체의99%를 보장하지 않는다. 필터 제외·정책 제외·무효 질량을 각각 보존한다. Context limit 질량을 유효/무효로 확정하지 않는다.

영상/query KV는 query당 한 번 공유한다. 계산한 answer prefix의 text-only KV suffix를 최대8GiB에서 LRU 재사용하며 다른 prefix의 상태를 근사하지 않는다. Smoke에서 복원 결과와 fresh prefill의 확률·admitted token을 대조한다. Cache의 저장·복원 비용과 eviction·재실행 token도 측정한다. Node expansion 보장은 GPU 비용 보장과 다르다.

완성 interval은[0,actual metadata duration]으로 clipping하고 start<end와 EOS를 요구한다. 뒤집힌 구간을 정렬하지 않는다. Interval entropy=−Σp log₂p는 exact clipped (start,end) atom의 관측 유효 질량 조건부 분포에서 계산한다. 추가 entropy binning은 없다. 시작·끝 covariance는 같은 가중 분포의 population covariance이며 단위는s²다. 미탐색 부분이 있는 통계를 전체 분포의 정확한 통계로 부르지 않는다.

[LevinTS](https://proceedings.neurips.cc/paper/2018/file/52c5189391854c93e8a0e1326e56c14f-Paper.pdf), [PHS의 sub-probability 정의](https://arxiv.org/html/2103.11505), [DORA](https://arxiv.org/html/2506.15707v1)가 방법 참고다.
'''

def stratified(records,budget=1024):
 methods=['greedy']+METHODS;de=[0,30,60,120,180,math.inf];fe=[0,.05,.1,.2,.4,math.inf]
 def bucket(x,edges):return next(i for i in range(5) if edges[i]<=x<edges[i+1])
 def label(edges,i):return f'[{edges[i]:g},{edges[i+1]:g})' if math.isfinite(edges[i+1]) else f'≥{edges[i]:g}'
 def value(item,method):
  row,q=item;span=q['greedy']['span'] if method=='greedy' else q['methods'][method]['snapshots'][str(budget)]['selected_span'];return iou(span,row['span'][0])
 lines=['','## Dataset·영상 길이·GT 비율별 결과','','기본 예산1,024의 최대 native-mass 후보를 평가한다. n<10은 작은 표본이며 n=0은 빈 칸이다. 고정 cohort는 층화 진단 표본이며 전체 benchmark 분포의 무작위 평균이 아니다.','']
 cats=[('Dataset',sorted({r['dataset'] for r,q in records}),lambda r:r['dataset']),('영상 길이 s',[label(de,i) for i in range(5)],lambda r:label(de,bucket(r['duration'],de))),('GT/영상 길이',[label(fe,i) for i in range(5)],lambda r:label(fe,bucket((r['span'][0][1]-r['span'][0][0])/r['duration'],fe)))]
 for title,labels,assignment in cats:
  lines += [f'### {title}','','| 범위 | n | 방법 | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |','|---|---:|---|---:|---:|---:|---:|']
  for label_ in labels:
   subset=[x for x in records if assignment(x[0])==label_];base=metrics([value(x,'greedy') for x in subset])
   for method in methods:
    m=metrics([value(x,method) for x in subset]);vals=[f'{v:.4f} ({100*(v-b):+.2f})' if v is not None else '—' for v,b in zip(m,base)];lines.append('| '+label_+(' · 작은 표본' if 0<len(subset)<10 else ' · 빈 칸' if not subset else '')+f' | {len(subset)} | {method} | '+' | '.join(vals)+' |')
 for k,title in enumerate(('mIoU','R@.3','R@.5','R@.7')):
  lines += ['',f'### 영상 길이 × GT 비율 · {title}','','각 칸은 n; greedy / levin / native_mass / numeric_novelty 순서다. 괄호는 해당 칸의 greedy 대비Δpp이며 n<10은 작은 표본이다.','', '| 영상 s | '+' | '.join(label(fe,i) for i in range(5))+' |','|---|'+'---|'*5]
  for d in range(5):
   cells=[]
   for ff in range(5):
    subset=[x for x in records if bucket(x[0]['duration'],de)==d and bucket((x[0]['span'][0][1]-x[0]['span'][0][0])/x[0]['duration'],fe)==ff];vals=[metrics([value(x,m) for x in subset])[k] for m in methods];b=vals[0];cells.append(f'n={len(subset)}'+(' 작은 표본' if 0<len(subset)<10 else '')+'; '+(' / '.join(f'{v:.3f} ({100*(v-b):+.2f})' for v in vals) if subset else '—'))
   lines.append('| '+label(de,d)+' | '+' | '.join(cells)+' |')
 return lines

def report():
 from result_layout import relocate_path
 state=read(OUT/'metadata/status.json');active=state.get('active_queries',300);rows=read(COHORT);scope_summaries=[];new_scope=False
 for count in [300]+([900] if active>=900 else []):
  result=ROOT/f'results/rollout/probability_budget/pilot/{count}';artifact=ROOT/f'artifacts/runs/rollout/probability_budget/pilot/{count}';new_scope=new_scope or not result.exists();result.mkdir(parents=True,exist_ok=True);catalog=[];records=[]
  for row in rows[:count]:
   key=row['sample_key'];p=OUT/'queries'/f'{key}.json';q=read(p) if p.exists() else None;original=read(SOURCE/'web/samples'/f'{key}.json');video=relocate_path((ROOT/'results/rollout/deterministic/pilot/900'/original['video']).resolve());catalog.append(dict(key=key,query=row['query'],dataset=row['dataset'],gt=row['span'][0],duration=row['duration'],video=os.path.relpath(video,ROOT.parent),complete=bool(q and q['complete']),methods=[] if q is None else list(q['methods'])))
   if q and q['complete']:records.append((row,q))
  write(artifact/'web/samples.json',catalog);write(artifact/'metadata/source_records.json',dict(queries=count,cohort=str(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{count}/samples.json'),cohort_sha256=sha(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{count}/samples.json'),canonical=str(OUT),query_keys=[r['sample_key'] for r in rows[:count]]))
  if artifact!=OUT:write(artifact/'metadata/status.json',dict(state,state='complete' if len(records)==count else 'running',completed=len(records),total=count))
  lines=[f'# Probability budget · pilot{count}','',f'{count} query 중 세 방법을 모두 완료한 query {len(records)}개. 고유 영상 {len({r["video_path"] for r in rows[:count]})}개. 조건 반복 수와 query 수를 구분한다. 완료 전 수치는 진행 중 공통 부분집합이며 전체 성능으로 해석하지 않는다.','', '## 후보 품질과 zero-shot 대비 차이','','IoU=예측·GT 교집합 길이÷합집합 길이,mIoU=query별 IoU 평균,R@τ=IoU≥τ 비율이다. 높을수록 좋다. 괄호 안Δpp=100×(결과−같은 query의zero-shot greedy). Greedy는 동일 입력의 새 native generation이며 clipping·평가 정책도 같다. GT는 탐색 후 평가에만 사용한다.','', '최대 native-mass 후보는 관측된 sequence 질량을 exact interval별로 합친 뒤 가장 큰 interval을 고른 진단값이다. 독립 verifier가 선택한 final answer가 아니다. Candidate oracle은 생성 후보 중 사후 최고 GT IoU이며 실제 selector 성능이 아니다.','', '| 예산 | 방법·평가 | 공통 n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |','|---:|---|---:|---:|---:|---:|---:|']
  base=metrics([iou(q['greedy']['span'],r['span'][0]) for r,q in records]);lines.append('| — | zero-shot greedy | '+str(len(records))+' | '+' | '.join(f'{v:.4f} (+0.00)' if v is not None else '—' for v in base)+' |')
  # Search raw files only once, avoiding response copies and repeated JSON reads.
  oracles={}
  for row,q in records:
   for method in METHODS:
    raw=read(OUT/'search'/row['sample_key']/f'{method}.json')
    for budget in BUDGETS:
     s=raw['snapshots'][str(budget)];oracles[row['sample_key'],method,budget]=max([iou(l['span'],row['span'][0]) for l in raw['leaves'][:s['leaf_count']] if l['valid']]+[0.])
  for budget in BUDGETS:
   for method in METHODS:
    for evaluation in ('native_mass','candidate_oracle'):
     vals=[]
     for row,q in records:
      s=q['methods'][method]['snapshots'][str(budget)];gt=row['span'][0]
      if evaluation=='native_mass':v=iou(s['selected_span'],gt)
      else:
       v=oracles[row['sample_key'],method,budget]
      vals.append(v)
     m=metrics(vals);lines.append(f'| {budget} | {method} · {evaluation} | {len(vals)} | '+' | '.join(f'{v:.4f} ({100*(v-b):+.2f})' if v is not None else '—' for v,b in zip(m,base))+' |')
  lines+=['','## 실제 연산 비용과1% 달성률','','비용은 같은 공통 query의 누적 실측이다. 예산 checkpoint끼리 합산하지 않는다. Decoder inputs는 실제 모델 입력 position 수이며 replay도 포함한다. Prefill·KV 복사·reward 계산은 아래와 별도/함께 표시하므로 forward 수를 FLOPs나 wall time과 동일시하지 않는다.','', '| 예산 | 방법 | n | 평균 decoder inputs | 평균 search wall s | CUDA decode 합 s | KV 저장·복원 합 s | Reward 계산 합 s | 1% 도달 | 평균 미탐색 상한 |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for budget in BUDGETS:
   for method in METHODS:
    ss=[q['methods'][method]['snapshots'][str(budget)] for r,q in records];cs=[s['cost'] for s in ss];lines.append(f'| {budget} | {method} | {len(ss)} | {f(mean([c["decoder_inputs"] for c in cs]),1)} | {f(mean([c["wall_seconds"] for c in cs]))} | {sum(c["decode_cuda_seconds"] for c in cs):.3f} | {sum(c["kv_store_cuda_seconds"]+c["kv_restore_cuda_seconds"] for c in cs):.3f} | {sum(c["reward_wall_seconds"] for c in cs):.3f} | {sum(s["reason"]=="mass_target_reached" for s in ss)} / {len(ss)} | {f(mean([s["unseen_valid_mass_upper_bound"] for s in ss if s["unseen_valid_mass_upper_bound"] is not None]))} |')
  lines+=['','## Greedy 대비 오버헤드','','배수는同じ query들의 합계 비율이다. Decode 배수는 prefill 이후의 실제 입력 position 비율,전체 inputs 배수는 각 방법에 shared prefill을 한 번 귀속한 값이다. 서로 다른 context 길이에서 같은 입력 position 수가 같은 FLOPs라는 뜻은 아니다.','', '| 예산 | 방법 | Decode inputs 배수 | Prefill 포함 inputs 배수 | Search+prefill wall 배수 |','|---:|---|---:|---:|---:|']
  greedy_inputs=sum(q['greedy']['cost']['decoder_inputs'] for r,q in records);prefill_inputs=sum(q['input']['input_tokens'] for r,q in records);prefill_wall=sum(q['input']['prefill_wall_seconds'] for r,q in records);greedy_wall=sum(q['greedy']['cost']['wall_seconds'] for r,q in records)
  for budget in BUDGETS:
   for method in METHODS:
    cs=[q['methods'][method]['snapshots'][str(budget)]['cost'] for r,q in records];inputs=sum(c['decoder_inputs'] for c in cs);wall=sum(c['wall_seconds'] for c in cs);lines.append(f'| {budget} | {method} | {f(inputs/greedy_inputs if greedy_inputs else None)} | {f((inputs+prefill_inputs)/(greedy_inputs+prefill_inputs) if greedy_inputs+prefill_inputs else None)} | {f((wall+prefill_wall)/(greedy_wall+prefill_wall) if greedy_wall+prefill_wall else None)} |')
  lines+=['',f'공유 입력 준비 {sum(q["input"]["input_prepare_wall_seconds"] for r,q in records):.3f}s; video/query prefill {prefill_wall:.3f}s; greedy decode {greedy_wall:.3f}s; peak allocated {max([q["peak_allocated_bytes"] for r,q in records]+[0])/2**30:.2f}GiB.','', '## 분기 coverage와 확률 질량','','Coverage는 초기 numeric fork의 subtree 중 유효 후보를 하나 이상 완성한 그룹 비율이다. 그룹은 semantic mode가 아니다. 확률 가중 coverage는 그 그룹 root의 native 질량 합으로 가중한다. 균등 coverage 극대화를 목표로 하지 않고 고확률 그룹에 더 많은 연산을 배분한다.','', '| 예산 | 방법 | 평균 그룹 coverage | 평균 확률 가중 coverage | 평균 원래 유효 질량 | 평균 정책 제외 질량 |','|---:|---|---:|---:|---:|---:|']
  for budget in BUDGETS:
   for method in METHODS:
    cs=[];ws=[];vs=[];ps=[]
    for r,q in records:
     s=q['methods'][method]['snapshots'][str(budget)];gs=[g for g in s['groups'] if g['root'] is not None] or s['groups'];cs.append(sum(g['valid_leaves']>0 for g in gs)/len(gs));total=sum(g['root_native_mass'] for g in gs);ws.append(sum(g['root_native_mass'] for g in gs if g['valid_leaves']>0)/total if total else 0.);vs.append(s['mass']['valid']);ps.append(s['mass']['policy'])
    lines.append(f'| {budget} | {method} | {f(mean(cs))} | {f(mean(ws))} | {f(mean(vs))} | {f(mean(ps))} |')
  scope_summaries.append(dict(scope=count,n=len(records),metrics={method:metrics([iou(q['greedy']['span'] if method=='greedy' else q['methods'][method]['snapshots']['1024']['selected_span'],r['span'][0]) for r,q in records]) for method in ['greedy']+METHODS}))
  lines+=stratified(records)
  lines+=observation_report(rows[:count],records)
  lines+=['','## 해석 범위','','현재 얻은 interval entropy·variance는 관측 유효 분포에 조건부다. 종료 시 남은 질량·context limit·정책 제외 질량을 함께 확인한다. Candidate oracle 향상이 최대 native-mass 답의 성능 향상을 의미하지 않는다. Novelty는 시간적 차이이며 사건 semantic이나 정답 여부를 판별하지 않는다.','', '[실행 명세와 지표 정의](SPEC.md)']
  text='\n'.join(lines).replace('同じ','같은');(result/'REPORT.md').write_text(text+'\n');(result/'SPEC.md').write_text(spec(count))
  rawurl=os.path.relpath(OUT,result);caturl=os.path.relpath(artifact/'web/samples.json',result);theme=os.path.relpath(ROOT.parent/'_lib/experiment_theme.css',result);script=os.path.relpath(OUT/'web/viewer.js',result);rooturl=os.path.relpath(ROOT.parent,result)
  assets=Path(__file__).parent
  for suffix in ('js','css'):
   source=assets/f'probability_budget_viewer.{suffix}'
   target=OUT/f'web/viewer.{suffix}'
   if source.read_bytes()!=target.read_bytes() if target.exists() else True:
    tmp=target.with_name(target.name+f'.{os.getpid()}.tmp');tmp.write_bytes(source.read_bytes());tmp.replace(target)
  style=os.path.relpath(OUT/'web/viewer.css',result)
  traces=os.path.relpath(SOURCE/'token_traces',result)
  template=(assets/'probability_budget_viewer.html').read_text()
  replacements={'__DEFAULT__':'data-family-default' if count==300 else '', '__COUNT__':str(count),'__THEME__':theme,'__STYLE__':style+'?sha='+sha(OUT/'web/viewer.css')[:12],'__SCRIPT__':script+'?sha='+sha(OUT/'web/viewer.js')[:12],'__CONFIG__':json.dumps(dict(store=rawurl,catalog=caturl,root=rooturl,traces=traces))}
  for old,new in replacements.items():template=template.replace(old,new)
  target=result/'detail.html';tmp=target.with_name(target.name+f'.{os.getpid()}.tmp');tmp.write_text(template);tmp.replace(target)

 if active>=900:
  overview=ROOT/'results/rollout/probability_budget/pilot/overview';overview.mkdir(parents=True,exist_ok=True);lines=['# Probability budget · pilot300 / pilot900 누적 비교','','같은 원자료의 중첩 cohort이며 두 범위를 독립 표본처럼 합산하지 않는다. 기본 예산1,024에서 최대 native-mass 후보를 평가한다. mIoU와R은 높을수록 좋고 괄호는 같은 범위의 greedy 대비Δpp다. 진행 중인 pilot900은 완료된 n만의 값이다.','', '| 범위 | 완료 n | 방법 | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |','|---|---:|---|---:|---:|---:|---:|']
  for item in scope_summaries:
   base=item['metrics']['greedy']
   for method,m in item['metrics'].items():lines.append(f'| pilot{item["scope"]} | {item["n"]} | {method} | '+' | '.join(f'{v:.4f} ({100*(v-b):+.2f})' if v is not None else '—' for v,b in zip(m,base))+' |')
  lines+=['','[pilot300 상세](../300/REPORT.md) · [pilot900 상세](../900/REPORT.md)'];(overview/'REPORT.md').write_text('\n'.join(lines)+'\n');write(ROOT/'artifacts/runs/rollout/probability_budget/pilot/overview/metadata/source_records.json',dict(canonical=str(OUT),cohorts=[300,900],overlapping=True,default_budget=1024))
 if new_scope:
  import subprocess,sys
  subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
 print('REPORT_UPDATED',len(records),active,flush=True)

if __name__=='__main__':
 import fcntl
 with (OUT/'metadata/report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);report()
