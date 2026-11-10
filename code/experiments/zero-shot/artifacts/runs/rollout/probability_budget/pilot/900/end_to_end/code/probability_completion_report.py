"""End-to-end evidence and overhead; never confuse proposal oracle with final answers."""
import html,os
from probability_completion_core import *
def f(v,k=4):return f'{v:.{k}f}' if v is not None and math.isfinite(v) else '—'
def metrics(values):return [sum(values)/len(values)]+[sum(v>=t for v in values)/len(values) for t in (.3,.5,.7)] if values else [None]*4
def mcell(vals,base):return ' | '.join(f'{v:.4f} ({100*(v-b):+.2f})' if v is not None else '—' for v,b in zip(vals,base))
def specs(count):
 return f'''# End-to-end completion · pilot{count} · 실행 명세

기존 probability budget의 후보를 그대로 사용한다. 후보 생성의 512·1,024·2,048 checkpoint와 LevinTS·native mass·numeric novelty를 구분한다. 고정 pilot300을 모두 처리하고 보고서·viewer까지 검증한 뒤 나머지600의 후속 계산을 수행한다. 기존 탐색은 재실행하거나 GT로 수정하지 않는다.

## 독립 구간 검증과 최종 생성

Qwen2.5-VL-7B-Instruct가 모든 고유 후보 구간을 독립적으로 다시 본다. 제안 모델인 Qwen3-VL-8B-Instruct와 가중치가 다른 모델이다. 검증 입력은 해당 crop과 query뿐이다. 다른 후보·원래 path probability·GT를 넣지 않는다. Native next-token의 0=다른 사건/없음, 1=부분적·경계 부정확, 2=완전한 사건·적절한 경계 확률을 측정한다. score=(p1+2p2)/2이며 p는 세 label 안의 조건부 확률이다. 원래 세 label의 전체 vocabulary 질량도 저장한다. 점수가 높을수록 이 prompt에서 모델이 긍정적으로 평가했다는 뜻이며 정답 확률의 보정값이 아니다.

각 조건에 대해 검증 점수 최대 후보, 후보만 제공한 Qwen3-VL의 최종 재생성(proposal_only), 검증 점수도 제공한 최종 재생성(reviewed)을 평가한다. 원래 full video/query 입력·tensor hash·greedy 답변은 동일하게 공유하고 매 prompt는 저장한다. Semantic cosine은 어느 선택에도 쓰지 않는다. Final answer는 후보의 경계를 수정할 수 있으므로 candidate oracle은 final regeneration의 엄밀한 상한이 아니다. EOS 없는 생성·parsing 실패는 유효 결과로 바꾸지 않고 IoU0으로 평가한다. 생성은 greedy·최대64 token이며 [0,실제 영상 길이] clipping을 적용한다.

## Crop과 연산 재사용

원래2 FPS frame sampling에서 구간 내부의 프레임을 모두 사용한다. 내부 프레임이 없으면 원본 영상에서 구간 중간 시점에 실제 표시되는 프레임을 추출한다. 한 프레임은 Qwen의 시간축 병합을 위해 두 번 넣으며 이 경우를 기록한다. Crop timestamp는 첫 프레임부터의 상대 시각으로 바꾼다. 같은 frame IDs·상대 timestamp를 갖는 입력은 query 안에서 한 번 계산한다. 이는 서로 다른 사건이라는 판단이나 decoding tree pruning이 아니다. Crop의 실제 시간 해상도 때문에 서로 다른 숫자가 같은 시각 입력이 될 수 있다.

Qwen3의 원래 H×W를 보존한다. Qwen2.5 verifier에는 원래 면적과 가까운28 배수 H×W로 resize한다. 프레임 수를 줄이지 않는다. Batch의 최대 padded 입력은16,384 token·최대8 clip이다. 단일 긴 clip은 자르지 않고 단독 처리한다. 첫 실제 batch는 serial 결과와 대조한다. 검증 확률의 최대 절대 차이가1%p 이상이거나 semantic cosine이0.999 이하이면 해당 모델은 serial native 실행으로 전환하고, 검사와 버린 batch의 계산 비용도 기록한다. 같은 최종 prompt는 해시로 한 번 생성한다. 본 실험 안의 검증 통과 여부만 남기고 별도 smoke-test 실험은 만들지 않는다.

## Query-free semantic 측정

Qwen3-VL-8B-Instruct의 마지막 layer에서 video token만 mean pooling하고 L2 정규화한다. 공간 격자 크기가 일정하므로 시간별 격자에 같은 가중치를 주는 frame-balanced mean과 같다. 마지막 nonpadding token pooling도 같은 forward에서 보조 측정한다. Qwen3-VL-Embedding-8B는 공식 구현과 같은 generic instruction·chat template·마지막 nonpadding token pooling을 사용한다. 이 tokenizer에서 마지막 token이 문자 그대로 EOS인지는 따로 가정하지 않는다. 임의의 EOS를 추가하거나8192 token으로 자르지 않는다. 두 모델 모두 query·GT 없이 crop만 받는다.

실제 벡터는 query 처리 중 RAM에만 두고 폐기한다. 모델별 scalar 통계를 계산한 즉시 checkpoint로 저장하며 재개할 때 끝난 모델은 건너뛴다. 후보가0개면 비교를 만들지 않고,1개면 서로 다른 두 구간의 cosine 통계는 n=0·미정의로 남긴다. 이를 cosine0으로 대체하지 않는다. 동일 모델 안의 cosine에 대해 native-mass 가중 mean·std·10/50/90% quantile, temporal IoU 구간별 통계, 시간 그룹 내부/사이 통계, greedy와의 유사도, 가장 다른12쌍을 저장한다. 서로 다른 모델의 벡터를 직접 비교하지 않는다. Cosine이 높다는 사실만으로 같은 사건이라고 확정하지 않는다.

## 시간 분포와 분기별 불확실성

Entropy와 covariance는 원래 exact clipped(start,end) atom에서 계산한다. KDE를 위한 시간 격자나 entropy 재binning을 추가하지 않는다. 시간 그룹은 endpoint 공간의 Gaussian KDE(h=1초)와8-nearest support graph(거리≤4h)로 구한다. Edge saddle은 두 끝과 중간의 밀도 중 최솟값이다. Saddle÷두 peak 중 낮은 밀도≥0.5이면 합친다. 관측 support graph로 근사한 통계적 mode이며 별개 semantic 사건의 GT가 아니다. h=0.5·2초에서 그룹 수의 민감도도 함께 기록한다.

각 실제 numeric fork에서 완성된 유효 descendant의 질량을 모아 최종 interval entropy(bits), start/end population covariance(s²), 시간 그룹 entropy를 계산한다. 큰 interval entropy는 끝까지 이어간 유효 후보가 여러 숫자 구간에 퍼졌다는 뜻이다. 분기 token과 시간 그룹의 관측 MI=H(C)−ΣP(a|관측 valid)H(C|a)도 저장한다. 미탐색·무효 부분을 채운 전체 분포의 정확한 entropy/MI라고 부르지 않는다. 소수 자릿수를 새로운 branch로 만들거나 fractional branch entropy를 측정하지 않는다.

## 평가와 비용

IoU=교집합 길이÷합집합 길이, mIoU=query별 IoU 평균, R@τ=IoU≥τ 비율이며 높을수록 좋다. Δpp=100×(해당 결과−같은 query의 greedy). Changed는 clipped span이 정확히 달라진 query 비율이며 개선을 뜻하지 않는다. Invalid final answer는 IoU0으로 포함한다. GT는 이 평가에서만 읽는다.

검증·Instruct semantic·Embedding semantic·최종 생성·CPU 통계의 비용을 분리한다. 실제 unique batch forward의 nonpadding/padded tokens·CUDA·wall을 기록한다. Clip별 amortized 시간은 batch 시간을 입력 token 비율로 나눈 회계값이며 clip 단독 latency나 FLOPs가 아니다. 조건별 final generation 비용은 고정 실행 순서에서 실제 소비한 incremental 연산이며 동일 prompt 재사용을 독립 실행처럼 합산하지 않는다. 같은 GPU에서 기존 proposal search와 실행될 수 있으므로 wall·CUDA event 시간에는 자원 경합이 포함된다.

## 참고한 공개 구현

'''+ '\n'.join(f'- [{r["name"]}]({r["url"]}): {r["use"]}.' for r in REFERENCES)+'\n'

def report():
 state=read(OUT/'metadata/status.json');active=state.get('active_queries',300);rows=read(COHORT)
 for count in [300]+([900] if active>=900 else []):
  data=[];catalog=[]
  parent_catalog=read(ROOT/f'artifacts/runs/rollout/probability_budget/pilot/{300 if count==300 else 900}/web/samples.json');lookup={r['key']:r for r in parent_catalog}
  for row in rows[:count]:
   path=OUT/'queries'/f'{row["sample_key"]}.json';d=read(path) if path.exists() else None;catalog.append(dict(lookup[row['sample_key']],complete=bool(d and d.get('complete'))))
   if d and d.get('complete'):data.append((row,d,read(PARENT/'queries'/f'{row["sample_key"]}.json')))
  if not data:continue
  result=ROOT/f'results/rollout/probability_budget/pilot/{count}/end_to_end';artifact=ROOT/f'artifacts/runs/rollout/probability_budget/pilot/{count}/end_to_end';result.mkdir(parents=True,exist_ok=True);write(artifact/'web/samples.json',catalog)
  base=metrics([iou(q['greedy']['span'],r['span'][0]) for r,d,q in data]);gate=OUT/'validation'/f'pilot{count}.json';verified=gate.exists() and read(gate).get('reports_and_viewer_verified',False)
  if verified:
   proposal_gate=PARENT/'validation'/f'pilot{count}.json'
   if proposal_gate.exists():
    record=read(proposal_gate);record.update(verified_scope='candidate_proposal_only',full_pipeline_complete=True,end_to_end_validation=f'end_to_end/validation/pilot{count}.json');write(proposal_gate,record)
  lines=[f'# End-to-end completion · pilot{count}','',f'후속 계산 완료 {len(data)} / {count} query. 보고서·viewer 최종 검증: '+('통과' if verified else '대기 중')+'. 후보 생성의 완료 수와 구분한다.','', '[방법·예산별 추가 연산](../APPENDIX_COMPUTE.md)', '', '## 최종 답변의 실제 성능','','검증 모델이 고른 구간(verified_selection), 후보만 받은 최종 생성(proposal_only), 검증 점수까지 받은 최종 생성(reviewed)을 구분한다. mIoU와 R은 높을수록 좋으며 괄호는 같은 query의 greedy 대비Δpp다. Candidate oracle은 GT를 본 후보 품질 진단이며 final answer가 아니다.','', '| 방법 | 예산 | 결과 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |','|---|---:|---|---:|---:|---:|---:|---:|',f'| greedy | — | 기준 답변 | {len(data)} | '+mcell(base,base)+' |']
  for method in METHODS:
   for budget in BUDGETS:
    key=f'{method}:{budget}'
    for variant in ('verified_selection','proposal_only','reviewed','candidate_oracle'):
     vals=[]
     for row,d,q in data:
      c=d['conditions'][key];gt=row['span'][0]
      if variant=='candidate_oracle':v=max([iou(d['spans'][m['id']],gt) for m in c['members']]+[0.])
      else:
       pair=d['spans'][c['verified_selected_id']] if variant=='verified_selection' and c['verified_selected_id'] is not None else None
       if variant in ('proposal_only','reviewed'):pair=d['generations'][c[variant]]['span'] if d['generations'][c[variant]]['valid'] else None
       v=iou(pair,gt)
      vals.append(v)
     lines.append(f'| {method} | {budget} | {variant} | {len(data)} | '+mcell(metrics(vals),base)+' |')
  lines+=['','## 답변 변화와 최종 생성 비용','','Changed는 정확한 clipped span의 변화이며 개선률이 아니다. Invalid는 EOS·parsing·구간 유효성 실패다. 비용은 실제 incremental 생성 연산이며 같은 prompt 재사용을 여러 번 실행한 것처럼 합산하지 않는다.','', '| 방법 | 예산 | greedy→reviewed 변화 | proposal_only→reviewed 변화 | reviewed 무효 | proposal_only 평균 입력 | reviewed 평균 입력 | reviewed 평균 wall s |','|---|---:|---:|---:|---:|---:|---:|---:|']
  for method in METHODS:
   for budget in BUDGETS:
    entries=[(d['generations'][d['conditions'][f'{method}:{budget}']['proposal_only']],d['generations'][d['conditions'][f'{method}:{budget}']['reviewed']],q) for row,d,q in data];n=len(entries)
    lines.append(f'| {method} | {budget} | {sum(a["span"]!=q["greedy"]["span"] for c,a,q in entries)} / {n} | {sum(c["span"]!=a["span"] for c,a,q in entries)} / {n} | {sum(not a["valid"] for c,a,q in entries)} / {n} | {f(sum(c["cost"]["input_tokens"] for c,a,q in entries)/n,1)} | {f(sum(a["cost"]["input_tokens"] for c,a,q in entries)/n,1)} | {f(sum(a["cost"]["wall_seconds"] for c,a,q in entries)/n,3)} |')
  lines+=['','## 독립 구간 검증·semantic의 추가 비용','','서로 다른 숫자 구간 수와 실제 고유 crop 수를 구분한다. Batch validation의 반복 비용도 아래 합계에 포함한다. 통계용 semantic 계산을 final 선택 비용으로 숨기지 않는다.','',f'구간 합 {sum(len(d["spans"]) for r,d,q in data)}개, 고유 crop 합 {sum(len(d["crops"]) for r,d,q in data)}개.','', '| 단계 | 실제 forward 합 | 실제 입력 token 합 | Padded token 합 | CUDA 합 s | Model wall 합 s |','|---|---:|---:|---:|---:|---:|']
  for kind in ('verifier','instruct','embedding'):
   batches=[b for row,d,q in data for b in d['phase_costs'][kind]['batches']+d['phase_costs'][kind]['validation']];lines.append('| '+kind+' | '+' | '.join(str(sum(b[k] for b in batches)) for k in ('forward_calls','input_tokens','padded_tokens'))+' | '+f(sum(b['cuda_seconds'] for b in batches),3)+' | '+f(sum(b['wall_seconds'] for b in batches),3)+' |')
  total_wall=sum(d['postprocess_wall_seconds'] for r,d,q in data);final_wall=sum(g['cost']['wall_seconds'] for r,d,q in data for g in d['generations'].values());semantic_wall=sum(b['wall_seconds'] for r,d,q in data for kind in ('instruct','embedding') for b in d['phase_costs'][kind]['batches']+d['phase_costs'][kind]['validation'])
  lines+=['',f'완료 query의 후속 처리 wall 합은 {total_wall:.3f}s ({total_wall/3600:.3f}시간), query당 평균은 {total_wall/len(data):.3f}s다. 이는 세 방법×세 예산과 semantic 진단을 공유 실행한 전체 작업량이며 단일 조건의 단독 latency가 아니다. 이 중 semantic 모델 forward wall 합은 {semantic_wall:.3f}s ({semantic_wall/3600:.3f}시간), 고유 final generation wall 합은 {final_wall:.3f}s ({final_wall/3600:.3f}시간)다. CPU 입력 준비·공유 prefill 등은 각각의 model forward wall과 구분한다. 중단 전에 저장되지 못한 작업의 비용은 아래 재개 기록의 제한을 따른다.','']
  lines+=['', '## 시간 그룹과 semantic 차이','','Cosine은 같은 모델의 벡터끼리 비교한다. 1에 가까울수록 유사하며, 높은 유사도가 같은 사건의 확정 근거는 아니다. 아래 within−between은 시간 그룹 내부 mean cosine에서 그룹 사이 mean cosine을 뺀 값이다. 양수는 이 embedding이 시간 그룹의 차이와 같은 방향으로 구분한다는 진단이며 GT 검증 점수가 아니다. 그룹이 하나뿐이면 between은 정의되지 않는다.','', '| 방법 | 예산 | Pooling | 관측 pair 수 합 | 평균 cosine (유효query n) | 평균 within−between (유효query n) | 평균 시간 그룹 수 |','|---|---:|---|---:|---:|---:|---:|']
  for method in METHODS:
   for budget in BUDGETS:
    cs=[d['conditions'][f'{method}:{budget}'] for r,d,q in data]
    for pool in ('instruct_visual_mean','instruct_last','embedding_last'):
     ss=[c['semantic'].get(pool) for c in cs if c['semantic'].get(pool)];means=[s['all']['mean'] for s in ss if s['all']['mean'] is not None];gaps=[s['within_mode']['mean']-s['between_modes']['mean'] for s in ss if s['within_mode']['mean'] is not None and s['between_modes']['mean'] is not None]
     lines.append(f'| {method} | {budget} | {pool} | {sum(s["all"]["n"] for s in ss)} | {f(sum(means)/len(means) if means else None)} (n={len(means)}) | {f(sum(gaps)/len(gaps) if gaps else None)} (n={len(gaps)}) | {f(sum(len(c["modes"]["groups"]) for c in cs)/len(cs),2)} |')
  recovery=OUT/'metadata/recovery.json'
  if recovery.exists():
   event=read(recovery);lines+=['','## 중단·재개 기록','',f'완료된 {event["completed_before"]}개 query를 보존하고 {event["resume_key"]}부터 재개했다. 후보1개인 조건의 빈 pair mask 처리 오류를 수정했으며 과거 다중 후보 통계는 그대로다. 해당 query에서 저장되지 못했던 semantic 계산은 다시 실행했다. 위 비용 합은 checkpoint에 남은 계산 기준이며, 중단 전에 저장되지 못한 semantic 작업의 CUDA 비용까지 복원한 수치가 아니다. 전체 worker 경과 시간에는 중단 전 작업 시간도 포함한다. 이후부터 모델별 scalar checkpoint를 저장해 재개 시 완료된 모델을 반복하지 않는다.','']
  lines+=['','## 해석 범위','','본 실험은 모든 후보의 독립 영상 검증과 실제 final generation을 수행한다. 후속 계산이 완료되지 않은 query를 최종 성능 평균에 넣지 않는다. Semantic은 진단 전용이며 pruning·후보 선정·score에 사용하지 않는다. 관측 interval entropy·variance·MI는 완성된 valid descendant에 조건부이며 미탐색 분포의 정확한 값이 아니다. 비용은 동시 실행 환경에서의 실제 측정이며 FLOPs 또는 전용 GPU latency와 동일시하지 않는다.','', '[실행 명세·지표 정의·공개 구현 출처](SPEC.md) · [방법·예산별 추가 연산](../APPENDIX_COMPUTE.md)']
  (result/'REPORT.md').write_text('\n'.join(lines)+'\n');(result/'SPEC.md').write_text(specs(count));web=OUT/'web';web.mkdir(exist_ok=True)
  sys.path.insert(0,str(ROOT.parent/'_lib'))
  from report_appendices import split_report
  split_report(result/'REPORT.md')
  for suffix in ('js','css'):
   source=Path(__file__).with_name('probability_completion_viewer.'+suffix);target=web/('viewer.'+suffix);tmp=target.with_name(target.name+'.tmp');tmp.write_bytes(source.read_bytes());tmp.replace(target)
  template=Path(__file__).with_name('probability_completion_viewer.html').read_text();cfg=dict(store=os.path.relpath(OUT,result),parent=os.path.relpath(PARENT,result),catalog=os.path.relpath(artifact/'web/samples.json',result),root=os.path.relpath(ROOT.parent,result))
  replaces={'__COUNT__':str(count),'__THEME__':os.path.relpath(ROOT.parent/'_lib/experiment_theme.css',result),'__STYLE__':os.path.relpath(web/'viewer.css',result)+'?sha='+sha(web/'viewer.css')[:12],'__SCRIPT__':os.path.relpath(web/'viewer.js',result)+'?sha='+sha(web/'viewer.js')[:12],'__CONFIG__':json.dumps(cfg)}
  for old,new in replaces.items():template=template.replace(old,new)
  (result/'detail.html').write_text(template)
 from probability_compute_appendix import build as build_compute_appendix
 build_compute_appendix()
 print('END_TO_END_REPORT',active,len(data),flush=True)
if __name__=='__main__':
 import fcntl
 with (OUT/'metadata/report.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX);report()
