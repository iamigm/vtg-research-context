"""Additional rollout diagnostics from saved completions only; no model calls."""
import collections,csv,hashlib,json,statistics
from pathlib import Path

THRESHOLDS=[.3,.5,.7]
SELECTORS=['max_probability','medoid','probability_weighted_medoid']

def average(xs):return statistics.mean(xs) if xs else None
def number(x,digits=3):return '—' if x is None else f'{x:.{digits}f}'
def fraction(n,d):return f'{n}/{d} ({100*n/d:.2f}%)' if d else f'{n}/0 (—)'
def role_name(role):
 parts=role.split('.');endpoint={'start':'시작 시각','end':'종료 시각'}.get(parts[0],parts[0])
 if parts[-1]=='clock_separator':return endpoint+' · 시:분:초 구분 콜론'
 unit={'seconds':'초 필드','minutes':'분 필드','hours':'시 필드'}.get(parts[1],parts[1])
 position={'decimal_separator':'소수점 자체','10^2':'백의 자리','10^1':'십의 자리','10^0':'일의 자리','10^-1':'소수 첫째 자리'}.get(parts[-1],parts[-1])
 return f'{endpoint} · {unit} · {position}'
def save_csv(path,rows):
 if not rows:return
 with path.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def transitions(group,threshold,oracle):
 good=[x['baseline_iou']>=threshold for x in group]
 chosen=[x['iou']>=threshold for x in group]
 available=[oracle[x['sample_key']]>=threshold for x in group]
 assert all(not g or a for g,a in zip(good,available))
 assert all(not s or a for s,a in zip(chosen,available))
 fixed=sum(not g and s for g,s in zip(good,chosen))
 harmed=sum(g and not s for g,s in zip(good,chosen))
 opportunities=sum(not g and a for g,a in zip(good,available))
 failed=sum(a and not s for a,s in zip(available,chosen))
 total=len(group)
 assert sum(chosen)-sum(good)==fixed-harmed
 return dict(n=total,baseline_wrong=total-sum(good),baseline_correct=sum(good),
  corrected=fixed,harmed=harmed,oracle_correct=sum(available),selection_failures=failed,
  new_answer_opportunities=opportunities,new_answer_misses=opportunities-fixed,
  recovery_rate=fixed/(total-sum(good)) if total>sum(good) else None,
  harm_rate=harmed/sum(good) if sum(good) else None,
  selection_failure_rate=failed/sum(available) if sum(available) else None,
  new_answer_adoption_rate=fixed/opportunities if opportunities else None,
  net_success_pp=100*(fixed-harmed)/total)

def diagnostic_sections(records,rows,out):
 # Import after the report module has initialized, avoiding a circular import.
 from deterministic_rollout300_report import interval,overlap
 from deterministic_rollout300 import write,BUDGETS
 n=len(records);bykey={r['sample']['sample_key']:r for r in records}
 index={(x['budget'],x['method']):[] for x in rows}
 for x in rows:index[x['budget'],x['method']].append(x)
 results=[];lines=['','## 오답 복구·정답 훼손·후보 선택 실패','',
  '각 기준 τ에서 G는 greedy IoU≥τ, S는 최종 선택 IoU≥τ, O는 후보 중 하나라도 IoU≥τ인 질문이다. 아래 세 선택기는 GT를 쓰지 않는다. 성공 판정에만 GT를 사용한다. 각 행은 전체 질문을 대상으로 하며 분모가0이면 —로 표시한다.', '',
  '- 오답 복구율 = |¬G∧S| / |¬G|. 기존 오답을 고친 비율로 높을수록 좋다.',
  '- 정답 훼손율 = |G∧¬S| / |G|. 기존 정답을 잃은 비율로 낮을수록 좋다.',
  '- 선택 실패율 = |O∧¬S| / |O|. 후보 안에 정답이 있는데도 선택에 실패한 비율로 낮을수록 좋다.',
  '- 새 정답 채택률 = |¬G∧S| / |¬G∧O|. greedy는 틀렸고 새 정답 후보가 생긴 경우 그 답을 고른 비율로 높을수록 좋다. 미채택률은1−채택률이다.',
  '- 순증 성공률(pp) = 100×(복구 수−훼손 수)/N. 복구와 훼손을 상쇄한 전체 성능 변화다.', '']
 for threshold in THRESHOLDS:
  lines += [f'### IoU≥{threshold}', '',
   '| 후보 상한 | 선택 | 복구 수/기존 오답 | 훼손 수/기존 정답 | 선택 실패/정답 후보 존재 | 새 정답 채택/기회 | 순증(pp) |',
   '|---|---|---:|---:|---:|---:|---:|']
  for budget in BUDGETS[1:]:
   oracle={x['sample_key']:x['iou'] for x in index[budget,'oracle']}
   for method in SELECTORS:
    d=transitions(index[budget,method],threshold,oracle)
    results.append(dict(budget=budget,method=method,threshold=threshold,**d))
    lines.append(f'| {budget} | {method} | {fraction(d["corrected"],d["baseline_wrong"])} | {fraction(d["harmed"],d["baseline_correct"])} | {fraction(d["selection_failures"],d["oracle_correct"])} | {fraction(d["corrected"],d["new_answer_opportunities"])} | {d["net_success_pp"]:+.2f} |')
 d=next(x for x in results if x['budget']==32 and x['method']=='probability_weighted_medoid' and x['threshold']==.5)
 lines += ['',f'IoU≥.5·32후보·확률가중 medoid 기준으로 기존 오답 {d["baseline_wrong"]}개 중 {d["corrected"]}개를 고쳤지만, 기존 정답 {d["baseline_correct"]}개 중 {d["harmed"]}개를 잃었다. 후보에 정답이 있는 {d["oracle_correct"]}개 중 {d["selection_failures"]}개는 최종 선택이 틀렸다. 새 정답 후보가 생긴 {d["new_answer_opportunities"]}개 중 {d["corrected"]}개를 채택했다.','']
 save_csv(out/'metrics/recovery_and_selection.csv',results)

 # Absolute and signed endpoint errors, normalized by GT length when valid.
 boundary=[]
 lines += ['## 시작·종료 경계의 오차와 편향','',
  '오차는 예측 시각−GT 시각(초)이다. 양수는 늦게, 음수는 일찍 예측함을 뜻한다. MAE=평균 |오차|, 정규화 MAE=평균(|오차|/GT 구간 길이)로 낮을수록 좋다. MAE·편향은 파싱 가능하고 양의 길이인 유효 응답에 조건부이며 유효 N을 병기한다. 영상 길이 초과 응답도 기존 IoU 평가와 동일하게 포함한다. 이 값은 같은 사건에 조건부인 refinement 성능이 아니다.', '',
  '| 후보 상한 | 선택 | 유효 N | 시작 MAE(초) | 종료 MAE(초) | 시작 편향(초) | 종료 편향(초) | 시작 정규화 MAE | 종료 정규화 MAE |',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for budget in BUDGETS:
  for method in ['greedy']+SELECTORS+['oracle']:
   if budget>1 and method=='greedy':continue
   if budget==1 and method!='greedy':continue
   group=index[budget,method];valid=[];hits={t:0 for t in [1,2,5]}
   for x in group:
    rec=bykey[x['sample_key']];c=rec['result']['candidates'][x['selected_index']-1]
    pair=interval(c)
    if pair is None:continue
    gt=rec['sample']['span'][0];length=gt[1]-gt[0];assert length>0
    se,ee=pair[0]-gt[0],pair[1]-gt[1];valid.append((se,ee,length))
    for t in hits:hits[t]+=abs(se)<=t and abs(ee)<=t
   d=dict(budget=budget,method=method,n=n,valid_n=len(valid),
    start_mae=average([abs(v[0]) for v in valid]),end_mae=average([abs(v[1]) for v in valid]),
    start_bias=average([v[0] for v in valid]),end_bias=average([v[1] for v in valid]),
    start_normalized_mae=average([abs(v[0])/v[2] for v in valid]),
    end_normalized_mae=average([abs(v[1])/v[2] for v in valid]),
    start_early=sum(v[0]<-1e-9 for v in valid),start_late=sum(v[0]>1e-9 for v in valid),
    end_early=sum(v[1]<-1e-9 for v in valid),end_late=sum(v[1]>1e-9 for v in valid),
    both_within1s=hits[1],both_within2s=hits[2],both_within5s=hits[5])
   boundary.append(d)
   lines.append(f'| {budget} | {method} | {len(valid)} | '+' | '.join(number(d[k]) for k in ['start_mae','end_mae','start_bias','end_bias','start_normalized_mae','end_normalized_mae'])+' |')
 lines += ['', '시작과 종료 모두 허용 오차 안에 드는 성공률도 함께 측정한다. 아래 성공률의 분모는 유효 응답 수가 아닌 전체 N이며, 파싱/완성 실패는 실패다. 이른/늦은 예측 비율은 유효 응답 수를 분모로 쓴다. 정확히 일치한 경우는 이른/늦은 쪽 어디에도 포함하지 않는다.', '',
  '| 후보 상한 | 선택 | 양쪽 ±1초 | 양쪽 ±2초 | 양쪽 ±5초 | 시작 일찍/유효 N | 시작 늦게/유효 N | 종료 일찍/유효 N | 종료 늦게/유효 N |',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for d in boundary:
  lines.append(f'| {d["budget"]} | {d["method"]} | '+' | '.join(fraction(d[k],n) for k in ['both_within1s','both_within2s','both_within5s'])+' | '+' | '.join(fraction(d[k],d['valid_n']) for k in ['start_early','start_late','end_early','end_late'])+' |')
 save_csv(out/'metrics/boundary_errors.csv',boundary)

 # Matched parent-child comparisons for the actual forced-token intervention.
 branches=[];role_groups=collections.defaultdict(list);endpoint_groups=collections.defaultdict(list)
 for record in records:
  gt=record['sample']['span'][0];cs=record['result']['candidates'];key=record['sample']['sample_key']
  greedy=interval(cs[0]);greedy_iou=overlap(greedy,gt) if greedy else 0.
  for c in cs[1:]:
   b=c['branch'];assert 1<=b['parent']<c['index']
   parent=cs[b['parent']-1];a,z=interval(parent),interval(c)
   aiou=overlap(a,gt) if a else 0.;ziou=overlap(z,gt) if z else 0.
   roles=b['roles'];endpoints=sorted({r.split('.')[0] for r in roles})
   d=dict(sample_key=key,candidate=c['index'],parent=b['parent'],
    source_roles=';'.join(roles),target_roles=';'.join(b.get('target_roles',[])),
    endpoint=';'.join(endpoints),parent_iou=aiou,child_iou=ziou,delta_iou=ziou-aiou,
    greedy_iou=greedy_iou,delta_vs_greedy=ziou-greedy_iou,
    start_shift=z[0]-a[0] if a and z else None,end_shift=z[1]-a[1] if a and z else None,
    raw_alternative_probability=b['raw_probability'])
   branches.append(d)
   for role in roles:role_groups[role].append(d)
   endpoint_groups[d['endpoint']].append(d)
 def summarize(group):
  valid=[x for x in group if x['start_shift'] is not None]
  return dict(branches=len(group),questions=len({x['sample_key'] for x in group}),
    improved=sum(x['delta_iou']>1e-9 for x in group),worsened=sum(x['delta_iou']<-1e-9 for x in group),
    unchanged=sum(abs(x['delta_iou'])<=1e-9 for x in group),mean_delta_iou=average([x['delta_iou'] for x in group]),
    mean_delta_vs_greedy=average([x['delta_vs_greedy'] for x in group]),
    improved_vs_greedy=sum(x['delta_vs_greedy']>1e-9 for x in group),
    worsened_vs_greedy=sum(x['delta_vs_greedy']<-1e-9 for x in group),
    parent_is_greedy=sum(x['parent']==1 for x in group),
    valid_pairs=len(valid),mean_abs_start_shift=average([abs(x['start_shift']) for x in valid]),
    mean_abs_end_shift=average([abs(x['end_shift']) for x in valid]),
    end_changed=sum(abs(x['end_shift'])>1e-6 for x in valid),
    end_changed_over1s=sum(abs(x['end_shift'])>1 for x in valid))
 lines += ['', '## 분기 위치별 개선과 시작→종료 변화','',
  '여기서 분기는 생성 도중 한 토큰을 다른 토큰으로 바꾸고 나머지를 끝까지 생성하는 것이다. 부모 후보는 바꾸기 전 완성 답, 자식 후보는 바꾼 뒤 완성 답이다. 부모는 최초 greedy일 수도 있고 탐색 중 생성한 다른 후보일 수도 있다.', '',
  '표의 내부 코드는 분기 전 부모 답에서 그 토큰이 차지한 위치를 적은 이름이다. `start`는 시작 시각, `end`는 종료 시각이다. `seconds`는 초 필드, `minutes`는 분 필드, `hours`는 시 필드다. `10^2`, `10^1`, `10^0`, `10^-1`은 각각 백·십·일·소수 첫째 자리다. 예를 들어 `135.7`에서 `3`은 십의 자리이고 `7`은 소수 첫째 자리다.', '',
  '`decimal_separator`는 소수점 아래 숫자가 아니라 소수점 기호 `.` 자체다. 따라서 `end.seconds.decimal_separator`는 종료 시각의 초 필드에서 소수점 자체를 바꾼 분기를 뜻한다. `clock_separator`는 예외적인 시:분:초 표기의 구분 콜론 `:`이다. 예를 들어 `2:11`은 분 필드2와 초 필드11로 나눈다.', '',
  '원자료의 `source_roles`는 분기 전 위치 이름, `target_roles`는 변경 후 완성 답에서 같은 토큰의 위치 이름이다. 소수점 대신 숫자를 넣으면 정수 부분이 길어질 수 있으므로 분기 전후 자릿수 역할이 같다는 뜻은 아니다. 표는 한국어 이름과 내부 코드를 함께 표시한다.', '',
  '개선/악화 비율과 부모 대비 평균 ΔIoU는 자식 IoU−부모 IoU 기준이다. 최초 greedy 대비 평균 ΔIoU는 자식 IoU−최초 greedy IoU로 별도 표시한다. 개선/악화는 차이가 ±1e−9를 넘는 경우이고 실패 응답의 IoU는0이다. 여러 후보를 최종 선택하기 전의 후보별 진단이므로 실제 선택 성능과 구분한다.', '',
  '한 질문에서 여러 분기가 나오며 각 분기는 서로 독립 표본이 아니다. 여러 자릿수를 담는 토큰은 각 역할 행에 중복 집계될 수 있다. 아래 비율은 이 탐색 정책이 생성한 분기들에 조건부인 빈도이며, 자릿수의 일반적인 인과 효과나 모델 전체 분포의 확률로 해석하지 않는다.', '',
  '| 부모의 분기 위치: 한국어 이름 (내부 코드) | 분기 수 | 고유 질문 수 | 부모 대비 개선/분기 | 부모 대비 악화/분기 | 변화 없음/분기 | 부모 대비 평균 ΔIoU | 최초 greedy 대비 평균 ΔIoU |',
  '|---|---:|---:|---:|---:|---:|---:|---:|']
 role_stats=[]
 for role,group in sorted(role_groups.items()):
  d=dict(role=role,**summarize(group));role_stats.append(d)
  lines.append(f'| {role_name(role)} (`{role}`) | {d["branches"]} | {d["questions"]} | {fraction(d["improved"],d["branches"])} | {fraction(d["worsened"],d["branches"])} | {fraction(d["unchanged"],d["branches"])} | {d["mean_delta_iou"]:+.4f} | {d["mean_delta_vs_greedy"]:+.4f} |')
 dot_group=role_groups.get('end.seconds.decimal_separator',[])
 if dot_group:
  d=summarize(dot_group)
  numeric=sum(bool(x['target_roles']) and all(r.split('.')[-1].startswith('10^') for r in x['target_roles'].split(';')) for x in dot_group)
  lines += ['', '### 종료 시각 소수점 분기의 해석', '',
   f'종료 시각의 소수점 위치를 바꾼 분기는 {d["questions"]}질문에서 {d["branches"]}개였다. 그중 {numeric}개는 변경 후 해당 토큰이 숫자 자릿수가 됐다. 소수점을 숫자로 바꾸면 예를 들어 `1.3`이 `107.3`처럼 길어질 수 있다. 이는 소수 첫째 자리를 미세 조정하는 것과 다르며, 정수 부분을 너무 일찍 끝낸 후보를 고치는 분기일 수 있다.', '',
   f'부모보다 좋아진 분기는 {d["improved"]}/{d["branches"]}, 부모 대비 평균 ΔIoU는 {d["mean_delta_iou"]:+.4f}다. 다만 부모가 최초 greedy인 것은 {d["parent_is_greedy"]}개이고 나머지는 다른 탐색 후보다. 최초 greedy와 비교하면 좋아진 분기는 {d["improved_vs_greedy"]}개, 악화된 분기는 {d["worsened_vs_greedy"]}개이며 평균 ΔIoU는 {d["mean_delta_vs_greedy"]:+.4f}다. 따라서 부모 대비 평균이 양수라는 사실을 곧바로 최초 greedy보다 유리한 전략이라고 해석하면 안 된다.', '',
   '다른 자릿수의 부모 대비 평균이 음수라는 것은 그 자릿수에 개선 사례가 없다는 뜻이 아니다. 좋아진 분기보다 나빠진 분기의 횟수나 악화 크기가 더 크게 반영된 평균이다. 각 행의 개선/악화 개수와 최초 greedy 대비 값도 함께 봐야 한다.', '',
   '| 질문 ID | GT 구간(초) | 부모 후보 응답 | 분기 후 응답 | 부모 대비 ΔIoU |',
   '|---|---|---|---|---:|']
  examples=sorted(dot_group,key=lambda x:-x['delta_iou'])[:2]+sorted(dot_group,key=lambda x:x['delta_iou'])[:1]
  for x in examples:
   rec=bykey[x['sample_key']];cs=rec['result']['candidates']
   parent=cs[x['parent']-1]['answer'].replace('|','\\|').replace('\n',' ')
   child=cs[x['candidate']-1]['answer'].replace('|','\\|').replace('\n',' ')
   lines.append(f'| {x["sample_key"]} | {rec["sample"]["span"][0]} | {parent} | {child} | {x["delta_iou"]:+.4f} |')
 lines += ['', '시작을 바꿨을 때 종료도 바뀌는 비율은 다음과 같다. 변화량은 유효한 부모·자식 쌍만 비교하며 시간 수치 차이1e−6초 초과를 변경으로 센다.', '',
  '| 분기 대상 | 전체 분기 | 유효 비교 쌍 | 평균 시작 변화 절댓값(초) | 평균 종료 변화 절댓값(초) | 종료 변경/유효 쌍 | 종료 1초 초과 변경/유효 쌍 |',
  '|---|---:|---:|---:|---:|---:|---:|']
 coupling=[]
 for endpoint,group in sorted(endpoint_groups.items()):
  d=dict(endpoint=endpoint,**summarize(group));coupling.append(d)
  name={'start':'시작 시각','end':'종료 시각'}.get(endpoint,endpoint)
  lines.append(f'| {name} | {d["branches"]} | {d["valid_pairs"]} | {number(d["mean_abs_start_shift"])} | {number(d["mean_abs_end_shift"])} | {fraction(d["end_changed"],d["valid_pairs"])} | {fraction(d["end_changed_over1s"],d["valid_pairs"])} |')
 save_csv(out/'metrics/branch_parent_comparisons.csv',branches)
 save_csv(out/'metrics/branch_role_summary.csv',role_stats)
 save_csv(out/'metrics/start_end_coupling.csv',coupling)

 efficiency=[]
 lines += ['', '## 후보를 더 만들 때 추가 비용과 얻은 정답','',
  '인접 예산 1→4→8→16→32를 같은 질문에서 비교한다. 추가 decoder 토큰은 전체 질문의 계산량 증가 합이다. 새 정답 수는 이전 예산에서 틀렸으나 현재 맞힌 수, 잃은 정답 수는 그 반대다. 순증=새 정답−잃은 정답이다. 1개 복구당 비용=추가 토큰/새 정답 수, 순증1개당 비용=추가 토큰/순증이며 낮을수록 좋다. 순증≤0이면 순개선 없음으로 표시한다. 분모가0인 경우 —를 표시한다. 선택기를 바꿔 새 추론을 한 것이 아니라 같은 후보 집합에서 선택만 달리한 비교다.', '']
 for threshold in THRESHOLDS:
  lines += [f'### IoU≥{threshold}: 인접 예산의 효율', '',
   '| 예산 증가 | 선택 | 추가 토큰 합 | 새 정답 | 잃은 정답 | 순증 | 순증(pp) | 1개 복구당 토큰 | 순증1개당 토큰 |',
   '|---|---|---:|---:|---:|---:|---:|---:|---:|']
  for prev,budget in zip(BUDGETS,BUDGETS[1:]):
   for method in SELECTORS+['oracle']:
    before={x['sample_key']:x for x in index[prev,method]};after=index[budget,method]
    cost=sum(x['decoder_tokens']-before[x['sample_key']]['decoder_tokens'] for x in after)
    assert cost>=0
    gained=sum(x['iou']>=threshold and before[x['sample_key']]['iou']<threshold for x in after)
    lost=sum(x['iou']<threshold and before[x['sample_key']]['iou']>=threshold for x in after)
    net=gained-lost
    d=dict(previous_budget=prev,budget=budget,method=method,threshold=threshold,n=n,
     additional_tokens=cost,gained=gained,lost=lost,net=net,net_pp=100*net/n,
     tokens_per_recovered=cost/gained if gained else None,tokens_per_net=cost/net if net>0 else None)
    efficiency.append(d)
    lines.append(f'| {prev}→{budget} | {method} | {cost} | {gained} | {lost} | {net:+d} | {d["net_pp"]:+.2f} | {number(d["tokens_per_recovered"],1)} | '+(number(d['tokens_per_net'],1) if net>0 else '순개선 없음')+' |')
 save_csv(out/'metrics/marginal_efficiency.csv',efficiency)

 lines += ['', '## 영상 검토가 필요한 지표: 아직 측정하지 않음','',
  '- 다른 사건 후보 발견율: greedy가 잘못된 사건 occurrence를 가리킨 질문 중 정답 occurrence 후보가 생긴 비율. 영상 검토로 사건 occurrence 라벨을 정해야 한다.',
  '- 같은 사건 내 경계 개선율: 부모·자식이 같은 occurrence를 가리키는 분기 중 시작/종료 MAE가 감소한 비율과 감소량. 위 전체 경계 MAE로 대체하지 않는다.',
  '- 사건 이탈률: 올바른 occurrence였던 답이 분기/선택 후 다른 occurrence로 바뀐 비율. 단순한 시간 거리나 낮은 두 후보 IoU만으로 판정하지 않는다.',
  '현재 기록에는 occurrence 라벨이 없어 위 세 지표의 수치는 미측정이다. 조건부 종료 변화는 측정했지만 다른 사건 발견 또는 같은 사건 refinement 성공으로 단정하지 않는다.', '']
 summary=dict(queries=n,recovery=results,boundary=boundary,role_summary=role_stats,
    coupling=coupling,efficiency=efficiency,event_metrics='unmeasured: no occurrence labels')
 write(out/'metrics/additional_diagnostics.json',summary)
 write(out/'validation/additional_diagnostics.json',dict(passed=True,queries=n,
   recovery_rows=len(results),boundary_rows=len(boundary),branches=len(branches),
   transition_identities_verified=True,oracle_contains_greedy=True,actual_selection_within_oracle=True,
   matched_parent_indices_verified=True,inference_rerun=False))
 # Preserve the original inference snapshots and version current reporting separately.
 source_dir=Path(__file__).parent;provenance=[]
 for name in ['deterministic_rollout_diagnostics.py','deterministic_rollout300_report.py']:
  source=source_dir/name;destination=out/'code/reporting'/name
  destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.read_bytes())
  provenance.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
 write(out/'metadata/reporting_code.json',dict(files=provenance,scope='report regeneration only; saved inference unchanged'))
 return lines
