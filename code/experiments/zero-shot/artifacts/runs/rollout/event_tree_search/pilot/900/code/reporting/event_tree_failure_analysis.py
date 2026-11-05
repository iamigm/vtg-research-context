"""CPU-only audit of proposal admissibility, visual scoring, selection and search failure."""
from event_tree_core import *
import collections,itertools,statistics,sys

def valid(span):return bool(span and len(span)==2 and all(math.isfinite(x) for x in span) and 0<=span[0]<span[1])
def score(span,gt):return iou(span,gt) if valid(span) else 0.
def mean(values):return sum(values)/len(values) if values else 0.
def median(values):return statistics.median(values) if values else None
def pct(a,b):return 100*a/b if b else 0.
def pool_oracle(nodes,gt):return max((score(n.get('span'),gt) for n in nodes),default=0.)

def analyze(n=900):
 cohort=read(COHORT)[:n];dest=ROOT/'results'/FAMILY/'pilot'/str(n);out=ROOT/'artifacts/runs'/FAMILY/'pilot'/str(n);stats=collections.defaultdict(collections.Counter);ratios=collections.defaultdict(list);proposals=collections.Counter();overshoots=[];value_pairs=collections.Counter();binary=collections.defaultdict(list);per_query=[];cases=collections.defaultdict(list);unique_pair_stats=collections.Counter();native_check=[];qconcentration=[];method_equal=collections.Counter();baseline_scores=[];strict_scores=[]
 reference='nucleus_frontier@8192'
 for row in cohort:
  key=row['sample_key'];gt=row['span'][0];duration=row['duration'];old=read(SOURCE/'rollouts'/f'{key}.json');data=read(OUT/'responses'/f'{key}.json');events=data['proposals']['events'];records=data['model_calls'];greedy=pair(old['result']['candidates'][0]);base=score(greedy,gt);baseline_scores.append(base);strict_scores.append(base if usable(greedy,duration) else 0.)
  def sample_case(**extras):return dict(key=key,query=row['query'],dataset=row['dataset'],duration=duration,gt=gt,greedy=greedy,greedy_iou=base,**extras)
  if valid(greedy) and not usable(greedy,duration):
   excess=max(0.,greedy[1]-duration);overshoots.append(excess);cases['duration'].append(sample_case(excess=excess))
  for record in records.values():
   if record.get('option_mass') is not None:binary[record['kind']].append(record['option_mass'])
  initial=[dict(span=e['span'],value=e['assessment']['value'],event=e['assessment']['event']) for e in events if usable(e['span'],duration)]
  for a,b in itertools.combinations(initial,2):
   difference=score(a['span'],gt)-score(b['span'],gt)
   if abs(difference)<.1:continue
   delta=a['value']-b['value'];value_pairs['n']+=1;value_pairs['agreement']+=.5 if abs(delta)<1e-12 else int(delta*difference>0)
  old8=[pair(c) for c in old['result']['candidates'][:8]];positive=[e for e in events if valid(e['span'])];legal=[e for e in positive if usable(e['span'],duration)];eligible=[e for e in legal if e['assessment']['eligible']]
  first=pool_oracle([dict(span=p) for p in old8],gt);raw=pool_oracle(positive,gt);admissible=pool_oracle(legal,gt);gated=pool_oracle(eligible,gt)
  proposals.update(query=1,old8=first,raw_all=raw,duration_valid=admissible,gated=gated,positive_count=len(positive),legal_count=len(legal),eligible_count=len(eligible),zero_eligible=int(not eligible),duration_lost_best=int(raw>admissible+1e-9),critic_lost_best=int(admissible>gated+1e-9),duration_lost50=int(raw>=.5 and admissible<.5),critic_lost50=int(admissible>=.5 and gated<.5),all_raw_bad=int(raw<.5))
  if not eligible:
   proposals['zero_eligible_no_legal' if not legal else 'zero_eligible_with_legal']+=1
  leading=[p['span'] for p in data['proposals']['leading_digit_paths']];sem=[e['span'] for e in events if any(p['source']=='semantic_proposal' for p in e['provenance'])]
  proposals['leading_paths']+=len(leading);proposals['leading_valid']+=sum(usable(p,duration) for p in leading);proposals['semantic_unique']+=len(sem)
  semantic_record=records[data['proposals']['semantic_call']];proposals['semantic_complete_array']+=isinstance(semantic_record.get('json'),list)
  fallback=pair(semantic_record)
  if not sem and valid(fallback):
   proposals['semantic_plain_timestamp_discarded']+=1;proposals['semantic_plain_timestamp_valid']+=usable(fallback,duration);proposals['semantic_plain_timestamp_recovers50']+=raw<.5 and score(fallback,gt)>=.5
  for label,spans in [('old8',old8),('leading',leading),('semantic',sem),('old8_leading',old8+leading),('old8_semantic',old8+sem)]:
   val=max((score(p,gt) for p in spans),default=0.);proposals['source_oracle_'+label]+=val;proposals['source_r50_'+label]+=int(val>=.5)
  for p in data['proposals']['leading_digit_paths']:
   record=records[p['score_call']];j=len(p['prefix_ids']);native_check.append(abs(math.exp(record['log_probabilities'][j])-p['forced_probability']))
  if eligible:
   peak=max(e['prior_logp'] for e in eligible);weights=[math.exp(e['prior_logp']-peak) for e in eligible];qconcentration.append(max(weights)/sum(weights));qs=[v/sum(weights) for v in weights];entropy=-sum(p*math.log(max(p,1e-300)) for p in qs)
   def support(order):
    chosen=[];mass=0.
    for i in order:
     chosen.append(i);mass+=qs[i]
     if mass>=.95:break
    return set(chosen)
   ns=support(sorted(range(len(qs)),key=lambda i:(-qs[i],i)));ts=support(sorted(range(len(qs)),key=lambda i:(abs(-math.log(max(qs[i],1e-300))-entropy),i)))
   method_equal['same_initial_support']+=ns==ts;method_equal['initial_support_n']+=1;method_equal['initial_nucleus_size']+=len(ns)
  outcomes={}
  for label,r in data['proposal_only'].items():outcomes['proposal_only_'+label]=dict(result=r,control=True)
  for method,variants in data['methods'].items():
   for cap,r in variants.items():outcomes[method+'@'+cap]=dict(result=r,control=False)
  for cap in ('4096','8192'):
   a=data['methods']['nucleus_frontier'][cap];b=data['methods']['typicality_frontier'][cap];method_equal['nucleus_typical_same_'+cap]+=a['span']==b['span']
  query_pairs=set()
  for name,wrapped in outcomes.items():
   r=wrapped['result'];nodes=r['nodes'];selected=nodes[r['selected']] if r['selected'] is not None else None;result_span=r['span'];result_iou=score(result_span,gt);s=stats[name];s['n']+=1;s['iou']+=result_iou;s['failure']+=result_span is None
   all_nodes=nodes[1:];legal_nodes=[x for x in all_nodes if usable(x['span'],duration)];allowed=[x for x in legal_nodes if x['assessment']['eligible'] and not nodes[x['event_id']]['assessment']['blocked']];pool=[nodes[i] for i in r['selection_pool']]
   oa=pool_oracle(all_nodes,gt);ov=pool_oracle(legal_nodes,gt);og=pool_oracle(allowed,gt);op=pool_oracle(pool,gt)
   unique_pool={tuple(x['span']) for x in pool};s['pool_n']+=len(pool);s['pool_unique_n']+=len(unique_pool);s['pool_nonempty']+=bool(pool);s['duplicate_pool']+=len(unique_pool)<len(pool);s['only_one_unique']+=bool(pool) and len(unique_pool)==1
   dedup_candidates={}
   for node in allowed:
    if name.startswith('commit_then_refine') and (not pool or node['event_id']!=pool[0]['event_id']):continue
    identity=tuple(node['span']);previous=dedup_candidates.get(identity)
    if previous is None or (node['assessment']['value'],-node['id'])>(previous['assessment']['value'],-previous['id']):dedup_candidates[identity]=node
   dedup_pool=diversity_select(list(dedup_candidates.values()),4,.1);dedup_oracle=pool_oracle(dedup_pool,gt);s['dedup_pool_oracle']+=dedup_oracle;s['dedup_has50']+=dedup_oracle>=.5
   if name==reference and dedup_oracle>=.7 and op<.3:cases['duplicate_pool'].append(sample_case(method=name,final=result_span,final_iou=result_iou,original_pool=[dict(id=x['id'],span=x['span'],iou=score(x['span'],gt),value=x['assessment']['value']) for x in pool],dedup_pool=[dict(id=x['id'],span=x['span'],iou=score(x['span'],gt),value=x['assessment']['value']) for x in dedup_pool],dedup_pool_oracle=dedup_oracle))
   assert oa+1e-9>=ov>=og-1e-9 and og+1e-9>=op and op+1e-9>=result_iou,(key,name,oa,ov,og,op,result_iou)
   for field,value in dict(oracle_all=oa,oracle_legal=ov,oracle_gate=og,oracle_pool=op,loss_duration=oa-ov,loss_gate=ov-og,loss_pool=og-op,loss_selection=op-result_iou).items():s[field]+=value
   s['pool_has50']+=op>=.5;s['pool_good_but_wrong']+=op>=.5 and result_iou<.5;s['better']+=result_iou>base+.01;s['worse']+=result_iou<base-.01;s['baseline_good']+=base>=.5;s['harmful']+=base>=.5 and result_iou<.5;s['baseline_bad']+=base<.1;s['recovery']+=base<.1 and result_iou>=.5
   if result_span:
    ratio=(result_span[1]-result_span[0])/max(1e-12,gt[1]-gt[0]);ratios[name].append(ratio);s['length_gt2x']+=ratio>2;s['valid']+=1;s['selected_initial']+=selected['source_event'] is not None;s['selected_boundary']+=selected['kind']=='boundary';s['selected_event_switch']+=selected['kind']=='event_switch'
   if not wrapped['control']:
    new=[x for x in nodes[1:] if x['source_event'] is None];seen={tuple(x['span']) for x in nodes[1:] if x['source_event'] is not None and valid(x['span'])}
    for node in new:
     s['generated_nodes']+=1
     if valid(node['span']):
      s['generated_valid']+=1;s['parent_identical']+=node['span']==nodes[node['parent']].get('span');s['repeated_interval']+=tuple(node['span']) in seen;seen.add(tuple(node['span']))
    root=nodes[selected['event_id']] if selected else None
    if selected and selected['kind']=='boundary' and iou(root['span'],selected['span'])>=.5:
     s['refined_same_event_n']+=1;s['refined_root_iou']+=score(root['span'],gt);s['refined_final_iou']+=result_iou
    refs=r['search_calls'];s['search_tokens']+=r['search_cost']['decoder_input_tokens'];s['expansions']+=r['expansions'];s['selector_tokens']+=r['selection_cost']['decoder_input_tokens'];committed={x['call'] for x in nodes[1:] if x.get('call')};generation_indices=[j for j,k in enumerate(refs) if records[k]['kind'] in ('boundary_refinement','new_event_proposal')]
    if generation_indices:
     last=generation_indices[-1];call=refs[last];failed_json=any(a['action']=='proposal_failed' and a.get('call')==call for a in r['actions'])
     if call not in committed and r['stop']=='token_budget' and not failed_json:
      start=last-1 if last and records[refs[last-1]]['kind']=='neural_controller' else last;s['budget_discard_queries']+=1;s['budget_discard_tokens']+=sum(records[k]['cost']['decoder_input_tokens'] for k in refs[start:])
    for i,action in enumerate(r['actions']):
     if action['action']=='contradiction_backtrack':
      s['contradiction_backtracks']+=1;event=nodes[action['event']];matched=score(event['span'],gt);s['pruned_gt50']+=matched>=.5
      if matched>=.7 and name==reference:cases['prune'].append(sample_case(method=name,pruned=event['span'],pruned_iou=matched,final=result_span,final_iou=result_iou,assessment=nodes[action['from_node']]['assessment']))
   for comparison in r['comparisons']:
    a=nodes[comparison['a']];b=nodes[comparison['b']];left,right=[records[k] for k in comparison['calls']];order_disagree=(left['probabilities'][0]>=.5)!=(right['probabilities'][1]>=.5);s['comparisons']+=1;s['order_disagree']+=order_disagree
    unique=tuple(comparison['calls'])
    if unique not in query_pairs:
     query_pairs.add(unique);unique_pair_stats['n']+=1;unique_pair_stats['order_disagree']+=order_disagree
     diff=score(a['span'],gt)-score(b['span'],gt)
     if abs(diff)>=.1:
      correct=(comparison['winner']==a['id'])==(diff>0);unique_pair_stats['clear_iou_difference']+=1;unique_pair_stats['correct']+=correct
      longer=a if a['span'][1]-a['span'][0]>b['span'][1]-b['span'][0] else b;shorter=b if longer is a else a
      if longer['span'][1]-longer['span'][0]>1.2*(shorter['span'][1]-shorter['span'][0]):unique_pair_stats['length_different']+=1;unique_pair_stats['chose_longer']+=comparison['winner']==longer['id'];unique_pair_stats['chose_longer_and_worse']+=comparison['winner']==longer['id'] and score(longer['span'],gt)<score(shorter['span'],gt)
   if name==reference:
    if op>=.7 and result_iou<.3:cases['selection'].append(sample_case(method=name,final=result_span,final_iou=result_iou,pool=[dict(span=x['span'],iou=score(x['span'],gt),value=x['assessment']['value']) for x in pool],oracle_pool=op))
    if selected and result_iou<.3 and raw>=.7:cases['new_candidate_unused'].append(sample_case(method=name,final=result_span,final_iou=result_iou,root=nodes[selected['event_id']]['span'],best_proposals=sorted([dict(span=e['span'],source=e['source'],iou=score(e['span'],gt),assessment=e['assessment']) for e in events],key=lambda x:-x['iou'])[:4]))
  if key=='qvhighlights-timelens_00507':
   one=next(p for p in data['proposals']['leading_digit_paths'] if p['digit']=='1');after_one=records[one['call']]['tokens'][0];best_old=max(old['result']['candidates'],key=lambda x:score(pair(x),gt))
   cases['requested_reference'].append(sample_case(proposals=[dict(span=e['span'],source=e['source'],provenance=e['provenance'],iou=score(e['span'],gt),assessment=e['assessment']) for e in sorted(events,key=lambda x:-score(x['span'],gt))[:8]],leading_digit_paths=[dict(digit=p['digit'],span=p['span'],answer=p['answer']) for p in data['proposals']['leading_digit_paths']],after_start_one=after_one,old32_best=dict(index=best_old['index'],span=pair(best_old),iou=score(pair(best_old),gt),branch=best_old['branch']),semantic_output=records[data['proposals']['semantic_call']].get('text'),methods={m:data['methods'][m]['8192']['span'] for m in METHODS}))
  per_query.append(dict(key=key,duration=duration,gt=gt,greedy=greedy,greedy_iou=base,greedy_usable=usable(greedy,duration),proposal_unfiltered_oracle=raw,proposal_valid_oracle=admissible,proposal_gate_oracle=gated,proposal_count=len(events),eligible_count=len(eligible)))
  if len(per_query)%150==0:print(json.dumps(dict(stage='failure_analysis',completed=len(per_query),total=n)),flush=True)
 stats={k:dict(v) for k,v in stats.items()}
 for name,s in stats.items():s['length_ratio_median']=median(ratios[name]);s['length_ratio_mean']=mean(ratios[name])
 cases['duration']=sorted(cases['duration'],key=lambda x:(not (x['excess']<=.1),-x['greedy_iou']))[:5]
 for name in ('selection','new_candidate_unused'):cases[name]=sorted(cases[name],key=lambda x:x.get('oracle_pool',1)-x['final_iou'],reverse=True)[:4]
 cases['duplicate_pool']=sorted(cases['duplicate_pool'],key=lambda x:x['dedup_pool_oracle']-x['final_iou'],reverse=True)[:4]
 cases['prune']=sorted(cases['prune'],key=lambda x:x['pruned_iou']-x['final_iou'],reverse=True)[:3]
 bins={str(x):sum(v<=x+1e-9 for v in overshoots) for x in (.05,.1,.5,1,5)};binary_summary={k:dict(n=len(v),median=median(v),below_01=pct(sum(x<.01 for x in v),len(v)),below_10=pct(sum(x<.1 for x in v),len(v))) for k,v in binary.items()}
 result=dict(queries=n,baseline_miou=mean(baseline_scores),strict_duration_baseline_miou=mean(strict_scores),greedy_out_of_duration=len(overshoots),overshoot_cumulative_bins=bins,overshoot_median=median(overshoots),proposals=dict(proposals),methods=stats,value_pair_ranking=dict(value_pairs),unique_pairwise=dict(unique_pair_stats),binary_mass=binary_summary,canonical_probability_check=dict(n=len(native_check),mean_absolute_error=mean(native_check),max_absolute_error=max(native_check,default=0)),finite_list_prior=dict(n=len(qconcentration),top1_above_95=sum(x>=.95 for x in qconcentration),median_top1=median(qconcentration)),method_equality=dict(method_equal),cases=dict(cases))
 write(out/'metrics/failure_analysis.json',result);write(out/'metrics/failure_analysis_per_query.json',per_query)
 md=[f'# Failure analysis · pilot{n}','',f'기존 고정 {n} query의 완료된 원자료를 CPU로 분석했다. 새로운 GPU 추론·prompt 변경·응답 수정은 하지 않았다. 결과의 분모는 query 또는 명시한 조건 내 candidate/비교 수이며, 같은 query의 여러 방법을 독립 query로 합산하지 않는다.','', '## 확인된 운영 오류','', 'GPU worker 두 개는900개 추론을 완료했지만 attached worker를 처리하는 supervisor가 `returncode` 속성을 참조해 최종 보고가 중단됐다. `poll()` 반환값으로 처리하도록 고치고 정상/실패 종료 회귀 검증을 수행했다. 추론 원자료와 search 정책은 변경하지 않았다.','', '## 1. Duration 기준의 불일치','',f"기존 greedy mIoU {result['baseline_miou']:.4f}. 새 후보와 같은 duration admissibility(0≤start<end≤duration+.001)를 기존 greedy에도 사후 적용해 실패를0점 처리하면 {result['strict_duration_baseline_miou']:.4f}다. 이는 정책 영향을 분리하는 진단이며 원래 benchmark 점수를 대체하지 않는다.",f"기존 greedy가 이 기준에서 제외되는 query는 {len(overshoots)}/{n}. 영상 길이 초과가≤.05초 {bins['0.05']}개, ≤.1초 {bins['0.1']}개, ≤.5초 {bins['0.5']}개, ≤1초 {bins['1']}개다(누적 수). 작은 초과는 timestamp 반올림 규모와 맞지만 반올림이 원인이라는 직접 검증은 아니다.",'', '이 불일치는 baseline 대비 손실의 일부를 만든다. 다만 출력 단계의 기준을 바꾸면 당시 수행하지 않은 평가·탐색도 달라지므로, 원자료만으로 전체 알고리즘의 수정 후 성능을 재구성할 수 없다. 새 policy로 재실행해야 한다.','', '## 2. Proposal 자체와 gate','', '| 단계 | oracle mIoU | R@.5 정보를 잃은 query |','|---|---:|---:|',f"| 기존8 | {proposals['old8']/n:.4f} | — |",f"| 모든 신규 proposal, positive interval | {proposals['raw_all']/n:.4f} | — |",f"| duration 기준 통과 | {proposals['duration_valid']/n:.4f} | {int(proposals['duration_lost50'])} |",f"| model gate 통과 | {proposals['gated']/n:.4f} | {int(proposals['critic_lost50'])} |",'', '여기서 oracle은 생성된 후보 중 GT IoU가 가장 높은 값이다. 각 단계에서 실제로 존재한 후보만 보므로 원래 존재하지 않은 경로의 반사실적 성능이 아니다. R@.5 정보 손실은 직전 단계에서는 oracle≥.5인데 해당 단계에서<.5가 된 query 수다.',f"Gate 통과 후보가 없는 query {int(proposals['zero_eligible'])}/{n}; 그중 duration-valid 후보 자체가 없는 경우 {int(proposals['zero_eligible_no_legal'])}, valid 후보는 있지만 model gate가 모두 제외한 경우 {int(proposals['zero_eligible_with_legal'])}다.",f"Leading digit 경로는 {int(proposals['leading_paths'])}개 중 {int(proposals['leading_valid'])}개가 duration-valid였다. Semantic proposal의 unique interval은 query당 평균 {proposals['semantic_unique']/n:.2f}개다.",f"Semantic 요청에 완성된 JSON array로 응답한 query는 {int(proposals['semantic_complete_array'])}/{n}다. JSON 후보는 없지만 통상적인 timestamp 응답이 나온 {int(proposals['semantic_plain_timestamp_discarded'])}개는 proposal로 사용되지 않았다. 그중 duration-valid {int(proposals['semantic_plain_timestamp_valid'])}개, 기존 신규 proposal oracle<.5에서 그 버려진 응답이 IoU≥.5인 경우 {int(proposals['semantic_plain_timestamp_recovers50'])}개다. 이는 응답 형식 예외 처리의 한계다.",'', '| 후보 구성(continuous IoU, duration cut 이전) | oracle mIoU | R@.5 |','|---|---:|---:|']
 for source in ('old8','leading','semantic','old8_leading','old8_semantic'):md.append(f"| {source} | {proposals['source_oracle_'+source]/n:.4f} | {100*proposals['source_r50_'+source]/n:.2f}% |")
 md+=['','## 3. 어느 단계에서 좋은 후보를 잃었는가','','각 method의 실제 생성된 후보에서 최대 IoU를 구하고, duration → critic/blocked → 최종 비교 pool → 실제 선택 순서로 평균을 계산했다. 각 차이는 해당 기록에서의 oracle 손실 분해이며 gate를 없애 재탐색했을 때의 인과효과는 아니다. commit 대조군의 pool 손실에는 사건 고정 제약도 포함된다.','','| method | all oracle | valid oracle | eligible oracle | pool oracle | selected mIoU | pool≥.5인데 최종<.5 |','|---|---:|---:|---:|---:|---:|---:|']
 for name in ('proposal_only_token_seed','proposal_only_wide','nucleus_frontier@4096','nucleus_frontier@8192','value_best_first@8192','puct@8192','submodular_coverage@8192','token_seed_best_first@8192'):
  s=stats[name];md.append(f"| {name} | "+' | '.join(f"{s[k]/n:.4f}" for k in ('oracle_all','oracle_legal','oracle_gate','oracle_pool','iou'))+f" | {int(s['pool_good_but_wrong'])}/{int(s['pool_has50'])} |")
 md+=['','## 4. 최종 pool의 동일 구간 중복','','현재 final selector는 context/token 경로가 다르면 같은 interval도 별도 후보로 받아들인다. Soft IoU diversity penalty만 적용하므로 동일한 interval이 최대4개의 비교 자리를 차지할 수 있다. 아래 dedup은 정확히 같은 interval을 하나로 묶고, 기존 visual value와 동일한 selection rule로4개를 다시 구성한 CPU 진단이다. 새 pool의 oracle은 실제 model selection 성능이 아니라 상한이며, 새로운 pairwise inference를 수행하지 않았다.','','| method | 중복 pool / nonempty | unique=1 pool / nonempty | 평균 unique/pool size | 원래 pool oracle | dedup pool oracle |','|---|---:|---:|---:|---:|---:|']
 for name in ('proposal_only_wide','reward_balanced@4096','reward_balanced@8192','nucleus_frontier@8192','value_best_first@8192','puct@8192','submodular_coverage@8192','token_seed_best_first@8192'):
  s=stats[name];md.append(f"| {name} | {int(s['duplicate_pool'])}/{int(s['pool_nonempty'])} | {int(s['only_one_unique'])}/{int(s['pool_nonempty'])} | {s['pool_unique_n']/max(1,s['pool_nonempty']):.2f}/{s['pool_n']/max(1,s['pool_nonempty']):.2f} | {s['oracle_pool']/n:.4f} | {s['dedup_pool_oracle']/n:.4f} |")
 md+=['','## 5. Visual score와 pairwise 선택의 진단','',f"초기 valid 후보의 query 내부에서 IoU 차이≥.1인 {int(value_pairs['n'])}쌍 중 visual value 순위가 GT IoU 순위와 맞는 비율은 {pct(value_pairs['agreement'],value_pairs['n']):.2f}%다(tie=.5). 사건이 포함된다는 판단과구간이 정확하다는 판단은 다른 목표이므로, 이 수치를 사건 판별의 semantic accuracy로 해석하지 않는다.",f"여러 방법·예산에서 재사용한 동일 model comparison을 query별로 중복 제거하면 {int(unique_pair_stats['n'])}쌍이다. Presentation order를 바꿨을 때 판단이 불일치한 비율 {pct(unique_pair_stats['order_disagree'],unique_pair_stats['n']):.2f}%. GT IoU 차이≥.1인 {int(unique_pair_stats['clear_iou_difference'])}쌍에서 더 높은 IoU 후보를 고른 비율 {pct(unique_pair_stats['correct'],unique_pair_stats['clear_iou_difference']):.2f}%다.",f"길이 차이≥20% 및 IoU 차이≥.1인 비교에서 긴 구간을 고른 경우 {int(unique_pair_stats['chose_longer'])}/{int(unique_pair_stats['length_different'])}, 긴 구간을 골랐지만 그 구간의 IoU가 더 낮은 경우 {int(unique_pair_stats['chose_longer_and_worse'])}/{int(unique_pair_stats['length_different'])}다. 이는 길이 편향에 대한 관찰이며 원인 확정은 아니다.",'', '| model criterion | unique calls | median raw option mass | raw mass<.01 |','|---|---:|---:|---:|']
 for kind in ('event_presence','event_contradiction','boundary_fit','final_pairwise'):
  if kind in binary_summary:
   b=binary_summary[kind];md.append(f"| {kind} | {b['n']} | {b['median']:.4f} | {b['below_01']:.2f}% |")
 md+=['','## 6. 탐색과 refinement가 실제로 한 일','','동일 parent interval을 다시 생성한 비율은 generated-valid node가 분모다. 반복 interval은 해당 query·method의 초기 후보 및 앞선 node와 같은 구간을 뜻한다. 서로 다른 token/context 경로일 수 있으므로 동일 model evaluation과 동일시하지 않는다. cut-off 폐기는 token cap에서 생성은 끝났지만 후속 assessment/prior가 끝나지 않아 atomic tree commit을 하지 못한 경우다.','','| method | mean expansions | final이 초기 coarse 후보 | parent와 같은 생성 | 반복 interval | cut-off 폐기 query | 폐기 token/search token |','|---|---:|---:|---:|---:|---:|---:|']
 for method in ('value_best_first','nucleus_frontier','puct','neural_controller','submodular_coverage','self_refine'):
  for cap in ('4096','8192'):
   name=method+'@'+cap;s=stats[name];md.append(f"| {name} | {s.get('expansions',0)/n:.2f} | {pct(s.get('selected_initial',0),s.get('valid',0)):.2f}% | {pct(s.get('parent_identical',0),s.get('generated_valid',0)):.2f}% | {pct(s.get('repeated_interval',0),s.get('generated_valid',0)):.2f}% | {int(s.get('budget_discard_queries',0))}/{n} | {pct(s.get('budget_discard_tokens',0),s.get('search_tokens',0)):.2f}% |")
 md+=['','## 7. 방법 간 차이가 작아지는 이유의 단서','',f"Nucleus와 typicality 최종 interval은4096 cap에서 {method_equal['nucleus_typical_same_4096']}/{n},8192 cap에서 {method_equal['nucleus_typical_same_8192']}/{n} query가 같았다. 초기 eligible finite-list prior의 top1 probability≥.95는 {sum(x>=.95 for x in qconcentration)}/{len(qconcentration)}, median top1={median(qconcentration) if qconcentration else None}. 두 방식의 초기 .95 support 집합이 동일한 경우 {method_equal['same_initial_support']}/{method_equal['initial_support_n']}, nucleus support 크기 평균 {method_equal['initial_nucleus_size']/max(1,method_equal['initial_support_n']):.2f}다. 이번 구현은 support 안의 방문 순서를 같은 q/(1+visits)로 정하므로 support가 같으면 이 단계에서 차이를 만들지 못한다. 이는 token-level nucleus와 typical sampling 일반의 동등성을 뜻하지 않는다.",f"원래 leading digit의 stepwise probability와 같은 경로를 batched teacher forcing한 probability 차이는 평균 {mean(native_check):.6g}, 최대 {max(native_check,default=0):.6g}였다. 입력 hash·greedy replay와 함께 implementation drift를 점검한 진단이며 batch/stepwise 수치 차이를 완전히 배제한 검증은 아니다.",'', '## 8. GT 기준 회복과 손상','','회복 분모는 기존 greedy IoU<.1, 손상 분모는 기존 greedy IoU≥.5다. 전체 query의 성능과 함께 해석한다.','','| method | recovery to IoU≥.5 | harmful to IoU<.5 | final length>2×GT / valid | median length/GT |','|---|---:|---:|---:|---:|']
 for name in ('proposal_only_wide','nucleus_frontier@8192','value_best_first@8192','puct@8192','submodular_coverage@8192','self_refine@8192','commit_then_refine@8192','token_seed_best_first@8192'):
  s=stats[name];md.append(f"| {name} | {int(s['recovery'])}/{int(s['baseline_bad'])} | {int(s['harmful'])}/{int(s['baseline_good'])} | {int(s.get('length_gt2x',0))}/{int(s.get('valid',0))} | {s['length_ratio_median'] if s['length_ratio_median'] is not None else '—'} |")
 if cases['requested_reference']:
  case=cases['requested_reference'][0];p=case['after_start_one']['numeric_probabilities'];old_best=case['old32_best'];one_path=next(x for x in case['leading_digit_paths'] if x['digit']=='1')
  md+=['','## 핵심 사례: 정수 prefix의 내부 대안을 놓침','',f"{case['key']}의 GT는 {case['gt']}다. start의 첫 token으로1을 선택한 직후 P('.')={p['.']:.4f}, P('4')={p['4']:.4f}, P('1')={p['1']:.4f}, P('2')={p['2']:.4f}였다. 이번 leading-digit proposal은 가장 높은 decimal point를 따라 {one_path['span']}로 완성했다. 그 뒤 정수 자리의1·4·2 대안을 체계적으로 열지 않았다.",f"기존32개에는 이미 candidate#{old_best['index']}={old_best['span']}, IoU={old_best['iou']:.4f}가 있었다. 이번 seed는 첫8개만 사용했고 해당 좋은 경로를 초기 pool에 포함하지 않았다. 새 semantic proposal 원문은 `{case['semantic_output']}`였고 요청한 JSON array가 아니었다. 모든 방법이 최종 유효 응답을 내지 못했다.",'', '따라서 이번 실행의 event/action tree 탐색 결과로 숫자 prefix tree 내부의 재검토·확장 가설을 기각할 수 없다. 첫 자리 뒤를 greedy로 완성한 설계가 사용자가 지적한 중요한 내부 분기를 놓쳤다. 이 구현 한계와 candidate 선택 문제를 먼저 수정해야 한다.']
 md+=['','## 9. 실패 원인 추정과 해석 한계','','확인된 사실은 위 수치와 코드 경로다. 다음은 그 사실로부터의 추정이다.','', '- Strict duration gate와 benchmark baseline의 admissibility 불일치가 일부 손실을 만든다. endpoint를 임의 clipping해 성능을 올리지 말고, 허용오차 정책을 모든 조건에 맞춘 재실험으로 확인해야 한다.','- Final comparison pool에서 같은 interval을 별도 후보로 유지한 것이 유효한 선택 폭을 줄였다. token/context 경로는 보존하되 interval만 제시하는 최종 비교에서는 exact interval을 중복 제거하는 대조가 필요하다.','- 초기 coarse 후보의 boundary probability는 실제로 평가하지 않고 .5로 둔다. 같은 interval을 refinement에서 다시 평가해 boundary probability가1에 가까워지면 다른 값이 같아도 value의 boundary multiplier는 .75에서1로 올라간다. 이런 깊이별 평가 기준 차이가 반복 생성의 점수를 높이고 초기 다른 사건 후보를 밀어낼 가능성이 있다.','- Boundary child에서 확인한 contradiction을 event root 전체에 전파한다. 이 때문에 원래 GT에 가까운 parent·siblings까지 함께 제거될 수 있다. 원래 사건이 잘못됐다는 판단과 특정 refinement가 잘못됐다는 판단을 분리하는 대조가 필요하다.','- Visual value와 pairwise judgement가 localization IoU에 충분히 정렬되지 않았다면, 더 강한 탐색은 잘못된 proxy를 더 잘 최적화할 수 있다. 사건 포함 여부와 정확한 사건·경계를 나눠 검증할 필요가 있다.','- Same full-video input을 모든 검토에 사용했다. 시간 구간별 evidence를 새로 분리해서 보는 실험을 하지 않았으므로, 영상 정보 부족/FPS가 원인이라고 확정할 수 없다.','- Candidate-level tree/beam/UCT/PUCT/typicality adaptations의 결과이며 원 논문의 token-level 또는 학습된 value-model 성능 전체로 일반화할 수 없다. 이번 초기 numeric expansion은 leading digit 뒤를 greedy로 완성하며 둘째·셋째 integer token의 대안을 체계적으로 열지 않는다.','- Budget cut-off의 atomic rollback과 반복 boundary 생성은 실제 계산의 일부를 후보 증가로 연결하지 못한다. 재실험에서는 생성된 후보와 완료된 critic을 단계별로 checkpoint하는 비교가 필요하다.','- 수학적 coverage의 approximation guarantee는 고정 F에 관한 것이다. F의 kernel·quality가 GT localization 목표와 다르면 그 목적함수의 최적화 성공과 실제 성능은 다를 수 있다.','', '## 사례','']
 for label in ('duration','duplicate_pool','selection','prune','new_candidate_unused','requested_reference'):
  md+=['### '+label,'']
  for case in cases[label]:md+=['```json',json.dumps(case,ensure_ascii=False,indent=2),'```','']
 (dest/'FINDINGS.md').write_text('\n'.join(md))
 synchronize_report_notes(dest/'FINDINGS.md')
 archive=out/'code/reporting';archive.mkdir(parents=True,exist_ok=True);(archive/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 write(out/'validation/failure_analysis.json',dict(passed=True,queries=n,source_sha256=sha(Path(__file__)),oracle_decomposition_monotonic=True,inference_rerun=False,timestamp=stamp()))
 print(json.dumps(dict(stage='failure_analysis_complete',queries=n,baseline=mean(baseline_scores),strict_baseline=mean(strict_scores),duration_excluded=len(overshoots),proposals=dict(proposals),unique_pairwise=dict(unique_pair_stats)),ensure_ascii=False),flush=True)
 return result

def synchronize_report_notes(findings):
 for size in (300,600,900):
  dest=ROOT/'results'/FAMILY/'pilot'/str(size);report_path=dest/'REPORT.md';pfile=ROOT/'artifacts/runs'/FAMILY/'pilot'/str(size)/'metrics/proposals.json'
  if not report_path.exists():continue
  text=report_path.read_text();relative=os.path.relpath(findings,dest);warning='해석 주의: 실패 분석에서 기존 greedy와 신규 후보의 duration 기준 불일치, 최종 비교 pool의 동일 interval 중복, 숫자 prefix 내부 확장 부족을 확인했다. 아래는 원래 실행 정책의 결과이며, 알고리즘 일반의 우열이나 초기 가설 기각으로 해석하지 않는다. [실패 원인 분석]('+relative+').'
  if '해석 주의: 실패 분석에서' not in text:text=text.replace('\n\n[설계·수식·문헌]', '\n\n'+warning+'\n\n[설계·수식·문헌]',1)
  text=text.replace('raw는 gate 이전, gated는 사건 적합성 gate 이후다.','duration-valid는 시간 범위 검사 통과 후 model gate 이전이며, gated는 사건 적합성 gate 이후다. 시간 범위 검사 이전의 oracle은 실패 분석에 별도 표시한다.').replace('| new proposals raw |','| new proposals duration-valid |')
  if pfile.exists():
   props=read(pfile)
   for p in props:
    cs=read(SOURCE/'rollouts'/f'{p["key"]}.json')['result']['candidates'][:8];p['old8_unique']=len({tuple(pair(c)) for c in cs if pair(c)})
   write(pfile,props);count=mean([p['old8_unique'] for p in props]);text=re.sub(r'\| old paths first8 \| [0-9.]+ \|',f'| old paths first8 | {count:.2f} |',text)
  report_path.write_text(text)

if __name__=='__main__':analyze(int(sys.argv[1]) if len(sys.argv)>1 else 900)
