"""Automatic quantitative reports and interactive inspection for reconsideration."""
import collections,csv,hashlib,json,math,os,re,shutil,statistics,subprocess,sys
from html import escape
from pathlib import Path
from rollout_method_docs import formalize, refresh_spec
from rollout_reconsideration import ROOT,OUT,REPORT,SOURCE,COHORT,CONFIG,VARIANTS,FAMILY,ANALYSIS,read,write,sha,stamp,interval,iou,clusters
LITERATURE=[
 ('Snell et al., Scaling LLM Test-Time Compute Optimally','https://arxiv.org/abs/2408.03314','proposal revision과 verifier-guided search를 구분하고, 평가·수정 연산을 포함한 예산을 비교한다. 이 pilot은 학습된 PRM을 재현하지 않으며 frozen VLM judge로 두 축을 시험한다.'),
 ('Qin et al., Pairwise Ranking Prompting','https://arxiv.org/abs/2306.17563','pairwise selection을 이용하고 두 후보의 presentation order를 뒤집어 평가한다. 문서 ranking에서 video interval selection으로 적용한 실험 설계이며 원 논문 성능의 재현이 아니다.'),
 ('Yao et al., ReAct','https://arxiv.org/abs/2210.03629','agent의 action과 observation을 교대로 수행한다. 여기서는 event/boundary visual inspection의 순서를 controller가 고른다.'),
 ('Liu et al., VideoMind','https://arxiv.org/abs/2503.13444','planner·grounder·verifier 역할의 분리를 참고한다. 해당 논문의 LoRA나 학습 가중치를 사용하지 않으므로 zero-shot workflow 비교로 한정한다.')]

def init():
 from rollout_reconsideration import setup
 cohort=setup()
 (REPORT/'SPEC.md').write_text(formalize('', 'reconsideration',len(cohort),spec=True))
 if not (REPORT/'REPORT.md').exists():
  (REPORT/'REPORT.md').write_text('# Rollout reconsideration · pilot900\n\n상태: running. 완료된 pilot900 성능 결과가 아니다.\n\n[실험 설계](SPEC.md)\n')
 return cohort


def assigned_cluster(pair,groups,threshold=.5):
 if not pair:return None
 scores=[max(iou(pair,interval(c)) for c in group) for group in groups]
 if scores and max(scores)>=threshold:return int(max(range(len(scores)),key=lambda i:scores[i]))
 return -1

def summarize(rows):
 n=len(rows)
 if not n:return None
 result=dict(n=n,miou=sum(r['iou'] for r in rows)/n)
 for t in (.3,.5,.7):result[f'r{int(t*100)}']=sum(r['iou']>=t for r in rows)/n
 result.update(failure=sum(r['invalid'] for r in rows)/n,switches=sum(r.get('switch',False) for r in rows),harmful=sum(r['baseline_iou']>=.5 and r['iou']<.5 for r in rows))
 return result

def table(rows,methods):
 base=summarize([r for r in rows if r['name']=='greedy']);out=['| method | n | mIoU (Δ) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | model evaluations | decoder tokens (proposal+review) | review model seconds | failure |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name in methods:
  group=[r for r in rows if r['name']==name];s=summarize(group)
  if not s:continue
  out.append(f"| {name} | {s['n']} | {s['miou']:.4f} ({s['miou']-base['miou']:+.4f}) | "+' | '.join(f"{s[k]:.4f} ({100*(s[k]-base[k]):+.2f})" for k in ('r30','r50','r70'))+f" | {statistics.mean(r['evaluations'] for r in group):.1f} | {statistics.mean(r['tokens'] for r in group):.1f} | {statistics.mean(r['seconds'] for r in group):.3f} | {s['failure']:.3f} |")
 return out

def bootstrap(rows,n_boot=400,reference="greedy"):
 import numpy as np
 methods=sorted({r['name'] for r in rows});videos=sorted({r['video'] for r in rows});index={v:i for i,v in enumerate(videos)};result={};rng=np.random.default_rng(42)
 counts=np.bincount([index[r['video']] for r in rows if r['name']==reference],minlength=len(videos));weights=rng.multinomial(len(videos),np.ones(len(videos))/len(videos),size=n_boot);denom=weights@counts
 base={r['key']:r for r in rows if r['name']==reference}
 for method in methods:
  group=[r for r in rows if r['name']==method]
  for metric,threshold in [('miou',None),('r30',.3),('r50',.5),('r70',.7)]:
   values=np.zeros(len(videos))
   for r in group:
    b=base[r['key']];difference=r['iou']-b['iou'] if threshold is None else float(r['iou']>=threshold)-float(b['iou']>=threshold);values[index[r['video']]]+=difference
   draws=(weights@values)/denom;low,high=np.quantile(draws,[.025,.975]);result.setdefault(method,{})[metric]=dict(delta=float(values.sum()/counts.sum()),lower=float(low),upper=float(high))
 return result

def report(n=900,refresh=True):
 cohort=read(COHORT)[:n];out=ROOT/'artifacts/runs'/FAMILY/'pilot'/str(n)/ANALYSIS;dest=ROOT/'results'/FAMILY/'pilot'/str(n)/ANALYSIS;dest.mkdir(parents=True,exist_ok=True);rows=[];details=[];validated=[]
 for sample in cohort:
  key=sample['sample_key'];rec=read(SOURCE/'rollouts'/f'{key}.json');data=read(OUT/'responses'/f'{key}.json');assert set(data['variants'])=={v[0] for v in VARIANTS};assert data['input_hashes']==rec['input_hashes'];assert sha(data['baseline_reference'])==data['baseline_sha256']
  cs=rec['result']['candidates'];gt=sample['span'][0];greedy=interval(cs[0]);base_iou=iou(greedy,gt);groups,assignment=clusters(cs);old_cluster=assigned_cluster(greedy,groups);sensitivity={t:clusters(cs,t)[0] for t in (.3,.7)};sample_detail=dict(sample=sample,greedy=greedy,variants={},response=str(OUT/'responses'/f'{key}.json'))
  variants={'greedy':dict(answer=cs[0],cost_decision=dict(model_evaluations=0,decoder_input_tokens=cs[0]['decoder_tokens'],model_seconds=0),pool=1)}
  from deterministic_rollout300_report import select
  for pool in (8,32):
   for method in ('max_probability','probability_weighted_medoid','oracle'):
    chosen=select(cs[:pool],method,gt);variants[f'{method}_pool{pool}']=dict(answer=chosen,cost_decision=dict(model_evaluations=0,decoder_input_tokens=cs[min(pool,len(cs))-1]['decoder_tokens'],model_seconds=0),pool=pool)
  variants.update(data['variants'])
  for name,v in variants.items():
   pair=interval(v['answer']);score=iou(pair,gt);new_cluster=assigned_cluster(pair,groups);switch=bool(greedy and pair and (new_cluster==-1 or new_cluster!=old_cluster));cost=v['cost_decision'];same=bool(greedy and pair and new_cluster==old_cluster and new_cluster!=-1)
   root=dict(key=key,name=name,dataset=sample['dataset'],video=sample['video_path'],duration=sample['duration'],ratio=(gt[1]-gt[0])/sample['duration'],iou=score,baseline_iou=base_iou,invalid=pair is None,switch=switch,same_cluster=same,event_recovery=bool(base_iou<.1 and score>=.5 and switch),evaluations=cost['model_evaluations'],tokens=cost['decoder_input_tokens'],seconds=cost['model_seconds'],endpoint_mae=(abs(pair[0]-gt[0])+abs(pair[1]-gt[1]))/2 if pair else None,baseline_mae=(abs(greedy[0]-gt[0])+abs(greedy[1]-gt[1]))/2 if greedy else None)
   for t,sg in sensitivity.items():
    old=assigned_cluster(greedy,sg,t);new=assigned_cluster(pair,sg,t);root[f'switch_t{int(t*100)}']=bool(greedy and pair and (new==-1 or old!=new))
   rows.append(root)
   if name in data['variants']:
    root['proposal_tokens']=v['proposal_cost']['decoder_tokens'];root['end_to_end_tokens']=root['proposal_tokens']+v['cost_total']['decoder_input_tokens'];root['diagnostic_seconds']=v['cost_diagnostics']['model_seconds'];root['tokens']=root['proposal_tokens']+cost['decoder_input_tokens'];root['historical_proposal_seconds']=v['proposal_cost']['original_generation_seconds'];root['selected_candidate_iou']=iou(interval(cs[v['selected_candidate']-1]),gt) if v['selected_candidate'] else None
    sample_detail['variants'][name]=dict(pair=pair,iou=score,delta=score-base_iou,switch=switch,same_cluster=same,method=v['method'],pool=v['pool'],model_seconds=cost['model_seconds'],model_evaluations=cost['model_evaluations'],probes=v['probes'])
    for probe in v['probes']:
     for node in probe['nodes']:
      path=OUT/node['file'];assert path.is_file() and sha(path)==node['sha256'] and abs(node['total_probability']-1)<3e-5
   validated.append(name)
  details.append(sample_detail)
 for category in ('metrics','validation','web','metadata'):(out/category).mkdir(parents=True,exist_ok=True)
 write(out/'metadata/cohort_reference.json',dict(path=str(COHORT),queries=n,indices=f'0:{n}',sha256=sha(COHORT),responses=str(OUT/'responses')))
 write(out/'metrics/per_query.json',rows);write(out/'metrics/viewer_details.json',details)
 fields=sorted(set().union(*(r.keys() for r in rows)))
 with (out/'metrics/per_query.csv').open('w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 methods=['greedy','max_probability_pool8','probability_weighted_medoid_pool8']+[v[0] for v in VARIANTS]+['max_probability_pool32','probability_weighted_medoid_pool32','oracle_pool8','oracle_pool32']
 ci=bootstrap(rows);pairs={}
 for reference in ('agent_pool8','verifier_pool8','agent_pool32','verifier_pool32'):
  compared=bootstrap(rows,reference=reference);pool=8 if reference.endswith('pool8') else 32
  for target in (f'verifier_pool{pool}',f'tts_pool{pool}_select{4 if pool==8 else 8}'):
   if target!=reference:pairs[target+' minus '+reference]=compared[target]
 for reference,targets in [('tts_pool8_select4',['tts_pool8_select8','tts_pool32_select4']),('tts_pool8_select8',['tts_pool32_select8']),('tts_pool32_select4',['tts_pool32_select8'])]:
  compared=bootstrap(rows,reference=reference)
  for target in targets:pairs[target+' minus '+reference]=compared[target]
 write(out/'metrics/paired_video_bootstrap.json',dict(against_greedy=ci,pairwise=pairs))
 text=[f'# Rollout reconsideration · pilot{n}','',f'완료: {n} query, dataset별{n//3}. 3개 접근과 direct regeneration control을 같은 고정 cohort에서 평가했다. 생성 input hash는 기존 deterministic rollout과 일치하며 GT는 보고 단계에서만 사용했다.','', '[실험 설계](SPEC.md)' if (dest/'SPEC.md').exists() else '[실험 설계](../../900/reconsideration/SPEC.md)', '',
 '## 지표와 비교 기준','',f'mIoU는 {n} query의 temporal IoU 평균, R@τ는 IoU≥τ query 비율이다. IoU=intersection duration/union duration이며 높을수록 좋다. Δ는 같은 query의 greedy 대비 차이, Δpp는 성공률 차이×100이다. failure는 파싱 실패·invalid interval·EOS 미도달 비율이고0점으로 포함한다. model seconds는 CUDA 동기화로 측정한 의사결정 model wall time이며 공통 video prefill과 사후 probability diagnostic을 제외한다. decoder tokens는 기존 proposal generation 비용과 새 review의 prompt·generated token·KV replay를 합산한다. probability diagnostics 비용은 별도 원자료에 보존한다. 기존 baseline은 추가 review time이0이다. 평가 호출 수를 맞춘 비교가 FLOPs 일치를 뜻하지 않는다.','',
 '## 전체 성능','']+table(rows,methods)
 text+=['','## Paired comparison · 95% bootstrap CI','','video 단위 paired bootstrap의 방법 간 차이다. CI가0을 포함하면 순위 차이를 확정하지 않는다.','','| comparison | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |','|---|---|---|']
 for name,values in pairs.items():
  a,b=values['miou'],values['r50'];text.append(f"| {name} | {a['delta']:+.4f} [{a['lower']:+.4f}, {a['upper']:+.4f}] | {b['delta']:+.4f} [{b['lower']:+.4f}, {b['upper']:+.4f}] |")
 text+=['','## Clustering sensitivity','','semantic gold가 아닌 temporal proxy의 threshold 민감도다. 분모는 두 응답이 모두 valid인 query다.','','| method | n | switch @.3 | switch @.5 | switch @.7 |','|---|---:|---:|---:|---:|']
 for name in methods:
  valid=[r for r in rows if r['name']==name and not r['invalid'] and r['baseline_mae'] is not None];nn=len(valid);text.append(f"| {name} | {nn} | {sum(r['switch_t30'] for r in valid)/nn if nn else 0:.4f} | {sum(r['switch'] for r in valid)/nn if nn else 0:.4f} | {sum(r['switch_t70'] for r in valid)/nn if nn else 0:.4f} |")
 text+=['','## Model selection와 final regeneration','','verifier/TTS가 실제로 선택한 기존 candidate의 IoU와 final regeneration의 IoU를 구분한다. agent의 inspection 대상은 선택 결과로 간주하지 않는다.','','| method | n selected | selected mIoU | final mIoU | model order disagreement |','|---|---:|---:|---:|---:|']
 for name,method,_,_ in VARIANTS:
  group=[r for r in rows if r['name']==name and r.get('selected_candidate_iou') is not None]
  if not group:continue
  comparisons=[v for d in details for v in read(d['response'])['variants'][name]['reviews'] if 'forward' in v]
  disagreement=sum((r['forward']['conditional_probabilities'][0]>=.5)!=(r['reverse']['conditional_probabilities'][1]>=.5) for r in comparisons)/len(comparisons) if comparisons else None
  text.append(f"| {name} | {len(group)} | {statistics.mean(r['selected_candidate_iou'] for r in group):.4f} | {statistics.mean(r['iou'] for r in group):.4f} | {disagreement if disagreement is not None else '—'} |")
 text+=['','## event change와 boundary refinement','', 'temporal clustering은 semantic event identity gold가 아닌 proxy다. switch는 valid 두 응답의 cluster 변경, event recovery는 greedy IoU<0.1에서 switch 후 IoU≥0.5가 된 경우다. same-cluster boundary 지표는 greedy IoU≥0.5인 paired-valid query에 조건부이며 분모를 별도로 표시한다. endpoint MAE는 두 endpoint의 절대오차 평균(seconds), 낮을수록 좋다.','', '| method | switch / paired-valid | beneficial / switch | event recovery / greedy IoU<.1 | harmful / greedy IoU≥.5 | same-cluster boundary n | MAE before → after | ΔIoU |','|---|---:|---:|---:|---:|---:|---|---:|']
 for name in methods:
  group=[r for r in rows if r['name']==name];valid=[r for r in group if not r['invalid'] and r['baseline_mae'] is not None];switches=[r for r in valid if r['switch']];boundary=[r for r in valid if r['same_cluster'] and r['baseline_iou']>=.5];low=[r for r in group if r['baseline_iou']<.1 and r['baseline_mae'] is not None];high=[r for r in group if r['baseline_iou']>=.5]
  mae=f"{statistics.mean(r['baseline_mae'] for r in boundary):.3f} → {statistics.mean(r['endpoint_mae'] for r in boundary):.3f}" if boundary else '—';delta=f"{statistics.mean(r['iou']-r['baseline_iou'] for r in boundary):+.4f}" if boundary else '—'
  text.append(f"| {name} | {len(switches)}/{len(valid)} | {sum(r['iou']-r['baseline_iou']>.01 for r in switches)}/{len(switches)} | {sum(r['event_recovery'] for r in low)}/{len(low)} | {sum(r['iou']<.5 for r in high)}/{len(high)} | {len(boundary)} | {mae} | {delta} |")
 text+=['','## token distribution과 path probability','', '같은 token prefix를 review context에서 재평가했다. 아래 positive log-odds Δ는 원래 선택에 비해 대체 token을 더 선호하게 됐다는 뜻이다. GT에 가까운지와는 별도다. JS divergence는 전체 어휘 분포 차이이며0이면 같다.','', '| method | axis | n probes | mean JS divergence | mean alternative probability Δ | mean log-odds Δ | alternative full-path preference Δ |','|---|---|---:|---:|---:|---:|---:|']
 for name,_,_,_ in VARIANTS:
  for axis in ('event','boundary'):
   probes=[p for d in details for p in d['variants'][name]['probes'] if p['axis']==axis]
   if not probes:continue
   roots=[p['nodes'][0] for p in probes];logodds=[math.log(max(x['alternative_probability_after'],1e-38)/max(x['original_probability_after'],1e-38))-math.log(max(x['alternative_probability_before'],1e-38)/max(x['original_probability_before'],1e-38)) for x in roots]
   paths=[(p['paths']['alternative']['after_log_probability']-p['paths']['parent']['after_log_probability'])-(p['paths']['alternative']['before_log_probability']-p['paths']['parent']['before_log_probability']) for p in probes]
   text.append(f"| {name} | {axis} | {len(probes)} | {statistics.mean(x['js_divergence'] for x in roots):.6f} | {statistics.mean(x['alternative_probability_after']-x['alternative_probability_before'] for x in roots):+.6f} | {statistics.mean(logodds):+.4f} | {statistics.mean(paths):+.4f} |")
 text+=['','## 추가 분석','', '[dataset·video duration·GT ratio·교차표](STRATA.md). [video별 paired bootstrap 원자료]('+os.path.relpath(out/'metrics/paired_video_bootstrap.json',dest)+'). 동일 video query를 묶어400회 재표집했다.','', '선택만의 이득과 regeneration의 이득은 interactive detail에서 verifier/TTS selected candidate와 final answer를 비교한다. agent의 마지막 inspection 대상은 최종 선택 후보라고 간주하지 않는다. '+('qvhighlights-timelens_00507은 이 고정 cohort에 포함되어 정상 평가했다.' if any(s['sample_key']=='qvhighlights-timelens_00507' for s in cohort) else 'qvhighlights-timelens_00507은 이 고정 cohort 밖이므로 통계에서 제외했다.'),'']
 (dest/'REPORT.md').write_text(formalize('\n'.join(text),'reconsideration',n));refresh_spec(dest,'reconsideration',n);strata=['# Stratified results · pilot'+str(n),'','mIoU와 R@0.3/0.5/0.7은 각 칸 query를 분모로 한다. n<10은 small cell, n=0은 empty다. Δ는 동일 칸의 greedy 대비다.','']
 axes=[('dataset',sorted({r['dataset'] for r in rows})),('duration',[(0,30),(30,60),(60,120),(120,180),(180,float('inf'))]),('ratio',[(0,.05),(.05,.1),(.1,.2),(.2,.4),(.4,1.000001)])]
 for axis,bins in axes:
  for cell in bins:
   subset=[r for r in rows if r['dataset']==cell] if axis=='dataset' else [r for r in rows if cell[0]<=r[axis]<cell[1]]
   strata+=['## '+axis+' '+str(cell),'']
   if subset:strata+=table(subset,methods)+['']
   else:strata+=['empty','']
 for metric in ('miou','r30','r50','r70'):
  strata+=['## duration × ratio · '+metric,'','| method | duration | ratio | n | value | Δgreedy |','|---|---|---|---:|---:|---:|']
  for db in axes[1][1]:
   for rb in axes[2][1]:
    subset=[r for r in rows if db[0]<=r['duration']<db[1] and rb[0]<=r['ratio']<rb[1]];base=summarize([r for r in subset if r['name']=='greedy'])
    for method in methods:
     val=summarize([r for r in subset if r['name']==method]);nn=val['n'] if val else 0;mark=' small' if 0<nn<10 else ' empty' if nn==0 else ''
     strata.append(f"| {method} | {db} | {rb} | {nn}{mark} | "+(f"{val[metric]:.4f} | {val[metric]-base[metric]:+.4f} |" if val else '— | — |'))
 (dest/'STRATA.md').write_text('\n'.join(strata));report_source=Path(__file__);(out/'code/reporting').mkdir(parents=True,exist_ok=True);(out/'code/reporting'/report_source.name).write_bytes(report_source.read_bytes())
 write(out/'validation/report.json',dict(passed=True,report_source_sha256=sha(report_source),queries=n,variants=len(VARIANTS),input_hashes_identical=True,original_records_unchanged=True,probability_vectors_validated=True,paired_video_bootstrap=True))
 build_view(n,details)
 if refresh:subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
 print(json.dumps(dict(stage='report_complete',queries=n,report=str(dest/'REPORT.md')),ensure_ascii=False),flush=True)


def build_view(n,details=None):
 assert n in (300,600,900)
 out=ROOT/'artifacts/runs'/FAMILY/'pilot'/str(n)/ANALYSIS
 dest=ROOT/'results'/FAMILY/'pilot'/str(n)/ANALYSIS;web=out/'web'
 web.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
 # UI reads deterministic manifests and original responses; do not duplicate their payloads.
 for old in ('details.json','samples.json'):
  (web/old).unlink(missing_ok=True)
 # Reuse case definitions and stored trajectories, while giving reconsideration its own process UI.
 baseline=ROOT/'artifacts/runs'/FAMILY/'pilot'/str(n)/'web';samples=SOURCE/'web/samples'
 shared=Path(__file__).with_name('deterministic_rollout_viewer.html').read_text()
 finder=shared[shared.index('<aside class="panel case-finder"'):shared.index('</aside>')+8]
 template=Path(__file__).with_name('rollout_reconsideration_viewer.html').read_text().replace('__CASE_FINDER__',finder)
 css=Path(__file__).with_name('rollout_reconsideration_viewer.css');js=css.with_suffix('.js')
 config=dict(store=os.path.relpath(OUT,dest),queries=n,sourceViewer=os.path.relpath(ROOT/'results'/FAMILY/'viewer.html',dest)+'?view=pilot/'+str(n)+'/viewer.html')
 boot='<script>window.ROLLOUT_EXPLORER_OPTIONS={caseOnly:true};window.ROLLOUT_VIDEO_BASE='+json.dumps(os.path.relpath(ROOT/'results'/FAMILY/'pilot/900/viewer.html',dest))+';window.RECONSIDERATION_CONFIG='+json.dumps(config)+';</script>'
 for a,b in [('__PILOT_SIZE__',str(n)),('__ASSET_BASE__',os.path.relpath(baseline,dest)),('__SAMPLE_BASE__',os.path.relpath(samples,dest)),('__REVIEW_ASSETS__',os.path.relpath(web,dest)),('__SPEC__','SPEC.md' if (dest/'SPEC.md').exists() else '../../900/reconsideration/SPEC.md'),('__BOOT__',boot)]:template=template.replace(a,b)
 sys.path.insert(0,str(ROOT.parent/'_lib'));from build_index import add_navigation
 (dest/'viewer.html').write_text(add_navigation(dest/'viewer.html',template))
 shutil.copyfile(css,web/'reconsideration.css');shutil.copyfile(js,web/'reconsideration.js')
 sources=[]
 for name in ('rollout_reconsideration_report.py','rollout_reconsideration_viewer.html','rollout_reconsideration_viewer.js','rollout_reconsideration_viewer.css','rollout_reconsideration_viewer_check.py'):
  source=Path(__file__).with_name(name);target=out/'code/reporting'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  sources.append(dict(path=str(source),sha256=sha(source)))
 write(out/'validation/viewer_sources.json',dict(passed=True,queries=n,analysis='reconsideration',deterministic_manifest=str(baseline/'manifest.json'),response_source=str(OUT/'responses'),sources=sources))
 report=dest/'REPORT.md'
 if report.exists():
  text=re.sub(r'\n<!-- reconsideration-viewer:start -->.*?<!-- reconsideration-viewer:end -->\n?','',report.read_text(),flags=re.S)
  block='\n<!-- reconsideration-viewer:start -->\n## Reconsideration 프로세스 탐색\n\n[접근별 프로세스](viewer.html)는 deterministic rollout의 하위 분석이다. 기존 사례 유형·수치 필터와 같은 질문을 사용하며 원래 trajectory 문장·분기 지도·baseline 분포는 원본 뷰 링크에서 확인한다. 이 페이지는 그 화면을 복제하지 않는다.\n\nagent는 실제 controller action과 inspector observation을 시간순으로 표시한다. 명시적 selected candidate가 없으므로 마지막 inspection을 최종 선택으로 간주하지 않는다. verifier는 후보별 event/경계 Yes 확률과 그 곱, 실제 선택 및 final regeneration을 연결한다. test-time scaling은 양쪽 presentation order의 pairwise 점수, round별 승자와 bye를 대진 구조로 표시한다. direct regeneration은 추가 평가 없이 같은 ledger로 생성한 두 pool 조건의 대조다. 각 접근 안에서 예산 조건을 비교한다.\n\npool은 기존 deterministic 후보의 앞8/32개다. 유효하고 중복되지 않은 구간에서 GT 없이 farthest-first로 다양성을 확보하며, ledger의 후보 목록은 최대8개다. verifier는 별도로 최대3/7개를 평가하고 TTS는 최대4/8개를 대진에 넣는다. 실제 목록·평가 수·원문은 저장된 기록을 사용한다. final answer는 선택 후보와 다를 수 있다. 사후 token distribution 진단은 기본 검토 과정과 분리한다. 원본 응답·영상·baseline token distribution는 복제하지 않는다.\n<!-- reconsideration-viewer:end -->\n'
  report.write_text(text.rstrip()+'\n'+block)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--init',action='store_true');p.add_argument('--queries',type=int,default=900);p.add_argument('--view-only',action='store_true');a=p.parse_args()
 if a.view_only:build_view(a.queries)
 elif a.init:init()
 else:report(a.queries)
