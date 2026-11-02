"""Aggregate all saved zero-shot run versions; CPU only, never launch inference."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
from collections import defaultdict,Counter
import argparse,csv,hashlib,json,statistics,sys,time,os,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/all_zeroshot_v1')
sys.path[:0]=[str(relocate_path(ROOT/'runtime/timelens_deps')),str(relocate_path(ROOT/'vendor/TimeLens')),str(relocate_path(ROOT/'_lib'))]
from timelens.utils import extract_time,iou
from artifact_store import register

def read(p):return json.loads(p.read_text())
def now():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');write_result_text(tmp, json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def savecsv(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def aggregate():
 OUT.mkdir(parents=True,exist_ok=True);started=time.time();sources=[];records=[];seen=set();pilot=read(relocate_path(ROOT/'predictions/cohorts/pilot_300/samples.json'));pilotkeys={s['sample_key'] for s in pilot};samples={s['sample_key']:s for s in read(relocate_path(ROOT/'predictions/qwen3/samples.json'))}
 status=lambda stage,**kw:write(relocate_path(OUT/'status.json'),dict(stage=stage,timestamp=now(),elapsed_seconds=time.time()-started,records=len(records),**kw))
 def add(r,path,run,model,order,fps,policy,original='rounded',line=None):
  sample=r.get('sample',r);key=sample['sample_key'];ds=sample['dataset'];s=samples[key];gt=r.get('gt',s['span'][0]);assert gt==s['span'][0]
  identity=(run,model,key,order);assert identity not in seen,identity;seen.add(identity)
  parsed=extract_time(r['answer']) or [];raw=iou(gt,parsed[0]) if parsed else 0.;rounded=iou(gt,[round(t) for t in parsed[0]]) if parsed else 0.
  expected=raw if original=='raw' else rounded;assert abs(expected-r['iou'])<1e-10,(identity,expected,r['iou'])
  rel=os.path.relpath(path,ROOT);records.append(dict(run=run,model=model,order=order,fps=fps,policy=policy,dataset=ds,sample_key=key,is_pilot=key in pilotkeys,iou_raw=raw,iou_rounded=rounded,iou_original=r['iou'],original_scoring=original,parse_failed=not parsed,generated_tokens=r.get('generated_tokens',len(r.get('generated_ids',[]))),truncated=r.get('truncated',r.get('generation_limit_reached',not r.get('eos_reached',r.get('stopped_on_eos',True)))) ,source=rel,source_line=line))
 def load(path):
  b=path.read_bytes();sources.append(dict(path=os.path.relpath(path,ROOT),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)));return json.loads(b)
 status('legacy')
 for order in ['video_query','query_video']:
  for s in samples.values():
   p=relocate_path(ROOT/'predictions/qwen3'/s['dataset']/s['sample_key']/order/'answer.json');add(load(p),p,'legacy_qwen3','Qwen3-VL-8B-Instruct',order,2,'greedy, max512')
 for s in pilot:
  p=relocate_path(ROOT/'predictions/qwen25/predictions'/f"{s['sample_key']}.json");add(load(p),p,'legacy_qwen25_pilot','Qwen2.5-VL-7B-Instruct','video_query',2,'greedy, max512, rep1.05')
 status('full_benchmark')
 for m,label in [('qwen25','Qwen2.5-VL-7B-Instruct'),('qwen3','Qwen3-VL-8B-Instruct')]:
  paths=sorted(result_glob((relocate_path(ROOT/'artifacts/predictions/full_benchmark_v1'/m)), '*/*.json'));assert len(paths)==9404
  for p in paths:add(load(p),p,'full_benchmark_v1',label,'video_query',2,'greedy, max512'+(', rep1.05' if m=='qwen25' else ''))
 status('thinking_and_restricted')
 for p in sorted(result_glob((relocate_path(ROOT.parent/'attention/results/thinking_pilot_orders_v1/samples')), '*/*/*/result.json')):
  r=load(p);add(r,p,'thinking_pilot_2fps','Qwen3-VL-8B-Thinking',r['order'],2,'sample T1 p.95 k20 seed42, max16384','raw')
 for m,label in [('qwen25','Qwen2.5-VL-7B-Instruct'),('qwen3','Qwen3-VL-8B-Instruct')]:
  for p in sorted(result_glob((relocate_path(ROOT/'results/timelens_span_head'/m)), 'shard*/predictions.jsonl')):
   b=p.read_bytes();sources.append(dict(path=os.path.relpath(p,ROOT),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
   for lineno,line in enumerate(b.splitlines(),1):add(json.loads(line),p,'restricted_head_526',label,'video_query',2,'greedy restricted526, max512',line=lineno)
 # Snapshot only atomically completed 8fps records; incomplete dataset groups remain explicit.
 paths=sorted(result_glob((relocate_path(ROOT/'results/thinking_vs_instruct_8fps_v1/samples')), '*/*/*/*/result.json'))
 for p in paths:
  r=load(p);m=r['model'];add(r,p,'pilot_8fps_v1','Qwen3-VL-8B-'+('Instruct' if m=='instruct' else 'Thinking'),r['order'],8,'greedy, max512' if m=='instruct' else 'sample T1 p.95 k20 seed42, max16384','raw')
 status('metrics')
 grouped=defaultdict(list)
 for r in records:
  cohorts=['pilot_300'] if r['run'] in ['legacy_qwen25_pilot','thinking_pilot_2fps','restricted_head_526','pilot_8fps_v1'] else ['full_9404']+(['pilot_300'] if r['is_pilot'] else [])
  for cohort in cohorts:
   for ds in [r['dataset'],'all']:grouped[(r['run'],r['model'],cohort,r['order'],r['fps'],r['policy'],ds)].append(r)
 metrics=[]
 for (run,model,cohort,order,fps,policy,ds),rr in sorted(grouped.items()):
  expected=(300 if ds=='all' else 100) if cohort=='pilot_300' else (9404 if ds=='all' else sum(s['dataset']==ds for s in samples.values()))
  row=dict(run=run,model=model,cohort=cohort,order=order,fps=fps,policy=policy,dataset=ds,n=len(rr),expected=expected,complete=len(rr)==expected,parse_failures=sum(r['parse_failed'] for r in rr),truncated=sum(r['truncated'] for r in rr),mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in rr))
  for scoring in ['raw','rounded','original']:
   vals=[r['iou_'+scoring] for r in rr];row['miou_'+scoring]=statistics.mean(vals)
   for t in [.3,.5,.7]:row[f'r{int(t*10):02d}_{scoring}']=statistics.mean(v>=t for v in vals)
  metrics.append(row)
 # Trace capture paths contain the same responses as the indexed full/pilot runs.
 # Compare run versions without pooling their observations or assuming identical generations.
 comparisons=[]
 for label in ['Qwen2.5-VL-7B-Instruct','Qwen3-VL-8B-Instruct']:
  legacy={r['sample_key']:r for r in records if r['model']==label and r['run'].startswith('legacy_') and r['order']=='video_query'}
  full={r['sample_key']:r for r in records if r['model']==label and r['run']=='full_benchmark_v1'}
  pairs=[(a,full[k]) for k,a in legacy.items() if k in full]
  comparisons.append(dict(model=label,matched=len(pairs),same_raw_iou=sum(a['iou_raw']==b['iou_raw'] for a,b in pairs),different_raw_iou=sum(a['iou_raw']!=b['iou_raw'] for a,b in pairs)))
 rollpath=relocate_path(ROOT/'results/timelens_rollout/summary.json');roll=load(rollpath);rollval=load(relocate_path(ROOT/'results/timelens_rollout/validation.json'));assert rollval['passed'] and roll['runs']==6000
 rolloutfiles=list(result_glob((relocate_path(ROOT/'results/timelens_rollout/records')), '*/*/*.json'));assert len(rolloutfiles)==6000
 inventory=[dict(run=run,records=sum(r['run']==run for r in records),source='raw responses, re-scored') for run in sorted({r['run'] for r in records})]+[dict(run='stochastic_rollout',records=6000,source='existing validated summary; 6000 response files counted, candidate traces not reloaded')]
 expectedcounts={'legacy_qwen3':18808,'legacy_qwen25_pilot':300,'full_benchmark_v1':18808,'thinking_pilot_2fps':600,'restricted_head_526':600}
 for run,n in expectedcounts.items():assert sum(r['run']==run for r in records)==n
 savecsv(relocate_path(OUT/'metrics.csv'),metrics);savecsv(relocate_path(OUT/'sample_metrics.csv'),records);savecsv(relocate_path(OUT/'rollout_metrics.csv'),roll['groups'])
 summary=dict(timestamp=now(),response_records_rescored=len(records),rollout_records_in_existing_validated_summary=6000,inventory=inventory,metrics=metrics,rollout=roll,run_version_comparisons=comparisons,source_count=len(sources),scoring='raw and integer-rounded, first predicted interval vs single GT; parse failures zero; no clipping',note='Cohorts overlap; pilot rows are subsets of full rows. Run versions and sampling repeats must never be pooled as independent queries.')
 write(relocate_path(OUT/'summary.json'),summary);write(relocate_path(OUT/'sources.json'),sources)
 write(relocate_path(OUT/'validation.json'),dict(passed=True,records_rescored=len(records),unique_run_model_sample_order=True,stored_iou_reproduced=True,gt_checked=True,expected_run_counts=expectedcounts,source_sha256_recorded=True,rollout_validation='existing validated summary and counted 6000 files; not a fresh candidate-trace verification'))
 lines=['# 기존 zero-shot 전체 집계','',f'집계 시각: {now()}. GPU 추론 없이 기존 응답 {len(records):,}개를 재채점하고, 별도 sampling rollout 6,000개는 기존 검증된 통계를 통합했다. 8fps는 집계 시점에 저장된 완료분만 포함한다.','', '동일 실행의 response·top30 trace·attention 사본은 한 번만 집계했다. pilot은 full의 부분집합이며, 서로 다른 실행 버전과 rollout을 합쳐 하나의 성능으로 계산하지 않았다.','', '## 전체 실행 목록','', '| Run | 응답 수 | 근거 |','|---|---:|---|']
 for x in inventory:lines.append(f"| {x['run']} | {x['records']:,} | {x['source']} |")
 def table(title,selected):
  lines.extend(['',f'## {title}','','모든 성능은 %. 본문은 반올림 없는 동일 parser 재채점 값이다. 원래 반올림 규칙 및 저장 점수는 CSV·JSON에서 함께 제공한다. `all`은 query 수에 따른 가중 평균이다.','', '| Run | 모델 | FPS | 입력 순서 | Dataset | N / 목표 | mIoU | R@.3 | R@.5 | R@.7 |','|---|---|---:|---|---|---:|---:|---:|---:|---:|'])
  for r in selected:lines.append(f"| {r['run']} | {r['model']} | {r['fps']} | {r['order']} | {r['dataset']} | {r['n']}/{r['expected']} | "+' | '.join(f"{100*r[k+'_raw']:.2f}" for k in ['miou','r03','r05','r07'])+' |')
 table('전체 벤치마크', [r for r in metrics if r['cohort']=='full_9404'])
 table('동일 pilot 300개', [r for r in metrics if r['cohort']=='pilot_300' and r['run']!='restricted_head_526' and r['fps']==2])
 table('8fps 진행 중인 결과', [r for r in metrics if r['fps']==8])
 lines+=['','부분 완료 행은 선택된 일부 샘플의 결과이므로 완료된 300개 결과와 직접 비교하면 안 된다. Instruct greedy(max512)와 Thinking sampling(T=1,p=.95,k=20,seed42,max16384)은 생성 정책과 예산이 달라 checkpoint만의 효과가 아니다. 모든 실행의 요청 FPS는 2 또는 8이며, 실제 프레임 sampling은 원본 fps 및 기존 최대 768프레임 제한을 따른다. min_tokens=64,total_tokens=14336을 유지한다.']
 table('별도 비교군: 제한 head', [r for r in metrics if r['run']=='restricted_head_526'])
 lines+=['','## 별도 비교군: stochastic rollout','', '기존 검증된 summary.json과 validation.json을 사용했다. 이 섹션은 원래 반올림 채점 값이며 raw 재채점이 아니다. 각 query 5회 생성, 벤치마크별 100 query / 500 rollout이다. Oracle any-of-5는 GT를 본 상한이며 실제 추론 성능이 아니다.','', '| 모델 | 방식 | Dataset | Query | Rollout | mIoU % | R@.5 % | Oracle any5 R@.5 % |','|---|---|---|---:|---:|---:|---:|---:|']
 for r in roll['groups']:lines.append(f"| {r['model']} | {r['variant']} | {r['dataset']} | {r['queries']} | {r['rollouts']} | {r['miou']:.2f} | {r['r50']:.2f} | {r['oracle_any5_r50']:.2f} |")
 lines+=['','## 해석 및 중복 처리','']
 for x in comparisons:lines.append(f"- {x['model']}: 기존 baseline과 full_benchmark_v1의 대응 {x['matched']:,}개 중 raw IoU가 달라진 응답은 {x['different_raw_iou']:,}개다. 실행 버전을 분리해 보존했다. 동일 IoU는 동일 생성 토큰을 뜻하지 않는다.")
 for o in ['video_query','query_video']:
  rr=[r for r in metrics if r['cohort']=='pilot_300' and r['dataset']=='all' and r['order']==o and r['run'] in ['legacy_qwen3','thinking_pilot_2fps']];by={r['run']:r for r in rr}
  a,b=by['legacy_qwen3'],by['thinking_pilot_2fps'];lines.append(f"- 2fps pilot {o}: Instruct {100*a['miou_raw']:.2f}%, Thinking {100*b['miou_raw']:.2f}%, 차이 {100*(b['miou_raw']-a['miou_raw']):+.2f}pp mIoU.")
 lines+=['','원본 예측과 파생 측정은 변경하지 않았다. smoke 검사·중단된 old cohort·성능과 무관한 entropy/attention 파생값은 별도 성능 실행으로 세지 않았다. top30 pilot은 full_benchmark_v1에서 재사용한 동일 응답이므로 중복 합산하지 않았다.','', '[지표 CSV](metrics.csv) · [샘플별 재채점 CSV](sample_metrics.csv) · [전체 JSON](summary.json) · [출처 SHA-256](sources.json) · [검증](validation.json) · [rollout 원 보고서](../timelens_rollout/REPORT.md)']
 write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
 register('analysis.all_zeroshot.v1','analyses',OUT,'All historical zero-shot run versions, matched pilot/full cohorts, raw and rounded scoring, rollout appendix','_lib/aggregate_all_zeroshot.py','multiple cohorts','complete',['legacy.qwen3','legacy.qwen25.pilot','full_benchmark_v1.predictions','thinking_vs_instruct.pilot300.v1','pilot.rollouts'])
 status('complete',metric_rows=len(metrics));print(json.dumps({'records':len(records),'metrics':len(metrics),'inventory':inventory,'report':str(relocate_path(OUT/'REPORT.md'))},ensure_ascii=False),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--watch-8fps',action='store_true');args=ap.parse_args()
 aggregate()
 if args.watch_8fps:
  while True:
   p=relocate_path(ROOT/'results/thinking_vs_instruct_8fps_v1/status.json');s=read(p) if p.exists() else {}
   write(relocate_path(OUT/'watcher.json'),dict(timestamp=now(),pid=os.getpid(),state='watching_8fps',inference_stage=s.get('stage'),inference_completed=s.get('completed')))
   if s.get('stage') in ['complete','failed']:
    aggregate();write(relocate_path(OUT/'watcher.json'),dict(timestamp=now(),pid=os.getpid(),state='refreshed',inference_stage=s['stage']));break
   time.sleep(30)
if __name__=='__main__':
 try:main()
 except BaseException:
  write(relocate_path(OUT/'status.json'),dict(stage='failed',timestamp=now(),error=traceback.format_exc()));raise
