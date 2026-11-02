"""CPU-only paired analysis of historical Qwen3 input-order predictions."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import collections, csv, hashlib, importlib.util, json, math, statistics, time
from datetime import datetime, timezone
OUT=Path('/home/irteam/experiments/zero-shot/results/input_order/consistency_v1/analyze.py').resolve().parent
ROOT=OUT.parents[2]
SOURCE=relocate_path(ROOT/'predictions/qwen3')
PARSER=relocate_path(ROOT.parent/'shared/timelens/vendor/TimeLens/timelens/utils.py')
spec=importlib.util.spec_from_file_location('original_parser',PARSER)
parser=importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens']
rows=[]; hashes=[]; started=time.monotonic()
def dump(name,obj): write_result_text((relocate_path(OUT/name)), json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def progress(state):
 d=dict(timestamp=datetime.now(timezone.utc).isoformat(),state=state,stage='CPU paired prediction analysis',completed=len(rows),total=9404,percent=100*len(rows)/9404,elapsed_seconds=time.monotonic()-started,counts=dict(collections.Counter(r['dataset'] for r in rows)))
 dump('status.json',d)
 with (relocate_path(OUT/'progress.log')).open('a') as f: f.write(json.dumps(d)+'\n')
def read(p):
 data=p.read_bytes(); hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())); return json.loads(data)
def overlap(a,b):
 if a is None or b is None or a[1]<=a[0] or b[1]<=b[0]: return None
 inter=max(0,min(a[1],b[1])-max(a[0],b[0])); return inter/((a[1]-a[0])+(b[1]-b[0])-inter)
def quant(vals,q):
 vals=sorted(vals); k=(len(vals)-1)*q; lo=int(k); hi=math.ceil(k); return vals[lo]+(vals[hi]-vals[lo])*(k-lo) if vals else None
progress('running')
for ds in DATASETS:
 aa={p.parent.parent.name:p for p in result_glob((relocate_path(SOURCE/ds)), '*/video_query/answer.json')}
 bb={p.parent.parent.name:p for p in result_glob((relocate_path(SOURCE/ds)), '*/query_video/answer.json')}
 assert aa.keys()==bb.keys()
 for key in sorted(aa):
  a,b=read(aa[key]),read(bb[key])
  assert all(a[k]==b[k] for k in ['sample_key','dataset','video_name','query','gt','duration'])
  x,y=a['timestamps'][0],b['timestamps'][0]
  px,py=parser.extract_time(a['answer']),parser.extract_time(b['answer'])
  for d,p in [(a,px),(b,py)]:
   assert [[round(s),round(e)] for s,e in (p or [[d['duration']+10,d['duration']+20]])]==d['timestamps']
   assert abs(parser.iou(d['gt'],d['timestamps'][0])-d['iou'])<1e-10
  rx,ry=(px[0] if px else None),(py[0] if py else None)
  row=dict(dataset=ds,sample_key=key,video=a['video_name'],query=a['query'],duration=a['duration'],gt=a['gt'],video_query=x,query_video=y,raw_video_query=rx,raw_query_video=ry,answer_video_query=a['answer'],answer_query_video=b['answer'],text_equal=a['answer']==b['answer'],tokens_equal=a['generated_ids']==b['generated_ids'],rounded_equal=x==y,raw_equal=rx==ry if rx and ry else False,pair_iou=overlap(x,y) if px and py else None,raw_pair_iou=overlap(rx,ry),start_abs_diff=abs(x[0]-y[0]),end_abs_diff=abs(x[1]-y[1]),start_shift=y[0]-x[0],end_shift=y[1]-x[1],iou_video_query=a['iou'],iou_query_video=b['iou'],iou_delta=b['iou']-a['iou'])
  row['both_parsed']=bool(px and py)
  row['normalized_max_endpoint_diff']=max(row['start_abs_diff'],row['end_abs_diff'])/a['duration'] if px and py else None
  for tag,d,p in [('vq',a,px),('qv',b,py)]:
   row['parse_failed_'+tag]=not bool(p); row['truncated_'+tag]=d['truncated']; row['nonpositive_'+tag]=d['timestamps'][0][1]<=d['timestamps'][0][0]
   row['out_of_bounds_'+tag]=d['timestamps'][0][0]<0 or d['timestamps'][0][1]>d['duration']
  rows.append(row)
 progress('running')
assert len(rows)==9404

def summarize(rs):
 n=len(rs); pct=lambda count:100*count/n
 s=dict(n=n)
 for k in ['text_equal','tokens_equal','rounded_equal','raw_equal','parse_failed_vq','parse_failed_qv','truncated_vq','truncated_qv','nonpositive_vq','nonpositive_qv','out_of_bounds_vq','out_of_bounds_qv']:
  count=sum(r[k] for r in rs); s[k]=dict(count=count,pct=pct(count))
 for k in ['pair_iou','raw_pair_iou','start_abs_diff','end_abs_diff','start_shift','end_shift','normalized_max_endpoint_diff','iou_video_query','iou_query_video','iou_delta']:
  v=[r[k] for r in rs if r[k] is not None and (k not in ['start_abs_diff','end_abs_diff','start_shift','end_shift'] or r['both_parsed'])]; s[k]=dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),p90=quant(v,.9))
 for t in [.3,.5,.7,.9]:
  valid=[r for r in rs if r['pair_iou'] is not None]
  s['pair_iou_ge_'+str(t)]=dict(count=sum(r['pair_iou']>=t for r in valid),denominator=len(valid),pct=100*sum(r['pair_iou']>=t for r in valid)/len(valid))
 for t in [1,2,5]:
  valid=[r for r in rs if r['both_parsed']]; count=sum(max(r['start_abs_diff'],r['end_abs_diff'])<=t for r in valid)
  s['both_endpoints_within_'+str(t)+'s']=dict(count=count,denominator=len(valid),pct=100*count/len(valid))
 s['zero_overlap']=dict(count=sum(r['pair_iou']==0 for r in rs),denominator=s['pair_iou']['n'])
 s['gt_comparison']={label:dict(count=sum(test(r['iou_delta']) for r in rs),pct=pct(sum(test(r['iou_delta']) for r in rs))) for label,test in [('query_first_better',lambda v:v>1e-10),('video_first_better',lambda v:v< -1e-10),('tie',lambda v:abs(v)<=1e-10)]}
 s['success']={}
 for t in [.3,.5,.7]:
  counts=collections.Counter(('both_success' if a and b else 'video_only' if a else 'query_only' if b else 'both_fail') for r in rs for a,b in [(r['iou_video_query']>=t,r['iou_query_video']>=t)])
  assert sum(counts.values())==n
  s['success'][str(t)]={k:dict(count=counts[k],pct=pct(counts[k])) for k in ['both_success','video_only','query_only','both_fail']}
 return s
summary={ds:summarize([r for r in rows if r['dataset']==ds]) for ds in DATASETS}; summary['all']=summarize(rows)
# Independently reproduce published per-dataset metrics.
published=json.loads((relocate_path(OUT.parent/'summary.json')).read_text())['metrics']
for m in published:
 if m['cohort']!='full_9404': continue
 s=summary[m['dataset']]; assert abs(s['iou_'+m['order']]['mean']-m['miou'])<1e-10
 rs=[r for r in rows if r['dataset']==m['dataset']]
 for t,k in [(.3,'r03'),(.5,'r05'),(.7,'r07')]: assert abs(sum(r['iou_'+m['order']]>=t for r in rs)/len(rs)-m[k])<1e-10

def csvwrite(name,rs):
 with (relocate_path(OUT/name)).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,tuple)) else v for k,v in r.items()} for r in rs)
csvwrite('paired_samples.csv',rows)
csvwrite('largest_gt_degradation.csv',sorted(rows,key=lambda r:r['iou_delta'])[:50])
csvwrite('largest_gt_improvement.csv',sorted(rows,key=lambda r:r['iou_delta'],reverse=True)[:50])
csvwrite('largest_normalized_shift.csv',sorted([r for r in rows if r['both_parsed']],key=lambda r:r['normalized_max_endpoint_diff'],reverse=True)[:50])
dump('summary.json',summary); dump('source_manifest.json',dict(model='Qwen3-VL-8B-Instruct',cohort='legacy full_9404',orders=['video_query','query_video'],parser_path=str(PARSER),parser_sha256=hashlib.sha256(PARSER.read_bytes()).hexdigest(),sources=hashes))
dump('validation.json',dict(passed=True,paired_samples=len(rows),source_records=len(hashes),same_sample_identity=True,original_parser_rounding_verified=True,gt_iou_recomputed=True,published_metrics_reproduced=True,success_tables_sum_verified=True))
S=summary['all']; fmt=lambda x:f'{x:.2f}'
lines=['# Qwen3 입력 순서에 따른 예측 일관성','',f"동일한 영상–쿼리 9,404쌍에서 video → query와 query → video를 비교했다. 평가용 구간 완전 일치율은 {S['rounded_equal']['pct']:.2f}%, 두 예측의 평균 상호 IoU는 {S['pair_iou']['mean']:.4f}다. 정답 IoU 0.5 기준 성공 여부가 바뀐 비율은 {S['success']['0.5']['video_only']['pct']+S['success']['0.5']['query_only']['pct']:.2f}%다. 이 실행의 예측은 입력 순서에 민감하다.",'',
'기존 Qwen3-VL-8B-Instruct 생성 결과만 CPU로 분석했다. 표본 단위는 영상–쿼리 쌍이며 전체 값은 샘플 수로 가중한 집계다. 두 순서의 샘플 ID, 영상, 쿼리, 정답 구간, 영상 길이가 모두 일치한다.','',
'## 구간 일관성','',
'평가용 timestamps는 원문 시간에 Python round를 적용한 정수 초다. 원문 구간은 기존 TimeLens extract_time으로 재파싱했다. 구간 일치와 응답 텍스트 일치는 별도 지표다. 상호 IoU는 두 예측 구간의 교집합 길이 / 합집합 길이이며, 양쪽 모두 파싱에 성공하고 양의 길이를 가진 쌍만 포함한다. 파싱 실패의 평가용 대체 구간은 상호 IoU에서 제외한다. 길이가 0 이하인 쌍은 제외하고 개수를 별도 표시한다. 영상 범위로 클리핑하지 않는다.','',
'| 데이터셋 | N | 정수 구간 일치 % | 원문 구간 일치 % | 텍스트 일치 % | 상호 IoU 평균 | 상호 IoU 중앙값 | 상호 IoU ≥ .5 % | 유효 쌍 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for ds,s in summary.items(): lines.append(f"| {ds} | {s['n']} | {s['rounded_equal']['pct']:.2f} | {s['raw_equal']['pct']:.2f} | {s['text_equal']['pct']:.2f} | {s['pair_iou']['mean']:.4f} | {s['pair_iou']['median']:.4f} | {s['pair_iou_ge_0.5']['pct']:.2f} | {s['pair_iou']['n']} |")
lines += ['',f"전체 생성 토큰 ID 완전 일치율: {S['tokens_equal']['pct']:.2f}%. 원문 소수 구간의 상호 IoU 평균: {S['raw_pair_iou']['mean']:.4f} (유효 {S['raw_pair_iou']['n']}쌍). 평가용 유효 구간의 무겹침: {S['zero_overlap']['count']} / {S['zero_overlap']['denominator']}쌍.",'','## 시작 및 종료 시각 차이','', '양쪽 파싱에 성공한 9,336쌍의 정수 초 평가 구간 기준이다. 실패 시 대체 구간은 시간 차이 집계에서 제외한다. 허용 오차는 시작과 종료가 모두 조건을 만족해야 한다.','', '| 데이터셋 | 시작 차이 평균 / 중앙값 / P90 초 | 종료 차이 평균 / 중앙값 / P90 초 | 둘 다 1초 이내 % | 둘 다 2초 이내 % | 둘 다 5초 이내 % |','|---|---:|---:|---:|---:|---:|']
for ds,s in summary.items():
 vals=[' / '.join(fmt(s[k][v]) for v in ['mean','median','p90']) for k in ['start_abs_diff','end_abs_diff']]
 lines.append(f"| {ds} | {vals[0]} | {vals[1]} | {s['both_endpoints_within_1s']['pct']:.2f} | {s['both_endpoints_within_2s']['pct']:.2f} | {s['both_endpoints_within_5s']['pct']:.2f} |")
lines += ['','## 정답 기준 성공 여부의 변화','','성공은 정답과의 IoU가 임계값 이상인 경우다. 아래 비율의 분모는 해당 데이터셋 전체 표본이다.','', '| 데이터셋 | 임계값 | 둘 다 성공 % | video 먼저만 성공 % | query 먼저만 성공 % | 둘 다 실패 % |','|---|---:|---:|---:|---:|---:|']
for ds,s in summary.items():
 for t,c in s['success'].items(): lines.append('| '+ds+' | '+t+' | '+' | '.join(fmt(c[k]['pct']) for k in ['both_success','video_only','query_only','both_fail'])+' |')
lines += ['','## 정답 IoU 변화','','Δ는 query 먼저 − video 먼저다. 평균은 0–1 스케일이다.','', '| 데이터셋 | video 먼저 mIoU | query 먼저 mIoU | 평균 Δ | query 먼저 개선 % | 악화 % | 동일 % |','|---|---:|---:|---:|---:|---:|---:|']
for ds,s in summary.items(): lines.append(f"| {ds} | {s['iou_video_query']['mean']:.4f} | {s['iou_query_video']['mean']:.4f} | {s['iou_delta']['mean']:.4f} | {s['gt_comparison']['query_first_better']['pct']:.2f} | {s['gt_comparison']['video_first_better']['pct']:.2f} | {s['gt_comparison']['tie']['pct']:.2f} |")
lines += ['','## 파싱과 구간 품질','','범위 초과는 저장된 annotation duration 기준이며, 반올림 또는 영상 metadata 차이도 포함할 수 있다. 자동으로 생성 오류라고 판정하지 않는다.','', '| 데이터셋 | 파싱 실패 V/Q | 잘림 V/Q | 길이 0 이하 V/Q | 영상 범위 초과 V/Q |','|---|---:|---:|---:|---:|']
for ds,s in summary.items(): lines.append('| '+ds+' | '+' | '.join(str(s[k+'_vq']['count'])+' / '+str(s[k+'_qv']['count']) for k in ['parse_failed','truncated','nonpositive','out_of_bounds'])+' |')
lines += ['','## 해석과 재현','','두 예측이 일치해도 둘 다 틀릴 수 있으므로 상호 IoU와 정답 기준 성공 여부를 함께 봐야 한다. 데이터셋마다 영상 길이가 달라 초 단위 차이를 직접 비교할 때 주의가 필요하다. 샘플별 CSV에는 영상 길이로 정규화한 최대 경계 차이도 포함했다. 이 분석은 저장된 한 번의 deterministic 생성 비교이며 반복 실행 변동성이나 attention 변화의 인과를 측정하지 않는다.','',
'[전체 수치](summary.json) · [샘플별 비교](paired_samples.csv) · [검증](validation.json) · [소스 SHA-256](source_manifest.json) · [재현 코드](analyze.py)','',
'[정답 IoU 최대 악화 50개](largest_gt_degradation.csv) · [최대 개선 50개](largest_gt_improvement.csv) · [영상 길이 대비 최대 이동 50개](largest_normalized_shift.csv)','',
'재현: `python experiments/zero-shot/results/input_order/consistency_v1/analyze.py`']
write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
progress('complete')
print(json.dumps(summary,ensure_ascii=False,indent=2))
