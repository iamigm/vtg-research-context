"""Performance/cost reporting without any model calls or GT-dependent generation."""
import collections,csv,math,statistics,subprocess,sys
import numpy as np
from deterministic_rollout300 import ROOT,OUT,REPORT,BUDGETS,read,write

METHODS=['greedy','max_probability','medoid','probability_weighted_medoid','oracle']
def overlap(a,b):
 return max(0,min(a[1],b[1])-max(a[0],b[0]))/max(1e-12,max(a[1],b[1])-min(a[0],b[0]))
def interval(c):
 p=c['parsed']
 return [p['start'],p['end']] if c['stopped_on_eos'] and p and p['positive'] else None
def select(candidates,method,gt=None):
 valid=[c for c in candidates if interval(c) is not None]
 if method=='greedy':return candidates[0]
 if not valid:return candidates[0]
 if method=='max_probability':return max(valid,key=lambda c:(c['log_probability'],-c['index']))
 if method=='oracle':return max(valid,key=lambda c:(overlap(interval(c),gt),-c['index']))
 weights=[1.]*len(valid)
 if method=='probability_weighted_medoid':
  largest=max(c['log_probability'] for c in valid)
  weights=[math.exp(c['log_probability']-largest) for c in valid]
 # Select an existing candidate, maximizing pairwise temporal IoU agreement.
 scores=[sum(w*overlap(interval(c),interval(d)) for w,d in zip(weights,valid)) for c in valid]
 return valid[max(range(len(valid)),key=lambda i:(scores[i],valid[i]['log_probability'],-valid[i]['index']))]

def values(rows):
 return [statistics.mean(x['iou'] for x in rows)]+[sum(x['iou']>=t for x in rows)/len(rows) for t in [.3,.5,.7]]
def label(v):return f'{v:.4f}'

def report():
 records=[read(p) for p in sorted((OUT/'rollouts').glob('*.json'))]
 assert len(records)==300,'Do not publish complete results for an incomplete cohort'
 frozen=read(OUT/'metadata/cohort.json')
 assert {r['sample']['sample_key'] for r in records}=={s['sample_key'] for s in frozen}
 rows=[];roles=collections.defaultdict(list)
 for record in records:
  s=record['sample'];cs=record['result']['candidates'];gt=s['span'][0]
  assert len({tuple(c['ids']) for c in cs})==len(cs),'Duplicate response path'
  baseline=interval(cs[0]);baseline_score=overlap(baseline,gt) if baseline else 0.
  for c in cs[1:]:
   for role in c['branch']['roles']:
    pair=interval(c)
    roles[role].append(dict(sample_key=s['sample_key'],candidate=c['index'],
      candidate_iou=overlap(pair,gt) if pair else 0,baseline_iou=baseline_score,
      start_shift=pair[0]-baseline[0] if pair and baseline else None,
      end_shift=pair[1]-baseline[1] if pair and baseline else None))
  for budget in BUDGETS:
   pool=cs[:budget];last=pool[-1]
   for method in METHODS:
    chosen=select(pool,method,gt);pair=interval(chosen)
    # Greedy reference is costed as one answer regardless of the pool checkpoint.
    cost=cs[0] if method=='greedy' else last
    rows.append(dict(sample_key=s['sample_key'],dataset=s['dataset'],video=s['video_path'],
      duration=s['duration'],ratio=(gt[1]-gt[0])/s['duration'],budget=budget,
      available_candidates=len(pool),unique_intervals=len({tuple(interval(c)) for c in pool if interval(c)}),
      method=method,selected_index=chosen['index'],iou=overlap(pair,gt) if pair else 0.,
      parse_failure=pair is None,out_of_duration=bool(pair and pair[1]>s['duration']),
      start_mae=abs(pair[0]-gt[0]) if pair else None,end_mae=abs(pair[1]-gt[1]) if pair else None,
      decoder_tokens=cost['decoder_tokens'],seconds=cost['elapsed_seconds'],
      baseline_iou=baseline_score,clock=bool(chosen['parsed'] and chosen['parsed']['format']=='clock')))
 with (OUT/'metrics/per_query.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write(OUT/'metrics/digit_roles.json',dict(roles))
 base=[r for r in rows if r['budget']==1 and r['method']=='greedy'];bv=values(base)
 text=['# 결정적 분기 rollout: pilot300 성능과 계산량','',
 '고정 pilot300 전체(데이터셋별100개). 원래 프롬프트·2 FPS·공간 입력 유지. 시간 토큰의 대안을 중복 없이 선택한 뒤 원래 prefix 조건에서 EOS까지 greedy 완성했다. 숫자/소수점/콜론 위치를 기록하고 MM:SS/HH:MM:SS는 평가에서만 초로 환산했다.', '',
 '주 지표는 연속 시간 IoU=교집합 길이/합집합 길이, mIoU=300질문 평균, R@τ=IoU≥τ 질문 비율이다. 높을수록 좋다. clipping·반올림은 하지 않는다. 파싱 실패·역전 구간·EOS 미도달은0점이며 영상 길이 초과는 별도 표시한다. MAE는 유효 응답에 조건부인 초 단위 절대오차로 낮을수록 좋다.', '',
 'max_probability는 관측한 완성 경로 중 raw 전체 응답 확률이 가장 높은 답을 고른다. medoid는 후보 간 IoU 합의가 높은 답, probability_weighted_medoid는 관측 경로 확률로 가중한 합의가 높은 답이다. 이 세 선택기는 GT를 사용하지 않는다. oracle은 GT로 고른 후보 집합 상한이며 실제 성능과 구분한다. 결정적 후보 열거 빈도를 샘플링 빈도나 모델의 사건 확률로 해석하지 않는다.', '',
 '후보 예산에는 최초 greedy를 포함한다. 32개를 만들지 못하면 가용 후보 수를 기록하고 같은 질문을 제외하지 않는다. 같은 수치 구간도 다른 응답 토큰 경로면 별도 후보이며, 고유 수치 구간 수를 병기한다. decoder 토큰에는 rollback 재계산을 포함한다. 후보 선택 CPU 비용과 영상 prefill은 별도이며, 비교 표의 계산량은 실제 디코더 비용이다.', '',
 '| 후보 상한 | 선택 | mIoU (Δ) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | decoder 토큰 | 추가 토큰/greedy | 가용 후보 |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
 baseline_cost=statistics.mean(x['decoder_tokens'] for x in base)
 for budget in BUDGETS:
  for method in METHODS:
   if method=='greedy' and budget!=1:continue
   group=[r for r in rows if r['budget']==budget and r['method']==method];v=values(group)
   cost=statistics.mean(x['decoder_tokens'] for x in group)
   text.append(f'| {budget} | {method} | {v[0]:.4f} ({v[0]-bv[0]:+.4f}) | '+' | '.join(f'{v[j]:.4f} ({100*(v[j]-bv[j]):+.2f})' for j in range(1,4))+f' | {cost:.1f} | {cost-baseline_cost:.1f} | {statistics.mean(x["available_candidates"] for x in group):.2f} |')
 # Paired video bootstrap; all methods reuse identical resampled video IDs.
 videos=sorted({x['video'] for x in base});loc={v:i for i,v in enumerate(videos)}
 rng=np.random.default_rng(42);indices=rng.integers(len(videos),size=(2000,len(videos)))
 counts=np.array([sum(x['video']==v for x in base) for v in videos],dtype=float)
 denominator=counts[indices].sum(axis=1);cis=[]
 for budget in BUDGETS:
  for method in METHODS[1:]:
   group=[r for r in rows if r['budget']==budget and r['method']==method]
   sums=np.zeros((len(videos),4))
   for r in group:
    a=np.array([r['iou']]+[float(r['iou']>=t) for t in [.3,.5,.7]])
    b=np.array([r['baseline_iou']]+[float(r['baseline_iou']>=t) for t in [.3,.5,.7]])
    sums[loc[r['video']]]+=a-b
   draws=sums[indices].sum(axis=1)/denominator[:,None]
   ci=np.quantile(draws,[.025,.975],axis=0)
   cis.append(dict(budget=budget,method=method,delta_ci95=ci.tolist(),units=['IoU','fraction','fraction','fraction']))
 write(OUT/'metrics/paired_video_bootstrap.json',cis)
 text+=['', '## 32개 후보 선택의 오차·형식 진단','',
 '| 선택 | 파싱/완성 실패 | 영상 길이 초과 | 분:초 응답 | 시작 MAE | 종료 MAE |',
 '|---|---:|---:|---:|---:|---:|']
 for method in METHODS:
  group=[r for r in rows if r['budget']==32 and r['method']==method]
  valid=[r for r in group if r['start_mae'] is not None]
  mae=lambda k:f'{statistics.mean(r[k] for r in valid):.3f}' if valid else '—'
  text.append(f'| {method} | {sum(r["parse_failure"] for r in group)} | {sum(r["out_of_duration"] for r in group)} | {sum(r["clock"] for r in group)} | {mae("start_mae")} | {mae("end_mae")} |')
 text+=['', '자릿수별 분기 위치와 시작·종료 변화는 [digit_roles.json](../../artifacts/runs/'+OUT.name+'/metrics/digit_roles.json)에 있다. 자릿수 변화만으로 다른 사건이라고 판정하지 않으며 실제 사건 구분은 영상 검토가 필요하다.', '',
 '[영상 단위 paired bootstrap 95% CI](../../artifacts/runs/'+OUT.name+'/metrics/paired_video_bootstrap.json) · [길이/GT비율/교차표](STRATA.md) · [질문별 원자료 CSV](../../artifacts/runs/'+OUT.name+'/metrics/per_query.csv)', '',
 '이번 pilot300은 앞선 개발30개를 포함한다. 정책·후보 상한은 실행 전에 고정했고, 전체 결과를 독립 test 성능으로 부르지 않는다.']
 (REPORT/'REPORT.md').write_text('\n'.join(text)+'\n')
 strata(rows)
 # Validate aggregate denominators, original identities and GT-free selector invariance.
 assert len(rows)==300*len(BUDGETS)*len(METHODS)
 for record in records[:5]:
  for method in METHODS[1:-1]:
   cs=record['result']['candidates']
   assert select(cs,method,[0,1])['ids']==select(cs,method,[10000,10001])['ids']
 write(OUT/'validation/report.json',dict(passed=True,queries=300,rows=len(rows),
   unique_completions=True,gt_free_selectors=True,all_queries_in_denominator=True))
 subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)

def strata(rows):
 duration=[0,30,60,120,180,float('inf')];ratio=[0,.05,.1,.2,.4,1.000001]
 def bindex(v,bins):return next(i for i in range(5) if bins[i]<=v<bins[i+1])
 text=['# Pilot300 길이·GT비율별 성능','',
       '각 표는 같은 평가 기준의 mIoU·IoU≥.3/.5/.7 성공률이다. 교차표 셀은 n/점수, n<5는 small, 빈 셀은0/—. 실제 선택기와 oracle을 구분한다.','']
 for budget in BUDGETS:
  for method in METHODS:
   if method=='greedy' and budget!=1:continue
   group=[x for x in rows if x['budget']==budget and x['method']==method]
   for axis,bins in [('duration',duration),('ratio',ratio)]:
    text += [f'## 후보{budget} {method}: {axis}','', '| 구간 | n | mIoU | R@.3 | R@.5 | R@.7 |','|---|---:|---:|---:|---:|---:|']
    for i in range(5):
     cell=[x for x in group if bindex(x[axis],bins)==i]
     scores=' | '.join(map(label,values(cell))) if cell else '— | — | — | —'
     text.append(f'| [{bins[i]},{bins[i+1]}) | {len(cell)}'+(' small' if 0<len(cell)<5 else '')+' | '+scores+' |')
   for metric in range(4):
    text += [f'## 후보{budget} {method}: 길이×GT비율 '+['mIoU','R@.3','R@.5','R@.7'][metric],'', '| 영상 길이 | [0,.05) | [.05,.1) | [.1,.2) | [.2,.4) | [.4,1] |','|---|---|---|---|---|---|']
    for a in range(5):
     cells=[]
     for b in range(5):
      cell=[x for x in group if bindex(x['duration'],duration)==a and bindex(x['ratio'],ratio)==b]
      cells.append(f'{len(cell)}/'+(label(values(cell)[metric])+(' small' if len(cell)<5 else '') if cell else '—'))
     text.append(f'| [{duration[a]},{duration[a+1]}) | '+' | '.join(cells)+' |')
 (REPORT/'STRATA.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':report()
