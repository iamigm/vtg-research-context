"""Spatially controlled whole/boundary sweep: thresholds, axes, paired video CIs."""
import numpy as np
from adaptive_temporal_sampling import read,write,ROOT,objhash,need
from adaptive_temporal_report import score,csvwrite
from spatial_sweep900 import OUT,REPORT,CONFIG,RUN,AREAS,TEMPORAL
MET=[('miou','mIoU',None),('r03','IoU≥0.3',.3),('r05','IoU≥0.5',.5),('r07','IoU≥0.7',.7)]
TB=[(0,30,'<30'),(30,60,'30–60'),(60,120,'60–120'),(120,180,'120–180'),(180,float('inf'),'≥180')];QB=[(0,.05,'<5%'),(.05,.1,'5–10%'),(.1,.2,'10–20%'),(.2,.4,'20–40%'),(.4,1.000001,'≥40%')]
DS=[('charades-timelens','Charades'),('activitynet-timelens','ActivityNet'),('qvhighlights-timelens','QVHighlights')]
def table(h,rows):return ['','| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]+['']
def name(c):
 if c=='uniform':return 'Uniform'
 p=c.split('_');return ('GT' if p[0]=='gt' else 'Random')+' '+('전체' if p[1]=='whole' else '경계')+' '+p[2][1:]+'배'+(' 평균' if p[-1]=='mean' else (' seed'+p[-1][1:] if p[-1].startswith('s') else ''))
def report():
 REPORT.mkdir(parents=True,exist_ok=True);cohort=read(OUT/'metadata/cohort.json');ssmap={s['sample_key']:s for s in cohort};jobs=read(OUT/'metadata/active_jobs.json');records=[];cost=[]
 for j in jobs:
  k=j['sample_key'];r=read(OUT/'responses'/j['level']/j['condition']/(k+'.json'));detail=read(OUT/'inputs'/j['level']/j['condition']/(k+'.json'));need(r['identity']['input_hash']==objhash(detail['hashes']) and r['input_detail_hash']==objhash(detail),'Input binding mismatch');need((r['F'],r['B'],r['grid'])==(j['F'],j['B'],j['grid']),'Budget mismatch');s=ssmap[k]
  records.append(dict(sample_key=k,dataset=s['dataset'],level=j['level'],design=j['design'],condition=j['temporal_condition'],**score(r['answer'],s['span'][0],s['duration'])))
  cost.append(dict(sample_key=k,level=j['level'],condition=j['condition'],gpu_seconds=r['generation_seconds'],reused=r.get('reused_identical_input',False),tokens=r['generated_tokens'],input_tokens=r['total_input_tokens'],peak_GiB=r['peak_allocated_bytes']/2**30))
 write(OUT/'metrics/evaluation_records.json',records);csvwrite(OUT/'metrics/compute_cost.csv',cost)
 by={(r['design'],r['level'],r['sample_key'],r['condition']):r for r in records};T={s['sample_key']:read(OUT/'plans/A__uniform'/(s['sample_key']+'.json'))['T'] for s in cohort}
 conds=[c for c in TEMPORAL if not c.endswith('_s43')]
 metrics=[];axes=[];paired=[]
 for n in [300,600,900]:
  for ds,label in DS:
   ss=[s for s in cohort[:n] if s['dataset']==ds];videos=sorted({s['video_path'] for s in ss});vi={v:i for i,v in enumerate(videos)};membership=np.array([vi[s['video_path']] for s in ss]);counts=np.bincount(membership);draws=np.random.default_rng(42).multinomial(len(videos),np.full(len(videos),1/len(videos)),size=20000);den=draws@counts
   for d in ['A','B']:
    for level in AREAS:
     for m,ml,th in MET:
      vals={}
      for c in conds:
       members=[c]
       vals[c]=np.array([np.mean([by[d,level,s['sample_key'],cc]['iou'] if th is None else float(by[d,level,s['sample_key'],cc]['iou']>=th) for cc in members]) for s in ss])
       metrics.append(dict(size=n,dataset=ds,design=d,level=level,condition=c,metric=m,N=len(ss),percent=float(vals[c].mean()*100)))
      comparisons=[]
      for r in [2,4,8]:
       for mode in ['whole','boundary']:comparisons.extend([(f'gt_{mode}_r{r}','uniform'),(f'gt_{mode}_r{r}',f'random_{mode}_r{r}_s42')])
       comparisons.append((f'gt_boundary_r{r}',f'gt_whole_r{r}'))
      for a,b in comparisons:
       diff=vals[a]-vals[b];rep=(draws@np.bincount(membership,weights=diff,minlength=len(videos)))/den
       paired.append(dict(size=n,dataset=ds,design=d,level=level,a=a,b=b,metric=m,delta_pp=float(diff.mean()*100),ci95=list(np.quantile(rep*100,[.025,.975]))))
      if n!=900:continue
      for axis in ['duration','GT_fraction','joint']:
       for lo,hi,t in (TB if axis!='GT_fraction' else [(0,float('inf'),'all')]):
        for qlo,qhi,q in (QB if axis!='duration' else [(0,1.000001,'all')]):
         mask=np.array([lo<=T[s['sample_key']]<hi and qlo<=(s['span'][0][1]-s['span'][0][0])/T[s['sample_key']]<qhi for s in ss]);N=int(mask.sum())
         for c in [x for x in conds if x.startswith('gt_')]:
          random=c.replace('gt_','random_')+'_s42'
          axes.append(dict(dataset=ds,design=d,level=level,axis=axis,time=t,fraction=q,condition=c,metric=m,N=N,delta_uniform_pp=float((vals[c][mask]-vals['uniform'][mask]).mean()*100) if N else None,delta_random_pp=float((vals[c][mask]-vals[random][mask]).mean()*100) if N else None))
 csvwrite(OUT/'metrics/metrics.csv',metrics);csvwrite(OUT/'metrics/axes.csv',axes);write(OUT/'metrics/paired.json',paired)
 ix={(x['size'],x['dataset'],x['design'],x['level'],x['condition'],x['metric']):x['percent'] for x in metrics};ax={(x['dataset'],x['design'],x['level'],x['axis'],x['time'],x['fraction'],x['condition'],x['metric']):x for x in axes}
 lines=['# 두 공간 예산의 전체·경계 집중: pilot900','', '공간 해상도를 통제하지 않은 이전 실험 결과는 폐기했습니다. 이 보고서는 낮음(low)·높음(high) 두 목표 면적에서 수행한 결과만 포함합니다. 각 데이터셋300개, Random은 seed42 하나만 사용합니다. 실행 중 사용자 요청으로 seed43을 제외했으며, 이미 생성된 seed43 응답은 보존하되 이 보고서에는 포함하지 않습니다. 기존 공간 통제 실험과 입력이 동일한 응답은 출처를 기록해 재사용했습니다.','',
 'mIoU는 예측·정답 시간 구간의 교집합/합집합을 질문별 평균한 값입니다. IoU≥.3/.5/.7 성공률은 해당 기준 이상인 질문 수/전체 질문 수입니다. 모두 %이며 높을수록 좋습니다. 실패도 분모에 포함하고0점으로 셉니다. pp는 백분율 차이입니다. Random은 seed42의 점수입니다. 여러 seed에 따른 변동성은 평가하지 않습니다.','',
 'A는 GT 전체와 경계에 같은 수의 집중 프레임을 배정합니다. B는 각 집중 영역의 실제 길이로 안팎 밀도 비율2/4/8을 맞춥니다. 전체 조건과 Uniform은 두 방식에서 동일 입력이며 독립 반복이 아닙니다. 경계는 시작·끝 ±1초의 합집합입니다.','',
 'low/high 목표 면적은50,176/100,352픽셀입니다. H/W32배수, 면적±10% 안에서 원본 종횡비 오차를 최소화합니다. 실제 픽셀·종횡비 오차는 입력 표에 기록합니다. 총 영상 토큰 예산은 고정하지 않으며 같은 질문·공간 설정 내에서는 모든 조건의 F/grid/B를 맞춥니다.','',
 '95% 구간은 영상 단위 cluster bootstrap20,000회(seed42)입니다. 같은 영상 질문을 함께 재표집하며 다중비교 보정 없는 탐색 결과입니다.900개는 기존300개를 포함한 층화 진단 표본이며 원래 벤치마크 분포의 무작위 평균이 아닙니다.','',
 '## 모든 조건의900개 실제 점수','', '표의 괄호는 Uniform 대비 변화pp입니다. 각 설정별 상세 보고서에는 GT−Random, 경계−전체, 신뢰구간, 영상 길이별·GT 비율별·두 축 교차표가 있습니다.']
 for d in ['A','B']:
  for level in AREAS:
   detail=[f'# {d} / {level}: 상세 비교','', '[지표 정의와 전체 결과](REPORT.md)','', '각 축 표는 GT−Uniform / GT−대응 Random seed42의 pp입니다. 교차표도 두 차이를 함께 표시하며 괄호는 질문 수입니다. †는10개 미만, —는 빈 칸입니다. 왼쪽 구간 경계 포함·오른쪽 제외(마지막 GT 구간은100% 포함). 각 축을 따로 나누면 다른 축은 통제하지 않으며 교차표는 두 축을 동시에 나눕니다.']
   lines+=['',f'### {d} / {level}',f'[각 축·두 축 교차표·신뢰구간]({d}_{level}.md)']
   for ds,label in DS:
    lines+=['','#### '+label]+table(['조건']+[ml+' % (변화pp)' for _,ml,_ in MET],[[name(c)]+[f'{ix[900,ds,d,level,c,m]:.2f} ({ix[900,ds,d,level,c,m]-ix[900,ds,d,level,"uniform",m]:+.2f})' for m,_,_ in MET] for c in conds])
    for n in [300,600,900]:
     detail+=['',f'## {label} / 중첩{n}개 코호트 / 해당 데이터셋{n//3}개']+table(['조건']+[ml+' %' for _,ml,_ in MET],[[name(c)]+[f'{ix[n,ds,d,level,c,m]:.2f}' for m,_,_ in MET] for c in conds])
     detail+=table(['비교','지표','차이pp [95% 구간]'],[[name(x['a'])+' − '+name(x['b']),x['metric'],f"{x['delta_pp']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"] for x in paired if x['size']==n and x['dataset']==ds and x['design']==d and x['level']==level])
    for c in [c for c in conds if c.startswith('gt_')]:
     detail+=['',f'## {label} / {name(c)} / 900개 코호트']
     for axis,title,bins in [('duration','영상 길이별',TB),('GT_fraction','GT 비율별',QB)]:
      rs=[]
      for _,_,b in bins:
       t,q=(b,'all') if axis=='duration' else ('all',b);xs=[ax[ds,d,level,axis,t,q,c,m] for m,_,_ in MET]
       rs.append([b,xs[0]['N']]+['—' if not x['N'] else f"{x['delta_uniform_pp']:+.2f} / {x['delta_random_pp']:+.2f}" for x in xs])
      detail+=['','### '+title]+table(['구간','N']+[ml for _,ml,_ in MET],rs)
     for m,ml,_ in MET:
      rs=[]
      for _,_,q in QB:
       xs=[ax[ds,d,level,'joint',t,q,c,m] for _,_,t in TB];rs.append([q]+['— (0)' if not x['N'] else f"{x['delta_uniform_pp']:+.2f} / {x['delta_random_pp']:+.2f} ({x['N']})"+(' †' if x['N']<10 else '') for x in xs])
      detail+=['','### 두 축 교차표: '+ml]+table(['GT 비율']+[t+'초' for _,_,t in TB],rs)
   (REPORT/f'{d}_{level}.md').write_text('\n'.join(detail)+'\n')
 lines+=['','## 비용',f"논리 응답{len(jobs)}개, 기존 동일 입력 재사용{sum(x['reused'] for x in cost)}개. GPU 생성 시간 합계{sum(x['gpu_seconds'] for x in cost):.1f}초. 원자료와 입력별 비용은 artifacts/runs/{RUN}/metrics/에 있습니다."]
 (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 for p in REPORT.glob('*.md'):
  width=None
  for l in p.read_text().splitlines():
   if l.startswith('|'):
    n=len(l.split('|'))-2
    if width is None:width=n
    need(width==n,'Broken table '+str(p))
   else:width=None
 write(OUT/'validation/final.json',dict(passed=True,logical_responses=len(jobs),queries=900,thresholds=[.3,.5,.7],axes=['duration','GT_fraction','joint'],no_discarded_predictions_used=True))
if __name__=='__main__':report()
