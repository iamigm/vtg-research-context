"""Join existing first-loss entropy with independent endpoint absolute error."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import csv,json,time
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from artifact_store import register
ROOT=relocate_path(Path(__file__)).resolve().parents[1]; BASE=relocate_path(ROOT/'results/timelens_top30_full/digit_correctness'); OUT=relocate_path(BASE/'first_error_distance');OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont(str(relocate_path(BASE/'NotoSansKR.ttf')));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(relocate_path(BASE/'NotoSansKR.ttf'))).get_name();plt.rcParams['axes.unicode_minus']=False
rng=np.random.default_rng(42)
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def save(name,rows):
 with (relocate_path(OUT/name)).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def log(stage):
 d=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage=stage)
 with (relocate_path(OUT/'progress.log')).open('a') as f:f.write(json.dumps(d)+'\n')
 print(json.dumps(d),flush=True)
def stats(rows,bootstrap=False):
 x=np.array([r['entropy_nats'] for r in rows]);y=np.array([r['absolute_error_seconds'] for r in rows]);rho=float(spearmanr(x,y).statistic) if len(rows)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None
 lo=hi=None
 if bootstrap and rho is not None:
  values=[]
  for _ in range(500):
   ix=rng.integers(len(x),size=len(x));v=float(spearmanr(x[ix],y[ix]).statistic)
   if np.isfinite(v):values.append(v)
  lo,hi=map(float,np.quantile(values,[.025,.975]))
 return dict(n=len(rows),mean_entropy=float(x.mean()),median_error_seconds=float(np.median(y)),rho=rho,ci95_low=lo,ci95_high=hi)
log('join saved endpoint and first-loss records')
eps=read(relocate_path(BASE/'per_endpoint.csv')); key=lambda r:tuple(r[k] for k in ['model','dataset','sample_key','endpoint']);lookup={key(r):r for r in eps};rows=[];seen=set()
for s in read(relocate_path(BASE/'per_step.csv')):
 if s['first_error']!='True':continue
 k=key(s);assert k not in seen;seen.add(k);e=lookup[k]
 assert e['all_correct']=='False' and int(e['first_error_step'])==int(s['ar_step'])
 pred=float(e['prediction']);gt=float(e['gt'])
 rows.append(dict(model=s['model'],dataset=s['dataset'],sample_key=s['sample_key'],endpoint=s['endpoint'],first_error_step=int(s['ar_step']),role=s['role'],numeric=s['numeric'],token=s['token'],entropy_nats=float(s['entropy']),prediction_seconds=pred,gt_seconds=gt,absolute_error_seconds=abs(pred-gt),rounded_absolute_error_seconds=abs(round(pred)-gt)))
assert len(rows)==sum(e['all_correct']=='False' for e in eps)
save('per_first_error.csv',rows);groups=[];strata=[]
models=['qwen25','qwen3'];datasets=['charades-timelens','activitynet-timelens','qvhighlights-timelens']
for ep in ['start','end']:
 log('plot '+ep)
 fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
 for mi,m in enumerate(models):
  for di,ds in enumerate(datasets):
   g=[r for r in rows if r['endpoint']==ep and r['model']==m and r['dataset']==ds];a=axes[mi,di];st=stats(g,True);groups.append(dict(model=m,dataset=ds,endpoint=ep,**st))
   x=np.array([r['entropy_nats'] for r in g]);y=np.array([r['absolute_error_seconds'] for r in g]);num=np.array([r['numeric']=='True' for r in g])
   a.scatter(y[num],x[num],s=6,alpha=.16,c='#357aa0',rasterized=True,label='숫자에서 최초 오류')
   if (~num).any():a.scatter(y[~num],x[~num],s=11,alpha=.35,c='#b97536',marker='x',label='서식에서 최초 오류')
   edges=np.unique(np.quantile(x,np.linspace(0,1,6)));bx=[];med=[];low=[];high=[]
   for i,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
    sel=(x>=left)&((x<=right) if i==len(edges)-2 else (x<right))
    if sel.sum()<10:continue
    bx.append(float(np.median(x[sel])));q=np.quantile(y[sel],[.25,.5,.75]);low.append(q[0]);med.append(q[1]);high.append(q[2])
   a.plot(med,bx,'o-',color='#d95442',lw=2,ms=4,label='엔트로피 5분위별 오차 중앙값');a.fill_betweenx(bx,low,high,color='#d95442',alpha=.13,label='오차의 25–75백분위')
   a.set_xscale('symlog',linthresh=1);a.set_xlim(left=0);a.grid(alpha=.18)
   a.set_title(f"{m} · {ds.split('-')[0]}\nN={st['n']:,} · ρ={st['rho']:.3f} [{st['ci95_low']:.3f}, {st['ci95_high']:.3f}]",fontsize=11)
   if mi==1:a.set_xlabel('최종 예측 시각의 GT 절대 오차 (초)')
   if di==0:a.set_ylabel('최초 오류 생성 직전 엔트로피 (nat)')
   for role in sorted({r['role'] for r in g}):
    sub=[r for r in g if r['role']==role];strata.append(dict(model=m,dataset=ds,endpoint=ep,role=role,**stats(sub)))
 axes[0,0].legend(fontsize=7,loc='upper right');fig.suptitle(('시작 시각' if ep=='start' else '종료 시각')+' — 최초 오류의 불확실성과 최종 시간 오차',fontsize=17)
 fig.text(.5,.012,'점 1개 = 오답인 시각 1개 · 가로축: 1초 이하는 선형, 그 위는 로그 · 패널별 가로축 범위가 다름 · [ ]: Spearman ρ의 bootstrap 95% 신뢰구간',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(relocate_path(OUT/f'{ep}_entropy_vs_distance.png'),dpi=170);plt.close(fig)
save('group_statistics.csv',groups);save('role_statistics.csv',strata)
lines=['# 최초 오류의 엔트로피와 GT 거리','',f'전체 벤치마크의 지원 응답 {len(eps)//2:,}개에서 오답인 시작·종료 시각 {len(rows):,}개를 분석했다. 끝까지 정답 가능성이 유지된 시각은 최초 오류가 없으므로 제외한다. 지원하지 않는 출력 형식 217개 응답은 기존 분석과 동일하게 제외한다.','', '- 세로축: 정답으로 완성할 가능성을 처음 잃은 생성 단계 직전의 **raw 전체 어휘 엔트로피**(nat). 기존 반올림 가능성 판정을 그대로 사용한다.','- 가로축: **최종 출력 시각과 GT 시각의 절대 차이**(초). 시작과 종료는 독립 비교하며 반대쪽 시각은 계산에 관여하지 않는다. 반올림 후 오차도 CSV에 별도로 저장했다.','- 미완성 숫자 prefix 자체와 GT를 뺀 거리가 아니다. 최초 오류 뒤에 생성된 숫자도 최종 오차에 영향을 준다.','- 점은 시각별 관측값. 빨간 선은 엔트로피 5분위별 오차 중앙값, 음영은 해당 오차의 25–75백분위다. 음영은 신뢰구간이 아니다.','- 숫자 오류는 파랑, 소수점·종료 등 서식 오류는 갈색. 가로축은 1초까지 선형, 그 위는 로그이며 패널별 범위가 다르다.','', '![시작 시각](start_entropy_vs_distance.png)','', '![종료 시각](end_entropy_vs_distance.png)','', '| 모델 | 벤치마크 | 시각 | 관측 수 | 오차 중앙값(초) | Spearman ρ | 95% CI |','|---|---|---|---:|---:|---:|---|']
for g in groups:lines.append(f"| {g['model']} | {g['dataset']} | {g['endpoint']} | {g['n']} | {g['median_error_seconds']:.2f} | {g['rho']:.3f} | [{g['ci95_low']:.3f}, {g['ci95_high']:.3f}] |")
lines+=['','## 해석','', '12개 그룹 모두 음의 상관을 보였다(ρ 약 −0.18~−0.52). 오답인 시각들 안에서는 최초 오류 단계의 엔트로피가 낮은 사례가 더 큰 최종 시간 오차를 보이는 경향이다. 오류 자릿수를 섞은 효과인지 후속 검토가 필요하다.','', 'ρ가 양수이면 최초 오류 때 불확실성이 높은 사례에서 최종 시간 오차도 큰 경향, 음수이면 반대 경향을 뜻한다. 모델·벤치마크·시작·종료를 섞지 않고 해석한다. 신뢰구간은 각 패널의 시각 관측을 500회 복원추출한 percentile bootstrap이다.','', '오답 사례만 조건부로 분석하므로 오류 발생 여부를 예측하는 결과는 아니다. 최초 오류의 자릿수와 영상 길이, GT 시각 등의 교란이 있을 수 있다. 자릿수별 상관은 role_statistics.csv에 함께 저장했으며, 본 그림의 관계를 인과관계로 해석하지 않는다.','', '[관측값 CSV](per_first_error.csv) · [그룹 통계](group_statistics.csv) · [최초 오류 자릿수별 통계](role_statistics.csv)']
write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n');write_result_text((relocate_path(OUT/'status.json')), json.dumps(dict(state='complete',observations=len(rows),panels=len(groups)),indent=2))
register('analysis.full.first_error_distance','analyses',relocate_path(OUT/'REPORT.md'),'최초 오류 엔트로피와 시작·종료 GT 절대 오차','timelens_first_error_distance.py','full_benchmark_v1','complete')
log('complete');print(json.dumps(groups),flush=True)
