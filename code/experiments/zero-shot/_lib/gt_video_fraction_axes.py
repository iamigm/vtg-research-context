"""Existing-response comparison by duration, GT fraction, and their cross."""
from result_layout import relocate_path
import csv
import numpy as np
from adaptive_temporal_sampling import ROOT,read,write,fileinfo
from adaptive_temporal_report import csvwrite
from boundary_oracle_analysis import bootstrap_weights
RUN='gt_density_allocation_comparison_v1';SOURCE='gt_extent_boundary_density_pilot300_v1'
OUT=relocate_path(ROOT/'artifacts/runs'/RUN);REPORT=relocate_path(ROOT/'results'/RUN)
DS=['charades-timelens','activitynet-timelens','qvhighlights-timelens'];NAMES=['Charades','ActivityNet','QVHighlights']
TB=[(0,30,'30초 미만'),(30,60,'30–60초 미만'),(60,120,'60–120초 미만'),(120,180,'120–180초 미만'),(180,np.inf,'180초 이상')]
QB=[(0,.05,'5% 미만'),(.05,.1,'5–10% 미만'),(.1,.2,'10–20% 미만'),(.2,.4,'20–40% 미만'),(.4,1.000001,'40% 이상')]
def ci(rep,enabled):
    finite=rep[np.isfinite(rep)]
    return [float(v*100) for v in np.quantile(finite,[.025,.975])] if enabled and len(finite) else None

def main():
    source=relocate_path(ROOT/'artifacts/runs'/SOURCE/'metrics/evaluation_records.json');raw=read(source);lookup={(r['sample_key'],r['condition']):r for r in raw};qpath=OUT/'metrics/dataset_query_statistics.csv';query=list(csv.DictReader(qpath.open()));composition=[];metrics=[];effects=[];membership=[];reps={};comparisons=[]
    definitions=[('GT_fraction',None,j) for j in range(5)]+[('duration',i,None) for i in range(5)]+[('joint',i,j) for i in range(5) for j in range(5)]
    for di,ds in enumerate(DS):
        rr=sorted([r for r in query if r['dataset']==ds],key=lambda r:r['sample_key']);keys=[r['sample_key'] for r in rr];T=np.array([float(r['T']) for r in rr]);Q=np.array([float(r['GT_fraction']) for r in rr]);videos=[r['video_id'] for r in rr];w=bootstrap_weights(videos,42+di)
        matrices={}
        for metric in ['miou','r03','r05','r07']:
            th={'r03':.3,'r05':.5,'r07':.7}.get(metric)
            def val(k,c):
                v=lookup[k,c]['iou'];return v if th is None else float(v>=th)
            vv={'uniform':np.array([val(k,'uniform') for k in keys])}
            for r in [2,4,8]:
                vv[f'gt_r{r}']=np.array([val(k,f'gt_whole_r{r}') for k in keys]);vv[f'random_r{r}']=np.array([np.mean([val(k,f'random_whole_r{r}_s{s}') for s in [42,43]]) for k in keys])
            matrices[metric]=vv
        for axis,i,j in definitions:
            mask=np.ones(len(rr),bool)
            if i is not None:mask&=(T>=TB[i][0])&(T<TB[i][1])
            if j is not None:mask&=(Q>=QB[j][0])&(Q<QB[j][1])
            n=int(mask.sum());nv=len(set(np.array(videos)[mask]));group=dict(dataset=ds,axis=axis,duration_bin=TB[i][2] if i is not None else 'all',GT_fraction_bin=QB[j][2] if j is not None else 'all',N=n,N_video=nv,sparse=n<10)
            row=dict(group)
            for f,a in [('T',T),('GT_fraction',Q),('GT_length',np.array([float(r['L']) for r in rr])),('space_tokens',np.array([float(r['spatial_tokens_per_temporal_pair']) for r in rr]))]:
                for name,fun in [('median',np.median),('min',np.min),('max',np.max)]:row[f+'_'+name]=float(fun(a[mask])) if n else None
            composition.append(row)
            for k,m in zip(keys,mask):
                if m:membership.append(dict(group,sample_key=k))
            if not n:continue
            ww=w[:,mask];den=ww.sum(axis=1)
            for metric,vv in matrices.items():
                for c,v in vv.items():metrics.append(dict(group,metric=metric,condition=c,percent=float(v[mask].mean()*100)))
                for r in [2,4,8]:
                    a=f'gt_r{r}'
                    for b in ['uniform',f'random_r{r}']:
                        d=(vv[a]-vv[b])[mask];rep=np.divide(ww@d,den,out=np.full(len(w),np.nan),where=den>0)
                        e=dict(group,metric=metric,a=a,b=b,delta_pp=float(d.mean()*100),ci95=ci(rep,nv>=5),valid_bootstrap=int(np.isfinite(rep).sum()),improved=int((d>0).sum()),worsened=int((d<0).sum()),unchanged=int((d==0).sum()))
                        effects.append(e)
                        if metric=='miou' and r==8 and b=='uniform':reps[ds,axis,i,j]=(rep,e)
    for axis,i,j in definitions:
        for ai,bi in [(0,1),(0,2),(1,2)]:
            if (DS[ai],axis,i,j) not in reps or (DS[bi],axis,i,j) not in reps:continue
            ar,ae=reps[DS[ai],axis,i,j];br,be=reps[DS[bi],axis,i,j]
            comparisons.append(dict(axis=axis,duration_bin=TB[i][2] if i is not None else 'all',GT_fraction_bin=QB[j][2] if j is not None else 'all',dataset_a=DS[ai],dataset_b=DS[bi],N_a=ae['N'],N_b=be['N'],delta_gain_a_minus_b_pp=ae['delta_pp']-be['delta_pp'],ci95=ci(ar-br,min(ae['N_video'],be['N_video'])>=5),sparse=min(ae['N'],be['N'])<10))
    for name,rows in [('composition',composition),('metrics',metrics),('effects',effects),('pairwise',comparisons),('membership',membership)]:
        write(OUT/'metrics'/f'two_axes_{name}.json',rows);csvwrite(OUT/'metrics'/f'two_axes_{name}.csv',rows)
    write(OUT/'metadata/two_axes_analysis.json',dict(inference_calls=0,sources=[fileinfo(source),fileinfo(qpath),fileinfo(__file__)],duration_bins=[[a,b if np.isfinite(b) else None,c] for a,b,c in TB],GT_fraction_bins=QB,bootstrap='20000 video-cluster replicates within dataset, seeds42/43/44; planner seed44 not used',sparse_rule='N<10 marked; no CI when fewer than5 unique videos',common_support_all_three_duration=False,common_support_all_three_joint=False))
    for ds in DS:
        for axis in ['duration','GT_fraction','joint']:assert sum(x['N'] for x in composition if x['dataset']==ds and x['axis']==axis)==100
    render(composition,metrics,effects,comparisons)
    print('Duration effects:',[(x['dataset'],x['duration_bin'],x['N'],round(x['delta_pp'],2)) for x in effects if x['axis']=='duration' and x['metric']=='miou' and x['a']=='gt_r8' and x['b']=='uniform'])

def render(composition,metrics,effects,comparisons):
    def get(ds,axis,t='all',q='all',metric='miou'):
        return next((x for x in effects if (x['dataset'],x['axis'],x['duration_bin'],x['GT_fraction_bin'],x['metric'],x['a'],x['b'])==(ds,axis,t,q,metric,'gt_r8','uniform')),None)
    def cell(x):return '—' if x is None else f"{x['delta_pp']:+.2f} / {x['N']}개"+(' †' if x['sparse'] else '')
    lines=['## 7. 영상 길이만으로 나누면?','',
    '**영상 길이와 GT 비율을 각각 비교하고, 두 축을 동시에 비교했습니다. 추가 추론은 없습니다.** 아래는 GT 전체8배−Uniform의 mIoU 차이(pp)입니다. 각 칸은 **차이 / 질문 수**이며 †는10개 미만이라 불안정한 집단입니다. 빈 칸은 효과가0이라는 뜻이 아니라 해당 표본이 없다는 뜻입니다.2·4배 및 Random 대조도 원자료에 있습니다.','',
    '영상 길이 구간은30초 미만,30–60초,60–120초,120–180초,180초 이상으로 고정했습니다. 하한 포함·상한 제외입니다. GT 비율 구간은 앞선 분석과 동일합니다. 구간 경계를 바꿔 좋은 결과를 찾지 않았습니다.','',
    '| 영상 길이 | Charades | ActivityNet | QVHighlights |','|---|---|---|---|']
    for _,_,t in TB:lines.append('| '+t+' | '+' | '.join(cell(get(ds,'duration',t)) for ds in DS)+' |')
    lines+=['','**세 데이터셋이 함께 존재하는 영상 길이 구간은 없습니다.** Charades는 모두60초 미만, QVHighlights는 모두120–180초 구간입니다. 따라서 이번300개만으로 영상 길이 분포를 맞춘 세 데이터셋 평균을 계산할 수 없습니다. 결측 칸을 추정해서 채우거나0점으로 넣지 않았습니다.','',
    'Charades↔ActivityNet은30–60초, ActivityNet↔QVHighlights는120–180초에서만 구간 내 비교가 가능합니다. 다만 같은 구간이라도 실제 길이 분포가 같지는 않으며, 아래 실제 중앙값을 함께 봐야 합니다.','',
    '| 비교 구간 | 데이터셋 | 질문 수 | 영상 길이 중앙값 | GT 비율 중앙값 |','|---|---|---:|---:|---:|']
    for t,datasets in [(TB[1][2],DS[:2]),(TB[3][2],DS[1:])]:
        for ds in datasets:
            x=next(x for x in composition if x['dataset']==ds and x['axis']=='duration' and x['duration_bin']==t)
            lines.append(f"| {t} | {NAMES[DS.index(ds)]} | {x['N']} | {x['T_median']:.1f}초 | {100*x['GT_fraction_median']:.1f}% |")
    lines+=['','### 영상 길이별 threshold 변화','',
    '각 칸은 Uniform 대비 IoU≥0.3 / ≥0.5 / ≥0.7 성공 비율 변화(pp)입니다. 기준이 높을수록 정답 구간과 더 정확히 맞아야 합니다.','',
    '| 데이터셋 | 영상 길이 | 질문 수 | ≥0.3 / ≥0.5 / ≥0.7 변화 |','|---|---|---:|---|']
    for ds,name in zip(DS,NAMES):
        for _,_,t in TB:
            x=get(ds,'duration',t)
            if x:lines.append(f"| {name} | {t} | {x['N']} | "+' / '.join(f"{get(ds,'duration',t,metric=m)['delta_pp']:+.1f}" for m in ['r03','r05','r07'])+' |')
    lines+=['','## 8. GT 비율 × 영상 길이 교차표','',
    '교차표를 데이터셋별로 나눴습니다. **행은 GT 비율, 열은 영상 길이**입니다. 각 칸은 **mIoU 변화pp (질문 수)**이며 GT 전체8배에서 Uniform을 뺀 값입니다. 양수는 개선, 음수는 악화입니다. 예를 들어 +8.16 (7)은7개 질문에서 평균8.16pp 개선됐다는 뜻입니다.','',
    '**†는10개 미만의 작은 표본**, —는 해당 질문이 없음을 뜻합니다. 빈 칸을0점으로 처리하지 않았습니다. 구간은 하한 포함·상한 제외이고, 영상 길이 표시는 초 단위입니다.']
    for ds,name in zip(DS,NAMES):
        times=[(t, short) for (_,_,t),short in zip(TB,['<30','30–60','60–120','120–180','≥180']) if get(ds,'duration',t)]
        lines+=['','### '+name,'','| GT 비율 | '+' | '.join(short+'초' for t,short in times)+' |','|---|'+''.join('---:|' for _ in times)]
        for _,_,q in QB:
            cells=[]
            for t,_ in times:
                x=get(ds,'joint',t,q)
                cells.append('—' if x is None else f"{x['delta_pp']:+.2f} ({x['N']})"+(' †' if x['sparse'] else ''))
            lines.append('| '+q+' | '+' | '.join(cells)+' |')
        lines+=['']
    lines+=['데이터셋끼리 비교할 때는 같은 GT 비율 행과 같은 영상 길이 열을 맞춰 읽습니다. Charades와 QVHighlights는 겹치는 영상 길이 열이 없으므로 직접 비교할 수 없습니다.']
    lines+=['','**두 축을 함께 맞춘 세 데이터셋 비교도 불가능합니다.** 특히 ActivityNet의30–60초11개를 GT 비율까지 나누면2·2·2·0·5개입니다. Charades와 같은 칸의 숫자를 비교할 수는 있지만, 몇 질문의 변화로 큰 pp가 생기므로 원인 결론을 내리기 어렵습니다.','',
    '## 9. 같은 구간에서 개선 폭 자체의 차이','',
    '아래는 두 데이터셋 모두 표본이 있는 영상 길이 구간의 직접 비교입니다. 각 데이터셋에서GT−Uniform을 계산한 뒤 두 값을 뺐습니다. 신뢰구간은 영상 단위bootstrap20,000회이며 서로 다른 데이터셋을 독립적으로 재표집했습니다. 공간 해상도와 GT 비율이 통제된 비교는 아닙니다.','',
    '| 비교 | 영상 길이 | 개선 폭 차이 pp [95% 구간] |','|---|---|---|']
    for x in comparisons:
        if x['axis']=='duration':
            c=x['ci95'];ci_text=f"[{c[0]:+.2f}, {c[1]:+.2f}]" if c else '표본 부족으로 미표시'
            lines.append(f"| {NAMES[DS.index(x['dataset_a'])]} − {NAMES[DS.index(x['dataset_b'])]} | {x['duration_bin']} | {x['delta_gain_a_minus_b_pp']:+.2f} {ci_text} |")
    lines+=['','교차 칸별 직접 비교와 threshold의 모든 수치는 원자료에 있습니다. 고유 영상5개 미만의 칸은 신뢰구간을 표시하지 않았고, 그 이상이어도 작은 표본의 신뢰구간은 신뢰하기 어렵습니다. 빈 구간이 발생한 재표집은 제외하며 유효 횟수를 기록했습니다. 여러 구간을 살펴보는 사후 탐색 분석입니다.','',
    '**결론:** GT 비율 구성 차이만으로 데이터셋 격차가 사라지지는 않았습니다. 영상 길이까지 통제했을 때 격차가 남는지에 대해서는 세 데이터셋 공통 표본 범위가 없어 답할 수 없습니다. 후속 실험에서 공간 해상도를 맞추더라도 이 영상 길이 불균형은 남으므로, 길이와 GT 비율이 함께 겹치는 새로운 표본을 확보해야 두 요인을 동시에 통제한 세 데이터셋 비교가 가능합니다.','',
    f'[두 축의 표본 수·실제 길이·비율 분포](../../artifacts/runs/{RUN}/metrics/two_axes_composition.csv) · [모든2/4/8배·threshold 점수](../../artifacts/runs/{RUN}/metrics/two_axes_metrics.csv) · [조건별 변화와 신뢰구간](../../artifacts/runs/{RUN}/metrics/two_axes_effects.json) · [데이터셋 간 직접 비교](../../artifacts/runs/{RUN}/metrics/two_axes_pairwise.json) · [질문별 구간 소속](../../artifacts/runs/{RUN}/metrics/two_axes_membership.csv)']
    p=REPORT/'GT_FRACTION.md';marker='<!-- video_fraction_axes -->';s=p.read_text().split(marker)[0].rstrip();p.write_text(s+'\n\n'+marker+'\n'+'\n'.join(lines)+'\n')
if __name__=='__main__':main()
