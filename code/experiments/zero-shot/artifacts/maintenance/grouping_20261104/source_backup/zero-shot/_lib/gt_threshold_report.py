"""Threshold success metrics, recomputed from per-response raw IoU."""
import numpy as np
from adaptive_temporal_sampling import ROOT,read,write
from adaptive_temporal_report import csvwrite
RUN='gt_density_allocation_comparison_v1'
RUNS=['gt_extent_boundary_density_pilot300_v1','gt_actual_density_pilot300_v1']
LABELS={'charades-timelens':'Charades','activitynet-timelens':'ActivityNet','qvhighlights-timelens':'QVHighlights','all':'전체300개'}

def build():
    rows=[];paired=[];strata=[]
    for run in RUNS:
        data=read(ROOT/'artifacts/runs'/run/'metrics/evaluation_records.json');by={(x['sample_key'],x['condition']):x for x in data};conditions=sorted({x['condition'] for x in data});members={c:[c] for c in conditions}
        members.update({f'random_{mode}_r{r}_mean':[f'random_{mode}_r{r}_s{s}' for s in [42,43]] for r in [2,4,8] for mode in ['whole','boundary']})
        for dataset in LABELS:
            base=[x for x in data if x['condition']=='uniform' and (dataset=='all' or x['dataset']==dataset)];keys=[x['sample_key'] for x in base]
            for threshold in [.3,.5,.7]:
                vals={c:np.array([np.mean([by[k,part]['iou']>=threshold for part in parts]) for k in keys]) for c,parts in members.items()}
                for c,v in vals.items():
                    d=v-vals['uniform'];rows.append(dict(run=run,dataset=dataset,condition=c,N_query=len(keys),threshold=threshold,success_percent=100*float(v.mean()),delta_vs_uniform_pp=100*float(d.mean()),query_success_weight=float(v.sum()),raw_response_count=len(keys)*len(members[c])))
                comparisons=[(f'gt_{mode}_r{r}',b) for r in [2,4,8] for mode in ['whole','boundary'] for b in ['uniform',f'random_{mode}_r{r}_mean']]+[(f'gt_boundary_r{r}',f'gt_whole_r{r}') for r in [2,4,8]]
                for a,b in comparisons:
                    d=vals[a]-vals[b];paired.append(dict(run=run,dataset=dataset,threshold=threshold,a=a,b=b,N=len(keys),delta_pp=100*float(d.mean()),improved_queries=int((d>0).sum()),worsened_queries=int((d<0).sum()),unchanged_queries=int((d==0).sum())))
                for lo,hi in [(0,.1),(.1,.4),(.4,.8),(.8,1.00001)]:
                    mask=np.array([lo<=x['length_fraction']<hi for x in base])
                    if mask.any():
                        for c in ['uniform','gt_whole_r2','gt_whole_r4','gt_whole_r8']:
                            strata.append(dict(run=run,dataset=dataset,threshold=threshold,GT_fraction_lower=lo,GT_fraction_upper=hi,N=int(mask.sum()),condition=c,success_percent=100*float(vals[c][mask].mean()),delta_vs_uniform_pp=100*float((vals[c]-vals['uniform'])[mask].mean())))
    out=ROOT/'artifacts/runs'/RUN/'metrics'
    csvwrite(out/'threshold_metrics.csv',rows);csvwrite(out/'threshold_paired_changes.csv',paired);csvwrite(out/'threshold_GT_fraction_groups.csv',strata)
    def cell(run,ds,c,t):
        x=next(x for x in rows if (x['run'],x['dataset'],x['condition'],x['threshold'])==(run,ds,c,t))
        return f"{x['success_percent']:.1f}%"+(f" ({x['delta_vs_uniform_pp']:+.1f}pp)" if c!='uniform' else '')
    def section(run,include_boundary=False):
        lines=['## IoU 기준을 넘긴 질문은 얼마나 늘었나?','',
        '**R@1, IoU≥τ**는 첫 예측 구간의 IoU가 기준 τ 이상인 질문 비율입니다. 계산은 100×성공 질문 수/전체 질문 수이며 높을수록 좋습니다. 0.3은 느슨한 기준, 0.5는 중간, 0.7은 더 엄격한 구간 일치를 요구합니다. 파싱 실패·무효 예측은 실패로 포함합니다.','',
        '각 칸은 **성공 비율 (Uniform 대비 차이)**입니다. 같은 평균 겹침률 상승이라도 기준을 넘긴 질문이 늘었는지는 별도로 봐야 합니다. Random 평균은 seed별 성공 여부를 먼저 계산한 뒤 질문별 평균합니다. 두 IoU를 평균하고 threshold를 적용하지 않습니다.','',
        '| 데이터셋 | 배치 | IoU≥0.3 | IoU≥0.5 | IoU≥0.7 |','|---|---|---:|---:|---:|']
        conditions=[('uniform','균등')]+[(f'gt_whole_r{r}',f'GT 전체 {r}배') for r in [2,4,8]]
        if include_boundary:conditions += [(f'gt_boundary_r{r}',f'GT 경계 {r}배') for r in [2,4,8]]
        for ds,label in LABELS.items():
            for c,name in conditions:lines.append('| '+label+' | '+name+' | '+' | '.join(cell(run,ds,c,t) for t in [.3,.5,.7])+' |')
        lines+=['','이 표는 관측된 비율과 차이입니다. mIoU의 신뢰구간을 이 지표에 적용하지 않으며, 표의 작은 차이만으로 유의성을 주장하지 않습니다. 모든 Random·경계 조건과 성공↔실패 변화, GT 비율 집단별 수치는 CSV에 있습니다.','',f'[모든 조건의 기준별 성공률](../../artifacts/runs/{RUN}/metrics/threshold_metrics.csv) · [질문별 성공 여부의 변화 집계](../../artifacts/runs/{RUN}/metrics/threshold_paired_changes.csv) · [GT 길이 비율별 기준 통과율](../../artifacts/runs/{RUN}/metrics/threshold_GT_fraction_groups.csv)']
        return lines
    for run in RUNS:
        p=ROOT/'results'/run/'REPORT.md';s=p.read_text();marker='<!-- threshold_metrics_begin -->';s=s.split(marker)[0].rstrip();p.write_text(s+'\n\n'+marker+'\n'+'\n'.join(section(run,True))+'\n')
    lines=section(RUNS[0]);lines+=['','**경계 배정 공식도 같은 기준으로 비교:** 아래는 전체300개입니다. A는 집중 프레임 수 공유, B는 실제 집중 길이로 밀도 일치입니다.','', '| 공식 | 경계 조건 | IoU≥0.3 | IoU≥0.5 | IoU≥0.7 |','|---|---|---:|---:|---:|']
    for design,run in zip(['A','B'],RUNS):
        for r in [2,4,8]:lines.append('| '+design+f' | GT 경계 {r}배 | '+' | '.join(cell(run,'all',f'gt_boundary_r{r}',t) for t in [.3,.5,.7])+' |')
    return lines
