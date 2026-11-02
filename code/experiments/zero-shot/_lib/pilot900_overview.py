"""Automatic 300/600/900 and duration×GT-fraction reporting for both designs."""
import numpy as np
from adaptive_temporal_sampling import ROOT,read,write,fileinfo
from adaptive_temporal_report import csvwrite
from artifact_store import register
RUN='gt_density_allocation_comparison_pilot900_v1';A='gt_extent_boundary_density_pilot900_v1';B='gt_actual_density_pilot900_v1'
TB=[(0,30,'<30'),(30,60,'30–60'),(60,120,'60–120'),(120,180,'120–180'),(180,float('inf'),'≥180')];QB=[(0,.05,'<5%'),(.05,.1,'5–10%'),(.1,.2,'10–20%'),(.2,.4,'20–40%'),(.4,1.000001,'≥40%')]
DS=['charades-timelens','activitynet-timelens','qvhighlights-timelens'];LABELS=['Charades','ActivityNet','QVHighlights']
def main():
    out=ROOT/'artifacts/runs'/RUN;report=ROOT/'results'/RUN;report.mkdir(parents=True,exist_ok=True);metrics=[];cells=[];sources=[]
    cohorts={n:read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json') for n in [300,600,900]}
    assert cohorts[900][:600]==cohorts[600] and cohorts[600][:300]==cohorts[300]
    for design,run in [('A',A),('B',B)]:
        p=ROOT/'artifacts/runs'/run;assert read(p/'metadata/status.json')['state']=='complete';data=read(p/'metrics/evaluation_records.json');assert len(data)==17100;sources.append(fileinfo(p/'metrics/evaluation_records.json'));by={(x['sample_key'],x['condition']):x for x in data};conditions=sorted({x['condition'] for x in data});durations={s['sample_key']:read(p/'plans/uniform'/(s['sample_key']+'.json'))['T'] for s in cohorts[900]}
        conditions+= [f'random_{mode}_r{ratio}_mean' for mode in ['whole','boundary'] for ratio in [2,4,8]]
        for size,cohort in cohorts.items():
            for ds in DS+['all']:
                ss=[s for s in cohort if ds=='all' or s['dataset']==ds];keys=[s['sample_key'] for s in ss]
                for c in conditions:
                    members=[c.replace('_mean','_s42'),c.replace('_mean','_s43')] if c.endswith('_mean') else [c]
                    for metric,threshold in [('miou',None),('r03',.3),('r05',.5),('r07',.7)]:
                        def v(k,d):
                            x=by[k,d]['iou'];return x if threshold is None else float(x>=threshold)
                        values=np.array([np.mean([v(k,d) for d in members]) for k in keys]);uniform=np.array([v(k,'uniform') for k in keys])
                        metrics.append(dict(size=size,dataset=ds,design=design,condition=c,metric=metric,N=len(keys),percent=float(values.mean()*100),delta_uniform_pp=float((values-uniform).mean()*100)))
                for axis in ['duration','GT_fraction','joint']:
                    for ti,(t0,t1,tl) in enumerate(TB if axis!='GT_fraction' else [(0,float('inf'),'all')]):
                        for qi,(q0,q1,ql) in enumerate(QB if axis!='duration' else [(0,1.000001,'all')]):
                            subset=[s for s in ss if t0<=durations[s['sample_key']]<t1 and q0<=by[s['sample_key'],'uniform']['length_fraction']<q1]
                            for c in ['gt_whole_r2','gt_whole_r4','gt_whole_r8','gt_boundary_r2','gt_boundary_r4','gt_boundary_r8']:
                                for metric,threshold in [('miou',None),('r03',.3),('r05',.5),('r07',.7)]:
                                    def val(s,d):
                                        x=by[s['sample_key'],d]['iou'];return x if threshold is None else float(x>=threshold)
                                    diff=[val(s,c)-val(s,'uniform') for s in subset]
                                    cells.append(dict(size=size,dataset=ds,design=design,axis=axis,duration_bin=tl,GT_fraction_bin=ql,condition=c,metric=metric,N=len(subset),delta_pp=float(np.mean(diff)*100) if diff else None))
    csvwrite(out/'metrics/nested_metrics.csv',metrics);csvwrite(out/'metrics/nested_axes.csv',cells);write(out/'metadata/sources.json',dict(files=sources,cohort_files=[fileinfo(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json') for n in cohorts]))
    lines=['# 900개 확장: 사건 전체·경계 집중 비교','',
    '기존300개를 그대로 포함하는600개·900개 결과입니다. 각각 데이터셋별100·200·300개 질문이며, 추가 표본은 성능을 보지 않고 영상 길이×GT 비율의 부족한 칸을 보완했습니다. 독립 test나 원래 벤치마크 분포의 평균이 아닙니다. 질문 수 증가와 구성 변화가 함께 있으므로 크기별 점수 변화만으로 성능 향상을 주장하지 않습니다.','',
    'mIoU는 예측과 정답의 교집합 길이/합집합 길이를 질문별로 평균한 값입니다. 높을수록 좋습니다. IoU≥0.3/0.5/0.7은 해당 기준을 넘긴 질문 비율이며 실패도 분모에 포함합니다. pp는 백분율 차이입니다. Random은42·43 두 seed이며 threshold를 seed별 적용 후 평균합니다.','',
    'A는 GT 전체와 경계가 집중 프레임 수를 공유하고, B는 각 집중 영역의 실제 길이로 안팎 밀도 비율을 맞춥니다. 전체 조건과 Uniform은 동일 입력을 공유하므로 독립 반복이 아닙니다. 공간 해상도는 기존 질문별 입력 설정을 유지했습니다.','',
    '| 전체 질문 수 | 데이터셋 | 균등 mIoU | GT전체8배 mIoU | 변화pp | ≥0.3 변화 | ≥0.5 변화 | ≥0.7 변화 |','|---:|---|---:|---:|---:|---:|---:|---:|']
    def get(size,ds,c,metric='miou',design='A'):return next(x for x in metrics if (x['size'],x['dataset'],x['condition'],x['metric'],x['design'])==(size,ds,c,metric,design))
    for size in [300,600,900]:
        for ds,label in zip(DS+['all'],LABELS+['전체']):
            g=get(size,ds,'gt_whole_r8');u=get(size,ds,'uniform');lines.append(f"| {size} | {label} | {u['percent']:.2f} | {g['percent']:.2f} | {g['delta_uniform_pp']:+.2f} | "+' | '.join(f"{get(size,ds,'gt_whole_r8',m)['delta_uniform_pp']:+.2f}" for m in ['r03','r05','r07'])+' |')
    lines+=['','## 900개 두 축 교차표','', '각 칸은 GT전체8배−Uniform의 mIoU 변화pp (질문 수)입니다. 10개 미만은†, 빈 칸은—입니다. 행은GT 비율, 열은영상 길이입니다. 데이터가 없는 시간 구간을 채워서 비교하지 않았습니다.']
    for ds,label in zip(DS,LABELS):
        times=[tl for _,_,tl in TB if any(x['N'] for x in cells if x['size']==900 and x['dataset']==ds and x['axis']=='joint' and x['duration_bin']==tl)]
        lines+=['','### '+label,'','| GT 비율 | '+' | '.join(t+'초' for t in times)+' |','|---|'+''.join('---:|' for _ in times)]
        for _,_,q in QB:
            values=[]
            for t in times:
                x=next(x for x in cells if (x['size'],x['dataset'],x['design'],x['axis'],x['duration_bin'],x['GT_fraction_bin'],x['condition'],x['metric'])==(900,ds,'A','joint',t,q,'gt_whole_r8','miou'))
                values.append('—' if not x['N'] else f"{x['delta_pp']:+.2f} ({x['N']})"+(' †' if x['N']<10 else ''))
            lines.append('| '+q+' | '+' | '.join(values)+' |')
    lines+=['','## 전체 조건과 불확실성','','2·4·8배, 두 경계 배정 방식, Random 결과와 영상 단위bootstrap 신뢰구간은 아래 실행별 보고서에 있습니다.900개·600개·기존300개 모든 조건의 threshold와 교차 칸 원자료도 제공합니다. 작은 칸의 수치만으로 원인을 단정하지 않습니다.',f'[A: 집중 프레임 수 공유](../{A}/REPORT.md) · [B: 실제 집중 길이로 밀도 일치](../{B}/REPORT.md)',f'[300/600/900 모든 조건 점수](../../artifacts/runs/{RUN}/metrics/nested_metrics.csv) · [길이·GT 비율·두 축 비교](../../artifacts/runs/{RUN}/metrics/nested_axes.csv)']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from pilot900_report_tables import render
    render()
    register(RUN+'.analyses','analyses',report,'Nested300/600/900 integrated density experiment','_lib/pilot900_overview.py','pilot_900','complete',[A+'.analyses',B+'.analyses'])
    write(out/'metadata/status.json',dict(state='complete',source_runs=[A,B]))
if __name__=='__main__':main()
