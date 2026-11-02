"""No-inference GT-fraction stratification and common-bin standardization."""
import csv
from pathlib import Path
import numpy as np
from adaptive_temporal_sampling import ROOT,read,write,fileinfo
from adaptive_temporal_report import csvwrite
from boundary_oracle_analysis import bootstrap_weights
RUN='gt_density_allocation_comparison_v1';SOURCE='gt_extent_boundary_density_pilot300_v1'
OUT=ROOT/'artifacts/runs'/RUN;REPORT=ROOT/'results'/RUN
DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens'];NAMES=['Charades','ActivityNet','QVHighlights']
BINS=[(0,.05,'0–5% 미만'),(.05,.1,'5–10% 미만'),(.1,.2,'10–20% 미만'),(.2,.4,'20–40% 미만'),(.4,1.000001,'40% 이상')]
def interval(v):
    v=np.asarray(v);v=v[np.isfinite(v)];return [float(x) for x in np.quantile(v,[.025,.975])]
def main():
    source=ROOT/'artifacts/runs'/SOURCE/'metrics/evaluation_records.json';data=read(source);by={(r['sample_key'],r['condition']):r for r in data}
    querypath=OUT/'metrics/dataset_query_statistics.csv';query=list(csv.DictReader(querypath.open()));stats=[];metrics=[];effects=[];standard=[];diffs=[];replicas={};details=[]
    for di,ds in enumerate(DATASETS):
        rows=sorted([r for r in query if r['dataset']==ds],key=lambda r:r['sample_key']);keys=[r['sample_key'] for r in rows];fraction=np.array([float(r['GT_fraction']) for r in rows]);videos=[r['video_id'] for r in rows]
        masks=[(fraction>=lo)&(fraction<hi) for lo,hi,_ in BINS]
        assert all(mask.any() for mask in masks),'Empty bin: common support must be explicitly redefined'
        weights=bootstrap_weights(videos,seed=42+di)
        for bi,mask in enumerate(masks):
            rr=[r for r,m in zip(rows,mask) if m];f=fraction[mask]
            st=dict(dataset=ds,bin=BINS[bi][2],N=len(rr),N_video=len({r['video_id'] for r in rr}),fraction_median=float(np.median(f)),fraction_q25=float(np.quantile(f,.25)),fraction_q75=float(np.quantile(f,.75)),fraction_min=float(f.min()),fraction_max=float(f.max()),T_median=float(np.median([float(r['T']) for r in rr])),L_median=float(np.median([float(r['L']) for r in rr])),space_tokens_median=float(np.median([float(r['spatial_tokens_per_temporal_pair']) for r in rr])))
            stats.append(st)
            for r in rr:details.append(dict(dataset=ds,sample_key=r['sample_key'],bin=BINS[bi][2],GT_fraction=float(r['GT_fraction']),standardized_query_weight=1/(5*len(rr))))
        for metric in ['miou','r03','r05','r07']:
            threshold={'r03':.3,'r05':.5,'r07':.7}.get(metric)
            def value(k,c):
                v=by[k,c]['iou'];return v if threshold is None else float(v>=threshold)
            vals={'uniform':np.array([value(k,'uniform') for k in keys])}
            for ratio in [2,4,8]:
                vals[f'gt_r{ratio}']=np.array([value(k,f'gt_whole_r{ratio}') for k in keys]);vals[f'random_r{ratio}']=np.array([np.mean([value(k,f'random_whole_r{ratio}_s{s}') for s in [42,43]]) for k in keys])
            reps={};point={}
            for c,v in vals.items():
                group_reps=[];group_means=[]
                for bi,mask in enumerate(masks):
                    w=weights[:,mask];den=w.sum(axis=1);rep=np.divide(w@v[mask],den,out=np.full(len(w),np.nan),where=den>0);mu=float(v[mask].mean());reps[c,bi]=rep
                    group_reps.append(rep);group_means.append(mu)
                    metrics.append(dict(dataset=ds,bin=BINS[bi][2],N=int(mask.sum()),metric=metric,condition=c,percent=100*mu))
                standardized=np.mean(group_reps,axis=0);reps[c,'standard']=standardized;point[c]=np.mean(group_means)
                standard.append(dict(dataset=ds,metric=metric,condition=c,raw_percent=float(v.mean()*100),standardized_percent=float(point[c]*100),ci95=[x*100 for x in interval(standardized)],valid_bootstrap=int(np.isfinite(standardized).sum())))
            for ratio in [2,4,8]:
                a=f'gt_r{ratio}'
                for b in ['uniform',f'random_r{ratio}']:
                    for bi,mask in enumerate(masks):
                        d=vals[a][mask]-vals[b][mask];rep=reps[a,bi]-reps[b,bi]
                        effects.append(dict(dataset=ds,bin=BINS[bi][2],N=int(mask.sum()),metric=metric,a=a,b=b,delta_pp=float(d.mean()*100),ci95=[x*100 for x in interval(rep)],valid_bootstrap=int(np.isfinite(rep).sum()),improved=int((d>0).sum()),worsened=int((d<0).sum())))
                    rep=reps[a,'standard']-reps[b,'standard'];effects.append(dict(dataset=ds,bin='equal_bin_weight',N=100,metric=metric,a=a,b=b,delta_pp=float((point[a]-point[b])*100),ci95=[x*100 for x in interval(rep)],valid_bootstrap=int(np.isfinite(rep).sum())))
                    replicas[ds,metric,a,b]=rep
        assert abs(sum(x['standardized_query_weight'] for x in details if x['dataset']==ds)-1)<1e-9
    for ds in DATASETS[1:]:
        for metric in ['miou','r03','r05','r07']:
            for ratio in [2,4,8]:
                a=f'gt_r{ratio}';b='uniform';rep=replicas[ds,metric,a,b]-replicas[DATASETS[0],metric,a,b]
                def get(d):return next(x['delta_pp'] for x in effects if (x['dataset'],x['metric'],x['a'],x['b'],x['bin'])==(d,metric,a,b,'equal_bin_weight'))
                diffs.append(dict(dataset_a=ds,dataset_b=DATASETS[0],metric=metric,condition=a,standardized_gain_difference_pp=get(ds)-get(DATASETS[0]),ci95=[x*100 for x in interval(rep)]))
    for name,value in [('composition',stats),('metrics',metrics),('effects',effects),('standardized',standard),('dataset_differences',diffs),('membership',details)]:
        write(OUT/'metrics'/f'GT_fraction_{name}.json',value);csvwrite(OUT/'metrics'/f'GT_fraction_{name}.csv',value)
    write(OUT/'metadata/GT_fraction_analysis.json',dict(sources=[fileinfo(source),fileinfo(querypath),fileinfo(Path(__file__))],inference_calls=0,bins=BINS,standardization='each of five bins weight0.2 within each dataset; all 100 queries retained',bootstrap='20000 video-cluster draws per dataset, independent seeds42/43/44; replicate ratio means recomputed; missing-bin replicates excluded',note='bootstrap seed44 is not an inference/planner random seed; planner seeds remain42/43'))
    render(stats,metrics,effects,standard,diffs)
    from gt_video_fraction_axes import main as add_video_axis
    add_video_axis()
    print('COMPOSITION',[(x['dataset'],x['bin'],x['N']) for x in stats])
    print('STANDARDIZED',[x for x in effects if x['bin']=='equal_bin_weight' and x['metric']=='miou' and x['a']=='gt_r8'])
    print('DIFFERENCES',[x for x in diffs if x['metric']=='miou' and x['condition']=='gt_r8'])

def render(stats,metrics,effects,standard,diffs):
    def metric(ds,bin,c,m='miou'):return next(x['percent'] for x in metrics if (x['dataset'],x['bin'],x['condition'],x['metric'])==(ds,bin,c,m))
    def effect(ds,bin,m='miou',b='uniform'):return next(x for x in effects if (x['dataset'],x['bin'],x['a'],x['b'],x['metric'])==(ds,bin,'gt_r8',b,m))
    def fmt(e):return f"{e['delta_pp']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]"
    lines=['# GT가 영상에서 차지하는 비율을 맞추면 데이터셋 차이가 남는가?','',
    '**추가 추론 없이 기존300개 질문의 응답만 다시 분석했습니다.** GT는 정답 구간이며 비율은 정답 길이/영상 길이입니다. 정답 구간이나 영상을 잘라 비율을 바꾸지 않았습니다. 공간 해상도는 기존 그대로이므로 이번 분석으로 공간 해상도 효과를 분리할 수는 없습니다.','',
    'mIoU는 예측과 정답의 겹친 길이를 합집합 길이로 나눈 IoU의 질문 평균입니다. 높을수록 좋고 표에는%로 표시합니다. pp는 백분율 차이입니다. IoU≥0.3/0.5/0.7 비율은 첫 예측이 해당 기준 이상인 질문 비율이며 실패는0점으로 포함합니다. Random은 GT 길이만 알고 위치는 무작위인42·43 두 seed의 질문별 평균입니다. threshold는 seed별로 먼저 적용한 뒤 평균합니다.','',
    '## 1. 실제로 비교하는 질문들은 얼마나 비슷한가?','',
    '범위는0–5%,5–10%,10–20%,20–40%,40% 이상으로 고정했습니다. 아래 칸은 **질문 수 / 실제 비율 중앙값 / 영상 길이 중앙값**입니다. 같은 구간이어도 정확히 같은 비율이나 영상 길이는 아닙니다. 특히40% 이상은 넓은 구간입니다.','', '| GT 비율 | Charades | ActivityNet | QVHighlights |','|---|---|---|---|']
    for _,_,bin in BINS:
        cells=[]
        for ds in DATASETS:
            x=next(x for x in stats if x['dataset']==ds and x['bin']==bin);cells.append(f"{x['N']}개 / {100*x['fraction_median']:.1f}% / {x['T_median']:.1f}초")
        lines.append('| '+bin+' | '+' | '.join(cells)+' |')
    lines+=['','실제 비율의25/75백분위수·범위, 정답 길이·공간 토큰 수 중앙값, 고유 영상 수도 composition.csv에 저장했습니다.','',
    '## 2. 각 구간에서 GT 전체8배가 얼마나 좋아졌나?','',
    '각 칸은 **균등 점수 → GT 점수 (차이pp)**입니다. 기존 결과에서 평균이 높았던8배를 상세 분석하며,2·4배도 동일 계산한 원자료를 함께 제공합니다. 사후 분석이므로8배의 최적성을 검증한 것은 아닙니다.','', '| GT 비율 | Charades | ActivityNet | QVHighlights |','|---|---|---|---|']
    for _,_,bin in BINS:lines.append('| '+bin+' | '+' | '.join(f"{metric(ds,bin,'uniform'):.2f} → {metric(ds,bin,'gt_r8'):.2f} ({effect(ds,bin)['delta_pp']:+.2f})" for ds in DATASETS)+' |')
    lines+=['','## 3. 더 엄격한 성공 기준에서도 같은가?','',
    '각 칸은 Uniform 대비 **IoU≥0.3 / ≥0.5 / ≥0.7 비율 변화(pp)**입니다. 절대 성공률은 metrics.csv, 성공↔실패 변화와 구간별 신뢰구간은 effects.json에 있습니다.','', '| GT 비율 | Charades | ActivityNet | QVHighlights |','|---|---|---|---|']
    for _,_,bin in BINS:lines.append('| '+bin+' | '+' | '.join(' / '.join(f"{effect(ds,bin,m)['delta_pp']:+.1f}" for m in ['r03','r05','r07']) for ds in DATASETS)+' |')
    lines+=['','## 4. 구간별 구성 비중까지 같게 만들면?','',
    '**다섯 구간에 각각20%의 동일 가중치를 적용**했습니다. 구간 안에서는 질문에 같은 가중치를 줍니다. 기존100개를 모두 쓰되, 질문이 적은 구간의 한 질문은 더 큰 가중치를 갖습니다. 이것은 실제 데이터셋의 원래 평균이 아니라 같은 구간 구성을 가정한 비교입니다. 작은 구간을 크게 반영할 수 있어 원래 평균과 함께 봐야 합니다.','',
    '95% 구간은 영상 단위 bootstrap20,000회로 계산합니다. 같은 영상의 질문들을 함께 뽑고 각 재표집에서 구간별 평균과 가중 평균을 다시 계산했습니다. 빈 구간이 생긴 재표집은 제외하고 유효 횟수를 저장했습니다. 다중비교 보정 없는 탐색 분석이며 표본 수가 적은 구간의 구간 추정은 불안정합니다.','',
    '| 데이터셋 | 원래 GT−Uniform | 구간 비중 일치 GT−Uniform [95% 구간] | 구간 비중 일치 GT−Random [95% 구간] |','|---|---:|---|---|']
    for ds,name in zip(DATASETS,NAMES):
        get=lambda c:next(x for x in standard if x['dataset']==ds and x['metric']=='miou' and x['condition']==c)
        raw=get('gt_r8')['raw_percent']-get('uniform')['raw_percent'];lines.append(f"| {name} | {raw:+.2f}pp | {fmt(effect(ds,'equal_bin_weight'))} | {fmt(effect(ds,'equal_bin_weight',b='random_r8'))} |")
    lines+=['','**개선 폭 자체의 차이도 직접 비교했습니다.** 아래는 구간 비중을 맞춘 뒤 다른 데이터셋의 개선 폭에서 Charades의 개선 폭을 뺀 값입니다.','', '| 비교 | 개선 폭 차이 pp [95% 구간] |','|---|---|']
    for x in diffs:
        if x['metric']=='miou' and x['condition']=='gt_r8':lines.append(f"| {NAMES[DATASETS.index(x['dataset_a'])]} − Charades | {x['standardized_gain_difference_pp']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}] |")
    lines+=['','## 해석의 범위','',
    '구간을 나누고 구간 비중을 맞추는 것은 GT 비율을 정확히 일치시키는 것과 다릅니다. 구간 내부 비율 분포, 영상·정답 절대 길이, 공간 해상도, 질문·사건 특성이 여전히 다릅니다. 따라서 남는 차이를 데이터셋 고유의 원인이나 공간 해상도 효과로 확정할 수 없습니다. 다음 단계의 공통 공간 해상도 실험은 아직 실행하지 않았습니다.','',
    '표의 평균 차이와 신뢰구간을 함께 보고, 한쪽만 유의하다는 이유로 두 데이터셋의 효과가 다르다고 결론내리지 않습니다. 위 개선 폭 차이가 이를 직접 비교하는 통계입니다.','',
    f'[질문 수·길이·비율 분포](../../artifacts/runs/{RUN}/metrics/GT_fraction_composition.csv) · [모든2/4/8배 점수](../../artifacts/runs/{RUN}/metrics/GT_fraction_metrics.csv) · [구간별 차이·신뢰구간](../../artifacts/runs/{RUN}/metrics/GT_fraction_effects.json) · [질문별 구간·가중치](../../artifacts/runs/{RUN}/metrics/GT_fraction_membership.csv) · [기존 통합 보고서](REPORT.md)']
    lines[2:2]=['**결론: GT 비율이 작은 구간에서 개선 신호는 있지만, GT 비율 구간의 구성 차이만으로 Charades와 다른 두 데이터셋의 격차를 설명하기는 어렵습니다.** Charades의5% 미만18개는 평균 +5.03pp였지만 IoU≥0.7 성공률은 그대로였습니다. 비율에 따라 개선이 단조롭게 증가·감소하지 않으며 작은 집단의 결과는 탐색적으로 봐야 합니다.','']
    (REPORT/'GT_FRACTION.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
