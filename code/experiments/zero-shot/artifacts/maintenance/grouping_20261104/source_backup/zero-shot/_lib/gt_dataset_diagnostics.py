"""Pilot-only descriptive statistics and plain-language report regeneration."""
from pathlib import Path
import numpy as np
from adaptive_temporal_sampling import ROOT,read,write,fileinfo,sha
from adaptive_temporal_report import csvwrite
A='gt_extent_boundary_density_pilot300_v1';B='gt_actual_density_pilot300_v1';RUN='gt_density_allocation_comparison_v1'
LABELS={'charades-timelens':'Charades','activitynet-timelens':'ActivityNet','qvhighlights-timelens':'QVHighlights'}
OUT=ROOT/'artifacts/runs'/RUN;REPORT=ROOT/'results'/RUN

def describe(values):
    a=np.asarray(values,float)
    return dict(N=len(a),mean=float(a.mean()),median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),min=float(a.min()),max=float(a.max()))

def main():
    pa=ROOT/'artifacts/runs'/A;pb=ROOT/'artifacts/runs'/B
    cohort=read(pa/'metadata/cohort.json');ev=read(pa/'metrics/evaluation_records.json');by={(x['sample_key'],x['condition']):x for x in ev};rows=[];alloc=[]
    for sample in cohort:
        key=sample['sample_key'];s,e=sample['span'][0];L=e-s;u=read(pa/'plans/uniform'/(key+'.json'));T=u['T'];F=u['F'];response=read(pa/'responses/uniform'/(key+'.json'));base=by[key,'uniform'];pts=u['pts'];inside=sum(s<=t<=e for t in pts)
        row=dict(sample_key=key,dataset=sample['dataset'],video_id=by[key,'uniform']['video_id'],annotation_T=sample['duration'],T=T,L=L,GT_fraction=L/T,F=F,B=response['B'],spatial_tokens_per_temporal_pair=response['grid'][1]*response['grid'][2]/4,uniform_fps=F/T,uniform_inside_frames=inside,uniform_inside_fps=inside/L,query_words=len(sample['query'].split()),GT_touches_edge=int(s<=.5 or T-e<=.5),baseline_iou=base['iou'],baseline_start_normalized=base['normalized_start_error'],baseline_end_normalized=base['normalized_end_error'])
        for r in [2,4,8]:
            p=read(pa/'plans'/f'gt_whole_r{r}'/(key+'.json'));g=by[key,f'gt_whole_r{r}'];rand=np.mean([by[key,f'random_whole_r{r}_s{seed}']['iou'] for seed in [42,43]])
            row.update({f'gain_r{r}':100*(g['iou']-base['iou']),f'gt_minus_random_r{r}':100*(g['iou']-rand),f'GT_inside_frames_r{r}':p['inside_actual'],f'GT_inside_fps_r{r}':p['inside_actual']/L,f'extra_inside_r{r}':p['inside_actual']-inside})
            for design,path in [('same_count',pa),('same_density',pb)]:
                q=read(path/'plans'/f'gt_boundary_r{r}'/(key+'.json'));alloc.append(dict(sample_key=key,dataset=sample['dataset'],design=design,ratio=r,focus_duration=q['focus_duration'],focus_fraction=q['focus_duration']/T,requested=q['inside_requested'],actual=q['inside_actual'],transfer=q['capacity_transfer'],minimum_adjustment=q['minimum_adjustment'],actual_density_ratio=q['density_ratio_actual']))
        rows.append(row)
    fields=['T','L','GT_fraction','F','B','spatial_tokens_per_temporal_pair','uniform_fps','uniform_inside_frames','uniform_inside_fps','query_words','baseline_iou','GT_inside_frames_r8','GT_inside_fps_r8','extra_inside_r8']
    stats={};groups=[];associations=[]
    for ds in LABELS:
        rr=[r for r in rows if r['dataset']==ds];stats[ds]={f:describe([r[f] for r in rr]) for f in fields};stats[ds].update(n_query=len(rr),n_video=len({r['video_id'] for r in rr}),GT80_count=sum(r['GT_fraction']>=.8 for r in rr),edge_count=sum(r['GT_touches_edge'] for r in rr))
        for field,bins in [('T',[(0,60),(60,120),(120,float('inf'))]),('GT_fraction',[(0,.1),(.1,.4),(.4,.8),(.8,1.00001)]),('baseline_iou',[(0,.3),(.3,.7),(.7,1.00001)])]:
            for lo,hi in bins:
                sub=[r for r in rr if lo<=r[field]<hi]
                if sub:groups.append(dict(dataset=ds,variable=field,lower=lo,upper=hi if np.isfinite(hi) else None,N=len(sub),uniform_miou=100*np.mean([r['baseline_iou'] for r in sub]),gain_r2=np.mean([r['gain_r2'] for r in sub]),gain_r4=np.mean([r['gain_r4'] for r in sub]),gain_r8=np.mean([r['gain_r8'] for r in sub])))
    # Descriptive standardized association, no causal claim or inferential p-value.
    for ds in LABELS:
        rr=[r for r in rows if r['dataset']==ds]
        for field in ['T','L','GT_fraction','uniform_inside_frames','baseline_iou']:
            x=np.array([r[field] for r in rr]);y=np.array([r['gain_r8'] for r in rr]);corr=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None
            associations.append(dict(dataset=ds,variable=field,pearson_r=corr,N=len(rr)))
    csvwrite(OUT/'metrics/dataset_query_statistics.csv',rows);csvwrite(OUT/'metrics/boundary_allocations.csv',alloc);csvwrite(OUT/'metrics/dataset_strata.csv',groups);csvwrite(OUT/'metrics/descriptive_associations.csv',associations)
    write(OUT/'metrics/dataset_statistics.json',stats)
    write(OUT/'metadata/dataset_diagnostics_sources.json',dict(code=fileinfo(Path(__file__)),cohort=fileinfo(pa/'metadata/cohort.json'),evaluations=fileinfo(pa/'metrics/evaluation_records.json'),plans_roots=[str(pa/'plans'),str(pb/'plans')],scope='fixed pilot 100 queries per dataset, not whole benchmark',posthoc=True))
    render(stats,groups,alloc)
    return stats

def render(stats,groups,alloc):
    summaries=[read(ROOT/'artifacts/runs'/run/'metrics/summary.json') for run in [A,B]]
    lookup=[{(x['dataset'],x['condition']):x for x in s['metrics']} for s in summaries]
    def metric(ds,c,d=0):return lookup[d][ds,c]['miou']*100
    def pair(ds,a,b,d=0):return next(x for x in summaries[d]['comparisons'] if x['dataset']==ds and x['a']==a and x['b']==b)
    def ci(x):return f"{x['delta_pp']:+.2f} [{x['lower_pp']:+.2f}, {x['upper_pp']:+.2f}]"
    def med(ds,f,scale=1):return stats[ds][f]['median']*scale
    lines=['# 정답 전체를 볼까, 시작·끝을 볼까? — pilot300 결과','',
    '**현재 가장 뚜렷한 결과는 정답 전체 집중의 이득이 ActivityNet과 QVHighlights에서 나왔고, Charades에서는 거의 없었다는 것입니다.** 경계만 강조하는 배치는 전체 집중보다 대체로 낮았습니다. 경계 배정 공식을 바꿔도 전체 평균의 우열은 분명해지지 않았습니다.','',
    '이 보고서는 세 벤치마크에서 이미 고정한 질문 100개씩, 총300개를 분석합니다. 데이터셋 전체의 통계가 아닙니다. GT는 정답 시간 구간입니다. GT를 이용한 배치는 가능성을 보는 진단이며 실용적인 추론 방법이나 최적 성능 상한이 아닙니다.','',
    '## 1. 먼저, 무엇이 좋아졌나?','',
    '**구간 겹침률(IoU)** = 예측과 정답이 겹친 길이 ÷ 두 구간의 합집합 길이입니다. 질문별 IoU의 평균이 **mIoU**이며, 아래는 %로 표시하고 높을수록 좋습니다. **pp**는 백분율 차이입니다. 40%에서45%가 되면 +5pp입니다. 파싱 실패·잘못된 구간은0점이며 원본 초 단위로 평가합니다.','',
    '다음 표는 **전체에 균등하게 보여줬을 때와 정답 전체를 강조했을 때**를 비교합니다. 2·4·8배는 정답 안의 초당 프레임 수를 밖보다 얼마나 높게 할지 정한 목표입니다. 총 프레임 수와 영상 토큰 수는 같은 질문 안에서 같습니다.','',
    '| 데이터셋 | 균등 배치 | 정답 전체 2배 | 정답 전체 4배 | 정답 전체 8배 | 8배−균등 차이 |','|---|---:|---:|---:|---:|---:|']
    for ds,label in list(LABELS.items())+[('all','전체300개')]:lines.append(f"| {label} | {metric(ds,'uniform'):.2f} | {metric(ds,'gt_whole_r2'):.2f} | {metric(ds,'gt_whole_r4'):.2f} | {metric(ds,'gt_whole_r8'):.2f} | {pair(ds,'gt_whole_r8','uniform')['delta_pp']:+.2f}pp |")
    lines+=['', '위 표의 8배는 시험한 값 중 전체 평균이 가장 높았다는 뜻이지, 최적값으로 확정했다는 뜻은 아닙니다. 정답과 같은 길이의 무작위 구간을 강조한 Random도 seed42·43으로 비교했습니다. 전체300개의 정답 전체8배는 Random 평균보다 +7.14pp 높았습니다. 단순히 아무 곳이나 강조해서 얻은 개선만으로 설명되지는 않습니다.','',
    '## 2. 두 실험은 무엇이 다른가?','',
    '**실험 A — 집중 프레임 수를 공유:** 정답 전체에 줄 프레임 수를 먼저 정하고, 경계 조건에도 같은 수를 줍니다. “같은 수의 집중 프레임을 전체 동작에 펼칠까, 시작·끝에 몰아줄까?”를 묻습니다.','',
    '**실험 B — 안팎의 밀도 비율을 일치:** 집중 영역의 실제 길이에 따라 프레임 수를 따로 계산합니다. 짧은 경계 영역에는 더 적은 프레임을 주고 나머지를 밖에 남깁니다. “전체 동작과 시작·끝을 각각 주변보다 똑같이 강조하면 어느 쪽이 나을까?”를 묻습니다.','',
    '예를 들어 영상30초, 정답12초, 경계영역 합계4초, 총60장,4배 설정이면: 실험A는 전체와 경계에 각각44장, 실험B는 전체44장·경계23장입니다. 두 실험 모두 총60장입니다.','',
    '공식은 K=round(F×rD/(rD+T−D))입니다. F는 총 프레임 수, T는 영상 길이, r은2·4·8입니다. 실험A는 경계에서도 D=정답 전체 길이로 계산하고, 실험B는 D=실제 집중 영역 길이를 씁니다. 경계는 시작·끝 ±1초의 합집합이며 영상 밖과 겹치는 부분을 정리한 뒤 길이를 셉니다.','',
    '**기존 보고서의 두 표는 서로 다른 질문을 섞어 보여줬습니다.** 첫 표는 “경계 배정 공식을 A→B로 바꾸면 좋아지나?”, 둘째 표는 “각 공식 안에서 경계가 전체보다 좋은가?”였습니다. 비교 대상을 먼저 설명하지 않아 읽기 어려웠습니다. 아래는 같은 점수표 안에 함께 놓았습니다.','',
    '| 전체300개 평균 겹침률 | 2배 설정 | 4배 설정 | 8배 설정 |','|---|---:|---:|---:|',f"| 균등 배치 | {metric('all','uniform'):.2f} | {metric('all','uniform'):.2f} | {metric('all','uniform'):.2f} |"]
    for label,c,d in [('정답 전체 — A·B 공통','gt_whole',0),('정답 경계 — A: 집중 프레임 수 공유','gt_boundary',0),('정답 경계 — B: 밀도 비율 일치','gt_boundary',1)]:lines.append('| '+label+' | '+' | '.join(f"{metric('all',f'{c}_r{r}',d):.2f}" for r in [2,4,8])+' |')
    lines+=['','**세로로 읽으면 전체 대 경계 비교**, 경계 두 행끼리 읽으면 **공식 A 대 B 비교**입니다. A의 경계 행에서2·4·8은 집중 프레임 수를 결정하는 설정이며 경계의 실제 밀도 비율이 그 값이라는 뜻은 아닙니다. 전체 조건과 균등 배치는 두 실험이 같은 입력을 공유하므로 두 번의 독립적인 확인이 아닙니다.','',
    '경계도4·8배에서는 균등 배치와 대응 Random보다 좋아졌습니다. 그러나 전체 집중과 비교하면2·8배에서 낮았고4배에서는 우열이 불확실했습니다. 따라서 “경계가 중요하지 않다”가 아니라 “이번 ±1초 배치는 전체 집중보다 덜 유리했다”가 결과에 맞는 설명입니다.','',
    '## 3. 세 데이터셋의 입력은 얼마나 다른가?','',
    '다음은 **이번100개 질문의 중앙값**입니다. 중앙값은 값을 작은 순서로 정렬했을 때 가운데 값이라 극단적으로 긴 영상의 영향을 덜 받습니다. 평균·25/75백분위수·최솟값·최댓값은 연결된 JSON에 모두 저장했습니다. 같은 영상의 질문이 여러 개면 질문별 통계에 각각 포함합니다.','',
    '| 통계 | Charades | ActivityNet | QVHighlights |','|---|---:|---:|---:|']
    for label,f,scale,unit in [('영상 길이','T',1,'초'),('정답 구간 길이','L',1,'초'),('정답 길이 / 영상 길이','GT_fraction',100,'%'),('입력 프레임 수','F',1,'장'),('균등 배치의 초당 프레임 수','uniform_fps',1,'장/초'),('균등 배치에서 정답 안의 프레임 수','uniform_inside_frames',1,'장'),('정답 전체8배에서 정답 안의 프레임 수','GT_inside_frames_r8',1,'장'),('정답 안에 추가된 프레임 수:8배−균등','extra_inside_r8',1,'장'),('실제 영상 토큰 수','B',1,'개'),('시간 패치 하나당 공간 토큰 수','spatial_tokens_per_temporal_pair',1,'개')]:lines.append('| '+label+' | '+' | '.join(f'{med(ds,f,scale):.2f}{unit}' for ds in LABELS)+' |')
    lines.append('| 서로 다른 영상 수 | '+' | '.join(str(stats[ds]['n_video']) for ds in LABELS)+' |')
    lines.append('| 정답이 영상80% 이상인 질문 수 | '+' | '.join(str(stats[ds]['GT80_count']) for ds in LABELS)+' |')
    lines+=['','각 행의 중앙값은 따로 계산했으므로 중앙값끼리 나눠 다른 행을 재현할 수는 없습니다. 영상 토큰은 모델이 영상을 입력 특징으로 바꾼 단위입니다. 같은 질문의 조건 간 예산은 같지만, 데이터셋 사이의 영상 길이·공간 grid·프레임 수까지 같은 것은 아닙니다.','',
    '## 4. 실제 프레임 부족 때문에 경계 배치가 달라졌나?','',
    '아래 숫자는 경계 조건에서 후보 프레임이 부족해 집중 예산을 밖으로 옮긴 질문 수입니다. 분모는 데이터셋마다100개입니다. 목표 배정과 실제 배정이 달라진 정도를 확인하려는 표이며 성능 점수가 아닙니다.','',
    '| 데이터셋 | A:2/4/8배 부족 건수 | B:2/4/8배 부족 건수 |','|---|---|---|']
    for ds,label in LABELS.items():
        cells=[]
        for design in ['same_count','same_density']:cells.append(' / '.join(str(sum(x['transfer']!=0 for x in alloc if x['dataset']==ds and x['design']==design and x['ratio']==r)) for r in [2,4,8]))
        lines.append(f'| {label} | {cells[0]} | {cells[1]} |')
    lines+=['','## 5. 길이 차이만으로 설명할 수 있나?','',
    '아래는 정답이 영상에서 차지하는 비율을 비슷하게 나눠 본 **사후 분석**입니다. 각 칸은 “질문 수 / 정답 전체8배−균등의 평균 차이(pp)”입니다. 작은 집단은 몇 질문에 크게 흔들릴 수 있으므로 원인 검증으로 해석하지 않습니다.','',
    '| 정답 길이 비율 | Charades | ActivityNet | QVHighlights |','|---|---|---|---|']
    for lo,hi,label in [(0,.1,'10% 미만'),(.1,.4,'10–40% 미만'),(.4,.8,'40–80% 미만'),(.8,1.00001,'80% 이상')]:
        cells=[]
        for ds in LABELS:
            g=next((g for g in groups if g['dataset']==ds and g['variable']=='GT_fraction' and g['lower']==lo),None);cells.append(f"{g['N']}개 / {g['gain_r8']:+.2f}pp" if g else '해당 없음')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','**통계로 좁힐 수 있는 원인 후보와 반례**','',
    '- ActivityNet은 영상 중앙값132초에 정답5.5초로 정답 비율 중앙값이4.8%입니다. 긴 영상의 좁은 구간으로 관측을 옮기는 효과와 맞는 패턴입니다. 정답 비율10% 미만64개에서는 +15.54pp였습니다.',
    '- QVHighlights는 정답 길이 중앙값14.5초이며 정답 안에 추가된 프레임 수 중앙값이103장입니다. Charades의18장보다 재배치 규모가 큽니다. 같은8배라도 실제 추가 관측 수와 절대 밀도는 같지 않습니다.',
    '- Charades는 평균 +0.04pp지만 정답 비율10% 미만44개에서는 +2.65pp입니다. 비율40% 이상11개에서는 평균적으로 악화돼 개선을 상쇄합니다. 모든 질문에서 이득이 없었다는 뜻은 아닙니다.',
    '- 하지만 ActivityNet의60초 미만11개에서도 +17.66pp가 나왔습니다. 정답 비율10% 미만으로 제한해도 Charades +2.65pp, ActivityNet +15.54pp, QVHighlights +11.29pp로 차이가 남습니다. 영상 길이나 정답 비율 하나만으로 충분히 설명되지는 않습니다.',
    '- 공간 해상도도 다릅니다. 인접 프레임을 묶는 시간 패치 하나당 공간 토큰 수 중앙값은 Charades360개, ActivityNet93.5개, QVHighlights91개입니다. 전체 토큰 예산이 비슷해도 시간 관측 수와 공간 상세도의 구성이 다릅니다. 데이터셋 차이를 영상 길이 하나의 효과로 분리할 수는 없습니다.',
    '- Charades는 정답 전체8배가 Uniform보다 거의 개선되지 않았지만 Random보다 +3.29pp 높았습니다. 정답 위치가 무의미하다기보다 기존 균등 배치보다 나아지기 어려웠다는 표현이 더 정확합니다.','',
    '**경계 배치의 후보 부족도 유일한 설명은 아닙니다.** 실험A의8배 경계 배치는 QVHighlights58/100개에서 집중 프레임을 채우지 못했습니다. 실험B에서는0개로 줄었지만 전체300개 경계 점수는 A보다0.55pp 낮고 그 차이도 불확실했습니다.','',
    '이 분석은 원인 후보를 좁히는 관찰입니다. 원인을 더 구분하려면 영상 길이·정답 비율·기존 점수가 비슷한 질문끼리 비교하고 시간 관측 수와 공간 해상도를 따로 바꿔야 합니다. GT에 따른 timestamp 배치 자체도 정답 위치 단서를 줄 수 있으므로 픽셀 관측 증가만의 효과로 분리했다고 볼 수 없습니다.']
    lines+=['','영상 길이와 원래 균등 배치 점수별 집단도 dataset_strata.csv에 저장했습니다. 원래 점수가 낮은 질문에서 개선 여지가 크다는 관계는 점수 상한과 평균으로의 회귀 영향을 포함하므로 원인 증거로 쓰지 않습니다. 영상 내용의 복잡성·동작 유형·정답 표현 방식은 이 통계만으로 확인하지 않았습니다.','',
    '## 6. 결과를 어디까지 믿을 수 있나?','',
    '같은 영상의 질문을 함께 뽑는 영상 단위 bootstrap(재표집)20,000회로95% 신뢰구간을 계산했습니다. 0을 포함하면 방향이 불확실합니다. 여러 설정을 탐색했으며 다중비교 보정이나 새 영상 확증은 하지 않았습니다.','',
    '| 데이터셋 | 정답 전체8배−균등 pp [95% 구간] | 정답 전체8배−Random 평균 pp [95% 구간] |','|---|---|---|']
    for ds,label in list(LABELS.items())+[('all','전체300개')]:lines.append(f"| {label} | {ci(pair(ds,'gt_whole_r8','uniform'))} | {ci(pair(ds,'gt_whole_r8','random_whole_r8_mean'))} |")
    lines+=['','두 공식의 경계 성능 차이는 전체300개에서2배 +0.24pp,4배 +0.01pp,8배 −0.55pp이며 모두 신뢰구간이0을 포함합니다. 더 좋은 공식이 확정된 것은 아닙니다.','',
    'GPU 속도 개선 과정에서 최대4개 배치를 사용했습니다. 세 벤치마크8개 점검 입력의 단일/배치 생성 토큰은 일치했지만 전량 동등성을 확인한 것은 아닙니다. 전체11400개 조건별 결과 중 동일 입력 응답은 재사용했으며 독립 반복으로 세지 않았습니다. annotation 길이보다 GT 끝이 긴 한 건은 GT를 자르지 않고 검증된 파일 재생 길이를 계획에 사용했습니다.','',
    '## 원자료와 재현','',f'[질문별 데이터·입력·개선량](../../artifacts/runs/{RUN}/metrics/dataset_query_statistics.csv) · [분포 통계 전체](../../artifacts/runs/{RUN}/metrics/dataset_statistics.json) · [집단별 결과](../../artifacts/runs/{RUN}/metrics/dataset_strata.csv) · [경계 실제 배정](../../artifacts/runs/{RUN}/metrics/boundary_allocations.csv)',f'[모든 조건의 점수](../../artifacts/runs/{RUN}/metrics/combined_metrics.csv) · [공식 간 비교 상세](../../artifacts/runs/{RUN}/metrics/cross_design_comparisons.json)',f'[실험A 상세](../{A}/REPORT.md) · [실험B 상세](../{B}/REPORT.md)','', '재생성: `.venv/bin/python _lib/gt_dataset_diagnostics.py` 또는 기존 통합 생성기 `_lib/gt_density_combined.py`.']
    from gt_threshold_report import build as build_thresholds
    threshold_lines=build_thresholds()
    insert_at=lines.index('## 2. 두 실험은 무엇이 다른가?')
    lines[insert_at:insert_at]=threshold_lines+['']
    lines[2:2]=['[추가 추론 없는 GT 비율 구간별 비교: 표본 수·threshold·구간 비중 통제](GT_FRACTION.md)','']
    axes_report=REPORT/'GT_FRACTION.md'
    if axes_report.exists() and '## 8. GT 비율 × 영상 길이 교차표' in axes_report.read_text():
        cross=axes_report.read_text().split('## 8. GT 비율 × 영상 길이 교차표',1)[1].split('## 9.',1)[0]
        lines+=['','## GT 비율과 영상 길이를 함께 나눈 교차표','',cross.strip(),'','[영상 길이별·GT 비율별 분석과 비교의 한계](GT_FRACTION.md)']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    stats=main()
    for ds,s in stats.items():print(ds,{f:round(s[f]['median'],3) for f in ['T','L','GT_fraction','F','uniform_fps','uniform_inside_frames','extra_inside_r8']},'videos',s['n_video'])
