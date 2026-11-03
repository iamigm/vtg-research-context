"""Post-hoc analysis of the completed pilot; no inference and no source writes."""
from result_layout import write_result_text
from result_layout import relocate_path
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import adaptive_temporal_sampling as old
from adaptive_temporal_report import score,csvwrite
import numpy as np

RUN='temporal_sampling_pilot300_v2'
OUT=relocate_path(old.ROOT/'artifacts/runs'/RUN)
REPORT=relocate_path(old.ROOT/'results'/RUN)

def bootstrap_weights(videos,seed=42):
    unique=sorted(set(videos));rng=np.random.default_rng(seed);draws=rng.integers(len(unique),size=(20000,len(unique)))
    counts=np.stack([(draws==i).sum(axis=1) for i in range(len(unique))],axis=1)
    membership=np.array([[v==u for u in unique] for v in videos],dtype=np.float64)
    return counts@membership.T

def estimate(values,weights):
    values=np.asarray(values,dtype=float);mask=np.isfinite(values);denom=weights[:,mask].sum(axis=1)
    reps=weights[:,mask]@values[mask]/denom
    return dict(n=int(mask.sum()),mean=float(values[mask].mean()),median=float(np.median(values[mask])),ci95=[float(x) for x in np.quantile(reps,[.025,.975])])

def analyze_previous():
    records=old.read(old.OUT/'metadata/baseline_records.json');source_manifest=old.read(old.OUT/'metadata/source_manifest.json');vm={v['path']:v for v in source_manifest['videos']}
    rows=[];near=[];sources=[];videos=[r['video_id'] for r in records];weights=bootstrap_weights(videos)
    for r in records:
        s=r['sample'];key=s['sample_key'];gt=s['span'][0];vals={}
        for condition in ['uniform','gt','random_s42','random_s43']:
            path=old.OUT/'responses'/condition/(key+'.json');raw=old.read(path);sources.append(old.fileinfo(path));vals[condition]=score(raw['answer'],gt,s['duration'])
        u=vals['uniform'];g=vals['gt'];row=dict(sample_key=key,video_id=r['video_id'],uniform_start_error=u['start_error'],uniform_end_error=u['end_error'],gt_start_error=g['start_error'],gt_end_error=g['end_error'],start_error_reduction=u['start_error']-g['start_error'] if u['valid'] and g['valid'] else None,end_error_reduction=u['end_error']-g['end_error'] if u['valid'] and g['valid'] else None,iou_change=g['iou']-u['iou'])
        for condition in ['random_s42','random_s43']:
            x=vals[condition];row[condition+'_start_reduction']=u['start_error']-x['start_error'] if u['valid'] and x['valid'] else None;row[condition+'_end_reduction']=u['end_error']-x['end_error'] if u['valid'] and x['valid'] else None;row[condition+'_iou_change']=x['iou']-u['iou']
        for endpoint in ['start','end']:
            v=[row[c+'_'+endpoint+'_reduction'] for c in ['random_s42','random_s43']]
            row['random_mean_'+endpoint+'_reduction']=float(np.mean(v)) if all(x is not None for x in v) else None
        rows.append(row)
        base=set(old.read(old.OUT/'plans/uniform'/(key+'.json'))['frame_ids']);fps=vm[s['video_path']]['fps']
        for condition in ['gt','random_s42','random_s43']:
            ids=old.read(old.OUT/'plans'/condition/(key+'.json'))['frame_ids'];new=[i/fps for i in ids if i not in base];uniform=[i/fps for i in sorted(base)];alltimes=[i/fps for i in ids]
            for label,t in zip(['start','end'],gt):
                for eps in [.05,.1,.25,.5,1.]:
                    near.append(dict(sample_key=key,condition=condition,endpoint=label,radius_seconds=eps,uniform_count=sum(abs(x-t)<=eps for x in uniform),condition_count=sum(abs(x-t)<=eps for x in alltimes),new_source_frames=sum(abs(x-t)<=eps for x in new),new_before=sum(t-eps<=x<t for x in new),new_at_or_after=sum(t<=x<=t+eps for x in new),uniform_nearest_seconds=min(abs(x-t) for x in uniform),condition_nearest_seconds=min(abs(x-t) for x in alltimes)))
    stats={}
    for endpoint in ['start','end']:
        a=np.array([r[endpoint+'_error_reduction'] if r[endpoint+'_error_reduction'] is not None else np.nan for r in rows]);b=np.array([r['random_mean_'+endpoint+'_reduction'] if r['random_mean_'+endpoint+'_reduction'] is not None else np.nan for r in rows])
        stats['gt_'+endpoint+'_reduction']=estimate(a,weights);stats['random_mean_'+endpoint+'_reduction']=estimate(b,weights);stats['gt_minus_random_mean_'+endpoint+'_reduction']=estimate(a-b,weights)
    er=np.array([r['end_error_reduction'] for r in rows]);sr=np.array([r['start_error_reduction'] for r in rows]);rm=np.array([r['random_mean_end_reduction'] for r in rows]);order=np.argsort(-er,kind='stable');positive=np.maximum(er,0);top=order[:5]
    concentration=dict(improved=int((er>0).sum()),worsened=int((er<0).sum()),equal=int((er==0).sum()),end_improved_start_worsened=int(((er>0)&(sr<0)).sum()),end_and_start_improved=int(((er>0)&(sr>0)).sum()),gt_and_random_mean_end_improved=int(((er>0)&(rm>0)).sum()),gt_only_end_improved=int(((er>0)&(rm<=0)).sum()),random_only_end_improved=int(((er<=0)&(rm>0)).sum()),top5_share_of_positive_reduction=float(positive[top].sum()/positive.sum()),mean_reduction_without_top5=float(np.delete(er,top).mean()),median_reduction=float(np.median(er)),top5=[rows[i] for i in top])
    old.write(OUT/'metrics/previous_boundary_analysis.json',dict(estimates=stats,concentration=concentration,bootstrap=dict(unit='video',n_video=len(set(videos)),n_query=100,replicates=20000,seed=42),sources=sources));csvwrite(OUT/'metrics/previous_query_changes.csv',rows);csvwrite(OUT/'metrics/previous_added_boundary_frames.csv',near)
    lines=['# 기존 100개 결과: 종료 시각 오차는 왜 줄었나?','',
        '기존 응답만 다시 분석했습니다. 추가 GPU 추론은 없습니다. 아래 결과는 다음 실험을 설계하는 데 사용한 사후 분석이며, 원인을 확정하는 실험은 아닙니다.','',
        '먼저 용어를 설명합니다. 시작 시각의 절대 오차는 |예측 시작 시각 − 정답 시작 시각|이고, 종료 시각도 같은 방식으로 계산합니다. 질문별 절대 오차를 평균한 값이 **평균 절대 오차(Mean Absolute Error, MAE)**입니다. 단위는 초이며 작을수록 정확합니다. 예를 들어 정답이 10초이고 예측이 13초이면 절대 오차는 3초입니다.','',
        '이 보고서의 **오차 감소량 = Uniform의 절대 오차 − 비교 방법의 절대 오차**입니다. 양수는 개선, 음수는 악화입니다. 정의를 바꾼 새 평가 지표가 아니라 두 방법의 절대 오차 차이입니다. 무효 응답은 경계 오차 평균에서 제외하고 비교 가능한 질문 수를 표시합니다. 기존 100개는 모두 비교 가능합니다.','',
        '95% 신뢰구간은 영상을 20,000번 다시 뽑는 bootstrap(재표집, seed 42)으로 구했습니다. 같은 영상의 질문은 함께 뽑습니다. 0을 포함하면 평균 개선의 방향이 불확실합니다. 여러 비교를 탐색하므로 확증적 검정으로 해석하지 않습니다.','',
        '| 비교 | 줄어든 시작 오차: 평균 초 [95% 구간] | 줄어든 종료 오차: 평균 초 [95% 구간] |','|---|---|---|']
    for name,label in [('gt','정답 중심 집중 − Uniform'),('random_mean','Random 두 seed 평균 − Uniform'),('gt_minus_random_mean','정답 중심 집중 − Random 평균')]:
        a=stats[name+'_start_reduction'];b=stats[name+'_end_reduction'];lines.append(f"| {label} | {a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {b['mean']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] |")
    c=concentration;end=stats['gt_end_reduction']
    lines += ['',f"정답 중심 집중에서 종료 오차가 줄어든 질문은 {c['improved']}개, 늘어난 질문은 {c['worsened']}개, 같은 질문은 {c['equal']}개입니다. 질문별 종료 오차 감소량의 중앙값은 {c['median_reduction']:.3f}초입니다.",
        f"종료 오차 감소가 가장 큰 5개 질문은 모든 양의 감소량 합계 중 {100*c['top5_share_of_positive_reduction']:.1f}%를 차지합니다. 이 5개를 제외한 나머지 95개의 평균 감소량은 {c['mean_reduction_without_top5']:+.3f}초입니다. 이 제외 분석은 큰 사례의 영향을 설명하는 사후 진단이며, 새 주평가가 아닙니다.",
        f"종료 오차가 줄어든 대신 시작 오차가 늘어난 질문은 {c['end_improved_start_worsened']}개입니다. GT와 Random 평균에서 함께 종료 오차가 줄어든 질문은 {c['gt_and_random_mean_end_improved']}개, GT에서만 줄어든 질문은 {c['gt_only_end_improved']}개입니다.",
        f"평균 종료 오차 감소의 구간이 {'0을 포함하므로 개선 방향을 확정할 수 없습니다' if end['ci95'][0]<=0<=end['ci95'][1] else '0보다 높으므로 이 표본에서는 평균 감소 신호가 있습니다'}. 이것만으로 정답 위치를 알려준 효과라고 단정하지 않고, 위 Random 대조와 시작 오차 변화를 함께 봐야 합니다.",'',
        '| 정답 경계 주변 범위 | GT에서 새로 추가된 시작 주변 프레임: 질문당 평균 | 종료 주변: 질문당 평균 |','|---|---:|---:|']
    for eps in [.05,.1,.25,.5,1.]:
        values=[float(np.mean([r['new_source_frames'] for r in near if r['condition']=='gt' and r['endpoint']==ep and r['radius_seconds']==eps])) for ep in ['start','end']];lines.append(f'| ±{eps:g}초 | {values[0]:.2f} | {values[1]:.2f} |')
    lines += ['', '새 프레임은 기존 Uniform에 없던 원본 frame ID로 정의했습니다. 단순히 집중 슬롯에 들어갔다고 새 관측으로 세지 않았습니다. CSV에는 경계 직전·직후 개수, 기존·변경 입력의 근접 프레임 개수와 가장 가까운 프레임까지의 시간 거리도 있습니다.','',
        '| 종료 오차 감소가 큰 5개 질문 | 종료 오차 감소 초 | 시작 오차 감소 초 | 시간 구간 겹침률 변화 |','|---|---:|---:|---:|']
    for r in c['top5']:lines.append(f"| {r['sample_key']} | {r['end_error_reduction']:+.2f} | {r['start_error_reduction']:+.2f} | {r['iou_change']:+.3f} |")
    lines += ['', '[질문별 변화](../../artifacts/runs/'+RUN+'/metrics/previous_query_changes.csv) · [실제로 추가된 경계 프레임](../../artifacts/runs/'+RUN+'/metrics/previous_added_boundary_frames.csv) · [계산값과 원본 hash](../../artifacts/runs/'+RUN+'/metrics/previous_boundary_analysis.json)']
    REPORT.mkdir(exist_ok=True,parents=True);write_result_text(REPORT/'PREVIOUS_RESULTS.md', '\n'.join(lines)+'\n')
    return dict(estimates=stats,concentration=concentration)

if __name__=='__main__':
    result=analyze_previous();print(old.encoded(dict(estimates=result['estimates'],concentration={k:v for k,v in result['concentration'].items() if k!='top5'})).decode())
