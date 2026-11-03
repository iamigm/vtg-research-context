"""Plain-language report for all pilot300 experiments; saved responses only."""
from result_layout import write_result_text
from boundary_oracle_sampling import *
from adaptive_temporal_report import score,csvwrite
from boundary_oracle_analysis import bootstrap_weights,estimate
import numpy as np
from statistics import NormalDist

def fmt(x,d=2):return '—' if x is None else f'{x:.{d}f}'

def evaluate(records,cache_index,include_add=False):
    from temporal_pilot300_run import response_path,MID
    groups=[('midpoint','full',MID),*[( 'boundary',b,CONDITIONS) for b in BUDGETS]]
    if include_add:groups.append(('addition','full',CONDITIONS))
    rows=[];paired=[];metrics=[];diagnostics=[];costs=[];seen=set();values={};hints=[];power=[]
    for phase,budget,conditions in groups:
        for record in records:
            s=record['sample'];key=s['sample_key'];gt=s['span'][0];T=s['duration'];cache=read(cache_index[key]);base_condition='uniform'
            uniform_path=response_path(phase,budget,'uniform',key);u=read(uniform_path);uniform=score(u['answer'],gt,T)
            uplan_path=OUT/'plans'/('shared' if budget=='full' else phase)/budget/'uniform'/(key+'.json');uniform_ids=set(read(uplan_path)['frame_ids'])
            for condition in conditions:
                p=response_path(phase,budget,condition,key);r=read(p);evaluation=score(r['answer'],gt,T)
                input_path=OUT/'inputs'/r['identity']['phase']/r['identity']['budget']/r['identity']['condition']/(key+'.json');detail=read(input_path)
                need(r['identity']['input_hash']==objhash(detail['hashes']),'Input integrity failed during evaluation')
                row=dict(dataset=s['dataset'],sample_key=key,video_id=record['video_id'],phase=phase,budget=budget,condition=condition,**evaluation,F=r['F'],B=r['B'],grid=r['grid'],generated_tokens=r['generated_tokens'],eos_reached=r['eos_reached'])
                rows.append(row);values[(phase,budget,key,condition)]=evaluation
                plan_phase=r['identity']['phase'];plan_path=OUT/'plans'/plan_phase/budget/condition/(key+'.json');planned=read(plan_path)
                need(objhash(planned)==r['identity']['plan_hash'],'Plan integrity failed')
                new=[t for i,t in zip(planned['frame_ids'],planned['pts']) if i not in uniform_ids]
                diag=dict(dataset=s['dataset'],sample_key=key,phase=phase,budget=budget,condition=condition,F=r['F'],B=r['B'],spatial_grid=str(r['grid'][1:]),fill_fraction=planned['fill_fraction'],new_source_frames=len(new),clipped=planned.get('clipped',False),overlap=planned.get('overlap',False))
                for label,boundary in zip(['start','end'],gt):
                    diag[label+'_nearest_seconds']=min(abs(t-boundary) for t in planned['pts'])
                    for eps in [.05,.1,.25,.5,1.]:
                        diag[label+'_new_within_'+str(eps)]=sum(abs(t-boundary)<=eps for t in new)
                        diag[label+'_all_within_'+str(eps)]=sum(abs(t-boundary)<=eps for t in planned['pts'])
                diagnostics.append(diag)
                if str(p) not in seen:
                    seen.add(str(p));costs.append(dict(dataset=s['dataset'],sample_key=key,phase=r['identity']['phase'],budget=r['identity']['budget'],condition=r['identity']['condition'],F=r['F'],B=r['B'],gpu_generation_seconds=r['generation_seconds'],gpu_prefill_seconds=r['prefill_seconds'],gpu_decode_seconds=r['decode_seconds'],cpu_input_seconds=r['input_prepare_seconds'],peak_allocated_GiB=r['peak_allocated_bytes']/2**30,output_tokens=r['generated_tokens']))
    labels=sorted({r['sample']['dataset'] for r in records})+['all']
    for dataset in labels:
        rr=[r for r in records if dataset=='all' or r['sample']['dataset']==dataset];keys=[r['sample']['sample_key'] for r in rr];videos=[r['video_id'] for r in rr];weights=bootstrap_weights(videos)
        for phase,budget,conditions in groups:
            iv={c:np.array([values[(phase,budget,k,c)]['iou'] for k in keys]) for c in conditions}
            randoms=[c for c in conditions if 'random' in c];iv['random_mean']=np.mean([iv[c] for c in randoms],axis=0)
            for c in conditions:
                vv=[values[(phase,budget,k,c)] for k in keys];se=[v['start_error'] for v in vv if v['valid']];ee=[v['end_error'] for v in vv if v['valid']]
                metrics.append(dict(dataset=dataset,phase=phase,budget=budget,condition=c,N=len(keys),n_video=len(set(videos)),miou=float(iv[c].mean()),r03=float((iv[c]>=.3).mean()),r05=float((iv[c]>=.5).mean()),r07=float((iv[c]>=.7).mean()),start_absolute_error_mean=float(np.mean(se)) if se else None,end_absolute_error_mean=float(np.mean(ee)) if ee else None,boundary_valid_N=len(se),parse_failures=sum(v['parse_failed'] for v in vv),invalid=sum(v['invalid_interval'] for v in vv),out_of_range=sum(v['out_of_range'] for v in vv),both_endpoints_within_1s=sum(v['boundary_within_1s'] for v in vv),both_endpoints_within_2s=sum(v['boundary_within_2s'] for v in vv)))
            metrics.append(dict(dataset=dataset,phase=phase,budget=budget,condition='random_mean',N=len(keys),n_video=len(set(videos)),miou=float(iv['random_mean'].mean()),r03=float(np.mean([(iv[c]>=.3).mean() for c in randoms])),r05=float(np.mean([(iv[c]>=.5).mean() for c in randoms])),r07=float(np.mean([(iv[c]>=.7).mean() for c in randoms])),start_absolute_error_mean=float(np.mean([values[(phase,budget,k,c)]['start_error'] for k in keys for c in randoms if values[(phase,budget,k,c)]['valid']])),end_absolute_error_mean=float(np.mean([values[(phase,budget,k,c)]['end_error'] for k in keys for c in randoms if values[(phase,budget,k,c)]['valid']])),boundary_valid_N=None,parse_failures=None,invalid=None,out_of_range=None,both_endpoints_within_1s=None,both_endpoints_within_2s=None))
            oracle='gt_midpoint' if phase=='midpoint' else 'gt_boundary'
            contrasts=[(oracle,'uniform'),(oracle,'random_mean'),('random_mean','uniform')]
            if phase=='midpoint':contrasts+=[('pred_midpoint','uniform'),('pred_midpoint','random_mean')]
            estimates={}
            for a,b in contrasts:
                delta=iv[a]-iv[b];e=estimate(delta,weights);estimates[(a,b)]=e
                paired.append(dict(dataset=dataset,phase=phase,budget=budget,comparison=a+'-'+b,N=len(keys),n_video=len(set(videos)),mean_iou_change_pp=100*e['mean'],ci95_lower_pp=100*e['ci95'][0],ci95_upper_pp=100*e['ci95'][1],improved=int((delta>0).sum()),worsened=int((delta<0).sum()),unchanged=int((delta==0).sum())))
                # Absolute-error reductions are positive when the boundary is better localized.
                for endpoint in ['start','end']:
                    def ev(k,c):
                        if c=='random_mean':
                            v=[values[(phase,budget,k,rc)][endpoint+'_error'] for rc in randoms];return np.mean(v) if all(x is not None for x in v) else np.nan
                        v=values[(phase,budget,k,c)][endpoint+'_error'];return v if v is not None else np.nan
                    reductions=np.array([ev(k,b)-ev(k,a) for k in keys]);be=estimate(reductions,weights) if np.isfinite(reductions).any() else dict(n=0,mean=None,ci95=[None,None])
                    write(OUT/'metrics/boundary_intervals'/dataset/phase/budget/(a+'-'+b+'_'+endpoint+'.json'),dict(definition='absolute error of comparison B minus A; positive seconds mean improvement',**be))
            if dataset!='all' and phase=='boundary' and estimates[(oracle,'uniform')]['mean']>=.02 and estimates[(oracle,'random_mean')]['mean']>0:
                hints.append(dict(dataset=dataset,budget=budget,delta_uniform_pp=100*estimates[(oracle,'uniform')]['mean'],delta_random_pp=100*estimates[(oracle,'random_mean')]['mean'],exploratory_only=True))
                cluster=[]
                for vid in sorted(set(videos)):
                    where=[i for i,v in enumerate(videos) if v==vid];cluster.append([float((iv[oracle]-iv[b])[where].mean()) for b in ['uniform','random_mean']])
                sd=np.std(np.array(cluster),axis=0,ddof=1);z=NormalDist().inv_cdf(.9875)+NormalDist().inv_cdf(.8);n=int(math.ceil(max((z*x/.03)**2 for x in sd)))
                power.append(dict(dataset=dataset,budget=budget,setting_frozen=CONFIG,approximate_new_unique_videos=n,assumed_effect_pp=3,power=.8,comparisons=2,two_sided_family_alpha=.05,variance_source='paired video-mean changes in this pilot',exclude_current_pilot_videos=videos,one_query_per_new_video=True,automatic_expansion=False,reason='Quick exploratory screening; next run should use this frozen setting on disjoint videos, not tune this pilot again'))
    write(OUT/'metrics/evaluation_records.json',rows);csvwrite(OUT/'metrics/metrics.csv',metrics);csvwrite(OUT/'metrics/paired_comparisons.csv',paired);csvwrite(OUT/'metrics/sampling_diagnostics.csv',diagnostics);csvwrite(OUT/'metrics/compute_cost.csv',costs)
    write(OUT/'metadata/confirmation_plan.json',power)
    summary=dict(state='evaluated',n_query=300,random_seeds=[42,43],hints=hints,addition_executed=include_add,new_responses=len(seen),gpu_generation_seconds=sum(r['gpu_generation_seconds'] for r in costs),cpu_input_seconds=sum(r['cpu_input_seconds'] for r in costs),peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in costs),metrics=metrics,paired=paired)
    write(OUT/'metrics/summary.json',summary)
    lines=['# 프레임을 어디에 배치하면 시간 구간을 더 잘 맞힐까?','',
        f'pilot 300개 전체: Charades-TimeLens·ActivityNet-TimeLens·QVHighlights-TimeLens 각 100개 질문. Random은 seed 42·43만 사용했습니다. 평가 시각: {stamp()}.','',
        '먼저 지표를 설명합니다. **시간 구간 겹침률(IoU)**은 예측 구간과 정답 구간이 겹치는 길이를 두 구간의 합집합 길이로 나눈 값입니다. 1이면 정확히 겹치고 0이면 겹치지 않습니다. 이를 질문 전체에서 평균한 값이 **mIoU**이며, 표에서는 %로 표시하고 높을수록 좋습니다. **IoU≥0.5 비율**은 첫 예측 구간이 정답과 절반 이상 겹친 질문의 비율입니다.','',
        '**평균 절대 오차(Mean Absolute Error, MAE)**는 질문마다 |예측 시각−정답 시각|을 계산한 뒤 평균한 값입니다. 단위는 초이고 작을수록 좋습니다. 아래에는 시작 시각과 종료 시각을 따로 적었습니다. 예측을 읽을 수 없는 응답은 겹침률 0으로 포함하고, 경계 오차 평균에서는 제외한 질문 수를 별도로 CSV에 기록합니다.','',
        '**차이(pp)**는 두 백분율의 차이입니다. 예를 들어 40%→42%는 +2pp입니다. 95% 구간은 같은 영상의 질문을 묶어 20,000회 재표집한 탐색적 bootstrap 신뢰구간입니다. 0을 포함하면 개선 방향이 불확실합니다. 여러 조건을 살펴보는 빠른 가능성 점검이며 확증적 검정은 아닙니다.','',
        '원본 프레임을 시간순으로 한 번 준비한 캐시를 모든 조건이 공유합니다. 과거 sparse seek 입력과 픽셀이 다르면 과거 점수는 역사적 참고로만 두고, 이번에 다시 생성한 Uniform을 비교 기준으로 사용합니다. 모델·프롬프트·BF16·FlashAttention2·greedy 생성은 유지했습니다. 기존 결과는 덮어쓰지 않았습니다.','',
        '현재 예산은 기존 2fps 입력 프레임 수입니다. 절반·4분의 1은 모델의 2프레임 묶음에 맞춰 내림했습니다. 원래 18장이면 18→8→4장입니다. 공간 해상도는 원래 값으로 고정했으며, 같은 질문·같은 예산의 비교 조건끼리는 프레임 수와 visual-token 수가 정확히 같습니다.','',
        '시작·끝 집중은 전체에 고르게 남긴 절반의 프레임과 정답 시작·끝 각각 ±1초의 프레임을 합칩니다. Random도 무작위 두 시점에 같은 규칙을 씁니다. 영상 밖은 잘라내고, 겹치는 구간에서 같은 프레임을 중복 선택하지 않습니다. 후보가 부족하면 가까운 바깥 프레임으로 채우고 그 비율을 기록합니다.','',
        '## 기존 중심 집중 방식도 300개로 다시 확인','', '| 벤치마크 | 방식 | 평균 겹침률 % | IoU≥0.5 비율 % | 시작 오차 평균 초 | 종료 오차 평균 초 |','|---|---|---:|---:|---:|---:|']
    names={'uniform':'전체 고르게','gt_midpoint':'정답 중심 주변 0.4T','pred_midpoint':'Uniform 예측 중심 주변 0.4T','random_mean':'Random 두 seed 평균','gt_boundary':'정답 시작·끝 ±1초'}
    for dataset in labels[:-1]:
        for m in metrics:
            if m['dataset']==dataset and m['phase']=='midpoint' and m['condition'] in names:lines.append(f"| {dataset} | {names[m['condition']]} | {100*m['miou']:.2f} | {100*m['r05']:.2f} | {fmt(m['start_absolute_error_mean'])} | {fmt(m['end_absolute_error_mean'])} |")
    lines+=['','## 시작·끝 집중과 프레임 예산 축소','', '| 벤치마크 | 프레임 예산 | Uniform 평균 겹침률 % | 정답 경계 집중 % | Random 평균 % | 정답−Uniform pp [95% 구간] |','|---|---|---:|---:|---:|---|']
    for dataset in labels[:-1]:
        for budget in BUDGETS:
            mm={m['condition']:m for m in metrics if m['dataset']==dataset and m['phase']=='boundary' and m['budget']==budget};p=next(p for p in paired if p['dataset']==dataset and p['phase']=='boundary' and p['budget']==budget and p['comparison']=='gt_boundary-uniform')
            lines.append(f"| {dataset} | {dict(full='현재',half='절반',quarter='4분의 1')[budget]} | {100*mm['uniform']['miou']:.2f} | {100*mm['gt_boundary']['miou']:.2f} | {100*mm['random_mean']['miou']:.2f} | {p['mean_iou_change_pp']:+.2f} [{p['ci95_lower_pp']:+.2f}, {p['ci95_upper_pp']:+.2f}] |")
    lines+=['', '현재 예산의 Uniform과 저예산 조건의 수치 차이도 함께 읽어야 합니다. 저예산에서 Random보다 낫다는 사실만으로 원래 성능을 유지했다고 단정하지 않습니다. 모든 seed별 수치·시작/종료 오차·실패 수는 metrics.csv에 있습니다.','',
        '## 다음 행동','',
        '빠른 선별 규칙은 정답 경계 집중이 같은 예산 Uniform보다 2pp 이상 높고 Random 평균보다도 높은 경우를 “후속 확인 후보”로 남기는 것입니다. 이것은 통계적으로 검증됐다는 기준이 아닙니다. 신뢰구간은 별도로 보고합니다.']
    if hints:
        for h in hints:lines.append(f"- {h['dataset']} / {dict(full='현재 예산',half='절반 예산',quarter='4분의 1 예산')[h['budget']]}: Uniform 대비 {h['delta_uniform_pp']:+.2f}pp, Random 대비 {h['delta_random_pp']:+.2f}pp. 이 설정을 고정하고 겹치지 않는 새 영상에서 확인할 후보입니다.")
        lines+=['','새 영상의 대략적인 필요 수는 이 pilot의 영상별 성능 차이 분산으로 계산해 confirmation_plan.json에 저장했습니다. 목표 효과 3pp, 검정력 80%, 두 주비교를 가정한 근사치이며 실제 후속 실행 전에 정할 표본 수 계획입니다. 이번 빠른 실행에서 새 영상을 자동으로 추가해 설정을 계속 바꾸지는 않습니다.']
    elif include_add:lines+=['','세 예산에서 선별 기준을 넘는 후보가 없어, 기존 Uniform 프레임을 전부 보존하고 경계 주변 프레임만 추가하는 진단을 실시했습니다. 이 조건은 원래 Uniform보다 프레임·토큰 예산이 큽니다. GT 추가와 Random 추가끼리만 같은 증가 예산입니다.','', '| 벤치마크 | 정답 추가 평균 겹침률 % | Random 추가 평균 % | 정답 추가−Random 추가 pp [95% 구간] |','|---|---:|---:|---|']
    if include_add:
        for dataset in labels[:-1]:
            mm={m['condition']:m for m in metrics if m['dataset']==dataset and m['phase']=='addition'};p=next(p for p in paired if p['dataset']==dataset and p['phase']=='addition' and p['comparison']=='gt_boundary-random_mean');lines.append(f"| {dataset} | {100*mm['gt_boundary']['miou']:.2f} | {100*mm['random_mean']['miou']:.2f} | {p['mean_iou_change_pp']:+.2f} [{p['ci95_lower_pp']:+.2f}, {p['ci95_upper_pp']:+.2f}] |")
        lines.append('추가만 하는 조건에서도 Random과 차이가 작거나 방향이 불확실하면, 이 고정 방식의 확대보다 다음 가설을 검토하는 편이 낫습니다. 반대로 추가만 할 때 좋아지는 패턴은 기존 전체 장면 프레임을 빼는 손실이 있었을 가능성을 보여주지만, 인과 원인을 확정하지는 않습니다.')
    lines+=['',f"실제로 생성한 고유 응답은 {len(seen):,}개이며, GPU 생성 경과시간 합계는 {summary['gpu_generation_seconds']:.1f}초, 실제로 할당받은 GPU 메모리의 최대값는 {summary['peak_allocated_GiB']:.2f}GiB입니다. 공유 Uniform 응답은 비용을 중복 합산하지 않았습니다. Pred 중심 집중만 Uniform 첫 응답을 추가로 이용하므로 자연스러운 single-pass Uniform과 전체 비용이 같지 않습니다.",'',
        '[기존 100개 종료 오차 분석](PREVIOUS_RESULTS.md) · [상태](../../artifacts/runs/'+RUN+'/metadata/status.json) · [전체 수치](../../artifacts/runs/'+RUN+'/metrics/metrics.csv) · [질문별 평가](../../artifacts/runs/'+RUN+'/metrics/evaluation_records.json) · [대응 비교와 신뢰구간](../../artifacts/runs/'+RUN+'/metrics/paired_comparisons.csv) · [추가된 경계 프레임](../../artifacts/runs/'+RUN+'/metrics/sampling_diagnostics.csv) · [비용](../../artifacts/runs/'+RUN+'/metrics/compute_cost.csv)']
    write_result_text(REPORT/'REPORT.md', '\n'.join(lines)+'\n');return summary
