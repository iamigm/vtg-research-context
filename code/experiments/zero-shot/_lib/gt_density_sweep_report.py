"""Readable evaluation for the prespecified whole/boundary density sweep."""
from result_layout import write_result_text
import numpy as np
from adaptive_temporal_report import score,csvwrite
from boundary_oracle_analysis import bootstrap_weights,estimate

def evaluate_run(m,rows):
    data=[];sampling=[];costs=[]
    for r in rows:
        s=r['sample'];key=s['sample_key'];gt=s['span'][0]
        for c in m.CONDITIONS:
            response=m.read(m.OUT/'responses'/c/(key+'.json'));p=m.read(m.OUT/'plans'/c/(key+'.json'));detail=m.read(m.OUT/'inputs'/c/(key+'.json'))
            m.need(response['identity']['plan_hash']==m.objhash(p) and response['identity']['input_hash']==m.objhash(detail['hashes']),'Input/plan hash mismatch')
            m.need((response['F'],response['B'],response['grid'])==(r['F'],r['B'],r['grid']),'Budget mismatch')
            data.append(dict(dataset=s['dataset'],sample_key=key,video_id=r['video_id'],condition=c,length_fraction=(gt[1]-gt[0])/r['planning_T'],**score(response['answer'],gt,s['duration'])))
            overlap=max(0,min(gt[1],p['interval'][1])-max(gt[0],p['interval'][0])) if p['interval'] else None
            sampling.append(dict(sample_key=key,dataset=s['dataset'],condition=c,F=r['F'],B=r['B'],ratio=p.get('ratio'),focus_duration=p.get('focus_duration'),requested=p.get('inside_requested'),actual=p.get('inside_actual'),capacity_transfer=p.get('capacity_transfer',0),minimum_adjustment=p.get('minimum_adjustment',0),density_ratio_actual=p.get('density_ratio_actual'),random_gt_overlap_fraction=overlap/(gt[1]-gt[0]) if c.startswith('random') else None,focus_mean_target_deviation_seconds=float(np.mean(p['inside_target_time_absolute_deviations'])) if p.get('inside_target_time_absolute_deviations') else None))
            costs.append(dict(sample_key=key,condition=c,gpu_seconds=response['generation_seconds'],input_seconds=response['input_prepare_seconds'],tokens=response['generated_tokens'],peak_GiB=response['peak_allocated_bytes']/2**30,reused=response.get('reused_identical_input',False)))
    metrics=[];paired=[];strata=[]
    means={f'random_{mode}_r{ratio}_mean':[f'random_{mode}_r{ratio}_s{seed}' for seed in [42,43]] for ratio in m.RATIOS for mode in ['whole','boundary']}
    members={c:[c] for c in m.CONDITIONS};members.update(means)
    comparisons=[]
    for ratio in m.RATIOS:
        for mode in ['whole','boundary']:
            comparisons.extend([(f'gt_{mode}_r{ratio}','uniform'),(f'gt_{mode}_r{ratio}',f'random_{mode}_r{ratio}_mean')])
        comparisons.append((f'gt_boundary_r{ratio}',f'gt_whole_r{ratio}'))
    for mode in ['whole','boundary']:
        comparisons.extend([(f'gt_{mode}_r4',f'gt_{mode}_r2'),(f'gt_{mode}_r8',f'gt_{mode}_r4'),(f'gt_{mode}_r8',f'gt_{mode}_r2')])
    for dataset in sorted({r['sample']['dataset'] for r in rows})+['all']:
        rr=[r for r in rows if dataset=='all' or r['sample']['dataset']==dataset];keys=[r['sample']['sample_key'] for r in rr];videos=[r['sample']['dataset']+'/'+r['video_id'] for r in rr];w=bootstrap_weights(videos)
        by={(x['sample_key'],x['condition']):x for x in data};values={}
        for c,cc in members.items():
            values[c]=np.array([np.mean([by[k,d]['iou'] for d in cc]) for k in keys]);er=[by[k,d] for k in keys for d in cc];valid=[x for x in er if x['valid']]
            metrics.append(dict(dataset=dataset,condition=c,N=len(keys),N_video=len(set(videos)),miou=float(values[c].mean()),r05=float(np.mean([x['iou']>=.5 for x in er])),start_error_seconds=float(np.mean([x['start_error'] for x in valid])) if valid else None,end_error_seconds=float(np.mean([x['end_error'] for x in valid])) if valid else None,valid_boundary_responses=len(valid),total_responses=len(er),parse_failures=sum(x['parse_failed'] for x in er)))
        for a,b in comparisons:
            delta=values[a]-values[b];e=estimate(delta,w);row=dict(dataset=dataset,a=a,b=b,N=len(keys),delta_pp=e['mean']*100,lower_pp=e['ci95'][0]*100,upper_pp=e['ci95'][1]*100,improved=int((delta>0).sum()),worsened=int((delta<0).sum()),unchanged=int((delta==0).sum()))
            for endpoint in ['start','end']:
                def errors(c):
                    return np.array([np.mean([by[k,d][endpoint+'_error'] for d in members[c]]) if all(by[k,d][endpoint+'_error'] is not None for d in members[c]) else np.nan for k in keys])
                v=errors(b)-errors(a);row[endpoint+'_error_reduction']=estimate(v,w) if np.isfinite(v).any() else None
            paired.append(row)
        fractions=np.array([(r['sample']['span'][0][1]-r['sample']['span'][0][0])/r['planning_T'] for r in rr])
        for lower,upper in [(0,.1),(.1,.4),(.4,.8),(.8,1.000001)]:
            mask=(fractions>=lower)&(fractions<upper)
            if mask.any():
                for c,v in values.items():strata.append(dict(dataset=dataset,GT_fraction_lower=lower,GT_fraction_upper=upper,condition=c,N=int(mask.sum()),miou=float(v[mask].mean())))
    for name,value in [('evaluation_records',data),('paired_comparisons',paired),('summary',dict(metrics=metrics,comparisons=paired,logical_responses=len(data),new_gpu_responses=sum(not x['reused'] for x in costs)))]:m.write(m.OUT/'metrics'/(name+'.json'),value)
    for name,value in [('metrics',metrics),('sampling_diagnostics',sampling),('compute_cost',costs),('length_groups',strata)]:csvwrite(m.OUT/'metrics'/(name+'.csv'),value)
    def name(c):
        if c=='uniform':return '전체 균등'
        p=c.split('_');return ('GT' if p[0]=='gt' else 'Random')+' '+('전체' if p[1]=='whole' else '경계')+' '+p[2][1:]+'배'+(' / 두 seed 평균' if p[-1]=='mean' else (' / seed '+p[-1][1:] if p[-1].startswith('s') else ''))
    def f(x):return '—' if x is None else f'{x:.2f}'
    lines=['# GT 전체와 경계 집중: 배정 비율 2·4·8 비교','', 'pilot 300개 전체, Charades·ActivityNet·QVHighlights 각 100개. Random seed는 42·43만 사용했습니다. 이전 3.5배 실험의 예측은 이번 결과에 포함하지 않았습니다.','',
    '시간 구간 겹침률(IoU)은 예측과 정답이 겹친 길이 ÷ 두 구간의 합집합 길이입니다. 전체 질문의 평균인 mIoU는 높을수록 좋습니다. 파싱 실패와 무효 구간은 0점으로 포함하며 반올림·clipping 없이 평가합니다. IoU≥0.5 비율은 겹침률이 0.5 이상인 질문 비율입니다.','',
    '평균 절대 오차(Mean Absolute Error, MAE)는 |예측 시각−정답 시각|의 평균이며 초 단위, 낮을수록 좋습니다. 시작·종료를 따로 계산하며 유효 응답만 사용합니다. 유효 응답 수는 CSV에 기록했습니다. pp는 백분율 차이로 40%→42%는 +2pp입니다.','',
    '2·4·8배는 GT 전체 조건의 안/밖 목표 밀도 비율입니다. K=round(F×rL/(rL+T−L))로 집중 프레임 수를 정하고, 경계 조건도 같은 K를 시작·끝 ±1초의 합집합에 배치합니다. 따라서 경계 조건의 실제 안/밖 밀도 비율이 2·4·8이라는 뜻은 아닙니다. Random 전체·경계는 같은 seed의 같은 구간을 사용하며 비율 사이에서도 위치를 고정했습니다.','',
    '경계 창은 영상 범위와 교집합을 취하고 겹치는 창은 합칩니다. 후보 부족으로 K를 채울 수 없으면 나머지 영역으로 옮깁니다. 전체 프레임·공간 grid·visual-token 예산은 같지만, 이런 항목은 집중 예산이 다른 예외입니다. GT가 영상 전체인 경우 전체 집중은 Uniform과 같고 경계 집중은 시작·끝을 관측하는 별도 배치입니다.','',
    f"집중 예산의 후보 부족 조정이 발생한 조건·질문 쌍: {sum(x['capacity_transfer']!=0 for x in sampling)}개. 자세한 실제 배정과 시간 오차는 sampling_diagnostics.csv에 기록했습니다.",'',
    '95% 구간은 영상 단위 bootstrap 20,000회(seed42)로 계산했습니다. 같은 영상의 질문을 함께 뽑고 모든 비교에 같은 재표집을 사용했습니다. Random은 질문마다 두 seed를 평균합니다. 여러 비율·조건을 살펴보는 탐색 분석이며 다중비교 보정이나 새 영상 확증은 하지 않았습니다.','',
    '| 벤치마크 | 조건 | mIoU % | IoU≥0.5 % | 시작 오차 평균 초 | 종료 오차 평균 초 |','|---|---|---:|---:|---:|---:|']
    for x in metrics:
        if x['condition'].startswith('random') and not x['condition'].endswith('mean'):continue
        lines.append(f"| {x['dataset']} | {name(x['condition'])} | {100*x['miou']:.2f} | {100*x['r05']:.2f} | {f(x['start_error_seconds'])} | {f(x['end_error_seconds'])} |")
    lines+=['','| 벤치마크 | 비교 | mIoU 차이 pp [95% 구간] | 개선/악화/동일 |','|---|---|---|---|']
    for x in paired:lines.append(f"| {x['dataset']} | {name(x['a'])} − {name(x['b'])} | {x['delta_pp']:+.2f} [{x['lower_pp']:+.2f}, {x['upper_pp']:+.2f}] | {x['improved']}/{x['worsened']}/{x['unchanged']} |")
    lines+=['','해석: GT가 Uniform과 대응 Random 모두보다 좋아야 정확한 위치 정보의 이득을 주장할 근거가 생깁니다. 경계−전체가 양수면 경계 배치가 유리한 방향이며, 구간이 0을 포함하면 불확실합니다. 가장 높은 비율 하나만 골라 확정적 결론을 내리지 않습니다. 후보 부족 조정과 긴 GT 집단은 sampling_diagnostics.csv와 length_groups.csv를 함께 확인해야 합니다.','']
    for ratio in m.RATIOS:
        x=next(x for x in paired if x['dataset']=='all' and x['a']==f'gt_boundary_r{ratio}' and x['b']==f'gt_whole_r{ratio}')
        verdict='경계 집중이 유리한 신호' if x['lower_pp']>0 else ('전체 집중이 유리한 신호' if x['upper_pp']<0 else '두 배치의 우열이 불확실')
        lines.append(f"- {ratio}배 배정: 경계−전체 {x['delta_pp']:+.2f}pp, {verdict}입니다.")
    lines+=['',f"논리적 응답 {len(data)}개, 신규 GPU 생성 {sum(not x['reused'] for x in costs)}개. 동일 질문의 입력이 완전히 같은 경우만 응답을 공유했습니다. GPU 생성 시간 합계 {sum(x['gpu_seconds'] for x in costs):.1f}초.",'', '원본 annotation과 영상 길이 차이 한 건은 GT를 수정하지 않고 실제 컨테이너 재생 길이를 계획에 사용했습니다. 자세한 근거는 validation/annotation_checks.json에 있습니다.','',f'[수치와 질문별 평가](../../artifacts/runs/{m.RUN}/metrics/) · [검증 기록](../../artifacts/runs/{m.RUN}/validation/)']
    write_result_text(m.REPORT/'REPORT.md', '\n'.join(lines)+'\n')
