"""Saved-response evaluation only; never generates or edits model answers."""
from result_layout import write_result_text
from adaptive_temporal_sampling import *
import csv
import numpy as np
from timelens.utils import extract_time,iou

def csvwrite(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def parse(answer):
    spans=extract_time(answer) or [];p=list(spans[0]) if spans else None
    valid=bool(p and all(math.isfinite(x) for x in p) and p[0]<p[1])
    return p,valid,spans

def score(answer,gt,T):
    p,valid,spans=parse(answer)
    v=iou(gt,p) if valid else 0.
    independent=max(0,min(gt[1],p[1])-max(gt[0],p[0]))/(max(gt[1],p[1])-min(gt[0],p[0])) if valid else 0.
    need(abs(v-independent)<1e-12,'Independent evaluator IoU mismatch')
    nonfinite=bool(p and not all(math.isfinite(x) for x in p))
    rounded=[round(x) for x in p] if p and not nonfinite else [T+10,T+20]
    rv=iou(gt,rounded) if not nonfinite and rounded[0]<rounded[1] else 0.
    es=abs(p[0]-gt[0]) if valid else None;ee=abs(p[1]-gt[1]) if valid else None
    safe_parsed=[x if math.isfinite(x) else str(x) for x in p] if p else None
    return dict(iou=float(v),rounded_iou=float(rv),parsed=safe_parsed,valid=valid,parse_failed=not spans,invalid_interval=bool(spans and not valid),nonfinite_interval=nonfinite,multiple_intervals=len(spans)>1,out_of_range=bool(p and (p[0]<0 or p[1]>T)),start_error=es,end_error=ee,normalized_start_error=es/T if valid else None,normalized_end_error=ee/T if valid else None,boundary_within_1s=bool(valid and es<=1 and ee<=1),boundary_within_2s=bool(valid and es<=2 and ee<=2))

def coverage(times,gt,eps=.5):
    times=sorted(times);a,b=gt
    intervals=sorted((max(a,t-eps),min(b,t+eps)) for t in times if t+eps>a and t-eps<b)
    length=0.;last=a
    for lo,hi in intervals:length+=max(0,hi-max(last,lo));last=max(last,hi)
    ds=min((abs(t-a) for t in times),default=None);de=min((abs(t-b) for t in times),default=None)
    return dict(gt_frame_fraction=sum(a<=t<=b for t in times)/len(times) if times else None,temporal_coverage=length/(b-a),start_boundary_coverage=ds is not None and ds<=eps,end_boundary_coverage=de is not None and de<=eps,both_boundary_coverage=ds is not None and ds<=eps and de<=eps,start_min_distance=ds,end_min_distance=de)

def fixture_checks():
    texts=['The event happens in 1.5 - 3.25 seconds','1 - 3','-1 - 2','00:01.5 to 00:03.0','00:00:01 - 00:00:03','1 - 3 and 5 - 8','','4 - 2','0 - 100','unparseable','The event happens in 2 - 4 seconds'+' x'*512,'9'*400+' - 1','2 - 2']
    return [dict(answer=t,**score(t,[1,3],10)) for t in texts]

def build():
    records=read(OUT/'metadata/baseline_records.json');cohort=read(OUT/'metadata/cohort.json');N=len(cohort)
    need(N==100,'Unexpected report cohort')
    config_hash=objhash(read(OUT/'metadata/config.json'));cohort_hash=objhash(cohort)
    by={};paired=[];diagnostics=[];costs=[];allrows=[]
    baseline_by={r['sample']['sample_key']:r for r in records}
    for s in cohort:
        key=s['sample_key'];gt=s['span'][0];T=s['duration'];by[key]={};rec=baseline_by[key]
        for cond in CONDITIONS:
            r=read(OUT/'responses'/cond/(key+'.json'));p=read(OUT/'plans'/cond/(key+'.json'));inp=read(OUT/'inputs'/cond/(key+'.json'));ident=r['identity']
            need(ident['config_hash']==config_hash and ident['cohort_hash']==cohort_hash and ident['plan_hash']==objhash(p),'Report identity mismatch')
            need(ident['input_hash']==objhash(inp['hashes']) and r['input_detail_hash']==objhash(inp),'Report input hash mismatch')
            need(r['F']==rec['F'] and r['B']==rec['B'] and r['grid']==rec['grid'] and inp['frame_ids']==p['frame_ids'],'Report budget/plan mismatch')
            val=score(r['answer'],gt,T);by[key][cond]=val
            allrows.append(dict(sample_key=key,video_id=rec['video_id'],condition=cond,**{k:v for k,v in val.items() if k!='parsed'},generated_tokens=r['generated_tokens'],eos_reached=r['eos_reached']))
            ts=p['pts'];focusids=set(p['focus']+p['fill']);focus=[t for i,t in zip(p['frame_ids'],ts) if i in focusids];anchorids=set(p['anchors']);ats=[t for i,t in zip(p['frame_ids'],ts) if i in anchorids]
            gaps=np.diff(ts)
            diagnostics.append(dict(sample_key=key,video_id=rec['video_id'],condition=cond,F=r['F'],B=r['B'],grid=str(r['grid']),unique_frames=len(set(p['frame_ids'])),padding=inp['padding'],fill_fraction=p['fill_fraction'],pred_fallback=p['pred_fallback'],center_clamped=p['center_clamped'],interval_min=float(min(gaps)),interval_median=float(np.median(gaps)),interval_max=float(max(gaps)),window_frames=sum(p['window'][0]<=t<=p['window'][1] for t in ts) if p['window'] else None,**coverage(ts,gt),**{'focus_'+k:v for k,v in coverage(focus,gt).items()},**{'anchor_'+k:v for k,v in coverage(ats,gt).items()}))
            costs.append(dict(sample_key=key,condition=cond,stage='final',F=r['F'],B=r['B'],policy_F=2*r['F'],policy_B=2*r['B'],single_pass_uniform_F=r['F'],single_pass_uniform_B=r['B'],generation_seconds=r['generation_seconds'],prefill_seconds=r.get('prefill_seconds'),decode_seconds=r.get('decode_seconds'),peak_allocated_bytes=r['peak_allocated_bytes'],generated_tokens=r['generated_tokens'],input_tokens=r['total_input_tokens'],reused=False,input_prepare_seconds=r.get('input_prepare_seconds'),peak_reserved_bytes=r['peak_reserved_bytes'],cache_state='warm immutable candidates'))
        scout=read(OUT/'scouts'/(key+'.json'));need(scout['generated_ids']==read(OUT/'responses/uniform'/(key+'.json'))['generated_ids'],'Repeat Uniform mismatch')
        costs.append(dict(sample_key=key,condition='shared',stage='scout',F=rec['F'],B=rec['B'],policy_F=None,policy_B=None,single_pass_uniform_F=None,single_pass_uniform_B=None,generation_seconds=scout['generation_seconds'],prefill_seconds=scout['prefill_seconds'],decode_seconds=scout['decode_seconds'],peak_allocated_bytes=scout['peak_allocated_bytes'],generated_tokens=len(scout['generated_ids']),input_tokens=scout['total_input_tokens'],reused=False,input_prepare_seconds=scout['input_prepare_seconds'],peak_reserved_bytes=scout['peak_reserved_bytes'],cache_state='fresh from immutable uniform RGB'))
        cache=read(OUT/'frames'/(key+'.json'))
        costs.append(dict(sample_key=key,condition='shared',stage='cpu_candidates_and_uniform_bridge',F=None,B=None,policy_F=None,policy_B=None,single_pass_uniform_F=None,single_pass_uniform_B=None,generation_seconds=cache['seconds'],prefill_seconds=None,decode_seconds=None,peak_allocated_bytes=None,generated_tokens=None,input_tokens=None,reused=False,input_prepare_seconds=None,peak_reserved_bytes=None,cache_state=cache['cache_state']))
    keys=[s['sample_key'] for s in cohort];videos=[baseline_by[k]['video_id'] for k in keys];unique=sorted(set(videos))
    vals={c:np.array([by[k][c]['iou'] for k in keys]) for c in CONDITIONS}
    vals['random_mean']=np.mean([vals[c] for c in ['random_s42','random_s43','random_s44']],axis=0)
    contrasts=[('gt','uniform'),('pred','uniform'),('random_s42','uniform'),('gt','random_s42'),('pred','random_s42'),('random_s43','uniform'),('random_s44','uniform'),('random_mean','uniform'),('gt','random_mean'),('pred','random_mean')]
    # Draw video clusters and retain query weighting in each replicate. The same draws serve every contrast.
    mat=np.array([[1. if v==u else 0. for u in unique] for v in videos]);rng=np.random.default_rng(42)
    draws=rng.integers(0,len(unique),size=(20000,len(unique)))
    cluster_counts=np.zeros((20000,len(unique)),dtype=np.int16)
    for j in range(len(unique)):cluster_counts[:,j]=(draws==j).sum(axis=1)
    weights=cluster_counts@mat.T;denom=weights.sum(axis=1)
    comparisons={}
    for a,b in contrasts:
        d=vals[a]-vals[b];rep=weights@d/denom;lo,hi=np.quantile(rep,[.025,.975]);name=a+'-'+b
        comparisons[name]=dict(delta_miou_pp=float(100*d.mean()),ci95_pp=[float(100*lo),float(100*hi)],improved=int((d>0).sum()),worsened=int((d<0).sum()),equal=int((d==0).sum()))
        for ix,k in enumerate(keys):
            va=by[k].get(a);vb=by[k].get(b);both=va and vb and va['valid'] and vb['valid']
            paired.append(dict(sample_key=k,video_id=videos[ix],comparison=name,delta_iou=float(d[ix]),iou_a=float(vals[a][ix]),iou_b=float(vals[b][ix]),failure_to_success=bool(vals[b][ix]<.5 and vals[a][ix]>=.5),success_to_failure=bool(vals[b][ix]>=.5 and vals[a][ix]<.5),both_valid=bool(both),start_error_delta=va['start_error']-vb['start_error'] if both else None,end_error_delta=va['end_error']-vb['end_error'] if both else None,normalized_start_error_delta=va['normalized_start_error']-vb['normalized_start_error'] if both else None,normalized_end_error_delta=va['normalized_end_error']-vb['normalized_end_error'] if both else None,same_prediction=va['parsed']==vb['parsed'] if va and vb else None))
        comparisons[name].update(failure_to_success=sum(r['failure_to_success'] for r in paired if r['comparison']==name) if 'random_mean' not in [a,b] else None,success_to_failure=sum(r['success_to_failure'] for r in paired if r['comparison']==name) if 'random_mean' not in [a,b] else None,paired_boundary_n=sum(r['both_valid'] for r in paired if r['comparison']==name))
        qa=np.mean([(vals[c]>=.5).astype(float) for c in ['random_s42','random_s43','random_s44']],axis=0) if a=='random_mean' else (vals[a]>=.5).astype(float)
        qb=np.mean([(vals[c]>=.5).astype(float) for c in ['random_s42','random_s43','random_s44']],axis=0) if b=='random_mean' else (vals[b]>=.5).astype(float)
        rd=qa-qb;rrep=weights@rd/denom;rlo,rhi=np.quantile(rrep,[.025,.975]);comparisons[name].update(delta_r05_pp=float(100*rd.mean()),r05_ci95_pp=[float(100*rlo),float(100*rhi)])
    metrics={}
    for c in CONDITIONS:
        rr=[r for r in allrows if r['condition']==c];metrics[c]=dict(n=100,miou=float(vals[c].mean()),r03=float((vals[c]>=.3).mean()),r05=float((vals[c]>=.5).mean()),r07=float((vals[c]>=.7).mean()),rounded_miou=float(np.mean([r['rounded_iou'] for r in rr])),rounded_r03=float(np.mean([r['rounded_iou']>=.3 for r in rr])),rounded_r05=float(np.mean([r['rounded_iou']>=.5 for r in rr])),rounded_r07=float(np.mean([r['rounded_iou']>=.7 for r in rr])))
        for name in ['parse_failed','invalid_interval','nonfinite_interval','multiple_intervals','out_of_range','boundary_within_1s','boundary_within_2s']:metrics[c][name]=sum(r[name] for r in rr)
        metrics[c]['eos_not_reached']=sum(not r['eos_reached'] for r in rr)
        for name in ['start_error','end_error','normalized_start_error','normalized_end_error']:
            vv=[r[name] for r in rr if r[name] is not None];metrics[c][name]=dict(n=len(vv),mean=float(np.mean(vv)) if vv else None,median=float(np.median(vv)) if vv else None,p90=float(np.quantile(vv,.9)) if vv else None)
    # Random-mean recalls average each query's three success indicators, rather than thresholding mean IoU.
    metrics['random_mean']=dict(n=100,miou=float(vals['random_mean'].mean()),r03=float(np.mean([metrics[c]['r03'] for c in ['random_s42','random_s43','random_s44']])),r05=float(np.mean([metrics[c]['r05'] for c in ['random_s42','random_s43','random_s44']])),r07=float(np.mean([metrics[c]['r07'] for c in ['random_s42','random_s43','random_s44']])))
    gt=comparisons['gt-uniform'];pred=comparisons['pred-uniform']
    oracle=gt['delta_miou_pp']>=3 and gt['ci95_pp'][0]>0 and comparisons['gt-random_s42']['delta_miou_pp']>0
    practical=pred['delta_miou_pp']>=2 and pred['ci95_pp'][0]>0 and comparisons['pred-random_s42']['delta_miou_pp']>0 and metrics['pred']['r05']>=metrics['uniform']['r05']-.01-1e-10
    decision=dict(oracle_signal=oracle,pred_signal=practical,oracle='강한 위치 oracle 신호' if oracle else ('v1 확장 보류: CI 상한 < +3pp' if gt['ci95_pp'][1]<3 else '불확실'),pred='후속 별도 cohort 진행 기준 충족' if practical else '불확실 또는 사전 진행 기준 미충족',random_mean_direction_gt=comparisons['gt-random_mean']['delta_miou_pp']>0,random_mean_direction_pred=comparisons['pred-random_mean']['delta_miou_pp']>0)
    total_seconds=sum(c['generation_seconds'] for c in costs if c['stage']=='final');cpu_seconds=sum(c['generation_seconds'] for c in costs if c['stage']=='cpu_candidates_and_uniform_bridge')
    planning_seconds=sum(x['seconds'] for x in read(OUT/'metrics/planning_cost.json'))
    input_prepare_seconds=sum(c['input_prepare_seconds'] for c in costs if c['stage']=='final')
    frozen=read(OUT/'metadata/source_manifest.json');verified=[]
    for source in frozen['files']+frozen['videos']:
        need(sha(source['path'])==source['sha256'],'Frozen source changed: '+source['path']);verified.append(source['path'])
    for rec in records:
        entry=rec['baseline_entry']
        from result_layout import relocate_path
        for p,h in [(relocate_path(ROOT/entry['response_path']),entry['response_sha256']),(rec['eight_input']['path'],rec['eight_input']['sha256']),(rec['eight_response']['path'],rec['eight_response']['sha256'])]:need(sha(p)==h,'Historical input/response changed: '+str(p))
    for f in (OUT/'frames').glob('*.json'):
        c=read(f);need(sha(c['path'])==c['sha256'],'Candidate cache hash changed')
    write(OUT/'validation/source_preservation.json',dict(passed=True,verified_sources=verified,historical_responses_and_inputs=300,candidate_cache_records=100))
    scout_seconds=sum(c['generation_seconds'] for c in costs if c['stage']=='scout')
    summary=dict(state='evaluated',n_query=100,n_video=len(unique),new_finals=600,new_scouts=100,reused_scouts=0,logical_responses=700,metrics=metrics,comparisons=comparisons,decision=decision,cost=dict(final_gpu_wall_seconds=total_seconds,cpu_candidate_and_bridge_seconds=cpu_seconds,cpu_planning_seconds=planning_seconds,cpu_final_input_seconds=input_prepare_seconds,scout_new_gpu_seconds=scout_seconds,final_generated_tokens=sum(r['generated_tokens'] for r in allrows)),budget_equality=True,uniform_repeat_mismatches=0)
    csvwrite(OUT/'metrics/metrics.csv',[dict(condition=c,**{k:v for k,v in m.items() if not isinstance(v,dict)}) for c,m in metrics.items() if c!='random_mean'])
    csvwrite(OUT/'metrics/paired_samples.csv',paired);csvwrite(OUT/'metrics/per_response.csv',allrows);csvwrite(OUT/'metrics/sampling_diagnostics.csv',diagnostics);csvwrite(OUT/'metrics/compute_cost.csv',costs)
    write(OUT/'metrics/summary.json',summary);write(OUT/'metrics/bootstrap.json',dict(unit='video ID',replicates=20000,seed=42,interval='95% percentile',n_query=100,n_video=len(unique),query_weighted_within_replicate=True,shared_draws_all_conditions=True,comparisons=comparisons,random='per-query mean over three seeds; not 300 independent queries'))
    write(OUT/'validation/evaluation_fixtures.json',fixture_checks())
    lines=['# VTG Adaptive Temporal Sampling Pilot v1','',f'평가 완료: {stamp()}. Charades-TimeLens **100 query / {len(unique)} videos**. 신규 공통 scout 100개, 신규 final 600개, 신규·논리적 응답 모두 700개. 최종 검증·registry 상태는 실행 status를 참조한다.','',
        'Qwen3-VL-8B-Instruct full-head, video_query, greedy, max512, repetition_penalty=1, seed42, BF16·FlashAttention2. anchor 50%, focus window 0.4T. 각 query의 기존 2fps F/B/grid를 모든 조건에 고정했다. GT-guided는 진단용 위치 oracle reference이며 실용 방법 또는 최적 상한이 아니다.','',
        '주평가: 공용 TimeLens extract_time의 첫 구간, raw seconds, 반올림·clipping 없음. tIoU=교집합 길이/합집합 길이, 파싱 실패·비유한·역전·길이0은 0점. mIoU와 R@.3/.5/.7은 전체 100개 기준이며 높을수록 좋다. R@t는 IoU≥t인 비율이다.','',
        '| 조건 | N | mIoU % | R@.3 % | R@.5 % | R@.7 % |','|---|---:|---:|---:|---:|---:|']
    for c,m in metrics.items():lines.append(f"| {c} | 100 | {m['miou']*100:.2f} | {m['r03']*100:.2f} | {m['r05']*100:.2f} | {m['r07']*100:.2f} |")
    lines+=['','| 대응 비교 (앞−뒤) | ΔmIoU pp | video-cluster 95% CI pp | 개선/악화/동일 | .5 실패→성공/성공→실패 |','|---|---:|---|---|---|']
    for c,m in comparisons.items():lines.append(f"| {c} | {m['delta_miou_pp']:+.2f} | [{m['ci95_pp'][0]:+.2f}, {m['ci95_pp'][1]:+.2f}] | {m['improved']}/{m['worsened']}/{m['equal']} | {m['failure_to_success']}/{m['success_to_failure']} |")
    lines+=['','Video ID cluster bootstrap 20,000회, seed42, 95% percentile CI. 각 재표집은 같은 video의 query들을 함께 포함하고 query 가중 평균을 다시 계산한다. Random 평균은 query별 세 seed 평균이며 300개 독립 query로 취급하지 않았다. 탐색적 CI이고 다중비교 보정·생성 seed 불확실성·새 영상/모델 일반화를 포함하지 않는다. CI가 0을 포함하면 효과는 불확실하다.','',
        '| 조건 | rounded mIoU % | rounded R@.3/.5/.7 % | 파싱/무효/범위초과/EOS미도달 | 양경계≤1s/2s (전체100) |','|---|---:|---|---|---|']
    for c in CONDITIONS:
        m=metrics[c];lines.append(f"| {c} | {m['rounded_miou']*100:.2f} | {m['rounded_r03']*100:.1f}/{m['rounded_r05']*100:.1f}/{m['rounded_r07']*100:.1f} | {m['parse_failed']}/{m['invalid_interval']}/{m['out_of_range']}/{m['eos_not_reached']} | {m['boundary_within_1s']}/{m['boundary_within_2s']} |")
    lines+=['','호환성 평가는 같은 parser + Python round(ties-to-even), 파싱 실패 시 [T+10,T+20] fallback을 적용한다. GT는 원래 값을 유지한다. raw 주지표와 섞지 않는다.','',
        '| 조건 | start MAE/중앙값/P90 s (유효 N) | end MAE/중앙값/P90 s | 정규화 start/end MAE | GT frame fraction | temporal coverage | start/end/both coverage | fill/fallback |','|---|---|---|---|---:|---:|---|---|']
    for c in CONDITIONS:
        m=metrics[c];d=[r for r in diagnostics if r['condition']==c];a=m['start_error'];b=m['end_error']
        lines.append(f"| {c} | {a['mean']:.3f}/{a['median']:.3f}/{a['p90']:.3f} ({a['n']}) | {b['mean']:.3f}/{b['median']:.3f}/{b['p90']:.3f} | {m['normalized_start_error']['mean']:.4f}/{m['normalized_end_error']['mean']:.4f} | {np.mean([r['gt_frame_fraction'] for r in d]):.3f} | {np.mean([r['temporal_coverage'] for r in d]):.3f} | {np.mean([r['start_boundary_coverage'] for r in d]):.2f}/{np.mean([r['end_boundary_coverage'] for r in d]):.2f}/{np.mean([r['both_boundary_coverage'] for r in d]):.2f} | {np.mean([r['fill_fraction'] for r in d]):.3f}/{sum(r['pred_fallback'] for r in d)} |")
    lines+=['','절대 경계오차=|예측−GT| (초), 정규화 오차=절대오차/T이며 낮을수록 좋다. 무효 예측은 경계오차 평균에서 제외하되 유효 N을 표기하며 양경계 within 지표에서는 실패(false)로 포함한다. paired 경계차이는 두 조건 모두 유효한 쌍만 사용하고 N을 CSV에 보존했다. GT frame fraction=GT 안의 유일 관측수/전체 유일 관측수. temporal coverage=GT∩union([t−0.5,t+0.5]) 길이/GT 길이, boundary coverage=각 GT endpoint와 관측의 최소거리≤0.5초이며 높을수록 좋다. 전체/focus/anchor별 수치는 sampling_diagnostics.csv에 보존했다. coverage 개선만으로 모델 이해 또는 decoding의 인과 원인을 확정하지 않는다.','',
        f'Final GPU wall time 합계 {total_seconds:.1f}초; 공통 scout GPU wall time {scout_seconds:.1f}초; CPU 후보 준비+Uniform bridge {cpu_seconds:.1f}초. final 생성 토큰 {summary["cost"]["final_generated_tokens"]:,}개. GPU prefill/decode 및 peak VRAM은 compute_cost.csv에 있다. CPU preparation은 cold decoding·resize·bridge를 포함하며 순수 decode만의 시간과 혼동하지 않는다.','',
        '모든 final은 F_i/B_i, 동일 grid이며 scout+final 정책은 조건당 2F_i/2B_i, 생성 상한1024다. 실제 계산에서 scout는 1회만 공유했다. 자연스러운 single-pass Uniform은 F_i/B_i, max512다. Pred는 이 single-pass와 동일한 전체 추론 비용이 아니다. GT annotation 취득 비용은 계측하지 않았다. timestamp/특수/텍스트 토큰은 visual B와 별도로 inputs에 기록했다.','',
        f'사전 판단: 위치 oracle={decision["oracle"]}; Pred={decision["pred"]}. GT 기준은 Δ≥+3pp, CI 하한>0, GT−Random42>0. Pred 기준은 Δ≥+2pp, CI 하한>0, Pred−Random42>0, R@.5 하락≤1pp다. 폭·예산·임계값을 중간 성능으로 조정하지 않았다. 이미 연구에 사용한 pilot이므로 미관측 test로 간주하지 않는다.','',
        '역사적 비교는 새 equal-budget 조건과 구분한다. 같은100개 raw mIoU: 기존2fps 46.90%, 기존8fps 42.81%; R@.5 48%,45%. 과거 rounded 47.53%/49%와 평가 규칙이 다르다. 서버 폴더 정리로 공용 경로는 _shared/timelens, registry는 artifacts/common/registry다. GitHub main은 기준 a0824288d3fab917db04d016eeb32d32e170f3c1과 동일했다.','',
        '입력 검증: 100개 fresh prepare_pair와 Uniform adapter의 pixels/IDs/grid/attention/rotary positions 일치; 원본 CFR 시각과 native indices/fps가 Decord float32 PTS 정밀도에서 일치; 같은 source RGB는 공유 immutable cache와 독립 baseline seek를 비교했다. native timestamp는 두 프레임 평균을 0.1초 텍스트로 표현하는 기존 계약을 그대로 따른다. 모든 Uniform final 생성 IDs는 검증된 scout와 일치했다. 계획은 첫 final 전에 고정하고 GT 없는 public schema로 Uniform/Random/Pred를 생성했다.','',
        f'[상태](../../artifacts/runs/{RUN}/metadata/status.json) · [사전 검증](../../artifacts/runs/{RUN}/metadata/preflight.json) · [전량 검증](../../artifacts/runs/{RUN}/validation/final.json) · [summary](../../artifacts/runs/{RUN}/metrics/summary.json) · [paired samples](../../artifacts/runs/{RUN}/metrics/paired_samples.csv) · [sampling](../../artifacts/runs/{RUN}/metrics/sampling_diagnostics.csv) · [cost](../../artifacts/runs/{RUN}/metrics/compute_cost.csv) · [코드](../../_lib/adaptive_temporal_sampling.py)']
    lines += ['',f'CPU 계획 {planning_seconds:.3f}초, final 입력 준비 {input_prepare_seconds:.1f}초. GPU forward CUDA-event prefill/decode 시간과 응답 전체 wall time을 구분한다. cold/warm 후보 준비 표시는 비용 CSV에 보존했다. 과거 입력에는 픽셀 hash가 없어 역사적 응답을 scout로 재사용하지 않고 100개 모두 새로 생성했다.','',
        f'실행 코드: `_lib/adaptive_temporal_sampling.py`; 평가 코드: `_lib/adaptive_temporal_report.py`. 실행 명령: `.venv/bin/python -u _lib/adaptive_temporal_sampling.py run`; 저장 응답 재평가: `.venv/bin/python _lib/adaptive_temporal_report.py`. config SHA-256 `{config_hash}`, cohort canonical SHA-256 `{cohort_hash}`. model/processor/evaluator/source SHA-256와 환경 lock은 metadata/source_manifest.json 및 environment/에 있다.']
    REPORT.mkdir(parents=True,exist_ok=True);write_result_text(REPORT/'REPORT.md', '\n'.join(lines)+'\n')
    output_files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and not any(part in ['logs'] for part in p.relative_to(OUT).parts) and p.name not in ['status.json','run.lock','output_manifest.json']:output_files.append(fileinfo(p))
    write(OUT/'validation/output_manifest.json',dict(files=output_files,report=fileinfo(REPORT/'REPORT.md'),mutable_excluded=['logs','status.json','run.lock','output_manifest.json']))
    return summary

if __name__=='__main__':build()
