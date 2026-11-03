"""Pilot300: uniform versus whole-GT interval density versus length-matched random.

No anchors, no boundary focus, no Pred. Two planner seeds: 42 and 43.
"""
from result_layout import write_result_text
from result_layout import relocate_path
import sys,os,json,time,math,copy,hashlib,random,bisect,threading,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import boundary_oracle_inputs as inputs
import adaptive_temporal_sampling as old
from adaptive_temporal_sampling import read,write,immutable,sha,objhash,fileinfo,need,stamp
RUN='gt_interval_density_pilot300_v1'
ROOT=old.ROOT;OUT=relocate_path(ROOT/'artifacts/runs'/RUN);REPORT=relocate_path(ROOT/'results'/RUN)
inputs.RUN=RUN;inputs.OUT=OUT;inputs.REPORT=REPORT
CONDITIONS=['uniform','gt','random_s42','random_s43']
CONFIG=dict(run_id=RUN,cohort='existing pilot_300, 100 per benchmark',density_ratio=3.5,random_seeds=[42,43],model='Qwen3-VL-8B-Instruct',revision=old.CONFIG['revision'],do_sample=False,max_new_tokens=512,repetition_penalty=1.,generation_seed=42,dtype='BF16',backend='FlashAttention2',order='video_query',frame_budget='original 2fps actual F; original H/W and actual visual tokens',rounding='Python round, ties to even',inside_boundary='s <= PTS <= e',outside_axis='concatenate [0,s) and (e,T]; bin centers in concatenated time',source_selection='ordered nearest distinct source frame, ties lower source ID, retain enough candidates for remaining targets',min_allocation='one per nonempty side if F>=2; then clamp to source capacity and transfer remainder',whole_video='GT and Random use exactly the Uniform frame list',GT_usage='GT location only in GT planner/evaluator; Random is diagnostic and uses GT LENGTH only',uniform='F equal-duration cells over [0,T], nearest source frame to each center',bootstrap_replicates=20000,bootstrap_seed=42,full_oracle_ratio=.8,inference='single pass; identical inputs within a query may reuse an earlier deterministic response, explicitly recorded')

def choose(candidates,targets):
    """Candidates (source ID, time coordinate), sorted in source/time order."""
    need(len(candidates)>=len(targets),'Insufficient distinct source frames')
    coords=[x[1] for x in candidates];selected=[];errors=[];last=-1
    for j,target in enumerate(targets):
        lo=last+1;hi=len(candidates)-(len(targets)-j);at=bisect.bisect_left(coords,target,lo,hi+1)
        options={max(lo,min(hi,at)),max(lo,min(hi,at-1))};last=min(options,key=lambda k:(abs(coords[k]-target),candidates[k][0]))
        selected.append(candidates[last][0]);errors.append(abs(coords[last]-target))
    return selected,errors

def make_plan(public,condition,gt=None,length=None):
    need(set(public)=={'sample_key','T','F','ids','pts'},'Planner schema must not contain GT')
    key,T,F=public['sample_key'],public['T'],public['F'];mapping=dict(zip(public['ids'],public['pts']));allc=[(i,t) for i,t in zip(public['ids'],public['pts']) if 0<=t<=T]
    uniform_targets=[(j+.5)*T/F for j in range(F)];uniform,ue=choose(allc,uniform_targets)
    if condition=='uniform':return dict(sample_key=key,condition=condition,frame_ids=uniform,pts=[mapping[i] for i in uniform],F=F,T=T,target_times=uniform_targets,target_time_absolute_deviations=ue,interval=None)
    if condition=='gt':
        need(gt is not None,'GT interval missing');s,e=gt;L=e-s
    else:
        need(gt is None and length is not None,'Random may use GT length but not its location');L=length
        rng=random.Random(int.from_bytes(hashlib.sha256(f'{RUN}|{key}|{condition[-2:]}'.encode()).digest()[:8],'big'));s=rng.random()*(T-L);e=s+L
    need(0<=s<e<=T+1e-10 and 0<L<=T,'Invalid interval; do not silently clip GT')
    if abs(L-T)<=1e-10:
        return dict(sample_key=key,condition=condition,frame_ids=uniform,pts=[mapping[i] for i in uniform],F=F,T=T,interval=[0.,T],length=L,inside_requested=F,inside_actual=F,outside_actual=0,minimum_adjustment=0,capacity_transfer=0,full_video=True,target_time_absolute_deviations=ue,inside_target_interval=T/F,outside_target_interval=None,density_ratio_actual=None)
    inside=[(i,t-s) for i,t in allc if s<=t<=e]
    outside=[(i,t if t<s else s+(t-e)) for i,t in allc if t<s or t>e]
    requested=round(F*3.5*L/(3.5*L+(T-L)));minimum=requested
    if inside and outside and F>=2:minimum=max(1,min(F-1,minimum))
    elif inside and not outside:minimum=F
    elif outside and not inside:minimum=0
    lo=max(0,F-len(outside));hi=min(F,len(inside));need(lo<=hi,'Total frame budget exceeds candidate source frames')
    n_in=max(lo,min(hi,minimum));n_out=F-n_in
    ti=[(j+.5)*L/n_in for j in range(n_in)] if n_in else []
    to=[(j+.5)*(T-L)/n_out for j in range(n_out)] if n_out else []
    selected_i,ei=choose(inside,ti);selected_o,eo=choose(outside,to);ids=sorted(selected_i+selected_o)
    need(len(ids)==F and len(set(ids))==F,'Budget or source-ID uniqueness failure')
    def spacing(selected,coords,step):
        if len(selected)<2:return None
        c=dict(coords);actual=[c[i] for i in selected];return sum(abs(b-a-step) for a,b in zip(actual,actual[1:]))/(len(actual)-1)
    return dict(sample_key=key,condition=condition,frame_ids=ids,pts=[mapping[i] for i in ids],F=F,T=T,interval=[s,e],length=L,inside_requested=requested,inside_actual=n_in,outside_actual=n_out,inside_candidates=len(inside),outside_candidates=len(outside),minimum_adjustment=minimum-requested,capacity_transfer=n_in-minimum,full_video=False,inside_target_interval=L/n_in if n_in else None,outside_target_interval=(T-L)/n_out if n_out else None,density_ratio_actual=(n_in/L)/(n_out/(T-L)) if n_in and n_out else None,inside_target_time_absolute_deviations=ei,outside_target_time_absolute_deviations=eo,inside_spacing_absolute_deviation=spacing(selected_i,inside,L/n_in) if n_in else None,outside_spacing_absolute_deviation=spacing(selected_o,outside,(T-L)/n_out) if n_out else None,outside_before_count=sum(mapping[i]<s for i in selected_o),outside_after_count=sum(mapping[i]>e for i in selected_o),outside_target_coordinates=to,inside_target_coordinates=ti)

def checks():
    examples=[]
    for frac,expected in [(.1,17),(.4,42),(.8,56),(1.,60)]:
        p=dict(sample_key='fixture',T=100.,F=60,ids=list(range(10001)),pts=[i/100 for i in range(10001)]);r=make_plan(p,'gt',gt=[0.,100*frac]);need(r['inside_actual']==expected,'Allocation formula example failed');examples.append(r)
    p=dict(sample_key='30second',T=30.,F=60,ids=list(range(901)),pts=[i/30 for i in range(901)]);r=make_plan(p,'gt',gt=[6.,18.]);need(r['inside_actual']==42 and r['outside_before_count']==6 and r['outside_after_count']==12,'Outside concatenation example failed')
    for interval in [[0,5],[25,30],[0,30],[10,10.01]]:
        r=make_plan(p,'gt',gt=interval);need(len(r['frame_ids'])==60 and len(set(r['frame_ids']))==60,'Edge fixture failed')
    u=make_plan(p,'uniform');g=make_plan(p,'gt',gt=[0,30]);a=make_plan(p,'random_s42',length=30);need(u['frame_ids']==g['frame_ids']==a['frame_ids'],'Whole-video equality failed')
    return dict(passed=True,allocation_examples=[dict(length_fraction=r['length']/r['T'],inside=r['inside_actual'],outside=r['outside_actual']) for r in examples],outside_example='30s, GT6..18, F60: 42 in, 6 before, 12 after; passed')

class Progress:
    def __init__(self):self.lock=threading.RLock();self.start=time.monotonic();self.stop=threading.Event();self.events=[];self.state=dict(state='building',stage='starting',completed=0,total=1200,new_gpu_responses=0,reused_identical_inputs=0,counts={})
    def update(self,**kw):
        with self.lock:self.state.update(kw);self.emit()
    def emit(self):
        with self.lock:
            n=self.state['completed'];eta=(1200-n)*(self.events[-1][0]-self.events[0][0])/(self.events[-1][1]-self.events[0][1]) if len(self.events)>1 else None
            self.state.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start,percentage=n/12,eta_seconds=eta,eta_scope='remaining current 1200-response generation; excludes input preparation and report')
            write(OUT/'metadata/status.json',self.state)
            with (OUT/'logs/progress.log').open('a') as f:f.write(json.dumps(self.state,ensure_ascii=False)+'\n')
            print(f"{self.state['timestamp']} {self.state['stage']} {n}/1200 {self.state.get('sample_key','')}",flush=True)
    def done(self,dataset,condition,reused):
        with self.lock:
            self.state['completed']+=1;key=dataset+'/'+condition;self.state['counts'][key]=self.state['counts'].get(key,0)+1
            field='reused_identical_inputs' if reused else 'new_gpu_responses';self.state[field]+=1;self.events.append((time.monotonic(),self.state['completed']));self.events=self.events[-100:];self.emit()
    def heartbeat(self):
        while not self.stop.wait(20):self.emit()

def evaluate(rows,cache_index):
    import numpy as np
    from adaptive_temporal_report import score,csvwrite
    from boundary_oracle_analysis import bootstrap_weights,estimate
    evaluations=[];sampling=[];costs=[];metrics=[];paired=[]
    for record in rows:
        sample=record['sample'];key=sample['sample_key'];gt=sample['span'][0];T=record['planning_T'];L=gt[1]-gt[0]
        for condition in CONDITIONS:
            response=read(OUT/'responses'/condition/(key+'.json'));p=read(OUT/'plans'/condition/(key+'.json'));detail=read(OUT/'inputs'/condition/(key+'.json'))
            need(response['identity']['plan_hash']==objhash(p) and response['identity']['input_hash']==objhash(detail['hashes']),'Saved plan/input identity mismatch')
            v=score(response['answer'],gt,sample['duration']);evaluations.append(dict(dataset=sample['dataset'],sample_key=key,video_id=record['video_id'],condition=condition,length_fraction=L/T,**v))
            inside=p['interval'];overlap=max(0,min(gt[1],inside[1])-max(gt[0],inside[0])) if inside else None
            sampling.append(dict(dataset=sample['dataset'],sample_key=key,condition=condition,length_fraction=L/T,weak_location_identifiability=L/T>=.8,random_gt_interval_iou=overlap/(2*L-overlap) if overlap is not None and condition.startswith('random') else None,random_overlap_fraction=overlap/L if overlap is not None and condition.startswith('random') else None,F=response['F'],B=response['B'],grid=str(response['grid']),inside_requested=p.get('inside_requested'),inside_actual=p.get('inside_actual'),outside_actual=p.get('outside_actual'),minimum_adjustment=p.get('minimum_adjustment',0),capacity_transfer=p.get('capacity_transfer',0),inside_spacing_goal=p.get('inside_target_interval'),outside_spacing_goal=p.get('outside_target_interval'),inside_spacing_absolute_deviation=p.get('inside_spacing_absolute_deviation'),outside_spacing_absolute_deviation=p.get('outside_spacing_absolute_deviation'),inside_mean_target_time_deviation=float(np.mean(p.get('inside_target_time_absolute_deviations',p.get('target_time_absolute_deviations',[])))) if p.get('inside_target_time_absolute_deviations',p.get('target_time_absolute_deviations',[])) else None,outside_mean_target_time_deviation=float(np.mean(p['outside_target_time_absolute_deviations'])) if p.get('outside_target_time_absolute_deviations') else None,density_ratio_actual=p.get('density_ratio_actual')))
            costs.append(dict(dataset=sample['dataset'],sample_key=key,condition=condition,gpu_seconds=response['generation_seconds'],prefill_seconds=response['prefill_seconds'],decode_seconds=response['decode_seconds'],input_seconds=response['input_prepare_seconds'],peak_allocated_GiB=response['peak_allocated_bytes']/2**30,generated_tokens=response['generated_tokens'],reused=response.get('reused_identical_input',False),F=response['F'],B=response['B']))
    labels=sorted({r['sample']['dataset'] for r in rows})+['all'];length_rows=[]
    for label in labels:
        rr=[r for r in rows if label=='all' or r['sample']['dataset']==label];keys=[r['sample']['sample_key'] for r in rr];videos=[r['video_id'] for r in rr];weights=bootstrap_weights(videos);by={(r['sample_key'],r['condition']):r for r in evaluations if r['sample_key'] in set(keys)}
        vals={c:np.array([by[(k,c)]['iou'] for k in keys]) for c in CONDITIONS};vals['random_mean']=(vals['random_s42']+vals['random_s43'])/2
        for condition in CONDITIONS+['random_mean']:
            er=[by[(k,c)] for k in keys for c in (['random_s42','random_s43'] if condition=='random_mean' else [condition])];valid=[r for r in er if r['valid']]
            metrics.append(dict(dataset=label,condition=condition,N=len(keys),videos=len(set(videos)),miou=float(vals[condition].mean()),r03=float(np.mean([r['iou']>=.3 for r in er])),r05=float(np.mean([r['iou']>=.5 for r in er])),r07=float(np.mean([r['iou']>=.7 for r in er])),start_mean_absolute_error=float(np.mean([r['start_error'] for r in valid])) if valid else None,end_mean_absolute_error=float(np.mean([r['end_error'] for r in valid])) if valid else None,valid_boundary_responses=len(valid),parse_failures=sum(r['parse_failed'] for r in er),invalid_intervals=sum(r['invalid_interval'] for r in er),out_of_range=sum(r['out_of_range'] for r in er)))
        for a,b in [('gt','uniform'),('gt','random_mean'),('gt','random_s42'),('gt','random_s43'),('random_mean','uniform')]:
            d=vals[a]-vals[b];est=estimate(d,weights);row=dict(dataset=label,comparison=a+'-'+b,N=len(keys),n_video=len(set(videos)),delta_miou_pp=100*est['mean'],ci95_lower_pp=100*est['ci95'][0],ci95_upper_pp=100*est['ci95'][1],improved=int((d>0).sum()),worsened=int((d<0).sum()),unchanged=int((d==0).sum()))
            for endpoint in ['start','end']:
                def err(k,c):
                    cc=['random_s42','random_s43'] if c=='random_mean' else [c];v=[by[(k,x)][endpoint+'_error'] for x in cc];return float(np.mean(v)) if all(x is not None for x in v) else np.nan
                reduction=np.array([err(k,b)-err(k,a) for k in keys]);e=estimate(reduction,weights) if np.isfinite(reduction).any() else dict(mean=None,ci95=[None,None]);row[endpoint+'_error_reduction_seconds']=e['mean'];row[endpoint+'_reduction_ci95_lower']=e['ci95'][0];row[endpoint+'_reduction_ci95_upper']=e['ci95'][1]
            paired.append(row)
        bins=[('0–10%',lambda x:x<.1),('10–40%',lambda x:.1<=x<.4),('40–80%',lambda x:.4<=x<.8),('80–100% 미만',lambda x:.8<=x<1-1e-10),('100%',lambda x:abs(x-1)<1e-10)]
        for name,test in bins:
            kk=[r['sample']['sample_key'] for r in rr if test((r['sample']['span'][0][1]-r['sample']['span'][0][0])/r['planning_T'])]
            if kk:
                index=[keys.index(k) for k in kk];length_rows.append(dict(dataset=label,length_group=name,N=len(kk),uniform_miou=float(vals['uniform'][index].mean()),gt_miou=float(vals['gt'][index].mean()),random_mean_miou=float(vals['random_mean'][index].mean())))
    write(OUT/'metrics/evaluation_records.json',evaluations);csvwrite(OUT/'metrics/metrics.csv',metrics);csvwrite(OUT/'metrics/paired_comparisons.csv',paired);csvwrite(OUT/'metrics/sampling_diagnostics.csv',sampling);csvwrite(OUT/'metrics/length_groups.csv',length_rows);csvwrite(OUT/'metrics/compute_cost.csv',costs)
    summary=dict(state='evaluated',queries=300,conditions=CONDITIONS,density_ratio=3.5,seeds=[42,43],metrics=metrics,paired=paired,logical_responses=1200,new_gpu_responses=sum(not r['reused'] for r in costs),identical_input_reuses=sum(r['reused'] for r in costs),gpu_seconds=sum(r['gpu_seconds'] for r in costs),peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in costs),length_groups=length_rows)
    write(OUT/'metrics/summary.json',summary)
    lines=['# 정답 구간 전체를 더 촘촘히 보면 시간 구간을 더 잘 맞힐까?','',f'기존 pilot 300개 전체: Charades·ActivityNet·QVHighlights 각 100개 질문. 밀도비 3.5:1, Random seed 42·43. 평가 시각 {stamp()}.','',
        '## 지표를 읽는 법','',
        '**시간 구간 겹침률(IoU)**은 예측과 정답이 겹치는 길이를 두 구간의 합집합 길이로 나눈 값입니다. 이를 전체 질문에서 평균한 값이 **mIoU**입니다. 표에는 %로 표시하며 높을수록 좋습니다. **IoU≥0.5 비율**은 첫 예측 구간의 겹침률이 0.5 이상인 질문 비율입니다. 파싱 실패·역전·0길이·비유한 예측은 0점으로 전체 분모에 포함합니다.','',
        '**평균 절대 오차(Mean Absolute Error, MAE)**는 질문마다 |예측 시각−정답 시각|을 계산한 뒤 평균한 값입니다. 단위는 초이고 낮을수록 좋습니다. 예를 들어 정답 10초, 예측 13초면 절대 오차는 3초입니다. 시작·종료 시각을 따로 계산하며 무효 응답은 이 오차 평균에서 제외하고 수를 별도로 기록했습니다.','',
        '**pp**는 백분율 간 차이입니다. 40%→42%는 +2pp입니다. 영상 단위로 같은 영상의 질문들을 함께 다시 뽑는 bootstrap을 20,000회 시행해 탐색적 95% 신뢰구간을 구했습니다. 0을 포함하면 개선 방향이 불확실합니다. Random 두 응답은 질문별로 평균하며 600개 독립 질문으로 세지 않습니다.','',
        '## 이번 배치 방식','',
        '전체에 남기는 anchors는 없습니다. GT 안과 밖의 목표 초당 프레임 수 비율을 3.5:1로 고정하고, 각 영역을 배정된 수의 같은 길이 칸으로 나눠 중심 시각에 가장 가까운 서로 다른 원본 프레임을 골랐습니다. 바깥은 GT 앞·뒤 시간을 이어 붙인 축에서 선택했습니다. Random은 GT 길이만 알고 위치는 모르는 진단용 대조군이며, 같은 길이의 구간 시작점을 영상 안에서 균등하게 뽑습니다. Random과 GT가 겹치는 것은 허용했습니다. 실용적인 위치 선택기로 해석하면 안 됩니다.','',
        'Uniform은 영상 전체를 같은 길이 칸으로 나눈 중심에서 선택합니다. 모든 조건은 질문별 원래 2fps 입력과 같은 프레임 수·공간 해상도·visual-token 수를 사용합니다. 영상 토큰(visual token)은 영상이 모델 입력 특징으로 바뀐 단위입니다. 같은 source frame의 RGB는 공유 캐시에서 읽습니다. GT가 전체 영상이면 모든 방법이 정확히 같은 Uniform 프레임 목록을 씁니다.','',
        '목표 안쪽 프레임 수는 round(F×3.5L/[3.5L+T−L])입니다. 원본 프레임 수가 모자라면 반대쪽으로 예산을 옮기고, 양쪽에 후보가 있으면 가능할 때 최소 1장씩 배정했습니다. 실제 조정량·목표 간격과 실제 간격의 차이·Random과 GT의 겹침은 sampling_diagnostics.csv에 있습니다.','',
        '## 전체 결과','', '| 벤치마크 | 방식 | 평균 겹침률 % | IoU≥0.5 비율 % | 시작 오차 평균 초 | 종료 오차 평균 초 |','|---|---|---:|---:|---:|---:|']
    names=dict(uniform='전체 균등',gt='GT 전체 구간',random_s42='Random 42',random_s43='Random 43',random_mean='Random 두 seed 평균')
    def f(v):return '—' if v is None else f'{v:.2f}'
    for m in metrics:lines.append(f"| {m['dataset']} | {names[m['condition']]} | {100*m['miou']:.2f} | {100*m['r05']:.2f} | {f(m['start_mean_absolute_error'])} | {f(m['end_mean_absolute_error'])} |")
    lines+=['','## 정확한 위치를 아는 것이 도움이 됐나?','', '| 벤치마크 | 비교 | 평균 겹침률 차이 pp [95% 구간] | 개선/악화/동일 질문 수 |','|---|---|---|---|']
    for p in paired:
        if p['comparison'] not in ['gt-uniform','gt-random_mean']:continue
        lines.append(f"| {p['dataset']} | {'GT−Uniform' if p['comparison']=='gt-uniform' else 'GT−Random 평균'} | {p['delta_miou_pp']:+.2f} [{p['ci95_lower_pp']:+.2f}, {p['ci95_upper_pp']:+.2f}] | {p['improved']}/{p['worsened']}/{p['unchanged']} |")
    lines+=['']
    for label in labels:
        u=next(p for p in paired if p['dataset']==label and p['comparison']=='gt-uniform');r=next(p for p in paired if p['dataset']==label and p['comparison']=='gt-random_mean')
        if u['delta_miou_pp']>0 and r['delta_miou_pp']>0:
            text='두 비교 모두 개선 구간이 0보다 높아 후속 새 영상 확인 후보입니다.' if u['ci95_lower_pp']>0 and r['ci95_lower_pp']>0 else '좋아지는 방향은 있지만 구간이 0을 포함해 불확실합니다. 이 pilot 안에서 비율을 튜닝하기보다 새 영상 확인 여부를 결정해야 합니다.'
        elif u['delta_miou_pp']>0:text='Uniform보다 높아도 Random보다 낫지 않아, 정확한 GT 위치를 골랐기 때문이라고 말할 수 없습니다.'
        else:text='Uniform보다 평균 겹침률이 높지 않습니다. 이 고정 배치를 확대할 근거가 약하므로 다음 가설을 검토할 수 있습니다.'
        lines.append(f'- {label}: {text}')
    lines+=['','## GT가 긴 질문도 포함한 결과','', 'GT가 영상의 80% 이상이면 Random도 GT와 비슷한 위치를 고를 수밖에 없어 위치 효과를 구분하기 어렵습니다. 아래 집단도 전체 평가에서 제외하지 않았습니다.','', '| 벤치마크 | GT 길이/영상 길이 | 질문 수 | Uniform % | GT % | Random 평균 % |','|---|---|---:|---:|---:|---:|']
    for r in length_rows:lines.append(f"| {r['dataset']} | {r['length_group']} | {r['N']} | {100*r['uniform_miou']:.2f} | {100*r['gt_miou']:.2f} | {100*r['random_mean_miou']:.2f} |")
    exceptions=read(OUT/'validation/annotation_checks.json')['duration_resolutions']
    if exceptions:lines+=['','## 원본 annotation 확인','', 'GT를 조용히 자르지 않았습니다. 다음 항목은 원본 annotation과 영상 파일의 대응을 확인한 후 GT 값을 유지하고 파일의 실제 재생 길이로 계획 구간을 정의했습니다. 평가 GT와 원본 데이터는 수정하지 않았습니다.']+[f"- {x['sample_key']}: annotation 길이 {x['annotation_duration']:.6f}초, GT 끝 {x['gt'][1]}초, 파일 재생 길이 {x['container_duration']:.6f}초. 프레임 계획 T={x['planning_T']:.6f}초." for x in exceptions]
    lines+=['',f"논리적 응답 {summary['logical_responses']}개, 신규 GPU 생성 {summary['new_gpu_responses']}개, 같은 질문·모델·설정·입력 hash가 완전히 같은 응답 재사용 {summary['identical_input_reuses']}개입니다. GPU 생성시간 합계 {summary['gpu_seconds']:.1f}초, 최대 할당 GPU 메모리 {summary['peak_allocated_GiB']:.2f}GiB입니다. 재사용은 모델 입력이 같은 경우에만 허용했고 통계를 독립 반복으로 해석하지 않았습니다.",'',
        '표본·모델·밀도비 하나의 진단입니다. GT는 실용 방법이나 최적 상한이 아닙니다. 결과를 본 뒤 밀도비·시드를 바꾸지 않았습니다.','',
        '[상태](../../artifacts/runs/'+RUN+'/metadata/status.json) · [질문별 원문 평가](../../artifacts/runs/'+RUN+'/metrics/evaluation_records.json) · [전체 수치](../../artifacts/runs/'+RUN+'/metrics/metrics.csv) · [대응 비교·시작/종료 오차 변화와 신뢰구간](../../artifacts/runs/'+RUN+'/metrics/paired_comparisons.csv) · [배정 조정·실제 간격·구간 겹침](../../artifacts/runs/'+RUN+'/metrics/sampling_diagnostics.csv) · [비용](../../artifacts/runs/'+RUN+'/metrics/compute_cost.csv)']
    write_result_text(REPORT/'REPORT.md', '\n'.join(lines)+'\n');return summary

def main():
    import torch,fcntl,av,subprocess
    from artifact_store import register
    for category in ['metadata','plans','inputs','responses','metrics','validation','logs','code','environment']:(OUT/category).mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);progress=Progress();thread=threading.Thread(target=progress.heartbeat,daemon=True);thread.start()
        try:
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Whole-GT density 3.5 pilot300; seeds 42/43',str(Path(__file__).relative_to(ROOT)),'pilot_300','building',['pilot_300.cohort'])
            rows=inputs.records();resolutions=[]
            for record in rows:
                s=record['sample'];gt=s['span'][0];T=s['duration'];need(0<=gt[0]<gt[1],'Invalid GT annotation')
                if gt[1]>T:
                    with av.open(s['video_path']) as c:media_T=c.duration/1e6
                    need(gt[1]<=media_T,'GT outside both annotation duration and actual media; annotation needs review')
                    resolutions.append(dict(sample_key=s['sample_key'],annotation_duration=T,gt=gt,container_duration=media_T,planning_T=media_T,reason='Original annotation/video correspondence verified; unmodified GT fits actual file playback duration'))
                    T=media_T
                record['planning_T']=T
            write(OUT/'validation/annotation_checks.json',dict(passed=True,queries=300,duration_resolutions=resolutions,GT_modified=False))
            write(OUT/'validation/sampler_checks.json',checks());immutable(OUT/'metadata/config.json',CONFIG);immutable(OUT/'metadata/cohort.json',[r['sample'] for r in rows])
            progress.update(stage='source_frame_preparation')
            if not (OUT/'metadata/frame_cache_index.json').exists():inputs.prepare_pilot()
            index=read(OUT/'metadata/frame_cache_index.json');need(len(index)==300,'Incomplete pilot source inputs')
            sourcefiles=[Path(__file__),Path(inputs.__file__),Path(old.__file__),ROOT/'_lib/adaptive_temporal_report.py',ROOT/'_lib/boundary_oracle_analysis.py'];sources=[fileinfo(p) for p in sourcefiles];code_hash=objhash(sources);config_hash=objhash(CONFIG);cohort_hash=objhash([r['sample'] for r in rows])
            for p in sourcefiles:(OUT/'code'/(sha(p)[:12]+'_'+p.name)).write_bytes(p.read_bytes())
            write(OUT/'metadata/source_manifest.json',dict(files=sources,cohort=fileinfo(old.LEGACY/'cohorts/pilot_300/samples.json'),frame_cache_index=fileinfo(OUT/'metadata/frame_cache_index.json')))
            jobs=[]
            for record in rows:
                s=record['sample'];key=s['sample_key'];cache=read(index[key]);public=dict(sample_key=key,T=record['planning_T'],F=record['F'],ids=cache['ids'],pts=cache['pts']);gt=s['span'][0];L=gt[1]-gt[0]
                for condition in CONDITIONS:
                    p=make_plan(public,condition,gt=gt if condition=='gt' else None,length=L if condition.startswith('random') else None)
                    if condition.startswith('random'):need(old.encoded(p)==old.encoded(make_plan(copy.deepcopy(public),condition,length=L)),'Random plan not reproducible')
                    immutable(OUT/'plans'/condition/(key+'.json'),p);jobs.append(dict(sample_key=key,dataset=s['dataset'],condition=condition,plan_hash=objhash(p)))
            immutable(OUT/'metadata/jobs.json',jobs);write(OUT/'validation/plans_frozen.json',dict(passed=True,plans=1200,config_hash=config_hash,cohort_hash=cohort_hash,code_hash=code_hash))
            from transformers import AutoProcessor,AutoModelForImageTextToText
            torch.set_num_threads(2);progress.update(stage='waiting_for_gpu_memory')
            while torch.cuda.mem_get_info()[0]<28*2**30:time.sleep(20)
            progress.update(stage='loading_model');processor=AutoProcessor.from_pretrained(old.MODEL,padding_side='left',do_resize=False,local_files_only=True);model=AutoModelForImageTextToText.from_pretrained(old.MODEL,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
            write(OUT/'environment/runtime.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),backend='flash_attention_2',model_revision=CONFIG['revision']))
            (OUT/'environment/nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True));lookup={r['sample']['sample_key']:r for r in rows};input_reuse={};smoke=[]
            for job in jobs:
                key=job['sample_key'];condition=job['condition'];record=lookup[key];p=read(OUT/'plans'/condition/(key+'.json'));progress.update(stage='generation',sample_key=key,dataset=job['dataset'],condition=condition)
                t=time.perf_counter();cpu,detail=inputs.make_input(record,read(index[key]),p,processor);prep_seconds=time.perf_counter()-t;h=objhash(detail['hashes']);identity=dict(run_id=RUN,sample_key=key,condition=condition,config_hash=config_hash,cohort_hash=cohort_hash,code_hash=code_hash,plan_hash=job['plan_hash'],input_hash=h)
                dest=OUT/'responses'/condition/(key+'.json');write(OUT/'inputs'/condition/(key+'.json'),detail)
                reuse_key=(key,h);reused=False
                if dest.exists():result=read(dest);need(result['identity']==identity,'Resume config/input/plan mismatch');reused=result.get('reused_identical_input',False)
                elif reuse_key in input_reuse:
                    path=input_reuse[reuse_key];result=copy.deepcopy(read(path));result.update(identity=identity,input_detail_hash=objhash(detail),reused_identical_input=True,reuse_source=str(path),reuse_source_sha256=sha(path),generation_seconds=0.,prefill_seconds=0.,decode_seconds=0.,input_prepare_seconds=prep_seconds);write(dest,result);reused=True
                else:
                    result=old.generate_fresh(cpu,detail,identity,model,processor,dict(do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512,repetition_penalty=1.,use_cache=True),prep_seconds);result['reused_identical_input']=False;write(dest,result)
                input_reuse[reuse_key]=dest
                need(result['F']==record['F'] and result['grid']==record['grid'] and result['B']==record['B'],'Condition frame/token/grid mismatch')
                if len(smoke)<24:smoke.append(dict(sample_key=key,condition=condition,F=result['F'],B=result['B'],grid=result['grid']));write(OUT/'validation/smoke.json',dict(passed=True,records=smoke))
                progress.done(job['dataset'],condition,reused)
            del model;torch.cuda.empty_cache();progress.update(stage='evaluation_and_report');evaluate(rows,index)
            for source in sources:need(sha(source['path'])==source['sha256'],'Source changed during inference')
            write(OUT/'validation/final.json',dict(passed=True,queries=300,logical_responses=1200,budget_equality=True,source_frame_duplicates=0,ratio=3.5,seeds=[42,43]))
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Complete whole-GT density pilot300; seeds42/43',str(Path(__file__).relative_to(ROOT)),'pilot_300','complete',['pilot_300.cohort'])
            progress.update(state='complete',stage='complete',report=str(REPORT/'REPORT.md'))
        except BaseException:
            error=traceback.format_exc();progress.update(state='aborted',stage='aborted',error=error);write_result_text(REPORT/'ERROR.md', '# 실행 중단\n\n완료분과 기존 원본은 보존했습니다.\n\n```text\n'+error+'\n```\n')
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Aborted density pilot; see status/error',str(Path(__file__).relative_to(ROOT)),'pilot_300','aborted',['pilot_300.cohort'])
            raise
        finally:progress.stop.set();thread.join()

if __name__=='__main__':main()
