"""Rounded endpoint correctness with an assigned decimal-point error boundary, reusing saved logits."""
from result_layout import relocate_path, write_result_text, copy_result_file, result_glob
from pathlib import Path
from decimal import Decimal
from collections import defaultdict,Counter
import json,csv,re,time,traceback,os,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
_FONT=relocate_path(relocate_path(Path(__file__)).resolve().parent/'entropy_viewer/NotoSansKR.ttf')
font_manager.fontManager.addfont(str(_FONT))
plt.rcParams['font.family']=font_manager.FontProperties(fname=str(_FONT)).get_name()
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/timelens_top30');OUT=relocate_path(BASE/'digit_correctness');ASSETS=relocate_path(ROOT/'_lib/digit_correctness_viewer')
MODELS=['qwen25','qwen3'];DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens']
CONDITIONS=['first_digit_correct','first_digit_wrong','previous_correct_now_correct','previous_correct_now_wrong','after_error']
NAMES={'first_digit_correct':'첫 숫자 정답','first_digit_wrong':'첫 숫자 오답','previous_correct_now_correct':'이전까지 정답 → 정답','previous_correct_now_wrong':'이전까지 정답 → 최초 오답','after_error':'이전 오류 이후 숫자','format_correct':'서식 유지','format_first_error':'이전까지 정답 가능 → 최초 오류 (서식)','format_after_error':'이전 오류 이후 서식'}
ROLES=['hundreds','tens','ones','decimal_point','tenths','closing_boundary'];rng=np.random.default_rng(42)
def read(p):return json.loads(p.read_text())
def write(name,data):
    p=relocate_path(OUT/name);p.parent.mkdir(parents=True,exist_ok=True);write_result_text(p, json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def csvfile(name,rows):
    with (relocate_path(OUT/name)).open('w') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def role_at(number,index):
    if index>=len(number):return 'extra_digit'
    integer=number.split('.')[0]
    if number[index]=='.':return 'decimal_point'
    if index<len(integer):return {0:'ones',1:'tens',2:'hundreds'}.get(len(integer)-index-1,'higher_integer')
    return 'tenths' if index==len(integer)+1 else 'later_fraction'
def error_boundary(number,gt,steps):
    # Same supported grammar as the source analysis: nonnegative seconds,
    # canonical integer part and at most one fractional digit.
    assert re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9])?',number),number
    candidates=[str(int(gt))]+[f'{k//10}.{k%10}' for k in range(max(0,int(gt)-1)*10,(int(gt)+1)*10) if round(float(f'{k//10}.{k%10}'))==gt]
    feasible=[(round(float(number))==gt if t['closed'] else any(c.startswith(t['prefix']) for c in candidates)) for t in steps]
    loss=next((i for i,v in enumerate(feasible) if not v),None)
    return loss,loss,feasible

def classify(number,gt,steps):
    assert float(gt)==int(gt);target=f'{int(gt)}.0';failed=False;offset=0;digit_index=0;result=[];rounded_equal=round(float(number))==gt
    boundary,loss,feasible=error_boundary(number,gt,steps)
    for step_index,source in enumerate(steps):
        s={k:v for k,v in source.items() if k not in ['label_30','label_50','label_70','oracle_before','oracle_after','witness_endpoint','fixed_gt_endpoint']};prefix=s['prefix'];intro=prefix[offset:];assert prefix.startswith(number[:offset]) and number.startswith(prefix)
        numeric=bool(re.search(r'\d',intro));previous_correct=not failed;expected=target[offset:offset+len(intro)]
        local_match=intro==expected
        current_error=step_index==boundary
        if numeric:
            ordinal=digit_index+1
            if not previous_correct:condition='after_error'
            elif digit_index==0:condition='first_digit_wrong' if current_error else 'first_digit_correct'
            else:condition='previous_correct_now_wrong' if current_error else 'previous_correct_now_correct'
            positions=[j for j in range(offset,len(prefix)) if number[j].isdigit()]
            actual_roles=[role_at(number,j) for j in positions];gt_roles=[role_at(target,j) for j in positions]
            role=actual_roles[0] if len(actual_roles)==1 else 'multi_digit';gt_role=gt_roles[0] if len(gt_roles)==1 else 'multi_digit'
            digit_index+=len(positions)
        else:
            ordinal=None;role='closing_boundary' if s['closed'] else 'decimal_point';gt_role='closing_boundary' if s['closed'] else role_at(target,offset)
            condition='format_after_error' if failed else ('format_first_error' if current_error else 'format_correct')
        first_error=previous_correct and current_error;failed=failed or current_error
        assert first_error==(step_index==loss) and (not failed)==feasible[step_index]
        s.update(target=target,introduced_text=intro,expected_text=expected if not s['closed'] else '(end / optional .0)',digit_ordinal=ordinal,digit_count_so_far=digit_index,numeric=numeric,role=role,gt_aligned_role=gt_role,previous_correct=previous_correct,local_text_match=local_match,rounded_value=round(float(number)),classification_rule='rounded_prefix_first_feasibility_loss',completion_feasible=feasible[step_index],actual_feasibility_loss=step_index==loss,first_error=first_error,prefix_correct=not failed,condition=condition)
        result.append(s);offset=len(prefix)
    assert offset==len(number)
    exact=not failed;value_equal=rounded_equal
    assert exact==value_equal,(number,target,exact,value_equal)
    return target,result,exact

def validate():
    def make(text):
        seq=[]
        for i in range(len(text)):seq.append(dict(prefix=text[:i+1],closed=False))
        seq.append(dict(prefix=text,closed=True));return seq
    fixtures=[('23.9',24,None),('24.0',24,None),('24',24,None),('24.4',24,None),('23.5',24,None),('24.5',24,None),('23.4',24,3),('35.0',24,0),('25.5',24,1),('23',24,2),('2.5',24,1),('240.0',24,2),('25.5',25,3),('24.5',25,3),('24.6',25,None),('0.5',0,None),('0.6',0,2)]
    for number,gt,boundary in fixtures:
        _,ss,ok=classify(number,gt,make(number))
        assert ok==(round(float(number))==gt)
        failures=[i for i,v in enumerate(ss) if v['first_error']]
        assert failures==([] if boundary is None else [boundary]),(number,gt,failures,boundary)
        if ok:assert all(v['prefix_correct'] for v in ss)
        else:assert all(v['prefix_correct'] for v in ss[:boundary]) and all(not v['prefix_correct'] for v in ss[boundary:])
    # Check the entire supported one-decimal grammar over a small domain.
    for gt in range(31):
        for k in range(401):
            number=f'{k//10}.{k%10}';_,ss,ok=classify(number,gt,make(number))
            assert ok==(round(float(number))==gt)
            assert sum(v['first_error'] for v in ss)==(not ok)



def statistics(rows):
    by=defaultdict(list)
    for r in rows:by[r['sample_key']].append(r['entropy'])
    a=np.asarray([np.mean(x) for x in by.values()]);raw=[r['entropy'] for r in rows]
    boot=rng.choice(a,(1000,len(a)),replace=True).mean(axis=1)
    return dict(n_queries=len(a),n_steps=len(rows),mean_entropy=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),step_mean=float(np.mean(raw)),step_median=float(np.median(raw)))

def main():
    OUT.mkdir(exist_ok=True);started=time.time();validate();source=read(relocate_path(BASE/'endpoint_gt_entropy/trajectories.json'));source_meta=read(relocate_path(BASE/'endpoint_gt_entropy/summary.json'));allsteps=[];endpoints=[];cases=defaultdict(dict)
    for i,q in enumerate(source):
        n=1 if q['endpoint']=='start' else 2;match=re.search(r'(?<![\d:])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d:])',q['answer']);number=match.group(n);gt=q['gt'][n-1]
        target,ss,exact=classify(number,gt,q['steps']);row=dict(model=q['model'],dataset=q['dataset'],sample_key=q['sample_key'],endpoint=q['endpoint'],gt=gt,target=target,prediction=number,gt_digits=len(str(int(gt))),generated_digits=len(number.split('.')[0]),all_correct=exact,rounded_value_equal=q['predicted_timestamps'][n-1]==gt,endpoint_iou=q['endpoint_iou'],joint_iou=q['joint_iou'],first_error_step=next((s['ar_step'] for s in ss if s['first_error']),None),first_error_numeric=next((s['numeric'] for s in ss if s['first_error']),None))
        assert row['all_correct']==row['rounded_value_equal']
        endpoints.append(row)
        for s in ss:
            s.update({k:row[k] for k in ['model','dataset','sample_key','endpoint','gt_digits','generated_digits','all_correct']});allsteps.append(s)
        cases[(q['model'],q['sample_key'])][q['endpoint']]=dict(row,steps=ss)
        if (i+1)%100==0:
            elapsed=time.time()-started;state=dict(state='running',timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage='rounded endpoint classification',counts=dict(Counter(x['model']+'/'+x['dataset'] for x in endpoints)),eta_scope='classification',completed=i+1,total=len(source),percent=100*(i+1)/len(source),elapsed_seconds=elapsed,eta_seconds=elapsed*(len(source)-i-1)/(i+1));write('status.json',state);print(json.dumps(state),flush=True)
    numeric=[s for s in allsteps if s['numeric']];condition_stats=[];role_stats=[];position_stats=[];endpoint_metrics=[]
    for model in MODELS:
        for ds in ['all']+DATASETS:
            for ep in ['start','end']:
                es=[q for q in endpoints if q['model']==model and q['endpoint']==ep and (ds=='all' or q['dataset']==ds)];rs=[s for s in numeric if s['model']==model and s['endpoint']==ep and (ds=='all' or s['dataset']==ds)]
                first=[s for s in rs if s['digit_ordinal']==1];assert len(first)==len(es)
                endpoint_metrics.append(dict(model=model,dataset=ds,endpoint=ep,n=len(es),first_digit_correct=sum(s['condition']=='first_digit_correct' for s in first),all_correct=sum(q['all_correct'] for q in es),rounded_correct=sum(q['rounded_value_equal'] for q in es),first_error_numeric=sum(q['first_error_numeric'] is True for q in es),first_error_format=sum(q['first_error_numeric'] is False for q in es)))
                for condition in CONDITIONS+['format_first_error','all_correct_endpoints','not_all_correct_endpoints']:
                    ss=[s for s in allsteps if s['model']==model and s['endpoint']==ep and (ds=='all' or s['dataset']==ds) and s['condition']==condition] if condition in CONDITIONS+['format_first_error'] else [s for s in rs if s['all_correct']==(condition=='all_correct_endpoints')]
                    if ss:condition_stats.append(dict(model=model,dataset=ds,endpoint=ep,condition=condition,**statistics(ss)))
                for role in ROLES:
                    for condition in CONDITIONS+['format_first_error']:
                        ss=[s for s in allsteps if s['model']==model and s['endpoint']==ep and (ds=='all' or s['dataset']==ds) and s['role']==role and s['condition']==condition]
                        if ss:role_stats.append(dict(model=model,dataset=ds,endpoint=ep,role=role,condition=condition,**statistics(ss)))
                for ordinal in sorted({s['digit_ordinal'] for s in rs}):
                    eligible=[s for s in rs if s['digit_ordinal']==ordinal and s['previous_correct']];right=[s for s in eligible if s['prefix_correct']];wrong=[s for s in eligible if not s['prefix_correct']]
                    position_stats.append(dict(model=model,dataset=ds,endpoint=ep,digit_ordinal=ordinal,eligible_n=len(eligible),correct_n=len(right),wrong_n=len(wrong),conditional_accuracy=len(right)/len(eligible) if eligible else None,correct_entropy=float(np.mean([s['entropy'] for s in right])) if right else None,wrong_entropy=float(np.mean([s['entropy'] for s in wrong])) if wrong else None))
    stratified=[];buckets=defaultdict(list)
    for s in allsteps:buckets[tuple(s[k] for k in ['model','dataset','endpoint','gt_digits','generated_digits','role','condition'])].append(s)
    for key,rs in sorted(buckets.items()):
        model,ds,ep,gd,pd,role,condition=key;stratified.append(dict(model=model,dataset=ds,endpoint=ep,gt_digits=gd,generated_digits=pd,role=role,condition=condition,**statistics(rs)))
    csvfile('per_step.csv',allsteps);csvfile('per_endpoint.csv',endpoints);csvfile('conditions.csv',condition_stats);csvfile('role_conditions.csv',role_stats);csvfile('position_conditional_accuracy.csv',position_stats);csvfile('digit_stratified.csv',stratified)
    summary=dict(endpoint_metrics=endpoint_metrics,conditions=condition_stats,roles=role_stats,positions=position_stats,digit_stratified=stratified,excluded=source_meta['excluded'],responses=594,endpoints=len(endpoints),numeric_steps=len(numeric),all_steps=len(allsteps));write('summary.json',summary)
    for ep in ['start','end']:
        fig,axes=plt.subplots(2,3,figsize=(16,8),sharey=True)
        for mi,m in enumerate(MODELS):
            for di,ds in enumerate(DATASETS):
                ax=axes[mi,di]
                for offset,condition,color in [(-.22,'previous_correct_now_correct','#357aa0'),(0,'first_error','#c65243'),(.22,'after_error','#929292')]:
                    for j,role in enumerate(ROLES):
                        ss=[s for s in allsteps if s['model']==m and s['dataset']==ds and s['endpoint']==ep and s['role']==role and (s['first_error'] if condition=='first_error' else (s['condition'] in ['first_digit_correct','previous_correct_now_correct'] if condition=='previous_correct_now_correct' else s['condition']==condition))]
                        if ss:ax.scatter(j+offset+rng.uniform(-.055,.055,len(ss)),[s['entropy'] for s in ss],s=13,alpha=.4,color=color,label=condition if j==0 else None)
                ax.set_xlim(-.5,5.5)
                for category in range(6):
                    ax.axvspan(category-.5,category+.5,color=('#f0f4f8' if category%2==0 else '#ffffff'),zorder=-2)
                    if category<5:ax.axvline(category+.5,color='#b9c4cf',lw=.8,zorder=-1)
                ax.set_xticks(range(6),['백의 자리','십의 자리','일의 자리','소수점','소수 첫째 자리','종료']);ax.set_title(f'{m} / {ds.split("-")[0]}');ax.grid(axis='y',alpha=.2)
                if di==0:ax.set_ylabel('Single-step raw entropy (nats)')
        fig.suptitle(f'{ep.upper()}: 자릿수별 단일 step 엔트로피 (첫 숫자부터 모두 포함)\n파랑: 정답 가능성 유지 / 빨강: 이전까지 정답 가능 → 최초 오류 / 회색: 이전 오류 이후')
        fig.tight_layout(rect=(0,0,1,.91));fig.savefig(relocate_path(OUT/f'{ep}_roles.png'),dpi=160);plt.close(fig)
    # Each sample has its own lazy-loaded trace and actual playable video.
    samples=read(relocate_path(ROOT/'predictions/cohorts/pilot_300/samples.json'));media=read(relocate_path(BASE/'media_manifest.json'));manifest=[]
    for sample in samples:
        entry=dict(sample,media=media[sample['video_path']]['media'],models={});dest=relocate_path(OUT/entry['media']);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(relocate_path(BASE/entry['media']),dest)
        for m in MODELS:
            raw=read(relocate_path(BASE/m/(sample['sample_key']+'.json')));eps=cases.get((m,sample['sample_key']),{});data=dict(model=m,sample=sample,answer=raw['answer'],timestamps=raw['timestamps'],iou=raw['iou'],endpoints=eps)
            key=m+'/'+sample['sample_key'];p=relocate_path(OUT/'traces'/(key+'.js'));p.parent.mkdir(parents=True,exist_ok=True);write_result_text(p, 'window.putTrace('+json.dumps(key)+','+json.dumps(data,ensure_ascii=False)+');\n');entry['models'][m]=dict(iou=raw['iou'],supported=bool(eps),start_correct=eps.get('start',{}).get('all_correct'),end_correct=eps.get('end',{}).get('all_correct'))
        manifest.append(entry)
    write_result_text((relocate_path(OUT/'manifest.js')), 'window.MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';\nwindow.CONDITION_STATS='+json.dumps(condition_stats,ensure_ascii=False)+';\nwindow.ROLE_STRATA='+json.dumps(stratified,ensure_ascii=False)+';\n')
    for model in MODELS:
        raw_dir=relocate_path(OUT/'raw'/model);raw_dir.mkdir(parents=True,exist_ok=True)
        for source in result_glob((relocate_path(BASE/model)), '*.json'):
            if not (raw_dir/source.name).exists():os.link(source,raw_dir/source.name)
    for name in ['report.html','viewer.js']:copy_result_file(relocate_path(ASSETS/name),relocate_path(OUT/name))
    for name in ['style.css','NotoSansKR.ttf','OFL.txt']:copy_result_file(relocate_path(ROOT/'_lib/entropy_viewer'/name),relocate_path(OUT/name))
    with (relocate_path(OUT/'style.css')).open('a') as f:f.write((relocate_path(ASSETS/'style.css')).read_text())
    write_report(summary)
    write('validation.json',dict(passed=True,synthetic_fixtures=17,exhaustive_small_domain_cases=12431,absorbing_error=True,optional_zero_decimal=True,original_entropy_unchanged=True,endpoint_matches_evaluator_rounding=True,red_equals_first_feasibility_loss=True))
    write('status.json',dict(state='built',stage='browser validation pending',completed=len(source),total=len(source),percent=100,elapsed_seconds=time.time()-started))

def write_report(s):
    lines=['# 반올림 기준 자릿수별 불확실성','', '[샘플별 영상·토큰 HTML](report.html)','',
        '예측 endpoint를 기존 평가와 동일한 Python `round(float(value))`로 반올림한 결과가 GT와 같으면 모든 생성 step을 정답으로 처리한다. 다르면 각 prefix에서 반올림 정답으로 완성할 수 있는지 검사한다. 정수부에서 가능성이 사라지면 해당 숫자가 최초 오류다. 소수부도 동일하게 정답 가능성이 실제로 처음 사라지는 숫자 step을 최초 오류로 표시한다. 소수점으로 앞당기지 않는다. 정수 종료로 가능성이 사라지면 종료 단계가 최초 오류다. 후보 표기는 원자료와 동일하게 음이 아닌 초, 정수 또는 소수 한 자리, 불필요한 선행 0 없음으로 제한한다. 시작과 종료를 각각 판정한다. Python 반올림은 정확히 .5일 때 가까운 짝수로 처리한다.','',
        '예: GT 24에 대한 23.9는 모두 정답. 23.4는 23.까지 정답 가능, 숫자 4가 최초 오류다. 35.0은 첫 숫자 3, 25.0은 숫자 5에서 정답 가능성이 사라진다. 빨간 표시 first_error와 실제 가능성 상실 actual_feasibility_loss는 항상 일치한다. GT 문자 일치 여부는 참고 필드로만 남긴다.','',
        '기존 동일 pilot 300개 × 두 모델의 저장된 분포를 재사용했다. 초 단위 표기 594개 응답 / 1,188개 endpoint를 분석하고 MM:SS 6개는 제외했다. 새 추론 없이 raw 전체 어휘 엔트로피를 유지했다.','',
        '## endpoint 결과','', '| 모델 | 데이터셋 | 대상 | N | 반올림 정답 N | 최초 오류 숫자/소수점·종료 N |','|---|---|---|---:|---:|---|']
    for r in s['endpoint_metrics']:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['endpoint']} | {r['n']} | {r['all_correct']} | {r['first_error_numeric']}/{r['first_error_format']} |")
    lines+=['','## 경우별 엔트로피','', '각 query 내부에서 해당 상태의 step 엔트로피를 평균한 뒤 query 간 평균한다. 95% CI는 query bootstrap 1,000회. 최초 오류 행은 소수점 또는 종료 토큰도 포함한다. 최종 정답/오답 endpoint 행은 모든 숫자 step으로 별도 집계하며 다른 행과 중복된다.','', '| 모델 | 대상 | 경우 | N query / N step | 평균 H [95% CI] |','|---|---|---|---|---|']
    for r in s['conditions']:
        if r['dataset']=='all':
            label=NAMES.get(r['condition'],r['condition'])
            lines.append(f"| {r['model']} | {r['endpoint']} | {label} | {r['n_queries']} / {r['n_steps']} | {r['mean_entropy']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    lines+=['','![시작](start_roles.png)','', '![종료](end_roles.png)','',
        '가로축은 구분선으로 나눈 자릿수·서식 범주이며 시간이 아니다. 칸 안 색상별 좌우 위치는 가독성을 위한 배치다. 파랑은 첫 숫자를 포함해 정답 가능성을 유지한 숫자, 빨강은 정답 가능성이 처음 사라진 step, 회색은 이전에 이미 가능성이 사라진 뒤의 숫자다. 생성 전 빈 문자열은 정답 가능 상태로 두므로 첫 숫자도 파랑 또는 빨강으로 표시한다. 첫 숫자별 정량 집계는 별도로 유지한다.','',
        '## 해석','', '오류 전 상태는 GT 문자 자체의 일치가 아니라 반올림 정답으로 완성 가능한 prefix를 뜻한다. 소수점이나 종료 구분자 자체가 가능한 완성 후보를 없애는 경우에는 그 서식 step이 최초 오류다. 첫 숫자도 동일한 가능성 기준으로 파랑 또는 빨강으로 표시한다. 반올림 endpoint 일치와 전체 구간 IoU는 다른 지표다. 엔트로피는 원래 생성 경로의 분포이며 인과 효과를 뜻하지 않는다.','',
        '[전체 step](per_step.csv) · [endpoint](per_endpoint.csv) · [경우별](conditions.csv) · [자리별](role_conditions.csv) · [조건부 상태 비율](position_conditional_accuracy.csv) · [자릿수 통제](digit_stratified.csv)']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write('status.json',dict(state='failed',error=traceback.format_exc()));raise
