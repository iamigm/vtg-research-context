"""Endpoint-specific prefix feasibility; opposite endpoint fixed to exact GT.
Uses saved greedy distributions. Does NOT condition generation on GT.
"""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import json,csv,re,math,time,traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timelens_prefix_entropy import metric,MODELS,DATASETS,THRESHOLDS
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/timelens_top30');OUT=relocate_path(BASE/'endpoint_gt_entropy')
def read(p):return json.loads(p.read_text())
def write(name,obj):write_result_text((relocate_path(OUT/name)), json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def csvfile(name,rows):
    if rows:
        with (relocate_path(OUT/name)).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(vals):return float(np.mean(vals)) if vals else None

def best_iou(gt,values,endpoint):
    if len(values)==0:return 0.,None
    v=np.asarray(values,dtype=float);a,b=gt
    if endpoint=='start':scores=np.maximum(b-np.maximum(a,v),0)/(b-np.minimum(a,v));scores[v>=b]=0
    else:scores=np.maximum(np.minimum(b,v)-a,0)/(np.maximum(b,v)-a);scores[v<=a]=0
    j=int(np.argmax(scores));return float(scores[j]),float(v[j])
def validate():
    rng=np.random.default_rng(3)
    for _ in range(1000):
        a=float(rng.uniform(0,30));gt=[a,a+float(rng.uniform(.1,30))];vals=np.unique(rng.integers(0,80,30))
        for ep in ['start','end']:
            scores=[metric(gt,v,gt[1]) if ep=='start' else metric(gt,gt[0],v) for v in vals]
            assert abs(best_iou(gt,vals,ep)[0]-max(scores))<1e-12

def main():
    OUT.mkdir(exist_ok=True);started=time.time();validate();samples=read(relocate_path(ROOT/'predictions/cohorts/pilot_300/samples.json'));trajectories=[];steps=[];excluded=[];done=0
    for model in MODELS:
        for sample in samples:
            r=read(relocate_path(BASE/model/(sample['sample_key']+'.json')));gt=sample['span'][0];assert gt[1]>gt[0]
            match=re.search(r'(?<![\d:])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d:])',r['answer']);done+=1
            if ':' in r['answer'] or not match or any(len(v.partition('.')[2])>1 for v in match.groups()):
                excluded.append(dict(model=model,dataset=sample['dataset'],sample_key=sample['sample_key'],answer=r['answer'],reason='outside seconds with at most one decimal grammar'));continue
            text=''.join(t['token'] for t in r['steps']);assert text.startswith(r['answer'])
            predicted=[round(float(v)) for v in match.groups()];assert predicted==r['timestamps'][0]
            cap=math.ceil(max(sample['duration'],gt[0]+(gt[1]-gt[0])/.3,*map(float,match.groups())))+2
            labels=[];vals=[]
            for k in range(cap*10+1):
                label=f'{k//10}.{k%10}';labels.append(label);vals.append(round(float(label)))
                if k%10==0:labels.append(str(k//10));vals.append(k//10)
            vals=np.asarray(vals);allvalues=np.unique(vals)
            for n,endpoint in [(1,'start'),(2,'end')]:
                a,b=match.span(n);cache={};initial,_=best_iou(gt,allvalues,endpoint);previous=initial;states=[];offset=0
                for t in r['steps']:
                    limit=offset+len(t['token'])
                    if limit>a and offset<=b:
                        prefix=text[a:min(limit,b)];closed=limit>b
                        if closed:values=np.asarray([predicted[n-1]])
                        else:
                            if prefix not in cache:cache[prefix]=np.unique(vals[[v.startswith(prefix) for v in labels]])
                            values=cache[prefix]
                        best,witness=best_iou(gt,values,endpoint);assert best<=previous+1e-10
                        overlap=text[max(a,offset):min(b,limit)];numeric=bool(re.search(r'\d',overlap))
                        row=dict(model=model,dataset=sample['dataset'],sample_key=sample['sample_key'],endpoint=endpoint,endpoint_step=len(states)+1,ar_step=t['step'],token=t['token'],numeric=numeric,closed=closed,prefix=prefix,entropy=t['raw']['entropy_nats'],processed_entropy=t['processed']['entropy_nats'],oracle_before=previous,oracle_after=best,witness_endpoint=witness,fixed_gt_endpoint=gt[1] if endpoint=='start' else gt[0],joint_iou=r['iou'])
                        for th in THRESHOLDS:row['label_'+str(int(th*100))]='initially_unattainable' if initial+1e-12<th else ('retains_feasibility' if best+1e-12>=th else ('first_loss' if previous+1e-12>=th else 'after_loss'))
                        states.append(row);previous=best
                    offset=limit
                final=metric(gt,predicted[0],gt[1]) if endpoint=='start' else metric(gt,gt[0],predicted[1]);assert abs(states[-1]['oracle_after']-final)<1e-10
                for th in THRESHOLDS:
                    if initial+1e-12>=th:assert any(s['label_'+str(int(th*100))]=='first_loss' for s in states)==(final+1e-12<th)
                trajectories.append(dict(model=model,dataset=sample['dataset'],sample_key=sample['sample_key'],endpoint=endpoint,gt=gt,answer=r['answer'],predicted_timestamps=predicted,joint_iou=r['iou'],endpoint_iou=final,absolute_error_seconds=abs(predicted[n-1]-gt[n-1]),unrounded_absolute_error_seconds=abs(float(match.group(n))-gt[n-1]),initial_oracle=initial,steps=states))
                steps.extend(states)
            if done%50==0:
                elapsed=time.time()-started;status=dict(state='running',stage='endpoint prefix evaluation',completed=done,total=600,percent=done/6,elapsed_seconds=elapsed,eta_seconds=elapsed*(600-done)/done);write('status.json',status);print(json.dumps(status),flush=True)
    groups=[];rng=np.random.default_rng(42)
    for endpoint in ['start','end']:
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        for mi,model in enumerate(MODELS):
            for di,ds in enumerate(DATASETS):
                qs=[q for q in trajectories if q['endpoint']==endpoint and q['model']==model and q['dataset']==ds];fault=[];numeric_fault=[];paired=[];aligned=[];boundary=0
                for q in qs:
                    loss=next((s for s in q['steps'] if s['label_50']=='first_loss'),None)
                    if loss:
                        fault.append(loss['entropy']);boundary+=not loss['numeric']
                        if loss['numeric']:numeric_fault.append(loss['entropy'])
                        preceding=[s for s in q['steps'] if s['numeric'] and s['ar_step']<loss['ar_step']]
                        if preceding:paired.append(loss['entropy']-preceding[-1]['entropy'])
                        for s in q['steps']:
                            relative=s['ar_step']-loss['ar_step']
                            if abs(relative)<=3:aligned.append(dict(sample_key=q['sample_key'],relative_step=relative,entropy=s['entropy'],numeric=s['numeric']))
                g=dict(model=model,dataset=ds,endpoint=endpoint,n=len(qs),r50=100*mean([q['endpoint_iou']>=.5 for q in qs]),mean_endpoint_iou=mean([q['endpoint_iou'] for q in qs]),mean_absolute_error_seconds=mean([q['absolute_error_seconds'] for q in qs]),joint_failure_endpoint_success=sum(q['joint_iou']<.5 and q['endpoint_iou']>=.5 for q in qs),first_loss_count=len(fault),numeric_first_loss_count=len(numeric_fault),boundary_first_loss_count=boundary,first_loss_entropy=mean(fault),numeric_first_loss_entropy=mean(numeric_fault),paired_count=len(paired),first_loss_minus_last_feasible_numeric_entropy=mean(paired))
                if paired:
                    arr=np.asarray(paired);boots=rng.choice(arr,(2000,len(arr))).mean(axis=1);g['paired_difference_ci95']=np.quantile(boots,[.025,.975]).tolist()
                else:g['paired_difference_ci95']=[None,None]
                groups.append(g);ax=axes[mi,di]
                for key in {v['sample_key'] for v in aligned}:
                    pts=[v for v in aligned if v['sample_key']==key];ax.plot([v['relative_step'] for v in pts],[v['entropy'] for v in pts],color='gray',alpha=.12,lw=.6)
                for numeric in [False,True]:
                    for side,color in [(-1,'#2474a7'),(0,'#bc3333'),(1,'#949494')]:
                        pts=[v for v in aligned if v['numeric']==numeric and np.sign(v['relative_step'])==side]
                        if pts:ax.scatter([v['relative_step']+rng.uniform(-.1,.1) for v in pts],[v['entropy'] for v in pts],s=16,alpha=.45,color=color,marker='o' if numeric else 'x')
                ax.axvline(0,color='#bc3333',alpha=.3);ax.set_xticks(range(-3,4));ax.set_title(f'{model} / {ds.split("-")[0]}: first loss N={len(fault)}');ax.grid(alpha=.2)
                if mi==1:ax.set_xlabel('Actual step relative to this endpoint\'s first loss')
                if di==0:ax.set_ylabel('Single-step raw entropy (nats)')
        fig.suptitle(f'{endpoint.upper()} endpoint: opposite endpoint fixed to GT (IoU >= 0.5)\nNo temporal averaging. Blue: feasible; red: first loss; gray: after. Circle: number; cross: punctuation/boundary.')
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(relocate_path(OUT/f'{endpoint}_step_entropy.png'),dpi=160);fig.savefig(relocate_path(OUT/f'{endpoint}_step_entropy.svg'));plt.close(fig)
    rows=[{k:v for k,v in q.items() if k!='steps'} for q in trajectories];csvfile('per_endpoint.csv',rows);csvfile('per_step.csv',steps);csvfile('excluded.csv',excluded);write('trajectories.json',trajectories)
    write('summary.json',dict(groups=groups,excluded=excluded,thresholds=THRESHOLDS,validation=dict(exhaustive_oracle_checks=2000,monotone_prefix_oracle=True,final_matches_fixed_gt_endpoint_iou=True,original_distributions_unchanged=True)))
    lines=['# 반대쪽 시각을 GT로 고정한 endpoint별 엔트로피','',f'동일 pilot 300개 × 두 모델의 저장된 greedy 분포를 사용했다. 초 단위/소수 한 자리 범위의 {len(rows)//2}개 응답에서 시작·종료를 각각 평가했다. MM:SS 등 {len(excluded)}개 응답은 명시적으로 제외했다. GPU 재추론은 없다.','',
        '- 시작 평가: 예측 시작 시각 + **GT 종료 시각**.','- 종료 평가: **GT 시작 시각** + 예측 종료 시각.','',
        'GT는 원래 정밀도 그대로 고정한다. 예측 숫자만 기존 평가와 동일하게 Python round한다. 각 endpoint의 prefix에서 완성 가능한 최대 IoU를 계산하고, 0.5 아래로 처음 내려가는 토큰을 독립적으로 표시한다. 0.3/0.7 기준도 CSV에 포함했다. 반대쪽 시각이 이미 틀렸더라도 이 endpoint가 정답 가능한 상태이면 오류 이후로 분류하지 않는다.','',
        '**생성된 시작 시각을 GT로 바꾸어 모델을 다시 실행한 실험은 아니다.** 종료 시각의 엔트로피는 여전히 원래 생성한 시작 시각에 조건화되어 있다. GT 고정은 평가에만 적용하므로 실제 토큰 분포를 그대로 재사용한다.','',
        '![시작 시각](start_step_entropy.png)','', '![종료 시각](end_step_entropy.png)','',
        '가로축 0은 해당 endpoint의 최초 가능성 상실 step이고, 점 하나는 평균하지 않은 단일 step의 선택 직전 엔트로피다. 각 그래프에는 그 endpoint에서 실패한 응답만 포함한다. 반대쪽 endpoint의 step은 섞지 않았다. 원은 숫자, ×는 소수점/구분자다.','',
        '| 모델 | 벤치마크 | 평가 대상 | N | GT 고정 R@.5 (%) | 평균 절대오차 (초) | 기존 구간 실패→이 endpoint 성공 N | 최초 오류 N (숫자/경계) | 최초 오류 H |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'NA' if x is None else f'{x:.3f}'
    for g in groups:lines.append(f"| {g['model']} | {g['dataset']} | {g['endpoint']} | {g['n']} | {g['r50']:.1f} | {g['mean_absolute_error_seconds']:.3f} | {g['joint_failure_endpoint_success']} | {g['first_loss_count']} ({g['numeric_first_loss_count']}/{g['boundary_first_loss_count']}) | {fmt(g['first_loss_entropy'])} |")
    lines+=['','`기존 구간 실패→이 endpoint 성공`은 원래 전체 구간 IoU<0.5이지만 반대쪽을 GT로 고정하면 해당 endpoint가 IoU≥0.5인 응답 수다. 이는 기존 공동 판정에서 가려졌던 endpoint 성능을 보여준다.','',
        '실패 step의 H에서 직전 정답 가능 상태의 **마지막 숫자 step** H를 뺀 paired 차이를 summary.json에 저장했다. 직전 숫자 step이 없는 응답은 이 차이에서 제외한다. Bootstrap CI는 query 단위 2,000회 재표집한 탐색적 구간이며, 숫자의 자리·token 종류가 다를 수 있어 인과적 오류 탐지 효과로 해석하지 않는다.','']
    for endpoint in ['start','end']:
        gs=[g for g in groups if g['endpoint']==endpoint];rescued=sum(g['joint_failure_endpoint_success'] for g in gs)
        lines.append(f"- {endpoint}: 원래 구간은 실패했지만 반대쪽을 GT로 고정하면 성공하는 응답은 총 {rescued}개(두 모델 합산)다.")
    lines+=['','판정은 음이 아닌 초 단위 숫자, 정수/소수 한 자리, 선행 0 없음이라는 표기 범위 내에서만 유효하다. 다른 문법, 추가 소수 자리, 수정 문장을 허용하면 회복 가능성이 달라질 수 있다. 표본의 분포는 고정된 실제 greedy 경로의 조건부 분포이며, 가능한 completion이 실제 모델에서 선택될 확률까지 계산한 것은 아니다. 기존 Decord 디코딩 한계도 유지된다.','',
        '[endpoint별 CSV](per_endpoint.csv) · [step별 CSV](per_step.csv) · [전체 경로 JSON](trajectories.json) · [집계·검증 JSON](summary.json) · [제외 목록](excluded.csv)']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n');write('status.json',dict(state='complete',completed=600,total=600,percent=100,analyzed_responses=len(rows)//2,endpoint_evaluations=len(rows),excluded=len(excluded),elapsed_seconds=time.time()-started))
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write('status.json',dict(state='failed',error=traceback.format_exc()));raise
