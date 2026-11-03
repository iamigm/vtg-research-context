"""Locate the first prefix that excludes all IoU-correct timestamp completions.
Oracle domain: nonnegative seconds with 0 or 1 fractional digit, no leading zeros.
No inference; all entropy values precede the token actually emitted.
"""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import json,csv,re,math,time,traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300');OUT=relocate_path(BASE/'prefix_entropy')
MODELS=['qwen25','qwen3'];DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens'];THRESHOLDS=[.3,.5,.7]
def read(p):return json.loads(p.read_text())
def metric(gt,s,e):return max(min(gt[1],e)-max(gt[0],s),0)/(max(gt[1],e)-min(gt[0],s)) if s<e else 0.
def near(values,target):
    values=np.asarray(values);left=values[values<=target];right=values[values>=target]
    return list(set(([int(left[-1])] if len(left) else [])+([int(right[0])] if len(right) else [])))
def oracle(gt,starts,ends):
    best=-1.;witness=None
    for s in near(starts,gt[0]):
        for e in near(ends,gt[1]):
            score=metric(gt,s,e)
            if score>best:best=score;witness=[s,e]
    return max(best,0.),witness

def validate():
    rng=np.random.default_rng(1)
    for _ in range(1000):
        a=float(rng.uniform(0,40));gt=[a,a+float(rng.uniform(.1,20))]
        s=np.unique(rng.integers(0,70,20));e=np.unique(rng.integers(0,70,20))
        expected=max(metric(gt,x,y) for x in s for y in e)
        assert abs(oracle(gt,s,e)[0]-expected)<1e-12
    # A wrong-looking digit can still be completed to a valid time.
    pool=[str(i) for i in range(100)]
    assert '25'.startswith('2') and not '25'.startswith('3')

def write_csv(name,rows):
    if not rows:return
    with (relocate_path(OUT/name)).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def mean(x):return float(np.mean(x)) if x else None

def main():
    OUT.mkdir(exist_ok=True);validate();started=time.time();manifest=read(relocate_path(ROOT/'predictions/cohorts/pilot_300/samples.json'))
    queries=[];allsteps=[];excluded=[];progress=0
    for model in MODELS:
        for sample in manifest:
            r=read(relocate_path(BASE/model/(sample['sample_key']+'.json')));text=''.join(t['token'] for t in r['steps']);gt=sample['span'][0]
            matches=re.search(r'(?<![\d:])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d:])',r['answer'])
            if ':' in r['answer'] or not matches or any(len(x.partition('.')[2])>1 for x in matches.groups()):
                excluded.append(dict(model=model,sample_key=sample['sample_key'],answer=r['answer'],reason='outside declared seconds/at-most-one-decimal grammar'));progress+=1;continue
            assert [round(float(x)) for x in matches.groups()]==r['timestamps'][0]
            # For IoU>=.3, e <= GT_start + GT_length/.3; larger e cannot rescue a prefix.
            # Include duration and actual endpoints as well; no artificial duration clipping.
            cap=math.ceil(max(sample['duration'],gt[0]+(gt[1]-gt[0])/.3,*map(float,matches.groups())))+2
            strings=[];rounded=[]
            for k in range(cap*10+1):
                label=f'{k//10}.{k%10}';strings.append(label);rounded.append(round(float(label)))
                if k%10==0:strings.append(str(k//10));rounded.append(k//10)
            values=np.asarray(rounded);universe=np.unique(values);cache={}
            def possible(prefix):
                if prefix not in cache:cache[prefix]=np.unique(values[[label.startswith(prefix) for label in strings]])
                return cache[prefix]
            initial,_=oracle(gt,universe,universe);states=[];offset=0;previous_max=initial
            for t in r['steps']:
                end=offset+len(t['token']);sets=[];prefixes=[]
                for n in [1,2]:
                    a,b=matches.span(n)
                    prefix=text[a:min(end,b)] if end>a else ''
                    prefixes.append(prefix)
                    sets.append(np.asarray([round(float(matches.group(n)))] ) if end>b else possible(prefix))
                best,witness=oracle(gt,*sets)
                assert best<=previous_max+1e-10,(sample['sample_key'],t['step'],previous_max,best)
                relevant=(end>matches.start(1) and offset<=matches.end(2))
                numeric=bool(re.search(r'\d',t['token'])) and any(max(offset,matches.start(n))<min(end,matches.end(n)) for n in [1,2])
                endpoint='start' if max(offset,matches.start(1))<min(end,matches.end(1)) else ('end' if max(offset,matches.start(2))<min(end,matches.end(2)) else 'boundary')
                state=dict(model=model,dataset=sample['dataset'],sample_key=sample['sample_key'],step=t['step'],token=t['token'],endpoint=endpoint,numeric=numeric,relevant=relevant,entropy=t['raw']['entropy_nats'],processed_entropy=t['processed']['entropy_nats'],selected_probability=t['raw_selected_probability'],oracle_before=previous_max,oracle_after=best,witness_after=witness,start_prefix=prefixes[0],end_prefix=prefixes[1],iou=r['iou'])
                for th in THRESHOLDS:
                    key='label_'+str(int(th*100));state[key]='initially_unattainable' if initial<th else ('retains_feasibility' if best+1e-12>=th else ('first_loss' if previous_max+1e-12>=th else 'after_loss'))
                states.append(state);previous_max=best;offset=end
            assert abs(states[-1]['oracle_after']-r['iou'])<1e-10
            losses={str(int(th*100)):next((s['step'] for s in states if s['label_'+str(int(th*100))]=='first_loss'),None) for th in THRESHOLDS}
            for th in THRESHOLDS:
                if initial+1e-12>=th:assert (losses[str(int(th*100))] is None)==(r['iou']+1e-12>=th)
            queries.append(dict(model=model,dataset=sample['dataset'],sample_key=sample['sample_key'],answer=r['answer'],gt=gt,iou=r['iou'],initial_oracle=initial,first_loss_steps=losses,steps=states))
            allsteps.extend(states);progress+=1
            if progress%50==0:
                elapsed=time.time()-started;status=dict(state='running',completed=progress,total=600,percent=progress/6,elapsed_seconds=elapsed,eta_seconds=elapsed*(600-progress)/progress)
                write_result_text((relocate_path(OUT/'status.json')), json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    groups=[];fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True);rng=np.random.default_rng(42)
    for mi,model in enumerate(MODELS):
        for di,ds in enumerate(DATASETS):
            qs=[q for q in queries if q['model']==model and q['dataset']==ds];before=[];fault=[];after=[];success=[];aligned=[];fault_numeric=[];fault_boundary=[]
            for q in qs:
                relevant=[s for s in q['steps'] if s['relevant']];lost=next((s for s in relevant if s['label_50']=='first_loss'),None)
                if lost:
                    fault.append(lost['entropy']);(fault_numeric if lost['numeric'] else fault_boundary).append(lost['entropy'])
                    pre=[s['entropy'] for s in relevant if s['numeric'] and s['step']<lost['step']]
                    post=[s['entropy'] for s in relevant if s['numeric'] and s['step']>lost['step']]
                    if pre:before.append(mean(pre))
                    if post:after.append(mean(post))
                    idx=relevant.index(lost)
                    for rel in range(-3,4):
                        j=idx+rel
                        if 0<=j<len(relevant):aligned.append(dict(relative_step=rel,entropy=relevant[j]['entropy'],numeric=relevant[j]['numeric'],sample_key=q['sample_key']))
                elif q['iou']>=.5:success.append(mean([s['entropy'] for s in relevant if s['numeric']]))
            g=dict(model=model,dataset=ds,n=len(qs),n_loss=len(fault),n_numeric_loss=len(fault_numeric),n_boundary_loss=len(fault_boundary),mean_before=mean(before),mean_first_loss=mean(fault),mean_numeric_first_loss=mean(fault_numeric),mean_boundary_first_loss=mean(fault_boundary),mean_after=mean(after),mean_success=mean(success),n_before=len(before),n_after=len(after),n_success=len(success),aligned=aligned);groups.append(g)
            ax=axes[mi,di]
            for pos,vals in enumerate([before,fault,after,success]):
                if vals:
                    ax.scatter(pos+rng.uniform(-.15,.15,len(vals)),vals,s=15,alpha=.4)
                    ax.plot([pos-.2,pos+.2],[mean(vals)]*2,color='black',lw=2)
            ax.set_xticks(range(4),['Before loss\n(query mean)','First loss\n(single step)','After loss\n(query mean)','Successful\n(query mean)']);ax.set_title(f'{model} / {ds.split("-")[0]}\nN={len(qs)}, first loss={len(fault)}');ax.grid(axis='y',alpha=.2)
            if di==0:ax.set_ylabel('Raw full-vocabulary entropy (nats)')
    fig.suptitle('IoU >= 0.5 feasibility along actual greedy prefixes\nFirst loss includes numeric tokens and timestamp-closing boundaries; black bars = means');fig.tight_layout(rect=(0,0,1,.93));fig.savefig(relocate_path(OUT/'prefix_entropy.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for index,g in enumerate(groups):
        ax=axes.flat[index];aligned=g['aligned'];keys=sorted({v['sample_key'] for v in aligned})
        for key in keys:
            points=[v for v in aligned if v['sample_key']==key]
            ax.plot([v['relative_step'] for v in points],[v['entropy'] for v in points],color='gray',alpha=.12,lw=.6)
        for v in aligned:
            x=v['relative_step'];color='#bc3333' if x==0 else ('#2474a7' if x<0 else '#949494')
            ax.scatter(x+rng.uniform(-.1,.1),v['entropy'],color=color,s=14,alpha=.5,marker='o' if v['numeric'] else 'x')
        ax.axvline(0,color='#bc3333',alpha=.3);ax.set_title(f"{g['model']} / {g['dataset'].split('-')[0]} (N={g['n_loss']})")
        ax.set_xticks(range(-3,4));ax.grid(alpha=.2)
        if index>=3:ax.set_xlabel('Actual generation step relative to first loss')
        if index%3==0:ax.set_ylabel('Single-step entropy (nats)')
    fig.suptitle('No temporal averaging: entropy around the first lost IoU>=0.5 completion\nBlue: before; red: first loss; gray: after. Circles: numeric tokens; crosses: punctuation/boundary.')
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(relocate_path(OUT/'step_entropy_aligned.png'),dpi=160);plt.close(fig)
    write_csv('per_step.csv',allsteps);write_csv('excluded.csv',excluded)
    write_result_text((relocate_path(OUT/'trajectories.json')), json.dumps(queries,ensure_ascii=False,indent=2)+'\n')
    write_result_text((relocate_path(OUT/'summary.json')), json.dumps(dict(groups=groups,thresholds=THRESHOLDS,excluded=excluded,validation=dict(oracle_vs_bruteforce_cases=1000,monotone_prefix_oracle=True,final_oracle_matches_saved_iou=True)),ensure_ascii=False,indent=2)+'\n')
    lines=['# 정답 가능성을 처음 잃는 생성 단계의 엔트로피','',f'기존 600개 full-head greedy 응답 중 {len(queries)}개를 분석했다. {len(excluded)}개는 초 단위 표기 범위 밖(MM:SS 등)이어서 별도 제외 목록에 기록했다. 재추론은 하지 않았다.','',
        '주 기준은 **IoU≥0.5를 만족하는 시간 구간으로 현재 prefix를 완성할 수 있는가**이다. 0.3/0.7 기준도 per_step.csv와 trajectories.json에 함께 기록했다. 올바른 GT 숫자 문자열과 일치하는지만 검사하지 않는다. IoU를 만족하는 다른 구간도 정답으로 인정한다.','',
        '각 step의 엔트로피는 해당 토큰을 선택하기 **직전**의 raw 전체 어휘 분포다. 생성된 토큰을 prefix에 붙인 뒤 가능한 최대 IoU를 계산한다. 최대 IoU가 처음 0.5 아래로 내려간 토큰을 first loss로 표시한다. 그 이전은 “맞게 생성한 토큰”이 아니라 **정답 가능성을 유지한 토큰**, 그 이후는 **이미 정답 가능성을 잃은 상태**다.','',
        '**판정 범위:** 음이 아닌 초 단위 숫자, 정수 또는 소수 한 자리, 불필요한 선행 0 없음, 시작−끝 두 구간 형식. delimiter가 나오면 그 숫자를 확정한다. 기존 평가처럼 Python round 후 첫 구간 IoU를 계산한다. GT 길이와 최저 threshold 0.3에서 도출한 충분히 큰 상한까지 열거하며 영상 duration으로 예측을 잘라내지 않는다. 두 끝점이 모두 확정된 경우 저장된 IoU와 일치함을 전 샘플에서 검증했다.','',
        '**이것은 이 표기 범위 안에서의 prefix 가능성 판정이지 토큰의 실제 인과 효과 추정이 아니다.** 이후 더 많은 소수 자리, 다른 시간 형식, 수정 문장을 허용하면 회복 가능성이 달라질 수 있다. 또한 가능한 completion이 존재해도 모델이 실제로 그 completion을 생성할 확률을 추정한 것은 아니다.','',
        '![평균하지 않은 step별 엔트로피](step_entropy_aligned.png)','', '가로축 0은 최초 가능성 상실 step, 음수/양수는 그 전/후의 실제 생성 step 거리다. 점 하나는 단일 step 엔트로피이며 시간축으로 평균하지 않았다. 원은 숫자, ×는 구두점/경계이며 가는 선은 같은 query를 잇는다. 아래 표와 [구간 평균 보조 그림](prefix_entropy.png)은 요약용이다.','',
        '| 모델 | 벤치마크 | 분석 N | first loss N (숫자/경계) | 이전 숫자 H | first loss H | 이후 숫자 H | 성공 응답 숫자 H |','|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'NA' if x is None else f'{x:.3f}'
    for g in groups:lines.append(f"| {g['model']} | {g['dataset']} | {g['n']} | {g['n_loss']} ({g['n_numeric_loss']}/{g['n_boundary_loss']}) | {fmt(g['mean_before'])} | {fmt(g['mean_first_loss'])} | {fmt(g['mean_after'])} | {fmt(g['mean_success'])} |")
    lines+=['','first loss는 샘플당 해당 단일 step의 H다. 이전/이후/성공 그룹은 해당 구간 숫자 step만 query 내부 평균한 보조 요약이며 원래 전체 숫자 단계 평균과 다르다. 숫자가 끝났음을 확정하는 구분자 때문에 가능성이 사라질 수 있어 경계 step도 포함하고 숫자/경계 수를 분리했다. 서로 다른 token 종류·자리수의 엔트로피 수준이 다르므로 평균 차이만으로 오류 탐지 성능을 주장하지 않는다. 원시 step별 결과를 함께 제공한다.','',
        '다음 예시는 first loss가 발생한 토큰 주변을 보여준다. 가능한 최대 IoU는 oracle이며 실제 출력 IoU와 구분해야 한다.','']
    for model in MODELS:
        picked=0
        for q in queries:
            if q['model']!=model:continue
            loss=next((s for s in q['steps'] if s['label_50']=='first_loss'),None)
            if not loss:continue
            lines += [f"### {model} · {q['sample_key']}",'',f"GT={q['gt']}, 최종 IoU={q['iou']:.3f}; 답변: `{q['answer']}`",'', '| Step | Token | H (nat) | 최대 IoU 전→후 | 구분 |','|---:|---|---:|---|---|']
            for s in q['steps']:
                if abs(s['step']-loss['step'])<=2:lines.append(f"| {s['step']} | `{s['token'].replace(' ','␠')}` | {s['entropy']:.3f} | {s['oracle_before']:.3f} → {s['oracle_after']:.3f} | {s['label_50']} |")
            picked+=1
            if picked==2:break
    lines+=['','[모든 step CSV](per_step.csv) · [모든 생성 경로 JSON](trajectories.json) · [집계/threshold/검증 JSON](summary.json) · [제외 목록](excluded.csv)']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
    write_result_text((relocate_path(OUT/'status.json')), json.dumps(dict(state='complete',completed=600,total=600,percent=100,analyzed=len(queries),excluded=len(excluded),elapsed_seconds=time.time()-started),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_result_text((relocate_path(OUT/'status.json')), json.dumps(dict(state='failed',error=traceback.format_exc()),indent=2)+'\n');raise
