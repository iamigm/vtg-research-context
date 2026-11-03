"""Role- and digit-length-stratified entropy from fixed-opposite-GT endpoint traces."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import json,re,csv,time,traceback
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/timelens_top30');OUT=relocate_path(BASE/'digit_role_entropy')
MODELS=['qwen25','qwen3'];DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens']
ROLES=['hundreds','tens','ones','decimal_point','tenths','closing_boundary'];NAMES={'hundreds':'백의 자리','tens':'십의 자리','ones':'일의 자리','decimal_point':'소수점','tenths':'소수 첫째 자리','closing_boundary':'종료 구분자','multi_role':'여러 역할을 포함한 토큰','format_other':'기타 서식'}
LABELS=['retains_feasibility','first_loss','after_loss','initially_unattainable']
def read(p):return json.loads(p.read_text())
def write(name,obj):write_result_text((relocate_path(OUT/name)), json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def csvfile(name,rows):
    with (relocate_path(OUT/name)).open('w') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digit_count(value):return len(str(int(value)))
def roles_for(number,a,b):
    integer=number.split('.')[0];roles=[]
    for i in range(a,b):
        char=number[i]
        if char=='.':role='decimal_point'
        elif i<len(integer):role={0:'ones',1:'tens',2:'hundreds'}.get(len(integer)-i-1,'integer_10pow_'+str(len(integer)-i-1))
        else:role='tenths' if i==len(integer)+1 else 'fraction_'+str(i-len(integer))
        roles.append(role)
    return roles

def main():
    OUT.mkdir(exist_ok=True);started=time.time();source=read(relocate_path(BASE/'endpoint_gt_entropy/trajectories.json'));source_summary=read(relocate_path(BASE/'endpoint_gt_entropy/summary.json'));rows=[];endpoint_rows=[]
    assert roles_for('25.0',0,4)==['tens','ones','decimal_point','tenths']
    assert roles_for('105.7',0,5)==['hundreds','tens','ones','decimal_point','tenths']
    for q in source:
        match=re.search(r'(?<![\d:])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d:])',q['answer']);n=1 if q['endpoint']=='start' else 2;number=match.group(n);gt=q['gt'][n-1]
        gd=digit_count(gt);pd=len(number.split('.')[0]);rounded_pd=digit_count(q['predicted_timestamps'][n-1]);prev='';seen=[]
        for s in q['steps']:
            prefix=s['prefix'];assert prefix.startswith(prev) and number.startswith(prefix)
            roles=roles_for(number,len(prev),len(prefix))
            if s['closed']:roles.append('closing_boundary')
            if not roles:role='format_other'
            elif len(roles)==1:role=roles[0]
            else:role='multi_role'
            row=dict(model=q['model'],dataset=q['dataset'],sample_key=q['sample_key'],endpoint=q['endpoint'],gt=gt,generated_number=number,gt_digits=gd,generated_digits=pd,rounded_generated_digits=rounded_pd,digits_match=gd==pd,ar_step=s['ar_step'],token=s['token'],introduced_text=prefix[len(prev):],role=role,role_components='|'.join(roles),entropy=s['entropy'],processed_entropy=s['processed_entropy'],label_30=s['label_30'],label_50=s['label_50'],label_70=s['label_70'],oracle_before=s['oracle_before'],oracle_after=s['oracle_after'],endpoint_iou=q['endpoint_iou'],joint_iou=q['joint_iou'])
            rows.append(row);prev=prefix
            if role in ROLES:seen.append(role)
        assert len(seen)==len(set(seen)),(q['sample_key'],q['endpoint'],seen)
        assert prev==number
        endpoint_rows.append(dict(model=q['model'],dataset=q['dataset'],sample_key=q['sample_key'],endpoint=q['endpoint'],gt=gt,generated_number=number,gt_digits=gd,generated_digits=pd,rounded_generated_digits=rounded_pd,digits_match=gd==pd,endpoint_iou=q['endpoint_iou'],first_loss_role=next((r['role'] for r in rows[-len(q['steps']):] if r['label_50']=='first_loss'),None)))
    csvfile('per_step_roles.csv',rows);csvfile('per_endpoint_digits.csv',endpoint_rows)
    buckets=defaultdict(list)
    for r in rows:buckets[tuple(r[k] for k in ['model','dataset','endpoint','gt_digits','generated_digits','role','label_50'])].append(r['entropy'])
    stats=[]
    for key,vals in sorted(buckets.items()):
        model,ds,ep,gd,pd,role,label=key
        stats.append(dict(model=model,dataset=ds,endpoint=ep,gt_digits=gd,generated_digits=pd,digits_match=gd==pd,role=role,label=label,n=len(vals),mean_entropy=float(np.mean(vals)),median_entropy=float(np.median(vals)),q25=float(np.quantile(vals,.25)),q75=float(np.quantile(vals,.75))))
    csvfile('role_statistics.csv',stats)
    contrasts=[]
    for key in sorted({tuple(r[k] for k in ['model','dataset','endpoint','gt_digits','generated_digits','role']) for r in rows}):
        kept=buckets.get(key+('retains_feasibility',),[]);lost=buckets.get(key+('first_loss',),[]);after=buckets.get(key+('after_loss',),[])
        if not lost:continue
        model,ds,ep,gd,pd,role=key
        contrasts.append(dict(model=model,dataset=ds,endpoint=ep,gt_digits=gd,generated_digits=pd,role=role,n_retained=len(kept),n_first_loss=len(lost),n_after_loss=len(after),retained_mean=float(np.mean(kept)) if kept else None,first_loss_mean=float(np.mean(lost)),difference=float(np.mean(lost)-np.mean(kept)) if kept else None,enough_for_descriptive_comparison=len(kept)>=5 and len(lost)>=5))
    csvfile('first_loss_vs_retained.csv',contrasts)
    patterns=sorted({(r['endpoint'],r['gt_digits'],r['generated_digits']) for r in rows});plots=[];rng=np.random.default_rng(42)
    for i,(ep,gd,pd) in enumerate(patterns):
        fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True)
        for mi,model in enumerate(MODELS):
            for di,ds in enumerate(DATASETS):
                ax=axes[mi,di];subset=[r for r in rows if r['model']==model and r['dataset']==ds and r['endpoint']==ep and r['gt_digits']==gd and r['generated_digits']==pd];n=len({r['sample_key'] for r in subset})
                for label,color,offset,marker in [('retains_feasibility','#2474a7',-.20,'o'),('first_loss','#c33535',0,'D'),('after_loss','#999999',.20,'o')]:
                    ss=[r for r in subset if r['label_50']==label and r['role'] in ROLES]
                    xs=[ROLES.index(r['role'])+offset+rng.uniform(-.065,.065) for r in ss]
                    ax.scatter(xs,[r['entropy'] for r in ss],color=color,alpha=.55,s=17,marker=marker,label=label)
                    for role in ROLES:
                        vals=[r['entropy'] for r in ss if r['role']==role]
                        if vals:
                            x=ROLES.index(role)+offset;ax.plot([x-.08,x+.08],[np.median(vals)]*2,color=color,lw=2)
                ax.set_title(f'{model} / {ds.split("-")[0]}: N={n}');ax.set_xticks(range(len(ROLES)),['100s','10s','1s','decimal .','0.1s','close'],rotation=20);ax.grid(axis='y',alpha=.2)
                if di==0:ax.set_ylabel('Single-step full-vocabulary entropy (nats)')
                if mi==1:ax.set_xlabel('Role in the emitted timestamp (not relative step)')
        axes[0,0].legend(fontsize=8);fig.suptitle(f'{ep.upper()}: GT integer digits={gd}, generated integer digits={pd}; opposite endpoint fixed to GT\nEach point is one token step. Blue: feasible; red: first loss; gray: after loss. Short bars: medians.')
        fig.tight_layout(rect=(0,0,1,.92));name=f'{ep}_gt{gd}_pred{pd}.png';fig.savefig(relocate_path(OUT/name),dpi=160);plt.close(fig);plots.append(dict(endpoint=ep,gt_digits=gd,generated_digits=pd,path=name))
        elapsed=time.time()-started;status=dict(state='running',stage='role-stratified plots',completed=i+1,total=len(patterns),percent=100*(i+1)/len(patterns),elapsed_seconds=elapsed,eta_seconds=elapsed*(len(patterns)-i-1)/(i+1));write('status.json',status);print(json.dumps(status),flush=True)
    counts=Counter((q['endpoint'],q['gt_digits'],q['generated_digits']) for q in endpoint_rows);mixed=[r for r in rows if r['role']=='multi_role']
    summary=dict(endpoint_evaluations=len(endpoint_rows),responses=len(endpoint_rows)//2,steps=len(rows),multi_role_steps=len(mixed),plots=plots,excluded_responses=source_summary['excluded'],digit_strata=[dict(endpoint=k[0],gt_digits=k[1],generated_digits=k[2],n=v) for k,v in sorted(counts.items())],contrasts=contrasts,validation=dict(role_examples_checked=True,one_step_per_single_role_per_endpoint=True,prefix_reconstruction=True,original_entropy_and_endpoint_labels_reused=True))
    write('summary.json',summary)
    lines=['# 자릿수·토큰 역할별 엔트로피: 반대쪽 시각 GT 고정','',f'기존 {len(endpoint_rows)//2}개 응답의 시작·종료 {len(endpoint_rows)}개 endpoint를 재분석했다. 기존 분석에서 제외한 MM:SS 형식 6개 응답은 그대로 제외한다. 모델 추론 없이 저장된 엔트로피와 endpoint 독립 판정을 재사용했다.','',
        '**그룹:** 모델 × 벤치마크 × 시작/종료 × GT 정수부 자릿수 × 실제 생성 정수부 자릿수. GT와 생성 자릿수가 다른 경우를 별도 그림으로 분리했다. 같은 그룹 안에서 백·십·일의 자리, 소수점, 소수 첫째 자리, 종료 구분자를 비교한다.','',
        '**점 하나는 해당 토큰을 고르기 직전의 전체 어휘 raw 엔트로피**다. 시간축 평균이나 최초 실패 위치 정렬을 하지 않았다. 파랑은 IoU≥0.5 정답 가능성을 유지한 step, 빨강은 해당 endpoint에서 처음 잃은 step, 회색은 이미 잃은 이후다. 짧은 가로선은 같은 역할/상태의 중앙값이다. 최초 실패가 없는 성공 응답도 파란 점으로 포함한다.','',
        '역할은 **최종 생성된 숫자 문자열을 기준으로 사후 지정**한다. 예를 들어 `25.0`의 `2`는 십의 자리, `5`는 일의 자리, `.`은 소수점, `0`은 소수 첫째 자리다. 생성 당시 모델이 숫자의 길이를 미리 확정했다는 뜻은 아니다. 자릿수 분류는 반올림 전 표기를 사용하며, IoU 평가는 기존과 동일하게 예측만 반올림한다. 예: `99.9`는 생성 자릿수 2지만 반올림 후 100이 된다. GT는 원 정밀도로 고정한다.','',
        f'여러 역할을 한 토큰이 동시에 포함하면 entropy를 역할별로 복제하지 않고 `multi_role`로 분리했다. 해당 step은 {len(mixed)}개다. 단일 역할 그림에서는 제외하고 원시 CSV에는 보존했다.','',
        '## GT와 생성 자릿수가 같은 경우','']
    for p in plots:
        if p['gt_digits']==p['generated_digits']:lines += [f"### {p['endpoint']} · {p['gt_digits']}자리",'',f"![{p['endpoint']} {p['gt_digits']} digits]({p['path']})",'']
    lines+=['## GT와 생성 자릿수가 다른 경우','', '| 평가 대상 | GT 자릿수 | 생성 자릿수 | endpoint N (두 모델 합산) | 그림 |','|---|---:|---:|---:|---|']
    for p in plots:
        if p['gt_digits']!=p['generated_digits']:lines.append(f"| {p['endpoint']} | {p['gt_digits']} | {p['generated_digits']} | {counts[(p['endpoint'],p['gt_digits'],p['generated_digits'])]} | [보기]({p['path']}) |")
    lines+=['','## 같은 역할 내 최초 실패 vs 가능성 유지','',
        '아래는 두 상태에 각각 최소 5개 관측이 있는 셀이다. 평균은 동일 역할의 서로 다른 query 간 요약이며, 한 응답의 여러 생성 step을 평균한 값이 아니다. 이 수량 기준은 유의성 검정이 아니다. 나머지 희소 셀을 포함한 전체 통계는 CSV에 보존했다.','',
        '| 모델 | 벤치마크 | 대상 | GT/생성 자릿수 | 역할 | 유지 N / 평균 H | 최초실패 N / 평균 H | 차이 H |','|---|---|---|---|---|---|---|---:|']
    useful=[r for r in contrasts if r['enough_for_descriptive_comparison'] and r['role'] in ROLES]
    for r in useful:lines.append(f"| {r['model']} | {r['dataset']} | {r['endpoint']} | {r['gt_digits']}/{r['generated_digits']} | {NAMES[r['role']]} | {r['n_retained']} / {r['retained_mean']:.3f} | {r['n_first_loss']} / {r['first_loss_mean']:.3f} | {r['difference']:+.3f} |")
    if not useful:lines.append('| 충분한 관측이 있는 셀 없음 | | | | | | | |')
    lines+=['',f'두 상태 각각 5개 이상인 단일 역할 셀은 {len(useful)}개다. 자릿수·역할을 분리하면 셀당 관측이 크게 줄어든다. 빨간 점이 파란 점보다 항상 높다고 전제하지 말고, 같은 패널·같은 역할의 겹침과 표본 수를 확인해야 한다.','',
        '이 분석은 위치/자릿수 혼합을 줄였지만 GT 구간 길이·영상 내용·이미 생성된 prefix까지 통제한 인과 실험은 아니다. 종료 분포는 여전히 실제 생성한 시작 시각에 조건화되어 있고, GT 고정은 평가에만 적용했다. 기존의 초 단위/소수 한 자리 prefix 가능성 판정 범위와 Decord 한계도 유지된다.','',
        '[모든 step·역할 CSV](per_step_roles.csv) · [모든 역할별 통계 CSV](role_statistics.csv) · [유지/최초실패 비교 CSV](first_loss_vs_retained.csv) · [endpoint 자릿수 CSV](per_endpoint_digits.csv) · [집계 및 그림 목록 JSON](summary.json)']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n');write('status.json',dict(state='complete',completed=len(patterns),total=len(patterns),percent=100,responses=len(endpoint_rows)//2,endpoint_evaluations=len(endpoint_rows),steps=len(rows),plots=len(plots),elapsed_seconds=time.time()-started))
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write('status.json',dict(state='failed',error=traceback.format_exc()));raise
