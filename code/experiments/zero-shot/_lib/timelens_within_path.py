"""Within-trajectory uncertainty comparisons from saved top-30 digit analyses."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from artifact_store import register
from result_layout import relocate_path
from token_uncertainty_metrics import DEFINITION
from token_uncertainty_output import write_result_text
from within_path_explanation import explain

ROOT=Path('/home/irteam/experiments/zero-shot')
FAMILY=Path('uncertainty/token_analysis')
RESULT=ROOT/'results'/FAMILY
ART=ROOT/'artifacts/runs'/FAMILY
ANALYSIS=Path('digit_correctness/within_path')
JOB=ART/'full/9404'/ANALYSIS
UPSTREAM=ART/'full/9404/top30_renormalized/metadata/report_refresh_status.json'
SEED=42


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def status(stage,**extra):
    row=dict(state='running',stage=stage,timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    row.update(extra)
    save(JOB/'metadata/status.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def read_steps(scope):
    source=relocate_path(RESULT/scope/'digit_correctness/per_step.csv')
    with source.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for field in ('numeric','previous_correct','first_error','prefix_correct','all_correct'):
            r[field]=r[field]=='True'
        for field in ('entropy','processed_entropy'):r[field]=float(r[field])
        for field in ('ar_step','endpoint_step'):r[field]=int(r[field])
        r['digit_ordinal']=int(r['digit_ordinal']) if r['digit_ordinal'] else None
        assert -1e-10<=r['entropy']<=math.log(30)+1e-10
    return source,rows


def path_records(rows):
    by=defaultdict(list);response=defaultdict(list)
    for r in rows:
        by[(r['model'],r['sample_key'],r['endpoint'])].append(r)
        if r['numeric'] and r['previous_correct']:
            response[(r['model'],r['sample_key'])].append(r['entropy'])
    records=[]
    for key,steps in sorted(by.items()):
        steps=sorted(steps,key=lambda r:r['ar_step'])
        numeric=[r for r in steps if r['numeric']]
        risk=[r for r in numeric if r['previous_correct']]
        failures=[r for r in steps if r['first_error']]
        assert len(failures)<=1
        error=failures[0] if failures else None
        prior=[r for r in numeric if error and r['ar_step']<error['ar_step'] and r['prefix_correct']]
        pair=bool(error and error['numeric'] and prior)
        item=dict(model=key[0],dataset=steps[0]['dataset'],sample_key=key[1],endpoint=key[2],
            numeric_steps=len(numeric),risk_steps=len(risk),prior_safe_steps=len(prior),
            risk_mean_entropy=float(np.mean([r['entropy'] for r in risk])),
            response_risk_mean_entropy=float(np.mean(response[key[:2]])),
            error_kind='none' if error is None else ('numeric' if error['numeric'] else 'format'),
            error_step=error['ar_step'] if error else None,error_role=error['role'] if error else None,
            paired=pair)
        if pair:
            item.update(previous_step=prior[-1]['ar_step'],previous_role=prior[-1]['role'],
                error_entropy=error['entropy'],previous_entropy=prior[-1]['entropy'],
                prior_safe_mean_entropy=float(np.mean([r['entropy'] for r in prior])),
                delta_previous=error['entropy']-prior[-1]['entropy'],
                delta_prior_mean=error['entropy']-float(np.mean([r['entropy'] for r in prior])),
                processed_delta_previous=error['processed_entropy']-prior[-1]['processed_entropy'],
                ordinal_gap=error['digit_ordinal']-prior[-1]['digit_ordinal'],
                error_rank_vs_prior=float(np.mean([float(error['entropy']>r['entropy'])+.5*float(error['entropy']==r['entropy']) for r in prior])))
        records.append(item)
    return records


def bootstrap(values,rng):
    a=np.asarray(values,dtype=float)
    if not len(a):return None,None,None
    draws=[]
    for _ in range(20):draws.extend(rng.choice(a,size=(100,len(a)),replace=True).mean(axis=1))
    lo,hi=np.quantile(draws,[.025,.975])
    return float(a.mean()),float(lo),float(hi)


def paired_groups(paths):
    rng=np.random.default_rng(SEED);groups=[]
    for model,ds in sorted({(r['model'],r['dataset']) for r in paths}):
        for ep in ('start','end','all'):
            base=[r for r in paths if r['model']==model and r['dataset']==ds and (ep=='all' or r['endpoint']==ep)]
            pairs=[r for r in base if r['paired']]
            row=dict(model=model,dataset=ds,endpoint=ep,endpoint_paths=len(base),paired_paths=len(pairs),
                no_error_paths=sum(r['error_kind']=='none' for r in base),
                format_error_paths=sum(r['error_kind']=='format' for r in base),
                numeric_error_without_predecessor=sum(r['error_kind']=='numeric' and not r['paired'] for r in base))
            for metric in ('delta_previous','delta_prior_mean','processed_delta_previous','error_rank_vs_prior'):
                by=defaultdict(list)
                for r in pairs:by[r['sample_key']].append(r[metric])
                values=[float(np.mean(v)) for v in by.values()]
                estimate,lo,hi=bootstrap(values,rng)
                row[metric]=dict(mean=estimate,ci95=[lo,hi],queries=len(values))
            groups.append(row)
    return groups


def fixed_effect(rows,kind):
    """Query-cluster sandwich after weighted within transformation; no causal claim."""
    n=len(rows)
    y=np.array([r['entropy'] for r in rows]);error=np.array([r['first_error'] for r in rows],dtype=float)
    cols=[error];names=['first_error']
    controls=['endpoint'] if kind in ('pooled','response') else []
    if kind=='endpoint_role_position':controls=['role','digit_ordinal']
    for key in controls:
        levels=sorted({str(r[key]) for r in rows})
        for level in levels[1:]:
            cols.append(np.array([str(r[key])==level for r in rows],dtype=float));names.append(key+'='+level)
    if kind=='pooled':cols.append(np.ones(n));names.append('intercept')
    x=np.column_stack(cols)
    grouping=defaultdict(list)
    for i,r in enumerate(rows):
        key=(r['sample_key'],r['endpoint']) if kind.startswith('endpoint') else (r['sample_key'],)
        grouping[key].append(i)
    weights=np.zeros(n);varying=0
    for idx in grouping.values():
        weights[idx]=1/len(idx)
        varying+=int(np.ptp(error[idx])>0)
        if kind!='pooled':
            x[idx]-=x[idx].mean(axis=0);y[idx]-=y[idx].mean()
    xw=x*np.sqrt(weights[:,None]);yw=y*np.sqrt(weights)
    beta,_,rank,_=np.linalg.lstsq(xw,yw,rcond=None)
    # The first-error coefficient must be separately identified after controls.
    other_rank=np.linalg.matrix_rank(xw[:,1:]) if xw.shape[1]>1 else 0
    result=dict(kind=kind,steps=n,queries=len({r['sample_key'] for r in rows}),
                centering_groups=len(grouping),groups_with_error_variation=varying,design_rank=int(rank),
                columns=names,identified=bool(rank>other_rank))
    if not result['identified']:return dict(result,beta=None,ci95=[None,None])
    residual=yw-xw@beta;bread=np.linalg.pinv(xw.T@xw)
    cluster=defaultdict(list)
    for i,r in enumerate(rows):cluster[r['sample_key']].append(i)
    meat=np.zeros((xw.shape[1],xw.shape[1]))
    for idx in cluster.values():
        score=xw[idx].T@residual[idx];meat+=np.outer(score,score)
    g=len(cluster)
    if g<=1:return dict(result,beta=float(beta[0]),ci95=[None,None])
    covariance=(g/(g-1))*bread@meat@bread
    se=float(np.sqrt(max(0,covariance[0,0])))
    margin=float(student_t.ppf(.975,g-1))*se
    return dict(result,beta=float(beta[0]),ci95=[float(beta[0]-margin),float(beta[0]+margin)],cluster_se=se)


def regressions(rows):
    risk=[r for r in rows if r['numeric'] and r['previous_correct']]
    groups=[]
    for model,ds in sorted({(r['model'],r['dataset']) for r in risk}):
        selected=[r for r in risk if (r['model'],r['dataset'])==(model,ds)]
        for kind in ('pooled','response','endpoint','endpoint_role_position'):
            groups.append(dict(model=model,dataset=ds,**fixed_effect(selected,kind)))
    return risk,groups


def variance_groups(risk):
    groups=[]
    for model,ds in sorted({(r['model'],r['dataset']) for r in risk}):
        by=defaultdict(list)
        for r in risk:
            if (r['model'],r['dataset'])==(model,ds):by[r['sample_key']].append(r['entropy'])
        means=[np.mean(v) for v in by.values()]
        between=float(np.var(means));within=float(np.mean([np.var(v) for v in by.values()]))
        total=between+within
        groups.append(dict(model=model,dataset=ds,queries=len(by),between_variance=between,
                           within_variance=within,between_fraction=between/total if total else None))
    return groups


def f(value):return 'NA' if value is None else f'{value:.4f}'


def interval(row):return f"{f(row['mean'])} [{f(row['ci95'][0])}, {f(row['ci95'][1])}]"


def report(scope,paths,paired,regression,variances,source,rows):
    out=ART/scope/ANALYSIS;dest=RESULT/scope/ANALYSIS
    dest.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(parents=True,exist_ok=True)
    count=int(scope.split('/')[1]);mode=scope.split('/')[0]
    save(out/'metrics/per_path.json',paths)
    save(out/'metrics/summary.json',dict(scope=scope,queries=count,source_responses=len({(r['model'],r['sample_key']) for r in rows}),
        paired=paired,regressions=regression,variance_decomposition=variances,bootstrap_seed=SEED,bootstrap_repeats=2000))
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,model in zip(axes,('qwen25','qwen3')):
        selected=[r for r in paired if r['model']==model and r['endpoint']!='all']
        for j,r in enumerate(selected):
            for shift,key,color in [(-.12,'delta_previous','#247ba0'),(.12,'delta_prior_mean','#c75a38')]:
                stats=r[key]
                if stats['mean'] is None:continue
                lo,hi=stats['ci95'];mu=stats['mean']
                ax.errorbar(j+shift,mu,yerr=[[mu-lo],[hi-mu]],fmt='o',color=color,capsize=3,
                            label=key if j==0 else None)
        ax.axhline(0,color='gray',linewidth=1);ax.set_title(model)
        ax.set_xticks(range(len(selected)),[r['dataset'].split('-')[0]+'\n'+r['endpoint'] for r in selected],rotation=25)
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('First-error H30 minus earlier-safe H30 (nats)')
    fig.tight_layout();fig.savefig(out/'figures/paired_entropy.svg');plt.close(fig)
    lines=[f'# Within-path entropy 비교 · {mode} · {count:,} queries','',DEFINITION,'',
        f'동일 {count:,} query × 두 모델의 저장된 greedy decoding을 사용한다. '
        f'지원 표기의 {len({(r["model"],r["sample_key"]) for r in rows}):,}개 응답, '
        f'{len(paths):,}개 endpoint 경로가 분석 대상이다. 모델·dataset을 섞지 않는다.','',
        '## 무엇을 비교했나','',
        '주 분석 단위는 **model × query × endpoint**다. 같은 start 또는 end 경로에서 최초 오류가 숫자 step인 경우, '
        'Δprevious = H₃₀(first error) − H₃₀(직전 정답 가능 numeric step), '
        'Δprior = H₃₀(first error) − mean H₃₀(이전 정답 가능 numeric steps)를 계산한다. '
        '직전은 numeric step 순서 기준이며 중간 소수점·서식 step이 있을 수 있다. 양수면 같은 경로의 이전 숫자보다 최초 오류 때 더 불확실했다는 뜻이다. '
        '최초 오류 이후 step은 주 비교에서 제외한다. 첫 숫자부터 오류인 경로와 서식에서 처음 오류가 난 경로는 paired 비교에서 제외하고 수를 표시한다.','',
        '표의 N은 paired query 수다. start/end를 합치는 보조 집계는 query 안에서 두 Δ를 먼저 평균한다. '
        '95% CI는 query 단위 percentile bootstrap 2,000회(seed 42)다. 여러 query가 같은 영상을 공유할 수 있어 video 단위 독립성을 보장하지 않는다. '
        '여러 비교의 유의성 보정은 하지 않은 탐색적 분석이다.','',
        '## 같은 endpoint 경로의 paired 결과','',
        '| 모델 | dataset | endpoint | N paired query | Δprevious (nat) [95% CI] | Δprior (nat) [95% CI] |',
        '|---|---|---|---:|---:|---:|']
    for r in paired:
        if r['endpoint']=='all':continue
        lines.append(f"| {r['model']} | {r['dataset']} | {r['endpoint']} | {r['delta_previous']['queries']} | {interval(r['delta_previous'])} | {interval(r['delta_prior_mean'])} |")
    lines+=['','![같은 경로의 entropy 차이]('+os.path.relpath(out/'figures/paired_entropy.svg',dest)+')','',
        '## 경로별 기준 수준과 생성 위치를 통제한 비교','',
        'risk set은 numeric이며 선택 직전까지 정답 가능했던 step이다. 유지 step과 최초 numeric 오류 step만 포함한다. '
        'Hᵢₜ = αᵢ + β·first_errorᵢₜ + γ·roleᵢₜ + δ·digit_ordinalᵢₜ + εᵢₜ로 비교한다. '
        'response FE는 query·model별 평균을 빼고 start/end indicator를 통제한다. endpoint FE는 query·model·endpoint별 평균을 뺀다. '
        '위치 통제 모형은 endpoint FE에 실제 출력 자릿수 role과 numeric 순서의 categorical 효과를 더한다. '
        'pooled는 response를 같은 가중치로 두고 endpoint indicator만 통제한다. 각 FE 모형은 해당 경로 안의 risk step 가중치를 1/nᵢ로 둔다. '
        '경로 안에서 오류 여부가 변하지 않는 경우 단순 endpoint FE의 β에 직접 기여하지 않는다. role·순서 효과는 다른 경로와 함께 추정하므로 완전히 같은 자릿수의 matched comparison은 아니다.','',
        'β의 단위는 nat이며 양수면 최초 오류 step의 entropy가 더 높다. CI는 query-cluster sandwich covariance에 G/(G−1) 보정과 t(G−1) 임계값을 사용했다. '
        'β가 별도로 식별되지 않으면 NA다. 같은 관측 risk set을 사용하지만 모형별 경로 가중치와 통제 조건은 다르다.','',
        '| 모델 | dataset | 모형 | risk step 수 | query 수 | β (nat) [95% CI] |',
        '|---|---|---|---:|---:|---:|']
    for r in regression:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['kind']} | {r['steps']} | {r['queries']} | {f(r['beta'])} [{f(r['ci95'][0])}, {f(r['ci95'][1])}] |")
    comparable=[r for r in paired if r['endpoint']!='all' and r['delta_previous']['mean'] is not None]
    positive=sum(r['delta_previous']['ci95'][0]>0 for r in comparable)
    negative=sum(r['delta_previous']['ci95'][1]<0 for r in comparable)
    lines+=['','## 해석과 한계','',
        f'Δprevious의 CI가 0보다 큰 model·dataset·endpoint 그룹은 {positive}/{len(comparable)}개, '
        f'0보다 작은 그룹은 {negative}/{len(comparable)}개다. 나머지는 이 분석에서 방향이 뚜렷하지 않다. '
        '그룹별 paired 결과와 위치 통제 β를 함께 읽는다.','',
        '경로 평균을 빼는 것은 sample마다 일정한 additive baseline 차이를 제거한다. '
        '**sample 고유 불확실성 자체를 측정하거나 모두 제거했다는 뜻은 아니다.** '
        '이전 prefix·자릿수·후보 집합은 step마다 달라지고, 각 query에는 모델별 저장된 greedy 경로 한 개만 있다. '
        '오류가 발생한 경로 및 이전에 정답 가능한 숫자가 있는 경로에 조건화한 비교이므로 오류 예측 성능이나 인과 효과로 해석하지 않는다. '
        '경로 평균·role은 완성된 경로를 사용하는 사후 통계이며 온라인 탐지기나 test-set 성능이 아니다.','',
        '[경로별 interactive 비교](report.html) · [제외 범위·분산 분해·보조 지표](APPENDIX.md)','',
        '[경로별 파생 지표]('+os.path.relpath(out/'metrics/per_path.json',dest)+') · '
        '[전체 집계]('+os.path.relpath(out/'metrics/summary.json',dest)+')']
    appendix=['# Within-path 보조 결과','',DEFINITION,'',
        '| 모델 | dataset | endpoint | 경로 수 | paired | 오류 없음 | 서식 최초 오류 | numeric 오류·이전 숫자 없음 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in paired:
        if r['endpoint']=='all':continue
        appendix.append(f"| {r['model']} | {r['dataset']} | {r['endpoint']} | {r['endpoint_paths']} | {r['paired_paths']} | {r['no_error_paths']} | {r['format_error_paths']} | {r['numeric_error_without_predecessor']} |")
    appendix+=['','## 경로 간·경로 내부 분산','',
        'response별 risk step 평균 μᵢ와 분산 σ²ᵢ를 구한다. 경로에 같은 가중치를 부여하면 총 분산은 '
        'Varᵢ(μᵢ) + meanᵢ(σ²ᵢ)다. 아래 between 비중은 앞 항을 총 분산으로 나눈 값이다. '
        '경로 간 항에는 sample뿐 아니라 숫자·길이·prefix 구성 차이가 섞여 있으므로 intrinsic uncertainty 비중이라고 부르지 않는다.','',
        '| 모델 | dataset | query 수 | between variance (nat²) | within variance (nat²) | between 비중 |',
        '|---|---|---:|---:|---:|---:|']
    for r in variances:appendix.append(f"| {r['model']} | {r['dataset']} | {r['queries']} | {f(r['between_variance'])} | {f(r['within_variance'])} | {f(r['between_fraction'])} |")
    appendix+=['','## Processed 및 경로 내부 순위','',
        'rank는 최초 오류 H가 이전 safe numeric H보다 큰 쌍의 비율이며 동률은 0.5로 센다. '
        '0.5 초과면 오류가 이전 숫자보다 대체로 높다. 이 값도 query 내부 평균 후 query 간 평균한다.','',
        '| 모델 | dataset | endpoint | processed Δprevious [95% CI] | raw rank [95% CI] |',
        '|---|---|---|---:|---:|']
    for r in paired:appendix.append(f"| {r['model']} | {r['dataset']} | {r['endpoint']} | {interval(r['processed_delta_previous'])} | {interval(r['error_rank_vs_prior'])} |")
    lines,appendix=explain(lines,appendix,scope,paths,paired,regression,variances)
    write_result_text(dest/'REPORT.md','\n'.join(lines)+'\n')
    write_result_text(dest/'APPENDIX.md','\n'.join(appendix)+'\n')
    build_viewer(scope,out,dest)
    save(out/'metadata/provenance.json',dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        producer=Path(__file__).name,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        explanation_producer='within_path_explanation.py',
        explanation_sha256=hashlib.sha256(Path(__file__).with_name('within_path_explanation.py').read_bytes()).hexdigest(),
        viewer_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'_lib/token_viewer/report.html',ROOT/'_lib/token_viewer/viewer.js',ROOT/'_lib/token_viewer/style.css',ROOT/'_lib/publish_token_viewer.py']},
        entropy_basis='top30_renormalized',bootstrap_seed=SEED,bootstrap_repeats=2000,queries=count))


def build_viewer(scope,out,dest):
    from publish_token_viewer import publish
    publish()


def self_check():
    rows=[]
    for sample,offset in [('a',.1),('b',1.5),('c',.7)]:
        for j in range(3):
            rows.append(dict(model='m',dataset='d',sample_key=sample,endpoint='start',ar_step=j+1,
                entropy=offset+.2*j,processed_entropy=offset+.2*j,numeric=True,previous_correct=True,
                prefix_correct=j<2,first_error=j==2,role=['tens','ones','tenths'][j],digit_ordinal=j+1))
    paths=path_records(rows)
    assert all(abs(r['delta_previous']-.2)<1e-12 and abs(r['delta_prior_mean']-.3)<1e-12 for r in paths)
    baseline=fixed_effect(rows,'endpoint')['beta']
    shifted=[dict(r,entropy=r['entropy']+{'a':10,'b':-2,'c':4}[r['sample_key']]) for r in rows]
    assert abs(fixed_effect(shifted,'endpoint')['beta']-baseline)<1e-10
    assert not fixed_effect(rows,'endpoint_role_position')['identified']
    return dict(passed=True,paired_formula=True,additive_offset_invariance=True,collinearity_detected=True)


def refresh_reports():
    """Regenerate prose from saved statistics without rerunning estimation."""
    for scope in ('pilot/300','full/9404'):
        out=ART/scope/ANALYSIS
        files=[out/'metrics/summary.json',out/'metrics/per_path.json']
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        data=json.loads(files[0].read_text());paths=json.loads(files[1].read_text())
        source,rows=read_steps(scope)
        provenance=json.loads((out/'metadata/provenance.json').read_text())
        assert hashlib.sha256(source.read_bytes()).hexdigest()==provenance['source_sha256']
        report(scope,paths,data['paired'],data['regressions'],data['variance_decomposition'],source,rows)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==before[p] for p in files)
        print(scope+': reports refreshed; source and statistics unchanged',flush=True)


def build(scope):
    source,rows=read_steps(scope);paths=path_records(rows)
    status('paired_comparison',scope=scope,completed_queries=len({r['sample_key'] for r in rows}),total_queries=int(scope.split('/')[1]))
    paired=paired_groups(paths);risk,regression=regressions(rows);variances=variance_groups(risk)
    for r in paths:
        if r['paired']:
            assert r['previous_step']<r['error_step'] and abs(r['delta_previous']-(r['error_entropy']-r['previous_entropy']))<1e-12
    report(scope,paths,paired,regression,variances,source,rows)
    save(ART/scope/ANALYSIS/'validation/numerical.json',dict(passed=True,endpoint_paths=len(paths),
        risk_steps=len(risk),paired_paths=sum(r['paired'] for r in paths),self_check=self_check()))


def browser_check():
    from verify_within_path_viewer import check
    return check()


def run():
    try:
        self_check()
        while True:
            upstream=json.loads(UPSTREAM.read_text())
            if upstream['state']=='complete':break
            if upstream['state']=='failed':raise RuntimeError('Upstream report refresh failed: '+upstream.get('error',''))
            status('waiting_for_top30_reports',upstream_stage=upstream.get('stage'));time.sleep(20)
        for scope in ('pilot/300','full/9404'):build(scope)
        from entropy_error_appendix import build as build_error_appendix
        for scope in ('pilot/300','full/9404'):build_error_appendix(scope)
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
        browser_check()
        for scope in ('pilot/300','full/9404'):
            dest=RESULT/scope/ANALYSIS
            for name in ('REPORT.md','APPENDIX.md'):
                with urlopen('http://127.0.0.1:8780/'+str((dest/name).relative_to(ROOT.parent)),timeout=10) as response:assert response.status==200
            register('analysis.'+scope.replace('/','')+'.within_path','analyses',ART/scope/ANALYSIS,
                     'Within-path paired and fixed-effect comparisons of top-30 entropy',Path(__file__).name,scope,'complete')
        status('complete',state='complete')
    except Exception:
        status('failed',state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['launch','run','check','report']);args=parser.parse_args()
    if args.command=='check':print(json.dumps(self_check()))
    elif args.command=='report':refresh_reports()
    elif args.command=='run':run()
    else:
        (JOB/'logs').mkdir(parents=True,exist_ok=True)
        with (JOB/'logs/run.log').open('a') as stream:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,TZ='Asia/Seoul',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'))
        print(json.dumps(dict(pid=process.pid,log=str(JOB/'logs/run.log'),status=str(JOB/'metadata/status.json'))))
