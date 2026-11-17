"""Reproducible descriptive and within-path entropy/error appendix; no model inference."""
from collections import defaultdict, Counter
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from result_layout import relocate_path
from token_uncertainty_output import write_result_text

ROOT=Path('/home/irteam/experiments/zero-shot')
RESULT=ROOT/'results/uncertainty/token_analysis'
ART=ROOT/'artifacts/runs/uncertainty/token_analysis'
ANALYSIS=Path('entropy_error')
SEED=42
REPEATS=1000
BASES=('token','numeric')
BASE_NAMES={'token':'직전 decoding token','numeric':'직전 정답 가능 숫자'}
FILTER_NAMES={'all':'전체 경로','paired':'오류 전후 숫자 비교 가능','integer':'정수부 최초 오류','fractional':'소수부 최초 오류','format':'소수점·서식 최초 오류','none':'오류 없음','increase':'직전 숫자 대비 상승','decrease':'직전 숫자 대비 감소','token_increase':'직전 token 대비 상승','token_decrease':'직전 token 대비 감소','low_error':'낮은 entropy 오류 (H≤0.5)','high_safe':'높은 entropy 유지 (H≥1.5)','next_failure':'유지 직후 다음 숫자 오류','disagree':'두 모델 판정 불일치'}


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'
def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def interval(r):return f"{fmt(r['mean'])} [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]"


def query_stat(rows,field):
    by=defaultdict(list)
    for r in rows:
        if r.get(field) is not None:by[r['sample_key']].append(r[field])
    a=np.array([np.mean(v) for v in by.values()],float)
    if not len(a):return dict(queries=0,mean=None,ci95=[None,None])
    rng=np.random.default_rng(SEED);draws=[]
    for start in range(0,REPEATS,50):draws.extend(rng.choice(a,size=(min(50,REPEATS-start),len(a)),replace=True).mean(axis=1))
    # A singleton supplies a descriptive value, not an estimated population interval.
    ci=[float(x) for x in np.quantile(draws,[.025,.975])] if len(a)>1 else [None,None]
    return dict(queries=len(a),mean=float(a.mean()),ci95=ci)


def class_token(t):
    if t and all(c.isdigit() for c in t):return 'numeric'
    if t.strip()=='.':return 'decimal_point'
    if not t.strip():return 'whitespace'
    return 'other_format'


def sources(scope):
    base=RESULT/scope;out=ART/scope
    return dict(digit_steps=relocate_path(base/'digit_correctness/per_step.csv'),
                all_steps=out/'top30_renormalized/metrics/per_step.csv.gz',
                paths=out/'digit_correctness/within_path/metrics/per_path.json',
                catalog=out/'metrics/viewer_catalog.json',
                manifest=relocate_path(base/'digit_correctness/manifest.js'))


def load(scope):
    src=sources(scope)
    with src['digit_steps'].open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('numeric','first_error','previous_correct','prefix_correct'):r[k]=r[k]=='True'
        for k in ('ar_step','endpoint_step'):r[k]=int(r[k])
        r['entropy']=float(r['entropy']);r['digit_ordinal']=int(r['digit_ordinal']) if r['digit_ordinal'] else 0
    h={}
    with gzip.open(src['all_steps'],'rt') as f:
        for r in csv.DictReader(f):
            if r['mode']=='raw':h[(r['model'],r['sample_key'],int(r['step']))]=float(r['top30_entropy_nats'])
    catalog=json.loads(src['catalog'].read_text())
    text=src['manifest'].read_text();manifest,_=json.JSONDecoder().raw_decode(text,text.index('=')+1)
    samples={s['sample_key']:s for s in manifest}
    by=defaultdict(list)
    for r in rows:by[(r['model'],r['sample_key'],r['endpoint'])].append(r)
    paths=[];risk=[];checked=0
    for key,steps in sorted(by.items()):
        steps.sort(key=lambda r:r['ar_step']);model,sample,endpoint=key
        cat=catalog[model+'/'+sample]['endpoints'][endpoint]
        p=dict(cat,model=model,dataset=steps[0]['dataset'],sample_key=sample,endpoint=endpoint)
        m=samples[sample]['models'];p['disagree']=bool(m['qwen25']['supported'] and m['qwen3']['supported'] and m['qwen25'][endpoint+'_correct']!=m['qwen3'][endpoint+'_correct'])
        prev_numeric=None;path_risk=[]
        step_map={r['ar_step']:r for r in steps}
        assert sum(r['first_error'] for r in steps)<=1
        for r in steps:
            k=r['ar_step'];hh=h[(model,sample,k)]
            assert abs(hh-r['entropy'])<1e-9
            checked+=1
            if not r['previous_correct']:continue
            previous_h=h.get((model,sample,k-1))
            previous_r=step_map.get(k-1)
            rr={k:r[k] for k in ('model','dataset','sample_key','endpoint','ar_step','role','digit_ordinal','numeric','first_error','entropy')}
            rr['region']='fractional' if r['role']=='tenths' else 'integer' if r['numeric'] else 'format'
            rr['previous_kind']=class_token(previous_r['token']) if previous_r else 'outside_endpoint'
            rr['token_delta']=hh-previous_h if previous_h is not None else None
            rr['numeric_delta']=hh-prev_numeric['entropy'] if r['numeric'] and prev_numeric else None
            path_risk.append(rr);risk.append(rr)
            if r['numeric'] and r['prefix_correct']:prev_numeric=r
        for basis in BASES:
            eligible=[r for r in path_risk if r[basis+'_delta'] is not None]
            errors=[r for r in eligible if r['first_error']];safe=[r for r in eligible if not r['first_error']]
            assert len(errors)<=1
            p[basis+'_risk_steps']=len(eligible);p[basis+'_safe_steps']=len(safe)
            p[basis+'_error_delta']=errors[0][basis+'_delta'] if errors else None
            p[basis+'_safe_delta']=float(np.mean([r[basis+'_delta'] for r in safe])) if safe else None
            p[basis+'_matched']=p[basis+'_error_delta']-p[basis+'_safe_delta'] if errors and safe else None
            p[basis+'_error_rise']=float(errors[0][basis+'_delta']>0) if errors else None
            p[basis+'_safe_rise']=float(np.mean([r[basis+'_delta']>0 for r in safe])) if safe else None
            p[basis+'_rise_gap']=p[basis+'_error_rise']-p[basis+'_safe_rise'] if errors and safe else None
        if cat['delta_previous'] is not None:assert abs(p['numeric_error_delta']-cat['delta_previous'])<1e-9
        if cat['delta_previous_token'] is not None:assert abs(p['token_error_delta']-cat['delta_previous_token'])<1e-9
        paths.append(p)
    return src,rows,risk,paths,dict(queries=len(samples),responses=len({(r['model'],r['sample_key']) for r in rows}),endpoint_paths=len(paths),source_steps=len(rows),risk_steps=len(risk),matched_entropy_checks=checked,unique_videos=len({s['video_path'] for s in manifest}))


def filter_path(p,name):
    if name=='all':return True
    if name in ('integer','fractional','format','none'):return p['error_region']==name
    if name=='increase':return p['delta_previous'] is not None and p['delta_previous']>0
    if name=='decrease':return p['delta_previous'] is not None and p['delta_previous']<0
    if name=='token_increase':return p['delta_previous_token'] is not None and p['delta_previous_token']>0
    if name=='token_decrease':return p['delta_previous_token'] is not None and p['delta_previous_token']<0
    if name=='low_error':return p['token_error_entropy'] is not None and p['token_error_entropy']<=.5
    if name=='high_safe':return p['safe_high']
    return p[name]


def fixed_effect(rows,basis):
    selected=[r for r in rows if r[basis+'_delta'] is not None]
    if not selected:return dict(beta=None,ci95=[None,None],queries=0,paths=0,steps=0,identified=False)
    group=defaultdict(list)
    for i,r in enumerate(selected):group[(r['sample_key'],r['endpoint'])].append(i)
    # Only paths containing both safe and error steps identify the error contrast.
    selected=[r for r in selected if any(selected[j]['first_error'] for j in group[(r['sample_key'],r['endpoint'])]) and any(not selected[j]['first_error'] for j in group[(r['sample_key'],r['endpoint'])])]
    if not selected:return dict(beta=None,ci95=[None,None],queries=0,paths=0,steps=0,identified=False)
    group=defaultdict(list);cluster=defaultdict(list)
    for i,r in enumerate(selected):group[(r['sample_key'],r['endpoint'])].append(i);cluster[r['sample_key']].append(i)
    y=np.array([r[basis+'_delta'] for r in selected]);cols=[np.array([r['first_error'] for r in selected],float)];names=['first_error']
    for field in ('role','previous_kind','digit_ordinal'):
        levels=sorted({str(r[field]) for r in selected})
        for level in levels[1:]:cols.append(np.array([str(r[field])==level for r in selected],float));names.append(field+'='+level)
    x=np.column_stack(cols);weights=np.zeros(len(selected))
    for idx in group.values():
        x[idx]-=x[idx].mean(axis=0);y[idx]-=y[idx].mean();weights[idx]=1/len(idx)
    xw=x*np.sqrt(weights[:,None]);yw=y*np.sqrt(weights)
    beta,_,rank,_=np.linalg.lstsq(xw,yw,rcond=None)
    identified=int(rank)>np.linalg.matrix_rank(xw[:,1:])
    result=dict(queries=len(cluster),paths=len(group),steps=len(selected),identified=bool(identified),rank=int(rank),controls=names,beta=None,ci95=[None,None])
    if not identified:return result
    residual=yw-xw@beta;bread=np.linalg.pinv(xw.T@xw);meat=np.zeros((xw.shape[1],xw.shape[1]))
    for idx in cluster.values():score=xw[idx].T@residual[idx];meat+=np.outer(score,score)
    g=len(cluster);result['beta']=float(beta[0])
    if g>1:
        cov=g/(g-1)*bread@meat@bread;se=math.sqrt(max(0,float(cov[0,0])));margin=student_t.ppf(.975,g-1)*se
        result['ci95']=[float(beta[0]-margin),float(beta[0]+margin)]
    return result


def summarize(rows,risk,paths):
    filters=[];paired=[];detection=[];adjusted=[]
    for model in ('qwen25','qwen3'):
        pp=[p for p in paths if p['model']==model]
        for name in FILTER_NAMES:
            rr=[p for p in pp if filter_path(p,name)]
            x=dict(model=model,filter=name,queries=len({p['sample_key'] for p in rr}),paths=len(rr),errors=sum(p['error_kind']!='none' for p in rr))
            for basis in BASES:
                valid=[p for p in rr if p[basis+'_error_delta'] is not None]
                x[basis]=dict(eligible=len(valid),rise=sum(p[basis+'_error_delta']>0 for p in valid),fall=sum(p[basis+'_error_delta']<0 for p in valid),equal=sum(p[basis+'_error_delta']==0 for p in valid),mean_delta=float(np.mean([p[basis+'_error_delta'] for p in valid])) if valid else None)
            filters.append(x)
        for ds in sorted({p['dataset'] for p in pp}):
            dd=[p for p in pp if p['dataset']==ds];steps=[r for r in risk if (r['model'],r['dataset'])==(model,ds)]
            for basis in BASES:
                adjusted.append(dict(model=model,dataset=ds,basis=basis,**fixed_effect(steps,basis)))
                for endpoint in ('start','end'):
                    ps=[p for p in dd if p['endpoint']==endpoint and p[basis+'_matched'] is not None]
                    paired.append(dict(model=model,dataset=ds,endpoint=endpoint,basis=basis,paths=len(ps),contrast=query_stat(ps,basis+'_matched'),rise_gap=query_stat(ps,basis+'_rise_gap')))
                for region in ('all','integer','fractional','format'):
                    ss=[r for r in steps if r[basis+'_delta'] is not None and (region=='all' or r['region']==region)]
                    err=[r for r in ss if r['first_error']];safe=[r for r in ss if not r['first_error']]
                    er=sum(r[basis+'_delta']>0 for r in err);sr=sum(r[basis+'_delta']>0 for r in safe)
                    rise=[r for r in ss if r[basis+'_delta']>0];fall=[r for r in ss if r[basis+'_delta']<=0]
                    detection.append(dict(model=model,dataset=ds,basis=basis,region=region,queries=len({r['sample_key'] for r in ss}),steps=len(ss),error_steps=len(err),safe_steps=len(safe),error_rise=er,safe_rise=sr,sensitivity=er/len(err) if err else None,fpr=sr/len(safe) if safe else None,precision=er/len(rise) if rise else None,nonrise_error_rate=sum(r['first_error'] for r in fall)/len(fall) if fall else None,path_balanced_error_rise=query_stat(dd,basis+'_error_rise') if region=='all' else None,path_balanced_safe_rise=query_stat(dd,basis+'_safe_rise') if region=='all' else None))
    return dict(filters=filters,paired=paired,detection=detection,adjusted=adjusted)


def plot(summary,out):
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,model in zip(axes,('qwen25','qwen3')):
        groups=sorted({(r['dataset'],r['endpoint']) for r in summary['paired'] if r['model']==model})
        for j,(ds,ep) in enumerate(groups):
            for shift,basis,color in [(-.12,'token','#247ba0'),(.12,'numeric','#c75a38')]:
                r=next(r for r in summary['paired'] if (r['model'],r['dataset'],r['endpoint'],r['basis'])==(model,ds,ep,basis));st=r['contrast']
                if st['mean'] is None:continue
                lo,hi=st['ci95'];mu=st['mean'];err=None if lo is None else [[max(0,mu-lo)],[max(0,hi-mu)]]
                ax.errorbar(j+shift,mu,yerr=err,fmt='o',color=color,capsize=3,label=basis if j==0 else None)
        ax.axhline(0,color='#777',linewidth=1);ax.set_title(model);ax.grid(axis='y',alpha=.2)
        ax.set_xticks(range(len(groups)),[d.split('-')[0]+'\n'+e for d,e in groups],rotation=20);ax.legend(title='Previous comparator')
    axes[0].set_ylabel('Error delta - mean earlier-safe delta (nat)')
    fig.tight_layout();out.mkdir(parents=True,exist_ok=True);fig.savefig(out/'within_path_delta_contrast.svg');plt.close(fig)


def report(scope,src,counts,summary):
    dest=RESULT/scope/ANALYSIS;out=ART/scope/ANALYSIS;dest.mkdir(parents=True,exist_ok=True)
    def table(headers,rows,caption):return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows],'','**표 해설.** '+caption,'']
    headline=[]
    for model in ('qwen25','qwen3'):
        for region in ('integer','fractional'):
            rr=[r for r in summary['detection'] if r['model']==model and r['basis']=='token' and r['region']==region]
            errors=sum(r['error_steps'] for r in rr);safe=sum(r['safe_steps'] for r in rr)
            rise=sum(r['error_rise'] for r in rr);false=sum(r['safe_rise'] for r in rr)
            headline.append([model,region,f'{rise}/{errors}',pct(rise/errors if errors else None),f'{false}/{safe}',pct(false/safe if safe else None)])
    lead=['## 먼저 읽을 결과','',
          '오류 때 entropy가 올라가는 사례는 많지만, 숫자 step에서는 정답 가능성을 유지할 때도 자주 올라간다. 특히 소수점 뒤 숫자는 이전 소수점의 entropy가 낮아 상승하기 쉽다. 따라서 상승 자체를 오류 신호로 바로 쓰기 전에 아래 오경보율을 확인해야 한다.','']
    lead+=table(['모델','현재 숫자 위치','오류 상승/오류','민감도','safe 상승/safe','오경보율'],headline,'직전 decoding token 기준이며 모든 dataset·start/end의 해당 숫자 step을 합친 기술통계다. 소수부에서 민감도와 오경보율이 둘 다 높으면 소수점→숫자 전이의 정상적인 상승을 오류와 구분하지 못한다. 정수부도 동일하다. dataset별 수치, 경로 내부 비교, 통제 결과는 뒤 표에 따로 제시한다.')
    for basis in BASES:
        pair=[r['contrast'] for r in summary['paired'] if r['basis']==basis and r['contrast']['mean'] is not None]
        adjusted=[r for r in summary['adjusted'] if r['basis']==basis and r['identified']]
        lead += [f'- {BASE_NAMES[basis]} 기준: Cpath 평균이 양수인 조건 {sum(r["mean"]>0 for r in pair)}/{len(pair)}, 95% CI 전체가 0보다 큰 조건 {sum(r["ci95"][0] is not None and r["ci95"][0]>0 for r in pair)}/{len(pair)}다(model×dataset×endpoint). role·직전 token 종류·숫자 순서 통제 후 β의 CI 전체가 양수인 조건은 식별 가능한 {len(adjusted)}개 model×dataset 중 {sum(r["ci95"][0] is not None and r["ci95"][0]>0 for r in adjusted)}개, 전체가 음수인 조건은 {sum(r["ci95"][1] is not None and r["ci95"][1]<0 for r in adjusted)}개다.']
    lead+=['','같은 경로의 차이가 일부 조건에서 양수여도 상승 여부만으로 오류를 잘 분류한다는 뜻은 아니다. 위 숫자 step의 오경보율과 함께 판단한다.','']
    lead+=['단순한 “entropy 감소 방향의 decoding”은 목표로 충분하지 않다. 후속 실험의 성공 조건은 **관련 영상 증거가 들어왔을 때 정답 가능 후보의 확률이 늘고 오류가 줄어드는가**다. entropy 감소는 그와 함께 확인할 보조 지표이며, 관련 증거 때문에 기존의 잘못된 확신이 깨지면 오히려 entropy가 증가할 수도 있다.','']
    lines=[f'# Entropy 상승과 최초 오류 · {scope.split("/")[0]} · {int(scope.split("/")[1]):,} queries','',
           '**측정된 결과와 후속 실험 제안을 구분한다.** 이 appendix의 앞부분은 기존 greedy decoding 기록의 재집계다. 영상 증거를 바꿔 얻은 결과가 아니므로 영상 활용의 인과 효과는 아직 측정하지 않았다.','',
           f'실제 query 수는 {counts["queries"]:,}, 고유 영상 수는 {counts["unique_videos"]:,}다. 지원 표기의 {counts["responses"]:,}개 model×query 응답에서 {counts["endpoint_paths"]:,}개 model×query×endpoint 경로와 {counts["source_steps"]:,}개 endpoint step을 읽었다. 두 모델의 반복 응답 수를 query 수로 세지 않으며, pilot과 full을 합산하지 않는다.','',
           '## 1. 어떤 값이 오류의 신호인가','',
           'Hₖ는 **raw top-30 재정규화 entropy**다. 해당 step의 top-30 확률 pᵢ를 qᵢ=pᵢ/Σ₁³⁰pᵢ로 바꾸고 Hₖ=−Σ₁³⁰qᵢlnqᵢ를 계산한다. 단위는 nat, 범위는 0–ln30≈3.401이다. top-30 밖의 불확실성은 포함하지 않는다. entropy는 다음 token 선택 **전**의 분포이고 오류 판정은 실제 token을 고른 **후**의 판정이다.','',
           '오류는 허용된 표기법 안에서 반올림 endpoint 정답으로 완성할 수 없게 되는 최초 step이다. 최종 구간 IoU 오류, 영상 이해 오류, 정답 token과의 문자 불일치와 같지 않다. start/end 경로를 분리한다. risk step은 이전 prefix가 정답 가능성을 유지한 step(정답 유지 또는 최초 오류)이며 최초 오류 이후는 모두 제외한다.','',
           '두 변화량을 따로 계산한다. Δtokenₖ=Hₖ−Hₖ₋₁은 바로 앞 decoding token을 쓰며 endpoint 밖 token도 포함한다. Δnumericₖ=Hₖ−Hⱼ는 같은 endpoint 안의 직전 정답 가능 숫자 j를 쓰며 현재 step도 숫자여야 한다. 따라서 첫 숫자와 format 오류에는 Δnumeric이 없다. Δ>0은 상승, Δ<0은 감소, Δ=0은 둘 다 아니다.','',
           '가장 중요한 비교는 **같은 경로에서 오류 때의 변화가 평소 정답 유지 step의 변화보다 컸는가**다. Cpath=Δ(first error)−mean Δ(그 경로의 앞선 safe steps). 오류와 비교 가능한 safe 변화가 둘 다 있는 경로만 포함한다. 양수는 오류 순간에 상대적으로 더 많이 상승했다는 뜻이며, 오류 때 Δ 자체가 양수라는 뜻은 아니다. token 기준의 safe step에는 format도 포함되어 아래 role 통제를 함께 확인해야 한다.','',
           'CI는 model·dataset·endpoint 안에서 query 단위 percentile bootstrap 1,000회(seed 42)다. 두 endpoint를 합친 path-balanced 지표는 query 안에서 먼저 평균한다. 여러 query가 같은 영상을 공유할 수 있어 video 의존성까지 보정하지는 않는다. 단일 query는 CI를 제시하지 않는다. 다중 비교 보정 없는 탐색적 집계이며, 작은 표본 결과를 일반화하지 않는다.','',
           '## 2. 같은 경로의 평소 변화와 최초 오류 변화','']
    first_section=lines.index('## 1. 어떤 값이 오류의 신호인가')
    from saved_decoding_entropy_links import links_from
    extra_links=links_from(dest)
    if extra_links:
        lead=['다른 저장 자료의 추가 확인: '+' · '.join(extra_links), '']+lead
    lines[first_section:first_section]=lead
    lines+=table(['모델','dataset','endpoint','비교 기준','paired 경로 / query','Cpath 평균 [95% CI] nat','상승 비율 차이 pp [95% CI]'],[[r['model'],r['dataset'],r['endpoint'],BASE_NAMES[r['basis']],f"{r['paths']} / {r['contrast']['queries']}",interval(r['contrast']),f"{fmt(None if r['rise_gap']['mean'] is None else 100*r['rise_gap']['mean'],1)} [{fmt(None if r['rise_gap']['ci95'][0] is None else 100*r['rise_gap']['ci95'][0],1)}, {fmt(None if r['rise_gap']['ci95'][1] is None else 100*r['rise_gap']['ci95'][1],1)}]"] for r in summary['paired']], '각 행은 같은 model·dataset·endpoint 안의 비교다. Cpath가 0보다 크면 그 sample에서 정답 가능성을 유지하던 step보다 최초 오류 step의 entropy 변화가 컸다. 상승 비율 차이는 1[오류 Δ>0]−그 경로의 safe step 상승 비율을 query별 평균한 값이다. 이전 safe 변화가 없는 첫 숫자 오류 등은 이 표에서 빠지므로 전체 오류의 탐지 성능으로 해석하면 안 된다.')
    fig=out/'figures/within_path_delta_contrast.svg'
    lines += [f'![Within-path entropy 변화 대비]({os.path.relpath(fig,dest)})','', '**그래프 해설.** 점은 위 표의 Cpath 평균, 오차 막대는 query bootstrap 95% CI다. 파랑(token)은 직전 decoding step, 주황(numeric)은 직전 정답 가능 숫자를 비교 대상으로 쓴다. 0선 위의 점은 오류 순간 변화가 같은 경로의 이전 변화보다 컸다는 뜻이다. 두 색이 다르면 소수점·문법 token과 숫자를 오가는 효과가 섞였는지 확인해야 한다.','', '## 3. 숫자 위치와 직전 token 종류를 통제해도 남는가','',
      'Δₚₖ = αpath + β·1[first error] + 현재 role + 직전 token 종류 + digit ordinal + ε로 비교한다. 각 경로의 평균을 빼서 αpath를 제거하고, 경로별 전체 가중치를 1로 맞춘 weighted least squares를 사용한다. 경로 안에 safe와 error 변화가 모두 있어야 한다. 95% CI는 query-cluster sandwich SE와 t 분포로 계산한다. 이전 token이 endpoint 밖이면 outside_endpoint로 묶고, 나머지는 숫자·소수점·공백·기타 서식으로 분류한다. β가 양수면 이 통제 아래 최초 오류 변화가 더 큰 연관성이 남았다는 뜻이다. 인과효과가 아니다.','']
    lines+=table(['모델','dataset','비교 기준','query / 경로 / step','β [95% CI] nat'],[[r['model'],r['dataset'],BASE_NAMES[r['basis']],f"{r['queries']} / {r['paths']} / {r['steps']}",f"{fmt(r['beta'])} [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]" ] for r in summary['adjusted']], '소수점은 보통 후보가 좁아 entropy가 낮으므로 소수점→숫자라는 이유만으로 Δtoken이 커질 수 있다. 이 표는 현재 숫자/format role과 직전 token 종류, 숫자 순서를 통제한 민감도 분석이다. 단순 paired 결과와 방향이 다르면 “entropy 상승=오류”라는 일반 규칙보다 전이 유형·자릿수의 영향에 주의해야 한다. 범주 통제로 모든 혼란 요인이 제거되는 것은 아니다. β가 —인 행은 통제변수와 오류 위치가 겹쳐 오류 계수를 별도로 식별할 수 없는 경우다. 효과가 0이라는 뜻이 아니다.')
    lines+=['## 4. 상승하면 실제 오류인가: 정답 유지 step도 분모에 넣기','',
       '아래 지표는 사후에 오류인 경로만 고르지 않고 모든 eligible risk step에서 계산한다. 민감도=P(Δ>0 | 최초 오류), 오경보율=P(Δ>0 | safe), precision=P(최초 오류 | Δ>0), 비상승 오류율=P(최초 오류 | Δ≤0)이다. 이 표의 비율은 step 기준이며 긴 경로는 더 많은 step을 기여한다. 배포 성능 검증이나 새로운 threshold 학습은 아니다.','']
    for basis in BASES:
        lines += [f'### {BASE_NAMES[basis]} 기준','']
        selected=[r for r in summary['detection'] if r['basis']==basis and r['region']=='all']
        lines+=table(['모델','dataset','최초 오류 / safe step','오류 상승 / safe 상승','민감도','오경보율','precision','비상승 오류율'],[[r['model'],r['dataset'],f"{r['error_steps']} / {r['safe_steps']}",f"{r['error_rise']} / {r['safe_rise']}",pct(r['sensitivity']),pct(r['fpr']),pct(r['precision']),pct(r['nonrise_error_rate'])] for r in selected], '민감도가 높아도 오경보율이 비슷하게 높으면 오류를 잘 구분하는 신호가 아니다. precision은 오류의 기본 비율과 함께 달라진다. 숫자 기준은 현재 숫자+이전 safe 숫자가 있는 step만 포함하고 token 기준은 format과 첫 숫자도 포함하므로 두 행 집합의 분모가 다르다.')
        lines+=table(['모델','dataset','오류 상승률 query 평균 [95% CI]','safe 상승률 query 평균 [95% CI]'],[[r['model'],r['dataset'],interval(r['path_balanced_error_rise']),interval(r['path_balanced_safe_rise'])] for r in selected], '비율을 먼저 endpoint 경로 안에서 평균하고 두 endpoint가 있으면 query 안에서 다시 평균해 긴 경로의 영향력을 줄였다. 수치는 0–1 비율이다. 오류 열과 safe 열은 자료가 있는 query 집합이 다를 수 있으므로 둘의 차이가 동일 경로 paired 효과는 아니다. 동일 경로 비교는 2절을 본다.')
    lines+=['### 현재 step의 정수부·소수부·서식을 나누면','']
    lines+=table(['모델','dataset','기준','현재 위치','오류 / safe','민감도','오경보율','precision'],[[r['model'],r['dataset'],r['basis'],r['region'],f"{r['error_steps']} / {r['safe_steps']}",pct(r['sensitivity']),pct(r['fpr']),pct(r['precision'])] for r in summary['detection'] if r['region']!='all' and r['steps']], '여기의 위치는 현재 risk step의 role이며, viewer의 “미래 최초 오류가 어디인가” 필터와 다르다. tenths가 소수부 숫자, decimal_point와 경계 token 등은 format이다. 같은 위치의 정답 유지와 오류를 비교하므로 소수점 뒤 숫자가 원래 상승하기 쉬운 현상을 확인할 수 있다. 오류 수가 적은 format 행의 극단 비율은 표본 수와 함께 읽는다.')
    lines+=['## 5. Viewer 특수 필터별 최초 오류의 상승·감소','',
       '이 절은 viewer와 같은 사후 선택 조건의 **기술통계**다. 예를 들어 “직전 token 대비 상승”을 고르면 그 상승 비율은 정의상 100%다. “최초 오류 위치”, “낮은 entropy 오류”, “유지 직후 다음 숫자 오류” 등은 정답/미래 결과를 알아야 정할 수 있어 online trigger나 오류 예측의 근거로 쓸 수 없다. 필터는 서로 겹치며 행을 합산하지 않는다. 모델 안에서 dataset을 합친 탐색용 집계이므로 dataset 분포에 민감하다.','']
    for model in ('qwen25','qwen3'):
        lines+=[f'### {model}','']
        lines+=table(['조건','query / endpoint 경로','최초 오류 수','token 상승/감소/동일 (eligible)','numeric 상승/감소/동일 (eligible)','오류 Δtoken 평균 nat'],[[FILTER_NAMES[r['filter']],f"{r['queries']} / {r['paths']}",r['errors'],f"{r['token']['rise']}/{r['token']['fall']}/{r['token']['equal']} ({r['token']['eligible']})",f"{r['numeric']['rise']}/{r['numeric']['fall']}/{r['numeric']['equal']} ({r['numeric']['eligible']})",fmt(r['token']['mean_delta'])] for r in summary['filters'] if r['model']==model], 'eligible은 해당 비교 대상이 존재하는 최초 오류 경로 수다. format 오류는 token 기준에 들어가지만 numeric 기준에는 들어가지 않는다. high-safe는 오류 여부와 무관하게 H≥1.5인 safe numeric step이 하나 이상인 경로, next-failure는 safe numeric step 다음 숫자에서 최초 오류가 난 경로다. 모델 불일치는 두 모델 모두 지원 표기인 경우의 선택 endpoint 최종 반올림 정답 판정 불일치다. 이 평균은 endpoint 경로 가중 평균이며 검증된 오류 탐지 임계값은 아니다.')
    lines+=['## 6. 현재 자료에서 말할 수 있는 것과 없는 것','',
       'entropy 상승과 오류의 연관성을 판단하려면 ① error-only 필터의 비율이 아니라 safe step을 포함한 오경보율, ② 동일 경로 paired 차이, ③ role·전이 종류를 통제한 결과를 함께 봐야 한다. 오류 순간에 entropy가 낮아지는 사례가 있다는 사실은 단순 entropy 최소화가 정답을 보장하지 않음을 뜻한다. 낮은 entropy는 영상 근거에 맞는 확신일 수도 있고 잘못된 답에 대한 확신일 수도 있다.','',
       '앞에서 집계한 greedy 기록은 영상 입력을 고정한 자료다. Hₖ−Hₖ₋₁에는 prefix 변화, 출력 문법, 자릿수, 새로 선택한 token의 영향이 섞이므로 이를 “영상 증거가 줄인 entropy”로 부를 수 없다. 별도로 저장된 영상 perturbation probe의 재집계는 아래 링크에서 구분해 확인한다. 자연스러운 관련 증거의 효과는 별도 대조 실험이 필요하다.','']
    from saved_decoding_entropy_links import markdown_from
    lines += markdown_from(dest)
    from entropy_evidence_design import evidence_markdown
    evidence=evidence_markdown()
    lines += evidence.splitlines() if isinstance(evidence,str) else evidence
    lines+=['## 재현과 자료','',
       f'생성기: `{ROOT.relative_to(ROOT.parent)}/_lib/entropy_error_appendix.py`. bootstrap seed={SEED}, repeats={REPEATS}. 모델 재추론 없이 기존 자료만 읽는다. 원본 step entropy와 top-30 자료를 {counts["matched_entropy_checks"]:,}개 step에서 대조하고 viewer의 최초 오류 Δ와도 대조했다.','',
       f'[집계 JSON]({os.path.relpath(out/"metrics/summary.json",dest)}) · [출처와 SHA-256]({os.path.relpath(out/"metadata/sources.json",dest)}) · [검증 기록]({os.path.relpath(out/"validation/checks.json",dest)})','']
    for label,path in src.items():lines.append(f'- {label}: [{path.name}]({os.path.relpath(path,dest)})')
    write_result_text(dest/'REPORT.md','\n'.join(lines)+'\n')


def build(scope):
    print(scope+': reading saved steps',flush=True)
    src,rows,risk,paths,counts=load(scope);hashes={k:digest(p) for k,p in src.items()}
    print(scope+': summarizing '+str(len(risk))+' risk steps',flush=True)
    summary=summarize(rows,risk,paths);summary.update(scope=scope,counts=counts,bootstrap_seed=SEED,bootstrap_repeats=REPEATS)
    out=ART/scope/ANALYSIS
    save(out/'metrics/summary.json',summary)
    save(out/'metadata/sources.json',dict(sources={k:dict(path=str(p.relative_to(ROOT)),sha256=hashes[k]) for k,p in src.items()},generator_sha256=digest(Path(__file__)),evidence_design_sha256=digest(ROOT/'_lib/entropy_evidence_design.py'),saved_decoding_links_sha256=digest(ROOT/'_lib/saved_decoding_entropy_links.py'),scope=scope))
    assert all(digest(p)==hashes[k] for k,p in src.items())
    save(out/'validation/checks.json',dict(passed=True,source_unchanged=True,step_entropy_checks=counts['matched_entropy_checks'],viewer_error_delta_agreement=True,post_error_excluded=True,first_error_unique=True,generated_tables_from_summary=True))
    plot(summary,out/'figures');report(scope,src,counts,summary)
    print(scope+': complete '+str(RESULT/scope/ANALYSIS/'REPORT.md'),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('pilot/300','full/9404'));args=parser.parse_args()
    for scope in ([args.scope] if args.scope else ['pilot/300','full/9404']):build(scope)
