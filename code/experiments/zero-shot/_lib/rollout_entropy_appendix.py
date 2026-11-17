"""Entropy diagnostics on saved deterministic alternative decoding paths; CPU only."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from result_layout import relocate_path
from token_uncertainty_output import write_result_text
from timelens_full_digit_impl import classify

ROOT=Path('/home/irteam/experiments/zero-shot')
ART=ROOT/'artifacts/runs/rollout/deterministic/pilot/900'
DEST=ROOT/'results/rollout/deterministic/pilot/900'
BASES=('token','numeric')
SEED=42


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fmt(v,n=3):return '—' if v is None else f'{v:.{n}f}'
def pct(v):return '—' if v is None else f'{100*v:.1f}%'


def entropy(node):
    values=[x['probability'] for x in node['top']];mass=sum(values)
    return -sum((v/mass)*math.log(v/mass) for v in values if v>0)


def ci(values):
    a=np.asarray(values,float)
    if not len(a):return {'queries':0,'mean':None,'ci95':[None,None]}
    if len(a)==1:return {'queries':1,'mean':float(a[0]),'ci95':[None,None]}
    rng=np.random.default_rng(SEED);draws=[]
    for _ in range(20):draws.extend(rng.choice(a,(50,len(a)),replace=True).mean(axis=1))
    return {'queries':len(a),'mean':float(a.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist()}


def stat(rows,key):
    by=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:by[r['sample_key']].append(r[key])
    return ci([np.mean(v) for v in by.values()])


def interval(r):return f"{fmt(r['mean'])} [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]"


def iou(candidate,gt):
    p=candidate['parsed']
    if not p or not p['positive'] or not candidate['stopped_on_eos']:return 0.
    a,b=p['start'],p['end'];return max(0,min(b,gt[1])-max(a,gt[0]))/max(1e-12,max(b,gt[1])-min(a,gt[0]))


def endpoint_steps(candidate,index,sample,entropies):
    answer=candidate['answer']
    match=re.search(r'(?<![\d:])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d:])',answer)
    if ':' in answer or not match or any(not re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9])?',v) for v in match.groups()):return None
    identifiers=index['candidates'][str(candidate['index'])]['nodes'];pieces=[]
    for chosen,key in zip(candidate['ids'],identifiers):
        node=index['nodes'][key];found=next(t for t in node['observed'] if t['id']==chosen)
        pieces.append(found['token'])
    text=''.join(pieces);assert text.startswith(answer)
    output={}
    for n,ep in ((1,'start'),(2,'end')):
        a,b=match.span(n);offset=0;states=[]
        for j,piece in enumerate(pieces):
            end=offset+len(piece)
            if end>a and offset<=b:
                states.append(dict(ar_step=j+1,token=piece,prefix=text[a:min(end,b)],closed=end>b,
                                   entropy=entropies[identifiers[j]],node=identifiers[j],chosen=candidate['ids'][j]))
            offset=end
        if not states or not states[-1]['closed']:return None
        _,classified,correct=classify(match.group(n),sample['span'][0][n-1],states)
        output[ep]=classified
    return output


def load():
    references=ART/'metadata/source_records.json';refs=json.loads(references.read_text())
    counts=Counter();excluded=Counter();sources=[];allrisk=[];pathstats=[];pairs=[];samples=[]
    for ordinal,ref in enumerate(refs):
        source=relocate_path(Path(ref['path']));assert sha(source)==ref['sha256']
        rec=json.loads(source.read_text());sample=rec['sample'];key=sample['sample_key'];samples.append(sample)
        indexpath=ART/'token_traces'/key/'index.json';index=json.loads(indexpath.read_text())
        assert index['raw_record_sha256']==ref['sha256'] and index['validation']['passed']
        sources.append({'sample_key':key,'record':str(source.relative_to(ROOT)),'record_sha256':ref['sha256'],
                        'index':str(indexpath.relative_to(ROOT)),'index_sha256':sha(indexpath)})
        hs={k:entropy(n) for k,n in index['nodes'].items()};counts['unique_prefixes']+=len(hs)
        assert all(-1e-12<=v<=math.log(30)+1e-10 for v in hs.values())
        cs=rec['result']['candidates'];by={c['index']:c for c in cs};classified={};scores={c['index']:iou(c,sample['span'][0]) for c in cs}
        counts['candidates']+=len(cs);counts['queries']+=1;seen=set()
        for c in cs:
            kind='greedy' if c['branch'] is None else 'alternative_free'
            counts[kind+'_candidates']+=1
            forced=len(c['branch']['forced_prefix_ids']) if c['branch'] else 0
            trace=index['candidates'][str(c['index'])]['nodes'];assert len(trace)==len(c['ids'])
            for j,nodekey in enumerate(trace):assert index['nodes'][nodekey]['prefix_ids']==c['ids'][:j]
            eps=endpoint_steps(c,index,sample,hs);classified[c['index']]=eps
            if eps is None:excluded[kind+'_unsupported_candidates']+=1;continue
            counts[kind+'_supported_candidates']+=1
            for ep,steps in eps.items():
                previous_numeric=None;eligible=[]
                for r in steps:
                    j=r['ar_step']-1
                    if not r['previous_correct']:
                        if j>=forced:counts[kind+'_free_posterror_occurrences']+=1
                        continue
                    dd=dict(sample_key=key,dataset=sample['dataset'],kind=kind,endpoint=ep,candidate=c['index'],
                            role=r['role'],region='fractional' if r['role']=='tenths' else 'integer' if r['numeric'] else 'format',
                            first_error=r['first_error'],token_delta=hs[trace[j]]-hs[trace[j-1]] if j else None,
                            numeric_delta=r['entropy']-previous_numeric['entropy'] if r['numeric'] and previous_numeric else None)
                    if j>=forced:
                        counts[kind+'_free_risk_occurrences']+=1;eligible.append(dd)
                        identity=(kind,ep,trace[j],c['ids'][j])
                        if identity not in seen:seen.add(identity);allrisk.append(dd)
                        else:counts[kind+'_shared_risk_occurrences_removed']+=1
                    else:
                        counts['conditioning_risk_occurrences_excluded']+=1
                        if r['first_error']:counts['conditioning_first_errors_excluded']+=1
                    if r['numeric'] and r['prefix_correct']:previous_numeric=r
                for basis in BASES:
                    rr=[r for r in eligible if r[basis+'_delta'] is not None]
                    errors=[r for r in rr if r['first_error']];safe=[r for r in rr if not r['first_error']]
                    if errors and safe:
                        assert len(errors)==1
                        pathstats.append(dict(sample_key=key,dataset=sample['dataset'],kind=kind,endpoint=ep,basis=basis,
                            candidate=c['index'],contrast=errors[0][basis+'_delta']-float(np.mean([r[basis+'_delta'] for r in safe]))))
        for c in cs:
            if not c['branch']:continue
            b=c['branch'];parent=by[b['parent']];j=b['position'];cn=index['candidates'][str(c['index'])]['nodes'];pn=index['candidates'][str(parent['index'])]['nodes']
            assert cn[j]==pn[j];counts['same_prechoice_distribution_pairs']+=1
            endpoint_names={r.split('.')[0] for r in b['roles']}
            if len(endpoint_names)!=1:excluded['branch_multiple_endpoints']+=1;continue
            ep=next(iter(endpoint_names));ce=classified[c['index']];pe=classified[parent['index']]
            if ce is None or pe is None:excluded['branch_unsupported_endpoint']+=1;continue
            a=next((r for r in ce[ep] if r['numeric'] and r['ar_step']-1>j),None)
            p=next((r for r in pe[ep] if r['numeric'] and r['ar_step']-1>j),None)
            if a is None or p is None:excluded['branch_no_later_numeric_same_endpoint']+=1;continue
            if (a['role'],a['digit_ordinal'])!=(p['role'],p['digit_ordinal']):excluded['branch_later_role_mismatch']+=1;continue
            role='format' if any('separator' in r for r in b['roles']) else 'fractional' if any('10^-' in r for r in b['roles']) else 'integer'
            pairs.append(dict(sample_key=key,dataset=sample['dataset'],endpoint=ep,fork_role=role,parent_is_greedy=b['parent']==1,
                child=c['index'],parent=parent['index'],next_role=a['role'],entropy_delta=a['entropy']-p['entropy'],
                iou_delta=scores[c['index']]-scores[parent['index']]))
        if (ordinal+1)%100==0:print(f'read {ordinal+1}/{len(refs)} queries',flush=True)
    counts['unique_videos']=len({s['video_path'] for s in samples});counts['unique_risk_decisions']=len(allrisk)
    return dict(counts=dict(counts),excluded=dict(excluded),risk=allrisk,pathstats=pathstats,pairs=pairs,sources=sources,
                references_sha256=sha(references))


def summarize(data):
    detection=[];within=[];branches=[]
    datasets=['all']+sorted({r['dataset'] for r in data['risk']})
    for dataset in datasets:
        risk=[r for r in data['risk'] if dataset=='all' or r['dataset']==dataset]
        for kind in ('greedy','alternative_free'):
            for basis in BASES:
                for region in ('all','integer','fractional','format'):
                    rr=[r for r in risk if r['kind']==kind and r[basis+'_delta'] is not None and (region=='all' or r['region']==region)]
                    if not rr:continue
                    tp=sum(r['first_error'] and r[basis+'_delta']>0 for r in rr);fn=sum(r['first_error'] and r[basis+'_delta']<=0 for r in rr)
                    fp=sum(not r['first_error'] and r[basis+'_delta']>0 for r in rr);tn=sum(not r['first_error'] and r[basis+'_delta']<=0 for r in rr)
                    detection.append(dict(dataset=dataset,kind=kind,basis=basis,region=region,queries=len({r['sample_key'] for r in rr}),tp=tp,fn=fn,fp=fp,tn=tn,
                        sensitivity=tp/(tp+fn) if tp+fn else None,fpr=fp/(fp+tn) if fp+tn else None,precision=tp/(tp+fp) if tp+fp else None))
                pp=[p for p in data['pathstats'] if p['kind']==kind and p['basis']==basis and (dataset=='all' or p['dataset']==dataset)]
                within.append(dict(dataset=dataset,kind=kind,basis=basis,endpoint_paths=len(pp),contrast=stat(pp,'contrast')))
        for role in ('all','integer','fractional','format'):
            pp=[p for p in data['pairs'] if (dataset=='all' or p['dataset']==dataset) and (role=='all' or p['fork_role']==role)]
            if not pp:continue
            for sign in ('lower','higher','equal'):
                rr=[p for p in pp if (p['entropy_delta']<0 if sign=='lower' else p['entropy_delta']>0 if sign=='higher' else p['entropy_delta']==0)]
                branches.append(dict(dataset=dataset,fork_role=role,entropy_direction=sign,pairs=len(rr),queries=len({p['sample_key'] for p in rr}),
                    improved=sum(p['iou_delta']>0 for p in rr),worsened=sum(p['iou_delta']<0 for p in rr),unchanged=sum(p['iou_delta']==0 for p in rr),
                    mean_delta_iou=stat(rr,'iou_delta'),mean_delta_entropy=stat(rr,'entropy_delta')))
    return dict(counts=data['counts'],excluded=data['excluded'],detection=detection,within=within,branches=branches,
        branch_pairs=len(data['pairs']),branch_queries=len({p['sample_key'] for p in data['pairs']}),seed=SEED,bootstrap_repeats=1000)


def plot(summary):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    rows=[r for r in summary['detection'] if r['dataset']=='all' and r['basis']=='token' and r['region'] in ('integer','fractional')]
    for j,r in enumerate(rows):
        axes[0].bar(j-.17,100*(r['sensitivity'] or 0),.34,color='#c75a38',label='First error rises' if j==0 else None)
        axes[0].bar(j+.17,100*(r['fpr'] or 0),.34,color='#247ba0',label='Safe decision rises' if j==0 else None)
    axes[0].set_xticks(range(len(rows)),[r['kind'].replace('_',' ')+'\n'+r['region'] for r in rows],rotation=12)
    axes[0].set_ylabel('Immediate-token entropy rise (%)');axes[0].legend(fontsize=8);axes[0].set_ylim(0,105)
    rows=[r for r in summary['branches'] if r['dataset']!='all' and r['fork_role']=='all' and r['entropy_direction']=='lower']
    for j,r in enumerate(rows):
        st=r['mean_delta_iou'];lo,hi=st['ci95'];mu=st['mean']
        if mu is not None:axes[1].errorbar(j,100*mu,yerr=None if lo is None else [[100*max(0,mu-lo)],[100*max(0,hi-mu)]],fmt='o',capsize=4,color='#247ba0')
    axes[1].set_xticks(range(len(rows)),[r['dataset'].split('-')[0] for r in rows],rotation=12)
    axes[1].axhline(0,color='#777',linewidth=1);axes[1].set_ylabel('Child - parent IoU (pp)');axes[1].set_title('When next numeric H30 is lower')
    fig.tight_layout();path=ART/'figures/entropy_saved_paths.svg';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path);plt.close(fig)
    return path


def report(summary):
    c=summary['counts'];e=summary['excluded'];figure=plot(summary)
    def table(headers,rows,explanation):return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows],'','**표 해설.** '+explanation,'']
    lower=next(r for r in summary['branches'] if r['dataset']=='all' and r['fork_role']=='all' and r['entropy_direction']=='lower')
    native={r['region']:r for r in summary['detection'] if r['dataset']=='all' and r['kind']=='alternative_free' and r['basis']=='token'}
    lines=['# 저장된 다른 decoding 경로의 entropy와 오류 · pilot · 900 queries','',
        '기존 deterministic rollout의 Qwen3-VL-8B-Instruct 분포를 다시 읽은 **사후 분석**이다. 모델 재추론·새 영상 개입 없이 저장된 top-30을 재정규화했다. Token uncertainty의 greedy full 결과와 별도 입력·후보 수집 조건이므로 두 표본을 독립 표본처럼 합산하지 않는다.','',
        '원래 실행은 2 FPS, spatial target 100,352 pixels, video_query input order, 공식 TimeLens prompt를 사용했다. 이번 집계는 저장된 입력 hash와 분포 검증에 연결하며 입력이나 generation 정책을 바꾸지 않는다.','',
        f'고정 pilot 900 queries(각 dataset 300), 고유 영상 {c["unique_videos"]:,}개, 후보 응답 {c["candidates"]:,}개, 최초 greedy 900개와 분기 {c["alternative_free_candidates"]:,}개, 고유 prefix {c["unique_prefixes"]:,}개를 확인했다. 같은 prefix의 분포는 index에서 한 번 읽고 전체 vocabulary 배열은 다시 읽지 않았다.','',
        '## 먼저 읽을 결과','',
        f'대체 경로의 free suffix에서도 직전 token 대비 상승만으로 오류를 가려내기는 어렵다. 정수부의 민감도/오경보율은 {pct(native["integer"]["sensitivity"])}/{pct(native["integer"]["fpr"])}, 소수부는 {pct(native["fractional"]["sensitivity"])}/{pct(native["fractional"]["fpr"])}다. 정답 가능성을 유지한 숫자에서도 상승이 매우 흔하다.','',
        f'같은 prefix에서 다른 token을 택하기 **전** entropy는 부모와 자식이 동일하다({c["same_prechoice_distribution_pairs"]:,}쌍 검증). 따라서 대체 token을 고르기 전의 entropy만으로 그 후보가 더 좋은지 알 수 없다. 분기 뒤 같은 endpoint의 다음 숫자 위치를 맞춘 비교는 {summary["branch_pairs"]:,}쌍, {summary["branch_queries"]:,}queries에서 가능했다.','',
        f'그중 자식의 다음 숫자 entropy가 더 낮은 {lower["pairs"]:,}쌍에서도 최종 IoU 개선 {lower["improved"]:,}쌍, 악화 {lower["worsened"]:,}쌍, 동일 {lower["unchanged"]:,}쌍이 함께 있었다. query 안에서 평균한 IoU 변화는 {interval(lower["mean_delta_iou"])}다. entropy가 낮은 branch를 고르는 규칙만으로 정확도 개선이 보장되지 않는다.','',
        '## 비교 정의와 분모','',
        'Hₜ=−Σqᵢlnqᵢ, qᵢ=pᵢ/Σtop30 pᵢ인 raw top-30 재정규화 entropy를 사용한다. 단위는 nat, 범위는 0–ln30다. 기존 index의 entropy_nats는 full-vocabulary 값이므로 주 분석에 사용하지 않았다.','',
        '첫 후보는 greedy baseline이다. 대체 후보는 마지막 강제 prefix 길이 L 이후 j≥L인 **free suffix**만 model의 선택으로 센다(여기서 j는 0부터 시작). 강제 token과 그 앞의 조건화 prefix를 모델의 새로운 오류로 세지 않는다. 각 endpoint의 rounded 정답으로 완성할 가능성이 이전 prefix까지 남은 risk step만 포함하며 최초 오류 이후는 제외한다. 원래 자릿수별 정답과 같은 canonical 비음수 정수 또는 소수 한 자리 seconds 문법을 사용한다. clock·여러 소수 자리 등은 이 오류 분석에서 제외한다.','',
        'Δtokenₜ=Hₜ−Hₜ₋₁은 바로 앞 decoding step을 비교한다. Δnumericₜ=Hₜ−Hⱼ는 같은 endpoint의 직전 정답 가능 숫자 j를 비교하며 현재도 숫자여야 한다. 비교 상대는 이미 고정된 conditioning prefix 안에 있을 수 있다. Δ>0을 상승으로 정의한다.','',
        'step 비율은 query 안의 (endpoint, prefix hash, 실제 선택 token)을 중복 제거해 공통 prefix가 후보 수만큼 반복되지 않게 했다. greedy와 alternative-free는 별도 집계다. 민감도=오류 중 상승 비율, 오경보율=safe 중 상승 비율, precision=상승 중 오류 비율이다. 강제 선택을 제외한 탐색 후보의 사후 표본이므로 자연 sampling 확률이나 online detector 성능은 아니다.','',
        'Cpath=최초 free 오류 Δ−앞선 free safe Δ 평균이다. 둘 다 있는 candidate×endpoint만 포함하고, query 안에서 candidate×endpoint를 먼저 평균한 뒤 query를 같은 가중치로 평균한다. 공통 prefix 중복은 이 경로별 비교에는 남을 수 있지만 query를 복제 독립 표본으로 세지 않는다. CI는 query bootstrap 1,000회(seed42), video 의존성과 다중 비교는 보정하지 않았다.','',
        '## 저장 범위와 제외','']
    lines+=table(['구분','greedy','alternative free'],[[name,c.get('greedy_'+k,0),c.get('alternative_free_'+k,0)] for name,k in [('후보 수','candidates'),('지원 문법 후보 수','supported_candidates'),('free risk step 발생 수','free_risk_occurrences'),('공통 risk step 중복 제거 수','shared_risk_occurrences_removed'),('오류 이후 free step 제외 수','free_posterror_occurrences')]],'후보 수는 query 수와 다르다. 동일 prefix의 여러 후보 참조를 step 비율의 독립 표본으로 세지 않는다. 강제·조건화 prefix의 risk step 발생 '+str(c.get('conditioning_risk_occurrences_excluded',0))+'개와 그 안의 최초 오류 '+str(c.get('conditioning_first_errors_excluded',0))+'개도 model free 선택 통계에서 제외했다. 지원하지 않는 후보는 greedy '+str(e.get('greedy_unsupported_candidates',0))+'개, alternative '+str(e.get('alternative_free_unsupported_candidates',0))+'개다.')
    lines+=['## 다른 decoding step에서도 상승이 오류를 구분하는가','']
    selected=[r for r in summary['detection'] if r['dataset']=='all']
    lines+=table(['경로','기준','현재 위치','query','오류 / safe','민감도','오경보율','precision'],[[r['kind'],r['basis'],r['region'],r['queries'],f"{r['tp']+r['fn']} / {r['fp']+r['tn']}",pct(r['sensitivity']),pct(r['fpr']),pct(r['precision'])] for r in selected], '같은 숫자 위치에서 민감도와 오경보율이 비슷하면 상승 여부가 오류를 잘 구분하지 못한다. format을 함께 묶은 all은 숫자와 다른 전이 빈도에 영향을 받으므로 integer·fractional 행을 우선 확인한다. numeric 기준은 이전 safe 숫자가 존재하는 일부 step만 포함한다.')
    lines+=['## 같은 경로의 평소 변화와 최초 free 오류 변화','']
    lines+=table(['dataset','경로','기준','candidate×endpoint 수','query 수','Cpath nat [95% CI]'],[[r['dataset'],r['kind'],r['basis'],r['endpoint_paths'],r['contrast']['queries'],interval(r['contrast'])] for r in summary['within']], 'Cpath가 양수면 같은 경로의 앞선 free safe step보다 최초 free 오류 때 entropy 변화가 더 컸다. 첫 free step부터 오류이면 이 비교에 들어가지 않는다. 현재 role·숫자 순서를 맞춘 회귀가 아니므로 문법 전이가 남아 있으며, 기존 greedy full 분석의 통제 회귀와 동일한 추정량이 아니다.')
    lines+=['## 같은 부모에서 분기한 다음 숫자 entropy와 최종 성능','',
        '분기 위치 직후 같은 endpoint에서 처음 나타나는 numeric step을 부모·자식에서 찾고 role과 numeric ordinal이 같은 경우만 비교한다. 서로 다른 token prefix에 조건화된 분포이며 영상은 동일하다. 마지막 강제 위치 이후 자식은 greedy로 완성되므로 추가 강제 선택이 없는 해당 child의 직접 continuation을 사용한다. 후손 후보를 parent의 직접 continuation으로 혼합하지 않는다. 정답 여부와 무관하게 모든 비교 가능한 branch를 포함한다.','',
        'ΔH=H(child 다음 숫자)−H(parent 다음 숫자), ΔIoU=IoU(child)−IoU(parent)다. IoU는 원래 deterministic 평가대로 연속 초·clipping/반올림 없이 계산하고 파싱 실패·역전·EOS 미도달은 0이다. 다음 숫자의 entropy는 실제 suffix를 관측한 값이므로 즉시 선택 가능한 비용 없는 점수가 아니다.','']
    rows=[r for r in summary['branches'] if r['fork_role']=='all']
    lines+=table(['dataset','다음 숫자 H','비교 쌍 / query','IoU 개선/악화/동일','query 평균 ΔIoU [95% CI]'],[[r['dataset'],r['entropy_direction'],f"{r['pairs']} / {r['queries']}",f"{r['improved']}/{r['worsened']}/{r['unchanged']}",interval(r['mean_delta_iou'])] for r in rows], '낮은 entropy에서 개선·악화가 함께 나오면 entropy 최소화만으로 좋은 branch를 고를 수 없다. 쌍의 개수 비율과 query 평균은 가중치가 다르며 query별 branch 수가 결과를 지배하지 않게 평균 변화는 query 안에서 먼저 평균했다. 같은 query가 lower/higher 두 그룹에 모두 들어갈 수 있어 행을 독립 비교하지 않는다.')
    rows=[r for r in summary['branches'] if r['dataset']=='all' and r['fork_role']!='all']
    lines+=table(['분기 token 위치','다음 숫자 H','비교 쌍 / query','IoU 개선/악화/동일','query 평균 ΔIoU [95% CI]'],[[r['fork_role'],r['entropy_direction'],f"{r['pairs']} / {r['queries']}",f"{r['improved']}/{r['worsened']}/{r['unchanged']}",interval(r['mean_delta_iou'])] for r in rows], '분기한 token의 정수부·소수부·서식을 분리했다. 같은 endpoint에 뒤따르는 숫자가 없으면 비교 자체가 불가능하므로 마지막 소수 자리 분기는 대체로 제외된다. 이 빈 표본을 효과 없음으로 해석하지 않는다.')
    lines+=table(['branch 비교 제외 사유','쌍 수'],[[k,v] for k,v in e.items() if k.startswith('branch_')],'26,560개의 전체 branch 중 위 제외 조건을 통과한 쌍만 다음 숫자 비교에 들어간다. 마지막 숫자 분기와 표기 변경을 제거한 선택된 부분집합이다.')
    lines += [f'![다른 경로의 entropy 진단]({os.path.relpath(figure,DEST)})','',
        '**그래프 해설.** 왼쪽은 오류와 safe 숫자에서 직전 token 대비 entropy 상승이 얼마나 자주 나오는지 비교한다. 두 막대가 비슷하면 상승만으로 오류를 구분하기 어렵다. 오른쪽은 다음 숫자 entropy가 낮아진 branch의 query 평균 ΔIoU(pp)와 bootstrap 95% CI다. 0보다 낮으면 그 조건에서는 평균적으로 부모보다 나빠졌다는 뜻이다.','',
        '## 영상 증거에 대해 아직 말할 수 없는 것','',
        '여기서는 영상 입력을 바꾸지 않고 같은 영상에서 여러 prefix·선택 경로를 비교했다. 따라서 분기 후 entropy 감소를 영상 증거 활용의 인과 효과로 부를 수 없다. 이 저장 자료는 “어떤 후보를 선택하면 이후 분포가 더 집중되는가”를 검사하는 데 유용하며, 실제 영상 증거 효과는 같은 prefix에서 관련/대조 구간 제거를 별도로 비교해야 한다. 낮은 entropy가 정답을 강화했는지 최종 IoU·rounded endpoint와 함께 확인해야 한다.','',
        '## 재현과 원자료','',
        '[집계 JSON]('+os.path.relpath(ART/'metrics/entropy_summary.json',DEST)+') · [출처 SHA-256]('+os.path.relpath(ART/'metadata/entropy_sources.json',DEST)+') · [검증]('+os.path.relpath(ART/'validation/entropy_checks.json',DEST)+')','',
        '재생성: `.venv/bin/python _lib/rollout_entropy_appendix.py`. 입력은 canonical pilot900의 `metadata/source_records.json`, `rollouts/<query>.json`, `token_traces/<query>/index.json`이다. 원래 응답·분포·정답 판정을 덮어쓰지 않으며 새 추론은 실행하지 않는다.','']
    write_result_text(DEST/'APPENDIX_ENTROPY.md','\n'.join(lines)+'\n')


def build():
    data=load();summary=summarize(data)
    save(ART/'metrics/entropy_summary.json',summary)
    save(ART/'metadata/entropy_sources.json',dict(scope='pilot/900',model='Qwen3-VL-8B-Instruct',sources=data['sources'],references_sha256=data['references_sha256'],generator_sha256=sha(Path(__file__)),classification_source_sha256=sha(Path(__file__).with_name('timelens_full_digit_impl.py')),definition='raw top30 renormalized; forced conditioning excluded; shared risk decisions deduplicated'))
    for source in data['sources']:
        assert sha(ROOT/source['record'])==source['record_sha256']
        assert sha(ROOT/source['index'])==source['index_sha256']
    save(ART/'validation/entropy_checks.json',dict(passed=True,sources_unchanged=True,counts=summary['counts'],same_prechoice_prefix_verified=True,forced_steps_excluded=True,shared_risk_steps_deduplicated=True,canonical_classifier_reused=True))
    report(summary);print(json.dumps({'complete':True,'counts':summary['counts'],'branch_pairs':summary['branch_pairs']},ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':build()
