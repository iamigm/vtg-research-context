"""Comparable power-of-two tables and a budget selector within the family viewer."""
import collections
import copy
import json
import os
from pathlib import Path
from temporal_mode_capped_core import *
import temporal_mode_report as common


def specification():
    return '''# Temporal mode search · capped rollouts · pilot600

## 범위와 변경

고정 pilot600,세 데이터셋별200 query,seed42·43,probability/token_entropy/temporal_mode 세 방법이다. 이전 token-budget 실행은 중단하고 원자료를 보존했다. 생성 정책이 바뀐 새 조건이므로 이전 결과와 합산하지 않는다. frame IDs·PTS·H×W·tensor hash,모델,0.9% absolute floor,0.01 relative coefficient,90% main nucleus,[0,D] clipping은 유지한다. 0–9 enumeration과 새 interval proposal prompt는 사용하지 않는다.

## Rollout 예산

rollout을 시작해 한 응답을 완성하거나 유효성 실패로 끝낸 시도 한 번을1회로 센다. 중복 sequence·동일 interval도 실제 시도이므로 포함한다. unique sequence·unique interval·temporal mode 수는 별도 기록한다.

| 총 rollout 상한 | Main | Tail |
|---:|---:|---:|
|32|16|16|
|64|32|32|
|128|64|64|
|256|128|128|

각 방법·seed는 main128+tail128을 최대 한 번 실행한다. 작은 예산은 같은 draw stream의 누적 checkpoint다. 3방법×2seed를 합해 query당 최대1,536 rollout 시도다. 원래 약65,500회 규모보다 약43배 작은 횟수 상한이며 시간 단축 배율을 보장하는 값은 아니다. Main65,536/tail131,072 decoder input tokens는 이차 안전 상한으로만 유지한다. 응답 최대512 tokens다. 지원된 branch가 없거나 token guard를 만나면 실제 시도 수가 상한보다 작을 수 있으며 이를 표시한다.

## 초기화와 우선순위

Main root sampling4회,공통 coverage를 합해 초기화는 최대8회다. 새 prefix마다8회를 요구하지 않는다. 실제 main 선택지가2개 이상인 prefix만 main task로 등록한다. 이후 probability는 raw prefix mass/cost와 visit penalty,token_entropy는 추가 raw entropy,temporal_mode는 추가 weighted novelty와 uncertainty bonus를 사용한다. 미조사 edge의 periodic coverage는8회당 최대1회다. Tail도 초기 coverage는최대4회이며 모든 tail entry에32회를 배정하지 않는다.

Checkpoint마다 root/initialization/coverage/adaptive 선택 횟수와 미조사 edge 수를 기록한다. 방법별 adaptive 선택이 예산 내에 실제 발생하는지 테스트한다. 선택 점수가 달라도 모델이 같은 결과를 낼 수 있으며 실제 결과를 인위적으로 다르게 만들지 않는다.

## Tail 격리와 예산 간 무결성

각 checkpoint에서 main 후보를 먼저 고정·판단한 뒤 같은 횟수의 tail을 조사한다. Tail entry는 그 시점까지 main이 방문한 prefix에서만 얻는다. 이후 큰 예산에서 발견할 prefix를 작은 예산의 tail에 미리 사용하지 않는다. Main RNG와 선택 통계는 tail 호출 수 및 중간 judge 호출과 분리한다. 과거 checkpoint는 deep copy로 고정하며 이후 확장으로 바뀌지 않는다. Tail 후보는 모델 판단이나 main 선택에 전달하지 않는다.

## Mode와 모델 판단

부호를 보존해 parsing한 후 endpoint를[0,D]로 clipping하고 start<end를 요구한다. exact interval 통합 뒤 모든 쌍의 IoU≥.5와 endpoint 차이≤max(1초,.1×짧은 interval 길이)를 만족하는 mode로 묶는다. 원래 token IDs·raw interval·path probability는 유지한다. Mode는 semantic event GT가 아닌 시간 가설이다.

관측 main mass 상위6 mode에서 실제 interval medoid와 boundary 대안을 고른다. 같은 frozen VLM이 event match/onset/offset과 짧은 근거를 판단하고,mode 내부 비교 후 mode 대표 전원을 양쪽 A/B 순서로 round robin 비교한다. Tail·token rank·greedy 출처는 제시하지 않는다. 최종 답은 기존 clipped interval이며 새 timestamp를 생성하지 않는다. 각 checkpoint의 judge cap은32,768 tokens이며 동일 prompt는 정확한 context cache로 재사용한다.

## 평가와 비용

IoU=교집합/합집합,mIoU=query별 IoU 평균,R@τ=IoU≥τ 비율(τ=.3/.5/.7)이다. 높을수록 좋다. 실패는0점이며 두 seed를 query 안에서 평균한다. 다른 예산의 값을 한 평균으로 합치지 않는다. 진행 중 표는 해당 방법·예산에서 두 seed가 모두 완료된 query만 집계한다.600 query×3방법×2seed=3,600 route,각4checkpoint=14,400개 예산별 선택이며 독립14,400 query가 아니다.

Main/pool/diagnostic oracle을 구분하고 GT는 모든 선택 이후 평가에만 사용한다. MI는 p/q-weighted 조건부 mode 정보량(bits) 진단이다. Native model mass와 관측 경로 질량 하한을 구분한다. Main/tail token 비용은 해당 checkpoint까지 누적이며 judge는 그 checkpoint만 독립 배치했을 때 필요한 logical 비용이다. 이전 작은 예산의 judge 비용을 큰 예산의 독립 비용에 더하지 않는다. 실제 전체 sweep GPU 비용은 raw actual_cost에 따로 기록한다.

## 운영

영상 hash modulo3으로 GPU worker3개에 겹치지 않게 배정한다. worker별 Torch threads2와 CPU input prefetch1개를 사용한다. 고정 code snapshot·조건 및 예산별 atomic checkpoint·20초 heartbeat·최대3회 재개를 사용한다. 완료 후 MD·HTML·raw validation·viewer validation·registry·통합 index를 자동 갱신한다. 동일 계열의 capped_rollouts 조건이며 이전 결과는 별도 선택 항목으로 남긴다.
'''


def collect(cohort):
    rows=[]
    completed=[]
    for sample in cohort:
        final=OUT/'responses'/f'{sample["sample_key"]}.json'
        partial=final.with_suffix('.partial.json')
        path=final if final.exists() else partial
        try:
            data=read(path)
        except FileNotFoundError:
            try:
                data=read(final)
            except FileNotFoundError:
                continue
        assert data['config_hash']==digest(CONFIG)
        if is_complete(data):
            completed.append(sample['sample_key'])
        gt=sample['span'][0]
        baseline=iou(data['baseline']['span'],gt)
        for method in METHODS:
            for budget in CHECKPOINTS:
                pair=[data['results'].get(f'{method}:{seed}',{}).get('checkpoints',{}).get(str(budget)) for seed in CONFIG['seeds']]
                if not all(pair):
                    continue
                for seed,result in zip(CONFIG['seeds'],pair):
                    main=result['main_modes']
                    tail=result['tail_modes']
                    main_oracle=max((iou(interval,gt) for group in main for interval in group['intervals']),default=0.)
                    pool_oracle=max((iou(interval,gt) for group in result['selection']['pool'] for interval in group['representatives']),default=0.)
                    diagnostic=max([main_oracle]+[iou(interval,gt) for group in tail for interval in group['intervals']])
                    score=iou(result['selection']['span'],gt)
                    tail_only=sum(not any(all(compatible(a,b) for a in group['intervals'] for b in other['intervals']) for other in main) for group in tail)
                    tokens=result['main']['spent']+result['tail']['spent']+result['selection']['cost']+len(data['baseline']['ids'])
                    draws=data['results'][f'{method}:{seed}']['draws'][:result['draw_cutoff']]
                    unique={kind:len({draw['sequence'] for draw in draws if draw['kind']==kind}) for kind in ('main','tail')}
                    rows.append(dict(key=sample['sample_key'],dataset=sample['dataset'],video=sample['video_path'],
                        duration=sample['duration'],gt_ratio=(gt[1]-gt[0])/sample['duration'],method=method,seed=seed,
                        budget=budget,total_rollout_budget=2*budget,iou=score,baseline=baseline,
                        r03=float(score>=.3),r05=float(score>=.5),r07=float(score>=.7),failed=result['selection']['span'] is None,
                        main_oracle=main_oracle,pool_oracle=pool_oracle,diagnostic_oracle=diagnostic,
                        main_modes=len(main),tail_modes=len(tail),tail_only_modes=tail_only,
                        main_tokens=result['main']['spent'],tail_tokens=result['tail']['spent'],judge_tokens=result['selection']['cost'],
                        total_tokens=tokens,main_rollouts=result['main']['rollouts'],tail_rollouts=result['tail']['rollouts'],
                        main_unique_sequences=unique['main'],tail_unique_sequences=unique['tail'],
                        main_adaptive=result['main']['scheduler'].get('adaptive',0),
                        status=result['selection']['status']))
    return rows,completed


def initialize():
    for path in (RESULT,OUT/'web',OUT/'metrics'):
        path.mkdir(parents=True,exist_ok=True)
    (RESULT/'SPEC.md').write_text(specification())
    write(OUT/'web/samples.json',common.sample_catalog(read(COHORT)))
    template=(LEGACY/'code/temporal_mode_viewer.html').read_text()
    template=template.replace('<html lang="ko">','<html lang="ko" data-family-default>')
    template=template.replace('Temporal mode search · pilot600','Temporal mode search · capped rollouts · pilot600')
    template=template.replace('<option>44</option>','')
    budget='<label>총 rollout 예산 <select id="budget">'+''.join(f'<option value="{k}"'+(' selected' if k==128 else '')+f'>{2*k} (main{k}+tail{k})</option>' for k in CHECKPOINTS)+'</select></label>'
    template=template.replace('<a id="raw"',budget+'<a id="raw"')
    template=template.replace('<div id="heading">','<section><h2>Rollout 예산별 mIoU</h2><div id="budget-curve">완료된 checkpoint를 기다리는 중…</div></section><div id="heading">')
    responsive='''<style>.cards{grid-template-columns:repeat(4,minmax(0,1fr))}.card,.mode,.status,.muted,.evidence,#heading{min-width:0;overflow-wrap:anywhere}.controls>*{max-width:100%}.mode-grid{grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr))}#budget-curve svg{width:100%;max-height:250px}@media(max-width:850px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>'''
    template=template.replace('</head>',responsive+'</head>')
    template=template.replace('__THEME__',os.path.relpath(ROOT.parent/'_lib/experiment_theme.css',RESULT))
    template=template.replace('__SCRIPT__',os.path.relpath(OUT/'web/viewer.js',RESULT))
    template=template.replace('__STORE__',os.path.relpath(OUT,RESULT))
    (RESULT/'detail.html').write_text(template)
    script=(LEGACY/'code/temporal_mode_viewer.js').read_text()
    needle='result=data?.results?.[key];'
    assert needle in script
    script=script.replace(needle,"run=data?.results?.[key],result=run?.checkpoints?.[$('budget').value];if(result)result.draws=run.draws.slice(0,result.draw_cutoff);")
    script=script.replace("['method','seed']","['method','seed','budget']")
    script=script.replace("if(params.has('seed'))$('seed').value=params.get('seed');","if(['42','43'].includes(params.get('seed')))$('seed').value=params.get('seed');if(['16','32','64','128'].includes(params.get('budget')))$('budget').value=params.get('budget');")
    script=script.replace("url.searchParams.set('seed',$('seed').value);","url.searchParams.set('seed',$('seed').value);url.searchParams.set('budget',$('budget').value);")
    script=script.replace('${s.completed_conditions??0}/5400 반복 선택','${s.completed_conditions??0}/${s.total_conditions??3600} route · ${s.completed_budget_selections??0}/${s.total_budget_selections??14400} budget 선택')
    script=script.replace('최소 조사량 미달: main ${result.main.unfinished_minimum},tail ${result.tail.unfinished_minimum}.','Rollout main ${result.main.rollouts}/${result.main.rollout_cap},tail ${result.tail.rollouts}/${result.tail.rollout_cap}. Main adaptive 선택 ${result.main.scheduler.adaptive??0}회. 미조사 edge: main ${result.main.unvisited_edges},tail ${result.tail.unvisited_edges}.')
    script+='''
async function drawBudgetCurve(){const x=await get(new URL('metrics/summary.json',store));const groups=x.groups.filter(g=>g.queries>0);if(!groups.length)return;const methods=['probability','token_entropy','temporal_mode'],colors=['#3974b9','#a66723','#22806a'];let svg='<svg viewBox="0 0 660 230" role="img" aria-label="총 rollout 예산별 mIoU"><path d="M45 15V195H625" fill="none" stroke="#aab5c4"/>';for(let i=0;i<4;i++){svg+=`<text x="${65+i*175}" y="219" font-size="12">${32*2**i}</text>`}methods.forEach((m,i)=>{const points=groups.filter(g=>g.method===m).sort((a,b)=>a.budget-b.budget).map(g=>`${65+Math.log2(g.budget/16)*175},${195-g.iou*175}`).join(' ');svg+=`<polyline points="${points}" fill="none" stroke="${colors[i]}" stroke-width="2"/>`;});svg+='</svg>';document.getElementById('budget-curve').innerHTML=svg+'<p class="muted">가로축: 총 rollout 상한 (main+tail),세로축:mIoU 0–1. Probability 파랑 · token entropy 갈색 · temporal mode 초록. 진행 중에는 조건별 완료 query 수가 다를 수 있습니다.</p>';}
drawBudgetCurve().catch(()=>{});setInterval(()=>drawBudgetCurve().catch(()=>{}),30000);
'''
    (OUT/'web/viewer.js').write_text(script)
    report()


def report():
    rows,complete=collect(read(COHORT))
    write(OUT/'metrics/per_query.json',rows)
    groups=[]
    for budget in CHECKPOINTS:
        for method in METHODS:
            subset=[r for r in rows if r['budget']==budget and r['method']==method]
            summary=common.summarize(subset)
            summary.update(method=method,budget=budget,total_rollout_budget=2*budget)
            summary.update({field:sum(r[field] for r in subset)/max(1,len(subset)) for field in
                ('main_rollouts','tail_rollouts','main_unique_sequences','tail_unique_sequences','main_adaptive')})
            groups.append(summary)
    write(OUT/'metrics/summary.json',dict(queries_completed=len(complete),target_queries=600,complete=len(complete)==600,
        checkpoints=CHECKPOINTS,total_rollout_budgets=[2*k for k in CHECKPOINTS],groups=groups))
    lines=['# Temporal mode search · capped rollouts · pilot600','',
        f'상태: {"완료" if len(complete)==600 else "실행 중·미완료"}.600 query 중 {len(complete)} query의 전체 조건 완료. '
        '세 데이터셋별200 query,3방법×2seed×4예산 checkpoint를 독립 query로 합산하지 않는다.','',
        '동일 source stream에서 main/tail 각각16·32·64·128회,총32·64·128·256회로 비교한다. '
        '중복·무효 응답을 포함한 시도 수가 예산이며 unique sequence·interval·mode는 별도 집계한다. '
        'Main 초기화는최대8회이며 새 prefix마다8회를 요구하지 않는다.','',
        'IoU=교집합/합집합,mIoU=query 평균 IoU,R@τ=IoU≥τ 비율(τ=.3/.5/.7). 높을수록 좋으며 실패는0점이다. '
        '같은 query의 seed42·43을 먼저 평균한다. 진행 중에는 두 seed가 완료된 checkpoint만 표에 포함한다. '
        'Δ는 동일 clipping greedy 대비다. Main oracle/pool oracle은 실제 main 후보/shortlist의 사후 최고 IoU이며 diagnostic oracle은 tail까지 포함한 비운영 상한이다.','',
        '## 예산 비교','',
        '| 총 rollout 상한 | Main/Tail 상한 | 방법 | query / seed 응답 | mIoU | R@.3 | R@.5 | R@.7 | main oracle |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        if group['queries']:
            lines.append(f'| {group["total_rollout_budget"]} | {group["budget"]}/{group["budget"]} | {group["method"]} | '
                f'{group["queries"]}/{group["responses"]} | {group["iou"]:.4f} | {group["r03"]:.4f} | {group["r05"]:.4f} | {group["r07"]:.4f} | {group["main_oracle"]:.4f} |')
        else:
            lines.append(f'| {group["total_rollout_budget"]} | {group["budget"]}/{group["budget"]} | {group["method"]} | 0/0 | — | — | — | — | — |')
    lines+=['','## 실제 시도·중복·적응 선택','',
        '| 총 상한 | 방법 | Main 시도 | Tail 시도 | Main unique sequence | Tail unique sequence | Main adaptive 선택 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for group in groups:
        lines.append(f'| {group["total_rollout_budget"]} | {group["method"]} | '+
            ' | '.join(f'{group[field]:.2f}' for field in ('main_rollouts','tail_rollouts','main_unique_sequences','tail_unique_sequences','main_adaptive'))+' |')
    for budget in CHECKPOINTS:
        selected=[r for r in rows if r['budget']==budget]
        lines+=['',f'## 총 rollout 상한 {2*budget} · 세부 집계','',*common.table(selected),'','### Dataset별','']
        for dataset in sorted({r['dataset'] for r in read(COHORT)}):
            lines+=[f'#### {dataset}','',*common.table([r for r in selected if r['dataset']==dataset]),'']
        lines+=['### Seed별','']
        for seed in CONFIG['seeds']:
            lines+=[f'#### seed {seed}','',*common.table([r for r in selected if r['seed']==seed]),'']
        lines+=['### 영상 길이별','']
        for index in range(5):
            lines+=[f'#### [{common.DURATIONS[index]:g},{common.DURATIONS[index+1]:g}) seconds','',
                *common.table([r for r in selected if common.band(r['duration'],common.DURATIONS)==index]),'']
        lines+=['### GT ratio별','']
        for index in range(5):
            lines+=[f'#### [{common.RATIOS[index]:g},{common.RATIOS[index+1]:g})','',
                *common.table([r for r in selected if common.band(r['gt_ratio'],common.RATIOS)==index]),'']
        lines+=['### duration × GT ratio',*common.cross_tables(selected)]
        if len(complete)==600:
            lines+=['','### Greedy 대비 paired ΔmIoU 95% CI','',str(common.bootstrap(selected))]
    lines+=['','## 재현과 비용','',
        '[실행 명세](SPEC.md) · [상세 결과](detail.html). 이전 token-budget 결과와 원자료를 합산하지 않는다. '
        'Main/tail token 비용은 해당 checkpoint까지 누적,judge 비용은 그 예산만 독립 실행하는 기준이다. '
        '실제 전체 sweep 비용과cache 재사용은 raw actual_cost에 따로 보존한다. 관측 mode 수는 실제 모든 사건의 coverage가 아니다.']
    (RESULT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'web/completed.json',complete)


if __name__=='__main__':
    import sys
    initialize() if '--initialize' in sys.argv else report()
