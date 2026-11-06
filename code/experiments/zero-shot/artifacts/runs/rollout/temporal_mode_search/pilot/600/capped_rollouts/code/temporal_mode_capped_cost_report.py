"""Add fair trace work and observed execution costs to the frozen result report."""
import functools
import statistics

from temporal_mode_capped_cost import *
import temporal_mode_capped_report as original

_report = original.report


def definitions():
    return '''## 연산 비용과 성능 대비 비용

비용의 분모는 query·method·seed·예산 한 경우다. seed42·43을 query 안에서 먼저 평균한다. 진행 중 비교는 3방법×4예산×2seed가 모두 완료된 같은 query만 사용한다. 다음 두 비용을 구분한다.

**Independent trace work**는 저장된 token ID·분기 prefix·judge prompt를 따라 fresh video KV와 빈 distribution/prompt cache에서 greedy → main K회 → 해당 K의 judge → tail K회만 실행하는 경우의 연산 횟수를 재구성한다. 다른 method·seed·작은 예산 judge의 cache를 빌려 쓰지 않는다. 영상 prefill은 한 번 포함한다. 이는 시간 실측이나 FLOPs 추정이 아니며 실제 native forward 계측과 대조한다. Tail이 중간 main에 영향을 주지 않는 고정 draw stream을 사용한다.

- `decoder inputs`: native decoder forward에 실제 투입하는 token position 수의 합. 재생한 prefix, judge text prefill·근거 생성, 영상 prefill 뒤의 multimodal decoder positions를 포함한다. sampling 횟수나 출력 token 수와 다르다.
- `forwards`: native model forward 호출 수. 하나의 긴 judge text prefill도1회이므로 token 수와 함께 해석한다.
- `attention pairs`: 각 forward의 causal query-key position 쌍 `q(k−q)+q(q+1)/2`의 합. q는 새 input 길이,k는 전체 KV 길이다. layer/head 곱을 생략한 context 길이 workload proxy이며 정확한 FLOPs가 아니다. vision encoder의 별도 연산은 포함하지 않는다.
- `operational`: video prefill+greedy+main+judge. `research total`: operational+진단용 tail. Tail을 후보로 쓰지 않아도 이번 실험의 총비용에서 제외하지 않는다.
- `greedy ×`: 같은 query의 video prefill+greedy 대비 decoder inputs 또는 forward 호출 수 배율. 원래 입력의 hash와 greedy ID가 동일하다. 입력 prefill과 추가 text 비용도 분리한다.

**Observed execution**은 지금 사용하는 worker3개·공유 cache의 nested sweep에서 각 main/tail/judge 구간의 wall seconds,native forward CUDA stream seconds,forward 수,decoder inputs,process별 peak allocated/reserved memory를 기록한다. CUDA events를 각 forward 전후에 배치하며 GPU 동시 실행·queue 대기의 영향을 받을 수 있다. CUDA stream seconds는 GPU 전체 사용시간이나 정확한 kernel active time이 아니다. wall−forward wall은 Python/cache/mode grouping 등 host-side 잔여시간의 근사다. peak memory는 worker별 model/KV를 포함하고 다른 worker의 점유량과 합산한 장치 전체 peak가 아니다.

Observed main/tail은 해당 K까지의 누적,judge는 해당 K의 구간이다. `sweep`은 이전 모든 작은 예산 judge를 포함한 phase 합계이고,`sweep_elapsed_seconds`는 이전 checkpoint 저장과 phase 밖 Python 처리까지 포함한 경과시간(현재 checkpoint 저장 직전)이다. 독립 K 실행시간으로 부르지 않는다. method 실행 순서·이미 저장된 cache·재개에 따라 관측 비용이 달라지므로 method 공정 비교는 independent trace work를 사용한다. 재개 실행의 재탐색·실패 비용은 누적 실행 로그와 raw actual_cost에 보존한다. 계측 도입 이전 checkpoint의 실측은 결측으로 남기며0초로 처리하지 않는다.

효율 지표는 greedy 대비 `ΔmIoU(pp)=100×mean(IoU−IoU_greedy)`,추가1,000 decoder inputs당 향상 `ΔmIoU(pp)/(mean(extra decoder inputs)/1000)`다. 음수는 비용을 써도 성능이 악화됐다는 뜻이다. `marginal Δ`와 `추가 비용`은 같은 query에서 직전 절반 예산 대비 값이다. 같은 query 집합에서 더 낮은 비용·같거나 높은 mIoU의 다른 조건이 있으면 dominated로 표시한다. 평균 개선만으로 연구적 기여를 단정하지 않으며 비용 배율,실측 시간,seed·dataset별 결과와 불확실성을 함께 본다.
'''


@functools.lru_cache(maxsize=1)
def dialog_builder():
    import fps_spatial_control  # Select the exact, already installed tokenizer implementation.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    end_turn = tok.convert_tokens_to_ids('<|im_end|>')

    @functools.lru_cache(maxsize=16000)
    def dialog(prompt):
        text = tok.apply_chat_template([dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True)
        return [end_turn] + tok.encode('\n'+text, add_special_tokens=False)
    return dialog


def cost_rows():
    rows = []
    for sample in read(COHORT):
        path = OUT/'responses'/f'{sample["sample_key"]}.json'
        partial = path.with_suffix('.partial.json')
        try:
            data = read(path if path.exists() else partial)
        except FileNotFoundError:
            continue
        assert data['config_hash'] == digest(CONFIG)
        inputs = read(OUT/'inputs'/f'{sample["sample_key"]}.json')
        for method in METHODS:
            for budget in CHECKPOINTS:
                routes = [data['results'].get(f'{method}:{seed}', {}) for seed in CONFIG['seeds']]
                points = [route.get('checkpoints', {}).get(str(budget)) for route in routes]
                if not all(points):
                    continue
                for seed, route, point in zip(CONFIG['seeds'], routes, points):
                    independent = standalone_work(data, route, point, inputs['input_tokens'], dialog_builder())
                    score = iou(point['selection']['span'], sample['span'][0])
                    rows.append(dict(key=sample['sample_key'], dataset=sample['dataset'], method=method,
                        seed=seed, budget=budget, total_rollout_budget=2*budget,
                        iou=score, baseline=iou(data['baseline']['span'], sample['span'][0]),
                        standalone=independent, observed=point.get('cost')))
    return rows


def avg(rows, fn):
    by_key = collections.defaultdict(list)
    for row in rows:
        by_key[row['key']].append(fn(row))
    return statistics.mean(statistics.mean(values) for values in by_key.values()) if by_key else None


def summarize(rows):
    required = {(m, k, s) for m in METHODS for k in CHECKPOINTS for s in CONFIG['seeds']}
    present = collections.defaultdict(set)
    for row in rows:
        present[row['key']].add((row['method'], row['budget'], row['seed']))
    common = {key for key, values in present.items() if values == required}
    paired = [row for row in rows if row['key'] in common]
    measured_present = collections.defaultdict(set)
    for row in paired:
        if row['observed']:
            measured_present[row['key']].add((row['method'], row['budget'], row['seed']))
    measured_common = {key for key, values in measured_present.items() if values == required}
    groups = []
    for method in METHODS:
        for budget in CHECKPOINTS:
            subset = [r for r in paired if r['method'] == method and r['budget'] == budget]
            measured = [r for r in subset if r['key'] in measured_common]
            group = dict(method=method, budget=budget, total_rollout_budget=budget*2, queries=len(common),
                         responses=len(subset), measured_queries=len(measured_common))
            if not subset:
                groups.append(group)
                continue
            group.update(iou=avg(subset, lambda r:r['iou']), delta_pp=avg(subset, lambda r:100*(r['iou']-r['baseline'])))
            for scope in ('baseline', 'operational', 'research'):
                group[scope] = {field:avg(subset, lambda r:r['standalone'][scope][field])
                    for field in ('decoder_inputs', 'forwards', 'attention_pairs')}
            group['baseline']['iou'] = avg(subset, lambda r:r['baseline'])
            group['phases'] = {phase:{field:avg(subset, lambda r:r['standalone']['phases'][phase][field])
                for field in ('decoder_inputs', 'forwards', 'attention_pairs')}
                for phase in ('video_prefill', 'greedy_decode', 'main', 'tail', 'judge')}
            for scope in ('operational', 'research'):
                group[scope]['decoder_ratio'] = avg(subset, lambda r:r['standalone'][scope]['decoder_inputs']/max(1,r['standalone']['baseline']['decoder_inputs']))
                group[scope]['forward_ratio'] = avg(subset, lambda r:r['standalone'][scope]['forwards']/max(1,r['standalone']['baseline']['forwards']))
                extra = group[scope]['decoder_inputs']-group['baseline']['decoder_inputs']
                group[scope]['delta_pp_per_1000_extra_inputs'] = group['delta_pp']/(extra/1000) if extra>0 else None
            group['observed'] = {}
            if measured:
                for phase in ('main', 'tail', 'judge', 'sweep'):
                    group['observed'][phase] = {field:avg(measured, lambda r:r['observed'][phase][field]) for field in ADDITIVE}
                group['observed']['baseline'] = {phase:{field:avg(measured,lambda r:r['observed']['baseline'][phase][field]) for field in ADDITIVE}
                    for phase in ('video_prefill','greedy_decode','total')}
                def peak(row, field):
                    obs = row['observed']
                    return max([obs[phase][field] for phase in ('main','tail','judge')]+
                        [obs['baseline'][phase][field] for phase in ('video_prefill','greedy_decode')])/2**30
                group['observed']['peak_allocated_gib'] = avg(measured, lambda r:peak(r,'peak_allocated_bytes'))
                group['observed']['peak_reserved_gib'] = avg(measured, lambda r:peak(r,'peak_reserved_bytes'))
                group['observed']['max_peak_allocated_gib'] = max(peak(r,'peak_allocated_bytes') for r in measured)
                group['observed']['sweep_elapsed_seconds'] = avg(measured, lambda r:r['observed']['sweep_elapsed_seconds'])
                group['observed']['deployment_from_sweep_not_standalone'] = combine([group['observed'][phase] for phase in ('main','judge')])
                group['observed']['all_from_sweep_not_standalone'] = combine([group['observed'][phase] for phase in ('main','tail','judge')])
            previous = next((g for g in groups if g['method']==method and g['budget']==budget//2 and g['queries']), None)
            group['marginal_delta_pp'] = 100*(group['iou']-previous['iou']) if previous else group['delta_pp']
            group['marginal_extra_inputs'] = group['research']['decoder_inputs']-(previous['research']['decoder_inputs'] if previous else group['baseline']['decoder_inputs'])
            groups.append(group)
    for group in groups:
        if not group['queries']:
            continue
        group['dominated'] = (group['baseline']['iou']>=group['iou'] and group['baseline']['decoder_inputs']<group['research']['decoder_inputs']) or any(other.get('queries') and other['research']['decoder_inputs']<=group['research']['decoder_inputs']
            and other['iou']>=group['iou'] and (other['research']['decoder_inputs']<group['research']['decoder_inputs'] or other['iou']>group['iou']) for other in groups)
    return dict(schema=SCHEMA, common_queries=len(common), measured_common_queries=len(measured_common),
        raw_cost_rows=len(rows), paired_cost_rows=len(paired), common_keys=sorted(common),
        measured_common_keys=sorted(measured_common), groups=groups, timestamp=stamp())


def tables(summary):
    lines = [definitions(), '', f'비교 공통 query: **{summary["common_queries"]}/600**. 전체12조건의 두 seed 실측까지 있는 공통 query: **{summary["measured_common_queries"]}/600**. 계측 전 결과도 independent trace work를 계산하며 실측은 소급 생성하지 않는다.', '',
        '### Independent trace work · query당 평균', '',
        '| 총 rollout | 방법 | mIoU / Δpp | Main inputs | Tail inputs | Judge inputs | Prefill / greedy inputs | Operational inputs / greedy× | Research inputs / greedy× | Research forwards / greedy× | Attention pairs | 향상 pp / 추가1k inputs | Marginal Δpp / inputs | Dominated |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for group in summary['groups']:
        if not group['queries']:
            continue
        p, op, full = group['phases'], group['operational'], group['research']
        efficiency = full['delta_pp_per_1000_extra_inputs']
        formatted_efficiency = f'{efficiency:+.3f}' if efficiency is not None else '—'
        lines.append(f'| {group["total_rollout_budget"]} | {group["method"]} | {group["iou"]:.4f} / {group["delta_pp"]:+.2f} | '
            f'{p["main"]["decoder_inputs"]:.1f} | {p["tail"]["decoder_inputs"]:.1f} | {p["judge"]["decoder_inputs"]:.1f} | '
            f'{p["video_prefill"]["decoder_inputs"]:.1f} / {p["greedy_decode"]["decoder_inputs"]:.1f} | '
            f'{op["decoder_inputs"]:.1f} / {op["decoder_ratio"]:.2f}× | {full["decoder_inputs"]:.1f} / {full["decoder_ratio"]:.2f}× | '
            f'{full["forwards"]:.1f} / {full["forward_ratio"]:.2f}× | {full["attention_pairs"]:.0f} | '
            f'{formatted_efficiency} | {group["marginal_delta_pp"]:+.2f} / {group["marginal_extra_inputs"]:+.1f} | {group["dominated"]} |')
    lines += ['', '### Observed execution · shared-cache nested sweep', '',
        '아래 시간은 CUDA event 계측을 추가한 뒤의 공통 query에서 측정한다. 독립 조건 실행시간이나 단일 worker latency로 해석하지 않는다. CUDA stream·wall 각각 main/tail/judge를 표시하고 전체 nested sweep에는 이전 judge 비용도 포함한다.', '',
        '| 총 rollout | 방법 | 공통 n | Main CUDA / wall s | Tail CUDA / wall s | Judge CUDA / wall s | Sweep phase 합 CUDA / wall s | Sweep elapsed s | Sweep forwards | Mean peak allocated / reserved (max allocated) GiB |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in summary['groups']:
        obs = group.get('observed')
        if not obs:
            continue
        values = ' | '.join(f'{obs[phase]["cuda_forward_seconds"]:.3f} / {obs[phase]["wall_seconds"]:.3f}' for phase in ('main','tail','judge','sweep'))
        lines.append(f'| {group["total_rollout_budget"]} | {group["method"]} | {group["measured_queries"]} | {values} | '
            f'{obs["sweep_elapsed_seconds"]:.3f} | {obs["sweep"]["forwards"]:.1f} | '
            f'{obs["peak_allocated_gib"]:.2f} / {obs["peak_reserved_gib"]:.2f} ({obs["max_peak_allocated_gib"]:.2f}) |')
    if not summary['measured_common_queries']:
        lines += ['', '전체12조건의 두 seed 실측이 완료된 공통 query를 기다리는 중이다. 원자료에는 완료되는 checkpoint부터 구간별 실측을 즉시 기록한다.']
    else:
        observed = next(g['observed'] for g in summary['groups'] if g.get('observed'))['baseline']
        prefill, greedy = observed['video_prefill'], observed['greedy_decode']
        lines += ['', f'같은 공통 query의 video prefill 평균 CUDA/wall={prefill["cuda_forward_seconds"]:.3f}/{prefill["wall_seconds"]:.3f}s, '
            f'greedy decode={greedy["cuda_forward_seconds"]:.3f}/{greedy["wall_seconds"]:.3f}s다. '
            '이 기준 비용은 query마다 한 번 발생하며 위 Search sweep에는 포함하지 않았다. 전체 inference 비용에서는 함께 더한다.']
    return '\n'.join(lines)+'\n\n'


def report():
    _report()
    rows = cost_rows()
    summary = summarize(rows)
    write(OUT/'metrics/cost_per_query.json', rows)
    write(OUT/'metrics/costs.json', summary)
    path = RESULT/'REPORT.md'
    text = path.read_text()
    text = text.replace('## 실제 시도·중복·적응 선택', tables(summary)+'## 실제 시도·중복·적응 선택')
    text = text.replace('실제 전체 sweep 비용과cache 재사용은 raw actual_cost에 따로 보존한다.',
        '실제 전체 sweep 비용은 raw actual_cost·measured_actual_cost·checkpoint.cost에 보존한다.')
    path.write_text(text)
    spec = original.specification().replace('실제 전체 sweep GPU 비용은 raw actual_cost에 따로 기록한다.',
        '실제 forward의 CUDA stream 시간은 raw measured_actual_cost와 checkpoint.cost,기존 token·forward·wall 비용은 actual_cost에 기록한다.')
    (RESULT/'SPEC.md').write_text(spec+'\n'+definitions())


def initialize():
    original.report = report
    original.initialize()
    path = RESULT/'detail.html'
    html = path.read_text().replace('<div id="heading">', '<section><h2>성능과 연산 비용</h2><div id="cost-frontier">공통 query의 비용 집계를 기다리는 중…</div></section><div id="heading">')
    html = html.replace('</head>', '<style>#cost-frontier svg{width:100%;max-height:290px}#cost-frontier table{display:block;overflow:auto;max-width:100%}#case-cost{overflow-wrap:anywhere}</style></head>')
    path.write_text(html)
    js = OUT/'web/viewer.js'
    script = js.read_text()
    script += r'''
async function drawCostFrontier(){
 const summary=await get(new URL('metrics/costs.json',store));
 const groups=summary.groups.filter(g=>g.queries>0);if(!groups.length)return;
 const methods=['probability','token_entropy','temporal_mode'],colors=['#3974b9','#a66723','#22806a'];
 const baseline=groups[0].baseline;
 const xs=[baseline.decoder_inputs,...groups.map(g=>g.research.decoder_inputs)],xmin=Math.min(...xs),xmax=Math.max(...xs),dx=Math.max(1,xmax-xmin);
 let svg='<svg viewBox="0 0 690 260" role="img" aria-label="Independent trace work 대비 mIoU"><path d="M55 20V215H655" fill="none" stroke="#aab5c4"/>';
 for(let j=0;j<=4;j++)svg+=`<text x="${50+j*145}" y="236" font-size="10">${Math.round(xmin+dx*j/4)}</text><text x="20" y="${215-j*45}" font-size="10">${(j/4).toFixed(2)}</text>`;
 methods.forEach((method,i)=>{const rows=groups.filter(g=>g.method===method).sort((a,b)=>a.budget-b.budget);let pts=[];rows.forEach(g=>{const x=55+(g.research.decoder_inputs-xmin)/dx*580,y=215-g.iou*180;pts.push(`${x},${y}`);svg+=`<circle cx="${x}" cy="${y}" r="${g.dominated?3:5}" fill="${colors[i]}"><title>${method} · 총${g.total_rollout_budget} · n=${g.queries} · mIoU=${g.iou.toFixed(4)} · ${g.research.decoder_inputs.toFixed(1)} inputs · ${g.research.decoder_ratio.toFixed(2)}× greedy</title></circle><text x="${x+5}" y="${y-5}" font-size="10">${g.total_rollout_budget}</text>`;});svg+=`<polyline points="${pts.join(' ')}" stroke="${colors[i]}" fill="none"/>`;});
 svg+=`<circle class="greedy-reference" cx="${55+(baseline.decoder_inputs-xmin)/dx*580}" cy="${215-baseline.iou*180}" r="5" fill="#555"><title>Greedy · mIoU=${baseline.iou.toFixed(4)} · ${baseline.decoder_inputs.toFixed(1)} inputs</title></circle><text x="62" y="${210-baseline.iou*180}" font-size="11">greedy</text></svg>`;let table='<table><thead><tr><th>총 rollout</th><th>Method</th><th>ΔmIoU pp</th><th>Research inputs</th><th>Greedy ×</th><th>Forwards</th><th>향상 pp/추가1k inputs</th></tr></thead><tbody>';
 groups.forEach(g=>{table+=`<tr><td>${g.total_rollout_budget}</td><td>${g.method}</td><td>${g.delta_pp.toFixed(2)}</td><td>${g.research.decoder_inputs.toFixed(1)}</td><td>${g.research.decoder_ratio.toFixed(2)}</td><td>${g.research.forwards.toFixed(1)}</td><td>${g.research.delta_pp_per_1000_extra_inputs?.toFixed(3)??'—'}</td></tr>`;});
 document.getElementById('cost-frontier').innerHTML=svg+`<p class="muted">모든 조건의 공통 n=${summary.common_queries}/600. 가로축: independent research decoder inputs,세로축:mIoU. 점의 숫자는 총 rollout 상한. Probability 파랑 · token entropy 갈색 · temporal mode 초록. 시간 실측과 FLOPs가 아닌 trace workload입니다. 구간별 실제 CUDA/wall 시간과 memory는 보고서에 표시합니다. 실측 공통 n=${summary.measured_common_queries}.</p>`+table+'</tbody></table><p id="case-cost" class="muted"></p>';
}
function showCaseCost(){const output=document.getElementById('case-cost');if(!output)return;const point=data?.results?.[$('method').value+':'+$('seed').value]?.checkpoints?.[$('budget').value];const obs=point?.cost;if(!point){output.textContent='선택한 query의 checkpoint를 기다리는 중입니다.';return;}if(!obs){output.textContent='선택한 checkpoint는 계측 도입 전 결과이므로 시간 실측은 결측입니다. Independent trace work는 보고서에 포함됩니다.';return;}output.textContent='선택 사례 실제 비용 · '+['main','tail','judge'].map(phase=>`${phase}: ${obs[phase].forwards} forwards, ${obs[phase].decoder_inputs} inputs, CUDA ${obs[phase].cuda_forward_seconds.toFixed(3)}s / wall ${obs[phase].wall_seconds.toFixed(3)}s`).join(' · ')+` · sweep CUDA ${obs.sweep.cuda_forward_seconds.toFixed(3)}s / wall ${obs.sweep.wall_seconds.toFixed(3)}s. 이전 작은 예산의 judge 비용까지 포함한 관측값입니다.`;}
const renderWithoutCost=render;render=function(){renderWithoutCost();showCaseCost();};
async function refreshCost(){await drawCostFrontier();showCaseCost();}
refreshCost().catch(()=>{});setInterval(()=>refreshCost().catch(()=>{}),30000);['method','seed','budget'].forEach(id=>$(id).addEventListener('change',()=>showCaseCost()));
'''
    js.write_text(script)


if __name__ == '__main__':
    import sys
    initialize() if '--initialize' in sys.argv else report()
