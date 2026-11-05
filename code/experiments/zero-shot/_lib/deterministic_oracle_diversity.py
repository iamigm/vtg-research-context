"""CPU analysis of strict interval overlap between greedy, oracle and completed branches."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
FAMILY = 'rollout/deterministic'
ANALYSIS = 'oracle_diversity'
SOURCE = ROOT / 'artifacts/runs' / FAMILY / 'pilot/900/rollouts'
COHORT = ROOT / 'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'
BUDGETS = [1, 4, 8, 16, 32]
CATEGORIES = {
    'identical': 'Identical interval · greedy와 같은 구간',
    'overlapping': 'Overlapping intervals · 구간이 바뀌었지만 겹침',
    'disjoint': 'Disjoint intervals · 전혀 겹치지 않음',
    'invalid_greedy': 'Invalid greedy · 비교할 유효 구간 없음',
    'invalid_oracle': 'Invalid oracle · 유효 후보 없음',
}
MARKER = '<!-- oracle-diversity-reference -->'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(candidate):
    p = candidate.get('parsed')
    return [p['start'], p['end']] if candidate.get('stopped_on_eos') and p and p.get('positive') else None


def intersection(a, b):
    return max(0., min(a[1], b[1]) - max(a[0], b[0])) if a and b else 0.


def iou(a, b):
    return intersection(a, b) / max(1e-12, max(a[1], b[1]) - min(a[0], b[0])) if a and b else 0.


def category(a, b):
    if not b:
        return 'invalid_oracle'
    if not a:
        return 'invalid_greedy'
    if a == b:
        return 'identical'
    return 'overlapping' if intersection(a, b) > 0 else 'disjoint'


def counts(ids, spans, oracle):
    valid = [j for j in ids if spans[j] is not None]
    overlaps = sum(intersection(spans[j], oracle) > 0 for j in valid)
    return dict(total=len(ids), overlap=overlaps,
                disjoint=len(valid) - overlaps if oracle else 0,
                invalid=len(ids) - len(valid))


def evaluate(record, row, budget):
    cs = record['result']['candidates'][:budget]
    spans = [interval(c) for c in cs]
    scores = [iou(s, row['span'][0]) for s in spans]
    valid = [j for j, s in enumerate(spans) if s]
    chosen = max(valid, key=lambda j: (scores[j], -cs[j]['index'])) if valid else 0
    oracle, greedy = spans[chosen], spans[0]
    relation = category(greedy, oracle)
    other = [j for j in range(len(cs)) if j != chosen]
    branches = list(range(1, len(cs)))
    other_branches = [j for j in branches if j != chosen]
    unique_other = {tuple(s) for s in spans if s and s != oracle}
    ties = [j for j in valid if math.isclose(scores[j], scores[chosen], rel_tol=0., abs_tol=1e-12)]
    tie_categories = sorted({category(greedy, spans[j]) for j in ties})
    chain = []
    node = cs[chosen]
    while node.get('branch'):
        parent = cs[node['branch']['parent'] - 1]
        chain.append(dict(parent=parent['index'], child=node['index'], roles=node['branch']['roles'],
                          raw_token_probability=node['branch']['raw_probability'],
                          parent_overlaps_oracle=intersection(interval(parent), oracle) > 0,
                          child_overlaps_oracle=intersection(interval(node), oracle) > 0,
                          parent_child_relation=category(interval(parent), interval(node))))
        assert parent['index'] < node['index']
        node = parent
    delta = scores[chosen] - scores[0]
    assert delta >= -1e-12
    return dict(budget=budget, candidates=len(cs), oracle_index=cs[chosen]['index'],
                greedy_iou=scores[0], oracle_iou=scores[chosen], delta_iou=delta,
                improved=delta > 1e-12, material_improvement=delta > .01,
                oracle_correct=scores[chosen] >= .5, relation=relation,
                greedy_oracle_iou=iou(greedy, oracle),
                other_candidates=counts(other, spans, oracle),
                all_branches=counts(branches, spans, oracle),
                other_branches=counts(other_branches, spans, oracle),
                unique_other_intervals=len(unique_other),
                overlapping_unique_other_intervals=sum(intersection(s, oracle) > 0 for s in unique_other),
                oracle_ties=len(ties), tie_categories=tie_categories,
                tie_category_ambiguous=len(tie_categories) > 1,
                oracle_out_of_duration=bool(oracle and oracle[1] > row['duration']),
                exact_oracle_paths=sum(s == oracle for s in spans) if oracle else 0,
                oracle_chain=list(reversed(chain)))


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def rate(n, d):
    return f'{n}/{d} ({100*n/d:.2f}%)' if d else '0/0 (—)'


def summarize(rows, budget):
    data = [row['budgets'][str(budget)] for row in rows]
    improved = [x for x in data if x['improved']]
    material = [x for x in data if x['material_improvement']]
    correct = [x for x in data if x['oracle_correct']]
    gain = sum(x['delta_iou'] for x in data)
    categories = {}
    for key in CATEGORIES:
        subset = [x for x in data if x['relation'] == key]
        categories[key] = dict(n=len(subset), improved=sum(x['improved'] for x in subset),
            material_improvement=sum(x['material_improvement'] for x in subset),
            oracle_correct=sum(x['oracle_correct'] for x in subset),
            improved_and_correct=sum(x['oracle_correct'] and x['improved'] for x in subset),
            contribution_to_miou=sum(x['delta_iou'] for x in subset) / len(data),
            share_of_gain=sum(x['delta_iou'] for x in subset) / gain if gain else None)
    groups = {}
    for field in ['other_candidates','all_branches','other_branches']:
        total = sum(x[field]['total'] for x in data)
        overlap = sum(x[field]['overlap'] for x in data)
        invalid = sum(x[field]['invalid'] for x in data)
        groups[field] = dict(total=total, overlap=overlap, invalid=invalid,
            disjoint=sum(x[field]['disjoint'] for x in data),
            pooled_fraction=overlap / total if total else None,
            macro_fraction=mean([x[field]['overlap']/x[field]['total'] for x in data if x[field]['total']]),
            macro_n=sum(x[field]['total'] > 0 for x in data),
            average_overlaps=mean([x[field]['overlap'] for x in data]))
    return dict(queries=len(data),budget=budget,
        greedy_miou=mean([x['greedy_iou'] for x in data]),oracle_miou=mean([x['oracle_iou'] for x in data]),
        greedy_r={str(t):mean([x['greedy_iou'] >= t for x in data]) for t in [.3,.5,.7]},
        oracle_r={str(t):mean([x['oracle_iou'] >= t for x in data]) for t in [.3,.5,.7]},
        improved=len(improved),material_improvement=len(material),oracle_correct=len(correct),
        singleton_queries=sum(x['other_candidates']['overlap'] == 0 for x in data),
        correct_singleton_queries=sum(x['other_candidates']['overlap'] == 0 for x in correct),
        singleton_non_greedy=sum(x['other_candidates']['overlap'] == 0 and x['oracle_index'] > 1 for x in data),
        out_of_duration=sum(x['oracle_out_of_duration'] for x in data),
        tied_queries=sum(x['oracle_ties'] > 1 for x in data),
        tied_correct_queries=sum(x['oracle_ties'] > 1 for x in correct),
        tie_category_ambiguous=sum(x['tie_category_ambiguous'] for x in correct),
        categories=categories,groups=groups,
        overlap_count_histogram=dict(sorted(collections.Counter(x['other_candidates']['overlap'] for x in data).items())),
        macro_unique_interval_fraction=mean([x['overlapping_unique_other_intervals']/x['unique_other_intervals'] for x in data if x['unique_other_intervals']]))


def definitions(n):
    return f'''## Measurement — 무엇을 비교했는가

고정 pilot{n} query의 기존 deterministic rollout을 CPU로 분석했다. 모델은 Qwen3-VL-8B-Instruct, 원래 2 FPS·spatial target 100,352 pixels 입력이다. 새로운 GPU 추론·prompt 변경은 하지 않았다. pilot300·600·900은 canonical900 원자료의 고정 prefix subset이다.

**구간이 실제로 겹치는지만 측정한다.** 두 interval의 교집합 길이가 양수이면 overlapping, 0이면 disjoint다. 끝점만 닿는 것도 disjoint에 포함한다. IoU 0.1·0.5 cutoff나 center distance·semantic identity 판정은 사용하지 않는다.

interval a=[s_a,e_a], b=[s_b,e_b]의 교집합 길이는 max(0, min(e_a,e_b)−max(s_a,s_b))다. 이 값이 양수인지로 아래 모든 overlap 비율을 계산한다. **예측과 oracle의 overlap**과 **예측과 GT의 IoU**는 다른 값이다.

### Oracle와 greedy의 비교

각 query에서 앞선 K개 경로 중 GT IoU 최대 후보가 oracle이다(K=1/4/8/16/32). 동률이면 작은 candidate index를 선택한다. EOS 도달·parsing 성공·0≤start<end를 유효 조건으로 쓰고 duration 초과는 기존 평가와 같이 제외하지 않는다. 유효 후보가 없으면 IoU=0이다.

- Identical interval: oracle와 greedy의 start/end가 정확히 같다.
- Overlapping intervals: interval은 달라졌지만 교집합 길이가 양수다.
- Disjoint intervals: 유효한 두 interval의 교집합 길이가 0이다.
- Invalid greedy/oracle: 유효 interval이 없어 비교할 수 없다.

여기서는 겹치는 변화와 겹치지 않는 이동만 구분하며, action의 실제 의미를 판정하지 않는다. 개선 query는 oracle GT IoU−greedy GT IoU>10⁻¹²인 경우다. 아주 작은 개선의 영향을 확인하도록 ΔIoU>0.01도 병기한다. oracle IoU≥0.5 subset도 따로 표시한다.

mIoU gain contribution은 해당 범주의 (oracle IoU−greedy IoU) 합 / 전체 query 수다. 모든 범주를 더하면 전체 ΔmIoU다. gain share는 해당 범주의 gain 합 / 전체 gain 합이다.

### 전체 분기 중 oracle와 겹치는 비율

**분기**는 실제 EOS까지 생성한 non-greedy 완성 경로다. 미완성 queue prefix는 interval을 알 수 없어 분모에 넣지 않는다. 같은 interval로 이어지는 서로 다른 token 경로는 각각 센다.

| 집계 대상 | 포함·제외 기준 |
|---|---|
| Other candidates | K개에서 oracle 자신 한 경로만 제외한다. greedy는 포함한다. |
| All branches | non-greedy 경로 전체다. oracle가 non-greedy라면 그 경로도 포함한다. |
| Other branches | non-greedy 경로에서 oracle 자신도 제외한다. 주 분기 유사 비율은 이 집계다. |

- **Macro fraction**: query마다 (oracle와 겹치는 경로 수 / 해당 경로 수)를 구한 뒤 query를 동일 가중 평균한다. 분모 0인 query는 평균에서 제외하고 N을 표시한다.
- **Pooled fraction**: 전체 query의 겹치는 경로 수 합 / 전체 경로 수 합이다. macro와 구분한다.
- **Singleton**: oracle 자신을 뺀 나머지 모든 후보 중 oracle와 겹치는 것이 하나도 없는 query다. greedy도 이 비교에 포함한다. 이것이 후보 집합 안에서 oracle가 얼마나 혼자 떨어져 있는지를 나타낸다.
- **Unique interval fraction**: 동일 start/end를 합치고 oracle interval을 제외한 뒤 overlap 비율을 query별 평균한다. 중복 token 경로의 영향을 확인하는 보조 지표다.

invalid 경로는 전체 경로 수 분모에는 포함하되 overlap이나 disjoint로 분류하지 않고 별도 수를 표시한다. deterministic 경로 빈도는 sampling probability가 아니다. oracle 자신의 자명한 일치를 제외한 결과와 포함한 결과를 모두 제공한다.

성능 지표 mIoU는 query별 GT IoU의 동일 가중 평균이며 높을수록 좋다. R@τ는 GT IoU≥τ query 비율(τ=.3/.5/.7)이며 표에서는 %다. Δ는 같은 query의 greedy 대비이며 R의 변화는 pp다. 실패는0점이다.
'''


def performance_table(rows):
    lines=['| K | greedy mIoU | oracle mIoU (Δ) | oracle R@.3 (Δpp) | oracle R@.5 (Δpp) | oracle R@.7 (Δpp) | 개선 query |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for k in BUDGETS:
        s=summarize(rows,k)
        lines.append(f"| {k} | {s['greedy_miou']:.4f} | {s['oracle_miou']:.4f} ({s['oracle_miou']-s['greedy_miou']:+.4f}) | "+' | '.join(f"{100*s['oracle_r'][str(t)]:.2f}% ({100*(s['oracle_r'][str(t)]-s['greedy_r'][str(t)]):+.2f})" for t in [.3,.5,.7])+f" | {s['improved']}/{len(rows)} |")
    return lines


def report(rows, n):
    dest=ROOT/'results'/FAMILY/'pilot'/str(n)/ANALYSIS
    out=ROOT/'artifacts/runs'/FAMILY/'pilot'/str(n)/ANALYSIS
    dest.mkdir(parents=True,exist_ok=True)
    summaries={str(k):summarize(rows,k) for k in BUDGETS}
    save(out/'metrics/summary.json',summaries)
    lines=[f'# Oracle interval overlap · pilot{n}', '', definitions(n), '## Oracle performance', '']+performance_table(rows)
    for k in [8,32]:
        s=summaries[str(k)]
        lines += ['',f'## Greedy → oracle · K={k}', '',
          '| interval relation | 전체 query | 개선 query 중 | ΔIoU>0.01 중 | 개선 & oracle GT IoU≥.5 | ΔmIoU 기여 | 전체 gain 중 비율 |',
          '|---|---:|---:|---:|---:|---:|---:|']
        for key,label in CATEGORIES.items():
            c=s['categories'][key]
            lines.append(f"| {label} | {rate(c['n'],n)} | {rate(c['improved'],s['improved'])} | {rate(c['material_improvement'],s['material_improvement'])} | {c['improved_and_correct']} | {c['contribution_to_miou']:.4f} | {fmt(100*c['share_of_gain'] if c['share_of_gain'] is not None else None,2)}% |")
        lines += ['',f"Oracle GT IoU≥.5: {s['oracle_correct']}/{n}. duration 초과 oracle: {s['out_of_duration']}/{n}.",
            f"최대 GT IoU에 동률인 경로가 2개 이상인 query는 {s['tied_queries']}/{n}; oracle IoU≥.5에서는 {s['tied_correct_queries']}/{s['oracle_correct']}다. 이 subset에서 동률 선택에 따라 위 interval relation이 달라질 수 있는 query는 {s['tie_category_ambiguous']}개다. 표는 작은 index tie-break를 고정한 결과다.", '',
            f'## 다른 후보·분기와 oracle의 overlap · K={k}', '',
            '| 대상 | 겹치는 경로 / 전체 경로 | 겹치지 않는 유효 경로 | invalid | macro overlap fraction (N) | 평균 겹치는 경로 수/query |',
            '|---|---:|---:|---:|---:|---:|']
        for field,label in [('other_candidates','Other candidates · greedy 포함, oracle 제외'),('all_branches','All branches · oracle 포함'),('other_branches','Other branches · greedy·oracle 제외')]:
            a=s['groups'][field]
            lines.append(f"| {label} | {rate(a['overlap'],a['total'])} | {a['disjoint']} | {a['invalid']} | {fmt(100*a['macro_fraction'] if a['macro_fraction'] is not None else None,2)}% ({a['macro_n']}) | {a['average_overlaps']:.2f} |")
        lines += ['', f"다른 후보가 oracle와 하나도 겹치지 않는 singleton: {rate(s['singleton_queries'],n)}. oracle GT IoU≥.5인 경우만 보면 {rate(s['correct_singleton_queries'],s['oracle_correct'])}. 동일 interval 경로를 합친 unique interval fraction은 {fmt(100*s['macro_unique_interval_fraction'] if s['macro_unique_interval_fraction'] is not None else None,2)}%다."]
        if k == 32:
            lines += ['', '## Greedy와 겹치지 않는 oracle만 따로 본 결과 · K=32', '',
                '전체 평균은 greedy 주변 후보가 많은 query의 영향을 받으므로 disjoint subset도 따로 집계한다. 다른 분기에서는 greedy와 oracle 자신을 제외한다.', '',
                '| subset | query 수 | 다른 분기 overlap macro | pooled 겹치는 수/분기 수 | 다른 겹치는 후보 수 평균 | singleton |',
                '|---|---:|---:|---:|---:|---:|']
            for label,subset in [('모든 query',rows),
                ('greedy와 disjoint',[r for r in rows if r['budgets']['32']['relation']=='disjoint']),
                ('greedy와 disjoint & oracle GT IoU≥.5',[r for r in rows if r['budgets']['32']['relation']=='disjoint' and r['budgets']['32']['oracle_correct']])]:
                if not subset:
                    continue
                q=summarize(subset,32);a=q['groups']['other_branches']
                lines.append(f"| {label} | {len(subset)} | {100*a['macro_fraction']:.2f}% | {rate(a['overlap'],a['total'])} | {q['groups']['other_candidates']['average_overlaps']:.2f} | {rate(q['singleton_queries'],len(subset))} |")
    lines += ['', '## Overlap by budget', '', '| K | greedy와 disjoint인 oracle | 다른 분기 overlap macro | 다른 분기 overlap pooled | oracle GT IoU≥.5 중 singleton |', '|---|---:|---:|---:|---:|']
    for k in BUDGETS[1:]:
        s=summaries[str(k)];a=s['groups']['other_branches']
        lines.append(f"| {k} | {rate(s['categories']['disjoint']['n'],n)} | {fmt(100*a['macro_fraction'] if a['macro_fraction'] is not None else None,2)}% | {fmt(100*a['pooled_fraction'] if a['pooled_fraction'] is not None else None,2)}% | {rate(s['correct_singleton_queries'],s['oracle_correct'])} |")
    lines += ['', '## Dataset · K=32', '', '| dataset | N | identical | overlapping change | disjoint | 개선 중 disjoint | 다른 분기 overlap macro |', '|---|---:|---:|---:|---:|---:|---:|']
    for ds in sorted({r['dataset'] for r in rows}):
        subset=[r for r in rows if r['dataset']==ds];s=summarize(subset,32)
        lines.append(f"| {ds} | {len(subset)} | "+' | '.join(str(s['categories'][key]['n']) for key in ['identical','overlapping','disjoint'])+f" | {rate(s['categories']['disjoint']['improved'],s['improved'])} | {100*s['groups']['other_branches']['macro_fraction']:.2f}% |")
    lines += ['', '## Cases · K=32', '', '아래는 큰 개선이 있었던 query의 원자료 참조이며 대표성을 보장하는 표본은 아니다.', '', '| sample | relation | greedy GT IoU → oracle GT IoU | oracle candidate | other overlapping candidates |', '|---|---|---|---:|---:|']
    cases=[]
    for key in ['disjoint','overlapping']:
        eligible=[r for r in rows if r['budgets']['32']['relation']==key and r['budgets']['32']['improved'] and r['budgets']['32']['oracle_correct']]
        cases += sorted(eligible,key=lambda r:(-r['budgets']['32']['delta_iou'],r['sample_key']))[:4]
    for r in cases:
        x=r['budgets']['32'];href='../viewer.html#preset=all&sample='+r['sample_key']+'&candidate='+str(x['oracle_index'])+'&panel=distributions'
        lines.append(f"| [{r['sample_key']}]({href}) | {CATEGORIES[x['relation']]} | {x['greedy_iou']:.4f} → {x['oracle_iou']:.4f} | {x['oracle_index']} | {x['other_candidates']['overlap']} |")
    prefix='../../../../../../artifacts/runs/rollout/deterministic/pilot/'
    canonical=prefix+'900/oracle_diversity/'
    lines += ['', '## Reproducibility', '',
        '[조건별 성능](STRATA.md) · [summary]('+prefix+str(n)+'/oracle_diversity/metrics/summary.json) · [query별 파생값]('+canonical+'metrics/per_query.json) · [완성 분기별 비교]('+canonical+'metrics/branches.csv)', '',
        'query별 파생값·source SHA-256·oracle의 parent chain은 canonical900 analysis에 한 번만 저장한다. 작은 cohort는 같은 배열의 앞부분을 참조한다. 원래 응답·GT·frame·token distribution은 복제하거나 변경하지 않았다.', '',
        '재생성: `python experiments/zero-shot/_lib/deterministic_oracle_diversity.py`.']
    from rollout_boundary_clipping import integrate_text
    (dest/'REPORT.md').write_text(integrate_text('\n'.join(lines)+'\n',n,'oracle'))
    strata(rows,dest)
    return summaries


def strata(rows,dest):
    lines=[f'# Stratified oracle interval overlap · pilot{len(rows)}', '', '각 표는 해당 조건의 query만 분모로 사용한다. n<10은 small, n=0은 empty다. R은 %이고 Δ는 같은 subset의 greedy 대비다. overlap의 정의와 분모는 REPORT.md를 따른다.', '']
    for ds in sorted({r['dataset'] for r in rows}):
        sub=[r for r in rows if r['dataset']==ds]
        lines += ['## '+ds, '']+performance_table(sub)+['']
    db=[(0,30),(30,60),(60,120),(120,180),(180,float('inf'))]
    rb=[(0,.05),(.05,.1),(.1,.2),(.2,.4),(.4,1.000001)]
    for axis,bins in [('duration',db),('gt_ratio',rb)]:
        for a,b in bins:
            sub=[r for r in rows if a<=r[axis]<b]
            lines += [f'## {axis} [{a},{b}) · n={len(sub)}'+(' small' if 0<len(sub)<10 else ' empty' if not sub else ''),'']+(performance_table(sub) if sub else ['empty'])+['']
    for metric,t in [('mIoU',None),('R@.3',.3),('R@.5',.5),('R@.7',.7)]:
        lines += ['## duration × GT/video duration · '+metric,'','| duration | GT ratio | n | K | greedy | oracle | Δ / Δpp |','|---|---|---:|---:|---:|---:|---:|']
        for da,dbb in db:
            for ra,rbb in rb:
                sub=[r for r in rows if da<=r['duration']<dbb and ra<=r['gt_ratio']<rbb]
                for k in BUDGETS:
                    g=mean([r['budgets'][str(k)]['greedy_iou'] if t is None else 100*(r['budgets'][str(k)]['greedy_iou']>=t)for r in sub])
                    o=mean([r['budgets'][str(k)]['oracle_iou'] if t is None else 100*(r['budgets'][str(k)]['oracle_iou']>=t)for r in sub])
                    mark=' small' if 0<len(sub)<10 else ' empty' if not sub else ''
                    lines.append(f'| [{da},{dbb}) | [{ra},{rbb}) | {len(sub)}{mark} | {k} | {fmt(g)} | {fmt(o)} | {fmt(o-g if o is not None else None)} |')
    lines += ['', '## Interval overlap · K=32', '', '| duration | GT ratio | n | disjoint oracle | overlapping change | identical | other-branch overlap macro |','|---|---|---:|---:|---:|---:|---:|']
    for da,dbb in db:
        for ra,rbb in rb:
            sub=[r for r in rows if da<=r['duration']<dbb and ra<=r['gt_ratio']<rbb]
            if not sub:
                lines.append(f'| [{da},{dbb}) | [{ra},{rbb}) | 0 empty | — | — | — | — |')
                continue
            s=summarize(sub,32)
            lines.append(f'| [{da},{dbb}) | [{ra},{rbb}) | {len(sub)}'+(' small'if len(sub)<10 else '')+' | '+' | '.join(str(s['categories'][key]['n'])for key in ['disjoint','overlapping','identical'])+f" | {100*s['groups']['other_branches']['macro_fraction']:.2f}% |")
    (dest/'STRATA.md').write_text('\n'.join(lines)+'\n')


def analysis_reference(count):
    dest=ROOT/'results'/FAMILY/'pilot'/str(count)/ANALYSIS/'REPORT.md'
    if not dest.exists():
        return []
    return ['',MARKER,'## Oracle interval overlap', '',
        '[Oracle와 greedy가 겹치는지, 전체 완성 분기 중 oracle와 겹치는 비율](oracle_diversity/REPORT.md). 교집합 길이>0만 overlap으로 사용한 CPU 분석이다.']


def main():
    cohort=read(COHORT);assert len(cohort)==900
    rows=[];branches=[];source_hashes={}
    for row in cohort:
        path=SOURCE/(row['sample_key']+'.json');record=read(path)
        assert record['sample']==row
        assert len({tuple(c['ids']) for c in record['result']['candidates']})==len(record['result']['candidates'])
        sha=digest(path);source_hashes[row['sample_key']]=sha
        item=dict(sample_key=row['sample_key'],dataset=row['dataset'],video=row['video_path'],duration=row['duration'],
                  gt_ratio=(row['span'][0][1]-row['span'][0][0])/row['duration'],source=str(path.relative_to(ROOT)),source_sha256=sha,
                  budgets={str(k):evaluate(record,row,k) for k in BUDGETS})
        assert all(item['budgets'][str(a)]['oracle_iou']<=item['budgets'][str(b)]['oracle_iou']+1e-12 for a,b in zip(BUDGETS,BUDGETS[1:]))
        for x in item['budgets'].values():
            for field in ['other_candidates','all_branches','other_branches']:
                a=x[field]
                assert a['overlap']+a['disjoint']+a['invalid']==a['total']
        rows.append(item)
        o=interval(record['result']['candidates'][item['budgets']['32']['oracle_index']-1])
        for c in record['result']['candidates'][1:]:
            s=interval(c)
            branches.append(dict(sample_key=row['sample_key'],candidate=c['index'],parent=c['branch']['parent'],oracle_index=item['budgets']['32']['oracle_index'],
                                 valid=s is not None,is_oracle=c['index']==item['budgets']['32']['oracle_index'],overlaps_oracle=bool(s and intersection(s,o)>0),
                                 iou_to_oracle=iou(s,o),iou_to_gt=iou(s,row['span'][0]),roles=';'.join(c['branch']['roles'])))
    previous=ROOT/'artifacts/runs'/FAMILY/'pilot/900/metrics/per_query.csv'
    bykey={r['sample_key']:r for r in rows};verified=0
    with previous.open() as f:
        for row in csv.DictReader(f):
            if row['method']=='oracle':
                assert math.isclose(bykey[row['sample_key']]['budgets'][row['budget']]['oracle_iou'],float(row['iou']),abs_tol=1e-12)
                verified+=1
    assert verified==4500
    out=ROOT/'artifacts/runs'/FAMILY/'pilot/900'/ANALYSIS
    save(out/'metrics/per_query.json',rows)
    with (out/'metrics/branches.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(branches[0]));writer.writeheader();writer.writerows(branches)
    summaries={}
    for n in [300,600,900]:
        frozen=read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json')
        assert frozen==cohort[:n]
        summaries[n]=report(rows[:n],n)
        dest=ROOT/'results'/FAMILY/'pilot'/str(n)/'REPORT.md'
        text=dest.read_text().split(MARKER)[0].rstrip()
        dest.write_text(text+'\n'+'\n'.join(analysis_reference(n))+'\n')
    assert all(digest(SOURCE/(key+'.json'))==sha for key,sha in source_hashes.items())
    save(out/'validation/analysis.json',dict(passed=True,queries=900,branches=len(branches),budgets=BUDGETS,
         original_oracle_rows_identical=verified,source_hashes_verified=900,cohort_prefixes=[300,600,900],
         source_code_sha256=digest(Path(__file__)),cohort_sha256=digest(COHORT),
         conditions=dict(overlap='intersection length > 0; touching endpoints are disjoint',
                         improvement=1e-12,material_improvement=.01,oracle_correct=.5)))
    from artifact_store import register
    register('deterministic.oracle_diversity.pilot900','analyses',out/'metrics',
             'Strict interval overlap of greedy/oracle and completed branches; fixed900 with300/600 prefixes; no semantic inference.',
             producer='deterministic_oracle_diversity.py',cohort=str(COHORT))
    print(json.dumps(dict(passed=True,queries=900,branches=len(branches),main=summaries[900]['32']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
