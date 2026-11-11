"""Cumulative MD reports and a lazy-loading, source-referencing attribution viewer."""
import argparse
import collections
import json
import math
import os
from pathlib import Path

from greedy_attribution_core import BASE, COHORT, CONFIG, GROUPS, RESULTS, ROOT, SOURCE, iou, read, sha, stamp, storage, write


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def fmt(value, digits=4):
    if value is None:
        return '—'
    if value != 0 and abs(value) < 10**(-digits):
        return f'{value:.3e}'
    return f'{value:.{digits}f}'


def relative(path, start):
    # /home/irteam and /home1/irteam are aliases of the same workspace.
    # Resolve both sides before computing browser URLs, including future files.
    return os.path.relpath(Path(path).resolve(), Path(start).resolve())


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def record_summary(row, record):
    steps = record.get('steps', [])
    targets = record.get('targets', [])
    candidates = [candidate for step in steps for candidate in step.get('candidates', [])]
    kinds = collections.Counter(t.get('kind', 'unknown') for t in targets)
    branches = [t for t in targets if t.get('alternative') is not None]
    modes = [t.get('source_mode', t.get('mode', t.get('branch', {}).get('mode', {}))) for t in branches]
    groups = collections.defaultdict(list)
    for target in targets:
        if target.get('kind') != 'chosen_log_probability':
            continue
        for group in target.get('groups', []):
            groups[group['group']].append(group)
    group_means = {}
    for name, values in groups.items():
        group_means[name] = dict(targets=len(values), grad_l2=mean([v.get('grad_l2') for v in values]),
                                 signed_sum=mean([v.get('signed_sum') for v in values]),
                                 absolute_sum=mean([v.get('absolute_sum') for v in values]))
    validation = record.get('validation', {})
    cost = record.get('cost', {})
    greedy = record.get('greedy', {})
    return dict(key=row['sample_key'], dataset=row['dataset'], duration=row['duration'],
                gt=row['span'][0], video_path=row['video_path'], greedy_span=greedy.get('span'),
                iou=iou(greedy.get('span'), row['span'][0]), steps=len(steps), targets=len(targets),
                kinds=dict(kinds), branch_targets=len(branches),
                current_candidates=sum(bool(c.get('current_competitor')) for c in candidates),
                potential_candidates=sum(bool(c.get('potential_competitor')) for c in candidates),
                potential_only_candidates=sum(bool(c.get('potential_competitor')) and not c.get('current_competitor') for c in candidates),
                distinct_mode_candidates=sum(bool(m.get('distinct_mode_candidate')) for m in modes),
                unknown_mode_candidates=sum(m.get('status', 'unobserved') == 'unobserved' for m in modes),
                replay_passed=bool(validation.get('passed', validation.get('input_hashes_match') and validation.get('causality_passed') and validation.get('greedy_argmax_match') and validation.get('replay_max_probability_error', math.inf) <= CONFIG['replay_probability_tolerance'] and validation.get('gradient_forward_max_logit_error', math.inf)==0)),
                strict_argmax_match=bool(validation.get('greedy_argmax_match')), near_tie_steps=len(validation.get('greedy_near_tie_steps', [])),
                top30_mass=mean([s.get('top30_mass') for s in steps]), omitted_mass=mean([s.get('omitted_mass') for s in steps]),
                source_drift_sensitive_candidates=sum(bool(c.get('margin_sensitive_to_replay')) for c in candidates),
                source_drift_sensitive_targets=sum(bool(t.get('branch', {}).get('margin_sensitive_to_replay')) for t in branches),
                max_top30_probability_error=validation.get('replay_max_probability_error'),
                gradient_forward_max_logit_error=validation.get('gradient_forward_max_logit_error'),
                validation=validation, cost=cost, groups=group_means)


def accuracy(records):
    values = [r['iou'] if r['iou'] is not None else 0.0 for r in records]
    return [mean(values)] + [mean([float(v >= threshold) for v in values]) for threshold in (.3, .5, .7)]


def strata_appendix(records):
    duration_edges = [0, 30, 60, 120, 180, math.inf]
    fraction_edges = [0, .05, .10, .20, .40, math.inf]

    def bucket(value, edges):
        return next((i for i in range(len(edges)-1) if edges[i] <= value < edges[i+1]), len(edges)-2)

    def interval(edges, index):
        return f'[{edges[index]:g}, {edges[index+1]:g})' if math.isfinite(edges[index+1]) else f'≥ {edges[index]:g}'

    def fraction(row):
        return (row['gt'][1]-row['gt'][0])/row['duration']

    lines = ['# Greedy attribution · 상세 정의와 층화 결과', '',
             '이 표는 새 답변의 성능 개선을 측정하지 않는다. 고정된 source greedy 답변의 진단 표본 성능과 계측 coverage를 함께 기록한다. ' 
             'IoU=예측·GT 교집합 길이÷합집합 길이, mIoU=query별 IoU 평균, R@τ=IoU≥τ인 query 수÷평가 query 수다. ' 
             '유효 구간이 없으면 IoU=0으로 평가한다. 네 지표 모두 높을수록 좋으며 동일 source greedy 대비 Δpp=0.00이다. ' 
             'n<10은 작은 표본, n=0은 빈 칸이다. 고정 cohort는 층화 진단 표본이며 전체 benchmark의 무작위 평균이 아니다.', '']
    categories = [('Dataset', sorted({r['dataset'] for r in records}), lambda r: r['dataset']),
                  ('영상 길이 s', [interval(duration_edges, i) for i in range(5)],
                   lambda r: interval(duration_edges, bucket(r['duration'], duration_edges))),
                  ('GT/영상 길이 비율', [interval(fraction_edges, i) for i in range(5)],
                   lambda r: interval(fraction_edges, bucket(fraction(r), fraction_edges)))]
    for title, labels, assign in categories:
        lines += [f'## {title}', '', '| 범위 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | Backward targets |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for label in labels:
            subset = [r for r in records if assign(r) == label]
            suffix = ' · 빈 칸' if not subset else ' · 작은 표본' if len(subset) < 10 else ''
            values = [f'{v:.4f} (+0.00)' if v is not None else '—' for v in accuracy(subset)]
            lines.append(f'| {label}{suffix} | {len(subset)} | ' + ' | '.join(values) + f' | {sum(r["targets"] for r in subset)} |')
        lines.append('')
    for metric, name in enumerate(('mIoU', 'R@.3', 'R@.5', 'R@.7')):
        lines += [f'## 영상 길이 × GT 비율 · {name}', '',
                  '각 칸은 완료 query 수 n, 해당 범위의 source greedy 값과 동일 source 대비 Δpp다.', '',
                  '| 영상 길이 s | ' + ' | '.join(interval(fraction_edges, i) for i in range(5)) + ' |',
                  '|---|'+'---|'*5]
        for d in range(5):
            cells = []
            for f in range(5):
                subset = [r for r in records if bucket(r['duration'], duration_edges) == d and bucket(fraction(r), fraction_edges) == f]
                value = accuracy(subset)[metric]
                cells.append(f'n={len(subset)}'+(' 작은 표본' if 0 < len(subset) < 10 else ' 빈 칸' if not subset else '')+
                             (f'; {value:.4f} (+0.00)' if value is not None else '; —'))
            lines.append('| '+interval(duration_edges, d)+' | '+' | '.join(cells)+' |')
        lines.append('')
    return lines


def definitions(config):
    return [
        '## 목적과 계측 정의', '',
        '원래 입력·frame sampling·spatial resolution·greedy prefix를 고정하고 모든 생성 token의 선택과 선행 입력의 연결을 계측한다. '
        '영상·query·timestamp·instruction·special token·이미 생성된 prefix를 포함한다. GT는 사후 평가에만 사용한다.', '',
        '- **Chosen log probability**: 각 step t의 Lₜ=log P(yₜ* | V,Q,y*<ₜ). Source greedy의 format·숫자·소수점·EOS를 고정한 계측 replay에서 측정하며 값이 클수록 해당 token의 replay probability가 높다.',
        '- **Endpoint log probability**: 시작 또는 종료 숫자 token들의 Lₜ 합. 종료는 이미 생성된 시작 숫자를 조건으로 한다. 다른 길이의 token열 사이에서는 길이 효과를 포함한다.',
        '- **Logit margin**: mₜ,k=zₜ(g)−zₜ(k)=log(Pₜ(g)/Pₜ(k)). 양수는 source greedy token g의 우세, 작을수록 두 선택이 가깝다. Margin gradient는 계측 replay 계산점에서 이 격차에 대한 민감도다. Source KV margin=log(p_source(g)/p_source(k))를 별도로 기록한다.',
        '- **Input gradient norm**: 입력 위치 i의 ‖∂L/∂xᵢ‖₂. 위치 또는 표시한 입력 group에서의 L2 norm이며 값이 클수록 지정한 표현의 작은 변화에 민감하다. 픽셀·embedding의 단위가 달라 raw 값을 곧바로 의미적 기여 비율로 비교하지 않는다.',
        '- **Signed activation × gradient**: ⟨xᵢ,∂L/∂xᵢ⟩. 해당 표현을 아주 조금 배율 확대하는 방향의 1차 변화량이다. 양수는 목표 증가, 음수는 목표 감소 방향이다. 그 신호의 제거 효과나 인과적 증거량은 아니다.',
        '- **Top30 entropy contribution**: H₃₀=−Σᵥ∈top30 pᵥ log pᵥ, 단위 nats. 계측 replay의 native probability를 사용한 부분합이며 full entropy 또는 top30 내부 재정규화 entropy가 아니다. Top30 mass=Σᵥ∈top30 pᵥ, omitted mass=1−top30 mass다. Omitted mass의 개별 token은 계측하지 않는다.',
        '', '## 현재 경쟁과 잠재 경쟁', '',
        f'현재 경쟁 candidate는 Source KV에서 고정한 top30 안에서 greedy 외 token 중 계측 replay native probability≥{config["current_min_probability"]:g} 또는 logit gap≤{config["current_max_logit_gap"]:.6f}인 token이다. '
        '분기 집합 안에서 probability를 재정규화하지 않는다.', '',
        f'Screening은 입력 종류별 표현을 ±{100*config["screen_epsilon"]:g}% 배율 변경해 source KV에서 고정한 top30 token의 계측 replay logits만 계측한다. '
        '영상은 normalized pixels, 텍스트는 입력 embedding에 대한 내부 진단이다. '
        '변화 이후 top30을 다시 선택하지 않는다. 방향별 중앙 유한차분으로 greedy와 경쟁 token의 gap 감소량을 추정하고, 가장 크게 줄어드는 방향과 부호를 저장한다. '
        f'Screen gap=계측 replay baseline logit gap−측정 방향의 최대 gap 감소 추정이며, {config["potential_max_screen_gap"]:g} 이하인 후보를 잠재 경쟁 대상으로 고려한다. '
        f'고정 top30에서 greedy 외 모든 candidate의 screening 수치를 보존하고 query당 최대 {config["detailed_branch_targets"]}개 분기에 상세 backward를 수행한다. '
        '현재 경쟁 후보와 현재 경쟁 기준을 통과하지 않은 후보의 최소 screen gap을 각각 우선 선택한다. 나머지는 기존 continuation의 시간적 분리와 screen gap 순으로 채우며 가능한 한 다른 step을 조사한다. '
        '따라서 상세 target에는 잠재 경쟁 threshold까지 도달하지 않은 낮은 probability probe도 포함될 수 있다. 낮은 probability를 곱해 민감도를 억제하지 않는다.', '',
        '이 screening은 source KV의 고정 top30 안의 token과 지정한 유한 개 변화 방향에서의 진단이다. Top30 밖의 분기는 미계측이다. 자연스러운 query 수정·실제 영상 변화나 모든 입력 방향을 포괄하지 않으며, '
        '낮은 sensitivity로도 분기를 영구 제외할 수 없다. 큰 변화의 비선형 급변이나 실제 top1 역전은 별도 재계산 확인이 필요하다.', '',
        '## Temporal mode 후보', '',
        f'정확히 같은 token prefix와 분기 token을 공유하는 기존 deterministic continuation을 참조한다. '
        f'Greedy 구간 대비 max(|Δstart|,|Δend|)>max({config["mode_endpoint_tolerance_seconds"]:g}s, '
        f'{config["mode_relative_tolerance"]:g}×영상 길이)이면 distinct mode candidate로 표시한다. '
        '이 기준은 구간의 시간적 분리를 판정하며 사건 의미의 차이를 확정하지 않는다. '
        '기존 continuation이 없으면 unknown으로 유지한다. 한 분기의 관측 continuation이나 다음 token probability를 최종 mode probability로 해석하지 않는다.', '',
        '## Source KV 재현과 gradient 계산', '',
        'Source KV probability는 기존 greedy 실행의 incremental decoding 기록이다. 계측 replay는 같은 원영상 prefill 이후 source token을 한 번에 하나씩 입력하는 계산 순서를 유지한다. '
        'KV는 immutable differentiable chunks로 연결해 prefill부터 모든 생성 step까지 gradient가 이어지도록 한다. 기존에 분리 저장한 inference KV를 가져와 gradient graph를 대체하지 않는다. '
        'Source KV와 계측 replay probability를 나란히 보존하며 어느 쪽도 top30 내부에서 재정규화하지 않는다. '
        'Max top30 |Δp|=query의 모든 step과 고정 top30 token에 대한 |p_replay−p_source|의 최댓값으로 낮을수록 재현 오차가 작다. '
        f'이 값은 {config["replay_probability_tolerance"]:g} 이하여야 하며, 모든 step의 전체-vocabulary argmax가 source greedy token과 정확히 일치해야 한다. '
        '이 argmax 검사는 top30 밖 분기의 sensitivity를 계측하는 것이 아니다. '
        '같은 계측 replay의 gradient-enabled forward와 no-grad forward 사이의 top30 logit 차이는 별도로 검사하며 0이어야 한다. '
        'Margin 차이 |m_replay−m_source|≥max(|m_replay|,10⁻⁶)이거나 두 margin의 부호가 다르면 재현 오차에 민감한 후보로 표시한다.', '',
        '## 내부 계측 coverage', '',
        'Layers 표는 각 decoder layer의 attention output, MLP output, residual input에서 선택한 prediction receiver의 signed activation × gradient다. '
        'Message score=Aᵢⱼ〈∂L/∂oᵢ,vⱼ〉는 o_proj 이전 head 좌표에서 메시지를 미소 강화하는 방향이다. '
        'Message 계측은 decoder layer 0·17·35의 prediction receivers에 한정하고 head·receiver·부호별 상위 source edge를 보존한다. '
        'Visual-to-visual 및 그 외 간접 receiver의 edge는 미계측이다. Attention row 재구성 오차를 원자료에 남긴다. '
        'Visual embedding gradient는 visual replacement 이후, DeepStack 추가 주입 이전의 language-model 입력에서 측정한다. '
        'Video pixel gradient는 vision encoder와 DeepStack 경로를 포함하므로 두 지도를 구분한다.', '',
        '## 해석과 재현 범위', '',
        '선행 입력 변화 이후에도 원래 prefix를 teacher-force하므로 현재 step에 대한 conditional 분석이다. '
        '실제 입력 변화가 더 앞선 생성부터 바꾸는 효과는 이 기록으로 확정할 수 없다. '
        'Gradient가 작다고 영상 근거가 없다고 판단하지 않는다. 이전 생성 위치의 표현에도 영상·query 정보가 들어 있을 수 있고, '
        'MLP를 곧바로 language prior로 이름 붙이지 않는다. 층별·경로별 signed 점수를 모두 합쳐 전체 증거량을 만들지 않는다.', '',
        '모델 파라미터는 고정하고 원영상 prefill부터 gradient 계산 그래프를 다시 구성한다. '
        '기존 inference KV를 독립적인 입력 증거로 사용하지 않는다. 원자료에는 입력 tensor hash, '
        'source greedy argmax 일치 여부, source KV와 계측 replay probability drift, 같은 계측 replay 계산의 gradient/no-grad 일치, 연산 시간과 memory를 기록한다. '
        f'계측 replay는 모든 source greedy argmax 일치와 top30 probability 최대 절대 오차≤{config["replay_probability_tolerance"]:g}를 요구한다.', '',
        '참고: [DeepFool의 margin·국소 선형화](https://arxiv.org/html/1511.04599v3). '
        '본 실험의 source-group screening과 attribution 계측은 해당 논문의 VTG 재현 실험이 아니다.', ''
    ]


def _video(row):
    source = SOURCE/'web/samples'/f'{row["sample_key"]}.json'
    if not source.exists():
        return None
    old = read(source)
    if not old.get('video'):
        return None
    from result_layout import relocate_path
    path = relocate_path((ROOT/'results/rollout/deterministic/pilot/900'/old['video']).resolve())
    return relative(path, ROOT.parent)


def report(scope=None, allow_partial=False, canonical=None):
    canonical = Path(canonical) if canonical is not None else storage()
    config_path = canonical/'metadata/config.json'
    config = read(config_path) if config_path.exists() else CONFIG
    rows = read(COHORT)
    scopes = [int(scope)] if scope is not None else [n for n in CONFIG['stages'] if n <= int(canonical.name)]
    published = []
    for count in scopes:
        scope_rows = rows[:count]
        catalog, records, refs = [], [], []
        for row in scope_rows:
            path = canonical/'queries'/f'{row["sample_key"]}.json'
            q = read(path) if path.exists() else None
            complete = bool(q and q.get('complete'))
            entry = dict(key=row['sample_key'], query=row['query'], dataset=row['dataset'], gt=row['span'][0],
                         duration=row['duration'], video=_video(row), complete=complete,
                         record=relative(path, ROOT.parent) if complete else None,
                         greedy_span=q.get('greedy', {}).get('span') if q else None,
                         greedy_iou=iou(q.get('greedy', {}).get('span'), row['span'][0]) if q else None)
            catalog.append(entry)
            if complete:
                records.append(record_summary(row, q))
                refs.append(dict(sample_key=row['sample_key'], path=str(path), sha256=sha(path)))
        if not records:
            continue
        if not allow_partial and len(records) != count:
            raise ValueError(f'pilot{count}: only {len(records)} complete query records')
        result = RESULTS/'pilot'/str(count)
        artifacts = BASE/str(count)
        summary = dict(completed=len(records), queries=count,
                       unique_videos=len({row['video_path'] for row in scope_rows}),
                       completed_unique_videos=len({r['video_path'] for r in records}),
                       steps=sum(r['steps'] for r in records), targets=sum(r['targets'] for r in records),
                       branch_targets=sum(r['branch_targets'] for r in records),
                       source_drift_sensitive_candidates=sum(r['source_drift_sensitive_candidates'] for r in records),
                       source_drift_sensitive_targets=sum(r['source_drift_sensitive_targets'] for r in records),
                       current_candidates=sum(r['current_candidates'] for r in records),
                       potential_candidates=sum(r['potential_candidates'] for r in records),
                       potential_only_candidates=sum(r['potential_only_candidates'] for r in records),
                       distinct_mode_candidates=sum(r['distinct_mode_candidates'] for r in records),
                       unknown_mode_candidates=sum(r['unknown_mode_candidates'] for r in records),
                       replay_passed=sum(r['replay_passed'] for r in records),
                       top30_mass=mean([r['top30_mass'] for r in records]), omitted_mass=mean([r['omitted_mass'] for r in records]),
                       greedy_metrics=accuracy(records))
        write(artifacts/'web/catalog.json', dict(scope=count, generated_at=stamp(), summary=summary,
              config=config, config_path=relative(config_path, ROOT.parent), samples=catalog))
        write(artifacts/'metadata/source_records.json', dict(queries=count, completed=len(records),
              cohort=str(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{count}/samples.json'),
              cohort_sha256=sha(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{count}/samples.json'),
              canonical=str(canonical), records=refs))
        write(artifacts/'metadata/report_summary.json', summary)
        lines = [f'# Greedy attribution · pilot{count}', '',
                 f'고정 pilot **{count} query** 중 **{len(records)} query**의 계측을 완료했다. '
                 f'전체 범위의 고유 영상은 {summary["unique_videos"]}개이며 완료 query에 포함된 고유 영상은 {summary["completed_unique_videos"]}개다. '
                 'Query 수·영상 수·backward target 수를 구분한다. '+
                 ('이 범위의 원자료 계측이 완료되었다. Viewer 검증의 완료 여부는 아래 검증 기록을 따른다.' if len(records)==count else
                  '진행 중 부분집합의 수치이며 pilot 전체 결과로 해석하지 않는다.'), '',
                 '## 목적과 현재 범위', '',
                 'Source greedy token열을 고정한 계측 replay에서 모든 선행 입력의 gradient를 계산하고, 현재 경쟁 분기와 지정한 입력 변화에 민감한 잠재 분기를 함께 추적한다. '
                 '원영상 prefill 이후 one-token decoding을 재현하고 differentiable KV로 모든 선행 입력까지 gradient를 연결한다. Source KV와 계측 replay probability를 함께 기록한다. 새 답변을 생성하거나 성능을 개선하는 실험이 아니다. 기존 deterministic continuation은 mode 후보의 사후 참조로만 사용한다.', '',
                 '[계측 정의·실행 조건·층화 결과](APPENDIX.md)', '',
                 '## 계측 coverage', '',
                 '| 항목 | 실제 수 | 분모·해석 |', '|---|---:|---|',
                 f'| 완료 query | {len(records)} / {count} | 같은 고정 cohort의 완료 query 수 |',
                 f'| Greedy step | {summary["steps"]} | 완료 query의 출력 token 위치 합계 |',
                 f'| Backward target | {summary["targets"]} | Chosen token·endpoint 합계·선별 margin 목표의 합계 |',
                 f'| 평균 계측 replay top30 mass | {fmt(summary["top30_mass"],6)} | step 내 합계를 query 평균 후 query 간 평균; 재정규화 없음 |',
                 f'| 평균 계측 replay omitted mass | {fmt(summary["omitted_mass"],6)} | 1−top30 mass; top30 밖 개별 분기는 미계측 |',
                 f'| 상세 branch target | {summary["branch_targets"]} | 다른 token과의 margin을 계측한 목표 수 |',
                 f'| 현재 경쟁 candidate | {summary["current_candidates"]} | step×token 후보 수; query 수가 아님 |',
                 f'| 잠재 경쟁 candidate | {summary["potential_candidates"]} | 현재 경쟁과 중복 가능 |',
                 f'| 잠재 경쟁만 해당 | {summary["potential_only_candidates"]} | 현재 경쟁 기준을 통과하지 않은 잠재 후보 |',
                 f'| 분리된 구간을 관측한 branch target | {summary["distinct_mode_candidates"]} / {summary["branch_targets"]} | 기존 continuation 기준 distinct mode candidate |',
                 f'| 재현 오차에 민감한 branch target | {summary["source_drift_sensitive_targets"]} / {summary["branch_targets"]} | replay margin에 비해 source와의 차이가 큰 목표 |',
                 f'| 기존 continuation 미관측 branch target | {summary["unknown_mode_candidates"]} / {summary["branch_targets"]} | mode를 unknown으로 유지 |', '',
                 '현재 경쟁은 source KV의 고정 top30 안에서 계측 replay native probability≥1% 또는 greedy와 logit gap≤log(100)인 후보이며, 잠재 경쟁은 지정한 source-group 변화 방향의 screening에서 선택된 후보다. '
                 '세부 threshold·후보 상한·source와 replay probability는 appendix와 실행 설정에 기록한다. Top30 밖은 미계측이며 이 수치는 full mode probability나 전체 입력 방향에 대한 coverage가 아니다.', '',
                 '## Source greedy 성능', '',
                 'IoU=예측·GT 교집합 길이÷합집합 길이, mIoU=query별 IoU 평균, R@τ=IoU≥τ인 query 수÷완료 query 수다. '
                 '유효 구간이 없으면 IoU=0으로 평가하며 높을수록 좋다. 같은 source greedy 답변을 유지하므로 Δpp=100×(현재−동일 source)=0.00이다.', '',
                 '| 조건 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |',
                 '|---|---:|---:|---:|---:|---:|',
                 f'| Source greedy · 답변 고정 | {len(records)} | '+' | '.join(f'{v:.4f} (+0.00)' if v is not None else '—' for v in summary['greedy_metrics'])+' |', '',
                 '## 입력 종류별 계측', '',
                 '아래 값은 chosen-token target에서 입력 group별 값을 먼저 query 내 평균한 뒤 query 간 평균한 값이다. '
                 '각 query를 동일 가중하며 endpoint 합계와 branch margin은 섞지 않는다. Gradient L2는 민감도이고 signed sum은 해당 group 표현을 '
                 '미소 배율 확대하는 방향의 변화량이다. 절댓값 합은 위치별 signed 값의 절댓값 합으로 상쇄 정도를 보조한다. '
                 '세 값은 성공률이 아니며 크기가 클수록 좋은 지표도 아니다. 서로 다른 표현 단위의 raw 값과 그룹 크기에 영향을 받는다.', '',
                 '| 입력 group | 계측 query n | 평균 gradient L2 | 평균 signed sum | 평균 absolute sum |',
                 '|---|---:|---:|---:|---:|']
        for group in GROUPS:
            values=[r['groups'][group] for r in records if group in r['groups']]
            lines.append(f'| {group} | {len(values)} | '+ ' | '.join(fmt(mean([g.get(field) for g in values]),6) for field in ('grad_l2','signed_sum','absolute_sum'))+' |')
        wall = [r['cost'].get('total_seconds', r['cost'].get('wall_seconds', r['cost'].get('total_wall_seconds'))) for r in records]
        peak = [r['cost'].get('peak_allocated_bytes') for r in records if r['cost'].get('peak_allocated_bytes') is not None]
        checks = canonical/'validation'/f'pilot{count}.json'
        check = read(checks) if checks.exists() else None
        lines += ['', '## 재현 검증과 실제 비용', '',
                  '| 항목 | 값 | 정의 |', '|---|---:|---|',
                  f'| 계측 실행 기준 통과 query | {summary["replay_passed"]} / {len(records)} | 입력 hash·causality·source argmax 일치·probability 오차·gradient forward 일치 |',
                  f'| 최대 top30 absolute Δp | {fmt(max((r["max_top30_probability_error"] for r in records if r["max_top30_probability_error"] is not None),default=None),6)} | 모든 완료 query·step·고정 top30의 max absolute(p_replay−p_source); 낮을수록 일치 |',
                  f'| 동일 계측 replay gradient/no-grad 최대 logit 차이 | {fmt(max((r["gradient_forward_max_logit_error"] for r in records if r["gradient_forward_max_logit_error"] is not None),default=None),6)} | Source KV와의 비교와 별도; 0을 요구 |',
                  f'| Strict greedy argmax 일치 query | {sum(r["strict_argmax_match"] for r in records)} / {len(records)} | 모든 step의 argmax가 source greedy와 일치 |',
                  f'| 평균 query wall s | {fmt(mean(wall),3)} | 계측 query별 실측 평균; 대기·전체 작업 경과 시간과 구분 |',
                  f'| Query wall 합 s | {fmt(sum(v for v in wall if v is not None),3)} | 완료 query의 wall 실측 합 |',
                  f'| 최대 GPU allocated GiB | {fmt(max(peak)/2**30 if peak else None,3)} | query별 peak allocated 중 최댓값 |',
                  f'| Scope·viewer 검증 | {"통과" if check and check.get("passed") else "완료 기록 대기"} | 계측 완료와 후처리 검증 완료를 구분 |', '',
                  '각 query의 정확한 입력 hash·source KV와 계측 replay probability drift·future-position gradient 검증과 비용은 원자료에 남긴다. '
                  f'Source greedy argmax는 모든 step에서 정확히 일치해야 한다. Top30 probability 최대 절대 오차는 {config["replay_probability_tolerance"]:g} 이하여야 하며, 허용 범위의 실제 차이도 그대로 기록한다.', '',
                  '## 해석 범위', '',
                  '이 결과는 source greedy token열을 고정한 계측 replay 계산점의 local sensitivity와 내부 신호 방향이다. 큰 gradient가 인과적 증거를 확정하지 않고, '
                  '작은 영상 gradient도 영상 근거가 없다는 증거가 아니다. 이전 생성 위치의 표현에 영상 정보가 들어 있을 수 있다. '
                  'Layers의 점수를 합쳐 전체 증거량으로 해석하지 않는다. 잠재 경쟁 screening은 정한 변화 방향만 조사하므로 다른 방향의 급변을 배제하지 않는다. '
                  '실제 mode 전환은 변경 입력으로 이전 경로부터 다시 생성해야 확인할 수 있다.', '',
                  f'원자료: [{count} query source references]({relative(artifacts/"metadata/source_records.json",result)}) · '
                  f'[동결 실행 설정]({relative(config_path,result)}) · '
                  f'[계측 상태]({relative(canonical/"metadata/status.json",result)})', '']
        atomic_text(result/'REPORT.md','\n'.join(lines))
        appendix = strata_appendix(records)+definitions(config)
        atomic_text(result/'APPENDIX.md','\n'.join(appendix)+'\n')
        assets = Path(__file__).parent
        for suffix in ('js','css'):
            atomic_text(canonical/f'web/viewer.{suffix}', (assets/f'greedy_attribution_viewer.{suffix}').read_text())
        template=(assets/'greedy_attribution_viewer.html').read_text()
        replacements = {'__DEFAULT__':'data-family-default' if count==300 else '', '__COUNT__':str(count),
                        '__THEME__':relative(ROOT.parent/'_lib/experiment_theme.css',result),
                        '__STYLE__':relative(canonical/'web/viewer.css',result)+'?sha='+sha(canonical/'web/viewer.css')[:12],
                        '__SCRIPT__':relative(canonical/'web/viewer.js',result)+'?sha='+sha(canonical/'web/viewer.js')[:12],
                        '__CONFIG__':json.dumps(dict(scope=count,store=relative(canonical,result),
                              root=relative(ROOT.parent,result),catalog=relative(artifacts/'web/catalog.json',result)),ensure_ascii=False)}
        for old,new in replacements.items():
            template=template.replace(old,new)
        atomic_text(result/'detail.html',template)
        published.append(summary)
    # Only existing scopes enter the cumulative comparison; no empty future scopes.
    summaries=[]
    for count in CONFIG['stages']:
        source=BASE/str(count)/'metadata/report_summary.json'
        if source.exists() and (RESULTS/'pilot'/str(count)/'REPORT.md').exists():
            summaries.append(read(source))
    if len(summaries)>1:
        overview=RESULTS/'pilot/overview'
        lines=['# Greedy attribution · 누적 pilot 비교','','300 ⊂ 600 ⊂ 900의 고정 prefix cohort다. '
               '각 범위는 같은 누적 원자료를 참조하며 중첩 query를 독립 표본처럼 합산하지 않는다. '
               '완료 query 수와 계획 query 수를 구분한다. 상세 정의·한계·층화 결과는 각 범위 보고서에 있다.','',
               '| 범위 | 완료 query | Greedy step | Backward target | 잠재 경쟁만 해당 | mIoU |','|---|---:|---:|---:|---:|---:|']
        for s in summaries:
            lines.append(f'| [pilot{s["queries"]}](../{s["queries"]}/REPORT.md) | {s["completed"]} / {s["queries"]} | '
                         f'{s["steps"]} | {s["targets"]} | {s["potential_only_candidates"]} | {fmt(s["greedy_metrics"][0])} |')
        atomic_text(overview/'REPORT.md','\n'.join(lines)+'\n')
        write(BASE/'overview/metadata/source_records.json',dict(canonical=str(canonical),scopes=[s['queries'] for s in summaries],overlapping=True))
    return published


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',type=int,choices=CONFIG['stages'])
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--canonical',type=Path)
    args=parser.parse_args()
    print(json.dumps(report(args.scope,args.allow_partial,args.canonical),ensure_ascii=False))
