"""Generate the CPU-only oracle-switch pilot report and interactive detail view.

The JSON query records remain authoritative raw measurements; REPORT.md records
metric definitions and the interpretation of the measured results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / 'artifacts/runs/rollout/oracle_switch/pilot/3'
OUTPUT = PROJECT / 'results/rollout/oracle_switch/pilot/3'
DIRECTIONS = ('all_video', 'top_frame_pair', 'top_spatial_cell')
LABELS = {
    'all_video': '전체 영상', 'top_frame_pair': '선택한 frame pair',
    'top_spatial_cell': '선택한 spatial cell',
    'charades-timelens': 'Charades', 'activitynet-timelens': 'ActivityNet',
    'qvhighlights-timelens': 'QVHighlights',
}


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def esc(value):
    return html.escape(str(value if value is not None else ''), quote=True)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=4):
    if not finite(value):
        return '—'
    if value and abs(value) < 10 ** -digits:
        return f'{value:.2e}'
    return f'{value:.{digits}f}'


def span(value):
    if isinstance(value, dict):
        value = value.get('span')
    return f'{fmt(value[0], 2)}–{fmt(value[1], 2)} s' if isinstance(value, list) and len(value) == 2 else '미관측'


def gt(sample):
    value = sample.get('row', {}).get('span', [])
    return value[0] if value and isinstance(value[0], list) else value or None


def url(path, output, base=ARTIFACTS):
    if not path:
        return None
    if isinstance(path, dict):
        path = path.get('path') or path.get('file')
    if not path:
        return None
    if str(path).startswith(('http://', 'https://')):
        return str(path)
    path = Path(path)
    path = path if path.is_absolute() else base / path
    return quote(os.path.relpath(path.resolve(), Path(output).resolve()), safe='/')


def existing_url(path, output, base=ARTIFACTS):
    if not path:
        return None
    candidate = path.get('path') or path.get('file') if isinstance(path, dict) else path
    if not candidate:
        return None
    if str(candidate).startswith(('https://', 'http://')):
        return str(candidate)
    candidate = Path(candidate)
    candidate = candidate if candidate.is_absolute() else base / candidate
    return url(candidate, output) if candidate.exists() else None


def epsilon(point):
    if not point:
        return None
    value = point.get('epsilon_255')
    return value if finite(value) else point.get('epsilon', 0) * 255


def direction_id(direction):
    return direction.get('id', direction.get('direction', 'unknown'))


def directions(query):
    value = query.get('directions', [])
    if isinstance(value, dict):
        return [dict(entry, id=key) for key, entry in value.items()]
    return value


def boundary_values(direction):
    boundary = direction.get('boundary') or {}
    low, high = boundary.get('low_epsilon_255'), boundary.get('high_epsilon_255')
    if low is None and finite(boundary.get('low_epsilon')):
        low = boundary['low_epsilon'] * 255
    if high is None and finite(boundary.get('high_epsilon')):
        high = boundary['high_epsilon'] * 255
    return low, high


def scan_points(direction):
    return [point for point in direction.get('trials', []) if finite(epsilon(point)) and epsilon(point) >= 0 and point.get('phase') != 'opposite_control']


def first_observed(direction, predicate):
    points = [point for point in scan_points(direction) if predicate(point)]
    return min((epsilon(point) for point in points), default=None)


def support_text(direction):
    support = direction.get('support', {})
    if not support:
        return '선택 대기'
    pieces = []
    for key, label in [('slots', '입력 slot'), ('frame_ids', 'Source frame ID'), ('timestamps', '실제 timestamp'), ('spatial_bounds', 'Spatial bounds')]:
        if key in support:
            value = support[key]
            if key == 'timestamps' and isinstance(value, list):
                value = ', '.join(fmt(item, 3) + ' s' for item in value) if len(value) <= 8 else f'{len(value)}개 · {fmt(min(value), 3)}–{fmt(max(value), 3)} s'
            else:
                value = (f'{len(value)}개 · ' + json.dumps(value[:3], ensure_ascii=False) + ' … ' + json.dumps(value[-3:], ensure_ascii=False)) if isinstance(value, list) and len(value) > 8 else json.dumps(value, ensure_ascii=False)
            pieces.append(f'{label}: {value}')
    return ' · '.join(pieces) if pieces else json.dumps(support, ensure_ascii=False)


def query_summary(query):
    tested = directions(query)
    return {
        'complete': bool(query.get('complete')),
        'trials': sum(len(item.get('trials', [])) for item in tested),
        'directions': len(tested),
        'pair_reversal': sum(any(finite(point.get('margin')) and point['margin'] < 0 for point in scan_points(item)) for item in tested),
        'branch_reached': sum(any(point.get('branch_reached') for point in scan_points(item)) for item in tested),
        'target_top1': sum(any(point.get('target_top1') for point in scan_points(item)) for item in tested),
        'strict_target_top1': sum(any(point.get('strict_target_top1') for point in scan_points(item)) for item in tested),
    }


def chain_note(sample):
    chain = sample.get('oracle', {}).get('chain', [])
    if len(chain) <= 1:
        return '저장된 oracle continuation과 최초 분기를 비교합니다.'
    direct = sample.get('root_branch_continuation', {})
    path = ' → '.join(str(chain[0].get('parent', '?')) if index == 0 else str(item.get('child', '?')) for index, item in enumerate([chain[0]] + chain))
    result = f'최초 분기만 바꾼 기존 결과는 {span(direct)}, GT IoU {fmt(direct.get("iou"))}입니다. ' if direct else ''
    return (f'저장된 oracle은 여러 강제 분기({path})를 거친 경로입니다. ' + result +
            '최초 oracle token 하나의 전환이 전체 oracle 경로 재현을 보장하지 않습니다.')


def methods():
    return [
        '## 측정 대상과 해석', '',
        '이 pilot은 기존 deterministic rollout에서 dataset별 oracle−greedy IoU 차이가 큰 query를 한 개씩 고른 '
        '**총 3 query의 진단 실험**이다. Oracle은 저장된 후보 중 GT IoU가 가장 높은 결과다. GT를 사용해 고른 극단 사례이며 '
        'dataset 평균 성능이나 일반적인 전환 성공률을 추정하는 표본이 아니다.', '',
        '원래 greedy prefix를 그대로 고정하고 최초 oracle 분기의 token g와 o를 비교한다. '
        '원래 모델 입력으로 resize한 float RGB x∈[0,1]에서 m(x)=z_g−z_o의 gradient를 계산한다. '
        '고정 방향 d=−sign(∇ₓm)에 변경 범위 mask M을 적용하여 x′(ε)=clip(x+εMd,0,1)로 바꾼다. '
        'Text token·text embedding·FPS·frame IDs·timestamp·spatial resolution·token 구성은 바꾸지 않는다. Pixel 변화에 따라 visual embedding은 달라질 수 있다. '
        '같은 source frame이 반복 입력된 slot은 gradient를 합산해 방향을 정하고 동일한 perturbation을 적용한다.', '',
        '변경 범위는 전체 영상, gradient L1 합이 가장 큰 frame pair, 그 pair에서 L1 합이 가장 큰 32×32 spatial cell이다. '
        '변형은 resize 이후 모델 입력의 float RGB에 적용하며 압축된 원본 영상 자체를 편집한 것이 아니다. '
        'Processor의 실제 mean/std를 사용해 모델 입력으로 변환한다. ε₂₅₅=255ε는 8-bit intensity level 단위이며 '
        '소수도 허용한다. 추론 입력을 uint8로 양자화하지 않는다. Preview PNG만 표시를 위해 반올림될 수 있다.', '',
        '초기 ε₂₅₅ grid는 0.25, 0.5, 1, 2, 4, 8, 16, 32다. 한 방향에서 처음 관측한 branch reached의 '
        '실패·성공 구간을 6회 추가 계측한다. 이 구간은 시험한 고정 gradient ray에서의 관측 경계이며, '
        '미계측 구간의 단조성이나 전역 최소 adversarial distance를 보장하지 않는다. '
        '전체 영상 방향에서 실패해도 국소 변경 범위의 방향은 별도로 조사한다.', '',
        '| 지표 | 정의와 해석 |', '|---|---|',
        '| Paired margin | m=z_g−z_o. 음수이면 oracle token이 원래 greedy token보다 높다. 다른 token보다 높은지는 별도다. |',
        '| Oracle margin | r=z_o−max_{v≠o}z_v. 양수이면 전체 vocabulary에서 oracle token이 단독 top1이다. 0은 동률이다. |',
        '| Target top1 | 모델의 실제 전체 vocabulary argmax ID가 o인지 검사한다. 동률에서 o가 선택될 수 있으므로 strict target top1과 구분한다. |',
        '| Prefix argmax match | 원래 prefix를 teacher-forcing으로 평가할 때 분기 이전 모든 step의 실제 argmax가 원래 token과 일치하는지 검사한다. |',
        '| Branch reached | target top1 AND prefix argmax match. 이 조건을 만족하는 첫 관측 경계를 세분화하며 응답 처음부터 같은 greedy decoding으로 다시 확인한다. |',
        '| Strict target top1 | Oracle margin>0. 모든 다른 token보다 o의 logit이 엄격히 높다. |',
        '| Probability | 원래 고정 top30 ID의 native softmax probability. Top30 안에서 재정규화하지 않는다. 전체 분포는 저장하지 않고 전체 argmax ID 등 검증 scalar만 별도 기록한다. |',
        '| RGB L∞ | maxᵢ|x′ᵢ−xᵢ|. Clipping 이후의 실제 최대 변경량이며 계획 ε보다 작을 수 있다. |',
        '| RGB RMS | sqrt(meanᵢ((x′ᵢ−xᵢ)²)). 분모는 중복 input slot을 한 번만 센 전체 physical source frame의 RGB scalar 수이며 변경 범위 밖의 0 변경도 포함한다. |',
        '| Changed fraction | 변경된 RGB scalar 수 / 중복 slot을 제외한 전체 physical source frame의 RGB scalar 수. |',
        '| Temporal IoU | length(예측 구간∩GT) / length(예측 구간∪GT). 0–1이며 클수록 GT와 겹친다. |',
        '| Oracle span IoU | 새 예측 구간과 저장된 oracle 구간의 temporal IoU. GT IoU와 다른 비교다. |', '',
        '**Fixed-prefix probe**는 원래 prefix를 teacher-forcing으로 유지한 다음 token 평가이며, 응답을 생성하는 decoding 조건이 아니다. '
        '결과표의 모든 행은 응답 처음부터 같은 greedy decoding을 사용한다. 조건은 입력 변경량의 차이다. '
        '원래 prefix를 조건으로 한 token 역전, 실제 top1 전환, greedy 재생성의 prefix 도달 여부와 최종 구간·GT IoU를 각각 구분한다. '
        'Oracle branch reached와 oracle path equal도 서로 다른 검사다. Token 전환을 최종 mode 전환이나 정답 개선과 동일시하지 않는다.', '',
        'Branch reached 경계 아래·위 입력 변경량에서 같은 greedy decoding 결과를 비교하고, 경계 위 변경량에서 반복 계측·반대 방향 control을 확인한다. '
        '경계를 못 찾은 방향도 최대 시험 ε에서 같은 greedy decoding을 실행하며 입력 변경 조건을 시험 최대 변경량으로 표시한다. 미관측 결과는 실패로 세지 않는다. 전환을 못 찾은 방향은 해당 변경 범위·고정 방향·시험한 ε 범위에서의 음성 결과다. '
        '자연스러운 영상 수정의 효과, 다른 방향의 불가능성, 전역 최소 perturbation을 주장하지 않는다.', '',
        '## PNG 비교의 의미', '',
        '원본 입력과 실제 변경 입력의 ×1 PNG는 float RGB를 8-bit 표시값으로 반올림한 것이다. 미세한 float 변경은 이 반올림 뒤 같아 보이거나 완전히 같은 PNG가 될 수 있다. '
        'PNG 일치 여부와 실제 float RGB의 최대 변경량을 함께 표시하며, PNG가 같다는 이유만으로 모델 입력이 같다고 판단하지 않는다.', '',
        '확대 이미지는 x_display=clip(x+G(x′−x),0,1)로 변경 차이만 G배 확대해 원본 위에 표시한 것이다. '
        '**표시용 확대이며 모델에 넣은 영상이 아니다.** 표시 배율 G와 확대 후 clipping 비율을 따로 명시한다. '
        '실제 ×1 이미지를 일부러 다르게 만들지 않으며 수치·응답·원자료는 변경하지 않는다.', '',
    ]


def direction_rows(query):
    rows = []
    for item in directions(query):
        paired = first_observed(item, lambda point: finite(point.get('margin')) and point['margin'] < 0)
        target = first_observed(item, lambda point: bool(point.get('target_top1')))
        strict = first_observed(item, lambda point: bool(point.get('strict_target_top1')))
        reached = first_observed(item, lambda point: bool(point.get('branch_reached')))
        low, high = boundary_values(item)
        limits = [epsilon(point) for point in scan_points(item)]
        measured_limit = max(limits, default=None)
        rows.append([LABELS.get(direction_id(item), item.get('label', direction_id(item))),
                     fmt(paired, 3), fmt(target, 3), fmt(strict, 3), fmt(reached, 3),
                     f'({fmt(low, 4)}, {fmt(high, 4)}]' if high is not None else f'미관측 · 최대 {fmt(measured_limit, 3)}',
                     fmt((item.get('free_low') or {}).get('iou')),
                     fmt((item.get('free_high') or {}).get('iou')),
                     fmt((item.get('free_limit') or {}).get('iou'))])
    return rows


def md_table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return ['| ' + ' | '.join(map(cell, headers)) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
        '| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]


def outcome_rows(query):
    rows = []
    for direction in directions(query):
        low, high = boundary_values(direction)
        for field, label, value in [('free_low', '경계 아래 변경량', low), ('free_high', '경계 위 변경량', high), ('free_limit', '시험 최대 변경량', direction.get('evaluated_limit_255', max((epsilon(point) for point in scan_points(direction)), default=None)))]:
            outcome = direction.get(field)
            if not outcome:
                continue
            if finite(outcome.get('epsilon_255')) or finite(outcome.get('epsilon')):
                value = epsilon(outcome)
            rows.append([LABELS.get(direction_id(direction), direction_id(direction)), label, fmt(value, 4), span(outcome),
                         fmt(outcome.get('iou')), fmt(outcome.get('oracle_span_iou')),
                         str(outcome.get('first_divergence_step', '—')),
                         '예' if outcome.get('oracle_branch_reached') else '아니오' if 'oracle_branch_reached' in outcome else '—',
                         '예' if outcome.get('oracle_path_equal') else '아니오' if 'oracle_path_equal' in outcome else '—'])
    return rows


def markdown(selection, config, status, cases, artifacts, output):
    total = len(cases)
    completed = sum(bool(query.get('complete')) for _, query in cases)
    lines = ['# Oracle switch · pilot' + str(total), '',
             f'**측정 완료 {completed}/{total} query · 상태 `{status.get("state", status.get("stage", "queued"))}`**', '',
             '선택된 기존 greedy·oracle 결과와 새 RGB perturbation 측정을 구분한다. 대기 중인 query의 곡선과 새 decoding 결과는 아직 없다.', '',
             '## 선택된 사례', '']
    rows = []
    for sample, _ in cases:
        branch = sample.get('branch', {})
        rows.append([LABELS.get(sample.get('dataset'), sample.get('dataset')), sample['sample_key'],
                     span(sample.get('greedy')), fmt(sample.get('greedy', {}).get('iou')),
                     span(sample.get('oracle')), fmt(sample.get('oracle', {}).get('iou')), fmt(sample.get('delta_iou')),
                     f'Step {branch.get("step", "—")}: {branch.get("greedy_token_text", branch.get("greedy_token"))} → {branch.get("oracle_token_text", branch.get("oracle_token"))}',
                     branch.get('source_rank', '—')])
    lines += md_table(['Dataset', 'Query', 'Greedy span', 'Greedy IoU', 'Oracle span', 'Oracle IoU', 'ΔIoU', '최초 분기', 'Source rank'], rows)
    lines += ['', 'ΔIoU=저장된 oracle IoU−저장된 greedy IoU이며 새 perturbation의 효과가 아니다. Dataset별 1 query씩 총 3 query다. '
              '완료 수는 실제 새 query 측정 완료 수이며 trial·방향·생성 응답 수와 구분한다.', '']
    lines += methods()
    for sample, query in cases:
        summary = query_summary(query)
        lines += [f'## {LABELS.get(sample.get("dataset"), sample.get("dataset"))} · {sample["sample_key"]}', '',
                  str(sample.get('row', {}).get('query', '')), '', chain_note(sample), '',
                  f'원래 GT {span(gt(sample))}, greedy {span(sample.get("greedy"))}, oracle {span(sample.get("oracle"))}.', '']
        if not query:
            lines += ['**실행 대기:** 원본 사례 선택만 완료했다. Gradient·ε scan·greedy 재생성은 미계측이다.', '']
            continue
        lines += [f'새 측정 상태: {"완료" if summary["complete"] else "진행 중"}. 기록된 방향 {summary["directions"]}개, trial {summary["trials"]}개. '
                  f'Paired reversal 관측 {summary["pair_reversal"]}/{summary["directions"]} 방향, target top1 관측 '
                  f'{summary["target_top1"]}/{summary["directions"]}, branch reached 관측 {summary["branch_reached"]}/{summary["directions"]}, strict target top1 관측 {summary["strict_target_top1"]}/{summary["directions"]}.', '',
                  '아래 ε는 모두 ε₂₅₅ 단위다. 최소 관측 값은 저장된 trial 중의 최솟값이며 미계측 ε의 최솟값이 아니다.', '']
        repeat = query.get('gradient', {}).get('repeat', {})
        if repeat:
            lines += [f'동일 graph의 두 backward 반복: gradient 상대 L2 차이 {fmt(repeat.get("relative_l2"))}, cosine {fmt(repeat.get("cosine"))}, sign 일치율 {fmt(repeat.get("sign_agreement"))}. 첫 gradient로 방향을 고정하며 실제 유한 변화의 forward 결과로 전환을 판단한다.', '']
        if query.get('baseline_free'):
            equal = query['baseline_free'].get('ids') == sample.get('greedy', {}).get('ids')
            lines += [f'변형 전 free greedy token 재현: {"전체 일치" if equal else "불일치"}. 고정 top30 probability 최대 오차 {fmt(query.get("baseline_probability_error"))}.', '']
        lines += md_table(['변경 범위', '첫 paired reversal ε', '첫 target top1 ε', '첫 strict top1 ε', '첫 branch reached ε', '관측 branch 경계', '경계 아래 IoU', '경계 위 IoU', '시험 최대 IoU'], direction_rows(query))
        outcomes = outcome_rows(query)
        if outcomes:
            lines += ['', '모든 행은 응답 처음부터 같은 greedy decoding이다. 입력 변경 조건은 경계 아래·위 또는 시험 최대 변경량을 구분한다.', '']
            lines += md_table(['변경 범위', '입력 변경 조건', 'ε₂₅₅', '새 구간', 'GT IoU', 'Oracle span IoU', '첫 token 차이 step', 'Oracle branch 도달', 'Oracle path 일치'], outcomes)
        for direction in directions(query):
            lines += ['', f'**{LABELS.get(direction_id(direction), direction_id(direction))} 변경 범위:** {support_text(direction)}.']
            media = direction.get('media') or {}
            if 'png_identical' in media or 'amplification_gain' in media:
                lines += [' '.join(preview_facts(media))]
            repeat, opposite = direction.get('repeat_high'), direction.get('control_opposite')
            if repeat:
                lines += [f'Repeat high: margin {fmt(repeat.get("margin"))}, oracle margin {fmt(repeat.get("oracle_margin"))}, target top1={repeat.get("target_top1")}.']
            if opposite:
                lines += [f'Opposite control: ε₂₅₅={fmt(epsilon(opposite), 4)}, margin {fmt(opposite.get("margin"))}, target top1={opposite.get("target_top1")}.']
        lines += ['', f'[Query 원자료]({url(artifacts / "queries" / (sample["sample_key"] + ".json"), output)})', '']
    lines += ['## 원자료와 재현', '']
    for filename, label in [('selection.json', '선택 규칙·source hashes'), ('config.json', '실행 설정'), ('status.json', '현재 실행 상태')]:
        path = artifacts / 'metadata' / filename
        if path.exists():
            lines += [f'- [{label}]({url(path, output)})']
    manifest = artifacts / 'analyses/preview_manifest.json'
    if manifest.exists():
        lines += [f'- [PNG 비교·표시 배율·원자료 대조]({url(manifest, output)})']
    lines += ['', '새 데이터는 원본 greedy·oracle source를 덮어쓰지 않는다. 입력 hash·실제 preprocessing·baseline 재현·support와 '
              '중복 source frame 처리·실제 clipping 이후 거리·runtime cost는 query 원자료에 보존한다. '
              '이 보고서의 곡선은 측정점만 나타내며 점 사이 연결선은 미계측 결과를 뜻하지 않는다.', '']
    return '\n'.join(lines)


def html_table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{esc(cell)}</th>' for cell in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(cell)}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def curve(query, direction):
    points = list(scan_points(direction))
    if not any(epsilon(point) == 0 for point in points):
        points.insert(0, dict(query.get('baseline', {}), epsilon_255=0))
    points = sorted((point for point in points if finite(point.get('margin')) and finite(epsilon(point))), key=lambda point: epsilon(point))
    if not points:
        return '<div class="empty">ε scan은 아직 실행되지 않았습니다.</div>'
    width, height = 680, 290
    left, right, top, bottom = 65, 20, 20, 48
    xmax = max(32, max(epsilon(point) for point in points))
    values = [point['margin'] for point in points] + [0]
    lower, upper = min(values), max(values)
    padding = max((upper - lower) * .12, .1)
    lower -= padding
    upper += padding
    x = lambda value: left + value / xmax * (width - left - right)
    y = lambda value: top + (upper - value) / (upper - lower) * (height - top - bottom)
    content = [f'<svg class="margin-curve" viewBox="0 0 {width} {height}" role="img" aria-label="RGB epsilon에 따른 paired margin 측정 곡선">']
    for tick in [0, 8, 16, 24, 32]:
        content += [f'<line x1="{x(tick):.2f}" y1="{top}" x2="{x(tick):.2f}" y2="{height-bottom}" class="grid-line"/>', f'<text x="{x(tick):.2f}" y="{height-bottom+22}" text-anchor="middle">{tick}</text>']
    for index in range(5):
        tick = lower + (upper - lower) * index / 4
        content += [f'<line x1="{left}" y1="{y(tick):.2f}" x2="{width-right}" y2="{y(tick):.2f}" class="grid-line"/>', f'<text x="{left-8}" y="{y(tick)+4:.2f}" text-anchor="end">{fmt(tick, 2)}</text>']
    content += [f'<line x1="{left}" y1="{y(0):.2f}" x2="{width-right}" y2="{y(0):.2f}" class="zero-line"/>',
                '<polyline fill="none" stroke="#42768a" stroke-width="1.5" points="' + ' '.join(f'{x(epsilon(point)):.2f},{y(point["margin"]):.2f}' for point in points) + '"/>']
    for point in points:
        color = '#246b4b' if point.get('target_top1') else '#ad7132' if point['margin'] < 0 else '#42768a'
        label = f'ε₂₅₅={fmt(epsilon(point), 4)}, margin={fmt(point["margin"])}, oracle margin={fmt(point.get("oracle_margin"))}, target top1={point.get("target_top1", False)}'
        content += [f'<circle cx="{x(epsilon(point)):.2f}" cy="{y(point["margin"]):.2f}" r="4" fill="{color}" stroke="white" stroke-width="1"><title>{esc(label)}</title></circle>']
    content += [f'<text x="{(left+width-right)/2}" y="{height-5}" text-anchor="middle">ε₂₅₅ · RGB intensity level</text>', '<text x="15" y="145" text-anchor="middle" transform="rotate(-90 15 145)">z_g − z_oracle</text>', '</svg>']
    return ''.join(content)


def preview_facts(media):
    """Describe display-only media without inferring a change from PNG appearance."""
    identical = media.get('png_identical')
    maximum = media.get('actual_delta_max_255')
    if identical is True:
        identity = '8-bit 반올림 뒤 원본과 실제 변경 PNG가 동일합니다.'
        if finite(maximum) and maximum > 0:
            identity += ' Float RGB에는 아래 크기의 변경이 있습니다.'
    elif identical is False:
        identity = '8-bit PNG에서도 원본과 실제 변경 입력의 차이가 남습니다.'
    else:
        identity = '8-bit PNG 일치 여부는 별도 검증 전입니다.'
    facts = [identity]
    if finite(maximum):
        facts.append(f'실제 float 최대 RGB 변경량: {fmt(maximum, 6)} /255.')
    gain = media.get('amplification_gain')
    if finite(gain):
        facts.append(f'변경 차이 표시 배율: {fmt(gain, 3)}×. 표시용 확대이며 모델에 넣은 영상이 아닙니다.')
    clipped = media.get('display_clipped_fraction')
    if finite(clipped):
        facts.append(f'확대 표시 frame의 RGB scalar 중 clipping 비율: {fmt(100 * clipped, 3)}%.')
    return facts


def merge_preview_media(query, supplemental):
    """Return a view model; never mutate measured query dictionaries or their files."""
    if not supplemental:
        return query
    combined = dict(query)
    combined['directions'] = [dict(direction, media={
        **(direction.get('media') or {}), **(supplemental.get(direction_id(direction)) or {})
    }) for direction in directions(query)]
    return combined


def media_previews(direction, artifacts, output):
    media = direction.get('media') or {}
    figures = []
    gain = media.get('amplification_gain')
    labels = [
        ('original', '원본 입력 · ×1', '원래 float RGB의 8-bit PNG 표시'),
        ('perturbed', '실제 변경 입력 · ×1', '모델에 넣은 float RGB의 8-bit PNG 표시'),
        ('amplified', f'변경 차이 확대 · ×{fmt(gain, 2)}', '표시용 확대, 모델에 넣은 영상 아님'),
    ]
    for name, label, description in labels:
        source = existing_url(media.get(name), output, artifacts)
        if source:
            figures.append(f'<figure class="preview-{name}" data-preview-kind="{name}"><figcaption><b>{esc(label)}</b><span>{esc(description)}</span></figcaption><a href="{esc(source)}" target="_blank" rel="noopener"><img src="{esc(source)}" alt="{esc(label)} · {esc(description)}" loading="lazy"></a></figure>')
    if not figures:
        return ''
    identity_class = ' identical-png' if media.get('png_identical') else ''
    note = f'<p class="preview-note{identity_class}">' + ' '.join(esc(fact) for fact in preview_facts(media)) + '</p>'
    extra = []
    for name, label in [('difference', 'Absolute RGB difference'), ('signed_difference', 'Signed RGB difference')]:
        source = existing_url(media.get(name), output, artifacts)
        if source:
            extra.append(f'<figure><figcaption>{label}</figcaption><img src="{esc(source)}" alt="{label}" loading="lazy"></figure>')
    if extra:
        scale = media.get('difference_gain', media.get('difference_scale', media.get('scale')))
        scale_note = f'Absolute difference 표시 배율 {fmt(scale, 3)}×.' if finite(scale) else 'Difference 표시 배율은 metadata에 기록된 값만 해석합니다.'
        extra = '<details class="difference-previews"><summary>차이만 표시한 이미지 보기</summary><div class="difference-grid">' + ''.join(extra) + f'</div><p class="small">{esc(scale_note)} 별도 시각화이며 모델에 넣은 입력이 아닙니다.</p></details>'
    else:
        extra = ''
    return '<div class="preview-grid comparison-previews">' + ''.join(figures) + '</div>' + note + f'<p class="small">표시 source slot {esc(media.get("slot", "—"))} · {esc(fmt(media.get("timestamp"), 3))} s. 실제 추론 입력은 float RGB이며 PNG는 표시용입니다.</p>' + extra


def direction_html(query, direction, index, artifacts, output):
    key = direction_id(direction)
    low, high = boundary_values(direction)
    points = direction.get('trials', [])
    summary = f'관측 branch reached 경계 ({fmt(low, 4)}, {fmt(high, 4)}] /255' if high is not None else '시험한 범위에서 branch reached 경계 미관측'
    if high is not None:
        verified = (direction.get('boundary') or {}).get('verified')
        summary += ' · Greedy 재생성·반복 검증 ' + ('통과' if verified is True else '미통과' if verified is False else '대기')
    details = []
    for point in points:
        perturbation = point.get('perturbation', {})
        details.append([fmt(epsilon(point), 4), fmt(point.get('margin')), fmt(point.get('oracle_margin')),
                        fmt(point.get('greedy_p')), fmt(point.get('oracle_p')), point.get('top1_id', '—'),
                        '예' if point.get('target_top1') else '아니오', '예' if point.get('strict_target_top1') else '아니오',
                        '예' if point.get('prefix_argmax_match') else '아니오', '예' if point.get('branch_reached') else '아니오',
                        fmt(perturbation.get('rgb_linf')), fmt(perturbation.get('rgb_rms')), fmt(perturbation.get('changed_fraction')), point.get('phase', '')])
    return f'<section class="direction-view" data-direction="{esc(key)}" {"hidden" if index else ""}><h3>{esc(LABELS.get(key, direction.get("label", key)))}</h3><p class="small">{esc(support_text(direction))}</p><p><b>{esc(summary)}</b></p>{curve(query, direction)}<p class="curve-legend"><span class="blue">● Paired margin ≥ 0</span><span class="amber">● Paired reversal</span><span class="green">● 실제 target top1</span></p><p class="small">음수 margin은 두 token의 순서만 바뀌었다는 뜻입니다. 초록 점은 전체 vocabulary argmax가 oracle token입니다. 이전 prefix까지 유지한 branch reached 여부는 상세표에서 구분합니다. 점 사이 선은 측정점 연결이며 보간 예측이 아닙니다.</p>{media_previews(direction, artifacts, output)}<details><summary>측정점·실제 RGB 변경량 펼치기 ({len(points)} trial)</summary>{html_table(["ε₂₅₅", "g−oracle", "Oracle margin", "p(g)", "p(oracle)", "Argmax ID", "Target top1", "Strict top1", "Prefix 유지", "Branch reached", "RGB L∞", "RGB RMS", "변경 비율", "Phase"], details)}</details></section>'


def timeline_row(label, value, duration, kind):
    if not isinstance(value, list) or len(value) != 2:
        return ''
    start, end = value
    width = max(0, min(end, duration) - max(0, start)) / duration * 100
    return f'<div class="timeline-row"><span>{esc(label)}</span><div class="timeline-track"><button class="timeline-span {kind}" data-seek="{max(0, start)}" aria-label="{esc(label)} 시작 시각으로 이동" style="left:{max(0, start)/duration*100:.4f}%;width:{width:.4f}%"></button></div><small>{esc(span(value))}</small></div>'


def case_html(sample, query, index, artifacts, output):
    key, row = sample['sample_key'], sample.get('row', {})
    branch = sample.get('branch', {})
    duration = max(float(row.get('duration', 1)), .001)
    video = existing_url(sample.get('video_path') or sample.get('video'), output, PROJECT.parent)
    video_html = f'<video controls muted preload="metadata" src="{esc(video)}" aria-label="원본 영상"></video>' if video else '<div class="empty">연결 가능한 원본 영상이 없습니다.</div>'
    bars = [timeline_row('GT', gt(sample), duration, 'gt'), timeline_row('Greedy', sample.get('greedy', {}).get('span'), duration, 'greedy'), timeline_row('Oracle', sample.get('oracle', {}).get('span'), duration, 'oracle')]
    new_spans = []
    for item in directions(query):
        for field, name in [('free_low', '경계 아래 변경량'), ('free_high', '경계 위 변경량'), ('free_limit', '시험 최대 변경량')]:
            outcome = item.get(field) or {}
            if outcome.get('span'):
                new_spans.append((f'{LABELS.get(direction_id(item), direction_id(item))} · {name}', outcome['span']))
    bars += [timeline_row(name, value, duration, 'new') for name, value in new_spans]
    measured = directions(query)
    options = ''.join(f'<option value="{esc(direction_id(item))}">{esc(LABELS.get(direction_id(item), item.get("label", direction_id(item))))}</option>' for item in measured)
    panels = ''.join(direction_html(query, item, position, artifacts, output) for position, item in enumerate(measured))
    if not measured:
        panels = '<div class="empty"><b>GPU 실행 대기 · 아직 새 측정 결과가 없습니다.</b><p>계측할 변경 범위는 전체 영상 → 선택한 frame pair → 선택한 32×32 spatial cell입니다. 기존 결과만 아래 사례의 출처로 표시합니다.</p></div>'
    outcomes = outcome_rows(query)
    outcomes_html = html_table(['변경 범위', '입력 변경 조건', 'ε₂₅₅', '새 구간', 'GT IoU', 'Oracle span IoU', '첫 token 차이 step', 'Oracle branch 도달', 'Oracle path 일치'], outcomes) if outcomes else '<p class="empty">새 입력 조건의 greedy decoding 결과는 아직 없습니다.</p>'
    raw = artifacts / 'queries' / (key + '.json')
    raw_link = f'<a href="{esc(url(raw, output))}" target="_blank" rel="noopener">Query 원자료 ↗</a>' if raw.exists() else ''
    return f'''<article class="case" data-case="{esc(key)}" {"hidden" if index else ""}>
<section class="panel"><div class="case-heading"><div><p class="eyebrow">{esc(LABELS.get(sample.get('dataset'), sample.get('dataset')))} · {esc(key)}</p><h2>{esc(row.get('query', key))}</h2></div><span class="badge">{'새 측정 완료' if query.get('complete') else '측정 진행 중' if query else '실행 대기'}</span></div>
<div class="case-overview"><div>{video_html}<p class="small">원본 영상입니다. 아래 RGB perturbation은 resize한 모델 입력에만 적용합니다.</p></div><div><div class="scores"><div><small>저장된 greedy IoU</small><b>{fmt(sample.get('greedy', {}).get('iou'))}</b></div><div><small>저장된 oracle IoU</small><b>{fmt(sample.get('oracle', {}).get('iou'))}</b></div><div><small>선택 기준 ΔIoU</small><b>{fmt(sample.get('delta_iou'))}</b></div></div>{''.join(bars)}<p class="small">구간 막대를 누르면 같은 원본 시각으로 이동합니다. Oracle은 GT를 사용해 선택한 기존 후보입니다.</p></div></div>
<p><b>Step {branch.get('step', '—')}:</b> <code>{esc(branch.get('greedy_token_text', branch.get('greedy_token')))}</code> → <code>{esc(branch.get('oracle_token_text', branch.get('oracle_token')))}</code> · Source rank {esc(branch.get('source_rank', '—'))}</p><p class="notice">{esc(chain_note(sample))}</p></section>
<section class="panel"><div class="case-heading"><div><p class="eyebrow">01 / RGB 변화와 token 선택</p><h2>얼마나 바꿨을 때 경쟁이 바뀌는가</h2></div>{'<label>변경 범위 <select class="direction-picker">'+options+'</select></label>' if measured else ''}</div><p class="small">원래 prefix에서 계산한 고정 −sign gradient 방향을 시험합니다. ε₂₅₅는 RGB intensity level이며 자연스러운 영상 편집량이나 전체 영상의 변화 비율이 아닙니다.</p>{panels}</section>
<section class="panel"><p class="eyebrow">02 / 응답을 다시 시작한 결과</p><h2>Token 역전이 최종 구간으로 이어졌는가</h2><p class="small">모든 행은 응답 처음부터 같은 greedy decoding을 사용하며, 조건은 입력 변경량의 차이입니다. Fixed-prefix probe는 원래 prefix의 다음 token을 평가하는 별도 측정이며 decoding 방식이 아닙니다. Oracle branch 도달·전체 path 일치·최종 GT IoU를 따로 봅니다.</p>{outcomes_html}</section>
<section class="panel raw"><details><summary>방향별 요약·실행 기록</summary>{html_table(['변경 범위', '첫 paired reversal ε', '첫 target top1 ε', '첫 strict top1 ε', '첫 branch reached ε', '관측 branch 경계', '경계 아래 IoU', '경계 위 IoU', '시험 최대 IoU'], direction_rows(query))}<p>{raw_link}</p><pre>{esc(json.dumps({'baseline': query.get('baseline'), 'gradient_repeat': query.get('gradient', {}).get('repeat'), 'cost': query.get('cost'), 'validation': query.get('validation')},ensure_ascii=False,indent=2))}</pre></details></section>
</article>'''


STYLE = '''
main{max-width:1500px;margin:auto;padding:24px}h1,h2,h3,p{margin-top:0}h2{font-size:21px}.page-header{margin:-24px -24px 24px}.eyebrow{font-size:11px;letter-spacing:.08em;font-weight:700;color:var(--ui-accent)}.small{font-size:12px;line-height:1.8;color:var(--ui-muted)}.notice{font-size:13px;line-height:1.8;border-left:3px solid #bb9858;padding:8px 12px;background:#faf8f1}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}.summary>div,.scores>div{background:var(--ui-surface);border:1px solid var(--ui-line);border-radius:9px;padding:14px}.summary b{font-size:24px;display:block}.summary small{color:var(--ui-muted)}.query-tabs{display:flex;flex-wrap:wrap;gap:9px;margin:18px 0}.query-tabs button{padding:10px 16px;text-align:left}.query-tabs small{display:block;font-size:10px}.panel{background:var(--ui-surface);padding:22px;border:1px solid var(--ui-line);border-radius:12px;margin:16px 0;min-width:0}.case-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:15px}.case-heading h2{margin-bottom:0}.case-heading label{font-size:12px;display:flex;gap:8px;align-items:center}.case-heading select{padding:7px;max-width:230px}.badge{font-size:11px;padding:5px 9px;border-radius:6px;background:var(--ui-soft);white-space:nowrap}.case-overview{display:grid;grid-template-columns:minmax(240px,1fr) minmax(300px,1.2fr);gap:24px}video{display:block;width:100%;max-height:310px;background:#14221d;border-radius:8px}.scores{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:18px}.scores>div{padding:10px}.scores b{display:block;font-size:21px}.scores small{font-size:10px;color:var(--ui-muted)}.timeline-row{display:grid;grid-template-columns:100px minmax(80px,1fr) 120px;gap:8px;align-items:center;font-size:11px;margin:10px 0}.timeline-row>span{overflow-wrap:anywhere}.timeline-track{height:16px;background:#edf1ec;position:relative;border-radius:4px}.case .timeline-track button{position:absolute;height:14px;top:1px;border:0;border-radius:3px;padding:0;min-height:0;background:#42768a}.case .timeline-track button.gt{background:#8b958e}.case .timeline-track button.oracle{background:#b8823b}.case .timeline-track button.new{background:#32745b}.table-wrap{overflow:auto;margin:12px 0}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid var(--ui-line)}th{white-space:nowrap;font-size:11px}.empty{padding:22px;background:var(--ui-soft);border-radius:8px;font-size:13px;line-height:1.8}.empty p:last-child{margin-bottom:0}.margin-curve{display:block;width:100%;max-height:330px;background:#fbfcfa;border:1px solid var(--ui-line);border-radius:8px}.margin-curve text{font-size:11px;fill:#62736b}.grid-line{stroke:#e5ebe5;stroke-width:1}.zero-line{stroke:#ad7132;stroke-dasharray:5 4;stroke-width:1}.curve-legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:14px 0 8px}.blue{color:#42768a}.amber{color:#ad7132}.green{color:#246b4b}details{margin-top:15px}summary{cursor:pointer;font-size:13px;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;background:#f7f9f6;padding:12px;max-height:380px;overflow:auto}.preview-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.preview-grid figure{margin:0}.preview-grid img{width:100%;max-height:240px;object-fit:contain}.preview-grid figcaption{font-size:11px;color:var(--ui-muted);line-height:1.6;margin-bottom:8px}.preview-grid figcaption b,.preview-grid figcaption span{display:block}.preview-grid figcaption b{font-size:12px;color:var(--ui-ink);margin-bottom:3px}.preview-amplified figcaption span{color:#926331}.preview-note{font-size:12px;line-height:1.8;background:#f3f6f1;border-radius:8px;padding:12px}.preview-note.identical-png{background:#faf5e9}.difference-previews{margin:16px 0}.difference-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:12px}.difference-grid figure{margin:0}.difference-grid img{display:block;width:100%;max-height:240px;object-fit:contain}.difference-grid figcaption{font-size:11px;color:var(--ui-muted);margin-bottom:8px}body.embedded .page-header{display:none}.footer{font-size:12px;color:var(--ui-muted)}
@media(max-width:950px){.case-overview{grid-template-columns:1fr}.case-heading{flex-wrap:wrap}.timeline-row{grid-template-columns:100px 1fr 120px}.preview-grid{grid-template-columns:1fr}}
@media(max-width:600px){main{padding:14px}.page-header{margin:-14px -14px 16px}.panel{padding:15px}.summary{gap:7px}.summary>div{padding:10px}.summary b{font-size:20px}.summary small{font-size:10px}.query-tabs{display:grid;grid-template-columns:1fr}.query-tabs button{width:100%}.scores b{font-size:18px}.scores>div{padding:8px}.timeline-row{grid-template-columns:66px minmax(40px,1fr) 98px;gap:5px;font-size:10px}.timeline-row small{font-size:9px}.curve-legend{gap:10px;font-size:10px}.case-heading select{max-width:190px}}
'''

SCRIPT = '''
(() => {
  const cases = [...document.querySelectorAll('[data-case]')];
  if (window.self !== window.top) document.body.classList.add('embedded');
  function select(key, update = true) {
    const selected = cases.find(item => item.dataset.case === key) || cases[0];
    if (!selected) return;
    for (const item of cases) {
      item.hidden = item !== selected;
      if (item !== selected) item.querySelectorAll('video').forEach(video => video.pause());
    }
    document.querySelectorAll('[data-sample]').forEach(button => {
      const active = button.dataset.sample === selected.dataset.case;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    if (update) {
      const locationURL = new URL(location.href);
      locationURL.hash = new URLSearchParams({sample: selected.dataset.case});
      history.replaceState(null, '', locationURL);
      window.dispatchEvent(new Event('rollout:context'));
    }
  }
  document.querySelector('.query-tabs').addEventListener('click', event => {
    const button = event.target.closest('[data-sample]');
    if (button) select(button.dataset.sample);
  });
  document.querySelectorAll('.direction-picker').forEach(picker => picker.addEventListener('change', () => {
    picker.closest('.panel').querySelectorAll('[data-direction]').forEach(panel => {panel.hidden = panel.dataset.direction !== picker.value;});
  }));
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-seek]');
    if (!button) return;
    const video = button.closest('[data-case]').querySelector('video');
    if (video) {video.pause(); video.currentTime = Number(button.dataset.seek);}
  });
  window.addEventListener('hashchange', () => select(new URLSearchParams(location.hash.slice(1)).get('sample'), false));
  select(new URLSearchParams(location.hash.slice(1)).get('sample'));
})();
'''


def detail_html(selection, config, status, cases, artifacts, output):
    completed = sum(bool(query.get('complete')) for _, query in cases)
    total = len(cases)
    trials = sum(query_summary(query)['trials'] for _, query in cases)
    theme = url(PROJECT.parent / '_lib/experiment_theme.css', output)
    tabs = ''.join(f'<button data-sample="{esc(sample["sample_key"])}"><b>{esc(LABELS.get(sample.get("dataset"), sample.get("dataset")))}</b><small>{esc(sample["sample_key"])}</small></button>' for sample, _ in cases)
    state = status.get('state', status.get('stage', 'queued'))
    body = ''.join(case_html(sample, query, index, artifacts, output) for index, (sample, query) in enumerate(cases))
    return f'''<!doctype html><html lang="ko" data-family-default data-experiment-detail><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Oracle switch · pilot{total}</title><style>{STYLE}</style><link rel="stylesheet" href="{esc(theme)}"></head><body><main>
<header class="page-header"><p class="eyebrow">ROLLOUT / ORACLE SWITCH / PILOT {total}</p><h1>Oracle 쪽 선택을 영상 변화로 유도할 수 있는가</h1><p>기존 greedy와 oracle이 크게 다른 세 사례에서, 모델 입력 RGB의 작은 고정 방향 변화를 조사합니다.</p></header>
<div class="summary"><div><b>{completed} / {total}</b><small>새 측정 완료 query</small></div><div><b>{trials}</b><small>기록된 ε trial</small></div><div><b>{esc(state)}</b><small>실행 상태</small></div></div>
<p class="notice">GT를 사용해 고른 dataset별 극단 사례 1개씩입니다. 기존 greedy·oracle 결과는 이미 관측한 경로이며 새 입력 변형의 결과와 구분합니다. 전체 benchmark 평균으로 해석하지 않습니다.</p>
<nav class="query-tabs" aria-label="Dataset별 query 선택">{tabs}</nav>{body}<p class="footer">지표 정의·계산식·한계는 대표 viewer의 보고서 탭에 있습니다. 이 화면은 원본 영상·측정 곡선·경로 결과를 함께 탐색합니다.</p>
</main><script>{SCRIPT}</script></body></html>'''


def generate(artifacts=None, output=None):
    artifacts, output = Path(artifacts or ARTIFACTS), Path(output or OUTPUT)
    selection_path = artifacts / 'metadata/selection.json'
    selection = read(selection_path)
    if not selection or not selection.get('samples'):
        raise FileNotFoundError(f'Actual selected source cases are required before creating a report: {selection_path}')
    config = read(artifacts / 'metadata/config.json', {})
    status = read(artifacts / 'metadata/status.json', {'state': 'queued'})
    preview_manifest = read(artifacts / 'analyses/preview_manifest.json', {})
    cases = []
    for sample in selection['samples']:
        query = read(artifacts / 'queries' / (sample['sample_key'] + '.json'), {})
        supplemental = preview_manifest.get('samples', {}).get(sample['sample_key'], {}).get('directions', {})
        cases.append((sample, merge_preview_media(query, supplemental)))
    atomic_text(output / 'REPORT.md', markdown(selection, config, status, cases, artifacts, output))
    atomic_text(output / 'detail.html', detail_html(selection, config, status, cases, artifacts, output))
    return {'scope': 'pilot', 'queries': len(cases), 'completed': sum(bool(query.get('complete')) for _, query in cases),
            'report': str((output / 'REPORT.md').resolve()), 'detail': str((output / 'detail.html').resolve()),
            'generated_at': dt.datetime.now(dt.timezone.utc).isoformat()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.artifacts, args.output), ensure_ascii=False))
