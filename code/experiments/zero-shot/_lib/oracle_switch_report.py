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


def first_observed(direction, predicate):
    points = [point for point in direction.get('trials', []) if predicate(point) and finite(epsilon(point))]
    return min((epsilon(point) for point in points), default=None)


def support_text(direction):
    support = direction.get('support', {})
    if not support:
        return '選択待ち'.replace('選択', '선택').replace('待ち', ' 대기')
    pieces = []
    for key, label in [('slots', '입력 slot'), ('frame_ids', 'Source frame ID'), ('timestamps', '실제 timestamp'), ('spatial_bounds', 'Spatial bounds')]:
        if key in support:
            value = support[key]
            if key == 'timestamps' and isinstance(value, list):
                value = ', '.join(fmt(item, 3) + ' s' for item in value)
            else:
                value = json.dumps(value, ensure_ascii=False)
            pieces.append(f'{label}: {value}')
    return ' · '.join(pieces) if pieces else json.dumps(support, ensure_ascii=False)


def query_summary(query):
    tested = directions(query)
    return {
        'complete': bool(query.get('complete')),
        'trials': sum(len(item.get('trials', [])) for item in tested),
        'directions': len(tested),
        'pair_reversal': sum(any(finite(point.get('margin')) and point['margin'] < 0 for point in item.get('trials', [])) for item in tested),
        'target_top1': sum(any(point.get('target_top1') for point in item.get('trials', [])) for item in tested),
        'strict_target_top1': sum(any(point.get('strict_target_top1') for point in item.get('trials', [])) for item in tested),
    }


def chain_note(sample):
    chain = sample.get('oracle', {}).get('chain', [])
    if len(chain) <= 1:
        return '저장된 oracle continuation과 최초 분기를 비교합니다.'
    return ('저장된 oracle은 여러 강제 분기를 거친 경로입니다. 최초 분기의 oracle token 하나만 선택해도 '
            '그 전체 oracle 경로가 재현된다고 가정하지 않습니다.')


def methods():
    return [
        '## 측정 대상과 해석', '',
        '이 pilot은 기존 deterministic rollout에서 dataset별 oracle−greedy IoU 차이가 큰 query를 한 개씩 고른 '
        '**총 3 query의 진단 실험**이다. Oracle은 저장된 후보 중 GT IoU가 가장 높은 결과다. GT를 사용해 고른 극단 사례이며 '
        'dataset 평균 성능이나 일반적인 전환 성공률을 추정하는 표본이 아니다.', '',
        '원래 greedy prefix를 그대로 고정하고 최초 oracle 분기의 token g와 o를 비교한다. '
        '원래 모델 입력으로 resize한 float RGB x∈[0,1]에서 m(x)=z_g−z_o의 gradient를 계산한다. '
        '고정 방향 d=−sign(∇ₓm)에 support mask M을 적용하여 x′(ε)=clip(x+εMd,0,1)로 바꾼다. '
        'Text·embedding·FPS·frame IDs·timestamp·spatial resolution·token 구성은 바꾸지 않는다. '
        '같은 source frame이 반복 입력된 slot은 gradient를 합산해 방향을 정하고 동일한 perturbation을 적용한다.', '',
        'Support는 전체 영상, gradient L1 합이 가장 큰 frame pair, 그 pair에서 L1 합이 가장 큰 32×32 spatial cell이다. '
        '변형은 resize 이후 모델 입력의 float RGB에 적용하며 압축된 원본 영상 자체를 편집한 것이 아니다. '
        'Processor의 실제 mean/std를 사용해 모델 입력으로 변환한다. ε₂₅₅=255ε는 8-bit intensity level 단위이며 '
        '소수도 허용한다. 추론 입력을 uint8로 양자화하지 않는다. Preview PNG만 표시를 위해 반올림될 수 있다.', '',
        '초기 ε₂₅₅ grid는 0.25, 0.5, 1, 2, 4, 8, 16, 32다. 한 방향에서 처음 관측한 target top1의 '
        '실패·성공 구간을 추가 계측한다. 이 구간은 시험한 고정 gradient ray에서의 관측 경계이며, '
        '미계측 구간의 단조성이나 전역 최소 adversarial distance를 보장하지 않는다. '
        '전체 영상 방향에서 실패해도 국소 support 방향은 별도로 조사한다.', '',
        '| 지표 | 정의와 해석 |', '|---|---|',
        '| Paired margin | m=z_g−z_o. 음수이면 oracle token이 원래 greedy token보다 높다. 다른 token보다 높은지는 별도다. |',
        '| Oracle margin | r=z_o−max_{v≠o}z_v. 양수이면 전체 vocabulary에서 oracle token이 단독 top1이다. 0은 동률이다. |',
        '| Target top1 | 모델의 실제 전체 vocabulary argmax ID가 o인지 검사한다. 동률에서 o가 선택될 수 있으므로 strict target top1과 구분한다. |',
        '| Strict target top1 | Oracle margin>0. 모든 다른 token보다 o의 logit이 엄격히 높다. |',
        '| Probability | 원래 고정 top30 ID의 native softmax probability. Top30 안에서 재정규화하지 않는다. 전체 분포는 저장하지 않고 전체 argmax ID 등 검증 scalar만 별도 기록한다. |',
        '| RGB L∞ | maxᵢ|x′ᵢ−xᵢ|. Clipping 이후의 실제 최대 변경량이며 계획 ε보다 작을 수 있다. |',
        '| RGB RMS | sqrt(meanᵢ((x′ᵢ−xᵢ)²)). 분모는 전체 모델 입력 RGB scalar 수이며 support 밖의 0 변경도 포함한다. |',
        '| Changed fraction | 변경된 RGB scalar 수 / 전체 모델 입력 RGB scalar 수. |',
        '| Temporal IoU | |예측 구간∩GT| / |예측 구간∪GT|. 0–1이며 클수록 GT와 겹친다. |',
        '| Oracle span IoU | 새 예측 구간과 저장된 oracle 구간의 temporal IoU. GT IoU와 다른 비교다. |', '',
        '**Conditional continuation**은 원래 prefix를 강제한 상태에서 분기 이후를 이어간 결과다. '
        '**Free decoding**은 변경된 입력에서 응답 시작부터 greedy decoding한 결과다. '
        '원래 prefix를 조건으로 한 token 역전, 실제 top1 전환, free decoding의 prefix 도달 여부와 최종 구간·GT IoU를 각각 구분한다. '
        'Oracle branch reached와 oracle path equal도 서로 다른 검사다. Token 전환을 최종 mode 전환이나 정답 개선과 동일시하지 않는다.', '',
        '성공 구간의 두 endpoint에서 free decoding을 비교하고, high endpoint에서 conditional continuation·반복 계측·반대 방향 control을 확인한다. '
        '미관측 결과는 실패로 세지 않는다. 전환을 못 찾은 방향은 해당 support·고정 방향·시험한 ε 범위에서의 음성 결과다. '
        '자연스러운 영상 수정의 효과, 다른 방향의 불가능성, 전역 최소 perturbation을 주장하지 않는다.', '',
    ]


def direction_rows(query):
    rows = []
    for item in directions(query):
        paired = first_observed(item, lambda point: finite(point.get('margin')) and point['margin'] < 0)
        target = first_observed(item, lambda point: bool(point.get('target_top1')))
        strict = first_observed(item, lambda point: bool(point.get('strict_target_top1')))
        low, high = boundary_values(item)
        limits = [epsilon(point) for point in item.get('trials', []) if finite(epsilon(point))]
        measured_limit = max(limits, default=None)
        rows.append([LABELS.get(direction_id(item), item.get('label', direction_id(item))),
                     fmt(paired, 3), fmt(target, 3), fmt(strict, 3),
                     f'({fmt(low, 4)}, {fmt(high, 4)}]' if high is not None else f'미관측 · 최대 {fmt(measured_limit, 3)}',
                     fmt((item.get('free_low') or {}).get('iou')),
                     fmt((item.get('free_high') or {}).get('iou')),
                     fmt((item.get('conditional_high') or {}).get('iou'))])
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
        for field, label, value in [('free_low', 'Free · low', low), ('free_high', 'Free · high', high), ('conditional_high', 'Conditional · high', high)]:
            outcome = direction.get(field)
            if not outcome:
                continue
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
            lines += ['**실행 대기:** 원본 사례 선택만 완료했다. Gradient·ε scan·conditional continuation·free decoding은 미계측이다.', '']
            continue
        lines += [f'새 측정 상태: {"완료" if summary["complete"] else "진행 중"}. 기록된 방향 {summary["directions"]}개, trial {summary["trials"]}개. '
                  f'Paired reversal 관측 {summary["pair_reversal"]}/{summary["directions"]} 방향, target top1 관측 '
                  f'{summary["target_top1"]}/{summary["directions"]}, strict target top1 관측 {summary["strict_target_top1"]}/{summary["directions"]}.', '',
                  '아래 ε는 모두 ε₂₅₅ 단위다. 최소 관측 값은 저장된 trial 중의 최솟값이며 미계측 ε의 최솟값이 아니다.', '']
        lines += md_table(['Support', '첫 paired reversal ε', '첫 target top1 ε', '첫 strict top1 ε', '관측 top1 경계', 'Free low IoU', 'Free high IoU', 'Conditional high IoU'], direction_rows(query))
        outcomes = outcome_rows(query)
        if outcomes:
            lines += ['', '원래 prefix를 강제했는지에 따라 결과를 따로 비교한다.', '']
            lines += md_table(['Support', 'Decoding', 'ε₂₅₅', '새 구간', 'GT IoU', 'Oracle span IoU', '첫 token 차이 step', 'Oracle branch 도달', 'Oracle path 일치'], outcomes)
        for direction in directions(query):
            lines += ['', f'**{LABELS.get(direction_id(direction), direction_id(direction))} support:** {support_text(direction)}.']
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
    lines += ['', '새 데이터는 원본 greedy·oracle source를 덮어쓰지 않는다. 입력 hash·실제 preprocessing·baseline 재현·support와 '
              '중복 source frame 처리·실제 clipping 이후 거리·runtime cost는 query 원자료에 보존한다. '
              '이 보고서의 곡선은 측정점만 나타내며 점 사이 연결선은 미계측 결과를 뜻하지 않는다.', '']
    return '\n'.join(lines)


def html_table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{esc(cell)}</th>' for cell in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(cell)}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def curve(query, direction):
    points = [dict(query.get('baseline', {}), epsilon_255=0)] + list(direction.get('trials', []))
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


def media_previews(direction, artifacts, output):
    media = direction.get('media') or {}
    figures = []
    for name, label in [('original', '원래 입력 preview'), ('perturbed', '변경된 입력 preview'), ('difference', 'Difference preview')]:
        source = existing_url(media.get(name), output, artifacts)
        if source:
            figures.append(f'<figure><img src="{esc(source)}" alt="{label}" loading="lazy"><figcaption>{label}</figcaption></figure>')
    if not figures:
        return ''
    scale = media.get('difference_scale', media.get('scale'))
    note = f'Difference 표시 배율: {esc(scale)}.' if scale is not None else 'Difference의 증폭 여부와 배율은 연결된 원자료 metadata를 확인하세요.'
    return '<div class="preview-grid">' + ''.join(figures) + f'</div><p class="small">PNG는 표시용 반올림 결과이며 실제 추론 입력은 float RGB입니다. {note}</p>'


def direction_html(query, direction, index, artifacts, output):
    key = direction_id(direction)
    low, high = boundary_values(direction)
    points = direction.get('trials', [])
    summary = f'관측 target top1 경계 ({fmt(low, 4)}, {fmt(high, 4)}] /255' if high is not None else '시험한 범위에서 target top1 경계 미관측'
    details = []
    for point in points:
        perturbation = point.get('perturbation', {})
        details.append([fmt(epsilon(point), 4), fmt(point.get('margin')), fmt(point.get('oracle_margin')),
                        fmt(point.get('greedy_p')), fmt(point.get('oracle_p')), point.get('top1_id', '—'),
                        '예' if point.get('target_top1') else '아니오', '예' if point.get('strict_target_top1') else '아니오',
                        fmt(perturbation.get('rgb_linf')), fmt(perturbation.get('rgb_rms')), fmt(perturbation.get('changed_fraction')), point.get('phase', '')])
    return f'<section class="direction-view" data-direction="{esc(key)}" {"hidden" if index else ""}><h3>{esc(LABELS.get(key, direction.get("label", key)))}</h3><p class="small">{esc(support_text(direction))}</p><p><b>{esc(summary)}</b></p>{curve(query, direction)}<p class="curve-legend"><span class="blue">● Paired margin ≥ 0</span><span class="amber">● Paired reversal</span><span class="green">● 실제 target top1</span></p><p class="small">음수 margin은 두 token의 순서만 바뀌었다는 뜻입니다. 초록 점은 전체 vocabulary argmax가 oracle token입니다. 점 사이 선은 측정점 연결이며 보간 예측이 아닙니다.</p>{media_previews(direction, artifacts, output)}<details><summary>측정점·실제 RGB 변경량 펼치기 ({len(points)} trial)</summary>{html_table(["ε₂₅₅", "g−oracle", "Oracle margin", "p(g)", "p(oracle)", "Argmax ID", "Target top1", "Strict top1", "RGB L∞", "RGB RMS", "변경 비율", "Phase"], details)}</details></section>'


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
        for field, name in [('free_low', 'Free low'), ('free_high', 'Free high'), ('conditional_high', 'Conditional high')]:
            outcome = item.get(field) or {}
            if outcome.get('span'):
                new_spans.append((f'{LABELS.get(direction_id(item), direction_id(item))} · {name}', outcome['span']))
    bars += [timeline_row(name, value, duration, 'new') for name, value in new_spans]
    measured = directions(query)
    options = ''.join(f'<option value="{esc(direction_id(item))}">{esc(LABELS.get(direction_id(item), item.get("label", direction_id(item))))}</option>' for item in measured)
    panels = ''.join(direction_html(query, item, position, artifacts, output) for position, item in enumerate(measured))
    if not measured:
        panels = '<div class="empty"><b>GPU 실행 대기 · 아직 새 측정 결과가 없습니다.</b><p>계측할 support는 전체 영상 → 선택한 frame pair → 선택한 32×32 spatial cell입니다. 기존 결과만 아래 사례의 출처로 표시합니다.</p></div>'
    outcomes = outcome_rows(query)
    outcomes_html = html_table(['Support', 'Decoding', 'ε₂₅₅', '새 구간', 'GT IoU', 'Oracle span IoU', '첫 token 차이 step', 'Oracle branch 도달', 'Oracle path 일치'], outcomes) if outcomes else '<p class="empty">새 conditional continuation과 free decoding은 아직 없습니다.</p>'
    raw = artifacts / 'queries' / (key + '.json')
    raw_link = f'<a href="{esc(url(raw, output))}" target="_blank" rel="noopener">Query 원자료 ↗</a>' if raw.exists() else ''
    return f'''<article class="case" data-case="{esc(key)}" {"hidden" if index else ""}>
<section class="panel"><div class="case-heading"><div><p class="eyebrow">{esc(LABELS.get(sample.get('dataset'), sample.get('dataset')))} · {esc(key)}</p><h2>{esc(row.get('query', key))}</h2></div><span class="badge">{'새 측정 완료' if query.get('complete') else '측정 진행 중' if query else '실행 대기'}</span></div>
<div class="case-overview"><div>{video_html}<p class="small">원본 영상입니다. 아래 RGB perturbation은 resize한 모델 입력에만 적용합니다.</p></div><div><div class="scores"><div><small>저장된 greedy IoU</small><b>{fmt(sample.get('greedy', {}).get('iou'))}</b></div><div><small>저장된 oracle IoU</small><b>{fmt(sample.get('oracle', {}).get('iou'))}</b></div><div><small>선택 기준 ΔIoU</small><b>{fmt(sample.get('delta_iou'))}</b></div></div>{''.join(bars)}<p class="small">구간 막대를 누르면 같은 원본 시각으로 이동합니다. Oracle은 GT를 사용해 선택한 기존 후보입니다.</p></div></div>
<p><b>Step {branch.get('step', '—')}:</b> <code>{esc(branch.get('greedy_token_text', branch.get('greedy_token')))}</code> → <code>{esc(branch.get('oracle_token_text', branch.get('oracle_token')))}</code> · Source rank {esc(branch.get('source_rank', '—'))}</p><p class="notice">{esc(chain_note(sample))}</p></section>
<section class="panel"><div class="case-heading"><div><p class="eyebrow">01 / RGB 변화와 token 선택</p><h2>얼마나 바꿨을 때 경쟁이 바뀌는가</h2></div>{'<label>Support <select class="direction-picker">'+options+'</select></label>' if measured else ''}</div><p class="small">원래 prefix에서 계산한 고정 −sign gradient 방향을 시험합니다. ε₂₅₅는 RGB intensity level이며 자연스러운 영상 편집량이나 전체 영상의 변화 비율이 아닙니다.</p>{panels}</section>
<section class="panel"><p class="eyebrow">02 / 응답을 다시 시작한 결과</p><h2>Token 역전이 최종 구간으로 이어졌는가</h2><p class="small">Conditional은 원래 prefix를 유지합니다. Free는 변경 입력에서 응답 처음부터 greedy decoding합니다. Oracle branch 도달·전체 path 일치·최종 GT IoU를 따로 봅니다.</p>{outcomes_html}</section>
<section class="panel raw"><details><summary>방향별 요약·실행 기록</summary>{html_table(['Support', '첫 paired reversal ε', '첫 target top1 ε', '첫 strict top1 ε', '관측 top1 경계', 'Free low IoU', 'Free high IoU', 'Conditional high IoU'], direction_rows(query))}<p>{raw_link}</p><pre>{esc(json.dumps({'baseline': query.get('baseline'), 'cost': query.get('cost'), 'validation': query.get('validation')},ensure_ascii=False,indent=2))}</pre></details></section>
</article>'''


STYLE = '''
main{max-width:1500px;margin:auto;padding:24px}h1,h2,h3,p{margin-top:0}h2{font-size:21px}.page-header{margin:-24px -24px 24px}.eyebrow{font-size:11px;letter-spacing:.08em;font-weight:700;color:var(--ui-accent)}.small{font-size:12px;line-height:1.8;color:var(--ui-muted)}.notice{font-size:13px;line-height:1.8;border-left:3px solid #bb9858;padding:8px 12px;background:#faf8f1}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}.summary>div,.scores>div{background:var(--ui-surface);border:1px solid var(--ui-line);border-radius:9px;padding:14px}.summary b{font-size:24px;display:block}.summary small{color:var(--ui-muted)}.query-tabs{display:flex;flex-wrap:wrap;gap:9px;margin:18px 0}.query-tabs button{padding:10px 16px;text-align:left}.query-tabs small{display:block;font-size:10px}.panel{background:var(--ui-surface);padding:22px;border:1px solid var(--ui-line);border-radius:12px;margin:16px 0;min-width:0}.case-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:15px}.case-heading h2{margin-bottom:0}.case-heading label{font-size:12px;display:flex;gap:8px;align-items:center}.case-heading select{padding:7px;max-width:230px}.badge{font-size:11px;padding:5px 9px;border-radius:6px;background:var(--ui-soft);white-space:nowrap}.case-overview{display:grid;grid-template-columns:minmax(240px,1fr) minmax(300px,1.2fr);gap:24px}video{display:block;width:100%;max-height:310px;background:#14221d;border-radius:8px}.scores{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:18px}.scores>div{padding:10px}.scores b{display:block;font-size:21px}.scores small{font-size:10px;color:var(--ui-muted)}.timeline-row{display:grid;grid-template-columns:100px minmax(80px,1fr) 120px;gap:8px;align-items:center;font-size:11px;margin:10px 0}.timeline-row>span{overflow-wrap:anywhere}.timeline-track{height:16px;background:#edf1ec;position:relative;border-radius:4px}.case .timeline-track button{position:absolute;height:14px;top:1px;border:0;border-radius:3px;padding:0;min-height:0;background:#42768a}.case .timeline-track button.gt{background:#8b958e}.case .timeline-track button.oracle{background:#b8823b}.case .timeline-track button.new{background:#32745b}.table-wrap{overflow:auto;margin:12px 0}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid var(--ui-line)}th{white-space:nowrap;font-size:11px}.empty{padding:22px;background:var(--ui-soft);border-radius:8px;font-size:13px;line-height:1.8}.empty p:last-child{margin-bottom:0}.margin-curve{display:block;width:100%;max-height:330px;background:#fbfcfa;border:1px solid var(--ui-line);border-radius:8px}.margin-curve text{font-size:11px;fill:#62736b}.grid-line{stroke:#e5ebe5;stroke-width:1}.zero-line{stroke:#ad7132;stroke-dasharray:5 4;stroke-width:1}.curve-legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:14px 0 8px}.blue{color:#42768a}.amber{color:#ad7132}.green{color:#246b4b}details{margin-top:15px}summary{cursor:pointer;font-size:13px;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;background:#f7f9f6;padding:12px;max-height:380px;overflow:auto}.preview-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.preview-grid figure{margin:0}.preview-grid img{width:100%;max-height:240px;object-fit:contain}.preview-grid figcaption{font-size:11px;color:var(--ui-muted)}body.embedded .page-header{display:none}.footer{font-size:12px;color:var(--ui-muted)}
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
    cases = [(sample, read(artifacts / 'queries' / (sample['sample_key'] + '.json'), {})) for sample in selection['samples']]
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
