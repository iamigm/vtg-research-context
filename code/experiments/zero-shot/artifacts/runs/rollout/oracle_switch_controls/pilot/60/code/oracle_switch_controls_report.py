"""CPU-only reporting for the prespecified 60-query visual control study."""
from __future__ import annotations

import argparse
import collections
import datetime
import html
import json
import math
import os
from pathlib import Path
import random
import statistics
from urllib.parse import quote

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / 'artifacts/runs/rollout/oracle_switch_controls/pilot/60'
OUTPUT = PROJECT / 'results/rollout/oracle_switch_controls/pilot/60'
DATASETS = ('charades-timelens', 'activitynet-timelens', 'qvhighlights-timelens')
DATASET_LABELS = dict(zip(DATASETS, ('Charades', 'ActivityNet', 'QVHighlights')))
POPULATION = dict(zip(DATASETS, (277, 266, 262)))
BUDGETS = (.25, 1.)
SEEDS = (42, 43, 44)
METHODS = ('targeted', 'opposite', 'random', 'repeat_gradient', 'rounded8_targeted')
LABELS = {
    'targeted': 'Targeted gradient', 'opposite': '반대 방향', 'random': 'Random · 3 seed 평균',
    'repeat_gradient': '반복 gradient', 'rounded8_targeted': 'Targeted 후 8-bit roundtrip',
    'native': 'Native BF16', 'patch_fp32_bf16_input': '첫 projection FP32 · BF16 입력',
    'patch_fp32_float_input': '첫 projection FP32 · FP32 입력',
}
BOOTSTRAPS = 1000


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_text(path, text):
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
    return f'{value:.2e}' if value and abs(value) < 10 ** -digits else f'{value:.{digits}f}'


def percent(value):
    return f'{100*value:.1f}%' if finite(value) else '—'


def yes(value):
    return '예' if value is True else '아니오' if value is False else '—'


def span(value):
    if isinstance(value, dict):
        value = value.get('span')
    return f'{fmt(value[0], 2)}–{fmt(value[1], 2)} s' if isinstance(value, list) and len(value) == 2 else '미관측'


def relative(path, output, base=ARTIFACTS):
    if not path:
        return ''
    if isinstance(path, dict):
        path = path.get('path')
    if not path:
        return ''
    if str(path).startswith(('https://', 'http://')):
        return str(path)
    path = Path(path)
    path = path if path.is_absolute() else base / path
    return quote(os.path.relpath(path.resolve(), Path(output).resolve()), safe='/')


def mean(values):
    values = [value for value in values if finite(value)]
    return sum(values)/len(values) if values else None


def interval_text(interval):
    return f'{percent(interval[0])}–{percent(interval[1])}' if interval else '—'


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    proportion = successes/n
    denominator = 1+z*z/n
    center = (proportion+z*z/(2*n))/denominator
    half = z*math.sqrt(proportion*(1-proportion)/n+z*z/(4*n*n))/denominator
    return max(0, center-half), min(1, center+half)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*q
    low, high = int(math.floor(index)), int(math.ceil(index))
    return values[low]*(high-index)+values[high]*(index-low) if high != low else values[low]


def weighted_mean(rows, field, population, dataset='all'):
    datasets = DATASETS if dataset == 'all' else (dataset,)
    means = {name: mean([row.get(field) for row in rows if row['dataset'] == name]) for name in datasets}
    if any(value is None for value in means.values()):
        return None
    total = sum(population[name] for name in datasets)
    return sum(population[name]/total*means[name] for name in datasets)


def cluster_interval(rows, field, population, dataset='all'):
    """Resample videos within each dataset; repeated seeds stay in their query row."""
    datasets = DATASETS if dataset == 'all' else (dataset,)
    clusters = {}
    for name in datasets:
        mapping = collections.defaultdict(list)
        for row in rows:
            if row['dataset'] == name and finite(row.get(field)):
                mapping[row['video']].append(row[field])
        if not mapping:
            return None
        clusters[name] = list(mapping.values())
    rng = random.Random(42)
    total = sum(population[name] for name in datasets)
    estimates = []
    for _ in range(BOOTSTRAPS):
        estimate = 0.
        for name in datasets:
            units = clusters[name]
            numerator, denominator = 0., 0
            for _ in units:
                values = units[rng.randrange(len(units))]
                numerator += sum(values)
                denominator += len(values)
            estimate += population[name]/total*numerator/denominator
        estimates.append(estimate)
    return percentile(estimates, .025), percentile(estimates, .975)


def load_cases(selection, artifacts):
    cases = []
    for sample in selection['samples']:
        key = sample['sample_key']
        cases.append(dict(sample=sample,
                          native=read(artifacts/'queries'/f'{key}.json', {}),
                          controls=read(artifacts/'analyses/controls'/f'{key}.json', {})))
    return cases


def baseline_iou(case):
    controls = case['controls']
    for source in (controls.get('baseline_free'), controls.get('details', {}).get('baseline_free'),
                   case['native'].get('baseline_free'), case['sample'].get('greedy')):
        if isinstance(source, dict) and finite(source.get('iou')):
            return source['iou']
    return None


def matching_trial(case, method, budget, seed=None):
    matches = [trial for trial in case['controls'].get('trials', [])
               if trial.get('method') == method and finite(trial.get('epsilon_255'))
               and abs(trial['epsilon_255']-budget) < 1e-9
               and (method != 'random' or trial.get('seed') == seed)]
    return next((trial for trial in reversed(matches) if isinstance(trial.get('free'), dict)), None)


def observed_free(trial):
    if not trial:
        return None
    outcome = trial.get('free')
    if not isinstance(outcome, dict) or not isinstance(outcome.get('oracle_branch_reached'), bool):
        return None
    return outcome


def condition_row(case, method, budget):
    sample, controls = case['sample'], case['controls']
    row = dict(key=sample['sample_key'], dataset=sample['dataset'],
               video=sample.get('row', {}).get('video_path', sample['sample_key']),
               failed=bool(controls.get('failed')), resolved=False, evaluable=False,
               score=None, iou=None, delta=None)
    if row['failed']:
        row.update(resolved=True, score=0.)
        return row
    trials = [matching_trial(case, method, budget, seed) for seed in SEEDS] if method == 'random' else [matching_trial(case, method, budget)]
    outcomes = [observed_free(trial) for trial in trials]
    if any(outcome is None for outcome in outcomes):
        return row
    row.update(resolved=True, evaluable=True, score=mean([float(outcome['oracle_branch_reached']) for outcome in outcomes]))
    if all(finite(outcome.get('iou')) for outcome in outcomes):
        row['iou'] = mean([outcome['iou'] for outcome in outcomes])
        baseline = baseline_iou(case)
        row['delta'] = row['iou']-baseline if baseline is not None else None
    return row


def aggregate(rows, population, dataset='all', random_method=False):
    selected = [row for row in rows if dataset == 'all' or row['dataset'] == dataset]
    resolved = [row for row in selected if row['resolved']]
    observed = [row for row in selected if row['evaluable']]
    n = len(selected)
    pending = n-len(resolved)
    scores_with_pending_zero = [dict(row, score=row['score'] if row['resolved'] else 0.) for row in selected]
    confirmed = weighted_mean(scores_with_pending_zero, 'score', population, dataset) if resolved else None
    observed_rate = weighted_mean(observed, 'score', population, dataset)
    interval = None
    if n and pending == 0:
        interval = cluster_interval(selected, 'score', population, dataset) if dataset == 'all' or random_method else wilson(sum(row['score'] for row in selected), n)
    return dict(n=n, resolved=len(resolved), evaluable=len(observed), pending=pending,
                failed=sum(row['failed'] for row in selected), confirmed=confirmed,
                observed_rate=observed_rate, interval=interval,
                iou=weighted_mean(observed, 'iou', population, dataset),
                delta=weighted_mean(observed, 'delta', population, dataset),
                iou_n=sum(finite(row.get('iou')) for row in observed),
                clusters=len({(row['dataset'], row['video']) for row in selected}))


def comparisons(cases, population):
    summaries, rows_by_condition, paired = {}, {}, {}
    for budget in BUDGETS:
        for method in METHODS:
            rows = [condition_row(case, method, budget) for case in cases]
            rows_by_condition[budget, method] = rows
            for dataset in ('all',)+DATASETS:
                summaries[budget, method, dataset] = aggregate(rows, population, dataset, method == 'random')
        for control in ('random', 'opposite'):
            contrasts = []
            for targeted, compared in zip(rows_by_condition[budget, 'targeted'], rows_by_condition[budget, control]):
                contrasts.append(dict(key=targeted['key'], dataset=targeted['dataset'], video=targeted['video'],
                                      resolved=targeted['resolved'] and compared['resolved'],
                                      score=targeted['score']-compared['score'] if targeted['resolved'] and compared['resolved'] else None,
                                      delta=targeted['iou']-compared['iou'] if finite(targeted['iou']) and finite(compared['iou']) else None))
            for dataset in ('all',)+DATASETS:
                relevant = [row for row in contrasts if dataset == 'all' or row['dataset'] == dataset]
                finished = bool(relevant) and all(row['resolved'] for row in relevant)
                paired[budget, control, dataset] = dict(
                    n=sum(row['resolved'] for row in relevant), selected=len(relevant),
                    rate_difference=weighted_mean(relevant, 'score', population, dataset),
                    rate_interval=cluster_interval(relevant, 'score', population, dataset) if finished else None,
                    iou_difference=weighted_mean(relevant, 'delta', population, dataset),
                    iou_n=sum(finite(row['delta']) for row in relevant),
                    iou_interval=cluster_interval(relevant, 'delta', population, dataset) if finished else None)
    return summaries, paired


def metric_rows(summaries, budget, dataset):
    rows = []
    for method in METHODS:
        value = summaries[budget, method, dataset]
        rows.append([LABELS[method], f'{value["resolved"]}/{value["n"]}',
                     percent(value['confirmed']), interval_text(value['interval']),
                     percent(value['observed_rate']), fmt(value['iou']), fmt(value['delta']),
                     value['iou_n'], f'{value["failed"]} / {value["pending"]}'])
    return rows


HEADERS = ['입력 변경 조건', '확정/선정 query', '확인 전환률', '95% 구간', '관측 전환률', '관측 GT IoU', '관측 ΔIoU', 'IoU 관측 n', '실패 / 대기']
SUPPORT_HEADERS = ['변경 범위', '검증 경계 도달/검사 query', '검증 upper ε₂₅₅ 중앙값', '검사 중 경계 미확인', '미관측 query']


def support_rows(cases, dataset='all'):
    selected = [case for case in cases if dataset == 'all' or case['sample']['dataset'] == dataset]
    rows = []
    for support, label in [('all_video', '전체 영상'), ('top_frame_pair', '선택 frame pair'), ('top_spatial_cell', '선택 spatial cell')]:
        measured = []
        verified = []
        for case in selected:
            direction = next((item for item in case['native'].get('directions', []) if item.get('id') == support and item.get('complete')), None)
            if not direction:
                continue
            measured.append(direction)
            boundary = direction.get('boundary') or {}
            if boundary.get('verified') and (direction.get('free_high') or {}).get('oracle_branch_reached') is True and finite(boundary.get('high_epsilon_255')):
                verified.append(boundary['high_epsilon_255'])
        rows.append([label, f'{len(verified)}/{len(measured)}', fmt(statistics.median(verified) if verified else None), len(measured)-len(verified), len(selected)-len(measured)])
    return rows


def interpretation(cases, summaries, paired):
    """Use one evidence summary for the authoritative MD and the interactive view."""
    completed = sum(bool(case['controls'].get('complete')) for case in cases)
    progress = ', '.join(f'{DATASET_LABELS[name]} {sum(bool(case["controls"].get("complete")) for case in cases if case["sample"]["dataset"] == name)}/20'
                         for name in DATASETS)
    lines = []
    if completed < len(cases):
        lines.append(f'현재 control 확정은 {completed}/{len(cases)} query다({progress}). Dataset별 진행량이 다를 수 있으므로 먼저 처리된 사례의 결과를 모집단 전체의 경향으로 읽지 않는다. 아직 최종 비교와 원인 해석은 미확정이며, 아래 표의 확인 전환률은 현재까지 확인된 전환의 하한이다.')
    else:
        failures = sum(bool(case['controls'].get('failed')) for case in cases)
        lines.append(f'선정 {len(cases)} query의 처리가 확정됐다({progress}; baseline 재현 실패 {failures}). Branch token 도달과 저장된 oracle 전체 경로 재현·GT IoU 개선은 별개다.')
        for budget in BUDGETS:
            target, random_result = summaries[budget, 'targeted', 'all'], summaries[budget, 'random', 'all']
            contrast = paired[budget, 'random', 'all']
            interval = contrast['rate_interval']
            interval_label = f'[{fmt(interval[0])}, {fmt(interval[1])}]' if interval else '미산출'
            lines.append(f'ε₂₅₅={budget:g}에서 모집단 가중 관측 branch 도달률은 targeted {percent(target["observed_rate"])} / random {percent(random_result["observed_rate"])}다. 선정 query 기준 paired targeted−random 차이는 {fmt(contrast["rate_difference"])}(95% cluster bootstrap {interval_label})이며, 관측 ΔGT IoU는 targeted {fmt(target["delta"])} / random {fmt(random_result["delta"])}다. 전환률 차이와 성능 차이를 각각 읽어야 한다.')
    roundtrips = [matching_trial(case, 'rounded8_targeted', .25) for case in cases]
    identical = [(trial.get('roundtrip') or {}).get('normalized_tensor_identical_to_baseline') for trial in roundtrips if trial]
    identical = [value for value in identical if isinstance(value, bool)]
    if identical:
        lines.append(f'ε₂₅₅=0.25의 8-bit roundtrip 후 normalized tensor가 원본과 완전히 같은 경우는 관측 {sum(identical)}/{len(identical)} query다. 동일한 경우 모델에 남은 입력 변경이 없음을 뜻하며, 양자화 전 변경이나 다른 변경량의 효과까지 없다고 뜻하지 않는다.')
    for mode_name in ('patch_fp32_bf16_input', 'patch_fp32_float_input'):
        modes, pairs = [], []
        for case in cases:
            audit_modes = (case['controls'].get('precision_audit') or {}).get('modes', [])
            mode = next((item for item in audit_modes if item.get('mode') == mode_name), None)
            native = next((item for item in audit_modes if item.get('mode') == 'native'), None)
            if not mode:
                continue
            modes.append(mode)
            for condition in mode.get('conditions', []):
                outcome = (condition.get('free') or {}).get('oracle_branch_reached')
                native_condition = next((item for item in (native or {}).get('conditions', []) if item.get('epsilon_255') == condition.get('epsilon_255')), None)
                native_outcome = ((native_condition or {}).get('free') or {}).get('oracle_branch_reached')
                if isinstance(outcome, bool) and isinstance(native_outcome, bool):
                    pairs.append(outcome == native_outcome)
        if modes:
            matches = [item.get('baseline_matches_native_prefix_and_token') for item in modes]
            matches = [value for value in matches if isinstance(value, bool)]
            pair_text = f' 두 예산의 실제 branch 결과가 native와 같은 비교 {sum(pairs)}/{len(pairs)}, 다른 비교 {len(pairs)-sum(pairs)}/{len(pairs)}다.' if pairs else ' Native와의 실제 branch 결과 비교는 아직 미관측이다.'
            lines.append(f'사전 지정 precision 진단의 {LABELS[mode_name]}에서 native baseline prefix·선택 token 일치는 관측 {sum(matches)}/{len(matches)} query다.'+pair_text)
    if any((case['controls'].get('precision_audit') or {}).get('modes') for case in cases):
        lines.append('Precision 진단은 사전 지정6 query의 첫 projection만 바꾼 비교이며 전체 FP32 모델 실험이 아니다. 각 mode의 자체 baseline 기준 Δoracle margin과 baseline 일치 여부를 함께 읽는다. 방향·양자화·첫 계산 정밀도의 영향을 좁혀 보는 증거이며 한 원인을 확정하지 않는다.')
    return lines


def table(headers, rows, markdown=False, classes=''):
    if markdown:
        clean = lambda item: str(item).replace('|', '\\|').replace('\n', ' ')
        return '\n'.join(['| '+' | '.join(map(clean, headers))+' |', '| '+' | '.join('---' for _ in headers)+' |']+
                         ['| '+' | '.join(map(clean, row))+' |' for row in rows])
    return '<div class="table-wrap"><table class="'+classes+'"><thead><tr>'+''.join('<th>'+esc(item)+'</th>' for item in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(item)+'</td>' for item in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def definitions():
    return '''## 모집단과 표본

기존 fixed pilot900 중 저장된 oracle의 GT IoU가 greedy보다 높고, 최초 oracle 분기 token이 원래 top30에 있는 query에서 기존 극단 사례 3개를 제외했다. 이 조건부 모집단은 **805 query**(Charades 277, ActivityNet 266, QVHighlights 262)다. Dataset별로 SHA256(`oracle_switch_controls|42|`+sample_key)가 작은 20개씩, 총 60개를 사전에 고정했다. 기존 3개를 60개에 합산하지 않는다. 결과는 이 조건부 모집단을 대상으로 하며 전체 benchmark·oracle 개선이 없는 query·다른 모델·자연스러운 영상 수정으로 일반화하지 않는다.

## 같은 변경량에서 비교하는 조건

주 비교 예산은 ε₂₅₅=0.25, 1이며 전체 영상의 같은 변경 범위를 사용한다. 모든 결과 행은 변경된 입력에서 **응답 처음부터 같은 greedy decoding**으로 생성한다. Targeted gradient, 반대 방향, random seed42·43·44, 반복 gradient, targeted 후 8-bit roundtrip을 비교한다. 고정 prefix의 다음 token을 평가하는 probe는 별도 진단이며 decoding 방식이 아니다. 요청 RGB L∞를 맞춰도 clipping 뒤 실제 RGB L∞·RMS가 같다고 가정하지 않는다.

Random의 세 seed는 같은 query의 반복 측정이다. 주 표에는 세 seed의 결과를 query 안에서 평균한 뒤 query를 집계한다. 세 seed가 모두 완료되어야 그 query의 random 조건을 평가 가능으로 센다. 180개의 독립 query로 취급하지 않는다. Query 상세에서 각 seed를 따로 확인할 수 있다.

모델에 넣는 resized floating RGB x∈[0,1]에서 m=z_g−z_oracle의 첫 gradient를 고정하고 targeted 방향은 x′=clip(x−ε·sign(∂m/∂x),0,1)로 만든다. ε₂₅₅=255ε이며 추론 입력을 uint8로 반올림하지 않는다. 같은 source frame의 반복 입력 slot은 같은 변경을 공유한다. 실제 RGB L∞=max|x′−x|, RMS=√mean((x′−x)²)이며 원자료의 physical RGB 분모와 packed normalized input 분모를 구분한다. Cast 전후 변경 비율은 반복 slot을 포함한 전체 packed normalized scalar 중 값이 달라진 비율이다.

## 변경 범위와 관측 경계

Native scan은 전체 영상, gradient 크기로 선택한 frame pair, 선택한 spatial cell에서 각각 하나의 고정 gradient 방향을 시험한다. ε₂₅₅∈{0.25,0.5,1,2,4,8,16,32}에서 처음 관측된 branch 도달 구간을6회 좁히며 upper에서 반복 probe와 실제 greedy 응답을 확인한다. 표의 검증 upper 중앙값은 **검증된 사례만의 중앙값**이다. 검사 완료 중 경계가 없거나 검증되지 않은 경우와 아직 미관측인 경우를 따로 센다. 최대32/255까지 경계를 못 찾았다고 해당 branch가 불가능한 것은 아니다. 이 값은 선택한 gradient ray의 탐색 결과이며 전역 최소 변경량이 아니다.

원인 후보는 네 축으로 읽는다. ①같은 예산의 targeted−random 차이는 시험한 방향의 특이성, ②전체 영상과 pair/cell의 경계는 변경 범위와 누적 효과, ③8-bit roundtrip은 실제 양자화 이후의 효과 유지, ④첫 projection precision은 입력 반올림과 첫 계산 정밀도의 영향을 각각 조사한다. 범위마다 gradient와 clipping도 달라지므로 범위 비교만으로 누적 효과를 유일 원인으로 확정하지 않는다. 이 통제들은 원인 후보를 좁히며 하나의 설명을 자동으로 확정하지 않는다.

## 지표·분모와 불확실성

- **Branch reached**: 실제 응답 전체를 생성한 `free.oracle_branch_reached`. 원래 prefix를 그대로 따르고 해당 위치에서 oracle token을 선택했는지 검사한다. 고정 prefix probe의 top1 전환만으로 성공을 세지 않는다. **분기 token 도달은 저장된 oracle의 전체 경로 재현이나 GT IoU 개선을 뜻하지 않는다.** 이후 token이 달라져 IoU가 낮아질 수 있으므로 실제 생성 구간·GT IoU·ΔIoU를 함께 확인한다.
- **확인 전환률**: 확인된 branch reached 수 / 선정 query 수. Random은 query별 세 seed 성공 비율을 합한다. Baseline 재현 실패는 확인 성공 0으로 포함하되 실패 수를 별도 표시한다. 아직 대기 중인 query를 실패로 분류하지 않는다. 진행 중 값은 현재까지 확인된 전환의 하한이며 완료된 성공률이 아니다.
- **관측 전환률**: 실제 decoding을 평가할 수 있었던 query의 전환률. 실패와 대기를 분모에서 제외하므로 확인 전환률과 함께 읽는다.
- **GT IoU**: length(예측 구간∩GT)/length(예측 구간∪GT). **ΔIoU**는 같은 query의 실제 native baseline greedy IoU를 뺀 값이다. 생성된 응답의 parse 실패가 원자료에서 IoU0으로 정의되어 있으면 그 관측값을 사용한다. 추론 자체가 실패해 응답이 없으면 IoU0을 임의로 채우지 않으며 관측 평균·관측 n·미관측 실패 수를 별도로 표시한다.
- **모집단 가중값**: Σ_d (N_d/805)×dataset d의 평균, N_d∈{277,266,262}. Dataset별 표본 수20의 단순 pooled 평균과 다르다. 확인 전환률의 진행 중 하한은 각 dataset의 선정20 query를 분모로 계산한다. 가중 관측 평균은 세 dataset 모두에 해당 지표의 관측이 있을 때만 표시한다. 가중 관측 IoU는 관측된 응답 평균이며 미관측 실패를 포함한 805개 전체의 결과를 복원하지 않는다.
- **95% Wilson 구간**: binary 조건의 dataset별 확인 전환에 사용한다. n=선정 query 수, p=확인 성공/n, z=1.959964이며 중심=(p+z²/(2n))/(1+z²/n), 반폭=z√(p(1−p)/n+z²/(4n²))/(1+z²/n)이다. 실패를 확인 성공0으로 둔 보수적 프로토콜 지표의 구간이다. 대기 query가 있으면 구간을 계산하지 않는다.
- **Bootstrap 구간**: random query 평균, 모집단 가중 전환률, paired 차이에 사용한다. Dataset별 video cluster를 복원추출하며 같은 영상의 query와 query 안의 모든 seed를 함께 유지한다. 1,000회, seed42, percentile2.5–97.5%다. Targeted−random 및 targeted−opposite는 같은 query 안에서 차이를 구해 재표집한다. IoU 차이는 두 조건 모두 관측된 query를 사용하고 그 n을 명시한다. 미완료 비교의 95% 구간은 표시하지 않는다. 관측값이 모두 같아 empirical bootstrap 구간의 폭이0이어도 모집단 빈도가 확정된 것은 아니다.

## 수치 정밀도 진단

Dataset별 hash 순서 첫 두 query, 총 6개를 결과 전에 지정했다. Native, 첫 Conv3D projection만 FP32로 계산하되 입력을 BF16으로 반올림한 조건, 같은 projection에 FP32 입력을 유지한 조건을 비교한다. 기존 BF16 가중치를 FP32로 올려 사용하며 projection 출력과 이후 모델은 BF16 그대로다. **전체 FP32 모델 실험이 아니다.**

각 mode마다 자체 변형 전 baseline을 구한다. Δoracle margin = r_mode(x′)−r_mode(x), r=z_oracle−max_{v≠oracle}z_v이며 자체 baseline 기준이다. Native와 baseline prefix·선택 token이 일치하는지도 별도 표시한다. Baseline이 다른 mode의 원래 margin을 그대로 같은 기준인 것처럼 비교하지 않는다. 이 6개는 진단 사례이며 정밀도 결과를 805개 전체의 빈도로 해석하지 않는다.

실제 RGB 변경량, native BF16 cast 전후 normalized input 차이, 8-bit roundtrip 뒤 원본 tensor와의 일치를 함께 기록한다. 이 비교는 원인 후보를 좁히기 위한 통제다. 한 조건의 성공이나 실패만으로 의미 이해·수치 정밀도·특정 layer를 유일한 원인으로 단정하지 않는다.
'''


def precision_rows(cases, budget=None):
    rows = []
    for case in cases:
        audit = case['controls'].get('precision_audit') or {}
        for mode in audit.get('modes', []):
            base = mode.get('baseline') or {}
            for condition in mode.get('conditions', []):
                if budget is not None and abs(condition.get('epsilon_255', -1)-budget) > 1e-9:
                    continue
                probe, outcome = condition.get('probe') or {}, condition.get('free') or {}
                rows.append([case['sample']['sample_key'], LABELS.get(mode.get('mode'), mode.get('mode')),
                             fmt(condition.get('epsilon_255'), 2), yes(mode.get('baseline_matches_native_prefix_and_token')),
                             fmt(base.get('oracle_margin')), fmt(probe.get('oracle_margin')), fmt(condition.get('delta_oracle_margin')),
                             yes(outcome.get('oracle_branch_reached')), fmt(outcome.get('iou'))])
    return rows


PRECISION_HEADERS = ['Query', '계산 조건', 'ε₂₅₅', 'Native baseline 일치', '자체 baseline r', '변경 후 r', '자체 기준 Δr', '실제 branch 도달', 'GT IoU']


def markdown(selection, status, cases, summaries, paired, artifacts, output):
    complete = sum(bool(case['controls'].get('complete')) for case in cases)
    failures = sum(bool(case['controls'].get('failed')) for case in cases)
    lines = ['# Visual perturbation controls · pilot60', '',
             f'**Control 확정 {complete}/60 query · 이 중 baseline 재현 실패 {failures} · 상태 `{status.get("state", "queued")}`**', '',
             '같은 RGB 변경 예산에서 targeted 방향의 효과를 반대·random·반복 gradient·8-bit roundtrip과 비교한다. 기존 극단 3사례는 이 60개의 집계에 포함하지 않는다.', '',
             '## 현재 관측의 해석', '', '\n\n'.join(interpretation(cases, summaries, paired)), '',
             definitions(), '## 변경 범위별 native scan · 선정60 query', '', table(SUPPORT_HEADERS, support_rows(cases), True), '']
    for budget in BUDGETS:
        lines += [f'## 공통 예산 ε₂₅₅={budget:g}', '', '### 모집단 가중 요약', '',
                  table(HEADERS, metric_rows(summaries, budget, 'all'), True), '']
        for dataset in DATASETS:
            lines += [f'### {DATASET_LABELS[dataset]} · 선정20 query', '', table(HEADERS, metric_rows(summaries, budget, dataset), True), '']
        contrasts = []
        for dataset in ('all',)+DATASETS:
            for other in ('random', 'opposite'):
                result = paired[budget, other, dataset]
                contrasts.append(['모집단 가중' if dataset == 'all' else DATASET_LABELS[dataset],
                                  'Targeted − '+LABELS[other], f'{result["n"]}/{result["selected"]}',
                                  fmt(result['rate_difference']), str(tuple(round(value, 4) for value in result['rate_interval'])) if result['rate_interval'] else '—',
                                  fmt(result['iou_difference']), result['iou_n'],
                                  str(tuple(round(value, 4) for value in result['iou_interval'])) if result['iou_interval'] else '—'])
        lines += ['### 같은 query의 paired 차이', '', table(['범위', '비교', '확정 pair/선정', '전환률 차이', '95% 구간', '관측 IoU 차이', 'IoU pair n', '95% 구간'], contrasts, True), '']
    lines += ['## 사전 지정 정밀도 진단 · 6 query', '',
              '지정 목록: '+', '.join(selection.get('prespecified_precision_subset', [])), '']
    precision = precision_rows(cases)
    lines += [table(PRECISION_HEADERS, precision, True) if precision else '아직 정밀도 진단 결과가 없다. 미측정을 실패로 해석하지 않는다.', '', '## 원자료와 상태', '']
    for name, label in [('selection.json', '표본·805개 조건부 모집단·hash 선택'), ('config.json', '실행 설정'), ('status.json', '진행·실패 상태')]:
        path = artifacts/'metadata'/name
        if path.exists():
            lines.append(f'- [{label}]({relative(path, output)})')
    lines += [f'- [기존 3사례와 PNG 비교]({relative(PROJECT/"results/rollout/oracle_switch/viewer.html", output)})', '',
              'Query별 원본 참조·control 기록·정밀도 원자료는 결과 화면에서 선택해 연다. 기존 3개 query를 복제하거나 60개와 독립 합산하지 않는다.', '']
    return '\n'.join(lines)


def query_control_html(case, budget):
    rows = []
    sample = case['sample']
    for method in METHODS:
        for seed in SEEDS if method == 'random' else (None,):
            trial = matching_trial(case, method, budget, seed)
            label = f'Random · seed{seed}' if method == 'random' else LABELS[method]
            if trial is None:
                rows.append([label, '재현 실패' if case['controls'].get('failed') else '미측정']+['—']*10)
                continue
            outcome, probe = trial.get('free') or {}, trial.get('probe') or {}
            cast, rounding, rgb = trial.get('precision_input') or {}, trial.get('roundtrip') or {}, trial.get('rgb') or {}
            score = outcome.get('iou')
            base = baseline_iou(case)
            delta = score-base if finite(score) and base is not None else None
            rows.append([label, yes(outcome.get('oracle_branch_reached')), fmt(score), fmt(delta), span(outcome),
                         yes(probe.get('branch_reached')), fmt(probe.get('oracle_margin')),
                         fmt(rgb.get('actual_rgb_linf')), percent(cast.get('precast_changed_fraction')),
                         percent(cast.get('postcast_changed_fraction')), yes(cast.get('postcast_equal')),
                         yes(rounding.get('normalized_tensor_identical_to_baseline'))])
    return table(['입력 변경 조건', '실제 branch 도달', 'GT IoU', 'ΔIoU', '생성 구간', '고정-prefix probe 도달', 'Oracle margin', '실제 RGB L∞', 'Cast 전 변경 비율', 'Cast 후 변경 비율', 'Cast 후 동일', '8-bit 후 원 tensor 동일'], rows, classes='query-controls')


def case_html(case, artifacts, output):
    sample, controls = case['sample'], case['controls']
    key, row = sample['sample_key'], sample.get('row', {})
    gt = row.get('span', [])
    gt = gt[0] if gt and isinstance(gt[0], list) else gt
    video = relative(sample.get('video_path'), output)
    links = []
    for path, label in [(artifacts/'queries'/f'{key}.json', 'Native scan 원자료'),
                        (artifacts/'analyses/controls'/f'{key}.json', 'Control 원자료')]:
        if path.exists():
            links.append(f'<a href="{esc(relative(path, output))}" target="_blank" rel="noopener">{label} ↗</a>')
    for name, value in sample.get('source', {}).items():
        path = value.get('path') if isinstance(value, dict) else value
        if path:
            links.append(f'<a href="{esc(relative(path, output))}" target="_blank" rel="noopener">Source {esc(name)} ↗</a>')
    failure = f'<p class="notice">Baseline 재현 실패: {esc(controls.get("error"))}. 선정 분모에 남기며 관측하지 못한 IoU를 0으로 채우지 않습니다.</p>' if controls.get('failed') else ''
    panels = ''.join(f'<div class="query-budget" data-budget="{budget:g}">{query_control_html(case, budget)}</div>' for budget in BUDGETS)
    precision = ''.join(f'<div class="precision-budget" data-budget="{budget:g}">{table(PRECISION_HEADERS[1:], [line[1:] for line in precision_rows([case], budget)], classes="precision-table")}</div>' for budget in BUDGETS)
    has_precision = bool((controls.get('precision_audit') or {}).get('modes'))
    return f'''<article class="query-case" data-key="{esc(key)}" data-dataset="{esc(sample['dataset'])}" data-video="{esc(video)}" hidden>
<h3>{esc(row.get('query', key))}</h3><p class="small">{esc(key)} · {esc(DATASET_LABELS.get(sample['dataset'], sample['dataset']))} · {fmt(row.get('duration'), 2)} s</p>
<div class="query-overview"><span>GT <b>{esc(span(gt))}</b></span><span>Native greedy <b>{esc(span((controls.get('baseline_free') or sample.get('greedy'))))}</b> · IoU {fmt(baseline_iou(case))}</span><span>저장된 oracle <b>{esc(span(sample.get('oracle')))}</b> · IoU {fmt(sample.get('oracle', {}).get('iou'))}</span></div>{failure}
<p class="small">모든 행은 같은 greedy decoding입니다. Random seed는 개별 표시하지만 주 집계에서는 query 안에서 평균합니다. 고정-prefix probe는 실제 응답 생성과 별도입니다.</p>{panels}
{'<details class="precision-detail"><summary>이 query의 정밀도 진단 · 자체 baseline 기준</summary><p class="small">첫 projection만 FP32이며 전체 FP32 모델이 아닙니다.</p>'+precision+'</details>' if has_precision else ''}
<details><summary>원자료·기존 source 참조</summary><p class="source-links">{' '.join(links)}</p><pre>{esc(json.dumps({'validation':controls.get('validation'),'cost':controls.get('cost'),'precision_selected':(controls.get('precision_audit') or {}).get('selected')},ensure_ascii=False,indent=2))}</pre></details></article>'''


STYLE = '''
main{max-width:1600px;margin:auto;padding:24px}h1,h2,h3,p{margin-top:0}.page-header{margin:-24px -24px 24px}.small{font-size:12px;line-height:1.8;color:var(--ui-muted)}.eyebrow{font-size:11px;letter-spacing:.08em;color:var(--ui-accent);font-weight:700}.panel{border:1px solid var(--ui-line);border-radius:12px;background:var(--ui-surface);padding:22px;margin:18px 0;min-width:0}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.summary>div{background:var(--ui-surface);border:1px solid var(--ui-line);border-radius:9px;padding:14px}.summary b{display:block;font-size:24px}.summary small{color:var(--ui-muted);font-size:11px}.notice{border-left:3px solid #b88c42;background:#faf7ef;padding:12px 15px;font-size:13px;line-height:1.8;margin:18px 0}.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:18px}.toolbar label{font-size:12px;display:flex;align-items:center;gap:8px}.toolbar select{padding:7px;max-width:100%}.table-wrap{overflow:auto;margin:12px 0}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px;text-align:left;vertical-align:top;border-bottom:1px solid var(--ui-line)}th{font-size:11px;white-space:nowrap}.panel h2{font-size:21px}.table-wrap td{font-variant-numeric:tabular-nums}.section-heading{display:flex;gap:14px;align-items:center;justify-content:space-between}.section-heading h2{margin-bottom:0}#query-select{max-width:100%;width:520px;padding:8px}.query-media{margin:14px 0}#query-video{display:block;width:100%;max-height:270px;background:#14221d;border-radius:8px}.query-overview{display:flex;flex-wrap:wrap;gap:12px 24px;font-size:12px;padding:12px;background:var(--ui-soft);border-radius:8px}.query-controls td{min-width:55px}.query-controls td:first-child{min-width:130px}details{margin-top:16px}summary{cursor:pointer;font-weight:600;font-size:13px}.source-links a{display:inline-block;margin:5px 14px 5px 0;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;background:#f7f9f6;padding:12px;max-height:260px;overflow:auto}.empty{padding:18px;background:var(--ui-soft);font-size:13px;border-radius:8px}body.embedded .page-header{display:none}.footer{font-size:12px;color:var(--ui-muted)}
@media(max-width:720px){main{padding:14px}.page-header{margin:-14px -14px 16px}.panel{padding:15px}.summary{grid-template-columns:repeat(2,1fr);gap:8px}.summary>div{padding:10px}.summary b{font-size:21px}.toolbar{gap:10px}.section-heading{display:block}#query-select{width:100%}.section-heading h2{margin-bottom:12px}table{font-size:11px}.query-overview{display:block}.query-overview>span{display:block;margin:4px 0}}
'''

SCRIPT = '''
(() => {
 const $=id=>document.getElementById(id);
 const cases=[...document.querySelectorAll('.query-case')];
 const originalOptions=[...$('query-select').options].map(option=>option.cloneNode(true));
 if(window.self!==window.top)document.body.classList.add('embedded');
 function showQuery(key,update=true){
  const selected=cases.find(item=>item.dataset.key===key)||cases.find(item=>$('dataset-filter').value==='all'||item.dataset.dataset===$('dataset-filter').value);
  if(!selected)return;
  cases.forEach(item=>item.hidden=item!==selected);
  $('query-select').value=selected.dataset.key;
  const video=$('query-video');
  if(video.dataset.sample!==selected.dataset.key){video.pause();video.dataset.sample=selected.dataset.key;if(selected.dataset.video){video.src=selected.dataset.video;}else{video.removeAttribute('src');video.load();}}
  if(update){const url=new URL(location.href);url.hash=new URLSearchParams({sample:selected.dataset.key});history.replaceState(null,'',url);window.dispatchEvent(new Event('rollout:context'));}
 }
 function applyFilters(){
  const dataset=$('dataset-filter').value,budget=$('budget-filter').value;
  document.querySelectorAll('.summary-slice').forEach(panel=>panel.hidden=panel.dataset.dataset!==dataset||panel.dataset.budget!==budget);
  document.querySelectorAll('.support-slice').forEach(panel=>panel.hidden=panel.dataset.dataset!==dataset);
  document.querySelectorAll('.query-budget,.precision-budget').forEach(panel=>panel.hidden=panel.dataset.budget!==budget);
  const selected=$('query-select').value;
  $('query-select').replaceChildren(...originalOptions.filter(option=>dataset==='all'||option.dataset.dataset===dataset).map(option=>option.cloneNode(true)));
  showQuery([...$('query-select').options].some(option=>option.value===selected)?selected:$('query-select').value);
 }
 $('dataset-filter').onchange=applyFilters;$('budget-filter').onchange=applyFilters;
 $('query-select').onchange=()=>showQuery($('query-select').value);
 window.addEventListener('hashchange',()=>{const key=new URLSearchParams(location.hash.slice(1)).get('sample');if(cases.some(item=>item.dataset.key===key)){$('dataset-filter').value='all';applyFilters();showQuery(key);}});
 const initial=new URLSearchParams(location.hash.slice(1)).get('sample');
 applyFilters();if(initial)showQuery(initial);
})();
'''


def html_document(selection, status, cases, summaries, paired, artifacts, output):
    completed = sum(bool(case['controls'].get('complete')) for case in cases)
    failed = sum(bool(case['controls'].get('failed')) for case in cases)
    native = sum(bool(case['native'].get('complete')) for case in cases)
    query_options = ''.join(f'<option value="{esc(case["sample"]["sample_key"])}" data-dataset="{esc(case["sample"]["dataset"])}">{esc(DATASET_LABELS[case["sample"]["dataset"]])} · {esc(case["sample"]["sample_key"])} · {esc(case["sample"].get("row",{}).get("query",""))}</option>' for case in cases)
    slices = []
    for budget in BUDGETS:
        for dataset in ('all',)+DATASETS:
            title = '805개 조건부 모집단 가중' if dataset == 'all' else DATASET_LABELS[dataset]+' · 선정20 query'
            contrast_rows = []
            for other in ('random', 'opposite'):
                value = paired[budget, other, dataset]
                rate_interval = ' … '.join(fmt(x) for x in value['rate_interval']) if value['rate_interval'] else '—'
                iou_interval = ' … '.join(fmt(x) for x in value['iou_interval']) if value['iou_interval'] else '—'
                contrast_rows.append(['Targeted − '+LABELS[other], f'{value["n"]}/{value["selected"]}', fmt(value['rate_difference']), rate_interval,fmt(value['iou_difference']),value['iou_n'],iou_interval])
            slices.append(f'<div class="summary-slice" data-dataset="{dataset}" data-budget="{budget:g}" hidden><h3>{esc(title)} · ε₂₅₅={budget:g}</h3>{table(HEADERS, metric_rows(summaries,budget,dataset))}<details><summary>같은 query의 paired 차이</summary>{table(["비교","확정/선정 pair","전환률 차이","95% 구간","관측 IoU 차이","IoU pair n","95% 구간"],contrast_rows)}</details></div>')
    old_view = relative(PROJECT/'results/rollout/oracle_switch/viewer.html', output)
    state = status.get('state', 'queued')
    precision = precision_rows(cases)
    precision_section = table(PRECISION_HEADERS, precision, classes='precision-table') if precision else '<p class="empty">사전 지정 6 query의 정밀도 진단은 아직 없습니다.</p>'
    state_note = f'<p class="notice">실행 상태: {esc(state)}. {esc(status.get("error", ""))}</p>' if state not in ('complete','completed') else ''
    support_sections = ''.join(f'<div class="support-slice" data-dataset="{dataset}" hidden>{table(SUPPORT_HEADERS,support_rows(cases,dataset))}</div>' for dataset in ('all',)+DATASETS)
    interpretation_html = ''.join(f'<p class="small">{esc(line)}</p>' for line in interpretation(cases, summaries, paired))
    return f'''<!doctype html><html lang="ko" data-experiment-detail data-family-default><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual perturbation controls · pilot60</title><style>{STYLE}</style><link rel="stylesheet" href="{esc(relative(PROJECT.parent/'_lib/experiment_theme.css',output))}"></head><body><main>
<header class="page-header"><p class="eyebrow">ROLLOUT / ORACLE SWITCH CONTROLS / PILOT 60</p><h1>같은 영상 변경량에서 무엇이 재현되는가</h1><p>Targeted 방향을 반대·random·반복 gradient·8-bit roundtrip과 비교합니다.</p></header>
<div id="study-summary" class="summary"><div><b>{completed} / 60</b><small>Control 확정 query</small></div><div><b>{native} / 60</b><small>Native scan 완료 query</small></div><div><b>{failed}</b><small>확정 query 중 재현 실패</small></div><div><b>805 → 60</b><small>조건부 모집단 → 선정 query</small></div></div>{state_note}
<p class="notice">Dataset별 고정 hash 순서로 20 query씩 골랐습니다. 원래 greedy보다 좋은 oracle과 top30 분기가 있는 805개만의 조건부 모집단입니다. 기존 극단 3사례와 합산하지 않습니다.</p>
<section class="panel" id="interpretation"><h2>현재 관측의 해석</h2>{interpretation_html}</section>
<section class="panel"><div class="section-heading"><h2>공통 예산에서의 결과</h2><div class="toolbar"><label>Dataset<select id="dataset-filter"><option value="all">모집단 가중 · 전체</option>{''.join(f'<option value="{name}">{DATASET_LABELS[name]}</option>' for name in DATASETS)}</select></label><label>변경량<select id="budget-filter"><option value="0.25">0.25 /255</option><option value="1">1 /255</option></select></label></div></div>
<p class="small">실제 응답의 oracle branch 도달을 셉니다. <strong>분기 token 도달은 oracle 전체 경로 재현이나 GT IoU 개선을 뜻하지 않습니다.</strong> 확인 전환률은 선정 query 분모이며 실패는 확인 성공0, 대기는 아직 미측정입니다. 진행 중 값은 확인된 전환의 하한입니다. IoU는 관측 응답만 평균하며 실패를0으로 채우지 않습니다.</p>
<div id="condition-summary">{''.join(slices)}</div><p class="small">Random은 seed42·43·44를 query 안에서 평균합니다. Dataset binary 조건은95% Wilson, random·모집단 가중·paired 차이는 dataset별 video-cluster bootstrap1,000회(seed42)입니다. 대기가 남은 비교는 구간을 표시하지 않습니다.</p></section>
<section class="panel"><h2>변경 범위별로 관측한 경계</h2>{support_sections}<p class="small">Native scan의 검증 upper ε₂₅₅ 중앙값은 경계 도달이 확인된 사례만 사용합니다. 최대32/255까지의 선택한 gradient ray 탐색이며 전역 최소 변경량이 아닙니다. 전체 영상·pair/cell 비교는 변경 범위와 누적 효과를, targeted−random은 방향성을, 8-bit와 projection precision은 양자화·계산 정밀도를 각각 조사합니다. 하나의 원인으로 확정하지 않습니다.</p></section>
<section class="panel" id="query-section"><div class="section-heading"><h2>Query별 조건 비교</h2><select id="query-select" aria-label="Query 선택">{query_options}</select></div><div class="query-media"><video id="query-video" controls muted preload="metadata" aria-label="선택 query의 원본 영상"></video></div>{''.join(case_html(case,artifacts,output) for case in cases)}</section>
<section class="panel"><h2>첫 projection 정밀도 진단 · 사전 지정6 query</h2><p class="small">모든 mode의 자체 baseline을 따로 측정합니다. 첫 Conv3D projection만 FP32이며 기존 BF16 가중치를 upcast합니다. 이후 계산은BF16이므로 전체 FP32 모델 실험이 아닙니다. Native baseline 일치 여부와 자체 기준 Δoracle margin을 함께 봅니다.</p>{precision_section}</section>
<p class="footer"><a href="{esc(old_view)}" target="_blank" rel="noopener">기존 극단3사례와 PNG 비교 ↗</a> · 지표·모집단·분모·한계는 대표 viewer의 보고서 탭에서 확인하세요.</p>
</main><script>{SCRIPT}</script></body></html>'''


def generate(artifacts=None, output=None):
    artifacts, output = Path(artifacts or ARTIFACTS), Path(output or OUTPUT)
    selection = read(artifacts/'metadata/selection.json')
    if not selection or len(selection.get('samples', [])) != 60:
        raise ValueError('The fixed 60-query selection is required before publishing this study.')
    population = selection.get('counts', {}).get('population_by_dataset', POPULATION)
    if sum(population.values()) != 805:
        raise ValueError('Unexpected target population; update report definitions before publishing.')
    cases = load_cases(selection, artifacts)
    status = read(artifacts/'metadata/status.json', {'state':'queued'})
    summaries, paired = comparisons(cases, population)
    write_text(output/'REPORT.md', markdown(selection,status,cases,summaries,paired,artifacts,output))
    write_text(output/'detail.html', html_document(selection,status,cases,summaries,paired,artifacts,output))
    return dict(scope='pilot',queries=60,controls_complete=sum(bool(case['controls'].get('complete')) for case in cases),
                failed=sum(bool(case['controls'].get('failed')) for case in cases),
                report=str((output/'REPORT.md').resolve()),detail=str((output/'detail.html').resolve()),
                generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(generate(args.artifacts,args.output),ensure_ascii=False))
