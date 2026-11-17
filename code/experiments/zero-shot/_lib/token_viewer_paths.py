"""Lazy path metadata for Token uncertainty; original distributions stay canonical.

The public lookup contains only references to existing rollout records/indexes
and derived endpoint labels. The probability arrays and token sequences are
never copied into this adapter's outputs. Public saved-path views use the
original fixed pilot300/600/900 cohorts; full data supply sample metadata only.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/home/irteam/experiments/zero-shot')
SOURCE = ROOT / 'artifacts/runs/rollout/deterministic/pilot/900'
FAMILY = ROOT / 'artifacts/runs/uncertainty/token_analysis'
OVERVIEW = FAMILY / 'overview'
OUT = OVERVIEW / 'web/paths'
LOOKUP = OUT / 'lookup.json'
PROVENANCE = OVERVIEW / 'metadata/token_viewer_paths.json'
VALIDATION = OVERVIEW / 'validation/token_viewer_paths.json'
CLASSIFIER = ROOT / '_lib/timelens_full_digit_impl.py'
SCOPES = ('pilot/300', 'full/9404')
STAGES = (300, 600, 900)
COHORT_ROOT = ROOT / 'artifacts/common/cohorts/nested_pilot_v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
    if path.exists() and path.read_text() == text:
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(value):
    # Canonical record paths also occur with the workspace's /home1 alias.
    return Path(value.replace('/home1/irteam/', '/home/irteam/'))


def manifest(scope):
    path = FAMILY / scope / 'digit_correctness/web/manifest.js'
    text = path.read_text()
    rows, _ = json.JSONDecoder().raw_decode(text, text.index('=') + 1)
    return path, rows


def entropy(node):
    values = [row['probability'] for row in node['top']]
    assert len(values) == 30 and all(math.isfinite(p) and p >= 0 for p in values)
    mass = sum(values)
    assert mass > 0
    h = -sum(p / mass * math.log(p / mass) for p in values if p > 0)
    assert -1e-12 <= h <= math.log(30) + 1e-10
    return h


def interval_iou(gt, span):
    if not span or span[0] >= span[1]:
        return 0.0
    union = max(gt[1], span[1]) - min(gt[0], span[0])
    return max(0, min(gt[1], span[1]) - max(gt[0], span[0])) / union if union > 0 else 0.0


def forced_positions(candidate, candidates):
    """Actual forced choices inherited along the saved ancestor chain, 1-based."""
    positions = []
    current = candidate
    visited = set()
    while current['branch']:
        assert current['index'] not in visited
        visited.add(current['index'])
        branch = current['branch']
        position = branch['position']
        assert candidate['ids'][position] == branch['forced_token']
        positions.append(position + 1)
        parent = candidates[branch['parent']]
        assert current['ids'][:position] == parent['ids'][:position]
        current = parent
    return sorted(set(positions))


def selection_kind(step, prefix_length, forced_steps):
    if step in forced_steps:
        return 'forced'
    return 'conditioned' if step <= prefix_length else 'free'


def endpoint_metrics(steps, node_ids, entropies):
    error = next((s for s in steps if s['first_error']), None)
    numeric = [s for s in steps if s['numeric']]
    risk = [s for s in numeric if s['previous_correct']]
    prior = [s for s in numeric if error and s['ar_step'] < error['ar_step'] and s['prefix_correct']]
    paired = bool(error and error['numeric'] and prior)
    previous = prior[-1] if paired else None
    error_h = error['entropy'] if error else None
    error_step = error['ar_step'] if error else None
    previous_token_h = entropies[node_ids[error_step - 2]] if error_step and error_step > 1 else None
    kind = 'none' if error is None else 'numeric' if error['numeric'] else 'format'
    return dict(
        paired=paired, error_kind=kind, error_role=error['role'] if error else None,
        error_region='none' if not error else 'format' if not error['numeric'] else 'fractional' if error['role'] == 'tenths' else 'integer',
        error_step=error_step, error_entropy=error_h if paired else None,
        token_error_entropy=error_h, previous_step=previous['ar_step'] if previous else None,
        previous_entropy=previous['entropy'] if previous else None,
        delta_previous=error_h - previous['entropy'] if previous else None,
        delta_prior_mean=error_h - sum(s['entropy'] for s in prior) / len(prior) if paired else None,
        risk_mean_entropy=sum(s['entropy'] for s in risk) / len(risk) if risk else None,
        response_risk_mean_entropy=None, prior_safe_steps=len(prior),
        previous_token_step=error_step - 1 if previous_token_h is not None else None,
        previous_token_entropy=previous_token_h,
        delta_previous_token=error_h - previous_token_h if previous_token_h is not None else None,
        safe_high=any(s['numeric'] and s['prefix_correct'] and s['entropy'] >= 1.5 for s in steps),
        next_failure=bool(paired), min_safe=None, max_unsafe=None,
        error_selection_kind=error['selection_kind'] if error else None,
        error_is_model_generated=bool(error and error['selection_kind'] == 'free'),
        free_error_step=error_step if error and error['selection_kind'] == 'free' else None,
    )


def candidate_metadata(candidate, candidates, index, sample, entropies, classify):
    identifiers = index['candidates'][str(candidate['index'])]['nodes']
    assert len(identifiers) == len(candidate['ids'])
    pieces = []
    for j, (chosen, key) in enumerate(zip(candidate['ids'], identifiers)):
        node = index['nodes'][key]
        assert node['prefix_ids'] == candidate['ids'][:j]
        pieces.append(next(t['token'] for t in node['observed'] if t['id'] == chosen))
    forced = len(candidate['branch']['forced_prefix_ids']) if candidate['branch'] else 0
    actual_forced = forced_positions(candidate, candidates)
    parsed = candidate['parsed']
    predicted = [parsed['start'], parsed['end']] if parsed and parsed['positive'] and candidate['stopped_on_eos'] else None
    item = dict(
        id=candidate['index'], parent=candidate['branch']['parent'] if candidate['branch'] else None,
        branch_step=candidate['branch']['position'] + 1 if candidate['branch'] else None,
        branch_roles=candidate['branch']['roles'] if candidate['branch'] else [],
        forced_prefix_length=forced, forced_steps=actual_forced, trace_step_count=len(identifiers),
        iou=interval_iou(sample['span'][0], predicted), supported=False, endpoints={},
    )
    text = ''.join(pieces)
    assert text.startswith(candidate['answer'])
    match = re.search(r'(?<![\d:])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d:])', candidate['answer'])
    if ':' in candidate['answer'] or not match or any(not re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9])?', value) for value in match.groups()):
        return item
    for n, endpoint in ((1, 'start'), (2, 'end')):
        a, b = match.span(n)
        offset = 0
        states = []
        for j, piece in enumerate(pieces):
            limit = offset + len(piece)
            if limit > a and offset <= b:
                states.append(dict(ar_step=j + 1, endpoint_step=len(states) + 1, token=piece,
                                   prefix=text[a:min(limit, b)], closed=limit > b,
                                   entropy=entropies[identifiers[j]]))
            offset = limit
        if not states or not states[-1]['closed']:
            item['endpoints'] = {}
            return item
        target, classified, correct = classify(match.group(n), sample['span'][0][n - 1], states)
        keep = ('ar_step', 'endpoint_step', 'prefix', 'closed', 'entropy', 'numeric', 'role',
                'first_error', 'prefix_correct', 'previous_correct', 'digit_ordinal', 'condition')
        steps = []
        for state in classified:
            row = {key: state[key] for key in keep}
            row['selection_kind'] = selection_kind(row['ar_step'], forced, actual_forced)
            steps.append(row)
        target_span = list(sample['span'][0])
        target_span[n - 1] = round(float(match.group(n)))
        item['endpoints'][endpoint] = dict(
            endpoint=endpoint, prediction=match.group(n), target=target, all_correct=correct,
            gt_digits=len(str(int(sample['span'][0][n - 1]))), generated_digits=len(match.group(n).split('.')[0]),
            endpoint_iou=interval_iou(sample['span'][0], target_span), steps=steps,
            metrics=endpoint_metrics(steps, identifiers, entropies),
        )
    response_risk = [s['entropy'] for ep in item['endpoints'].values() for s in ep['steps'] if s['numeric'] and s['previous_correct']]
    response_mean = sum(response_risk) / len(response_risk) if response_risk else None
    for ep in item['endpoints'].values():
        ep['metrics']['response_risk_mean_entropy'] = response_mean
    item['supported'] = True
    return item


def build(worker=False):
    references_path = SOURCE / 'metadata/source_records.json'
    references = json.loads(references_path.read_text())
    assert len(references) == 900
    scope_manifests = {scope: manifest(scope) for scope in SCOPES}
    inputs = [references_path, CLASSIFIER] + [item[0] for item in scope_manifests.values()]
    cohorts = {}
    for n in STAGES:
        cohort_path = COHORT_ROOT / f'pilot_{n}/samples.json'
        cohorts[n] = json.loads(cohort_path.read_text())
        inputs.append(cohort_path)
        assert len(cohorts[n]) == n
    assert all(cohorts[900][:n] == cohorts[n] for n in STAGES)
    samples = {}
    for reference in references:
        record = source_path(reference['path'])
        key = record.stem
        index = SOURCE / 'token_traces' / key / 'index.json'
        inputs.extend((record, index))
        samples[key] = dict(record=os.path.relpath(record, OUT), index=os.path.relpath(index, OUT),
                            metadata='samples/' + key + '.json')
    stamp = {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns] for p in inputs}
    producer_hash = sha(Path(__file__))
    if LOOKUP.exists() and PROVENANCE.exists():
        previous = json.loads(PROVENANCE.read_text())
        if previous.get('source_stats') == stamp and previous.get('generator_sha256') == producer_hash and all((OUT / s['metadata']).exists() for s in samples.values()):
            return LOOKUP
    try:
        from timelens_full_digit_impl import classify
    except ModuleNotFoundError:
        if worker:
            raise
        # build_index runs in a lightweight Python environment; reuse the
        # existing scientific environment only when derived labels need updating.
        subprocess.run([str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()), '--worker'], check=True)
        assert LOOKUP.exists()
        return LOOKUP
    counts = Counter()
    sources = []
    for ordinal, reference in enumerate(references):
        record_path = source_path(reference['path'])
        record_bytes = record_path.read_bytes()
        assert hashlib.sha256(record_bytes).hexdigest() == reference['sha256']
        record = json.loads(record_bytes)
        sample = record['sample']
        key = sample['sample_key']
        index_path = SOURCE / 'token_traces' / key / 'index.json'
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
        assert index['sample_key'] == key and index['raw_record_sha256'] == reference['sha256']
        assert index['input_hashes'] == record['input_hashes'] and index['validation']['passed']
        hs = {name: entropy(node) for name, node in index['nodes'].items()}
        candidates = {c['index']: c for c in record['result']['candidates']}
        derived = {}
        for candidate in candidates.values():
            data = candidate_metadata(candidate, candidates, index, sample, hs, classify)
            derived[str(candidate['index'])] = data
            counts['candidates'] += 1
            counts['supported_candidates'] += int(data['supported'])
            counts['unsupported_candidates'] += int(not data['supported'])
            counts['steps'] += data['trace_step_count']
            counts['forced_markers'] += len(data['forced_steps'])
            if candidate['branch']:
                parent = candidates[candidate['branch']['parent']]
                j = candidate['branch']['position']
                assert index['candidates'][str(candidate['index'])]['nodes'][j] == index['candidates'][str(parent['index'])]['nodes'][j]
                counts['same_prechoice_prefix_checks'] += 1
            for ep in data['endpoints'].values():
                failures = [s for s in ep['steps'] if s['first_error']]
                assert len(failures) <= 1
                assert ep['all_correct'] == (not failures)
                if failures:
                    counts['first_errors_' + failures[0]['selection_kind']] += 1
        samples[key]['paths'] = len(derived)
        samples[key]['nodes'] = len(hs)
        save(OUT / samples[key]['metadata'], dict(sample_key=key, source='deterministic', candidates=derived))
        sources.append(dict(sample_key=key, record_sha256=reference['sha256'], index_sha256=hashlib.sha256(index_bytes).hexdigest()))
        counts['queries'] += 1
        counts['unique_prefixes'] += len(hs)
        if (ordinal + 1) % 100 == 0:
            print(f'decoding path metadata: {ordinal + 1}/900 queries', flush=True)
    source_audit = {}
    for scope, (_, rows) in scope_manifests.items():
        available = {r['sample_key'] for r in rows}
        source_audit[scope] = dict(queries=len(rows), overlapping_saved_queries=len(available & samples.keys()))
    assert source_audit['full/9404']['overlapping_saved_queries'] == 900
    assert [s['sample_key'] for s in scope_manifests['pilot/300'][1]] == [s['sample_key'] for s in cohorts[300]]
    coverage = {}
    for n, cohort in cohorts.items():
        keys = [s['sample_key'] for s in cohort]
        assert all(key in samples for key in keys)
        coverage[f'pilot/{n}'] = dict(queries=n, saved_path_queries=n, unavailable_queries=0,
                                      paths=sum(samples[k]['paths'] for k in keys),
                                      nodes=sum(samples[k]['nodes'] for k in keys),
                                      cohort_sha256=sha(COHORT_ROOT / f'pilot_{n}/samples.json'))
    lookup = dict(
        source=dict(id='deterministic', model='qwen3', scope='pilot/900', queries=900,
                    label='deterministic rollout · Qwen3-VL-8B-Instruct',
                    input_description='2 FPS · spatial target 100,352 pixels · video_query input order. 기존 Token uncertainty greedy와 입력 조건이 다릅니다.',
                    distribution='raw top-30 재정규화 entropy · 저장된 전체 vocabulary 분포의 top-30'),
        samples=samples, counts=dict(counts), coverage=coverage,
    )
    save(LOOKUP, lookup)
    assert all([p.stat().st_size, p.stat().st_mtime_ns] == stamp[str(p.relative_to(ROOT))] for p in inputs)
    save(PROVENANCE, dict(generator_sha256=producer_hash, classifier_sha256=sha(CLASSIFIER),
                          source_stats=stamp, source_sha256=sources,
                          sample_metadata_source_audit=source_audit,
                          definition='Native raw H30; original record/index references; rounded endpoint labels; conditioned/actual forced/free selection provenance.'))
    save(VALIDATION, dict(passed=True, source_unchanged=True, original_record_sha256_verified=True,
                         input_hashes_identical=True, canonical_classifier_reused=True,
                         probabilities_not_copied=True, step_prefixes_verified=True,
                         coverage=coverage, counts=dict(counts)))
    return LOOKUP


def build_stage(scope, lookup=None):
    """Reference one fixed pilot cohort without copying distributions or video.

    pilot300 retains its original two-model catalog and manifest. The additional
    pilot600/900 views use the deterministic run's greedy candidate for sidebar
    filters; selected-path metadata still comes from the shared lazy lookup.
    Media references are relative to results/<scope>/digit_correctness/, just
    like the original manifest and the publisher's media_base setting.
    """
    assert scope in {f'pilot/{n}' for n in STAGES}, scope
    lookup_path = Path(lookup) if lookup else build()
    lookup_data = json.loads(lookup_path.read_text())
    n = int(scope.split('/')[1])
    cohort_path = COHORT_ROOT / f'pilot_{n}/samples.json'
    cohort = json.loads(cohort_path.read_text())
    summary = lookup_data['coverage'][scope]
    assert summary['queries'] == len(cohort) == n
    result = dict(queries=n, paths=summary['paths'], nodes=summary['nodes'],
                  cohort_sha256=sha(cohort_path), source='deterministic')
    out = FAMILY / scope
    if n == 300:
        result.update(manifest=out / 'digit_correctness/web/manifest.js',
                      catalog=out / 'metrics/viewer_catalog.json')
        assert result['manifest'].exists() and result['catalog'].exists()
        return result

    manifest_path = out / 'web/manifest.js'
    catalog_path = out / 'metrics/viewer_catalog.json'
    provenance_path = out / 'metadata/viewer_stage_sources.json'
    full_manifest_path, full_rows = manifest('full/9404')
    inputs = (cohort_path, lookup_path, PROVENANCE, full_manifest_path, source_path(str(Path(__file__).resolve())))
    stamp = {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns] for p in inputs}
    result.update(manifest=manifest_path, catalog=catalog_path)
    if manifest_path.exists() and catalog_path.exists() and provenance_path.exists():
        if json.loads(provenance_path.read_text()).get('source_stats') == stamp:
            return result
    by_key = {s['sample_key']: s for s in full_rows}
    rows = []
    catalog = {}
    for original in cohort:
        key = original['sample_key']
        reference = lookup_data['samples'][key]
        metadata = json.loads((lookup_path.parent / reference['metadata']).read_text())
        greedy = metadata['candidates']['1']
        row = dict(by_key[key])
        assert all(row[k] == original[k] for k in original)
        source_base = ROOT / 'results/uncertainty/token_analysis/full/9404/digit_correctness'
        target_base = ROOT / 'results/uncertainty/token_analysis' / scope / 'digit_correctness'
        # Normalize URL segments lexically: resolving the workspace symlink
        # would change /home to /home1 and produce a path outside the web root.
        row['media'] = os.path.relpath(os.path.normpath(source_base / row['media']), target_base)
        row['models'] = dict(qwen3=dict(
            iou=greedy['iou'], supported=greedy['supported'],
            start_correct=greedy['endpoints'].get('start', {}).get('all_correct'),
            end_correct=greedy['endpoints'].get('end', {}).get('all_correct'),
        ))
        rows.append(row)
        catalog['qwen3/' + key] = dict(
            endpoints={ep: data['metrics'] for ep, data in greedy['endpoints'].items()},
            blue_ranges=[],
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = 'window.MANIFEST=' + json.dumps(rows, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + ';\n'
    if not manifest_path.exists() or manifest_path.read_text() != payload:
        temp = manifest_path.with_suffix('.js.tmp')
        temp.write_text(payload)
        temp.replace(manifest_path)
    save(catalog_path, catalog)
    save(provenance_path, dict(
        source_stats=stamp, cohort_path=str(cohort_path.relative_to(ROOT)),
        cohort_sha256=result['cohort_sha256'], fixed_cohort_order=True,
        queries=n, paths=result['paths'], nodes=result['nodes'],
        filter_source='deterministic greedy candidate 1',
        metadata_source=str(full_manifest_path.relative_to(ROOT)),
        source_lookup=str(lookup_path.relative_to(ROOT)),
        probabilities_copied=False, video_copied=False,
    ))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    print(build(worker=args.worker), flush=True)
