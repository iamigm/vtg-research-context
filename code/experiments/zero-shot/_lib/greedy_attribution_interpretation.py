"""Read-only branch interpretation derived from query and source records.

No arrays, model execution, filesystem access, or changes to raw records are
needed. A matching descendant is not evidence of a direct conditional greedy
completion: its last forced token may occur later than the branch being shown.
"""
from __future__ import annotations

import math
from collections import Counter


READING_GUIDE = {
    'probability': '같은 prefix에서 다음 token을 선택하는 native probability다. 완성 구간 전체의 probability가 아니다.',
    'direct_continuation': '표시한 token 하나를 대안으로 고정한 뒤 나머지를 greedy로 완성한 기존 결과다.',
    'descendant': '같은 분기를 공유하지만 이후 token도 추가로 바꾼 경로다. 표시한 분기 하나의 직접 결과와 구분한다.',
    'signed_scaling': 'Margin은 greedy−alternative다. 표현을 미소 확대할 때 양수는 greedy 쪽, 음수는 alternative 쪽으로 기운다.',
    'sensitivity': 'Gradient norm은 작은 변화에 대한 민감도다. 부호나 인과적 근거량을 뜻하지 않는다.',
    'cancellation': 'Signed 합이 작아도 위치별 양수·음수 값이 상쇄되었을 수 있다. 절댓값 합과 함께 읽는다.',
    'layers': 'Attention과 MLP는 내부 연산 종류다. 영상과 언어를 각각 뜻하지 않으며 서로 다른 층의 점수를 합산하지 않는다.',
    'screening': '지정한 표현 배율 변화만 검사했다. Pairwise margin 역전은 전체 vocabulary의 top1 역전이나 최종 mode 전환을 확정하지 않는다.',
}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _span(value):
    if isinstance(value, dict):
        if value.get('positive') is False:
            return None
        value = [value.get('start'), value.get('end')]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = map(_number, value)
    return [start, end] if start is not None and end is not None and end > start else None


def _direction(value):
    value = _number(value)
    return None if value is None else 'greedy' if value > 0 else 'alternative' if value < 0 else 'zero'


def _source_context(query_record, source_record):
    candidates = source_record.get('result', {}).get('candidates', []) if source_record else []
    source_greedy = candidates[0] if candidates else None
    greedy_ids = query_record.get('greedy', {}).get('ids', [])
    if source_greedy is not None and source_greedy.get('ids') != greedy_ids:
        raise ValueError('Query and source greedy token sequences do not match')
    query_key = query_record.get('sample_key')
    source_key = source_record.get('sample_key') if source_record else None
    if source_key is not None and source_key != query_key:
        raise ValueError('Query and source sample keys do not match')
    query_hashes = query_record.get('input', {}).get('hashes')
    source_hashes = source_record.get('input_hashes') if source_record else None
    hashes_match = query_hashes == source_hashes if query_hashes is not None and source_hashes is not None else None
    if hashes_match is False:
        raise ValueError('Query and source input hashes do not match')
    return candidates, source_greedy, hashes_match


def _candidate_summary(candidate, classification, checks):
    branch = candidate.get('branch') or {}
    return {
        'candidate': candidate.get('index'),
        'classification': classification,
        'span': _span(candidate.get('parsed')),
        'answer': candidate.get('answer'),
        'last_forced_step': branch.get('position'),
        'parent_candidate': branch.get('parent'),
        'source_roles': list(branch.get('roles') or []),
        'alternative_roles': list(branch.get('target_roles') or []),
        'provenance': checks,
    }


def interpret_branch(query_record, source_record, target):
    """Return new display facts for one measured logit-margin target.

    Direct predicate, after exact candidate prefix/token matching:
    ``branch.position == step`` AND ``branch.forced_token == alternative`` AND
    ``branch.forced_prefix_ids == greedy_ids[:step] + [alternative]``.
    Parent identity is recorded separately: exact actual prefix equality is
    sufficient even when the same prefix was reached through another parent.
    """
    if target.get('kind') != 'logit_margin' or target.get('alternative') is None:
        raise ValueError('interpret_branch expects a measured logit-margin target')
    candidates, source_greedy, hashes_match = _source_context(query_record, source_record)
    step = int(target['step'])
    alternative_id = int(target['alternative'])
    greedy_ids = query_record['greedy']['ids']
    if not 0 <= step < len(greedy_ids):
        raise ValueError('Branch step is outside the saved greedy sequence')
    prefix = list(greedy_ids[:step])
    forced_prefix = prefix + [alternative_id]
    step_record = next((item for item in query_record.get('steps', []) if item.get('step') == step), {})
    branch = target.get('branch') or {}
    chosen = step_record.get('chosen') or {}
    greedy_span = _span(query_record.get('greedy', {}).get('span'))
    if greedy_span is None and source_greedy is not None:
        greedy_span = _span(source_greedy.get('parsed'))
    direct, descendants, unverified = [], [], []
    direct_token_sequences = set()
    for candidate in candidates:
        ids = candidate.get('ids') or []
        if len(ids) <= step or ids[:step] != prefix or ids[step] != alternative_id:
            continue
        metadata = candidate.get('branch') or {}
        last_step = metadata.get('position')
        stored_prefix = metadata.get('forced_prefix_ids')
        checks = {
            'exact_saved_greedy_prefix': True,
            'alternative_token_matches': True,
            'last_forced_step_is_selected_step': last_step == step,
            'forced_prefix_is_single_selected_change': stored_prefix == forced_prefix,
            'parent_is_source_greedy': metadata.get('parent') == source_greedy.get('index') if source_greedy else None,
        }
        if last_step == step and metadata.get('forced_token') == alternative_id and stored_prefix == forced_prefix:
            direct.append(_candidate_summary(candidate, 'direct_conditional_greedy', checks))
            direct_token_sequences.add(tuple(ids))
        elif (isinstance(last_step, int) and last_step > step and
              isinstance(stored_prefix, list) and len(stored_prefix) == last_step + 1 and
              stored_prefix == ids[:last_step + 1] and stored_prefix[:step + 1] == forced_prefix and
              metadata.get('forced_token') == ids[last_step]):
            descendants.append(_candidate_summary(candidate, 'later_forced_descendant', checks))
        else:
            unverified.append(_candidate_summary(candidate, 'matching_path_unverified', checks))
    direct.sort(key=lambda row: (row['candidate'] is None, str(row['candidate'])))
    conflict = len(direct_token_sequences) > 1
    representative = direct[0] if direct and not conflict else None
    if conflict:
        status = 'conflicting_direct'
    elif representative is not None:
        status = 'direct'
    elif descendants:
        status = 'descendants_only'
    elif unverified or (source_greedy is None and branch.get('mode', {}).get('candidates')):
        status = 'provenance_unverified'
    else:
        status = 'unobserved'
    alternative_span = representative['span'] if representative else None
    tolerance = _number(branch.get('mode', {}).get('endpoint_tolerance'))
    max_difference = max(abs(a-b) for a, b in zip(greedy_span, alternative_span)) if greedy_span and alternative_span else None
    distinct = max_difference > tolerance if max_difference is not None and tolerance is not None else None
    current, potential = bool(branch.get('current_competitor')), bool(branch.get('potential_competitor'))
    observed_gap = _number(branch.get('observed_best_gap'))
    groups = []
    for group in target.get('groups') or []:
        signed, absolute = _number(group.get('signed_sum')), _number(group.get('absolute_sum'))
        groups.append({
            'group': group.get('group'), 'count': group.get('count'),
            'signed_sum': signed, 'absolute_sum': absolute, 'grad_l2': _number(group.get('grad_l2')),
            'scaling_direction': _direction(signed),
            'signed_cancellation_fraction': max(0.0, min(1.0, 1-abs(signed)/absolute)) if signed is not None and absolute is not None and absolute > 0 else None,
            'representation': 'language_model_input_embedding',
        })
    return {
        'target_id': target.get('id'), 'step': step, 'prefix': step_record.get('prefix'),
        'greedy': {'id': greedy_ids[step], 'token': chosen.get('token'),
                   'probability': _number(chosen.get('probability')), 'source_probability': _number(chosen.get('source_probability')),
                   'span': greedy_span},
        'alternative': {'id': alternative_id, 'token': branch.get('token'),
                        'probability': _number(branch.get('probability')), 'source_probability': _number(branch.get('source_probability'))},
        'margin': {'value': _number(target.get('score', branch.get('margin'))), 'source': _number(branch.get('source_margin')),
                   'sensitive_to_replay': bool(branch.get('margin_sensitive_to_replay'))},
        'competition': {
            'current': current, 'potential': potential, 'potential_only': potential and not current,
            'probe_only': not current and not potential, 'selection_reason': branch.get('selection_reason'),
            'screen_gain': _number(branch.get('screen_gain')), 'screen_gap': _number(branch.get('screen_gap')),
            'screen_direction': branch.get('screen_direction'), 'screen_sign': branch.get('screen_sign'),
            'observed_best_gap': observed_gap,
            'observed_pairwise_reversal': observed_gap < 0 if observed_gap is not None else None,
            'observed_pairwise_tie': observed_gap == 0 if observed_gap is not None else None,
            'observed_best_gap_scope': 'minimum over tested directions and signs; winning direction is not stored in the query record',
            'full_vocabulary_top1_reversal_measured': False,
            'candidate_scope': 'fixed source top30; outside unmeasured',
        },
        'continuation': {
            'status': status, 'direct': representative, 'direct_candidates': direct,
            'descendant_count': len(descendants), 'descendant_examples': descendants,
            'unverified_count': len(unverified), 'unverified_examples': unverified,
            'provenance': {'source_record_available': source_greedy is not None,
                           'greedy_token_ids_match': source_greedy is not None,
                           'input_hashes_match': hashes_match,
                           'direct_predicate': 'exact prefix and alternative; last forced position equals step; forced prefix equals greedy prefix plus alternative'},
        },
        'comparison': {
            'available': max_difference is not None, 'greedy_span': greedy_span, 'alternative_span': alternative_span,
            'max_endpoint_difference': max_difference, 'endpoint_tolerance': tolerance,
            'direct_distinct_mode_candidate': distinct,
            'legacy_matching_path_distinct': bool(branch.get('mode', {}).get('distinct_mode_candidate')),
            'criterion': 'temporal endpoint separation only; event meaning and complete mode probability are not established',
        },
        'groups': groups,
        'pixels': {'signed_sum': _number(target.get('pixel_signed_sum')), 'grad_l2': _number(target.get('pixel_gradient_l2')),
                   'scaling_direction': _direction(target.get('pixel_signed_sum')),
                   'representation': 'normalized_pixels_including_vision_encoder_and_deepstack'},
    }


def interpret_query(query_record, source_record):
    """Summarize measured branches without mutating either supplied record."""
    branches = [interpret_branch(query_record, source_record, target) for target in query_record.get('targets', [])
                if target.get('kind') == 'logit_margin' and target.get('alternative') is not None]
    statuses = Counter(row['continuation']['status'] for row in branches)
    return {
        'sample_key': query_record.get('sample_key'),
        'greedy_span': _span(query_record.get('greedy', {}).get('span')),
        'branches': branches,
        'counts': {
            'measured_branches': len(branches), 'direct_confirmed': statuses['direct'],
            'direct_distinct': sum(row['comparison']['direct_distinct_mode_candidate'] is True for row in branches),
            'direct_non_distinct': sum(row['comparison']['direct_distinct_mode_candidate'] is False for row in branches),
            'direct_without_comparable_span': sum(row['continuation']['status'] == 'direct' and not row['comparison']['available'] for row in branches),
            'descendants_only': statuses['descendants_only'], 'unobserved': statuses['unobserved'],
            'provenance_unverified': statuses['provenance_unverified'], 'conflicting_direct': statuses['conflicting_direct'],
            'current': sum(row['competition']['current'] for row in branches),
            'potential_only': sum(row['competition']['potential_only'] for row in branches),
            'probe_only': sum(row['competition']['probe_only'] for row in branches),
        },
        'reading_guide': dict(READING_GUIDE),
    }
