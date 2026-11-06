"""Active execution scope over immutable stochastic-generation configuration."""
import hashlib
import json
import sys
from pathlib import Path

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())
RUN = PROJECT/'artifacts/runs/rollout/temporal_mode_search/pilot/600'
if str(RUN/'code') not in sys.path:
    sys.path.insert(0, str(RUN/'code'))
from temporal_mode_core import CONFIG, COHORT, METHODS, OUT, ROOT, RESULT, read, write, digest, sha, stamp


def scope():
    value = read(OUT/'metadata/execution_scope.json')
    assert value['queries'] == 600
    assert value['methods'] == METHODS
    assert value['generation_config_hash'] == digest(CONFIG)
    assert value['seeds'] and set(value['seeds']).issubset(CONFIG['seeds'])
    return value


def conditions(value=None):
    value = value or scope()
    return {f'{method}:{seed}' for method in value['methods'] for seed in value['seeds']}


def rank_for(video, world):
    return int(hashlib.sha256(video.encode()).hexdigest(), 16) % world


def assigned(cohort, rank, world):
    return [r for r in cohort if rank_for(r['video_path'], world) == rank]


def is_complete(data, value=None):
    return conditions(value).issubset(data.get('results', {}))


def active_rows(rows, value=None):
    value = value or scope()
    return [r for r in rows if r['seed'] in value['seeds'] and r['method'] in value['methods']]


def freeze_scope():
    value = dict(queries=600, methods=METHODS, seeds=[42,43], excluded_seeds=[44], worker_count=3,
        total_conditions=3600, generation_config_hash=digest(CONFIG),
        generation_config_reference=str(OUT/'metadata/config.json'),
        cohort_sha256=sha(COHORT), partition='SHA256(video_path) modulo worker_count',
        change='User requested two seeds and increased GPU utilization; generation policy and budgets unchanged.',
        legacy_seed44='Preserved in original raw responses; excluded from generation, metrics, validation denominator and viewer choices.',
        timestamp=stamp())
    effective = dict(CONFIG, seeds=value['seeds'], workers=value['worker_count'],
                     original_generation_config_hash=digest(CONFIG), execution_scope=str(OUT/'metadata/execution_scope.json'))
    target = OUT/'metadata/execution_scope.json'
    if target.exists():
        old = read(target)
        for key in ('queries','methods','seeds','worker_count','generation_config_hash'):
            assert old[key] == value[key]
        value = old
    else:
        write(target, value)
    write(OUT/'metadata/effective_config.json', effective)
    return value
