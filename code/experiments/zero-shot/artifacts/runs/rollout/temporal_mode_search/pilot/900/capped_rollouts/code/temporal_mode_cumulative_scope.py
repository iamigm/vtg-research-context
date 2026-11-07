"""Canonical pilot900 storage; preserve pilot600 config hashes and subset views."""
import collections
import json
import sys
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())
FAMILY=ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot'
OLD=FAMILY/'600/capped_rollouts'
CANONICAL=FAMILY/'900/capped_rollouts'
COHORT_ROOT=ROOT/'artifacts/common/cohorts/nested_pilot_v1'
_configured=False


def json_read(path):
    return json.loads(Path(path).read_text())


def source_config():
    reuse=CANONICAL/'metadata/reuse.json'
    return json_read(reuse)['source_config'] if reuse.exists() else json_read(OLD/'metadata/config.json')


def configure(scope=900,reporting=False):
    global CONFIG,OLD_HASH,NEW_HASH,COHORT_ROWS,OLD_KEYS,SCOPE,OUT,RESULT,COHORT,_configured
    if _configured:
        raise RuntimeError('Configure one cohort scope per process')
    sys.path.insert(0,str(CANONICAL/'code' if (CANONICAL/'code').exists() else OLD/'code'))
    import temporal_mode_capped_core as core
    SCOPE=scope
    origin=source_config()
    CONFIG=dict(origin,queries=900)
    OLD_HASH=core.digest(origin)
    NEW_HASH=core.digest(CONFIG)
    COHORT_ROWS=json_read(COHORT_ROOT/'pilot_900/samples.json')
    assert json_read(COHORT_ROOT/'pilot_600/samples.json')==COHORT_ROWS[:600]
    assert json_read(COHORT_ROOT/'pilot_300/samples.json')==COHORT_ROWS[:300]
    OLD_KEYS={row['sample_key'] for row in COHORT_ROWS[:600]}
    OUT=FAMILY/str(scope)/'capped_rollouts'
    RESULT=ROOT/f'results/rollout/temporal_mode_search/pilot/{scope}/capped_rollouts'
    COHORT=COHORT_ROOT/f'pilot_{scope}/samples.json'
    core.CONFIG=CONFIG
    core.OUT=OUT
    core.RESULT=RESULT
    core.COHORT=COHORT
    core.is_complete=is_complete
    _configured=True
    return core


def accepted_record(data,key=None):
    key=key or data.get('sample',{}).get('sample_key')
    if data.get('sample',{}).get('sample_key')!=key:
        return False
    return data.get('config_hash')==NEW_HASH or (key in OLD_KEYS and data.get('config_hash')==OLD_HASH)


def is_complete(data):
    if not accepted_record(data):
        return False
    expected={'16','32','64','128'}
    return all(set(data.get('results',{}).get(f'{method}:{seed}',{}).get('checkpoints',{}))==expected
        for method in CONFIG['methods'] for seed in CONFIG['seeds'])


def canonical_read(path):
    path=Path(path)
    try:
        relative=path.relative_to(OUT)
    except ValueError:
        return json_read(path)
    if relative.parts[0] in ('responses','inputs','code','environment','logs'):
        path=CANONICAL/relative
        if relative.parts[0]=='responses' and path.name.endswith('.partial.json'):
            final=path.with_name(path.name.replace('.partial.json','.json'))
            if final.exists():
                path=final
    elif relative.parts[0]=='metadata' and not path.exists():
        path=CANONICAL/relative
    value=json_read(path)
    if relative.parts[0]=='responses':
        assert accepted_record(value),path
        # Analysis uses the scope-equivalent config; source JSON bytes stay unchanged.
        if value['config_hash']!=NEW_HASH:
            value=dict(value,source_config_hash=value['config_hash'],config_hash=NEW_HASH)
    return value


def preflight():
    import temporal_mode_capped_core as core
    origin=source_config()
    rows=json_read(COHORT_ROOT/'pilot_900/samples.json')
    assert len(rows)==900 and json_read(COHORT_ROOT/'pilot_600/samples.json')==rows[:600]
    assert json_read(COHORT_ROOT/'pilot_300/samples.json')==rows[:300]
    assert origin['seeds']==[42,43] and origin['checkpoints']==[16,32,64,128]
    assert origin['absolute_floor']==.009 and origin['relative_floor']==.01
    counts=collections.Counter(row['dataset'] for row in rows)
    assert set(counts.values())=={300}
    for row in rows:
        assert (core.SOURCE/'inputs'/f'{row["sample_key"]}.json').is_file()
        assert (core.SOURCE/'rollouts'/f'{row["sample_key"]}.json').is_file()
    return dict(passed=True,queries=900,source_queries=600,additional_queries=300,
        datasets=dict(counts),cohort_sha256=core.sha(COHORT_ROOT/'pilot_900/samples.json'),
        source_cohort_sha256=core.sha(COHORT_ROOT/'pilot_600/samples.json'),
        generation_changes=[],scope_only_config_changes=['queries'],seeds=origin['seeds'],
        checkpoints=origin['checkpoints'],timestamp=core.stamp())


def register_canonical(count,state='building'):
    """Relocate the same registered resource after the verified physical move."""
    from artifact_store import register,STORE,relative,atomic,now
    resource=f'temporal_mode_search.capped_rollouts.pilot{count}.responses'
    target=STORE/'registry'/f'{resource}.json'
    if target.exists():
        old=json_read(target)
        canonical=relative(CANONICAL/'responses')
        if old['path']!=canonical:
            assert count==600 and not (OLD/'responses').exists()
            assert old['path'].endswith('/pilot/600/capped_rollouts/responses')
            assert json_read(CANONICAL/'metadata/reuse.json')['source_config']['queries']==600
            old.update(path=canonical,updated_at=now())
            atomic(target,old)
    register(resource,'rollouts',CANONICAL/'responses',
        f'Fixed cumulative pilot{count}; canonical900 source; two seeds.',
        producer='temporal_mode_cumulative_runtime.py',
        cohort=str(COHORT_ROOT/f'pilot_{count}/samples.json'),state=state)
