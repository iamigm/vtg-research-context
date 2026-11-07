"""Validate canonical900 and unchanged reused pilot600 records."""
import argparse
from pathlib import Path
import temporal_mode_cumulative_scope as scope


def preflight():
    result=scope.preflight()
    root=scope.CANONICAL if scope.CANONICAL.exists() else scope.OLD
    import temporal_mode_capped_core as core
    core.write(root/'validation/cumulative_preflight.json',result)
    print('CUMULATIVE_PREFLIGHT_PASSED',result['queries'])


def validate():
    core=scope.configure(900)
    cohort=core.read(core.COHORT)
    total_draws=0
    reused=core.read(core.OUT/'metadata/reuse.json')
    for old in reused['responses']:
        assert core.sha(core.OUT/'responses'/old['name'])==old['sha256'],old['name']
    for row in cohort:
        data=core.read(core.OUT/'responses'/f'{row["sample_key"]}.json')
        assert core.is_complete(data)
        assert data['sample']['query']==row['query'] and data['sample']['video_path']==row['video_path']
        source_input=core.read(core.SOURCE/'inputs'/f'{row["sample_key"]}.json')
        assert data['input_hashes']==source_input['hashes']
        assert data['baseline']['ids']==core.read(core.SOURCE/'rollouts'/f'{row["sample_key"]}.json')['result']['candidates'][0]['ids']
        distributions={tuple(d['prefix']):d for d in data['distributions']}
        for route in data['results'].values():
            assert route['seed'] in [42,43]
            for k in core.CHECKPOINTS:
                point=route['checkpoints'][str(k)]
                assert point['tail_isolation_verified'] and core.digest(point['main_modes'])==point['main_candidate_hash']
                allowed=[x for group in point['main_modes'] for x in group['intervals']]
                assert point['selection']['span'] is None or point['selection']['span'] in allowed
                for kind in ('main','tail'):
                    assert point[kind]['rollouts']<=k and point[kind]['spent']<=core.CONFIG[kind+'_budget']
            for draw in route['draws']:
                sequence=data['sequences'][draw['sequence']]
                tail=False
                for i,token in enumerate(sequence['ids']):
                    dist=distributions[tuple(sequence['ids'][:i])]
                    edge=next(e for e in dist['accepted'] if e['token']==token)
                    assert edge['p']>=max(.009,.01*dist['pmax'])-1e-12
                    tail|=not edge['core']
                assert tail==(draw['kind']=='tail')
            total_draws+=len(route['draws'])
    for count in (300,600,900):
        run=scope.FAMILY/str(count)/'capped_rollouts'
        records=core.read(run/'metadata/source_records.json')
        assert records['queries']==count and len(records['records'])==count
        assert len(core.read(run/'metrics/per_query.json'))==count*24
        assert all(g['queries']==count for g in core.read(run/'metrics/zero_shot_comparison.json')['groups'])
    core.write(core.OUT/'validation/cumulative_final.json',dict(passed=True,queries=900,
        reused_queries=600,route_count=5400,budget_selections=21600,rollout_attempts=total_draws,
        seeds=[42,43],source_responses_byte_identical=True,timestamp=core.stamp()))
    print('CUMULATIVE900_VALIDATION_PASSED')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    preflight() if args.preflight else validate()
