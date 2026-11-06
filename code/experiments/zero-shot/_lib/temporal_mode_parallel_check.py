"""Partition, seed filtering, immutable generation and final parallel validation."""
import argparse
import collections
import json
import unittest
from pathlib import Path
from temporal_mode_scope import *
from temporal_mode_core import compatible,parse_clip


class ParallelTests(unittest.TestCase):
    def test_video_partition_disjoint_and_complete(self):
        cohort=read(COHORT)
        partitions=[assigned(cohort,rank,3) for rank in range(3)]
        self.assertEqual(sum(map(len,partitions)),600)
        query_sets=[{r['sample_key'] for r in rows} for rows in partitions]
        video_sets=[{r['video_path'] for r in rows} for rows in partitions]
        for i in range(3):
            for j in range(i+1,3):
                self.assertFalse(query_sets[i]&query_sets[j])
                self.assertFalse(video_sets[i]&video_sets[j])
        self.assertEqual(set.union(*query_sets),{r['sample_key'] for r in cohort})

    def test_seed44_excluded_and_completed_active_prefix_reusable(self):
        value=dict(methods=METHODS,seeds=[42,43])
        data=dict(results={f'{method}:{seed}':{} for method in METHODS for seed in [42,43,44]})
        self.assertTrue(is_complete(data,value))
        subset=dict(results={key:result for key,result in data['results'].items() if key.endswith((':42',':43'))})
        self.assertTrue(is_complete(subset,value))
        rows=[dict(method=method,seed=seed,iou=float(seed==44)) for method in METHODS for seed in [42,43,44]]
        selected=active_rows(rows,value)
        self.assertEqual(len(selected),6)
        self.assertEqual(sum(r['iou'] for r in selected),0.)
        del subset['results']['probability:43']
        self.assertFalse(is_complete(subset,value))

    def test_existing_raw_hashes_and_scope_aggregation(self):
        import temporal_mode_scope_report as reporting
        cohort=read(COHORT)
        paths=[OUT/'responses'/f'{r["sample_key"]}.json' for r in cohort]
        hashes={path:sha(path) for path in paths if path.exists()}
        rows,completed=reporting.selected_rows(cohort)
        self.assertTrue(all(r['seed'] in (42,43) for r in rows))
        self.assertEqual(len(rows),len(completed)*6)
        self.assertEqual(CONFIG['seeds'],[42,43,44])
        self.assertEqual(hashes,{path:sha(path) for path in hashes})


def validate():
    active=scope()
    expected=conditions(active)
    cohort=read(COHORT)
    assert sha(COHORT)==active['cohort_sha256']
    manifest=read(OUT/'metadata/manifest.json')
    for source in manifest['files']:
        assert sha(OUT/'code'/Path(source['path']).name)==source['sha256']
    parallel=read(OUT/'metadata/parallel_manifest.json')
    for source in parallel['files']:
        assert sha(source['snapshot'])==source['sha256']
    checked=0
    excluded=0
    for row in cohort:
        data=read(OUT/'responses'/f'{row["sample_key"]}.json')
        assert data['config_hash']==digest(CONFIG)
        assert expected.issubset(data['results'])
        excluded+=sum(result['seed'] not in active['seeds'] for result in data['results'].values())
        inputs=read(OUT/'inputs'/f'{row["sample_key"]}.json')
        from temporal_mode_core import SOURCE
        assert data['input_hashes']==read(SOURCE/'inputs'/f'{row["sample_key"]}.json')['hashes']
        distributions={tuple(item['prefix']):item for item in data['distributions']}
        for distribution in distributions.values():
            cumulative=0.
            for edge in distribution['accepted']:
                assert edge['p']>=max(.009,.01*distribution['pmax'])-1e-12
                assert edge['core']==(cumulative<.9)
                cumulative+=edge['p']
        for condition in expected:
            result=data['results'][condition]
            assert result['seed'] in active['seeds']
            assert result['tail_isolation_verified']
            assert digest(result['main_modes'])==result['main_candidate_hash']
            intervals=[interval for group in result['main_modes'] for interval in group['intervals']]
            assert result['selection']['span'] is None or result['selection']['span'] in intervals
            for kind in ('main','tail'):
                for group in result[kind+'_modes']:
                    assert len(group['intervals'])==len({tuple(x) for x in group['intervals']})
                    assert all(compatible(a,b) for a in group['intervals'] for b in group['intervals'])
            for draw in result['draws']:
                sequence=data['sequences'][draw['sequence']]
                assert parse_clip(sequence['text'],inputs['clipping_duration'],sequence['stopped_on_eos'])['span']==sequence['span']
                tail=False
                for index,token in enumerate(sequence['ids']):
                    distribution=distributions[tuple(sequence['ids'][:index])]
                    edge=next(item for item in distribution['accepted'] if item['token']==token)
                    tail|=not edge['core']
                assert tail==(draw['kind']=='tail')
            assert result['main']['spent']<=CONFIG['main_budget']
            assert result['tail']['spent']<=CONFIG['tail_budget']
            assert result['selection']['cost']<=CONFIG['judge_budget']
            checked+=1
    metrics=read(OUT/'metrics/per_query.json')
    assert len(metrics)==3600 and {r['seed'] for r in metrics}=={42,43}
    write(OUT/'validation/final.json',dict(passed=True,queries=600,conditions=checked,active_seeds=[42,43],
        excluded_legacy_conditions=excluded,worker_count=3,partition_disjoint=True,tail_isolation=True,
        generation_policy_unchanged=True,timestamp=stamp()))
    print(json.dumps(dict(passed=True,queries=600,active_conditions=checked,excluded_conditions=excluded)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    validate() if args.validate else unittest.main(argv=['temporal_mode_parallel_check'],verbosity=2)
