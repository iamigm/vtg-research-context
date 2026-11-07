"""Regression checks for caps, policy activation, nested budgets and tail isolation."""
import argparse
import collections
import copy
import math
import types
import unittest
from temporal_mode_capped_core import *


class FakeBackend:
    generate=Backend.generate
    remember=Backend.remember

    def __init__(self):
        self.sequences={}
        self.distributions={}
        self.eos={9}
        self.duration=100.
        self.tok=types.SimpleNamespace(decode=self.decode)

    def decode(self,ids,**kwargs):
        if ids[0]==3:
            return '90 - 95'
        if len(ids)>1 and ids[1]==6:
            return '40 - 45'
        return { (1,4):'10 - 20',(1,5):'10.3 - 20.2',(2,4):'70 - 80',(2,5):'70.4 - 80.3'}[tuple(ids[:2])]

    def distribution(self,prefix):
        prefix=tuple(prefix)
        if not prefix:
            edges=[(1,.6,True),(2,.3,True),(3,.1,False)]
        elif len(prefix)==1 and prefix[0]!=3:
            edges=[(4,.55,True),(5,.35,True),(6,.1,False)]
        else:
            edges=[(9,1.,True)]
        value=dict(prefix=list(prefix),accepted=[dict(token=t,p=p,core=c) for t,p,c in edges],
            entropy_bits=1. if not prefix else (.2 if prefix[0]==1 else 3.))
        self.distributions[prefix]=value
        return value


def fake_judge(backend,sample,modes):
    return dict(span=modes[0]['representatives'][0] if modes else None,status='test_only',pool=modes,
                cost=0,comparisons=[],assessments=[],calls=[])


class CappedTests(unittest.TestCase):
    def test_power_checkpoints_and_bound(self):
        search=Search(FakeBackend(),dict(sample_key='fake',query='action'),'temporal_mode',42)
        output=search.sweep(judge_fn=fake_judge)
        self.assertEqual(len(output['draws']),256)
        for target in CHECKPOINTS:
            value=output['checkpoints'][str(target)]
            self.assertEqual(value['main']['rollouts'],target)
            self.assertEqual(value['tail']['rollouts'],target)
            self.assertEqual(value['draw_cutoff'],2*target)
            self.assertEqual(target&(target-1),0)
            self.assertTrue(value['tail_isolation_verified'])

    def test_initialization_cannot_starve_policy(self):
        search=Search(FakeBackend(),dict(sample_key='fake',query='action'),'probability',42)
        output=search.sweep(judge_fn=fake_judge)
        for value in output['checkpoints'].values():
            stats=value['main']['scheduler']
            self.assertLessEqual(stats.get('root',0)+stats.get('initialization',0),8)
            self.assertGreaterEqual(stats.get('adaptive',0),8)
        self.assertGreaterEqual(output['checkpoints']['128']['main']['scheduler']['adaptive'],100)

    def test_three_policies_really_use_distinct_scores(self):
        tasks=[]
        for token,p,entropy,new_weight in [(1,.6,.1,0.),(2,.3,8.,0.),(3,.1,.1,900000.)]:
            tasks.append(dict(prefix=[token],parent=[],token=token,p=p,logp=math.log(p),
                entropy=entropy,trials=1000000,attempts=1,costs=[1.],new_weight=new_weight))
        selected=[]
        for method in METHODS:
            search=Search(FakeBackend(),dict(sample_key='fake',query='action'),method,42)
            search.counts['main']=10
            selected.append(search.select(tasks,'main')['token'])
            self.assertEqual(search.last_scheduler,'adaptive')
        self.assertEqual(selected,[1,2,3])

    def test_tail_and_intermediate_checkpoints_do_not_change_main_stream(self):
        a=Search(FakeBackend(),dict(sample_key='fake',query='action'),'temporal_mode',42)
        a.advance_main(128)
        b=Search(FakeBackend(),dict(sample_key='fake',query='action'),'temporal_mode',42)
        b.sweep(judge_fn=fake_judge)
        a_main=[(d['sequence'],d['task']) for d in a.draws if d['kind']=='main']
        b_main=[(d['sequence'],d['task']) for d in b.draws if d['kind']=='main']
        self.assertEqual(a_main,b_main)

    def test_old_checkpoint_is_immutable_after_extension(self):
        saved=[]
        search=Search(FakeBackend(),dict(sample_key='fake',query='action'),'token_entropy',42)
        def save(result):
            if '16' in result['checkpoints'] and not saved:
                saved.append(digest(result['checkpoints']['16']))
        output=search.sweep(save=save,judge_fn=fake_judge)
        self.assertEqual(saved[0],digest(output['checkpoints']['16']))

    def test_video_partition_is_disjoint(self):
        cohort=read(COHORT)
        partitions=[assigned(cohort,i) for i in range(3)]
        self.assertEqual(sum(map(len,partitions)),600)
        for i in range(3):
            for j in range(i+1,3):
                self.assertFalse({r['video_path'] for r in partitions[i]}&{r['video_path'] for r in partitions[j]})


def validate():
    manifest=read(OUT/'metadata/manifest.json')
    assert manifest['config_hash']==digest(CONFIG)
    assert manifest['cohort_sha256']==sha(COHORT)
    for entry in manifest['files']+manifest['base_files']:
        assert sha(entry['snapshot'])==entry['sha256']
    counts=collections.Counter()
    used=0
    adaptive=0
    for row in read(COHORT):
        data=read(OUT/'responses'/f'{row["sample_key"]}.json')
        assert is_complete(data)
        inputs=read(OUT/'inputs'/f'{row["sample_key"]}.json')
        assert inputs['hashes']==read(SOURCE/'inputs'/f'{row["sample_key"]}.json')['hashes']==data['input_hashes']
        distributions={tuple(d['prefix']):d for d in data['distributions']}
        for route in data['results'].values():
            assert route['seed'] in [42,43]
            assert len([d for d in route['draws'] if d['kind']=='main'])<=128
            assert len([d for d in route['draws'] if d['kind']=='tail'])<=128
            previous=None
            for target in CHECKPOINTS:
                value=route['checkpoints'][str(target)]
                assert value['tail_isolation_verified']
                assert digest(value['main_modes'])==value['main_candidate_hash']
                allowed=[interval for group in value['main_modes'] for interval in group['intervals']]
                assert value['selection']['span'] is None or value['selection']['span'] in allowed
                for kind in ('main','tail'):
                    state=value[kind]
                    assert state['rollouts']<=target
                    assert state['spent']<=CONFIG[kind+'_budget']
                stats=value['main']['scheduler']
                assert stats.get('root',0)+stats.get('initialization',0)<=8
                if value['main']['tasks'] and value['main']['rollouts']>=16:
                    assert stats.get('adaptive',0)>0
                adaptive+=stats.get('adaptive',0)
                cutoff=value['draw_cutoff']
                selected=route['draws'][:cutoff]
                assert len([d for d in selected if d['kind']=='main'])==value['main']['rollouts']
                assert len([d for d in selected if d['kind']=='tail'])==value['tail']['rollouts']
                if previous:
                    assert previous['draw_cutoff']<=cutoff
                previous=value
                counts[str(target)]+=1
            for draw in route['draws']:
                sequence=data['sequences'][draw['sequence']]
                assert parse_clip(sequence['text'],inputs['clipping_duration'],sequence['stopped_on_eos'])['span']==sequence['span']
                is_tail=False
                for i,token in enumerate(sequence['ids']):
                    distribution=distributions[tuple(sequence['ids'][:i])]
                    edge=next(e for e in distribution['accepted'] if e['token']==token)
                    assert edge['p']>=max(.009,.01*distribution['pmax'])-1e-12
                    is_tail|=not edge['core']
                assert is_tail==(draw['kind']=='tail')
            used+=len(route['draws'])
    assert all(counts[str(k)]==3600 for k in CHECKPOINTS)
    metrics=read(OUT/'metrics/per_query.json')
    assert len(metrics)==14400 and {r['seed'] for r in metrics}=={42,43}
    write(OUT/'validation/final.json',dict(passed=True,queries=600,route_count=3600,
        budget_selections=dict(counts),rollout_attempts=used,maximum_attempts=600*1536,
        adaptive_selections_exercised=adaptive,tail_isolation=True,nested_checkpoints=True,timestamp=stamp()))
    print('CAPPED_FINAL_VALIDATION_PASSED')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    validate() if args.validate else unittest.main(argv=['temporal_mode_capped_check'],verbosity=2)
