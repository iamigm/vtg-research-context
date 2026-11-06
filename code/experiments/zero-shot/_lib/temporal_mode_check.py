"""Scientific invariants and final raw-data validation, independent of GPU runs."""
import argparse
import copy
import json
import math
import types
import unittest
from temporal_mode_core import *


class CoreTests(unittest.TestCase):
    def test_signed_clip_and_no_reordering(self):
        self.assertEqual(parse_clip('The event happens in -2 - 8 seconds.', 10)['span'], [0., 8.])
        self.assertEqual(parse_clip('The event happens in 8 - 14 seconds.', 10)['span'], [8., 10.])
        self.assertIsNone(parse_clip('The event happens in -4 - -1 seconds.', 10)['span'])
        self.assertIsNone(parse_clip('The event happens in 8 - 3 seconds.', 10)['span'])
        self.assertIsNone(parse_clip('The event happens in 1 - 5 seconds.', 10, False)['span'])
        self.assertEqual(parse_clip('-00:02 - 00:08', 10)['span'], [0., 8.])

    def test_information_distinguishes_token_entropy_from_mode_information(self):
        self.assertAlmostEqual(mutual_information({'a': {'early': .5}, 'b': {'late': .5}}), 1.)
        self.assertAlmostEqual(mutual_information({'a': {'early': .25, 'late': .25}, 'b': {'early': .25, 'late': .25}}), 0.)
        self.assertAlmostEqual(mutual_information({'a': {'early': .99}, 'b': {'late': .01}}), .08079313589591118)
        self.assertIsNone(mutual_information({}))

    def test_clipped_alias_does_not_duplicate_interval_or_raw_mass(self):
        sequences = {}
        registry = Modes(sequences)
        for index, raw in enumerate(['-1 - 8', '0 - 8', '0.0 - 8.0']):
            value = dict(id=str(index), raw_logp=math.log(.1), **parse_clip(raw, 10))
            sequences[str(index)] = value
            registry.add(value)
            registry.add(value)
        summary = registry.summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]['unique_intervals'], 1)
        self.assertEqual(len(summary[0]['paths']), 3)
        self.assertAlmostEqual(math.exp(summary[0]['observed_log_mass']), .3)

    def test_complete_link_prevents_chaining(self):
        sequences = {}
        registry = Modes(sequences)
        for index, interval in enumerate([[10., 20.], [11., 21.], [12., 22.]]):
            value = dict(id=str(index), span=interval, raw_logp=-1., valid=True)
            sequences[value['id']] = value
            registry.add(value)
        self.assertEqual(len(registry.groups), 2)

    def test_raw_floor_before_normalization(self):
        import torch
        instance = object.__new__(Backend)
        instance.torch, instance.distributions = torch, {}
        values = torch.tensor([.98, .0095, .009, .0015], dtype=torch.float64)
        instance.get = lambda prefix: values.log()
        distribution = instance.distribution(())
        self.assertEqual([item['token'] for item in distribution['accepted']], [0])
        self.assertAlmostEqual(distribution['threshold'], .0098)
        self.assertAlmostEqual(distribution['excluded_mass'], .02)

    def test_position_bias_cancels(self):
        class PositionBiased:
            def choice(self, prompt, budget, calls):
                return dict(probabilities=[1., 0., 0.], option_mass=1.)
        result = preference(PositionBiased(), 'action', [1, 2], [8, 9], Budget(1000), [])
        self.assertEqual(result['probability_left'], .5)
        self.assertTrue(result['order_conflict'])

    def test_tail_never_mutates_main_candidates(self):
        class Fake:
            def __init__(self):
                self.sequences, self.distributions, self.eos, self.duration = {}, {}, {9}, 100.
                self.tok = types.SimpleNamespace(decode=lambda ids, **kw: {1:'10 - 20', 2:'10.2 - 20.1', 3:'80 - 90'}[ids[0]])
            def distribution(self, prefix):
                prefix = tuple(prefix)
                accepted = [dict(token=1,p=.6,core=True), dict(token=2,p=.3,core=True), dict(token=3,p=.1,core=False)] if not prefix else [dict(token=9,p=1.,core=True)]
                result = dict(prefix=list(prefix), accepted=accepted, entropy_bits=1.)
                self.distributions[prefix] = result
                return result
            remember = Backend.remember
            generate = Backend.generate
        backend = Fake()
        frozen = copy.deepcopy(CONFIG)
        try:
            CONFIG.update(main_budget=120, tail_budget=120, root_probes=4, main_min_continuations=2, tail_min_continuations=2)
            search = Search(backend, dict(sample_key='fake', query='action'), 'temporal_mode', 42)
            search.run_main()
            before = digest(search.main.summary())
            result = search.run_tail()
            self.assertEqual(digest(search.main.summary()), before)
            self.assertEqual(len(search.main.groups), 1)
            self.assertEqual(search.tail.groups[0]['intervals'], [[80., 90.]])
            self.assertTrue(result['diagnostic_only'])
            self.assertTrue(any(draw['kind']=='tail' for draw in search.draws))
        finally:
            CONFIG.clear()
            CONFIG.update(frozen)


def validate():
    cohort = read(COHORT)
    assert len(cohort) == 600
    manifest = read(OUT/'metadata/manifest.json')
    assert sha(COHORT) == manifest['cohort_sha256']
    for source in manifest['files']:
        assert sha(OUT/'code'/Path(source['path']).name) == source['sha256']
    checked = 0
    for row in cohort:
        data = read(OUT/'responses'/f'{row["sample_key"]}.json')
        assert data['config_hash'] == digest(CONFIG)
        assert len(data['results']) == len(METHODS)*len(CONFIG['seeds'])
        original = read(SOURCE/'inputs'/f'{row["sample_key"]}.json')
        assert data['input_hashes'] == original['hashes']
        inputs = read(OUT/'inputs'/f'{row["sample_key"]}.json')
        duration = inputs['clipping_duration']
        distributions = {tuple(d['prefix']): d for d in data['distributions']}
        for distribution in distributions.values():
            cumulative = 0.
            for edge in distribution['accepted']:
                assert edge['p'] >= max(.009, .01*distribution['pmax']) - 1e-12
                assert edge['core'] == (cumulative < .9)
                cumulative += edge['p']
        for sequence in data['sequences'].values():
            expected = parse_clip(sequence['text'], duration, sequence['stopped_on_eos'])
            assert expected['span'] == sequence['span'] and expected['raw'] == sequence['raw']
        for result in data['results'].values():
            assert result['tail_isolation_verified']
            assert digest(result['main_modes']) == result['main_candidate_hash']
            intervals = [interval for group in result['main_modes'] for interval in group['intervals']]
            assert result['selection']['span'] is None or result['selection']['span'] in intervals
            assert result['main']['spent'] <= CONFIG['main_budget']
            assert result['tail']['spent'] <= CONFIG['tail_budget']
            assert result['selection']['cost'] <= CONFIG['judge_budget']
            for kind in ('main', 'tail'):
                groups = result[kind+'_modes']
                for group in groups:
                    assert len(group['intervals']) == len({tuple(x) for x in group['intervals']})
                    assert all(compatible(a,b) for a in group['intervals'] for b in group['intervals'])
            for draw in result['draws']:
                sequence = data['sequences'][draw['sequence']]
                tail = False
                for index, token in enumerate(sequence['ids']):
                    distribution = distributions[tuple(sequence['ids'][:index])]
                    edge = next(item for item in distribution['accepted'] if item['token']==token)
                    tail |= not edge['core']
                assert tail == (draw['kind']=='tail')
            checked += 1
    write(OUT/'validation/final.json', dict(passed=True, queries=600, conditions=checked,
        cohort_hash=sha(COHORT), main_tail_isolation=True, native_token_floors=True,
        clipping_before_dedup=True, input_hashes_equal=True, code_frozen=True, timestamp=stamp()))
    print(json.dumps(dict(passed=True, queries=600, conditions=checked)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.validate:
        validate()
    else:
        unittest.main(argv=['temporal_mode_check'], verbosity=2)
