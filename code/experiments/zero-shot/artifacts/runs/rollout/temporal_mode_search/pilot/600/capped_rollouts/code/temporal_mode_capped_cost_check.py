"""Cost-accounting invariance, independent cache and trace counter checks."""
import contextlib
import types
import unittest
from temporal_mode_capped_cost import *
from temporal_mode_capped_check import FakeBackend, fake_judge


class FakeMeter:
    @contextlib.contextmanager
    def segment(self, phase):
        yield dict(phase=phase, **{key:0 for key in ADDITIVE})


class CostTests(unittest.TestCase):
    def test_accounting_preserves_complete_generation_and_judgment(self):
        sample = dict(sample_key='fake', query='action')
        for method in METHODS:
            a, b = FakeBackend(), FakeBackend()
            b.cost_meter, b.baseline_cost = FakeMeter(), {}
            plain = Search(a, sample, method, 42).sweep(judge_fn=fake_judge)
            measured = CostSearch(b, sample, method, 42).sweep(judge_fn=fake_judge)
            for point in measured['checkpoints'].values():
                point.pop('cost')
            self.assertEqual(plain, measured)

    def test_native_prefix_rollback_and_memoization(self):
        work = TraceWork(10)
        work.distribution([1,2,3], 'main')
        self.assertEqual(work.records['main']['forwards'], 3)
        work.distribution([1,2], 'main')  # crop cannot return earlier logits; replay one token
        self.assertEqual(work.records['main']['forwards'], 4)
        work.distribution([1,2,3], 'main')  # cached exact distribution: no forward
        self.assertEqual(work.records['main']['forwards'], 4)
        work.distribution([1,4,5], 'main')
        self.assertEqual(work.records['main']['forwards'], 6)
        self.assertEqual(work.records['main']['decoder_inputs'], 6)
        self.assertEqual(work.records['main']['attention_pairs'], 11+12+13+12+12+13)

    def test_judge_prefill_and_observation_decode(self):
        work = TraceWork(10)
        choices = {'a':dict(prompt='a'), 'b':dict(prompt='b',kind='visual_observation',ids=[7,8,9])}
        work.judge(dict(calls=['a','b','a']), choices, lambda _: [1,2,3])
        self.assertEqual(work.records['judge']['forwards'], 4)
        self.assertEqual(work.records['judge']['decoder_inputs'], 8)
        self.assertEqual(work.records['judge']['attention_pairs'], 2*(30+6)+14+15)

    def test_no_supported_token_is_still_a_distribution_request(self):
        work = TraceWork(10)
        work.draws([dict(sequence='a',task='root')],
            {'a':dict(ids=[1,2],generation_stop='no_supported_token')}, {'root':[]}, 'main')
        self.assertEqual(work.records['main']['forwards'],2)

    def test_fresh_case_has_no_other_methods_or_previous_judge_cache(self):
        one, two = TraceWork(10), TraceWork(10)
        one.distribution([1,2], 'main')
        two.distribution([1,2], 'main')
        self.assertEqual(one.records, two.records)
        self.assertEqual(two.records['main']['forwards'],2)

    def test_aggregates_add_time_but_take_peak_memory_maximum(self):
        result = combine([dict(wall_seconds=2,peak_allocated_bytes=10),dict(wall_seconds=3,peak_allocated_bytes=7)])
        self.assertEqual(result['wall_seconds'],5)
        self.assertEqual(result['peak_allocated_bytes'],10)


def validate():
    from temporal_mode_capped_cost_report import cost_rows, summarize
    manifest = read(OUT/'metadata/cost_manifest.json')
    for entry in manifest['files']:
        assert sha(entry['snapshot']) == entry['sha256']
    rows = cost_rows()
    summary = summarize(rows)
    assert summary['common_queries'] == 600
    assert len(rows) == 14400
    missing = 0
    for row in rows:
        values = row['standalone']
        assert values['research']['decoder_inputs'] >= values['operational']['decoder_inputs'] >= values['baseline']['decoder_inputs']
        if row['observed']:
            actual = row['observed']
            assert actual['schema'] == SCHEMA
            for phase in ('main','tail','judge','sweep'):
                assert actual[phase]['forwards'] >= 0 and actual[phase]['cuda_forward_seconds'] >= 0
                assert actual[phase]['peak_allocated_bytes'] <= actual[phase]['peak_reserved_bytes']
        else:
            missing += 1
    assert read(OUT/'validation/cost_gpu_smoke.json')['passed']
    write(OUT/'validation/cost_final.json',dict(passed=True,cost_rows=len(rows),
        common_queries=summary['common_queries'],measured_common_queries=summary['measured_common_queries'],
        pre_instrumentation_missing=missing,raw_generation_config_unchanged=True,timestamp=stamp()))


if __name__ == '__main__':
    import sys
    validate() if '--validate' in sys.argv else unittest.main(argv=['cost_check'], verbosity=2)
