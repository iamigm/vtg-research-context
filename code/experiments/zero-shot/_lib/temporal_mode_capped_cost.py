"""Observe native forwards and reconstruct independent-case work from saved traces.

No logits, RNG, model inputs, generation policy, or candidate ordering are changed.
CUDA stream time is measured around each native forward, not around Python loops.
"""
import collections
import copy
import time
from contextlib import contextmanager

from temporal_mode_capped_core import *

SCHEMA = 'forward_cost_accounting'
ADDITIVE = ('forwards', 'decoder_inputs', 'attention_pairs', 'cuda_forward_seconds',
            'forward_wall_seconds', 'wall_seconds', 'distribution_requests',
            'distribution_hits', 'judge_requests', 'judge_hits')


def combine(records):
    records = list(records)
    value = {key: sum(record.get(key, 0) for record in records) for key in ADDITIVE}
    for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'extra_peak_allocated_bytes'):
        value[key] = max((record.get(key, 0) for record in records), default=0)
    return value


class Meter:
    def __init__(self, torch):
        self.torch = torch
        self.current = None
        self.records = []

    @contextmanager
    def segment(self, phase):
        assert self.current is None
        cuda = self.torch.cuda
        cuda.synchronize()
        cuda.reset_peak_memory_stats()
        starting_allocated = cuda.memory_allocated()
        record = dict(phase=phase, **{key: 0 for key in ADDITIVE})
        self.current = record
        started = time.perf_counter()
        try:
            yield record
        finally:
            cuda.synchronize()
            record.update(wall_seconds=time.perf_counter()-started,
                starting_allocated_bytes=starting_allocated,
                peak_allocated_bytes=cuda.max_memory_allocated(),
                peak_reserved_bytes=cuda.max_memory_reserved(),
                extra_peak_allocated_bytes=max(0, cuda.max_memory_allocated()-starting_allocated))
            self.records.append(record)
            self.current = None

    def count(self, name, amount=1):
        if self.current is not None:
            self.current[name] = self.current.get(name, 0) + amount


class MeasuredModel:
    def __init__(self, model, meter):
        self.model, self.meter = model, meter

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        # Native call arguments and output objects pass through unchanged.
        torch = self.meter.torch
        q = int(kwargs['input_ids'].shape[-1])
        kv = int(kwargs['attention_mask'].shape[-1])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        output = self.model(*args, **kwargs)
        end.record()
        end.synchronize()
        self.meter.count('cuda_forward_seconds', start.elapsed_time(end)/1000.)
        self.meter.count('forward_wall_seconds', time.perf_counter()-started)
        self.meter.count('forwards')
        self.meter.count('decoder_inputs', q)
        self.meter.count('attention_pairs', q*(kv-q)+q*(q+1)//2)
        return output


class CostBackend(Backend):
    def __init__(self, model, inputs, torch, tokenizer, eos):
        self.cost_meter = Meter(torch)
        with self.cost_meter.segment('video_prefill') as record:
            super().__init__(MeasuredModel(model, self.cost_meter), inputs, torch, tokenizer, eos)
        self.baseline_cost = dict(schema=SCHEMA, video_prefill=record)

    def greedy(self):
        with self.cost_meter.segment('greedy_decode') as record:
            result = super().greedy()
        self.baseline_cost['greedy_decode'] = record
        self.baseline_cost['total'] = combine([self.baseline_cost['video_prefill'], record])
        # Keep the sequence registry itself free of operational measurements.
        return dict(result, cost=copy.deepcopy(self.baseline_cost))

    def distribution(self, path):
        self.cost_meter.count('distribution_requests')
        self.cost_meter.count('distribution_hits', int(tuple(path) in self.distributions))
        return super().distribution(path)

    def choice(self, prompt, budget, calls):
        self.cost_meter.count('judge_requests')
        self.cost_meter.count('judge_hits', int(digest(prompt) in self.choices))
        return super().choice(prompt, budget, calls)

    def observation(self, prompt, budget, calls):
        self.cost_meter.count('judge_requests')
        self.cost_meter.count('judge_hits', int(digest(['observation', prompt]) in self.choices))
        return super().observation(prompt, budget, calls)


class CostSearch(Search):
    def sweep(self, save=None, checkpoints=None, judge_fn=judge):
        output = dict(method=self.method, seed=self.seed, checkpoints={}, draws=self.draws)
        segments = collections.defaultdict(list)
        sweep_started = time.perf_counter()
        for target in checkpoints or CHECKPOINTS:
            with self.b.cost_meter.segment('main') as main_cost:
                self.advance_main(target)
                frozen = copy.deepcopy(self.main.summary())
                before = digest(frozen)
                main = self.state('main', target)
            segments['main'].append(main_cost)
            self.notify(stage='model_judgment', method=self.method, seed=self.seed,
                        rollout_budget=2*target, main_rollouts=self.counts['main'], tail_rollouts=self.counts['tail'])
            with self.b.cost_meter.segment('judge') as judge_cost:
                selection = judge_fn(self.b, self.sample, frozen)
            segments['judge'].append(judge_cost)
            with self.b.cost_meter.segment('tail') as tail_cost:
                self.advance_tail(target)
                tail_state = self.state('tail', target)
                tail_modes = copy.deepcopy(self.tail.summary())
                assert digest(self.main.summary()) == before, 'Tail changed main candidate state'
            segments['tail'].append(tail_cost)
            actual = dict(schema=SCHEMA, scope='observed shared-cache sweep; method/seed/order dependent',
                baseline=copy.deepcopy(self.b.baseline_cost),
                main=combine(segments['main']), tail=combine(segments['tail']), judge=copy.deepcopy(judge_cost),
                incremental=dict(main=main_cost, tail=tail_cost, judge=judge_cost),
                sweep=combine(record for phase in segments.values() for record in phase),
                sweep_elapsed_seconds=time.perf_counter()-sweep_started)
            output['checkpoints'][str(target)] = dict(method=self.method, seed=self.seed,
                budget_per_kind=target, total_rollout_budget=2*target, main=main, tail=tail_state,
                main_modes=frozen, tail_modes=tail_modes, selection=selection, draw_cutoff=len(self.draws),
                main_candidate_hash=before, tail_isolation_verified=True, cost=actual)
            self.notify(stage='budget_checkpoint', method=self.method, seed=self.seed, rollout_budget=2*target,
                main_rollouts=self.counts['main'], tail_rollouts=self.counts['tail'],
                adaptive_main_selections=self.scheduler['main']['adaptive'],
                measured_cuda_forward_seconds=actual['sweep']['cuda_forward_seconds'],
                measured_forwards=actual['sweep']['forwards'])
            if save:
                save(output)
        return output


class TraceWork:
    """Count native decoder work under the specified independent-case cache schedule.

    Starts at a fresh video KV; runs greedy, main up to K, judge at K, then tail up
    to K. Exact distribution/prompt caches exist inside this case only. No GPU
    timing or memory value is inferred. attention_pairs is per layer/head causal
    query-key pairs, a context-length workload proxy, not FLOPs.
    """
    def __init__(self, n):
        self.n, self.path = n, ()
        self.distributions, self.choices = set(), set()
        self.records = collections.defaultdict(lambda: dict(forwards=0, decoder_inputs=0, attention_pairs=0))
        self.add('video_prefill', n, n)

    def add(self, phase, q, kv):
        value = self.records[phase]
        value['forwards'] += 1
        value['decoder_inputs'] += q
        value['attention_pairs'] += q*(kv-q)+q*(q+1)//2

    def get(self, path, phase):
        path = tuple(path)
        if not path or path == self.path:
            return
        common = 0
        for a, b in zip(self.path, path):
            if a != b:
                break
            common += 1
        if common == len(path):
            common -= 1
        for j in range(common, len(path)):
            self.add(phase, 1, self.n+j+1)
        self.path = path

    def distribution(self, path, phase):
        path = tuple(path)
        if path not in self.distributions:
            self.get(path, phase)
            self.distributions.add(path)

    def greedy(self, ids):
        for j in range(len(ids)):
            self.get(ids[:j], 'greedy_decode')

    def draws(self, draws, sequences, prefixes, phase):
        for draw in draws:
            sequence = sequences[draw['sequence']]
            ids = sequence['ids']
            prefix = prefixes[draw['task']]
            assert ids[:len(prefix)] == prefix
            stop = len(ids) + int(sequence.get('generation_stop') == 'no_supported_token' and len(ids) < CONFIG['max_new_tokens'])
            for j in range(len(prefix), stop):
                self.distribution(ids[:j], phase)

    def judge(self, selection, choices, dialog):
        for key in selection['calls']:
            if key in self.choices:
                continue
            value = choices[key]
            prefix = dialog(value['prompt'])
            self.add('judge', len(prefix), self.n+len(prefix))
            self.path = tuple(prefix)
            if value.get('kind') == 'visual_observation':
                for j in range(1, len(value['ids'])):
                    self.get(prefix+value['ids'][:j], 'judge')
            self.choices.add(key)


def standalone_work(data, route, checkpoint, n, dialog):
    work = TraceWork(n)
    work.greedy(data['baseline']['ids'])
    prefixes = {digest([])[:24]: []}
    for kind in ('main', 'tail'):
        prefixes.update({digest(task['prefix'])[:24]: task['prefix'] for task in checkpoint[kind]['tasks']})
    draws = route['draws'][:checkpoint['draw_cutoff']]
    work.draws([d for d in draws if d['kind'] == 'main'], data['sequences'], prefixes, 'main')
    work.judge(checkpoint['selection'], data['choices'], dialog)
    work.draws([d for d in draws if d['kind'] == 'tail'], data['sequences'], prefixes, 'tail')
    # defaultdict lookup materializes zero-cost phases, including no-tail/no-judge.
    phases = {name: dict(work.records[name]) for name in ('video_prefill', 'greedy_decode', 'main', 'judge', 'tail')}
    baseline = combine([phases['video_prefill'], phases['greedy_decode']])
    operational = combine([phases[name] for name in ('video_prefill', 'greedy_decode', 'main', 'judge')])
    return dict(schema=SCHEMA, scope='trace-derived independent case; main then judge then diagnostic tail',
        phases=phases, baseline=baseline, operational=operational, research=combine(phases.values()))
