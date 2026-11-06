"""Probability-supported stochastic token trees; GT-free temporal-mode discovery.

No digit enumeration, proposal prompt, or review-conditioned regeneration.  Raw
model probabilities and the actual truncated sampling probabilities are distinct.
"""
import collections
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent/'_lib').is_dir() and (parent/'artifacts').is_dir())
if str(ROOT/'_lib') not in sys.path:
    sys.path.append(str(ROOT/'_lib'))
FAMILY = 'rollout/temporal_mode_search'
OUT = ROOT / 'artifacts/runs' / FAMILY / 'pilot/600'
RESULT = ROOT / 'results' / FAMILY / 'pilot/600'
SOURCE = ROOT / 'artifacts/runs/rollout/deterministic/pilot/900'
COHORT = ROOT / 'artifacts/common/cohorts/nested_pilot_v1/pilot_600/samples.json'
MODEL = Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct')
METHODS = ['probability', 'token_entropy', 'temporal_mode']
CONFIG = dict(family=FAMILY, queries=600, model=str(MODEL), fps=2,
    target_pixels=100352, input_order='video_query', methods=METHODS,
    seeds=[42, 43, 44], absolute_floor=.009, relative_floor=.01, main_mass=.90,
    main_budget=65536, tail_budget=131072, judge_budget=32768,
    main_min_continuations=8, tail_min_continuations=32,
    max_new_tokens=512, root_probes=16, maximum_modes=6,
    mode_iou=.5, mode_absolute_seconds=1., mode_relative_endpoint=.1,
    option_mass_floor=.01, temperature=1., presence_penalty=0., repetition_penalty=1.,
    clipping='each parsed endpoint to [0, metadata duration]; no sorting; require start<end',
    raw_probability='full-vocabulary softmax before all filters; EOS included in path',
    sampling='with replacement conditional categorical; duplicate draws retained; exact logits reused',
    mass_estimator='raw unique observed path mass is a lower bound; stratified p/q-weighted suffix probes for MI and novelty',
    main_tail='all core tokens=main; any supported noncore token=tail; tail never proposed',
    budget_scope='logical decoder input tokens: exact text prefix replay plus generated tokens; video prefill separate',
    input_policy='reuse fixed deterministic tensors/frame IDs/PTS/HW; assert input hashes',
    final_selector='two boundary representatives per mode; swapped-order three-way pairwise; round robin across up to six main modes',
    judge_text_prefill='fixed full text prompt batch over unchanged video KV; rollout rollback uses single-token replay',
    max_retries=3, gpu_initial_free_gib=55, cpu_threads=2)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    import datetime
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def iou(a, b):
    if not a or not b:
        return 0.
    return max(0., min(a[1], b[1]) - max(a[0], b[0])) / max(1e-12, max(a[1], b[1]) - min(a[0], b[0]))


def compatible(a, b):
    tolerance = max(CONFIG['mode_absolute_seconds'], CONFIG['mode_relative_endpoint'] * min(a[1]-a[0], b[1]-b[0]))
    return iou(a, b) >= CONFIG['mode_iou'] and max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= tolerance


NUMBER = r'[+-]?(?:\d+(?::\d{1,2}){0,2}(?:\.\d+)?|\.\d+)'
PAIR = re.compile(r'(?<![\w.])(' + NUMBER + r')\s*(?:-|–|—|\bto\b)\s*(' + NUMBER + r')(?![\d.:])')


def seconds(value):
    sign = -1 if value.startswith('-') else 1
    parts = value.lstrip('+-').split(':')
    if len(parts) > 1 and any(not 0 <= float(x) < 60 for x in parts[1:]):
        raise ValueError('Invalid clock field')
    result = 0.
    for part in parts:
        result = result * 60 + float(part)
    return sign * result


def parse_clip(text, duration, eos=True):
    if not eos:
        return dict(raw=None, span=None, valid=False, reason='no_eos', clipped=False)
    matches = list(PAIR.finditer(text))
    if len(matches) != 1:
        return dict(raw=None, span=None, valid=False, reason='ambiguous_or_missing_interval', clipped=False)
    try:
        raw = [seconds(matches[0].group(j)) for j in (1, 2)]
        if not all(math.isfinite(x) for x in raw):
            raise ValueError('Nonfinite')
    except ValueError:
        return dict(raw=None, span=None, valid=False, reason='invalid_number', clipped=False)
    span = [max(0., min(duration, x)) for x in raw]
    valid = span[0] < span[1]
    return dict(raw=raw, span=span if valid else None, valid=valid,
                reason=None if valid else 'empty_or_reversed_after_clipping', clipped=span != raw)


def logsum(values):
    values = list(values)
    if not values:
        return -1e300
    maximum = max(values)
    return maximum + math.log(sum(math.exp(x-maximum) for x in values))


def normalized_logs(values):
    norm = logsum(values)
    return [math.exp(x-norm) for x in values]


def mutual_information(matrix):
    """Joint nonnegative mass by token and valid mode; bits, normalized here only."""
    total = sum(sum(row.values()) for row in matrix.values())
    if total <= 0:
        return None
    rows = {a: sum(row.values()) / total for a, row in matrix.items()}
    cols = collections.defaultdict(float)
    for row in matrix.values():
        for c, value in row.items():
            cols[c] += value / total
    return max(0., sum((value/total) * math.log2((value/total) / (rows[a]*cols[c]))
                      for a, row in matrix.items() for c, value in row.items() if value > 0))


class Modes:
    def __init__(self, sequences):
        self.sequences = sequences
        self.groups = []
        self.seen = set()

    def add(self, sequence):
        key = sequence['id']
        if key in self.seen:
            return self.find(sequence['span']), False
        self.seen.add(key)
        span = sequence['span']
        if not span:
            return None, False
        eligible = [g for g in self.groups if all(compatible(span, item) for item in g['intervals'])]
        if eligible:
            group = min(eligible, key=lambda g: (min(max(abs(span[j]-i[j]) for j in (0, 1)) for i in g['intervals']), g['id']))
            new = False
        else:
            group = dict(id=len(self.groups), intervals=[], paths=[])
            self.groups.append(group)
            new = True
        if span not in group['intervals']:
            group['intervals'].append(span)
        group['paths'].append(key)
        return group['id'], new

    def find(self, span):
        if not span:
            return None
        return next((g['id'] for g in self.groups if span in g['intervals']), None)

    def summary(self):
        output = []
        for group in self.groups:
            paths = [self.sequences[k] for k in group['paths']]
            interval_logs = collections.defaultdict(list)
            for path in paths:
                interval_logs[tuple(path['span'])].append(path['raw_logp'])
            intervals = sorted(interval_logs)
            logs = [logsum(interval_logs[i]) for i in intervals]
            weights = normalized_logs(logs)
            central = max(range(len(intervals)), key=lambda j: (sum(w*iou(intervals[j], other) for w, other in zip(weights, intervals)), logs[j], tuple(-v for v in intervals[j])))
            alternative = max(range(len(intervals)), key=lambda j: (max(abs(intervals[j][k]-intervals[central][k]) for k in (0, 1)), logs[j]))
            representatives = [list(intervals[central])]
            if alternative != central:
                representatives.append(list(intervals[alternative]))
            output.append(dict(**group, observed_log_mass=logsum(p['raw_logp'] for p in paths),
                               representatives=representatives, unique_intervals=len(intervals)))
        return sorted(output, key=lambda g: (-g['observed_log_mass'], g['id']))


class BudgetEnd(Exception):
    pass


class Budget:
    def __init__(self, cap):
        self.cap = cap
        self.spent = 0

    def charge(self, amount):
        if self.spent + amount > self.cap:
            raise BudgetEnd()
        self.spent += amount


class Backend:
    """One video KV, deterministic single-token rollback/replay, compact CPU cache."""
    def __init__(self, model, inputs, torch, tokenizer, eos):
        self.model, self.inputs, self.torch, self.tok = model, inputs, torch, tokenizer
        self.eos = set(eos)
        self.n = inputs.input_ids.shape[1]
        self.path = ()
        self.distributions = {}
        self.sequences = {}
        self.choices = {}
        self.actual_tokens = 0
        self.actual_forwards = 0
        self.started = time.monotonic()
        torch.cuda.synchronize()
        with torch.inference_mode():
            output = model(**inputs, use_cache=True, logits_to_keep=1)
        self.cache = output.past_key_values
        self.root = output.logits[0, -1].float().cpu()
        self.last = self.root
        torch.cuda.synchronize()
        self.prefill_seconds = time.monotonic() - self.started
        self.choice_ids = [tokenizer.encode(x, add_special_tokens=False) for x in ('A', 'B', 'C')]
        assert all(len(x) == 1 for x in self.choice_ids), self.choice_ids
        self.choice_ids = [x[0] for x in self.choice_ids]
        self.end_turn = tokenizer.convert_tokens_to_ids('<|im_end|>')

    def get(self, path):
        path = tuple(path)
        if not path:
            return self.root
        if path == self.path:
            return self.last
        common = 0
        for a, b in zip(self.path, path):
            if a != b:
                break
            common += 1
        if common == len(path):
            common -= 1
        self.cache.crop(self.n + common)
        torch = self.torch
        for j in range(common, len(path)):
            with torch.inference_mode():
                output = self.model(input_ids=torch.tensor([[path[j]]], device='cuda'),
                    attention_mask=torch.ones((1, self.n+j+1), device='cuda', dtype=self.inputs.attention_mask.dtype),
                    past_key_values=self.cache, cache_position=torch.tensor([self.n+j], device='cuda'),
                    use_cache=True, logits_to_keep=1)
            self.cache = output.past_key_values
            self.last = output.logits[0, -1].float().cpu()
            self.path = path[:j+1]
            self.actual_tokens += 1
            self.actual_forwards += 1
        return self.last

    def distribution(self, path):
        path = tuple(path)
        if path in self.distributions:
            return self.distributions[path]
        torch = self.torch
        probabilities = torch.softmax(self.get(path), dim=-1)
        values, ids = torch.topk(probabilities, min(256, len(probabilities)))
        values, ids = values.tolist(), ids.tolist()
        floor = max(CONFIG['absolute_floor'], CONFIG['relative_floor'] * values[0])
        retained = []
        cumulative = 0.
        for token, probability in zip(ids, values):
            core = cumulative < CONFIG['main_mass']
            cumulative += probability
            if probability >= floor:
                retained.append(dict(token=token, p=probability, core=core,
                    text=self.tok.decode([token], clean_up_tokenization_spaces=False) if hasattr(self, 'tok') else str(token)))
        positive = probabilities[probabilities > 0]
        result = dict(prefix=list(path), prefix_text=self.tok.decode(path, skip_special_tokens=True, clean_up_tokenization_spaces=False) if hasattr(self, 'tok') else '',
            accepted=retained, pmax=values[0], threshold=floor,
            entropy_bits=float(-(positive*positive.log2()).sum()),
            admitted_mass=sum(x['p'] for x in retained), core_mass=sum(x['p'] for x in retained if x['core']),
            excluded_mass=max(0., 1-sum(x['p'] for x in retained)),
            top30=[dict(token=i, p=p) for i, p in zip(ids[:30], values[:30])])
        self.distributions[path] = result
        return result

    def remember(self, ids, raw_logp, stopped, reason=None):
        key = digest(list(ids))[:24]
        if key not in self.sequences:
            text = self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            value = parse_clip(text, self.duration, stopped)
            self.sequences[key] = dict(id=key, ids=list(ids), text=text, raw_logp=raw_logp,
                stopped_on_eos=stopped, generation_stop=reason, **value)
        return self.sequences[key]

    def generate(self, prefix, prefix_logp, rng, kind, budget):
        ids = list(prefix)
        budget.charge(len(prefix))
        raw_logp = prefix_logp
        steps = []
        log_weight = 0.
        reason = 'length_cap'
        # All token IDs, including mixed numeric/separator tokens, remain atomic.
        while len(ids) < CONFIG['max_new_tokens'] and (not ids or ids[-1] not in self.eos):
            distribution = self.distribution(ids)
            choices = [x for x in distribution['accepted'] if kind == 'tail' or x['core']]
            if not choices:
                reason = 'no_supported_token'
                break
            budget.charge(1)
            mass = sum(x['p'] for x in choices)
            chosen = rng.choices(choices, weights=[x['p'] for x in choices], k=1)[0]
            p, token = chosen['p'], chosen['token']
            q = p/mass
            steps.append(dict(prefix=tuple(ids), token=token, log_weight_before=log_weight,
                              raw_prefix_logp=raw_logp, p=p, q=q))
            log_weight += math.log(mass)
            raw_logp += math.log(p)
            ids.append(token)
        stopped = bool(ids and ids[-1] in self.eos)
        if stopped:
            reason = 'eos'
        sequence = self.remember(ids, raw_logp, stopped, reason)
        return sequence, steps, log_weight

    def greedy(self):
        ids, logp = [], 0.
        for _ in range(CONFIG['max_new_tokens']):
            probabilities = self.torch.softmax(self.get(ids), dim=-1)
            token = int(probabilities.argmax())
            logp += math.log(max(float(probabilities[token]), 1e-300))
            ids.append(token)
            if token in self.eos:
                break
        return self.remember(ids, logp, bool(ids and ids[-1] in self.eos), 'greedy')

    def dialog(self, prompt):
        template = self.tok.apply_chat_template([dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True)
        return [self.end_turn] + self.tok.encode('\n'+template, add_special_tokens=False)

    def get_judge_prompt(self, prefix):
        """Fixed-shape causal text prefill, never used for rollout probabilities."""
        self.cache.crop(self.n)
        torch = self.torch
        with torch.inference_mode():
            output = self.model(input_ids=torch.tensor([prefix], device='cuda'),
                attention_mask=torch.ones((1, self.n+len(prefix)), device='cuda', dtype=self.inputs.attention_mask.dtype),
                past_key_values=self.cache, cache_position=torch.arange(self.n, self.n+len(prefix), device='cuda'),
                use_cache=True, logits_to_keep=1)
        self.cache = output.past_key_values
        self.last = output.logits[0,-1].float().cpu()
        self.path = tuple(prefix)
        self.actual_tokens += len(prefix)
        self.actual_forwards += 1
        return self.last

    def choice(self, prompt, budget, calls):
        prefix = self.dialog(prompt)
        budget.charge(len(prefix)+1)
        key = digest(prompt)
        if key not in self.choices:
            probabilities = self.torch.softmax(self.get_judge_prompt(prefix), dim=-1)[self.choice_ids].tolist()
            mass = sum(probabilities)
            self.choices[key] = dict(prompt=prompt, raw_probabilities=probabilities, option_mass=mass,
                probabilities=[x/mass for x in probabilities] if mass else [1/3]*3)
        calls.append(key)
        return self.choices[key]

    def observation(self, prompt, budget, calls):
        prefix = self.dialog(prompt)
        key = digest(['observation', prompt])
        if key in self.choices:
            result = self.choices[key]
            budget.charge(len(prefix)+len(result['ids']))
        else:
            # Reserve a complete bounded observation; partial notes do not change candidates.
            if budget.spent + len(prefix) + 48 > budget.cap:
                raise BudgetEnd()
            budget.charge(len(prefix))
            ids = []
            for _ in range(48):
                logits = self.get(prefix+ids) if ids else self.get_judge_prompt(prefix)
                token = int(logits.argmax())
                budget.charge(1)
                ids.append(token)
                if token in self.eos:
                    break
            result = dict(kind='visual_observation', prompt=prompt, ids=ids,
                text=self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            self.choices[key] = result
        calls.append(key)
        return result['text']


class Search:
    def __init__(self, backend, sample, method, seed, notify=None):
        self.b, self.sample, self.method, self.seed = backend, sample, method, seed
        self.tasks = {}
        self.prefix_logp = {}
        self.main = Modes(backend.sequences)
        self.tail = Modes(backend.sequences)
        self.draws = []
        self.notify = notify or (lambda **kw: None)
        self.draw_index = 0

    def rng(self, task, kind):
        # Common random numbers at identical conditional probes, regardless of scheduling.
        index = task['attempts'] if task else self.draw_index
        key = digest([self.sample['sample_key'], self.seed, kind, task['prefix'] if task else [], index])
        return random.Random(int(key[:16], 16))

    def discover(self, steps):
        for step in steps:
            prefix = step['prefix']
            self.prefix_logp[prefix] = step['raw_prefix_logp']
            distribution = self.b.distribution(prefix)
            if len(distribution['accepted']) < 2:
                continue
            for edge in distribution['accepted']:
                if not edge['core']:
                    continue
                key = prefix + (edge['token'],)
                if key not in self.tasks:
                    self.tasks[key] = dict(prefix=list(key), parent=list(prefix), token=edge['token'],
                        logp=step['raw_prefix_logp']+math.log(edge['p']), p=edge['p'],
                        entropy=distribution['entropy_bits'], attempts=0, trials=0, probes=[], costs=[], new_weight=0.)

    def probe(self, task, kind, budget):
        prefix = task['prefix'] if task else []
        logp = task['logp'] if task else 0.
        before = budget.spent
        sequence, steps, log_weight = self.b.generate(prefix, logp, self.rng(task, kind), kind, budget)
        registry = self.main if kind == 'main' else self.tail
        mode, new = registry.add(sequence)
        record = dict(sequence=sequence['id'], kind=kind, task=digest(prefix)[:24],
                      suffix_log_weight=log_weight, new_mode=new, mode=mode,
                      logical_tokens=budget.spent-before)
        self.draws.append(record)
        self.draw_index += 1
        if task:
            task['attempts'] += 1
        if kind == 'main':
            self.discover(steps)
        # Opportunistic suffix probes are conditioned on their exact visited edge.
        for index, step in enumerate(steps):
            key = step['prefix']+(step['token'],)
            current = self.tasks.get(key) if kind == 'main' else None
            if current is not None:
                suffix_weight = log_weight - (step['log_weight_before'] + math.log(step['p']/step['q']))
                self.update_task(current, sequence, suffix_weight, new, budget.spent-before)
        if task:
            self.update_task(task, sequence, log_weight, new, budget.spent-before)
        if self.draw_index % 16 == 0:
            self.notify(stage=kind+'_search', method=self.method, seed=self.seed, decoder_tokens=budget.spent,
                        main_modes=len(self.main.groups), tail_modes=len(self.tail.groups), draws=self.draw_index)

    def update_task(self, task, sequence, log_weight, new, cost):
        task['trials'] += 1
        task['probes'].append([sequence['id'], log_weight])
        task['costs'].append(cost)
        if new and sequence['valid']:
            task['new_weight'] += math.exp(log_weight)

    def select(self, tasks, kind):
        minimum = CONFIG['main_min_continuations'] if kind == 'main' else CONFIG['tail_min_continuations']
        pending = [t for t in tasks if t['trials'] < minimum]
        if pending:
            return max(pending, key=lambda t: (t['logp'], -t['trials'], tuple(-x for x in t['prefix'])))
        def score(task):
            cost = sum(task['costs'])/max(1, len(task['costs']))
            value = task['logp'] - math.log(max(1., cost))
            if self.method == 'token_entropy' and kind == 'main':
                value += math.log(max(1e-6, task['entropy']))
            elif self.method == 'temporal_mode' or kind == 'tail':
                n = max(1, task['trials'])
                upper = min(1., task['new_weight']/n + math.sqrt(math.log(2+self.draw_index)/(2*n)))
                value += math.log(max(upper, 1e-12))
            # Diminishing repeated allocation also applies to probability/entropy controls.
            return value - .5*math.log(1+task['trials'])
        return max(tasks, key=lambda t: (score(t), tuple(-x for x in t['prefix'])))

    def mode_information(self, tasks, registry):
        by_parent = collections.defaultdict(dict)
        for task in tasks:
            bins = collections.defaultdict(float)
            for sequence_id, log_weight in task['probes']:
                sequence = self.b.sequences[sequence_id]
                mode = registry.find(sequence['span'])
                if mode is not None:
                    bins[str(mode)] += task['p'] * math.exp(log_weight)/max(1, task['trials'])
            by_parent[tuple(task['parent'])][str(task['token'])] = dict(bins)
        return [dict(prefix=list(prefix), mi_bits=mutual_information(matrix), joint_observed_mass=matrix)
                for prefix, matrix in by_parent.items()]

    def run_main(self):
        budget = Budget(CONFIG['main_budget'])
        stop = 'token_budget'
        try:
            for _ in range(CONFIG['root_probes']):
                self.probe(None, 'main', budget)
            while self.tasks:
                self.probe(self.select(list(self.tasks.values()), 'main'), 'main', budget)
            stop = 'frontier_exhausted'
        except BudgetEnd:
            pass
        return dict(spent=budget.spent, cap=budget.cap, stop=stop,
                    unfinished_minimum=sum(t['trials'] < CONFIG['main_min_continuations'] for t in self.tasks.values()),
                    tasks=list(self.tasks.values()), mi=self.mode_information(self.tasks.values(), self.main))

    def run_tail(self):
        # First-tail edges from main-only prefixes form disjoint tail entry strata.
        tasks = []
        for prefix, logp in self.prefix_logp.items():
            for edge in self.b.distribution(prefix)['accepted']:
                if edge['core']:
                    continue
                tasks.append(dict(prefix=list(prefix+(edge['token'],)), parent=list(prefix), token=edge['token'],
                    logp=logp+math.log(edge['p']), p=edge['p'], entropy=0., attempts=0, trials=0, probes=[], costs=[], new_weight=0.))
        budget = Budget(CONFIG['tail_budget'])
        try:
            while tasks:
                self.probe(self.select(tasks, 'tail'), 'tail', budget)
        except BudgetEnd:
            pass
        return dict(spent=budget.spent, cap=budget.cap, stop='token_budget' if tasks else 'no_supported_tail_entries',
                    tasks=tasks, unfinished_minimum=sum(t['trials'] < CONFIG['tail_min_continuations'] for t in tasks),
                    mi=self.mode_information(tasks, self.tail), diagnostic_only=True)


def preference(backend, query, left, right, budget, calls):
    def ask(a, b):
        prompt = (f'Target event: {query}\nInterval A: {a} seconds. Interval B: {b} seconds.\n'
            'Use the video. First compare the requested action, actor, object and occurrence; '
            'if both match, compare precise onset and offset, penalizing missing action and unrelated duration. '
            'Answer one letter only: A = interval A is more accurate; B = interval B is more accurate; '
            'C = equivalent or insufficient visual evidence.')
        result = backend.choice(prompt, budget, calls)
        return result['probabilities'] if result['option_mass'] >= CONFIG['option_mass_floor'] else [0., 0., 1.]
    forward, reverse = ask(left, right), ask(right, left)
    probability = .5*(forward[0]+.5*forward[2]+reverse[1]+.5*reverse[2])
    forward_delta, reverse_delta = forward[0]-forward[1], reverse[0]-reverse[1]
    conflict = abs(forward_delta) > 1e-6 and abs(reverse_delta) > 1e-6 and (forward_delta > 0) == (reverse_delta > 0)
    return dict(left=left, right=right, probability_left=probability, forward=forward, reverse=reverse,
                order_conflict=conflict)


def judge(backend, sample, main_modes):
    pool = main_modes[:CONFIG['maximum_modes']]
    if not pool:
        return dict(span=None, status='no_valid_main_candidate', pool=[], assessments=[], comparisons=[], cost=0, calls=[])
    fallback = pool[0]['representatives'][0]
    if len(pool) == 1 and len(pool[0]['representatives']) == 1:
        return dict(span=fallback, status='single_main_candidate', pool=pool, assessments=[], comparisons=[], cost=0, calls=[])
    budget = Budget(CONFIG['judge_budget'])
    calls, assessments, comparisons, winners = [], [], [], []
    try:
        for mode in pool:
            for interval in mode['representatives']:
                record = dict(mode=mode['id'], interval=interval)
                for name, question in [('event_match', 'Does this interval show the requested action, correct actor/object and occurrence?'),
                    ('onset_fit', 'Does the onset include the beginning of the requested event without unrelated lead-in?'),
                    ('offset_fit', 'Does the offset include the end of the requested event without unrelated trailing action?')]:
                    prompt = (f'Target event: {sample["query"]}\nCandidate interval: {interval} seconds. Use the video. '
                        f'{question} Answer one letter only: A = supported by visible evidence; '
                        'B = contradicted by visible evidence; C = uncertain or not observable.')
                    result = backend.choice(prompt, budget, calls)
                    record[name] = dict(probabilities=result['probabilities'], option_mass=result['option_mass'])
                record['visual_evidence'] = backend.observation(
                    f'Target event: {sample["query"]}\nInterval: {interval} seconds. '
                    'Use the video. In at most 24 words describe visible action and onset/offset evidence or uncertainty. '
                    'Do not propose or generate another interval.', budget, calls)
                assessments.append(record)
            representatives = mode['representatives']
            winner = representatives[0]
            if len(representatives) > 1:
                comparison = preference(backend, sample['query'], representatives[0], representatives[1], budget, calls)
                comparisons.append(dict(stage='boundary', mode=mode['id'], **comparison))
                if comparison['probability_left'] < .5:
                    winner = representatives[1]
            winners.append(dict(mode=mode['id'], interval=winner, observed_log_mass=mode['observed_log_mass']))
        scores = [0.]*len(winners)
        for i in range(len(winners)):
            for j in range(i+1, len(winners)):
                comparison = preference(backend, sample['query'], winners[i]['interval'], winners[j]['interval'], budget, calls)
                comparisons.append(dict(stage='mode', mode_left=winners[i]['mode'], mode_right=winners[j]['mode'], **comparison))
                scores[i] += comparison['probability_left']
                scores[j] += 1-comparison['probability_left']
        selected = max(range(len(winners)), key=lambda i: (scores[i], winners[i]['observed_log_mass'], tuple(-x for x in winners[i]['interval'])))
        for item, score in zip(winners, scores):
            item['mean_preference'] = score/max(1, len(winners)-1)
        return dict(span=winners[selected]['interval'], status='complete', pool=pool, winners=winners,
                    assessments=assessments, comparisons=comparisons, cost=budget.spent, calls=calls)
    except BudgetEnd:
        return dict(span=fallback, status='judge_budget_fallback', pool=pool, assessments=assessments,
                    comparisons=comparisons, cost=budget.spent, calls=calls)
