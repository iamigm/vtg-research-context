"""Power-of-two rollout checkpoints with bounded initialization and real policies."""
import collections
import copy
import hashlib
import math
import random
import sys
from pathlib import Path

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())
LEGACY=PROJECT/'artifacts/runs/rollout/temporal_mode_search/pilot/600'
sys.path.insert(0,str(LEGACY/'code'))
import temporal_mode_core as base
from temporal_mode_core import Backend,Budget,BudgetEnd,Modes,judge,iou,compatible,parse_clip,read,write,digest,sha,stamp

ROOT=PROJECT
OUT=LEGACY/'capped_rollouts'
RESULT=PROJECT/'results/rollout/temporal_mode_search/pilot/600/capped_rollouts'
COHORT=base.COHORT
SOURCE=base.SOURCE
MODEL=base.MODEL
METHODS=base.METHODS
CHECKPOINTS=[16,32,64,128]
CONFIG=dict(base.CONFIG,condition='capped_rollouts',seeds=[42,43],worker_count=3,
    checkpoints=CHECKPOINTS,main_rollout_cap=128,tail_rollout_cap=128,
    root_probes=4,main_initialization_cap=8,tail_initialization_cap=4,coverage_every=8,
    main_min_continuations=None,tail_min_continuations=None,
    budget_scope='rollout attempts (duplicates/invalid included) per kind; main=tail=16,32,64,128; one nested sweep; token caps are secondary guards',
    generation_change='bounded initialization and policy scheduling; no quota on every newly discovered prefix',
    base_code_reference=str(LEGACY/'code/temporal_mode_core.py'),
    base_code_sha256=sha(LEGACY/'code/temporal_mode_core.py'))


def condition_keys():
    return {f'{method}:{seed}' for method in METHODS for seed in CONFIG['seeds']}


def is_complete(data):
    return data.get('config_hash')==digest(CONFIG) and all(
        key in data.get('results',{}) and set(data['results'][key].get('checkpoints',{}))==set(map(str,CHECKPOINTS))
        for key in condition_keys())


def rank_for(video,world=3):
    return int(hashlib.sha256(video.encode()).hexdigest(),16)%world


def assigned(cohort,rank,world=3):
    return [row for row in cohort if rank_for(row['video_path'],world)==rank]


class Search(base.Search):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.counts=dict(main=0,tail=0)
        self.scheduler={kind:collections.Counter() for kind in ('main','tail')}
        self.budgets={kind:Budget(CONFIG[kind+'_budget']) for kind in ('main','tail')}
        self.tail_tasks={}
        self.exhausted={}
        self.last_scheduler=None

    def rng(self,task,kind):
        # Tail calls and intermediate judge calls never alter the main RNG stream.
        index=task['attempts'] if task else self.counts[kind]
        value=digest([self.sample['sample_key'],self.seed,kind,task['prefix'] if task else [],index])
        return random.Random(int(value[:16],16))

    def discover(self,steps):
        for step in steps:
            prefix=step['prefix']
            self.prefix_logp[prefix]=step['raw_prefix_logp']
            distribution=self.b.distribution(prefix)
            core=[edge for edge in distribution['accepted'] if edge['core']]
            # A sole main continuation is not a main branching decision.
            if len(core)<2:
                continue
            for edge in core:
                key=prefix+(edge['token'],)
                if key not in self.tasks:
                    self.tasks[key]=dict(prefix=list(key),parent=list(prefix),token=edge['token'],
                        logp=step['raw_prefix_logp']+math.log(edge['p']),p=edge['p'],
                        entropy=distribution['entropy_bits'],attempts=0,trials=0,probes=[],costs=[],new_weight=0.)

    def policy_score(self,task,kind):
        n=max(1,task['trials'])
        cost=sum(task['costs'])/max(1,len(task['costs']))
        score=task['logp']-math.log(max(1.,cost))-.5*math.log(1+task['trials'])
        if self.method=='token_entropy' and kind=='main':
            score+=math.log(max(1e-6,task['entropy']))
        elif self.method=='temporal_mode' or kind=='tail':
            upper=min(1.,task['new_weight']/n+math.sqrt(math.log(2+self.counts[kind])/(2*n)))
            score+=math.log(max(1e-12,upper))
        return score

    def select(self,tasks,kind):
        count=self.counts[kind]
        initial=CONFIG['main_initialization_cap'] if kind=='main' else CONFIG['tail_initialization_cap']
        unseen=[task for task in tasks if task['trials']==0]
        periodic=count>initial and count%CONFIG['coverage_every']==0
        if count<initial or (periodic and unseen):
            self.last_scheduler='initialization' if count<initial else 'coverage'
            return max(unseen or tasks,key=lambda task:(task['logp'],-task['trials'],tuple(-v for v in task['prefix'])))
        self.last_scheduler='adaptive'
        return max(tasks,key=lambda task:(self.policy_score(task,kind),tuple(-v for v in task['prefix'])))

    def perform(self,task,kind,reason):
        try:
            super().probe(task,kind,self.budgets[kind])
        except BudgetEnd:
            self.exhausted[kind]='token_guard'
            return False
        self.counts[kind]+=1
        self.scheduler[kind][reason]+=1
        self.draws[-1].update(kind_draw=self.counts[kind],scheduler=reason)
        return True

    def advance_main(self,target):
        if 'main' in self.exhausted:
            return
        while self.counts['main']<target:
            if self.counts['main']<CONFIG['root_probes']:
                task,reason=None,'root'
            elif self.tasks:
                task=self.select(list(self.tasks.values()),'main')
                reason=self.last_scheduler
            else:
                self.exhausted['main']='no_main_branch'
                return
            if not self.perform(task,'main',reason):
                return

    def refresh_tail(self):
        # Only prefixes actually encountered by main at the current checkpoint.
        for prefix,logp in self.prefix_logp.items():
            for edge in self.b.distribution(prefix)['accepted']:
                if edge['core']:
                    continue
                key=prefix+(edge['token'],)
                if key not in self.tail_tasks:
                    self.tail_tasks[key]=dict(prefix=list(key),parent=list(prefix),token=edge['token'],
                        logp=logp+math.log(edge['p']),p=edge['p'],entropy=0.,
                        attempts=0,trials=0,probes=[],costs=[],new_weight=0.)

    def advance_tail(self,target):
        self.refresh_tail()
        if self.exhausted.get('tail')=='token_guard':
            return
        while self.counts['tail']<target and self.tail_tasks:
            task=self.select(list(self.tail_tasks.values()),'tail')
            if not self.perform(task,'tail',self.last_scheduler):
                return

    def state(self,kind,target):
        tasks=list(self.tasks.values()) if kind=='main' else list(self.tail_tasks.values())
        registry=self.main if kind=='main' else self.tail
        budget=self.budgets[kind]
        return copy.deepcopy(dict(spent=budget.spent,cap=budget.cap,rollout_cap=target,
            rollouts=self.counts[kind],scheduler=dict(self.scheduler[kind]),
            unvisited_edges=sum(task['trials']==0 for task in tasks),unfinished_minimum=0,
            stop=self.exhausted.get(kind,'rollout_cap' if self.counts[kind]>=target else 'no_supported_entry'),
            tasks=tasks,mi=self.mode_information(tasks,registry),diagnostic_only=kind=='tail'))

    def sweep(self,save=None,checkpoints=None,judge_fn=judge):
        output=dict(method=self.method,seed=self.seed,checkpoints={},draws=self.draws)
        for target in checkpoints or CHECKPOINTS:
            self.advance_main(target)
            frozen=copy.deepcopy(self.main.summary())
            before=digest(frozen)
            main=self.state('main',target)
            self.notify(stage='model_judgment',method=self.method,seed=self.seed,rollout_budget=2*target,
                        main_rollouts=self.counts['main'],tail_rollouts=self.counts['tail'])
            selection=judge_fn(self.b,self.sample,frozen)
            self.advance_tail(target)
            assert digest(self.main.summary())==before,'Tail changed main candidate state'
            output['checkpoints'][str(target)]=dict(method=self.method,seed=self.seed,budget_per_kind=target,
                total_rollout_budget=2*target,main=main,tail=self.state('tail',target),
                main_modes=frozen,tail_modes=copy.deepcopy(self.tail.summary()),selection=selection,
                draw_cutoff=len(self.draws),main_candidate_hash=before,tail_isolation_verified=True)
            self.notify(stage='budget_checkpoint',method=self.method,seed=self.seed,rollout_budget=2*target,
                main_rollouts=self.counts['main'],tail_rollouts=self.counts['tail'],
                adaptive_main_selections=self.scheduler['main']['adaptive'])
            if save:
                save(output)
        return output
