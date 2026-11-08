"""Frozen native-probability search specification and scalar statistics."""
import collections,datetime,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())
OUT=ROOT/'artifacts/runs/rollout/probability_budget/pilot/900'
SOURCE=ROOT/'artifacts/runs/rollout/deterministic/pilot/900'
COHORT=ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'
MODEL=Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct')
METHODS=['levin','native_mass','numeric_novelty']
BUDGETS=[512,1024,2048]
CONFIG=dict(family='rollout/probability_budget',queries=900,stages=[300,900],methods=METHODS,budgets=BUDGETS,default_budget=1024,per_query_per_method_cap=2048,early_stop_unseen_valid_mass=.01,top_p=.99,absolute_floor=.01,temperature=1.,sampling=False,inference_seeds=None,frame_sampling_fps=2,input_order='video_query',input_policy='reuse original frame IDs, PTS, HxW and tensor hashes',branch_policy='integer digits and decimal separator only; all other positions greedy; no fractional branch entropy',grammar='no extra grammar mask',clipping='[0,actual metadata duration]; require start<end and EOS; do not sort reversed endpoints',levin='global priority log(answer token depth) minus native log path probability; compare every token, including greedy positions',allocation='native frontier mass weighted service credits; root first numeric fork defines disjoint fixed groups; best native prefix then greedy completion within group',allocation_quantum=32,novelty_lambda=.1,novelty='native-mass-weighted mean of 1-max temporal IoU against completed intervals in other groups; unknown=0 bonus; native mass never divided by candidate count',kv_suffix_cache_bytes=8*2**30,checkpoint_policy='one cumulative search; actual decoder input positions including all replay; partial prefix materialization remains pending',entropy='exact clipped endpoint pair, conditional on observed valid mass; no entropy rebinning',scope='candidate proposal only; native-mass answer is a diagnostic proxy, not an independent verifier',prefill='shared once per query; attributed once per standalone method comparison; separately measured',time_limit=None)
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def iou(a,b):return max(0.,min(a[1],b[1])-max(a[0],b[0]))/max(1e-12,max(a[1],b[1])-min(a[0],b[0])) if a and b else 0.
def interval_distribution(leaves):
 masses=collections.defaultdict(float)
 for leaf in leaves:
  if leaf['valid'] and leaf['logp']> -740:masses[tuple(leaf['span'])]+=math.exp(leaf['logp'])
 total=sum(masses.values());out=[dict(id=digest(list(k))[:20],span=list(k),native_mass=v,weight=v/total) for k,v in sorted(masses.items())] if total else []
 if not out:return out,dict(entropy=None,mean=None,covariance=None)
 x=np.asarray([r['span'] for r in out]);w=np.asarray([r['weight'] for r in out]);mean=w@x;delta=x-mean;cov=(delta*w[:,None]).T@delta
 return out,dict(entropy=float(-(w*np.log2(w)).sum()),mean=mean.tolist(),covariance=cov.tolist())
def selected_interval(leaves):
 intervals,_=interval_distribution(leaves)
 return max(intervals,key=lambda r:(r['native_mass'],tuple(-x for x in r['span'])))['span'] if intervals else None
