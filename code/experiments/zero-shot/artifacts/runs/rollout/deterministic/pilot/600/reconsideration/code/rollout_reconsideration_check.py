"""GT-leakage, budget topology, full600 input references, and frozen cohort checks."""
import copy,json,math
from pathlib import Path
from rollout_reconsideration import *

def run():
 cohort=read(COHORT);assert len(cohort)==600
 assert read(ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json')[:600]==cohort
 counts=collections.Counter(s['dataset'] for s in cohort);assert sorted(counts.values())==[200,200,200]
 assert all(s['sample_key']!='qvhighlights-timelens_00507' for s in cohort)
 failures=[]
 for s in cohort:
  key=s['sample_key'];rec=read(SOURCE/'rollouts'/f'{key}.json');trace=SOURCE/'token_traces'/key/'index.json';inp=read(SOURCE/'inputs'/f'{key}.json');cache_path=relocate_path(Path(inp['cache_metadata']));cache=read(cache_path)
  assert rec['sample']==s and trace.exists() and relocate_path(Path(cache['array'])).is_file()
  assert inp['hashes']==rec['input_hashes']
  censored=copy.deepcopy(rec);censored['sample']['span']=[[99990,99999]]
  assert select_probes(rec)==select_probes(censored)
  cs=rec['result']['candidates'];assert len({tuple(c['ids']) for c in cs})==len(cs)
  for n in (4,8):
   chosen=diverse(cs,n);assert len({tuple(interval(c)) for c in chosen})==len(chosen)
  _,a=clusters(cs);assert all(c['index'] in a for c in cs if interval(c))
 # Budget arithmetic uses a fake deterministic judge, testing orchestration without GT or a GPU.
 rec=read(SOURCE/'rollouts'/(cohort[0]['sample_key']+'.json'));trace=read(SOURCE/'token_traces'/cohort[0]['sample_key']/'index.json')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 class Fake:
  def __init__(self):self.tok=tok;self.logical_calls=[];self.prompts=[]
  def snapshot(self):return dict(model_evaluations=len(self.logical_calls))
  def choice(self,text,options,kind):self.prompts.append(text);self.logical_calls.append(kind);return dict(conditional_probabilities=[.6,.4],raw_probabilities=[.3,.2],option_mass=.5)
  def generate(self,turns,limit,kind,json_stop=False):
   self.prompts.append(str(turns));self.logical_calls.append(kind)
   text='{"action":"boundary","a":1,"b":1}' if json_stop else 'The event happens in 1.0 - 2.0 seconds.'
   return dict(text=text,ids=[1],parsed=parse(text),stopped_on_eos=True,context_ids=[])
 for name,method,pool,amount in VARIANTS:
  f=Fake();out=run_variant(f,rec,trace,name,method,pool,amount,[])
  g=Fake();redacted=copy.deepcopy(rec);redacted['sample']['span']=[[99990,99999]];run_variant(g,redacted,trace,name,method,pool,amount,[]);assert f.prompts==g.prompts
  cap=1 if method=='direct' else 2*amount+1 if method in ('agent','verifier') else 2*(amount-1)+1
  assert out['cost_decision']['model_evaluations']<=cap
 assert iou([0,10],[5,15])==1/3 and iou([0,1],[3,4])==0
 write(OUT/'validation/cpu.json',dict(passed=True,queries=600,dataset_counts=dict(counts),all_input_and_trace_references_exist=True,cohort_unchanged=True,gt_redaction_leaves_probes_and_prompts_unchanged=True,topology_budgets_checked=True,reference_case_excluded=True))
 print('CPU CHECK PASSED',flush=True)
if __name__=='__main__':run()
