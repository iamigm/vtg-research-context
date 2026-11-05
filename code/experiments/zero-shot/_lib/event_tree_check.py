"""CPU checks for budget accounting, backtracking, tree invariants and coverage math."""
from event_tree_core import *
import collections,itertools

class FakeBackend:
 def __init__(self):self.records={};self.meter()
 def meter(self,cap=None):self.active=[];self.spent=0;self.cap=cap
 def cost(self,refs=None):
  refs=self.active if refs is None else refs
  return dict(decoder_input_tokens=50*len(refs),model_seconds=0.,forward_calls=len(refs),generated_tokens=0,model_evaluations=len(refs))
 def take(self,kind,prompt,result):
  if self.cap is not None and self.spent+50>self.cap:raise BudgetEnd()
  key=signature([kind,prompt]);r=dict(id=key,kind=kind,cost=dict(decoder_input_tokens=50,model_seconds=0.,forward_calls=1,generated_tokens=0),**result);self.records[key]=r;self.active.append(key);self.spent+=50;return r
 def generate(self,prompt,limit=64,kind='generation',json_kind=None):
  if kind=='boundary_refinement':
   span=json.loads(re.search(r'Current occurrence interval: (\[[^]]+\])',prompt).group(1));text=f'The event happens in {span[0]} - {span[1]} seconds.'
   result=dict(text=text,ids=[1,2],parsed=parse(text),stopped_on_eos=True)
  elif kind=='new_event_proposal':result=dict(text='',json=dict(start=20,end=22,evidence='another action'))
  elif kind=='neural_controller':result=dict(text='',json=dict(action='refine',node=1,focus='joint'))
  else:result=dict(text='Visible action contradicts the proposed occurrence.')
  return self.take(kind,prompt,result)
 def choice(self,prompt,kind='critic'):
  wrong='Candidate interval: [0, 2]' in prompt
  p=.99 if wrong and kind in ('event_contradiction','contradiction_confirmation') else .02 if kind in ('event_contradiction','contradiction_confirmation') else .95
  return self.take(kind,prompt,dict(probabilities=[p,1-p],raw_probabilities=[.8*p,.8*(1-p)],option_mass=.8))
 def prior(self,span):return self.take('prior',str(span),dict(log_probability=-2-span[0]/100))
 def operation(self,kind,prompt,**kw):return self.take(kind,prompt,dict(probabilities=[.7,.3],raw_probabilities=[.56,.24],option_mass=.8))

def run():
 cohort=read(COHORT);assert len(cohort)==900;assert collections.Counter(r['dataset'] for r in cohort)==dict.fromkeys(['charades-timelens','activitynet-timelens','qvhighlights-timelens'],300)
 for n in (300,600):assert cohort[:n]==read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json')
 for row in cohort:
  rec=read(SOURCE/'rollouts'/f'{row["sample_key"]}.json');inp=read(SOURCE/'inputs'/f'{row["sample_key"]}.json');assert rec['sample']==row;assert rec['input_hashes']==inp['hashes'];assert relocate_path(Path(inp['cache_metadata'])).exists()
  altered=dict(row,span=[[-9999,99999]],iou=1,answer='not a video fact');assert public_sample(row)==public_sample(altered);assert 'span' not in public_sample(row)
 assert parse('The event happens in 01:59 - 02:16 seconds.')['start']==119
 assert parse('The event happens in 1'+'19 - 136 seconds.')['start']==119
 roots=[]
 for i,(span,value) in enumerate([([0,2],.95),([10,12],.8),([30,34],.7),([30.2,34.1],.69),([70,80],.3),([90,98],.1)]):roots.append(dict(id=i+1,span=span,assessment=dict(value=value,event=value,conflict=.01,boundary=.5,eligible=True,blocked=False,reason=None,calls=[]),prior_logp=-i-1,provenance=[dict(source='token_seed')],source='token_seed',evidence=''))
 duration=100;subsets=[list(c) for k in range(len(roots)+1) for c in itertools.combinations(roots,k)]
 for a in subsets:
  fa=coverage_value(roots,a,duration)
  for node in roots:
   if node in a:continue
   increment=coverage_value(roots,a+[node],duration)-fa;assert increment>=-1e-12
   for b in subsets:
    if node in b or not all(x in b for x in a):continue
    assert increment+1e-12>=coverage_value(roots,b+[node],duration)-coverage_value(roots,b,duration)
 selected=submodular_select(roots,4,duration);greedy=coverage_value(roots,selected,duration);optimal=max(coverage_value(roots,list(c),duration) for c in itertools.combinations(roots,4));assert greedy>=optimal*(1-1/math.e)-1e-12
 sample=dict(sample_key='fake',dataset='test',query='target',duration=100,video_path='not_used');proposals=dict(events=roots[:2]);all_results={}
 for method in METHODS:
  result=Search(FakeBackend(),sample,proposals,method).run();all_results[method]=result
  for cap,r in result.items():
   assert r['search_cost']['decoder_input_tokens']<=int(cap);assert r['selection_cost']['decoder_input_tokens']<=CONFIG['selection_token_cap'];assert r['nodes'][0]['visits']==r['expansions']+r['proposal_rounds']
   assert all(n['parent']<n['id'] for n in r['nodes'][1:]);assert all(n['event_id']<=n['id'] for n in r['nodes'][1:])
   assert not r['selected'] or not r['nodes'][r['nodes'][r['selected']]['event_id']]['assessment']['blocked']
 value=all_results['value_best_first']['8192'];assert any(a['action']=='contradiction_backtrack' for a in value['actions']);assert value['span'] and value['span'][0]>0
 committed=all_results['commit_then_refine']['8192'];assert committed['span'] is None
 b=FakeBackend();b.meter(100);x=b.choice('same');b.choice('same');assert b.spent==100 and len(b.active)==2 and b.active[0]==b.active[1]
 try:b.choice('same');raise AssertionError('Budget must stop cache replay too')
 except BudgetEnd:pass
 from event_tree_runtime import verify_worker_exit
 class PollOnly:
  pid=123
  def __init__(self,value):self.value=value
  def poll(self):return self.value
 verify_worker_exit(PollOnly(None));verify_worker_exit(PollOnly(0))
 try:verify_worker_exit(PollOnly(1));raise AssertionError('Failed worker must raise')
 except RuntimeError:pass
 result=dict(passed=True,queries=900,methods=len(METHODS),source_inputs_verified=True,cohort_prefixes_identical=True,public_input_gt_redaction=True,clock_parser=True,leading_digit_prefix_reconstruction=True,tree_backup_invariants=True,cache_cost_charged=True,budget_caps=True,contradiction_backtracking=True,commit_ablation=True,submodularity_exhaustive_subset_check=True,coverage_greedy=greedy,coverage_optimal=optimal,coverage_ratio=greedy/optimal,timestamp=stamp())
 write(OUT/'validation/cpu.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':run()
