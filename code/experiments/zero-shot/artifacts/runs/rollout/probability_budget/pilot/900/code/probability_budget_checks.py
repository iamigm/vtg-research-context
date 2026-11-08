"""Meaningful deterministic tree, budget, conservation and scope checks."""
from probability_budget_core import *
def cpu_check():
 import torch
 import probability_budget_search as search
 from probability_budget_backend import BudgetBoundary
 class Tok:
  pieces={0:'',1:'The event happens in ',2:'1',3:'2',4:'.',5:'0',6:' - ',7:'4',8:'5',9:' seconds.',10:'<|im_end|>'}
  def decode(self,ids,skip_special_tokens=False,**kw):return ''.join(self.pieces.get(i,'x') for i in ids if not(skip_special_tokens and i==10))
 class Fake:
  tokenizer=Tok();n=100;max_context=1000;eos={10}
  def reset(self,limit):self.tokens=0;self.limit=limit
  def costs(self):return dict(native_forwards=self.tokens,decoder_inputs=self.tokens,replay_inputs=0,decode_cuda_seconds=0.,kv_restore_cuda_seconds=0.,kv_store_cuda_seconds=0.,kv_cache_wall_seconds=0.,kv_restore_hits=0,kv_evictions=0,kv_suffix_cache_bytes=0)
  def get(self,path):
   if path:
    if self.tokens>=self.limit:raise BudgetBoundary()
    self.tokens+=1
   t=self.tokenizer.decode(path);role=search.numeric_role(t)
   if not path:options={1:1.}
   elif role=='start.integer_or_decimal' and t.endswith('in '):options={2:.7,3:.3}
   elif role=='end.integer_or_decimal' and t.endswith(' - '):options={7:.8,8:.2}
   elif role.endswith('integer_or_decimal'):options={4:1.}
   elif t.endswith(' seconds.'):options={10:1.}
   elif t.endswith('.'):options={5:.6,8:.4}
   elif ' - ' not in t:options={6:1.}
   else:options={9:1.}
   result=torch.full((40,),-torch.inf)
   for token,p in options.items():result[token]=math.log(p)
   return result
 old=search.BUDGETS
 try:
  search.BUDGETS=[8,16,128];results={}
  for method in METHODS:
   result=search.Search(Fake(),method,10.).run();final=result['snapshots']['128'];assert final['reason']=='mass_target_reached';assert len([l for l in result['leaves'] if l['valid']])==4;assert abs(final['mass']['valid']-.36)<1e-6;assert abs(final['mass_total']-1)<1e-6
   for budget,s in result['snapshots'].items():assert s['actual_inputs']<=int(budget)
   for n in result['nodes']:
    if 'entropy' in n:assert n['role'].endswith('integer_or_decimal')
   for g in final['groups']:assert 0<=g['novelty']<=1
   results[method]=dict(inputs=final['actual_inputs'],valid_mass=final['mass']['valid'])
  a=search.eligible(np.array([.97,.015,.009,.006]));assert [v['token'] for v in a]==[0,1]
 finally:search.BUDGETS=old
 return dict(passed=True,synthetic_tree=results,strict_global_levin=True,exact_budget_cap=True,partial_prefix_stays_pending=True,fractional_branching_disabled=True,native_mass_conserved=True)

def validate_scope(count):
 rows=read(COHORT)[:count];assert rows==read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{count}/samples.json')
 for row in rows:
  data=read(OUT/'queries'/f'{row["sample_key"]}.json');assert data['complete'] and data['config_hash']==digest(CONFIG)
  for method in METHODS:
   m=read(OUT/'search'/row['sample_key']/f'{method}.json');assert m['complete'] and m['config_hash']==digest(CONFIG)
   for budget,s in m['snapshots'].items():
    assert s['actual_inputs']<=int(budget) and abs(s['mass_total']-1)<5e-4
    assert s['reason']!='mass_target_reached' or s['unseen_valid_mass_upper_bound']<=.01
    if method=='levin':assert s['levin_order_verified']
   for node in m['nodes']:
    if 'entropy' in node:assert node['role'].endswith('integer_or_decimal')
  hashes=read(SOURCE/'inputs'/f'{row["sample_key"]}.json')['hashes'];assert data['input']['hashes']==hashes
 write(OUT/'validation'/f'raw_pilot{count}.json',dict(passed=True,queries=count,methods=METHODS,budgets=BUDGETS,mass_conservation=True,input_hashes=True,no_fractional_branches=True,timestamp=stamp()))
if __name__=='__main__':print(cpu_check())
