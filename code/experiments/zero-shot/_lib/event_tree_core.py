"""Frozen-VLM event proposals, visual critics and budgeted deterministic tree search."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import copy,gc,hashlib,json,math,re,time
from pathlib import Path
from deterministic_rollout300 import parse,sites
from result_layout import relocate_path

ROOT=Path('/home/irteam/experiments/zero-shot')
FAMILY='rollout/event_tree_search'
OUT=ROOT/'artifacts/runs'/FAMILY/'pilot/900'
REPORT=ROOT/'results'/FAMILY/'pilot/900'
SOURCE=ROOT/'artifacts/runs/rollout/deterministic/pilot/900'
COHORT=ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'
MODEL=Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct')
METHODS=['likelihood_best_first','value_best_first','value_beam','diverse_beam','uct','puct','neural_controller','self_refine','submodular_coverage','reward_balanced','sequential_halving','nucleus_frontier','typicality_frontier','commit_then_refine','token_seed_best_first']
CONFIG=dict(family=FAMILY,queries=900,cohort=str(COHORT),model=str(MODEL),fps=2,target_pixels=100352,input_order='video_query',
 methods=METHODS,search_token_caps=[4096,8192],selection_token_cap=2048,seed_candidates=8,leading_digits=list(range(10)),semantic_proposals=6,
 proposal_max_tokens=320,answer_max_tokens=64,observation_max_tokens=48,controller_max_tokens=48,max_depth=3,children_per_node=3,beam_width=4,diversity_penalty=.25,
 event_gate=.5,contradiction_threshold=.8,minimum_binary_mass=.01,exploration=1.4,progressive_widening='min(root_count,2+floor(2*sqrt(root_visits+1)))',
 coverage_k=4,coverage_quality_weight=.2,coverage_uniform_weight=.1,nucleus_mass=.95,typical_mass=.95,reward_temperature=.25,halving_initial_arms=8,
 decoder='deterministic greedy completions; explicit unique forced digit branches; no random sampling',
 budget='new text decoder inputs including prompt replay and generation; common proposals separately charged; selectors reserve2048; diagnostics excluded and separately logged',
 critic='same frozen VLM; binary normalized probabilities and raw masses; decisive contradiction requires two independent prompts',
 reuse='exact per-query operation cache; conceptual cold-cache cost charged on every use, executed cache cost separately recorded',
 event_identity='temporal occurrence proxy, not semantic ground truth; same-context token trie nested in context/action search tree',
 version_identity='event_proposal_then_boundary_backtracking')

def read(p):return json.loads(relocate_path(Path(p)).read_text())
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def signature(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(relocate_path(Path(p)).read_bytes()).hexdigest()
def stamp():
 import datetime
 return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def iou(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0]))/max(1e-12,max(a[1],b[1])-min(a[0],b[0])) if a and b else 0.
def pair(c):
 p=c.get('parsed')
 return [p['start'],p['end']] if p and p.get('positive') and c.get('stopped_on_eos',True) else None
def public_sample(row):return {k:row[k] for k in ('sample_key','dataset','video_path','query','duration')}
def json_value(text,kind='object'):
 begin='{' if kind=='object' else '[';decoder=json.JSONDecoder()
 for m in re.finditer(re.escape(begin),text):
  try:return decoder.raw_decode(text[m.start():])[0]
  except ValueError:pass
 return None
class BudgetEnd(Exception):pass

class Backend:
 """One video prefill; fresh text KV per operation, exact operation memoization."""
 def __init__(self,model,inputs,torch,tok,eos):
  self.model=model;self.inputs=inputs;self.torch=torch;self.tok=tok;self.eos=eos;self.n=inputs.input_ids.shape[1];self.records={};self.active=[];self.spent=0;self.cap=None;self.actual_tokens=0;self.actual_seconds=0.;self.actual_forwards=0;self.cache_hits=0
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode():o=model(**inputs,use_cache=True,logits_to_keep=1)
  self.cache=o.past_key_values;self.root=o.logits[0,-1].float().cpu();torch.cuda.synchronize();self.prefill_seconds=time.perf_counter()-t;del o
  self.end_turn=tok.convert_tokens_to_ids('<|im_end|>');assert self.end_turn in eos
  self.numeric_ids={c:tok.encode(c,add_special_tokens=False)[0] for c in '0123456789.:'}
  self.reset_execution()
 def reset_execution(self):self.cache.crop(self.n);self.path=();self.last=self.root
 def meter(self,cap=None):self.active=[];self.spent=0;self.cap=cap
 def cost(self,refs=None):
  refs=self.active if refs is None else refs
  return dict(decoder_input_tokens=sum(self.records[r]['cost']['decoder_input_tokens'] for r in refs),model_seconds=sum(self.records[r]['cost']['model_seconds'] for r in refs),forward_calls=sum(self.records[r]['cost']['forward_calls'] for r in refs),model_evaluations=len(refs),generated_tokens=sum(self.records[r]['cost'].get('generated_tokens',0) for r in refs))
 def dialog(self,text):
  # Close the empty original assistant turn, not a saved completed answer.
  return [self.end_turn]+self.tok.encode('\n'+self.tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True),add_special_tokens=False)
 def get(self,path,all_logits=False):
  path=tuple(path)
  if not all_logits and path==self.path:return self.last
  common=0
  for a,b in zip(path,self.path):
   if a!=b:break
   common+=1
  if common==len(path):common=max(0,common-1)
  self.cache.crop(self.n+common);torch=self.torch
  for start in range(common,len(path),128):
   ids=path[start:start+128];size=len(ids);torch.cuda.synchronize();t=time.perf_counter()
   with torch.inference_mode():o=self.model(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,self.n+start+size),device='cuda',dtype=self.inputs.attention_mask.dtype),past_key_values=self.cache,cache_position=torch.arange(self.n+start,self.n+start+size,device='cuda'),use_cache=True,logits_to_keep=0 if all_logits else 1)
   self.cache=o.past_key_values;self.path=path[:start+size];self.last=o.logits[0,-1].float().cpu();torch.cuda.synchronize();self.actual_seconds+=time.perf_counter()-t;self.actual_tokens+=size;self.actual_forwards+=1
   if all_logits:return o.logits[0].float()
   del o
  return self.last
 def operation(self,kind,prompt,limit=0,options=None,prefix=None,json_kind=None,score_ids=None):
  key=signature(dict(kind=kind,prompt=prompt,limit=limit,options=options,prefix=prefix,json_kind=json_kind,score_ids=score_ids))
  context=self.dialog(prompt) if prefix is None else list(prefix)
  upper=len(context)+(len(score_ids) if score_ids else limit)
  if self.cap is not None and self.spent+upper>self.cap:raise BudgetEnd()
  if key in self.records:
   self.cache_hits+=1;r=self.records[key]
  else:
   self.reset_execution();before=(self.actual_tokens,self.actual_seconds,self.actual_forwards);torch=self.torch
   r=dict(id=key,kind=kind,prompt=prompt,context_ids=context,context_hash=signature(context),options=options)
   if score_ids:
    ids=score_ids;lp0=torch.log_softmax(self.root,-1)[ids[0]].item()
    logits=self.get(ids[:-1],all_logits=True);lp=torch.log_softmax(logits,-1);targets=torch.tensor(ids[1:],device=lp.device);values=lp.gather(1,targets[:,None]).squeeze(1).cpu().tolist();del logits,lp
    vals=[lp0]+values;r.update(ids=ids,log_probabilities=vals,log_probability=sum(vals),mean_log_probability=sum(vals)/len(vals))
   elif options:
    lp=torch.log_softmax(self.get(context),-1);ids=[self.tok.encode(x,add_special_tokens=False) for x in options];assert all(len(x)==1 for x in ids)
    raw=[math.exp(float(lp[x[0]])) for x in ids];mass=sum(raw);r.update(raw_probabilities=raw,probabilities=[x/mass if mass else 1/len(raw) for x in raw],option_mass=mass)
   else:
    ids=[];tokens=[];logp=0.;eos=False
    for j in range(limit):
     lp=torch.log_softmax(self.get(context+ids),-1);token=int(lp.argmax());v=float(lp[token]);logp+=v;top=torch.topk(lp,8)
     tokens.append(dict(step=j,id=token,token=self.tok.decode([token]),probability=math.exp(v),cumulative_log_probability=logp,top=[dict(id=int(i),token=self.tok.decode([int(i)]),probability=math.exp(float(x))) for x,i in zip(top.values,top.indices)]));ids.append(token)
     if kind in ('boundary_refinement','leading_digit_rollout'):tokens[-1]['numeric_probabilities']={c:math.exp(float(lp[i])) for c,i in self.numeric_ids.items()}
     if token in self.eos:eos=True;break
     if json_kind and json_value(self.tok.decode(ids,skip_special_tokens=True),json_kind) is not None:break
    text=self.tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    r.update(text=text,ids=ids,tokens=tokens,log_probability=logp,parsed=parse(text),stopped_on_eos=eos,json=json_value(text,json_kind) if json_kind else None)
   # Consume the final sampled token only when needed by a later operation; cost is actual decoder input.
   r['cost']=dict(decoder_input_tokens=self.actual_tokens-before[0],model_seconds=self.actual_seconds-before[1],forward_calls=self.actual_forwards-before[2],generated_tokens=len(r.get('ids',[])) if not score_ids else 0)
   assert r['cost']['decoder_input_tokens']<=upper,(kind,r['cost'],upper)
   self.records[key]=r
  self.active.append(key);self.spent+=r['cost']['decoder_input_tokens'];return r
 def generate(self,prompt,limit=64,kind='generation',json_kind=None):return self.operation(kind,prompt,limit=limit,json_kind=json_kind)
 def choice(self,prompt,kind='critic'):return self.operation(kind,prompt,options=['Yes','No'])
 def prior(self,span):
  text=f'The event happens in {span[0]:g} - {span[1]:g} seconds.';ids=self.tok.encode(text,add_special_tokens=False)+[self.end_turn]
  return self.operation('canonical_path_prior','',prefix=[],score_ids=ids)

def usable(span,duration):return bool(span and 0<=span[0]<span[1]<=duration+1e-3)
def assess(b,sample,span,refined=False):
 if not usable(span,sample['duration']):return dict(event=0.,conflict=1.,boundary=0.,value=0.,eligible=False,blocked=True,reason='invalid_or_out_of_duration',calls=[])
 text=f"Target event: {sample['query']}\nCandidate interval: [{span[0]:g}, {span[1]:g}] seconds. Refer to the video. "
 start=len(b.active)
 event=b.choice(text+'Does this interval contain the requested action with the correct actor and object? Answer Yes or No only.','event_presence')
 conflict=b.choice(text+'Is there decisive visual evidence that this is a different action, actor, object, or occurrence, rather than merely an imprecise boundary? Answer Yes or No only.','event_contradiction')
 boundary=b.choice(text+'Do the onset and offset accurately delimit the target action, without missing the action or adding unrelated portions? Answer Yes or No only.','boundary_fit') if refined else None
 p,c=event['probabilities'][0],conflict['probabilities'][0];confirmed=False;confirmation=None
 if c>=CONFIG['contradiction_threshold'] and conflict['option_mass']>=CONFIG['minimum_binary_mass']:
  confirmation=b.choice(text+'Check the visible action again independently: would changing only its start/end fail to fix a fundamental mismatch with the target? Answer Yes or No only.','contradiction_confirmation')
  confirmed=confirmation['probabilities'][0]>=CONFIG['contradiction_threshold'] and confirmation['option_mass']>=CONFIG['minimum_binary_mass']
 bp=boundary['probabilities'][0] if boundary else .5;eligible=p>=CONFIG['event_gate'] and not confirmed
 return dict(event=p,conflict=c,boundary=bp,value=p*(1-c)*(.5+.5*bp) if eligible else 0.,eligible=eligible,blocked=confirmed,reason='model_confirmed_contradiction' if confirmed else None,calls=b.active[start:])

def initial_proposals(b,sample,record):
 start=len(b.active);events=[];paths=[]
 def add(span,source,evidence='',call=None):
  if span is None:return
  old=next((x for x in events if x['span']==list(span)),None)
  provenance=dict(source=source,call=call,evidence=evidence)
  if old:old['provenance'].append(provenance);return
  events.append(dict(id=len(events)+1,span=list(span),source=source,evidence=evidence,provenance=[provenance]))
 cs=record['result']['candidates']
 for c in cs[:8]:add(pair(c),'token_seed',call='baseline:'+str(c['index']))
 baseline=cs[0];numeric=sites(baseline['ids'],b.tok);first=numeric[0][0] if numeric else None
 if first is not None:
  prefix=baseline['ids'][:first]
  distribution=b.operation('leading_digit_distribution','',prefix=prefix,options=list('0123456789'))
  for digit in '0123456789':
   ids=b.tok.encode(digit,add_special_tokens=False);assert len(ids)==1
   r=b.operation('leading_digit_rollout','',prefix=prefix+ids,limit=64)
   full_ids=prefix+ids+r['ids'];full_text=b.tok.decode(full_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
   score=b.operation('original_digit_path','',prefix=[],score_ids=full_ids)
   span=pair(dict(parsed=parse(full_text),stopped_on_eos=r['stopped_on_eos']));paths.append(dict(digit=digit,prefix_ids=prefix,forced_token=ids[0],forced_probability=distribution['raw_probabilities'][int(digit)],distribution_call=distribution['id'],call=r['id'],score_call=score['id'],prefix_log_probability=sum(score['log_probabilities'][:len(prefix)]),path_log_probability=score['log_probability'],full_ids=full_ids,answer=full_text,span=span))
   add(span,'leading_digit',call=r['id'])
 prompt=f"Target event: {sample['query']}\nVideo duration: {sample['duration']:g} seconds. Identify up to6 genuinely plausible, distinct occurrences anywhere in this video, including lower-confidence alternatives. Use visible evidence, not just nearby decimal variants. Do not invent an occurrence to fill the list. Return a JSON array ordered by time: [{{\"start\":number,\"end\":number,\"evidence\":\"at most8 words\"}}]. All times are seconds."
 r=b.generate(prompt,CONFIG['proposal_max_tokens'],'multi_event_proposal','array')
 items=r['json'] if isinstance(r.get('json'),list) else []
 if not items:
  for m in re.finditer(r'\{[^{}]*\}',r.get('text','')):
   try:items.append(json.loads(m.group()))
   except ValueError:pass
 for item in items[:6]:
  try:span=[float(item['start']),float(item['end'])];assert all(math.isfinite(x) for x in span)
  except (KeyError,TypeError,ValueError,AssertionError):continue
  add(span,'semantic_proposal',str(item.get('evidence',''))[:200],r['id'])
 for event in events:
  event['assessment']=assess(b,sample,event['span'])
  if usable(event['span'],sample['duration']):
   pr=b.prior(event['span']);event['prior_logp']=pr['log_probability'];event['prior_mean_logp']=pr['mean_log_probability'];event['prior_call']=pr['id']
  else:event.update(prior_logp=-1e6,prior_mean_logp=-1e6)
 return dict(events=events,leading_digit_paths=paths,semantic_call=r['id'],calls=b.active[start:])

def diversity_select(nodes,n,penalty=.25):
 selected=[];remaining=list(nodes)
 while remaining and len(selected)<n:
  chosen=max(remaining,key=lambda x:(x['assessment']['value']-penalty*max((iou(x['span'],z['span']) for z in selected),default=0),-x['id']))
  selected.append(chosen);remaining.remove(chosen)
 return selected

def coverage_value(universe,selected,duration):
 """Normalized nonnegative monotone submodular facility-location + modular quality."""
 if not selected:return 0.
 quality=[max(n['assessment']['value'],0.) for n in universe];denom=sum(quality) or 1.;count=max(1,len(universe));tau=max(2.,.05*duration)
 total=0.
 for node,q in zip(universe,quality):
  weight=.9*q/denom+.1/count
  similarity=max(.5*iou(node['span'],other['span'])+.5*math.exp(-abs(sum(node['span'])/2-sum(other['span'])/2)/tau) for other in selected)
  total+=weight*similarity
 return total+.2*sum(n['assessment']['value'] for n in selected)

def submodular_select(nodes,k,duration):
 selected=[];remaining=list(nodes)
 while remaining and len(selected)<k:
  best=max(remaining,key=lambda n:(coverage_value(nodes,selected+[n],duration),-n['id']));selected.append(best);remaining.remove(best)
 return selected

class Search:
 def __init__(self,backend,sample,proposals,method):
  self.b=backend;self.sample=sample;self.method=method;self.nodes=[dict(id=0,parent=None,kind='root',depth=0,children=[],visits=0,total_value=0.,expanded=0)];self.log=[];self.current=None;self.committed_event=None;self.beam=[];self.halving_active=[];self.halving_target=1;self.expansions=0;self.proposal_rounds=0;self.max_depth=CONFIG['max_depth']
  for event in proposals['events']:
   if method=='token_seed_best_first' and not any(p['source']=='token_seed' for p in event['provenance']):continue
   self.add(None,event['span'],'event',event['assessment'],source_event=event)
  self.initial_count=len(self.nodes);self.b.meter()
 def add(self,parent,span,kind,assessment,call=None,observation=None,source_event=None):
  parent=0 if parent is None else parent;index=len(self.nodes);depth=self.nodes[parent]['depth']+1
  event_id=index if kind in ('event','event_switch') else self.nodes[parent]['event_id']
  node=dict(id=index,parent=parent,kind=kind,event_id=event_id,depth=depth,span=span,assessment=copy.deepcopy(assessment),children=[],visits=0,total_value=0.,expanded=0,call=call,observation=observation,prior_logp=source_event.get('prior_logp',-100.) if source_event else self.nodes[parent].get('prior_logp',-100.),source_event=source_event.get('id') if source_event else None)
  self.nodes.append(node);self.nodes[parent]['children'].append(index);return node
 def available(self,node):
  if node['id']==0:return False
  root=self.nodes[node['event_id']]
  return node['assessment']['eligible'] and not root['assessment']['blocked'] and node['depth']<=self.max_depth and node['expanded']<3
 def candidates(self):return [n for n in self.nodes[1:] if n['assessment']['eligible'] and not self.nodes[n['event_id']]['assessment']['blocked']]
 def backup(self,index,value):
  while index is not None:
   n=self.nodes[index];n['visits']+=1;n['total_value']+=value;index=n['parent']
 def select(self):
  available=[n for n in self.nodes[1:] if self.available(n)]
  if not available:return None
  if self.method in ('submodular_coverage','reward_balanced','sequential_halving','nucleus_frontier','typicality_frontier'):
   roots=[n for n in self.nodes[1:] if n['event_id']==n['id'] and any(x['event_id']==n['id'] for x in available)];root=None
   if self.method=='submodular_coverage':
    pool=submodular_select(roots,CONFIG['coverage_k'],self.sample['duration']);root=min(pool,key=lambda n:(n['visits'],-n['assessment']['value'],n['id']))
   elif self.method=='reward_balanced':root=max(roots,key=lambda n:(math.exp(n['assessment']['value']/CONFIG['reward_temperature'])/(1+n['visits']),-n['id']))
   elif self.method=='sequential_halving':
    active=[n for n in roots if n['id'] in self.halving_active]
    if not active:active=diversity_select(roots,CONFIG['halving_initial_arms'],.15);self.halving_active=[n['id'] for n in active];self.halving_target=min(n['visits'] for n in active)+1
    if len(active)>1 and all(n['visits']>=self.halving_target for n in active):
     active=sorted(active,key=lambda n:(-max(x['assessment']['value'] for x in self.descendants(n['id'])) ,n['id']))[:math.ceil(len(active)/2)];self.halving_active=[n['id'] for n in active];self.halving_target=2*self.halving_target+1
    root=min(active,key=lambda n:(n['visits'],-n['assessment']['value'],n['id']))
   else:
    # Conditional distribution over this finite list of unique canonical event answers.
    unique=[];seen=set()
    for n in roots:
     if tuple(n['span']) not in seen:unique.append(n);seen.add(tuple(n['span']))
    peak=max(n['prior_logp'] for n in unique);weights=[math.exp(n['prior_logp']-peak) for n in unique];total=sum(weights);ps={n['id']:w/total for n,w in zip(unique,weights)}
    entropy=-sum(p*math.log(max(p,1e-300)) for p in ps.values())
    ordered=sorted(unique,key=lambda n:(-ps[n['id']],n['id'])) if self.method=='nucleus_frontier' else sorted(unique,key=lambda n:(abs(-math.log(max(ps[n['id']],1e-300))-entropy),n['id']))
    pool=[];mass=0.
    for n in ordered:
     pool.append(n);mass+=ps[n['id']]
     if mass>=CONFIG['nucleus_mass']:break
    root=max(pool,key=lambda n:(ps[n['id']]/(1+n['visits']),-n['id']))
   return max([n for n in available if n['event_id']==root['id']],key=lambda n:(n['assessment']['value']-.05*n['expanded'],-n['id']))
  if self.method in ('uct','puct'):
   def descend(parent):
    children=[self.nodes[i] for i in parent['children'] if any(self.available(n) for n in self.descendants(i))]
    if not children:return parent if self.available(parent) else None
    if parent['id']==0:
     ordered=diversity_select(children,len(children),.25);limit=min(len(children),2+int(2*math.sqrt(parent['visits']+1)));children=ordered[:limit]
    denom=sum(max(x['assessment']['event'],1e-6) for x in children)
    def ucb(n):
     q=n['total_value']/n['visits'] if n['visits'] else n['assessment']['value'];prior=.9*max(n['assessment']['event'],1e-6)/denom+.1/len(children)
     bonus=CONFIG['exploration']*(prior*math.sqrt(parent['visits']+1)/(1+n['visits']) if self.method=='puct' else math.sqrt(math.log(parent['visits']+2)/(1+n['visits'])))
     return q+bonus,-n['id']
    child=max(children,key=ucb)
    # Progressive child widening opens a genuinely new refinement, never a duplicate simulation.
    if self.available(child) and (not child['children'] or child['expanded']<min(3,1+int(math.sqrt(child['visits'])))):return child
    return descend(child)
   return descend(self.nodes[0])
  if self.method in ('value_beam','diverse_beam'):
   current=[self.nodes[i] for i in self.beam if self.available(self.nodes[i])]
   if not current:
    children=[n for n in available if n['parent'] in self.beam]
    layer=children if children else [n for n in available if n['depth']==min(z['depth'] for z in available)]
    current=diversity_select(layer,4,CONFIG['diversity_penalty'] if self.method=='diverse_beam' else 0);self.beam=[n['id'] for n in current]
   return min(current,key=lambda n:(n['expanded'],-n['assessment']['value'],n['id']))
  if self.method in ('self_refine','commit_then_refine'):
   if self.current is not None:
    target=self.committed_event if self.method=='commit_then_refine' else self.nodes[self.current]['event_id']
    subtree=[n for n in available if n['event_id']==target]
    if subtree:return max(subtree,key=lambda n:(n['depth'],n['assessment']['value'],-n['expanded'],-n['id']))
    if self.method=='commit_then_refine':return None
   n=max(available,key=lambda n:(n['assessment']['value'],-n['id']));self.current=n['id']
   if self.method=='commit_then_refine':self.committed_event=n['event_id']
   return n
  if self.method=='likelihood_best_first':return max(available,key=lambda n:(n['prior_logp']-.3*n['expanded'],-n['depth'],-n['id']))
  return max(available,key=lambda n:(n['assessment']['value']-.05*n['expanded'],-n['id']))
 def descendants(self,index):
  result=[self.nodes[index]]
  for child in self.nodes[index]['children']:result.extend(self.descendants(child))
  return result
 def propose(self):
  avoided=[dict(interval=n['span'],status=n['assessment']['reason'],evidence=(n.get('observation') or '')[:120]) for n in self.nodes[1:] if n['kind'] in ('event','event_switch') or n['assessment']['blocked']][-12:]
  prompt=f"Target: {self.sample['query']}\nDuration: {self.sample['duration']:g} seconds.\nAlready considered: {json.dumps(avoided,ensure_ascii=False)}\nFind a different plausible occurrence supported by the video, including an earlier low-confidence possibility. Avoid merely shifting a decimal of the same event. Return one JSON object {{\"start\":number,\"end\":number,\"evidence\":\"at most12 words\"}} with seconds."
  r=self.b.generate(prompt,96,'new_event_proposal','object');item=r.get('json');self.proposal_rounds+=1
  try:span=[float(item['start']),float(item['end'])];assert all(math.isfinite(x) for x in span)
  except (TypeError,KeyError,ValueError,AssertionError):self.log.append(dict(action='proposal_failed',call=r['id']));return
  a=assess(self.b,self.sample,span);prior=self.b.prior(span) if usable(span,self.sample['duration']) else None
  node=self.add(0,span,'event',a,r['id'],str(item.get('evidence',''))[:200],dict(id=None,prior_logp=prior['log_probability'] if prior else -1e6));self.log.append(dict(action='new_event',node=node['id'],call=r['id']));self.backup(node['id'],a['value'])
 def expand(self,node,focus=None):
  focus=focus or ('onset','offset','joint')[node['expanded']];node['expanded']+=1
  history=[];p=node
  while p['id']:
   if p.get('observation'):history.append(p['observation'])
   p=self.nodes[p['parent']]
  prompt=f"Target event: {self.sample['query']}\nDuration: {self.sample['duration']:g} seconds.\nCurrent occurrence interval: {node['span']} seconds.\nVisual observations: {'; '.join(history[-3:])}\nReinspect this occurrence and refine its {focus}. Keep the same occurrence unless decisive visual evidence shows it is wrong; in that case you may identify another occurrence. Recompute the end conditioned on any changed start. Reply only: The event happens in <start> - <end> seconds."
  r=self.b.generate(prompt,64,'boundary_refinement');span=pair(r)
  note=self.b.generate(f"Target: {self.sample['query']}\nCandidate interval: {span} seconds. Inspect the video. In at most24 words state visible onset/offset evidence and any fundamental mismatch of actor, object, action or occurrence. Do not assume the candidate is correct.",48,'visual_boundary_observation') if span else None
  a=assess(self.b,self.sample,span,True);changed=span and iou(span,node['span'])<.2;kind='event_switch' if changed else 'boundary'
  if changed and self.method=='commit_then_refine':a.update(eligible=False,blocked=True,value=0.,reason='commit_event_switch_disallowed')
  prior=self.b.prior(span) if usable(span,self.sample['duration']) else None
  child=self.add(node['id'],span,kind,a,r['id'],note['text'] if note else 'invalid generation');self.expansions+=1
  if prior:child.update(prior_logp=prior['log_probability'],prior_call=prior['id'])
  if a['blocked'] and kind!='event_switch':
   self.nodes[node['event_id']]['assessment']['blocked']=True
   self.log.append(dict(action='contradiction_backtrack',from_node=child['id'],event=node['event_id'],reason='two_model_contradiction_checks'))
  if self.method in ('self_refine','commit_then_refine'):self.current=child['id']
  self.log.append(dict(action='refine',parent=node['id'],child=child['id'],focus=focus,eligible=a['eligible'],blocked=a['blocked']));self.backup(child['id'],a['value'])
 def step(self):
  node=self.select();focus=None
  if self.method=='neural_controller' and node:
   ledger=[dict(node=n['id'],interval=n['span'],event=n['assessment']['event'],value=n['assessment']['value'],depth=n['depth'],expanded=n['expanded']) for n in diversity_select([x for x in self.nodes[1:] if self.available(x)],8,.1)]
   prompt=f"Target: {self.sample['query']}\nSearch frontier: {json.dumps(ledger)}\nRecent actions: {json.dumps(self.log[-4:])}\nChoose the next inspection. Refine only a plausible occurrence; if there is a fundamental mismatch return to another occurrence, or propose a new one. Return JSON {{\"action\":\"refine\" or \"propose\",\"node\":integer,\"focus\":\"onset\" or \"offset\" or \"joint\"}}."
   plan=self.b.generate(prompt,48,'neural_controller','object');item=plan.get('json') or {};self.log.append(dict(action='controller',call=plan['id'],plan=item))
   if item.get('action')=='propose':self.propose();return
   chosen=item.get('node');focus=item.get('focus')
   if isinstance(chosen,int) and 0<chosen<len(self.nodes) and self.available(self.nodes[chosen]):node=self.nodes[chosen]
   else:self.log.append(dict(action='controller_fallback',node=node['id']))
   if focus not in ('onset','offset','joint'):focus=None
  if node is None:
   if self.method in ('commit_then_refine','token_seed_best_first') or self.proposal_rounds>=3:raise StopIteration()
   self.propose();return
  # New proposals are not restricted to the original leading digits.
  if self.method not in ('commit_then_refine','token_seed_best_first') and self.expansions and self.expansions%4==0 and self.proposal_rounds<self.expansions//4:
   self.propose();return
  self.expand(node,focus)
 def final_select(self):
  candidates=self.candidates()
  if self.method=='commit_then_refine' and self.committed_event is not None:candidates=[n for n in candidates if n['event_id']==self.committed_event]
  pool=diversity_select(candidates,4,.1);rounds=[];current=pool
  while len(current)>1:
   following=[]
   for i in range(0,len(current),2):
    if i+1==len(current):following.append(current[i]);continue
    a,c=current[i:i+2];base=f"Target: {self.sample['query']}\nCompare visual evidence for the requested occurrence and precise boundaries. "
    def judge(x,y):return self.b.operation('final_pairwise',base+f'A: {x["span"]} seconds. B: {y["span"]} seconds. Is A more accurate than B? Answer Yes or No only.',options=['Yes','No'])
    first=judge(a,c);reverse=judge(c,a);score=.5*(first['probabilities'][0]+reverse['probabilities'][1]);winner=a if score>=.5 else c;rounds.append(dict(a=a['id'],b=c['id'],probability_a=score,winner=winner['id'],calls=[first['id'],reverse['id']]));following.append(winner)
   current=following
  selected=current[0] if current else None
  return dict(selected=selected['id'] if selected else None,span=selected['span'] if selected else None,selection_pool=[n['id'] for n in pool],comparisons=rounds)
 def run(self,save=None):
  results={};initial=copy.deepcopy((self.nodes,self.log,self.current,self.committed_event,self.beam,self.halving_active,self.halving_target,self.expansions,self.proposal_rounds))
  for cap in CONFIG['search_token_caps']:
   self.nodes,self.log,self.current,self.committed_event,self.beam,self.halving_active,self.halving_target,self.expansions,self.proposal_rounds=copy.deepcopy(initial);self.b.meter(cap)
   stop=None
   while self.b.spent<cap:
    # A state expansion must commit atomically; failed partial calls still consume budget.
    snapshot=copy.deepcopy((self.nodes,self.log,self.current,self.committed_event,self.beam,self.halving_active,self.halving_target,self.expansions,self.proposal_rounds));before=self.b.spent
    try:self.step()
    except BudgetEnd:
     self.nodes,self.log,self.current,self.committed_event,self.beam,self.halving_active,self.halving_target,self.expansions,self.proposal_rounds=snapshot;stop='token_budget';break
    except StopIteration:stop='frontier_exhausted';break
    if self.b.spent==before:stop='no_new_inference';break
   search_refs=list(self.b.active);search_cost=self.b.cost();nodes=copy.deepcopy(self.nodes);logs=copy.deepcopy(self.log)
   self.b.meter(CONFIG['selection_token_cap'])
   try:selection=self.final_select();selector_status='complete'
   except BudgetEnd:selection=dict(selected=None,span=None,selection_pool=[],comparisons=[]);selector_status='budget_failure'
   results[str(cap)]=dict(method=self.method,search_cap=cap,selection_cap=CONFIG['selection_token_cap'],nodes=nodes,actions=logs,**selection,search_calls=list(search_refs),selection_calls=list(self.b.active),search_cost=search_cost,selection_cost=self.b.cost(),selector_status=selector_status,stop=stop,expansions=self.expansions,proposal_rounds=self.proposal_rounds)
   if save:save(results)
  return results
