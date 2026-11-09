"""Strict token-level LevinTS and two deterministic mass-weighted schedulers."""
import collections,heapq,math,time
import numpy as np
from probability_budget_core import *
from probability_budget_inputs import numeric_role,choose_edges,parse_clip
from probability_budget_backend import BudgetBoundary

def eligible(probabilities):
 p=np.asarray(probabilities,dtype=np.float64);ids=np.flatnonzero(p>=CONFIG['absolute_floor']);order=ids[np.lexsort((ids,-p[ids]))];out=[];mass=0.
 for token in order:
  if mass>=CONFIG['top_p']:break
  value=float(p[token]);mass+=value;out.append(dict(token=int(token),p=value))
 return out

class Search:
 def __init__(self,backend,method,duration,progress=None):
  self.backend=backend;self.method=method;self.duration=duration;self.progress=progress;self.tokenizer=backend.tokenizer;self.started=time.monotonic();self.last_progress=self.started;self.backend.reset(BUDGETS[0]);self.step=0;self.nodes=[];self.paths=[];self.pending=set();self.heap=[];self.groups={};self.grouped=False;self.frontier=0.;self.mass=dict(valid=0.,invalid=0.,filtered=0.,policy=0.,context=0.);self.leaves=[];self.snapshots={};self.active=None;self.jobs=[];self.reward_wall=0.;self.scheduler_wall=0.;self.pieces={};self.spans=[];self.span_ids={};self.sim=np.zeros((128,128),np.float32);self.group_spans=collections.defaultdict(dict);self.last_levin_key=None;self.levin_order_verified=True
  self.add((),None,None,1.,0.,0)
 def group(self,gid):
  if gid not in self.groups:self.groups[gid]=dict(id=gid,root=None,heap=[],pending_mass=0.,spent=0,credit=0.,novelty=0.,valid_mass=0.,valid_leaves=0)
  return self.groups[gid]
 def piece(self,token):
  if token not in self.pieces:self.pieces[token]=self.tokenizer.decode([token],clean_up_tokenization_spaces=False)
  return self.pieces[token]
 def add(self,path,parent,token,p,logp,gid):
  ident=len(self.nodes);node=dict(id=ident,parent=parent,token=token,p=p,logp=logp,depth=len(path),group=gid,piece=self.piece(token) if token is not None else '',status='pending',created_step=self.step,visited_step=None);self.nodes.append(node);self.paths.append(path);self.pending.add(ident);mass=math.exp(logp);self.frontier+=mass;group=self.group(gid);group['pending_mass']+=mass
  key=math.log(max(1,len(path)))-logp;heapq.heappush(self.heap,(key,ident));heapq.heappush(group['heap'],(-logp,ident))
  return ident
 def group_top(self,gid):
  heap=self.groups[gid]['heap']
  while heap and heap[0][1] not in self.pending:heapq.heappop(heap)
  return heap[0][1] if heap else None
 def remove(self,ident):
  node=self.nodes[ident];mass=math.exp(node['logp']);self.pending.remove(ident);self.frontier=max(0.,self.frontier-mass);group=self.groups[node['group']];group['pending_mass']=max(0.,group['pending_mass']-mass);self.step+=1;node['visited_step']=self.step
 def prefix(self,ident):return self.tokenizer.decode(self.paths[ident],skip_special_tokens=True,clean_up_tokenization_spaces=False)
 def remember_interval(self,gid,span,mass):
  span=tuple(span)
  if span not in self.span_ids:
   index=len(self.spans);self.span_ids[span]=index
   if index>=len(self.sim):
    bigger=np.zeros((len(self.sim)+128,len(self.sim)+128),np.float32);bigger[:len(self.sim),:len(self.sim)]=self.sim;self.sim=bigger
   for j,other in enumerate(self.spans):self.sim[index,j]=self.sim[j,index]=iou(span,other)
   self.sim[index,index]=1.;self.spans.append(span)
  index=self.span_ids[span];group=self.group_spans[gid];group[index]=group.get(index,0.)+mass
 def update_novelty(self):
  if self.method!='numeric_novelty':return
  start=time.monotonic()
  for gid,group in self.groups.items():
   inside=self.group_spans[gid];outside=set().union(*(set(v) for g,v in self.group_spans.items() if g!=gid)) if len(self.group_spans)>1 else set()
   if not inside or not outside:group['novelty']=0.;continue
   ids=list(inside);weights=np.array([inside[i] for i in ids]);novelty=1-self.sim[np.ix_(ids,list(outside))].max(axis=1);group['novelty']=float(weights@novelty/weights.sum())
  self.reward_wall+=time.monotonic()-start
 def finish_job(self):
  if self.active is None:return
  job=self.active;spent=self.backend.tokens-job['started_inputs'];weights=job['weights']
  if weights:
   for gid,w in weights.items():self.groups[gid]['credit']+=w*spent
   self.groups[job['group']]['credit']-=spent
  self.jobs.append(dict(group=job['group'],start_inputs=job['started_inputs'],end_inputs=self.backend.tokens,allocation_weight=weights.get(job['group']),start_frontier_mass=job['start_mass'],start_novelty=job['start_novelty'],completed=True));self.active=None;self.update_novelty()
 def choose(self):
  if self.method=='levin':
   while self.heap and self.heap[0][1] not in self.pending:heapq.heappop(self.heap)
   if not self.heap:return None
   key,ident=self.heap[0]
   if self.last_levin_key is not None and key+1e-10<self.last_levin_key:self.levin_order_verified=False
   self.last_levin_key=key;self.nodes[ident]['selection_log_score']=key
   return ident
  if self.active is not None:return self.active['node']
  start=time.monotonic();active=[g for g in self.groups if self.group_top(g) is not None];lam=CONFIG['novelty_lambda'] if self.method=='numeric_novelty' else 0.;scores={g:self.groups[g]['pending_mass']*(1+lam*self.groups[g]['novelty']) for g in active};total=sum(scores.values());weights={g:v/total for g,v in scores.items()} if total else {g:1/len(active) for g in active};gid=max(active,key=lambda g:(self.groups[g]['credit']+CONFIG['allocation_quantum']*weights[g],weights[g],-g));ident=self.group_top(gid);self.active=dict(group=gid,node=ident,started_inputs=self.backend.tokens,weights=weights if self.grouped else {},start_mass=self.groups[gid]['pending_mass'],start_novelty=self.groups[gid]['novelty']);self.scheduler_wall+=time.monotonic()-start;return ident
 def process(self,ident):
  node=self.nodes[ident];gid=node['group'];path=self.paths[ident];mass=math.exp(node['logp']);group=self.groups[gid]
  if path and path[-1] in self.backend.eos:
   self.remove(ident);text=self.prefix(ident);parsed=parse_clip(text,self.duration,True);node.update(status='valid' if parsed['valid'] else 'invalid',text=text,span=parsed['span'],reason=parsed['reason']);self.mass[node['status']]+=mass;self.leaves.append(dict(node=ident,group=gid,logp=node['logp'],valid=parsed['valid'],span=parsed['span']))
   if parsed['valid']:group['valid_mass']+=mass;group['valid_leaves']+=1;self.remember_interval(gid,parsed['span'],mass)
   self.finish_job();return
  if self.backend.n+len(path)>=self.backend.max_context:
   self.remove(ident);node['status']='context_limit';self.mass['context']+=mass;self.finish_job();return
  before=self.backend.tokens
  try:logits=self.backend.get(path)
  finally:group['spent']+=self.backend.tokens-before
  import torch
  prob=torch.softmax(logits,dim=-1);raw=eligible(prob.numpy());text=self.prefix(ident);chosen,role=choose_edges(raw,self.tokenizer,text);retained=sum(e['p'] for e in raw);kept=sum(e['p'] for e in chosen);self.remove(ident);node.update(status='expanded' if chosen else 'dead',role=role,accepted_mass=kept);self.mass['filtered']+=mass*max(0.,1-retained);self.mass['policy']+=mass*max(0.,retained-kept)
  if len(chosen)>1:
   assert role.endswith('integer_or_decimal'),role
   positive=prob[prob>0];node['entropy']=float(-(positive*positive.log2()).sum());node['prefix_text']=text;values,ids=torch.topk(prob,30);allowed={e['token'] for e in chosen};node['top30']=[dict(token=int(i),text=self.piece(int(i)),p=float(p),admitted=int(i) in allowed) for p,i in zip(values.tolist(),ids.tolist())]
  first_fork=not self.grouped and len(chosen)>1
  if first_fork:self.grouped=True
  children=[]
  for edge in chosen:
   child_group=len(self.groups) if first_fork else gid;child=self.add(path+(edge['token'],),ident,edge['token'],edge['p'],node['logp']+math.log(edge['p']),child_group);children.append(child)
   if first_fork:self.groups[child_group]['root']=child
  node['children']=children
  if self.method!='levin':
   if not children or first_fork:self.finish_job()
   else:self.active['node']=children[0]
 def unseen(self):
  u=self.frontier+self.mass['context'];den=self.mass['valid']+u
  return u/den if den else None
 def snapshot(self,budget,reason):
  intervals,stats=interval_distribution(self.leaves);counts=collections.Counter(n['status'] for n in self.nodes);group_rows=[]
  for gid,g in self.groups.items():group_rows.append(dict({k:g[k] for k in ('id','root','pending_mass','spent','credit','novelty','valid_mass','valid_leaves')},root_native_mass=math.exp(self.nodes[g['root']]['logp']) if g['root'] is not None else 1.))
  total=self.frontier+sum(self.mass.values());assert abs(total-1)<5e-4,total
  costs=self.backend.costs();costs.update(wall_seconds=time.monotonic()-self.started,reward_wall_seconds=self.reward_wall,scheduler_wall_seconds=self.scheduler_wall)
  return dict(budget=budget,actual_inputs=self.backend.tokens,reason=reason,logical_step=self.step,leaf_count=len(self.leaves),nodes=len(self.nodes),counts=dict(counts),unique_intervals=len(intervals),distribution=stats,selected_span=max(intervals,key=lambda r:(r['native_mass'],tuple(-x for x in r['span'])))['span'] if intervals else None,frontier_mass=self.frontier,mass=dict(self.mass),mass_total=total,unseen_valid_mass_upper_bound=self.unseen(),groups=group_rows,cost=costs,levin_order_verified=self.levin_order_verified)
 def run(self):
  bi=0
  while True:
   bound=self.unseen()
   if self.mass['valid'] and bound is not None and bound<=CONFIG['early_stop_unseen_valid_mass']:reason='mass_target_reached';break
   if not self.pending:reason='context_limit' if self.mass['context'] else 'frontier_exhausted';break
   ident=self.choose()
   try:self.process(ident)
   except BudgetBoundary:
    self.snapshots[str(BUDGETS[bi])]=self.snapshot(BUDGETS[bi],'decoder_budget_exhausted');bi+=1
    if bi==len(BUDGETS):reason='decoder_budget_exhausted';break
    self.backend.limit=BUDGETS[bi]
   if self.progress and time.monotonic()-self.last_progress>=20:
    self.progress(stage='searching',method=self.method,decoder_inputs=self.backend.tokens,expanded_nodes=self.step,valid_paths=len(self.leaves),pending_prefixes=len(self.pending),unseen_valid_mass_upper_bound=self.unseen());self.last_progress=time.monotonic()
  for budget in BUDGETS:
   if str(budget) not in self.snapshots:self.snapshots[str(budget)]=self.snapshot(budget,reason)
  assert self.levin_order_verified
  return dict(method=self.method,complete=True,config_hash=digest(CONFIG),nodes=self.nodes,leaves=self.leaves,snapshots=self.snapshots,allocation_jobs=self.jobs,unfinished_job=None if self.active is None else dict(group=self.active['group'],start_inputs=self.active['started_inputs'],end_inputs=self.backend.tokens),scope='proposal search; no GT or semantic embeddings in allocation',timestamp=stamp())
