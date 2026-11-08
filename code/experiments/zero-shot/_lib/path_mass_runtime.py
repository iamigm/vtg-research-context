"""Short native probability-mass diagnostic, with censored 45-second controls."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import collections,gc,heapq,json,math,re,shutil,sys,time,traceback,fcntl
from pathlib import Path
import regex
from path_mass_core import *
from path_mass_inputs import Backend,parse_clip,source_frames,processor_input

PATTERN=regex.compile(r'The event happens in ([0-9]+\.[0-9]) - ([0-9]+\.[0-9]) seconds\.?')
def grammar(text,duration,eos=False):
 m=PATTERN.fullmatch(text,partial=not eos)
 if not m:return False
 start=re.match(r'The event happens in ([0-9]+\.[0-9]) - ',text)
 if start and float(start[1])>=duration:return False
 if not m.partial:return parse_clip(text,duration,True)['valid']
 end=re.match(r'The event happens in ([0-9]+\.[0-9]) - ([0-9]+\.[0-9]) ',text)
 return not end or min(duration,float(end[1]))<min(duration,float(end[2]))

def numeric_role(text):
 for endpoint,anchor in [('start','The event happens in '),('end',' - ')]:
  if anchor not in text:continue
  tail=text.rsplit(anchor,1)[1]
  if re.fullmatch(r'[0-9]*',tail):return endpoint+'.integer_or_decimal'
  if re.fullmatch(r'[0-9]+\.[0-9]*',tail):return endpoint+'.fraction'
 return 'format'

def choose_edges(edges,tokenizer,text):
 role=numeric_role(text)
 if not edges:return [],role
 if not role.endswith('integer_or_decimal'):return edges[:1],role
 top=edges[0];selected=[];keys=set()
 for edge in edges:
  token_text=tokenizer.decode([edge['token']],clean_up_tokenization_spaces=False)
  if edge is not top and not re.fullmatch(r'[0-9.]+',token_text):continue
  # If an atomic token crosses the decimal point, fractional-only alternatives
  # share one key; keep the most likely native token rather than splitting it.
  projected=token_text.split('.',1)[0]+('.' if '.' in token_text else '')
  key=projected if re.fullmatch(r'[0-9.]+',token_text) else ('other',edge['token'])
  if key in keys:continue
  keys.add(key);selected.append(edge)
 return selected,role

def status(**kw):
 state=read(OUT/'metadata/status.json') if (OUT/'metadata/status.json').exists() else {};state.update(kw,timestamp=stamp());write(OUT/'metadata/status.json',state)
 with (OUT/'logs/progress.log').open('a') as f:f.write(json.dumps(state,ensure_ascii=False)+'\n')
 print(json.dumps(kw,ensure_ascii=False),flush=True)

def intervals_from(masses):
 z=sum(masses.values());return [dict(id=digest(list(k))[:20],span=list(k),native_mass=v,conditional_weight=v/z) for k,v in sorted(masses.items())] if z else []

def search(backend,processor,duration,variant):
 import torch
 backend.cache.crop(backend.n);backend.path=();backend.last=backend.root;backend.actual_tokens=0;backend.actual_forwards=0
 queue=[(-1.,0,())];serial=1;frontier=1.;valid=0.;filtered=0.;grammar_removed=0.;invalid=0.;context=0.;policy_removed=0.;branch_audit=[];masses=collections.defaultdict(float);snapshots={};expanded=0;leaves=0;gpu_seconds=0.;started=time.monotonic();last_update=started;special=set(processor.tokenizer.all_special_ids);text_cache={};max_context=int(backend.model.config.text_config.max_position_embeddings)
 def text_of(path):
  if path not in text_cache:text_cache[path]=processor.tokenizer.decode(path,skip_special_tokens=True,clean_up_tokenization_spaces=False)
  return text_cache[path]
 def snapshot(reason):
  data=intervals_from(masses);logs={tuple(r['span']):math.log(r['native_mass']) for r in data};bound=(frontier+context)/(valid+frontier+context) if valid+frontier+context>0 else None
  return dict(reason=reason,unseen_valid_mass_upper_bound=bound,valid_native_mass=valid,frontier_mass=max(0.,frontier),context_limit_mass=context,filtered_mass=filtered,grammar_rejected_mass=grammar_removed,invalid_mass=invalid,policy_rejected_mass=policy_removed,mass_total=valid+frontier+context+filtered+grammar_removed+invalid+policy_removed,branch_points=len(branch_audit),fractional_branch_points=0,expanded_nodes=expanded,complete_paths=leaves,frontier_prefixes=len(queue),intervals=data,distribution=distribution_stats(logs),cost=dict(native_forwards=backend.actual_forwards,decoder_inputs=backend.actual_tokens,cuda_stream_seconds=gpu_seconds,wall_seconds=time.monotonic()-started))
 while queue:
  if time.monotonic()-started>=CONFIG['diagnostic_seconds_per_variant']:break
  neg,_,path=heapq.heappop(queue);mass=-neg;frontier=max(0.,frontier-mass)
  # Complete one most-likely spine before returning to the largest frontier prefix.
  while True:
   if path and path[-1] in backend.eos:
    parsed=parse_clip(text_of(path),duration,True);leaves+=1
    if parsed['valid']:valid+=mass;masses[tuple(parsed['span'])]+=mass
    else:invalid+=mass
    break
   if time.monotonic()-started>=CONFIG['diagnostic_seconds_per_variant']:
    heapq.heappush(queue,(-mass,serial,path));serial+=1;frontier+=mass;break
   if backend.n+len(path)>=max_context:context+=mass;break
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();logits=backend.get(path);b.record();b.synchronize();gpu_seconds+=a.elapsed_time(b)/1000;raw_edges,entropy=admitted(torch.softmax(logits,dim=-1).numpy(),include_entropy=numeric_role(text_of(path)).endswith('integer_or_decimal'));expanded+=1;retained=sum(e['p'] for e in raw_edges);filtered+=mass*max(0.,1-retained);edges,role=choose_edges(raw_edges,processor.tokenizer,text_of(path));policy_removed+=mass*(retained-sum(e['p'] for e in edges))
   allowed=[]
   for e in edges:
    child=path+(e['token'],)
    if variant=='format_and_mass' and ((e['token'] in special and e['token'] not in backend.eos) or not grammar(text_of(child),duration,e['token'] in backend.eos)):
     grammar_removed+=mass*e['p']
    else:allowed.append(e)
   if len(allowed)>1:
    assert role.endswith('integer_or_decimal'),role
    branch_audit.append(dict(prefix=text_of(path),role=role,entropy=entropy,alternatives=[dict(token=e['token'],text=processor.tokenizer.decode([e['token']],clean_up_tokenization_spaces=False),p=e['p']) for e in allowed]))
   if not allowed:break
   for edge in allowed[1:]:
    p=mass*edge['p'];heapq.heappush(queue,(-p,serial,path+(edge['token'],)));serial+=1;frontier+=p
   edge=allowed[0];path=path+(edge['token'],);mass*=edge['p']
  bound=(frontier+context)/(valid+frontier+context) if valid+frontier+context else 1.
  for eps in CONFIG['epsilons']:
   if str(eps) not in snapshots and valid and bound<=eps:snapshots[str(eps)]=snapshot('mass_target_reached')
  if str(min(CONFIG['epsilons'])) in snapshots:break
  if time.monotonic()-last_update>=5:
   status(stage='searching',variant=variant,expanded_nodes=expanded,unique_intervals=len(masses),native_forwards=backend.actual_forwards,unseen_valid_mass_upper_bound=bound,variant_elapsed_seconds=time.monotonic()-started);last_update=time.monotonic()
 reason='mass_target_reached' if '0.01' in snapshots else 'diagnostic_time_limit' if queue else 'frontier_exhausted'
 final=snapshot(reason)
 for s in list(snapshots.values())+[final]:assert abs(s['mass_total']-1)<3e-4,s['mass_total']
 return dict(variant=variant,snapshots=snapshots,final=final,branch_audit=branch_audit,branch_policy=CONFIG['branch_policy'],spine_scheduling='largest native-prefix mass, then greedy spine; no random sampling',stopping_formula='(frontier_mass+context_limit_mass)/(valid_native_mass+frontier_mass+context_limit_mass) <= epsilon')

def cpu_check():
 assert numeric_role('The event happens in 2')=='start.integer_or_decimal'
 assert numeric_role('The event happens in 2.')=='start.fraction'
 assert numeric_role('The event happens in 2.0 - 3')=='end.integer_or_decimal'
 assert numeric_role('The event happens in 2.0 - 3.')=='end.fraction'
 assert numeric_role('The event happens')=='format'
 assert grammar('The event happens in 2',10)
 assert grammar('The event happens in 2.0 - 5.0 seconds.',10,True)
 assert grammar('The event happens in 2.0 - 5.0 seconds',10,True)
 assert not grammar('The event happens in 2.05',10)
 assert not grammar('The event happens in 11.0 - ',10)
 assert not grammar('The event happens in 5.0 - 2.0 ',10)
 assert not grammar('The event happens in 5.0 - 2.0 seconds.',10,True)
 assert grammar('The event happens in 5.0 - 12.0 seconds.',10,True)
 assert [e['token'] for e in admitted([.97,.015,.009,.006])[0]]==[0,1]
 # The unseen valid-mass bound holds for any valid fraction of each prefix.
 for valid in (.01,.2,.8):
  for pending in (.001,.1,.3):
   bound=pending/(valid+pending)
   for survival in (0.,.1,.5,1.):assert pending*survival/(valid+pending*survival)<=bound+1e-14
 return dict(passed=True,grammar=True,native_probability_floor=True,missing_mass_bound=True,fractional_and_format_branching_disabled=True)

def run():
 import torch
 from transformers import AutoProcessor,AutoModelForImageTextToText
 from adaptive_temporal_sampling import tensorhash
 cpu=cpu_check();write(OUT/'validation/cpu.json',cpu);status(state='running',stage='loading_model',completed=0,total=3)
 torch.set_num_threads(2);start=time.monotonic();processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False);model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval();load=time.monotonic()-start
 rows=read(OUT/'metadata/cohort.json')
 for number,row in enumerate(rows):
  key=row['sample_key'];path=OUT/'queries'/f'{key}.json'
  if path.exists():continue
  status(stage='preparing_input',sample_key=key,completed=number)
  start=time.monotonic();prior,cache,frames=source_frames(row);plan=cache['plans']['2'];cpu=processor_input(processor,frames,plan['frame_ids'],plan['source_pts'],cache['source_fps'],cache['N'],row['query']);assert {k:tensorhash(v) for k,v in cpu.items()}==prior['hashes'];prepare=time.monotonic()-start;gpu=cpu.to('cuda');eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];torch.cuda.reset_peak_memory_stats();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();backend=Backend(model,gpu,torch,processor.tokenizer,eos);b.record();b.synchronize();prefill_cuda=a.elapsed_time(b)/1000
  start=time.monotonic();ids=[]
  while backend.n+len(ids)<model.config.text_config.max_position_embeddings:
   token=int(backend.get(ids).argmax());ids.append(token)
   if token in backend.eos:break
  greedy=dict(answer=processor.tokenizer.decode(ids,skip_special_tokens=True),span=parse_clip(processor.tokenizer.decode(ids,skip_special_tokens=True),cache['duration'],bool(ids and ids[-1] in backend.eos))['span'],ids=ids,cost=dict(native_forwards=backend.actual_forwards,decoder_inputs=backend.actual_tokens,wall_seconds=time.monotonic()-start))
  data=dict(sample=row,input=dict(hashes=prior['hashes'],metadata_duration=cache['duration'],input_tokens=backend.n,hw=cache['hw'],frames=len(plan['frame_ids']),input_prepare_wall_seconds=prepare,prefill_wall_seconds=backend.prefill_seconds,prefill_cuda_stream_seconds=prefill_cuda),greedy=greedy,variants={},model_load_wall_seconds=load if number==0 else 0.)
  for variant in CONFIG['variants']:
   status(stage='searching',sample_key=key,variant=variant);data['variants'][variant]=search(backend,processor,cache['duration'],variant);write(path,data)
  data['peak_allocated_bytes']=torch.cuda.max_memory_allocated();data['complete']=True;write(path,data)
  del backend,gpu,cpu,frames;gc.collect();torch.cuda.empty_cache();status(stage='query_complete',completed=number+1,sample_key=key)
 status(stage='reporting',completed=3)
 from path_mass_report import report
 report();write(OUT/'validation/final.json',dict(passed=True,input_hashes_verified=True,queries=3,mass_conservation=True,all_comparisons_finished=True,timestamp=stamp()));status(state='complete',stage='complete',completed=3)

if __name__=='__main__':
 try:
  with (OUT/'metadata/run.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
 except BaseException:
  status(state='failed',stage='failed',error=traceback.format_exc());raise
