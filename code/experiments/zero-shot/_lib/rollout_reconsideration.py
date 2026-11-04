"""Pilot600: adaptive agent, fixed verifier, and model-selected test-time scaling."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,collections,fcntl,gc,gzip,hashlib,json,math,re,struct,subprocess,sys,threading,time,traceback
from pathlib import Path
from result_layout import relocate_path
from deterministic_rollout300 import parse,sites
ROOT=Path('/home/irteam/experiments/zero-shot')
FAMILY='rollout/reconsideration'
OUT=ROOT/'artifacts/runs'/FAMILY/'pilot/600'
REPORT=ROOT/'results'/FAMILY/'pilot/600'
SOURCE=ROOT/'artifacts/runs/rollout/deterministic/pilot/900'
COHORT=ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_600/samples.json'
MODEL=Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct')
VARIANTS=[('direct_pool8','direct',8,1),('agent_pool8','agent',8,3),('verifier_pool8','verifier',8,3),('tts_pool8_select4','tts',8,4),('tts_pool8_select8','tts',8,8),('direct_pool32','direct',32,1),('agent_pool32','agent',32,7),('verifier_pool32','verifier',32,7),('tts_pool32_select4','tts',32,4),('tts_pool32_select8','tts',32,8)]
CONFIG=dict(family=FAMILY,queries=600,cohort=str(COHORT),model=str(MODEL),fps=2,target_pixels=100352,input_order='video_query',
 proposal_pools=[8,32],variants=VARIANTS,decoding='greedy; no random sampling, no training, no new model',
 agent='adaptive controller chooses event/boundary inspection; same-VLM visual inspector returns observations; 3/7 action-observation rounds then regeneration',
 verifier='3/7 fixed diverse candidates; independent event and boundary Yes/No likelihoods then regeneration',
 tts='4/8 diverse candidates from pool8/32; model pairwise knockout; both presentation orders per comparison; final regeneration',
 decision_budget='agent/verifier low/high ceilings 7/15 model evaluations; tts select4/8 ceilings 7/15; actual costs recorded, NOT FLOPs matched',
 control='direct regeneration with the same candidate ledger but no review; existing greedy/max_probability/weighted_medoid/oracle references',
 candidate_selection='GT-free unique-interval farthest-first, greedy first; ties by original log probability then candidate index',
 clustering='complete-link temporal IoU >= .5; temporal occurrence proxy, not semantic event identity gold',
 probes='at most one disjoint event proxy and one local boundary proxy, selected without GT from pool32; root and both immediate child FP32 distributions',
 generation_limits=dict(controller=48,inspector=48,final=64),new_review_prompt_chunk=128,seed='no sampling; bootstrap42 only',
 additional_diagnostics='qvhighlights-timelens_00507 not in frozen600; inspect saved trace only, excluded from all600 outcomes')

def stamp():
 import datetime
 return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def read(path):return json.loads(relocate_path(Path(path)).read_text())
def sha(path):return hashlib.sha256(relocate_path(Path(path)).read_bytes()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def signature(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def interval(c):
 p=c.get('parsed');return [p['start'],p['end']] if p and p.get('positive') and c.get('stopped_on_eos',True) else None
def iou(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0]))/max(1e-12,max(a[1],b[1])-min(a[0],b[0])) if a and b else 0.
def unique_candidates(cs):
 result=[];seen=set()
 for c in cs:
  pair=interval(c)
  if pair and tuple(pair) not in seen:result.append(c);seen.add(tuple(pair))
 return result

def diverse(cs,n):
 available=unique_candidates(cs)
 if not available:return []
 first=next((c for c in available if c['index']==1),max(available,key=lambda c:c['log_probability']))
 selected=[first];available=[c for c in available if c['index']!=first['index']]
 while available and len(selected)<n:
  chosen=max(available,key=lambda c:(1-max(iou(interval(c),interval(d)) for d in selected),c['log_probability'],-c['index']))
  selected.append(chosen);available.remove(chosen)
 return selected

def clusters(cs,threshold=.5):
 groups=[];assignment={}
 for c in unique_candidates(cs):
  matches=[(min(iou(interval(c),interval(d)) for d in group),i) for i,group in enumerate(groups)]
  matches=[x for x in matches if x[0]>=threshold]
  if matches:j=max(matches)[1];groups[j].append(c)
  else:j=len(groups);groups.append([c])
  assignment[c['index']]=j
 for c in cs:
  if interval(c):assignment[c['index']]=next(assignment[d['index']] for d in unique_candidates(cs) if interval(d)==interval(c))
 return groups,assignment

def select_probes(record):
 cs=record['result']['candidates'];duration=record['sample']['duration'];event=[];boundary=[]
 for c in cs[1:]:
  b=c['branch'];parent=cs[b['parent']-1];a,z=interval(parent),interval(c)
  if not a or not z:continue
  overlap=iou(a,z);shift=max(abs(a[0]-z[0]),abs(a[1]-z[1]))
  if overlap<.1 and abs(sum(a)/2-sum(z)/2)>max(2,.05*duration):event.append(c)
  if overlap>=.5 and shift>0 and shift<=max(2,.1*(a[1]-a[0])):boundary.append(c)
 def best(items):return max(items,key=lambda c:(c['branch']['raw_probability'],-c['index'])) if items else None
 result=[]
 for axis,items in [('event',event),('boundary',boundary)]:
  c=best(items)
  if c:result.append(dict(axis=axis,parent=c['branch']['parent'],candidate=c['index'],position=c['branch']['position'],roles=c['branch']['roles'],raw_probability=c['branch']['raw_probability']))
 return result

def ledger_text(cs,probes,tok,trace):
 groups,assignment=clusters(cs);rows=[]
 for c in diverse(cs,8):
  pair=interval(c);rows.append(f"#{c['index']} [{pair[0]:g},{pair[1]:g}] group{assignment[c['index']]} logP={c['log_probability']:.2f}")
 for p in probes:
  parent=cs[p['parent']-1] if p['parent']<=len(cs) else None;child=cs[p['candidate']-1] if p['candidate']<=len(cs) else None
  if parent is None or child is None:continue
  node=trace['nodes'][trace['candidates'][str(parent['index'])]['nodes'][p['position']]]
  original=parent['ids'][p['position']];alternative=child['ids'][p['position']]
  def prob(i):return next(t['probability'] for t in node['observed'] if t['id']==i)
  rows.append('prefix '+repr(node['prefix_text'])+f"; {tok.decode([original])!r}:{prob(original):.3f} -> #{parent['index']}; {tok.decode([alternative])!r}:{prob(alternative):.3f} -> #{child['index']}")
 return 'Candidates in seconds (group=temporal overlap, logP=original LM likelihood):\n'+'\n'.join(rows)

class Progress:
 def __init__(self,rank,total):
  self.rank=rank;self.total=total;self.start=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event();self.state=dict(state='running',stage='initializing',completed=0,total=total,counts={})
  threading.Thread(target=self.heartbeat,daemon=True).start()
 def update(self,**kw):
  with self.lock:
   self.state.update(kw);self.state.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start,percentage=100*self.state['completed']/max(1,self.total))
   write(OUT/'metadata'/f'worker{self.rank}.json',self.state)
   with (OUT/'logs'/f'worker{self.rank}.log').open('a') as f:f.write(json.dumps(self.state,ensure_ascii=False)+'\n')
   print(json.dumps(self.state,ensure_ascii=False),flush=True)
 def heartbeat(self):
  while not self.stop.wait(20):self.update()

class Session:
 """Common video prefill; deterministic chunked review prompts and greedy decoding."""
 def __init__(self,model,inputs,torch,tok,eos,baseline):
  self.model=model;self.inputs=inputs;self.torch=torch;self.tok=tok;self.eos=eos;self.baseline=baseline;self.n=inputs.input_ids.shape[1]
  torch.cuda.synchronize();started=time.perf_counter()
  with torch.inference_mode():o=model(**inputs,use_cache=True,logits_to_keep=1)
  self.cache=o.past_key_values;self.root=o.logits[0,-1].float().cpu();torch.cuda.synchronize();self.prefill_seconds=time.perf_counter()-started;del o
  self.reset()
 def reset(self):
  self.cache.crop(self.n);self.path=();self.logits={():self.root};self.tokens=0;self.forwards=0;self.seconds=0.;self.context_work=0;self.logical_calls=[];self.generated=0
 def snapshot(self):return dict(decoder_input_tokens=self.tokens,forward_calls=self.forwards,model_seconds=self.seconds,attention_context_work=self.context_work,generated_tokens=self.generated,model_evaluations=len(self.logical_calls))
 def get(self,path):
  path=tuple(path)
  if path in self.logits:return self.logits[path]
  common=0
  for a,b in zip(self.path,path):
   if a!=b:break
   common+=1
  if common==len(path):common=max(0,len(path)-1)
  self.cache.crop(self.n+common);torch=self.torch
  for start in range(common,len(path),CONFIG['new_review_prompt_chunk']):
   tokens=path[start:start+CONFIG['new_review_prompt_chunk']];size=len(tokens);torch.cuda.synchronize();began=time.perf_counter()
   with torch.inference_mode():o=self.model(input_ids=torch.tensor([tokens],device='cuda'),attention_mask=torch.ones((1,self.n+start+size),device='cuda',dtype=self.inputs.attention_mask.dtype),past_key_values=self.cache,cache_position=torch.arange(self.n+start,self.n+start+size,device='cuda'),use_cache=True,logits_to_keep=1)
   self.cache=o.past_key_values;self.path=path[:start+size];self.logits[self.path]=o.logits[0,-1].float().cpu();torch.cuda.synchronize()
   self.seconds+=time.perf_counter()-began;self.tokens+=size;self.forwards+=1;self.context_work+=sum(self.n+start+i+1 for i in range(size));del o
  return self.logits[path]
 def dialog(self,turns):
  text='\n'+self.tok.apply_chat_template(turns,tokenize=False,add_generation_prompt=True)
  return self.baseline['ids']+self.tok.encode(text,add_special_tokens=False)
 def generate(self,turns,limit,kind,json_stop=False):
  prefix=self.dialog(turns);ids=[];before=self.snapshot();started=time.monotonic();eos=False
  for _ in range(limit):
   token=int(self.get(prefix+ids).argmax());ids.append(token);self.generated+=1
   if token in self.eos:eos=True;break
   if json_stop and self.tok.decode(ids,skip_special_tokens=True).strip().endswith('}'):break
  text=self.tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
  self.logical_calls.append(dict(kind=kind,prompt_tokens=len(prefix),response_tokens=len(ids),elapsed_seconds=time.monotonic()-started,decoder_input_tokens=self.tokens-before['decoder_input_tokens']))
  return dict(text=text,ids=ids,parsed=parse(text),stopped_on_eos=eos,truncated=not eos,context_ids=prefix)
 def choice(self,text,options,kind):
  prefix=self.dialog([dict(role='user',content=text)]);before=self.snapshot();logits=self.get(prefix);lp=self.torch.log_softmax(logits,-1)
  ids=[self.tok.encode(o,add_special_tokens=False) for o in options];assert all(len(x)==1 for x in ids)
  raw=[math.exp(float(lp[x[0]])) for x in ids];relative=[x/sum(raw) for x in raw] if sum(raw)>0 else [1/len(raw)]*len(raw)
  result=dict(options=options,raw_probabilities=raw,conditional_probabilities=relative,option_mass=sum(raw))
  self.logical_calls.append(dict(kind=kind,prompt_tokens=len(prefix),response_tokens=0,decoder_input_tokens=self.tokens-before['decoder_input_tokens'],result=result))
  return result
 def score_path(self,context,ids):
  # Teacher-forced path probability under the revised context, including EOS.
  value=0.
  for j,token in enumerate(ids):value+=float(self.torch.log_softmax(self.get(context+ids[:j]),-1)[token])
  return value

FINAL='Reconsider the target event in the video using the candidates and observations. You may change the occurrence or either boundary. Reply only: The event happens in <start> - <end> seconds.'
def final_answer(session,ledger,notes,turns=None):
 turns=list(turns or []);turns.append(dict(role='user',content=ledger+'\n'+notes+'\n'+FINAL))
 return session.generate(turns,CONFIG['generation_limits']['final'],'final_regeneration')

def run_variant(session,record,trace,name,method,pool_size,amount,probes):
 cs=record['result']['candidates'][:pool_size];ledger=ledger_text(cs,[p for p in probes if p['candidate']<=len(cs) and p['parent']<=len(cs)],session.tok,trace);reviews=[];selected=None;agent_history=[]
 if method=='direct':answer=final_answer(session,ledger,'No additional assessment is available.')
 elif method=='verifier':
  candidates=diverse(cs,amount)
  for c in candidates:
   span=interval(c);base=f"Target event: {record['sample']['query']}\nCandidate interval: [{span[0]:g}, {span[1]:g}] seconds. Inspect the video, not the model likelihood. "
   event=session.choice(base+'Does this interval show the requested event occurrence? Answer Yes or No only.',['Yes','No'],'verify_event')
   boundary=session.choice(base+'Are both boundaries well placed without unrelated portions or missing the action? Answer Yes or No only.',['Yes','No'],'verify_boundaries')
   reviews.append(dict(candidate=c['index'],event=event,boundary=boundary,score=event['conditional_probabilities'][0]*boundary['conditional_probabilities'][0]))
  selected=max(reviews,key=lambda r:(r['score'],-r['candidate']))['candidate'] if reviews else 1
  notes='Independent verifier scores (noisy, not ground truth): '+ '; '.join(f"#{r['candidate']} event={r['event']['conditional_probabilities'][0]:.3f}, boundaries={r['boundary']['conditional_probabilities'][0]:.3f}" for r in reviews)+f'. Preferred candidate #{selected}.'
  answer=final_answer(session,ledger,notes)
 elif method=='tts':
  current=diverse(cs,amount);rounds=[]
  while len(current)>1:
   following=[]
   for i in range(0,len(current),2):
    if i+1==len(current):following.append(current[i]);continue
    a,b=current[i:i+2];aa,bb=interval(a),interval(b)
    def judge(x,y):
     return session.choice(f"Target event: {record['sample']['query']}\nA: [{x[0]:g}, {x[1]:g}] seconds. B: [{y[0]:g}, {y[1]:g}] seconds. Which interval better identifies the requested occurrence and its exact boundaries in the video? Compare visual evidence. Answer A or B only.",['A','B'],'pairwise_selection')
    forward=judge(aa,bb);reverse=judge(bb,aa);score=.5*(forward['conditional_probabilities'][0]+reverse['conditional_probabilities'][1]);winner=a if score>=.5 else b
    rounds.append(dict(a=a['index'],b=b['index'],forward=forward,reverse=reverse,p_a=score,winner=winner['index']));following.append(winner)
   current=following
  selected=current[0]['index'] if current else 1;reviews=rounds
  notes='Pairwise visual comparisons, both presentation orders: '+'; '.join(f"#{r['a']} vs #{r['b']}: P(first wins)={r['p_a']:.3f}, winner #{r['winner']}" for r in rounds)+f'. Tournament winner #{selected}. Original LM likelihood was not used to rank the comparisons.'
  answer=final_answer(session,ledger,notes)
 else:
  candidates=diverse(cs,8);byid={c['index']:c for c in candidates};available=list(byid);used=set()
  for r in range(amount):
   prompt=ledger+'\nSelect the next visual inspection. Distinguish a different occurrence from a small boundary error. Return JSON only: {"action":"event" or "boundary" or "done", "a":candidate_id, "b":candidate_id}. Use event to compare two occurrences, boundary to inspect onset/offset of a. Use done only after an observation if sufficient.'
   history=agent_history+[dict(role='user',content=prompt)]
   plan=session.generate(history,CONFIG['generation_limits']['controller'],'agent_plan',json_stop=True)
   try:action=json.loads(re.search(r'\{.*\}',plan['text'],re.S).group())
   except Exception:action={}
   if action.get('action')=='done' and reviews:break
   fallback=False;a=action.get('a');b=action.get('b');mode=action.get('action')
   if a not in byid or mode not in ('event','boundary') or mode=='event' and (b not in byid or a==b) or (mode,a,b) in used:
    fallback=True;options=[('boundary',x,x) for x in available]+[('event',x,y) for x in available for y in available if x<y];unused=[x for x in options if x not in used]
    if not unused:break
    mode,a,b=unused[0]
   used.add((mode,a,b));aa=interval(byid[a]) if a in byid else interval(cs[0]);bb=interval(byid[b]) if b in byid else aa
   if mode=='event':inspection=f"Compare the event occurrences in {aa} and {bb} seconds. Which contains the target '{record['sample']['query']}'? State visible action evidence and distinguish repeated or unrelated occurrences in at most 30 words."
   else:inspection=f"Inspect the onset and offset around {aa} seconds for '{record['sample']['query']}'. State visual boundary evidence and propose more precise start/end if needed, in at most 30 words."
   observation=session.generate([dict(role='user',content=inspection)],CONFIG['generation_limits']['inspector'],'visual_inspector')
   reviews.append(dict(round=r+1,action=mode,a=a,b=b,fallback=fallback,plan=plan['text'],observation=observation['text']))
   agent_history += [dict(role='user',content=prompt),dict(role='assistant',content=plan['text']),dict(role='user',content='Visual inspection observation: '+observation['text']),dict(role='assistant',content='Observation recorded.')]
  answer=final_answer(session,ledger,'Use the accumulated visual observations. They can be mistaken; verify against the video.',agent_history)
 decision=session.snapshot();calls=list(session.logical_calls);context=answer.pop('context_ids')
 return dict(name=name,method=method,pool=pool_size,amount=amount,answer=answer,selected_candidate=selected,reviews=reviews,ledger=ledger,cost_decision=decision,model_calls=calls,final_context_ids=context,probes=[])


def capture_probes(session,outcome,record,trace,probes,key):
 import numpy as np
 dest=OUT/'token_traces'/key;dest.mkdir(parents=True,exist_ok=True);before=session.snapshot();context=outcome['final_context_ids'];torch=session.torch
 for probe in probes:
  parent=record['result']['candidates'][probe['parent']-1];child=record['result']['candidates'][probe['candidate']-1];j=probe['position'];prefix=parent['ids'][:j];old=parent['ids'][j];alt=child['ids'][j];points=[]
  for tag,suffix,baseline_candidate,step in [('root',prefix,parent['index'],j),('original_child',prefix+[old],parent['index'],j+1),('alternative_child',prefix+[alt],child['index'],j+1)]:
   baseline_nodes=trace['candidates'][str(baseline_candidate)]['nodes']
   if step>=len(baseline_nodes):continue
   source_node=trace['nodes'][baseline_nodes[step]];baseline_file=SOURCE/'token_traces'/key/source_node['file']
   base=np.frombuffer(gzip.decompress(baseline_file.read_bytes()),dtype='<f4').astype(np.float64)
   logits=session.get(context+suffix);prob=torch.softmax(logits,-1).numpy().astype('<f4',copy=False);total=float(prob.sum(dtype=np.float64));assert abs(total-1)<3e-5
   new=prob.astype(np.float64);mid=.5*(base+new);positive=base>0;positive_new=new>0;js=.5*float((base[positive]*np.log(base[positive]/mid[positive])).sum())+.5*float((new[positive_new]*np.log(new[positive_new]/mid[positive_new])).sum())
   filename=f"{outcome['name']}.{probe['axis']}.{tag}.f32.gz";compressed=gzip.compress(prob.tobytes(),compresslevel=1,mtime=0);path=dest/filename;path.write_bytes(compressed)
   ids=np.lexsort((np.arange(len(prob)),-prob))[:10]
   points.append(dict(kind=tag,file=str(path.relative_to(OUT)),sha256=hashlib.sha256(compressed).hexdigest(),bytes=len(compressed),suffix_ids=suffix,baseline_file=str(baseline_file),js_divergence=js,total_probability=total,top=[dict(id=int(i),token=session.tok.decode([int(i)]),before=float(base[i]),after=float(prob[i])) for i in ids],original_probability_before=float(base[old]),original_probability_after=float(prob[old]),alternative_probability_before=float(base[alt]),alternative_probability_after=float(prob[alt])))
  paths={}
  for role,c in [('parent',parent),('alternative',child)]:paths[role]=dict(candidate=c['index'],before_log_probability=c['log_probability'],after_log_probability=session.score_path(context,c['ids']))
  outcome['probes'].append(dict(**probe,nodes=points,paths=paths))
 outcome['cost_total']=session.snapshot();outcome['cost_diagnostics']={k:outcome['cost_total'][k]-before[k] for k in before};outcome['completed_at']=stamp()


def setup():
 for category in ('metadata','responses','token_traces','logs','inputs','validation','code'):(OUT/category).mkdir(parents=True,exist_ok=True)
 REPORT.mkdir(parents=True,exist_ok=True)
 cohort=read(COHORT);assert len(cohort)==600 and sorted(collections.Counter(s['dataset'] for s in cohort).values())==[200,200,200]
 full=read(ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json');assert full[:600]==cohort
 for name,obj in [('config.json',CONFIG),('cohort.json',cohort)]:
  path=OUT/'metadata'/name
  if path.exists():assert read(path)==json.loads(json.dumps(obj)),f'Frozen config mismatch {path}'
  else:write(path,obj)
 manifest=OUT/'metadata/provenance.json'
 if not manifest.exists():write(manifest,dict(created_at=stamp(),cohort_sha256=sha(COHORT),parent_store=str(SOURCE),model_manifest_sha256=sha('/home/irteam/backbones/manifests/registry.json'),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),cpuset=Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text().strip(),workers=2,torch_threads_per_worker=2,ground_truth_excluded_from_generation=True))
 for file in ('rollout_reconsideration.py','rollout_reconsideration_report.py','rollout_reconsideration_viewer.html','rollout_reconsideration_check.py'):
  source=Path(__file__).with_name(file)
  if source.exists():(OUT/'code'/file).write_bytes(source.read_bytes())
 write(OUT/'metadata/source_manifest.json',dict(files=[dict(path=str(Path(__file__).with_name(n)),sha256=sha(Path(__file__).with_name(n))) for n in ('rollout_reconsideration.py','rollout_reconsideration_report.py','rollout_reconsideration_viewer.html','rollout_reconsideration_check.py','fps_spatial_control.py','conditional_rollout_budget.py')]))
 return cohort

def worker(rank,world):
 cohort=read(COHORT);rows=[s for s in cohort if int(hashlib.sha256(s['video_path'].encode()).hexdigest(),16)%world==rank];p=Progress(rank,len(rows));done=0;counts=collections.Counter();fresh=0;started=time.monotonic()
 try:
  import torch
  import fps_spatial_control as source
  from transformers import AutoProcessor,AutoModelForImageTextToText
  torch.set_num_threads(2);p.update(stage='loading_model')
  (OUT/'environment').mkdir(exist_ok=True)
  import transformers
  write(OUT/'environment'/f'worker{rank}.json',dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),threads=2))
  proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
  model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
  eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];validated=set()
  for row in rows:
   key=row['sample_key'];dest=OUT/'responses'/f'{key}.json'
   if dest.exists():
    saved=read(dest);assert saved['config_hash']==signature(CONFIG);done+=1;counts[row['dataset']]+=1;continue
   p.update(stage='preparing_input',sample_key=key,variant=None,completed=done)
   record=read(SOURCE/'rollouts'/f'{key}.json');assert record['sample']==row
   old_input=read(SOURCE/'inputs'/f'{key}.json');cache_path=relocate_path(Path(old_input['cache_metadata']));cache=read(cache_path)
   public={k:row[k] for k in ('sample_key','dataset','video_path','query','duration')}
   cpu,detail=source.prepare_input(public,cache,2,'video_query',proc);assert detail['hashes']==record['input_hashes']==old_input['hashes'],'Video/text input changed'
   write(OUT/'inputs'/f'{key}.json',dict(source_input=str(SOURCE/'inputs'/f'{key}.json'),hashes=detail['hashes'],fps=2,hw=detail['hw'],frame_ids=detail['plan']['frame_ids'],timestamps=detail['plan']['source_pts'],visual_tokens=detail['visual_tokens'],input_tokens=detail['input_tokens'],uses_gt=False))
   while torch.cuda.mem_get_info()[0]<16*2**30:p.update(state='waiting',stage='waiting_for_gpu_memory');time.sleep(20)
   p.update(state='running')
   assert detail['input_tokens']+12000<model.config.text_config.max_position_embeddings,'Review context may exceed model capacity'
   gpu=cpu.to('cuda');session=Session(model,gpu,torch,proc.tokenizer,eos,record['result']['candidates'][0]);p.update(stage='prefill_complete',input_tokens=session.n,prefill_seconds=session.prefill_seconds)
   if row['dataset'] not in validated:
    path=[]
    for token in record['result']['candidates'][0]['ids']:
     predicted=int(session.get(path).argmax());assert predicted==token,(key,'Greedy replay changed',len(path),predicted,token);path.append(token)
    write(OUT/'validation'/f'input_greedy_worker{rank}_{row["dataset"]}.json',dict(passed=True,sample_key=key,input_hashes=detail['hashes'],greedy_ids_identical=True,prompt_chunk_size=CONFIG['new_review_prompt_chunk']))
    validated.add(row['dataset'])
   trace=read(SOURCE/'token_traces'/key/'index.json');probes=select_probes(record)
   partial=dest.with_suffix('.partial.json')
   result=read(partial) if partial.exists() else dict(sample=public,config_hash=signature(CONFIG),baseline_reference=str(SOURCE/'rollouts'/f'{key}.json'),baseline_sha256=sha(SOURCE/'rollouts'/f'{key}.json'),input_hashes=detail['hashes'],prefill_seconds=session.prefill_seconds,probes=probes,variants={})
   for name,method,pool,amount in VARIANTS:
    if name in result['variants']:continue
    session.reset();p.update(stage='reviewing',sample_key=key,variant=name,variants_completed=len(result['variants']))
    outcome=run_variant(session,record,trace,name,method,pool,amount,probes)
    p.update(stage='probing_distribution',sample_key=key,variant=name,decision_calls=outcome['cost_decision']['model_evaluations'],decision_seconds=outcome['cost_decision']['model_seconds'])
    capture_probes(session,outcome,record,trace,probes,key);outcome['proposal_cost']=dict(candidates=min(pool,len(record['result']['candidates'])),decoder_tokens=record['result']['candidates'][min(pool,len(record['result']['candidates']))-1]['decoder_tokens'],original_prefill_seconds=record['result']['prefill_seconds'],original_generation_seconds=record['result']['candidates'][min(pool,len(record['result']['candidates']))-1]['elapsed_seconds'])
    result['variants'][name]=outcome;write(partial,result)
   write(dest,result);partial.unlink(missing_ok=True);done+=1;fresh+=1;counts[row['dataset']]+=1
   p.update(stage='inference',completed=done,counts=dict(counts),sample_key=key,variants_completed=len(VARIANTS),eta_seconds=(time.monotonic()-started)/fresh*(len(rows)-done),eta_scope='remaining assigned inference including review and probability diagnostics; excludes initial preprocessing/model load and final reporting')
   del session,gpu,cpu;gc.collect();torch.cuda.empty_cache()
  p.update(state='complete',stage='complete',completed=len(rows),eta_seconds=0)
 except BaseException:p.update(state='failed',stage='failed',error=traceback.format_exc());raise
 finally:p.stop.set()


def supervise():
 cohort=setup()
 with (OUT/'metadata/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);workers=[]
  for rank in range(2):
   stream=(OUT/'logs'/f'worker{rank}_runner.log').open('a')
   process=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(rank),'--world','2'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);workers.append(process);stream.close()
  write(OUT/'metadata/launcher.json',dict(supervisor_pid=os.getpid(),worker_pids=[p.pid for p in workers],started_at=stamp()))
  started=time.monotonic();reported300=False
  try:
   while True:
    states=[read(OUT/'metadata'/f'worker{i}.json') if (OUT/'metadata'/f'worker{i}.json').exists() else {} for i in range(2)]
    completed={p.stem for p in (OUT/'responses').glob('*.json') if not p.name.endswith('.partial.json')};n=sum(s['sample_key'] in completed for s in cohort)
    failed=[s for s in states if s.get('state')=='failed']
    counts=collections.Counter(s['dataset'] for s in cohort if s['sample_key'] in completed)
    etas=[s['eta_seconds'] for s in states if s.get('eta_seconds') is not None]
    status=dict(state='failed' if failed else 'running',stage='inference',completed=n,total=600,percentage=n/6,counts=dict(counts),elapsed_seconds=time.monotonic()-started,eta_seconds=max(etas) if etas else None,eta_scope='remaining worker inference; excludes final reporting',timestamp=stamp(),workers=states)
    write(OUT/'metadata/status.json',status)
    with (OUT/'logs/progress.log').open('a') as f:f.write(json.dumps(status,ensure_ascii=False)+'\n')
    if failed:raise RuntimeError('Worker failed; see worker logs')
    if not reported300 and all(s['sample_key'] in completed for s in cohort[:300]):
     from rollout_reconsideration_report import report
     report(300,refresh=True);reported300=True
    if n==600:break
    for process in workers:
     if process.poll() is not None and process.returncode!=0:raise RuntimeError(f'Worker {process.pid} exited {process.returncode}')
    time.sleep(20)
   for process in workers:assert process.wait()==0
   from rollout_reconsideration_report import report
   report(600,refresh=True)
   write(OUT/'metadata/status.json',dict(state='complete',stage='complete',completed=600,total=600,percentage=100,counts=dict(counts),timestamp=stamp(),elapsed_seconds=time.monotonic()-started,eta_seconds=0))
  except BaseException:
   write(OUT/'metadata/status.json',dict(state='failed',stage='failed',completed=n if 'n' in locals() else 0,total=600,error=traceback.format_exc(),timestamp=stamp()));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--world',type=int,default=2);args=parser.parse_args()
 if args.worker is not None:worker(args.worker,args.world)
 else:supervise()
