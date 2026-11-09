"""Supplement missing observations without rerunning search or altering its answers."""
import argparse,collections,fcntl,gc,os,sys,time,traceback,subprocess
from probability_budget_core import *
from probability_budget_runtime import load_model,prepare
from probability_budget_backend import Backend
from probability_budget_search import eligible
from probability_budget_inputs import choose_edges


def collect(key):
 required={};native={};sources={}
 for method in METHODS:
  path=OUT/'search'/key/f'{method}.json'
  if not path.exists():return None
  record=read(path);sources[method]=sha(path);prefixes={}
  for n in record['nodes']:
   prefix=() if n['parent'] is None else prefixes[n['parent']]+(n['token'],);prefixes[n['id']]=prefix
   if n.get('top30'):native[prefix]=n['top30']
   if n['status'] not in ('expanded','dead'):continue
   entry=required.setdefault(prefix,dict(role=n['role'],edges={},native=n.get('top30')))
   for child in n.get('children',[]):
    c=record['nodes'][child];entry['edges'][c['token']]=c['p']
 return required,native,sources


def capture(row,model,processor,heartbeat):
 import torch
 key=row['sample_key'];data=read(OUT/'queries'/f'{key}.json');collected=collect(key)
 if not collected:return
 required,native,sources=collected;target=OUT/'token_distributions'/f'{key}.json'
 if target.exists() and read(target).get('sources')==sources:return
 started=time.monotonic();observation_started=stamp();oldpath=SOURCE/'token_traces'/key/'index.json';reference={}
 if oldpath.exists():
  old=read(oldpath)
  if old.get('validation',{}).get('passed') and old['input_hashes']==data['input']['hashes']:reference={tuple(n['prefix_ids']):n for n in old['nodes'].values()}
 rows=[];todo={};counts=collections.Counter();max_error=0.
 for prefix,info in required.items():
  if info['native'] is not None:counts['native_search']+=1;continue
  if prefix in native:
   rows.append(dict(prefix_ids=list(prefix),top30=native[prefix],source='same_input_prefix_other_method'));counts['other_method']+=1;continue
  old=reference.get(prefix)
  if old:
   probs={t['id']:t['probability'] for t in old['top']+old.get('observed',[])}
   error=max([abs(probs.get(t,-1)-p) for t,p in info['edges'].items()]+[0.])
   if error<=.0001:counts['existing_reference']+=1;continue
  todo[prefix]=info
 heartbeat(stage='supplementing',sample_key=key,required=len(required),missing=len(todo))
 if todo:
  cpu,input_info=prepare(row,processor);gpu=cpu.to('cuda');backend=Backend(model,gpu,processor);backend.reset(10**9);prefill=backend.prefill_seconds;prefill_tokens=backend.n
  # Match the search's single-token KV execution. Parallel teacher forcing was
  # tested but changed some BF16 distributions, so it is not used for this view.
  last_heartbeat=time.monotonic()
  for index,prefix in enumerate(sorted(todo)):
   logits=backend.get(prefix);prob=torch.softmax(logits.float(),dim=-1);info=todo[prefix]
   error=max([abs(float(prob[t])-p) for t,p in info['edges'].items()]+[0.]);max_error=max(max_error,error)
   assert error<.001,(key,prefix,error)
   raw=eligible(prob.numpy());chosen,policy=choose_edges(raw,processor.tokenizer,processor.tokenizer.decode(prefix,skip_special_tokens=True,clean_up_tokenization_spaces=False));assert {e['token'] for e in chosen}==set(info['edges']),(key,prefix,'admitted tokens changed')
   why=('numeric_fork' if len(chosen)>1 else 'no_admitted_token' if not chosen else 'fraction_greedy' if policy.endswith('fraction') else 'format_greedy' if policy=='format' else 'single_probability_candidate' if len(raw)==1 else 'numeric_policy_single_candidate')
   values,ids=torch.topk(prob,min(30,prob.numel()));rows.append(dict(prefix_ids=list(prefix),probability_candidates=len(raw),decision_reason=why,top30=[dict(token=int(i),text=processor.tokenizer.decode([int(i)],clean_up_tokenization_spaces=False),p=float(p)) for p,i in zip(values.tolist(),ids.tolist())],source='single_token_native_replay',max_edge_probability_error=error))
   if time.monotonic()-last_heartbeat>=20:
    heartbeat(stage='supplementing',sample_key=key,recorded=index+1,missing=len(todo));last_heartbeat=time.monotonic()
  torch.cuda.synchronize();cost=dict(backend.costs(),parallel_teacher_forcing_inputs=0,prefill_inputs=prefill_tokens,prefill_wall_seconds=prefill,wall_seconds=time.monotonic()-started)
  del backend,gpu,cpu;gc.collect();torch.cuda.empty_cache()
 else:cost=dict(wall_seconds=time.monotonic()-started,prefill_inputs=0,prefill_wall_seconds=0.,decoder_inputs=0,parallel_teacher_forcing_inputs=0)
 counts['supplemental_replay']=len(todo)
 assert len(rows)==counts['other_method']+len(todo),(key,len(rows),counts)
 result=dict(complete=True,sample_key=key,config_hash=data['config_hash'],sources=sources,input_hashes=data['input']['hashes'],nodes=rows,counts=dict(counts),cost=cost,max_edge_probability_error=max_error,gpu_observation_started_at=observation_started,gpu_observation_finished_at=stamp(),scope='observation only; raw search nodes, probabilities, allocation and answers are unchanged',timestamp=stamp())
 write(target,result);heartbeat(stage='query_supplemented',sample_key=key,counts=dict(counts),cost=cost,max_edge_probability_error=max_error)


def main(once=False,key=None):
 with (OUT/'metadata/distribution_capture.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=dict(state='running',completed=0,total=900,scope='supplemental observation; separate from search budget and wall time');started=time.monotonic()
  def heartbeat(**kw):
   state.update(kw,timestamp=stamp(),elapsed_seconds=time.monotonic()-started);write(OUT/'metadata/distribution_status.json',state)
   line=json.dumps(state,ensure_ascii=False)
   with (OUT/'logs/distribution_capture.log').open('a') as f:f.write(line+'\n')
   print(line,flush=True)
  model=processor=None
  try:
   heartbeat(stage='loading_model');model,processor,_=load_model();rows=read(COHORT)
   if key:rows=[r for r in rows if r['sample_key']==key]
   last_report=time.monotonic()
   while True:
    pending=[]
    for row in rows:
     key=row['sample_key'];path=OUT/'queries'/f'{key}.json';out=OUT/'token_distributions'/f'{key}.json'
     if path.exists() and read(path).get('complete') and not out.exists():pending.append(row)
    if not pending:
     status=read(OUT/'metadata/status.json')
     if once or status.get('state') in ('complete','failed'):break
     heartbeat(stage='waiting_for_new_queries');time.sleep(20);continue
    for row in pending:
     capture(row,model,processor,heartbeat);state['completed']=len(list((OUT/'token_distributions').glob('*.json')))
     if time.monotonic()-last_report>300:
      with (OUT/'logs/report.log').open('ab',buffering=0) as stream:subprocess.run([sys.executable,str(Path(__file__).with_name('probability_budget_report.py'))],stdout=stream,stderr=subprocess.STDOUT,check=True)
      last_report=time.monotonic()
     # Short CUDA work gaps keep proposal search responsive on the same device.
     time.sleep(max(.3,state.get('cost',{}).get('wall_seconds',0)))
    if once:break
   heartbeat(state='complete',stage='complete')
   execution=OUT/'metadata/distribution_execution.json'
   if execution.exists():
    history=read(execution)
    for session in history['sessions']:
     if session.get('pid')==os.getpid():session.update(finished_at=stamp(),state='complete')
    write(execution,history)
   with (OUT/'logs/report.log').open('ab',buffering=0) as stream:subprocess.run([sys.executable,str(Path(__file__).with_name('probability_budget_report.py'))],stdout=stream,stderr=subprocess.STDOUT,check=True)
  except BaseException:
   heartbeat(state='failed',stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--key');a=p.parse_args();main(a.once,a.key)
