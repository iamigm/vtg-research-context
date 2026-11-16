"""Detached 300-to-900 native search comparison with automatic reporting gates."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,collections,fcntl,gc,json,sys,threading,time,traceback,subprocess
from pathlib import Path
from probability_budget_core import *
from probability_budget_inputs import source_frames,processor_input,parse_clip
from probability_budget_backend import Backend,BudgetBoundary
from probability_budget_search import Search,eligible
class Progress:
 def __init__(self):
  clock=OUT/'metadata/experiment_clock.json';elapsed=max(0,time.time()-__import__('datetime').datetime.fromisoformat(read(clock)['started_at']).timestamp()) if clock.exists() else 0
  self.lock=threading.RLock();self.start=time.monotonic()-elapsed;self.stop=threading.Event();self.state=dict(state='running',stage='starting',active_queries=300,completed=0,total=900,method_completed={m:0 for m in METHODS},eta_seconds=None,eta_scope='remaining proposal queries based on completed-prefix throughput; excludes report validation');threading.Thread(target=self.heartbeat,daemon=True).start()
 def __call__(self,**kw):
  with self.lock:
   if 'sample_key' in kw and kw['sample_key']!=self.state.get('sample_key'):
    for k in ('decoder_inputs','expanded_nodes','valid_paths','pending_prefixes','unseen_valid_mass_upper_bound'):self.state.pop(k,None)
   self.state.update(kw,timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start);write(OUT/'metadata/status.json',self.state)
   with (OUT/'logs/progress.log').open('a') as f:f.write(json.dumps(self.state,ensure_ascii=False)+'\n')
   print(json.dumps(kw,ensure_ascii=False),flush=True)
 def heartbeat(self):
  while not self.stop.wait(20):self()

def load_model():
 import torch
 from transformers import AutoProcessor,AutoModelForImageTextToText
 torch.set_num_threads(2);start=time.monotonic();processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False);model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
 return model,processor,time.monotonic()-start

def prepare(row,processor):
 from adaptive_temporal_sampling import tensorhash
 start=time.monotonic();prior,cache,frames=source_frames(row);plan=cache['plans']['2'];cpu=processor_input(processor,frames,plan['frame_ids'],plan['source_pts'],cache['source_fps'],cache['N'],row['query']);hashes={k:tensorhash(v) for k,v in cpu.items()};assert hashes==prior['hashes'],row['sample_key']
 info=dict(source=str(SOURCE/'inputs'/f'{row["sample_key"]}.json'),hashes=hashes,duration=cache['duration'],frame_ids=plan['frame_ids'],timestamps=plan['source_pts'],hw=cache['hw'],visual_tokens=int(cpu.video_grid_thw.prod())//4,input_tokens=int(cpu.input_ids.shape[1]),input_prepare_wall_seconds=time.monotonic()-start,uses_gt=False);del frames
 return cpu,info

def greedy(backend,duration):
 backend.reset(2048);ids=[];start=time.monotonic()
 try:
  while backend.n+len(ids)<backend.max_context:
   token=int(backend.get(ids,store=False).argmax());ids.append(token)
   if token in backend.eos:break
 except BudgetBoundary:pass
 text=backend.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);parsed=parse_clip(text,duration,bool(ids and ids[-1] in backend.eos));cost=backend.costs();cost['wall_seconds']=time.monotonic()-start
 return dict(answer=text,ids=ids,span=parsed['span'],stopped_on_eos=bool(ids and ids[-1] in backend.eos),cost=cost)

def smoke(model,processor,progress):
 import torch
 import probability_budget_search as search
 row=read(COHORT)[0];cpu,info=prepare(row,processor);gpu=cpu.to('cuda');backend=Backend(model,gpu,processor);source=read(SOURCE/'rollouts'/f'{row["sample_key"]}.json')['result']['candidates'];ids=source[0]['ids'];other=source[1]['ids'];backend.reset(2048)
 for depth in range(1,min(14,len(ids))):backend.get(ids[:depth])
 backend.get(other[:12]);cached=torch.softmax(backend.get(ids[:14]),dim=-1);hits=backend.restore_hits
 reference=Backend(model,gpu,processor);reference.reset(2048);fresh=torch.softmax(reference.get(ids[:14],store=False),dim=-1);delta=float((cached-fresh).abs().max());assert delta<.005 and int(cached.argmax())==int(fresh.argmax()) and hits>0,(delta,hits)
 assert [e['token'] for e in eligible(cached.numpy())]==[e['token'] for e in eligible(fresh.numpy())]
 backend.reset(3)
 try:backend.get(ids[:8],store=False);raise AssertionError('Expected a hard boundary during replay')
 except BudgetBoundary:pass
 assert backend.tokens==3 and len(backend.path)==3 and backend.cache.get_seq_length()==backend.n+3
 backend.limit=8;resumed=torch.softmax(backend.get(ids[:8],store=False),dim=-1);expected=torch.softmax(reference.get(ids[:8],store=False),dim=-1);resume_delta=float((resumed-expected).abs().max());assert backend.tokens==8 and resume_delta<.005
 del reference;gc.collect();torch.cuda.empty_cache();old=search.BUDGETS;search.BUDGETS=[32,64];results={}
 try:
  for method in METHODS:
   progress(stage='gpu_smoke',method=method);r=search.Search(backend,method,info['duration'],progress).run();s=r['snapshots']['64'];
   if method=='native_mass':write(OUT/'validation/smoke_tree.json',r)
   assert s['actual_inputs']<=64;assert abs(s['mass_total']-1)<5e-4;results[method]=dict(cost=s['cost'],nodes=s['nodes'],reason=s['reason'],levin_order_verified=s['levin_order_verified'])
 finally:search.BUDGETS=old
 write(OUT/'validation/gpu_smoke.json',dict(passed=True,config_hash=digest(CONFIG),tested_source_hashes={name:sha(Path(__file__).with_name(name)) for name in ('probability_budget_core.py','probability_budget_inputs.py','probability_budget_backend.py','probability_budget_search.py','probability_budget_runtime.py')},input_hashes=True,kv_restoration_max_probability_difference=delta,kv_restore_hits=hits,native_admission_identical=True,partial_replay_boundary_inputs=3,resumed_replay_total_inputs=8,resumed_probability_difference=resume_delta,smoke_budgets=[32,64],methods=results,timestamp=stamp()));del backend,gpu,cpu;gc.collect();torch.cuda.empty_cache()

_REPORT=None
_REPORT_TIME=0.
def report_partial(final=False):
 global _REPORT,_REPORT_TIME
 if _REPORT is not None:
  if final:
   if _REPORT.wait()!=0:raise RuntimeError('Report failed; logs/report.log')
  elif _REPORT.poll() is None:return
  elif _REPORT.returncode!=0:raise RuntimeError('Report failed; logs/report.log')
 if not final and time.monotonic()-_REPORT_TIME<300:return
 with (OUT/'logs/report.log').open('ab',buffering=0) as stream:_REPORT=subprocess.Popen([sys.executable,str(Path(__file__).with_name('probability_budget_report.py'))],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
 _REPORT_TIME=time.monotonic()
 if final and _REPORT.wait()!=0:raise RuntimeError('Final report failed; logs/report.log')

def run(smoke_only=False):
 import torch
 from probability_budget_checks import cpu_check,validate_scope
 manifest=read(OUT/'metadata/manifest.json');assert manifest['config']==CONFIG and manifest['cohort_sha256']==sha(COHORT)
 for entry in manifest['files']:assert sha(entry['snapshot'])==entry['sha256'],entry['snapshot']
 write(OUT/'validation/cpu.json',cpu_check());progress=Progress();progress(stage='loading_model');model,processor,load_seconds=load_model()
 if smoke_only:smoke(model,processor,progress);progress(state='smoke_complete',stage='smoke_complete');progress.stop.set();return
 records=read(OUT/'metadata/model_loads.json') if (OUT/'metadata/model_loads.json').exists() else [];records.append(dict(seconds=load_seconds,purpose='production',timestamp=stamp()));write(OUT/'metadata/model_loads.json',records);rows=read(COHORT);completed_keys={r['sample_key'] for r in rows if (OUT/'queries'/f'{r["sample_key"]}.json').exists() and read(OUT/'queries'/f'{r["sample_key"]}.json').get('complete')};completed_counts=collections.Counter(r['dataset'] for r in rows if r['sample_key'] in completed_keys)
 progress(stage='resuming' if completed_keys else 'inference',completed=len(completed_keys),completed_by_dataset=dict(completed_counts),method_completed={m:len(completed_keys) for m in METHODS})
 for scope in CONFIG['stages']:
  if (OUT/'validation'/f'pilot{scope}.json').exists():continue
  progress(stage='inference',active_queries=scope)
  for row in rows[:scope]:
   key=row['sample_key'];query_path=OUT/'queries'/f'{key}.json';old=read(query_path) if query_path.exists() else None
   if old and old.get('complete'):continue
   progress(stage='preparing_input',sample_key=key);cpu,info=prepare(row,processor);gpu=cpu.to('cuda');torch.cuda.reset_peak_memory_stats();backend=Backend(model,gpu,processor);info.update(prefill_wall_seconds=backend.prefill_seconds,prefill_cuda_seconds=backend.prefill_cuda_seconds)
   if old:data=old;data.setdefault('resume_input_costs',[]).append(info)
   else:data=dict(sample_key=key,input=info,greedy=greedy(backend,info['duration']),config_hash=digest(CONFIG),methods={},complete=False)
   write(query_path,data);query_started=time.monotonic()
   for method in METHODS:
    target=OUT/'search'/key/f'{method}.json'
    if target.exists():
     prior=read(target);assert prior['config_hash']==digest(CONFIG);data['methods'][method]=dict(snapshots=prior['snapshots'],source=str(target));continue
    progress(stage='searching',sample_key=key,method=method,decoder_inputs=0)
    result=Search(backend,method,info['duration'],progress).run();write(target,result);data['methods'][method]=dict(snapshots=result['snapshots'],source=str(target));write(query_path,data)
   data.update(complete=True,query_search_wall_seconds=time.monotonic()-query_started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),timestamp=stamp());write(query_path,data);del backend,gpu,cpu;gc.collect();torch.cuda.empty_cache()
   completed_keys.add(key);completed_counts[row['dataset']]+=1;progress(stage='query_complete',completed=len(completed_keys),completed_by_dataset=dict(completed_counts),method_completed={m:len(completed_keys) for m in METHODS},eta_seconds=(900-len(completed_keys))*(time.monotonic()-progress.start)/len(completed_keys));report_partial()
  progress(stage='reporting_and_validating',active_queries=scope);report_partial(final=True);validate_scope(scope)
  subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True);subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
  subprocess.run([sys.executable,str(Path(__file__).with_name('probability_budget_view_check.py')),'--scope',str(scope)],check=True)
  write(OUT/'validation'/f'pilot{scope}.json',dict(passed=True,queries=scope,reports_and_viewer_verified=True,timestamp=stamp()));progress(stage='pilot300_complete' if scope==300 else 'postprocessing',completed=scope)
 from artifact_store import register
 register('probability_budget.pilot900.search','rollouts',OUT/'search','LevinTS, native-mass allocation, bounded numeric novelty; per-query 512/1024/2048 decoder positions; nested300 then900','probability_budget_runtime.py',str(COHORT),'complete')
 progress(state='complete',stage='complete',completed=900);progress.stop.set()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
 try:
  with (OUT/'metadata/run.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.smoke)
 except BaseException:
  write(OUT/'metadata/status.json',dict(state='failed',stage='failed',error=traceback.format_exc(),timestamp=stamp()));raise
