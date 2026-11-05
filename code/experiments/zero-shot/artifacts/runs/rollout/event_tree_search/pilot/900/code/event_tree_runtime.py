"""Detached pilot900 execution, checkpoints, heartbeat and automatic nested reports."""
from event_tree_core import *
import argparse,collections,fcntl,subprocess,sys,threading,traceback

class Progress:
 def __init__(self,rank,total):
  self.rank=rank;self.total=total;self.started=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event();self.state=dict(state='running',stage='initializing',completed=0,total=total,counts={})
  threading.Thread(target=self.heartbeat,daemon=True).start()
 def update(self,**values):
  with self.lock:
   self.state.update(values);self.state.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.started,percentage=100*self.state['completed']/max(1,self.total))
   write(OUT/'metadata'/f'worker{self.rank}.json',self.state)
   with (OUT/'logs'/f'worker{self.rank}.log').open('a') as f:f.write(json.dumps(self.state,ensure_ascii=False)+'\n')
   print(json.dumps(self.state,ensure_ascii=False),flush=True)
 def heartbeat(self):
  while not self.stop.wait(20):self.update()

def setup():
 cohort=read(COHORT);assert len(cohort)==900
 assert collections.Counter(s['dataset'] for s in cohort)==dict.fromkeys(['charades-timelens','activitynet-timelens','qvhighlights-timelens'],300)
 for n in (300,600):assert read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json')==cohort[:n]
 for name in ('metadata','responses','inputs','logs','validation','code','web','metrics','environment'):(OUT/name).mkdir(parents=True,exist_ok=True)
 REPORT.mkdir(parents=True,exist_ok=True)
 config=OUT/'metadata/config.json'
 if config.exists():assert read(config)==CONFIG,'Frozen config mismatch'
 else:write(config,CONFIG)
 write(OUT/'metadata/cohort_reference.json',dict(path=str(COHORT),sha256=sha(COHORT),queries=900,unique_videos=len({s['video_path'] for s in cohort}),prefixes=[300,600,900]))
 for name in ('event_tree_core.py','event_tree_runtime.py','event_tree_report.py','event_tree_viewer.html','event_tree_check.py'):
  p=Path(__file__).with_name(name)
  if p.exists():(OUT/'code'/name).write_bytes(p.read_bytes())
 write(OUT/'metadata/source_manifest.json',dict(files=[dict(path=str(p),sha256=sha(p)) for p in Path(__file__).parent.glob('event_tree_*') if p.is_file()],backbone_manifest_sha256=sha('/home/irteam/backbones/manifests/registry.json'),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),cpuset=Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text().strip(),started_at=stamp()))
 from event_tree_report import initialize
 initialize(cohort)
 return cohort

def worker(rank,world,limit=None):
 cohort=read(COHORT);rows=[r for r in cohort if int(hashlib.sha256(r['video_path'].encode()).hexdigest(),16)%world==rank]
 # Validate each dataset early without changing the fixed cohort or report ordering.
 first=[next(r for r in rows if r['dataset']==d) for d in sorted({r['dataset'] for r in rows})];first_keys={r['sample_key'] for r in first};rows=first+[r for r in rows if r['sample_key'] not in first_keys]
 if limit:rows=rows[:limit]
 p=Progress(rank,len(rows));done=0;fresh=0;counts=collections.Counter();started=time.monotonic()
 try:
  import torch,transformers
  import fps_spatial_control as inputs_module
  from transformers import AutoProcessor,AutoModelForImageTextToText
  torch.set_num_threads(2);p.update(stage='loading_model')
  proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
  model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
  eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];validated=set()
  write(OUT/'environment'/f'worker{rank}.json',dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),threads=2))
  for row in rows:
   key=row['sample_key'];dest=OUT/'responses'/f'{key}.json';partial=dest.with_suffix('.partial.json')
   if dest.exists():
    data=read(dest);assert data['config_hash']==signature(CONFIG);done+=1;counts[row['dataset']]+=1;continue
   p.update(stage='preparing_input',sample_key=key,method=None,completed=done)
   record=read(SOURCE/'rollouts'/f'{key}.json');assert record['sample']==row
   old=read(SOURCE/'inputs'/f'{key}.json');cache=read(relocate_path(Path(old['cache_metadata'])));sample=public_sample(row)
   cpu,detail=inputs_module.prepare_input(sample,cache,2,'video_query',proc);assert detail['hashes']==old['hashes']==record['input_hashes']
   write(OUT/'inputs'/f'{key}.json',dict(source=str(SOURCE/'inputs'/f'{key}.json'),hashes=detail['hashes'],hw=detail['hw'],frame_ids=detail['plan']['frame_ids'],timestamps=detail['plan']['source_pts'],visual_tokens=detail['visual_tokens'],input_tokens=detail['input_tokens'],uses_gt=False))
   while torch.cuda.mem_get_info()[0]<18*2**30:p.update(state='waiting',stage='waiting_for_gpu_memory');time.sleep(20)
   p.update(state='running',stage='prefill');gpu=cpu.to('cuda');b=Backend(model,gpu,torch,proc.tokenizer,eos)
   if row['dataset'] not in validated:
    b.reset_execution();prefix=[]
    for token in record['result']['candidates'][0]['ids']:
     predicted=int(b.get(prefix).argmax());assert predicted==token,(key,'greedy replay changed',len(prefix),predicted,token);prefix.append(token)
    write(OUT/'validation'/f'input_greedy_worker{rank}_{row["dataset"]}.json',dict(passed=True,key=key,input_hashes=detail['hashes'],exact_greedy_ids=True));validated.add(row['dataset'])
   data=read(partial) if partial.exists() else dict(sample=sample,config_hash=signature(CONFIG),baseline_reference=str(SOURCE/'rollouts'/f'{key}.json'),baseline_sha256=sha(SOURCE/'rollouts'/f'{key}.json'),input_hashes=detail['hashes'],prefill_seconds=b.prefill_seconds,methods={},model_calls={})
   b.records=data.get('model_calls',{});b.meter()
   if 'proposals' not in data:
    p.update(stage='event_proposals');data['proposals']=initial_proposals(b,sample,record);data['model_calls']=b.records;write(partial,data)
   else:assert data['input_hashes']==detail['hashes']
   p.update(stage='proposal_complete',proposal_count=len(data['proposals']['events']),eligible_events=sum(e['assessment']['eligible'] for e in data['proposals']['events']))
   if 'proposal_only' not in data:
    data['proposal_only']={}
    for label,policy in [('wide','value_best_first'),('token_seed','token_seed_best_first')]:
     tree=Search(b,sample,data['proposals'],policy);b.meter(CONFIG['selection_token_cap']);selected=tree.final_select();data['proposal_only'][label]=dict(**selected,nodes=tree.nodes,calls=list(b.active),cost=b.cost())
    data['model_calls']=b.records;write(partial,data)
   methods=METHODS[:];offset=int(hashlib.sha256(key.encode()).hexdigest(),16)%len(methods);methods=methods[offset:]+methods[:offset]
   for method in methods:
    if method in data['methods'] and set(data['methods'][method])=={'4096','8192'}:continue
    p.update(stage='search',method=method,methods_completed=len(data['methods']))
    def save(results):
     data['methods'][method]=results;data['model_calls']=b.records;write(partial,data)
    data['methods'][method]=Search(b,sample,data['proposals'],method).run(save);data['model_calls']=b.records;write(partial,data)
   data['executed_cost']=dict(new_decoder_tokens=b.actual_tokens,new_model_seconds=b.actual_seconds,new_forward_calls=b.actual_forwards,cache_hits=b.cache_hits);data['completed_at']=stamp();write(dest,data);partial.unlink(missing_ok=True)
   done+=1;fresh+=1;counts[row['dataset']]+=1;p.update(stage='inference',completed=done,counts=dict(counts),sample_key=key,methods_completed=len(METHODS),eta_seconds=(time.monotonic()-started)/fresh*(len(rows)-done),eta_scope='assigned remaining inference; excludes final reports; early duration mix may change rate')
   del b,gpu,cpu,data;gc.collect();torch.cuda.empty_cache()
  p.update(state='complete',stage='complete',completed=len(rows),eta_seconds=0)
 except BaseException:p.update(state='failed',stage='failed',error=traceback.format_exc());raise
 finally:p.stop.set()

def supervise():
 cohort=setup()
 with (OUT/'metadata/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);workers=[]
  for rank in range(2):
   stream=(OUT/'logs'/f'worker{rank}_runner.log').open('a');proc=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(rank)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT);stream.close();workers.append(proc)
  write(OUT/'metadata/launcher.json',dict(supervisor_pid=os.getpid(),worker_pids=[p.pid for p in workers],timestamp=stamp()))
  started=time.monotonic();reported=set();n=0
  try:
   while True:
    states=[read(OUT/'metadata'/f'worker{i}.json') if (OUT/'metadata'/f'worker{i}.json').exists() else {} for i in range(2)]
    completed={p.stem for p in (OUT/'responses').glob('*.json') if not p.name.endswith('.partial.json')};n=sum(s['sample_key'] in completed for s in cohort);counts=collections.Counter(s['dataset'] for s in cohort if s['sample_key'] in completed);etas=[s.get('eta_seconds') for s in states if s.get('eta_seconds') is not None]
    status=dict(state='running',stage='inference',completed=n,total=900,percentage=n/9,counts=dict(counts),elapsed_seconds=time.monotonic()-started,timestamp=stamp(),workers=states,eta_seconds=max(etas) if etas else None,eta_scope='remaining worker inference; excludes reports')
    write(OUT/'metadata/status.json',status)
    with (OUT/'logs/progress.log').open('a') as f:f.write(json.dumps(status,ensure_ascii=False)+'\n')
    if any(s.get('state')=='failed' for s in states):raise RuntimeError('Worker failure; see worker runner logs')
    for proc in workers:
     if proc.poll() is not None and proc.returncode!=0:raise RuntimeError(f'Worker {proc.pid} exited {proc.returncode}')
    from event_tree_report import report
    for size in (300,600,900):
     if size not in reported and all(s['sample_key'] in completed for s in cohort[:size]):report(size);reported.add(size)
    if n==900:break
    time.sleep(20)
   for proc in workers:assert proc.wait()==0
   write(OUT/'metadata/status.json',dict(state='complete',stage='complete',completed=900,total=900,percentage=100,counts=dict(counts),elapsed_seconds=time.monotonic()-started,timestamp=stamp(),eta_seconds=0))
  except BaseException:
   write(OUT/'metadata/status.json',dict(state='failed',stage='failed',completed=n,total=900,timestamp=stamp(),error=traceback.format_exc()));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--world',type=int,default=2);parser.add_argument('--limit',type=int);args=parser.parse_args()
 if args.worker is None:
  try:supervise()
  except BaseException:
   write(OUT/'metadata/status.json',dict(state='failed',stage='setup_or_supervision',total=900,timestamp=stamp(),error=traceback.format_exc()));raise
 else:worker(args.worker,args.world,args.limit)
