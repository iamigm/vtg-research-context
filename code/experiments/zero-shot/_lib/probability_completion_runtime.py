"""Complete fixed pilot300 before processing the remaining nested900 queries."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,fcntl,gc,json,math,subprocess,sys,threading,time,traceback
from probability_completion_core import *
from probability_completion_models import Models,Crops,FinalSession
from probability_budget_inputs import processor_input

class Progress:
 def __init__(self):
  self.start=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event();self.state=dict(state='running',stage='starting',completed=0,total=900,active_queries=300,eta_seconds=None,eta_scope='remaining postprocessing at observed throughput; excludes waiting for proposal search and report validation');threading.Thread(target=self.heartbeat,daemon=True).start()
 def __call__(self,**kw):
  with self.lock:
   if kw.get('sample_key') and kw['sample_key']!=self.state.get('sample_key'):
    for key in ('clips_done','clips_total','condition','variant'):self.state.pop(key,None)
   self.state.update(kw,timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start);write(OUT/'metadata/status.json',self.state)
   with (OUT/'logs/progress.log').open('a') as stream:stream.write(json.dumps(self.state,ensure_ascii=False)+'\n')
   print(json.dumps(kw,ensure_ascii=False),flush=True)
 def heartbeat(self):
  while not self.stop.wait(20):self()

def condition_modes(spans,c,h=1.):
 ids=[m['id'] for m in c['members']];mode=temporal_modes([spans[i] for i in ids],[m['native_mass'] for m in c['members']],h);mode['assignment']={ids[i]:g for i,g in enumerate(mode['assignment'])}
 for group in mode['groups']:
  group['members']=[ids[i] for i in group['members']];group['representative']=ids[group['representative']]
 return mode

def process(row,models,progress):
 key=row['sample_key'];target=OUT/'queries'/f'{key}.json';q,trees,spans,sets=proposals(key);old=read(target) if target.exists() else None
 if old and old.get('complete'):return
 started=time.monotonic();progress(sample_key=key,stage='preparing_crops');public={k:row[k] for k in ('sample_key','query','video_path','duration','dataset')};crops=Crops(public,spans)
 source=dict(query_sha256=sha(PARENT/'queries'/f'{key}.json'),methods={m:sha(PARENT/'search'/key/f'{m}.json') for m in METHODS},input_hashes=q['input']['hashes'])
 if old:assert old['source']==source and old['settings_hash']==digest(SETTINGS);data=old
 else:data=dict(sample_key=key,source=source,settings_hash=digest(SETTINGS),spans=[list(s) for s in spans],crop_assignment=crops.assignment,crops=crops.specs,fallback_frames=crops.fallback_frame_count,conditions={},phase_costs={},generations={},complete=False)
 write(target,data)
 if 'verifications' not in data:
  values,cost=models.run_crops(crops,'verifier',progress);data['verifications']=[values[i] for i in range(len(crops.specs))];data['phase_costs']['verifier']=cost;write(target,data)
 if not data.get('semantic_complete'):
  vectors={}
  for kind in ('instruct','embedding'):
   values,cost=models.run_crops(crops,kind,progress);data['phase_costs'][kind]=cost
   for name in values[0] if values else []:
    if name!='cost':vectors[name]=np.stack([values[crops.assignment[i]][name] for i in range(len(spans))])
  progress(stage='temporal_and_semantic_statistics');before=time.monotonic();greedy=spans.index(tuple(q['greedy']['span'])) if q['greedy']['span'] and tuple(q['greedy']['span']) in spans else None
  for name,c in sets.items():
   mode=condition_modes(spans,c);members=c['members'];condition=dict(method=c['method'],budget=c['budget'],members=members,modes=mode,sensitivity={str(h):len(condition_modes(spans,c,h)['groups']) for h in (.5,1.,2.)},semantic=semantic_summary(spans,members,vectors,greedy,mode),branches=branch_statistics(trees[c['method']],c['snapshot'],spans,mode),proposal_selected_span=c['snapshot']['selected_span'])
   if members:condition['verified_selected_id']=max(members,key=lambda m:(data['verifications'][crops.assignment[m['id']]]['score'],m['native_mass'],-m['id']))['id']
   else:condition['verified_selected_id']=None
   data['conditions'][name]=condition
  data['phase_costs']['statistics_wall_seconds']=time.monotonic()-before;data['semantic_complete']=True;write(target,data);del vectors,values
 progress(stage='preparing_final_generation');before=time.monotonic();cache=crops.cache;plan=cache['plans']['2'];cpu=processor_input(models.processors['instruct'],crops.frames,plan['frame_ids'],plan['source_pts'],cache['source_fps'],cache['N'],row['query'])
 from adaptive_temporal_sampling import tensorhash
 assert {k:tensorhash(v) for k,v in cpu.items()}==q['input']['hashes']
 session=FinalSession(models.models['instruct'],models.processors['instruct'],cpu,q['greedy'],q['input']['duration']);data['phase_costs']['final_prefill']=dict(input_tokens=session.n,wall_seconds=session.prefill_wall,input_prepare_wall_seconds=time.monotonic()-before-session.prefill_wall)
 for name,c in data['conditions'].items():
  for variant in ('proposal_only','reviewed'):
   rows=[]
   for item in c['members']:
    ident=item['id'];a,b=spans[ident];text=f'#{ident} [{a:.6f}, {b:.6f}]'
    if variant=='reviewed':
     score=data['verifications'][crops.assignment[ident]];text+=f" independent_clip_score={score['score']:.4f}"
    rows.append(text)
   prompt='Target event: '+row['query']+'\nCandidate intervals in original video seconds:\n'+'\n'.join(rows)
   if variant=='reviewed':prompt+='\nEach independent clip score is the expected 0/1/2 evidence rating divided by2. It is noisy, not ground truth. Prefer clear visual evidence and verify boundaries against the full video.'
   prompt+='\nReconsider your original answer using the video and these proposals. You may select a proposal or refine its boundaries. Reply only: The event happens in <start> - <end> seconds.'
   signature=digest(prompt);c[variant]=signature
   if signature not in data['generations']:
    progress(stage='final_generation',condition=name,variant=variant);data['generations'][signature]=session.generate(prompt);write(target,data)
 data['phase_costs']['final_unique_input_tokens']=session.inputs;data['phase_costs']['final_unique_forward_calls']=session.calls;data['phase_costs']['final_unique_cuda_seconds']=session.cuda
 data.update(complete=True,postprocess_wall_seconds=time.monotonic()-started,timestamp=stamp());write(target,data);del session,crops,cpu;gc.collect();models.torch.cuda.empty_cache()

def report(final=False):
 with (OUT/'logs/report.log').open('ab',buffering=0) as stream:subprocess.run([sys.executable,str(Path(__file__).with_name('probability_completion_report.py'))],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def validate_scope(count):
 rows=read(COHORT)[:count]
 for row in rows:
  data=read(OUT/'queries'/f'{row["sample_key"]}.json');assert data['complete'] and data['semantic_complete'];assert len(data['conditions'])==len(METHODS)*len(BUDGETS);assert len(data['verifications'])==len(data['crops']);assert len(data['crop_assignment'])==len(data['spans']);assert data['source']['query_sha256']==sha(PARENT/'queries'/f'{row["sample_key"]}.json')
  for c in data['conditions'].values():
   assert c['proposal_only'] in data['generations'] and c['reviewed'] in data['generations'];assert set(c['semantic'])=={'instruct_visual_mean','instruct_last','embedding_last'} or not c['members']
  assert not any(p.suffix in ('.npy','.npz','.pt','.bin') for p in (OUT/'queries').glob('*'))
 write(OUT/'validation'/f'pilot{count}.json',dict(passed=True,queries=count,scope='isolated clip verification, both semantic models, proposal-only and reviewed final generation, report, viewer',timestamp=stamp()))

def run():
 import torch
 with (OUT/'metadata/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);progress=Progress();manifest=read(OUT/'metadata/manifest.json');assert manifest['settings']==json.loads(json.dumps(SETTINGS))
  for entry in manifest['files']:assert sha(entry['snapshot'])==entry['sha256'],entry['snapshot']
  models=Models(progress);write(OUT/'metadata/model_loads.json',models.loads);rows=read(COHORT);last_report=0
  for count in SETTINGS['stages']:
   gate=OUT/'validation'/f'pilot{count}.json'
   if gate.exists():continue
   progress(active_queries=count)
   for row in rows[:count]:
    target=OUT/'queries'/f'{row["sample_key"]}.json'
    if target.exists() and read(target).get('complete'):continue
    source=PARENT/'queries'/f'{row["sample_key"]}.json'
    while not source.exists() or not read(source).get('complete'):
     progress(stage='waiting_for_proposals',sample_key=row['sample_key']);time.sleep(20)
    for attempt in range(4):
     try:process(row,models,progress);break
     except torch.OutOfMemoryError:
      if models.batch_size<=1 or attempt==3:raise
      models.batch_size=max(1,models.batch_size//2);progress(stage='retry_after_oom',batch_size=models.batch_size,attempt=attempt+1);gc.collect();torch.cuda.empty_cache()
    done=sum(read(p).get('complete',False) for p in (OUT/'queries').glob('*.json'));progress(stage='query_complete',completed=done,eta_seconds=(900-done)*(time.monotonic()-progress.start)/max(1,done))
    if done==1 or time.monotonic()-last_report>300:report();last_report=time.monotonic()
   progress(stage='reporting_and_validating');report(True);validate_scope(count)
   subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True);subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
   subprocess.run([sys.executable,str(Path(__file__).with_name('probability_completion_view_check.py')),'--scope',str(count)],check=True)
   gate_data=read(gate);gate_data['reports_and_viewer_verified']=True;write(gate,gate_data)
   write(PARENT/'validation'/f'end_to_end_pilot{count}.json',gate_data);progress(stage=f'pilot{count}_end_to_end_complete',completed=count)
  progress(state='complete',stage='complete',completed=900);progress.stop.set()

if __name__=='__main__':
 try:run()
 except BaseException:
  write(OUT/'metadata/status.json',dict(state='failed',stage='failed',error=traceback.format_exc(),timestamp=stamp()));raise
