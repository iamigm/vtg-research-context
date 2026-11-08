"""Detached single-GPU exhaustive trace, followed by two embedding phases."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,collections,gc,json,math,sqlite3,sys,threading,time,traceback,fcntl
from pathlib import Path
from types import MethodType
from concurrent.futures import ThreadPoolExecutor
from interval_distribution_core import *
sys.path.insert(0,str(ROOT.parent/'_shared/timelens/runtime/interval_distribution_deps'))
import fps_spatial_control as inputs_module
sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
from temporal_mode_core import Backend,parse_clip
sys.path.insert(0,str(Path(__file__).resolve().parent))

class Progress:
 def __init__(self):
  self.lock=threading.RLock();self.started=time.monotonic();self.stop=threading.Event();self.state=dict(state='running',stage='starting',completed=0,total=900,decoding_completed=0,instruct_completed=0,embedding_completed=0,eta_seconds=None,eta_scope='Unavailable before representative completed queries; full admitted tree size is unknown')
  threading.Thread(target=self.heartbeat,daemon=True).start()
 def update(self,**kw):
  with self.lock:
   self.state.update(kw);self.state.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.started)
   write(OUT/'metadata/status.json',self.state)
   with (OUT/'logs/progress.log').open('a') as f:f.write(json.dumps(self.state,ensure_ascii=False)+'\n')
 def heartbeat(self):
  while not self.stop.wait(20):self.update()

def source_frames(row):
 from result_layout import relocate_path
 from shared_selected_frames import load_selected
 prior=read(SOURCE/'inputs'/f'{row["sample_key"]}.json');cache=read(relocate_path(Path(prior['cache_metadata'])));plan=cache['plans']['2'];ids=plan['frame_ids'];adapter=dict(cache_key=digest(dict(source=cache['key'],hw=cache['hw'],kind='fps_selected')),path=cache['array'],ids=cache['ids'])
 frames=load_selected(adapter,ids);loc={i:j for j,i in enumerate(cache['ids'])}
 assert all(hashlib.sha256(frame.tobytes()).hexdigest()==cache['rgb_hashes'][loc[ident]] for frame,ident in zip(frames,ids))
 return prior,cache,frames

def processor_input(processor,frames,ids,pts,sourcefps,N,query=None,direct_text=None):
 import torch
 if query is not None or direct_text is not None:
  from evaluation.utils import GROUNDER_PROMPT
  messages=[dict(role='user',content=[dict(type='video',video='frames'),dict(type='text',text=direct_text if direct_text is not None else GROUNDER_PROMPT.format(query))])]
 else:messages=[dict(role='system',content=[dict(type='text',text=CONFIG['semantic_instruction'])]),dict(role='user',content=[dict(type='video',video='frames')])]
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 if query is None and direct_text is None:text+='<|endoftext|>'
 mapping=dict(zip(ids,pts))
 def timestamps(self,indices,video_fps,merge_size=2):
  indices=list(indices);indices+=indices[-1:]*((-len(indices))%merge_size)
  return [(mapping[indices[j]]+mapping[indices[j+merge_size-1]])/2 for j in range(0,len(indices),merge_size)]
 original=processor._calculate_timestamps;processor._calculate_timestamps=MethodType(timestamps,processor)
 try:return processor(text=[text],videos=[torch.from_numpy(np.asarray(frames)).float()],video_metadata=[dict(fps=sourcefps,frames_indices=list(ids),total_num_frames=N,video_backend='decord')],do_resize=False,do_sample_frames=False,padding=True,return_tensors='pt')
 finally:processor._calculate_timestamps=original

def prepare(row,processor):
 from adaptive_temporal_sampling import tensorhash
 prior,cache,frames=source_frames(row);plan=cache['plans']['2'];cpu=processor_input(processor,frames,plan['frame_ids'],plan['source_pts'],cache['source_fps'],cache['N'],row['query']);hashes={k:tensorhash(v) for k,v in cpu.items()};assert hashes==prior['hashes'],row['sample_key']
 write(OUT/'inputs'/f'{row["sample_key"]}.json',dict(source=str(SOURCE/'inputs'/f'{row["sample_key"]}.json'),hashes=hashes,hw=cache['hw'],frame_ids=plan['frame_ids'],timestamps=plan['source_pts'],clipping_duration=cache['duration'],uses_gt=False))
 return cpu,cache['duration']

def trace_query(row,model,processor,progress,smoke_nodes=None):
 import torch
 key=row['sample_key'];dbpath=OUT/'trees'/f'{key}.sqlite';db=connect(dbpath)
 if read_meta(db,'trace_complete',False):db.close();return True
 prepare_started=time.monotonic();cpu,duration=prepare(row,processor);prepare_seconds=time.monotonic()-prepare_started;gpu=cpu.to('cuda');eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
 backend=Backend(model,gpu,torch,processor.tokenizer,eos);root_logits=backend.root;assert len(root_logits)>1000
 db.execute('INSERT OR IGNORE INTO nodes(id,parent,token,p,logp,depth,status) VALUES(0,NULL,NULL,1,0,0,\'pending\')');db.commit()
 node_count=0;cuda_seconds=0.;started=time.monotonic();inputtokens=int(cpu.input_ids.shape[1]);max_context=int(model.config.text_config.max_position_embeddings)
 def path_for(ident):
  ids=[]
  while ident:
   parent,token=db.execute('SELECT parent,token FROM nodes WHERE id=?',(ident,)).fetchone();ids.append(token);ident=parent
  return tuple(reversed(ids))
 while True:
  item=db.execute("SELECT id,parent,token,p,logp,depth FROM nodes WHERE status='pending' ORDER BY id DESC LIMIT 1").fetchone()
  if item is None:break
  ident,parent,token,p,logp,depth=item;path=path_for(ident)
  if token in eos:
   text=processor.tokenizer.decode(path,skip_special_tokens=True,clean_up_tokenization_spaces=False);parsed=parse_clip(text,duration,True);status='valid' if parsed['valid'] else 'invalid';db.execute('UPDATE nodes SET status=?,span=?,raw=?,reason=?,text=? WHERE id=?',(status,json.dumps(parsed['span']),json.dumps(parsed['raw']),parsed['reason'],text,ident))
  elif inputtokens+depth>=max_context:
   db.execute("UPDATE nodes SET status='context_limit',reason='model context capacity, not a search budget' WHERE id=?",(ident,))
  else:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();logits=backend.get(path);b.record();b.synchronize();cuda_seconds+=a.elapsed_time(b)/1000
   probabilities=torch.softmax(logits,dim=-1);edges,entropy=admitted(probabilities.numpy());
   for edge in edges:edge['text']=processor.tokenizer.decode([edge['token']],clean_up_tokenization_spaces=False)
   prefix=processor.tokenizer.decode(path,skip_special_tokens=True,clean_up_tokenization_spaces=False) if len(edges)>1 else None
   db.execute('UPDATE nodes SET status=?,entropy=?,accepted=?,prefix_text=? WHERE id=?',('expanded' if edges else 'dead',entropy,json.dumps(edges),prefix,ident))
   # Reverse insertion makes the largest native-probability continuation run first.
   for edge in reversed(edges):db.execute("INSERT INTO nodes(parent,token,p,logp,depth,status) VALUES(?,?,?,?,?,'pending')",(ident,edge['token'],edge['p'],logp+math.log(edge['p']),depth+1))
  node_count+=1
  if node_count%16==0 or smoke_nodes is not None:
   db.commit();counts=dict(db.execute('SELECT status,count(*) FROM nodes GROUP BY status'))
   progress.update(stage='decoding',sample_key=key,query_nodes=sum(counts.values()),query_counts=counts,maximum_depth=depth,query_elapsed_seconds=time.monotonic()-started,decoder_forwards=backend.actual_forwards,decoder_inputs=backend.actual_tokens,native_cuda_seconds=cuda_seconds)
  if smoke_nodes is not None and node_count>=smoke_nodes:
   db.commit();db.close();del backend,gpu,cpu;gc.collect();torch.cuda.empty_cache();return False
 db.commit();summary,intervals=summarize_tree(dbpath);groups,assign=grouped(intervals)
 for interval in intervals:interval['group']=assign[interval['id']]
 baseline=read(SOURCE/'rollouts'/f'{key}.json')['result']['candidates'][0];greedy_extra=max(0,len(baseline['ids'])-1)
 summary.update(sample_key=key,config_hash=digest(CONFIG),duration=duration,intervals=intervals,groups=groups,baseline_reference=dict(source=str(SOURCE/'rollouts'/f'{key}.json'),decoder_inputs=inputtokens+greedy_extra,native_forwards=1+greedy_extra,output_tokens=len(baseline['ids']),definition='one identical video/query prefill plus one token forward before each remaining greedy token, including EOS prediction'),cost=dict(input_prepare_wall_seconds=prepare_seconds,prefill_inputs=inputtokens,prefill_forwards=1,prefill_seconds=backend.prefill_seconds,decoder_forwards=backend.actual_forwards,decoder_inputs=backend.actual_tokens,native_cuda_seconds=cuda_seconds,wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()),timestamp=stamp())
 write(OUT/'queries'/f'{key}.json',summary);db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('trace_complete',json.dumps(True)));db.commit();db.close();del backend,gpu,cpu;gc.collect();torch.cuda.empty_cache();return summary['complete']

def read_meta(db,key,default=None):
 r=db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone();return json.loads(r[0]) if r else default

def load_model(path):
 import torch
 torch.set_num_threads(2)
 from transformers import AutoProcessor,AutoModelForImageTextToText
 load_started=time.monotonic();processor=AutoProcessor.from_pretrained(path,local_files_only=True,do_resize=False)
 if Path(path)==EMBED:
  import importlib.util
  source=EMBED/'scripts/qwen3_vl_embedding.py';spec=importlib.util.spec_from_file_location('interval_embedding_backbone',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  model=module.Qwen3VLForEmbedding.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
 else:model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
 record=dict(model=str(path),purpose='smoke' if '--smoke' in sys.argv else 'production',active_queries=read(OUT/'metadata/status.json').get('active_queries',300),wall_seconds=time.monotonic()-load_started,timestamp=stamp());p=OUT/'metadata/model_loads.json';records=read(p) if p.exists() else [];records.append(record);write(p,records)
 return model,processor

def check_manifest():
 manifest=read(OUT/'metadata/manifest.json')
 assert manifest['config']==CONFIG
 assert sha(COHORT)==manifest['cohort_sha256']
 for f in manifest['files']:assert sha(f['snapshot'])==f['sha256'],f['snapshot']

def run(smoke=False):
 import torch
 from interval_distribution_semantic import semantic_video
 check_manifest();progress=Progress();rows=read(COHORT)
 videos=collections.OrderedDict()
 for row in rows:videos.setdefault(row['video_path'],[]).append(row)
 # Start with one example per dataset, preserving cohort membership and order in storage.
 first=[next(v for v in videos if videos[v][0]['dataset']==ds) for ds in sorted({r['dataset'] for r in rows})]
 ordered=first+[v for v in videos if v not in first]
 if smoke:
  progress.update(stage='loading_instruct',model=str(MODEL));model,processor=load_model(MODEL)

  sample=rows[0];trace_query(sample,model,processor,progress,smoke_nodes=12)
  from interval_distribution_semantic import clip_catalog,encode
  from interval_distribution_verify import assess
  prior,cache,frames=source_frames(sample);plan=cache['plans']['2'];positions=[i for i,t in enumerate(plan['source_pts']) if 27<=t<35];positions=positions[:16]
  clip=dict(positions=positions,relative_pts=[plan['source_pts'][i]-plan['source_pts'][positions[0]] for i in positions])
  vectors,cost=encode(model,processor,frames,clip,'instruct');assert set(vectors)=={'instruct_mean','instruct_temporal_four','instruct_readout'}
  assert all(abs(float(np.linalg.norm(v))-1)<1e-5 for v in vectors.values());instruct_cost=cost
  del model,processor,vectors;gc.collect();torch.cuda.empty_cache()
  model,processor=load_model(EMBED);vectors,embedding_cost=encode(model,processor,frames,clip,'embedding');assert set(vectors)=={'embedding_readout'}
  del model,processor,vectors;gc.collect();torch.cuda.empty_cache()
  model,processor=load_model(VERIFIER);selected=np.array(frames[positions]);pts=clip['relative_pts'];verification=assess(model,processor,selected,pts,sample['query'],[0.,8.],progress,validate=True);assert all(0<=verification[k]<=1 for k in ('event_score','joint_score'))
  write(OUT/'validation/gpu_smoke.json',dict(passed=True,query=sample['sample_key'],trace_nodes=12,input_hashes_verified=True,instruct=instruct_cost,embedding=embedding_cost,independent_verifier=verification,timestamp=stamp()))
  progress.stop.set();return
 from interval_distribution_verify import verify_video
 from interval_distribution_check import validate
 import subprocess
 for scope in (300,900):
  if (OUT/'validation'/f'pilot{scope}.json').exists():continue
  cohort=rows[:scope];stage_videos=collections.OrderedDict()
  for row in cohort:stage_videos.setdefault(row['video_path'],[]).append(row)
  # Existing first300 records are referenced unchanged during the900 stage.
  progress.update(stage='loading_instruct',active_queries=scope,model=str(MODEL));model,processor=load_model(MODEL)
  for video,subset in stage_videos.items():
   for row in subset:
    if (OUT/'queries'/f'{row["sample_key"]}.json').exists():continue
    retries=0
    while True:
     try:trace_query(row,model,processor,progress);break
     except torch.cuda.OutOfMemoryError:
      retries+=1;progress.update(stage='waiting_for_gpu_memory',retry=retries,sample_key=row['sample_key'],error=traceback.format_exc());gc.collect();torch.cuda.empty_cache()
      if retries>=3:raise
      for _ in range(6):time.sleep(10);progress.update()
    counts=collections.Counter(r['dataset'] for r in rows if (OUT/'queries'/f'{r["sample_key"]}.json').exists());progress.update(decoding_completed=sum(counts.values()),decoding_by_dataset=dict(counts))
   semantic_video(subset,model,processor,'instruct',progress)
   progress.update(instruct_completed=sum((OUT/'semantic/instruct'/f'{r["sample_key"]}.json').exists() for r in rows));report_partial()
  del model,processor;gc.collect();torch.cuda.empty_cache();progress.update(stage='loading_embedding',model=str(EMBED));model,processor=load_model(EMBED)
  for subset in stage_videos.values():
   semantic_video(subset,model,processor,'embedding',progress)
   progress.update(embedding_completed=sum((OUT/'semantic/embedding'/f'{r["sample_key"]}.json').exists() for r in rows));report_partial()
  del model,processor;gc.collect();torch.cuda.empty_cache();progress.update(stage='loading_independent_verifier',model=str(VERIFIER));model,processor=load_model(VERIFIER)
  for subset in stage_videos.values():
   verify_video(subset,model,processor,progress)
   progress.update(completed=sum((OUT/'verification'/f'{r["sample_key"]}.json').exists() and read(OUT/'verification'/f'{r["sample_key"]}.json')['complete'] for r in rows));report_partial()
  del model,processor;gc.collect();torch.cuda.empty_cache();progress.update(stage='validating_pilot',active_queries=scope);report_partial(final=True)
  validate(scope)
  subprocess.run([sys.executable,str(Path(__file__).with_name('interval_distribution_view_check.py')),'--scope',str(scope)],check=True)
  subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True);subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
  write(OUT/'validation'/f'pilot{scope}.json',dict(passed=True,queries=scope,decoding_semantic_verification_complete=True,reports_and_viewer_passed=True,timestamp=stamp()))
  progress.update(stage='pilot300_complete' if scope==300 else 'postprocessing',completed=scope)
 from artifact_store import register
 register('interval_distribution.pilot900.trace','token_traces',OUT/'trees','Deterministic top-p .99, raw floor .01; nested300/900; independent4B clip verification','interval_distribution_runtime.py',str(COHORT),'complete')
 progress.update(state='complete',stage='complete',completed=900);progress.stop.set()

_REPORT_PROCESS=None
_LAST_REPORT=0.
def report_partial(final=False):
 global _REPORT_PROCESS,_LAST_REPORT
 import subprocess
 if _REPORT_PROCESS is not None:
  code=_REPORT_PROCESS.poll()
  if final:
   if _REPORT_PROCESS.wait()!=0:raise RuntimeError('Report generation failed; see logs/report.log')
  elif code is None:return
  elif code!=0:raise RuntimeError('Report generation failed; see logs/report.log')
 if not final and time.monotonic()-_LAST_REPORT<300:return
 with (OUT/'logs/report.log').open('ab',buffering=0) as stream:
  _REPORT_PROCESS=subprocess.Popen([sys.executable,str(Path(__file__).with_name('interval_distribution_report.py'))],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
 _LAST_REPORT=time.monotonic()
 if final and _REPORT_PROCESS.wait()!=0:raise RuntimeError('Final report failed; see logs/report.log')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
 try:
  with (OUT/'metadata/run.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.smoke)
 except BaseException:
  write(OUT/'metadata/status.json',dict(state='failed',stage='runtime_failed',error=traceback.format_exc(),timestamp=stamp(),total=900));raise
