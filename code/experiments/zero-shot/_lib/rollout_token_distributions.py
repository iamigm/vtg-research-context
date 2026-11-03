"""Replay saved pilot900 prefixes to capture raw full-vocabulary distributions."""
from result_layout import relocate_path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import collections,concurrent.futures,fcntl,gzip,hashlib,json,math,struct,subprocess,threading,time,traceback
from pathlib import Path
from deterministic_rollout300 import ROOT,MODEL,read,write,stamp,sha
from conditional_rollout_budget import Evaluator

OUT=relocate_path(ROOT/'artifacts/runs/deterministic_rollout_pilot900_v2')
TRACES=OUT/'token_traces'
STATUS=OUT/'metadata/distribution_status.json'
CONFIG=dict(schema=1,queries=900,model=str(MODEL),mode='teacher-forced replay of saved generated prefixes; no new answers',
 probability='raw full-vocabulary FP32 softmax, temperature1, no truncation or penalties',
 storage='gzip little-endian float32 probabilities per unique prefix; top30 and observed choices in index',
 replay='single token KV rollback/replay identical to original inference',
 validation='input tensor hashes identical; saved greedy suffix tokens identical; forced probabilities and full-answer log-probabilities checked',
 coverage='all saved candidate token positions, including original and forced continuations',
 resource_gate='>=60 GiB free before loading; >=24 GiB free before each next input',
 priority='100-second/1-second examples, default improvement example, all end-decimal samples, then frozen cohort order')
WRITERS=None

def stable_top(probs,k=30):
 import torch
 threshold=torch.topk(probs,k).values[-1]
 eligible=torch.where(probs>=threshold)[0]
 order=torch.argsort(probs[eligible],descending=True,stable=True)
 return eligible[order[:k]].tolist()

def save_vector(dest,filename,binary):
 raw=gzip.compress(binary,compresslevel=1,mtime=0)
 tmp=dest/(filename+'.tmp');tmp.write_bytes(raw);tmp.replace(dest/filename)
 return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

class Progress:
 def __init__(self,total=900,rank=None):
  self.start=time.monotonic();self.stop=threading.Event();self.lock=threading.RLock()
  self.path=STATUS if rank is None else OUT/'metadata'/f'distribution_worker{rank}.json'
  self.log=OUT/'logs'/('distribution_progress.log' if rank is None else f'distribution_progress_worker{rank}.log')
  self.state=dict(state='running',stage='initializing',completed=0,total=total,counts={},eta_seconds=None,
   eta_scope='remaining prefix replay at observed rate; excludes queue and final validation',model='qwen3-instruct')
  threading.Thread(target=self.heartbeat,daemon=True).start()
 def update(self,**kw):
  with self.lock:
   self.state.update(kw);self.state.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start,
    percentage=100*self.state['completed']/self.state['total'])
   write(self.path,self.state);line=json.dumps(self.state,ensure_ascii=False)
   with self.log.open('a') as f:f.write(line+'\n')
   print(line,flush=True)
 def heartbeat(self):
  while not self.stop.wait(20):self.update()

def free_mib():return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])

def capture(rec,reference,detail,ev,tok,p):
 import numpy as np
 torch=ev.torch;key=rec['sample']['sample_key'];dest=TRACES/key;dest.mkdir(parents=True,exist_ok=True)
 nodes={};steps={};checks=[];pending=collections.deque()
 observed=collections.defaultdict(set)
 for c in rec['result']['candidates']:
  for j,token in enumerate(c['ids']):observed[tuple(c['ids'][:j])].add(token)
 for c in rec['result']['candidates']:
  identifiers=[];logp=0.;forced_length=len(c['branch']['forced_prefix_ids']) if c['branch'] else 0
  for j,chosen in enumerate(c['ids']):
   prefix=tuple(c['ids'][:j]);nodekey=hashlib.sha256(struct.pack('<'+'i'*len(prefix),*prefix)).hexdigest()[:24]
   if nodekey not in nodes:
    logits=ev.get(prefix);lp=torch.log_softmax(logits,-1);probs=torch.exp(lp)
    arr=probs.numpy().astype('<f4',copy=False);total=float(arr.sum(dtype=np.float64))
    assert abs(total-1)<3e-5,(key,j,total)
    ids=stable_top(probs)
    def token_row(i):return dict(id=i,token=tok.decode([i],clean_up_tokenization_spaces=False),probability=float(probs[i]))
    node=dict(step=j,prefix_ids=list(prefix),prefix_text=tok.decode(list(prefix),clean_up_tokenization_spaces=False),
      prefix_log_probability=logp,top=[token_row(i) for i in ids],
      observed=[token_row(i) for i in sorted(observed[prefix],key=lambda i:(-float(probs[i]),i))],
      vocabulary_size=len(arr),top30_mass=float(probs[ids].sum()),total_probability=total,
      entropy_nats=float(-(probs[probs>0]*lp[probs>0]).sum()),file=nodekey+'.f32.gz')
    if WRITERS is None:node.update(save_vector(dest,node['file'],arr.tobytes()))
    else:
     pending.append((node,WRITERS.submit(save_vector,dest,node['file'],arr.tobytes())))
     if len(pending)>=8:
      target,future=pending.popleft();target.update(future.result())
    # Internal fields are removed before serialization.
    node['_observed_logp']={i:float(lp[i]) for i in observed[prefix]}
    node['_argmax']=int(lp.argmax());nodes[nodekey]=node
   node=nodes[nodekey]
   assert node['prefix_ids']==list(prefix)
   assert abs(node['prefix_log_probability']-logp)<.003
   if j>=forced_length:assert node['_argmax']==chosen,(key,c['index'],j,'Greedy suffix changed')
   logp+=node['_observed_logp'][chosen];identifiers.append(nodekey)
  difference=abs(logp-c['log_probability']);assert difference<.003,(key,c['index'],difference)
  if c['branch']:
   b=c['branch'];node=nodes[identifiers[b['position']]]
   actual=math.exp(node['_observed_logp'][b['forced_token']])
   error=abs(actual-b['raw_probability']);assert error<.001,(key,c['index'],error)
  else:error=0.
  steps[str(c['index'])]=dict(nodes=identifiers,log_probability=logp,saved_log_probability=c['log_probability'])
  checks.append(dict(candidate=c['index'],log_probability_error=difference,forced_probability_error=error))
  if c['index'] in [1,8,16,32]:p.update(stage='replaying',sample_key=key,candidate=c['index'],unique_prefixes=len(nodes))
 for target,future in pending:target.update(future.result())
 for node in nodes.values():node.pop('_observed_logp');node.pop('_argmax')
 index=dict(schema=1,sample_key=key,distribution=CONFIG['probability'],raw_record_sha256=reference['sha256'],
  input_hashes=detail['hashes'],nodes=nodes,candidates=steps,validation=dict(passed=True,checks=checks),
  decoder_tokens=ev.text_tokens,prefill_seconds=ev.prefill_seconds)
 write(dest/'index.json',index)
 return len(nodes),sum(n['bytes'] for n in nodes.values())

def main():
 global WRITERS
 rank=int(os.environ['ROLLOUT_DIST_RANK']) if 'ROLLOUT_DIST_RANK' in os.environ else None
 world=int(os.environ.get('ROLLOUT_DIST_WORLD','1'))
 TRACES.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 cfg=OUT/'metadata/distribution_capture.json'
 if cfg.exists():assert read(cfg)==CONFIG
 else:write(cfg,CONFIG)
 references=read(OUT/'metadata/source_records.json');assert len(references)==900
 manifest=read(OUT/'web/manifest.json');priority=['activitynet-timelens_01723','qvhighlights-timelens_00507']
 priority+=[max(manifest['branches'],key=lambda b:b['greedy_delta'])['sample']]
 priority += [s['key'] for s in manifest['samples'] if s['dot_count']]
 priority_rank={key:i for i,key in enumerate(dict.fromkeys(priority))}
 references.sort(key=lambda r:priority_rank.get(Path(r['path']).stem,10000))
 if world>1:
  references=[r for r in references if int(hashlib.sha256(Path(r['path']).stem.encode()).hexdigest()[:8],16)%world==rank]
 lockname='distribution_capture.lock' if world==1 else f'distribution_worker{rank}.lock'
 with (OUT/'metadata'/lockname).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=Progress(total=len(references),rank=rank)
  try:
   for name in ['rollout_token_distributions.py','conditional_rollout_budget.py','fps_spatial_control.py']:
    source=Path(__file__).with_name(name);target=OUT/'code/distribution_capture'/name
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
   while free_mib()<60*1024:p.update(state='waiting',stage='waiting_for_gpu',free_mib=free_mib());time.sleep(20)
   import fps_spatial_control as source
   import torch
   from transformers import AutoProcessor,AutoModelForImageTextToText
   torch.set_num_threads(2);p.update(state='running',stage='loading_model')
   processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
   model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
   WRITERS=concurrent.futures.ThreadPoolExecutor(max_workers=2)
   counts=collections.Counter();done=fresh=0;start=time.monotonic();total_nodes=total_bytes=0
   for reference in references:
    key=Path(reference['path']).stem;rec=read(reference['path']);row=rec['sample']
    assert sha(reference['path'])==reference['sha256']
    existing=TRACES/key/'index.json'
    if existing.exists():
     saved=read(existing);assert saved['raw_record_sha256']==reference['sha256'] and saved['validation']['passed']
     done+=1;counts[row['dataset']]+=1;total_nodes+=len(saved['nodes']);total_bytes+=sum(x['bytes'] for x in saved['nodes'].values());continue
    while free_mib()<24*1024:p.update(state='waiting',stage='waiting_for_gpu',free_mib=free_mib());time.sleep(20)
    p.update(state='running',stage='preparing_input',sample_key=key,candidate=0)
    original_run=Path(reference['path']).parents[1];saved_input=read(original_run/'inputs'/(key+'.json'))
    cache=read(saved_input['cache_metadata'])
    public={k:row[k] for k in ['sample_key','dataset','video_path','query','duration']}
    cpu,detail=source.prepare_input(public,cache,2,'video_query',processor)
    assert detail['hashes']==rec['input_hashes']==saved_input['hashes'],'Input changed'
    inputs=cpu.to('cuda');ev=Evaluator(model,inputs,torch)
    count,size=capture(rec,reference,detail,ev,processor.tokenizer,p)
    del ev,inputs,cpu;torch.cuda.empty_cache()
    done+=1;fresh+=1;counts[row['dataset']]+=1;total_nodes+=count;total_bytes+=size
    p.update(stage='generation',completed=done,counts=dict(counts),total_prefixes=total_nodes,
      total_compressed_bytes=total_bytes,eta_seconds=(time.monotonic()-start)/fresh*(len(references)-done))
   p.update(stage='validating')
   assert all((TRACES/Path(r['path']).stem/'index.json').exists() for r in references)
   if world>1:
    write(OUT/'validation'/f'distribution_worker{rank}.json',dict(passed=True,queries=len(references),prefixes=total_nodes,compressed_bytes=total_bytes))
    p.update(state='complete',stage='complete',completed=len(references),eta_seconds=0)
    return
   write(OUT/'validation/distribution_capture.json',dict(passed=True,queries=900,all_original_answers_preserved=True,
    input_hashes_identical=True,greedy_suffixes_identical=True,forced_probabilities_checked=True,
    full_answer_probability_products_checked=True,prefixes=total_nodes,compressed_bytes=total_bytes))
   from rollout_distribution_integrate import integrate
   integrate()
   p.update(state='complete',stage='complete',completed=900,eta_seconds=0)
  except BaseException:p.update(state='failed',stage='failed',error=traceback.format_exc());raise
  finally:
   p.stop.set()
   if WRITERS:WRITERS.shutdown(wait=True)

if __name__=='__main__':main()
