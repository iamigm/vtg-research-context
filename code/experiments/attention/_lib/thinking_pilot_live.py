"""Thinking pilot: fixed-token replay, per-head category masses and selected-layer maps."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, json, os, subprocess, sys, threading, time, traceback
ROOT=relocate_path(Path(__file__)).resolve().parents[1]
OUT=relocate_path(ROOT/'results/thinking_pilot_orders_v1')
MODEL=relocate_path(Path('/home1/irteam/backbones/models/Qwen3-VL-8B-Thinking'))
MANIFEST=relocate_path(ROOT.parent/'zero-shot/predictions/cohorts/pilot_300/samples.json')
sys.path[:0]=[str(relocate_path(ROOT/'runtime/timelens_deps')),str(relocate_path(ROOT/'vendor/TimeLens'))]
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['FORCE_QWENVL_VIDEO_READER']='decord'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
LAYERS=[0,7,14,21,28,35]
CATEGORIES=['video','input_query','input_other','reasoning','answer','generated_boundary','current_position']
ORDERS=['video_query','query_video']
def read(p):return json.loads(relocate_path(Path(p)).read_text())
def write(p,x):
 p=relocate_path(Path(p));p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
 write_result_text(t, json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with relocate_path(Path(p)).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def folder(s,o):return relocate_path(OUT/'samples'/s['dataset']/s['sample_key']/o)

def prepare(sample):
 import torch,decord
 from functools import partial
 from transformers import AutoProcessor
 from timelens_inputs import prepare_pair
 torch.set_num_threads(1)
 if not hasattr(prepare,'proc'):
  decord.VideoReader=partial(decordo,num_threads=1)
  prepare.proc=AutoProcessor.from_pretrained(MODEL,padding_side='left',do_resize=False,local_files_only=True)
 pair,info=prepare_pair(sample,prepare.proc)
 arrays={k:v.numpy() for k,v in pair[0].items()}
 second={k:(v.numpy() if k=='input_ids' else arrays[k]) for k,v in pair[1].items()}
 return sample,[arrays,second],info
# Decode function is initialized inside the spawned CPU worker.
def prep_init():
 global decordo
 import decord
 decordo=decord.VideoReader

def labels(proc,inputs,sample,generated):
 import numpy as np
 ids=inputs.input_ids[0].tolist();text=proc.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 encoded=proc.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
 assert encoded['input_ids']==ids
 begin=text.index(sample['query']);end=begin+len(sample['query'])
 query=[i for i,(s,e) in enumerate(encoded['offset_mapping']) if s<end and e>begin]
 assert query
 cats=np.full(len(ids)+len(generated),2,dtype=np.int64)
 cats[:len(ids)][np.asarray(ids)==proc.tokenizer.convert_tokens_to_ids('<|video_pad|>')]=0
 cats[query]=1
 close=proc.tokenizer.encode('</think>',add_special_tokens=False);opening=proc.tokenizer.encode('<think>',add_special_tokens=False)
 assert len(close)==1 and len(opening)==1
 eos=set([151643,151645]);closed=False;roles=[]
 for tok in generated:
  if tok in eos or tok in (close[0],opening[0]):role=5
  else:role=4 if closed else 3
  roles.append(role)
  if tok==close[0]:closed=True
 cats[len(ids):]=roles
 return cats,roles,text,query,closed

class Capture:
 def __init__(self,model):
  import transformers.models.qwen3_vl.modeling_qwen3_vl as m
  self.registry=m.ALL_ATTENTION_FUNCTIONS;self.original=self.registry['flash_attention_2']
  self.layer_ids={id(x.self_attn):i for i,x in enumerate(model.model.language_model.layers)}
  self.enabled=False;self.registry.register('flash_attention_2',self.forward)
 def configure(self,n,g,cats,h5,mode):
  import torch,numpy as np
  self.n=n;self.g=g;self.h5=h5;self.mode=mode;self.start=0;self.end=1
  self.cats=torch.as_tensor(cats,device='cuda',dtype=torch.long)
  self.onehot=torch.nn.functional.one_hot(self.cats,7).float()
  self.max_sum_error=0.;self.max_reference_error=0.;self.reference_done=False
  self.coverage=np.zeros((36,g),dtype=np.uint8)
  self.pending={i:[] for i in range(36)}
  self.enabled=True
 def forward(self,module,query,key,value,attention_mask,**kw):
  import torch,numpy as np
  result=self.original(module,query,key,value,attention_mask,**kw)
  if not self.enabled or id(module) not in self.layer_ids:return result
  layer=self.layer_ids[id(module)];s,e=self.start,self.end;rows=e-s
  assert query.shape[0]==1 and (query.shape[-2]==rows or (s==0 and rows==1))
  assert key.shape[-2]==self.n+e-1
  assert attention_mask is None or (attention_mask.ndim==2 and bool(attention_mask.all()))
  q=query[0,:,-rows:,:].float();k=key[0].float()
  heads,_,d=q.shape;kv=k.shape[0];groups=heads//kv
  scores=torch.matmul(q.reshape(kv,groups,rows,d),k[:,None].transpose(-2,-1)).reshape(heads,rows,-1)*kw['scaling']
  positions=torch.arange(self.n+s-1,self.n+e-1,device=q.device)
  visible=torch.arange(k.shape[-2],device=q.device)[None,:]<=positions[:,None]
  scores.masked_fill_(~visible[None],float('-inf'))
  weights=scores.softmax(-1)
  # Independent repeated-K reference verifies grouping and causal mask once per record.
  if layer==0 and not self.reference_done:
   rk=k.repeat_interleave(groups,dim=0)
   ref=(torch.matmul(q[:,:1],rk.transpose(-2,-1))*kw['scaling']).masked_fill(~visible[None,:1],float('-inf')).softmax(-1)
   self.max_reference_error=float((ref-weights[:,:1]).abs().max());assert self.max_reference_error<2e-6
   self.reference_done=True
  mass=torch.matmul(weights,self.onehot[:k.shape[-2]])
  current=weights.gather(-1,positions[None,:,None].expand(heads,-1,1))
  category=self.cats[positions][None,:,None].expand(heads,-1,1)
  mass.scatter_add_(-1,category,-current);mass[:,:,6]+=current[:,:,0]
  sums=mass.sum(-1);err=float((sums-1).abs().max());self.max_sum_error=max(self.max_sum_error,err)
  assert err<5e-5 and float(mass.min())>-2e-6
  mass_cpu=mass.transpose(0,1).cpu().numpy();detail_cpu=None
  if layer in LAYERS:
   if self.mode=='mean':
    detail=weights.mean(0,keepdim=True)
   else:detail=weights.reshape(8,4,rows,-1).mean(1)
   detail_cpu=detail.cpu().numpy()
  self.pending[layer].append((s,e,mass_cpu,detail_cpu))
  if len(self.pending[layer])>=32 or e==self.g:
   pending=self.pending[layer];lo=pending[0][0];hi=pending[-1][1]
   self.h5['category_mass'][layer,lo:hi]=np.concatenate([x[2] for x in pending],axis=0)
   if layer in LAYERS:
    channels=pending[0][3].shape[0];width=self.n+hi-1
    block=np.zeros((channels,hi-lo,width),dtype=np.float32)
    for a,b,_,array in pending:block[:,a-lo:b-lo,:array.shape[-1]]=array
    self.h5[f'layer_{layer:02d}'][:,lo:hi,:width]=block
   self.pending[layer]=[]
  self.coverage[layer,s:e]+=1
  return result

def open_capture(path,n,g,mode,cats,generated):
 import h5py,numpy as np
 f=h5py.File(path,'w');f.attrs['alignment']='predictive: row r uses absolute query N+r-1 to predict generated_ids[r]'
 f.attrs['head_compression']=mode;f.attrs['categories']=json.dumps(CATEGORIES)
 f.attrs['measurement']='FP32 softmax of post-RoPE BF16 Q/K from fixed-token replay; not bitwise live generation attention'
 f.attrs['detail_channels']=json.dumps(['head_mean'] if mode=='mean' else [f'heads_{4*i}_{4*i+3}_mean' for i in range(8)])
 f.attrs['future_key_fill']='zero; consult key_lengths to exclude future positions'
 f.create_dataset('key_lengths',data=n+np.arange(g));f.create_dataset('query_positions',data=n-1+np.arange(g))
 f.create_dataset('generated_ids',data=generated);f.create_dataset('key_category_ids',data=cats[:n+g-1])
 f.create_dataset('category_mass',shape=(36,g,32,7),dtype='f4',chunks=(1,min(32,g),32,7),compression='lzf')
 channels=1 if mode=='mean' else 8
 for layer in LAYERS:f.create_dataset(f'layer_{layer:02d}',shape=(channels,g,n+g-1),dtype='f4',chunks=(1,min(32,g),min(2048,n+g-1)),compression='lzf',fillvalue=0.)
 return f

def replay(model,capture,inputs,generated,cats,path,mode,chunk,update):
 import torch,numpy as np
 n=inputs.input_ids.shape[1];g=len(generated)
 with open_capture(path,n,g,mode,cats,generated) as h5:
  capture.configure(n,g,cats,h5,mode)
  with torch.inference_mode():
   out=model(**inputs,use_cache=True,logits_to_keep=1)
   cache=out.past_key_values;del out
   for s in range(1,g,chunk):
    e=min(g,s+chunk);capture.start=s;capture.end=e
    update(stage='attention_replay',attention_rows=e,generated_tokens=g)
    out=model(input_ids=torch.tensor([generated[s-1:e-1]],device='cuda'),attention_mask=torch.ones((1,n+e-1),device='cuda',dtype=torch.long),past_key_values=cache,cache_position=torch.arange(n+s-1,n+e-1,device='cuda'),use_cache=True,logits_to_keep=1)
    cache=out.past_key_values;del out
  capture.enabled=False
  assert np.all(capture.coverage==1)
  h5.flush()
  validation=dict(passed=True,rows=g,layers=36,heads=32,selected_layers=LAYERS,category_sum_max_error=capture.max_sum_error,grouped_qk_reference_max_error=capture.max_reference_error,coverage_all_once=True)
 del cache
 return validation

def validate_file(path,mode):
 import h5py,numpy as np
 with h5py.File(path,'r') as f:
  lengths=f['key_lengths'][:];g=len(lengths);maxerr=0.
  for s in range(0,g,32):
   e=min(s+32,g);mass=f['category_mass'][:,s:e]
   assert np.isfinite(mass).all() and mass.min()>-2e-6
   err=float(np.abs(mass.sum(-1)-1).max());maxerr=max(maxerr,err);assert err<5e-5
   for l in LAYERS:
    a=f[f'layer_{l:02d}'][:,s:e,:int(lengths[e-1])]
    assert np.isfinite(a).all() and a.min()>=0
    means=a[:1] if mode=='mean' else a
    assert np.allclose(means.sum(-1),1,atol=5e-5)
    for j,r in enumerate(range(s,e)):assert not np.any(a[:,j,int(lengths[r]):])
    # Aggregated detailed mean must equal the all-head summary categories.
    mean=means.mean(0);cats=f['key_category_ids'][:mean.shape[-1]]
    for j,r in enumerate(range(s,e)):
     rowcats=cats.copy();rowcats[int(lengths[r])-1]=6
     totals=np.bincount(rowcats,weights=mean[j],minlength=7)
     assert np.allclose(totals,mass[l,j].mean(0),atol=5e-5)
 return dict(passed=True,category_sum_max_error=maxerr,future_keys_zero=True,detail_matches_summary=True,finite_nonnegative=True)

def probe_replay(model,capture,inputs,generated,cats,where,mode,update):
 """Compare true live-generation attention and single-token fixed replay."""
 import torch,h5py,numpy as np
 from transformers import StoppingCriteria,StoppingCriteriaList
 ids=generated[:min(5,len(generated))];n=inputs.input_ids.shape[1]
 a=relocate_path(where/'probe_live.h5');b=relocate_path(where/'probe_replay.h5')
 class Advance(StoppingCriteria):
  def __call__(self,input_ids,scores,**kw):
   capture.start=input_ids.shape[1]-n;capture.end=capture.start+1
   return False
 with open_capture(a,n,len(ids),mode,cats[:n+len(ids)],ids) as f:
  capture.configure(n,len(ids),cats[:n+len(ids)],f,mode)
  torch.manual_seed(42)
  with torch.inference_mode():
   out=model.generate(**inputs,max_new_tokens=len(ids),do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,use_cache=True,stopping_criteria=StoppingCriteriaList([Advance()]))
  assert out[0,n:].tolist()==ids,'Seeded live prefix differs from saved generation'
  assert np.all(capture.coverage==1)
  capture.enabled=False
 replay(model,capture,inputs,ids,cats[:n+len(ids)],b,mode,1,update)
 errors={}
 with h5py.File(a,'r') as x,h5py.File(b,'r') as y:
  for name in ['category_mass']+[f'layer_{l:02d}' for l in LAYERS]:
   u=x[name][:];v=y[name][:];err=float(np.abs(u-v).max());errors[name]=err
   assert np.allclose(u,v,atol=2e-6,rtol=1e-4),(name,err)
 write(relocate_path(where/'replay_probe.json'),dict(passed=True,rows=len(ids),live_generated_prefix_identical=True,absolute_tolerance=2e-6,relative_tolerance=1e-4,max_abs_errors=errors,note='Single-token fixed replay compared to live generation for the first five tokens.'))
 a.unlink();b.unlink()


def finish_live(capture,h5,n,generated,cats):
 import numpy as np
 g=len(generated)
 for layer,pending in capture.pending.items():
  if not pending:continue
  lo=pending[0][0];hi=pending[-1][1]
  h5['category_mass'][layer,lo:hi]=np.concatenate([x[2] for x in pending],0)
  if layer in LAYERS:
   block=np.zeros((1,hi-lo,n+hi-1),dtype=np.float32)
   for a,b,_,arr in pending:block[:,a-lo:b-lo,:arr.shape[-1]]=arr
   h5[f'layer_{layer:02d}'][:,lo:hi,:n+hi-1]=block
 assert np.all(capture.coverage[:,:g]==1)
 h5['category_mass'].resize((36,g,32,7))
 for layer in LAYERS:h5[f'layer_{layer:02d}'].resize((1,g,n+g-1))
 for name,arr in [('generated_ids',generated),('key_lengths',n+np.arange(g)),('query_positions',n-1+np.arange(g)),('key_category_ids',cats[:n+g-1])]:
  del h5[name];h5.create_dataset(name,data=arr)
 h5.attrs['measurement']='FP32 attention reconstruction from live generation post-RoPE Q/K'

def live(model,proc,capture,inputs,sample,cap,path,update):
 import torch,numpy as np
 from transformers import StoppingCriteria,StoppingCriteriaList
 n=inputs.input_ids.shape[1]
 initial,_,_,_,_=labels(proc,inputs,sample,[])
 cats=np.full(n+cap,5,dtype=np.int64);cats[:n]=initial
 closed=False;close=proc.tokenizer.encode('</think>',add_special_tokens=False)[0]
 opening=proc.tokenizer.encode('<think>',add_special_tokens=False)[0]
 class Advance(StoppingCriteria):
  def __call__(self,input_ids,scores,**kw):
   nonlocal closed
   g=input_ids.shape[1]-n;tok=int(input_ids[0,-1])
   role=5 if tok in [151643,151645,close,opening] else (4 if closed else 3)
   cats[n+g-1]=role;capture.cats[n+g-1]=role
   capture.onehot[n+g-1].zero_();capture.onehot[n+g-1,role]=1
   if tok==close:closed=True
   capture.start=g;capture.end=g+1;update(generated_tokens=g,attention_rows=g)
   return False
 with open_capture(path,n,cap,'mean',cats,[0]*cap) as f:
  capture.configure(n,cap,cats,f,'mean')
  torch.manual_seed(42)
  with torch.inference_mode():
   out=model.generate(**inputs,max_new_tokens=cap,do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,use_cache=True,stopping_criteria=StoppingCriteriaList([Advance()]))
  generated=out[0,n:].tolist();capture.enabled=False
  actual,_,_,_,_=labels(proc,inputs,sample,generated)
  assert np.array_equal(actual,cats[:len(actual)])
  finish_live(capture,f,n,generated,cats);f.flush()
 return generated


def run_record(model,proc,capture,sample,inputs,info,order,args,update,probe=False):
 import torch,numpy as np
 from transformers import StoppingCriteria,StoppingCriteriaList
 from timelens.utils import extract_time,iou
 dest=folder(sample,order);dest.mkdir(parents=True,exist_ok=True)
 if (relocate_path(dest/'complete.json')).exists():return read(relocate_path(dest/'result.json'))
 started=time.time();n=inputs.input_ids.shape[1];inputs=inputs.to('cuda')
 temp=relocate_path(dest/'attention.partial.h5')
 method='replay_saved_generation' if (relocate_path(dest/'generation.json')).exists() else 'live'
 update(measurement_method=method)
 capture.enabled=False
 class Progress(StoppingCriteria):
  def __call__(self,input_ids,scores,**kw):
   update(generated_tokens=int(input_ids.shape[1]-n));return False
 torch.manual_seed(42)
 update(stage='generation',sample_key=sample['sample_key'],dataset=sample['dataset'],order=order,generated_tokens=0)
 rawpath=relocate_path(dest/'generation.json')
 if rawpath.exists():generated=read(rawpath)['generated_ids']
 else:
  update(stage='generation_with_attention')
  generated=live(model,proc,capture,inputs,sample,16384,temp,update)
  write(rawpath,dict(generated_ids=generated,seed=42,input_sha256=hashlib.sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest()))
 generation_seconds=time.time()-started
 cats,roles,text,qpos,closed=labels(proc,inputs,sample,generated)
 raw=proc.tokenizer.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 reasoning_ids=[t for t,r in zip(generated,roles) if r==3];answer_ids=[t for t,r in zip(generated,roles) if r==4]
 reasoning=proc.tokenizer.decode(reasoning_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 answer=proc.tokenizer.decode(answer_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 for name,content in [('expanded_input.txt',text),('output_raw.txt',raw),('reasoning.txt',reasoning),('answer.txt',answer)]: write_result_text((relocate_path(dest/name)), content)
 write(relocate_path(dest/'input.json'),dict(sample=sample,order=order,input_ids=inputs.input_ids[0].tolist(),metadata=info,input_query_positions=qpos))
 write(relocate_path(dest/'tokens.json'),dict(generated_ids=generated,generated_roles=roles,role_names=CATEGORIES,generated_tokens=[proc.tokenizer.decode([t],skip_special_tokens=False,clean_up_tokenization_spaces=False) for t in generated],predictive_query_positions=list(range(n-1,n+len(generated)-1)),alignment='row r predicts generated_ids[r], query position N+r-1'))
 if probe:probe_replay(model,capture,inputs,generated,cats,dest,args.head_mode,update)
 if method=='replay_saved_generation':
  update(stage='attention_replay_saved_generation',attention_rows=0)
  validation=replay(model,capture,inputs,generated,cats,temp,args.head_mode,args.chunk,update)
 else:
  validation=dict(passed=True,rows=len(generated),layers=36,heads=32,selected_layers=LAYERS,category_sum_max_error=capture.max_sum_error,grouped_qk_reference_max_error=capture.max_reference_error,coverage_all_once=True)
 validation['measurement_method']=method
 update(stage='validating')
 validation['saved_file']=validate_file(temp,args.head_mode)
 temp.replace(relocate_path(dest/'attention.h5'))
 spans=extract_time(answer) or [];score=iou(sample['span'][0],spans[0]) if spans else 0.
 result=dict(measurement_method=method,model='Qwen3-VL-8B-Thinking',dataset=sample['dataset'],sample_key=sample['sample_key'],order=order,input_tokens=n,visual_tokens=len(info['video_positions'][order]),reasoning_tokens=len(reasoning_ids),answer_tokens=len(answer_ids),generated_tokens=len(generated),reasoning_characters=len(reasoning),thinking_closed=closed,eos_reached=generated[-1] in [151643,151645],generation_limit_reached=len(generated)>=16384,answer=answer,predicted_spans=spans,iou=score,generation_seconds=generation_seconds if method=='replay_saved_generation' else None,generation_and_capture_seconds=generation_seconds if method=='live' else None,total_seconds=time.time()-started,attention_bytes=(relocate_path(dest/'attention.h5')).stat().st_size)
 write(relocate_path(dest/'result.json'),result);write(relocate_path(dest/'validation.json'),validation)
 write(relocate_path(dest/'complete.json'),dict(passed=True,attention_sha256=sha(relocate_path(dest/'attention.h5')),generation_sha256=sha(rawpath),result_sha256=sha(relocate_path(dest/'result.json'))))
 del inputs;torch.cuda.empty_cache()
 return result


def worker(args):
 import torch
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 import multiprocessing as mp
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 state=dict(stage='loading',completed=0,total=0,model='Qwen3-VL-8B-Thinking',shard=args.shard)
 begin=time.time();stop=threading.Event();lock=threading.Lock()
 def update(**kw):state.update(kw)
 def pulse():
  with lock:
   s={**state,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'elapsed_seconds':time.time()-begin}
   s['percentage']=100*s['completed']/max(1,s['total']);s['eta_seconds']=(time.time()-begin)*(s['total']-s['completed'])/s['completed'] if s['completed'] else None;s['eta_scope']='worker inference + attention capture + validation'
   write(relocate_path(OUT/f'worker_{args.shard}.json'),s);print(json.dumps(s),flush=True)
 def heartbeat():
  while not stop.wait(20):pulse()
 threading.Thread(target=heartbeat,daemon=True).start()
 try:
  samples=read(MANIFEST)[args.shard::args.shards]
  samples=[sample for sample in samples if not all((relocate_path(folder(sample,order)/'complete.json')).exists() for order in ORDERS)]
  if args.limit:samples=samples[:args.limit]
  state['total']=len(samples)*2;pulse()
  proc=AutoProcessor.from_pretrained(MODEL,padding_side='left',do_resize=False,local_files_only=True)
  model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='flash_attention_2',local_files_only=True).eval()
  capture=Capture(model)
  counts=collections.Counter();did_probe=False
  with concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=prep_init) as pool:
   pending=None
   for i,sample in enumerate(samples):
    if pending is None:pending=pool.submit(prepare,sample)
    update(stage='preprocessing',sample_key=sample['sample_key']);sample,pair,info=pending.result()
    from transformers.feature_extraction_utils import BatchFeature
    pair=[BatchFeature({k:torch.from_numpy(v) for k,v in item.items()}) for item in pair]
    pending=pool.submit(prepare,samples[i+1]) if i+1<len(samples) else None
    for order,inputs in zip(ORDERS,pair):
     r=run_record(model,proc,capture,sample,inputs,info,order,args,update,probe=args.probe and not did_probe)
     did_probe=True;state['completed']+=1;counts[sample['dataset']+'/'+order]+=1;state['counts']=dict(counts);pulse()
    del pair
  update(stage='completed');pulse()
 except Exception:
  update(stage='failed',error=traceback.format_exc());pulse();raise
 finally:stop.set()


def report():
 import numpy as np
 records=[]
 for p in result_glob((relocate_path(OUT/'samples')), '*/*/*/complete.json'):
  r=read(relocate_path(p.parent/'result.json'));v=read(relocate_path(p.parent/'validation.json'));assert v['passed'] and v['saved_file']['passed'];records.append(r)
 groups={}
 for order in ORDERS:
  for dataset in sorted(set(s['dataset'] for s in read(MANIFEST))):
   rows=[r for r in records if r['order']==order and r['dataset']==dataset]
   if not rows:continue
   lengths=[r['reasoning_tokens'] for r in rows]
   groups[dataset+'/'+order]=dict(completed=len(rows),mean_reasoning_tokens=float(np.mean(lengths)),median_reasoning_tokens=float(np.median(lengths)),p95_reasoning_tokens=float(np.percentile(lengths,95)),max_reasoning_tokens=max(lengths),mean_iou=float(np.mean([r['iou'] for r in rows])),truncated=sum(not r['eos_reached'] for r in rows),unclosed_thinking=sum(not r['thinking_closed'] for r in rows),attention_gib=sum(r['attention_bytes'] for r in rows)/2**30)
 write(relocate_path(OUT/'summary.json'),dict(completed=len(records),total=600,groups=groups))
 lines=['# Qwen3-VL-8B-Thinking pilot input-order attention','',f'검증 완료: {len(records)}/600 실행. 미완료 결과를 전체 결과로 해석하지 않는다.','', '300개 고정 pilot × video_query/query_video. seed=42, temperature=1, top_p=.95, top_k=20, 최대 16,384 생성 토큰. FPS=2, min_tokens=64, total_tokens=14336.','', '상세: L0/7/14/21/28/35, 모든 key, FP32. Head 압축 방식은 config.json 참조. 전체 36 layers × 32 heads에는 7개 범주의 attention mass를 저장한다. 각 생성 토큰을 예측하는 직전 위치의 attention이며 종료 태그와 EOS 예측 행도 포함한다.','', '전환 전 기록은 replay, 전환 후 신규 생성은 live 추출이다. 저장된 generation만 있는 미완료 항목은 replay로 복구한다. 각 신규 결과의 measurement_method와 migration_to_live 목록으로 구분한다. Attention은 post-RoPE Q/K의 FP32 softmax다. 실제 생성 실행의 attention과 비트 단위 동일함을 주장하지 않는다. 미래 key는 0이며 key_lengths로 유효 구간을 구분한다. 구간/head 평균은 key 축을 재정규화하지 않는다. 입력 순서에 따라 reasoning과 답변도 달라질 수 있어 attention 변화의 인과를 단정하지 않는다.','', '| Dataset/order | n | reasoning mean | median | p95 | max | mIoU | truncated | GiB |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for k,v in groups.items():lines.append(f"| {k} | {v['completed']} | {v['mean_reasoning_tokens']:.1f} | {v['median_reasoning_tokens']:.1f} | {v['p95_reasoning_tokens']:.1f} | {v['max_reasoning_tokens']} | {v['mean_iou']:.4f} | {v['truncated']} | {v['attention_gib']:.2f} |")
 lines+=['','category_mass: [layer, generated token, head, category]. layer_XX: [head channel, generated token, key token]. tokens.json/input.json에 위치·토큰·범주 매핑을 저장한다. mean이면 유일한 채널은 32개 head의 산술평균이다. 전체 레이어 요약은 head별로 보존한다.','', '표본 1개가 전체 길이 분포를 대표하지 않으므로 benchmark/order별 p95와 max, 종료 여부를 함께 확인한다.']
 write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
 return len(records),groups


def supervise(args):
 import fcntl
 OUT.mkdir(parents=True,exist_ok=True);(relocate_path(OUT/'logs')).mkdir(exist_ok=True)
 lock=(relocate_path(OUT/'supervisor.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.time();jobs=[];env=dict(os.environ);mps=False
 initial_completed=len(list(result_glob((relocate_path(OUT/'samples')), '*/*/*/complete.json')))
 (relocate_path(OUT/'source.py')).write_bytes(relocate_path(Path(__file__)).read_bytes())
 write(relocate_path(OUT/'samples.json'),read(MANIFEST))
 config=dict(measurement_method='live_with_saved_generation_replay_fallback',previous_config='migration_to_live/config_replay.json',model=str(MODEL),manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),samples=300,orders=ORDERS,total=600,detail_layers=LAYERS,head_mode=args.head_mode,detail_dtype='float32',summary_dtype='float32',summary_heads=32,summary_layers=36,categories=CATEGORIES,seed=42,max_new_tokens=16384,temperature=1.,top_p=.95,top_k=20,input_fps=2,min_tokens=64,total_tokens=14336,workers=args.shards,replay_chunk=args.chunk,source_sha256=sha(__file__),attention_alignment='predictive',backend='flash_attention_2')
 write(relocate_path(OUT/'config.json'),config)
 def progress(stage):
  n,groups=report();elapsed=time.time()-started
  workers=[read(p) for p in result_glob(OUT, 'worker_*.json')]
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
  r=dict(stage=stage,timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),completed=n,total=600,percentage=n/6,elapsed_seconds=elapsed,eta_seconds=elapsed*(600-n)/(n-initial_completed) if n>initial_completed else None,eta_scope='remaining 600 records including generation, attention and validation; throughput since live transition',counts={k:v['completed'] for k,v in groups.items()},model='Qwen3-VL-8B-Thinking',workers=workers,gpu_util_memory_power=gpu)
  write(relocate_path(OUT/'status.json'),r);print(json.dumps(r),flush=True)
  with (relocate_path(OUT/'logs/gpu.jsonl')).open('a') as f:f.write(json.dumps(dict(timestamp=r['timestamp'],gpu=gpu))+'\n')
 def launch(shard,shards,limit=0,probe=False):
  cmd=[sys.executable,__file__,'worker','--shard',str(shard),'--shards',str(shards),'--head-mode',args.head_mode,'--chunk',str(args.chunk)]
  if limit:cmd+=['--limit',str(limit)]
  if probe:cmd+=['--probe']
  log=(relocate_path(OUT/'logs'/f'worker_{shard}_{"smoke" if limit else "full"}.log')).open('a')
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);log.close();return p
 try:
  # Gate full launch on an actual paired sample, numerical replay and saved-file validation.
  progress('smoke_validation')
  log=(relocate_path(OUT/'logs/live_validation.log')).open('a')
  smoke=subprocess.Popen([sys.executable,__file__,'validate_live'],stdout=log,stderr=subprocess.STDOUT);log.close();jobs=[smoke]
  while smoke.poll() is None:progress('smoke_validation');time.sleep(15)
  if smoke.returncode:raise RuntimeError('Smoke validation failed; full experiment was not launched.')
  env.update(CUDA_MPS_PIPE_DIRECTORY=f'/tmp/thinking_pilot_mps_{os.getpid()}',CUDA_MPS_LOG_DIRECTORY=str(relocate_path(OUT/'logs/mps')))
  relocate_path(Path(env['CUDA_MPS_PIPE_DIRECTORY'])).mkdir();relocate_path(Path(env['CUDA_MPS_LOG_DIRECTORY'])).mkdir(exist_ok=True)
  subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True);mps=True
  jobs=[launch(s,args.shards) for s in range(args.shards)]
  write(relocate_path(OUT/'execution.json'),dict(supervisor_pid=os.getpid(),started_epoch=started,worker_pids=[p.pid for p in jobs],mps_pipe=env['CUDA_MPS_PIPE_DIRECTORY']))
  while any(p.poll() is None for p in jobs):progress('running');time.sleep(20)
  failed=[s for s,p in enumerate(jobs) if p.returncode]
  for s in failed:
   p=launch(s,args.shards);jobs.append(p)
   while p.poll() is None:progress('retrying_failed_shard');time.sleep(20)
   if p.returncode:raise RuntimeError(f'Shard {s} failed twice; see logs')
  n,_=report();assert n==600,n
  progress('completed')
 except Exception:
  write(relocate_path(OUT/'failure.json'),dict(error=traceback.format_exc()));progress('failed');raise
 finally:
  if mps and all(p.poll() is not None for p in jobs):subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,check=False)

def validate_live(args):
 global OUT
 import torch,decord,h5py,numpy as np
 from functools import partial
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from timelens_inputs import prepare_pair
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 production=OUT;OUT=relocate_path(production/'migration_to_live/smoke')
 sample=read(MANIFEST)[0]
 proc=AutoProcessor.from_pretrained(MODEL,padding_side='left',do_resize=False,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='flash_attention_2',local_files_only=True).eval()
 capture=Capture(model);decord.VideoReader=partial(decord.VideoReader,num_threads=1)
 pair,info=prepare_pair(sample,proc)
 result=run_record(model,proc,capture,sample,pair[0],info,'video_query',args,lambda **kw:None)
 dest=folder(sample,'video_query');old=relocate_path(production/'samples'/sample['dataset']/sample['sample_key']/'video_query')
 assert read(relocate_path(dest/'generation.json'))['generated_ids']==read(relocate_path(old/'generation.json'))['generated_ids']
 errors={}
 with h5py.File(relocate_path(dest/'attention.h5')) as a,h5py.File(relocate_path(old/'attention.h5')) as b:
  for name in ['category_mass']+[f'layer_{l:02d}' for l in LAYERS]:
   x=a[name][:];y=b[name][:];errors[name]=float(np.abs(x-y).max());assert np.allclose(x,y,atol=2e-6,rtol=1e-4)
 assert result['eos_reached'] and result['generated_tokens']<16384
 write(relocate_path(production/'migration_to_live/validation.json'),dict(passed=True,live_full_generation=True,early_eos_resize_verified=True,generated_ids_identical=True,max_abs_errors=errors,generated_tokens=result['generated_tokens']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','supervise','report','validate_live']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);p.add_argument('--limit',type=int,default=0);p.add_argument('--probe',action='store_true');p.add_argument('--chunk',type=int,default=1);p.add_argument('--head-mode',choices=['mean','groups8'],default='mean');a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 if a.action=='worker':worker(a)
 elif a.action=='supervise':supervise(a)
 elif a.action=='validate_live':validate_live(a)
 else:report()
