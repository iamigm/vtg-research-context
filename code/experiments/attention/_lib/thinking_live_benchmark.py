"""Bounded online-vs-replay benchmark; leaves production jobs untouched."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import os,sys,json,time,threading,traceback,importlib.util
ROOT=relocate_path(Path(__file__)).resolve().parents[1]
OUT=relocate_path(ROOT/'results/thinking_live_benchmark_v1');OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('pilot_reference',relocate_path(ROOT/'results/thinking_pilot_orders_v1/source.py'))
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
pilot.ROOT=ROOT
pilot.MANIFEST=relocate_path(ROOT.parent/'zero-shot/predictions/cohorts/pilot_300/samples.json')
sys.path[:0]=[str(relocate_path(ROOT/'runtime/timelens_deps')),str(relocate_path(ROOT/'vendor/TimeLens'))]
START=time.time();STOP=threading.Event();STATE={'stage':'starting','completed':0,'total':8,'model':'Qwen3-VL-8B-Thinking','benchmark':'charades-timelens','eta_scope':'8 benchmark trials including capture, writing and validation'}
def pulse():
 elapsed=time.time()-START
 state={**STATE,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'elapsed_seconds':elapsed,'percentage':relocate_path(STATE['completed']/8)*100,'eta_seconds':relocate_path(elapsed*(8-STATE['completed'])/STATE['completed']) if STATE['completed'] else None}
 pilot.write(relocate_path(OUT/'status.json'),state);print(json.dumps(state),flush=True)
def heartbeat():
 while not STOP.wait(20):pulse()
def update(**kw):STATE.update(kw)

def finish_live(capture,h5,n,generated,cats):
 import numpy as np
 g=len(generated)
 for layer,pending in capture.pending.items():
  if not pending:continue
  lo=pending[0][0];hi=pending[-1][1]
  h5['category_mass'][layer,lo:hi]=np.concatenate([x[2] for x in pending],0)
  if layer in pilot.LAYERS:
   block=np.zeros((1,hi-lo,n+hi-1),dtype=np.float32)
   for a,b,_,arr in pending:block[:,a-lo:b-lo,:arr.shape[-1]]=arr
   h5[f'layer_{layer:02d}'][:,lo:hi,:n+hi-1]=block
 assert np.all(capture.coverage[:,:g]==1)
 h5['category_mass'].resize((36,g,32,7))
 for layer in pilot.LAYERS:h5[f'layer_{layer:02d}'].resize((1,g,n+g-1))
 for name,arr in [('generated_ids',generated),('key_lengths',n+np.arange(g)),('query_positions',n-1+np.arange(g)),('key_category_ids',cats[:n+g-1])]:
  del h5[name];h5.create_dataset(name,data=arr)
 h5.attrs['measurement']='FP32 attention reconstruction from live generation post-RoPE Q/K'

def live(model,proc,capture,inputs,sample,cap,path):
 import torch,numpy as np
 from transformers import StoppingCriteria,StoppingCriteriaList
 n=inputs.input_ids.shape[1]
 initial,_,_,_,_=pilot.labels(proc,inputs,sample,[])
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
   capture.start=g;capture.end=g+1;STATE['generated_tokens']=g
   return False
 with pilot.open_capture(path,n,cap,'mean',cats,[0]*cap) as f:
  capture.configure(n,cap,cats,f,'mean')
  torch.manual_seed(42)
  with torch.inference_mode():
   out=model.generate(**inputs,max_new_tokens=cap,do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,use_cache=True,stopping_criteria=StoppingCriteriaList([Advance()]))
  generated=out[0,n:].tolist();capture.enabled=False
  actual,_,_,_,_=pilot.labels(proc,inputs,sample,generated)
  assert np.array_equal(actual,cats[:len(actual)])
  finish_live(capture,f,n,generated,cats);f.flush()
 return generated

def compare(a,b):
 import h5py,numpy as np
 errors={}
 with h5py.File(a) as x,h5py.File(b) as y:
  assert np.array_equal(x['generated_ids'][:],y['generated_ids'][:])
  for name in ['category_mass']+[f'layer_{l:02d}' for l in pilot.LAYERS]:
   largest=0.
   for s in range(0,x[name].shape[1],32):
    u=x[name][:,s:s+32];v=y[name][:,s:s+32]
    largest=max(largest,float(abs(u-v).max()))
    assert np.allclose(u,v,atol=2e-6,rtol=1e-4),(name,largest)
   errors[name]=largest
 return errors

def report(records,done=False):
 import statistics
 groups={}
 for order in pilot.ORDERS:
  rows=[r for r in records if r['order']==order]
  a=[r['seconds'] for r in rows if r['mode']=='replay'];b=[r['seconds'] for r in rows if r['mode']=='live']
  if a and b:groups[order]={'replay_mean_seconds':statistics.mean(a),'live_mean_seconds':statistics.mean(b),'speedup':statistics.mean(a)/statistics.mean(b),'replay_trials':len(a),'live_trials':len(b)}
 pilot.write(relocate_path(OUT/'results.json'),dict(complete=done,trials=records,groups=groups,conditions='Shares GPU via MPS with four production workers; timings are preliminary and include per-trial validation. Input preparation and model load excluded. ABBA per input order.'))
 lines=['# 생성 중 attention 추출 속도 비교','',f'완료된 trial: {len(records)}/8','', '동일한 모델·입력·seed·생성 한도·FP32 저장 설정. 원본 4개 작업과 GPU를 공유하므로 독점 GPU 속도나 전체 실험 ETA를 확정하지 않는다. replay/live/live/replay 순서로 시간 변화의 영향을 줄인다. 전처리와 모델 로드 시간은 제외하고 생성·attention 계산·파일 저장·검증을 포함한다.','', '| 입력 순서 | replay 평균(초) | live 평균(초) | 배속 |','|---|---:|---:|---:|']
 for k,v in groups.items():lines.append(f"| {k} | {v['replay_mean_seconds']:.2f} | {v['live_mean_seconds']:.2f} | {v['speedup']:.3f}× |")
 lines+=['','video_query는 최대 128토큰, query_video는 최대 512토큰으로 제한한 부분 trace 벤치마크다. 전체 답변 길이 실험과 구분한다. 모든 방식은 실제 sampling을 수행하며 원본 token IDs 동일성과 상세/요약 attention 오차를 확인한다.']
 write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')

def main():
 import torch,decord
 from functools import partial
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from timelens_inputs import prepare_pair
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 STATE['stage']='loading';pulse()
 proc=AutoProcessor.from_pretrained(pilot.MODEL,padding_side='left',do_resize=False,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(pilot.MODEL,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='flash_attention_2',local_files_only=True).eval()
 capture=pilot.Capture(model)
 decord.VideoReader=partial(decord.VideoReader,num_threads=1)
 sample=pilot.read(pilot.MANIFEST)[0];pair,info=prepare_pair(sample,proc)
 pilot.write(relocate_path(OUT/'config.json'),dict(model=str(pilot.MODEL),sample=sample,cases={'video_query':128,'query_video':512},trial_order=['replay','live','live','replay'],seed=42,head_mode='mean',detail_layers=pilot.LAYERS,summary_layers=36,summary_heads=32,measurement_dtype='float32',validation_tolerance=dict(atol=2e-6,rtol=1e-4),production_code_sha256=pilot.sha(relocate_path(ROOT/'results/thinking_pilot_orders_v1/source.py'))))
 records=[]
 # Warm CUDA kernels with 4 generated tokens before collecting timings.
 with torch.inference_mode():
  warm=pair[0].to('cuda');model.generate(**warm,max_new_tokens=4,do_sample=False);del warm
 for case,(order,cpu) in enumerate(zip(pilot.ORDERS,pair)):
  cap=[128,512][case];inputs=cpu.to('cuda');n=inputs.input_ids.shape[1];baseline=None;ids_reference=None
  for rep,mode in enumerate(['replay','live','live','replay']):
   path=relocate_path(OUT/f'{order}_{rep}_{mode}.h5');STATE.update(stage=mode,order=order,trial=rep,generated_tokens=0);pulse()
   capture.enabled=False;torch.cuda.synchronize();started=time.perf_counter()
   if mode=='replay':
    torch.manual_seed(42)
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=cap,do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,use_cache=True)
    generated=out[0,n:].tolist();del out
    cats,*_=pilot.labels(proc,inputs,sample,generated)
    pilot.replay(model,capture,inputs,generated,cats,path,'mean',1,lambda **kw:STATE.update(kw))
   else:generated=live(model,proc,capture,inputs,sample,cap,path)
   pilot.validate_file(path,'mean');torch.cuda.synchronize();seconds=time.perf_counter()-started
   if baseline is None:baseline=path;ids_reference=generated
   assert generated==ids_reference
   errors=compare(baseline,path)
   record=dict(order=order,mode=mode,trial=rep,cap=cap,generated_tokens=len(generated),seconds=seconds,bytes=path.stat().st_size,token_ids_identical=True,attention_max_abs_errors=errors)
   records.append(record);STATE['completed']=len(records);report(records);pulse()
  del inputs;torch.cuda.empty_cache()
 report(records,done=True);STATE['stage']='completed';pulse()

if __name__=='__main__':
 threading.Thread(target=heartbeat,daemon=True).start()
 try:main()
 except Exception:
  STATE.update(stage='failed',error=traceback.format_exc());pulse();raise
 finally:STOP.set()
