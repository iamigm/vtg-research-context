from result_layout import relocate_path, write_result_text
from pathlib import Path
import sys,time,json,importlib.util
ROOT=relocate_path(Path('/home/irteam/experiments/zero-shot'));OUT=relocate_path(ROOT/'results/thinking_vs_instruct_8fps_v1');PERF=relocate_path(OUT/'performance')
sys.path.insert(0,str(relocate_path(ROOT/'_lib')))
import pilot_8fps as run
import torch
from transformers import AutoProcessor,AutoModelForImageTextToText
from pilot_8fps_traces import TraceCapture as Fast
spec=importlib.util.spec_from_file_location('old_traces',relocate_path(PERF/'before_pilot_8fps_traces.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch.set_num_threads(2);mp='/home/irteam/backbones/models/Qwen3-VL-8B-Instruct';proc=AutoProcessor.from_pretrained(mp,padding_side='left',do_resize=False,local_files_only=True)
model=AutoModelForImageTextToText.from_pretrained(mp,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
s=run.read(run.MANIFEST)[0];_,pair,info=run.Prepared([s],mp)[0];inputs=pair[0].to('cuda');n=inputs.input_ids.shape[1];allowed=[t['id'] for t in run.read(relocate_path(ROOT/'results/timelens_span_head/qwen3/shard0/config.json'))['allowed_tokens']];eos=model.generation_config.eos_token_id
with torch.inference_mode():model.generate(**inputs,max_new_tokens=2,do_sample=False,temperature=None,top_p=None,top_k=None)
results=[]
for sampling in [False,True]:
 outputs=[];traces=[];times=[]
 for cls in [old.TraceCapture,Fast]:
  capture=cls(model,proc.tokenizer,allowed,eos);capture.model_label='instruct';capture.reset();torch.manual_seed(42)
  kw=dict(do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.) if sampling else dict(do_sample=False,temperature=None,top_p=None,top_k=None)
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=32,**kw,use_cache=True)
  generated=out[0,n:].tolist();steps=capture.finish(generated);torch.cuda.synchronize();times.append(time.perf_counter()-start);outputs.append(generated);traces.append(steps);capture.remove()
 assert outputs[0]==outputs[1], 'Instrumentation changed seeded generation'
 maxerr=0.
 for a,b in zip(*traces):
  for field in ['token_id','token','raw_selected_rank','processed_selected_rank']:assert a[field]==b[field],field
  for field in ['raw_selected_probability','processed_selected_probability']:maxerr=max(maxerr,abs(a[field]-b[field]))
  for mode in ['raw','processed']:
   for field in ['entropy_nats','top30_mass','span_alphabet_mass','eos_mass','probability_sum']:maxerr=max(maxerr,abs(a[mode][field]-b[mode][field]))
   assert a[mode]['top30']==b[mode]['top30']
 assert maxerr<1e-6
 result=dict(sampling=sampling,tokens=len(outputs[0]),before_seconds=times[0],after_seconds=times[1],speedup=times[0]/times[1],max_absolute_trace_error=maxerr,generated_ids_identical=True,top30_identical=True);results.append(result);print(json.dumps(result),flush=True)
run.write(relocate_path(PERF/'benchmark.json'),dict(passed=True,input_sha256=__import__('hashlib').sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest(),results=results,scope='single matched 8fps input; greedy and seeded sampling max32; timing includes trace capture, materialization and decoding; not an end-to-end cohort speedup'))
from pilot_8fps_traces import PairVisionCache
pair_outputs=[];pair_traces=[];pair_times=[]
for cached in [False,True]:
 vc=PairVisionCache(model) if cached else None
 capture=Fast(model,proc.tokenizer,allowed,eos);capture.model_label='instruct';outs=[];ts=[]
 torch.cuda.synchronize();started=time.perf_counter()
 for cpu in pair:
  inp=cpu.to('cuda');capture.reset();torch.manual_seed(42)
  with torch.inference_mode():output=model.generate(**inp,max_new_tokens=32,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True)
  ids=output[0,inp.input_ids.shape[1]:].tolist();ts.append(capture.finish(ids));outs.append(ids)
 torch.cuda.synchronize();pair_times.append(time.perf_counter()-started);pair_outputs.append(outs);pair_traces.append(ts);capture.remove()
 if vc:assert vc.calls==1 and vc.hits==1;vc.remove()
assert pair_outputs[0]==pair_outputs[1]
assert pair_traces[0]==pair_traces[1],'Vision cache changed trace'
p=relocate_path(PERF/'benchmark.json');result=run.read(p);result['vision_cache']=dict(before_seconds=pair_times[0],after_seconds=pair_times[1],speedup=pair_times[0]/pair_times[1],generated_ids_identical=True,all_trace_fields_identical=True,scope='one video/query input-order pair, exact visual feature reuse');run.write(p,result);print(json.dumps(result['vision_cache']),flush=True)
