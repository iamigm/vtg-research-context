"""Isolated clip verification and query-free embeddings; vectors stay in RAM."""
import time,sys,math,collections
from types import MethodType
import numpy as np
from probability_completion_core import *
from probability_budget_inputs import source_frames,processor_input,parse_clip

REVIEW='Target event: {query}\nInspect only this clip. Rate visual evidence and temporal coverage: 0 = absent or a different event; 1 = partly shown, incomplete action, or substantial unrelated content; 2 = the complete requested action with well placed boundaries and little unrelated content. Reply with one digit only: 0, 1, or 2.'

class Crops:
 def __init__(self,row,spans):
  import torch
  from result_layout import relocate_path
  from torchvision.transforms import functional as TF,InterpolationMode
  self.row=row;self.spans=spans;self.prior,self.cache,self.frames=source_frames(row);plan=self.cache['plans']['2'];ids=plan['frame_ids'];pts=np.array(plan['source_pts']);self.specs=[];self.assignment=[];self.extra={};keys={};fallback={}
  parent=read(relocate_path(Path(self.cache['parent_metadata'])));allpts=np.asarray(parent['pts'])
  for pair in spans:
   slots=np.flatnonzero((pts>=pair[0])&(pts<pair[1])).tolist()
   if slots:chosen=[ids[i] for i in slots];times=[float(pts[i]) for i in slots];fallback_id=None
   else:
    midpoint=sum(pair)/2;ident=max(0,min(len(allpts)-1,int(np.searchsorted(allpts,midpoint,side='right')-1)));chosen=[ident];times=[float(allpts[ident])];fallback_id=ident;fallback[ident]=None
   if len(chosen)==1:chosen*=2;times*=2
   times=[t-times[0] for t in times];key=(tuple(chosen),tuple(times));
   if key not in keys:
    keys[key]=len(self.specs);self.specs.append(dict(id=len(self.specs),frame_ids=chosen,timestamps=times,slots=slots,fallback_frame=fallback_id,frame_count=len(chosen),hw=self.cache['hw']))
   self.assignment.append(keys[key])
  if fallback:
   import decord
   vr=decord.VideoReader(row['video_path'],num_threads=1);ordered=sorted(fallback);rgb=vr.get_batch(ordered).asnumpy();pixels=TF.resize(torch.from_numpy(rgb).permute(0,3,1,2),self.cache['hw'],interpolation=InterpolationMode.BICUBIC,antialias=True).numpy()
   for ident,pixel,raw in zip(ordered,pixels,rgb):
    if parent.get('rgb_hashes'):assert hashlib.sha256(raw.tobytes()).hexdigest()==parent['rgb_hashes'][ident]
    self.extra[ident]=pixel
  self.fallback_frame_count=len(fallback)
 def pixels(self,spec,kind):
  import torch
  if spec['fallback_frame'] is not None:array=np.stack([self.extra[spec['fallback_frame']]]*2)
  else:
   array=self.frames[spec['slots']]
   if len(array)==1:array=np.concatenate([array,array],axis=0)
  if kind=='verifier':
   from torchvision.transforms import functional as TF,InterpolationMode
   hw=[max(28,round(d/28)*28) for d in array.shape[-2:]]
   if list(array.shape[-2:])!=hw:array=TF.resize(torch.from_numpy(array),hw,interpolation=InterpolationMode.BICUBIC,antialias=True).numpy()
  return torch.from_numpy(np.asarray(array)).float()
 def cpu(self,processor,spec,kind):
  frames=self.pixels(spec,kind);instruction="Represent the user's input."
  if kind=='verifier':messages=[dict(role='user',content=[dict(type='video',video='frames'),dict(type='text',text=REVIEW.format(query=self.row['query']))])]
  else:messages=[dict(role='system',content=instruction),dict(role='user',content=[dict(type='video',video='frames')])]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  if kind=='verifier':
   times=spec['timestamps'];rate=(len(times)-1)/(times[-1]-times[0]) if times[-1]>times[0] else 2.
   cpu=processor(text=[text],videos=[frames],fps=rate,do_resize=False,do_sample_frames=False,padding=True,return_tensors='pt')
  else:
   mapping=dict(enumerate(spec['timestamps']))
   def timestamps(self,indices,video_fps,merge_size=2):
    values=list(indices);values+=values[-1:]*((-len(values))%merge_size);return [(mapping[values[j]]+mapping[values[j+merge_size-1]])/2 for j in range(0,len(values),merge_size)]
   original=processor._calculate_timestamps;processor._calculate_timestamps=MethodType(timestamps,processor)
   try:cpu=processor(text=[text],videos=[frames],video_metadata=[dict(fps=2.,frames_indices=list(range(len(frames))),total_num_frames=len(frames),video_backend='decord')],do_resize=False,do_sample_frames=False,padding=True,return_tensors='pt')
   finally:processor._calculate_timestamps=original
  return cpu


def collate(items,pad):
 import torch
 size=max(int(x['input_ids'].shape[1]) for x in items);ids=torch.full((len(items),size),pad,dtype=items[0]['input_ids'].dtype);mask=torch.zeros_like(ids)
 for i,item in enumerate(items):n=item['input_ids'].shape[1];ids[i,-n:]=item['input_ids'][0];mask[i,-n:]=1
 out=dict(input_ids=ids,attention_mask=mask)
 for key in items[0]:
  if key in out:continue
  values=[x[key] for x in items]
  if isinstance(values[0],torch.Tensor):out[key]=torch.cat(values,dim=0)
  elif isinstance(values[0],list):out[key]=sum(values,[])
  else:raise TypeError((key,type(values[0])))
 return out


class Models:
 def __init__(self,progress):
  import torch
  from transformers import AutoProcessor,AutoModelForImageTextToText
  torch.set_num_threads(2);self.torch=torch;self.models={};self.processors={};self.validated=set();self.serial_kinds=set();self.batch_size=SETTINGS['batch_size'];loads={}
  for key,name in [('verifier',SETTINGS['verifier']),('instruct',SETTINGS['generator']),('embedding',SETTINGS['embedding'])]:
   progress(stage='loading_model',model=key);started=time.monotonic();path=BACKBONES/name;self.processors[key]=AutoProcessor.from_pretrained(path,local_files_only=True,do_resize=False,padding_side='left')
   if key=='embedding':
    sys.path.insert(0,str(EMBED_SOURCE.parents[2]));from src.models.qwen3_vl_embedding import Qwen3VLForEmbedding
    model=Qwen3VLForEmbedding.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'})
   else:model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'})
   self.models[key]=model.eval();loads[key]=time.monotonic()-started
  self.loads=loads
 def forward(self,kind,items):
  torch=self.torch;model=self.models[kind];processor=self.processors[kind];inp=collate(items,processor.tokenizer.pad_token_id);lengths=inp['attention_mask'].sum(-1).tolist();visual=int((inp['input_ids']==model.config.video_token_id).sum());started=time.monotonic();inp={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in inp.items()};a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
  with torch.inference_mode():
   if kind=='verifier':
    outputs=model(**inp,use_cache=False,logits_to_keep=1);logits=outputs.logits[:,-1].float();choices=[processor.tokenizer.encode(str(i),add_special_tokens=False) for i in range(3)];assert all(len(c)==1 for c in choices);lp=torch.log_softmax(logits,-1);selected=lp[:,[c[0] for c in choices]];probs=selected.softmax(-1);mass=selected.exp().sum(-1);values=[dict(probabilities=p.tolist(),score=float((p[1]+2*p[2])/2),option_mass=float(m),argmax=processor.tokenizer.decode([int(j)])) for p,m,j in zip(probs.cpu(),mass.cpu(),logits.argmax(-1).cpu())]
   else:
    outputs=model.model(**inp,use_cache=False,return_dict=True) if kind=='instruct' else model(**inp,use_cache=False);hidden=outputs.last_hidden_state.float();last=hidden[:,-1];last=torch.nn.functional.normalize(last,dim=-1)
    if kind=='instruct':
     mask=(inp['input_ids']==model.config.video_token_id)&inp['attention_mask'].bool();assert mask.any(-1).all();pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True);pooled=torch.nn.functional.normalize(pooled,dim=-1);values=[dict(instruct_visual_mean=x.cpu().numpy(),instruct_last=y.cpu().numpy()) for x,y in zip(pooled,last)]
    else:values=[dict(embedding_last=x.cpu().numpy()) for x in last]
  b.record();b.synchronize();cost=dict(forward_calls=1,input_tokens=int(sum(lengths)),padded_tokens=int(inp['input_ids'].numel()),visual_tokens=visual,cuda_seconds=a.elapsed_time(b)/1000,wall_seconds=time.monotonic()-started)
  for value,length in zip(values,lengths):value['cost']=dict(input_tokens=int(length),amortized_cuda_seconds=cost['cuda_seconds']*length/sum(lengths),amortized_wall_seconds=cost['wall_seconds']*length/sum(lengths))
  del outputs,inp;return values,cost
 def run_crops(self,crops,kind,progress):
  processor=self.processors[kind];specs=sorted(crops.specs,key=lambda s:(s['frame_count'],s['id']));out={};batches=[];validation_cost=[];prepared=[];tokens=0;prep=0.
  def flush():
   nonlocal prepared,tokens
   if not prepared:return
   batch=[x[1] for x in prepared];values,cost=self.forward(kind,batch);batches.append(cost)
   if kind not in self.validated and len(batch)>1:
    # Validate on the actual first production crop, without a separate smoke run.
    first,extra=self.forward(kind,[batch[0]]);validation_cost.append(extra)
    if kind=='verifier':
     error=max(abs(a-b) for a,b in zip(first[0]['probabilities'],values[0]['probabilities']));equivalent=error<.01
    else:
     error=max(1-float(first[0][key]@values[0][key]) for key in first[0] if key!='cost');equivalent=error<.001
    if not equivalent:
     self.serial_kinds.add(kind);progress(stage='batch_equivalence_fallback',model=kind,error=error,execution='serial native forward')
     values=[first[0]]
     for item in batch[1:]:
      one,extra=self.forward(kind,[item]);batches.append(extra);values.extend(one)
    self.validated.add(kind)
   for (spec,_),value in zip(prepared,values):out[spec['id']]=value
   progress(stage=kind,clips_done=len(out),clips_total=len(specs));prepared=[];tokens=0
  for spec in specs:
   start=time.monotonic();cpu=crops.cpu(processor,spec,kind);prep+=time.monotonic()-start;length=int(cpu.input_ids.shape[1])
   if prepared and (len(prepared)>=(1 if kind in self.serial_kinds else self.batch_size) or (max(tokens,length)*(len(prepared)+1)>SETTINGS['batch_tokens'])):flush()
   prepared.append((spec,cpu));tokens=max(tokens,length)
  flush();return out,dict(batches=batches,validation=validation_cost,cpu_prepare_seconds=prep)


class FinalSession:
 def __init__(self,model,processor,cpu,greedy,duration):
  import torch
  self.torch=torch;self.model=model;self.tokenizer=processor.tokenizer;self.n=int(cpu.input_ids.shape[1]);self.dtype=cpu.attention_mask.dtype;self.greedy=greedy;self.duration=duration;self.max_context=int(model.config.text_config.max_position_embeddings);self.base_ids=greedy['ids'];self.eos=model.generation_config.eos_token_id;self.eos=set(self.eos if isinstance(self.eos,list) else [self.eos]);gpu=cpu.to('cuda');start=time.monotonic();torch.cuda.synchronize()
  with torch.inference_mode():out=model(**gpu,use_cache=True,logits_to_keep=1)
  torch.cuda.synchronize();self.prefill_wall=time.monotonic()-start;self.cache=out.past_key_values;self.path=();self.last=out.logits[0,-1].float();self.inputs=0;self.calls=0;self.cuda=0.;self.generated=0
 def get(self,path):
  torch=self.torch;path=tuple(path)
  if path==self.path:return self.last
  common=0
  for a,b in zip(path,self.path):
   if a!=b:break
   common+=1
  if common==len(path):common=max(0,common-1)
  self.cache.crop(self.n+common);self.path=self.path[:common]
  for start in range(common,len(path),256):
   chunk=path[start:start+256];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
   with torch.inference_mode():out=self.model(input_ids=torch.tensor([chunk],device='cuda'),attention_mask=torch.ones((1,self.n+start+len(chunk)),device='cuda',dtype=self.dtype),past_key_values=self.cache,cache_position=torch.arange(self.n+start,self.n+start+len(chunk),device='cuda'),use_cache=True,logits_to_keep=1)
   b.record();b.synchronize();self.cuda+=a.elapsed_time(b)/1000;self.inputs+=len(chunk);self.calls+=1;self.cache=out.past_key_values;self.path=path[:start+len(chunk)];self.last=out.logits[0,-1].float()
  return self.last
 def generate(self,prompt):
  turns=self.tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True);suffix=self.tokenizer.encode('\n'+turns,add_special_tokens=False);prefix=self.base_ids+suffix;before=(self.inputs,self.calls,self.cuda);start=time.monotonic();ids=[]
  if self.n+len(prefix)+64>self.max_context:return dict(answer='',ids=[],span=None,valid=False,reason='context_limit',cost=dict(input_tokens=0,forward_calls=0,cuda_seconds=0,wall_seconds=0))
  for _ in range(64):
   token=int(self.get(prefix+ids).argmax());ids.append(token)
   if token in self.eos:break
  text=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);parsed=parse_clip(text,self.duration,bool(ids and ids[-1] in self.eos));return dict(answer=text,ids=ids,span=parsed['span'],valid=parsed['valid'],reason=parsed['reason'],cost=dict(input_tokens=self.inputs-before[0],forward_calls=self.calls-before[1],cuda_seconds=self.cuda-before[2],wall_seconds=time.monotonic()-start),prompt_sha256=digest(prompt))
