"""One shared video KV and bounded exact text-only prefix KV reuse."""
import collections,math,time
from probability_budget_core import CONFIG
class BudgetBoundary(Exception):pass
class Backend:
 def __init__(self,model,inputs,processor):
  import torch
  self.torch=torch;self.model=model;self.inputs=inputs;self.tokenizer=processor.tokenizer;self.n=int(inputs.input_ids.shape[1]);eos=model.generation_config.eos_token_id;self.eos=set(eos if isinstance(eos,list) else [eos]);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start=time.monotonic();a.record()
  with torch.inference_mode():out=model(**inputs,use_cache=True,logits_to_keep=1)
  b.record();b.synchronize();self.cache=out.past_key_values;self.root=out.logits[0,-1].float().cpu();self.prefill_seconds=time.monotonic()-start;self.prefill_cuda_seconds=a.elapsed_time(b)/1000
  assert all(type(layer).__name__=='DynamicLayer' for layer in self.cache.layers),[type(l).__name__ for l in self.cache.layers]
  self.base=[(layer.keys[...,:self.n,:],layer.values[...,:self.n,:]) for layer in self.cache.layers];self.max_context=int(model.config.text_config.max_position_embeddings);self.reset()
 def reset(self,limit=2048):
  self.cache.crop(self.n);self.path=();self.last=self.root;self.saved=collections.OrderedDict();self.saved_bytes=0;self.limit=limit;self.calls=0;self.tokens=0;self.replay_tokens=0;self.restore_hits=0;self.evictions=0;self.decode_cuda=0.;self.restore_cuda=0.;self.store_cuda=0.;self.cache_wall=0.
 def costs(self):return dict(native_forwards=self.calls,decoder_inputs=self.tokens,replay_inputs=self.replay_tokens,decode_cuda_seconds=self.decode_cuda,kv_restore_cuda_seconds=self.restore_cuda,kv_store_cuda_seconds=self.store_cuda,kv_cache_wall_seconds=self.cache_wall,kv_restore_hits=self.restore_hits,kv_evictions=self.evictions,kv_suffix_cache_bytes=self.saved_bytes)
 def _restore(self,path):
  torch=self.torch;start=time.monotonic();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();tails=self.saved[path];self.saved.move_to_end(path)
  for layer,(basek,basev),(k,v) in zip(self.cache.layers,self.base,tails):layer.keys=torch.cat((basek,k),dim=-2);layer.values=torch.cat((basev,v),dim=-2)
  b.record();b.synchronize();self.restore_cuda+=a.elapsed_time(b)/1000;self.cache_wall+=time.monotonic()-start;self.path=path;self.last=None;self.restore_hits+=1
 def _save(self,path):
  if not path or path in self.saved:return
  torch=self.torch;size=sum((l.keys[...,self.n:,:].numel()+l.values[...,self.n:,:].numel())*l.keys.element_size() for l in self.cache.layers)
  if size>CONFIG['kv_suffix_cache_bytes']:return
  while self.saved and self.saved_bytes+size>CONFIG['kv_suffix_cache_bytes']:
   _,old=self.saved.popitem(last=False);self.saved_bytes-=sum((k.numel()+v.numel())*k.element_size() for k,v in old);self.evictions+=1
  start=time.monotonic();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();tails=[(l.keys[...,self.n:,:].clone(),l.values[...,self.n:,:].clone()) for l in self.cache.layers];b.record();b.synchronize();self.store_cuda+=a.elapsed_time(b)/1000;self.cache_wall+=time.monotonic()-start;self.saved[path]=tails;self.saved_bytes+=size
 def get(self,path,store=True):
  path=tuple(path);torch=self.torch
  if not path:return self.root
  if path==self.path and self.last is not None:return self.last
  common=0
  for a,b in zip(self.path,path):
   if a!=b:break
   common+=1
  if common==len(path):common-=1
  restore=None
  for length in range(len(path)-1,common,-1):
   if path[:length] in self.saved:restore=path[:length];break
  if restore is not None:self._restore(restore);common=len(restore)
  else:self.cache.crop(self.n+common);self.path=self.path[:common]
  for j in range(common,len(path)):
   if self.tokens>=self.limit:raise BudgetBoundary()
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
   with torch.inference_mode():out=self.model(input_ids=torch.tensor([[path[j]]],device='cuda'),attention_mask=torch.ones((1,self.n+j+1),device='cuda',dtype=self.inputs.attention_mask.dtype),past_key_values=self.cache,cache_position=torch.tensor([self.n+j],device='cuda'),use_cache=True,logits_to_keep=1)
   b.record();b.synchronize();self.decode_cuda+=a.elapsed_time(b)/1000;self.cache=out.past_key_values;self.path=path[:j+1];self.last=out.logits[0,-1].float().cpu();self.calls+=1;self.tokens+=1;self.replay_tokens+=int(j<len(path)-1)
  if store:self._save(path)
  return self.last
