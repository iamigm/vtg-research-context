"""Native left-padded microbatch generation, gated by exact token bridge."""
import time,copy
import torch
from adaptive_temporal_sampling import stamp,objhash

def generate(items,model,processor,genkw):
    width=max(int(cpu.input_ids.shape[1]) for cpu,_,_ in items);batch={}
    for key in items[0][0]:
        tensors=[cpu[key] for cpu,_,_ in items]
        if key in ['input_ids','attention_mask']:
            pad=processor.tokenizer.pad_token_id if key=='input_ids' else 0
            batch[key]=torch.cat([torch.nn.functional.pad(t,(width-t.shape[1],0),value=pad) for t in tensors],dim=0)
        else:batch[key]=torch.cat(tensors,dim=0)
    events=[]
    def before(module,args,kwargs):
        e=torch.cuda.Event(enable_timing=True);e.record();events.append([e,None])
    def after(module,args,kwargs,output):
        e=torch.cuda.Event(enable_timing=True);e.record();events[-1][1]=e
    a=model.register_forward_pre_hook(before,with_kwargs=True);b=model.register_forward_hook(after,with_kwargs=True)
    torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    try:
        with torch.inference_mode():out=model.generate(**{k:v.to('cuda') for k,v in batch.items()},**genkw)
        torch.cuda.synchronize()
    finally:a.remove();b.remove()
    elapsed=time.perf_counter()-start;times=[x.elapsed_time(y)/1000 for x,y in events];n=len(items);eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];results=[]
    group=objhash([identity for _,_,identity in items])
    for row,(_,detail,identity) in zip(out,items):
        ids=row[width:].tolist()
        for i,t in enumerate(ids):
            if t in eos:ids=ids[:i+1];break
        results.append(dict(identity=identity,model_revision=model.config._commit_hash or 'local',answer=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),generated_ids=ids,generated_tokens=len(ids),eos_reached=bool(ids and ids[-1] in eos),generation_seconds=elapsed/n,prefill_seconds=times[0]/n,decode_seconds=sum(times[1:])/n,input_prepare_seconds=0.,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),F=detail['F'],B=detail['B'],grid=detail['grid'],total_input_tokens=detail['total_input_tokens'],completed_at=stamp(),input_detail_hash=objhash(detail),reused_identical_input=False,batch_size=n,batch_group=group,batch_wall_seconds=elapsed,cost_allocation='batch elapsed divided equally among rows'))
    return results
