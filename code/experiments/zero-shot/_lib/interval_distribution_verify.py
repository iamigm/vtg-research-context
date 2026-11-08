"""Independent checkpoint re-watches every unique candidate clip; no search feedback."""
import json,math,time,gc
import numpy as np
from interval_distribution_core import *

def assess(model,processor,frames,pts,query,relative_span,progress,validate=False):
 import torch
 from interval_distribution_runtime import processor_input
 torch.cuda.reset_peak_memory_stats()
 prompts=[];prepare_start=time.monotonic();first_cpu=None;first_prompt=None;expanded=None
 questions=[('event','Does the candidate show the requested action, actor and object?'),('onset','Does the candidate start at the beginning of the requested action, without missing its beginning or adding unrelated lead-in?'),('offset','Does the candidate end at the end of the requested action, without cutting it short or adding unrelated trailing action?')]
 for name,question in questions:
  prompt=f'Target event: {query}\nThe video includes context. Candidate interval within this clip: {relative_span} seconds.\n{question}\nUse only visible evidence. Answer one letter: A = supported; B = contradicted; C = uncertain or not observable.'
  if first_cpu is None:
   cpu=processor_input(processor,frames,list(range(len(frames))),pts,2.,len(frames),direct_text=prompt);first_cpu=cpu;first_prompt=prompt;expanded=processor.tokenizer.decode(cpu.input_ids[0],skip_special_tokens=False,clean_up_tokenization_spaces=False)
  elif expanded.count(first_prompt)==1:
   cpu=type(first_cpu)(data=dict(first_cpu));ids=processor.tokenizer(expanded.replace(first_prompt,prompt),return_tensors='pt').input_ids;cpu['input_ids']=ids;cpu['attention_mask']=torch.ones_like(ids)
  else:cpu=processor_input(processor,frames,list(range(len(frames))),pts,2.,len(frames),direct_text=prompt)
  prompts.append((name,cpu))
 prepare_seconds=time.monotonic()-prepare_start
 arrays=[cpu.input_ids[0].tolist() for _,cpu in prompts];common=0
 for tokens in zip(*arrays):
  if len(set(tokens))!=1:break
  common+=1
 pre=type(prompts[0][1])(data=dict(prompts[0][1])).to('cuda');pre['input_ids']=pre.input_ids[:,:common];pre['attention_mask']=pre.attention_mask[:,:common];started=time.monotonic();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
 with torch.inference_mode():out=model(**pre,use_cache=True,logits_to_keep=1)
 b.record();b.synchronize();cache=out.past_key_values;prefill=dict(input_prepare_wall_seconds=prepare_seconds,forwards=1,decoder_inputs=common,cuda_stream_seconds=a.elapsed_time(b)/1000,wall_seconds=time.monotonic()-started)
 option_ids=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'ABC'];assert all(len(x)==1 for x in option_ids);option_ids=[x[0] for x in option_ids];results={}
 for name,cpu in prompts:
  cache.crop(common);suffix=cpu.input_ids[:,common:].to('cuda');length=len(arrays[0]) if name=='event' else int(cpu.input_ids.shape[1]);start=time.monotonic();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
  with torch.inference_mode():out=model(input_ids=suffix,attention_mask=torch.ones((1,length),device='cuda',dtype=pre.attention_mask.dtype),past_key_values=cache,cache_position=torch.arange(common,length,device='cuda'),use_cache=True,logits_to_keep=1)
  b.record();b.synchronize();cache=out.past_key_values;raw=torch.softmax(out.logits[0,-1].float(),dim=-1)[option_ids].cpu().tolist();mass=sum(raw);prob=[x/mass for x in raw] if mass else [0.,0.,1.];score=prob[0]+.5*prob[2] if mass>=.01 else .5
  results[name]=dict(raw_probabilities=raw,option_mass=mass,probabilities=prob,score=score,cost=dict(forwards=1,decoder_inputs=int(suffix.shape[1]),cuda_stream_seconds=a.elapsed_time(b)/1000,wall_seconds=time.monotonic()-start))
 if validate:
  full=type(prompts[0][1])(data=dict(prompts[0][1])).to('cuda')
  with torch.inference_mode():reference=model(**full,use_cache=False,logits_to_keep=1)
  raw=torch.softmax(reference.logits[0,-1].float(),dim=-1)[option_ids].cpu().tolist();delta=max(abs(a-b) for a,b in zip(raw,results['event']['raw_probabilities']));assert delta<.005,delta
  results['event']['full_prefill_max_probability_difference']=delta
 peak=torch.cuda.max_memory_allocated();del cache,out,pre;return dict(assessments=results,prefill_cost=prefill,peak_allocated_bytes=peak,event_score=results['event']['score'],joint_score=math.prod(results[k]['score'] for k in results)**(1/3))

def verify_video(rows,model,processor,progress):
 from interval_distribution_runtime import source_frames
 _,cache,frames=source_frames(rows[0]);plan=cache['plans']['2'];unique=[];seen=set()
 for i,(ident,pts) in enumerate(zip(plan['frame_ids'],plan['source_pts'])):
  if ident not in seen:unique.append((ident,pts,i));seen.add(ident)
 for row in rows:
  key=row['sample_key'];path=OUT/'verification'/f'{key}.json'
  if path.exists() and read(path).get('complete'):continue
  data=read(OUT/'queries'/f'{key}.json');prior=read(SOURCE/'rollouts'/f'{key}.json');rawbase=prior['result']['candidates'][0];from temporal_mode_core import parse_clip
  baseline=parse_clip(rawbase['text'] if 'text' in rawbase else rawbase.get('answer',''),data['duration'],True)['span']
  if baseline is None:
   # Original candidate stores a parsed span under the source's own schema.
   value=rawbase.get('span');baseline=[max(0.,min(data['duration'],v)) for v in value] if value and value[0]<value[1] else None
  candidates=[dict(**r,baseline=False) for r in data['intervals']]
  if baseline is not None and not any(r['span']==baseline for r in candidates):candidates.append(dict(id='zero_shot',span=baseline,log_native_mass=None,baseline=True))
  elif baseline is not None:
   for r in candidates:r['baseline']=r['span']==baseline
  saved=read(path) if path.exists() else dict(sample_key=key,complete=False,model=str(VERIFIER),baseline=baseline,candidates=[],input_cache={},cost_definition='each video common prefix once plus separate short criterion suffix forwards; no generated rationale')
  done={r['id'] for r in saved['candidates']};lookup=saved['input_cache']
  for index,candidate in enumerate(candidates):
   if candidate['id'] in done:continue
   start,end=candidate['span'];context=CONFIG['verification_context_seconds'];chosen=[x for x in unique if max(0,start-context)<=x[1]<min(data['duration'],end+context)]
   progress.update(stage='independent_verification',sample_key=key,verification_candidates_done=index,verification_candidates_total=len(candidates))
   record=dict(candidate)
   if not chosen:record.update(status='no_input_frames',event_score=None,joint_score=None)
   else:
    relative=[start-chosen[0][1],end-chosen[0][1]];relpts=[x[1]-chosen[0][1] for x in chosen];fingerprint=digest(dict(model=str(VERIFIER),query=row['query'],frames=[x[0] for x in chosen],relative_span=relative,relative_pts=relpts));record['input_key']=fingerprint
    if fingerprint in lookup:result=lookup[fingerprint];record['reused']=True
    else:
     clip=np.array(frames[[x[2] for x in chosen]])
     if len(clip)%2:clip=np.concatenate([clip,clip[-1:]],axis=0);relpts+=relpts[-1:]
     result=assess(model,processor,clip,relpts,row['query'],relative,progress);lookup[fingerprint]=result;record['reused']=False
    record.update(event_score=result['event_score'],joint_score=result['joint_score'],status='complete')
   saved['candidates'].append(record)
   if index%16==0 or index==len(candidates)-1:write(path,saved)
  finals={}
  for name,scorefield in [('event_only','event_score'),('event_onset_offset','joint_score')]:
   eligible=[r for r in saved['candidates'] if r[scorefield] is not None]
   best=max(eligible,key=lambda r:(r[scorefield],r['baseline'],r['log_native_mass'] if r['log_native_mass'] is not None else -1e300,tuple(-v for v in r['span']))) if eligible else None
   finals[name]=dict(span=best['span'] if best else baseline,candidate=best['id'] if best else None,score=best[scorefield] if best else None,changed=bool(best and best['span']!=baseline))
  saved.update(complete=True,finals=finals,timestamp=stamp());write(path,saved)
 del frames;gc.collect()
