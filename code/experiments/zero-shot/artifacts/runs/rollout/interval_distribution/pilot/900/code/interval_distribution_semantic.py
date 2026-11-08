"""Per-video temporary vectors; permanent scalar statistics only."""
import collections,gc,json,math,sqlite3,time
from pathlib import Path
import numpy as np
from interval_distribution_core import *

def clip_catalog(rows,cache,frames,prior):
 plan=cache['plans']['2'];unique=[];seen=set()
 for j,(ident,pts) in enumerate(zip(plan['frame_ids'],plan['source_pts'])):
  if ident not in seen:unique.append((ident,pts,j));seen.add(ident)
 clips={};mapping={}
 for row in rows:
  query=read(OUT/'queries'/f'{row["sample_key"]}.json')
  for interval in query['intervals']:
   start,end=interval['span'];chosen=[x for x in unique if start<=x[1]<end]
   if not chosen:mapping[row['sample_key'],interval['id']]=None;continue
   ids=[x[0] for x in chosen];pts=[x[1]-chosen[0][1] for x in chosen];positions=[x[2] for x in chosen]
   fingerprint=dict(source=cache['source_sha256'],frame_ids=ids,relative_pts=pts,hw=cache['hw'],instruction=CONFIG['semantic_instruction'],readout='<|endoftext|>',padding='repeat last frame for temporal patch if odd')
   key=digest(fingerprint);clips.setdefault(key,dict(id=key,fingerprint=fingerprint,positions=positions,relative_pts=pts,unique_frames=len(ids),hw=cache['hw']))
   mapping[row['sample_key'],interval['id']]=key
 return clips,mapping

def encode(model,processor,frames,clip,phase):
 import torch
 from interval_distribution_runtime import processor_input
 prepare_start=time.monotonic();torch.cuda.reset_peak_memory_stats();positions=clip['positions'];selected=np.array(frames[positions]);pts=clip['relative_pts'][:];physical=len(selected)
 if physical%2:selected=np.concatenate([selected,selected[-1:]],axis=0);pts+=pts[-1:]
 cpu=processor_input(processor,selected,list(range(len(selected))),pts,2.,len(selected),query=None);gpu=cpu.to('cuda');n=int(cpu.input_ids.shape[1]);limit=int(model.config.text_config.max_position_embeddings)
 if n>=limit:return {},dict(status='context_limit',input_tokens=n,frames=physical)
 prepare_seconds=time.monotonic()-prepare_start;start=time.monotonic();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
 with torch.inference_mode():output=model.model(**gpu,use_cache=False)
 b.record();b.synchronize();cuda_seconds=a.elapsed_time(b)/1000;hidden=output.last_hidden_state[0].float();vectors={}
 def norm(x):
  v=x.float();length=torch.linalg.vector_norm(v)
  return (v/length).cpu().numpy().astype(np.float32) if float(length)>1e-12 else None
 pool_start=time.monotonic();vectors[phase+'_readout']=norm(hidden[-1]);pool_costs={phase+'_readout':time.monotonic()-pool_start}
 if phase=='instruct':
  pool_start=time.monotonic();mask=gpu.input_ids[0]==processor.tokenizer.convert_tokens_to_ids('<|video_pad|>');t,h,w=map(int,cpu.video_grid_thw[0]);visual=hidden[mask];assert len(visual)==t*h*w//4
  pertime=visual.reshape(t,h*w//4,-1).mean(dim=1)
  # Actual input timestamp cell widths; independent of raw interval endpoint aliases.
  centres=np.array([(pts[2*j]+pts[2*j+1])/2 for j in range(t)])
  if t==1:weights=np.ones(1)
  else:
   boundaries=np.r_[centres[0]-(centres[1]-centres[0])/2,(centres[:-1]+centres[1:])/2,centres[-1]+(centres[-1]-centres[-2])/2];weights=np.diff(boundaries)
  wt=torch.as_tensor(weights/weights.sum(),device=pertime.device,dtype=torch.float32);vectors['instruct_mean']=norm((pertime*wt[:,None]).sum(dim=0))
  pool_costs['instruct_mean']=time.monotonic()-pool_start;pool_start=time.monotonic()
  bins=np.minimum(3,((centres-centres.min())/(centres.max()-centres.min()+1e-12)*4).astype(int)) if t>1 else np.zeros(t,int)
  if set(bins)=={0,1,2,3}:
   pooled=[]
   for group in range(4):
    select=torch.as_tensor(bins==group,device=pertime.device);gwt=wt[select];pooled.append((pertime[select]*(gwt/gwt.sum())[:,None]).sum(dim=0))
   vectors['instruct_temporal_four']=norm(torch.cat(pooled));pool_costs['instruct_temporal_four']=time.monotonic()-pool_start
 stats=dict(status='complete',input_prepare_wall_seconds=prepare_seconds,pooling_wall_seconds=pool_costs,pooling_scope='incremental actual shared forward; not standalone GPU costs',input_tokens=n,visual_tokens=int(cpu.video_grid_thw.prod())//4,frames=physical,temporal_groups=int(cpu.video_grid_thw[0,0]),wall_seconds=time.monotonic()-start,cuda_stream_seconds=cuda_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated())
 del hidden,output,gpu,cpu;return {k:v for k,v in vectors.items() if v is not None},stats

def scalar_summary(mu):
 length2=float(np.dot(mu,mu));return dict(mean_norm_squared=length2,semantic_dispersion=max(0.,1-length2))

def query_statistics(row,phase,work,mapping):
 key=row['sample_key'];query=read(OUT/'queries'/f'{key}.json');intervals={r['id']:r for r in query['intervals']};pools=['embedding_readout'] if phase=='embedding' else ['instruct_mean','instruct_temporal_four','instruct_readout'];result=dict(sample_key=key,phase=phase,pools={},vectors_persisted=False)
 for pool in pools:
  vectors={}
  for ident in intervals:
   clip=mapping[key,ident]
   if clip is None:continue
   item=work.execute('SELECT vector FROM vectors WHERE clip=? AND pool=?',(clip,pool)).fetchone()
   if item:vectors[ident]=np.frombuffer(item[0],dtype=np.float32)
  means=[];groups=[]
  for group in query['groups']:
   ids=[k for k in group['interval_ids'] if k in vectors];covered=sum(intervals[k]['conditional_weight'] for k in ids)
   if not ids:groups.append(dict(group=group['id'],valid=False,covered_mass=0.));means.append(None);continue
   weights=np.array([intervals[k]['conditional_weight']/covered for k in ids]);x=np.stack([vectors[k] for k in ids]);mu=weights@x;alignment=x@mu;nearest=int(np.argmax(alignment));farthest=int(np.argmin(alignment));sumw2=float(weights@weights);scalars=scalar_summary(mu)
   groups.append(dict(group=group['id'],valid=True,covered_mass=covered,group_mass=group['mass'],unique_embedding_inputs=len({mapping[key,k] for k in ids}),intervals=len(ids),sum_squared_weights=sumw2,distinct_interval_mean_cosine=(scalars['mean_norm_squared']-sumw2)/(1-sumw2) if sumw2<1-1e-12 else None,near_interval=ids[nearest],far_interval=ids[farthest],**scalars));means.append(mu)
  pairs=[]
  for a in range(len(groups)):
   if means[a] is None:continue
   for b in range(a+1,len(groups)):
    if means[b] is None:continue
    similarity=float(means[a]@means[b]);den=math.sqrt(groups[a]['mean_norm_squared']*groups[b]['mean_norm_squared']);pairs.append(dict(left=groups[a]['group'],right=groups[b]['group'],weighted_mean_cosine=similarity,centroid_cosine=similarity/den if den>1e-12 else None))
  # Branch-level statistics are accumulated bottom-up; no vector is written to raw output.
  tree=connect(OUT/'trees'/f'{key}.sqlite');pending={};children=collections.defaultdict(list);branch_rows=[]
  for ident,parent,p,status,span,accepted in tree.execute('SELECT id,parent,p,status,span,accepted FROM nodes ORDER BY id DESC'):
   entries=pending.pop(ident,[])
   if status=='valid':
    span_id=digest(json.loads(span))[:20]
    if span_id in vectors:entries=[(0.,vectors[span_id])]
   if entries:
    logmass=lsum(v[0] for v in entries);mu=sum((math.exp(l-logmass)*v for l,v in entries),np.zeros_like(entries[0][1],dtype=np.float64))
    kids=children.pop(ident,[])
    if accepted and len(json.loads(accepted))>1:
     record=dict(node=ident,log_covered_valid_mass=logmass,**scalar_summary(mu));childpairs=[]
     for i in range(len(kids)):
      for j in range(i+1,len(kids)):childpairs.append(dict(left=kids[i][0],right=kids[j][0],weighted_mean_cosine=float(kids[i][1]@kids[j][1])))
     record['children']=childpairs;branch_rows.append(record)
    if parent is not None:pending.setdefault(parent,[]).append((math.log(p)+logmass,mu));children[parent].append((ident,mu))
  tree.close();result['pools'][pool]=dict(groups=groups,pairs=pairs,branches=branch_rows,embedding_coverage_mass=sum(intervals[k]['conditional_weight'] for k in vectors),valid_intervals=len(vectors),dimensions=int(next(iter(vectors.values())).shape[0]) if vectors else None)
 write(OUT/'semantic'/phase/f'{key}.json',result)
 summary=dict(result,pools={name:({k:v for k,v in value.items() if k!='branches'} if value else None) for name,value in result['pools'].items()})
 write(OUT/'semantic'/phase/f'{key}.summary.json',summary)
 return result

def semantic_video(rows,model,processor,phase,progress):
 from interval_distribution_runtime import source_frames
 vid=digest(rows[0]['video_path'])[:24];marker=OUT/'semantic'/phase/f'{vid}.complete.json'
 previous=read(marker) if marker.exists() else dict(queries=[],costs=[],unique_clips=0,wall_seconds=0.)
 rows=[r for r in rows if not (OUT/'semantic'/phase/f'{r["sample_key"]}.json').exists()]
 if not rows:return
 prior,cache,frames=source_frames(rows[0]);clips,mapping=clip_catalog(rows,cache,frames,prior);path=OUT/'work'/f'{vid}_{phase}.sqlite';work=sqlite3.connect(path);work.executescript('CREATE TABLE IF NOT EXISTS vectors(clip TEXT,pool TEXT,vector BLOB,PRIMARY KEY(clip,pool));CREATE TABLE IF NOT EXISTS costs(clip TEXT PRIMARY KEY,data TEXT);')
 started=time.monotonic()
 for index,(key,clip) in enumerate(clips.items()):
  if work.execute('SELECT 1 FROM costs WHERE clip=?',(key,)).fetchone():continue
  progress.update(stage=phase+'_encoding',video=rows[0]['video_path'],semantic_clips_done=index,semantic_clips_total=len(clips),semantic_elapsed_seconds=time.monotonic()-started)
  vectors,cost=encode(model,processor,frames,clip,phase);cost.update(clip=key,execution_queries=[r['sample_key'] for r in rows])
  for pool,vector in vectors.items():work.execute('INSERT OR REPLACE INTO vectors VALUES (?,?,?)',(key,pool,vector.tobytes()))
  work.execute('INSERT OR REPLACE INTO costs VALUES (?,?)',(key,json.dumps(cost)));work.commit()
 for row in rows:query_statistics(row,phase,work,mapping)
 costs=[json.loads(v[0]) for v in work.execute('SELECT data FROM costs')];clipledger=[dict(clip=k,**v['fingerprint'],unique_frames=v['unique_frames']) for k,v in clips.items()]
 ledger_path=OUT/'semantic'/phase/f'{vid}.inputs.json';old=read(ledger_path) if ledger_path.exists() else dict(clips=[],query_interval_mapping=[])
 merged={v['clip']:v for v in old['clips']+clipledger}
 write(ledger_path,dict(video=rows[0]['video_path'],clips=list(merged.values()),query_interval_mapping=old['query_interval_mapping']+[dict(query=k[0],interval=k[1],clip=v) for k,v in mapping.items()]))
 write(marker,dict(complete=True,queries=previous['queries']+[r['sample_key'] for r in rows],unique_clips=len(merged),costs=previous['costs']+costs,wall_seconds=previous['wall_seconds']+time.monotonic()-started,permanent_vectors=False,timestamp=stamp()))
 work.close();path.unlink();del frames;gc.collect()
