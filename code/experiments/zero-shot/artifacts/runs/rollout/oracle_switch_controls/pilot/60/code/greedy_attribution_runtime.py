"""Full-prefill gradients, frozen top-30 directional screening, staged reports."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse, collections, concurrent.futures, contextlib, fcntl, gc, gzip, importlib, json, math, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path
import numpy as np
from greedy_attribution_core import *
RUN_PROGRESS=None

class Progress:
    def __init__(self):
        self.start=time.monotonic(); self.stop=threading.Event(); self.lock=threading.RLock()
        self.state=dict(state='running',stage='initializing',completed=0,total=900,active_queries=300,
            model='Qwen3-VL-8B-Instruct',eta_seconds=None,eta_scope='active pilot remaining queries; excludes report validation')
        threading.Thread(target=self.heartbeat,daemon=True).start()
    def __call__(self,**kw):
        with self.lock:
            self.state.update(kw,timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start)
            self.state['percentage']=100*self.state['completed']/self.state['active_queries']
            write(storage()/'metadata/status.json',self.state)
            write(ROOT/'artifacts/common/registry/greedy_attribution.live.json',dict(canonical=str(storage()),
                status=str(storage()/'metadata/status.json'),log=str(storage()/'logs/run.log'),**self.state))
            with (storage()/'logs/progress.log').open('a') as f:f.write(json.dumps(self.state,ensure_ascii=False)+'\n')
            if kw:print(json.dumps(kw,ensure_ascii=False),flush=True)
    def heartbeat(self):
        while not self.stop.wait(20): self()

def setup():
    out=storage()
    for name in ('metadata','logs','queries','targets','arrays','validation','code'):(out/name).mkdir(parents=True,exist_ok=True)
    if (out/'metadata/config.json').exists():assert read(out/'metadata/config.json')==CONFIG
    else:write(out/'metadata/config.json',CONFIG)
    rows=read(COHORT)
    for n in (300,600,900):
        assert read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json')==rows[:n]
    write(out/'metadata/cohort.json',rows[:int(out.name)])
    sources=[]
    for row in rows:
        path=SOURCE/'rollouts'/f'{row["sample_key"]}.json'
        sources.append(dict(sample_key=row['sample_key'],path=str(path),sha256=sha(path)))
    write(out/'metadata/source_manifest.json',sources)
    files=[]
    for source in sorted((ROOT/'_lib').glob('greedy_attribution_*')):
        if source.is_file():
            target=out/'code'/source.name;shutil.copyfile(source,target)
            files.append(dict(path=str(source),snapshot=str(target),sha256=sha(source)))
    write(out/'metadata/provenance.json',dict(timestamp=stamp(),config_hash=digest(CONFIG),cohort_sha256=sha(COHORT),
        cohort_hashes={str(n):sha(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json') for n in (300,600,900)},
        model_config_sha256=sha(MODEL/'config.json'),model_tokenizer_sha256=sha(MODEL/'tokenizer.json'),
        model_registry_sha256=sha('/home/irteam/backbones/manifests/registry.json'),files=files))
    write(out/'validation/cohort.json',dict(passed=True,nested_prefixes=[300,600,900],queries=900,timestamp=stamp()))
    return rows

def load():
    from probability_budget_runtime import load_model
    import torch
    from torch.utils.checkpoint import checkpoint
    model,processor,seconds=load_model();model.requires_grad_(False)
    model._attribution_checkpoint=False
    model._attribution_captures=[];model._attribution_offset=0
    for layer in list(model.model.language_model.layers)+list(model.model.visual.blocks):
        original=layer.forward
        def forward(*args,_original=original,**kwargs):
            captures=tuple(model._attribution_captures);offset=model._attribution_offset
            def call(*inner):
                old=[capture.segment_offset for capture in captures]
                try:
                    for capture in captures:capture.set_segment(offset)
                    return _original(*inner,**kwargs)
                finally:
                    for capture,previous in zip(captures,old):capture.set_segment(previous)
            if model._attribution_checkpoint and torch.is_grad_enabled():
                return checkpoint(call,*args,use_reentrant=False)
            return call(*args)
        layer.forward=forward
    return model,processor,seconds

def sequential_forward(model,gpu,prompt_length,count,capture=None,messages=None):
    """Native prefill followed by single-token decode with connected gradients."""
    import torch
    from greedy_attribution_cache import SnapshotDynamicCache
    captures=[item for item in (capture,messages) if item is not None]
    for item in captures:item.begin_forward()
    model._attribution_captures=captures
    cache=None;outputs=[]
    for step in range(count):
        offset=0 if step==0 else prompt_length+step-1
        model._attribution_offset=offset
        for item in captures:item.set_segment(offset)
        cache=SnapshotDynamicCache(previous=cache,config=model.config)
        if step==0:
            inputs=dict(gpu);inputs['input_ids']=gpu['input_ids'][:,:prompt_length]
            inputs['attention_mask']=gpu['attention_mask'][:,:prompt_length]
        else:
            inputs=dict(input_ids=gpu['input_ids'][:,offset:offset+1],attention_mask=gpu['attention_mask'][:,:offset+1],
                cache_position=torch.tensor([offset],device=gpu['input_ids'].device))
        out=model(**inputs,past_key_values=cache,use_cache=True,logits_to_keep=1)
        outputs.append(out.logits[0,-1])
    return torch.stack(outputs)

def saved_probabilities(key):
    trace_path=SOURCE/'token_traces'/key/'index.json';trace=read(trace_path);nodes=trace['nodes']
    ids=[];probabilities=[]
    for node in trace['candidates']['1']['nodes']:
        top=nodes[node]['top'][:CONFIG['candidate_top_k']]
        ids.append([r['id'] for r in top]);probabilities.append([r['probability'] for r in top])
    return np.asarray(ids),np.asarray(probabilities,dtype=np.float32),trace_path

def preflight(rows,model,processor,progress):
    """Measure source-KV versus full-prefill drift, never change source tokens."""
    import torch
    from probability_budget_runtime import prepare
    path=storage()/'validation/source_replay.json'
    records=read(path).get('records',[]) if path.exists() else []
    seen={r['sample_key'] for r in records}
    for row in rows[:300]:
        key=row['sample_key']
        if key in seen:continue
        progress(stage='source_replay_calibration',sample_key=key,calibration_completed=len(records),calibration_total=300)
        cpu,info=prepare(row,processor);gpu=cpu.to('cuda');ids=read(SOURCE/'rollouts'/f'{key}.json')['result']['candidates'][0]['ids']
        gpu['input_ids']=torch.cat((gpu.input_ids,torch.tensor([ids[:-1]],device='cuda')),dim=1);gpu['attention_mask']=torch.ones_like(gpu.input_ids)
        top_ids,reference,_=saved_probabilities(key)
        with torch.no_grad():
            logits=model(**gpu,use_cache=False,logits_to_keep=len(ids)).logits[0].float()
            prob=logits.softmax(-1).gather(1,torch.as_tensor(top_ids,device='cuda')).cpu().numpy();argmax=logits.argmax(-1).cpu().tolist()
            mismatch=[i for i,(a,b) in enumerate(zip(argmax,ids)) if a!=b]
            gaps=[float(logits[i,argmax[i]]-logits[i,ids[i]]) for i in mismatch]
        records.append(dict(sample_key=key,dataset=row['dataset'],max_probability_error=float(np.abs(prob-reference).max()),
            argmax_mismatch_steps=mismatch,max_source_token_logit_gap=max(gaps,default=0.),input_hashes=info['hashes']))
        errors=np.asarray([r['max_probability_error'] for r in records])
        write(path,dict(records=records,completed=len(records),total=300,complete=len(records)==300,
            mean_probability_error=float(errors.mean()),max_probability_error=float(errors.max()),
            quantiles=dict(zip(['p50','p90','p95','p99'],np.quantile(errors,[.5,.9,.95,.99]).tolist())),
            mismatch_queries=sum(bool(r['argmax_mismatch_steps']) for r in records),timestamp=stamp()))
        del logits,gpu,cpu;gc.collect();torch.cuda.empty_cache()
    return read(path)

def screening(model,gpu,capture,context,base_logits,top_ids,progress,key):
    import torch
    epsilon=CONFIG['screen_epsilon'];slopes=[];changes=[];directions=[]
    pixels=gpu['pixel_values_videos']
    for group in CONFIG['screen_directions']:
        indices=[i for i,g in enumerate(context['groups']) if g==group]
        if group!='video_pixels' and not indices:continue
        pair=[]
        for sign in (-1,1):
            progress(stage='directional_screening',sample_key=key,direction=group,sign=sign)
            if group=='video_pixels':gpu['pixel_values_videos']=pixels*(1+sign*epsilon)
            else:capture.set_embedding_scale(indices,1+sign*epsilon)
            with torch.no_grad():
                full=sequential_forward(model,gpu,capture.prompt_length,base_logits.shape[0],capture=capture)
                logits=full.gather(1,torch.as_tensor(top_ids,device=full.device)).float().cpu().numpy()
                del full
            pair.append(logits)
            capture.clear_embedding_scale();gpu['pixel_values_videos']=pixels
        slopes.append((pair[1]-pair[0])/(2*epsilon))
        changes.append(np.stack([pair[0]-base_logits,pair[1]-base_logits]))
        directions.append(group)
    return directions,np.stack(slopes),np.stack(changes)

def candidate_steps(base,prob,top_ids,directions,slopes,changes,ids,tokenizer,source,duration):
    lp=np.log(np.maximum(prob,1e-45))
    steps=[];detailed=[];selected=np.asarray([list(top_ids[j]).index(token) for j,token in enumerate(ids)])
    greedy_scores=base[np.arange(len(ids)),selected]
    difference=slopes-slopes[:,np.arange(len(ids)),selected][:,:,None]
    gain=CONFIG['screen_epsilon']*np.max(np.abs(difference),axis=0)
    margins=greedy_scores[:,None]-base;gaps=margins-gain
    observed_change=changes-changes[:,:,np.arange(len(ids)),selected][:,:,:,None]
    best_observed_gap=margins-np.max(observed_change,axis=(0,1))
    for step,token in enumerate(ids):
        chosen_column=int(selected[step])
        ranking=np.argsort(-prob[step],kind='stable');top=ranking[:30].tolist()
        candidates=set(top);candidates.discard(chosen_column)
        for direction in range(len(directions)):
            local=margins[step]-CONFIG['screen_epsilon']*np.abs(difference[direction,step])
            for alt in np.argsort(local,kind='stable')[:CONFIG['screen_candidates_per_direction']+1]:
                if alt!=chosen_column:candidates.add(int(alt))
        # The best currently-low-probability response to every tested direction
        # is retained even if it never reaches the competition threshold.
        low=prob[step]<CONFIG['current_min_probability']
        for direction in range(len(directions)):
            local=np.where(low,margins[step]-CONFIG['screen_epsilon']*np.abs(difference[direction,step]),np.inf)
            alt=int(local.argmin())
            if alt!=chosen_column and np.isfinite(local[alt]):candidates.add(alt)
        rows=[]
        for alt in sorted(candidates,key=lambda i:(gaps[step,i],i)):
            d=int(np.argmax(np.abs(difference[:,step,alt])))
            current=bool(prob[step,alt]>=CONFIG['current_min_probability'] or margins[step,alt]<=CONFIG['current_max_logit_gap'])
            potential=bool(gaps[step,alt]<=CONFIG['potential_max_screen_gap'])
            alt_id=int(top_ids[step,alt])
            row=dict(id=alt_id,token=tokenizer.decode([alt_id],clean_up_tokenization_spaces=False),probability=float(prob[step,alt]),
                margin=float(margins[step,alt]),screen_gain=float(gain[step,alt]),screen_gap=float(gaps[step,alt]),
                screen_direction=directions[d],screen_sign=1 if difference[d,step,alt]>=0 else -1,
                observed_best_gap=float(best_observed_gap[step,alt]),current_competitor=current,potential_competitor=potential,
                selection_reason='current_and_potential' if current and potential else 'current' if current else 'potential' if potential else 'directional_probe',
                mode=source_mode(source,step,alt_id,ids[:step],duration))
            rows.append(row)
            detailed.append((step,row))
        def item(i):return dict(id=int(top_ids[step,i]),token=tokenizer.decode([int(top_ids[step,i])],clean_up_tokenization_spaces=False),probability=float(prob[step,i]))
        mass=float(prob[step].sum(dtype=np.float64))
        steps.append(dict(step=step,prefix=tokenizer.decode(ids[:step],clean_up_tokenization_spaces=False),chosen=item(chosen_column),
            top30_entropy_contribution=float(-(prob[step]*lp[step]).sum()),top=[item(i) for i in top],candidates=rows,
            tracked_tokens=30,top30_mass=mass,omitted_mass=max(0.,1-mass)))
    # Diversity in investigation: one currently competitive and one latent
    # branch, followed by mode-separated/low-gap alternatives at new steps.
    choices=[];seen=set()
    pools=[sorted([r for r in detailed if r[1]['current_competitor']],key=lambda r:(not r[1]['mode']['distinct_mode_candidate'],r[1]['margin'],r[0])),
        sorted([r for r in detailed if not r[1]['current_competitor']],key=lambda r:(r[1]['screen_gap'],r[0])),
        sorted(detailed,key=lambda r:(not r[1]['mode']['distinct_mode_candidate'],r[1]['screen_gap'],r[0]))]
    for pool in pools[:2]:
        if pool:
            pair=pool[0];choices.append(pair);seen.add((pair[0],pair[1]['id']))
    for step,row in pools[2]:
        if len(choices)>=CONFIG['detailed_branch_targets']:break
        if (step,row['id']) in seen:continue
        if any(step==s for s,_ in choices) and any(s not in {x[0] for x in choices} for s,_ in pools[2]):continue
        choices.append((step,row));seen.add((step,row['id']))
    return steps,choices,prob,gaps

def pixel_maps(pixels,gradient,grid):
    import torch
    t,h,w=grid
    # Processor order: time pair, merged y/x, within-block y/x, C, temporal, patch y/x.
    g=gradient.reshape(t,h//2,w//2,2,2,3,2,16,16).float()
    x=pixels.detach().reshape_as(g).float()
    square=g.square().sum(dim=(5,7,8));dot=(g*x).sum(dim=(5,7,8))
    def frame(a):return a.permute(0,5,1,3,2,4).reshape(2*t,h,w)
    return dict(gradient_norm=frame(square).sqrt().cpu().tolist(),activation_dot_gradient=frame(dot).cpu().tolist())

def validate_messages(data,summary,expected_layers):
    """Gate pre-projection message scores with reconstruction/roundoff bounds."""
    expected=set(int(layer) for layer in expected_layers)
    reconstructed={int(row['layer']):row for row in data['reconstruction']}
    measured={int(row['layer']):row for row in data['layers']}
    attention={int(row['layer']):row for row in summary['layers']['attention']}
    assert set(reconstructed)==set(measured)==expected,('message layer coverage',sorted(reconstructed),sorted(measured),sorted(expected))
    checks=[]
    for layer in sorted(expected):
        reconstruction=reconstructed[layer]
        errors=np.asarray([reconstruction[name] for name in ('maximum_absolute_difference','rms_difference','reference_rms','rms_relative_difference')],dtype=np.float64)
        assert np.isfinite(errors).all(),('nonfinite message reconstruction',layer)
        assert reconstruction['rms_relative_difference']<=.03,('message reconstruction RMS',layer,reconstruction)
        heads=np.asarray(measured[layer]['head_score_sum'],dtype=np.float64)
        actual_heads=np.asarray(measured[layer]['actual_attention_input_dot_gradient'],dtype=np.float64)
        reconstruction_bound=np.asarray(measured[layer]['reconstruction_dot_error_bound'],dtype=np.float64)
        roundoff_bound=np.asarray(measured[layer]['fp32_roundoff_error_bound'],dtype=np.float64)
        block=np.asarray(attention[layer]['activation_dot_gradient'],dtype=np.float64)
        assert heads.ndim==2 and heads.shape[1]==len(block),('message score shape',layer,heads.shape,block.shape)
        assert actual_heads.shape==reconstruction_bound.shape==roundoff_bound.shape==heads.shape,('pre-projection bound shape',layer)
        assert all(np.isfinite(value).all() for value in (heads,actual_heads,reconstruction_bound,roundoff_bound,block)),('nonfinite message score',layer)
        assert (reconstruction_bound>=0).all() and (roundoff_bound>=0).all(),('negative message error bound',layer)
        summed=heads.sum(axis=0);actual=actual_heads.sum(axis=0);delta=np.abs(summed-actual)
        bound=(reconstruction_bound+roundoff_bound).sum(axis=0)
        cpu_roundoff=np.finfo(np.float64).eps*(heads.shape[0]+2)*(np.abs(heads).sum(axis=0)+np.abs(actual_heads).sum(axis=0)+bound)
        tolerance=np.nextafter(bound+cpu_roundoff,np.inf)
        fraction=delta/tolerance
        assert np.all(delta<=tolerance),('message versus pre-projection score bound',layer,float(delta.max()),float(fraction.max()))
        after_projection_delta=np.abs(summed-block)
        for edge in measured[layer]['edges']:
            assert math.isfinite(edge['score']) and math.isfinite(edge['attention_weight']),('nonfinite message edge',layer)
            assert 0<=edge['source_position']<=edge['receiver_position'],('noncausal message edge',layer,edge)
        checks.append(dict(layer=layer,reconstruction_max_absolute_difference=float(errors[0]),
            reconstruction_rms_relative_difference=float(errors[3]),message_sum_max_absolute_difference=float(delta.max()),
            message_sum_max_error_bound=float(tolerance.max()),message_sum_max_tolerance_fraction=float(fraction.max()),
            after_projection_max_absolute_difference=float(after_projection_delta.max())))
    return dict(passed=True,method='pre_projection_reconstruction_bound',reconstruction_max_relative_rms=.03,
        fp32_unit_roundoff=2.0**-24,layers=checks,after_projection_diagnostic_only=True,
        error_bound='sum over heads and coordinates of |gradient|*|reconstructed pre-projection output-actual input| plus conservative FP32 contraction/reduction roundoff',
        identity_scope='within each selected head before o_proj at prediction positions; no cross-layer evidence sum')

def compact_floats(value):
    # JSON float formatting is compact already; preserve raw float32 values.
    return value

def gzip_write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    compressed=gzip.compress(raw,compresslevel=1,mtime=0)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(compressed);tmp.replace(path)
    return dict(sha256=hashlib.sha256(compressed).hexdigest(),bytes=len(compressed))

def process(row,model,processor,progress,dest):
    import torch
    from probability_budget_runtime import prepare
    from greedy_attribution_hooks import AttributionHooks
    from deterministic_rollout300 import sites
    start=time.monotonic();key=row['sample_key'];source_path=SOURCE/'rollouts'/f'{key}.json';source=read(source_path)
    saved=source['result']['candidates'][0];ids=saved['ids'];old_input=read(SOURCE/'inputs'/f'{key}.json')
    progress(stage='preparing_input',sample_key=key)
    cpu,info=prepare(row,processor);assert info['hashes']==source['input_hashes']==old_input['hashes']
    gpu=cpu.to('cuda');n=int(gpu.input_ids.shape[1]);length=n+len(ids)-1
    gpu['input_ids']=torch.cat((gpu.input_ids,torch.tensor([ids[:-1]],device='cuda')),dim=1)
    gpu['attention_mask']=torch.ones_like(gpu.input_ids)
    context=context_map(gpu.input_ids[0].tolist(),processor.tokenizer,n,row,old_input['grid'],info['timestamps'])
    predictors=list(range(n-1,length));groups={g:[i for i,x in enumerate(context['groups']) if x==g] for g in GROUPS}
    model._attribution_checkpoint=n>CONFIG['checkpoint_input_tokens'];torch.cuda.reset_peak_memory_stats()
    top_ids,expected,trace_path=saved_probabilities(key)
    targets=[];timings={};writer=concurrent.futures.ThreadPoolExecutor(max_workers=2);pending=[]
    with AttributionHooks(model,n,length,predictors,groups) as capture:
        progress(stage='replay_validation',sample_key=key)
        t=time.monotonic()
        with torch.no_grad():
            full=sequential_forward(model,gpu,n,len(ids),capture=capture).float()
            indices=torch.as_tensor(top_ids,device=full.device)
            base=full.gather(1,indices).cpu().numpy();actual=full.softmax(-1).gather(1,indices).cpu().numpy()
            argmax=full.argmax(-1).cpu().numpy()
            error=float(np.abs(actual-expected).max())
            match=bool(np.array_equal(argmax,ids));mismatches=[int(i) for i in np.where(argmax!=ids)[0]]
            near_ties=[i for i in mismatches if float(full[i,argmax[i]]-full[i,ids[i]])<=.125]
            del full
        assert error<=CONFIG['replay_probability_tolerance'],(key,'replay probability',error)
        assert match,(key,'source greedy argmax mismatch',mismatches)
        timings['replay_seconds']=time.monotonic()-t
        t=time.monotonic();directions,slopes,changes=screening(model,gpu,capture,context,base,top_ids,progress,key);timings['screen_seconds']=time.monotonic()-t
        steps,choices,probs,screen_gaps=candidate_steps(base,actual,top_ids,directions,slopes,changes,ids,processor.tokenizer,source,info['duration'])
        for step in steps:
            lookup=dict(zip(top_ids[step['step']].tolist(),expected[step['step']].tolist()))
            for candidate in step['top']+step['candidates']+[step['chosen']]:
                candidate['source_probability']=lookup[candidate['id']]
                if 'margin' in candidate:
                    source_margin=math.log(max(lookup[step['chosen']['id']],1e-45)/max(lookup[candidate['id']],1e-45))
                    candidate['source_margin']=source_margin
                    candidate['margin_sensitive_to_replay']=bool(abs(candidate['margin']-source_margin)>=max(abs(candidate['margin']),1e-6) or candidate['margin']*source_margin<0)
        array_path=dest/'arrays'/f'{key}.npz';array_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(array_path,token_ids=top_ids,logits=base,probabilities=probs,source_probabilities=expected,slopes=slopes,changes=changes,screen_gaps=screen_gaps,directions=np.asarray(directions))
        gpu['pixel_values_videos']=gpu['pixel_values_videos'].detach().requires_grad_(True)
        messages=None
        try:
            from greedy_attribution_messages import AttentionMessages
            messages=AttentionMessages(model,predictors)
        except ImportError:pass
        with messages if messages is not None else contextlib.nullcontext():
            t=time.monotonic();progress(stage='gradient_forward',sample_key=key)
            logits=sequential_forward(model,gpu,n,len(ids),capture=capture,messages=messages).float();lp=logits.log_softmax(-1)
            grad_replay_error=float((logits.detach().gather(1,indices).cpu()-torch.from_numpy(base)).abs().max());assert grad_replay_error==0
            timings['gradient_forward_seconds']=time.monotonic()-t
            objectives=[]
            for j,token in enumerate(ids):objectives.append((dict(id=f'chosen_{j}',kind='chosen_log_probability',step=j,alternative=None),lp[j,token]))
            role_map=dict(sites(ids,processor.tokenizer))
            for endpoint in ('start','end'):
                positions=[j for j,roles in role_map.items() if any(r.startswith(endpoint+'.') for r in roles)]
                if positions:objectives.append((dict(id=endpoint,kind=endpoint+'_log_probability',step=max(positions),steps=positions,alternative=None),lp[positions,torch.tensor([ids[j] for j in positions],device='cuda')].sum()))
            for j,alternative in choices:
                objectives.append((dict(id=f'margin_{j}_{alternative["id"]}',kind='logit_margin',step=j,alternative=alternative['id'],branch=alternative),logits[j,ids[j]]-logits[j,alternative['id']]))
            t=time.monotonic()
            for idx,(meta,objective) in enumerate(objectives):
                progress(stage='backward',sample_key=key,target=meta['id'],target_index=idx+1,target_count=len(objectives))
                capture.begin_target(meta['id'])
                if messages is not None:messages.begin_target(meta['id'])
                gradients=torch.autograd.grad(objective,tuple([gpu['pixel_values_videos']]+capture.embedding_leaves),retain_graph=idx<len(objectives)-1,allow_unused=True)
                pixel_gradient=gradients[0];assert pixel_gradient is not None
                summary=capture.collect();ctx=summary['context'];norm=np.asarray(ctx['gradient_norm_l2']);dots=np.asarray(ctx['activation_dot_gradient'])
                message_data=messages.collect() if messages is not None else dict(measured=False)
                message_validation=validate_messages(message_data,summary,messages.layers) if messages is not None else dict(passed=False,measured=False)
                end=n+meta['step'];future=float(norm[end:].max()) if end<length else 0.;assert future<1e-6,(key,meta['id'],future)
                group_rows=[]
                for group,indices in groups.items():
                    prior=[i for i in indices if i<end]
                    group_rows.append(dict(group=group,count=len(prior),grad_l2=float(np.linalg.norm(norm[prior])),signed_sum=float(dots[prior].sum()),absolute_sum=float(np.abs(dots[prior]).sum())))
                vg=np.asarray(context['visual_positions']);tt,hh,ww=old_input['grid']
                frame=dict(gradient_norm=norm[vg].reshape(tt,hh//2,ww//2).tolist(),activation_dot_gradient=dots[vg].reshape(tt,hh//2,ww//2).tolist())
                pm=pixel_maps(gpu['pixel_values_videos'],pixel_gradient,old_input['grid'])
                pixel_l2=float(pixel_gradient.float().norm());pixel_signed=float((pixel_gradient.float()*gpu['pixel_values_videos'].detach().float()).sum())
                target=dict(meta,score=float(objective.detach()),groups=group_rows,pixel_gradient_l2=pixel_l2,pixel_signed_sum=pixel_signed,
                    causality_max_future_gradient=future,message_validation=message_validation,data_file=f'targets/{key}/{meta["id"]}.json.gz')
                payload=dict(target,context=dict(grad_norm=ctx['gradient_norm_l2'],activation_dot_gradient=ctx['activation_dot_gradient']),
                    embedding=summary.get('embedding'),layers=summary['layers'],frames=frame,pixel_frames=pm,predictor_positions=predictors,
                    messages=message_data)
                pending.append((target,writer.submit(gzip_write,dest/target['data_file'],payload)));targets.append(target)
                if len(pending)>=2:
                    old,future_write=pending.pop(0);old.update(future_write.result())
                del pixel_gradient,gradients,payload,pm,summary,message_data
            timings['backward_seconds']=time.monotonic()-t
            del objectives,objective,logits,lp
    for meta,future in pending:meta.update(future.result())
    writer.shutdown();span=saved.get('parsed');span=[span['start'],span['end']] if span and span['positive'] else None
    info.update(prompt_length=n,total_length=length,grid=old_input['grid'],source=str(SOURCE/'inputs'/f'{key}.json'))
    rec=dict(sample_key=key,sample=row,input=info,context=context,greedy=dict(ids=ids,tokens=processor.tokenizer.batch_decode([[i] for i in ids],clean_up_tokenization_spaces=False),text=saved['answer'],span=span),
        steps=steps,targets=targets,source=dict(record=str(source_path),record_sha256=sha(source_path),trace=str(trace_path),trace_sha256=sha(trace_path)),
        arrays=dict(file=f'arrays/{key}.npz',sha256=sha(array_path),directions=directions,tracked_tokens=30),
        validation=dict(input_hashes_match=True,greedy_argmax_match=match,greedy_near_tie_steps=near_ties,
            replay_max_probability_error=error,gradient_forward_max_logit_error=grad_replay_error,causality_passed=True),
        cost=dict(timings,total_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpointing=model._attribution_checkpoint),
        config_hash=digest(CONFIG),complete=True,timestamp=stamp())
    write(dest/'queries'/f'{key}.json',rec)
    model._attribution_captures=[]
    del gpu,cpu,capture,base,slopes,changes,expected,probs,screen_gaps
    gc.collect();torch.cuda.empty_cache()
    return rec

def promote(scope,progress):
    old=storage();new=BASE/str(scope)
    if old==new:return
    with progress.lock:
        new.mkdir(parents=True,exist_ok=True)
        for category in ('queries','targets','arrays','metadata','logs','validation','code'):
            if (old/category).exists():shutil.move(str(old/category),str(new/category))
        write(new/'metadata/cohort.json',read(COHORT)[:scope])
        for n in (300,600):
            if n>=scope:continue
            write(BASE/str(n)/'metadata/source_records.json',dict(canonical=str(new),cohort=str(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json'),queries=n))

def postprocess(scope,partial=False):
    # Presentation and browser checks can evolve while a long run is active.
    # Refresh these modules only; frozen inference/configuration stays loaded.
    import greedy_attribution_check
    check = importlib.reload(greedy_attribution_check)
    check.validate_scope(scope,allow_partial=partial)
    refresh_report(scope,allow_partial=partial)
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
    check.check_viewer(scope,allow_partial=partial)
    if not partial:
        audit_path=storage()/'validation/pixel_sum_audit.json'
        audit=read(audit_path) if audit_path.exists() else None
        write(storage()/'validation'/f'pilot{scope}.json',dict(passed=True,queries=scope,report_and_viewer_verified=True,timestamp=stamp(),
            gradient_repeatability_verified=False,known_backward_audit=dict(path=str(audit_path),sample_key=audit.get('sample_key'),
            target_id=audit.get('target_id'),passed=audit.get('passed')) if audit else None))
        refresh_report(allow_partial=True)
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)

def refresh_report(scope=None,allow_partial=False):
    import greedy_attribution_report
    return importlib.reload(greedy_attribution_report).report(scope,allow_partial=allow_partial)

def run(args):
    global RUN_PROGRESS
    rows=setup();progress=Progress();progress(stage='loading_model');model,processor,load_seconds=load()
    RUN_PROGRESS=progress
    write(storage()/'metadata/environment.json',dict(model_load_seconds=load_seconds,python=sys.executable,timestamp=stamp()))
    if args.preflight:
        preflight(rows,model,processor,progress);progress(state='calibration_complete',stage='calibration_complete');progress.stop.set();return
    if args.smoke:
        dest=Path('/tmp/greedy_attribution_smoke');dest.mkdir(exist_ok=True)
        rec=process(rows[args.smoke_index],model,processor,progress,dest)
        write(storage()/'validation/gpu_smoke.json',dict(passed=True,config_hash=digest(CONFIG),sample_key=rec['sample_key'],validation=rec['validation'],cost=rec['cost'],targets=len(rec['targets']),timestamp=stamp()))
        progress(state='smoke_complete',stage='smoke_complete');progress.stop.set();return
    done={p.stem for p in (storage()/'queries').glob('*.json') if read(p).get('complete')}
    elapsed=[]
    for scope in (300,600,900):
        if (storage()/'validation'/f'pilot{scope}.json').exists():continue
        progress(active_queries=scope,completed=sum(r['sample_key'] in done for r in rows[:scope]))
        for row in rows[:scope]:
            if row['sample_key'] in done:continue
            previous_scope=int(storage().name)
            rec=process(row,model,processor,progress,storage())
            promote(scope,progress)
            done.add(row['sample_key']);elapsed.append(rec['cost']['total_seconds'])
            counts=collections.Counter(r['dataset'] for r in rows if r['sample_key'] in done)
            progress(stage='query_complete',completed=len(done),counts=dict(counts),eta_seconds=(scope-len(done))*sum(elapsed)/len(elapsed))
            # First actual result makes the running viewer reviewable; full
            # pilot gate is still required before proceeding to the next scope.
            if len(done)==1:postprocess(300,partial=True)
            elif len(done)%25==0 or previous_scope!=scope:
                refresh_report(allow_partial=True)
        progress(stage='reporting_and_validating');postprocess(scope)
        progress(stage=f'pilot{scope}_complete',completed=scope,eta_seconds=0)
    from artifact_store import register
    register('greedy_attribution.pilot900','analyses',storage()/'queries','Full-prefill greedy gradients and bounded directional mode-branch screening','greedy_attribution_runtime.py',str(COHORT),'complete')
    progress(state='complete',stage='complete',completed=900);progress.stop.set()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--preflight',action='store_true');parser.add_argument('--smoke-index',type=int,default=0);args=parser.parse_args()
    storage().mkdir(parents=True,exist_ok=True)
    try:
        with (BASE/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
    except BaseException:
        if RUN_PROGRESS is not None:RUN_PROGRESS.stop.set()
        state=dict(RUN_PROGRESS.state) if RUN_PROGRESS is not None else {}
        state.update(state='failed',stage='failed',error=traceback.format_exc(),timestamp=stamp())
        write(storage()/'metadata/status.json',state)
        write(ROOT/'artifacts/common/registry/greedy_attribution.live.json',dict(canonical=str(storage()),
            status=str(storage()/'metadata/status.json'),log=str(storage()/'logs/run.log'),**state))
        raise
