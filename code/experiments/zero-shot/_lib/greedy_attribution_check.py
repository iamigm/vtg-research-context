"""Validate original sources, gradient causality, reports and real browser views."""
from __future__ import annotations
import argparse, collections, gzip, json, math, os, sys
from pathlib import Path
import numpy as np
from greedy_attribution_core import BASE, COHORT, CONFIG, GROUPS, MODEL, RESULTS, ROOT, SOURCE, digest, read, sha, stamp, storage, write

COHORT_HASHES = {
    300:'b4232e420a9f9f1a5a4f07fc8f84dde2f372be29935f6dc1dc152a3a9e3d0fdf',
    600:'d7fb4b7cc1c28842be0fa81e9b277c688ccfddf862371a3253a1000db25e4752',
    900:'e92620e9b56da0ca62af58b2da07ab2fb3fa5bac58ab20aa880b709426124db7',
}

class ValidationError(ValueError): pass

def require(condition,message):
    if not condition: raise ValidationError(message)

def close(actual,expected,label,atol=1e-6,rtol=2e-5):
    require(math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),abs_tol=atol,rel_tol=rtol),
            f'{label}: {actual} != {expected}')

def finite(value,label='record'):
    if isinstance(value,float): require(math.isfinite(value),f'{label}: nonfinite number')
    elif isinstance(value,dict):
        for key,item in value.items(): finite(item,f'{label}.{key}')
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value): finite(item,f'{label}[{i}]')

def array(value,shape,label,nonnegative=False):
    data=np.asarray(value)
    require(data.shape==tuple(shape),f'{label}: shape {data.shape}, expected {tuple(shape)}')
    require(data.dtype.kind in 'iuf' and bool(np.isfinite(data).all()),f'{label}: nonnumeric/nonfinite data')
    if nonnegative: require(bool((data>=0).all()),f'{label}: negative norm')
    return data

def within(root,relative):
    path=(root/relative).resolve()
    require(path.is_relative_to(root.resolve()),f'Artifact path escapes store: {relative}')
    return path

# Both paths sum the same rounded FP32 products, in different reduction orders.
# Bounds use the pinned CUDA Reduce.cuh rather than gamma_N (N can exceed 1/u).
REDUCE_HEADER_SHA256='ce368277eee91fdf0ea30522949ad7094b914cae8a12aab78304095444013789'
PIXEL_WIDTH=3*16*16

def gamma(depth,unit=2.**-24):
    require(0<=depth*unit<1,'Floating-point error depth is outside gamma domain')
    return depth*unit/(1-depth*unit)

def scalar_reduction_depth(size):
    """Upper bound for a fresh contiguous float32 CUDA scalar sum in torch 2.6.

    Input is 4-aligned and a multiple of 768. A 512-thread block has four
    accumulators, three combines and nine tree additions. Global reduction
    uses C CTAs in [ceil(v/256),ceil(v/16)], v=ceil(N/512), then ceil(C/512)
    additions and another nine-level tree. Independent extrema over C are
    conservative and require no assumption about the GPU's SM count.
    """
    require(size>0 and size%PIXEL_WIDTH==0 and size<2**31,'Unexpected pixel reduction size')
    ceildiv=lambda a,b:(a+b-1)//b
    if size<2048: return size-1
    values=ceildiv(size,512)
    if values<256: return ceildiv(size,4*512)+3+9
    low=ceildiv(values,256); high=ceildiv(values,16)
    return ceildiv(size,4*512*low)+3+9+ceildiv(high,512)+9

def pixel_reduction_check(scalar,norms,dots,label):
    """Compare summaries using a conservative forward-error bound, not equality.

    Saved patch norms bound sum(abs(g*x)) by Cauchy with |normalized x|<=1.
    Norm underestimation from FP32 square/sum/sqrt is reversed first. Patch
    reductions use the arbitrary-order depth 767, independent of dispatch.
    min-normal additive terms also cover subnormal flush-to-zero; host sums
    use an FP64 bound. No per-pixel values are reconstructed or modified.
    """
    u=2.**-24; tiny=float(np.finfo(np.float32).tiny); eps64=float(np.finfo(np.float64).eps)
    norms=np.asarray(norms,dtype=np.float64).ravel(); dots=np.asarray(dots,dtype=np.float64).ravel()
    require(norms.size==dots.size and norms.size>0 and bool(np.isfinite(norms).all())
            and bool(np.isfinite(dots).all()) and bool((norms>=0).all()) and math.isfinite(scalar),f'{label}: invalid pixel summaries')
    size=int(norms.size)*PIXEL_WIDTH; depth=scalar_reduction_depth(size)
    patch_depth=PIXEL_WIDTH-1; norm_gamma=gamma(PIXEL_WIDTH)
    # There are at most 2M square/add operations, each with additive error tiny.
    norm_underflow=2*PIXEL_WIDTH*tiny/(1-PIXEL_WIDTH*u)
    norm_upper=np.sqrt((((norms+tiny)/(1-u))**2+norm_underflow)/(1-norm_gamma))
    product_absolute_bound=(1+u)*math.sqrt(PIXEL_WIDTH)*math.fsum(norm_upper.tolist())+size*tiny
    product_absolute_bound=float(np.nextafter(product_absolute_bound*(1+32*eps64),np.inf))
    fp32_bound=(gamma(depth)+gamma(patch_depth))*product_absolute_bound
    fp32_bound+=size*tiny*(1/(1-depth*u)+1/(1-patch_depth*u))
    expected=math.fsum(dots.tolist()); absolute_patch_sum=math.fsum(np.abs(dots).tolist())
    host_bound=gamma(len(dots)+2,2.**-53)*(absolute_patch_sum+abs(scalar))
    bound=float(np.nextafter((fp32_bound+host_bound)*(1+32*eps64),np.inf))
    observed=abs(float(scalar)-expected)
    require(observed<=bound,f'{label}: pixel reduction difference {observed} exceeds FP32 bound {bound}')
    return dict(observed_abs_error=observed,bound=bound,ratio=observed/bound if bound else 0.,
                global_reduction_depth=depth,patch_reduction_depth=patch_depth,
                absolute_product_bound=product_absolute_bound,
                legacy_tolerance_exceeded=not math.isclose(float(scalar),expected,abs_tol=2e-5,rel_tol=2e-4))

def pixel_reduction_contract():
    header=ROOT/'.venv/lib/python3.10/site-packages/torch/include/ATen/native/cuda/Reduce.cuh'
    require(sha(header)==REDUCE_HEADER_SHA256,'Pinned CUDA scalar-reduction implementation changed')
    processor=MODEL/'video_preprocessor_config.json'; config=read(processor)
    require(config['patch_size']==16 and config['temporal_patch_size']==2
            and config['image_mean']==config['image_std']==[.5,.5,.5],'Normalized pixel bound assumptions changed')
    return dict(method='fp32_reduction_forward_error_bound',fp32_unit_roundoff=2.**-24,
        formula='abs(global_sum - fsum(patch_sums)) <= (gamma(global_depth)+gamma(767))*S_upper + underflow_bound + fp64_host_bound',
        absolute_product_bound='S_upper=(1+u)*sqrt(768)*sum(patch_gradient_norm_upper)+N*tiny32; norms enlarged for FP32 square/sum/sqrt error',
        patch_reduction_depth=767,normalized_pixel_absolute_bound=1,
        scalar_reduction='torch 2.6 CUDA, contiguous aligned FP32 product; 512 threads, input vector 4, min/max values per thread 16/256',
        reduce_header_sha256=REDUCE_HEADER_SHA256,processor_sha256=sha(processor),
        limitation='Conservative summary-only Cauchy bound; passing is consistency within roundoff bounds, not exact equality. Per-pixel gradients were not saved, so the actual absolute product sum is unavailable.')

def validate_context(context,original,ids):
    prompt=len(original['input_ids']); expected=original['input_ids']+ids[:-1]; length=len(expected)
    require(context['ids']==expected,'Context differs from original prompt plus greedy prefix')
    groups=context['groups']
    require(len(context['tokens'])==length and len(groups)==length,'Context array length mismatch')
    require(''.join(context['tokens'][:prompt])==original['expanded_prompt'],'Context labels do not reconstruct the saved prompt')
    require(all(g in GROUPS for g in groups),'Unknown context group')
    require(all(g!='generated' for g in groups[:prompt]) and all(g=='generated' for g in groups[prompt:]),'Generated group overlap/gap')
    visual=[i for i,g in enumerate(groups) if g=='visual']
    require(visual==context['visual_positions']==[i for i,x in enumerate(expected) if x==151656],'Visual group coverage mismatch')
    t,h,w=original['grid']; require(h%2==w%2==0 and len(visual)==t*h*w//4,'Visual grid mismatch')
    mapping=context['visual_map']; require([m['position'] for m in mapping]==visual,'Visual map order mismatch')
    for j,entry in enumerate(mapping):
        frame,spatial=divmod(j,(h//2)*(w//2))
        require((entry['frame_pair'],entry['row'],entry['col'])==(frame,spatial//(w//2),spatial%(w//2)),'Visual coordinates mismatch')
        times=original['plan']['source_pts'][2*frame:2*frame+2]
        close(entry['timestamp'],sum(times)/len(times),'Visual timestamp')
    return prompt,length,groups

def validate_target(target,payload,prompt,length,groups,grid,count,pixel_audits=None):
    ident=target['id']; step=target['step']; require(isinstance(step,int) and 0<=step<count,f'{ident}: invalid step')
    for key in ('id','kind','step','alternative','score','groups','data_file'):
        require(payload[key]==target[key],f'{ident}: payload metadata differs: {key}')
    require(math.isfinite(target['score']),f'{ident}: nonfinite objective')
    norms=array(payload['context']['grad_norm'],(length,),f'{ident}.context.norm',True)
    dots=array(payload['context']['activation_dot_gradient'],(length,),f'{ident}.context.signed')
    end=prompt+step; future=float(np.max(norms[end:])) if end<length else 0.
    require(future<1e-6,f'{ident}: future gradient leakage {future}')
    close(target['causality_max_future_gradient'],future,f'{ident}.future',atol=1e-8)
    require(not len(dots[end:]) or float(np.max(np.abs(dots[end:])))<1e-5,f'{ident}: future signed attribution leakage')
    embedding=payload['embedding']; require(embedding is not None,f'{ident}: text embedding gradients missing')
    for name in ('gradient_norm_l2','activation_dot_gradient'):
        values=array(embedding[name],(length,),f'{ident}.embedding.{name}',name=='gradient_norm_l2')
        require(not len(values[end:]) or float(np.max(np.abs(values[end:])))<1e-5,f'{ident}: future embedding gradient')
    rows=target['groups']; require(len(rows)==len(GROUPS) and {r['group'] for r in rows}==set(GROUPS),f'{ident}: incomplete group partition')
    for row in rows:
        positions=np.asarray([i for i,g in enumerate(groups[:end]) if g==row['group']],dtype=np.int64)
        require(row['count']==len(positions),f'{ident}: prior group count mismatch')
        close(row['signed_sum'],float(np.sum(dots[positions],dtype=np.float64)),f'{ident}.signed_sum')
        close(row['absolute_sum'],float(np.sum(np.abs(dots[positions]),dtype=np.float64)),f'{ident}.absolute_sum')
        close(row['grad_l2'],float(np.linalg.norm(norms[positions].astype(np.float64))),f'{ident}.grad_l2')
    t,h,w=grid
    for field,shape in [('frames',(t,h//2,w//2)),('pixel_frames',(2*t,h,w))]:
        for metric in ('gradient_norm','activation_dot_gradient'):
            array(payload[field][metric],shape,f'{ident}.{field}.{metric}',metric=='gradient_norm')
    visual=np.asarray([i for i,g in enumerate(groups) if g=='visual'],dtype=np.int64)
    require(np.array_equal(np.asarray(payload['frames']['gradient_norm']).ravel(),norms[visual]),f'{ident}: visual map norm differs')
    require(np.array_equal(np.asarray(payload['frames']['activation_dot_gradient']).ravel(),dots[visual]),f'{ident}: visual map sign differs')
    pixel_norm=np.asarray(payload['pixel_frames']['gradient_norm'],dtype=np.float64)
    pixel_dot=np.asarray(payload['pixel_frames']['activation_dot_gradient'],dtype=np.float64)
    close(target['pixel_gradient_l2'],float(np.linalg.norm(pixel_norm)),f'{ident}.pixel_l2',rtol=2e-4)
    pixel_audit=pixel_reduction_check(target['pixel_signed_sum'],pixel_norm,pixel_dot,f'{ident}.pixel_signed_sum')
    if pixel_audits is not None: pixel_audits.append(dict(target=ident,**pixel_audit))
    require(payload['predictor_positions']==list(range(prompt-1,length)),f'{ident}: predictor position shift')
    layers=payload['layers']; require(set(layers)=={'attention','mlp','residual_input'},f'{ident}: decoder components missing')
    for kind,entries in layers.items():
        require([row['layer'] for row in entries]==list(range(36)),f'{ident}: incomplete {kind} layers')
        for row in entries:
            layer_signed=array(row['activation_dot_gradient'],(count,),f'{ident}.{kind}.signed')
            layer_norm=array(row['gradient_norm_l2'],(count,),f'{ident}.{kind}.norm',True)
            require(not len(layer_norm[step+1:]) or float(np.max(layer_norm[step+1:]))<1e-6,f'{ident}: future decoder gradient')
            require(not len(layer_signed[step+1:]) or float(np.max(np.abs(layer_signed[step+1:])))<1e-5,f'{ident}: future decoder signed signal')
    message_data=payload['messages']; finite(message_data,f'{ident}.messages')
    require(target['message_validation']['passed'] is True and payload['message_validation']==target['message_validation'],f'{ident}: message gate missing or failed')
    selected=message_data['selected_layers']; require(selected==[0,17,35],f'{ident}: message layer coverage differs')
    recon={r['layer']:r for r in message_data['reconstruction']}; measured={r['layer']:r for r in message_data['layers']}
    require(set(recon)==set(measured)==set(selected),f'{ident}: reconstructed/message layer coverage differs')
    for layer in selected:
        require(recon[layer]['rms_relative_difference']<=.03,f'{ident}: reconstructed attention RMS error')
        heads=array(measured[layer]['head_score_sum'],(32,count),f'{ident}.message_head_scores')
        summed=heads.sum(axis=0); block=np.asarray(layers['attention'][layer]['activation_dot_gradient'])
        method=target['message_validation'].get('method')
        if method=='pre_projection_reconstruction_bound':
            actual_heads=array(measured[layer]['actual_attention_input_dot_gradient'],(32,count),f'{ident}.actual_pre_projection')
            reconstruction_bound=array(measured[layer]['reconstruction_dot_error_bound'],(32,count),f'{ident}.reconstruction_bound',True)
            roundoff_bound=array(measured[layer]['fp32_roundoff_error_bound'],(32,count),f'{ident}.roundoff_bound',True)
            actual=actual_heads.sum(axis=0); delta=np.abs(summed-actual)
            bound=(reconstruction_bound+roundoff_bound).sum(axis=0)
            cpu_roundoff=np.finfo(np.float64).eps*(heads.shape[0]+2)*(np.abs(heads).sum(axis=0)+np.abs(actual_heads).sum(axis=0)+bound)
            tolerance=np.nextafter(bound+cpu_roundoff,np.inf)
            require(target['message_validation']['after_projection_diagnostic_only'] is True,f'{ident}: after-projection interpretation changed')
            require(target['message_validation']['fp32_unit_roundoff']==2.**-24,f'{ident}: FP32 roundoff constant differs')
        else:
            require(method is None,f'{ident}: unknown message validation method {method}')
            delta=np.abs(summed-block); tolerance=1e-4+.05*np.maximum(np.abs(summed),np.abs(block))
        require(bool((delta<=tolerance).all()),f'{ident}: message sum exceeds its reconstruction/roundoff bound')
        for edge in measured[layer]['edges']:
            require(0<=edge['source_position']<=edge['receiver_position']<length,f'{ident}: noncausal message edge')
            require(edge['receiver_position'] in payload['predictor_positions'],f'{ident}: unexpected message receiver')
            require(0<=edge['head']<32 and 0<=edge['attention_weight']<=1.0001,f'{ident}: invalid message head/weight')
    return future

def validate_record(record,original,saved_input,trace,canonical,reference):
    key=original['sample']['sample_key']; finite(record,key)
    require(record['complete'] is True and record['sample_key']==key and record['sample']==original['sample'],f'{key}: sample/incomplete mismatch')
    require(record['config_hash']==digest(CONFIG),f'{key}: config hash differs')
    source=record['source']; require(source['record_sha256']==reference['sha256'],f'{key}: source checksum differs')
    require(Path(source['record']).resolve()==(SOURCE/'rollouts'/f'{key}.json').resolve(),f'{key}: source record path differs')
    trace_path=SOURCE/'token_traces'/key/'index.json'
    require(Path(source['trace']).resolve()==trace_path.resolve() and source['trace_sha256']==sha(trace_path),f'{key}: source trace checksum differs')
    require(trace['raw_record_sha256']==reference['sha256'],f'{key}: trace source mismatch')
    greedy=original['result']['candidates'][0]; ids=greedy['ids']; count=len(ids)
    require(greedy['index']==1 and greedy['branch'] is None and record['greedy']['ids']==ids,f'{key}: greedy tokens changed')
    require(len(record['greedy']['tokens'])==count and record['greedy']['text']==greedy['answer'],f'{key}: greedy text changed')
    prompt,length,groups=validate_context(record['context'],saved_input,ids)
    inp=record['input']; require(inp['prompt_length']==prompt and inp['total_length']==length,f'{key}: length mismatch')
    require(inp['hashes']==saved_input['hashes']==original['input_hashes']==trace['input_hashes'],f'{key}: input hash mismatch')
    require(inp['grid']==saved_input['grid'] and inp['hw']==saved_input['hw'],f'{key}: input geometry changed')
    require(inp['timestamps']==saved_input['plan']['source_pts'],f'{key}: timestamp changed')
    require([s['step'] for s in record['steps']]==list(range(count)),f'{key}: greedy step coverage mismatch')
    require([s['chosen']['id'] for s in record['steps']]==ids,f'{key}: chosen IDs mismatch')
    validation=record['validation']; require(validation['input_hashes_match'] is True and validation['causality_passed'] is True,f'{key}: replay/causality failed')
    require(validation['gradient_forward_max_logit_error']==0,f'{key}: gradient forward changed logits')
    require(validation['replay_max_probability_error']<=CONFIG['replay_probability_tolerance'],f'{key}: probability tolerance exceeded')
    arrays=record['arrays']; array_path=within(canonical,arrays['file']); require(sha(array_path)==arrays['sha256'],f'{key}: array checksum differs')
    with np.load(array_path,allow_pickle=False) as saved:
        tracked=CONFIG['candidate_top_k']; require(arrays['tracked_tokens']==tracked==30,f'{key}: top30 scope differs')
        token_ids=array(saved['token_ids'],(count,tracked),f'{key}.token_ids')
        expected_ids=np.asarray([[r['id'] for r in trace['nodes'][node]['top'][:tracked]] for node in trace['candidates']['1']['nodes']])
        require(np.array_equal(token_ids,expected_ids),f'{key}: frozen original top30 IDs changed')
        logits=array(saved['logits'],(count,tracked),f'{key}.logits')
        prob=array(saved['probabilities'],(count,tracked),f'{key}.probabilities',True)
        source_prob=array(saved['source_probabilities'],(count,tracked),f'{key}.source_probabilities',True)
        expected_prob=np.asarray([[r['probability'] for r in trace['nodes'][node]['top'][:tracked]] for node in trace['candidates']['1']['nodes']],dtype=np.float32)
        require(np.array_equal(source_prob,expected_prob),f'{key}: source top30 probabilities changed')
        close(validation['replay_max_probability_error'],float(np.max(np.abs(prob-source_prob))),f'{key}.replay_max')
        require(bool((prob<=1).all()) and bool((prob.sum(-1,dtype=np.float64)<=1.00003).all()),f'{key}: invalid unnormalized top30 probability')
        argmax_index=logits.argmax(-1); argmax=token_ids[np.arange(count),argmax_index]
        chosen_indices=np.asarray([list(token_ids[j]).index(token) for j,token in enumerate(ids)])
        mismatch=np.flatnonzero(argmax!=np.asarray(ids)).tolist()
        require(validation['greedy_argmax_match']==(not mismatch) and validation['greedy_near_tie_steps']==mismatch,f'{key}: replay top1 audit inconsistent')
        require(all(logits[j,argmax_index[j]]-logits[j,chosen_indices[j]]<=.125 for j in mismatch),f'{key}: substantive greedy top1 mismatch')
        directions=saved['directions'].tolist(); require(directions==arrays['directions'],f'{key}: screening direction mismatch')
        for field,shape in [('slopes',(len(directions),count,tracked)),('changes',(len(directions),2,count,tracked)),('screen_gaps',(count,tracked))]:
            data=array(saved[field],shape,f'{key}.{field}'); del data
        for j,step in enumerate(record['steps']):
            close(step['chosen']['probability'],prob[j,chosen_indices[j]],f'{key}.chosen_probability')
            close(step['top30_mass'],float(prob[j].sum(dtype=np.float64)),f'{key}.top30_mass')
            close(step['omitted_mass'],max(0.,1-float(prob[j].sum(dtype=np.float64))),f'{key}.omitted_mass')
            require(step['tracked_tokens']==30 and {r['id'] for r in step['top']}==set(token_ids[j]),f'{key}: top30 step coverage differs')
            node=trace['nodes'][trace['candidates']['1']['nodes'][j]]
            require(node['prefix_ids']==ids[:j],f'{key}: source prefix mismatch')
            expected=next(r['probability'] for r in node['observed'] if r['id']==ids[j])
            require(abs(step['chosen']['probability']-expected)<=CONFIG['replay_probability_tolerance'],f'{key}: chosen replay error')
            alternatives=step['candidates']; require(len({a['id'] for a in alternatives})==len(alternatives),f'{key}: repeated alternative')
            for alternative in alternatives:
                alt=alternative['id']; require(alt!=ids[j] and alt in token_ids[j],f'{key}: alternative outside original top30')
                column=list(token_ids[j]).index(alt)
                close(alternative['probability'],prob[j,column],f'{key}.alternative_probability')
                close(alternative['margin'],logits[j,chosen_indices[j]]-logits[j,column],f'{key}.alternative_margin')
        del logits,prob
    targets=record['targets']; require(len({t['id'] for t in targets})==len(targets),f'{key}: duplicate target')
    chosen={t['step']:t for t in targets if t['kind']=='chosen_log_probability'}
    require(len(chosen)==count and sum(t['kind']=='chosen_log_probability' for t in targets)==count and sorted(chosen)==list(range(count)),f'{key}: chosen target coverage including EOS differs')
    require(sum(t['kind']=='logit_margin' for t in targets)<=CONFIG['detailed_branch_targets'],f'{key}: detailed margin budget exceeded')
    future=0.; pixel_audits=[]
    for target in targets:
        kind=target['kind']; require(kind in {'chosen_log_probability','start_log_probability','end_log_probability','logit_margin'},f'{key}: unknown objective')
        if kind=='logit_margin':
            branch=next((a for a in record['steps'][target['step']]['candidates'] if a['id']==target['alternative']),None)
            require(branch is not None,f'{key}: margin branch missing'); close(target['score'],branch['margin'],f'{key}.margin')
        elif kind!='chosen_log_probability':
            selected=target['steps']; require(selected and selected==sorted(set(selected)) and all(i in chosen for i in selected),f'{key}: endpoint steps invalid')
            require(target['step']==max(selected),f'{key}: endpoint causal boundary mismatch')
            close(target['score'],math.fsum(chosen[i]['score'] for i in selected),f'{key}.endpoint',atol=2e-4)
        data_path=within(canonical,target['data_file']); require(sha(data_path)==target['sha256'] and data_path.stat().st_size==target['bytes'],f'{key}: target checksum differs')
        with gzip.open(data_path,'rt') as stream: payload=json.load(stream)
        future=max(future,validate_target(target,payload,prompt,length,groups,inp['grid'],count,pixel_audits)); del payload
    if greedy.get('parsed'):
        require(all(sum(t['kind']==kind for t in targets)==1 for kind in ('start_log_probability','end_log_probability')),f'{key}: endpoint aggregate missing')
    return dict(sample_key=key,steps=count,targets=len(targets),future_max=future,pixel_reductions=pixel_audits)

def validate_scope(scope,allow_partial=False):
    require(scope in CONFIG['stages'],'Unsupported scope'); canonical=storage(); cohort=read(COHORT)
    pixel_contract=pixel_reduction_contract()
    require(read(canonical/'metadata/config.json')==CONFIG,'Frozen config differs')
    require(read(canonical/'metadata/cohort.json')==cohort[:int(canonical.name)],'Active cohort changed')
    for n,expected in COHORT_HASHES.items():
        path=ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json'
        require(sha(path)==expected and read(path)==cohort[:n],f'Frozen cohort{n} hash/prefix changed')
    require(read(SOURCE/'metadata/cohort.json')==cohort,'Original source cohort differs')
    refs=read(canonical/'metadata/source_manifest.json'); require(isinstance(refs,list) and len(refs)==900,'Source reference coverage differs')
    original_refs=read(SOURCE/'metadata/source_records.json')
    require([(r['path'],r['sha256']) for r in refs]==[(r['path'],r['sha256']) for r in original_refs],'Frozen source references differ')
    provenance=read(canonical/'metadata/provenance.json')
    require(provenance['config_hash']==digest(CONFIG) and provenance['cohort_sha256']==COHORT_HASHES[900],'Provenance mismatch')
    completed=[]; missing=[]
    for row,ref in zip(cohort[:scope],refs[:scope]):
        key=row['sample_key']; path=SOURCE/'rollouts'/f'{key}.json'
        require(ref['sample_key']==key and sha(path)==ref['sha256'],f'{key}: source checksum changed')
        query=canonical/'queries'/f'{key}.json'
        if not query.exists(): missing.append(key); continue
        record=read(query)
        if not record.get('complete'): missing.append(key); continue
        original=read(path); require(original['sample']==row,f'{key}: sample/cohort differs')
        completed.append(validate_record(record,original,read(SOURCE/'inputs'/f'{key}.json'),read(SOURCE/'token_traces'/key/'index.json'),canonical,ref))
        if len(completed)%25==0:
            print(json.dumps(dict(integrity_progress=len(completed),scope=scope,sample_key=key,timestamp=stamp())),file=sys.stderr,flush=True)
    require(allow_partial or not missing,f'pilot{scope}: {len(missing)} incomplete queries')
    counts=dict(completed=len(completed),steps=sum(r['steps'] for r in completed),targets=sum(r['targets'] for r in completed),
                unique_videos=len({r['video_path'] for r in cohort[:scope] if r['sample_key'] not in missing}))
    pixel_audits=[dict(sample_key=row['sample_key'],**audit) for row in completed for audit in row['pixel_reductions']]
    pixel_contract.update(checked_targets=len(pixel_audits),nonzero_difference_count=sum(r['observed_abs_error']>0 for r in pixel_audits),
        legacy_tolerance_exceeded_count=sum(r['legacy_tolerance_exceeded'] for r in pixel_audits),
        max_observed_abs_error=max(pixel_audits,key=lambda r:r['observed_abs_error'],default=None),
        max_bound=max(pixel_audits,key=lambda r:r['bound'],default=None),
        max_ratio=max(pixel_audits,key=lambda r:r['ratio'],default=None),
        legacy_tolerance_exceeded=[r for r in pixel_audits if r['legacy_tolerance_exceeded']])
    report=dict(passed=True,scope_queries=scope,complete=not missing,allow_partial=allow_partial,canonical=str(canonical),**counts,
                missing_queries=missing,source_hashes_verified=True,input_hashes_verified=True,all_greedy_tokens_including_eos=True,
                exclusive_context_groups=True,future_gradient_max=max((r['future_max'] for r in completed),default=None),
                pixel_signed_sum_validation=pixel_contract,timestamp=stamp())
    write(storage()/'validation'/f'integrity_pilot{scope}.json',report); return report

def validate_catalog(scope,allow_partial=False):
    canonical=storage(); cohort=read(COHORT)[:scope]; catalog=read(BASE/str(scope)/'web/catalog.json')
    require(catalog['scope']==scope and [s['key'] for s in catalog['samples']]==[r['sample_key'] for r in cohort],'Catalog cohort coverage/order differs')
    completed=[]; step_count=target_count=0; videos=set()
    for sample,row in zip(catalog['samples'],cohort):
        path=canonical/'queries'/f'{sample["key"]}.json'; record=read(path) if path.exists() else None
        actual=bool(record and record.get('complete'))
        # A running worker can finish another query after this catalog snapshot
        # was published. Its pending row is valid until the next report update.
        require(not sample['complete'] or actual,'Catalog advertises a missing/incomplete query')
        if not allow_partial: require(sample['complete']==actual,'Catalog completion state differs')
        if sample['complete']:
            completed.append(sample); step_count+=len(record['steps']); target_count+=len(record['targets']); videos.add(row['video_path'])
    expected=dict(completed=len(completed),steps=step_count,targets=target_count,
                  unique_videos=len({r['video_path'] for r in cohort}),completed_unique_videos=len(videos))
    for key,value in expected.items(): require(catalog['summary'][key]==value,f'Catalog {key} != actual unique-query aggregate')
    require((RESULTS/'pilot'/str(scope)/'REPORT.md').exists(),'MD report missing')
    return catalog,completed

def check_viewer(scope=300,allow_partial=False):
    catalog,completed=validate_catalog(scope,allow_partial); require(completed,'Browser validation requires a real completed query')
    require(allow_partial or len(completed)==scope,'Complete scope required for browser check')
    shared=ROOT.parent/'_shared/timelens'; sys.path.insert(0,str(shared/'browser_deps'))
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
    os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
    from playwright.sync_api import sync_playwright
    sample=completed[0]; record=read(storage()/'queries'/f'{sample["key"]}.json'); errors=[]
    margin_indices=[i for i,target in enumerate(record['targets']) if target['kind']=='logit_margin']
    require(margin_indices,'Real query has no measured branch to exercise')
    advanced=('group-details','position-details','token-details','layer-details','message-details')
    base='http://127.0.0.1:8780'; detail=f'{base}/zero-shot/results/rollout/greedy_attribution/pilot/{scope}/detail.html'
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1600,height=1100)); page.on('pageerror',lambda e:errors.append(str(e)))
        def selected_state():
            return page.evaluate("Object.fromEntries(new URLSearchParams(location.hash.slice(1)))")
        def wait_target(index):
            page.wait_for_function("""index => document.getElementById('target').value===String(index)
                && document.getElementById('attribution-panel').dataset.targetReady==='true'
                && !document.getElementById('frame-panel').hidden
                && document.querySelectorAll('#spatial-map .spatial-cell').length>0
                && new URLSearchParams(location.hash.slice(1)).get('target')===String(index)""",arg=index,timeout=60000)
        def assert_target(index,key=sample['key'],target_record=record):
            state=selected_state()
            require(state.get('sample')==key and state.get('step')==str(target_record['targets'][index]['step'])
                    and state.get('target')==str(index),'Sample/step/target state changed unexpectedly')
            require(page.locator('#target').input_value()==str(index),'Selected target control differs from deep link')
            require(page.locator('#attribution-panel').get_attribute('data-target-id')==target_record['targets'][index]['id'],
                    'Displayed payload differs from selected target')
        def toggle_details(identifier,opened):
            details=page.locator('#'+identifier)
            if details.evaluate('element=>element.open')!=opened:
                summary=page.locator('#'+identifier+' > summary'); summary.focus(); summary.press('Enter')
            require(details.evaluate('element=>element.open')==opened,f'{identifier}: keyboard disclosure failed')
        def assert_overlay_alignment():
            page.wait_for_function("""() => {
                const v=document.getElementById('overlay-video'),m=document.getElementById('spatial-map');
                const vr=v.getBoundingClientRect(),mr=m.getBoundingClientRect();
                if (!v.videoWidth || !vr.width || !mr.width) return false;
                const s=Math.min(vr.width/v.videoWidth,vr.height/v.videoHeight);
                const w=v.videoWidth*s,h=v.videoHeight*s;
                return Math.max(Math.abs(mr.width-w),Math.abs(mr.height-h),
                    Math.abs(mr.left-vr.left-(vr.width-w)/2),Math.abs(mr.top-vr.top-(vr.height-h)/2))<1.5;
            }""",timeout=10000)
            require(page.locator('#spatial-map').evaluate("m=>getComputedStyle(m).pointerEvents==='none'"),
                    'Heatmap blocks native video controls')

        page.goto(detail+'?sample='+sample['key'],wait_until='networkidle')
        page.wait_for_selector('#branch-cards button[data-branch-target]',timeout=60000)
        initial=int(page.locator('#target').input_value()); require(initial in margin_indices,'Default view did not select a measured branch')
        wait_target(initial)
        require(page.locator('#branch-cards button[data-branch-target]').count()==len(margin_indices),'Branch overview coverage differs')
        require(page.locator('#branch-cards button[aria-pressed="true"]').count()==1,'Branch selection lacks a unique accessible state')
        require(page.locator('#input-directions .direction-row').count()==6,'Input-direction overview missing source groups')
        legend=page.locator('#direction-legend').inner_text(); require('음수' in legend and '양수' in legend,'Signed direction legend relies on color alone')
        require(page.locator('#branch-narrative').inner_text().strip(),'Selected branch explanation empty')
        require(all(not page.locator('#'+identifier).evaluate('element=>element.open') for identifier in advanced),'Advanced details must begin collapsed')
        require(page.locator('#samples button[data-key]').count()<=10,'Query list not paginated')

        chosen=next((index for index in reversed(margin_indices) if index!=initial),initial)
        card=page.locator(f'#branch-cards button[data-branch-target="{chosen}"]'); card.focus(); card.press('Enter')
        wait_target(chosen); assert_target(chosen)
        require(page.locator(f'#branch-cards button[data-branch-target="{chosen}"]').get_attribute('aria-pressed')=='true','Keyboard branch selection not announced')
        prefix=page.locator('#prefix').text_content(); url=page.url
        page.reload(wait_until='networkidle'); wait_target(chosen); assert_target(chosen)
        require(page.locator('#prefix').text_content()==prefix,'Deep link lost the selected prefix')

        toggle_details('token-details',True)
        require(page.locator('#sentence button[data-step]').count()==len(record['steps']),'Viewer greedy token coverage differs')
        page.locator('#sentence button[data-step]').last.click()
        eos_index=next(i for i,target in enumerate(record['targets']) if target['kind']=='chosen_log_probability' and target['step']==len(record['steps'])-1)
        wait_target(eos_index); assert_target(eos_index)
        require(page.locator('#branches tr').count()==30,'Fixed top30 detail coverage differs')
        page.locator(f'#branch-cards button[data-branch-target="{chosen}"]').click(); wait_target(chosen); assert_target(chosen)
        toggle_details('group-details',True); toggle_details('position-details',True)
        page.wait_for_selector('#positions .position',timeout=60000)
        for metric in ('grad_l2','activation_gradient'):
            page.select_option('#metric',metric); wait_target(chosen); assert_target(chosen)
            require(page.locator('#positions .position').count()>0,'Metric switch lost positions')
        page.select_option('#source-type','query')
        require(page.locator('#positions .position').count()>0,'Query-only context filter empty')
        query_positions={i for i,group in enumerate(record['context']['groups']) if group=='query'}
        shown=page.locator('#positions .position').evaluate_all("els=>els.map(element=>Number(element.dataset.position))")
        require(set(shown)<=query_positions,'Context group filter mixed sources')
        page.select_option('#source-type','')
        require(page.locator('#spatial-map .spatial-cell').count()>0,'Spatial map empty')
        for space in ('pixels','visual'):
            page.select_option('#frame-space',space); assert_target(chosen)
            require(page.locator('#spatial-map .spatial-cell').count()>0,f'{space} map empty')
        toggle_details('layer-details',True)
        require(page.locator('#layers .layer-row').count()==37,'Decoder layer table incomplete')
        receivers=page.locator('#receiver option')
        if receivers.count()>1:
            page.select_option('#receiver',receivers.first.get_attribute('value'))
            require(page.locator('#layers .layer-row').count()==37,'Receiver switch lost layer table'); assert_target(chosen)
            page.select_option('#receiver',receivers.last.get_attribute('value'))
        toggle_details('message-details',True)
        require(page.locator('#messages').is_visible(),'Signed message detail missing')
        video=page.locator('#video'); src=video.get_attribute('src'); require(src,'Video source missing')
        response=page.request.get(src,headers={'Range':'bytes=0-127'}); require(response.status==206 and len(response.body())==128,'Video Range failed')
        page.wait_for_function("document.getElementById('video').readyState>=1",timeout=30000)
        overlay=page.locator('#overlay-video')
        require(overlay.get_attribute('src')==src and overlay.evaluate('v=>v.controls && v.muted'),
                'Overlay does not show the same video with native controls')
        page.wait_for_function("document.getElementById('overlay-video').readyState>=2",timeout=30000)
        assert_overlay_alignment()
        require(page.locator('#overlay-transparency').input_value()=='40','Heatmap default transparency differs')
        close(float(page.locator('#spatial-map').evaluate('m=>getComputedStyle(m).opacity')),.6,'Heatmap default opacity')
        require('투명도' in page.locator('label[for="overlay-transparency"]').inner_text(),'Heatmap transparency label differs')
        # The selected real payload need not contain an exact zero. Exercise
        # that boundary using the served pure color function, without changing
        # records or exposing a production-only testing hook.
        script_url=page.locator('script[src]').last.evaluate('script=>script.src')
        served_script=page.request.get(script_url).text()
        color_body=served_script.partition('  function overlayColor(')[2].partition('\n  }\n')[0]
        require(bool(color_body),'Served opaque color function missing')
        color_function='function overlayColor('+color_body+'\n  }'
        require(page.evaluate(r"""source=>{
            const color=new Function('return ('+source+')')();
            return [false,true].every(signed=>[0,1].every(maximum=>{
                const match=color(0,maximum,signed).match(/^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/);
                return match && match.slice(1).every(channel=>Number(channel)>=0 && Number(channel)<=255);
            }));
        }""",color_function),'Zero-valued heatmap colors are not opaque RGB')
        with gzip.open(within(storage(),record['targets'][chosen]['data_file']),'rt') as stream: overlay_payload=json.load(stream)
        timestamps=record['input']['timestamps']; t,h,w=record['input']['grid']
        for space,field,cells,frames in [('pixels','pixel_frames',h*w,2*t),('visual','frames',(h//2)*(w//2),t)]:
            page.select_option('#frame-space',space); assert_target(chosen)
            require(page.locator('#frame option').count()==frames,'Overlay frame coverage differs')
            require(page.locator('#spatial-map .spatial-cell').count()==cells,'Overlay spatial geometry differs')
            for metric,source_metric in [('grad_l2','gradient_norm'),('activation_gradient','activation_dot_gradient')]:
                page.select_option('#metric',metric); wait_target(chosen)
                maximum=float(np.max(np.abs(np.asarray(overlay_payload[field][source_metric],dtype=np.float64))))
                for frame_index in (0,1,frames-1):
                    page.select_option('#frame',str(frame_index))
                    source_index=frame_index if space=='pixels' else 2*frame_index
                    card=page.locator(f'#frame-timeline button[data-frame="{frame_index}"]')
                    require(card.locator('.frame-index').inner_text()==str(frame_index),'Frame card index differs')
                    actual_times=timestamps[source_index:source_index+(1 if space=='pixels' else 2)]
                    require(card.locator('.frame-time').inner_text()==' / '.join(f'{value:.3f}' for value in actual_times)+' s',
                            f'{space}: frame card does not show actual source timestamps to three decimals')
                    index_box=card.locator('.frame-index').bounding_box(); time_box=card.locator('.frame-time').bounding_box()
                    require(index_box and time_box and index_box['y']+index_box['height']<=time_box['y']+.5,
                            'Frame card timestamp is not below its index')
                    page.wait_for_function("i=>Number(document.getElementById('overlay-stage').dataset.frameIndex)===i",arg=frame_index)
                    page.wait_for_function("time=>['video','overlay-video'].every(id=>Math.abs(document.getElementById(id).currentTime-time)<.05 && !document.getElementById(id).seeking)",arg=timestamps[source_index],timeout=30000)
                    require(int(page.locator('#overlay-stage').get_attribute('data-sample-index'))==source_index,
                            'Manual frame selection lost its original input slot (including repeated PTS)')
                    close(float(page.locator('#spatial-map').get_attribute('data-color-maximum')),maximum,
                          'All-frame fixed color maximum',atol=1e-12,rtol=1e-12)
                    require(page.locator('#spatial-map .spatial-cell').evaluate_all("""cells=>cells.length>0 && cells.every(cell=>{
                        const style=getComputedStyle(cell),match=style.backgroundColor.match(/^rgba?\((.*)\)$/);
                        if (!match || Number(style.opacity)!==1) return false;
                        const channels=match[1].split(',').map(Number);
                        return channels.length===3 || (channels.length===4 && channels[3]===1);
                    })"""),f'{space}/{metric}: heatmap cell color is not fully opaque')
                    require(video.evaluate('v=>v.paused') and overlay.evaluate('v=>v.paused'),'Manual frame seek did not pause both players')
                if space=='visual':
                    second_source=2*(frames-1)+1
                    page.locator(f'#frame-details button[data-source-index="{second_source}"]').click()
                    page.wait_for_function("i=>Number(document.getElementById('overlay-stage').dataset.sampleIndex)===i",arg=second_source)
                    page.wait_for_function("time=>Math.abs(document.getElementById('overlay-video').currentTime-time)<.05 && !document.getElementById('overlay-video').seeking",arg=timestamps[second_source])
                    require(int(page.locator('#overlay-stage').get_attribute('data-frame-index'))==frames-1,
                            'Second source button changed its visual pair')
            # Exercise real media playback and natural timeupdate events.
            page.select_option('#frame','0')
            page.wait_for_function("document.getElementById('overlay-video').currentTime<.05 && !document.getElementById('overlay-video').seeking")
            overlay.evaluate('async v=>{v.playbackRate=1;await v.play()}')
            page.wait_for_function("time=>document.getElementById('overlay-video').currentTime>time",arg=timestamps[min(2,len(timestamps)-1)]+.08,timeout=15000)
            overlay.evaluate('v=>v.pause()')
            page.wait_for_function("""({times,space})=>{
                const v=document.getElementById('overlay-video'),stage=document.getElementById('overlay-stage');
                let k=0; for(let i=1;i<times.length;i++) if(Math.abs(times[i]-v.currentTime)<Math.abs(times[k]-v.currentTime)) k=i;
                return Number(stage.dataset.sampleIndex)===k && Number(stage.dataset.frameIndex)===(space==='pixels'?k:Math.floor(k/2));
            }""",arg=dict(times=timestamps,space=space),timeout=10000)
            require(video.evaluate('v=>v.paused'),'Overlay playback left the overview player playing')
            assert_target(chosen); assert_overlay_alignment()
        del overlay_payload
        page.locator('#overlay-visible').uncheck(); require(page.locator('#spatial-map').is_hidden(),'Heatmap visibility toggle failed')
        page.locator('#overlay-visible').check(); require(page.locator('#spatial-map').is_visible(),'Heatmap visibility restore failed')
        transparency=page.locator('#overlay-transparency'); transparency.focus(); transparency.press('Home')
        close(float(page.locator('#spatial-map').evaluate('m=>getComputedStyle(m).opacity')),1.,'Heatmap zero transparency')
        transparency.press('End')
        close(float(page.locator('#spatial-map').evaluate('m=>getComputedStyle(m).opacity')),0.,'Heatmap full transparency')
        transparency.evaluate("e=>{e.value='40';e.dispatchEvent(new Event('input',{bubbles:true}))}")
        close(float(page.locator('#spatial-map').evaluate('m=>getComputedStyle(m).opacity')),.6,'Heatmap restored default transparency')
        require(page.locator('#overlay-transparency-value').inner_text()=='40%','Heatmap transparency label differs')
        # The overview player must also advance the same cached overlay map.
        page.select_option('#frame-space','pixels'); page.select_option('#frame','0')
        page.wait_for_function("document.getElementById('video').currentTime<.05 && !document.getElementById('video').seeking")
        video.evaluate('async v=>{v.muted=true;await v.play()}')
        page.wait_for_function("document.getElementById('video').currentTime>.6",timeout=15000)
        video.evaluate('v=>v.pause()')
        page.wait_for_function("Math.abs(document.getElementById('video').currentTime-document.getElementById('overlay-video').currentTime)<.05")
        require(overlay.evaluate('v=>v.paused'),'Overview playback left the overlay player playing')

        if len(completed)>1:
            second=completed[1]; page.locator('#filter').fill(second['key'])
            page.locator(f'#samples button[data-key="{second["key"]}"]').click()
            page.wait_for_function("key=>document.getElementById('query-meta').textContent.includes(key)&&!document.getElementById('workspace').classList.contains('loading')",arg=second['key'])
            second_record=read(storage()/'queries'/f'{second["key"]}.json'); second_target=int(page.locator('#target').input_value())
            wait_target(second_target); assert_target(second_target,second['key'],second_record)
            page.reload(wait_until='networkidle'); wait_target(second_target); assert_target(second_target,second['key'],second_record)
        page.goto(url,wait_until='networkidle'); wait_target(chosen); assert_target(chosen)
        page.locator('#filter').fill('')
        page.locator(f'#branch-cards button[data-branch-target="{initial}"]').click()
        wait_target(initial); assert_target(initial)
        for identifier in advanced: toggle_details(identifier,False)
        page.locator('#branch-panel').scroll_into_view_if_needed()
        result=RESULTS/'pilot'/str(scope); page.screenshot(path=str(result/'preview_desktop.png'),full_page=True)
        for width in (390,768,1100):
            page.set_viewport_size(dict(width=width,height=844)); require(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),f'Overflow at {width}px')
            require(page.locator('#branch-cards button[data-branch-target]').count()==len(margin_indices),'Responsive view lost branch cards')
            assert_overlay_alignment()
        page.set_viewport_size(dict(width=390,height=844)); page.screenshot(path=str(result/'preview_mobile.png'),full_page=True)
        page.goto(f'{base}/zero-shot/results/rollout/greedy_attribution/viewer.html?view=pilot%2F{scope}%2Fdetail.html',wait_until='networkidle'); page.wait_for_selector('#analysis-tabs button')
        labels=page.locator('#analysis-tabs').inner_text(); require('결과' in labels and '보고서' in labels,'Common tabs missing')
        family_scopes=[]
        for n in CONFIG['stages']:
            if not (RESULTS/'pilot'/str(n)/'detail.html').exists(): continue
            family_scopes.append(n)
            page.locator('#view-browser > summary').click()
            page.locator(f'#view-tree-items button[data-view="pilot/{n}/detail.html"]').click()
            page.frame_locator('#detail-frame').locator('#branch-cards button[data-branch-target]').first.wait_for(timeout=60000)
            require(f'pilot/{n}/detail.html' in page.url.replace('%2F','/'),'Scope did not switch')
            page.locator('#tab-report').click(); page.wait_for_selector('#report h1')
            require(str(n) in page.locator('#report h1').inner_text(),'MD report scope differs')
            require(page.locator('#source-md').is_visible(),'MD source link missing on report tab')
            page.locator('#tab-result').click(); page.frame_locator('#detail-frame').locator('#branch-cards button[data-branch-target]').first.wait_for(timeout=60000)
            require(not page.locator('#source-md').is_visible(),'MD source link leaked into result tab')
        require(page.request.get(base+'/').status==200,'Unified index server did not respond')
        require(not errors,f'Browser errors: {errors}'); browser.close()
    report=dict(passed=True,scope_queries=scope,completed_queries=len(completed),allow_partial=allow_partial,real_sample=sample['key'],
                branch_overview=True,branch_keyboard_selection=True,selected_target_preserved=True,advanced_details_collapsed=True,
                keyboard_disclosures=True,sample_step_target_deep_link=True,token_navigation=True,context_positions=True,
                spatial_map=True,layer_table=True,deep_link=url,video_range=True,mobile_no_overflow=True,
                video_heatmap_overlay=True,overlay_real_playback_both_players=True,overlay_pixel_visual_sync=True,
                overlay_manual_input_slots=True,overlay_duplicate_timestamp_slots=any(a==b for a,b in zip(timestamps,timestamps[1:])),
                overlay_actual_pair_source_seek=True,overlay_all_frame_color_scale=True,overlay_transparency_visibility=True,
                overlay_all_cells_opaque=True,overlay_zero_color_boundary_opaque=True,frame_cards_actual_decimal_timestamps=True,
                overlay_object_fit_alignment=True,
                family_scopes=family_scopes,result_report_switch=True,unified_index=True,page_errors=errors,timestamp=stamp())
    write(storage()/'validation'/f'viewer_pilot{scope}.json',report); return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--scope',type=int,default=300); p.add_argument('--allow-partial',action='store_true'); p.add_argument('--viewer',action='store_true'); p.add_argument('--viewer-only',action='store_true'); a=p.parse_args()
    result=check_viewer(a.scope,a.allow_partial) if a.viewer_only else validate_scope(a.scope,a.allow_partial)
    if a.viewer and not a.viewer_only: result['viewer']=check_viewer(a.scope,a.allow_partial)
    print(json.dumps(result,ensure_ascii=False))
