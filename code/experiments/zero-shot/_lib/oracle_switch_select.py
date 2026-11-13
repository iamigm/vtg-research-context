"""Select three extreme saved greedy→oracle cases and reuse visual diagnostics.

No model inference: original responses, fixed top30 arrays and gradient maps
remain in their source stores. This is a GT-selected diagnostic cohort.
"""
from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from deterministic_oracle_diversity import evaluate, interval, iou

ROOT=Path('/home/irteam/experiments/zero-shot')
SOURCE=ROOT/'artifacts/runs/rollout/deterministic/pilot/900'
ATTRIBUTION=ROOT/'artifacts/runs/rollout/greedy_attribution/pilot/900'
COHORT=ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'
OUT=ROOT/'artifacts/runs/rollout/oracle_switch/pilot/3'
COHORT_SHA256='e92620e9b56da0ca62af58b2da07ab2fb3fa5bac58ab20aa880b709426124db7'

def read(path): return json.loads(Path(path).read_text())

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): result.update(chunk)
    return result.hexdigest()

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temporary.replace(path)

def require(condition,message):
    if not condition: raise ValueError(message)

def reference(path): return dict(path=str(path),sha256=sha(path))

def branch_at_greedy(record,oracle,trace):
    greedy=record['result']['candidates'][0]
    position=next((j for j,(g,o) in enumerate(zip(greedy['ids'],oracle['ids'])) if g!=o),None)
    if position is None: return None
    node_id=trace['candidates']['1']['nodes'][position]; node=trace['nodes'][node_id]
    require(node['prefix_ids']==greedy['ids'][:position]==oracle['ids'][:position],'First divergent prefix differs')
    top=node['top'][:30]; top_ids=[token['id'] for token in top]
    require(len(top_ids)==len(set(top_ids))==30,'Original greedy top30 coverage differs')
    greedy_token=greedy['ids'][position]; oracle_token=oracle['ids'][position]
    g=next(token for token in top if token['id']==greedy_token)
    alt=next((token for token in top if token['id']==oracle_token),None)
    return dict(step=position,step_indexing='zero_based',prefix_ids=node['prefix_ids'],prefix=node['prefix_text'],
        greedy_token=greedy_token,oracle_token=oracle_token,greedy_token_text=g['token'],
        oracle_token_text=alt['token'] if alt else None,source_rank=top_ids.index(oracle_token)+1 if alt else None,
        source_rank_indexing='one_based_saved_top30_order',top_ids=top_ids,
        greedy_source_probability=g['probability'],oracle_source_probability=alt['probability'] if alt else None,
        source_greedy_minus_oracle_logit_margin=math.log(g['probability']/alt['probability']) if alt else None,
        source_trace_node=node_id)

def region_summaries(target,payload,timestamps):
    norms=np.asarray(payload['pixel_frames']['gradient_norm'],dtype=np.float64)
    dots=np.asarray(payload['pixel_frames']['activation_dot_gradient'],dtype=np.float64)
    require(norms.shape==dots.shape and norms.ndim==3 and len(timestamps)==len(norms),'Pixel region geometry differs')
    require(np.isfinite(norms).all() and np.isfinite(dots).all() and (norms>=0).all(),'Invalid visual summaries')
    def patch(frame,index):
        row,col=divmod(int(index),norms.shape[2]); signed=float(dots[frame,row,col])
        return dict(frame_index=frame,timestamp=timestamps[frame],patch_row=row,patch_col=col,
            gradient_norm=float(norms[frame,row,col]),activation_dot_gradient=signed,
            pixel_y_range=[16*row,16*(row+1)],pixel_x_range=[16*col,16*(col+1)],
            scale_sign_to_reduce_greedy_minus_oracle_margin=-1 if signed>0 else 1 if signed<0 else 0)
    frames=[]
    for frame in range(len(norms)):
        flat_norm=norms[frame].ravel(); flat_dot=dots[frame].ravel()
        frames.append(dict(frame_index=frame,timestamp=timestamps[frame],
            gradient_norm_l2=float(np.linalg.norm(flat_norm)),activation_dot_gradient=math.fsum(flat_dot.tolist()),
            absolute_patch_dot_sum=math.fsum(np.abs(flat_dot).tolist()),
            strongest_gradient_patch=patch(frame,int(flat_norm.argmax())),
            strongest_absolute_dot_patch=patch(frame,int(np.abs(flat_dot).argmax()))))
    def top_patches(values):
        order=np.argsort(-values.ravel(),kind='stable')[:8]
        return [patch(int(index)//(norms.shape[1]*norms.shape[2]),int(index)%(norms.shape[1]*norms.shape[2])) for index in order]
    return dict(objective='greedy_logit_minus_oracle_logit',source_target=target['id'],
        source_target_ref=reference(ATTRIBUTION/target['data_file']),grid=list(norms.shape),patch_size=16,
        pixel_signed_sum=target['pixel_signed_sum'],pixel_gradient_l2=target['pixel_gradient_l2'],
        saved_patch_dot_sum=math.fsum(dots.ravel().tolist()),frames=frames,
        top_frames_by_gradient_norm=sorted(frames,key=lambda f:(-f['gradient_norm_l2'],f['frame_index']))[:3],
        top_frames_by_absolute_signed_sum=sorted(frames,key=lambda f:(-abs(f['activation_dot_gradient']),f['frame_index']))[:3],
        top_patches_by_gradient_norm=top_patches(norms),top_patches_by_absolute_dot=top_patches(np.abs(dots)),
        limitation='Saved single-backward local sensitivity; finite perturbation effects and repeatability are not established by this summary. Pixel gradient arrays are referenced, not copied.')

def visual_screen(sample,attribution,config):
    path=Path(sample['source']['arrays']); meta=attribution['arrays']; step=sample['branch']['step']
    require(sha(path)==meta['sha256'],'Attribution array checksum differs')
    with np.load(path,allow_pickle=False) as data:
        require(meta['tracked_tokens']==30 and data['token_ids'][step].tolist()==sample['branch']['top_ids'],'Frozen top30 changed')
        directions=data['directions'].tolist(); require(directions==meta['directions'],'Source screening directions differ')
        direction=directions.index('video_pixels'); ids=data['token_ids'][step].tolist()
        g=ids.index(sample['branch']['greedy_token']); o=ids.index(sample['branch']['oracle_token'])
        base=data['logits'][step].astype(np.float64)
        changes=data['changes'][direction,:,step].astype(np.float64)
        slopes=data['slopes'][direction,step].astype(np.float64)
        require(np.isfinite(base).all() and np.isfinite(changes).all() and np.isfinite(slopes).all(),'Nonfinite screening data')
        epsilon=float(config['screen_epsilon']); require(epsilon==.03125,'Initial screening epsilon differs')
        require(np.allclose(slopes,(changes[1]-changes[0])/(2*epsilon),atol=1e-6,rtol=1e-6),'Screening slope differs from probes')
        def summary(logits,alpha):
            others=np.array([index for index in range(30) if index!=o]); best=int(others[np.argmax(logits[others])])
            winners=[ids[index] for index in np.flatnonzero(logits==logits.max())]
            return dict(alpha=alpha,scale=1+alpha,greedy_logit=float(logits[g]),oracle_logit=float(logits[o]),
                greedy_minus_oracle_logit_margin=float(logits[g]-logits[o]),
                oracle_minus_best_other_logit_margin=float(logits[o]-logits[best]),best_other_token=ids[best],
                oracle_strict_top1_among_top30=bool(logits[o]>logits[best]),
                oracle_tied_for_top1_among_top30=sample['branch']['oracle_token'] in winners,
                tracked_top1_token_ids=winners,oracle_rank_by_strictly_greater=1+int((logits>logits[o]).sum()))
        result=dict(sample_key=sample['sample_key'],step=step,direction='video_pixels',epsilon=epsilon,
            perturbation='x_alpha=(1+alpha)*x for normalized video pixels only; unchanged original greedy prefix',
            baseline=summary(base,0.),measurements=[summary(base+changes[index],sign*epsilon) for index,sign in enumerate((-1,1))],
            oracle_minus_greedy_central_slope=float(slopes[o]-slopes[g]),source_arrays_ref=reference(path),
            array_slice=dict(direction_index=direction,step=step,greedy_column=g,oracle_column=o),
            candidate_scope='Only the fixed original greedy top30; untracked vocabulary is unmeasured in these saved probes.')
    exact=[target for target in attribution['targets'] if target['kind']=='logit_margin'
           and target['step']==step and target['alternative']==sample['branch']['oracle_token']]
    require(len(exact)<=1,'Repeated exact attribution target')
    if exact:
        target=exact[0]; target_path=ATTRIBUTION/target['data_file']
        require(sha(target_path)==target['sha256'],'Attribution target checksum differs')
        with gzip.open(target_path,'rt') as stream: payload=json.load(stream)
        matched={key:target[key] for key in ('id','data_file','sha256','pixel_signed_sum','pixel_gradient_l2','step','alternative')}
        matched.update(path=str(target_path),objective='greedy_logit_minus_oracle_logit')
        sample['matched_attribution_target']=matched
        result['visual_regions']=region_summaries(target,payload,attribution['input']['timestamps'])
    else:
        sample['matched_attribution_target']=None
        result['visual_regions']=None
        result['missing_exact_gradient_reason']='The exact first oracle branch was outside the four stored detailed margin targets; no other target is substituted.'
    return result

def build_selection(output=OUT):
    output=Path(output); cohort=read(COHORT)
    require(len(cohort)==900 and sha(COHORT)==COHORT_SHA256,'Frozen source cohort900 changed')
    require(read(SOURCE/'metadata/cohort.json')==cohort,'Deterministic source cohort differs')
    oracle_path=SOURCE/'oracle_diversity/metrics/per_query.json'; stored={r['sample_key']:r for r in read(oracle_path)}
    require(len(stored)==900,'Stored oracle coverage differs')
    source_refs=dict(cohort=reference(COHORT),oracle_analysis=reference(oracle_path),
        attribution_config=reference(ATTRIBUTION/'metadata/config.json'))
    records={}; rankings=[]
    for index,row in enumerate(cohort):
        key=row['sample_key']; path=SOURCE/'rollouts'/f'{key}.json'; raw=path.read_bytes()
        hashed=hashlib.sha256(raw).hexdigest(); saved=stored[key]
        require(hashed==saved['source_sha256'],'Original source response checksum differs')
        record=json.loads(raw); require(record['sample']==row,'Source cohort row differs')
        evaluated=evaluate(record,row,32); require(evaluated==saved['budgets']['32'],'Stored oracle metrics differ')
        candidates=record['result']['candidates']; greedy=candidates[0]; oracle=candidates[evaluated['oracle_index']-1]
        reasons=[]
        if interval(greedy) is None: reasons.append('invalid_greedy_interval_or_missing_eos')
        if interval(oracle) is None: reasons.append('invalid_oracle_interval_or_missing_eos')
        if evaluated['delta_iou']<=1e-12: reasons.append('no_positive_iou_gain')
        branch=None; trace_ref=None
        if not reasons:
            trace_path=SOURCE/'token_traces'/key/'index.json'; trace=read(trace_path); trace_ref=reference(trace_path)
            require(trace['raw_record_sha256']==hashed and trace['input_hashes']==record['input_hashes'],'Trace source differs')
            branch=branch_at_greedy(record,oracle,trace)
            if branch is None: reasons.append('no_first_token_divergence')
            elif branch['source_rank'] is None: reasons.append('first_oracle_token_outside_original_greedy_top30')
        rankings.append(dict(sample_key=key,dataset=row['dataset'],source_cohort_index=index,
            greedy_span=interval(greedy),oracle_span=interval(oracle),greedy_iou=evaluated['greedy_iou'],
            oracle_iou=evaluated['oracle_iou'],delta_iou=evaluated['delta_iou'],oracle_index=oracle['index'],
            oracle_chain=evaluated['oracle_chain'],oracle_ties=evaluated['oracle_ties'],relation=evaluated['relation'],
            branch=branch,eligible=not reasons,exclusion_reasons=reasons,
            source_rollout=dict(path=str(path),sha256=hashed),source_trace=trace_ref))
        records[key]=record
    rankings.sort(key=lambda r:(-r['delta_iou'],-r['oracle_iou'],r['sample_key']))
    dataset_ranks=collections.Counter(); selected=[]; selected_datasets=set()
    for rank,entry in enumerate(rankings,1):
        dataset_ranks[entry['dataset']]+=1; entry['global_rank']=rank; entry['dataset_rank']=dataset_ranks[entry['dataset']]
        if entry['eligible'] and entry['dataset'] not in selected_datasets:
            selected.append(entry); selected_datasets.add(entry['dataset'])
    require(len(selected)==3,'Expected one eligible query from each of three datasets')
    config=read(ATTRIBUTION/'metadata/config.json')
    catalog_path=ATTRIBUTION/'web/catalog.json'; media={r['key']:r for r in read(catalog_path)['samples']}
    source_refs['media_catalog']=reference(catalog_path)
    samples=[]; screens=[]
    for entry in sorted(selected,key=lambda r:r['source_cohort_index']):
        key=entry['sample_key']; record=records[key]; cs=record['result']['candidates']; greedy=cs[0]; oracle=cs[entry['oracle_index']-1]
        attribution_path=ATTRIBUTION/'queries'/f'{key}.json'; attribution=read(attribution_path)
        input_path=SOURCE/'inputs'/f'{key}.json'; input_record=read(input_path)
        require(attribution['complete'] and attribution['sample']==record['sample'] and attribution['greedy']['ids']==greedy['ids'],'Attribution source/greedy differs')
        require(attribution['input']['hashes']==record['input_hashes']==input_record['hashes'],'Original input hashes differ')
        source=dict(rollout=str(SOURCE/'rollouts'/f'{key}.json'),attribution=str(attribution_path),input=str(input_path),arrays=str(ATTRIBUTION/attribution['arrays']['file']))
        first=cs[entry['oracle_chain'][0]['child']-1]
        sample=dict(sample_key=key,dataset=entry['dataset'],source_cohort_index=entry['source_cohort_index'],
            source_global_rank=entry['global_rank'],source_dataset_rank=entry['dataset_rank'],row=record['sample'],
            greedy=dict(ids=greedy['ids'],span=interval(greedy),iou=entry['greedy_iou']),
            oracle=dict(ids=oracle['ids'],span=interval(oracle),iou=entry['oracle_iou'],index=oracle['index'],chain=entry['oracle_chain']),
            delta_iou=entry['delta_iou'],branch=entry['branch'],source=source,
            source_hashes={name:sha(path) for name,path in source.items()},source_trace=entry['source_trace'],
            input_hashes=record['input_hashes'],
            root_branch_continuation=dict(index=first['index'],span=interval(first),iou=iou(interval(first),record['sample']['span'][0]),
                equals_selected_oracle=first['ids']==oracle['ids']),
            video_url='/'+media[key]['video'].lstrip('/'),video_path=str(ROOT.parent/media[key]['video']))
        require(Path(sample['video_path']).is_file(),'Reused browser video missing')
        screens.append(visual_screen(sample,attribution,config)); samples.append(sample)
    policy=dict(oracle='Maximum raw GT IoU among up to 32 completed positive-interval candidates including greedy; exact ties choose lowest candidate index.',
        cohort='One maximum-delta eligible query per dataset from fixed source pilot900; a GT-selected extreme diagnostic, not a representative or independent test cohort.',
        eligibility='Valid greedy and oracle intervals with EOS, positive delta IoU, first divergent oracle token inside original greedy top30.',
        ranking=['delta_iou descending','oracle_iou descending','sample_key ascending'],source_cohort_indexing='zero_based',
        prediction_clipping=False,disjointness_filter=False,lineage_filter=False,
        perturbation_scope='visual input only; text input and generated-prefix perturbations excluded',
        branch_success='First oracle token switch is distinct from completing the full oracle path, especially for nested oracle lineages.')
    ranking_output=dict(source_queries=900,policy=policy,rankings=rankings)
    ranking_path=output/'metadata/rankings.json'; write(ranking_path,ranking_output)
    screen_path=output/'analyses/initial_screening.json'; write(screen_path,dict(scope_queries=3,perturbation_scope='video_pixels only',samples=screens))
    selection=dict(scope='pilot',scope_queries=3,source_queries=900,samples=samples,policy=policy,
        counts=dict(source_queries=900,selected_queries=3,selected_unique_videos=len({r['row']['video_path'] for r in samples}),
            eligible_queries=sum(r['eligible'] for r in rankings),
            excluded_by_reason=dict(collections.Counter(reason for r in rankings for reason in r['exclusion_reasons'])),
            selected_by_dataset=dict(collections.Counter(r['dataset'] for r in samples))),
        source_refs=source_refs,rankings_ref=reference(ranking_path),screening_ref=reference(screen_path),
        source_outputs_copied=False)
    write(output/'metadata/selection.json',selection)
    return selection

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=OUT); args=parser.parse_args()
    result=build_selection(args.output)
    print(json.dumps(dict(path=str(args.output/'metadata/selection.json'),counts=result['counts'],
        samples=[dict(sample_key=r['sample_key'],delta_iou=r['delta_iou'],branch=r['branch']['oracle_token'],
                      matched_attribution_target=r['matched_attribution_target']) for r in result['samples']]),ensure_ascii=False))
