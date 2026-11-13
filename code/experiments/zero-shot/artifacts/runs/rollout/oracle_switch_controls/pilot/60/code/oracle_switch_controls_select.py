"""Freeze a hash-selected, disjoint pilot60 for conditional oracle-switch controls.

Sampling population: the 805 eligible source queries after excluding the three
previously selected extreme cases. This does not estimate all-source accuracy.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from oracle_switch_select import (
    ROOT,SOURCE,ATTRIBUTION,COHORT,COHORT_SHA256,read,sha,write,reference,require,branch_at_greedy,
)
from deterministic_oracle_diversity import evaluate,interval,iou

EXTREMES=ROOT/'artifacts/runs/rollout/oracle_switch/pilot/3'
OUT=ROOT/'artifacts/runs/rollout/oracle_switch_controls/pilot/60'
DATASETS=('charades-timelens','activitynet-timelens','qvhighlights-timelens')
PREFIX='oracle_switch_controls|42|'
POPULATION_COUNTS={'charades-timelens':277,'activitynet-timelens':266,'qvhighlights-timelens':262}

def sampling_hash(key): return hashlib.sha256((PREFIX+key).encode()).hexdigest()

def build_selection(output=OUT):
    output=Path(output)
    cohort=read(COHORT); require(len(cohort)==900 and sha(COHORT)==COHORT_SHA256,'Fixed source cohort changed')
    require(read(SOURCE/'metadata/cohort.json')==cohort,'Original source cohort order changed')
    old_path=EXTREMES/'metadata/selection.json'; old=read(old_path)
    old_keys={sample['sample_key'] for sample in old['samples']}
    require(len(old_keys)==3 and old['scope_queries']==3,'Extreme pilot3 membership changed')
    ranking_path=Path(old['rankings_ref']['path'])
    require(sha(ranking_path)==old['rankings_ref']['sha256'],'Frozen source ranking checksum changed')
    ranking=read(ranking_path)['rankings']
    require(len(ranking)==900 and {r['sample_key'] for r in ranking}=={r['sample_key'] for r in cohort},'Ranking coverage changed')
    for entry in ranking:
        source=entry['source_rollout']
        require(sha(source['path'])==source['sha256'],f"Source response changed: {entry['sample_key']}")
    eligible=[entry for entry in ranking if entry['eligible']]
    population=[entry for entry in eligible if entry['sample_key'] not in old_keys]
    counts=dict(collections.Counter(entry['dataset'] for entry in population))
    require(len(eligible)==808 and len(population)==805 and counts==POPULATION_COUNTS,'Conditional sampling population changed')
    selected=[]; cutoffs={}
    for dataset in DATASETS:
        ordered=sorted((entry for entry in population if entry['dataset']==dataset),key=lambda r:(sampling_hash(r['sample_key']),r['sample_key']))
        selected.extend(ordered[:20]); cutoffs[dataset]=sampling_hash(ordered[19]['sample_key'])
    require(len(selected)==60 and len({entry['sample_key'] for entry in selected})==60,'Repeated selected query')
    require(not old_keys.intersection(entry['sample_key'] for entry in selected),'New cohort overlaps extreme pilot3')
    catalog_path=ATTRIBUTION/'web/catalog.json'; catalog={row['key']:row for row in read(catalog_path)['samples']}
    samples=[]; dataset_rank=collections.Counter()
    for entry in selected:
        key=entry['sample_key']; row=cohort[entry['source_cohort_index']]
        require(row['sample_key']==key,'Source cohort index changed')
        record=read(entry['source_rollout']['path']); require(record['sample']==row,'Original source sample changed')
        evaluated=evaluate(record,row,32)
        require(evaluated['oracle_index']==entry['oracle_index'] and evaluated['delta_iou']==entry['delta_iou'],
                f'{key}: recomputed oracle differs')
        candidates=record['result']['candidates']; greedy=candidates[0]; oracle=candidates[entry['oracle_index']-1]
        require(interval(greedy) and interval(oracle) and evaluated['delta_iou']>0,'Invalid selected source intervals')
        trace_ref=entry['source_trace']; require(sha(trace_ref['path'])==trace_ref['sha256'],'Source token trace changed')
        trace=read(trace_ref['path']); require(trace['raw_record_sha256']==entry['source_rollout']['sha256'],'Token trace response mismatch')
        branch=branch_at_greedy(record,oracle,trace)
        require(branch==entry['branch'] and branch['source_rank'] is not None,'Original greedy top30 divergence changed')
        attribution_path=ATTRIBUTION/'queries'/f'{key}.json'; attribution=read(attribution_path)
        input_path=SOURCE/'inputs'/f'{key}.json'; original_input=read(input_path)
        require(attribution['complete'] and attribution['sample']==row and attribution['greedy']['ids']==greedy['ids'],'Attribution replay source differs')
        require(record['input_hashes']==original_input['hashes']==attribution['input']['hashes']==trace['input_hashes'],'Original input hashes differ')
        arrays_path=ATTRIBUTION/attribution['arrays']['file']
        require(sha(arrays_path)==attribution['arrays']['sha256'],'Attribution array checksum changed')
        with np.load(arrays_path,allow_pickle=False) as arrays:
            require(arrays['token_ids'][branch['step']].tolist()==branch['top_ids'],'Attribution fixed top30 differs')
        source=dict(rollout=entry['source_rollout']['path'],attribution=str(attribution_path),input=str(input_path),arrays=str(arrays_path))
        first=candidates[entry['oracle_chain'][0]['child']-1]
        targets=[target for target in attribution['targets'] if target['kind']=='logit_margin'
                 and target['step']==branch['step'] and target['alternative']==branch['oracle_token']]
        require(len(targets)<=1,'Repeated exact saved attribution target')
        matched=None
        if targets:
            target=targets[0]
            matched={name:target[name] for name in ('id','data_file','sha256','pixel_signed_sum','pixel_gradient_l2','step','alternative')}
            matched.update(path=str(ATTRIBUTION/target['data_file']),objective='greedy_logit_minus_oracle_logit')
        dataset_rank[entry['dataset']]+=1
        sample=dict(sample_key=key,dataset=entry['dataset'],source_cohort_index=entry['source_cohort_index'],
            source_global_rank=entry['global_rank'],source_dataset_rank=entry['dataset_rank'],row=row,
            selection_hash=sampling_hash(key),selection_rank_within_dataset=dataset_rank[entry['dataset']],
            greedy=dict(ids=greedy['ids'],span=interval(greedy),iou=entry['greedy_iou']),
            oracle=dict(ids=oracle['ids'],span=interval(oracle),iou=entry['oracle_iou'],index=oracle['index'],chain=entry['oracle_chain']),
            delta_iou=entry['delta_iou'],branch=branch,source=source,
            source_hashes={name:sha(path) for name,path in source.items()},source_trace=trace_ref,input_hashes=record['input_hashes'],
            root_branch_continuation=dict(index=first['index'],span=interval(first),iou=iou(interval(first),row['span'][0]),
                equals_selected_oracle=first['ids']==oracle['ids']),matched_attribution_target=matched,
            video_url='/'+catalog[key]['video'].lstrip('/'),video_path=str(ROOT.parent/catalog[key]['video']))
        require(Path(sample['video_path']).is_file(),'Reused browser media missing')
        samples.append(sample)
    source_refs=dict(cohort=reference(COHORT),source_rankings=reference(ranking_path),excluded_extreme_selection=reference(old_path),
        attribution_config=reference(ATTRIBUTION/'metadata/config.json'),media_catalog=reference(catalog_path))
    old_video_paths={sample['row']['video_path'] for sample in old['samples']}
    population_rows=[cohort[entry['source_cohort_index']] for entry in population]
    frame=json.dumps(sorted((entry['sample_key'],entry['dataset']) for entry in population),separators=(',',':'))
    selection=dict(scope='pilot',scope_queries=60,source_queries=900,samples=samples,source_refs=source_refs,
        policy=dict(family='rollout/oracle_switch_controls',sampling='lowest SHA-256 hash per dataset, 20 each; no delta-IoU ranking inside the eligible population',
            hash_expression="SHA256('oracle_switch_controls|42|' + sample_key)",seed=42,
            sample_order='Charades, ActivityNet, QVHighlights; ascending hash within each dataset',source_cohort_indexing='zero_based',
            eligibility='Valid greedy/oracle intervals with EOS, positive oracle@32 GT IoU gain, first divergent oracle token in original greedy top30.',
            target_population='The 805 eligible fixed-pilot900 queries excluding the three prior extreme queries; conditional on a better saved oracle and tracked first divergence.',
            generalization_limit='Does not estimate all 900 queries, all benchmark queries, no-improvement cases, untracked branches, other models, or natural edited videos.',
            excluded_extremes=True,cumulative_with_extreme_pilot3=False,prediction_clipping=False,perturbation_scope='visual input only',
            failures='Retain baseline replay failures, no-switch cases, nonrepeatability and decoding failures in the selected 60-query denominator.'),
        counts=dict(source_queries=900,source_eligible_queries=808,excluded_extreme_queries=3,population_queries=805,
            source_eligible_by_dataset=dict(collections.Counter(entry['dataset'] for entry in eligible)),
            population_by_dataset=counts,population_unique_videos=len({row['video_path'] for row in population_rows}),
            selected_queries=60,selected_by_dataset=dict(collections.Counter(sample['dataset'] for sample in samples)),
            selected_unique_videos=len({sample['row']['video_path'] for sample in samples}),
            selected_video_overlap_with_extremes=sum(sample['row']['video_path'] in old_video_paths for sample in samples),
            selected_direct_oracle=sum(len(sample['oracle']['chain'])==1 for sample in samples),
            selected_nested_oracle=sum(len(sample['oracle']['chain'])>1 for sample in samples),
            selected_zero_source_margin=sum(sample['branch']['source_greedy_minus_oracle_logit_margin']==0 for sample in samples)),
        excluded_extreme_refs=[dict(sample_key=sample['sample_key'],query=reference(EXTREMES/'queries'/f"{sample['sample_key']}.json")) for sample in old['samples']],
        sampling_frame_sha256=hashlib.sha256(frame.encode()).hexdigest(),hash_cutoff_by_dataset=cutoffs,
        prespecified_precision_subset=[sample['sample_key'] for dataset in DATASETS for sample in samples if sample['dataset']==dataset and sample['selection_rank_within_dataset']<=2],
        inference_plan=dict(primary='Fixed-budget oracle-branch reach rate and free-decoding delta IoU on all selected 60; count failures in denominator.',
            per_dataset_interval='95% Wilson interval for binary query outcomes, n=20 per dataset.',
            population_weights={dataset:count/805 for dataset,count in counts.items()},
            paired_comparison='Gradient-minus-reverse/random query-level differences; random seeds are repeats, not independent queries; stratified video-cluster bootstrap for paired differences and mean delta IoU.',
            old_extremes='Report original pilot3 separately; do not pool its outcome-selected queries into pilot60 estimates.'),
        source_outputs_copied=False)
    destination=output/'metadata/selection.json'
    if destination.exists():
        require(read(destination)==selection,'Frozen pilot60 selection differs; refusing to overwrite')
    else: write(destination,selection)
    return selection

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=OUT); args=parser.parse_args()
    result=build_selection(args.output)
    print(json.dumps(dict(path=str(args.output/'metadata/selection.json'),counts=result['counts'],
        precision_subset=result['prespecified_precision_subset']),ensure_ascii=False))
