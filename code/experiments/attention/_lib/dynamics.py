from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Measure token-by-token attention from the previous actual generation run."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

PREV=ROOT
OUT=DYNAMICS
DATASETS=['Charades-STA','QVHighlights','ActivityNet-Captions']
SCOPES=['predictive','consumed']

def write(path,obj):
    write_result_text(path, json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')

def merged(windows,duration):
    result=[]
    for s,e in sorted((max(0,s),min(duration,e)) for s,e in windows):
        if e<=s:continue
        if result and s<=result[-1][1]:result[-1][1]=max(e,result[-1][1])
        else:result.append([s,e])
    return result

def fractions(edges,windows):
    return sum((np.maximum(0,np.minimum(edges[1:],e)-np.maximum(edges[:-1],s))/np.diff(edges) for s,e in windows),start=np.zeros(len(edges)-1))

def temporal_metrics(raw,t,edges,duration,gt,baseline):
    mass=raw.sum(-1)
    p=raw/mass[:,None]
    peak=t[raw.argmax(-1)]
    centroid=p@t
    overlap=fractions(edges,gt)
    entropy=-(p*np.log(np.maximum(p,1e-30))).sum(-1)/np.log(len(t)) if len(t)>1 else np.zeros(len(raw))
    result=dict(video_attention=mass.tolist(),raw_gt_attention=(raw@overlap).tolist(),
        gt_mass=(p@overlap).tolist(),peak_seconds=peak.tolist(),centroid_seconds=centroid.tolist(),
        peak_in_gt=[any(s<=x<=e for s,e in gt) for x in peak],
        early_peak=(peak<duration*.1).tolist(),normalized_entropy=entropy.tolist(),
        tv_from_query=(np.abs(p-baseline).sum(-1)/2).tolist(),
        tv_previous=[None]+(np.abs(np.diff(p,axis=0)).sum(-1)/2).tolist(),
        peak_jump_seconds=[None]+np.abs(np.diff(peak)).tolist(),
        centroid_shift_seconds=[None]+np.abs(np.diff(centroid)).tolist(),
        peak_bin_changed=[None]+(raw.argmax(-1)[1:]!=raw.argmax(-1)[:-1]).tolist())
    return result,p

def measure(sample):
    src=relocate_path(GENERATION/ sample['dataset'] / sample['sample_key']);dst=relocate_path(OUT/ sample['dataset'] / sample['sample_key']);dst.mkdir(parents=True, exist_ok=True)
    a=json.loads((relocate_path(src/'answer.json')).read_text());m=json.loads((relocate_path(src/'metadata.json')).read_text())
    t=np.array(m['sampled_timestamps']);duration=m['duration_video'];edges=np.r_[0,(t[:-1]+t[1:])/2,duration]
    assert np.all(np.diff(edges)>0)
    n=len(a['token_ids']);assert n>1
    gt=merged(m['gt_windows'],duration);ov=fractions(edges,gt)
    arrays={};validation={}
    with np.load(relocate_path(src/'attention.npz')) as z:
        query=z['query_temporal'][8:22].mean((0,1,2)).astype(float)
        pred=z['predictive_temporal'];cons=z['answer_temporal']
        assert pred.shape==cons.shape and pred.shape[2]==n
        assert np.array_equal(pred[:,:,1:],cons[:,:,:-1])
        for scope,array,source in [('predictive',pred,'predictive'),('consumed',cons,'answer')]:
            arrays[scope]=array[8:22].mean((0,1)).astype(float)
            arrays[scope+'_keys']=z[source+'_key_mass'][8:22].mean((0,1)).astype(float)
            assert np.allclose(arrays[scope].sum(-1),arrays[scope+'_keys'][:,0],atol=2e-6)
    qnorm=relocate_path(query/query.sum())
    data=dict(sample_key=sample['sample_key'],dataset=sample['dataset'],qid=sample['qid'],query=sample['query'],
        answer=a['text'],tokens=a['tokens'],token_ids=a['token_ids'],timestamp_token_indices=a['timestamp_token_indices'],
        duration=duration,timestamps=t.tolist(),edges=edges.tolist(),gt=m['gt_windows'],predictions=a['parsed_spans'],
        truncated=a['truncated'],query_raw=query.tolist(),query_gt_mass=float(qnorm@ov),
        query_peak_seconds=float(t[query.argmax()]),query_video_attention=float(query.sum()),scopes={})
    token_rows=[]
    for scope in SCOPES:
        metrics,p=temporal_metrics(arrays[scope],t,edges,duration,gt,qnorm)
        metrics['keys']=arrays[scope+'_keys'].tolist()
        data['scopes'][scope]=dict(metrics=metrics,raw=arrays[scope].tolist())
        for i,token in enumerate(a['tokens']):
            r=dict(sample_key=sample['sample_key'],dataset=sample['dataset'],scope=scope,step=i+1,token_id=a['token_ids'][i],token=token,
                is_timestamp_digit=i in a['timestamp_token_indices'],attention_row_position=m['sequence_length']+i-(scope=='predictive'))
            r.update({k:v[i] for k,v in metrics.items() if k!='keys'})
            r.update(dict(zip(['key_video','key_input_query','key_previous_answer','key_other_prefix'],metrics['keys'][i])))
            token_rows.append(r)
    write(relocate_path(dst/'dynamics.json'),data)
    with (relocate_path(dst/'token_metrics.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(token_rows[0]));w.writeheader();w.writerows(token_rows)
    write(relocate_path(dst/'validation.json'),dict(token_count=n,all_36_layers_predictive_consumed_alignment=True,
        source_npz_sha256=hashlib.sha256((relocate_path(src/'attention.npz')).read_bytes()).hexdigest(),
        generation_reused=True,gt_never_input=True))
    return data,token_rows

def summarize(data):
    result={}
    for ds in DATASETS:
        group=[d for d in data if d['dataset']==ds];summary=dict(samples=len(group),tokens=sum(len(d['tokens']) for d in group),
            truncated=sum(d['truncated'] for d in group),scopes={})
        for scope in SCOPES:
            rows=[]
            for d in group:
                m=d['scopes'][scope]['metrics'];nums=d['timestamp_token_indices'];other=[i for i in range(len(d['tokens'])) if i not in nums]
                def avg(key,idx):return float(np.mean([m[key][i] for i in idx])) if idx else None
                rows.append(dict(sample_key=d['sample_key'],n=len(d['tokens']),
                    first_video=m['video_attention'][0],last_video=m['video_attention'][-1],
                    first_gt=m['gt_mass'][0],last_gt=m['gt_mass'][-1],
                    mean_tv=float(np.mean(m['tv_previous'][1:])),max_tv=max(m['tv_previous'][1:]),
                    peak_change_rate=float(np.mean(m['peak_bin_changed'][1:])),
                    mean_peak_jump_fraction=float(np.mean(m['peak_jump_seconds'][1:]))/d['duration'],
                    mean_gt=float(np.mean(m['gt_mass'])),mean_early_peak=float(np.mean(m['early_peak'])),
                    numeric_gt=avg('gt_mass',nums),other_gt=avg('gt_mass',other),
                    numeric_video=avg('video_attention',nums),other_video=avg('video_attention',other)))
            median=lambda key:float(np.median([r[key] for r in rows]))
            paired=[r for r in rows if r['numeric_gt'] is not None and r['other_gt'] is not None]
            s=dict(median_sample_mean_adjacent_tv=median('mean_tv'),median_sample_max_adjacent_tv=median('max_tv'),
                samples_with_unchanged_peak=sum(r['peak_change_rate']==0 for r in rows),
                median_sample_peak_change_rate=median('peak_change_rate'),
                median_sample_mean_peak_jump_fraction=median('mean_peak_jump_fraction'),
                median_first_video_attention=median('first_video'),median_last_video_attention=median('last_video'),
                median_first_to_last_video_change=float(np.median([r['last_video']-r['first_video'] for r in rows])),
                median_first_to_last_gt_change=float(np.median([r['last_gt']-r['first_gt'] for r in rows])),
                mean_sample_early_peak_rate=float(np.mean([r['mean_early_peak'] for r in rows])),
                timestamp_paired_samples=len(paired),median_timestamp_minus_other_gt=float(np.median([r['numeric_gt']-r['other_gt'] for r in paired])),
                median_timestamp_minus_other_video=float(np.median([r['numeric_video']-r['other_video'] for r in paired])),sample_summaries=rows)
            progress=[]
            for b in range(10):
                values={key:[] for key in ['video_attention','gt_mass','tv_previous','early_peak']}
                for d in group:
                    n=len(d['tokens']);ix=np.minimum(9,np.floor(np.arange(n)/(n-1)*10).astype(int))
                    for key in values:
                        v=[x for i,x in enumerate(d['scopes'][scope]['metrics'][key]) if ix[i]==b and x is not None]
                        if v:values[key].append(float(np.mean(v)))
                progress.append({k:dict(n=len(v),median=float(np.median(v)) if v else None,q25=float(np.quantile(v,.25)) if v else None,q75=float(np.quantile(v,.75)) if v else None) for k,v in values.items()})
            s['progress_bins']=progress;summary['scopes'][scope]=s
        result[ds]=summary
    return result

def main():
    samples=json.loads((relocate_path(OUT/'samples.json')).read_text());assert samples==json.loads((relocate_path(GENERATION / 'samples.json')).read_text())
    all_data=[];all_rows=[]
    for i,s in enumerate(samples):
        d,rows=measure(s);all_data.append(d);all_rows+=rows
        if (i+1)%25==0:print('Measured',i+1,flush=True)
    write(relocate_path(OUT/'summary.json'),summarize(all_data))
    with (relocate_path(OUT/'all_token_metrics.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    print('Measured all 300 samples',flush=True)

if __name__=='__main__':main()
