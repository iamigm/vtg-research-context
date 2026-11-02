from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Known-answer metric tests and independent saved-artifact validation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import rankdata
from hidden_capture import HiddenCapture, SOURCE
from hidden_analysis import ROOT, RESULTS, compare, fractions, entropy, write_json


def unit_checks():
    # Explicit pair-cosine values also detect pooling vectors BEFORE cosine.
    capture=HiddenCapture.__new__(HiddenCapture)
    capture.qpos=torch.tensor([4,5]);capture.vpos=torch.tensor([0,1,2,3])
    capture.info={'video_grid_thw':[2,1,2],'spatial_tokens_per_bin':2}
    capture.cosine=[None];capture.token_cosine=[None];capture.norm_min=[]
    vectors=torch.tensor([[[1.,0.],[0.,2.],[-3.,0.],[0.,4.],[5.,0.],[0.,6.]]])
    capture.hook(0)(None,None,vectors)
    np.testing.assert_allclose(capture.cosine[0],[[.5,-.5],[.5,.5]],atol=1e-7)
    np.testing.assert_allclose(capture.token_cosine[0],[.5,.5,-.5,.5],atol=1e-7)
    # Identical distributions, reversed ranks, and average-rank ties.
    scores=np.array([-.5,0,.2,.8]);t=np.arange(4)
    identical=compare(scores,softmax(scores/.1),t,4,.1)
    assert identical['tv']<1e-14 and identical['js_bits']<1e-14
    assert identical['spearman']==1 and identical['top20_overlap']==1
    reverse=compare(scores,softmax(-scores/.1),t,4,.1)
    assert reverse['spearman']==-1 and reverse['top20_overlap']==0
    assert compare(np.ones(4),np.ones(4),t,4,.1)['spearman'] is None
    assert compare([1,1,2,3],[1,1,2,3],t,4,.1)['spearman']==1
    np.testing.assert_allclose(fractions(np.arange(5),[[-1,1.5],[1,2.5],[8,9]]),[1,1,.5,0])
    return dict(pairwise_cosine_and_pooling=True,identical_distribution=True,
                reversed_ranks=True,ties_and_constant=True,interval_union_clipping=True)


def artifacts():
    samples=json.loads((relocate_path(RESULTS/'samples.json')).read_text())
    assert len(samples)==300
    metrics=list(csv.DictReader((relocate_path(RESULTS/'metrics.csv')).open()))
    primary={r['sample_key']:r for r in metrics if r['scope']=='L8_21' and r['temperature']=='0.1' and r['region']=='full'}
    assert len(primary)==300
    checks=[]
    for s in samples:
        folder=relocate_path(RESULTS/s['dataset']/s['sample_key'])
        meta=json.loads((relocate_path(folder/'metadata.json')).read_text())
        val=json.loads((relocate_path(folder/'validation.json')).read_text())
        assert val['prior_attention_bit_identical'] and val['input_positions_and_text_identical']
        with np.load(relocate_path(folder/'distributions.npz')) as z:
            c=z['cosine_query_frame'].astype(float);a=z['attention_query_frame'].astype(float)
            tc=z['cosine_video_token'];ta=z['attention_video_token']
            t=z['sampled_timestamps']
        assert c.shape==a.shape==(36,len(meta['query_positions']),len(t))
        assert tc.shape==ta.shape==(36,len(meta['video_positions']))
        assert np.isfinite(c).all() and np.isfinite(tc).all() and np.abs(tc).max()<=1.00001
        assert np.isfinite(a).all() and (a>=0).all()
        hw=meta['spatial_tokens_per_bin']
        np.testing.assert_allclose(tc.reshape(36,len(t),hw).mean(-1),c.mean(1),atol=2e-7)
        np.testing.assert_allclose(ta.reshape(36,len(t),hw).sum(-1),a.mean(1),atol=2e-7)
        with np.load(relocate_path(GENERATION/s['dataset']/s['sample_key']/'attention.npz')) as z:
            prior=z['query_temporal'].mean(1)
        assert np.array_equal(a,prior)
        scores=c[8:22].mean((0,1));raw=a[8:22].mean((0,1))
        p=softmax(scores/.1);q=raw/raw.sum();m=(p+q)/2
        tv=np.abs(p-q).sum()/2
        js=(np.sum(p[p>0]*np.log2(relocate_path(p[p>0]/m[p>0])))+np.sum(q[q>0]*np.log2(q[q>0]/m[q>0])))/2
        # Independent average-rank normalized dot product for Spearman.
        x,y=rankdata(scores),rankdata(raw);x-=x.mean();y-=y.mean()
        rho=float(x@y/np.sqrt((x@x)*(y@y)))
        r=primary[s['sample_key']]
        for metric,value in [('tv',tv),('js_bits',js),('spearman',rho)]:
            np.testing.assert_allclose(float(r[metric]),value,atol=1e-12)
        curves=np.genfromtxt(relocate_path(folder/'curves.csv'),delimiter=',',names=True)
        np.testing.assert_allclose(curves['similarity_probability'],p,atol=1e-12)
        np.testing.assert_allclose(curves['attention_probability'],q,atol=1e-12)
        assert (relocate_path(folder/'comparison.png')).stat().st_size>10000
        checks.append(s['sample_key'])
    for r in csv.DictReader((relocate_path(RESULTS/'entropy_matched_metrics.csv')).open()):
        assert abs(float(r['similarity_entropy'])-float(r['attention_entropy']))<1e-9
    summary=json.loads((relocate_path(RESULTS/'summary.json')).read_text())
    for dataset in ['All','Charades-STA','QVHighlights','ActivityNet-Captions']:
        group=[r for r in primary.values() if dataset=='All' or r['dataset']==dataset]
        assert len(group)==(300 if dataset=='All' else 100)
        for key in ['tv','js_bits','spearman','top20_overlap']:
            np.testing.assert_allclose(np.mean([float(r[key]) for r in group]),summary[dataset]['L8_21__tau0.1__full'][key]['mean'])
    return dict(samples_checked=len(checks),prior_attention_bit_identical=True,
                frame_token_aggregation=True,independent_tv_js_spearman=True,
                saved_curves_recomputed=True,entropy_matching=True,summary_recomputed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--unit-only',action='store_true');args=parser.parse_args()
    result={'unit':unit_checks()}
    if not args.unit_only:result['artifacts']=artifacts()
    write_json(relocate_path(RESULTS/('unit_validation.json' if args.unit_only else 'final_validation.json')),result)
    print(json.dumps(result,indent=2))
