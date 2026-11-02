"""Independent integrity checks for the completed 300-pair experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from prepare import OUT, MANIFEST, video_key, write
from metrics import metrics, probabilities

def reference_ap(s,y):
    # Integrate the empirical precision staircase over recall jumps at distinct
    # score thresholds, independently of sklearn and the production metric code.
    thresholds=sorted(set(float(v) for v in s),reverse=True)
    last=0.;area=0.;total=int(y.sum())
    if not total:return None
    for t in thresholds:
        chosen=s>=t
        tp=int(y[chosen].sum());recall=tp/total
        area+=(recall-last)*tp/int(chosen.sum())
        last=recall
    return area

def main():
    # Nontrivial hand cases include ties, constant scores, no positives and a
    # confidently wrong top result. This also checks baseline/lift semantics.
    for scores,y,expected in [([.9,.8,.7],[0,1,1],7/12),([1,1,1],[0,1,1],2/3),
                               ([.1,.9,.8],[0,1,0],1.),([1,0],[0,0],None)]:
        r=metrics(scores,y)
        if expected is None:assert r['ap'] is None and r['first_gt_rank'] is None
        else:assert abs(r['ap']-expected)<1e-12
    uniform=metrics([1.,1.,1.,1.],[0,1,0,0])
    for v in uniform['mass'].values():assert v['gt_mass']==.25 and v['lift']==1.
    rows=json.loads((OUT/'samples.json').read_text())
    assert rows==json.loads(MANIFEST.read_text()) and rows
    assert len({s['sample_key'] for s in rows})==len(rows)
    recorded=json.loads((OUT/'sample_metrics.json').read_text())
    lookup={(r['model'],r['sample_key']):r for r in recorded}
    checked=0;zero=[]
    for model in ['clip','qwen']:
        with np.load(OUT/model/'queries.npz') as saved:
            q={name:saved[name] for name in saved.files}
        np.testing.assert_array_equal(q['sample_keys'],[s['sample_key'] for s in rows])
        np.testing.assert_allclose(np.linalg.norm(q['embeddings'],axis=1),1,atol=2e-5)
        assert json.loads((OUT/model/'status.json').read_text())['stage']=='complete'
        for i,s in enumerate(rows):
            a=np.load(OUT/model/'scores'/f'{s["sample_key"]}.npz')
            v=np.load(OUT/model/'video_embeddings'/f'{video_key(s["video_path"])}.npz')
            meta=json.loads((OUT/'frames'/video_key(s['video_path'])/'metadata.json').read_text())
            np.testing.assert_array_equal(a['timestamps'],meta['timestamps'])
            np.testing.assert_array_equal(a['timestamps'],v['timestamps'])
            assert len(a['scores'])==meta['count']
            assert np.isfinite(a['scores']).all()
            np.testing.assert_allclose(np.linalg.norm(v['embeddings'],axis=1),1,atol=2e-5)
            np.testing.assert_allclose(a['scores'],v['embeddings']@q['embeddings'][i],atol=1e-6)
            y=np.array([any(start<=t<end for start,end in s['span']) for t in a['timestamps']])
            r=lookup[model,s['sample_key']]
            assert r['n_gt']==int(y.sum())
            if y.any():
                assert abs(r['ap']-reference_ap(a['scores'],y))<1e-12
                order=sorted(range(len(y)),key=lambda j:(-float(a['scores'][j]),j))
                first=next(k+1 for k,j in enumerate(order) if y[j])
                assert r['first_gt_rank']==first
            else:zero.append([model,s['sample_key']])
            for tau,vals in r['mass'].items():
                p=probabilities(a['scores'],float(tau))
                assert abs(p.sum()-1)<1e-12
                assert abs(vals['gt_mass']-float(p[y].sum()))<1e-12
                assert vals['uniform_baseline']==float(y.mean())
                if y.any():assert abs(vals['lift']-p[y].sum()/y.mean())<1e-12
            checked+=1
    summary=json.loads((OUT/'summary.json').read_text())
    for ds,models in summary['groups'].items():
        for model,g in models.items():
            rr=[r for r in recorded if r['model']==model and (ds=='all' or r['dataset']==ds) and r['ap'] is not None]
            assert abs(g['map']-np.mean([r['ap'] for r in rr]))<1e-12
    write(OUT/'validation.json',dict(passed=True,samples=len(rows),records=checked,zero_gt=zero,
        cohort_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),checks=[
        'exact full manifest membership',f'{2*len(rows)} complete score arrays','identical timestamps across backbones',
        'embedding unit norms','cosine independently recomputed from embeddings',
        'AP independently integrated at score thresholds','first GT rank independently sorted',
        'softmax normalization, GT mass, uniform baseline and lift','macro group averages',
        'hand cases: ties, uniform scores, confidently wrong top, no positives']))
    print('VALIDATION PASSED',checked)

if __name__=='__main__':main()
