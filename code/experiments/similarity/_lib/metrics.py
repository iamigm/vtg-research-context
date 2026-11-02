"""Within-video frame retrieval metrics, macro averaged over video-query pairs."""
import numpy as np
from sklearn.metrics import average_precision_score

TEMPERATURES=[0.01,0.03,0.07,0.1,0.3,1.0]

def labels(timestamps,spans):
    ts=np.asarray(timestamps)
    y=np.zeros(len(ts),dtype=bool)
    for a,b in spans: y |= (ts>=a)&(ts<b)
    return y

def probabilities(scores,tau):
    z=np.asarray(scores,dtype=np.float64)
    z=(z-z.max())/tau
    p=np.exp(z)
    return p/p.sum()

def metrics(scores,y):
    s=np.asarray(scores,dtype=np.float64)
    y=np.asarray(y,dtype=bool)
    assert s.shape==y.shape and s.ndim==1 and len(s)>0
    assert np.isfinite(s).all()
    n=len(y); positives=int(y.sum()); baseline=positives/n
    # Exact score ties are ordered chronologically for first-GT rank.
    # AP uses sklearn threshold-grouped ties, not trapezoidal PR-AUC.
    order=np.argsort(-s,kind='stable')
    first=int(np.flatnonzero(y[order])[0]+1) if positives else None
    result=dict(n_frames=n,n_gt=positives,gt_fraction=baseline,
        ap=float(average_precision_score(y,s)) if positives else None,
        first_gt_rank=first, reciprocal_rank=1/first if first else None,
        first_gt_percentile=(first-1)/(n-1) if first and n>1 else (0. if first else None),
        score_gt_mean=float(s[y].mean()) if positives else None,
        score_out_mean=float(s[~y].mean()) if (~y).any() else None,
        mass={})
    for tau in TEMPERATURES:
        mass=float(probabilities(s,tau)[y].sum())
        result['mass'][str(tau)]=dict(gt_mass=mass,uniform_baseline=baseline,
                                    lift=mass/baseline if baseline else None)
    return result

def aggregate(records):
    valid=[r for r in records if r['ap'] is not None]
    def mean(key): return float(np.mean([r[key] for r in valid])) if valid else None
    result=dict(n=len(records),n_valid=len(valid),n_zero_gt=len(records)-len(valid),
        map=mean('ap'),mrr=mean('reciprocal_rank'),
        median_first_gt_rank=float(np.median([r['first_gt_rank'] for r in valid])) if valid else None,
        mean_first_gt_rank=mean('first_gt_rank'),mean_first_gt_percentile=mean('first_gt_percentile'),
        mean_gt_fraction=float(np.mean([r['gt_fraction'] for r in records])) if records else None,mass={})
    for tau in TEMPERATURES:
        key=str(tau)
        result['mass'][key]={name:float(np.mean([r['mass'][key][name] for r in valid])) if valid else None
                             for name in ['gt_mass','uniform_baseline','lift']}
    return result
