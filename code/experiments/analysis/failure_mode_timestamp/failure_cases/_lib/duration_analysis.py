"""Descriptive duration calibration, without claiming a causal duration prior."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=relocate_path(Path(__file__)).resolve().parents[1];dest=relocate_path(ROOT/'reports/pilot_100');rows=json.loads((relocate_path(dest/'case_metrics.json')).read_text());stats={}
fig,axes=plt.subplots(1,2,figsize=(11,5))
for ax,ds in zip(axes,['charades-timelens','qvhighlights-timelens']):
    for model,c in [('qwen25','#c47833')]:
        group=[r for r in rows if r['dataset']==ds and r['model_key']==model and r['timestamps'] and r['timestamps'][0][1]>r['timestamps'][0][0]]
        gt=np.array([r['gt_length'] for r in group]);pred=np.array([r['timestamps'][0][1]-r['timestamps'][0][0] for r in group]);rat=pred/gt
        stats[f'{model}/{ds}']={'n':len(group),'gt_seconds_p25_p50_p75':np.quantile(gt,[.25,.5,.75]).tolist(),'pred_seconds_p25_p50_p75':np.quantile(pred,[.25,.5,.75]).tolist(),'pred_to_gt_ratio_p25_p50_p75':np.quantile(rat,[.25,.5,.75]).tolist()}
        ax.scatter(gt,pred,c=c,label=model,s=23,alpha=.55)
    ax.plot([.8,160],[.8,160],c='gray',ls='--',lw=1);ax.set(xscale='log',yscale='log',xlim=(.8,160),ylim=(.8,160),xlabel='GT duration (seconds, log scale)',ylabel='Predicted duration (seconds, log scale)',title=ds);ax.legend();ax.grid(alpha=.2)
fig.suptitle('Duration calibration | same 100 queries per dataset | first rounded prediction');fig.tight_layout();fig.savefig(relocate_path(dest/'duration_calibration.png'),dpi=170);fig.savefig(relocate_path(dest/'duration_calibration.pdf'));plt.close(fig)
write_result_text((relocate_path(dest/'duration_calibration.json')), json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
