"""Same-checkpoint, same-query paired comparison to the immutable baseline outputs."""
from result_layout import relocate_path, write_result_text
import collections,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import measure
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT.parents[1]/'failure_mode_baseline/failure_cases');DEST=relocate_path(ROOT/'reports/pilot_100');DEST.mkdir(parents=True,exist_ok=True)
LABELS=['success','no_overlap','overextended','underextended','shifted','parse_failure','invalid_interval']
def metrics(rows):
 return {'n':len(rows),'mIoU':100*float(np.mean([r['iou'] for r in rows])),**{f'R1@{t}':100*float(np.mean([r['iou']>=t for r in rows])) for t in [.3,.5,.7]},'failure_counts':dict(collections.Counter(r['failure_type'] for r in rows)),'out_of_bounds':sum(r['out_of_bounds'] for r in rows),'out_of_bounds_over_1_second':sum(r['out_of_bounds_excess_seconds']>1 for r in rows),'rounding_flips_05':sum(r['rounding_flips_05'] for r in rows),'raw_R1_05':100*float(np.mean([r['raw_iou']>=.5 for r in rows])),'raw_mIoU':100*float(np.mean([r['raw_iou'] for r in rows])),'median_abs_center_error_seconds':float(np.median([abs(r['center_error_relative'])*r['duration'] for r in rows if r['center_error_relative'] is not None])),'pred_duration_median':float(np.median([r['timestamps'][0][1]-r['timestamps'][0][0] for r in rows if r['timestamps'] and r['timestamps'][0][1]>r['timestamps'][0][0]]))}
def main():
 assert (relocate_path(ROOT/'data/samples_100.json')).read_bytes()==(relocate_path(BASE/'data/samples_100.json')).read_bytes()
 pairs=[];groups={}
 for m in ['qwen25']:
  bp=relocate_path(BASE/'results/pilot_100'/m/'predictions.jsonl');tp=relocate_path(ROOT/'results/pilot_100'/m/'predictions.jsonl')
  if not tp.exists():continue
  bs={r['sample_id']:{**r,**measure(r),'model_key':m} for r in map(json.loads,bp.read_text().splitlines())}
  transformed=[{**r,**measure(r),'model_key':m} for r in map(json.loads,tp.read_text().splitlines())]
  assert len({r['sample_id'] for r in transformed})==len(transformed)
  for r in transformed:
   b=bs[r['sample_id']]
   for k in ['model','query','span','duration','video_path']:assert r[k]==b[k],(m,r['sample_id'],k)
   outcome='both_success' if b['success_05'] and r['success_05'] else 'both_fail' if not b['success_05'] and not r['success_05'] else 'recovered' if r['success_05'] else 'regressed'
   pairs.append({'model_key':m,'dataset':r['dataset'],'sample_id':r['sample_id'],'video_id':r['video_id'],'query':r['query'],'gt':r['span'][0],'baseline_iou':b['iou'],'timestamp_iou':r['iou'],'delta_iou':r['iou']-b['iou'],'baseline_prediction':b['timestamps'],'timestamp_prediction':r['timestamps'],'baseline_failure':b['failure_type'],'timestamp_failure':r['failure_type'],'outcome':outcome,'baseline_input_tokens':b['input_tokens'],'timestamp_input_tokens':r['input_tokens'],'timestamp_image_groups':r['input_audit']['image_groups']})
  for ds in sorted({r['dataset'] for r in transformed}):
   ts=[r for r in transformed if r['dataset']==ds];br=[bs[r['sample_id']] for r in ts];ps=[p for p in pairs if p['model_key']==m and p['dataset']==ds];bm,tm=metrics(br),metrics(ts)
   clusters=collections.defaultdict(list)
   for p in ps:clusters[p['video_id']].append(p)
   vals=list(clusters.values());rng=np.random.default_rng(20261007);boot=[]
   for _ in range(2000):
    chosen=[p for j in rng.integers(0,len(vals),len(vals)) for p in vals[j]]
    boot.append(100*np.mean([int(p['timestamp_iou']>=.5)-int(p['baseline_iou']>=.5) for p in chosen]))
   matrix=[[sum(p['baseline_failure']==a and p['timestamp_failure']==b for p in ps) for b in LABELS] for a in LABELS]
   groups[f'{m}/{ds}']={'n':len(ts),'expected':100,'baseline':bm,'timestamp':tm,'delta_R1_05_pp':tm['R1@0.5']-bm['R1@0.5'],'delta_mIoU_pp':tm['mIoU']-bm['mIoU'],'outcomes':dict(collections.Counter(p['outcome'] for p in ps)),'paired_cluster_bootstrap_95pct_R1_delta_pp':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_seed':20261007,'bootstrap_replicates':2000,'transition_labels':LABELS,'transition_matrix':matrix,'mean_input_tokens':{'baseline':float(np.mean([p['baseline_input_tokens'] for p in ps])),'timestamp':float(np.mean([p['timestamp_input_tokens'] for p in ps]))}}
 complete=len(groups)==2 and all(g['n']==100 for g in groups.values())
 summary={'complete':complete,'baseline_root':str(BASE),'timestamp_root':str(ROOT),'paired_queries_per_model':200,'comparison_scope':'combined input conversion: duplicate-even frames, independent-image modality, explicit seconds timestamps, timestamp prompt prefix; unchanged weights and common queries. Qwen2.5 only; Qwen3 is retained only as a native-input reference in the separate setting summary.','groups':groups}
 write_result_text((relocate_path(DEST/'baseline_comparison.json')), json.dumps(summary,indent=2));write_result_text((relocate_path(DEST/'baseline_paired_cases.json')), json.dumps(pairs,ensure_ascii=False,indent=2))
 if pairs:
  with (relocate_path(DEST/'baseline_paired_cases.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
  fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(len(groups));names=[k.replace('/','\n').replace('-timelens','') for k in groups]
  for ax,metric in zip(axes,['R1@0.5','mIoU']):
   for off,key,c in [(-.18,'baseline','#8b949e'),(.18,'timestamp','#3c82ae')]:
    bars=ax.bar(x+off,[g[key][metric] for g in groups.values()],.36,label=key,color=c);ax.bar_label(bars,fmt='%.1f',fontsize=8)
   ax.set_xticks(x,names,fontsize=8);ax.set_ylim(0,100);ax.set_title(metric+' (%)');ax.legend()
  fig.suptitle('TimeLens-style timestamp conversion vs native baseline | '+('complete' if complete else 'PARTIAL'));fig.tight_layout();fig.savefig(relocate_path(DEST/'baseline_comparison.png'),dpi=170);fig.savefig(relocate_path(DEST/'baseline_comparison.pdf'));plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(15,6))
  for ax,(k,g) in zip(axes.flat,groups.items()):
   arr=np.array(g['transition_matrix']);ax.imshow(arr,cmap='Blues');ax.set_xticks(range(len(LABELS)),LABELS,rotation=45,ha='right',fontsize=8);ax.set_yticks(range(len(LABELS)),LABELS,fontsize=8);ax.set_title(k);ax.set(xlabel='Timestamp conversion',ylabel='Baseline')
   for i in range(len(LABELS)):
    for j in range(len(LABELS)):
     if arr[i,j]:ax.text(j,i,str(arr[i,j]),ha='center',va='center',color='white' if arr[i,j]>arr.max()/2 else 'black')
  fig.tight_layout();fig.savefig(relocate_path(DEST/'failure_transitions.png'),dpi=150);fig.savefig(relocate_path(DEST/'failure_transitions.pdf'));plt.close(fig)
 print(json.dumps({'complete':complete,'groups':{k:{'n':v['n'],'delta_R1_05_pp':v['delta_R1_05_pp'],'outcomes':v['outcomes']} for k,v in groups.items()}},indent=2))
if __name__=='__main__':main()
