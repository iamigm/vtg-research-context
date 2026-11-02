"""Analyze the three retained settings; only Qwen2.5 has converted input."""
from result_layout import relocate_path, write_result_text
import collections,csv,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import measure
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT.parents[1]/'failure_mode_baseline/failure_cases');DEST=relocate_path(ROOT/'reports/pilot_100')
SETTINGS=[('qwen25_native','Qwen2.5 native',BASE,'qwen25','baseline'),('qwen25_timelens7b_input','Qwen2.5 + TimeLens-7B input',ROOT,'qwen25','timestamp'),('qwen3_native','Qwen3 native (text timestamps)',BASE,'qwen3','baseline')]
DATASETS=['charades-timelens','qvhighlights-timelens']
def summary(rows):
 return {'n':len(rows),'mIoU':100*float(np.mean([r['iou'] for r in rows])),**{f'R1@{t}':100*float(np.mean([r['iou']>=t for r in rows])) for t in [.3,.5,.7]},'failure_counts':dict(collections.Counter(r['failure_type'] for r in rows)),'out_of_bounds':sum(r['out_of_bounds'] for r in rows),'out_of_bounds_over_1_second':sum(r['out_of_bounds_excess_seconds']>1 for r in rows),'rounding_flips_05':sum(r['rounding_flips_05'] for r in rows)}
def main():
 expected=json.loads((relocate_path(ROOT/'data/samples_100.json')).read_text());ids={r['sample_id'] for r in expected};rows=[];groups={}
 for key,label,root,model,variant in SETTINGS:
  current=[]
  for r in map(json.loads,(relocate_path(root/'results/pilot_100'/model/'predictions.jsonl')).read_text().splitlines()):
   r.update(measure(r));r.update(setting=key,setting_label=label,model_key=model,variant=variant);current.append(r)
  assert len(current)==200 and {r['sample_id'] for r in current}==ids
  rows.extend(current)
  for ds in DATASETS:groups[key+'/'+ds]=summary([r for r in current if r['dataset']==ds])
 converted=[r for r in rows if r['variant']=='timestamp']
 for name,content in [('case_metrics',converted),('setting_case_metrics',rows)]:
  write_result_text((relocate_path(DEST/(name+'.json'))), json.dumps(content,ensure_ascii=False,indent=2))
  fields=['setting','sample_id','model_key','dataset','query','span','timestamps','answer','iou','raw_iou','failure_type','start_error','end_error','out_of_bounds']
  with (relocate_path(DEST/(name+'.csv'))).open('w') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(content)
 converted_groups={'qwen25/'+ds:summary([r for r in converted if r['dataset']==ds]) for ds in DATASETS}
 write_result_text((relocate_path(DEST/'summary.json')), json.dumps({'complete':True,'scope':'Qwen2.5 TimeLens-7B-style input only','groups':converted_groups},indent=2))
 write_result_text((relocate_path(DEST/'setting_summary.json')), json.dumps({'complete':True,'settings':[{'key':x[0],'label':x[1]} for x in SETTINGS],'groups':groups},indent=2))
 fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(2)
 for ax,metric in zip(axes,['R1@0.5','mIoU']):
  for i,(key,label,*_) in enumerate(SETTINGS):
   bars=ax.bar(x+(i-1)*.25,[groups[key+'/'+ds][metric] for ds in DATASETS],.25,label=label,color=['#929ba5','#b96518','#276abe'][i]);ax.bar_label(bars,fmt='%.1f',fontsize=9)
  ax.set_xticks(x,['Charades-TimeLens','QVHighlights-TimeLens']);ax.set_ylim(0,100);ax.set_title(metric+' (%)');ax.legend(fontsize=8)
 fig.suptitle('Three retained zero-shot settings | same 100 queries per dataset');fig.tight_layout();fig.savefig(relocate_path(DEST/'summary.png'),dpi=160);fig.savefig(relocate_path(DEST/'summary.pdf'));plt.close(fig)
 entries=[]
 for r in sorted(rows,key=lambda r:(r['dataset'],r['iou'])):
  values=[r['setting_label'],r['sample_id'],r['query'],str(r['span'][0]),str(r['timestamps']),f'{r["iou"]:.3f}',r['failure_type']]
  entries.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
 write_result_text((relocate_path(DEST/'report.html')), '<!doctype html><meta charset="utf-8"><title>Three-setting failure analysis</title><style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}input{padding:10px;width:500px}</style><h1>Qwen2.5 native / TimeLens-7B input · Qwen3 native</h1><p><a href="cases/report.html">Video gallery</a> · <a href="REPORT.md">Report</a> · <a href="setting_case_metrics.csv">CSV</a></p><img src="summary.png" width="1100"><p><input id="search" placeholder="Search query, model, or failure type"></p><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in ['Setting','Sample','Query','GT','Prediction','tIoU','Failure'])+'</tr></thead><tbody>'+''.join(entries)+'</tbody></table><script>document.querySelector("#search").oninput=e=>document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>')
 print(json.dumps({'complete':True,'groups':groups},indent=2))
if __name__=='__main__':main()
