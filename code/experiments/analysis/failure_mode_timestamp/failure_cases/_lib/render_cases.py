"""Render auditable video contact sheets plus GT/prediction timelines."""
from result_layout import relocate_path, write_result_text
import argparse,json,hashlib,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from decord import VideoReader,cpu
ROOT=relocate_path(Path(__file__)).resolve().parents[1]

def render(sid,rows,out):
    r=rows[0];vr=VideoReader(r['video_path'],ctx=cpu(0),num_threads=1);fps=vr.get_avg_fps();duration=len(vr)/fps
    times=np.linspace(0,max(0,duration-1/fps),20)
    frame_ids=np.minimum((times*fps).round().astype(int),len(vr)-1);frames=vr.get_batch(frame_ids.tolist()).asnumpy();times=frame_ids/fps
    fig=plt.figure(figsize=(16,12));gs=fig.add_gridspec(6,5,height_ratios=[1,1,1,1,.75,.7])
    gt=r['span'][0]
    for i,(frame,t) in enumerate(zip(frames,times)):
        ax=fig.add_subplot(gs[i//5,i%5]);ax.imshow(frame);ax.axis('off')
        marks=['GT'] if gt[0]<=t<=gt[1] else []
        for row in rows:
            if row['timestamps'] and row['timestamps'][0][0]<=t<=row['timestamps'][0][1]:pass
        ax.set_title(f'{t:.2f}s | '+', '.join(marks),fontsize=9)
    ax=fig.add_subplot(gs[4,:]);ax.broken_barh([(gt[0],gt[1]-gt[0])],(-.3,.6),facecolors='#33834a',label='GT')
    for i,row in enumerate(rows,1):
        if row['timestamps']:
            s,e=row['timestamps'][0];ax.broken_barh([(s,e-s)],(i-.3,.6),facecolors={'qwen25':'#c47833','qwen3':'#387da5'}[row['model_key']])
    ax.set_yticks(range(len(rows)+1),['GT']+[f'{x.get("display_label",x["model_key"])} IoU={x["iou"]:.3f}' for x in rows]);ax.set_xlim(0,max(duration,gt[1],*[x['timestamps'][0][1] for x in rows if x['timestamps']])+1);ax.axvline(duration,c='gray',ls='--');ax.set_xlabel('Time (seconds); dashed line = decoded video duration')
    ax=fig.add_subplot(gs[5,:]);ax.axis('off');text='\n'.join(f'{x.get("display_label",x["model_key"])}: {x["answer"][:220]} [{x["failure_type"]}]' for x in rows);ax.text(0,1,text,va='top',wrap=True,fontsize=10)
    fig.suptitle(sid+'\n'+r['query']+'\nUniform review frames (not necessarily model input frames)',fontsize=12);fig.subplots_adjust(left=.17,right=.99,top=.90,bottom=.04,hspace=.42);key=sid.replace('/','__');fig.savefig(relocate_path(out/f'{key}.jpg'),dpi=120);fig.savefig(relocate_path(out/f'{key}.pdf'));plt.close(fig)
    meta={'sample_id':sid,'query':r['query'],'video_path':r['video_path'],'annotation_duration':r['duration'],'decoded_duration':duration,'gt':gt,'review_frame_indices':frame_ids.tolist(),'review_timestamps':times.tolist(),'models':[{k:x[k] for k in ['model_key','variant','display_label','timestamps','iou','failure_type','answer']} for x in rows],'review_note':'Uniform contact sheet for inspection; semantic conclusions may require denser frames/full video.'}
    write_result_text((relocate_path(out/f'{key}.json')), json.dumps(meta,indent=2));return meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='pilot_100');ap.add_argument('--sample-id',action='append');ap.add_argument('--per-type',type=int,default=1);a=ap.parse_args()
    dest=relocate_path(ROOT/'reports'/a.run);rows=json.loads((relocate_path(dest/'case_metrics.json')).read_text());out=relocate_path(dest/'cases');out.mkdir(exist_ok=True)
    ids=set(a.sample_id or [])
    if not ids:
        review=relocate_path(dest/'manual_review.json')
        if review.exists():ids.update(r['sample_id'] for r in json.loads(review.read_text()))
        for model in ['qwen25','qwen3']:
            for ds in ['charades-timelens','qvhighlights-timelens']:
                for label in ['no_overlap','overextended','underextended','shifted']:
                    subset=sorted([r for r in rows if r['model_key']==model and r['dataset']==ds and r['failure_type']==label],key=lambda r:(r['iou'],r['sample_id']))
                    ids.update(r['sample_id'] for r in subset[:a.per_type])
    selected=[]
    for sid in sorted(ids):
        group=[r for r in rows if r['sample_id']==sid]
        if group:selected.append(render(sid,group,out));print(sid,flush=True)
    write_result_text((relocate_path(out/'selection.json')), json.dumps({'selection':'Lowest tIoU case in each model/dataset/geometric failure type; ties resolved by sample_id, unless explicit sample IDs supplied. Additional manually reviewed cases are included. Cases illustrate failures and are not a frequency sample.','cases':selected},indent=2))
    entries=[]
    for r in selected:
        name=r['sample_id'].replace('/','__')
        entries.append('<article><h2>'+html.escape(r['sample_id'])+'</h2><p>'+html.escape(r['query'])+'</p><a href="'+name+'.jpg"><img loading="lazy" src="'+name+'.jpg" width="1100"></a><p><a href="'+name+'.pdf">PDF</a> · <a href="'+name+'.json">Metadata</a></p></article>')
    write_result_text((relocate_path(out/'report.html')), '<!doctype html><meta charset="utf-8"><title>Reviewed failure cases</title><style>body{font:16px sans-serif;margin:24px}article{margin:35px 0;border-bottom:1px solid #ccc}img{max-width:100%}</style><h1>Failure case contact sheets</h1><p><a href="../report.html">All query metrics</a>. Review frames are sampled for inspection and may differ from the model input frames.</p>'+''.join(entries))
if __name__=='__main__':main()
