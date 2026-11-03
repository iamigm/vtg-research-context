"""Paired metrics for the fixed TimeLens restricted-head pilot."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import collections
import numpy as np

ROOT=relocate_path(Path(__file__)).resolve().parents[1]
sys.path.insert(0,str(relocate_path(ROOT/'runtime/timelens_deps')))
sys.path.insert(0,str(relocate_path(ROOT/'vendor/TimeLens')))
from timelens.utils import extract_time,iou
OUT=relocate_path(ROOT/'results/uncertainty/span_head/pilot/300')
from timelens_paths import QWEN3_ZERO, PILOT_SAMPLES, PILOT_SELECTION, pilot_export
BASE=QWEN3_ZERO
DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens']


def read(p):return json.loads(p.read_text())
def write(p,x):write_result_text(p, json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def stats(rows):
    return dict(n=len(rows),mIoU=100*float(np.mean([r['iou'] for r in rows])),**{f'R1@{t}':100*float(np.mean([r['iou']>=t for r in rows])) for t in [.3,.5,.7]},parse_failures=sum(not extract_time(r['answer']) for r in rows),invalid_intervals=sum(r['timestamps'][0][0]>=r['timestamps'][0][1] for r in rows),multiple_spans=sum(len(r['timestamps'])>1 for r in rows),out_of_bounds=sum(r['timestamps'][0][0]<0 or r['timestamps'][0][1]>r['duration'] for r in rows),truncated=sum(r['truncated'] for r in rows),mean_output_tokens=float(np.mean([len(r['generated_ids']) for r in rows])))


def paired_ci(rows):
    byvideo=collections.defaultdict(list)
    for r in rows:byvideo[r['video_name']].append(r)
    groups=list(byvideo.values());rng=np.random.default_rng(42);values=[]
    for _ in range(2000):
        draw=[r for i in rng.integers(0,len(groups),len(groups)) for r in groups[i]]
        values.append([100*np.mean([r['delta_iou'] for r in draw]),100*np.mean([r['delta_success_05'] for r in draw])])
    lo,hi=np.percentile(values,[2.5,97.5],axis=0)
    return dict(mIoU=[float(lo[0]),float(hi[0])],R1_05=[float(lo[1]),float(hi[1])])


def main():
    samples=read(PILOT_SAMPLES);expected={(m,s['sample_key']) for m in ['qwen25','qwen3'] for s in samples}
    rows=[json.loads(line) for p in sorted(result_glob(OUT, '*/shard*/predictions.jsonl')) for line in p.read_text().splitlines()]
    keys={(r['model_key'],r['sample_key']) for r in rows}
    assert len(rows)==len(keys)
    assert keys<=expected
    complete=keys==expected
    paired=[];original=[];checks=[]
    for p in result_glob(OUT, 'qwen3/*/baseline_validation.jsonl'):checks.extend(json.loads(line) for line in p.read_text().splitlines())
    for r in rows:
        p=relocate_path(ROOT/r['baseline_path']);b=read(p)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['baseline_sha256']
        assert all(r[k]==b[k] for k in ['sample_key','query','dataset','gt','duration','order'])
        assert r['input_matches_baseline']
        parsed=extract_time(r['answer']) or [[r['duration']+10,r['duration']+20]]
        assert r['timestamps']==[[round(s),round(e)] for s,e in parsed]
        assert abs(iou(r['gt'],r['timestamps'][0])-r['iou'])<1e-12
        assert abs(iou(b['gt'],b['timestamps'][0])-b['iou'])<1e-12
        b['model_key']=r['model_key']
        if r['model_key']=='qwen25':assert b['input_ids_sha256']==r['input_ids_sha256']
        config=read(relocate_path(OUT/r['model_key']/'shard0/config.json'))
        assert set(r['generated_ids'])<={t['id'] for t in config['allowed_tokens']}
        original.append(b)
        paired.append(dict(model=r['model_key'],sample_key=r['sample_key'],dataset=r['dataset'],video_name=r['video_name'],query=r['query'],gt=r['gt'],baseline_answer=b['answer'],restricted_answer=r['answer'],baseline_timestamps=b['timestamps'],restricted_timestamps=r['timestamps'],baseline_iou=b['iou'],restricted_iou=r['iou'],delta_iou=r['iou']-b['iou'],delta_success_05=int(r['iou']>=.5)-int(b['iou']>=.5),baseline_truncated=b['truncated'],restricted_truncated=r['truncated'],baseline_tokens=len(b['generated_ids']),restricted_tokens=len(r['generated_ids'])))
    summary={}
    groups=[(m,ds) for m in ['qwen25','qwen3'] for ds in DATASETS]
    for m,ds in groups:
        n=[r for r in rows if r['dataset']==ds and r['model_key']==m];b=[r for r in original if r['dataset']==ds and r['model_key']==m];p=[r for r in paired if r['dataset']==ds and r['model']==m]
        if not n:continue
        ns,bs=stats(n),stats(b)
        summary[m+'/'+ds]=dict(baseline=bs,restricted=ns,delta_percentage_points={k:ns[k]-bs[k] for k in ['mIoU','R1@0.3','R1@0.5','R1@0.7']},failure_to_success=sum(r['delta_success_05']==1 for r in p),success_to_failure=sum(r['delta_success_05']==-1 for r in p),identical_spans=sum(r['baseline_timestamps']==r['restricted_timestamps'] for r in p),bootstrap_95ci_percentage_points=paired_ci(p))
    result=dict(complete=complete,count=len(rows),expected=600,selection=read(PILOT_SELECTION),groups=summary,qwen3_baseline_reproduction=checks,qwen25_baseline='Fresh native-input full-head zero-shot on the exact same input tensors as restricted head; model default repetition penalty 1.05 for both arms',validation=dict(original_predictions_unchanged=True,qwen3_saved_input_text_and_sampling_metadata_checked=True,qwen25_same_input_tensors_and_input_ids_hash=True,official_parser_rounding_and_iou_match=True,all_generated_token_ids_allowed=True),timing_note='Historical Qwen3 baseline includes attention capture; concurrent process timings also depend on shared GPU load. Not a controlled head speed comparison.')
    if complete:
        import time
        execution=read(relocate_path(OUT/'execution.json'))
        gpu=list(csv.DictReader((relocate_path(OUT/'logs/gpu.csv')).open()))
        old_result=read(relocate_path(OUT/'summary.json')) if (relocate_path(OUT/'summary.json')).exists() else {}
        result['runtime']=old_result.get('runtime') if old_result.get('complete') and old_result.get('runtime') else dict(elapsed_seconds=time.time()-execution['started_epoch'],cuda_mps=True,initial_parallel_replicas=4,gpu='NVIDIA H200',monitor_scope='MPS phase; includes model-loading intervals',gpu_utilization_mean=float(np.mean([float(r['utilization_percent']) for r in gpu])),gpu_utilization_max=max(float(r['utilization_percent']) for r in gpu),gpu_memory_peak_mib=max(float(r['memory_mib']) for r in gpu),gpu_power_peak_watts=max(float(r['power_watts']) for r in gpu))
        # Invoke the unmodified upstream evaluator on both exact pilot predictions.
        for m,ds in groups:
            for label,content in [('baseline',original),('restricted',rows)]:
                path=pilot_export(m,ds) if label=='baseline' else relocate_path(OUT/f'{m}_{ds}_{label}.jsonl')
                selected=[r for r in content if r['dataset']==ds and r['model_key']==m]
                write_result_text(path, ''.join(json.dumps({f"{r['video_name']}>>>{r['query']}>>>{r['gt']}":dict(timestamps=r['timestamps'],answers=r['answer'])},ensure_ascii=False)+'\n' for r in selected))
                import os,re
                env=os.environ.copy();env['PYTHONPATH']=str(relocate_path(ROOT/'vendor/TimeLens'))
                output=subprocess.check_output([sys.executable,str(relocate_path(ROOT/'vendor/TimeLens/evaluation/compute_metrics.py')),'-f',str(path)],env=env,text=True)
                for label_off,key in [('IOU 0.3','R1@0.3'),('IOU 0.5','R1@0.5'),('IOU 0.7','R1@0.7'),('mIOU','mIoU')]:
                    value=float(re.search('^'+re.escape(label_off)+r': ([\d.eE+-]+)',output,re.M)[1])
                    assert abs(value-summary[m+'/'+ds][label][key])<1e-9
        result['validation']['official_evaluator_cli_matches_all_twelve_groups']=True
    write(relocate_path(OUT/'summary.json'),result);write(relocate_path(OUT/'paired.json'),paired)
    if paired:
        with (relocate_path(OUT/'paired.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(13,9));x=np.arange(3)
        for ax,(m,key) in zip(axes.flat,[(m,k) for m in ['qwen25','qwen3'] for k in ['R1@0.5','mIoU']]):
            for offset,setting,color in [(-.18,'baseline','#788897'),(.18,'restricted','#2a75a7')]:
                bars=ax.bar(x+offset,[summary[m+'/'+ds][setting][key] for ds in DATASETS],.36,label=setting,color=color);ax.bar_label(bars,fmt='%.1f')
            ax.set_xticks(x,['Charades','ActivityNet','QVHighlights']);ax.set_ylim(0,100);ax.set_title(m+' · '+key+' (%)');ax.legend()
        fig.suptitle('TimeLens | 100 paired queries per benchmark and model');fig.tight_layout();fig.savefig(relocate_path(OUT/'comparison.png'),dpi=160);fig.savefig(relocate_path(OUT/'comparison.pdf'));plt.close(fig)
    from timelens_span_markdown import render_report
    render_report(result)
    print(json.dumps(dict(complete=complete,count=len(rows),groups=summary),indent=2))


if __name__=='__main__':
    if sys.argv[1:] == ['--render-only']:
        from timelens_span_markdown import render_report
        render_report(read(relocate_path(OUT/'summary.json')))
    else:
        main()
