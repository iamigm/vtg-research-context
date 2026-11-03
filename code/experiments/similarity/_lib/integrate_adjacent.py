"""Update existing runs in place; no new experiment or model inference."""
from result_layout import relocate_path
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import torch
from adjacent_similarity import attach,precompute_many,markdown

ROOT=Path('/home/irteam/experiments/similarity')
def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def update(run):
    out=relocate_path(ROOT/'artifacts/runs'/run);view=relocate_path(ROOT/'results'/run)
    status=out/'logs/adjacent_status.json';start=time.monotonic();last=[0]
    def progress(stage,done,total,force=False):
        now=time.monotonic()
        if not force and done!=total and now-last[0]<10:return
        state=dict(run=run,stage=stage,completed=done,total=total,percentage=100*done/total if total else 0,
            elapsed_seconds=now-start,updated_at=stamp(),backend='CUDA batched cosine; 4 parallel ZIP readers')
        write(status,state);print(stamp(),json.dumps(state),flush=True);last[0]=now
    progress('loading_existing_report',0,1,True)
    path=view/'report.html';original=path.read_text()
    match=re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)',original,re.S);assert match
    payload=json.loads(match.group(2));samples=payload['samples']
    old_summary=json.dumps(payload['summary'],sort_keys=True)
    unique={s['video_key']:s for s in samples}
    dense=(out/'embeddings').exists()
    for model in ['clip','qwen']:
        paths=[out/'embeddings'/model/f'{v}.npz' if dense else out/model/'video_embeddings'/f'{v}.npz' for v in unique]
        specs=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
        progress('gpu_adjacent_'+model,0,len(paths),True)
        precompute_many(specs,lambda n,total:progress('gpu_adjacent_'+model,n,total))
    jobs=[(s,model) for s in samples for model in ['clip','qwen']]
    def work(job):
        s,model=job
        values,stats=attach(out,model,s['video_key'],s['sample_key'],s['timestamps'])
        return s,model,values,stats
    records=[];lookup={}
    progress('integrating_existing_scores',0,len(jobs),True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(s,model,values,stats) in enumerate(pool.map(work,jobs),1):
            s['models'][model]['adjacent_scores']=values
            s['models'][model]['metrics']['adjacent_similarity']=stats
            lookup[model,s['sample_key']]=stats
            records.append(dict(model=model,dataset=s['dataset'],sample_key=s['sample_key'],adjacent_similarity=stats))
            progress('integrating_existing_scores',i,len(jobs))
    assert json.dumps(payload['summary'],sort_keys=True)==old_summary
    template=(ROOT/'_lib/viewer.html').read_text()
    plot=re.search(r'^function plot\(.*?(?=^function histogram\()',template,re.M|re.S).group()
    updated=re.sub(r'^function plot\(.*?(?=^function histogram\()',lambda m:plot,original,flags=re.M|re.S)
    card=re.search(r'<section class="card" id="adjacentCard".*?</section>',template,re.S).group()
    if 'id="adjacentCard"' not in updated:
        anchor='<div class="grid"><section class="card"><h3>CLIP · GT 안/밖 score histogram</h3>'
        assert anchor in updated;updated=updated.replace(anchor,card+'\n'+anchor,1)
    updated=updated.replace("for(const id of ['raw','mass'])","for(const id of ['raw','mass','adjacent'])")
    updated=updated.replace("$('videoCard').hidden=true;return}","$('videoCard').hidden=true;$('adjacentCard').hidden=true;return}")
    updated=updated.replace('loadVideo(s)}','loadVideo(s);renderAdjacent()}')
    blob=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    updated=re.sub(r'(<script id="data" type="application/json">).*?(</script>)',lambda m:m.group(1)+blob+m.group(2),updated,flags=re.S)
    temp=path.with_suffix('.tmp');temp.write_text(updated);temp.replace(path)
    report=view/'REPORT.md';doc=report.read_text();heading='## 인접 frame embedding 유사도'
    if heading in doc:doc=doc[:doc.index(heading)].rstrip()+'\n'
    report.write_text(doc+'\n'+markdown(records))
    metrics=out/'sample_metrics.json'
    if metrics.exists():
        values=json.loads(metrics.read_text())
        for r in values:r['adjacent_similarity']=lookup[r['model'],r['sample_key']]
        write(metrics,values)
    csv_path=out/'measurements/per_sample.csv' if dense else out/'sample_metrics.csv'
    if csv_path.exists():
        with csv_path.open() as f:reader=csv.DictReader(f);cols=list(reader.fieldnames);rows=list(reader)
        for col in ['adjacent_pairs','adjacent_mean_cosine','adjacent_min_cosine']:
            if col not in cols:cols.append(col)
        for row in rows:
            stats=lookup.get((row['model'],row['sample_key']),{})
            row.update(adjacent_pairs=stats.get('n_pairs',''),adjacent_mean_cosine=stats.get('mean_cosine',''),adjacent_min_cosine=stats.get('min_cosine',''))
        with csv_path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    progress('browser_validation',len(samples),len(samples),True)
    env=dict(os.environ,SIMILARITY_RUN=run,OPENBLAS_NUM_THREADS='1')
    code="import sys;sys.path.insert(0,'/home/irteam/experiments/similarity/_lib');import check_viewer;"
    if dense:code+="check_viewer.OUT=check_viewer.OUT/'measurements';"
    code+='check_viewer.main()'
    subprocess.run([sys.executable,'-c',code],env=env,check=True)
    subprocess.run([sys.executable,'/home/irteam/experiments/_lib/build_index.py'],check=True)
    progress('complete',len(samples),len(samples),True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',default=['pilot_300_v1','pilot_300_16fps_cost_v1','full_9404_v1']);args=parser.parse_args()
    assert torch.cuda.is_available(),'CUDA is required for this integration run'
    import fcntl
    folder=ROOT/'artifacts/common/logs';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'adjacent_integration.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    print(stamp(),'GPU',torch.cuda.get_device_name(),'runs',args.runs,flush=True)
    for run in args.runs:
        try:update(run)
        except BaseException as e:
            status=relocate_path(ROOT/'artifacts/runs'/run/'logs/adjacent_status.json')
            write(status,dict(run=run,stage='failed',error=repr(e),updated_at=stamp()))
            raise

if __name__=='__main__':main()
