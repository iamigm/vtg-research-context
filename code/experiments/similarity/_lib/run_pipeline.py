"""Logged, resumable cohort pipeline with automatic reports and validation."""
from result_layout import relocate_path
import concurrent.futures
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from prepare import ROOT,VIEW,OUT,MANIFEST,video_key,write, RUN

def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

def log(message):print(stamp(),message,flush=True)

def link_tree(source,target):
    if not source.exists():return
    target.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        dest=target/p.name
        if p.is_dir():link_tree(p,dest)
        elif p.is_file() and p.suffix!='.tmp' and not dest.exists():os.link(p,dest)

def seed_pilot(rows):
    pilot=relocate_path(ROOT/'artifacts/runs/pilot_300_v1')
    if OUT==pilot or not pilot.exists():return {}
    wanted={video_key(s['video_path']) for s in rows}
    started=time.monotonic();count=0
    for folder in (pilot/'frames').iterdir():
        if folder.name not in wanted:continue
        link_tree(folder,OUT/'frames'/folder.name)
        count+=1
        if count%50==0:log(f'cache reuse frames {count} videos')
    counts={}
    for model in ['clip','qwen']:
        config=json.loads((pilot/model/'config.json').read_text())
        assert config['qwen_max_pixels']==262144 and config['model']==model
        counts[model]=0
        dest=OUT/model/'video_embeddings';dest.mkdir(parents=True,exist_ok=True)
        for p in (pilot/model/'video_embeddings').glob('*.npz'):
            if p.stem in wanted:
                if not (dest/p.name).exists():os.link(p,dest/p.name)
                counts[model]+=1
    result=dict(videos=count,model_video_embeddings=counts,source=str(pilot),
                seconds=time.monotonic()-started)
    log('cache reuse complete '+json.dumps(result))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--postprocess-only',action='store_true',help='Resume CPU scores/reports using completed embeddings; never load GPU weights')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    lock=(logs/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=json.loads(MANIFEST.read_text())
    assert rows and len({s['sample_key'] for s in rows})==len(rows)
    assert all(Path(s['video_path']).is_file() for s in rows)
    paths=list(dict.fromkeys(s['video_path'] for s in rows));keys={video_key(p) for p in paths}
    total=len(keys);datasets=dict(Counter(s['dataset'] for s in rows))
    write(OUT/'samples.json',rows)
    status=dict(run=RUN,stage='starting',samples=len(rows),videos=total,datasets=datasets,
                started_at=stamp(),manifest=str(MANIFEST),
                manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),pid=os.getpid())
    write(logs/'status.json',status)
    log(f'START {RUN}: {len(rows)} queries, {total} unique videos, datasets={datasets}')
    started=time.monotonic();children={};handles=[]
    try:
        status['cache_reuse']=seed_pilot(rows)
        base={m:len(list((OUT/m/'video_embeddings').glob('*.npz'))) for m in ['clip','qwen']}
        def launch(name,args):
            handle=(logs/f'{name}.log').open('a',buffering=1);handles.append(handle)
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'_lib'/args[0]),*args[1:]],
                               stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            children[name]=p;log(f'LAUNCH {name} pid={p.pid} log={logs/name}.log')
            return p
        if args.postprocess_only:
            assert all((OUT/'frames'/key/'metadata.json').exists() for key in keys)
            assert all((OUT/m/'video_embeddings'/f'{key}.npz').exists() for m in ['clip','qwen'] for key in keys)
            log('CPU-ONLY RECOVERY: all embeddings complete; GPU inference is skipped')
            for model in ['clip','qwen']:
                launch(model,['score.py','--model',model])
        else:
            launch('prepare',['prepare.py'])
            launch('clip',['infer.py','--model','clip','--batch-size','256','--workers','1'])
            launch('qwen',['infer.py','--model','qwen','--batch-size','128','--workers','3'])
        last_progress=None;last_change=time.monotonic()
        while True:
            for name,p in children.items():
                code=p.poll()
                if code is not None and code!=0:raise RuntimeError(f'{name} failed, exit={code}; see {logs/name}.log')
            counts={};model_states={}
            for model in ['clip','qwen']:
                available={p.stem for p in (OUT/model/'video_embeddings').glob('*.npz')}&keys
                per_dataset=Counter(s['dataset'] for s in rows if video_key(s['video_path']) in available)
                done=len(available);elapsed=time.monotonic()-started
                new=done-base[model]
                eta=(total-done)*elapsed/new if new>0 else None
                counts[model]=dict(videos=done,total_videos=total,percent=100*done/total,
                    query_pairs_with_embeddings=dict(per_dataset),
                    eta_embedding_seconds=eta,eta_scope='remaining distinct video embeddings; rough since launch; excludes report/validation')
                p=OUT/model/'status.json'
                if p.exists():model_states[model]=json.loads(p.read_text())
                counts[model]['scored_samples']=model_states.get(model,{}).get('completed_samples',0)
            prepared=len(list((OUT/'frames').glob('*/metadata.json')))
            status.update(stage='scoring' if args.postprocess_only else 'inference',prepared_videos=prepared,models=counts,
                model_states=model_states,elapsed_seconds=time.monotonic()-started,updated_at=stamp())
            write(logs/'status.json',status)
            log(f'PROGRESS prepare={prepared}/{total} elapsed={status["elapsed_seconds"]:.0f}s models='+json.dumps(counts))
            progress=(prepared,*[counts[m]['videos'] for m in ['clip','qwen']],
                      *[model_states.get(m,{}).get('completed_samples',0) for m in ['clip','qwen']])
            if progress!=last_progress:last_progress=progress;last_change=time.monotonic()
            if time.monotonic()-last_change>1800:raise RuntimeError('No preparation, embedding or scoring progress for 30 minutes; stopped rather than leaving a hung job.')
            if all(p.poll()==0 for p in children.values()):break
            time.sleep(20)
        status.update(stage='report_and_validation',updated_at=stamp());write(logs/'status.json',status)
        p=launch('finalize',['finalize.py'])
        while p.poll() is None:
            log('POSTPROCESS running: see finalize.log for report / quantitative / browser validation')
            time.sleep(20)
        if p.returncode:raise RuntimeError(f'finalize failed, exit={p.returncode}; see {logs}/finalize.log')
        status.update(stage='complete',updated_at=stamp(),elapsed_seconds=time.monotonic()-started)
        write(logs/'status.json',status);log('ALL COMPLETE '+str(VIEW))
    except BaseException as e:
        for p in children.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        status.update(stage='failed',error=repr(e),updated_at=stamp(),elapsed_seconds=time.monotonic()-started)
        write(logs/'status.json',status);log('FAILED '+repr(e))
        raise
    finally:
        for h in handles:h.close()

if __name__=='__main__':main()
