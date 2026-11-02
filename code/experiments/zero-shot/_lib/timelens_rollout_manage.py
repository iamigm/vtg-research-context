"""Detached launcher with live progress, bounded retries, and automatic reports."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import json,os,subprocess,sys,time,traceback,fcntl
from datetime import datetime,timezone
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/timelens_rollout');PYTHON=str(relocate_path(ROOT/'.venv/bin/python'))
def read(p):return json.loads(p.read_text())
def write(p,r):
    tmp=p.with_suffix('.tmp');write_result_text(tmp, json.dumps(r,indent=2)+'\n');tmp.replace(p)
def main():
    started=time.time();config=read(relocate_path(OUT/'config.json'));samples=read(relocate_path(OUT/'samples.json'));jobs=[];logs=[];total=600*2*config['rollouts_per_variant'];attempts={};active_stage='starting'
    def progress(state='running',**extra):
        counts={m:{v:{ds:0 for ds in {s['dataset'] for s in samples}} for v in ['top_k','top_p']} for m in ['qwen25','qwen3']}
        for m in counts:
            for p in result_glob((relocate_path(OUT/'records'/m)), '*/*.json'):
                ds=p.parent.name.rsplit('_',1)[0];v=p.stem.rsplit('_',1)[0];counts[m][v][ds]+=1
        done=sum(n for modes in counts.values() for datasets in modes.values() for n in datasets.values());elapsed=time.time()-started
        r=dict(state=state,stage=active_stage,timestamp=datetime.now(timezone.utc).isoformat(),completed=done,total=total,percent=round(100*done/total,2),elapsed_seconds=round(elapsed),inference_eta_seconds=round(elapsed*(total-done)/done) if done else None,counts=counts,**extra)
        write(relocate_path(OUT/'status.json'),r)
        with (relocate_path(OUT/'logs/progress.log')).open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    try:
        # Avoid colliding with an unrelated process if the GPU was occupied after launch.
        active_stage='waiting for GPU memory (110 GiB free required for four workers)'
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free>=110*1024:break
            progress(free_gpu_mib=free);time.sleep(15)
        env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY='/tmp/timelens_rollout_mps',CUDA_MPS_LOG_DIRECTORY=str(relocate_path(OUT/'logs/mps')),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        relocate_path(Path(env['CUDA_MPS_PIPE_DIRECTORY'])).mkdir(exist_ok=True);relocate_path(Path(env['CUDA_MPS_LOG_DIRECTORY'])).mkdir(exist_ok=True)
        subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
        def launch(model,shard):
            log=(relocate_path(OUT/'logs'/f'{model}_{shard}.log')).open('a');logs.append(log)
            p=subprocess.Popen([PYTHON,str(relocate_path(ROOT/'_lib/timelens_rollout.py')),'--model',model,'--shard',str(shard),'--shards','2'],stdout=log,stderr=subprocess.STDOUT,env=env)
            return dict(model=model,shard=shard,process=p)
        jobs=[launch(m,s) for m in ['qwen25','qwen3'] for s in range(2)]
        write(relocate_path(OUT/'execution.json'),dict(supervisor_pid=os.getpid(),started_epoch=started,jobs=[dict(model=j['model'],shard=j['shard'],pid=j['process'].pid) for j in jobs],mps_pipe=env['CUDA_MPS_PIPE_DIRECTORY']))
        active_stage='inference';progress()
        while any(j['process'].poll() is None for j in jobs):progress();time.sleep(15)
        # Retry failed shards once from their atomic per-rollout checkpoints.
        failed=[j for j in jobs if j['process'].returncode]
        if failed:
            active_stage='retrying failed shards';progress(failed_shards=[(j['model'],j['shard']) for j in failed])
            for j in failed:
                retry=launch(j['model'],j['shard'])
                while retry['process'].poll() is None:progress();time.sleep(15)
                if retry['process'].returncode:raise RuntimeError(f"{j['model']} shard {j['shard']} failed twice; see worker log")
        active_stage='report generation and validation';progress()
        for script in ['timelens_rollout_report.py','timelens_rollout_browser.py']:
            p=subprocess.Popen([PYTHON,str(relocate_path(ROOT/'_lib'/script))])
            while p.poll() is None:progress();time.sleep(15)
            if p.returncode:raise RuntimeError(f'{script} failed: exit {p.returncode}')
        active_stage='complete';progress('complete',html='report.html',markdown='REPORT.md')
    except Exception:progress('failed',error=traceback.format_exc());raise
    finally:
        # Own MPS server only; do not touch other workloads.
        if jobs and all(j['process'].poll() is not None for j in jobs):subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,check=False)
        for log in logs:log.close()
if __name__=='__main__':
    with (relocate_path(OUT/'supervisor.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
