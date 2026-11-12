"""Detached cumulative attribution supervision with durable failure logs."""
import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
from greedy_attribution_core import *

def launch():
    out=storage();prior=out/'metadata/launcher.json'
    if prior.exists():
        pid=read(prior)['pid']
        try:
            if b'greedy_attribution_manage.py' in Path(f'/proc/{pid}/cmdline').read_bytes():
                raise RuntimeError(f'Already running supervisor {pid}')
        except FileNotFoundError:pass
    smoke=read(out/'validation/gpu_smoke.json');assert smoke['passed'] and smoke.get('config_hash')==digest(CONFIG)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'_lib'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'logs/run.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'supervise'],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(prior,dict(pid=child.pid,detached=True,log=str(out/'logs/run.log'),timestamp=stamp()))
    print(json.dumps(read(prior),ensure_ascii=False))

def supervise():
    path=BASE/'supervisor.lock'
    try:
        with path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            attempts=[]
            for attempt in range(1,4):
                start=time.monotonic()
                result=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('greedy_attribution_runtime.py'))],cwd=ROOT)
                attempts.append(dict(attempt=attempt,exit_code=result.returncode,seconds=time.monotonic()-start,timestamp=stamp()))
                write(storage()/'metadata/supervision.json',dict(attempts=attempts))
                if result.returncode==0:return
                status=read(storage()/'metadata/status.json');error=status.get('error','')
                retry=result.returncode in (-9,137) or 'out of memory' in error.lower()
                if not retry or attempt==3:raise RuntimeError(f'Attribution stopped ({result.returncode}); see metadata/status.json')
                print('Retrying this worker after resource failure',attempt,flush=True);time.sleep(10)
    finally:
        path.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','supervise']);a=p.parse_args();globals()[a.action]()
