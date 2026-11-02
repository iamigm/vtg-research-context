"""Wait for the existing sweep, then run two disjoint singleton GPU workers."""
import os, sys, time, signal, subprocess, fcntl, json
from pathlib import Path
from fps_spatial_control import ROOT, OUT, REPORT, RUN, read, write, stamp
DEPENDENCY=ROOT/'artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json'

def state(**kw):
    write(OUT/'metadata/status.json',dict(timestamp=stamp(),**kw))
    with (OUT/'logs/supervisor.log').open('a') as f:f.write(json.dumps(dict(timestamp=stamp(),**kw),ensure_ascii=False)+'\n')

def finalize():
    from fps_spatial_report import build
    build()
    from artifact_store import register
    for kind,path in [('predictions',OUT/'responses'),('inputs',OUT/'inputs'),('manifests',OUT/'metadata'),('analyses',REPORT)]:
        register(RUN+'.'+kind,kind,path,'GT-blind zero-shot fixed spatial grid FPS sweep','_lib/fps_spatial_control.py','pilot_300','complete',['nested_pilot_v1.300.cohort'])
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)

def main():
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    with (OUT/'metadata/supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic()
        while True:
            dependency=read(DEPENDENCY) if DEPENDENCY.exists() else {}
            smoke=OUT/'validation/smoke.json'
            passed=smoke.exists() and read(smoke).get('passed') is True
            if dependency.get('state')=='complete' and dependency.get('stage')=='complete' and passed:break
            state(state='waiting',stage='waiting_for_previous_experiment',completed=0,total=6000,percentage=0,
                  dependency=str(DEPENDENCY),dependency_state=dependency.get('state'),dependency_stage=dependency.get('stage'),
                  dependency_completed=dependency.get('completed'),dependency_total=dependency.get('total'),smoke_passed=passed,
                  elapsed_seconds=time.monotonic()-start,eta_seconds=None,eta_scope='waiting on dependency; no inference ETA yet',counts={})
            time.sleep(20)
        start=time.monotonic();children=[];logs=[]
        try:
            for rank in range(2):
                log=(OUT/'logs'/f'worker{rank}.log').open('ab',buffering=0);logs.append(log)
                p=subprocess.Popen([sys.executable,'-u',str(ROOT/'_lib/fps_spatial_control.py')],cwd=ROOT,
                                   env=dict(os.environ,FPS_WORKER_RANK=str(rank)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(p)
            write(OUT/'metadata/processes.json',dict(timestamp=stamp(),supervisor=os.getpid(),workers=[p.pid for p in children],partition='SHA256(video_path) mod 2',dependency_satisfied=True))
            pressure_since=None
            while True:
                states=[read(OUT/'metadata'/f'status_worker{r}.json') for r in range(2) if (OUT/'metadata'/f'status_worker{r}.json').exists()]
                done=sum(s.get('completed',0) for s in states);counts={}
                for s in states:
                    for k,v in s.get('counts',{}).items():counts[k]=counts.get(k,0)+v
                elapsed=time.monotonic()-start
                state(state='running',stage='parallel_generation',completed=done,total=6000,percentage=done/60,counts=counts,
                      elapsed_seconds=elapsed,eta_seconds=elapsed*(6000-done)/done if done else None,
                      eta_scope='remaining inference; startup included, reporting excluded',workers=[p.pid for p in children])
                stats=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
                current=int(Path('/sys/fs/cgroup/memory.current').read_text());limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
                anon=int(stats['anon']);resource=dict(timestamp=stamp(),anon_GiB=anon/2**30,file_GiB=int(stats['file'])/2**30,total_GiB=current/2**30)
                write(OUT/'environment/resource_latest.json',resource)
                pressure=anon>110*2**30 or current>.92*limit
                pressure_since=(pressure_since or time.monotonic()) if pressure else None
                if pressure_since and time.monotonic()-pressure_since>=15:raise MemoryError('Memory guard: stop this FPS run only; spatial settings will not be reduced')
                codes=[p.poll() for p in children]
                if any(c is not None and c!=0 for c in codes):raise RuntimeError('FPS worker failed: '+str(codes))
                if all(c==0 for c in codes):break
                time.sleep(20)
            state(state='running',stage='reporting',completed=6000,total=6000)
            finalize();state(state='complete',stage='complete',completed=6000,total=6000,validation_passed=True,report=str(REPORT/'REPORT.md'))
        except BaseException:
            import traceback
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for p in children:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            state(state='failed',stage='failed',error=traceback.format_exc());raise
        finally:
            for log in logs:log.close()

if __name__=='__main__':main()
