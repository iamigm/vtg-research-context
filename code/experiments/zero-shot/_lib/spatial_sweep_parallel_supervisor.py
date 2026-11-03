"""Coordinate disjoint singleton workers with global resource guard and one final report."""
from result_layout import relocate_path
import os,sys,time,subprocess,signal,fcntl,json
from pathlib import Path
from adaptive_temporal_sampling import ROOT,read,write,stamp
OUT=relocate_path(ROOT/'artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900')

def main():
 with (OUT/'metadata/supervisor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  children=[];logs=[];pressure_since=None
  try:
   for rank in range(2):
    log=(OUT/'logs'/f'worker{rank}.log').open('ab',buffering=0);logs.append(log)
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'_lib/spatial_sweep_parallel_worker.py')],cwd=ROOT,env=dict(os.environ,VTG_RANK=str(rank)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(p)
   write(OUT/'metadata/parallel_processes.json',dict(timestamp=stamp(),supervisor=os.getpid(),workers=[p.pid for p in children]))
   while True:
    states=[]
    for rank in range(2):
     p=OUT/'metadata'/f'status_worker{rank}.json'
     if p.exists():states.append(read(p))
    mapping=read(OUT/'metadata/prepared.json')
    for rank in range(2):
     p=OUT/'metadata'/f'prepared_worker{rank}.json'
     if p.exists():mapping.update(read(p))
    write(OUT/'metadata/prepared.json',mapping)
    done=sum(s.get('completed',0) for s in states);counts={}
    for s in states:
     for k,v in s.get('counts',{}).items():counts[k]=counts.get(k,0)+v
    codes=[p.poll() for p in children]
    failed=any(c is not None and c!=0 for c in codes)
    status=dict(state='aborted' if failed else 'building',stage='parallel_generation',timestamp=stamp(),completed=done,total=46800,percentage=done/468,prepared_videos=len(mapping),total_videos=892,counts=counts,worker_pids=[p.pid for p in children],workers=[{k:s.get(k) for k in ['stage','completed','total','error']} for s in states],singleton_workers=2)
    write(OUT/'metadata/status.json',status)
    stats=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines());current=int(Path('/sys/fs/cgroup/memory.current').read_text());limit=int(Path('/sys/fs/cgroup/memory.max').read_text());anon=int(stats['anon'])
    resource=dict(timestamp=stamp(),anon_GiB=anon/2**30,file_GiB=int(stats['file'])/2**30,total_GiB=current/2**30,workers=[p.pid for p in children],memory_events=Path('/sys/fs/cgroup/memory.events').read_text())
    write(OUT/'environment/resource_latest.json',resource)
    with (OUT/'logs/resources.jsonl').open('a') as f:f.write(json.dumps(resource)+'\n')
    working=anon+int(stats.get('kernel',0))+int(stats.get('file_dirty',0))+int(stats.get('file_writeback',0))
    pressure=anon>min(110*2**30,.7*limit) or working>.8*limit
    pressure_since=(pressure_since or time.monotonic()) if pressure else None
    if failed:raise RuntimeError('Worker failed: '+str(codes))
    if pressure_since and time.monotonic()-pressure_since>=15:
     write(OUT/'logs/parallel_memory_guard.json',resource);raise MemoryError('Parallel memory guard; fallback to one inference worker')
    if all(c==0 for c in codes):break
    time.sleep(5)
   from spatial_sweep_report_s42 import report
   write(OUT/'metadata/status.json',dict(status,stage='reporting'));report()
   from artifact_store import register
   RUN='spatial_sampling_pilot900_v2';REPORT=relocate_path(ROOT/'results'/RUN)
   for kind,path in [('predictions',OUT/'responses'),('inputs',OUT/'inputs'),('manifests',OUT/'metadata'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Controlled spatial whole/boundary full pilot900 sweep','_lib/spatial_sweep900.py','pilot_900','complete',['spatial_sampling_sources_v2.inputs','spatial_budget_pilot900_v1.predictions'])
   subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True);write(OUT/'metadata/status.json',dict(status,state='complete',stage='complete',completed=46800,timestamp=stamp()))
  except BaseException as e:
   for p in children:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   for p in children:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   write(OUT/'logs/parallel_exit.json',dict(timestamp=stamp(),error=repr(e),codes=[p.returncode for p in children]))
   if isinstance(e,MemoryError):
    # Release supervisor lock before starting the existing memory-bounded singleton supervisor.
    fcntl.flock(lock,fcntl.LOCK_UN)
    os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'_lib/spatial_sweep_supervisor.py')])
   write(OUT/'metadata/status.json',dict(state='aborted',stage='parallel_error',error=repr(e),timestamp=stamp()));raise
  finally:
   for log in logs:log.close()
if __name__=='__main__':main()
