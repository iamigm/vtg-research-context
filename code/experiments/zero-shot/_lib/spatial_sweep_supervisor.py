"""Watch memory and child exits; checkpointed resume, without killing other jobs."""
from result_layout import relocate_path
import os,sys,time,subprocess,fcntl,signal,json
from pathlib import Path
from adaptive_temporal_sampling import ROOT,read,write,stamp
OUT=relocate_path(ROOT/'artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900')
def main():
    with (OUT/'metadata/supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for attempt,workers in enumerate([4,2,1],1):
            pressure_since=None;guard=False
            with (OUT/'logs/singleton_fallback.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'_lib/spatial_sweep_runtime.py')],cwd=ROOT,env=dict(os.environ,VTG_VIDEO_WORKERS=str(workers)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(OUT/'metadata/child_process.json',dict(pid=child.pid,supervisor_pid=os.getpid(),attempt=attempt,workers=workers,started_at=stamp()))
                while child.poll() is None:
                    resources={}
                    for name in ['memory.current','memory.peak','memory.events','memory.stat','cpu.stat']:
                        p=Path('/sys/fs/cgroup')/name
                        if p.exists():resources[name]=p.read_text().strip()
                    stat=dict(line.split() for line in resources['memory.stat'].splitlines());anon=int(stat['anon']);current=int(resources['memory.current']);limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
                    row=dict(timestamp=stamp(),child_pid=child.pid,workers=workers,anon_GiB=anon/2**30,file_GiB=int(stat['file'])/2**30,total_GiB=current/2**30,resources=resources)
                    write(OUT/'environment/resource_latest.json',row)
                    with (OUT/'logs/resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    # Guard actual working memory; clean file cache is reclaimable and is not a stop condition.
                    working=anon+int(stat.get('kernel',0))+int(stat.get('file_dirty',0))+int(stat.get('file_writeback',0))
                    pressure=anon>min(110*2**30,limit*.7) or working>limit*.8
                    pressure_since=(pressure_since or time.monotonic()) if pressure else None
                    if pressure_since and time.monotonic()-pressure_since>=15:
                        guard=True;write(OUT/'logs/memory_guard_latest.json',row);os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        break
                    time.sleep(5)
            exitrow=dict(timestamp=stamp(),attempt=attempt,pid=child.pid,returncode=child.returncode,memory_guard=guard)
            print(exitrow,flush=True);write(OUT/f'logs/exit_{child.pid}.json',exitrow)
            if child.returncode==0:return
            if (child.returncode>0 and not guard) or attempt==3:
                status=read(OUT/'metadata/status.json');status.update(state='aborted',timestamp=stamp(),supervisor_exit=exitrow);write(OUT/'metadata/status.json',status);return
            time.sleep(5)
if __name__=='__main__':main()
