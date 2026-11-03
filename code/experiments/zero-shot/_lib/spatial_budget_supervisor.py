"""Record exits and resume signal-killed inference; cannot survive container replacement."""
from result_layout import relocate_path
import os,sys,time,subprocess,fcntl
from pathlib import Path
from adaptive_temporal_sampling import ROOT,read,write,stamp
OUT=relocate_path(ROOT/'artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900')
def main():
    with (OUT/'metadata/supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for attempt in range(1,4):
            with (OUT/'logs/runner.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'_lib/spatial_budget_runtime.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                write(OUT/'metadata/child_process.json',dict(pid=child.pid,supervisor_pid=os.getpid(),attempt=attempt,started_at=stamp()))
                while child.poll() is None:
                    resources={}
                    for name in ['memory.current','memory.peak','memory.events','cpu.stat']:
                        p=Path('/sys/fs/cgroup')/name
                        if p.exists():resources[name]=p.read_text().strip()
                    write(OUT/'environment/resource_latest.json',dict(timestamp=stamp(),child_pid=child.pid,resources=resources));time.sleep(10)
            exitrow=dict(timestamp=stamp(),attempt=attempt,pid=child.pid,returncode=child.returncode)
            print(exitrow,flush=True);write(OUT/f'logs/exit_{attempt}.json',exitrow)
            if child.returncode==0:return
            if child.returncode>0 or attempt==3:
                status=read(OUT/'metadata/status.json');status.update(state='aborted',timestamp=stamp(),supervisor_exit=exitrow);write(OUT/'metadata/status.json',status);return
            time.sleep(5)
if __name__=='__main__':main()
