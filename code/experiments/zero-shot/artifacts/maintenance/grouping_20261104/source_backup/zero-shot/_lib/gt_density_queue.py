"""Detached sequential supervisor. Never alter the running predecessor."""
import os,sys,time,subprocess,traceback,fcntl
from pathlib import Path
from adaptive_temporal_sampling import ROOT,read,write,stamp,need
A='gt_extent_boundary_density_pilot300_v1';B='gt_actual_density_pilot300_v1';RUN='gt_density_allocation_comparison_v1'
out=ROOT/'artifacts/runs'/RUN

def state(stage,**kw):
    row=dict(state='building',stage=stage,timestamp=stamp(),**kw);write(out/'metadata/queue_status.json',row);print(row,flush=True)
def main():
    (out/'metadata').mkdir(parents=True,exist_ok=True)
    with (out/'metadata/queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                s=read(ROOT/'artifacts/runs'/A/'metadata/status.json')
                if s['state']=='complete':break
                need(s['state'] not in ['aborted','superseded'],'Predecessor failed or superseded; successor not started')
                state('waiting_for_first_experiment',completed=s.get('completed'),total=s.get('total'))
                time.sleep(20)
            dest=ROOT/'artifacts/runs'/B;dest.mkdir(parents=True,exist_ok=True)
            write(dest/'metadata/frame_cache_index.json',read(ROOT/'artifacts/runs'/A/'metadata/frame_cache_index.json'))
            state('running_actual_density_experiment')
            (dest/'logs').mkdir(parents=True,exist_ok=True)
            with (dest/'logs/runner.log').open('ab',buffering=0) as log:
                p=subprocess.Popen([sys.executable,'-u',str(ROOT/'_lib/gt_density_prefetch.py'),'gt_actual_density'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                write(dest/'metadata/process.json',dict(pid=p.pid,parent_pid=os.getpid(),started_at=stamp()))
                while p.poll() is None:
                    state('running_actual_density_experiment',child_pid=p.pid);time.sleep(20)
                need(p.returncode==0,'Actual-density experiment failed; inspect successor runner.log')
            state('building_combined_report');subprocess.run([sys.executable,str(ROOT/'_lib/gt_density_combined.py')],check=True)
            subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
            state('complete');write(out/'metadata/queue_status.json',dict(state='complete',stage='complete',timestamp=stamp()))
        except BaseException:
            write(out/'metadata/queue_status.json',dict(state='aborted',error=traceback.format_exc(),timestamp=stamp()));raise
if __name__=='__main__':main()
