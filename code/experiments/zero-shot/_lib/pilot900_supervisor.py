"""Detached sequential900-query sweep and automatic integrated reporting."""
from result_layout import relocate_path
import os,sys,time,subprocess,traceback,fcntl
from pathlib import Path
from adaptive_temporal_sampling import ROOT,read,write,stamp,need,fileinfo
RUN='gt_density_allocation_comparison_pilot900_v1';OUT=relocate_path(ROOT/'artifacts/runs'/RUN)
A='gt_extent_boundary_density_pilot900_v1';B='gt_actual_density_pilot900_v1'
def status(stage,state='building',**kw):
    x=dict(state=state,stage=stage,timestamp=stamp(),**kw);write(OUT/'metadata/status.json',x);print(x,flush=True)
def main():
    for cat in ['metadata','logs','code']:(OUT/cat).mkdir(parents=True,exist_ok=True)
    with (OUT/'metadata/supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for module,run in [('pilot900_extent',A),('pilot900_actual',B)]:
                dest=relocate_path(ROOT/'artifacts/runs'/run);(dest/'logs').mkdir(parents=True,exist_ok=True)
                if (dest/'metadata/status.json').exists() and read(dest/'metadata/status.json')['state']=='complete':continue
                if run==B:
                    need(read(relocate_path(ROOT/'artifacts/runs'/A/'metadata/status.json'))['state']=='complete','A not complete')
                    write(dest/'metadata/frame_cache_index.json',read(relocate_path(ROOT/'artifacts/runs'/A/'metadata/frame_cache_index.json')))
                with (dest/'logs/runner.log').open('ab',buffering=0) as log:
                    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'_lib/pilot900_prefetch.py'),module],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    write(dest/'metadata/process.json',dict(pid=p.pid,supervisor_pid=os.getpid(),started_at=stamp()))
                    while p.poll() is None:
                        st=read(dest/'metadata/status.json') if (dest/'metadata/status.json').exists() else {}
                        prep=read(dest/'metadata/preparation_status.json') if (dest/'metadata/preparation_status.json').exists() else {}
                        status(run,child_pid=p.pid,child_stage=st.get('stage','starting'),completed=st.get('completed',0),total=17100,preparation_completed=prep.get('completed'),preparation_total=900);time.sleep(20)
                    need(p.returncode==0,'Child failed; see '+str(dest/'logs/runner.log'))
            status('integrated_report');subprocess.run([sys.executable,str(ROOT/'_lib/pilot900_overview.py')],check=True)
            subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True);status('complete',state='complete')
        except BaseException:
            status('aborted',state='aborted',error=traceback.format_exc());raise
if __name__=='__main__':main()
