"""Two disjoint video shards, each preserving singleton inference and frozen identities."""
import os,hashlib,fcntl
from pathlib import Path
os.environ['VTG_VIDEO_WORKERS']='1'
import spatial_sweep_runtime as rt
m=rt.m
RANK=int(os.environ['VTG_RANK']);WORLD=2

def owner(video):return int(hashlib.sha256(video.encode()).hexdigest()[:8],16)%WORLD

def main():
    original_preflight=m.preflight;original_status=m.status;original_write=m.write
    def preflight():
        jobs,groups=original_preflight();return [j for j in jobs if owner(j['video'])==RANK],[g for g in groups if owner(g['video'])==RANK]
    def write(path,value):
        path=Path(path)
        if path==m.OUT/'metadata/prepared.json':path=m.OUT/'metadata'/f'prepared_worker{RANK}.json'
        elif path==m.OUT/'metadata/status.json':path=m.OUT/'metadata'/f'status_worker{RANK}.json'
        elif path.parent in [m.OUT/'environment',m.OUT/'validation']:
            path=path.with_name(path.stem+f'_worker{RANK}'+path.suffix)
        original_write(path,value)
    def status(**kw):
        kw.update(worker_rank=RANK,world_size=WORLD)
        original_status(**kw)
    m.preflight=preflight;m.write=write;m.status=status;m.prepare_all=rt.bounded_prepare_all
    rt.concurrent.futures.ThreadPoolExecutor=rt.ThreadPool
    # Each shard only generates records. The supervisor performs the one global report/registration.
    import spatial_sweep_report,artifact_store
    spatial_sweep_report.report=lambda:None
    artifact_store.register=lambda *a,**kw:None
    original_subprocess=m.subprocess.run
    def subprocess_run(args,*a,**kw):
        if any(str(x).endswith('build_index.py') for x in args):return None
        return original_subprocess(args,*a,**kw)
    m.subprocess.run=subprocess_run
    with (m.OUT/'metadata'/f'worker{RANK}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(m.OUT/'environment/parallel_runtime.json',dict(timestamp=m.stamp(),rank=RANK,world=WORLD,video_workers=1,input_workers=rt.INPUT_WORKERS,main_torch_threads=2,batch_size=1,partition='SHA256(video) first8 hex mod2',wrapper=m.fileinfo(__file__)))
        m.run()
if __name__=='__main__':main()
