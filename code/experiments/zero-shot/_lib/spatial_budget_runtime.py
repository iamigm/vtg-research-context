"""Operational concurrency overlay; frozen experiment/model/input identities unchanged."""
import os,math,concurrent.futures,fcntl
from pathlib import Path
import spatial_budget_pilot900 as m
quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
capacity=min(len(os.sched_getaffinity(0)),int(quota)//int(period) if quota!='max' else len(os.sched_getaffinity(0)))
VIDEO_WORKERS=max(1,capacity-4);INPUT_WORKERS=min(2,max(1,capacity//4))
OriginalProcessPool=concurrent.futures.ProcessPoolExecutor
OriginalThreadPool=concurrent.futures.ThreadPoolExecutor
class ProcessPool(OriginalProcessPool):
    def __init__(self,*args,**kw):
        kw['max_workers']=min(kw.get('max_workers',VIDEO_WORKERS),VIDEO_WORKERS);super().__init__(*args,**kw)
class ThreadPool(OriginalThreadPool):
    def __init__(self,*args,**kw):
        kw['max_workers']=min(kw.get('max_workers',INPUT_WORKERS),INPUT_WORKERS);super().__init__(*args,**kw)
def main():
    concurrent.futures.ProcessPoolExecutor=ProcessPool
    concurrent.futures.ThreadPoolExecutor=ThreadPool
    original_status=m.status;original_write=m.write
    def status(**kw):
        if 'preparation_workers' in kw:kw['preparation_workers']=VIDEO_WORKERS
        original_status(**kw)
    def write(path,value):
        if Path(path).name=='overlap_operation.json':value=dict(value,video_workers=VIDEO_WORKERS,input_workers=INPUT_WORKERS,cpu_quota=capacity)
        return original_write(path,value)
    m.status=status;m.write=write
    with (m.OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(m.OUT/'environment/effective_resources.json',dict(timestamp=m.stamp(),visible_cpus=os.cpu_count(),affinity_cpus=len(os.sched_getaffinity(0)),cpu_max=[quota,period],capacity=capacity,video_workers=VIDEO_WORKERS,input_workers=INPUT_WORKERS,main_torch_threads=2,memory_limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),wrapper=m.fileinfo(__file__)))
        m.run()
if __name__=='__main__':main()
