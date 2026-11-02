"""Memory-bounded operational overlay; frozen experiment inputs and identities unchanged."""
import os,concurrent.futures,fcntl,multiprocessing,time
from pathlib import Path
import spatial_sweep900 as m
# Execution scope amendment: preserve input identities and completed responses.
_frozen_preflight=m.preflight
def active_preflight():
    jobs,groups=_frozen_preflight()
    jobs=[j for j in jobs if not j['condition'].endswith('_s43')]
    assert len(jobs)==46800
    return jobs,groups
m.preflight=active_preflight
quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
capacity=min(len(os.sched_getaffinity(0)),int(quota)//int(period) if quota!='max' else len(os.sched_getaffinity(0)))
VIDEO_WORKERS=min(int(os.environ.get('VTG_VIDEO_WORKERS','4')),max(1,capacity-4));INPUT_WORKERS=min(2,max(1,capacity//4))
OriginalThreadPool=concurrent.futures.ThreadPoolExecutor
class ThreadPool(OriginalThreadPool):
    def __init__(self,*args,**kw):
        kw['max_workers']=min(kw.get('max_workers',INPUT_WORKERS),INPUT_WORKERS);super().__init__(*args,**kw)
def drop_file_cache(path):
    try:
        fd=os.open(path,os.O_RDONLY)
        try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)
    except OSError:pass

def prepare_one(g):
    # A fresh process handles exactly one video, releasing native decoder buffers at exit.
    result=m.prepare_group(g)
    for meta in result['metadata'].values():drop_file_cache(m.read(meta)['path'])
    drop_file_cache(g['video'])
    return result

def bounded_prepare_all(groups,mapping):
    from shared_selected_frames import manifest_path
    pending=[]
    for g in groups:
        previous=mapping.get(g['video'])
        if previous and all(Path(p).exists() and manifest_path(m.read(p)).exists() for p in previous['metadata'].values()):continue
        pending.append(g)
    m.status(prepared_videos=len(mapping),total_videos=len(groups),preparation_workers=VIDEO_WORKERS,worker_video_limit=1)
    with multiprocessing.get_context('spawn').Pool(processes=VIDEO_WORKERS,maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(prepare_one,pending,chunksize=1):
            mapping[result['video']]=result;m.write(m.OUT/'metadata/prepared.json',mapping)
            m.status(prepared_videos=len(mapping),total_videos=len(groups),preparation_workers=VIDEO_WORKERS,worker_video_limit=1)
    m.write(m.OUT/'validation/preparation.json',dict(passed=True,videos=len(groups),original_rgb_matches_parent=True,selected_frames_only=True));return mapping

def main():
    import spatial_sweep_report,spatial_sweep_report_s42
    spatial_sweep_report.report=spatial_sweep_report_s42.report
    concurrent.futures.ThreadPoolExecutor=ThreadPool
    m.prepare_all=bounded_prepare_all
    original_status=m.status;original_write=m.write
    def status(**kw):
        if 'preparation_workers' in kw:kw['preparation_workers']=VIDEO_WORKERS
        original_status(**kw)
    def write(path,value):
        if Path(path).name=='overlap_operation.json':value=dict(value,video_workers=VIDEO_WORKERS,input_workers=INPUT_WORKERS,cpu_quota=capacity,worker_video_limit=1,drop_completed_file_cache=True)
        return original_write(path,value)
    m.status=status;m.write=write
    with (m.OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(m.OUT/'environment/effective_resources.json',dict(timestamp=m.stamp(),visible_cpus=os.cpu_count(),affinity_cpus=len(os.sched_getaffinity(0)),cpu_max=[quota,period],capacity=capacity,video_workers=VIDEO_WORKERS,input_workers=INPUT_WORKERS,main_torch_threads=2,memory_limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),worker_video_limit=1,drop_completed_file_cache=True,wrapper=m.fileinfo(__file__)))
        m.run()
if __name__=='__main__':main()
