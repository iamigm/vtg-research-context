"""CPU-only preparation ahead of the serial runner, using its exact bridge.

Disjoint future videos only. Does not load a model, generate answers or plan.
"""
from adaptive_temporal_sampling import *

def initialize():
    global processor,videos
    import torch,decord
    from functools import partial
    from transformers import AutoProcessor
    torch.set_num_threads(2);decord.VideoReader=partial(decord.VideoReader,num_threads=1)
    processor=AutoProcessor.from_pretrained(str(MODEL),padding_side='left',do_resize=False,local_files_only=True)
    videos={v['path']:v for v in read(OUT/'metadata/source_manifest.json')['videos']}

def prepare_group(records):
    completed=[]
    for rec in records:
        key=rec['sample']['sample_key'];video=videos[rec['sample']['video_path']]
        public,cache,cpu,detail=candidate_cache(rec,video,processor,None)
        need(detail['bridge_passed'],'CPU preparation bridge failed')
        completed.append(key)
        del cpu
    return completed

def main():
    import multiprocessing,concurrent.futures
    records=read(OUT/'metadata/baseline_records.json')
    excluded={r['sample']['video_path'] for r in records[:60]};groups={}
    for rec in records[60:]:
        video=rec['sample']['video_path']
        if video not in excluded:groups.setdefault(video,[]).append(rec)
    completed=[];started=time.perf_counter()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
            futures=[pool.submit(prepare_group,g) for g in groups.values()]
            for future in concurrent.futures.as_completed(futures):
                completed.extend(future.result());s=dict(state='building',completed=len(completed),total=sum(map(len,groups.values())),sample_keys=completed,elapsed_seconds=time.perf_counter()-started,timestamp=stamp(),gpu_inference=False)
                write(OUT/'validation/cpu_prepare_status.json',s);print(json.dumps(s),flush=True)
        write(OUT/'validation/cpu_prepare_status.json',dict(s,state='complete',producer=fileinfo(Path(__file__))))
    except BaseException:
        write(OUT/'validation/cpu_prepare_status.json',dict(state='aborted',error=traceback.format_exc(),sample_keys=completed));raise

if __name__=='__main__':main()
