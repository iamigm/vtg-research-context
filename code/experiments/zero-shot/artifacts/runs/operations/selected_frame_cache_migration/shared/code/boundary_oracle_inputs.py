"""Read-only bridge to prior inputs; full-source-frame cache for boundary tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import adaptive_temporal_sampling as old
from adaptive_temporal_sampling import read,write,immutable,sha,objhash,fileinfo,need,stamp
import os,time,copy,json,threading,traceback
RUN='temporal_sampling_pilot300_v2'
ROOT=old.ROOT
OUT=ROOT/'artifacts/runs'/RUN
REPORT=ROOT/'results'/RUN

def records():
    rows=read(old.LEGACY/'cohorts/pilot_300/samples.json');previous={r['sample']['sample_key']:r for r in read(old.OUT/'metadata/baseline_records.json')};result=[]
    for sample in sorted(rows,key=lambda s:(s['dataset'],s['sample_key'])):
        key=sample['sample_key']
        if key in previous:result.append(previous[key]);continue
        metadata=read(old.LEGACY/'qwen3'/sample['dataset']/key/'metadata.json');grid=metadata['video_grid_thw'];need(metadata['sample']==sample,'Cohort metadata mismatch')
        result.append(dict(sample=sample,video_id=Path(sample['video_path']).stem,F=len(metadata['decoder_metadata']['frames_indices']),B=grid[0]*grid[1]*grid[2]//4,grid=grid,baseline_metadata=metadata,video_sha256=None))
    need(len(result)==300,'Pilot cohort must contain all 300 queries');return result

def input_record(record,F):
    need(F>=2 and F%2==0 and F<=768,'Unsupported temporal packing/frame budget')
    grid=[F//2,*record['grid'][1:]]
    return dict(sample={k:v for k,v in record['sample'].items() if k!='span'},F=F,grid=grid,B=grid[0]*grid[1]*grid[2]//4)

def make_input(record,cache,plan,processor):
    import numpy as np,torch,hashlib
    from types import MethodType,SimpleNamespace
    from evaluation.utils import GROUNDER_PROMPT
    r=input_record(record,len(plan['frame_ids']));ids=plan['frame_ids'];loc={i:j for j,i in enumerate(cache['ids'])};selected=np.array(np.load(cache['path'],mmap_mode='r')[[loc[i] for i in ids]])
    need(all(hashlib.sha256(a.tobytes()).hexdigest()==cache['resized_rgb_hashes'][loc[i]] for a,i in zip(selected,ids)),'Shared RGB cache checksum mismatch')
    source_pts=dict(zip(cache['ids'],cache['pts']))
    def actual_times(self,indices,video_fps,merge_size=2):
        indices=list(indices);indices+=indices[-1:]*((-len(indices))%merge_size)
        return [(source_pts[indices[j]]+source_pts[indices[j+merge_size-1]])/2 for j in range(0,len(indices),merge_size)]
    original=processor._calculate_timestamps;processor._calculate_timestamps=MethodType(actual_times,processor)
    text=processor.apply_chat_template([dict(role='user',content=[dict(type='video',video=r['sample']['video_path']),dict(type='text',text=GROUNDER_PROMPT.format(r['sample']['query']))])],tokenize=False,add_generation_prompt=True)
    try:
        cpu=processor(text=[text],videos=[torch.from_numpy(selected).float()],video_metadata=[dict(fps=cache['fps'],frames_indices=ids[:],total_num_frames=cache['N'],video_backend='decord')],do_sample_frames=False,padding=True,return_tensors='pt')
        native=processor._calculate_timestamps(ids,cache['fps'],processor.video_processor.merge_size)
    finally:processor._calculate_timestamps=original
    grid=cpu.video_grid_thw[0].tolist();B=int((cpu.input_ids==processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum());need(grid==r['grid'] and B==r['B'],'Frame/token/grid budget mismatch')
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    cfg=read(old.MODEL/'config.json');holder=SimpleNamespace(config=SimpleNamespace(**{k:cfg[k] for k in ['image_token_id','video_token_id','vision_start_token_id']},vision_config=SimpleNamespace(spatial_merge_size=2)))
    positions,delta=Qwen3VLModel.get_rope_index(holder,input_ids=cpu.input_ids,video_grid_thw=cpu.video_grid_thw,attention_mask=cpu.attention_mask)
    return cpu,dict(sample_key=r['sample']['sample_key'],frame_ids=ids,source_pts=[source_pts[i] for i in ids],native_patch_timestamps=native,timestamp_method='native Qwen3 timestamp helper uses measured source presentation times; container origin normalized to video time zero',F=len(ids),B=B,grid=grid,padding=0,hashes={k:old.tensorhash(v) for k,v in cpu.items()},input_ids=cpu.input_ids[0].tolist(),position_ids=positions.tolist(),rope_delta=delta.tolist(),expanded_prompt=processor.tokenizer.decode(cpu.input_ids[0],skip_special_tokens=False,clean_up_tokenization_spaces=False),cache_key=cache['cache_key'],plan_sha256=objhash(plan),total_input_tokens=int(cpu.input_ids.shape[1]))

def prepare_video(group):
    import numpy as np,torch,decord,av
    from fractions import Fraction
    from torchvision.transforms import functional as TF,InterpolationMode
    torch.set_num_threads(2)
    record=group[0];sample=record['sample'];video=sample['video_path'];started=time.perf_counter()
    actual=sha(video)
    if record['video_sha256']:need(actual==record['video_sha256'],'Original video SHA mismatch '+video)
    vr=decord.VideoReader(video,num_threads=1);N=len(vr);fps=vr.get_avg_fps()
    with av.open(video) as container:
        stream=container.streams.video[0];tb=stream.time_base;rate=stream.average_rate
        raw_pts=sorted(p.pts for p in container.demux(stream) if p.pts is not None)
    timing_source='container presentation timestamps'
    if len(raw_pts)==N and len(set(raw_pts))==N:
        origin=raw_pts[0];pts=[float(Fraction(t-origin)*tb) for t in raw_pts]
    else:
        # Some containers contain packets the decoder does not expose as frames.
        # Use the decoder's indexed frame timestamps, never a fabricated fps grid.
        measured=np.asarray(vr.get_frame_timestamp(list(range(N))))[:,0].astype(float);origin=float(measured[0]);pts=(measured-origin).tolist();timing_source='Decord indexed frame PTS (float32 precision); packet count differs from decoded frame count'
    need(len(pts)==N and all(np.isfinite(pts)) and all(a<=b for a,b in zip(pts,pts[1:])),'Invalid indexed source frame timestamps')
    H,W=record['grid'][1]*16,record['grid'][2]*16
    key=objhash(dict(video_sha256=actual,N=N,grid=[H,W],decoder='decord-0.6.0',resize='torchvision bicubic antialias uint8',processor=sha(old.MODEL/'video_preprocessor_config.json'),pts=pts))
    target=OUT/'frames'/(key+'.npy');meta_path=target.with_suffix('.json');target.parent.mkdir(parents=True,exist_ok=True)
    for reusable in ['boundary_oracle_budget_pilot_v1','temporal_sampling_pilot300_v2']:
        prepared=ROOT/'artifacts/runs'/reusable/'frames'/(key+'.json')
        if prepared.exists():
            metadata=read(prepared);need(metadata['sha256']==sha(metadata['path']),'Prepared frame cache checksum mismatch');return [(r['sample']['sample_key'],str(prepared)) for r in group]
    if target.exists() and meta_path.exists():
        metadata=read(meta_path);need(metadata['sha256']==sha(target),'New frame cache checksum mismatch');return [(r['sample']['sample_key'],str(meta_path)) for r in group]
    references=[]
    for r in group:
        key_sample=r['sample']['sample_key'];oldcache=old.OUT/'frames'/(key_sample+'.json')
        if oldcache.exists():
            ref=read(oldcache);need(sha(ref['path'])==ref['sha256'],'Previous candidate cache checksum mismatch')
            references.append((ref,np.load(ref['path'],mmap_mode='r')))
    tmp=target.with_name(target.name+f'.{os.getpid()}.tmp');array=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(N,3,H,W))
    raw_hashes=[];resized_hashes=[]
    import hashlib
    for start in range(0,N,32):
        ids=list(range(start,min(start+32,N)));raw=vr.get_batch(ids).asnumpy()
        resized=TF.resize(torch.from_numpy(raw).permute(0,3,1,2),[H,W],interpolation=InterpolationMode.BICUBIC,antialias=True).numpy()
        array[start:start+len(ids)]=resized
        raw_hashes.extend(hashlib.sha256(x.tobytes()).hexdigest() for x in raw)
        resized_hashes.extend(hashlib.sha256(x.tobytes()).hexdigest() for x in resized)
    array.flush();del array
    checked=0;different=0
    for ref,previous in references:
        for loc,i in enumerate(ref['ids']):
            different+=int(raw_hashes[i]!=ref['rgb_hashes'][loc] or resized_hashes[i]!=ref['resized_rgb_hashes'][loc])
            checked+=1
    os.replace(tmp,target)
    metadata=dict(cache_key=key,path=str(target),sha256=sha(target),ids=list(range(N)),pts=pts,raw_pts=raw_pts,pts_origin=origin,time_base=str(tb),average_rate=str(rate),fps=fps,N=N,rgb_hashes=raw_hashes,resized_rgb_hashes=resized_hashes,source_video_hash=actual,video_path=video,grid_hw=[H,W],preparation_seconds=time.perf_counter()-started,old_source_frames_checked=checked,historical_sparse_rgb_differences=different,timing_source=timing_source,decoder='Decord 0.6.0, num_threads=1; increasing 32-frame reads shared by every v2 condition; previous sparse outputs historical only',pts_contract='measured presentation times relative to first presentation time; native Qwen3 timestamp helper maps actual source IDs to source PTS')
    write(meta_path,metadata)
    return [(r['sample']['sample_key'],str(meta_path)) for r in group]

def prepare_pilot():
    import concurrent.futures,multiprocessing
    for name in ['metadata','validation','frames','logs','code','metrics']:(OUT/name).mkdir(parents=True,exist_ok=True)
    groups={}
    for r in records():groups.setdefault((r['sample']['video_path'],tuple(r['grid'][1:])),[]).append(r)
    state=dict(state='building',stage='source_frame_preparation',completed=0,total=300,timestamp=stamp());stop=threading.Event();lock=threading.Lock();start=time.perf_counter();mapping={}
    def emit():
        with lock:
            state.update(timestamp=stamp(),elapsed_seconds=time.perf_counter()-start,percentage=100*state['completed']/300,eta_seconds=(time.perf_counter()-start)*(300-state['completed'])/state['completed'] if state['completed'] else None,eta_scope='remaining CPU frame preparation only')
            write(OUT/'metadata/preparation_status.json',state)
            with (OUT/'logs/preparation.log').open('a') as f:f.write(json.dumps(state,ensure_ascii=False)+'\n')
    def heartbeat():
        while not stop.wait(20):emit()
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();emit()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(prepare_video,g) for g in groups.values()]
            for future in concurrent.futures.as_completed(futures):
                try:pairs=future.result()
                except BaseException:
                    for pending in futures:pending.cancel()
                    with lock:state.update(state='aborted',error=traceback.format_exc())
                    emit();raise
                mapping.update(dict(pairs))
                with lock:state['completed']=len(mapping)
                emit()
        write(OUT/'metadata/frame_cache_index.json',mapping)
        with lock:state.update(state='complete',stage='source_frame_preparation_complete')
        emit()
    except BaseException:
        with lock:state.update(state='aborted',error=traceback.format_exc())
        emit();raise
    finally:stop.set();thread.join()

if __name__=='__main__':prepare_pilot()
