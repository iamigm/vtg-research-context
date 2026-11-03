"""Zero-shot FPS sweep with a fixed per-video spatial grid and measured PTS."""
from result_layout import relocate_path
import os, sys, math, json, time, hashlib, threading, traceback, fcntl, subprocess
from pathlib import Path
from bisect import bisect_left
from types import MethodType
ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parent / '_shared/timelens'
sys.path[:0] = [str(SHARED/'runtime/timelens_deps'), str(SHARED/'vendor/TimeLens')]
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from adaptive_temporal_sampling import read, write, sha, objhash, stamp, need, immutable, tensorhash
RUN = 'fps_spatial_control_pilot300_v1'
OUT = relocate_path(ROOT/'artifacts/runs'/RUN)
REPORT = relocate_path(ROOT/'results'/RUN)
RANK = int(os.environ['FPS_WORKER_RANK']) if 'FPS_WORKER_RANK' in os.environ else None
COHORT = ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_300/samples.json'
FPS = [2,3,4,6,8]
ORDERS = ['video_query','query_video']
MODEL_ROOT = Path('/home/irteam/backbones/models')
MODELS = {m: MODEL_ROOT/('Qwen3-VL-8B-'+m.title()) for m in ['instruct','thinking']}
CONFIG = dict(run_id=RUN, fps=FPS, models=list(MODELS), orders=ORDERS, cohort='nested_pilot_v1/pilot_300',
              target_pixels=100352, area_tolerance=.1, spatial_policy='one fixed H,W per video for all FPS, models and orders; multiples of 32; aspect-ratio error first, then area error',
              temporal_policy='GT-blind nearest measured PTS to k/fps starting at zero, ties earlier; deduplicate; full duration; no 768 frame cap; duplicate final frame only for temporal patch packing',
              total_video_token_cap=None, resize='torchvision bicubic antialias uint8, directly from source RGB',
              dtype='bfloat16', attention='flash_attention_2', batch_size=1, seed=42, zero_shot=True,
              instruct=dict(do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,max_new_tokens=512,use_cache=True),
              thinking=dict(do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,max_new_tokens=16384,use_cache=True))

def grid(h,w):
    candidates=[];target=CONFIG['target_pixels']
    for a in range(1,math.floor(target*1.1/1024)+1):
        for b in range(max(1,math.ceil(target*.9/(1024*a))),math.floor(target*1.1/(1024*a))+1):
            candidates.append((abs(math.log((b/a)/(w/h))),abs(1024*a*b/target-1),a,b))
    _,_,a,b=min(candidates)
    return [a*32,b*32]

def temporal_plan(public, fps):
    # This API cannot receive query annotations, prior predictions or scores.
    need(set(public)=={'pts','ids','duration'},'Planner schema includes unapproved fields')
    original_pts=public['pts'];original_ids=public['ids']
    need(len(original_pts)==len(original_ids) and len(original_pts)>0,'Missing source timing')
    need(all(math.isfinite(t) for t in original_pts) and all(a<=b for a,b in zip(original_pts,original_pts[1:])),'Nonfinite or reversed source PTS')
    # Equal measured PTS are legitimate; the documented earlier-frame tie rule
    # selects their first source ID. Never invent or perturb a timestamp.
    keep=[i for i,t in enumerate(original_pts) if i==0 or t!=original_pts[i-1]]
    pts=[original_pts[i] for i in keep];ids=[original_ids[i] for i in keep];chosen=[]
    for k in range(math.ceil(public['duration']*fps)):
        t=k/fps
        j=bisect_left(pts,t)
        options=[i for i in [j-1,j] if 0<=i<len(pts)]
        ix=min(options,key=lambda i:(abs(pts[i]-t),i))
        if not chosen or ids[ix]!=chosen[-1]:chosen.append(ids[ix])
    need(len(chosen)>=2,'Video too short for specified temporal packing')
    packed=chosen+chosen[-1:] if len(chosen)%2 else chosen[:]
    loc=dict(zip(ids,pts))
    return dict(fps=fps,frame_ids=packed,source_pts=[loc[i] for i in packed],unique_frames=len(chosen),
                packed_frames=len(packed),padding_frames=len(packed)-len(chosen),duration=public['duration'],
                effective_unique_fps=len(chosen)/public['duration'],uses_gt=False)

def drop_cache(path):
    try:
        fd=os.open(relocate_path(Path(path)),os.O_RDONLY)
        try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)
    except OSError:pass

def prepare_video(g):
    """One short-lived CPU worker per video; decode in the frozen 32-frame order."""
    import numpy as np, torch, decord
    from torchvision.transforms import functional as TF, InterpolationMode
    torch.set_num_threads(1)
    dst=OUT/'media'/g['key'];meta_path=dst/'metadata.json'
    if meta_path.exists():
        m=read(meta_path);need(sha(m['array'])==m['array_sha256'],'Frame cache changed');drop_cache(m['array']);return m
    parent=read(g['parent_metadata']);need(sha(g['video'])==parent['source_video_hash'],'Source video changed')
    vr=decord.VideoReader(g['video'],num_threads=1);need(len(vr)==parent['N'],'Source frame count changed')
    # Read the first frame as part of the same sequential chunk path as all other frames.
    first=vr.get_batch(list(range(min(32,len(vr))))).asnumpy();h,w=first.shape[1:3];hw=grid(h,w)
    pts=parent['pts'];need(len(pts)==len(vr) and parent['ids']==list(range(len(vr))),'Incomplete source PTS')
    need(all(math.isfinite(t) for t in pts) and all(a<=b for a,b in zip(pts,pts[1:])),'Nonfinite or reversed source PTS')
    public=dict(pts=pts,ids=parent['ids'],duration=len(vr)/parent['fps'])
    plans={str(f):temporal_plan(public,f) for f in FPS};ids=sorted({i for p in plans.values() for i in p['frame_ids']});loc={i:j for j,i in enumerate(ids)}
    dst.mkdir(parents=True,exist_ok=True);target=dst/'selected_rgb.npy';tmp=dst/'selected_rgb.building.npy'
    arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(ids),3,*hw));hashes=['']*len(ids)
    for start in range(0,len(vr),32):
        batch=list(range(start,min(start+32,len(vr))));raw=first if start==0 else vr.get_batch(batch).asnumpy()
        selected=[i for i in batch if i in loc]
        if not selected:continue
        rgb=raw[[i-start for i in selected]]
        for i,a in zip(selected,rgb):need(hashlib.sha256(a.tobytes()).hexdigest()==parent['rgb_hashes'][i],'Decoded source RGB changed')
        resized=TF.resize(torch.from_numpy(rgb).permute(0,3,1,2),hw,interpolation=InterpolationMode.BICUBIC,antialias=True).numpy()
        for i,a in zip(selected,resized):arr[loc[i]]=a;hashes[loc[i]]=hashlib.sha256(a.tobytes()).hexdigest()
    arr.flush();del arr;os.replace(tmp,target)
    m=dict(video=g['video'],key=g['key'],source_sha256=parent['source_video_hash'],parent_metadata=g['parent_metadata'],parent_sha256=sha(g['parent_metadata']),
           source_hw=[h,w],hw=hw,actual_pixels=math.prod(hw),area_error=math.prod(hw)/CONFIG['target_pixels']-1,
           aspect_error=(hw[1]/hw[0])/(w/h)-1,ids=ids,rgb_hashes=hashes,array=str(target),array_sha256=sha(target),
           N=len(vr),source_fps=parent['fps'],duration=public['duration'],plans=plans,raw_rgb_verified=True,
           duplicate_source_pts=sum(a==b for a,b in zip(pts,pts[1:])),duplicate_pts_policy='earlier source ID at identical measured PTS')
    write(meta_path,m);drop_cache(target);drop_cache(g['video']);return m

def prepare_input(public, cache, fps, order, processor):
    import numpy as np, torch
    from evaluation.utils import GROUNDER_PROMPT
    need(set(public)=={'sample_key','dataset','video_path','query','duration'},'Annotations entered inference')
    plan=cache['plans'][str(fps)];ids=plan['frame_ids'];loc={i:j for j,i in enumerate(cache['ids'])}
    arr=np.load(relocate_path(Path(cache['array'])),mmap_mode='r');selected=arr[[loc[i] for i in ids]]
    need(all(hashlib.sha256(a.tobytes()).hexdigest()==cache['rgb_hashes'][loc[i]] for a,i in zip(selected,ids)),'Selected pixel hash changed')
    pts=dict(zip(ids,plan['source_pts']))
    def timestamps(self,indices,video_fps,merge_size=2):
        indices=list(indices);indices+=indices[-1:]*((-len(indices))%merge_size)
        return [(pts[indices[j]]+pts[indices[j+merge_size-1]])/2 for j in range(0,len(indices),merge_size)]
    content=[dict(type='video',video=public['video_path']),dict(type='text',text=GROUNDER_PROMPT.format(public['query']))]
    if order=='query_video':content.reverse()
    text=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
    original=processor._calculate_timestamps;processor._calculate_timestamps=MethodType(timestamps,processor)
    try:
        cpu=processor(text=[text],videos=[torch.from_numpy(selected).float()],video_metadata=[dict(fps=cache['source_fps'],frames_indices=ids[:],total_num_frames=cache['N'],video_backend='decord')],
                      do_resize=False,do_sample_frames=False,padding=True,return_tensors='pt')
    finally:processor._calculate_timestamps=original
    expected=[len(ids)//2,cache['hw'][0]//16,cache['hw'][1]//16]
    need(cpu.video_grid_thw[0].tolist()==expected,'Processor changed fixed spatial/temporal grid')
    tokens=int((cpu.input_ids==processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum())
    need(tokens==math.prod(expected)//4,'Visual token count mismatch')
    detail=dict(sample_key=public['sample_key'],fps=fps,order=order,hw=cache['hw'],plan=plan,grid=expected,visual_tokens=tokens,
                input_tokens=int(cpu.input_ids.shape[1]),input_ids=cpu.input_ids[0].tolist(),hashes={k:tensorhash(v) for k,v in cpu.items()},
                expanded_prompt=processor.tokenizer.decode(cpu.input_ids[0],skip_special_tokens=False,clean_up_tokenization_spaces=False),
                source_sha256=cache['source_sha256'],cache_metadata=str(OUT/'media'/cache['key']/'metadata.json'),uses_gt=False)
    return cpu,detail

class Progress:
    def __init__(self,total):
        self.start=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event();self.state=dict(state='running',stage='starting',completed=0,total=total,counts={})
        self.initial=0;self.generation_start=None
        threading.Thread(target=self.heartbeat,daemon=True).start()
    def update(self,**kw):
        with self.lock:
            self.state.update(kw);s=dict(self.state);s.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start,percentage=100*s['completed']/s['total'])
            n=s['completed']-self.initial
            s['eta_seconds']=(time.monotonic()-self.generation_start)*(s['total']-s['completed'])/n if self.generation_start and n>0 else None
            s['eta_scope']='remaining inference at observed rate; excludes preparation backlog, model reload and reporting'
            name='status.json' if RANK is None else f'status_worker{RANK}.json'
            write(OUT/'metadata'/name,s)
            log='progress.log' if RANK is None else f'progress_worker{RANK}.log'
            with (OUT/'logs'/log).open('a') as f:f.write(json.dumps(s,ensure_ascii=False)+'\n')
            print(json.dumps(s,ensure_ascii=False),flush=True)
    def heartbeat(self):
        while not self.stop.wait(20):self.update()

def setup():
    samples=read(COHORT);need(len(samples)==300,'Frozen pilot300 required')
    immutable(OUT/'metadata/config.json',CONFIG);immutable(OUT/'metadata/cohort.json',samples)
    parents={read(p)['video_path']:str(p) for p in (ROOT/'artifacts/common/spatial_sampling_sources_v2/source_frames').glob('*.json')}
    groups={}
    for s in samples:
        video=s['video_path'];need(video in parents,'Missing reusable source metadata')
        key=hashlib.sha256(video.encode()).hexdigest();groups[video]=dict(video=video,key=key,parent_metadata=parents[video])
    immutable(OUT/'metadata/groups.json',list(groups.values()))
    sources=[Path(__file__).resolve().with_name(name) for name in ['fps_spatial_control.py','fps_spatial_report.py','fps_spatial_supervisor.py','fps_spatial_smoke.py']]
    manifest=dict(cohort_sha256=sha(COHORT),files=[dict(path=str(p),sha256=sha(p)) for p in sources])
    immutable(OUT/'metadata/source_manifest.json',manifest)
    for p in sources:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    return samples,list(groups.values())

def run():
    import torch, multiprocessing, gc
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoProcessor,AutoModelForImageTextToText
    samples,groups=setup()
    if RANK is not None:
        groups=[g for g in groups if int(g['key'],16)%2==RANK]
        videos={g['video'] for g in groups};samples=[s for s in samples if s['video_path'] in videos]
    progress=Progress(len(samples)*len(FPS)*len(MODELS)*len(ORDERS))
    try:
        torch.set_num_threads(2)
        quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip();memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        write(OUT/'environment'/('resources.json' if RANK is None else f'resources_worker{RANK}.json'),dict(cpu_max=quota,memory_max=memory,preparation_workers=1,main_torch_threads=2,model_batch_size=1,rank=RANK))
        prepared={};ready=threading.Condition();prep_error=[]
        def prepare_all():
            try:
                pending=[]
                # Cache validation uses no decoder or Torch operations: avoid a
                # fresh spawned interpreter for each already prepared video.
                for g in groups:
                    meta=OUT/'media'/g['key']/'metadata.json'
                    if not meta.exists():pending.append(g);continue
                    m=read(meta);need(sha(m['array'])==m['array_sha256'],'Frame cache changed');drop_cache(m['array'])
                    with ready:prepared[m['video']]=m;ready.notify_all()
                    progress.update(prepared_videos=len(prepared),total_videos=len(groups))
                with multiprocessing.get_context('spawn').Pool(1,maxtasksperchild=1) as pool:
                    for m in pool.imap(prepare_video,pending,chunksize=1):
                        with ready:prepared[m['video']]=m;ready.notify_all()
                        progress.update(prepared_videos=len(prepared),total_videos=len(groups))
            except BaseException:
                with ready:prep_error.append(traceback.format_exc());ready.notify_all()
                raise
        executor=ThreadPoolExecutor(1);future=executor.submit(prepare_all)
        progress.update(stage='waiting_for_gpu',required_free_GiB=80)
        while torch.cuda.mem_get_info()[0]<80*2**30:time.sleep(20)
        config_hash=objhash(CONFIG);completed=0;counts={}
        for label in MODELS:
            for s in samples:
                for fps in FPS:
                    for order in ORDERS:
                        p=OUT/'responses'/label/str(fps)/order/(s['sample_key']+'.json')
                        if p.exists():
                            r=read(p);need(r['config_hash']==config_hash,'Resume config mismatch');completed+=1
                            k=label+'|'+s['dataset'];counts[k]=counts.get(k,0)+1
        progress.initial=completed;progress.generation_start=time.monotonic();progress.update(completed=completed,counts=counts)
        for label,model_path in MODELS.items():
            progress.update(stage='loading_model',model=label)
            processor=AutoProcessor.from_pretrained(model_path,padding_side='left',do_resize=False,local_files_only=True)
            model=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
            model_files={name:sha(model_path/name) for name in ['config.json','generation_config.json','tokenizer_config.json','video_preprocessor_config.json']}
            write(OUT/'environment'/(label+'.json'),dict(model_path=str(model_path),model_files=model_files,generation=model.generation_config.to_dict(),overrides=CONFIG[label],context_limit=model.config.text_config.max_position_embeddings))
            for sample in samples:
                public={k:sample[k] for k in ['sample_key','dataset','video_path','query','duration']}
                with ready:
                    while public['video_path'] not in prepared and not prep_error:ready.wait(10)
                    need(not prep_error,prep_error[0] if prep_error else '')
                    cache=prepared[public['video_path']]
                for fps in FPS:
                    for order in ORDERS:
                        relative=Path(label)/str(fps)/order/(sample['sample_key']+'.json');dest=OUT/'responses'/relative
                        if dest.exists():continue
                        progress.update(stage='generation',model=label,dataset=sample['dataset'],fps=fps,order=order,sample_key=sample['sample_key'])
                        cpu,detail=prepare_input(public,cache,fps,order,processor)
                        need(detail['input_tokens']+CONFIG[label]['max_new_tokens']<model.config.text_config.max_position_embeddings,'Context overflow: do not shrink input silently')
                        write(OUT/'inputs'/relative,detail)
                        torch.manual_seed(CONFIG['seed']);torch.cuda.synchronize();started=time.perf_counter()
                        gpu=cpu.to('cuda')
                        with torch.inference_mode():output=model.generate(**gpu,**CONFIG[label])
                        torch.cuda.synchronize();generated=output[0,cpu.input_ids.shape[1]:].tolist()
                        raw=processor.tokenizer.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                        if label=='thinking':answer=raw.split('</think>',1)[1].replace('<|im_end|>','').replace('<|endoftext|>','').strip() if '</think>' in raw else ''
                        else:answer=processor.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                        write(dest,dict(model=label,sample_key=sample['sample_key'],dataset=sample['dataset'],fps=fps,order=order,answer=answer,raw=raw,
                                        generated_ids=generated,generated_tokens=len(generated),eos_reached=bool(generated and generated[-1] in eos),
                                        generation_seconds=time.perf_counter()-started,input_hash=objhash(detail['hashes']),input_detail_sha256=sha(OUT/'inputs'/relative),
                                        config_hash=config_hash,source_manifest_hash=objhash(read(OUT/'metadata/source_manifest.json')),completed_at=stamp()))
                        completed+=1;k=label+'|'+sample['dataset'];counts[k]=counts.get(k,0)+1
                        progress.update(completed=completed,counts=dict(counts));del cpu,gpu,output
                drop_cache(cache['array'])
            del model,processor;gc.collect();torch.cuda.empty_cache()
        future.result();executor.shutdown()
        if RANK is not None:
            progress.update(state='complete',stage='worker_complete',completed=progress.state['total'])
            return
        progress.update(stage='reporting')
        from fps_spatial_report import build
        build()
        from artifact_store import register
        for kind,path in [('predictions',OUT/'responses'),('inputs',OUT/'inputs'),('manifests',OUT/'metadata'),('analyses',REPORT)]:
            register(RUN+'.'+kind,kind,Path(str(path).replace('/home1/irteam/','/home/irteam/')),'GT-blind zero-shot FPS sweep with fixed per-video spatial grid','_lib/fps_spatial_control.py','pilot_300','complete',['nested_pilot_v1.300.cohort'])
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
        progress.update(state='complete',stage='complete',completed=progress.state['total'],report=str(REPORT/'REPORT.md'),validation_passed=True)
    except BaseException:
        progress.update(state='failed',stage='failed',error=traceback.format_exc());raise
    finally:progress.stop.set()

if __name__=='__main__':
    for category in ['metadata','logs','validation','environment']:(OUT/category).mkdir(parents=True,exist_ok=True)
    with (OUT/'metadata'/('run.lock' if RANK is None else f'run_worker{RANK}.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if '--setup' in sys.argv:setup()
        else:run()
