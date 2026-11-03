"""Fixed per-frame spatial area, frozen parent temporal plans, selected-frame cache only."""
from result_layout import relocate_path
import os,sys,time,math,copy,hashlib,threading,traceback,fcntl,subprocess
from pathlib import Path
import adaptive_temporal_sampling as old
from adaptive_temporal_sampling import ROOT,read,write,immutable,objhash,sha,stamp,need,fileinfo
import spatial_sweep_inputs as bridge
RUN='spatial_sampling_pilot900_v2';PARENT='spatial_sampling_sources_v2'
OUT=relocate_path(ROOT/'artifacts/runs'/RUN);REPORT=relocate_path(ROOT/'results'/RUN);PR=relocate_path(ROOT/'artifacts/runs'/PARENT)
TEMPORAL=['uniform']+[f'{who}_{mode}_r{r}'+(f'_s{seed}' if who=='random' else '') for r in [2,4,8] for mode in ['whole','boundary'] for who in ['gt','random'] for seed in ([42,43] if who=='random' else [None])]
CONDITIONS=[d+'__'+c for d in ['A','B'] for c in TEMPORAL]
REUSE=relocate_path(ROOT/'artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900')
AREAS={'low':50176,'high':100352}
CONFIG=dict(run_id=RUN,parent=PARENT,areas=AREAS,conditions=CONDITIONS,cohort='nested_pilot_v1/pilot_900',model=str(old.MODEL),revision=old.CONFIG['revision'],seed=42,do_sample=False,max_new_tokens=512,repetition_penalty=1.,dtype='bfloat16',attention='flash_attention_2',resize='torchvision bicubic antialias uint8; directly from original RGB',grid_policy='H,W multiples32; area within10%; minimize abs(log((W/H)/(sourceW/sourceH))), then relative area error, then H,W',temporal_policy='exact parent frame IDs, source PTS, original frame count; no total video token cap',batch_policy='singleton; fixed-resolution prior batch4 probe failed exact generated IDs',gpu_free_required_GiB=60,cpu_workers=8)
GEN=dict(do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512,repetition_penalty=1.,use_cache=True)
STATE={};LOCK=threading.Lock();RUN_START=time.perf_counter();GEN_START=None;GEN_INITIAL=0
def status(**kw):
    with LOCK:
        STATE.update(kw,timestamp=stamp(),elapsed_seconds=time.perf_counter()-RUN_START)
        done=STATE.get('completed',0);total=STATE.get('total',7200);STATE['percentage']=100*done/total
        if STATE.get('stage')=='generation' and GEN_START is not None and done>GEN_INITIAL:
            STATE['eta_seconds']=(time.perf_counter()-GEN_START)*(total-done)/(done-GEN_INITIAL);STATE['eta_scope']='remaining inference only; excludes report'
        write(OUT/'metadata/status.json',STATE)
        print(str(STATE),flush=True)
def choose_grid(h,w,target):
    candidates=[]
    for a in range(1,int(target*1.1/1024)+1):
        for b in range(max(1,math.ceil(.9*target/(1024*a))),math.floor(1.1*target/(1024*a))+1):
            candidates.append((abs(math.log((b/a)/(w/h))),abs(1024*a*b/target-1),a,b))
    need(bool(candidates),'No grid within target tolerance');_,_,a,b=min(candidates);return a*32,b*32

def preflight():
    jobs=read(OUT/'metadata/jobs.json');groups=read(OUT/'metadata/groups.json')
    immutable(OUT/'metadata/config.json',CONFIG)
    need(len(jobs)==68400 and len(read(OUT/'metadata/cohort.json'))==900,'Sweep size mismatch')
    if not (OUT/'metadata/source_manifest.json').exists():
        files=[Path(__file__),ROOT/'_lib/spatial_sweep_report.py',ROOT/'_lib/spatial_sweep_inputs.py',ROOT/'_lib/pilot900_inputs.py',ROOT/'_lib/gt_density_batch.py']
        write(OUT/'metadata/source_manifest.json',dict(files=[fileinfo(p) for p in files]))
        for p in files:(OUT/'code'/p.name).write_bytes(p.read_bytes())
    for item in read(OUT/'metadata/source_manifest.json')['files']:need(sha(item['path'])==item['sha256'],'Frozen source changed')
    write(OUT/'metadata/preflight.json',dict(passed=True,config_hash=objhash(CONFIG),queries=900,logical_responses=len(jobs),videos=len(groups),context_limit=262144,max_visual_tokens=max(j['B'] for j in jobs),source_dependency='artifacts/common/spatial_sampling_sources_v2',temporal_plans_frozen=True))
    return jobs,groups

def prepare_group(g):
    import numpy as np,torch,decord
    from torchvision.transforms import functional as TF,InterpolationMode
    from shared_selected_frames import COMMON,manifest_path
    torch.set_num_threads(1);p=read(g['parent_metadata']);ids=g['ids'];start=time.perf_counter();result={};needed=[]
    for level,hw in g['grids'].items():
        ck=objhash(dict(video=p['source_video_hash'],ids=ids,hw=hw,decoder=p['decoder'],resize=CONFIG['resize'],version=RUN));meta=OUT/'metadata/cache'/(ck+'.json')
        if meta.exists() and manifest_path(read(meta)).exists():result[level]=str(meta);continue
        needed.append((level,hw,ck,meta))
    if not needed:return dict(video=g['video'],metadata=result,reused=True,seconds=time.perf_counter()-start)
    need(sha(g['video'])==p['source_video_hash'],'Source video changed');vr=decord.VideoReader(g['video'],num_threads=1);loc={i:j for j,i in enumerate(p['ids'])};wanted=set(ids);where={i:j for j,i in enumerate(ids)};arrays={};hashes={};paths={}
    for level,(h,w),ck,meta in needed:
        path=COMMON/'frames'/(ck+'.npy');path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.building.npy');arrays[level]=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(ids),3,h,w));hashes[level]=['']*len(ids);paths[level]=(path,tmp)
    for start_id in range(0,max(ids)+1,32):
        batch=list(range(start_id,min(start_id+32,len(vr))));rgb=vr.get_batch(batch).asnumpy();chosen=[i for i in batch if i in wanted]
        if not chosen:continue
        raw=rgb[[i-start_id for i in chosen]]
        for i,a in zip(chosen,raw):need(hashlib.sha256(a.tobytes()).hexdigest()==p['rgb_hashes'][loc[i]],'Original RGB differs from parent '+g['video'])
        tensor=torch.from_numpy(raw).permute(0,3,1,2)
        for level,hw,ck,meta in needed:
            resized=TF.resize(tensor,hw,interpolation=InterpolationMode.BICUBIC,antialias=True).numpy()
            for i,a in zip(chosen,resized):arrays[level][where[i]]=a;hashes[level][where[i]]=hashlib.sha256(a.tobytes()).hexdigest()
    for level,hw,ck,meta in needed:
        arrays[level].flush();del arrays[level];path,tmp=paths[level];os.replace(tmp,path)
        cache=dict(cache_key=ck,path=str(path),ids=ids,pts=[p['pts'][loc[i]] for i in ids],rgb_hashes=[p['rgb_hashes'][loc[i]] for i in ids],resized_rgb_hashes=hashes[level],grid_hw=hw,N=p['N'],fps=p['fps'],video_path=g['video'],source_video_hash=p['source_video_hash'],parent_metadata=g['parent_metadata'],original_rgb_equal=True,decoder=p['decoder'])
        write(manifest_path(cache),dict(cache_key=ck,path=str(path),ids=ids,rgb_hashes=hashes[level],sha256=sha(path),created_at=stamp(),provenance=RUN));write(meta,cache);result[level]=str(meta)
    return dict(video=g['video'],metadata=result,seconds=time.perf_counter()-start,reused=False)

def prepare_all(groups,mapping):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    completed=len(mapping)
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(prepare_group,g) for g in groups]
        for f in as_completed(futures):
            r=f.result();mapping[r['video']]=r;completed=len(mapping);write(OUT/'metadata/prepared.json',mapping);status(prepared_videos=completed,total_videos=len(groups),preparation_workers=8)
    write(OUT/'validation/preparation.json',dict(passed=True,videos=len(groups),original_rgb_matches_parent=True,selected_frames_only=True));return mapping

def run():
    global GEN_START,GEN_INITIAL
    import torch
    jobs,groups=preflight();status(state='building',stage='preparing_selected_frames',completed=0,total=len(jobs),prepared_videos=0)
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(20):status()
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        from concurrent.futures import ThreadPoolExecutor
        mapping=read(OUT/'metadata/prepared.json') if (OUT/'metadata/prepared.json').exists() else {}
        probe_videos={next(j['video'] for j in jobs if j['dataset']==ds and j['condition'].endswith('__uniform')) for ds in sorted({j['dataset'] for j in jobs})}
        groups.sort(key=lambda g:(g['video'] not in probe_videos,g['video'] not in mapping))
        prep_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='video-preparation')
        prep_future=prep_pool.submit(prepare_all,groups,mapping)
        write(OUT/'environment/overlap_operation.json',dict(timestamp=stamp(),video_workers=8,input_workers=2,input_prefetch=12,cpu_gpu_overlap=True,existing_caches_preserved=True,original_config_cpu_workers=CONFIG['cpu_workers']))
        status(stage='waiting_for_gpu',required_free_GiB=60)
        while torch.cuda.mem_get_info()[0]<60*2**30:time.sleep(20)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        torch.set_num_threads(2);processor=AutoProcessor.from_pretrained(old.MODEL,padding_side='left',do_resize=False,local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(old.MODEL,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
        write(OUT/'environment/runtime.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),generation_config=model.generation_config.to_dict(),processor_class=type(processor).__name__))
        rows={r['sample']['sample_key']:r for r in bridge.records()};cohort_hash=objhash(read(OUT/'metadata/cohort.json'));config_hash=objhash(CONFIG);code_hash=objhash(read(OUT/'metadata/source_manifest.json'))
        from concurrent.futures import ThreadPoolExecutor
        local=threading.local()
        def prepare(job):
            started=time.perf_counter()
            if not hasattr(local,'processor'):local.processor=AutoProcessor.from_pretrained(old.MODEL,padding_side='left',do_resize=False,local_files_only=True)
            r=copy.deepcopy(rows[job['sample_key']]);r.update(F=job['F'],B=job['B'],grid=job['grid'])
            p=read(OUT/'plans'/job['condition']/(job['sample_key']+'.json'));need(objhash(p)==job['plan_hash'],'Frozen plan changed')
            while job['video'] not in mapping:
                if prep_future.done():prep_future.result();need(job['video'] in mapping,'Missing prepared video')
                time.sleep(.1)
            cache=read(mapping[job['video']]['metadata'][job['level']])
            prior_path=REUSE/'inputs'/job['level']/job['temporal_condition']/(job['sample_key']+'.json')
            prior=read(prior_path) if prior_path.exists() else None
            if prior and prior['plan_sha256']==job['plan_hash'] and prior['grid']==job['grid'] and prior['frame_ids']==p['frame_ids']:
                detail=copy.deepcopy(prior);detail['cache_key']=cache['cache_key'];cpu=None
            else:cpu,detail=bridge.make_input(r,cache,p,local.processor)
            need(detail['total_input_tokens']+512<read(OUT/'metadata/preflight.json')['context_limit'],'Actual context overflow')
            detail.update(spatial_level=job['level'],target_pixels=job['target_pixels'],actual_pixels=job['actual_pixels'],area_error_fraction=job['area_error_fraction'],aspect_ratio_error_fraction=job['aspect_ratio_error_fraction'])
            identity=dict(run_id=RUN,sample_key=job['sample_key'],condition=job['condition'],spatial_level=job['level'],config_hash=config_hash,cohort_hash=cohort_hash,code_hash=code_hash,plan_hash=job['plan_hash'],input_hash=objhash(detail['hashes']))
            return cpu,detail,identity,time.perf_counter()-started
        def dest(job,kind):return OUT/kind/job['level']/job['condition']/(job['sample_key']+'.json')
        def source_path(job):
            c=job['temporal_condition'];p=REUSE/'responses'/job['level']/c/(job['sample_key']+'.json')
            return p if p.exists() else None
        input_reuse={}
        def save(job,item,result):
            cpu,detail,identity,seconds=item;result.update(identity=identity,input_detail_hash=objhash(detail),input_prepare_seconds=seconds,model_revision=CONFIG['revision']);write(dest(job,'inputs'),detail);write(dest(job,'responses'),result);input_reuse[(job['sample_key'],job['level'],identity['input_hash'])]=dest(job,'responses')
        complete=[];todo=[]
        for j in jobs:
            path=dest(j,'responses')
            if path.exists():
                x=read(path);need(x['identity']['config_hash']==config_hash and x['identity']['code_hash']==code_hash and x['identity']['plan_hash']==j['plan_hash'],'Resume identity mismatch');complete.append(j)
            else:todo.append(j)
        from gt_density_batch import generate
        gatepath=OUT/'validation/batch_bridge.json'
        if not gatepath.exists():
            status(stage='smoke_and_batch_bridge')
            probes=[next(j for j in jobs if j['dataset']==ds and j['level']==level and j['condition'].endswith('__uniform')) for ds in sorted({j['dataset'] for j in jobs}) for level in AREAS]
            items=[prepare(j) for j in probes];single=[old.generate_fresh(a,b,c,model,processor,GEN,t) for a,b,c,t in items];batched=[]
            try:
                for i in range(0,len(items),4):batched+=generate([x[:3] for x in items[i:i+4]],model,processor,GEN)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();batched=[]
            passed=len(batched)==len(single) and all(a['generated_ids']==b['generated_ids'] for a,b in zip(single,batched));write(gatepath,dict(passed=passed,batch_size=4 if passed else 1,probes=[dict(sample_key=j['sample_key'],level=j['level'],exact_ids=a['generated_ids']==b['generated_ids']) for j,a,b in zip(probes,single,batched)],batch_oom=len(batched)!=len(single)))
            for j,item,result in zip(probes,items,single):save(j,item,result)
            todo=[j for j in todo if not dest(j,'responses').exists()];complete=[j for j in jobs if dest(j,'responses').exists()]
            del items,single,batched
        batchsize=read(gatepath)['batch_size'];done=len(complete);GEN_INITIAL=done;GEN_START=time.perf_counter()
        from collections import Counter
        counts=Counter(j['dataset']+'|'+j['level']+'|'+j['condition'] for j in complete)
        status(stage='generation',batch_size=batchsize,completed=done,counts=dict(counts))
        # Bounded CPU prefetch keeps processor work off the GPU dispatch thread.
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={};next_submit=0
            for start in range(0,len(todo),batchsize):
                while next_submit<min(len(todo),start+12):pending[next_submit]=pool.submit(prepare,todo[next_submit]);next_submit+=1
                group=todo[start:start+batchsize];items=[pending.pop(i).result() for i in range(start,start+len(group))]
                results=[]
                for job,item in zip(group,items):
                    cpu,detail,identity,seconds=item
                    path=input_reuse.get((job['sample_key'],job['level'],identity['input_hash'])) or source_path(job)
                    prior=read(path) if path else None
                    if prior and prior['identity']['input_hash']==identity['input_hash'] and prior['grid']==job['grid'] and prior['F']==job['F']:
                        result=copy.deepcopy(prior);result.update(generation_seconds=0.,prefill_seconds=0.,decode_seconds=0.,reused_identical_input=True,reuse_source=str(path),reuse_source_sha256=sha(path))
                    else:
                        need(cpu is not None,'Missing tensor for fresh inference');result=old.generate_fresh(cpu,detail,identity,model,processor,GEN,seconds)
                    results.append(result)
                for j,item,result in zip(group,items,results):save(j,item,result)
                done+=len(group);counts.update(j['dataset']+'|'+j['level']+'|'+j['condition'] for j in group);status(completed=done,total=len(jobs),sample_key=group[-1]['sample_key'],counts=dict(counts))
                del items,results
        prep_future.result();prep_pool.shutdown();del model;torch.cuda.empty_cache();status(stage='reporting')
        from spatial_sweep_report import report
        report()
        from artifact_store import register
        for kind,path in [('predictions',OUT/'responses'),('inputs',OUT/'inputs'),('manifests',OUT/'metadata'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Fixed per-frame spatial area pilot900','_lib/spatial_budget_pilot900.py','pilot_900','complete',[PARENT+'.inputs'])
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True);status(state='complete',stage='complete',completed=len(jobs),report=str(REPORT/'REPORT.md'))
    except BaseException:
        status(state='aborted',error=traceback.format_exc());raise
    finally:stop.set()
if __name__=='__main__':
    OUT.joinpath('metadata').mkdir(parents=True,exist_ok=True)
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if '--preflight' in sys.argv:
            j,g=preflight();print(read(OUT/'metadata/preflight.json'))
        else:run()
