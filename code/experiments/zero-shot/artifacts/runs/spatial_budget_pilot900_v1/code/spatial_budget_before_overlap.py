"""Fixed per-frame spatial area, frozen parent temporal plans, selected-frame cache only."""
import os,sys,time,math,copy,hashlib,threading,traceback,fcntl,subprocess
from pathlib import Path
import adaptive_temporal_sampling as old
from adaptive_temporal_sampling import ROOT,read,write,immutable,objhash,sha,stamp,need,fileinfo
import pilot900_inputs as bridge
RUN='spatial_budget_pilot900_v1';PARENT='gt_extent_boundary_density_pilot900_v1'
OUT=ROOT/'artifacts/runs'/RUN;REPORT=ROOT/'results'/RUN;PR=ROOT/'artifacts/runs'/PARENT
CONDITIONS=['uniform','gt_whole_r8','random_whole_r8_s42','random_whole_r8_s43']
AREAS={'low':50176,'high':100352}
CONFIG=dict(run_id=RUN,parent=PARENT,areas=AREAS,conditions=CONDITIONS,cohort='nested_pilot_v1/pilot_900',model=str(old.MODEL),revision=old.CONFIG['revision'],seed=42,do_sample=False,max_new_tokens=512,repetition_penalty=1.,dtype='bfloat16',attention='flash_attention_2',resize='torchvision bicubic antialias uint8; directly from original RGB',grid_policy='H,W multiples32; area within10%; minimize abs(log((W/H)/(sourceW/sourceH))), then relative area error, then H,W',temporal_policy='exact parent frame IDs, source PTS, original frame count; no total video token cap',batch_policy='batch4 only if six prespecified inputs exactly match singleton; otherwise singleton',gpu_free_required_GiB=60,cpu_workers=4)
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
    import decord
    for cat in ['metadata','plans','inputs','responses','validation','environment','logs','code','metrics']:(OUT/cat).mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    immutable(OUT/'metadata/config.json',CONFIG)
    if (OUT/'metadata/preflight.json').exists():
        p=read(OUT/'metadata/preflight.json');need(p['config_hash']==objhash(CONFIG),'Changed config');return read(OUT/'metadata/jobs.json'),read(OUT/'metadata/groups.json')
    cohort=read(ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json');immutable(OUT/'metadata/cohort.json',cohort)
    rows={r['sample']['sample_key']:r for r in bridge.records()};index=read(PR/'metadata/frame_cache_index.json');groups={};jobs=[];sizes={};maxB=0
    for sample in sorted(cohort,key=lambda s:(s['dataset'],s['sample_key'])):
        key=sample['sample_key'];r=rows[key];parent=read(index[key]);video=sample['video_path']
        if video not in sizes:
            vr=decord.VideoReader(video,num_threads=1);h,w=vr[0].shape[:2];sizes[video]=(h,w)
        h,w=sizes[video]
        if video not in groups:groups[video]=dict(video=video,parent_metadata=index[key],ids=[],grids={level:choose_grid(h,w,target) for level,target in AREAS.items()},source_hw=[h,w])
        g=groups[video]
        for c in CONDITIONS:
            plan=read(PR/'plans'/c/(key+'.json'));ph=objhash(plan)
            immutable(OUT/'plans'/c/(key+'.json'),plan);g['ids']+=plan['frame_ids']
            for level in AREAS:
                H,W=g['grids'][level];F=len(plan['frame_ids']);grid=[F//2,H//16,W//16];B=grid[0]*grid[1]*grid[2]//4;maxB=max(maxB,B)
                jobs.append(dict(sample_key=key,dataset=sample['dataset'],video=video,level=level,condition=c,F=F,grid=grid,B=B,plan_hash=ph,source_hw=[h,w],target_pixels=AREAS[level],actual_pixels=H*W,area_error_fraction=H*W/AREAS[level]-1,aspect_ratio_error_fraction=(W/H)/(w/h)-1))
    for g in groups.values():g['ids']=sorted(set(g['ids']))
    # Stable video order; alternate spatial level to reduce execution-time confounding.
    jobs.sort(key=lambda j:(j['dataset'],j['sample_key'],CONDITIONS.index(j['condition']),list(AREAS).index(j['level']) ^ (int(hashlib.sha256(j['sample_key'].encode()).hexdigest()[:2],16)%2)))
    cfg=read(old.MODEL/'config.json');limit=cfg.get('text_config',cfg).get('max_position_embeddings',262144)
    need(maxB+10000+512<limit,'Projected context exceeds supported length')
    immutable(OUT/'metadata/jobs.json',jobs);immutable(OUT/'metadata/groups.json',list(groups.values()))
    write(OUT/'metadata/preflight.json',dict(passed=True,config_hash=objhash(CONFIG),queries=900,logical_responses=len(jobs),videos=len(groups),max_visual_tokens=maxB,context_limit=limit,area_tolerance=.1,parent_frame_ids_unchanged=True,source_cohort=fileinfo(ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json')))
    for p in [Path(__file__),ROOT/'_lib/spatial_budget_report.py',Path(bridge.__file__),ROOT/'_lib/gt_density_batch.py',ROOT/'_lib/shared_selected_frames.py']:
        (OUT/'code'/p.name).write_bytes(p.read_bytes())
    write(OUT/'metadata/source_manifest.json',dict(files=[fileinfo(p) for p in [Path(__file__),ROOT/'_lib/spatial_budget_report.py',Path(bridge.__file__),ROOT/'_lib/gt_density_batch.py',ROOT/'_lib/shared_selected_frames.py']],model_config=fileinfo(old.MODEL/'config.json'),processor=fileinfo(old.MODEL/'video_preprocessor_config.json')))
    (REPORT/'SPEC.md').write_text('# 공간 예산 통제 실험\n\n900개 × 두 공간 예산 × Uniform/GT 전체8배/Random42/43 = 7200개 논리 응답.\n\n목표 면적은 50,176·100,352픽셀. 32배수 H/W 중 목표 면적 ±10% 안에서 원본 종횡비 오차가 가장 작은 크기를 선택하며, 동률은 면적 오차, H, W 순. 실제 면적과 종횡비 오차를 모두 보고한다. 기존 temporal plan을 그대로 사용하며 원본 RGB에서 직접 resize한다. 모델/프롬프트/생성 설정은 기존과 같다. 전체 영상 토큰 예산을 다시 적용하지 않는다.\n\n전체·경계2/4/8 기존 sweep은 완료 결과로 보존한다. 이번 신규 실행은 승인된 전체8배 네 조건만 포함한다. 원래300/600/900 각각과 세 데이터셋의 mIoU, IoU≥.3/.5/.7 실제 점수·차이, 영상 길이·GT 비율·두 축 교차표, 영상 cluster bootstrap을 자동 보고한다.\n')
    (REPORT/'REPORT.md').write_text('# 공간 예산 통제 실험 — 실행 대기\n\n완료 후 자동 보고서로 교체됩니다. [명세](SPEC.md)\n')
    return jobs,list(groups.values())

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

def prepare_all(groups):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    mapping={};completed=0
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(prepare_group,g) for g in groups]
        for f in as_completed(futures):
            r=f.result();mapping[r['video']]=r;completed+=1;write(OUT/'metadata/prepared.json',mapping);status(stage='preparing_selected_frames',prepared_videos=completed,total_videos=len(groups))
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
        mapping=prepare_all(groups)
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
            cache=read(mapping[job['video']]['metadata'][job['level']]);cpu,detail=bridge.make_input(r,cache,p,local.processor)
            need(detail['total_input_tokens']+512<read(OUT/'metadata/preflight.json')['context_limit'],'Actual context overflow')
            detail.update(spatial_level=job['level'],target_pixels=job['target_pixels'],actual_pixels=job['actual_pixels'],area_error_fraction=job['area_error_fraction'],aspect_ratio_error_fraction=job['aspect_ratio_error_fraction'])
            identity=dict(run_id=RUN,sample_key=job['sample_key'],condition=job['condition'],spatial_level=job['level'],config_hash=config_hash,cohort_hash=cohort_hash,code_hash=code_hash,plan_hash=job['plan_hash'],input_hash=objhash(detail['hashes']))
            return cpu,detail,identity,time.perf_counter()-started
        def dest(job,kind):return OUT/kind/job['level']/job['condition']/(job['sample_key']+'.json')
        def save(job,item,result):
            cpu,detail,identity,seconds=item;result.update(identity=identity,input_detail_hash=objhash(detail),input_prepare_seconds=seconds,model_revision=CONFIG['revision']);write(dest(job,'inputs'),detail);write(dest(job,'responses'),result)
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
            probes=[next(j for j in jobs if j['dataset']==ds and j['level']==level and j['condition']=='uniform') for ds in sorted({j['dataset'] for j in jobs}) for level in AREAS]
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
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={};next_submit=0
            for start in range(0,len(todo),batchsize):
                while next_submit<min(len(todo),start+12):pending[next_submit]=pool.submit(prepare,todo[next_submit]);next_submit+=1
                group=todo[start:start+batchsize];items=[pending.pop(i).result() for i in range(start,start+len(group))]
                if len(items)>1 and sum(x[1]['total_input_tokens'] for x in items)<=60000:
                    try:results=generate([x[:3] for x in items],model,processor,GEN)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache();status(batch_oom_fallback=True)
                        results=[old.generate_fresh(*x[:3],model,processor,GEN,x[3]) for x in items]
                else:results=[old.generate_fresh(*x[:3],model,processor,GEN,x[3]) for x in items]
                for j,item,result in zip(group,items,results):save(j,item,result)
                done+=len(group);counts.update(j['dataset']+'|'+j['level']+'|'+j['condition'] for j in group);status(completed=done,total=len(jobs),sample_key=group[-1]['sample_key'],counts=dict(counts))
                del items,results
        del model;torch.cuda.empty_cache();status(stage='reporting')
        from spatial_budget_report import report
        report()
        from artifact_store import register
        for kind,path in [('predictions',OUT/'responses'),('inputs',OUT/'inputs'),('manifests',OUT/'metadata'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Fixed per-frame spatial area pilot900','_lib/spatial_budget_pilot900.py','pilot_900','complete',[PARENT+'.predictions'])
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
