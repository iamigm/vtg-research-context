"""Fixed-budget VTG temporal pilot. New outputs only; no legacy writes.

Run with the existing zero-shot .venv. Input integrity gates precede model load.
"""
from pathlib import Path
import os, sys, json, hashlib, time, copy, math, random, threading, traceback, inspect
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parent/'_shared/timelens'
sys.path[:0] = [str(ROOT/'_lib'), str(SHARED/'_lib'), str(SHARED/'runtime/timelens_deps'), str(SHARED/'vendor/TimeLens')]
os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[_k]='1'
RUN = 'adaptive_temporal_sampling_pilot_v1'
OUT = ROOT/'artifacts/runs'/RUN
REPORT = ROOT/'results'/RUN
STORE = ROOT/'artifacts/common'
MODEL = Path('/home1/irteam/backbones/models/Qwen3-VL-8B-Instruct')
LEGACY = STORE/'predictions/legacy'
CONDITIONS = ['uniform','random_s42','gt','pred','random_s43','random_s44']
CONFIG = dict(run_id=RUN, model='Qwen/Qwen3-VL-8B-Instruct', revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
    full_head=True, order='video_query', do_sample=False, max_new_tokens=512, repetition_penalty=1.0, seed=42,
    dtype='bfloat16', attention='flash_attention_2', min_tokens=64, total_tokens=14336, max_frames=768,
    anchor_fraction=.5, window_fraction=.4, planner_seeds=[42,43,44], coverage_epsilon=.5,
    bootstrap_replicates=20000, bootstrap_seed=42, benchmark='charades-timelens', query_count=100,
    timestamp_contract='native source frames_indices / original fps; CFR verified against decoder PTS; native pair average and one-decimal timestamp text',
    decode='unchanged decord 0.6.0 sparse get_batch; shared immutable candidate RGB with baseline bridge',
    response_limit=700, scout_policy='100 fresh scouts; historical pixel provenance insufficient for reuse',trace_capture=False, ground_truth='isolated planner/evaluator only')

def stamp(): return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def encoded(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def objhash(x): return hashlib.sha256(encoded(x)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def immutable(p,x):
    if Path(p).exists(): assert encoded(read(p))==encoded(x), f'Immutable JSON mismatch: {p}'
    else: write(p,x)
def fileinfo(p):
    p=Path(p); return dict(path=str(p),resolved=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def tensorhash(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def need(v,msg):
    if not v: raise RuntimeError(msg)

class Progress:
    def __init__(self):
        self.start=time.monotonic();self.state=dict(state='building',stage='initializing',model='qwen3-instruct',dataset='charades-timelens',completed=0,total=600,reused_scouts=0,counts={});self.lock=threading.Lock();self.stop=threading.Event()
    def update(self,**kw):
        with self.lock:self.state.update(kw)
        self.emit()
    def emit(self):
        with self.lock: s=copy.deepcopy(self.state)
        elapsed=time.monotonic()-self.start;n=s['completed'];s.update(timestamp=stamp(),elapsed_seconds=elapsed,percentage=100*n/600,eta_seconds=(elapsed*(600-n)/n if n else None),eta_scope='remaining final generation; estimate includes startup and excludes final report')
        write(OUT/'metadata/status.json',s)
        line=json.dumps(s,ensure_ascii=False)
        with (OUT/'logs/progress.log').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
    def heartbeat(self):
        while not self.stop.wait(20):self.emit()

def validate_indices(ids,pts):
    need(len(ids)==len(pts) and len(ids)>0,'Missing frame/PTS')
    need(len(set(ids))==len(ids) and all(a<b for a,b in zip(ids,ids[1:])),'Repeated/reversed source IDs')
    need(all(math.isfinite(t) for t in pts) and all(a<b for a,b in zip(pts,pts[1:])),'Invalid/reversed/duplicate PTS')

def select_plan(public,condition,seed=42,scout=None,gt=None):
    # Public schema intentionally excludes all annotations and performance labels.
    allowed={'sample_key','duration','baseline_ids','candidate_ids','candidate_pts'}
    need(set(public)==allowed,'Planner public schema violation')
    base=public['baseline_ids'];ids=public['candidate_ids'];pts=public['candidate_pts'];T=public['duration'];F=len(base)
    validate_indices(ids,pts);need(len(ids)>=F,'Insufficient distinct source frames')
    A=F//2;need(A>=2,'Unspecified A<2 case: abort')
    # Nearest integer with exact half ties toward the smaller source ID.
    anchors=[base[(2*j*(F-1)+(A-1)-1)//(2*(A-1))] for j in range(A)]
    fallback=False;clamped=False;c=None
    if condition=='uniform': selected=base[:];focus=[];fill=[];window=None
    else:
        if condition.startswith('random'):
            rng=random.Random(int.from_bytes(hashlib.sha256(f'{RUN}|{public["sample_key"]}|{seed}'.encode()).digest()[:8],'big'));c=rng.random()*T
        elif condition=='gt':need(gt is not None,'GT planner requires isolated interval');c=sum(gt)/2
        elif condition=='pred':
            p=scout
            fallback=not p or len(p)!=2 or not all(math.isfinite(x) for x in p) or p[0]>=p[1]
            if not fallback:c=sum(p)/2
        else:raise ValueError(condition)
        if fallback:selected=base[:];focus=[];fill=[];window=None
        else:
            clamped=c<0 or c>T;c=max(0,min(T,c));w=.4*T;lo=max(0,min(T-w,c-w/2));window=[lo,lo+w]
            inside=[i for i,t in zip(ids,pts) if i not in set(anchors) and lo<=t<=lo+w];K=F-A;n=len(inside)
            focus=[inside[math.floor((j+.5)*n/K)] for j in range(K)] if n>=K else inside[:]
            used=set(anchors+focus);remaining=[(max(lo-t,0,t-lo-w),i) for i,t in zip(ids,pts) if i not in used]
            fill=[i for _,i in sorted(remaining)[:K-len(focus)]];selected=sorted(anchors+focus+fill)
    need(len(selected)==F and len(set(selected))==F,'Frame budget/uniqueness failure')
    mapping=dict(zip(ids,pts))
    return dict(sample_key=public['sample_key'],condition=condition,planner_seed=seed if condition.startswith('random') else None,
        frame_ids=selected,pts=[mapping[i] for i in selected],anchors=anchors if condition!='uniform' and not fallback else [],
        focus=focus,fill=fill,fill_fraction=len(fill)/max(1,F-A),window=window,center=c,pred_fallback=fallback,center_clamped=clamped,
        F=F,source_unique_frames=len(set(selected)))

def sampler_checks():
    p=dict(sample_key='fixture',duration=10.,baseline_ids=list(range(0,100,10)),candidate_ids=list(range(100)),candidate_pts=[i/10 for i in range(100)])
    tests=[]
    for cond,gt,scout in [('uniform',None,None),('gt',[0,1],None),('gt',[9,10],None),('gt',[0,10],None),('pred',None,[-10,-2]),('pred',None,[5,4]),('pred',None,None)]:
        r=select_plan(p,cond,scout=scout,gt=gt);need(r['F']==10,'fixture budget');tests.append(dict(condition=cond,plan=r))
    rs=[select_plan(p,'random_s'+str(s),s) for s in [42,43,44]];need(len({objhash(x) for x in rs})==3,'Seeds not distinct')
    for ids,pts in [([1,2],None),([2,1],[.2,.1]),([1,1],[.1,.1]),([1,2],[.1,float('nan')])]:
        try:validate_indices(ids,pts or [])
        except RuntimeError:pass
        else:raise RuntimeError('Invalid PTS accepted')
    short=dict(p, candidate_ids=[0],candidate_pts=[0.])
    try:select_plan(short,'uniform')
    except RuntimeError:pass
    else:raise RuntimeError('Candidate shortage accepted')
    return dict(passed=True,fixtures=tests,random_plans=rs,invalid_pts_rejected=True)

def preflight(progress):
    import numpy as np,torch,transformers,decord,flash_attn,qwen_vl_utils
    from result_layout import relocate_path
    from artifact_store import iter_records,read_record,register
    from timelens.utils import extract_time,iou
    from timelens.dataset.timelens_data import parse_query
    from qwen_vl_utils import vision_process
    torch.set_num_threads(2)
    manifest=LEGACY/'cohorts/pilot_300/samples.json';selection=LEGACY/'cohorts/pilot_300/selection.json'
    allrows=read(manifest);rows=sorted([s for s in allrows if s['dataset']=='charades-timelens'],key=lambda s:s['sample_key'])
    need(len(rows)==100 and len({s['sample_key'] for s in rows})==100,'Cohort mismatch')
    original=LEGACY/'qwen3/samples.json';need(sha(original)==read(selection)['source_sha256'],'Selection source hash mismatch')
    originalmap={s['sample_key']:s for s in read(original)};need(all(originalmap[s['sample_key']]==s for s in rows),'Original mapping mismatch')
    annotation=Path('/data/video/MR/TimeLens/TimeLens-Bench/charades-timelens.json');ann=read(annotation)
    entries={e['sample_key']:e for e in iter_records(model='qwen3',dataset='charades-timelens')}
    model_registry=next(r for r in read('/home1/irteam/backbones/manifests/registry.json') if r['name']=='Qwen3-VL-8B-Instruct')
    need(model_registry['revision']==CONFIG['revision'],'Checkpoint revision mismatch')
    for name,h in model_registry['metadata_sha256'].items():need(sha(MODEL/name)==h,f'Model metadata hash mismatch: {name}')
    model_files=[fileinfo(p) for p in sorted(MODEL.iterdir()) if p.is_file()]
    env=dict(python=sys.version,executable=sys.executable,python_realpath=str(Path(sys.executable).resolve()),torch=torch.__version__,transformers=transformers.__version__,transformers_path=transformers.__file__,decord=decord.__version__,flash_attn=flash_attn.__version__,cuda=torch.version.cuda,bf16=torch.cuda.is_bf16_supported(),device=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    need(env['bf16'] and env['torch']=='2.6.0+cu124' and env['transformers']=='4.57.1','Environment differs from baseline')
    write(OUT/'environment/runtime.json',env)
    import subprocess
    gpu=subprocess.check_output(['nvidia-smi'],text=True);(OUT/'environment/nvidia-smi.txt').write_text(gpu)
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True);(OUT/'environment/pip-freeze.txt').write_text(freeze)
    codepaths=[Path(__file__),ROOT/'_lib/adaptive_temporal_report.py',ROOT/'_lib/artifact_store.py',ROOT/'_lib/result_layout.py',SHARED/'_lib/timelens_inputs.py',SHARED/'vendor/TimeLens/evaluation/utils.py',SHARED/'vendor/TimeLens/evaluation/eval_dataloader.py',SHARED/'vendor/TimeLens/timelens/utils.py',Path(vision_process.__file__),Path(transformers.__file__).parent/'models/qwen3_vl/processing_qwen3_vl.py',Path(transformers.__file__).parent/'models/qwen3_vl/video_processing_qwen3_vl.py',Path(transformers.__file__).parent/'models/qwen3_vl/modeling_qwen3_vl.py']
    codepaths += [SHARED/'vendor/TimeLens/timelens/dataset/timelens_data.py',REPORT/'SPEC.md']
    sources=[fileinfo(p) for p in codepaths]
    for i,p in enumerate(codepaths):
        dest=OUT/'code'/f'{i:02d}_{sha(p)[:12]}_{p.name}'
        if not dest.exists():dest.write_bytes(p.read_bytes())
    records=[];videos={};oldmetrics=[]
    for n,s in enumerate(rows):
        key=s['sample_key'];progress.update(stage='preflight_sources',current_sample=key,preflight_completed=n)
        video=Path(s['video_path']);vid=video.stem;a=ann[vid]
        need(any(parse_query(q)==s['query'] and span==s['span'][0] for q,span in zip(a['queries'],a['spans'])),'Annotation/query/GT/video mismatch '+key)
        if str(video) not in videos:
            v=decord.VideoReader(str(video),num_threads=1);pts=np.asarray(v.get_frame_timestamp(list(range(len(v))))).tolist();fps=v.get_avg_fps();N=len(v)
            err=float(np.max(np.abs(np.asarray(pts)[:,0]-np.arange(N)/fps)))
            # Decord exposes float32 PTS. Tolerance bounds its storage error only.
            tol=max(1e-6,float(np.spacing(np.float32(N/fps)))*2)
            need(err<=tol,'Native indices/fps cannot preserve source PTS (VFR/nonzero origin): '+key)
            need(abs(N/fps-s['duration'])<1e-8,'Annotation inference-duration mismatch '+key)
            videos[str(video)]=dict(**fileinfo(video),video_id=vid,N=N,fps=fps,pts=pts,pts_error_seconds=err,pts_float32_tolerance=tol,original_annotation_duration=a['duration'],inference_duration=s['duration'])
        old=read(LEGACY/'qwen3'/s['dataset']/key/'metadata.json')
        b=read_record(entries[key],verify=True);need(b['sample']==s and b['order']=='video_query','Full baseline mapping mismatch '+key)
        d=ROOT/'artifacts/runs/thinking_vs_instruct_8fps_v1/samples/instruct'/s['dataset']/key/'video_query'
        eight=read(d/'input.json');er=read(d/'result.json');need(eight['sample']==s and er['sample']==s,'8fps mapping mismatch '+key)
        need(old['sample']==s,'Legacy input mapping mismatch '+key)
        meta=old['decoder_metadata'];F=len(meta['frames_indices']);grid=old['video_grid_thw'];B=math.prod(grid)//4
        need(F==len(set(meta['frames_indices'])) and F%2==0,'Baseline duplicates/padding policy unresolved '+key)
        need(grid[0]*2==F and B==len(old['video_positions']['video_query']),'Baseline token/grid mismatch '+key)
        need(B<=14336 and grid[1]*grid[2]//4>=64,'Baseline token/min budget exceeds requested constraints '+key)
        for m in [meta,eight['metadata']['decoder_metadata']]:
            need(m['fps']==videos[str(video)]['fps'] and m['total_num_frames']==videos[str(video)]['N'],'Saved video metadata differs '+key)
        record=dict(sample=s,video_id=vid,video_sha256=videos[str(video)]['sha256'],F=F,B=B,grid=grid,padding=0,baseline_metadata=old,
            baseline_entry=entries[key],baseline_response_sha256=entries[key]['response_sha256'],baseline_input_sha256=b['input_ids_sha256'],eight_input=fileinfo(d/'input.json'),eight_response=fileinfo(d/'result.json'),eight_metadata=eight['metadata'],annotation_duration=a['duration'])
        records.append(record)
        vals=[]
        for r in [b,er]:
            spans=extract_time(r['answer']) or [];p=spans[0] if spans else None;valid=p and all(math.isfinite(x) for x in p) and p[0]<p[1]
            vals.append(float(iou(s['span'][0],p)) if valid else 0.)
        oldmetrics.append(dict(sample_key=key,iou_2fps=vals[0],iou_8fps=vals[1]))
    need(round(100*np.mean([v['iou_2fps'] for v in oldmetrics]),2)==46.90,'Historical full raw baseline does not reproduce 46.90')
    need(round(100*np.mean([v['iou_8fps'] for v in oldmetrics]),2)==42.81,'Historical 8fps raw baseline does not reproduce 42.81')
    pre=dict(state='building',main_sha='a0824288d3fab917db04d016eeb32d32e170f3c1',main_diff='identical',environment=env,model=model_registry,model_files=model_files,source_files=sources,
        paths=dict(home_symlink=str(Path('/home').resolve()),project=str(ROOT),shared=str(SHARED),requested_shared_exists=(ROOT.parent/'shared/timelens').exists(),registry=str(STORE/'registry')),
        cohort=fileinfo(manifest),selection=fileinfo(selection),annotation=fileinfo(annotation),n_query=100,n_video=len(videos),sampler_tests=sampler_checks(),
        timestamp_gate='CFR source PTS verified to decoder float32 precision; native nonuniform indices supported; processor synthetic and bridge pending',
        evaluation=dict(parser_path=str(SHARED/'vendor/TimeLens/timelens/utils.py'),primary='first interval, raw, no clipping; invalid IoU zero',legacy='Python round then first interval; out-of-video fallback'),pending=['uniform adapter pixel/ID/grid/position equality','decoder candidate/cache equality','all final budgets','GT isolation','generation repeatability','atomic resume'])
    immutable(OUT/'metadata/config.json',CONFIG);immutable(OUT/'metadata/cohort.json',rows)
    immutable(OUT/'metadata/gt_evaluator.json',{s['sample_key']:dict(span=s['span'],video_path=s['video_path']) for s in rows})
    immutable(OUT/'inputs/cohort_public.json',[{k:v for k,v in s.items() if k!='span'} for s in rows])
    write(OUT/'metadata/preflight.json',pre);write(OUT/'metadata/source_manifest.json',dict(files=sources+model_files+[fileinfo(manifest),fileinfo(selection),fileinfo(annotation)],videos=list(videos.values())))
    write(OUT/'metadata/baseline_records.json',records);write(OUT/'metrics/historical.json',oldmetrics)
    for kind in ['predictions','manifests','inputs','analyses']:
        path={'predictions':OUT/'responses','manifests':OUT/'metadata','inputs':OUT/'inputs','analyses':REPORT}[kind];path.mkdir(exist_ok=True,parents=True)
        register(RUN+'.'+kind,kind,path,'Equal-budget adaptive temporal sampling pilot',str(Path(__file__).relative_to(ROOT)),'pilot_300','building',['pilot_300.cohort','full_benchmark_v1.predictions','pilot8fps.v1.predictions'])
    return records,videos,pre

def candidate_cache(record,video,processor,progress):
    import torch,numpy as np,decord
    from torchvision.transforms import functional as TF,InterpolationMode
    from timelens_inputs import prepare_pair
    s=record['sample'];key=s['sample_key'];base=record['baseline_metadata']['decoder_metadata']['frames_indices'];eight=record['eight_metadata']['decoder_metadata']['frames_indices']
    ids=sorted(set(base+eight));validate_indices(ids,[video['pts'][i][0] for i in ids])
    # Cap extension is unnecessary on this cohort unless saved candidates exceed cap.
    need(len(ids)>=len(base),'Candidate count insufficient')
    cachekey=objhash(dict(video=video['sha256'],ids=ids,grid=record['grid'],decoder='decord-0.6.0',processor=sha(MODEL/'video_preprocessor_config.json')))
    path=OUT/'frames'/(cachekey+'.npy');path.parent.mkdir(exist_ok=True)
    begin=time.perf_counter()
    previous_cache=OUT/'frames'/(key+'.json');previous_input=OUT/'inputs/uniform'/(key+'.json')
    if previous_cache.exists() and previous_input.exists():
        saved_cache=read(previous_cache);saved_input=read(previous_input)
        if saved_input.get('bridge_passed') and saved_cache['cache_key']==cachekey and sha(path)==saved_cache['sha256']:
            public=dict(sample_key=key,duration=s['duration'],baseline_ids=base,candidate_ids=ids,candidate_pts=[video['pts'][i][0] for i in ids])
            cpu,detail=make_input(record,video,saved_cache,select_plan(public,'uniform'),processor)
            need(detail['hashes']==saved_input['hashes'] and detail['position_ids']==saved_input['position_ids'],'Saved Uniform bridge changed on resume')
            saved_cache['resume_bridge_check_seconds']=time.perf_counter()-begin
            write(previous_cache,saved_cache)
            return public,saved_cache,cpu,saved_input
    vr=decord.VideoReader(s['video_path'],num_threads=1);raw=vr.get_batch(ids).asnumpy()
    independent=decord.VideoReader(s['video_path'],num_threads=1).get_batch(base).asnumpy()
    loc={i:j for j,i in enumerate(ids)};need(np.array_equal(independent,raw[[loc[i] for i in base]]),'Sampler/seek RGB mismatch '+key)
    grid=record['grid'];resized=TF.resize(torch.from_numpy(raw).permute(0,3,1,2),[grid[1]*16,grid[2]*16],interpolation=InterpolationMode.BICUBIC,antialias=True)
    arr=resized.numpy()
    was_cached=path.exists()
    if was_cached:need(np.array_equal(np.load(path,mmap_mode='r'),arr),'Cache RGB mismatch '+key)
    else:
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.save(f,arr);f.flush();os.fsync(f.fileno())
        tmp.replace(path)
    # Verify a fresh legacy prepare_pair, without any vision/attention cache.
    pair,info=prepare_pair(s,processor,fps=2)
    for k in ['prompt','video_grid_thw','timestamps','duration_video','decoder_metadata','sequence_length']:
        need(info[k]==record['baseline_metadata'][k],'Fresh prepare_pair differs from legacy '+key+' '+k)
    public=dict(sample_key=key,duration=s['duration'],baseline_ids=base,candidate_ids=ids,candidate_pts=[video['pts'][i][0] for i in ids])
    cache=dict(cache_key=cachekey,path=str(path),sha256=sha(path),ids=ids,pts=public['candidate_pts'],rgb_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in raw],resized_rgb_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in arr],seconds=time.perf_counter()-begin,cache_state='warm' if was_cached else 'cold',source_video_hash=video['sha256'])
    plan=select_plan(public,'uniform');new,detail=make_input(record,video,cache,plan,processor)
    for k in pair[0]:need(torch.equal(pair[0][k],new[k]),'Uniform bridge tensor mismatch '+key+' '+k)
    need(tensorhash(new.input_ids)==record['baseline_input_sha256'],'Uniform IDs differ from full baseline '+key)
    # Compare actual rotary positions, independently computed for old and adapter inputs.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    class Holder:
        config=type('Cfg',(),dict(image_token_id=151655,video_token_id=151656,vision_start_token_id=151652,vision_config=type('V',(),dict(spatial_merge_size=2))()))()
    cfg=read(MODEL/'config.json');holder=Holder()
    for name in ['image_token_id','video_token_id','vision_start_token_id']:setattr(holder.config,name,cfg[name])
    oldpos=Qwen3VLModel.get_rope_index(holder,input_ids=pair[0].input_ids,video_grid_thw=pair[0].video_grid_thw,attention_mask=pair[0].attention_mask)
    newpos=Qwen3VLModel.get_rope_index(holder,input_ids=new.input_ids,video_grid_thw=new.video_grid_thw,attention_mask=new.attention_mask)
    need(all(torch.equal(a,b) for a,b in zip(oldpos,newpos)),'Uniform rotary position mismatch '+key)
    detail.update(bridge_passed=True,position_ids_sha256=tensorhash(newpos[0]),rope_delta_sha256=tensorhash(newpos[1]))
    write(OUT/'inputs/uniform'/(key+'.json'),detail);write(OUT/'frames'/(key+'.json'),cache)
    return public,cache,new,detail

def make_input(record,video,cache,plan,processor):
    import numpy as np,torch
    from evaluation.utils import GROUNDER_PROMPT
    ids=plan['frame_ids'];pts=plan['pts'];validate_indices(ids,pts)
    need(len(ids)==record['F'],'F budget mismatch')
    loc={i:j for j,i in enumerate(cache['ids'])}
    selected=np.array(np.load(cache['path'],mmap_mode='r')[[loc[i] for i in ids]])
    need(all(hashlib.sha256(a.tobytes()).hexdigest()==cache['resized_rgb_hashes'][loc[i]] for a,i in zip(selected,ids)),'Selected/cache RGB checksum mismatch')
    frames=torch.from_numpy(selected).float()
    meta=dict(fps=video['fps'],frames_indices=ids[:],total_num_frames=video['N'],video_backend='decord')
    messages=[dict(role='user',content=[dict(type='video',video=record['sample']['video_path']),dict(type='text',text=GROUNDER_PROMPT.format(record['sample']['query']))])]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],videos=[frames],video_metadata=[copy.deepcopy(meta)],do_sample_frames=False,padding=True,return_tensors='pt')
    grid=inputs.video_grid_thw[0].tolist();B=int((inputs.input_ids==processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum())
    need(grid==record['grid'] and B==record['B'],'Condition grid/token mismatch '+record['sample']['sample_key'])
    native=processor._calculate_timestamps(ids[:],video['fps'],processor.video_processor.merge_size)
    expanded=processor.tokenizer.decode(inputs.input_ids[0],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    # Native timestamp rendering is 0.1s; comparison uses float64 rational CFR time.
    expected=[(ids[j]+ids[j+1])/(2*video['fps']) for j in range(0,len(ids),2)]
    need(all(abs(a-b)<1e-10 for a,b in zip(native,expected)),'Native temporal patch mapping failure')
    need(all(f'<{t:.1f} seconds>' in expanded for t in native),'Timestamp token block absent')
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from types import SimpleNamespace
    cfg=read(MODEL/'config.json');holder=SimpleNamespace(config=SimpleNamespace(**{k:cfg[k] for k in ['image_token_id','video_token_id','vision_start_token_id']},vision_config=SimpleNamespace(spatial_merge_size=2)))
    positions,delta=Qwen3VLModel.get_rope_index(holder,input_ids=inputs.input_ids,video_grid_thw=inputs.video_grid_thw,attention_mask=inputs.attention_mask)
    detail=dict(sample_key=record['sample']['sample_key'],frame_ids=ids,source_pts=pts,source_fps=video['fps'],duration=record['sample']['duration'],native_patch_timestamps=native,timestamp_text=[f'<{t:.1f} seconds>' for t in native],grid=grid,F=len(ids),B=B,padding=0,unique_source_frames=len(set(ids)),position_ids=positions.tolist(),rope_delta=delta.tolist(),requested_baseline_fps=2,effective_fps=len(ids)/record['sample']['duration'],
        hashes={k:tensorhash(v) for k,v in inputs.items()},shapes={k:list(v.shape) for k,v in inputs.items()},input_ids=inputs.input_ids[0].tolist(),expanded_prompt=expanded,attention_mask=inputs.attention_mask[0].tolist(),cache_key=cache['cache_key'],plan_sha256=objhash(plan),total_input_tokens=inputs.input_ids.shape[1],source_video_sha256=video['sha256'])
    return inputs,detail

def synthetic_processor_gate(processor,record,video,cache):
    # Same pixels, known nonuniform source IDs; supported metadata only.
    import numpy as np,torch
    from evaluation.utils import GROUNDER_PROMPT
    arr=np.load(cache['path'],mmap_mode='r');frames=torch.from_numpy(np.array(arr[:4])).float()
    text=processor.apply_chat_template([dict(role='user',content=[dict(type='video',video='probe.mp4'),dict(type='text',text='probe')])],tokenize=False,add_generation_prompt=True)
    outputs=[]
    for ids in [[0,1,2,3],[0,1,8,9]]:
        meta=dict(fps=2.,frames_indices=ids,total_num_frames=20,video_backend='decord')
        x=processor(text=[text],videos=[frames],video_metadata=[meta],do_sample_frames=False,return_tensors='pt')
        outputs.append(dict(ids=ids,pixel_hash=tensorhash(x.pixel_values_videos),ids_hash=tensorhash(x.input_ids),expanded=processor.tokenizer.decode(x.input_ids[0],skip_special_tokens=False),timestamps=processor._calculate_timestamps(ids,2.,2)))
    need(outputs[0]['pixel_hash']==outputs[1]['pixel_hash'] and outputs[0]['ids_hash']!=outputs[1]['ids_hash'],'Synthetic timestamp change not reflected in native input')
    need('<4.2 seconds>' in outputs[1]['expanded'],'Unexpected native timestamp rounding')
    return dict(passed=True,probes=outputs,rejection='adapter rejects missing/nonfinite/reversed/duplicate PTS before processor; native metadata alone does not validate all such cases')

def generate_fresh(cpu,detail,identity,model,processor,genkw,prepare_seconds):
    import torch
    events=[]
    def before(module,args,kwargs):
        e=torch.cuda.Event(enable_timing=True);e.record();events.append([e,None])
    def after(module,args,kwargs,output):
        e=torch.cuda.Event(enable_timing=True);e.record();events[-1][1]=e
    a=model.register_forward_pre_hook(before,with_kwargs=True);b=model.register_forward_hook(after,with_kwargs=True)
    torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();gpu=cpu.to('cuda')
    try:
        with torch.inference_mode():output=model.generate(**gpu,**genkw)
    finally:a.remove();b.remove()
    torch.cuda.synchronize();seconds=time.perf_counter()-t;generated=output[0,cpu.input_ids.shape[1]:].tolist()
    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];timings=[x.elapsed_time(y)/1000 for x,y in events]
    return dict(identity=identity,model_revision=CONFIG['revision'],answer=processor.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False),generated_ids=generated,generated_tokens=len(generated),eos_reached=bool(generated and generated[-1] in eos),generation_seconds=seconds,prefill_seconds=timings[0],decode_seconds=sum(timings[1:]),input_prepare_seconds=prepare_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),F=detail['F'],B=detail['B'],grid=detail['grid'],total_input_tokens=detail['total_input_tokens'],completed_at=stamp(),input_detail_hash=objhash(detail))

def run(progress):
    import torch,numpy as np,decord
    from functools import partial
    # The original path also decodes with one thread.
    decord.VideoReader=partial(decord.VideoReader,num_threads=1)
    records,videos,pre=preflight(progress)
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from artifact_store import read_record,register
    from timelens.utils import extract_time
    processor=AutoProcessor.from_pretrained(str(MODEL),padding_side='left',do_resize=False,local_files_only=True)
    config_hash=objhash(CONFIG);cohort_hash=objhash(read(OUT/'metadata/cohort.json'))
    setup=[];validation=[];planning_costs=[]
    # Every original/adapter bridge precedes scout generation and final planning.
    for idx,record in enumerate(records):
        s=record['sample'];key=s['sample_key'];video=videos[s['video_path']];progress.update(stage='uniform_bridge_and_scout',current_sample=key,bridge_completed=idx)
        public,cache,cpu,detail=candidate_cache(record,video,processor,progress)
        if idx==0:pre['synthetic_timestamp_probe']=synthetic_processor_gate(processor,record,video,cache)
        setup.append(dict(record=record,cache=cache,public=public,plans={}))
        validation.append(dict(sample_key=key,uniform_bridge=True,source_rgb_equal=True,scout_reused=False))
        del cpu
        progress.update(bridge_completed=idx+1)
    pre.update(state='passed',pending=['fresh scouts','frozen final plans','smoke/final validation'],bridge_count=100,scouts_reused=0)
    write(OUT/'metadata/preflight.json',pre)
    progress.update(stage='loading_model',current_sample=None)
    t=time.perf_counter();model=AutoModelForImageTextToText.from_pretrained(str(MODEL),dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval();load_seconds=time.perf_counter()-t
    need(model.config._attn_implementation=='flash_attention_2','Model attention backend mismatch')
    genkw=dict(do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512,repetition_penalty=1.0,use_cache=True)
    write(OUT/'environment/generation_config.json',dict(model_generation_config=model.generation_config.to_dict(),overrides=genkw,model_load_seconds=load_seconds))
    scout_completed=0
    for item in setup:
        rec=item['record'];s=rec['sample'];key=s['sample_key'];video=videos[s['video_path']]
        progress.update(stage='fresh_scout_generation',current_sample=key,scout_completed=scout_completed,scout_total=100)
        plan=select_plan(item['public'],'uniform');t=time.perf_counter();cpu,detail=make_input(rec,video,item['cache'],plan,processor);prep=time.perf_counter()-t
        identity=dict(run_id=RUN,config_hash=config_hash,cohort_hash=cohort_hash,sample_key=key,condition='scout',planner_seed=None,input_hash=objhash(detail['hashes']),plan_hash=objhash(plan))
        dest=OUT/'scouts'/(key+'.json')
        if dest.exists():need(read(dest)['identity']==identity,'Scout resume identity mismatch')
        else:
            scout=generate_fresh(cpu,detail,identity,model,processor,genkw,prep)
            scout['parsed']=extract_time(scout['answer']) or [];scout['reuse_verified']=False
            write(dest,scout)
        scout_completed+=1;progress.update(scout_completed=scout_completed)
    # All GT-free plans depend solely on the fresh uniform scouts.
    for item in setup:
        s=item['record']['sample'];key=s['sample_key'];public=item['public'];scout=read(OUT/'scouts'/(key+'.json'));pred=scout['parsed'][0] if scout['parsed'] else None
        for c in CONDITIONS:
            seed=int(c[-2:]) if c.startswith('random') else 42;t=time.perf_counter()
            plan=select_plan(public,c,seed,scout=pred,gt=s['span'][0] if c=='gt' else None);elapsed=time.perf_counter()-t
            immutable(OUT/'plans'/c/(key+'.json'),plan);item['plans'][c]=plan;planning_costs.append(dict(sample_key=key,condition=c,seconds=elapsed))
            if c!='gt':
                for changed in [None,[[0,s['duration']]],[[999999,1000000]]]:
                    altered={k:v for k,v in s.items() if k!='span'}
                    if changed is not None:altered['span']=changed
                    sanitized=dict(public,sample_key=altered['sample_key'],duration=altered['duration'])
                    need(encoded(plan)==encoded(select_plan(sanitized,c,seed,scout=pred)),'GT leakage/planner reproducibility failure')
    plan_index=[dict(condition=c,sample_key=r['record']['sample']['sample_key'],sha256=objhash(r['plans'][c])) for r in setup for c in CONDITIONS]
    immutable(OUT/'metadata/plan_index.json',plan_index);write(OUT/'metrics/planning_cost.json',planning_costs)
    write(OUT/'validation/pre_generation.json',dict(passed=True,records=validation,gt_isolation=True,plan_index_hash=objhash(plan_index),all_plans_frozen_before_final=True))
    completed=0;counts={};smoke=[]
    for ix,item in enumerate(setup):
        rec=item['record'];s=rec['sample'];key=s['sample_key'];video=videos[s['video_path']]
        offset=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'big')%len(CONDITIONS)
        order=CONDITIONS[offset:]+CONDITIONS[:offset]
        for condition in order:
            progress.update(stage='smoke_generation' if ix<5 else 'final_generation',current_sample=key,condition=condition,completed=completed)
            plan=item['plans'][condition];prepare_start=time.perf_counter();cpu,detail=make_input(rec,video,item['cache'],plan,processor);input_prepare_seconds=time.perf_counter()-prepare_start
            input_hash=objhash(detail['hashes']);identity=dict(run_id=RUN,config_hash=config_hash,cohort_hash=cohort_hash,sample_key=key,condition=condition,planner_seed=plan['planner_seed'],input_hash=input_hash,plan_hash=objhash(plan))
            dest=OUT/'responses'/condition/(key+'.json');existing=read(dest) if dest.exists() else None
            if existing:need(existing['identity']==identity,'Resume identity/config/input/plan collision')
            else:
                write(OUT/'inputs'/condition/(key+'.json'),detail)
                events=[]
                def before_forward(module,args,kwargs):
                    event=torch.cuda.Event(enable_timing=True);event.record();events.append([event,None])
                def after_forward(module,args,kwargs,output):
                    event=torch.cuda.Event(enable_timing=True);event.record();events[-1][1]=event
                h1=model.register_forward_pre_hook(before_forward,with_kwargs=True);h2=model.register_forward_hook(after_forward,with_kwargs=True)
                torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();gpu=cpu.to('cuda')
                try:
                    with torch.inference_mode():answer_ids=model.generate(**gpu,**genkw)
                finally:h1.remove();h2.remove()
                torch.cuda.synchronize();seconds=time.perf_counter()-start;generated=answer_ids[0,cpu.input_ids.shape[1]:].tolist()
                eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                forward_seconds=[a.elapsed_time(b)/1000 for a,b in events]
                result=dict(identity=identity,model_revision=CONFIG['revision'],answer=processor.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False),generated_ids=generated,generated_tokens=len(generated),eos_reached=bool(generated and generated[-1] in eos),generation_seconds=seconds,prefill_seconds=forward_seconds[0],decode_seconds=sum(forward_seconds[1:]),input_prepare_seconds=input_prepare_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),F=detail['F'],B=detail['B'],grid=detail['grid'],total_input_tokens=detail['total_input_tokens'],completed_at=stamp(),input_detail_hash=objhash(detail))
                write(dest,result);need(read(dest)==result,'Atomic response readback mismatch')
                del gpu,answer_ids
            result=read(dest)
            if condition=='uniform':
                sc=read(OUT/'scouts'/(key+'.json'));need(result['generated_ids']==sc['generated_ids'],'Repeated Uniform differs from verified scout: '+key)
            need(result['F']==rec['F'] and result['B']==rec['B'] and result['grid']==rec['grid'],'Saved budget mismatch')
            completed+=1;counts[condition]=counts.get(condition,0)+1
            progress.update(completed=completed,counts=counts)
            if ix<5:smoke.append(dict(sample_key=key,condition=condition,passed=True,input_hash=input_hash))
        if ix==4:write(OUT/'validation/smoke.json',dict(passed=True,n_query=5,n_responses=30,records=smoke,repeat_uniform_matches_scout=5,conditions=CONDITIONS))
    del model;torch.cuda.empty_cache()
    progress.update(stage='evaluation_and_reporting',completed=completed)
    from adaptive_temporal_report import build
    summary=build()
    write(OUT/'validation/final.json',dict(passed=True,n_query=100,n_final=600,n_scout=100,new_scouts=100,reused_scouts=0,budget_equality=True,uniform_repeats_equal=100,config_hash=config_hash,cohort_hash=cohort_hash,plan_index_hash=objhash(plan_index)))
    for kind in ['predictions','manifests','inputs','analyses']:
        path={'predictions':OUT/'responses','manifests':OUT/'metadata','inputs':OUT/'inputs','analyses':REPORT}[kind]
        register(RUN+'.'+kind,kind,path,'Validated equal-budget temporal sampling pilot',str(Path(__file__).relative_to(ROOT)),'pilot_300','complete',['pilot_300.cohort','full_benchmark_v1.predictions','pilot8fps.v1.predictions'])
    progress.update(state='complete',stage='complete',report=str(REPORT/'REPORT.md'),validation_passed=True)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['run','sampler-check']);args=ap.parse_args()
    if args.command=='sampler-check':print(json.dumps(sampler_checks(),ensure_ascii=False));return
    for category in ['metadata','provenance','code','environment','logs','validation','metrics','inputs','plans','scouts','responses','frames']:(OUT/category).mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=Progress();thread=threading.Thread(target=progress.heartbeat,daemon=True);thread.start()
        try:run(progress)
        except BaseException:
            error=traceback.format_exc();progress.update(state='aborted',stage='aborted',error=error)
            write(OUT/'validation/abort.json',dict(state='aborted',error=error,timestamp=stamp(),preserved=True))
            (REPORT/'REPORT.md').write_text('# Adaptive Temporal Sampling Pilot v1\n\n상태: **aborted**. 추론·평가 전체 완료가 아닙니다.\n\n```text\n'+error+'\n```\n\n완료분과 기존 원본은 보존했습니다. 상태·검증 근거: [status](../../artifacts/runs/'+RUN+'/metadata/status.json), [abort](../../artifacts/runs/'+RUN+'/validation/abort.json).\n')
            from artifact_store import register
            for kind in ['predictions','manifests','inputs','analyses']:
                path={'predictions':OUT/'responses','manifests':OUT/'metadata','inputs':OUT/'inputs','analyses':REPORT}[kind]
                register(RUN+'.'+kind,kind,path,'Aborted temporal sampling pilot; see abort report',str(Path(__file__).relative_to(ROOT)),'pilot_300','aborted',['pilot_300.cohort'])
            raise
        finally:progress.stop.set();thread.join()

if __name__=='__main__':main()
