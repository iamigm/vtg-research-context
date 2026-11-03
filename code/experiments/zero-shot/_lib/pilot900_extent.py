"""Pilot300: uniform versus whole-GT interval density versus length-matched random.

Whole-event versus endpoint focus; no anchors or Pred. Two planner seeds: 42 and 43.
"""
from result_layout import write_result_text
from result_layout import relocate_path
import sys,os,json,time,math,copy,hashlib,random,bisect,threading,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pilot900_inputs as inputs
import adaptive_temporal_sampling as old
from adaptive_temporal_sampling import read,write,immutable,sha,objhash,fileinfo,need,stamp
RUN='gt_extent_boundary_density_pilot900_v1'
PARENT_RUN='gt_extent_boundary_density_pilot300_v1'
ROOT=old.ROOT;OUT=relocate_path(ROOT/'artifacts/runs'/RUN);REPORT=relocate_path(ROOT/'results'/RUN)
inputs.RUN=RUN;inputs.OUT=OUT;inputs.REPORT=REPORT
RATIOS=[2,4,8]
CONDITIONS=['uniform']+[f'{who}_{mode}_r{r}'+(f'_s{seed}' if who=='random' else '') for r in RATIOS for mode in ['whole','boundary'] for who in ['gt','random'] for seed in ([42,43] if who=='random' else [None])]
TOTAL=900*len(CONDITIONS)
CONFIG=dict(run_id=RUN,cohort='existing pilot_900, 300 per benchmark',density_ratios=RATIOS,boundary_radius_seconds=1.0,boundary_allocation="same requested K as whole-GT at each ratio; capacity transfers recorded",random_seeds=[42,43],model='Qwen3-VL-8B-Instruct',revision=old.CONFIG['revision'],do_sample=False,max_new_tokens=512,repetition_penalty=1.,generation_seed=42,dtype='BF16',backend='FlashAttention2',order='video_query',frame_budget='original 2fps actual F; original H/W and actual visual tokens',rounding='Python round, ties to even',inside_boundary='s <= PTS <= e',outside_axis='concatenate [0,s) and (e,T]; bin centers in concatenated time',source_selection='ordered nearest distinct source frame, ties lower source ID, retain enough candidates for remaining targets',min_allocation='one per nonempty side if F>=2; then clamp to source capacity and transfer remainder',whole_video='whole mode uses Uniform; boundary mode still probes endpoints',GT_usage='GT location only in GT planner/evaluator; Random is diagnostic and uses GT LENGTH only',uniform='F equal-duration cells over [0,T], nearest source frame to each center',bootstrap_replicates=20000,bootstrap_seed=42,full_oracle_ratio=.8,inference='single pass; identical inputs within a query may reuse an earlier deterministic response, explicitly recorded')

CONFIG.update(planner_rng_namespace='gt_extent_boundary_density_pilot300_v1',cohort_manifest='artifacts/common/cohorts/nested_pilot_v1/manifest.json',cohort_design='nested joint-cell-balanced expansion, original300 retained')

def choose(candidates,targets):
    """Candidates (source ID, time coordinate), sorted in source/time order."""
    need(len(candidates)>=len(targets),'Insufficient distinct source frames')
    coords=[x[1] for x in candidates];selected=[];errors=[];last=-1
    for j,target in enumerate(targets):
        lo=last+1;hi=len(candidates)-(len(targets)-j);at=bisect.bisect_left(coords,target,lo,hi+1)
        options={max(lo,min(hi,at)),max(lo,min(hi,at-1))};last=min(options,key=lambda k:(abs(coords[k]-target),candidates[k][0]))
        selected.append(candidates[last][0]);errors.append(abs(coords[last]-target))
    return selected,errors

def make_plan(public,condition,gt=None,length=None):
    need(set(public)=={'sample_key','T','F','ids','pts'},'Unexpected planner fields')
    key,T,F=public['sample_key'],public['T'],public['F']
    mapping=dict(zip(public['ids'],public['pts']));allc=[(i,t) for i,t in mapping.items() if 0<=t<=T]
    targets=[(j+.5)*T/F for j in range(F)];uniform,errors=choose(allc,targets)
    base=dict(sample_key=key,condition=condition,F=F,T=T)
    if condition=='uniform':return dict(base,frame_ids=uniform,pts=[mapping[i] for i in uniform],interval=None,focus_intervals=[[0,T]],target_time_absolute_deviations=errors)
    parts=condition.split('_');who,mode=parts[:2];ratio=int(parts[2][1:])
    if who=='gt':
        need(gt is not None,'Missing GT');start,end=gt;L=end-start
    else:
        need(gt is None and length is not None,'Random accepts length only');L=length
        seed=int(parts[3][1:]);rng=random.Random(int.from_bytes(hashlib.sha256(f'gt_extent_boundary_density_pilot300_v1|{key}|{seed}'.encode()).digest()[:8],'big'));start=rng.random()*(T-L);end=start+L
    need(0<=start<end<=T+1e-10,'Invalid GT or random interval')
    raw=[(start,end)] if mode=='whole' else [(max(0,start-1),min(T,start+1)),(max(0,end-1),min(T,end+1))]
    focus=[]
    for a,b in sorted(raw):
        if focus and a<=focus[-1][1]:focus[-1][1]=max(b,focus[-1][1])
        elif b>a:focus.append([a,b])
    outside=[];last=0.
    for a,b in focus:
        if a>last:outside.append([last,a])
        last=b
    if last<T:outside.append([last,T])
    D=sum(b-a for a,b in focus);requested=round(F*ratio*L/(ratio*L+T-L))
    def coordinate(t,regions):
        offset=0.
        for a,b in regions:
            if a<=t<=b:return offset+t-a
            offset+=b-a
        raise ValueError('Frame outside mapped regions')
    inc=[(i,coordinate(t,focus)) for i,t in allc if any(a<=t<=b for a,b in focus)]
    inset={i for i,_ in inc};outc=[(i,coordinate(t,outside)) for i,t in allc if i not in inset]
    adjusted=requested
    if inc and outc and F>=2:adjusted=max(1,min(F-1,adjusted))
    elif not outc:adjusted=F
    elif not inc:adjusted=0
    low=max(0,F-len(outc));high=min(F,len(inc));need(low<=high,'Insufficient candidates')
    ni=max(low,min(high,adjusted));no=F-ni
    ti=[(j+.5)*D/ni for j in range(ni)];to=[(j+.5)*(T-D)/no for j in range(no)]
    si,ei=choose(inc,ti);so,eo=choose(outc,to);ids=sorted(si+so)
    need(len(ids)==F and len(set(ids))==F,'Frame budget mismatch')
    return dict(base,frame_ids=ids,pts=[mapping[i] for i in ids],interval=[start,end],length=L,ratio=ratio,mode=mode,focus_intervals=focus,focus_duration=D,inside_requested=requested,inside_actual=ni,outside_actual=no,minimum_adjustment=adjusted-requested,capacity_transfer=ni-adjusted,inside_target_interval=D/ni if ni else None,outside_target_interval=(T-D)/no if no else None,inside_target_time_absolute_deviations=ei,outside_target_time_absolute_deviations=eo,density_ratio_actual=(ni/D)/(no/(T-D)) if ni and no else None)

def checks():
    public=dict(sample_key='fixture',T=30.,F=60,ids=list(range(901)),pts=[i/30 for i in range(901)])
    for span in [[6,18],[0,5],[25,30],[10,10.01],[0,30]]:
        for ratio in RATIOS:
            plans=[make_plan(public,f'gt_{mode}_r{ratio}',gt=span) for mode in ['whole','boundary']]
            need(plans[0]['inside_requested']==plans[1]['inside_requested'],'Requested budget differs')
            for plan in plans:need(len(set(plan['frame_ids']))==60,'Duplicate source frames')
            for seed in [42,43]:
                a=make_plan(public,f'random_whole_r{ratio}_s{seed}',length=span[1]-span[0]);b=make_plan(public,f'random_boundary_r{ratio}_s{seed}',length=span[1]-span[0])
                need(a['interval']==b['interval'],'Random locations not paired')
    u=make_plan(public,'uniform');g=make_plan(public,'gt_whole_r2',gt=[0,30]);need(u['frame_ids']==g['frame_ids'],'Whole-video whole-mode bridge failed')
    return dict(passed=True,ratios=RATIOS,seeds=[42,43],conditions=len(CONDITIONS),edge_capacity_overlap_and_random_pairing=True)


class Progress:
    def __init__(self):self.lock=threading.RLock();self.start=time.monotonic();self.stop=threading.Event();self.events=[];self.state=dict(state='building',stage='starting',completed=0,total=17100,new_gpu_responses=0,reused_identical_inputs=0,counts={})
    def update(self,**kw):
        with self.lock:self.state.update(kw);self.emit()
    def emit(self):
        with self.lock:
            n=self.state['completed'];eta=(17100-n)*(self.events[-1][0]-self.events[0][0])/(self.events[-1][1]-self.events[0][1]) if len(self.events)>1 else None
            self.state.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start,percentage=100*n/TOTAL,eta_seconds=eta,eta_scope='remaining current 17100-response generation; excludes input preparation and report')
            write(OUT/'metadata/status.json',self.state)
            with (OUT/'logs/progress.log').open('a') as f:f.write(json.dumps(self.state,ensure_ascii=False)+'\n')
            print(f"{self.state['timestamp']} {self.state['stage']} {n}/17100 {self.state.get('sample_key','')}",flush=True)
    def done(self,dataset,condition,reused):
        with self.lock:
            self.state['completed']+=1;key=dataset+'/'+condition;self.state['counts'][key]=self.state['counts'].get(key,0)+1
            field='reused_identical_inputs' if reused else 'new_gpu_responses';self.state[field]+=1;self.events.append((time.monotonic(),self.state['completed']));self.events=self.events[-100:];self.emit()
    def heartbeat(self):
        while not self.stop.wait(20):self.emit()

def evaluate(rows,cache_index):
    from pilot900_density_report import evaluate_run
    return evaluate_run(sys.modules[__name__],rows)

REUSE_RUNS=['gt_extent_boundary_density_pilot300_v1']
_REUSE_CONFIG={};_REUSE_COHORT={}
def reusable_source(key,condition,h,record):
    for source_run in REUSE_RUNS:
        root=relocate_path(ROOT/'artifacts/runs'/source_run);path=root/'responses'/condition/(key+'.json')
        if not path.exists():continue
        if source_run not in _REUSE_CONFIG:
            need(read(root/'metadata/status.json')['state']=='complete','Reuse source incomplete')
            _REUSE_CONFIG[source_run]=read(root/'metadata/config.json');_REUSE_COHORT[source_run]={s['sample_key']:s for s in read(root/'metadata/cohort.json')}
        need(_REUSE_COHORT[source_run][key]==record['sample'],'Source query/GT/video identity mismatch')
        for field in ['model','revision','do_sample','max_new_tokens','repetition_penalty','generation_seed','dtype','backend','order']:
            need(_REUSE_CONFIG[source_run][field]==CONFIG[field],'Source generation config mismatch')
        result=read(path)
        if result['identity']['input_hash']==h:
            detail=read(root/'inputs'/condition/(key+'.json'));need(objhash(detail['hashes'])==h and objhash(detail)==result['input_detail_hash'],'Reuse provenance mismatch');return path
    return None

def main():
    import torch,fcntl,av,subprocess
    from artifact_store import register
    for category in ['metadata','plans','inputs','responses','metrics','validation','logs','code','environment']:(OUT/category).mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);progress=Progress();thread=threading.Thread(target=progress.heartbeat,daemon=True);thread.start()
        try:
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'GT whole/boundary density 2/4/8 pilot900; seeds42/43',str(Path(__file__).relative_to(ROOT)),'pilot_900','building',['nested_pilot_v1.900.cohort'])
            rows=inputs.records();resolutions=[]
            for record in rows:
                s=record['sample'];gt=s['span'][0];T=s['duration'];need(0<=gt[0]<gt[1],'Invalid GT annotation')
                if gt[1]>T:
                    with av.open(s['video_path']) as c:media_T=c.duration/1e6
                    need(gt[1]<=media_T,'GT outside both annotation duration and actual media; annotation needs review')
                    resolutions.append(dict(sample_key=s['sample_key'],annotation_duration=T,gt=gt,container_duration=media_T,planning_T=media_T,reason='Original annotation/video correspondence verified; unmodified GT fits actual file playback duration'))
                    T=media_T
                record['planning_T']=T
            write(OUT/'validation/annotation_checks.json',dict(passed=True,queries=900,duration_resolutions=resolutions,GT_modified=False))
            write(OUT/'validation/sampler_checks.json',checks());immutable(OUT/'metadata/config.json',CONFIG);immutable(OUT/'metadata/cohort.json',[r['sample'] for r in rows])
            progress.update(stage='source_frame_preparation')
            if not (OUT/'metadata/frame_cache_index.json').exists():inputs.prepare_pilot()
            index=read(OUT/'metadata/frame_cache_index.json');need(len(index)==900,'Incomplete pilot source inputs')
            sourcefiles=[Path(__file__),Path(inputs.__file__),Path(old.__file__),ROOT/'_lib/adaptive_temporal_report.py',ROOT/'_lib/boundary_oracle_analysis.py',ROOT/'_lib/pilot900_density_report.py'];sources=[fileinfo(p) for p in sourcefiles];code_hash=objhash(sources);config_hash=objhash(CONFIG);cohort_hash=objhash([r['sample'] for r in rows])
            for p in sourcefiles:(OUT/'code'/(sha(p)[:12]+'_'+p.name)).write_bytes(p.read_bytes())
            write(OUT/'metadata/source_manifest.json',dict(files=sources,cohort=fileinfo(ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'),frame_cache_index=fileinfo(OUT/'metadata/frame_cache_index.json')))
            jobs=[]
            for record in rows:
                s=record['sample'];key=s['sample_key'];cache=read(index[key]);public=dict(sample_key=key,T=record['planning_T'],F=record['F'],ids=cache['ids'],pts=cache['pts']);gt=s['span'][0];L=gt[1]-gt[0]
                for condition in CONDITIONS:
                    p=make_plan(public,condition,gt=gt if condition.startswith('gt_') else None,length=L if condition.startswith('random') else None)
                    if condition.startswith('random'):need(old.encoded(p)==old.encoded(make_plan(copy.deepcopy(public),condition,length=L)),'Random plan not reproducible')
                    immutable(OUT/'plans'/condition/(key+'.json'),p);jobs.append(dict(sample_key=key,dataset=s['dataset'],condition=condition,plan_hash=objhash(p)))
            immutable(OUT/'metadata/jobs.json',jobs);write(OUT/'validation/plans_frozen.json',dict(passed=True,plans=17100,config_hash=config_hash,cohort_hash=cohort_hash,code_hash=code_hash))
            from transformers import AutoProcessor,AutoModelForImageTextToText
            torch.set_num_threads(2);progress.update(stage='waiting_for_gpu_memory')
            while torch.cuda.mem_get_info()[0]<28*2**30:time.sleep(20)
            progress.update(stage='loading_model');processor=AutoProcessor.from_pretrained(old.MODEL,padding_side='left',do_resize=False,local_files_only=True);model=AutoModelForImageTextToText.from_pretrained(old.MODEL,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
            write(OUT/'environment/runtime.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),backend='flash_attention_2',model_revision=CONFIG['revision']))
            (OUT/'environment/nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True));lookup={r['sample']['sample_key']:r for r in rows};input_reuse={};smoke=[]
            for job in jobs:
                key=job['sample_key'];condition=job['condition'];record=lookup[key];p=read(OUT/'plans'/condition/(key+'.json'));progress.update(stage='generation',sample_key=key,dataset=job['dataset'],condition=condition)
                t=time.perf_counter();cpu,detail=inputs.make_input(record,read(index[key]),p,processor);prep_seconds=time.perf_counter()-t;h=objhash(detail['hashes']);identity=dict(run_id=RUN,sample_key=key,condition=condition,config_hash=config_hash,cohort_hash=cohort_hash,code_hash=code_hash,plan_hash=job['plan_hash'],input_hash=h)
                dest=OUT/'responses'/condition/(key+'.json');write(OUT/'inputs'/condition/(key+'.json'),detail)
                reuse_key=(key,h);reused=False
                if dest.exists():result=read(dest);need(result['identity']==identity,'Resume config/input/plan mismatch');reused=result.get('reused_identical_input',False)
                elif reuse_key in input_reuse:
                    path=input_reuse[reuse_key];result=copy.deepcopy(read(path));result.update(identity=identity,input_detail_hash=objhash(detail),reused_identical_input=True,reuse_source=str(path),reuse_source_sha256=sha(path),generation_seconds=0.,prefill_seconds=0.,decode_seconds=0.,input_prepare_seconds=prep_seconds);write(dest,result);reused=True
                elif (source := reusable_source(key,condition,h,record)) is not None:
                    result=copy.deepcopy(read(source));result.update(identity=identity,input_detail_hash=objhash(detail),reused_identical_input=True,reuse_source=str(source),reuse_source_sha256=sha(source),generation_seconds=0.,prefill_seconds=0.,decode_seconds=0.,input_prepare_seconds=prep_seconds);write(dest,result);reused=True
                else:
                    result=old.generate_fresh(cpu,detail,identity,model,processor,dict(do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512,repetition_penalty=1.,use_cache=True),prep_seconds);result['reused_identical_input']=False;write(dest,result)
                input_reuse[reuse_key]=dest
                need(result['F']==record['F'] and result['grid']==record['grid'] and result['B']==record['B'],'Condition frame/token/grid mismatch')
                if len(smoke)<24:smoke.append(dict(sample_key=key,condition=condition,F=result['F'],B=result['B'],grid=result['grid']));write(OUT/'validation/smoke.json',dict(passed=True,records=smoke))
                progress.done(job['dataset'],condition,reused)
            del model;torch.cuda.empty_cache();progress.update(stage='evaluation_and_report');evaluate(rows,index)
            for source in sources:need(sha(source['path'])==source['sha256'],'Source changed during inference')
            write(OUT/'validation/final.json',dict(passed=True,queries=900,logical_responses=17100,budget_equality=True,source_frame_duplicates=0,ratios=RATIOS,seeds=[42,43]))
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Complete whole/boundary density sweep pilot900; seeds42/43',str(Path(__file__).relative_to(ROOT)),'pilot_900','complete',['nested_pilot_v1.900.cohort'])
            subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
            progress.update(state='complete',stage='complete',report=str(REPORT/'REPORT.md'))
        except BaseException:
            error=traceback.format_exc();progress.update(state='aborted',stage='aborted',error=error);write_result_text(REPORT/'ERROR.md', '# 실행 중단\n\n완료분과 기존 원본은 보존했습니다.\n\n```text\n'+error+'\n```\n')
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Aborted density pilot; see status/error',str(Path(__file__).relative_to(ROOT)),'pilot_900','aborted',['nested_pilot_v1.900.cohort'])
            raise
        finally:progress.stop.set();thread.join()

if __name__=='__main__':main()
