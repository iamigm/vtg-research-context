"""Detached pilot300: prior midpoint hypothesis + boundary and budget follow-up."""
from result_layout import write_result_text
from boundary_oracle_sampling import *
from boundary_oracle_inputs import prepare_pilot,records,input_record,make_input
import threading,traceback,time,json,collections,statistics,math

MID=['uniform','gt_midpoint','pred_midpoint','random_midpoint_s42','random_midpoint_s43']
CONFIG.update(run_id=RUN,pilot_queries=300,per_benchmark=100,random_seeds=[42,43],scouts='300 full-budget Uniform responses shared across midpoint and boundary full controls',midpoint_window_fraction=.4,historical_decode='v1 sparse outputs retained; v2 uses the same increasing-read canonical frame cache for every condition',confirmation='quick screen only: produce a fixed-setting power/confirmation plan when a hint exists; no automatic new-video expansion')

class Progress:
    def __init__(self):
        self.lock=threading.RLock();self.start=time.monotonic();self.events=[];self.stop=threading.Event();self.s=dict(state='building',stage='starting',completed=0,total=4800,counts={},timestamp=stamp())
    def update(self,**kw):
        with self.lock:self.s.update(kw);self.emit()
    def done(self,dataset,phase,budget,condition):
        with self.lock:
            self.s['completed']+=1;k='/'.join([dataset,phase,budget,condition]);self.s['counts'][k]=self.s['counts'].get(k,0)+1;self.events.append((time.monotonic(),self.s['completed']));self.events=self.events[-100:];self.emit()
    def emit(self):
        with self.lock:
            n=self.s['completed'];self.s.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start,percentage=100*n/self.s['total'])
            self.s['eta_seconds']=(self.s['total']-n)*(self.events[-1][0]-self.events[0][0])/(self.events[-1][1]-self.events[0][1]) if len(self.events)>1 else None
            self.s['eta_scope']='currently activated generation jobs; excludes conditional addition, CPU preparation and final report'
            write(OUT/'metadata/status.json',self.s)
            with (OUT/'logs/progress.log').open('a') as f:f.write(json.dumps(self.s,ensure_ascii=False)+'\n')
            print(f"{self.s['timestamp']} {self.s['stage']} {n}/{self.s['total']} ({self.s['percentage']:.1f}%) {self.s.get('sample_key','')}",flush=True)
    def heartbeat(self):
        while not self.stop.wait(20):self.emit()

def public_for(record,cache):return dict(sample_key=record['sample']['sample_key'],duration=record['sample']['duration'],baseline_ids=record['baseline_metadata']['decoder_metadata']['frames_indices'],ids=cache['ids'],pts=cache['pts'])

def midpoint(public,condition,scout,gt=None):
    base=public['baseline_ids'];F=len(base);anchors=even_ranks(base,F//2);T=public['duration'];fallback=False
    if condition=='gt_midpoint':center=sum(gt)/2
    elif condition=='pred_midpoint':
        from timelens.utils import extract_time
        spans=extract_time(scout['answer']) or [];p=spans[0] if spans else None;fallback=not p or not all(math.isfinite(v) for v in p) or p[0]>=p[1];center=sum(p)/2 if not fallback else 0.
    else:
        seed=int(condition[-2:]);rng=random.Random(int.from_bytes(hashlib.sha256(f'{RUN}|{public["sample_key"]}|midpoint|{seed}'.encode()).digest()[:8],'big'));center=rng.random()*T
    if fallback:
        p=plan(public,'full','uniform');p.update(condition=condition,pred_fallback=True);return p
    center=max(0,min(T,center));width=.4*T;lo=max(0,min(T-width,center-width/2));hi=lo+width;mapping=dict(zip(public['ids'],public['pts']))
    inside=[i for i in public['ids'] if i not in set(anchors) and lo<=mapping[i]<=hi];K=F-len(anchors);n=len(inside)
    focus=[inside[math.floor((j+.5)*n/K)] for j in range(K)] if n>=K else inside[:]
    used=set(anchors+focus);remaining=sorted((max(lo-mapping[i],0,mapping[i]-hi),i) for i in public['ids'] if i not in used);fill=[i for _,i in remaining[:K-len(focus)]];selected=sorted(anchors+focus+fill)
    return dict(sample_key=public['sample_key'],budget='full',condition=condition,frame_ids=selected,pts=[mapping[i] for i in selected],anchors=anchors,focus=focus,fill=fill,windows=[[lo,hi]],F=F,fill_fraction=len(fill)/K,mode='midpoint',pred_fallback=False)

def response_path(phase,budget,condition,key):
    if condition=='uniform' and budget=='full':phase='shared'
    return OUT/'responses'/phase/budget/condition/(key+'.json')

def freeze_job(record,cache,phase,budget,condition,planned,cohort_hash,code_hash):
    key=record['sample']['sample_key'];p=OUT/'plans'/phase/budget/condition/(key+'.json');immutable(p,planned)
    return dict(sample_key=key,dataset=record['sample']['dataset'],phase=phase,budget=budget,condition=condition,plan_path=str(p),plan_hash=objhash(planned),cohort_hash=cohort_hash,code_hash=code_hash)

def generate_job(job,record,cache,model,processor,config_hash,progress):
    key=job['sample_key'];planned=read(job['plan_path']);need(objhash(planned)==job['plan_hash'],'Frozen plan changed')
    t=time.perf_counter();cpu,detail=make_input(record,cache,planned,processor);prep=time.perf_counter()-t
    identity=dict(run_id=RUN,config_hash=config_hash,code_hash=job['code_hash'],cohort_hash=job['cohort_hash'],sample_key=key,phase=job['phase'],budget=job['budget'],condition=job['condition'],plan_hash=job['plan_hash'],input_hash=objhash(detail['hashes']))
    dest=response_path(job['phase'],job['budget'],job['condition'],key)
    if dest.exists():
        result=read(dest);need(result['identity']==identity,'Mixed config/input/plan on resume')
    else:
        write(OUT/'inputs'/job['phase']/job['budget']/job['condition']/(key+'.json'),detail)
        result=old.generate_fresh(cpu,detail,identity,model,processor,dict(do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512,repetition_penalty=1.,use_cache=True),prep)
        write(dest,result);need(read(dest)==result,'Atomic response readback failed')
    need(result['F']==len(planned['frame_ids']) and result['B']==result['grid'][0]*result['grid'][1]*result['grid'][2]//4,'Measured frame/token/grid mismatch')
    need(result['grid'][1:]==record['grid'][1:],'Spatial image grid changed')
    progress.done(record['sample']['dataset'],job['phase'],job['budget'],job['condition']);return result,detail

def main():
    import torch,subprocess,fcntl
    for name in ['metadata','plans','responses','inputs','logs','metrics','validation','code','environment']:(OUT/name).mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=Progress();thread=threading.Thread(target=progress.heartbeat,daemon=True);thread.start()
        from artifact_store import register
        try:
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Pilot300 temporal sampling v2; Random seeds 42/43 only',str(Path(__file__).relative_to(ROOT)),'pilot_300','building',['pilot_300.cohort'])
            from boundary_oracle_analysis import analyze_previous
            progress.update(stage='previous_result_analysis');analyze_previous()
            progress.update(stage='source_frame_preparation')
            # Resume the independent preparation work or prepare all 300 now.
            index=OUT/'metadata/frame_cache_index.json'
            if not index.exists():prepare_pilot()
            rows=records();cache_index=read(index);need(len(cache_index)==300,'Frame preparation incomplete')
            immutable(OUT/'metadata/config.json',CONFIG);immutable(OUT/'metadata/cohort.json',[r['sample'] for r in rows]);cohort_hash=objhash([r['sample'] for r in rows]);config_hash=objhash(CONFIG)
            write(OUT/'validation/sampler_fixtures.json',fixtures())
            codefiles=[Path(__file__),Path(__file__).with_name('boundary_oracle_inputs.py'),Path(__file__).with_name('boundary_oracle_sampling.py'),Path(__file__).with_name('boundary_oracle_report.py'),Path(__file__).with_name('boundary_oracle_analysis.py'),Path(old.__file__)]
            sources=[fileinfo(p) for p in codefiles];code_hash=objhash(sources)
            for source in sources:(OUT/'code'/(source['sha256'][:12]+'_'+Path(source['path']).name)).write_bytes(Path(source['path']).read_bytes())
            write(OUT/'metadata/source_manifest.json',dict(files=sources,previous_manifest=old.fileinfo(old.OUT/'metadata/source_manifest.json'),cohort=old.fileinfo(old.LEGACY/'cohorts/pilot_300/samples.json'),cache_index=old.fileinfo(index)))
            env=dict(torch=torch.__version__,python=sys.version,bf16=torch.cuda.is_bf16_supported(),gpu=torch.cuda.get_device_name(0));need(env['bf16'],'BF16 unsupported');write(OUT/'environment/runtime.json',env)
            (OUT/'environment/nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True));(OUT/'environment/pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
            from transformers import AutoProcessor,AutoModelForImageTextToText
            torch.set_num_threads(2);processor=AutoProcessor.from_pretrained(old.MODEL,padding_side='left',do_resize=False,local_files_only=True)
            progress.update(stage='loading_model');model=AutoModelForImageTextToText.from_pretrained(old.MODEL,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
            shared={};base_jobs=[]
            # Full-budget Uniform is a fresh shared control, also the Pred scout.
            for record in rows:
                key=record['sample']['sample_key'];cache=read(cache_index[key]);public=public_for(record,cache);p=plan(public,'full','uniform');job=freeze_job(record,cache,'shared','full','uniform',p,cohort_hash,code_hash)
                progress.update(stage='uniform_controls_and_scouts',sample_key=key,dataset=record['sample']['dataset']);result,detail=generate_job(job,record,cache,model,processor,config_hash,progress);shared[key]=result;base_jobs.append(job)
            jobs=[]
            for record in rows:
                key=record['sample']['sample_key'];cache=read(cache_index[key]);public=public_for(record,cache)
                for condition in MID[1:]:
                    p=midpoint(public,condition,shared[key],record['sample']['span'][0] if condition=='gt_midpoint' else None);jobs.append(freeze_job(record,cache,'midpoint','full',condition,p,cohort_hash,code_hash))
                for budget in BUDGETS:
                    for condition in CONDITIONS:
                        if condition=='uniform' and budget=='full':continue
                        p=plan(public,budget,condition,record['sample']['span'][0] if condition=='gt_boundary' else None)
                        if condition.startswith('random'):need(old.encoded(p)==old.encoded(plan(copy.deepcopy(public),budget,condition)),'Random/GT isolation failure')
                        jobs.append(freeze_job(record,cache,'boundary',budget,condition,p,cohort_hash,code_hash))
            # All 4500 non-shared plans are fixed before their answer generation.
            immutable(OUT/'metadata/jobs.json',base_jobs+jobs)
            first={r['sample']['sample_key'] for dataset in sorted({r['sample']['dataset'] for r in rows}) for r in [x for x in rows if x['sample']['dataset']==dataset][:2]}
            jobs.sort(key=lambda j:(j['sample_key'] not in first,j['sample_key'],j['phase'],j['budget'],j['condition']))
            lookup={r['sample']['sample_key']:r for r in rows};smoke=[]
            for job in jobs:
                record=lookup[job['sample_key']];cache=read(cache_index[job['sample_key']]);progress.update(stage='smoke_generation' if job['sample_key'] in first else 'generation',sample_key=job['sample_key'],dataset=job['dataset'],condition=job['condition'],budget=job['budget'])
                result,_=generate_job(job,record,cache,model,processor,config_hash,progress)
                if job['sample_key'] in first:smoke.append(dict(sample_key=job['sample_key'],phase=job['phase'],budget=job['budget'],condition=job['condition'],F=result['F'],B=result['B'],grid=result['grid']));write(OUT/'validation/smoke.json',dict(passed=True,queries_per_dataset=2,records=smoke))
            from boundary_oracle_report import evaluate
            progress.update(stage='pilot_evaluation');summary=evaluate(rows,cache_index,include_add=False)
            if not summary['hints']:
                progress.update(stage='add_only_diagnostic',total=5700);addition=[]
                for record in rows:
                    key=record['sample']['sample_key'];cache=read(cache_index[key]);public=public_for(record,cache)
                    for condition in CONDITIONS[1:]:addition.append(freeze_job(record,cache,'addition','full',condition,plan(public,'full',condition,record['sample']['span'][0] if condition=='gt_boundary' else None,add=True),cohort_hash,code_hash))
                immutable(OUT/'metadata/addition_jobs.json',addition)
                for job in addition:
                    progress.update(sample_key=job['sample_key'],condition=job['condition']);generate_job(job,lookup[job['sample_key']],read(cache_index[job['sample_key']]),model,processor,config_hash,progress)
                summary=evaluate(rows,cache_index,include_add=True)
            del model;torch.cuda.empty_cache();progress.update(stage='final_validation')
            for source in sources:need(sha(source['path'])==source['sha256'],'Implementation changed during run')
            write(OUT/'validation/final.json',dict(passed=True,queries=300,per_benchmark=100,seeds=[42,43],new_responses=progress.s['completed'],same_spatial_grid=True,budget_equality_within_condition_groups=True,code_hash=code_hash,config_hash=config_hash,cohort_hash=cohort_hash))
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Completed pilot300 temporal sampling v2; seeds 42/43',str(Path(__file__).relative_to(ROOT)),'pilot_300','complete',['pilot_300.cohort'])
            for temporary in (OUT/'frames').glob('*.tmp'):
                temporary.unlink(missing_ok=True)
            progress.update(state='complete',stage='complete',report=str(REPORT/'REPORT.md'))
        except BaseException:
            error=traceback.format_exc();progress.update(state='aborted',stage='aborted',error=error)
            write_result_text(REPORT/'ERROR.md', '# 실행 중단\n\n완료분과 기존 원본은 보존했습니다. 오류:\n\n```text\n'+error+'\n```\n')
            for kind,path in [('predictions',OUT/'responses'),('manifests',OUT/'metadata'),('inputs',OUT/'inputs'),('analyses',REPORT)]:register(RUN+'.'+kind,kind,path,'Aborted pilot300 v2; see status/error',str(Path(__file__).relative_to(ROOT)),'pilot_300','aborted',['pilot_300.cohort'])
            raise
        finally:progress.stop.set();thread.join()

if __name__=='__main__':main()
