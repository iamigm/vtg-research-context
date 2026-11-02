"""Operational CPU prefetch wrapper; original experiment sources/identity unchanged.

Four independent CPU processors prepare up to fifteen future inputs while the
main thread performs validated native microbatch generation of up to four rows.
"""
import sys,time,json,threading,importlib,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from adaptive_temporal_sampling import read,write,need,objhash,sha,stamp,ROOT

def install(module):
    # Reuse frozen plans on resume only when original sources/config/cache-index are intact.
    frozen=module.OUT/'validation/plans_frozen.json'
    if frozen.exists():
        need(objhash(module.CONFIG)==read(frozen)['config_hash'],'Resume config changed')
        manifest=read(module.OUT/'metadata/source_manifest.json')
        for source in manifest['files']:need(sha(source['path'])==source['sha256'],'Resume source changed')
        cache_index=manifest['frame_cache_index'];need(sha(cache_index['path'])==cache_index['sha256'],'Resume cache index changed')
        known={(j['sample_key'],j['condition']):j['plan_hash'] for j in read(module.OUT/'metadata/jobs.json')}
        original_plan=module.make_plan
        def existing_plan(public,condition,**kwargs):
            key=(public['sample_key'],condition)
            if key not in known:return original_plan(public,condition,**kwargs)
            plan=read(module.OUT/'plans'/condition/(public['sample_key']+'.json'))
            need(objhash(plan)==known[key] and plan['F']==public['F'] and plan['T']==public['T'],'Frozen plan changed')
            return plan
        module.make_plan=existing_plan
    # Keep the immutable job manifest in canonical order. Only dispatch order changes.
    execution_jobs=[]
    original_immutable=module.immutable
    def dispatch_immutable(path,value):
        original_immutable(path,value)
        if Path(path)!=module.OUT/'metadata/jobs.json':return
        completed=[];fresh=[];reusable=[]
        for job in value:
            relative=Path(job['condition'])/(job['sample_key']+'.json')
            if (module.OUT/'responses'/relative).exists():
                completed.append(job);continue
            candidate=False
            for source_run in module.REUSE_RUNS:
                source=ROOT/'artifacts/runs'/source_run/'responses'/relative
                if source.exists() and read(source)['identity']['plan_hash']==job['plan_hash']:
                    candidate=True;break
            (reusable if candidate else fresh).append(job)
        # Stable within each group: no performance- or GT-dependent ordering.
        value[:]=completed+fresh+reusable
        execution_jobs[:]=value
        write(module.OUT/'environment/dispatch_order.json',dict(timestamp=stamp(),
            policy='completed, new inference, parent reuse candidates; stable within group',
            completed=len(completed),new_inference_candidates=len(fresh),
            parent_reuse_candidates=len(reusable),canonical_manifest_unchanged=True,
            execution_jobs_sha256=objhash(value)))
    module.immutable=dispatch_immutable
    original=module.inputs.make_input
    executor=ThreadPoolExecutor(max_workers=4,thread_name_prefix='cpu-input-prefetch')
    pending={};state={};local=threading.local();timing_path=module.OUT/'logs/prefetch.jsonl'
    def emit(**row):
        row.update(timestamp=stamp())
        with timing_path.open('a') as f:f.write(json.dumps(row)+'\n')
    def initialize():
        from transformers import AutoProcessor
        state['processor']=AutoProcessor.from_pretrained(module.old.MODEL,padding_side='left',do_resize=False,local_files_only=True)
        state['jobs']=execution_jobs or read(module.OUT/'metadata/jobs.json');state['lookup']={j['plan_hash']:i for i,j in enumerate(state['jobs'])}
        state['rows']={r['sample']['sample_key']:r for r in module.inputs.records()}
        state['index']=read(module.OUT/'metadata/frame_cache_index.json');state['cache']={};state['cpu_seconds']={}
        for row in read(module.OUT/'validation/annotation_checks.json')['duration_resolutions']:state['rows'][row['sample_key']]['planning_T']=row['planning_T']
    def prepare(job):
        start=time.perf_counter();key=job['sample_key'];condition=job['condition']
        if not hasattr(local,'processor'):
            from transformers import AutoProcessor
            local.processor=AutoProcessor.from_pretrained(module.old.MODEL,padding_side='left',do_resize=False,local_files_only=True);local.cache={}
        if key not in local.cache:
            local.cache.clear();local.cache[key]=read(state['index'][key])
        plan=read(module.OUT/'plans'/condition/(key+'.json'))
        need(objhash(plan)==job['plan_hash'],'Frozen plan mismatch in CPU worker')
        cpu,detail=original(state['rows'][key],local.cache[key],plan,local.processor)
        state['cpu_seconds'][job['plan_hash']]=time.perf_counter()-start
        emit(event='prepared',sample_key=key,condition=condition,cpu_seconds=time.perf_counter()-start,input_hash=objhash(detail['hashes']))
        return cpu,detail
    def fast_source_detail(record,cache,plan):
        key=record['sample']['sample_key'];condition=plan['condition'];ph=objhash(plan)
        for source_run in module.REUSE_RUNS:
            root=ROOT/'artifacts/runs'/source_run;path=root/'responses'/condition/(key+'.json')
            if not path.exists():continue
            response=read(path)
            if response['identity']['plan_hash']!=ph:continue
            detail=read(root/'inputs'/condition/(key+'.json'))
            if detail.get('cache_key')!=cache['cache_key']:continue
            if (detail['F'],detail['B'],detail['grid'])!=(record['F'],record['B'],record['grid']):continue
            h=objhash(detail['hashes'])
            if module.reusable_source(key,condition,h,record) is None:continue
            need(objhash(detail)==response['input_detail_hash'],'Fast reuse metadata checksum mismatch')
            need(detail['plan_sha256']==ph,'Fast reuse plan binding mismatch')
            return detail
        return None
    def wrapped(record,cache,plan,processor):
        key=record['sample']['sample_key'];condition=plan['condition'];dest=module.OUT/'responses'/condition/(key+'.json');detailpath=module.OUT/'inputs'/condition/(key+'.json')
        if dest.exists() and detailpath.exists():
            detail=read(detailpath);response=read(dest)
            need(response['identity']['plan_hash']==objhash(plan),'Completed response plan mismatch')
            need(response['identity']['input_hash']==objhash(detail['hashes']),'Completed input hash mismatch')
            need(response['input_detail_hash']==objhash(detail),'Completed detail corruption')
            return None,detail
        reused_detail=fast_source_detail(record,cache,plan)
        if reused_detail is not None:
            emit(event='reuse_without_preprocessing',sample_key=key,condition=condition)
            return None,reused_detail
        if not state:initialize()
        ph=objhash(plan);position=state['lookup'][ph]
        # Bounded queue: prepare the next sixteen unfinished GPU inputs.
        for job in state['jobs'][position:]:
            if len(pending)>=16:break
            h=job['plan_hash'];path=module.OUT/'responses'/job['condition']/(job['sample_key']+'.json')
            if h not in pending and not path.exists():
                future_key=job['sample_key'];future_plan=read(module.OUT/'plans'/job['condition']/(future_key+'.json'))
                future_cache=read(state['index'][future_key])
                if fast_source_detail(state['rows'][future_key],future_cache,future_plan) is not None:continue
                pending[h]=executor.submit(prepare,job)
        start=time.perf_counter();value=pending.pop(ph).result();emit(event='consumed',sample_key=key,condition=condition,wait_seconds=time.perf_counter()-start)
        return value
    module.inputs.make_input=wrapped
    original_generate=module.old.generate_fresh
    from gt_density_batch import generate as batch_generate
    def generate(cpu,detail,identity,model,processor,genkw,prepare_seconds):
        import copy,torch
        prepare_seconds=state['cpu_seconds'].get(identity['plan_hash'],prepare_seconds)
        items=[(cpu,detail,identity)];future_details=[]
        # Keep a memory margin for other users; singleton path remains available.
        if torch.cuda.mem_get_info()[0]<24*2**30:
            return original_generate(cpu,detail,identity,model,processor,genkw,prepare_seconds)
        for ph,future in list(pending.items()):
            if len(items)>=4:break
            job=state['jobs'][state['lookup'][ph]];path=module.OUT/'responses'/job['condition']/(job['sample_key']+'.json')
            if path.exists():continue
            fc,fd=future.result()
            if module.reusable_source(job['sample_key'],job['condition'],objhash(fd['hashes']),state['rows'][job['sample_key']]) is not None:continue
            fi=dict(identity,sample_key=job['sample_key'],condition=job['condition'],plan_hash=ph,input_hash=objhash(fd['hashes']))
            items.append((fc,fd,fi));future_details.append((ph,path,fd))
        if len(items)==1:return original_generate(cpu,detail,identity,model,processor,genkw,prepare_seconds)
        try:results=batch_generate(items,model,processor,genkw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache();emit(event='batch_oom_fallback',batch_size=len(items))
            return original_generate(cpu,detail,identity,model,processor,genkw,prepare_seconds)
        for result in results:
            result['model_revision']=module.CONFIG['revision']
            result['input_prepare_seconds']=state['cpu_seconds'].get(result['identity']['plan_hash'],0.)
        for result,(ph,path,fd) in zip(results[1:],future_details):
            write(module.OUT/'inputs'/result['identity']['condition']/(result['identity']['sample_key']+'.json'),fd)
            write(path,result);pending.pop(ph)
        results[0]['input_prepare_seconds']=prepare_seconds
        emit(event='batch_generated',batch_size=len(items),wall_seconds=results[0]['batch_wall_seconds'],peak_bytes=results[0]['peak_allocated_bytes'])
        return results[0]
    gate=ROOT/'artifacts/runs/gt_extent_boundary_density_pilot300_v1/validation'
    need(read(gate/'batch4_bridge.json')['passed'] and read(gate/'batch4_other_benchmarks_bridge.json')['passed'],'Batch token bridge failed')
    module.old.generate_fresh=generate
    return executor

def main():
    module=importlib.import_module(sys.argv[1]);(module.OUT/'logs').mkdir(parents=True,exist_ok=True)
    write(module.OUT/'environment/prefetch_operation.json',dict(wrapper=str(Path(__file__).resolve()),sha256=sha(__file__),timestamp=stamp(),cpu_workers=4,prefetch_depth=15,batch_size=4,batch_helper_sha256=sha(ROOT/"_lib/gt_density_batch.py"),experiment_sources_unchanged=True,dispatch_policy='new inference before parent reuse'))
    executor=install(module)
    try:module.main()
    finally:executor.shutdown(wait=True,cancel_futures=True)
if __name__=='__main__':main()
