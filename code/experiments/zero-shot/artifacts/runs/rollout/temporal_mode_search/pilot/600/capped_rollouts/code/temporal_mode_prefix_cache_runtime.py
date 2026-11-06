"""Same fixed rollout policy with bounded exact native-prefix KV reuse."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ['CUDA_MPS_PIPE_DIRECTORY']='/tmp/irteam_temporal_mode_mps'
os.environ['CUDA_MPS_LOG_DIRECTORY']='/home1/irteam/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/logs/mps'
import argparse
import collections
import datetime
import fcntl
import gc
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from temporal_mode_capped_core import *
from temporal_mode_prefix_cache import PrefixBackend as Backend, PrefixSearch as Search, combine, SCHEMA, BACKEND


class Progress:
    def __init__(self,rank,total):
        self.rank=rank
        self.started=time.monotonic()
        self.stop=threading.Event()
        self.lock=threading.RLock()
        self.state=dict(state='running',stage='initializing',rank=rank,completed=0,total=total,
            completed_conditions=0,total_conditions=total*6,completed_budget_selections=0,
            total_budget_selections=total*24,seeds=CONFIG['seeds'],counts={},eta_seconds=None)
        threading.Thread(target=self.heartbeat,daemon=True).start()

    def update(self,**values):
        with self.lock:
            self.state.update(values)
            self.state.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.started,
                percentage=100*self.state['completed']/max(1,self.state['total']))
            write(OUT/'metadata'/f'worker{self.rank}.json',self.state)
            print(__import__('json').dumps(self.state,ensure_ascii=False),flush=True)

    def heartbeat(self):
        while not self.stop.wait(20):
            self.update()


def setup():
    rows=read(COHORT)
    assert len(rows)==600
    assert read(ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_300/samples.json')==rows[:300]
    for name in ('metadata','responses','inputs','logs','validation','code','metrics','web','environment'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    RESULT.mkdir(parents=True,exist_ok=True)
    path=OUT/'metadata/config.json'
    if path.exists():
        assert read(path)==CONFIG
    else:
        write(path,CONFIG)
    snapshot=(OUT/'code').resolve()
    if Path(__file__).resolve().parent==snapshot:
        manifest=read(OUT/'metadata/manifest.json')
        for entry in manifest['files']:
            assert sha(entry['snapshot'])==entry['sha256']
        for entry in manifest['base_files']:
            assert sha(entry['snapshot'])==entry['sha256']
        assert manifest['cohort_sha256']==sha(COHORT)
        cost_manifest=read(OUT/'metadata/cost_manifest.json')
        assert cost_manifest['generation_manifest_sha256']==sha(OUT/'metadata/manifest.json')
        for entry in cost_manifest['files']:
            assert sha(entry['snapshot'])==entry['sha256']
        assert read(OUT/'validation/cost_gpu_smoke.json')['passed']
        prefix_manifest=read(OUT/'metadata/prefix_cache_manifest.json')
        for entry in prefix_manifest['files']:
            assert sha(entry['snapshot'])==entry['sha256']
        assert read(OUT/'validation/prefix_cache_gpu.json')['passed']
        assert read(OUT/'validation/mps_real_inputs.json')['passed']
        assert read(OUT/'validation/mps_benchmark.json')['passed']
        smoke=read(OUT/'validation/gpu_smoke.json')
        assert smoke['passed'] and smoke['config_hash']==digest(CONFIG)
        return rows
    files=[]
    for source in Path(__file__).parent.glob('temporal_mode_capped*'):
        if not source.is_file():
            continue
        destination=OUT/'code'/source.name
        if destination.exists():
            assert sha(destination)==sha(source),'Frozen capped source changed'
        else:
            shutil.copyfile(source,destination)
        files.append(dict(source=str(source),snapshot=str(destination),sha256=sha(source)))
    old=read(LEGACY/'metadata/manifest.json')
    base_files=[dict(snapshot=str(LEGACY/'code'/Path(entry['path']).name),sha256=entry['sha256']) for entry in old['files']]
    for entry in base_files:
        assert sha(entry['snapshot'])==entry['sha256']
    manifest=dict(config_hash=digest(CONFIG),cohort=str(COHORT),cohort_sha256=sha(COHORT),
        queries=600,unique_videos=len({r['video_path'] for r in rows}),files=files,base_files=base_files,
        base_run=str(LEGACY),base_manifest_sha256=sha(LEGACY/'metadata/manifest.json'),
        backbone_manifest_sha256=sha('/home/irteam/backbones/manifests/registry.json'),
        started_at=stamp(),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip())
    if (OUT/'metadata/manifest.json').exists():
        assert read(OUT/'metadata/manifest.json')['files']==files
    else:
        write(OUT/'metadata/manifest.json',manifest)
    assert read(LEGACY/'validation/cpu_preflight.json')['passed']
    smoke=read(OUT/'validation/gpu_smoke.json')
    assert smoke['passed'] and smoke['config_hash']==digest(CONFIG)
    write(OUT/'validation/source_preflight.json',dict(passed=True,queries=600,
        reference=str(LEGACY/'validation/cpu_preflight.json'),sha256=sha(LEGACY/'validation/cpu_preflight.json'),
        input_policy_unchanged=True,timestamp=stamp()))
    from artifact_store import register
    register('temporal_mode_search.capped_rollouts.pilot600.responses','rollouts',OUT/'responses',
        'Fixed600; powers-of-two rollout attempts; bounded initialization; two seeds; three workers; running.',
        producer='temporal_mode_prefix_cache_runtime.py',cohort=str(COHORT),state='building')
    return rows


def prepare(row,processor):
    import fps_spatial_control as module
    from result_layout import relocate_path
    prior=read(SOURCE/'inputs'/f'{row["sample_key"]}.json')
    cache_path=relocate_path(Path(prior['cache_metadata']))
    cache=read(cache_path)
    public={key:row[key] for key in ('sample_key','dataset','video_path','query','duration')}
    cpu,detail=module.prepare_input(public,cache,2,'video_query',processor)
    assert detail['hashes']==prior['hashes']
    assert detail['hw']==prior['hw'] and detail['plan']['frame_ids']==prior['plan']['frame_ids']
    write(OUT/'inputs'/f'{row["sample_key"]}.json',dict(source=str(SOURCE/'inputs'/f'{row["sample_key"]}.json'),
        hashes=detail['hashes'],frame_ids=detail['plan']['frame_ids'],timestamps=detail['plan']['source_pts'],
        hw=detail['hw'],input_tokens=detail['input_tokens'],visual_tokens=detail['visual_tokens'],
        clipping_duration=cache['duration'],uses_gt=False))
    return cpu,detail,public,cache['duration']


def ordered(rows):
    early=[next(r for r in rows if r['dataset']==dataset) for dataset in sorted({r['dataset'] for r in rows})]
    keys={r['sample_key'] for r in early}
    return early+[r for r in rows if r['sample_key'] not in keys]


def worker(rank):
    rows=assigned(read(COHORT),rank)
    progress=Progress(rank,len(rows))
    try:
        import fps_spatial_control
        import torch
        import transformers
        from temporal_mode_runtime import wait_gpu
        from transformers import AutoProcessor,AutoModelForImageTextToText
        torch.set_num_threads(CONFIG['cpu_threads'])
        wait_gpu(progress,CONFIG['gpu_initial_free_gib'])
        progress.update(state='running',stage='loading_model')
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
        prefetch_processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
        model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
        eos=model.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        write(OUT/'environment'/f'worker{rank}.json',dict(torch=torch.__version__,transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(),threads=CONFIG['cpu_threads'],prefetch_workers=1,rank=rank,execution_service='cuda_mps',mps_pipe=os.environ['CUDA_MPS_PIPE_DIRECTORY']))
        pending=[]
        done=0
        completed_routes=0
        completed_selections=0
        counts=collections.Counter()
        for row in ordered(rows):
            path=OUT/'responses'/f'{row["sample_key"]}.json'
            if path.exists() and is_complete(read(path)):
                done+=1
                completed_routes+=6
                completed_selections+=24
                counts[row['dataset']]+=1
            else:
                pending.append(row)
        progress.update(completed=done,completed_conditions=completed_routes,
            completed_budget_selections=completed_selections,counts=dict(counts))
        started=time.monotonic()
        fresh=0
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(prepare,pending[0],prefetch_processor) if pending else None
            for index,row in enumerate(pending):
                key=row['sample_key']
                progress.update(stage='preparing_input',sample_key=key)
                cpu,detail,public,duration=future.result()
                future=pool.submit(prepare,pending[index+1],prefetch_processor) if index+1<len(pending) else None
                with (OUT/'metadata/gpu_prefill.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    required=max(18.,2.5*detail['input_tokens']*36*8*128*2*2/2**30+12.)
                    wait_gpu(progress,required)
                    progress.update(state='running',stage='prefill',input_tokens=detail['input_tokens'])
                    gpu=cpu.to('cuda')
                    backend=Backend(model,gpu,torch,processor.tokenizer,eos)
                    backend.duration=duration
                baseline=backend.greedy()
                original=read(SOURCE/'rollouts'/f'{key}.json')['result']['candidates'][0]['ids']
                assert baseline['ids']==original,(key,'greedy replay changed')
                write(OUT/'validation'/f'greedy_worker{rank}.json',dict(passed=True,sample_key=key,exact_ids=True))
                path=OUT/'responses'/f'{key}.json'
                partial=path.with_suffix('.partial.json')
                data=read(partial) if partial.exists() else dict(sample=public,config_hash=digest(CONFIG),
                    clipping_duration=duration,baseline=baseline,input_hashes=detail['hashes'],
                    results={},sequences={},distributions=[],choices={},prefill_seconds=backend.prefill_seconds)
                assert data['config_hash']==digest(CONFIG)
                backend.sequences.update(data['sequences'])
                backend.choices.update(data['choices'])
                backend.distributions.update({tuple(value['prefix']):value for value in data['distributions']})
                prior_cost=data.get('actual_cost',{})
                prior_measured=data.get('measured_actual_cost',{})
                data['cost_accounting']=dict(schema=SCHEMA,execution_backend=BACKEND,introduced_at=read(OUT/'metadata/cost_manifest.json')['introduced_at'],
                    caches_restored=partial.exists(),generation_config_unchanged=True)
                search=None
                for seed in CONFIG['seeds']:
                    for method in METHODS:
                        condition=f'{method}:{seed}'
                        existing=data['results'].get(condition,{})
                        if set(existing.get('checkpoints',{}))==set(map(str,CHECKPOINTS)):
                            completed_routes+=1
                            completed_selections+=len(CHECKPOINTS)
                            continue
                        progress.update(stage='main_search',method=method,seed=seed)
                        search=Search(backend,public,method,seed,progress.update)
                        # Incomplete sweeps replay deterministically; their checkpoints replace only this condition.
                        selection_base=completed_selections
                        def save(result):
                            data['results'][condition]=result
                            data['sequences']=backend.sequences
                            data['distributions']=list(backend.distributions.values())
                            data['choices']=backend.choices
                            data['actual_cost']=dict(decoder_inputs=prior_cost.get('decoder_inputs',0)+backend.actual_tokens,
                                forwards=prior_cost.get('forwards',0)+backend.actual_forwards,
                                wall_seconds=prior_cost.get('wall_seconds',0)+time.monotonic()-backend.started)
                            data['measured_actual_cost']=combine([prior_measured,combine(backend.cost_meter.records)])
                            write(partial,data)
                            progress.update(completed_budget_selections=selection_base+len(result['checkpoints']))
                        data['results'][condition]=search.sweep(save=save)
                        completed_routes+=1
                        completed_selections+=len(CHECKPOINTS)
                        progress.update(stage='condition_checkpoint',completed_conditions=completed_routes,
                                        completed_budget_selections=completed_selections)
                assert is_complete(data)
                data['actual_cost']=dict(decoder_inputs=prior_cost.get('decoder_inputs',0)+backend.actual_tokens,
                    forwards=prior_cost.get('forwards',0)+backend.actual_forwards,
                    wall_seconds=prior_cost.get('wall_seconds',0)+time.monotonic()-backend.started)
                data['measured_actual_cost']=combine([prior_measured,combine(backend.cost_meter.records)])
                data['completed_at']=stamp()
                write(path,data)
                partial.unlink(missing_ok=True)
                done+=1
                fresh+=1
                counts[row['dataset']]+=1
                progress.update(stage='query_complete',completed=done,completed_conditions=completed_routes,
                    completed_budget_selections=completed_selections,counts=dict(counts),
                    eta_seconds=(time.monotonic()-started)/fresh*(len(pending)-index-1))
                del backend,gpu,cpu,data,search
                gc.collect()
                torch.cuda.empty_cache()
        progress.update(state='complete',stage='inference_complete',completed=len(rows),
            completed_conditions=len(rows)*6,completed_budget_selections=len(rows)*24,eta_seconds=0)
    except BaseException:
        progress.update(state='failed',stage='worker_failed',error=traceback.format_exc())
        raise
    finally:
        progress.stop.set()


def supervise():
    cohort=setup()
    Path(os.environ['CUDA_MPS_PIPE_DIRECTORY']).mkdir(mode=0o700,exist_ok=True)
    Path(os.environ['CUDA_MPS_LOG_DIRECTORY']).mkdir(exist_ok=True)
    probe=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True)
    if probe.returncode!=0:
        subprocess.run(['nvidia-cuda-mps-control','-d'],check=True)
    write(OUT/'environment/mps.json',dict(enabled=True,private_pipe=os.environ['CUDA_MPS_PIPE_DIRECTORY'],worker_count=3,timestamp=stamp()))
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        processes={}
        retries=collections.Counter()
        def start(rank):
            stream=(OUT/'logs'/f'worker{rank}.log').open('ab',buffering=0)
            processes[rank]=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(rank)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            stream.close()
        for rank in range(CONFIG['worker_count']):
            start(rank)
        write(OUT/'metadata/launcher.json',dict(supervisor_pid=os.getpid(),worker_pids=[p.pid for p in processes.values()],
            worker_count=3,seeds=CONFIG['seeds'],timestamp=stamp()))
        origin=time.monotonic()
        last_report=-1
        while True:
            states=[read(OUT/'metadata'/f'worker{rank}.json') if (OUT/'metadata'/f'worker{rank}.json').exists() else {} for rank in processes]
            names={p.stem for p in (OUT/'responses').glob('*.json') if not p.name.endswith('.partial.json')}
            complete={r['sample_key'] for r in cohort if r['sample_key'] in names}
            status=dict(state='running',stage='capped_parallel_inference',cost_accounting=SCHEMA,execution_backend=BACKEND,execution_service='cuda_mps',completed=len(complete),total=600,
                completed_conditions=sum(s.get('completed_conditions',0) for s in states),total_conditions=3600,
                completed_budget_selections=sum(s.get('completed_budget_selections',0) for s in states),total_budget_selections=14400,
                percentage=len(complete)/6,workers=states,seeds=CONFIG['seeds'],worker_count=3,
                counts=dict(collections.Counter(r['dataset'] for r in cohort if r['sample_key'] in complete)),
                timestamp=stamp(),elapsed_seconds=time.monotonic()-origin)
            write(OUT/'metadata/status.json',status)
            write(OUT/'web/completed.json',[r['sample_key'] for r in cohort if r['sample_key'] in complete])
            with (OUT/'logs/progress.log').open('a') as log:
                log.write(__import__('json').dumps(status,ensure_ascii=False)+'\n')
            all_done=True
            for rank,process in list(processes.items()):
                code=process.poll()
                if code is None:
                    all_done=False
                elif code!=0:
                    if retries[rank]>=CONFIG['max_retries']:
                        raise RuntimeError(f'Worker {rank} exhausted retries')
                    retries[rank]+=1
                    start(rank)
                    all_done=False
            if len(complete)!=last_report and (len(complete)<=3 or len(complete)%10==0):
                subprocess.run([sys.executable,str(Path(__file__).with_name('temporal_mode_prefix_cache_report.py'))],check=True)
                last_report=len(complete)
            if all_done:
                assert len(complete)==600
                break
            time.sleep(20)
        status.update(stage='postprocessing')
        write(OUT/'metadata/status.json',status)
        for script,args in [('temporal_mode_prefix_cache_report.py',[]),('temporal_mode_capped_check.py',['--validate']),('temporal_mode_capped_cost_check.py',['--validate']),('temporal_mode_prefix_cache_check.py',['--validate'])]:
            subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*args],check=True)
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('temporal_mode_capped_view_check.py'))],check=True)
        from artifact_store import register
        register('temporal_mode_search.capped_rollouts.pilot600.responses','rollouts',OUT/'responses',
            'Fixed600; capped powers-of-two rollouts; bounded initialization; two seeds; three GPU workers; validated.',
            producer='temporal_mode_prefix_cache_runtime.py',cohort=str(COHORT),state='complete')
        status.update(state='complete',stage='complete',completed=600,completed_conditions=3600,
            completed_budget_selections=14400,timestamp=stamp())
        write(OUT/'metadata/status.json',status)
        subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,check=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int)
    parser.add_argument('--setup',action='store_true')
    args=parser.parse_args()
    if args.setup:
        setup()
    elif args.worker is not None:
        worker(args.worker)
    else:
        try:
            supervise()
        except BaseException:
            write(OUT/'metadata/status.json',dict(state='failed',stage='supervisor_failed',timestamp=stamp(),error=traceback.format_exc()))
            raise
