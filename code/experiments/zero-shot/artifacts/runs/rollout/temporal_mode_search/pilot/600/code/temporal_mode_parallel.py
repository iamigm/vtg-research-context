"""Three disjoint video-hash GPU workers, two active seeds, one finalizer."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse
import collections
import datetime
import fcntl
import gc
import json
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from temporal_mode_scope import *
from temporal_mode_core import Backend, Search, judge
from temporal_mode_runtime import prepare, wait_gpu


class WorkerProgress:
    def __init__(self, rank, total, active):
        self.rank, self.total = rank, total
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.state = dict(state='running',stage='initializing',rank=rank,completed=0,total=total,
            completed_conditions=0,total_conditions=total*len(conditions(active)),counts={},
            seeds=active['seeds'],eta_seconds=None,model=CONFIG['model'])
        threading.Thread(target=self.heartbeat,daemon=True).start()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.started,
                percentage=100*self.state['completed']/max(1,self.total))
            write(OUT/'metadata'/f'worker{self.rank}.json',self.state)
            print(json.dumps(self.state,ensure_ascii=False),flush=True)

    def heartbeat(self):
        while not self.stop.wait(20):
            self.update()


def ordered(rows):
    early = [next((r for r in rows if r['dataset']==d),None) for d in sorted({r['dataset'] for r in rows})]
    early = [r for r in early if r is not None]
    keys = {r['sample_key'] for r in early}
    return early+[r for r in rows if r['sample_key'] not in keys]


def worker(rank):
    active = scope()
    rows = assigned(read(COHORT),rank,active['worker_count'])
    progress = WorkerProgress(rank,len(rows),active)
    try:
        import fps_spatial_control
        import torch
        import transformers
        from transformers import AutoProcessor,AutoModelForImageTextToText
        torch.set_num_threads(CONFIG['cpu_threads'])
        progress.update(stage='checking_resources')
        wait_gpu(progress,CONFIG['gpu_initial_free_gib'])
        progress.update(state='running',stage='loading_model')
        processor = AutoProcessor.from_pretrained(CONFIG['model'],local_files_only=True,do_resize=False)
        # A distinct processor prevents mutable tokenizer/timestamp state crossing threads.
        input_processor = AutoProcessor.from_pretrained(CONFIG['model'],local_files_only=True,do_resize=False)
        model = AutoModelForImageTextToText.from_pretrained(CONFIG['model'],local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos,list) else [eos]
        write(OUT/'environment'/f'worker{rank}.json',dict(torch=torch.__version__,transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(),cpu_threads=CONFIG['cpu_threads'],prefetch_workers=1,rank=rank,
            seeds=active['seeds'],generation_config_hash=digest(CONFIG)))
        pending=[]
        done=0
        count=0
        counts=collections.Counter()
        expected=conditions(active)
        for row in ordered(rows):
            path=OUT/'responses'/f'{row["sample_key"]}.json'
            if path.exists():
                data=read(path)
                assert data['config_hash']==digest(CONFIG)
                if is_complete(data,active):
                    done+=1
                    count+=len(expected)
                    counts[row['dataset']]+=1
                    continue
            pending.append(row)
        progress.update(completed=done,completed_conditions=count,counts=dict(counts),stage='preparing_input')
        validated=set()
        started=time.monotonic()
        fresh=0
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(prepare,pending[0],input_processor) if pending else None
            for index,row in enumerate(pending):
                key=row['sample_key']
                progress.update(stage='preparing_input',sample_key=key)
                cpu,detail,public,duration=future.result()
                # At most one CPU input ahead, while this query runs on the GPU.
                future=pool.submit(prepare,pending[index+1],input_processor) if index+1<len(pending) else None
                kv_gib=detail['input_tokens']*36*8*128*2*2/2**30
                with (OUT/'metadata/gpu_prefill.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    wait_gpu(progress,max(18.,2.5*kv_gib+12.))
                    progress.update(state='running',stage='prefill',input_tokens=detail['input_tokens'])
                    gpu=cpu.to('cuda')
                    backend=Backend(model,gpu,torch,processor.tokenizer,eos)
                    backend.duration=duration
                baseline=backend.greedy()
                from temporal_mode_core import SOURCE
                original=read(SOURCE/'rollouts'/f'{key}.json')['result']['candidates'][0]['ids']
                assert baseline['ids']==original,(key,'greedy replay changed')
                if row['dataset'] not in validated:
                    write(OUT/'validation'/f'greedy_worker{rank}_{row["dataset"]}.json',
                        dict(passed=True,sample_key=key,input_hashes=detail['hashes'],greedy_ids_equal=True))
                    validated.add(row['dataset'])
                path=OUT/'responses'/f'{key}.json'
                partial=path.with_suffix('.partial.json')
                data=read(partial) if partial.exists() else (read(path) if path.exists() else
                    dict(sample=public,config_hash=digest(CONFIG),clipping_duration=duration,baseline=baseline,
                        baseline_reference=str(SOURCE/'rollouts'/f'{key}.json'),input_hashes=detail['hashes'],
                        prefill_seconds=backend.prefill_seconds,results={},sequences={},distributions=[],choices={}))
                assert data['config_hash']==digest(CONFIG)
                data['active_execution_scope']=dict(seeds=active['seeds'],scope_hash=digest(active))
                backend.sequences.update(data['sequences'])
                backend.choices.update(data['choices'])
                backend.distributions.update({tuple(d['prefix']):d for d in data['distributions']})
                prior_cost=data.get('actual_cost',{})
                search=None
                for seed in active['seeds']:
                    for method in active['methods']:
                        condition=f'{method}:{seed}'
                        if condition in data['results']:
                            count+=1
                            continue
                        progress.update(stage='main_search',method=method,seed=seed,completed_conditions=count)
                        search=Search(backend,public,method,seed,progress.update)
                        main=search.run_main()
                        frozen_main=search.main.summary()
                        before=digest(frozen_main)
                        progress.update(stage='model_judgment',main_modes=len(frozen_main))
                        selection=judge(backend,public,frozen_main)
                        progress.update(stage='tail_search')
                        tail=search.run_tail()
                        assert digest(search.main.summary())==before,'Tail changed main candidates'
                        data['results'][condition]=dict(method=method,seed=seed,main=main,tail=tail,
                            main_modes=frozen_main,tail_modes=search.tail.summary(),selection=selection,
                            draws=search.draws,main_candidate_hash=before,tail_isolation_verified=True)
                        data['sequences']=backend.sequences
                        data['distributions']=list(backend.distributions.values())
                        data['choices']=backend.choices
                        data['actual_cost']=dict(decoder_inputs=prior_cost.get('decoder_inputs',0)+backend.actual_tokens,
                            forwards=prior_cost.get('forwards',0)+backend.actual_forwards,
                            wall_seconds=prior_cost.get('wall_seconds',0)+time.monotonic()-backend.started)
                        write(partial,data)
                        count+=1
                        progress.update(stage='condition_checkpoint',completed_conditions=count)
                assert expected.issubset(data['results'])
                data['actual_cost']=dict(decoder_inputs=prior_cost.get('decoder_inputs',0)+backend.actual_tokens,
                    forwards=prior_cost.get('forwards',0)+backend.actual_forwards,
                    wall_seconds=prior_cost.get('wall_seconds',0)+time.monotonic()-backend.started)
                data['completed_at']=stamp()
                write(path,data)
                partial.unlink(missing_ok=True)
                done+=1
                fresh+=1
                counts[row['dataset']]+=1
                progress.update(stage='query_complete',completed=done,completed_conditions=count,counts=dict(counts),
                    eta_seconds=(time.monotonic()-started)/fresh*(len(pending)-index-1),
                    eta_scope='assigned remaining inference; excludes final reports; early input mix may change rate')
                del backend,gpu,cpu,data,search
                gc.collect()
                torch.cuda.empty_cache()
        progress.update(state='complete',stage='inference_complete',completed=len(rows),
            completed_conditions=len(rows)*len(expected),eta_seconds=0)
    except BaseException:
        progress.update(state='failed',stage='worker_failed',error=traceback.format_exc())
        raise
    finally:
        progress.stop.set()


def verify_snapshot():
    original=read(OUT/'metadata/manifest.json')
    for source in original['files']:
        assert sha(OUT/'code'/Path(source['path']).name)==source['sha256']
    runtime=read(OUT/'metadata/parallel_manifest.json')
    assert runtime['scope_hash']==digest(scope())
    for source in runtime['files']:
        assert sha(source['snapshot'])==source['sha256']


def supervise():
    active=scope()
    verify_snapshot()
    cohort=read(COHORT)
    world=active['worker_count']
    for rank in range(world):
        assert assigned(cohort,rank,world)
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        processes={}
        attempts=collections.Counter()
        def start(rank):
            stream=(OUT/'logs'/f'worker{rank}.log').open('ab',buffering=0)
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(rank)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            stream.close()
            processes[rank]=proc
        for rank in range(world):
            start(rank)
        write(OUT/'metadata/launcher.json',dict(supervisor_pid=os.getpid(),worker_pids=[processes[r].pid for r in range(world)],
            worker_count=world,seeds=active['seeds'],timestamp=stamp(),runtime=str(Path(__file__))))
        origin=datetime.datetime.fromisoformat(read(OUT/'metadata/manifest.json')['started_at']).timestamp()
        while True:
            states=[read(OUT/'metadata'/f'worker{rank}.json') if (OUT/'metadata'/f'worker{rank}.json').exists() else {} for rank in range(world)]
            complete={p.stem for p in (OUT/'responses').glob('*.json') if not p.name.endswith('.partial.json')}
            complete={r['sample_key'] for r in cohort if r['sample_key'] in complete}
            counts=collections.Counter(r['dataset'] for r in cohort if r['sample_key'] in complete)
            eta=[s.get('eta_seconds') for s in states if s.get('eta_seconds') is not None]
            status=dict(state='running',stage='parallel_inference',completed=len(complete),total=600,
                completed_conditions=max(len(complete)*len(conditions(active)),sum(s.get('completed_conditions',0) for s in states)),
                total_conditions=active['total_conditions'],seeds=active['seeds'],worker_count=world,workers=states,
                counts=dict(counts),percentage=len(complete)/6,timestamp=stamp(),elapsed_seconds=time.time()-origin,
                eta_seconds=max(eta) if eta else None,eta_scope='remaining parallel inference, excludes final reports',
                execution_scope_hash=digest(active))
            write(OUT/'metadata/status.json',status)
            write(OUT/'web/completed.json',[r['sample_key'] for r in cohort if r['sample_key'] in complete])
            with (OUT/'logs/progress.log').open('a') as log:
                log.write(json.dumps(status,ensure_ascii=False)+'\n')
            all_done=True
            for rank,proc in list(processes.items()):
                code=proc.poll()
                if code is None:
                    all_done=False
                elif code!=0:
                    if attempts[rank]>=CONFIG['max_retries']:
                        raise RuntimeError(f'Worker {rank} exhausted retries; see logs/worker{rank}.log')
                    attempts[rank]+=1
                    start(rank)
                    all_done=False
            if all_done:
                assert len(complete)==600
                break
            time.sleep(20)
        status.update(stage='postprocessing',timestamp=stamp())
        write(OUT/'metadata/status.json',status)
        folder=Path(__file__).parent
        subprocess.run([sys.executable,str(folder/'temporal_mode_scope_report.py')],check=True)
        subprocess.run([sys.executable,str(folder/'temporal_mode_parallel_check.py'),'--validate'],check=True)
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
        subprocess.run([sys.executable,str(folder/'temporal_mode_view_check.py')],check=True)
        from artifact_store import register
        register('temporal_mode_search.pilot600.responses','rollouts',OUT/'responses',
            'Fixed pilot600; active seeds42/43; three searches; three GPU workers; isolated diagnostic tails; final validation passed.',
            producer='temporal_mode_parallel.py',cohort=str(COHORT),state='complete')
        status.update(state='complete',stage='complete',completed=600,completed_conditions=3600,eta_seconds=0,timestamp=stamp())
        write(OUT/'metadata/status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    if args.worker is None:
        try:
            supervise()
        except BaseException:
            write(OUT/'metadata/status.json',dict(state='failed',stage='parallel_supervisor_failed',total=600,
                seeds=scope()['seeds'],timestamp=stamp(),error=traceback.format_exc()))
            raise
    else:
        worker(args.worker)
