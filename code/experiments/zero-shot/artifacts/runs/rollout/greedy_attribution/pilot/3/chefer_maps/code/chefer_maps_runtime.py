"""Detached, resource-gated Chefer decoder-map measurement and publication."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, fcntl, gc, gzip, json, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path
from chefer_maps_core import *

class Progress:
    def __init__(self):
        self.start=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event()
        self.state=dict(state='queued',stage='waiting_resources',completed=0,total=3,scope='pilot',
            eta_seconds=None,eta_scope='unknown while queued; excludes queue wait when measured')
        threading.Thread(target=self.heartbeat,daemon=True).start()
    def __call__(self,**kw):
        with self.lock:
            self.state.update(kw,timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start)
            self.state['percentage']=100*self.state['completed']/3
            write(OUT/'metadata/status.json',self.state)
            write(ROOT/'artifacts/common/registry/chefer_maps.live.json',dict(canonical=str(OUT),
                log=str(OUT/'logs/run.log'),**self.state))
            print(json.dumps(self.state,ensure_ascii=False,allow_nan=False),flush=True)
    def heartbeat(self):
        while not self.stop.wait(20):self()

def resources():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids=[int(v.strip()) for v in result.stdout.splitlines() if v.strip().isdigit() and int(v.strip())!=os.getpid()]
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free=min(int(v.strip()) for v in result.stdout.splitlines())
    stat={a:int(b) for a,b in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    limit_text=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    meminfo={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    pressure=stat.get('anon',0)+stat.get('kernel',0)+stat.get('file_dirty',0)+stat.get('file_writeback',0)
    headroom=min(meminfo['MemAvailable'],int(limit_text)-pressure) if limit_text!='max' else meminfo['MemAvailable']
    disk=shutil.disk_usage(OUT).free
    return dict(other_gpu_pids=pids,gpu_free_mib=free,host_nonreclaimable_bytes=pressure,
        host_headroom_gib=headroom/2**30,disk_free_gib=disk/2**30)

def wait_resources(progress):
    idle=0
    while True:
        r=resources();ok=not r['other_gpu_pids'] and r['gpu_free_mib']>=CONFIG['min_gpu_free_mib'] and r['host_headroom_gib']>=CONFIG['min_host_headroom_gib'] and r['disk_free_gib']>=CONFIG['min_disk_free_gib']
        idle=idle+1 if ok else 0
        progress(state='queued',stage='waiting_resources',resources=r,idle_checks=idle)
        if idle>=2:return
        time.sleep(20)

def publish(browser=False):
    from chefer_maps_report import generate
    generate()
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
    if browser:
        from chefer_maps_report import check_viewer
        check_viewer()

def setup():
    selection=setup_selection()
    assert read(OUT/'metadata/config.json')==CONFIG
    for folder in ('logs','queries','maps','arrays','validation','code'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    files=[]
    sources=list((ROOT/'_lib').glob('chefer_*.py'))
    sources += [ROOT/'_lib'/name for name in ('greedy_attribution_runtime.py','greedy_attribution_core.py',
        'greedy_attribution_cache.py','probability_budget_runtime.py','probability_budget_inputs.py')]
    for source in sorted(sources):
        target=OUT/'code'/source.name;shutil.copyfile(source,target)
        files.append(dict(path=str(source),snapshot=str(target),sha256=sha(source)))
    write(OUT/'metadata/provenance.json',dict(timestamp=stamp(),files=files,selection_sha256=sha(OUT/'metadata/selection.json'),
        config_hash=digest(CONFIG),official_reference=read(REFERENCE/'manifest.json'),
        model_config_sha256=sha(MODEL/'config.json'),model_registry_sha256=sha('/home/irteam/backbones/manifests/registry.json')))
    return selection

def save_result(sample,result,seconds):
    import numpy as np
    assert result.get('validation',{}).get('passed') is True, 'Scientific validation did not pass; maps cannot be published'
    key=sample['sample_key'];attribution=read(sample['source']['attribution'])
    positions=np.asarray(attribution['context']['visual_positions'],dtype=np.int64)
    t,h,w=sample['display']['grid'];shape=[t,h//2,w//2]
    vectors=result['maps'];names={'attention_rollout':'attention_rollout','final_c':'chefer_grad_relevance'}
    output=dict(shape=shape,maps_are='exact predictor row restricted to original decoder visual token positions',
        temporal_mapping='each spatial map belongs to two adjacent model input frames',layers=36)
    arrays={}
    for name,source in names.items():
        value=vectors[source]
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        value=np.asarray(value,dtype=np.float64).reshape(-1)
        assert np.isfinite(value).all() and (value>=-1e-9).all(),(key,name,'nonfinite or negative map')
        assert len(value)==sample['display']['prompt_length']+sample['branch']['step'],(key,name,'wrong context length')
        arrays[name]=value
        output[name]=value[positions].reshape(shape).tolist()
    array_path=OUT/'arrays'/f'{key}.npz';np.savez_compressed(array_path,**arrays,visual_positions=positions)
    map_path=OUT/'maps'/f'{key}.json.gz';temp=map_path.with_suffix('.tmp')
    temp.write_bytes(gzip.compress(json.dumps(output,ensure_ascii=False,allow_nan=False).encode(),mtime=0));temp.replace(map_path)
    metadata={k:v for k,v in result.items() if k!='maps'}
    write(OUT/'validation'/f'{key}.json',metadata)
    write(OUT/'queries'/f'{key}.json',dict(sample_key=key,complete=True,
        target=dict(step=sample['branch']['step'],token_id=sample['branch']['greedy_token'],text=sample['branch']['greedy_token_text'],objective='logit'),
        maps_file=f'maps/{key}.json.gz',maps_sha256=sha(map_path),array_file=str(array_path),array_sha256=sha(array_path),
        input_hashes=sample['input_hashes'],config_hash=digest(CONFIG),validation=metadata['validation'],
        measurement={k:v for k,v in metadata.items() if k!='validation'},
        map_stats={name:dict(total=float(value.sum()),visual_total=float(value[positions].sum()),maximum=float(value.max())) for name,value in arrays.items()},
        cost=dict(wall_seconds=seconds),timestamp=stamp()))

def validate():
    import numpy as np
    selection=read(OUT/'metadata/selection.json');count=0
    for sample in selection['samples']:
        for name,source in sample['source'].items():assert sha(source)==sample['source_hashes'][name]
        path=OUT/'queries'/f"{sample['sample_key']}.json"
        if not path.exists():continue
        query=read(path);assert query['complete'] and query['input_hashes']==sample['input_hashes'] and query['config_hash']==digest(CONFIG)
        assert query['validation']['passed'] is True
        assert sha(OUT/query['maps_file'])==query['maps_sha256'] and sha(query['array_file'])==query['array_sha256']
        maps=json.loads(gzip.decompress((OUT/query['maps_file']).read_bytes()))
        with np.load(query['array_file']) as arrays:
            for name in ('attention_rollout','final_c'):
                value=np.asarray(maps[name]);assert list(value.shape)==maps['shape'] and np.isfinite(value).all() and (value>=0).all()
                assert np.array_equal(value.ravel(),arrays[name][arrays['visual_positions']])
            assert abs(float(arrays['attention_rollout'].sum())-1)<1e-4
        count+=1
    result=dict(passed=True,completed=count,total=3,gpu_measurement_complete=count==3,timestamp=stamp())
    write(OUT/'validation/integrity.json',result);return result

def selfcheck():
    from chefer_lrp_engine import selftest
    result=selftest();assert result['passed']
    write(OUT/'validation/helpers.json',dict(**result,timestamp=stamp(),
        engine_sha256=sha(ROOT/'_lib/chefer_lrp_engine.py'),rules_sha256=sha(ROOT/'_lib/chefer_lrp_rules.py')))
    return result

def worker():
    progress=Progress()
    try:
        with (OUT/'metadata/run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            wait_resources(progress)
            provenance=read(OUT/'metadata/provenance.json')
            assert sha(OUT/'metadata/selection.json')==provenance['selection_sha256']
            assert read(OUT/'metadata/config.json')==CONFIG
            for source in provenance['official_reference']['files']:
                assert sha(source['path'])==source['sha256'],('official reference changed',source['path'])
            for source in provenance['files']:
                assert sha(source['path'])==source['sha256'],('source changed while queued',source['path'])
            import torch
            from chefer_lrp_engine import analyze
            from greedy_attribution_runtime import load
            from probability_budget_runtime import prepare
            torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
            progress(state='running',stage='loading_model')
            model,processor,seconds=load()
            write(OUT/'metadata/model_load.json',dict(seconds=seconds,timestamp=stamp()))
            selection=read(OUT/'metadata/selection.json');completed=0
            # Shortest original input first supplies the mandatory first real GPU validation.
            for sample in sorted(selection['samples'],key=lambda s:s['display']['prompt_length']):
                key=sample['sample_key'];path=OUT/'queries'/f'{key}.json'
                if path.exists() and read(path).get('complete'):completed+=1;continue
                start=time.monotonic();progress(stage='preparing_input',sample_key=key,completed=completed)
                for name,source in sample['source'].items():assert sha(source)==sample['source_hashes'][name],(key,name,'source changed')
                cpu,info=prepare(sample['row'],processor);assert info['hashes']==sample['input_hashes']
                workdir=OUT/'temporary'/key
                if workdir.exists():
                    progress(stage='discarding_incomplete_capture',temporary_path=str(workdir))
                    shutil.rmtree(workdir)
                workdir.mkdir(parents=True,exist_ok=True)
                result=analyze(model,processor,cpu,sample,workdir,progress)
                save_result(sample,result,time.monotonic()-start)
                validate();shutil.rmtree(workdir);del result,cpu;gc.collect();torch.cuda.empty_cache()
                completed+=1;progress(stage='query_complete',completed=completed)
                publish()
            del model,processor;gc.collect();torch.cuda.empty_cache()
            progress(stage='postprocessing');assert validate()['completed']==3
            publish(browser=True)
            from artifact_store import register
            register('greedy_attribution.chefer_maps.pilot3','analyses',OUT/'queries',
                'Native attention rollout and Chefer LRP-gradient final C on three source greedy branch tokens',
                'chefer_maps_runtime.py',str(OUT/'metadata/selection.json'),'complete')
            if (OUT/'temporary').exists(): (OUT/'temporary').rmdir()
            progress(state='complete',stage='complete',completed=3);publish()
    except BaseException:
        progress(state='failed',stage='failed',error=traceback.format_exc())
        try:publish()
        except Exception:traceback.print_exc()
        raise
    finally:progress.stop.set()

def launch():
    launcher=OUT/'metadata/launcher.json'
    if launcher.exists():
        pid=read(launcher)['pid']
        try:
            if b'chefer_maps_runtime.py' in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError(f'Already active: {pid}')
        except FileNotFoundError:pass
    setup();selfcheck();write(OUT/'metadata/status.json',dict(state='queued',stage='waiting_resources',completed=0,total=3,timestamp=stamp()))
    publish()
    with (OUT/'logs/run.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker'],cwd=ROOT,
            env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(launcher,dict(pid=child.pid,detached=True,log=str(OUT/'logs/run.log'),timestamp=stamp()))
    print(json.dumps(read(launcher),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','worker','publish','validate','selfcheck'])
    args=parser.parse_args();globals()[args.action]()
