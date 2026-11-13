"""Detached visual-only generalization and matched precision/control experiment."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,fcntl,gc,json,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
from oracle_switch_controls_core import *


class Progress:
    def __init__(self):
        self.start=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event()
        self.state=dict(state='queued',stage='waiting_gpu',completed=0,total=60,failed_queries=0,
            scope='pilot',eta_seconds=None,eta_scope='unknown until completed queries; includes scans and controls')
        threading.Thread(target=self.heartbeat,daemon=True).start()
    def __call__(self,**fields):
        with self.lock:
            self.state.update(fields,timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start)
            self.state['percentage']=100*self.state['completed']/60
            write(OUT/'metadata/status.json',self.state)
            write(ROOT/'artifacts/common/registry/oracle_switch_controls.live.json',
                dict(canonical=str(OUT),log=str(OUT/'logs/run.log'),**self.state))
            line=json.dumps(self.state,ensure_ascii=False)
            with (OUT/'logs/progress.log').open('a') as stream:stream.write(line+'\n')
            print(line,flush=True)
    def heartbeat(self):
        while not self.stop.wait(20):self()


def bind_native():
    import oracle_switch_runtime as native
    native.OUT=OUT;native.RESULTS=RESULTS;native.CONFIG=CONFIG
    return native


def publish():
    from oracle_switch_controls_report import generate
    generate()
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)


def setup():
    from oracle_switch_controls_select import build_selection
    selection_path=OUT/'metadata/selection.json'
    selection=read(selection_path) if selection_path.exists() else build_selection()
    assert len(selection['samples'])==60 and len({s['sample_key'] for s in selection['samples']})==60
    path=OUT/'metadata/config.json'
    if path.exists():assert read(path)==CONFIG
    else:write(path,CONFIG)
    for name in ('queries','arrays','media','analyses/controls','logs','validation','code'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    sources=list((ROOT/'_lib').glob('oracle_switch*.py'))
    sources += [ROOT/'_lib'/name for name in ('greedy_attribution_runtime.py','greedy_attribution_core.py',
        'greedy_attribution_cache.py','probability_budget_runtime.py','probability_budget_inputs.py')]
    files=[]
    for source in sorted(sources):
        target=OUT/'code'/source.name;shutil.copyfile(source,target)
        files.append(dict(path=str(source),snapshot=str(target),sha256=sha(source)))
    write(OUT/'metadata/provenance.json',dict(timestamp=stamp(),files=files,config_hash=digest(CONFIG),
        selection_sha256=sha(selection_path),model_manifest_sha256=sha('/home/irteam/backbones/manifests/registry.json'),
        processor_sha256=sha(MODEL/'video_preprocessor_config.json'),
        cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip()))
    return selection


def save_repeat_direction(sample,pixels,gradient,grid,frame_ids,times,mean,std,bundle):
    import numpy as np
    from oracle_switch_visual import make_directions
    repeated=make_directions(pixels,gradient,grid,frame_ids,times,mean,std)
    path=OUT/'arrays'/f"{sample['sample_key']}_repeat.npz"
    np.savez_compressed(path,direction_sign=repeated['sign'].cpu().numpy())
    return dict(repeat_sign_file=str(path),repeat_sign_sha256=sha(path),
        physical_sign_agreement=float((bundle['sign']==repeated['sign']).float().mean()))


def random_direction(reference,grid,frame_ids,seed,key):
    import hashlib,numpy as np,torch
    from oracle_switch_visual import unpack_pixels,pack_pixels
    reference_frames=unpack_pixels(reference,grid)
    actual_seed=int.from_bytes(hashlib.sha256(f'oracle_switch_controls|{seed}|{key}'.encode()).digest()[:8],'big')
    rng=np.random.Generator(np.random.PCG64(actual_seed))
    values=rng.integers(0,2,size=tuple(reference_frames.shape),dtype=np.int8)*2-1
    seen={}
    for slot,frame_id in enumerate(frame_ids):
        if frame_id in seen:values[slot]=values[seen[frame_id]]
        else:seen[frame_id]=slot
    frames=torch.from_numpy(values).to(reference.device)
    frames[reference_frames==0]=0
    return pack_pixels(frames,grid),actual_seed


def rgb_stats(original,changed,grid,support,mean,std):
    import torch
    from oracle_switch_visual import unpack_pixels
    x=unpack_pixels(original,grid).float();y=unpack_pixels(changed,grid).float()
    scale=torch.tensor(std,device=x.device).view(1,3,1,1)
    delta=((y-x)*scale)[support['all_physical_slots']]
    return dict(actual_rgb_linf=float(delta.abs().max()),
        actual_rgb_rms_full=float(delta.double().square().mean().sqrt()),
        changed_fraction=float((delta!=0).float().mean()),
        statistic_denominator='unique physical source frame RGB components, including unchanged components')


def rounded_pixels(original,changed,grid,mean,std,processor,sample,cache):
    """Use the original preprocessing path after an actual uint8 round trip."""
    import numpy as np,torch
    from probability_budget_inputs import processor_input
    from oracle_switch_visual import unpack_pixels
    scale=torch.tensor(std,device=original.device).view(1,3,1,1)
    shift=torch.tensor(mean,device=original.device).view(1,3,1,1)
    raw_original=unpack_pixels(original,grid).float()*scale+shift
    raw_changed=unpack_pixels(changed,grid).float()*scale+shift
    q0=(raw_original.clamp(0,1)*255).round().to(torch.uint8)
    q1=(raw_changed.clamp(0,1)*255).round().to(torch.uint8)
    identical=bool(torch.equal(q0,q1))
    record=dict(rounded_rgb_identical_to_baseline=identical,
        rounded_rgb_changed_fraction=float((q0!=q1).float().mean()),
        rounding='round(clamp(RGB,0,1)*255), original processor normalization and temporal packing')
    if identical:
        record['normalized_tensor_identical_to_baseline']=True
        return original,record
    plan=cache['plans']['2'];ids=plan['frame_ids'];frames=q1[:len(ids)].cpu().numpy()
    cpu=processor_input(processor,frames,ids,plan['source_pts'],cache['source_fps'],cache['N'],sample['row']['query'])
    assert cpu.video_grid_thw[0].tolist()==list(grid)
    assert cpu.input_ids[0].tolist()==read(sample['source']['input'])['input_ids']
    result=cpu.pixel_values_videos.to(original.device)
    record['normalized_tensor_identical_to_baseline']=bool(torch.equal(result,original))
    return result,record


def controls(sample,query,model,processor,progress,selection):
    import numpy as np,torch
    import greedy_attribution_runtime as replay
    from probability_budget_runtime import prepare
    from probability_budget_inputs import source_frames
    from oracle_switch_visual import apply
    from oracle_switch_precision import precision_mode,precision_stats
    native=bind_native();start=time.monotonic();key=sample['sample_key']
    dest=OUT/'analyses/controls'/f'{key}.json'
    cpu,info=prepare(sample['row'],processor);assert info['hashes']==sample['input_hashes']
    prompt=cpu.to('cuda');pixels=prompt.pixel_values_videos
    n=int(prompt.input_ids.shape[1]);step=sample['branch']['step'];grid=query['input']['grid']
    mean=query['input']['image_mean'];std=query['input']['image_std'];ids=query['input']['frame_ids']
    support=next(d['support'] for d in query['directions'] if d['id']=='all_video')
    with np.load(query['gradient']['direction_file']) as arrays:sign=torch.from_numpy(arrays['direction_sign']).to('cuda')
    with np.load(query['gradient']['repeat_sign_file']) as arrays:second_sign=torch.from_numpy(arrays['direction_sign']).to('cuda')
    _,cache,source=source_frames(sample['row']);del source
    teacher=dict(prompt)
    teacher['input_ids']=torch.cat((prompt.input_ids,torch.tensor([sample['branch']['prefix_ids']],device='cuda')),dim=1)
    teacher['attention_mask']=torch.ones_like(teacher['input_ids'])
    def evaluate(changed):
        teacher['pixel_values_videos']=changed
        with torch.no_grad():logits=replay.sequential_forward(model,teacher,n,step+1)
        return native.metrics(logits,sample)
    baseline=evaluate(pixels)
    assert baseline==query['baseline'],(key,'fresh control baseline differs')
    result=dict(sample_key=key,complete=False,config_hash=digest(CONFIG),
        input_hashes=info['hashes'],baseline=baseline,baseline_free=query['baseline_free'],
        trials=[],precision_audit=dict(selected=key in selection['prespecified_precision_subset'],modes=[]),timestamp=stamp())
    write(dest,result)
    dtypestr=str(model.model.visual.dtype)
    for epsilon_255 in CONFIG['common_epsilon_255']:
        epsilon=epsilon_255/255
        methods=[('targeted',None),('opposite',None),*[('random',s) for s in CONFIG['random_seeds']],
                 ('repeat_gradient',None),('rounded8_targeted',None)]
        for method,seed in methods:
            progress(stage='matched_control',sample_key=key,method=method,seed=seed,epsilon_255=epsilon_255)
            chosen=second_sign if method=='repeat_gradient' else -sign if method=='opposite' else sign
            actual_seed=None
            if method=='random':chosen,actual_seed=random_direction(sign,grid,ids,seed,key)
            changed,stats=apply(pixels,chosen,grid,support,epsilon,mean,std)
            rounding=None
            if method=='rounded8_targeted':
                continuous=changed;changed,rounding=rounded_pixels(pixels,continuous,grid,mean,std,processor,sample,cache)
                stats=rgb_stats(pixels,changed,grid,support,mean,std);del continuous
            probe=evaluate(changed)
            outcome=native.decode(model,processor,prompt,sample,changed)
            cast=precision_stats(pixels,changed,model.model.visual.dtype)
            trial=dict(method=method,seed=seed,random_generator_seed=actual_seed,epsilon_255=epsilon_255,
                probe=probe,free=outcome,rgb=stats,precision_input=cast)
            if rounding is not None:
                trial['roundtrip']=rounding
                if rounding['normalized_tensor_identical_to_baseline']:
                    assert probe==baseline and outcome['ids']==query['baseline_free']['ids'],(key,'identical input changed output')
            result['trials'].append(trial);write(dest,result)
            del changed,chosen;gc.collect()
    if result['precision_audit']['selected']:
        for mode in CONFIG['precision_modes']:
            progress(stage='precision_control',sample_key=key,method=mode,epsilon_255=None)
            with precision_mode(model,mode) as mode_meta:
                mode_base=evaluate(pixels)
                mode_free=native.decode(model,processor,prompt,sample,pixels)
                mode_record=dict(mode=mode,metadata=mode_meta,baseline=mode_base,baseline_free=mode_free,
                    baseline_matches_native_prefix_and_token=bool(mode_base['prefix_argmax_match'] and mode_base['top1_id']==baseline['top1_id']),
                    conditions=[])
                result['precision_audit']['modes'].append(mode_record)
                for epsilon_255 in CONFIG['common_epsilon_255']:
                    progress(stage='precision_control',method=mode,epsilon_255=epsilon_255)
                    changed,stats=apply(pixels,sign,grid,support,epsilon_255/255,mean,std)
                    probe=evaluate(changed);outcome=native.decode(model,processor,prompt,sample,changed)
                    mode_record['conditions'].append(dict(epsilon_255=epsilon_255,probe=probe,free=outcome,
                        delta_oracle_margin=probe['oracle_margin']-mode_base['oracle_margin'],rgb=stats))
                    del changed;write(dest,result)
        # Matched modes must restore all methods, flags and native output exactly.
        assert evaluate(pixels)==baseline,(key,'precision context did not restore native baseline')
    result.update(complete=True,cost=dict(wall_seconds=time.monotonic()-start),timestamp=stamp())
    write(dest,result)
    for name,path in sample['source'].items():assert sha(path)==sample['source_hashes'][name]
    del sign,second_sign,pixels,prompt,teacher,cpu;gc.collect();torch.cuda.empty_cache()
    return result


def selfcheck():
    import torch
    from oracle_switch_visual import selftest as visual_test,pack_pixels,unpack_pixels
    from oracle_switch_precision import selftest as precision_test
    grid=[2,2,2];reference=torch.ones((8,1536),dtype=torch.int8)
    x,seed=random_direction(reference,grid,[1,2,3,3],42,'mechanics_check')
    y,again=random_direction(reference,grid,[1,2,3,3],42,'mechanics_check')
    frames=unpack_pixels(x,grid)
    assert torch.equal(x,y) and seed==again and torch.equal(frames[2],frames[3])
    result=dict(passed=True,visual=visual_test(),precision=precision_test(),random_repeatable_and_duplicate_tied=True,timestamp=stamp())
    write(OUT/'validation/helpers.json',result)
    return result


def worker(smoke=False):
    progress=Progress();native=bind_native()
    try:
        with (OUT/'metadata/run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            native.wait_gpu(progress)
            for source in read(OUT/'metadata/provenance.json')['files']:
                if not source['path'].endswith(('_report.py','_check.py')):
                    assert sha(source['path'])==source['sha256'],('scientific source changed while queued',source['path'])
            selfcheck()
            import torch
            import greedy_attribution_runtime as replay
            from oracle_switch_precision import get_precision_source_metadata
            torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
            progress(state='running',stage='loading_model')
            model,processor,seconds=replay.load()
            write(OUT/'metadata/model_load.json',dict(seconds=seconds,precision=get_precision_source_metadata(model),timestamp=stamp()))
            selection=read(OUT/'metadata/selection.json')
            completed=failed=0;durations=[]
            for sample in selection['samples']:
                key=sample['sample_key'];query_path=OUT/'queries'/f'{key}.json';control_path=OUT/'analyses/controls'/f'{key}.json'
                if control_path.exists() and read(control_path).get('complete'):
                    completed+=1;failed+=int(read(control_path).get('failed',False));continue
                start=time.monotonic();progress(stage='native_scan',sample_key=key,method=None,epsilon_255=None)
                try:
                    query=read(query_path) if query_path.exists() else None
                    if not query or not query.get('complete'):
                        query=native.process(sample,model,processor,progress,gradient_observer=save_repeat_direction)
                    controls(sample,query,model,processor,progress,selection)
                except native.BaselineReplayError as error:
                    failure=dict(sample_key=key,complete=True,failed=True,failure_kind='baseline_replay',
                        error=str(error),details=error.data,config_hash=digest(CONFIG),timestamp=stamp())
                    write(control_path,failure);failed+=1;gc.collect();torch.cuda.empty_cache()
                completed+=1;durations.append(time.monotonic()-start)
                progress(stage='query_complete',completed=completed,failed_queries=failed,
                    eta_seconds=(60-completed)*sum(durations)/len(durations))
                if (OUT/'metadata/presentation_ready.json').exists():
                    from oracle_switch_controls_check import validate
                    validate()
                    if completed==1 or completed%5==0:publish()
                if smoke:break
            del model,processor;gc.collect();torch.cuda.empty_cache()
            from oracle_switch_controls_check import validate,check_viewer
            progress(stage='postprocessing')
            assert (OUT/'metadata/presentation_ready.json').exists(),'Report/browser implementation has not passed its publication gate'
            validate();publish();check_viewer()
            if smoke:
                progress(state='smoke_complete',stage='smoke_complete');return
            assert completed==60
            from artifact_store import register
            register('oracle_switch_controls.pilot60','analyses',OUT/'analyses/controls',
                'Hash-selected visual perturbation generalization with matched random, rounding, backward-repeat and input-projection precision controls',
                'oracle_switch_controls_runtime.py',str(OUT/'metadata/selection.json'),'complete')
            progress(state='complete',stage='complete',completed=60,failed_queries=failed,eta_seconds=0);publish()
    except BaseException:
        progress(state='failed',stage='failed',error=traceback.format_exc())
        try:publish()
        except Exception:traceback.print_exc()
        raise
    finally:progress.stop.set()


def launch():
    launcher=OUT/'metadata/launcher.json'
    if launcher.exists():
        try:
            pid=read(launcher)['pid']
            if b'oracle_switch_controls_runtime.py' in Path(f'/proc/{pid}/cmdline').read_bytes():
                raise RuntimeError(f'Already queued/running: {pid}')
        except FileNotFoundError:pass
    setup();selfcheck()
    write(OUT/'metadata/status.json',dict(state='queued',stage='waiting_gpu',completed=0,total=60,timestamp=stamp()))
    if (OUT/'metadata/presentation_ready.json').exists():publish()
    with (OUT/'logs/run.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker'],
            cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(launcher,dict(pid=child.pid,detached=True,log=str(OUT/'logs/run.log'),timestamp=stamp()))
    print(json.dumps(read(launcher),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','worker','publish','selfcheck','smoke'])
    args=parser.parse_args()
    if args.action=='smoke':worker(True)
    else:globals()[args.action]()
