"""Detached visual-only oracle-switch pilot with explicit GPU waiting state."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse
import fcntl
import gc
import json
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from oracle_switch_core import *


class Progress:
    def __init__(self):
        self.start=time.monotonic(); self.lock=threading.RLock(); self.stop=threading.Event()
        self.state=dict(state='queued',stage='waiting_gpu',completed=0,total=3,scope='pilot',
                        eta_seconds=None,eta_scope='unknown while another GPU job runs')
        self.thread=threading.Thread(target=self.heartbeat,daemon=True); self.thread.start()
    def __call__(self,**fields):
        with self.lock:
            self.state.update(fields,timestamp=stamp(),elapsed_seconds=time.monotonic()-self.start)
            self.state['percentage']=100*self.state['completed']/3
            write(OUT/'metadata/status.json',self.state)
            write(ROOT/'artifacts/common/registry/oracle_switch.live.json',
                  dict(canonical=str(OUT),log=str(OUT/'logs/run.log'),**self.state))
            line=json.dumps(self.state,ensure_ascii=False)
            with (OUT/'logs/progress.log').open('a') as stream: stream.write(line+'\n')
            print(line,flush=True)
    def heartbeat(self):
        while not self.stop.wait(20): self()


def publish():
    from oracle_switch_report import generate
    generate()
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)


def setup():
    from oracle_switch_select import build_selection
    selection=build_selection()
    assert len(selection['samples'])==CONFIG['queries']
    config=OUT/'metadata/config.json'
    if config.exists(): assert read(config)==CONFIG, 'Frozen experiment configuration differs'
    else: write(config,CONFIG)
    for name in ('logs','queries','arrays','media','validation','code'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    files=[]
    sources=list((ROOT/'_lib').glob('oracle_switch_*.py'))
    sources += [ROOT/'_lib'/name for name in ('greedy_attribution_runtime.py','greedy_attribution_core.py',
        'greedy_attribution_cache.py','probability_budget_runtime.py','probability_budget_inputs.py')]
    for source in sorted(sources):
        target=OUT/'code'/source.name; shutil.copyfile(source,target)
        files.append(dict(path=str(source),snapshot=str(target),sha256=sha(source)))
    write(OUT/'metadata/provenance.json',dict(timestamp=stamp(),files=files,
        config_hash=digest(CONFIG),selection_sha256=sha(OUT/'metadata/selection.json'),
        model_manifest_sha256=sha('/home/irteam/backbones/manifests/registry.json'),
        processor_sha256=sha(MODEL/'preprocessor_config.json'),
        cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip()))
    return selection


def gpu_state():
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    pids=[int(line.strip()) for line in processes.splitlines() if line.strip().isdigit()]
    stats=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    free,util=[int(value.strip()) for value in stats.splitlines()[0].split(',')]
    return dict(other_gpu_pids=[pid for pid in pids if pid!=os.getpid()],free_mib=free,utilization=util)


def wait_gpu(progress):
    stable=0
    while stable<2:
        state=gpu_state()
        idle=not state['other_gpu_pids'] and state['free_mib']>=CONFIG['gpu_min_free_mib'] and state['utilization']<10
        stable=stable+1 if idle else 0
        progress(state='queued',stage='waiting_gpu',gpu=state,idle_checks=stable)
        if stable<2: time.sleep(20)


def metrics(logits,sample):
    import torch
    branch=sample['branch']; step=branch['step']; target=branch['oracle_token']; greedy=branch['greedy_token']
    last=logits[step].float(); probs=last.softmax(-1); top1=int(last.argmax())
    top2=last.topk(2); rival=float(top2.values[1] if int(top2.indices[0])==target else top2.values[0])
    prefix_argmax=logits[:step].argmax(-1).tolist()
    prefix_match=prefix_argmax==branch['prefix_ids']; chosen=top1==target
    return dict(margin=float(last[greedy]-last[target]),oracle_margin=float(last[target])-rival,
        greedy_p=float(probs[greedy]),oracle_p=float(probs[target]),top1_id=top1,
        top1_in_original_top30=top1 in branch['top_ids'],paired_reversal=bool(last[target]>last[greedy]),
        target_top1=chosen,strict_target_top1=float(last[target])>rival,
        prefix_argmax_match=prefix_match,prefix_argmax=prefix_argmax,branch_reached=chosen and prefix_match,
        top30=[dict(id=token,logit=float(last[token]),probability=float(probs[token])) for token in branch['top_ids']])


def decode(model,processor,prompt_gpu,sample,pixels):
    import torch
    from greedy_attribution_cache import SnapshotDynamicCache
    from deterministic_rollout300 import parse
    n=int(prompt_gpu.input_ids.shape[1]); ids=[]; cache=None
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    with torch.no_grad():
        for step in range(CONFIG['max_new_tokens']):
            if step==0:
                inputs=dict(prompt_gpu); inputs['pixel_values_videos']=pixels
            else:
                inputs=dict(input_ids=torch.tensor([[ids[-1]]],device=pixels.device),
                    attention_mask=torch.ones((1,n+step),dtype=prompt_gpu.attention_mask.dtype,device=pixels.device),
                    cache_position=torch.tensor([n+step-1],device=pixels.device))
            cache=SnapshotDynamicCache(previous=cache,config=model.config)
            output=model(**inputs,past_key_values=cache,use_cache=True,logits_to_keep=1)
            ids.append(int(output.logits[0,-1].argmax()))
            if ids[-1] in eos: break
    answer=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    parsed=parse(answer); stopped=bool(ids and ids[-1] in eos)
    span=[parsed['start'],parsed['end']] if stopped and parsed and parsed['positive'] else None
    baseline=sample['greedy']['ids']; first=next((i for i,(a,b) in enumerate(zip(ids,baseline)) if a!=b),None)
    if first is None and len(ids)!=len(baseline): first=min(len(ids),len(baseline))
    branch=sample['branch']; step=branch['step']
    return dict(ids=ids,answer=answer,span=span,iou=iou(span,sample['row']['span'][0]) or 0.,
        stopped_on_eos=stopped,first_divergence_step=first,
        oracle_branch_reached=len(ids)>step and ids[:step]==branch['prefix_ids'] and ids[step]==branch['oracle_token'],
        oracle_path_equal=ids==sample['oracle']['ids'],oracle_span_iou=iou(span,sample['oracle']['span']) or 0.,
        endpoint_max_error=max(abs(a-b) for a,b in zip(span,sample['oracle']['span'])) if span else None)


def preview(pixels,changed,grid,support,mean,std,key,direction_id):
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from oracle_switch_visual import unpack_pixels
    slot=support['slots'][0]
    mean_t=torch.tensor(mean,device=pixels.device).view(1,3,1,1)
    std_t=torch.tensor(std,device=pixels.device).view(1,3,1,1)
    original=(unpack_pixels(pixels,grid)[slot:slot+1].float()*std_t+mean_t)[0].clamp(0,1)
    altered=(unpack_pixels(changed,grid)[slot:slot+1].float()*std_t+mean_t)[0].clamp(0,1)
    maximum=float((altered-original).abs().max())
    difference=(altered-original).abs()/(maximum or 1)
    paths={}
    folder=OUT/'media'/key;folder.mkdir(parents=True,exist_ok=True)
    for label,value in [('original',original),('perturbed',altered),('difference',difference)]:
        path=folder/f'{direction_id}_{label}.png'
        image=Image.fromarray((value.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
        image.save(path);paths[label]=str(path)
    paths.update(slot=slot,difference_gain=1/maximum if maximum else 1.,difference_label='Absolute RGB difference, independently amplified to fill [0,1]; PNG only is rounded to uint8')
    return paths


def process(sample,model,processor,progress):
    import numpy as np
    import torch
    import greedy_attribution_runtime as original
    from probability_budget_runtime import prepare
    from oracle_switch_visual import make_directions,apply
    started=time.monotonic();key=sample['sample_key'];dest=OUT/'queries'/f'{key}.json'
    for name,path in sample['source'].items(): assert sha(path)==sample['source_hashes'][name],(key,name)
    progress(stage='preparing_input',sample_key=key,direction=None)
    cpu,info=prepare(sample['row'],processor); assert info['hashes']==sample['input_hashes']
    prompt_gpu=cpu.to('cuda');pixels=prompt_gpu.pixel_values_videos.detach();n=int(prompt_gpu.input_ids.shape[1])
    grid=[int(v) for v in prompt_gpu.video_grid_thw[0].tolist()];step=sample['branch']['step']
    frame_ids=list(info['frame_ids']);times=list(info['timestamps'])
    frame_ids+=frame_ids[-1:]*(2*grid[0]-len(frame_ids));times+=times[-1:]*(2*grid[0]-len(times))
    info.update(grid=grid,frame_ids=frame_ids,timestamps=times)
    video_processor=getattr(processor,'video_processor',processor.image_processor)
    mean=[float(v) for v in video_processor.image_mean];std=[float(v) for v in video_processor.image_std]
    assert len(mean)==len(std)==3 and all(v>0 for v in std)
    assert video_processor.patch_size==16 and video_processor.temporal_patch_size==2 and video_processor.merge_size==2
    info.update(image_mean=mean,image_std=std)
    teacher=dict(prompt_gpu)
    teacher['input_ids']=torch.cat((prompt_gpu.input_ids,torch.tensor([sample['branch']['prefix_ids']],device='cuda')),dim=1)
    teacher['attention_mask']=torch.ones_like(teacher['input_ids'])
    model._attribution_checkpoint=n>original.CONFIG['checkpoint_input_tokens']; torch.cuda.reset_peak_memory_stats()
    def evaluate(changed):
        teacher['pixel_values_videos']=changed
        with torch.no_grad(): full=original.sequential_forward(model,teacher,n,step+1)
        result=metrics(full,sample); del full
        return result
    baseline=evaluate(pixels)
    expected=np.load(sample['source']['arrays'])
    probability_error=max(abs(point['probability']-float(expected['probabilities'][step,i])) for i,point in enumerate(baseline['top30']))
    expected.close()
    assert baseline['prefix_argmax_match'] and baseline['top1_id']==sample['branch']['greedy_token']
    assert probability_error<=original.CONFIG['replay_probability_tolerance'],(key,probability_error)
    progress(stage='baseline_free_decode')
    baseline_free=decode(model,processor,prompt_gpu,sample,pixels)
    assert baseline_free['ids']==sample['greedy']['ids'],(key,'unmodified full greedy replay differs')
    result=dict(sample_key=key,complete=False,config_hash=digest(CONFIG),input=info,baseline=baseline,
        baseline_free=baseline_free,baseline_probability_error=probability_error,directions=[],timestamp=stamp())
    write(dest,result)
    progress(stage='visual_gradient')
    teacher['pixel_values_videos']=pixels.detach().requires_grad_(True)
    full=original.sequential_forward(model,teacher,n,step+1)
    objective=full[step,sample['branch']['greedy_token']].float()-full[step,sample['branch']['oracle_token']].float()
    assert float(objective.detach())==baseline['margin']
    first=torch.autograd.grad(objective,teacher['pixel_values_videos'],retain_graph=True)[0].detach()
    second=torch.autograd.grad(objective,teacher['pixel_values_videos'])[0].detach()
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    a=first.float();b=second.float();denominator=float(a.norm())
    repeat=dict(relative_l2=float((a-b).norm())/max(denominator,1e-30),
        cosine=float((a*b).sum())/max(float(a.norm()*b.norm()),1e-30),
        sign_agreement=float((a.sign()==b.sign()).float().mean()),
        bit_equal=bool(torch.equal(first,second)),used_gradient='first_backward')
    bundle=make_directions(pixels,first,grid,frame_ids,times,mean,std)
    sign=bundle['sign'];supports=bundle['supports']
    result['gradient']=dict(bundle['gradient_summary'],repeat=repeat,checkpointing=model._attribution_checkpoint)
    array_path=OUT/'arrays'/f'{key}.npz'
    np.savez_compressed(array_path,direction_sign=sign.cpu().numpy())
    result['gradient'].update(direction_file=str(array_path),direction_sha256=sha(array_path))
    write(dest,result)
    del full,objective,first,second,a,b;teacher['pixel_values_videos']=pixels;gc.collect();torch.cuda.empty_cache()
    for support in supports:
        progress(stage='visual_scan',direction=support['id'],epsilon_255=0)
        direction=dict(id=support['id'],label=support['label'],support=support,trials=[],boundary=None)
        result['directions'].append(direction);write(dest,result)
        cache={0.:dict(baseline,epsilon=0.,epsilon_255=0.,phase='baseline',perturbation=dict(rgb_linf=0.,rgb_rms=0.,changed_fraction=0.))}
        direction['trials'].append(cache[0.])
        def probe(epsilon,phase='scan',repeat=False):
            if not repeat and epsilon in cache:return cache[epsilon]
            progress(stage=phase,direction=support['id'],epsilon_255=epsilon*255)
            changed,stats=apply(pixels,sign,grid,support,epsilon,mean,std)
            trial=dict(evaluate(changed),epsilon=epsilon,epsilon_255=epsilon*255,phase=phase,perturbation=stats)
            del changed
            if not repeat:cache[epsilon]=trial
            direction['trials'].append(trial);write(dest,result)
            return trial
        low=0.;high=None
        for value in CONFIG['epsilon_255']:
            epsilon=value/255;trial=probe(epsilon)
            if trial['branch_reached']:
                high=epsilon;break
            low=epsilon
        if high is not None:
            for _ in range(CONFIG['refinement_steps']):
                mid=(low+high)/2
                if probe(mid,'refining')['branch_reached']:high=mid
                else:low=mid
            repeated=probe(high,'repeat_boundary',repeat=True)
            direction['repeat_high']=repeated
            direction['boundary']=dict(observed=True,predicate='branch_reached',low_epsilon=low,high_epsilon=high,
                low_epsilon_255=low*255,high_epsilon_255=high*255,repeated_success=repeated['branch_reached'])
            direction['control_opposite']=probe(-high,'opposite_control',repeat=True)
            for label,epsilon in [('free_low',low),('free_high',high)]:
                progress(stage=label,direction=support['id'],epsilon_255=epsilon*255)
                changed,stats=apply(pixels,sign,grid,support,epsilon,mean,std)
                direction[label]=dict(decode(model,processor,prompt_gpu,sample,changed),epsilon=epsilon,epsilon_255=epsilon*255,perturbation=stats)
                if label=='free_high':direction['media']=preview(pixels,changed,grid,support,mean,std,key,support['id'])
                del changed;write(dest,result)
            direction['boundary']['verified']=bool(repeated['branch_reached'] and direction['free_high']['oracle_branch_reached'] and not direction['free_low']['oracle_branch_reached'])
        else:
            limit=CONFIG['epsilon_255'][-1]/255
            progress(stage='free_limit',direction=support['id'],epsilon_255=limit*255)
            changed,stats=apply(pixels,sign,grid,support,limit,mean,std)
            direction['free_limit']=dict(decode(model,processor,prompt_gpu,sample,changed),epsilon=limit,epsilon_255=limit*255,perturbation=stats)
            direction['media']=preview(pixels,changed,grid,support,mean,std,key,support['id'])
            direction['evaluated_limit_255']=CONFIG['epsilon_255'][-1]
            del changed
        direction['complete']=True;write(dest,result)
    result.update(complete=True,cost=dict(wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()),timestamp=stamp())
    write(dest,result)
    for name,path in sample['source'].items():assert sha(path)==sample['source_hashes'][name],(key,'modified source',name)
    del prompt_gpu,teacher,pixels,sign,cpu;gc.collect();torch.cuda.empty_cache()
    return result


def validate():
    import numpy as np
    selection=read(OUT/'metadata/selection.json');done=0
    for sample in selection['samples']:
        path=OUT/'queries'/f"{sample['sample_key']}.json"
        if not path.exists():continue
        result=read(path)
        assert result['input']['hashes']==sample['input_hashes']
        assert result['baseline_free']['ids']==sample['greedy']['ids']
        for direction in result['directions']:
            for trial in direction['trials']:
                assert [p['id'] for p in trial['top30']]==sample['branch']['top_ids']
                assert trial['branch_reached']==(trial['target_top1'] and trial['prefix_argmax_match'])
                assert all(np.isfinite(trial[k]) for k in ('epsilon','margin','oracle_margin','oracle_p'))
                assert trial['perturbation']['rgb_linf']<=abs(trial['epsilon'])+1e-6
            if direction.get('boundary'):
                boundary=direction['boundary'];assert boundary['low_epsilon']<boundary['high_epsilon']
                # Nonrepeatable boundaries are retained as failed verification, not silently accepted.
                assert 'verified' in boundary
        done+=int(result['complete'])
    record=dict(passed=True,completed_queries=done,total_queries=3,complete=done==3,timestamp=stamp())
    write(OUT/'validation/records.json',record);return record


def worker():
    progress=Progress()
    try:
        with (OUT/'metadata/run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            wait_gpu(progress)
            from oracle_switch_visual import selftest
            write(OUT/'validation/visual_helpers.json',selftest())
            import greedy_attribution_runtime as original
            progress(state='running',stage='loading_model',eta_scope='remaining selected queries; excludes postprocessing')
            model,processor,seconds=original.load()
            write(OUT/'metadata/model_load.json',dict(seconds=seconds,timestamp=stamp()))
            samples=read(OUT/'metadata/selection.json')['samples'];completed=0;costs=[]
            for sample in samples:
                path=OUT/'queries'/f"{sample['sample_key']}.json"
                if path.exists() and read(path).get('complete'):completed+=1;continue
                value=process(sample,model,processor,progress);completed+=1;costs.append(value['cost']['wall_seconds'])
                progress(stage='query_complete',completed=completed,eta_seconds=(3-completed)*sum(costs)/len(costs))
                validate();publish()
            del model,processor;gc.collect()
            progress(stage='validating');check=validate();assert check['complete']
            from oracle_switch_check import check as check_view
            progress(state='complete',stage='complete',completed=3,eta_seconds=0)
            publish();check_view()
            from artifact_store import register
            register('oracle_switch.pilot3','analyses',OUT/'queries','Visual pixel interventions for three extreme greedy-to-oracle branches',
                     'oracle_switch_runtime.py',str(OUT/'metadata/selection.json'),'complete')
    except BaseException:
        progress(state='failed',stage='failed',error=traceback.format_exc())
        try:publish()
        except Exception:traceback.print_exc()
        raise
    finally:progress.stop.set()


def launch():
    prior=OUT/'metadata/launcher.json'
    if prior.exists():
        pid=read(prior)['pid']
        try:
            if b'oracle_switch_runtime.py' in Path(f'/proc/{pid}/cmdline').read_bytes():
                raise RuntimeError(f'Already queued/running: {pid}')
        except FileNotFoundError:pass
    setup()
    from oracle_switch_visual import selftest
    write(OUT/'validation/visual_helpers.json',selftest())
    write(OUT/'metadata/status.json',dict(state='queued',stage='waiting_gpu',completed=0,total=3,timestamp=stamp()))
    publish()
    with (OUT/'logs/run.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker'],
            cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(prior,dict(pid=child.pid,detached=True,log=str(OUT/'logs/run.log'),timestamp=stamp()))
    print(json.dumps(read(prior),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','worker','validate','publish'])
    args=parser.parse_args();globals()[args.action]()
