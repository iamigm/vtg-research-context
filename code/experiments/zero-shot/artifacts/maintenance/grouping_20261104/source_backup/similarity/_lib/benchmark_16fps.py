"""Pilot 16fps similarity vs Instruct: isolated latency, inputs and analytic FLOPs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
import copy
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from compute_cost import clip_flops,qwen_flops
from metrics import metrics,labels

ROOT=Path('/home/irteam/experiments/similarity')
RUN='pilot_300_16fps_cost_v1'
OUT=ROOT/'artifacts/runs'/RUN
VIEW=ROOT/'results'/RUN
LOG=OUT/'logs'
MANIFEST=Path('/home/irteam/experiments/zero-shot/artifacts/common/predictions/legacy/cohorts/pilot_300/samples.json')
MODELS=Path('/home/irteam/backbones/models')
VENDOR=ROOT/'_lib/vendor/Qwen3-VL-Embedding'
TIMELENS=Path('/home/irteam/experiments/_shared/timelens/vendor/TimeLens')
sys.path[:0]=[str(VENDOR),str(TIMELENS)]
FPS=16

def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def key(path):return hashlib.sha256(path.encode()).hexdigest()[:20]
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def arrays(path,**values):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    temp.replace(path)
def log(*args):print(stamp(),*args,flush=True)

class Progress:
    def __init__(self):
        self.lock=threading.RLock();self.state=dict(started_at=stamp());self.stop=threading.Event()
        self.started=time.monotonic();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def set(self,**kw):
        with self.lock:self.state.update(kw)
    def emit(self):
        with self.lock:
            s=dict(self.state,updated_at=stamp(),elapsed_seconds=time.monotonic()-self.started)
            n=s.get('completed',0);total=s.get('total',0)
            s['percentage']=100*n/total if total else 0
            write(LOG/'status.json',s);log('PROGRESS',json.dumps(s,ensure_ascii=False))
    def loop(self):
        while not self.stop.wait(20):self.emit()

def prepare_video(path):
    import decord
    torch.set_num_threads(1)
    folder=OUT/'inputs/frames'/key(path);meta=folder/'metadata.json'
    stat=Path(path).stat();signature=dict(path=path,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
        target_fps=FPS,max_side=512,jpeg_quality=95,sampling='unique round(k*source_fps/16); no frame-count cap')
    if meta.exists():
        m=json.loads(meta.read_text());assert m['signature']==signature
        assert all((folder/f'{i:05d}.jpg').exists() for i in range(m['count']))
        return m
    folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    vr=decord.VideoReader(path,num_threads=1);fps=float(vr.get_avg_fps())
    ids=np.unique(np.rint(np.arange(0,len(vr),fps/FPS)).astype(int));ids=ids[ids<len(vr)]
    ts=np.asarray(vr.get_frame_timestamp(ids.tolist()))[:,0].astype(float)
    decode=0.;resize=0.
    for offset in range(0,len(ids),32):
        tick=time.perf_counter();batch=vr.get_batch(ids[offset:offset+32].tolist()).asnumpy();decode+=time.perf_counter()-tick
        tick=time.perf_counter()
        for i,frame in enumerate(batch,offset):
            im=Image.fromarray(frame).convert('RGB');im.thumbnail((512,512),Image.Resampling.LANCZOS)
            im.save(folder/f'{i:05d}.jpg',quality=95,subsampling=0)
        resize+=time.perf_counter()-tick
    m=dict(signature=signature,source_fps=fps,source_frames=len(vr),
        decoded_duration=float(vr.get_frame_timestamp(len(vr)-1)[1]),count=len(ids),
        frame_indices=ids.tolist(),timestamps=ts.tolist(),decode_seconds=decode,
        resize_and_jpeg_seconds=resize,shared_preparation_seconds=time.perf_counter()-start)
    write(meta,m);return m

def read_images(meta,start,end):
    folder=OUT/'inputs/frames'/key(meta['signature']['path']);images=[]
    for i in range(start,end):
        with Image.open(folder/f'{i:05d}.jpg') as im:images.append(im.convert('RGB'))
    return images

def timed_gpu(fn):
    torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    tick=time.perf_counter();a.record();result=fn();b.record();torch.cuda.synchronize()
    return result,time.perf_counter()-tick,a.elapsed_time(b)/1000

def transfer(inputs,dtype=None):
    torch.cuda.synchronize();tick=time.perf_counter()
    out={k:v.to(device='cuda',dtype=dtype if v.is_floating_point() else None) for k,v in inputs.items()}
    torch.cuda.synchronize();return out,time.perf_counter()-tick

class Embedding:
    def __init__(self,name):
        self.name=name
        if name=='clip':
            from transformers import CLIPProcessor,CLIPModel
            p=MODELS/'clip-vit-large-patch14';self.proc=CLIPProcessor.from_pretrained(p,local_files_only=True)
            self.model=CLIPModel.from_pretrained(p,dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
            self.batch=256
        else:
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
            self.embedder=Qwen3VLEmbedder(str(MODELS/'Qwen3-VL-Embedding-8B'),min_pixels=4096,max_pixels=262144,
                dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
            self.proc=self.embedder.processor;self.model=self.embedder.model;self.batch=128
        self.config=self.model.config.to_dict()
    def prepare(self,images=None,text=None):
        tick=time.perf_counter()
        if self.name=='clip':
            if images is None:inputs=self.proc(text=[text],return_tensors='pt',padding=True,truncation=True,max_length=77)
            else:inputs=self.proc(images=images,return_tensors='pt')
        else:
            from qwen_vl_utils import process_vision_info
            items=[dict(image=im) for im in images] if images is not None else [dict(text=text)]
            conversations=[self.embedder.format_model_input(**item) for item in items]
            prompts=self.proc.apply_chat_template(conversations,add_generation_prompt=True,tokenize=False)
            vision,videos=process_vision_info(conversations,image_patch_size=16);assert videos is None
            inputs=self.proc(text=prompts,images=vision,return_tensors='pt',padding=True,truncation=False,do_resize=False)
            assert inputs['input_ids'].shape[1]<=8192
        return inputs,time.perf_counter()-tick
    def forward(self,inputs,is_text=False):
        if self.name=='clip':
            out=self.model.get_text_features(**inputs) if is_text else self.model.get_image_features(**inputs)
        else:
            outputs=self.model(**inputs,use_cache=False)
            out=self.embedder._pooling_last(outputs.last_hidden_state,inputs['attention_mask'])
        return F.normalize(out.float(),dim=-1)
    def batch_input(self,meta,start,end):
        tick=time.perf_counter();images=read_images(meta,start,end);inputs,prep=self.prepare(images=images)
        for im in images:im.close()
        return inputs,time.perf_counter()-tick

def gpu_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(v.strip()) for v in text.splitlines() if v.strip().isdigit() and int(v.strip())!=os.getpid()]

@torch.inference_mode()
def similarity_sample(engine,s,meta,warm=False):
    if warm:
        images=read_images(meta,0,min(8,meta['count']))
        images=(images*math.ceil(engine.batch/len(images)))[:engine.batch]
        cpu,_=engine.prepare(images=images);device,_=transfer(cpu,torch.float16 if engine.name=='clip' else None)
        timed_gpu(lambda:engine.forward(device));return
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();cpu_seconds=0.;h2d=0.;gpu=0.;cuda=0.
    cpu,elapsed=engine.prepare(text=s['query']);cpu_seconds+=elapsed
    query_tokens=int(cpu['input_ids'].shape[1]);device,elapsed=transfer(cpu,torch.float16 if engine.name=='clip' else None);h2d+=elapsed
    q,wall,event=timed_gpu(lambda:engine.forward(device,True));gpu+=wall;cuda+=event
    q=q.cpu().numpy()[0];parts=[];grids=[];lengths=[query_tokens]
    # Prefetch the next batch within the same sample; samples never overlap.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(engine.batch_input,meta,0,min(engine.batch,meta['count']))
        for offset in range(0,meta['count'],engine.batch):
            cpu,elapsed=future.result();cpu_seconds+=elapsed
            next_offset=offset+engine.batch
            if next_offset<meta['count']:future=pool.submit(engine.batch_input,meta,next_offset,min(next_offset+engine.batch,meta['count']))
            if engine.name!='clip':
                grids.extend(cpu['image_grid_thw'].tolist());lengths.extend([int(cpu['input_ids'].shape[1])]*int(cpu['input_ids'].shape[0]))
            device,elapsed=transfer(cpu,torch.float16 if engine.name=='clip' else None);h2d+=elapsed
            out,wall,event=timed_gpu(lambda:engine.forward(device));gpu+=wall;cuda+=event
            parts.append(out.cpu().numpy())
    frames=np.concatenate(parts);tick=time.perf_counter();scores=frames@q;cosine=time.perf_counter()-tick
    cached_wall=time.perf_counter()-start
    cost=clip_flops(engine.config,meta['count'],query_tokens) if engine.name=='clip' else qwen_flops(engine.config,grids,lengths,cosine_frames=meta['count'])
    r=dict(sample_key=s['sample_key'],dataset=s['dataset'],model=engine.name,query=s['query'],fps=FPS,
        frame_count=meta['count'],query_tokens=query_tokens,input_tokens=sum(lengths),generated_tokens=0,
        gpu_forward_wall_seconds=gpu,cuda_event_seconds=cuda,h2d_seconds=h2d,
        cpu_input_work_seconds=cpu_seconds,cached_frame_wall_seconds=cached_wall,cosine_seconds=cosine,
        shared_source_preparation_seconds=meta['shared_preparation_seconds'],
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),flops=cost,
        similarity=metrics(scores,labels(meta['timestamps'],s['span'])))
    arrays(OUT/'scores'/engine.name/f'{s["sample_key"]}.npz',scores=scores,timestamps=np.asarray(meta['timestamps']))
    target=OUT/'embeddings'/engine.name/f'{key(s["video_path"])}.npz'
    if not target.exists():arrays(target,embeddings=frames,timestamps=np.asarray(meta['timestamps']))
    arrays(OUT/'embeddings'/engine.name/'queries'/f'{s["sample_key"]}.npz',embedding=q)
    return r

def generation_input(proc,s,meta,limit=None):
    from qwen_vl_utils.vision_process import smart_resize
    from evaluation.utils import GROUNDER_PROMPT
    count=meta['count'] if limit is None else min(limit,meta['count'])
    folder=OUT/'inputs/frames'/key(s['video_path'])
    with Image.open(folder/'00000.jpg') as im:width,height=im.size
    # Preserve the previous min_tokens=64 and total_tokens=14336 policy, but
    # bypass the default 768-frame cap. total_tokens is a soft spatial budget:
    # min_pixels takes precedence, so long 16fps videos can exceed that budget.
    min_pixels=64*1024;total_pixels=14336*1024
    max_pixels=max(min(768*1024,total_pixels/count*2),int(min_pixels*1.05))
    rh,rw=smart_resize(height,width,factor=32,min_pixels=min_pixels,max_pixels=max_pixels)
    frames=[]
    for i in range(count):
        with Image.open(folder/f'{i:05d}.jpg') as im:
            im=im.convert('RGB').resize((rw,rh),Image.Resampling.BICUBIC)
            frames.append(np.asarray(im).copy())
    ids=meta['frame_indices'][:count]
    if count%2:frames.append(frames[-1]);ids=ids+[ids[-1]]
    video=torch.from_numpy(np.stack(frames)).permute(0,3,1,2)
    conversation=[dict(role='user',content=[dict(type='video',video='shared-16fps-frames'),dict(type='text',text=GROUNDER_PROMPT.format(s['query']))])]
    prompt=proc.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
    video_metadata=dict(fps=meta['source_fps'],frames_indices=list(ids),total_num_frames=meta['source_frames'])
    inputs=proc(text=[prompt],videos=[video],video_metadata=[copy.deepcopy(video_metadata)],
        do_sample_frames=False,do_resize=False,return_tensors='pt')
    grid=inputs['video_grid_thw'][0].tolist()
    assert grid[0]*2==len(frames),('Temporal sampling changed',grid,len(frames))
    assert inputs['input_ids'].shape[1]+512<262144
    return inputs,dict(frame_count=count,padded_frames=len(frames),resized_height=rh,resized_width=rw,
        video_grid_thw=grid,visual_tokens=int(np.prod(grid)//4),frame_indices=ids,
        timestamps=meta['timestamps'][:count],source_fps=meta['source_fps'],
        actual_sampled_fps=count/meta['decoded_duration'],sampling_capped=False)

@torch.inference_mode()
def generation_sample(model,proc,s,meta,warm=False):
    from timelens.utils import extract_time,iou
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();tick=time.perf_counter()
    cpu,info=generation_input(proc,s,meta,limit=8 if warm else None);cpu_seconds=time.perf_counter()-tick
    inputs,h2d=transfer(cpu);n=int(inputs['input_ids'].shape[1]);events=[]
    def before(module,args,kwargs):
        if not events:
            event=torch.cuda.Event(enable_timing=True);event.record();events.append(event)
    def after(module,args,kwargs,output):
        if len(events)==1:
            event=torch.cuda.Event(enable_timing=True);event.record();events.append(event)
    a=model.register_forward_pre_hook(before,with_kwargs=True);b=model.register_forward_hook(after,with_kwargs=True)
    try:
        output,wall,cuda=timed_gpu(lambda:model.generate(**inputs,do_sample=False,temperature=None,top_p=None,top_k=None,
            max_new_tokens=2 if warm else 512,use_cache=True,logits_to_keep=1))
    finally:a.remove();b.remove()
    if warm:return
    generated=output[0,n:].tolist();answer=proc.tokenizer.decode(generated,skip_special_tokens=True)
    cached_wall=time.perf_counter()-start;spans=extract_time(answer) or []
    cost=qwen_flops(model.config.to_dict(),[info['video_grid_thw']],[n],generated=len(generated),lm_head=True)
    prefill=events[0].elapsed_time(events[1])/1000
    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
    r=dict(sample_key=s['sample_key'],dataset=s['dataset'],model='instruct',query=s['query'],fps=FPS,
        frame_count=meta['count'],input_tokens=n,generated_tokens=len(generated),
        gpu_forward_wall_seconds=wall,cuda_event_seconds=cuda,prefill_cuda_seconds=prefill,
        decode_cuda_seconds=max(0,cuda-prefill),h2d_seconds=h2d,cpu_input_work_seconds=cpu_seconds,
        cached_frame_wall_seconds=cached_wall,shared_source_preparation_seconds=meta['shared_preparation_seconds'],
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),flops=cost,answer=answer,
        predicted_spans=spans,iou=iou(s['span'][0],spans[0]) if spans else 0.,
        eos_reached=bool(generated and generated[-1] in eos),input=info)
    arrays(OUT/'inputs/token_ids'/f'{s["sample_key"]}.npz',input_ids=cpu['input_ids'].numpy(),generated_ids=np.asarray(generated))
    return r

def execute(progress,stages):
    rows=json.loads(MANIFEST.read_text());assert len(rows)==300
    assert all(v==100 for v in Counter(s['dataset'] for s in rows).values())
    write(OUT/'config/samples.json',rows)
    registry=json.loads(Path('/home/irteam/backbones/manifests/similarity_download.json').read_text())
    instruct_registry=[r for r in json.loads(Path('/home/irteam/backbones/manifests/registry.json').read_text()) if r['name']=='Qwen3-VL-8B-Instruct']
    config=dict(run=RUN,manifest=str(MANIFEST),manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        fps=FPS,models=['clip','qwen','instruct'],similarity=dict(batch_clip=256,batch_qwen=128,max_side=512,qwen_max_pixels=262144),
        generation=dict(model='Qwen3-VL-8B-Instruct',order='video_query',do_sample=False,max_new_tokens=512,
            attention='sdpa',min_tokens=64,total_tokens_soft_budget=14336,frame_cap=None),
        timing='one sample at a time, model stages sequential, warmup excluded; CPU batch prefetch within sample',
        flops='analytic matmul/conv forward estimate; multiply-add=2; causal attention triangle; excludes norm/activation/softmax/memory',
        model_provenance=registry['models']+instruct_registry,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'_lib/compute_cost.py']})
    saved=OUT/'config/config.json'
    if saved.exists():
        old=json.loads(saved.read_text());assert {k:v for k,v in old.items() if k!='source_sha256'}=={k:v for k,v in config.items() if k!='source_sha256'}
    else:write(saved,config)
    if 'prepare' in stages:
        paths=list(dict.fromkeys(s['video_path'] for s in rows));progress.set(stage='prepare_16fps',completed=0,total=len(paths),per_dataset={})
        with ProcessPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(prepare_video,path) for path in paths]
            for i,f in enumerate(as_completed(futures),1):
                meta=f.result();progress.set(completed=i,video=meta['signature']['path']);log('PREPARED',i,len(paths),meta['count'])
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    for name in ['clip','qwen','instruct']:
        if name not in stages:continue
        todo=[s for s in rows if not (OUT/'predictions'/name/f'{s["sample_key"]}.json').exists()]
        if not todo:log('REUSE COMPLETE STAGE',name);continue
        progress.set(stage='loading_'+name,completed=len(rows)-len(todo),total=len(rows));progress.emit()
        others=gpu_processes()
        if others:raise RuntimeError(f'Timing requires an idle GPU; other GPU process IDs: {others}')
        t=time.perf_counter()
        if name=='instruct':
            from transformers import AutoProcessor,AutoModelForImageTextToText
            proc=AutoProcessor.from_pretrained(MODELS/'Qwen3-VL-8B-Instruct',padding_side='left',do_resize=False,local_files_only=True)
            model=AutoModelForImageTextToText.from_pretrained(MODELS/'Qwen3-VL-8B-Instruct',dtype=torch.bfloat16,
                attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
            engine=None
        else:engine=Embedding(name);model=engine.model;proc=engine.proc
        load_seconds=time.perf_counter()-t
        first=todo[0];meta=json.loads((OUT/'inputs/frames'/key(first['video_path'])/'metadata.json').read_text())
        progress.set(stage='warmup_'+name)
        if engine:similarity_sample(engine,first,meta,warm=True)
        else:generation_sample(model,proc,first,meta,warm=True)
        counts=Counter(s['dataset'] for s in rows if (OUT/'predictions'/name/f'{s["sample_key"]}.json').exists())
        done=sum(counts.values());phase=time.perf_counter();initial=done
        for s in todo:
            progress.set(stage=name,sample_key=s['sample_key'],completed=done,per_dataset=dict(counts))
            others=gpu_processes()
            if others:raise RuntimeError(f'External GPU workload detected before sample: {others}; resume after GPU becomes idle')
            meta=json.loads((OUT/'inputs/frames'/key(s['video_path'])/'metadata.json').read_text())
            if engine:r=similarity_sample(engine,s,meta)
            else:r=generation_sample(model,proc,s,meta)
            r.update(completed_at=stamp(),other_gpu_pids=others)
            write(OUT/'predictions'/name/f'{s["sample_key"]}.json',r)
            done+=1;counts[s['dataset']]+=1;elapsed=time.perf_counter()-phase
            progress.set(completed=done,per_dataset=dict(counts),eta_seconds=(300-done)*elapsed/(done-initial),eta_scope='remaining samples in current model stage')
            log('SAMPLE',name,f'{done}/300',s['sample_key'],f"gpu={r['gpu_forward_wall_seconds']:.3f}s wall={r['cached_frame_wall_seconds']:.3f}s TFLOPs={r['flops']['total_flops']/1e12:.3f}")
        write(OUT/'measurements'/f'{name}_stage.json',dict(model=name,load_seconds=load_seconds,
            measured_stage_wall_seconds=time.perf_counter()-phase,measured_samples=len(todo),excluded_warmup=True,
            parameters=sum(p.numel() for p in model.parameters()),model_config=model.config.to_dict(),
            hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
            torch_version=torch.__version__))
        del model,proc,engine;gc.collect();torch.cuda.empty_cache()
    if 'report' in stages:
        progress.set(stage='report_and_validation',completed=0,total=1)
        subprocess.run([sys.executable,str(ROOT/'_lib/report_16fps.py')],check=True)
    progress.set(stage='complete',completed=300,total=300);progress.emit()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stages',default='prepare,clip,qwen,instruct,report');args=parser.parse_args()
    LOG.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(LOG/'benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    progress=Progress()
    try:execute(progress,args.stages.split(','))
    except BaseException as e:progress.set(stage='failed',error=repr(e));progress.emit();raise
    finally:progress.stop.set();progress.thread.join(timeout=2)

if __name__=='__main__':main()
