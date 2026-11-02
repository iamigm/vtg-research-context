"""Resumable independent frame/query embeddings, with no generation or joint input."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from prepare import ROOT, OUT, video_key, write

sys.path.insert(0,str(ROOT/'_lib/vendor/Qwen3-VL-Embedding'))
BACKBONES=Path('/home/irteam/backbones/models')

def stamp(): return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

def save_npz(path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f: np.savez_compressed(f,**arrays)
    path.with_suffix('.tmp').replace(path)

class Clip:
    def __init__(self):
        from transformers import CLIPModel,CLIPProcessor
        path=BACKBONES/'clip-vit-large-patch14'
        self.processor=CLIPProcessor.from_pretrained(path,local_files_only=True)
        self.model=CLIPModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,
                                           attn_implementation='sdpa').to('cuda').eval()
        self.truncated=[]
    def text(self,texts):
        for i,t in enumerate(texts):
            if len(self.processor.tokenizer(t)['input_ids'])>77: self.truncated.append(i)
        inp=self.processor(text=texts,return_tensors='pt',padding=True,truncation=True,max_length=77).to('cuda')
        return F.normalize(self.model.get_text_features(**inp).float(),dim=-1)
    def images(self,images):
        inp=self.processor(images=images,return_tensors='pt').to(device='cuda',dtype=torch.float16)
        return F.normalize(self.model.get_image_features(**inp).float(),dim=-1)

class Qwen:
    def __init__(self):
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        self.embedder=Qwen3VLEmbedder(str(BACKBONES/'Qwen3-VL-Embedding-8B'),
            min_pixels=4096,max_pixels=262144,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True)
        self.model=self.embedder.model
        self.truncated=[]
    def encode(self,items):
        from qwen_vl_utils import process_vision_info
        conversations=[self.embedder.format_model_input(**item) for item in items]
        text=self.embedder.processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=False)
        # Fail on corrupt inputs. The upstream convenience method catches vision
        # errors and replaces inputs with NULL; that fallback is unsuitable here.
        images,videos=process_vision_info(conversations,image_patch_size=16)
        assert videos is None
        inp=self.embedder.processor(text=text,images=images,return_tensors='pt',padding=True,
                                     truncation=False,do_resize=False)
        assert inp['input_ids'].shape[0]==len(items)
        if images is not None: assert int(inp['image_grid_thw'].shape[0])==len(items)
        assert inp['input_ids'].shape[1]<=8192
        inp=inp.to('cuda')
        outputs=self.model(**inp,use_cache=False)
        pooled=self.embedder._pooling_last(outputs.last_hidden_state,inp['attention_mask'])
        return F.normalize(pooled.float(),dim=-1)
    def text(self,texts): return self.encode([dict(text=t) for t in texts])
    def images(self,images): return self.encode([dict(image=im) for im in images])

def checked(array):
    array=array.detach().float().cpu().numpy()
    assert np.isfinite(array).all()
    np.testing.assert_allclose(np.linalg.norm(array,axis=-1),1,atol=2e-5)
    return array

def batched_frames(args,backbone,paths,dest):
    """Process prepared videos in bounded windows, overlapping CPU and GPU work."""
    from frame_loader import prefetched
    from concurrent.futures import ThreadPoolExecutor
    started=time.monotonic();frames_done=0;last_log=started
    # A window bounds worker descriptors and allows decoding to continue ahead.
    with ThreadPoolExecutor(max_workers=2) as saver:
        saves=[]
        for offset in range(0,len(paths),128):
            window=paths[offset:offset+128]
            for path in window:
                meta=OUT/'frames'/video_key(path)/'metadata.json'
                while not meta.exists():
                    print(stamp(),'waiting for prepared frames',path,flush=True)
                    time.sleep(5)
            dataset,loader=prefetched(args.model,window,args.batch_size,args.workers)
            if not len(dataset):continue
            buffers={};counts={}
            wait_start=time.monotonic()
            for bi,item in enumerate(loader):
                cpu_wait=time.monotonic()-wait_start
                tick=time.monotonic()
                dtype=torch.float16 if args.model=='clip' else None
                inputs={k:v.to(device='cuda',dtype=dtype if v.is_floating_point() else None,non_blocking=True) for k,v in item['inputs'].items()}
                if args.model=='clip':
                    embeddings=F.normalize(backbone.model.get_image_features(**inputs).float(),dim=-1)
                else:
                    outputs=backbone.model(**inputs,use_cache=False)
                    pooled=backbone.embedder._pooling_last(outputs.last_hidden_state,inputs['attention_mask'])
                    embeddings=F.normalize(pooled.float(),dim=-1)
                array=checked(embeddings)
                gpu_seconds=time.monotonic()-tick
                for row,(key,index) in zip(array,item['refs']):
                    if key not in buffers:
                        buffers[key]=np.empty((dataset.metadata[key]['count'],array.shape[1]),dtype=np.float32)
                        counts[key]=0
                    buffers[key][index]=row;counts[key]+=1
                    if counts[key]==len(buffers[key]):
                        meta=dataset.metadata[key];vectors=buffers.pop(key);counts.pop(key)
                        saves.append(saver.submit(save_npz,dest/'video_embeddings'/f'{key}.npz',
                            embeddings=vectors,timestamps=np.asarray(meta['timestamps']),frame_indices=np.asarray(meta['frame_indices'])))
                frames_done+=len(array)
                now=time.monotonic()
                if bi<3 or now-last_log>=20:
                    print(stamp(),args.model,'PREFETCH',f'window={offset+1}-{min(offset+128,len(paths))}/{len(paths)}',
                        f'batch={len(array)} cpu_wait={cpu_wait:.3f}s gpu={gpu_seconds:.3f}s',
                        f'new_frames={frames_done} fps={frames_done/(now-started):.1f}',flush=True)
                    write(dest/'status.json',dict(stage='embedding',pipeline='cross-video CPU-prefetch',
                        total_videos=len(paths),window_start=offset,frames_processed=frames_done,
                        cpu_wait_seconds=cpu_wait,gpu_batch_seconds=gpu_seconds,batch_size=len(array),
                        elapsed_seconds=now-started,updated_at=stamp()))
                    last_log=now
                wait_start=time.monotonic()
            assert not buffers
            # Propagate asynchronous disk errors and finish the window durably.
            for future in saves:future.result()
            saves.clear()
    return frames_done

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['clip','qwen'],required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-videos',type=int)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    rows=json.loads((OUT/'samples.json').read_text())
    dest=OUT/args.model
    dest.mkdir(parents=True,exist_ok=True)
    config=dict(embedding_spec_version=1,model=args.model,cohort_sha256=hashlib.sha256((OUT/'samples.json').read_bytes()).hexdigest(),
        image_source='shared 2fps JPEG quality95, RGB, max side512; GT independent',
        cosine='FP32 L2-normalized independent embeddings dot product; no learned logit scale',
        extraction='CLIP projected pooled features' if args.model=='clip' else 'official final-layer last valid token; float32 normalization',
        text='raw query' if args.model=='clip' else 'official chat template; system: Represent the user\'s input.',
        qwen_max_pixels=262144,qwen_min_pixels=4096,qwen_attention='sdpa',
        qwen_vendor_commit='393e2978d27852b0d0230d6994f37f9c15bed73c',
        versions={k:importlib.metadata.version(k) for k in ['torch','transformers','qwen-vl-utils','decord']},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (dest/'config.json').exists():
        # Cache identity follows the embedding specification, not storage paths.
        # The original source hash remains intact in the saved run provenance.
        stored=json.loads((dest/'config.json').read_text())
        stored.setdefault('embedding_spec_version',1)
        assert {k:v for k,v in stored.items() if k!='code_sha256'}=={k:v for k,v in config.items() if k!='code_sha256'}
    else: write(dest/'config.json',config)
    print(stamp(),'loading',args.model,flush=True)
    backbone=Clip() if args.model=='clip' else Qwen()
    print(stamp(),'loaded',args.model,flush=True)
    query_path=dest/'queries.npz'
    if not query_path.exists():
        queries=[]
        for i in range(0,len(rows),args.batch_size):
            queries.append(checked(backbone.text([s['query'] for s in rows[i:i+args.batch_size]])))
        save_npz(query_path,embeddings=np.concatenate(queries),sample_keys=np.array([s['sample_key'] for s in rows]))
        print('Text truncation indices (batch-local):',backbone.truncated,flush=True)
    paths=list(dict.fromkeys(s['video_path'] for s in rows))
    if args.max_videos: paths=paths[:args.max_videos]
    started=time.time()
    total_frames=0
    for vi,path in enumerate([] if args.workers else paths):
        key=video_key(path)
        target=dest/'video_embeddings'/f'{key}.npz'
        meta_path=OUT/'frames'/key/'metadata.json'
        while not meta_path.exists():
            write(dest/'status.json',dict(stage='waiting_frames',video=key,completed_videos=vi,total_videos=len(paths),updated_at=stamp()))
            time.sleep(5)
        meta=json.loads(meta_path.read_text())
        n=meta['count']
        if target.exists():
            cached=np.load(target)
            assert len(cached['embeddings'])==n
            np.testing.assert_array_equal(cached['timestamps'],meta['timestamps'])
        else:
            parts=[]
            video_started=time.time()
            for start in range(0,n,args.batch_size):
                ims=[]
                for i in range(start,min(start+args.batch_size,n)):
                    with Image.open(meta_path.parent/f'{i:05d}.jpg') as im: ims.append(im.convert('RGB'))
                parts.append(checked(backbone.images(ims)))
                for im in ims: im.close()
                write(dest/'status.json',dict(stage='embedding',video=key,completed_videos=vi,total_videos=len(paths),
                    frame=start+len(ims),frames_in_video=n,total_frames=total_frames+start+len(ims),
                    elapsed_seconds=time.time()-started,updated_at=stamp()))
            save_npz(target,embeddings=np.concatenate(parts),timestamps=np.asarray(meta['timestamps']),
                     frame_indices=np.asarray(meta['frame_indices']))
            print(stamp(),args.model,f'{vi+1}/{len(paths)}',key,'frames',n,'seconds',round(time.time()-video_started,2),flush=True)
        total_frames+=n
    if args.workers:
        batched_frames(args,backbone,paths,dest)
        total_frames=sum(json.loads((OUT/'frames'/video_key(p)/'metadata.json').read_text())['count'] for p in paths)
    if not args.max_videos:
        from score import score_model
        score_model(args.model)
    else:
        with np.load(query_path) as saved:queries=saved['embeddings']
        for i,s in enumerate(rows):
            fp=dest/'video_embeddings'/f'{video_key(s["video_path"])}.npz'
            if not fp.exists():continue
            with np.load(fp) as v:
                scores=v['embeddings']@queries[i]
                save_npz(dest/'scores'/f'{s["sample_key"]}.npz',scores=scores,timestamps=v['timestamps'])
    write(dest/'status.json',dict(stage='complete' if not args.max_videos else 'smoke_complete',
        completed_videos=len(paths),total_videos=len(paths),total_frames=total_frames,updated_at=stamp()))
    print(stamp(),args.model,'COMPLETE',flush=True)

if __name__=='__main__': main()
