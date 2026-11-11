from result_layout import relocate_path
"""Shared source-frame cache. Metadata/PTS remain full; RGB arrays contain only requested IDs."""
import os,fcntl,hashlib
from pathlib import Path
import numpy as np
from adaptive_temporal_sampling import ROOT,read,write,sha,need,stamp
COMMON=ROOT/'artifacts/common/selected_frames_v1'

def manifest_path(cache):return COMMON/'frames'/(cache['cache_key']+'.json')
def available_metadata(video,grid):
    index=COMMON/'video_index.json'
    if not index.exists():return None
    for x in read(index).get(str(Path(video).resolve())+'|'+','.join(map(str,grid)),[]):
        metadata=read(x);m=manifest_path(metadata)
        if m.exists() and metadata['source_video_hash']==sha(video):return x
    return None

def load_selected(cache,ids):
    """Return requested RGB; fill missing observations by the same increasing decoder path."""
    ids=list(ids)
    mp=manifest_path(cache)
    if not mp.exists():
        loc={i:j for j,i in enumerate(cache['ids'])}
        return np.array(np.load(relocate_path(Path(cache['path'])),mmap_mode='r')[[loc[i] for i in ids]])
    mp.parent.mkdir(parents=True,exist_ok=True)
    with mp.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        m=read(mp)
        if 'canonical_cache_key' in m:
            key=m['canonical_cache_key']
            need(isinstance(key,str) and len(key)==64 and all(c in '0123456789abcdef' for c in key),'Invalid canonical cache key')
            need(key!=cache['cache_key'],'Shared RGB cache cannot alias itself')
            canonical_cache=dict(cache,cache_key=key);cp=manifest_path(canonical_cache)
            need(cp.exists(),'Canonical shared RGB manifest missing')
            # Reject aliases before taking another lock, including A -> B -> A.
            need('canonical_cache_key' not in read(cp),'Shared RGB alias chains are unsupported')
            with cp.with_suffix('.lock').open('a') as canonical_lock:
                fcntl.flock(canonical_lock,fcntl.LOCK_EX)
                canonical=read(cp)
                need(canonical.get('cache_key')==key,'Canonical shared RGB cache key mismatch')
                need('canonical_cache_key' not in canonical,'Shared RGB alias chains are unsupported')
                # Resolve current rows under the canonical lock. Extensions update
                # only the canonical array and manifest, never an alias snapshot.
                selected=_load_manifest_selected(canonical_cache,ids,canonical)
            expected=dict(zip(m['ids'],m['rgb_hashes']))
            need(all(i not in expected or hashlib.sha256(a.tobytes()).hexdigest()==expected[i] for a,i in zip(selected,ids)),'Aliased shared RGB checksum mismatch')
            return selected
        return _load_manifest_selected(cache,ids,m)

def _load_manifest_selected(cache,ids,m):
    """Read or extend a physical cache while its manifest lock is held."""
    loc={i:j for j,i in enumerate(m['ids'])};missing=set(ids)-set(loc)
    if missing:m=extend(cache,m,missing);loc={i:j for j,i in enumerate(m['ids'])}
    selected=np.array(np.load(relocate_path(Path(m['path'])),mmap_mode='r')[[loc[i] for i in ids]])
    need(all(hashlib.sha256(a.tobytes()).hexdigest()==m['rgb_hashes'][loc[i]] for a,i in zip(selected,ids)),'Selected shared RGB checksum mismatch')
    return selected

def extend(cache,m,missing):
    import torch,decord
    from torchvision.transforms import functional as TF,InterpolationMode
    need(sha(cache['video_path'])==cache['source_video_hash'],'Original video changed')
    vr=decord.VideoReader(cache['video_path'],num_threads=1);need(len(vr)==cache['N'],'Source frame count changed');H,W=cache['grid_hw'];new={};locations={i:j for j,i in enumerate(cache['ids'])}
    # Preserve old decode batch boundaries, but resize/store requested frames only.
    for start in range(0,max(missing)+1,32):
        batch=list(range(start,min(start+32,len(vr))));rgb=vr.get_batch(batch).asnumpy();wanted=[i for i in batch if i in missing]
        if not wanted:continue
        frames=TF.resize(torch.from_numpy(rgb[[i-start for i in wanted]]).permute(0,3,1,2),[H,W],interpolation=InterpolationMode.BICUBIC,antialias=True).numpy()
        for i,a in zip(wanted,frames):
            expected=cache['resized_rgb_hashes'][locations[i]]
            need(hashlib.sha256(a.tobytes()).hexdigest()==expected,'On-demand RGB differs from original decoder cache');new[i]=a
    old=np.load(relocate_path(Path(m['path'])),mmap_mode='r');loc={i:j for j,i in enumerate(m['ids'])};allids=sorted(set(m['ids'])|missing);tmp=relocate_path(Path(m['path'])).with_suffix('.extending.npy');arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(allids),*old.shape[1:]));hashes=[]
    for j,i in enumerate(allids):
        a=new[i] if i in new else old[loc[i]];arr[j]=a;hashes.append(hashlib.sha256(a.tobytes()).hexdigest())
    arr.flush();del arr,old;os.replace(tmp,relocate_path(Path(m['path'])));m.update(ids=allids,rgb_hashes=hashes,sha256=sha(m['path']),updated_at=stamp());write(manifest_path(cache),m);return m
