"""Fixed, GT-independent 2fps sampling; shared JPEG inputs for both backbones."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from result_layout import canonical, relocate_path
import numpy as np
from PIL import Image

ROOT = Path(canonical(Path(__file__).resolve().parents[1]))
RUN = os.environ.get('SIMILARITY_RUN','pilot_300_v1')
assert RUN and Path(RUN).name==RUN
VIEW = relocate_path(ROOT/'results'/RUN)
OUT = relocate_path(ROOT/'artifacts/runs'/RUN)
MANIFEST = relocate_path(Path(os.environ.get('SIMILARITY_MANIFEST','/home/irteam/experiments/zero-shot/artifacts/common/predictions/legacy/cohorts/pilot_300/samples.json')))

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def video_key(path):
    return hashlib.sha256(str(path).encode()).hexdigest()[:20]

def prepare_video(path):
    import decord
    key = video_key(path)
    folder = OUT/'frames'/key
    meta = folder/'metadata.json'
    stat = Path(path).stat()
    signature = dict(path=path, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                     fps=2, max_side=512, jpeg_quality=95, sampling='nearest source index to k*fps_source/2')
    if meta.exists():
        m = json.loads(meta.read_text())
        assert m['signature'] == signature
        assert all((folder/f'{i:05d}.jpg').exists() for i in range(len(m['timestamps'])))
        return key, m
    folder.mkdir(parents=True, exist_ok=True)
    vr = decord.VideoReader(path, num_threads=1)
    fps = float(vr.get_avg_fps())
    ids = np.unique(np.rint(np.arange(0, len(vr), fps/2)).astype(int))
    ids = ids[ids < len(vr)]
    timestamps = np.asarray(vr.get_frame_timestamp(ids.tolist()))[:,0].astype(float)
    for start in range(0, len(ids), 32):
        frames = vr.get_batch(ids[start:start+32].tolist()).asnumpy()
        for j, frame in enumerate(frames, start):
            im = Image.fromarray(frame).convert('RGB')
            im.thumbnail((512,512), Image.Resampling.LANCZOS)
            im.save(folder/f'{j:05d}.jpg', quality=95, subsampling=0)
    m = dict(signature=signature, source_fps=fps, source_frames=len(vr),
             decoded_duration=float(vr.get_frame_timestamp(len(vr)-1)[1]),
             frame_indices=ids.tolist(), timestamps=timestamps.tolist(), count=len(ids))
    write(meta,m)
    return key,m

def main():
    rows = json.loads(MANIFEST.read_text())
    assert rows and len({s['sample_key'] for s in rows})==len(rows)
    write(OUT/'samples.json',rows)
    paths = list(dict.fromkeys(s['video_path'] for s in rows))
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get('SIMILARITY_PREP_WORKERS','6'))) as pool:
        futures = {pool.submit(prepare_video,p):p for p in paths}
        for f in concurrent.futures.as_completed(futures):
            key,m=f.result()
            completed+=1
            print(f'prepared {completed}/{len(paths)} {key} frames={m["count"]}',flush=True)
    coverage=[]
    for s in rows:
        m=json.loads((OUT/'frames'/video_key(s['video_path'])/'metadata.json').read_text())
        ts=np.asarray(m['timestamps'])
        y=np.zeros(len(ts),bool)
        for a,b in s['span']: y |= (ts>=a)&(ts<b)
        coverage.append(dict(sample_key=s['sample_key'],dataset=s['dataset'],n=len(ts),n_gt=int(y.sum()),
                             gt_fraction=float(y.mean()), gt=s['span'],video_key=video_key(s['video_path'])))
    print('GT coverage: zero-positive samples',
          [s['sample_key'] for s in coverage if s['n_gt']==0],flush=True)
    print('PREPARATION COMPLETE',flush=True)

if __name__=='__main__': main()
