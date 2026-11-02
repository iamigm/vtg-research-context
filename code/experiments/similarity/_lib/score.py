"""CPU-only, resumable cosine scoring; decompress each embedding array once."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo
import numpy as np
from prepare import OUT,video_key,write

def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

def score_model(model):
    rows=json.loads((OUT/'samples.json').read_text());dest=OUT/model
    with np.load(dest/'queries.npz') as saved:
        queries=saved['embeddings']
        np.testing.assert_array_equal(saved['sample_keys'],[s['sample_key'] for s in rows])
    assert np.isfinite(queries).all()
    np.testing.assert_allclose(np.linalg.norm(queries,axis=1),1,atol=2e-5)
    groups=defaultdict(list)
    for i,s in enumerate(rows):groups[s['video_path']].append(i)
    existing={p.stem for p in (dest/'scores').glob('*.npz')}
    done=sum(s['sample_key'] in existing for s in rows)
    counts=Counter(s['dataset'] for s in rows if s['sample_key'] in existing)
    resumed=done;started=time.monotonic();last=0
    def update(stage,force=False):
        nonlocal last
        now=time.monotonic()
        if not force and now-last<20:return
        elapsed=now-started;new=done-resumed
        state=dict(stage=stage,completed_samples=done,total_samples=len(rows),
            percent=100*done/len(rows),per_dataset=dict(counts),elapsed_seconds=elapsed,
            eta_seconds=(len(rows)-done)*elapsed/new if new else None,
            eta_scope='remaining CPU cosine scoring only',updated_at=stamp(),
            completed_videos=len(groups),total_videos=len(groups))
        write(dest/'status.json',state)
        print(stamp(),model,'SCORING',json.dumps(state),flush=True);last=now
    update('scoring',True)
    for path,indices in groups.items():
        pending=[i for i in indices if rows[i]['sample_key'] not in existing]
        if not pending:continue
        key=video_key(path)
        with np.load(dest/'video_embeddings'/f'{key}.npz') as saved:
            vectors=saved['embeddings'];timestamps=saved['timestamps']
        scores=vectors@queries[pending].T
        assert np.isfinite(scores).all() and np.max(np.abs(scores))<=1.00001
        for col,i in enumerate(pending):
            s=rows[i];target=dest/'scores'/f'{s["sample_key"]}.npz'
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,scores=scores[:,col],timestamps=timestamps)
            temp.replace(target)
            done+=1;counts[s['dataset']]+=1
        update('scoring')
    assert done==len(rows)
    update('complete',True)
    print(stamp(),model,'SCORING COMPLETE',done,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['clip','qwen'],required=True)
    score_model(parser.parse_args().model)
