"""Integrate query-independent adjacent-frame cosine into existing score files."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_CACHE={}

def precompute_many(specs,progress=None):
    """Parallel ZIP reads followed by CUDA batches spanning eight videos."""
    def load(spec):
        with np.load(spec[0]) as saved:return spec,saved['embeddings'],saved['timestamps']
    todo=[spec for spec in specs if spec not in _CACHE]
    with ThreadPoolExecutor(max_workers=4) as readers:
        for start in range(0,len(todo),8):
            batch=list(readers.map(load,todo[start:start+8]))
            if not batch:continue
            device=os.environ.get('SIMILARITY_ADJACENT_DEVICE','cuda' if torch.cuda.is_available() else 'cpu')
            assert len({x.shape[1] for spec,x,ts in batch})==1
            with torch.inference_mode():
                vectors=torch.from_numpy(np.concatenate([x for spec,x,ts in batch])).to(device)
                unit=F.normalize(vectors.float(),dim=-1)
                all_scores=torch.clamp((unit[:-1]*unit[1:]).sum(-1),-1,1).cpu().numpy()
            offset=0
            for spec,x,ts in batch:
                result=all_scores[offset:offset+len(x)-1].copy();offset+=len(x)
                assert len(x)==len(ts) and np.isfinite(result).all()
                for i in {0,len(result)//2,len(result)-1}:
                    if i<0:continue
                    a=x[i].astype(np.float64);b=x[i+1].astype(np.float64)
                    expected=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                    assert abs(float(result[i])-expected)<1e-6
                _CACHE[spec]=(result,ts)
            if progress:progress(min(start+8,len(todo)),len(todo))

def _compute(path,size,mtime):
    spec=(path,size,mtime)
    if spec not in _CACHE:precompute_many([spec])
    return _CACHE[spec]

def attach(out,model,video_key,sample_key,timestamps):
    out=Path(out)
    dense=(out/'embeddings'/model/f'{video_key}.npz').exists()
    embedding=out/'embeddings'/model/f'{video_key}.npz' if dense else out/model/'video_embeddings'/f'{video_key}.npz'
    target=out/'scores'/model/f'{sample_key}.npz' if dense else out/model/'scores'/f'{sample_key}.npz'
    stat=embedding.stat()
    with np.load(target) as saved:values={k:saved[k] for k in saved.files}
    ts=np.asarray(timestamps)
    np.testing.assert_array_equal(values['timestamps'],ts)
    valid=('adjacent_scores' in values and int(values.get('adjacent_source_size',-1))==stat.st_size
           and int(values.get('adjacent_source_mtime_ns',-1))==stat.st_mtime_ns)
    if valid:
        result=values['adjacent_scores']
        np.testing.assert_array_equal(values['adjacent_start_timestamps'],ts[:-1])
        np.testing.assert_array_equal(values['adjacent_end_timestamps'],ts[1:])
    else:
        result,source_ts=_compute(str(embedding),stat.st_size,stat.st_mtime_ns)
        np.testing.assert_array_equal(source_ts,ts)
        values.update(adjacent_scores=result,adjacent_start_timestamps=ts[:-1],adjacent_end_timestamps=ts[1:],
                      adjacent_source_size=np.asarray(stat.st_size),adjacent_source_mtime_ns=np.asarray(stat.st_mtime_ns))
        temp=target.with_suffix(f'.{os.getpid()}.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,**values)
        temp.replace(target)
    assert len(result)==max(0,len(ts)-1) and np.isfinite(result).all() and np.all(np.abs(result)<=1)
    summary=dict(n_pairs=len(result),mean_cosine=float(result.mean()) if len(result) else None,
                 min_cosine=float(result.min()) if len(result) else None,
                 max_cosine=float(result.max()) if len(result) else None)
    return result.tolist(),summary

def markdown(records):
    lines=['## 인접 frame embedding 유사도','',
        '`a_i = cos(f_(i-1), f_i)` (i=1,…,N−1). Query와 무관한 frame–frame cosine이다. 저장된 이미지 embedding을 사용하며 모델 추론은 추가로 수행하지 않았다.',
        '', '그래프의 점은 두 frame 중 나중 timestamp t_i에 표시한다. Hover에는 이전·현재 timestamp와 시간 간격을 표시하고, 클릭하면 두 frame을 함께 보여준다. 첫 frame에는 이전 frame이 없어 점이 없다. 연결선은 인접 측정값을 이은 표시이며 사이 시점에서 계산한 값이 아니다.',
        '', '아래 평균은 각 video–query 샘플에서 인접 pair cosine을 평균한 뒤 샘플에 동일 가중치를 부여한다. 같은 영상의 다른 query는 동일한 frame–frame 곡선을 공유한다. FPS가 다르면 인접 pair의 시간 간격도 달라진다. 기존 frame–query 지표 및 시간·FLOPs 측정은 유지한다.',
        '', '| Dataset | Model | N | Mean adjacent cosine | Mean pair count |', '|---|---|---:|---:|---:|']
    for ds in ['all']+sorted({r['dataset'] for r in records}):
        for model in ['clip','qwen']:
            rr=[r for r in records if r['model']==model and (ds=='all' or r['dataset']==ds)]
            valid=[r['adjacent_similarity']['mean_cosine'] for r in rr if r['adjacent_similarity']['n_pairs']]
            avg=float(np.mean(valid)) if valid else float('nan')
            lines.append(f"| {ds} | {model} | {len(rr)} | {avg:.6f} | {np.mean([r['adjacent_similarity']['n_pairs'] for r in rr]):.1f} |")
    return '\n'.join(lines)+'\n'
