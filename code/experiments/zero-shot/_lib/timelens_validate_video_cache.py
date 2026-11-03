from result_layout import relocate_path, write_result_text
from pathlib import Path
import json, tempfile, time, torch
from timelens_span_head import Prepared, ROOT, read, write
from timelens_video_cache import CachedPrepared

def same(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    else: assert a==b,(a,b)

def main():
    rows=read(relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404/samples.json'))
    reports=[]
    for model in ['qwen25','qwen3']:
        for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens']:
            row=next(r for r in rows if r['dataset']==ds)
            start=time.time(); plain=Prepared([row],model)[0]; seconds=time.time()-start
            with tempfile.TemporaryDirectory(prefix='timelens_cache_validation_') as folder:
                cached=CachedPrepared([row],model,folder,[row['video_path']])
                miss=cached[0]; hit=cached[0]
                for item in [miss,hit]:
                    same(dict(plain['inputs']),dict(item['inputs']))
                    same(plain['metadata'],item['metadata'])
                    assert plain['input_sha256']==item['input_sha256']
                assert miss['preparation']['cache'][0]['hit'] is False
                assert hit['preparation']['cache'][0]['hit'] is True
                reports.append(dict(model=model,sample=row['sample_key'],bitwise_equal=True,
                    uncached_seconds=seconds,miss=miss['preparation'],hit=hit['preparation']))
                print(json.dumps(reports[-1]),flush=True)
    write(relocate_path(ROOT/'results/operations/inference_performance/full/9404/cache_validation.json'),reports)
if __name__=='__main__': main()
