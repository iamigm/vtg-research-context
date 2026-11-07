"""Paired native/prefix-cache equivalence and timing on three original inputs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import gc
import fps_spatial_control
from temporal_mode_prefix_cache import *
from temporal_mode_capped_runtime import prepare


def run():
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    torch.set_num_threads(2)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
    eos=model.generation_config.eos_token_id
    eos=eos if isinstance(eos,list) else [eos]
    cohort=read(COHORT)
    cases=[]
    for dataset in sorted({r['dataset'] for r in cohort}):
        row=next(r for r in cohort if r['dataset']==dataset and (OUT/'responses'/f'{r["sample_key"]}.json').exists())
        cpu,detail,sample,duration=prepare(row,processor)
        pair=[]
        for backend_class,search_class in ((CostBackend,CostSearch),(PrefixBackend,PrefixSearch)):
            gpu=cpu.to('cuda')
            backend=backend_class(model,gpu,torch,processor.tokenizer,eos)
            backend.duration=duration
            baseline=backend.greedy()
            original=read(SOURCE/'rollouts'/f'{row["sample_key"]}.json')['result']['candidates'][0]['ids']
            assert baseline['ids']==original
            search=search_class(backend,sample,'probability',42)
            result=search.sweep(checkpoints=[16,32])
            point=result['checkpoints']['32']
            cost=copy.deepcopy(point['cost'])
            trace=None
            if backend_class is PrefixBackend:
                first=result['checkpoints']['16']
                trace=prefix_standalone_work(dict(baseline=baseline,sequences=backend.sequences,choices=backend.choices),
                    result,first,backend.n,backend.dialog)
                for phase in ('video_prefill','greedy_decode','main','tail','judge'):
                    actual=first['cost']['baseline'][phase] if phase in ('video_prefill','greedy_decode') else first['cost'][phase]
                    for field in ('forwards','decoder_inputs','attention_pairs'):
                        assert trace['phases'][phase][field]==actual[field],(dataset,phase,field,trace['phases'][phase],actual)
            for value in result['checkpoints'].values():
                value.pop('cost')
            pair.append(dict(result=copy.deepcopy(result),distributions=copy.deepcopy(backend.distributions),
                choices=copy.deepcopy(backend.choices),cost=cost,trace=trace))
            del backend,gpu,search
            gc.collect();torch.cuda.empty_cache()
        assert pair[0]['result']==pair[1]['result'],(dataset,'generation or selection changed')
        assert pair[0]['distributions']==pair[1]['distributions'],(dataset,'native probabilities changed')
        assert pair[0]['choices']==pair[1]['choices'],(dataset,'judge probabilities changed')
        native,fast=pair[0]['cost'],pair[1]['cost']
        record=dict(sample_key=row['sample_key'],dataset=dataset,passed=True,input_hashes_equal=True,
            greedy_ids_equal=True,draws_and_modes_equal=True,full_distribution_records_equal=True,
            judge_records_equal=True,trace_counts_equal=True,
            native=native,optimized=fast,
            forward_reduction=1-fast['sweep']['forwards']/max(1,native['sweep']['forwards']),
            wall_speedup=native['sweep_elapsed_seconds']/fast['sweep_elapsed_seconds'])
        cases.append(record)
        print('PREFIX_CACHE_CASE',row['sample_key'],'forwards',native['sweep']['forwards'],'->',fast['sweep']['forwards'],
            'wall speedup',record['wall_speedup'],flush=True)
    write(OUT/'validation/prefix_cache_gpu.json',dict(passed=True,config_hash=digest(CONFIG),
        backend=BACKEND,prefix_limit=PREFIX_LIMIT,cases=cases,timestamp=stamp()))
    print('PREFIX_CACHE_GPU_PASSED',flush=True)


def validate():
    manifest=read(OUT/'metadata/prefix_cache_manifest.json')
    for entry in manifest['files']:
        assert sha(entry['snapshot'])==entry['sha256']
    assert read(OUT/'validation/prefix_cache_gpu.json')['passed']
    observed=0
    checkpoints=0
    for row in read(COHORT):
        data=read(OUT/'responses'/f'{row["sample_key"]}.json')
        assert is_complete(data)
        for route in data['results'].values():
            for point in route['checkpoints'].values():
                cost=point.get('cost',{})
                if cost.get('execution_backend')!=BACKEND:
                    continue
                cache=cost['prefix_cache']
                assert 0<=cache['entries']<=PREFIX_LIMIT
                assert cache['sweep']['prefix_cache_cuda_seconds']>=0
                assert cost['baseline']['execution_backend']==BACKEND
                checkpoints+=1
        observed+=data.get('cost_accounting',{}).get('execution_backend')==BACKEND
    assert observed>0 and checkpoints>0
    write(OUT/'validation/prefix_cache_final.json',dict(passed=True,queries_using_backend=observed,
        optimized_checkpoints=checkpoints,prefix_limit=PREFIX_LIMIT,generation_config_unchanged=True,timestamp=stamp()))


if __name__=='__main__':
    import sys
    validate() if '--validate' in sys.argv else run()
