"""Real-input capped scheduler check; no GT-dependent tuning or selection."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import gc
import fps_spatial_control
from temporal_mode_capped_core import *
from temporal_mode_capped_runtime import prepare


def run():
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    torch.set_num_threads(2)
    print('CAPPED_GPU_SMOKE_LOADING',flush=True)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
    eos=model.generation_config.eos_token_id
    eos=eos if isinstance(eos,list) else [eos]
    cohort=read(COHORT)
    examples=[next(r for r in cohort if r['dataset']==d) for d in sorted({r['dataset'] for r in cohort})]
    records=[]
    for row in examples:
        print('SMOKE_INPUT',row['sample_key'],flush=True)
        cpu,detail,sample,duration=prepare(row,processor)
        gpu=cpu.to('cuda')
        backend=Backend(model,gpu,torch,processor.tokenizer,eos)
        backend.duration=duration
        greedy=backend.greedy()
        assert greedy['ids']==read(SOURCE/'rollouts'/f'{row["sample_key"]}.json')['result']['candidates'][0]['ids']
        methods=METHODS if row['dataset']=='charades-timelens' else ['temporal_mode']
        hashes={}
        for method in methods:
            search=Search(backend,sample,method,42)
            result=search.sweep(checkpoints=[16,32])
            assert len(result['draws'])<=64
            for point in result['checkpoints'].values():
                assert point['main']['rollouts']<=point['budget_per_kind']
                assert point['tail']['rollouts']<=point['budget_per_kind']
                if point['main']['tasks'] and point['main']['rollouts']>=16:
                    assert point['main']['scheduler'].get('adaptive',0)>0
            hashes[method]=digest([(d['sequence'],d['task']) for d in result['draws'] if d['kind']=='main'])
            records.append(dict(sample_key=row['sample_key'],method=method,passed=True,
                input_hashes_equal=True,greedy_ids_equal=True,main_rollouts=search.counts['main'],tail_rollouts=search.counts['tail'],
                adaptive_main=search.scheduler['main']['adaptive'],main_modes=len(search.main.groups),
                tail_isolation=True,trace_hash=hashes[method]))
            print(records[-1],flush=True)
        del backend,gpu,cpu,search
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/'validation/gpu_smoke.json',dict(passed=True,config_hash=digest(CONFIG),
        checkpoints=[16,32],cases=records,timestamp=stamp()))
    print('CAPPED_GPU_SMOKE_PASSED',flush=True)


if __name__=='__main__':
    run()
