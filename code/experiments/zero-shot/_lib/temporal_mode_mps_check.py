"""Check MPS native probabilities and decisions against completed non-MPS traces."""
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
        prior=read(OUT/'responses'/f'{row["sample_key"]}.json')
        cpu,detail,sample,duration=prepare(row,processor)
        gpu=cpu.to('cuda')
        backend=PrefixBackend(model,gpu,torch,processor.tokenizer,eos)
        backend.duration=duration
        baseline=backend.greedy()
        assert baseline['ids']==prior['baseline']['ids']
        search=PrefixSearch(backend,sample,'probability',42)
        result=search.sweep(checkpoints=[16,32])
        old=prior['results']['probability:42']
        assert result['draws']==old['draws'][:old['checkpoints']['32']['draw_cutoff']],(dataset,'draws changed')
        for k,point in result['checkpoints'].items():
            without_cost={key:value for key,value in point.items() if key!='cost'}
            reference={key:value for key,value in old['checkpoints'][k].items() if key!='cost'}
            assert without_cost==reference,(dataset,k,'mode or judgment changed')
        distributions={tuple(d['prefix']):d for d in prior['distributions']}
        assert all(value==distributions[prefix] for prefix,value in backend.distributions.items()),(dataset,'raw token distribution changed')
        assert all(value==prior['choices'][key] for key,value in backend.choices.items()),(dataset,'judge record changed')
        cases.append(dict(sample_key=row['sample_key'],dataset=dataset,passed=True,greedy_ids_equal=True,
            raw_distributions_equal=True,draws_modes_judgment_equal=True,checkpoints=[16,32]))
        print('MPS_REAL_INPUT_PASSED',row['sample_key'],flush=True)
        del backend,gpu,cpu,search
        gc.collect();torch.cuda.empty_cache()
    write(OUT/'validation/mps_real_inputs.json',dict(passed=True,cases=cases,config_hash=digest(CONFIG),timestamp=stamp()))
    print('MPS_REAL_INPUTS_PASSED',flush=True)


if __name__=='__main__':
    run()
