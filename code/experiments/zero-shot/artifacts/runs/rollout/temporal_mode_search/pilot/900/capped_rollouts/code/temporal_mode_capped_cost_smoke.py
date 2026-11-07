"""Real-input invariance and independently reconstructed native-operation counts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import gc
import fps_spatial_control
from temporal_mode_capped_cost import *
from temporal_mode_capped_runtime import prepare


def run():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, do_resize=False)
    model = AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos,list) else [eos]
    cohort = read(COHORT)
    cases = []
    for dataset in sorted({r['dataset'] for r in cohort}):
        row = next(r for r in cohort if r['dataset']==dataset and (OUT/'responses'/f'{r["sample_key"]}.json').exists())
        prior = read(OUT/'responses'/f'{row["sample_key"]}.json')
        cpu, detail, sample, duration = prepare(row,processor)
        gpu = cpu.to('cuda')
        backend = CostBackend(model,gpu,torch,processor.tokenizer,eos)
        backend.duration = duration
        greedy = backend.greedy()
        assert greedy['ids']==prior['baseline']['ids']
        search = CostSearch(backend,sample,'temporal_mode',42)
        result = search.sweep(checkpoints=[16])
        point = result['checkpoints']['16']
        old = prior['results']['temporal_mode:42']['checkpoints']['16']
        old_draws = prior['results']['temporal_mode:42']['draws'][:old['draw_cutoff']]
        assert [(d['sequence'],d['task']) for d in result['draws']]==[(d['sequence'],d['task']) for d in old_draws], 'instrumentation changed draw stream'
        assert point['selection']['span']==old['selection']['span'], 'instrumentation changed selection'
        data = dict(baseline=greedy,sequences=backend.sequences,choices=backend.choices)
        work = standalone_work(data,result,point,backend.n,backend.dialog)
        measured = point['cost']
        for phase in ('video_prefill','greedy_decode','main','tail','judge'):
            actual = measured['baseline'][phase] if phase in ('video_prefill','greedy_decode') else measured[phase]
            for field in ('decoder_inputs','forwards','attention_pairs'):
                assert work['phases'][phase][field]==actual[field], (row['sample_key'],phase,field,work['phases'][phase],actual)
        record = dict(sample_key=row['sample_key'],dataset=dataset,input_hashes_equal=True,greedy_ids_equal=True,
            draw_stream_equal=True,selection_equal=True,trace_counts_equal=True,
            independent_work=work,observed=measured)
        cases.append(record)
        print('COST_SMOKE_CASE',row['sample_key'],measured['sweep']['forwards'],measured['sweep']['cuda_forward_seconds'],flush=True)
        del backend,gpu,cpu,search
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/'validation/cost_gpu_smoke.json',dict(passed=True,config_hash=digest(CONFIG),
        generation_policy_unchanged=True,cases=cases,timestamp=stamp()))
    print('COST_GPU_SMOKE_PASSED',flush=True)


if __name__ == '__main__':
    run()
