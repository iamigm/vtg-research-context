"""Actual three-dataset tensor/greedy/rollback/sampling smoke before detaching."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import copy
import gc
import time
import fps_spatial_control
from temporal_mode_core import *
from temporal_mode_runtime import prepare


def run():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(2)
    cohort = read(COHORT)
    samples = [next(r for r in cohort if r['dataset']==d) for d in sorted({r['dataset'] for r in cohort})]
    print(json.dumps(dict(stage='smoke_loading_model', timestamp=stamp())), flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, do_resize=False)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map={'': 'cuda:0'}).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    original = copy.deepcopy(CONFIG)
    audits = []
    try:
        CONFIG.update(main_budget=1024, tail_budget=1024, judge_budget=8192, root_probes=4,
                      main_min_continuations=2, tail_min_continuations=2)
        for index, row in enumerate(samples):
            print(json.dumps(dict(stage='smoke_input', key=row['sample_key'], timestamp=stamp())), flush=True)
            cpu, detail, sample, duration = prepare(row, processor)
            gpu = cpu.to('cuda')
            backend = Backend(model, gpu, torch, processor.tokenizer, eos)
            backend.duration = duration
            baseline = backend.greedy()
            expected = read(SOURCE/'rollouts'/f'{row["sample_key"]}.json')['result']['candidates'][0]['ids']
            assert baseline['ids'] == expected, (row['sample_key'], 'greedy mismatch')
            rollback = None
            judge_batch = None
            if index == 0:
                prefix = expected[:min(5,len(expected)-1)]
                a = torch.softmax(backend.get(prefix), -1)
                reference = Backend(model, gpu, torch, processor.tokenizer, eos)
                b = torch.softmax(reference.get(prefix), -1)
                rollback = float((a-b).abs().sum())
                assert rollback < .001 and int(a.argmax()) == int(b.argmax())
                del reference
                gc.collect()
                torch.cuda.empty_cache()
                probe = backend.dialog('Use the video. Answer one letter only: A = a person is visible; B = no person is visible; C = uncertain.')
                single = torch.softmax(backend.get(probe), -1)[backend.choice_ids]
                batched = torch.softmax(backend.get_judge_prompt(probe), -1)[backend.choice_ids]
                sn, bn = single/single.sum(), batched/batched.sum()
                judge_batch = dict(option_l1=float((sn-bn).abs().sum()), label_equal=int(sn.argmax())==int(bn.argmax()))
                assert judge_batch['label_equal'] and judge_batch['option_l1'] < .02, judge_batch
            search = Search(backend, sample, 'temporal_mode', 42)
            main = search.run_main()
            main_modes = search.main.summary()
            before = digest(main_modes)
            selection = judge(backend, sample, main_modes)
            tail = search.run_tail()
            assert digest(search.main.summary()) == before
            assert selection['span'] is None or any(selection['span'] in g['intervals'] for g in main_modes)
            audit = dict(key=row['sample_key'], dataset=row['dataset'], passed=True, tensor_hashes_equal=True,
                greedy_ids_equal=True, rollback_l1=rollback, judge_batch=judge_batch, main_tokens=main['spent'], tail_tokens=tail['spent'],
                main_modes=len(main_modes), tail_modes=len(search.tail.groups), selection_status=selection['status'],
                sampled_sequences=len(backend.sequences), tail_isolation_verified=True)
            audits.append(audit)
            print(json.dumps(audit), flush=True)
            del backend, gpu, cpu, search
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        CONFIG.clear()
        CONFIG.update(original)
    write(OUT/'validation/gpu_smoke.json', dict(passed=True, datasets=audits,
        full_config_hash=digest(CONFIG), reduced_probe_budgets=True, timestamp=stamp()))
    print('GPU_SMOKE_PASSED', flush=True)


if __name__ == '__main__':
    run()
