"""Reproduce the targeted pixel-sum and backward-repeatability audit.

Reconstructed from the two temporary scripts executed on 2026-11-12. The
original execution-script hashes remain in pixel_sum_audit.json. This durable
runner performs the same comparisons: one backward without message hooks,
then a fresh model/forward with message hooks and two same-graph backwards.
It does not run screening, other backward targets, or change source records.

Run only while the experiment GPU is idle, using zero-shot/.venv/bin/python.
The default output overwrites the single canonical audit with the new run.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import time
from pathlib import Path

import greedy_attribution_runtime as runtime
from greedy_attribution_core import (
    COHORT, CONFIG, GROUPS, MODEL, ROOT, SOURCE, context_map, read, sha, stamp, write,
)


def execute_phase(query, row, target, stored, base, *, message_hooks, repetitions):
    import gc
    import numpy as np
    import torch
    from greedy_attribution_hooks import AttributionHooks
    from greedy_attribution_messages import AttentionMessages
    from probability_budget_runtime import prepare

    started = time.monotonic()
    print(json.dumps({'stage': 'loading_model', 'message_hooks': message_hooks}), flush=True)
    model, processor, load_seconds = runtime.load()
    cpu, info = prepare(row, processor)
    source = read(SOURCE / 'rollouts' / f"{query['sample_key']}.json")
    assert info['hashes'] == query['input']['hashes'] == source['input_hashes']
    gpu = cpu.to('cuda')
    ids = query['greedy']['ids']
    prompt = int(gpu.input_ids.shape[1])
    length = prompt + len(ids) - 1
    assert (prompt, length) == (query['input']['prompt_length'], query['input']['total_length'])
    gpu['input_ids'] = torch.cat((gpu.input_ids, torch.tensor([ids[:-1]], device='cuda')), dim=1)
    gpu['attention_mask'] = torch.ones_like(gpu.input_ids)
    gpu['pixel_values_videos'] = gpu['pixel_values_videos'].detach().requires_grad_(True)
    context = context_map(gpu.input_ids[0].tolist(), processor.tokenizer, prompt, row,
                          query['input']['grid'], info['timestamps'])
    predictors = list(range(prompt - 1, length))
    groups = {group: [i for i, value in enumerate(context['groups']) if value == group] for group in GROUPS}
    model._attribution_checkpoint = prompt > CONFIG['checkpoint_input_tokens']
    assert model._attribution_checkpoint == query['cost']['checkpointing']
    torch.cuda.reset_peak_memory_stats()
    messages = AttentionMessages(model, predictors) if message_hooks else None
    results = []
    with AttributionHooks(model, prompt, length, predictors, groups) as capture, (
        messages if messages is not None else contextlib.nullcontext()
    ):
        print(json.dumps({'stage': 'gradient_forward', 'message_hooks': message_hooks}), flush=True)
        logits = runtime.sequential_forward(model, gpu, prompt, len(ids), capture=capture, messages=messages).float()
        objective = logits[target['step'], ids[target['step']]] - logits[target['step'], target['alternative']]
        assert float(objective.detach()) == target['score']
        with np.load(base / query['arrays']['file']) as arrays:
            selected = logits.detach().gather(1, torch.as_tensor(arrays['token_ids'], device='cuda')).cpu().numpy()
            logit_error = float(np.max(np.abs(selected - arrays['logits'])))
        previous_gradient = None
        old_patch = np.asarray(stored['pixel_frames']['activation_dot_gradient'], dtype=np.float64)
        old_norm = np.asarray(stored['pixel_frames']['gradient_norm'], dtype=np.float64)
        for repetition in range(repetitions):
            capture.begin_target(target['id'])
            if messages is not None:
                messages.begin_target(target['id'])
            print(json.dumps({'stage': 'backward', 'message_hooks': message_hooks, 'repetition': repetition + 1}), flush=True)
            gradients = torch.autograd.grad(
                objective, tuple([gpu['pixel_values_videos']] + capture.embedding_leaves),
                retain_graph=repetition < repetitions - 1, allow_unused=True,
            )
            gradient = gradients[0]
            assert gradient is not None
            x, g = gpu['pixel_values_videos'].detach().float(), gradient.float()
            product = g * x
            maps = runtime.pixel_maps(gpu['pixel_values_videos'], gradient, query['input']['grid'])
            patch = np.asarray(maps['activation_dot_gradient'], dtype=np.float64)
            norms = np.asarray(maps['gradient_norm'], dtype=np.float64)
            global32 = float(product.sum())
            patch64 = float(patch.sum(dtype=np.float64))
            rounded64 = float(product.double().sum())
            exact64 = float((g.double() * x.double()).sum())
            result = dict(
                repetition=repetition + 1, message_hooks=message_hooks, objective=float(objective.detach()),
                global_fp32=global32, patch_fp32_host_fp64=patch64,
                rounded_fp32_products_fp64_sum=rounded64, fp64_product_sum=exact64,
                actual_fp32_product_absolute_sum=float(product.double().abs().sum()),
                fp64_products_absolute_sum=float((g.double() * x.double()).abs().sum()),
                gradient_l2=float(g.norm()), normalized_input_min=float(x.min()), normalized_input_max=float(x.max()),
                products=product.numel(), global_signed_bit_equal_to_stored=global32 == target['pixel_signed_sum'],
                patch_signed_bit_equal_to_stored=bool(np.array_equal(patch, old_patch)),
                patch_norm_bit_equal_to_stored=bool(np.array_equal(norms, old_norm)),
                patch_signed_max_difference_from_stored=float(np.max(np.abs(patch - old_patch))),
                patch_norm_max_difference_from_stored=float(np.max(np.abs(norms - old_norm))),
                global32_minus_patch32_host64=global32 - patch64,
                global32_minus_rounded_product_fp64=global32 - rounded64,
                patch32_host64_minus_rounded_product_fp64=patch64 - rounded64,
                fp32_product_rounding_sum=rounded64 - exact64,
            )
            if previous_gradient is not None:
                delta = g - previous_gradient
                result.update(
                    same_graph_backward_repeat_gradient_max_absolute_difference=float(delta.abs().max()),
                    same_graph_backward_repeat_gradient_relative_l2_difference=float(delta.norm() / previous_gradient.norm()),
                    same_graph_backward_repeat_gradient_bit_equal=bool(torch.equal(g, previous_gradient)),
                )
                del delta
            if repetition < repetitions - 1:
                previous_gradient = g.detach().clone()
            results.append(result)
            print(json.dumps(result), flush=True)
        phase = dict(message_hooks=message_hooks, measurements=results,
                     top30_logits_max_absolute_difference=logit_error,
                     checkpointing=model._attribution_checkpoint,
                     cost=dict(model_load_seconds=load_seconds, total_seconds=time.monotonic() - started,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    # Release this graph/model before the independent next phase.
    del model, processor, gpu, cpu, logits, objective, gradients, gradient, g, x, product, capture, messages, previous_gradient
    gc.collect()
    torch.cuda.empty_cache()
    return phase


def run(sample_key, target_id, base, output):
    import torch

    query_path = base / 'queries' / f'{sample_key}.json'
    query = read(query_path)
    target = next(item for item in query['targets'] if item['id'] == target_id)
    assert target['kind'] == 'logit_margin'
    payload_path = base / target['data_file']
    original_hashes = {str(path): sha(path) for path in (query_path, payload_path, base / 'metadata/config.json')}
    with gzip.open(payload_path, 'rt') as stream:
        stored = json.load(stream)
    row = next(item for item in read(COHORT) if item['sample_key'] == sample_key)
    phases = [execute_phase(query, row, target, stored, base, message_hooks=False, repetitions=1),
              execute_phase(query, row, target, stored, base, message_hooks=True, repetitions=2)]
    unchanged = all(sha(path) == value for path, value in original_hashes.items())
    assert unchanged
    measurements = [measurement for phase in phases for measurement in phase['measurements']]
    reproduced = all(item['global_signed_bit_equal_to_stored'] and item['patch_signed_bit_equal_to_stored']
                     and item['patch_norm_bit_equal_to_stored'] for item in measurements)
    repeated = phases[1]['measurements'][1]['same_graph_backward_repeat_gradient_bit_equal']
    passed = reproduced and repeated and all(phase['top30_logits_max_absolute_difference'] == 0 for phase in phases)
    record = dict(
        passed=passed, audit_completed=True, timestamp=stamp(), sample_key=sample_key, target_id=target_id,
        scope='One margin target; no screening or other targets; two model loads, one forward per phase',
        stored=dict(score=target['score'], pixel_signed_sum=target['pixel_signed_sum'],
                    pixel_gradient_l2=target['pixel_gradient_l2']),
        phases=phases, original_raw_and_config_unchanged=unchanged,
        provenance=dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                        allow_tf32=torch.backends.cuda.matmul.allow_tf32, model=str(MODEL),
                        model_config_sha256=sha(MODEL / 'config.json'), original_files_sha256=original_hashes,
                        runner=str(Path(__file__).resolve()), runner_sha256=sha(__file__),
                        source_files_sha256={name: sha(ROOT / '_lib' / name) for name in (
                            'greedy_attribution_runtime.py', 'greedy_attribution_hooks.py',
                            'greedy_attribution_messages.py', 'greedy_attribution_cache.py',
                            'probability_budget_runtime.py', 'probability_budget_inputs.py')}),
        interpretation='Stored-gradient reproduction, same-graph repeated-backward variation, and within-run scalar reduction error are separate checks. Between-run gradient changes are not explained by scalar sum roundoff.',
    )
    write(output, record)
    print(json.dumps({'audit_completed': True, 'passed': passed, 'output': str(output)}), flush=True)
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', default='activitynet-timelens_00261')
    parser.add_argument('--target', default='margin_12_20')
    parser.add_argument('--store', type=Path, default=ROOT / 'artifacts/runs/rollout/greedy_attribution/pilot/300')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.sample, args.target, args.store, args.output or args.store / 'validation/pixel_sum_audit.json')
