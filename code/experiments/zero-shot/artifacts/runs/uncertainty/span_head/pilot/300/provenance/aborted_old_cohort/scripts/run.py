"""Fixed-cohort, inference-only reduced LM-head ablation; no prompt changes."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/irteam/analysis/failure_mode_timestamp/failure_cases')
UPSTREAM = Path('/home/irteam/experiments/mtla_attention/vendor/TimeLens')
sys.path.insert(0, str(UPSTREAM))
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from evaluation.utils import GroundingDataset, GROUNDER_PROMPT
from timelens.utils import extract_time


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


class RestrictedHead(torch.nn.Module):
    """Project only allowed weight rows; scatter for HF token-ID compatibility."""
    def __init__(self, head, ids):
        super().__init__()
        self.full_head = head
        self.register_buffer('ids', torch.tensor(ids, device=head.weight.device))
        self.register_buffer('selected_weight', head.weight.detach()[self.ids].clone())
        self.register_buffer('selected_bias', None if head.bias is None else head.bias.detach()[self.ids].clone())
        self.enabled = True

    def forward(self, hidden):
        if not self.enabled:
            return self.full_head(hidden)
        reduced = F.linear(hidden, self.selected_weight, self.selected_bias)
        result = reduced.new_full((*reduced.shape[:-1], self.full_head.out_features), -float('inf'))
        return result.index_copy_(-1, self.ids, reduced)


def worker_init(worker_id):
    torch.set_num_threads(1)
    import decord
    from functools import partial
    decord.VideoReader = partial(decord.VideoReader, num_threads=1)


def identity(batch):
    return batch[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['qwen25', 'qwen3'], required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    torch.set_num_threads(2)
    set_seed(42)
    name = 'Qwen2.5-VL-7B-Instruct' if a.model == 'qwen25' else 'Qwen3-VL-8B-Instruct'
    mp = '/home/irteam/backbones/models/' + name
    manifest_path = SOURCE / 'data/samples_100.json'
    manifest = json.loads(manifest_path.read_text())
    old = json.loads((SOURCE / 'reports/pilot_100/setting_case_metrics.json').read_text())
    baseline = {r['sample_id']: r for r in old if r['setting'] == a.model + '_native'}
    assert len(manifest) == len(baseline) == 200
    rows = []
    for i, row in enumerate(manifest):
        assert all(row[k] == baseline[row['sample_id']][k] for k in ['query', 'span', 'dataset', 'video_id'])
        if i % a.shards != a.shard:
            continue
        row = dict(row)
        row['video_path'] = '/data/video/MR/TimeLens/TimeLens-Bench/videos/' + row['video_path'].split('/videos/')[1]
        assert Path(row['video_path']).is_file()
        rows.append(row)
    if a.limit:
        rows = rows[:a.limit]
    out = ROOT / 'results' / a.model / f'shard{a.shard}'
    out.mkdir(parents=True, exist_ok=True)
    pp = out / 'predictions.jsonl'
    done = {json.loads(s)['sample_id'] for s in pp.read_text().splitlines()} if pp.exists() else set()
    rows = [r for r in rows if r['sample_id'] not in done]
    if not rows:
        return
    started = time.time()
    proc = AutoProcessor.from_pretrained(mp, padding_side='left', do_resize=False, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(mp, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'}, local_files_only=True).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    ids = [i for i in range(len(proc.tokenizer)) if i not in proc.tokenizer.all_special_ids and re.fullmatch(r'[0-9.,\-\[\] :\n]+', proc.tokenizer.decode([i], clean_up_tokenization_spaces=False))]
    ids = sorted(set(ids + eos))
    original_head = model.lm_head
    head = RestrictedHead(original_head, ids)
    model.lm_head = head
    # Verify selected logits against the original projection on real dtype/device.
    with torch.inference_mode():
        test = torch.randn(2, 1, original_head.in_features, device='cuda', dtype=torch.bfloat16)
        full = original_head(test)
        restricted = head(test)
        torch.testing.assert_close(restricted[..., ids], full[..., ids], rtol=0.02, atol=0.02)
        projection_error = float((restricted[..., ids] - full[..., ids]).abs().max())
    config = dict(model=name, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), baseline_source=str(SOURCE / 'reports/pilot_100/setting_case_metrics.json'), baseline_source_sha256=hashlib.sha256((SOURCE / 'reports/pilot_100/setting_case_metrics.json').read_bytes()).hexdigest(), transformers=transformers.__version__, torch=torch.__version__, attention='sdpa', dtype='bfloat16', no_finetuning=True, prompt_unchanged=True, fps=2, min_tokens=64, total_tokens=14336, max_new_tokens=512, do_sample=False, repetition_penalty=model.generation_config.repetition_penalty, seed=42, full_vocabulary=original_head.out_features, allowed_count=len(ids), allowed_tokens=[{'id': i, 'text': proc.tokenizer.decode([i])} for i in ids], projection_max_abs_error=projection_error, shard=a.shard, shards=a.shards)
    write(out / 'config.json', config)
    dataset = GroundingDataset(rows, proc, SimpleNamespace(model_path=mp, min_tokens=64, total_tokens=14336, fps=2))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=a.workers, collate_fn=identity, prefetch_factor=2, pin_memory=True, multiprocessing_context='spawn', worker_init_fn=worker_init)
    validated = set()
    for idx, item in enumerate(loader):
        row = item['anno']
        base = baseline[row['sample_id']]
        inputs = item['inputs'].to('cuda', non_blocking=True)
        n = inputs.input_ids.shape[1]
        grid = inputs.video_grid_thw.tolist()
        assert n == base['input_tokens'], (row['sample_id'], n, base['input_tokens'])
        assert grid == base['video_grid_thw'], (grid, base['video_grid_thw'])
        generate_kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=512)
        with torch.inference_mode():
            if row['dataset'] not in validated:
                head.enabled = False
                orig = model.generate(**inputs, **generate_kw)
                answer = proc.batch_decode(orig[:, n:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                audit = dict(sample_id=row['sample_id'], saved_answer=base['answer'], reproduced_answer=answer, exact_match=answer == base['answer'], saved_timestamps=base['timestamps'], reproduced_timestamps=[[round(s), round(e)] for s, e in extract_time(answer)])
                with (out / 'baseline_reproduction.jsonl').open('a') as f:
                    f.write(json.dumps(audit) + '\n')
                print('BASELINE_CHECK', json.dumps(audit), flush=True)
                head.enabled = True
                validated.add(row['dataset'])
                del orig
            torch.cuda.synchronize()
            start = time.time()
            torch.cuda.reset_peak_memory_stats()
            output = model.generate(**inputs, **generate_kw)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            token_ids = output[0, n:].tolist()
            assert set(token_ids) <= set(ids)
            answer = proc.batch_decode(output[:, n:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            parsed = extract_time(answer)
            record = dict(row, model=name, model_key=a.model, prompt=GROUNDER_PROMPT.format(row['query']), answer=answer, generated_ids=token_ids, parsed_timestamps=parsed, timestamps=[[round(s), round(e)] for s, e in parsed], parse_failed=not bool(parsed), input_tokens=n, video_grid_thw=grid, output_tokens=len(token_ids), elapsed_seconds=elapsed, hit_token_limit=len(token_ids) >= 512 and token_ids[-1] not in eos, peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3, input_ids_sha256=hashlib.sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest())
        with pp.open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
        print(f'COMPLETE {idx+1}/{len(rows)} {row["sample_id"]} {elapsed:.2f}s {answer[:180]!r}', flush=True)
        del inputs, output, item
    write(out / 'complete.json', dict(count=len(rows)+len(done), wall_seconds=time.time()-started))


if __name__ == '__main__':
    main()
