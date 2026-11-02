"""TimeLens zero-shot input-order comparison with live answer-to-video attention.

The upstream dataset, prompt, processor and scoring functions are reused. The
only experimental input change is reversing the two user content blocks.
FlashAttention outputs are returned untouched; attention probabilities are
reconstructed separately from the actual post-RoPE, post-cache Q/K tensors.
"""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import argparse
import copy
import csv
import fcntl
import hashlib
import html
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = relocate_path(Path(__file__)).resolve().parent.parent
UPSTREAM = relocate_path(ROOT / 'vendor/TimeLens')
from timelens_paths import QWEN3_ZERO
OUT = relocate_path(ROOT / 'results/timelens_order')
from timelens_measurements import load_answer, save_answer
DATA = relocate_path(Path('/data/video/MR/TimeLens/TimeLens-Bench'))
MODEL = relocate_path(Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct'))
ORDERS = ['video_query', 'query_video']
DATASETS = ['charades-timelens', 'activitynet-timelens', 'qvhighlights-timelens']
os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, str(relocate_path(ROOT / 'runtime/timelens_deps')))
sys.path.insert(0, str(UPSTREAM))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    write_result_text(tmp, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    from timelens.dataset.timelens_data import DATASET_DICT
    samples = []
    for name in DATASETS:
        cls = DATASET_DICT[name]
        cls.ANNO_PATH_TEST = str(relocate_path(DATA / relocate_path(Path(cls.ANNO_PATH_TEST)).name))
        cls.VIDEO_ROOT = str(relocate_path(DATA / 'videos' / name.removesuffix('-timelens')))
        for index, anno in enumerate(cls.load_annos()):
            assert relocate_path(Path(anno['video_path'])).is_file(), anno['video_path']
            samples.append(dict(anno, dataset=name, sample_key=f'{name}_{index+1:05d}'))
    assert len(samples) == 9404
    path = relocate_path(QWEN3_ZERO / 'samples.json')
    if path.exists():
        assert read(path) == samples, 'TimeLens manifest changed'
    else:
        write(path, samples)
    return samples


from timelens_inputs import OrderProcessor, prepare_pair


class PreparedPairs:
    """CPU-only map dataset; each spawned worker owns its processor/decoder."""
    def __init__(self, samples):
        self.samples = samples
        self.processor = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        started = time.monotonic()
        sample = self.samples[index]
        try:
            if self.processor is None:
                import torch
                import decord
                from functools import partial
                from transformers import AutoProcessor
                torch.set_num_threads(2)
                # Same decoder and frames; avoid FFmpeg's automatic 16-thread
                # pool oversubscribing the container's CPU quota.
                decord.VideoReader = partial(decord.VideoReader, num_threads=2)
                self.processor = AutoProcessor.from_pretrained(
                    MODEL, padding_side='left', do_resize=False,
                    trust_remote_code=True, local_files_only=True)
            pair, info = prepare_pair(sample, self.processor)
            return dict(sample=sample, pair=pair, info=info,
                        prepare_seconds=time.monotonic()-started)
        except Exception:
            return dict(sample=sample, error=traceback.format_exc())


def keep_prepared_item(item):
    # Preserve BatchFeature objects and metadata without default collation.
    return item


def prepare_worker_init(worker_id):
    # The container has a 12-core quota despite exposing 128 host CPUs.
    # Bound decoder CPU use even when FFmpeg creates extra threads.
    cpus=sorted(os.sched_getaffinity(0))
    start=(worker_id*2)%len(cpus)
    os.sched_setaffinity(0,cpus[start:start+2])


def prepared_loader(samples, workers=4):
    from torch.utils.data import DataLoader
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    return DataLoader(PreparedPairs(samples), batch_size=None, shuffle=False,
                      num_workers=workers, prefetch_factor=2, pin_memory=True,
                      multiprocessing_context='spawn', collate_fn=keep_prepared_item,
                      worker_init_fn=prepare_worker_init)


class AnswerCapture:
    """Observe last-row Q/K while preserving the registered FA2 implementation."""
    def __init__(self, model):
        import transformers.models.qwen3_vl.modeling_qwen3_vl as module
        self.registry = module.ALL_ATTENTION_FUNCTIONS
        self.original = self.registry['flash_attention_2']
        self.layers = {id(layer.self_attn): i for i, layer in enumerate(model.model.language_model.layers)}
        self.enabled = False
        self.registry.register('flash_attention_2', self.forward)

    def configure(self, length, positions, grid):
        import torch
        self.length = length
        self.video_positions = torch.tensor(positions, device='cuda', dtype=torch.long)
        self.t, self.hw = grid[0], grid[1]*grid[2]//4
        self.rows = [[] for _ in self.layers]
        self.masses = [[] for _ in self.layers]
        self.positions = [[] for _ in self.layers]
        self.enabled = True

    def forward(self, module, query, key, value, attention_mask, **kwargs):
        import torch
        result = self.original(module, query, key, value, attention_mask, **kwargs)
        if not self.enabled or id(module) not in self.layers:
            return result
        layer = self.layers[id(module)]
        assert query.shape[0] == 1
        assert query.shape[-2] in (1, self.length)
        assert attention_mask is None or bool(attention_mask.all())
        # Last query row can see every cached key, in both prefill and decode.
        # FP32 measurement avoids BF16 probability rounding; logits retain the
        # BF16 Q/K values produced by the actual FA2 model, promoted to FP32.
        groups = module.num_key_value_groups
        temporal = torch.zeros(self.t, device=query.device, dtype=torch.float32)
        masses = torch.zeros(3, device=query.device, dtype=torch.float32)
        for start in range(0, query.shape[1], groups):
            q = query[:, start:start+groups, -1:, :].float()
            k = key[:, start//groups:start//groups+1].float()
            weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * kwargs['scaling'], -1)[0, :, 0]
            video = weights.index_select(-1, self.video_positions)
            temporal += video.reshape(groups, self.t, self.hw).sum((0, 2))
            masses += torch.stack([video.sum(), weights[:, self.length:].sum(), weights.sum()])
        temporal /= query.shape[1]; masses /= query.shape[1]
        self.rows[layer].append(temporal)
        self.masses[layer].append(masses)
        self.positions[layer].append(key.shape[-2]-1)
        return result

    def arrays(self, n):
        import torch
        import numpy as np
        expected = list(range(self.length-1, self.length+n))
        assert all(p == expected for p in self.positions), (n, self.positions[0])
        temporal = torch.stack([torch.stack(rows) for rows in self.rows]).cpu().numpy()
        masses = torch.stack([torch.stack(rows) for rows in self.masses]).cpu().numpy()
        assert np.isfinite(temporal).all() and (temporal >= 0).all()
        assert np.allclose(temporal.sum(-1), masses[..., 0], atol=2e-6)
        assert np.allclose(masses[..., 2], 1, atol=2e-6)
        assert temporal.shape[1] == n+1
        self.enabled = False
        self.rows = self.masses = None
        return dict(consumed_temporal=temporal[:, 1:], predictive_temporal=temporal[:, :-1],
                    consumed_key_mass=masses[:, 1:], predictive_key_mass=masses[:, :-1])


def measure(raw, times, duration, gt):
    import numpy as np
    raw = np.asarray(raw, dtype=float)
    if not len(raw) or raw.sum() <= 0:
        return None
    edges = np.clip(np.r_[0., (times[:-1]+times[1:])/2, duration], 0, duration)
    assert np.all(np.diff(edges) > 0)
    lo, hi = max(0, gt[0]), min(duration, gt[1])
    overlap = np.maximum(0, np.minimum(edges[1:], hi)-np.maximum(edges[:-1], lo))/np.diff(edges)
    p = raw/raw.sum()
    peak = float(times[raw.argmax()])
    return dict(video_mass=float(raw.sum()), gt_mass=float(p@overlap),
                raw_gt_mass=float(raw@overlap), peak_seconds=peak,
                peak_in_gt=bool(lo <= peak <= hi), early_peak=bool(peak < duration*.1),
                centroid_seconds=float(p@times),
                entropy=float(-(p[p>0]*np.log(p[p>0])).sum()/np.log(len(p))) if len(p)>1 else 0.)


def generate_one(model, processor, capture, inputs, info, sample, order, baseline_check=False):
    import torch
    import numpy as np
    from timelens.utils import extract_time, iou
    folder = relocate_path(OUT / sample['dataset'] / sample['sample_key'] / order)
    folder.mkdir(parents=True, exist_ok=True)
    inputs = inputs.to('cuda', non_blocking=True)
    length = inputs.input_ids.shape[-1]
    kwargs = dict(do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=512)
    baseline = None
    if baseline_check:
        capture.enabled = False
        with torch.inference_mode():
            baseline = model.generate(**inputs, **kwargs).cpu()
    capture.configure(length, info['video_positions'][order], info['video_grid_thw'])
    start = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs, return_dict_in_generate=True)
    if baseline is not None:
        assert torch.equal(output.sequences.cpu(), baseline), 'Instrumentation changed generated tokens'
    generated = output.sequences[0, length:].tolist()
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    stopped = bool(generated and generated[-1] in eos)
    ids = generated[:-1] if stopped else generated
    if not stopped and ids:
        pos = length+len(ids)-1
        with torch.inference_mode():
            model(input_ids=output.sequences[:, -1:], past_key_values=output.past_key_values,
                  attention_mask=torch.ones(1, pos+1, device='cuda', dtype=torch.long),
                  cache_position=torch.tensor([pos], device='cuda'), use_cache=True, logits_to_keep=1)
    arrays = capture.arrays(len(ids))
    times = np.array(info['timestamps'])
    text = processor.batch_decode(output.sequences[:, length:], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
    spans = extract_time(text) or [[sample['duration']+10, sample['duration']+20]]
    spans = [[round(s), round(e)] for s, e in spans]
    gt = sample['span'][0]
    score = iou(gt, spans[0])
    metrics = {}
    for scope in ['consumed', 'predictive']:
        metrics[scope] = measure(arrays[scope+'_temporal'][8:22].mean((0, 1)), times, info['duration_video'], gt) if ids else None
    prediction = dict(sample_key=sample['sample_key'], dataset=sample['dataset'], order=order,
        video_name=relocate_path(Path(sample['video_path'])).name, query=sample['query'], gt=gt,
        duration=sample['duration'], answer=text, token_ids=ids,
        tokens=[processor.tokenizer.decode([i]) for i in ids], generated_ids=generated,
        stopped_on_eos=stopped, truncated=not stopped, timestamps=spans, iou=score,
        attention=metrics, baseline_tokens_identical=True if baseline_check else None,
        elapsed_seconds=time.time()-start)
    np.savez_compressed(relocate_path(folder/'attention.npz'), **arrays, sampled_timestamps=times)
    save_answer(folder,prediction,processor.tokenizer.decode(inputs.input_ids[0],skip_special_tokens=False))
    write(relocate_path(folder/'validation.json'), dict(passed=True, layers=len(capture.layers), answer_tokens=len(ids),
        consumed_rows=len(ids), predictive_rows=len(ids), key_mass_sum_verified=True,
        predictive_consumed_shift_verified=bool(np.array_equal(arrays['predictive_temporal'][:, 1:], arrays['consumed_temporal'][:, :-1])),
        generation_backend='flash_attention_2', measurement='FP32 softmax of live post-RoPE post-cache Q/K, all causal keys',
        baseline_tokens_identical=prediction['baseline_tokens_identical'], attention_sha256=sha(relocate_path(folder/'attention.npz'))))
    del output, inputs, arrays
    torch.cuda.empty_cache()
    return prediction


def compare_pair(sample):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=relocate_path(OUT/sample['dataset']/sample['sample_key'])
    curves=[];answers=[]
    for order in ORDERS:
        answers.append(load_answer(relocate_path(folder/order)))
        with np.load(relocate_path(folder/order/'attention.npz')) as z:
            times=z['sampled_timestamps'].copy()
            curve=z['consumed_temporal'][8:22].mean((0,1)) if len(answers[-1]['token_ids']) else np.zeros(len(times))
            curves.append(curve.astype(float))
    probabilities=[c/c.sum() if c.sum()>0 else np.zeros_like(c) for c in curves]
    a,b=probabilities
    metrics=dict(sample_key=sample['sample_key'],dataset=sample['dataset'])
    if a.sum()>0 and b.sum()>0:
        middle=(a+b)/2
        metrics.update(attention_tv=float(np.abs(a-b).sum()/2),
            attention_js_bits=float((np.sum(a[a>0]*np.log2(a[a>0]/middle[a>0]))+np.sum(b[b>0]*np.log2(b[b>0]/middle[b>0])))/2),
            gt_mass_gain_query_first=answers[1]['attention']['consumed']['gt_mass']-answers[0]['attention']['consumed']['gt_mass'])
    write(relocate_path(folder/'comparison.json'),metrics)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for i,order in enumerate(ORDERS):
        axes[0].plot(times,curves[i],label=order)
        axes[1].plot(times,probabilities[i],label=order)
    for ax in axes:
        ax.axvspan(*sample['span'][0],color='green',alpha=.14,label='GT');ax.legend(fontsize=8)
    axes[0].set_ylabel('Raw answer → video mass');axes[1].set_ylabel('Video-conditional probability')
    axes[1].set_xlabel('Video time (seconds)')
    fig.suptitle(sample['sample_key']+' · consumed answer rows · L8–21',fontsize=12)
    fig.savefig(relocate_path(folder/'comparison.png'),dpi=120);plt.close(fig)
    return metrics


def summarize(samples):
    import numpy as np
    rows = []; summaries = {}; paired = []
    for sample in samples:
        by_order = {}
        for order in ORDERS:
            folder = relocate_path(OUT/sample['dataset']/sample['sample_key']/order)
            if not (relocate_path(folder/'validation.json')).exists():
                continue
            a = load_answer(folder); by_order[order] = a
            m = a['attention']['consumed'] or {}
            rows.append(dict(dataset=sample['dataset'], sample_key=sample['sample_key'], order=order,
                             answer_tokens=len(a['token_ids']), truncated=a['truncated'], **m))
        if len(by_order) == 2:
            v, q = (by_order[o] for o in ORDERS)
            pair = dict(dataset=sample['dataset'], sample_key=sample['sample_key'])
            if v['attention']['consumed'] and q['attention']['consumed']:
                pair['gt_mass_gain_query_first'] = q['attention']['consumed']['gt_mass']-v['attention']['consumed']['gt_mass']
            paired.append(pair)
            comparison=relocate_path(OUT/sample['dataset']/sample['sample_key']/'comparison.json')
            if comparison.exists():pair.update(read(comparison))
    for dataset in DATASETS:
        summaries[dataset] = {}
        for order in ORDERS:
            group = [r for r in rows if r['dataset']==dataset and r['order']==order]
            if not group:continue
            summaries[dataset][order] = dict(completed=len(group),
                mean_gt_mass=float(np.mean([r['gt_mass'] for r in group if 'gt_mass' in r])) if any('gt_mass' in r for r in group) else None,
                truncated=sum(r['truncated'] for r in group))
        pairs=[r for r in paired if r['dataset']==dataset]
        summaries[dataset]['paired'] = dict(completed=len(pairs),
            mean_attention_tv=float(np.mean([r['attention_tv'] for r in pairs if 'attention_tv' in r])) if any('attention_tv' in r for r in pairs) else None)
    write(relocate_path(OUT/'summary.json'), dict(expected_pairs=len(samples), completed_pairs=len(paired), completed_runs=len(rows), datasets=summaries))
    for name, data in [('sample_metrics',rows),('paired_metrics',paired)]:
        if data:
            with (relocate_path(OUT/(name+'.csv'))).open('w') as stream:
                fields=list(dict.fromkeys(k for row in data for k in row))
                writer=csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(data)
    body=[]
    for dataset, group in summaries.items():
        for order in ORDERS:
            x=group.get(order)
            if x:body.append(f"<tr><td>{dataset}</td><td>{order}</td><td>{x['completed']}</td><td>{x['mean_gt_mass']}</td></tr>")
    page='<!doctype html><meta charset="utf-8"><title>TimeLens input order</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto}td,th{padding:10px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>TimeLens: answer → video attention</h1>'
    page+=f'<p>완료 {len(paired):,} / {len(samples):,} pairs · {len(rows):,} runs · {now()}</p>'
    page+='<p>Primary: consumed answer rows, layers 8–21, mean heads/tokens. Predictive rows are also saved. Partial results are provisional; compare paired samples.</p>'
    page+='<p><a href="summary.json">Summary</a> · <a href="sample_metrics.csv">Metrics</a> · <a href="paired_metrics.csv">Paired metrics</a> · <a href="status.json">Status</a></p>'
    page+='<table><tr><th>Dataset</th><th>Order</th><th>n</th><th>GT attention mass</th></tr>'+''.join(body)+'</table>'
    page+='<h2>Paired attention plots</h2><ul>'+''.join(f'<li><a href="{r["dataset"]}/{r["sample_key"]}/comparison.png">{r["sample_key"]}</a></li>' for r in paired if (relocate_path(OUT/r['dataset']/r['sample_key']/'comparison.png')).exists())+'</ul>'
    write_result_text((relocate_path(OUT/'report.html')), page)
    return len(paired), len(rows)


def worker(smoke=False):
    import torch
    import numpy as np
    from nncore.engine import set_random_seed
    from transformers import AutoModelForImageTextToText, AutoProcessor
    OUT.mkdir(parents=True, exist_ok=True)
    lock=(relocate_path(OUT/'worker.lock')).open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(8)
    set_random_seed(42)
    samples=manifest()
    config=dict(upstream_commit='5d2110fb3faf74f4b0fc5957cd7ab73bea4763dc', model=str(MODEL),
        samples=len(samples), orders=ORDERS, fps=2,min_tokens=64,total_tokens=14336,
        dtype='bfloat16',attention_backend='flash_attention_2',max_new_tokens=512,do_sample=False,seed=42,
        primary_attention='consumed answer rows, excluding EOS; head mean, token mean, layers 8–21',
        secondary_attention='predictive answer rows, excluding EOS',
        measurement='FP32 softmax(QK^T/sqrt(d)) using live BF16 Q/K promoted to FP32; FA2 output untouched',
        environment={n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','qwen-vl-utils','flash-attn','nncore','accelerate','huggingface-hub']},
        annotations_sha256={p.name:sha(p) for p in result_glob(DATA, '*-timelens.json')})
    if (relocate_path(OUT/'run_config.json')).exists():assert read(relocate_path(OUT/'run_config.json'))==config
    else:write(relocate_path(OUT/'run_config.json'),config)
    chosen=[next(s for s in samples if s['dataset']==ds) for ds in DATASETS] if smoke else sorted(samples,key=lambda s:s['duration'],reverse=True)
    write(relocate_path(OUT/'status.json'),dict(state='loading_model',pid=os.getpid(),smoke=smoke,updated_at=now(),expected_pairs=len(samples)))
    processor=AutoProcessor.from_pretrained(MODEL,padding_side='left',do_resize=False,trust_remote_code=True,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map='auto',local_files_only=True).eval()
    capture=AnswerCapture(model)
    failures=[]; done=sum(all((relocate_path(OUT/s['dataset']/s['sample_key']/o/'validation.json')).exists() for o in ORDERS) for s in samples)
    pending=[]
    for sample in chosen:
        if all((relocate_path(OUT/sample['dataset']/sample['sample_key']/o/'validation.json')).exists() for o in ORDERS):
            if not (relocate_path(OUT/sample['dataset']/sample['sample_key']/'comparison.json')).exists():compare_pair(sample)
            continue
        pending.append(sample)
    if not smoke:
        write(relocate_path(OUT/'execution_policy.json'),dict(updated_at=now(),preparation_workers=4,
            preparation_cpu_affinity_per_worker=2,preparation_torch_threads=2,
            decoder_threads=2,
            prefetch_factor=2,pin_memory=True,multiprocessing_context='spawn',
            inference_batch_size=1,resumed_completed_pairs=done))
        prepared=iter(prepared_loader(pending))
        write(relocate_path(OUT/'status.json'),dict(state='prefetching',pid=os.getpid(),smoke=False,
            updated_at=now(),completed_pairs=done,expected_pairs=len(samples),failures=0))
    for sample in pending:
        pair_started=time.monotonic()
        if not smoke:
            item=next(prepared)
            assert item['sample']==sample
            wait_seconds=time.monotonic()-pair_started
            print(now(),'PREFETCH',sample['sample_key'],
                  'wait_seconds',round(wait_seconds,3),
                  'prepare_seconds',round(item.get('prepare_seconds',0),3),flush=True)
        for attempt in range(2):
            try:
                write(relocate_path(OUT/'status.json'),dict(state='running',pid=os.getpid(),smoke=smoke,updated_at=now(),
                    current=sample['sample_key'],attempt=attempt+1,completed_pairs=done,expected_pairs=len(samples),failures=len(failures)))
                print(now(),'PREPARE',sample['sample_key'],sample['duration'],flush=True)
                if not smoke and attempt==0:
                    if 'error' in item:raise RuntimeError(item['error'])
                    pair,info=item.pop('pair'),item.pop('info')
                else:
                    pair,info=prepare_pair(sample,processor)
                write(relocate_path(QWEN3_ZERO/sample['dataset']/sample['sample_key']/'metadata.json'),dict(sample=sample,**info))
                for order,inputs in zip(ORDERS,pair):
                    if (relocate_path(OUT/sample['dataset']/sample['sample_key']/order/'validation.json')).exists():continue
                    print(now(),'GENERATE',sample['sample_key'],order,info['sequence_length'],flush=True)
                    result=generate_one(model,processor,capture,inputs,info,sample,order,baseline_check=smoke)
                    print(now(),'SAVED',sample['sample_key'],order,repr(result['answer']),round(result['elapsed_seconds'],1),flush=True)
                del pair
                compare_pair(sample)
                done+=1
                print(now(),'PAIR_COMPLETE',sample['sample_key'],
                      'elapsed_seconds',round(time.monotonic()-pair_started,3),flush=True)
                break
            except Exception:
                capture.enabled=False
                error=traceback.format_exc();print(error,flush=True)
                torch.cuda.empty_cache()
                if smoke:raise
                if attempt==1:
                    failures.append(dict(sample_key=sample['sample_key'],error=error))
                    write(relocate_path(OUT/sample['dataset']/sample['sample_key']/'error.json'),failures[-1])
        if smoke or done%10==0:summarize(samples)
    paired,runs=summarize(samples)
    if not smoke:
        # Use the original metric CLI on the exact rounded predictions as a final check.
        for dataset in DATASETS:
            for order in ORDERS:
                records=[]
                for s in samples:
                    p=relocate_path(OUT/dataset/s['sample_key']/order/'answer.json')
                    if s['dataset']!=dataset or not (relocate_path(p.parent/'validation.json')).exists():continue
                    a=load_answer(p.parent);records.append({f"{a['video_name']}>>>{a['query']}>>>{a['gt']}":dict(timestamps=a['timestamps'],answers=a['answer'],duration=a['duration'])})
                dest=relocate_path(QWEN3_ZERO/f'{dataset}_{order}.jsonl')
                write_result_text(dest, ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
                env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(relocate_path(ROOT/'runtime/timelens_deps')),str(UPSTREAM)])
                subprocess.run([sys.executable,str(relocate_path(UPSTREAM/'evaluation/compute_metrics.py')),'-f',str(dest)],env=env,check=True)
    write(relocate_path(OUT/('smoke_validation.json' if smoke else 'completion.json')),dict(passed=not failures,
        smoke=smoke,baseline_generation_checked=smoke,completed_pairs=paired,completed_runs=runs,failures=failures,finished_at=now()))
    write(relocate_path(OUT/'status.json'),dict(state='smoke_complete' if smoke else ('completed' if not failures else 'completed_with_errors'),
        pid=os.getpid(),updated_at=now(),completed_pairs=paired,completed_runs=runs,expected_pairs=len(samples),failures=len(failures)))


def supervise():
    OUT.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,4):
        proc=subprocess.Popen([sys.executable,str(__file__),'worker'])
        write(relocate_path(OUT/'background.json'),dict(supervisor_pid=os.getpid(),worker_pid=proc.pid,attempt=attempt,started_at=now()))
        code=proc.wait()
        if code==0:return
        print(now(),'WORKER EXIT',code,'attempt',attempt,flush=True)
        time.sleep(10)
    write(relocate_path(OUT/'status.json'),dict(state='failed',reason='Worker failed three times',updated_at=now()))
    raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','smoke','worker','start','supervise','status','report'])
    args=parser.parse_args()
    if args.action=='prepare':
        print(len(manifest()),'TimeLens samples ready')
    elif args.action in ['worker','smoke']:
        worker(args.action=='smoke')
    elif args.action=='supervise':
        supervise()
    elif args.action=='start':
        assert (relocate_path(OUT/'smoke_validation.json')).exists() and read(relocate_path(OUT/'smoke_validation.json'))['passed'], 'Run smoke validation first'
        if (relocate_path(OUT/'background.json')).exists():
            old=read(relocate_path(OUT/'background.json'))
            for key in ['supervisor_pid','worker_pid']:
                cmd=relocate_path(Path(f'/proc/{old[key]}/cmdline'))
                if cmd.exists() and b'timelens_order.py' in cmd.read_bytes():
                    raise RuntimeError(f'Already running: {old}')
        with (relocate_path(OUT/'run.log')).open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(__file__),'supervise'],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True,close_fds=True)
        write(relocate_path(OUT/'launch.json'),dict(pid=proc.pid,started_at=now(),log=str(relocate_path(OUT/'run.log')),detached=True))
        print(json.dumps(read(relocate_path(OUT/'launch.json')),indent=2))
    elif args.action=='status':
        print(json.dumps(read(relocate_path(OUT/'status.json')),indent=2))
    else:
        samples=manifest()
        summarize(samples)
        if (relocate_path(OUT/'completion.json')).exists() and read(relocate_path(OUT/'completion.json'))['passed']:
            from timelens_report import build_final_report
            build_final_report(OUT,samples)


if __name__=='__main__':
    main()
