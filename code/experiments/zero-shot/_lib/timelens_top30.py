"""Capture full-vocabulary probabilities during the actual greedy generation."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import argparse
import json
import os
import re
import time

from timelens_span_head import ROOT, BASE, MODEL, Prepared, identity, read, write, sha
import torch
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText, LogitsProcessor, LogitsProcessorList, set_seed
from timelens.utils import extract_time, iou

OUT = relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300')
PILOT = relocate_path(ROOT/'results/uncertainty/span_head/pilot/300')
from timelens_paths import PILOT_SAMPLES, baseline_path


class Capture(LogitsProcessor):
    def __init__(self, tokenizer, allowed, eos):
        self.tokenizer = tokenizer
        self.allowed = torch.tensor(allowed, device='cuda')
        self.eos = torch.tensor(eos, device='cuda')
        self.text_cache = {}
        self.steps = []
        self.raw = None

    def text(self, idx):
        if idx not in self.text_cache:
            self.text_cache[idx] = self.tokenizer.decode([idx], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return self.text_cache[idx]

    def summarize(self, logits):
        lp = torch.log_softmax(logits.float(), dim=-1)
        probs = lp.exp()
        values, ids = torch.topk(lp, 30, sorted=True)
        numbers = values.exp().tolist()
        tokens = ids.tolist()
        # softmax covers every output-head row, including padded vocabulary rows.
        assert abs(float(probs.sum())-1) < 2e-5
        assert all(numbers[i] >= numbers[i+1] for i in range(29))
        entropy = -(probs * torch.where(torch.isfinite(lp), lp, 0)).sum()
        return lp, dict(top30=[dict(id=i, token=self.text(i), probability=p) for i,p in zip(tokens,numbers)],
                       top30_mass=sum(numbers), entropy_nats=float(entropy),
                       span_alphabet_mass=float(probs[self.allowed].sum()), eos_mass=float(probs[self.eos].sum()),
                       probability_sum=float(probs.sum()))

    def hook(self, module, args, output):
        assert self.raw is None, 'One raw forward must precede each decoding step'
        self.raw, self.raw_summary = self.summarize(output.logits[0,-1].detach())

    def __call__(self, input_ids, scores):
        assert self.raw is not None
        chosen = int(scores[0].argmax())
        lp, processed = self.summarize(scores[0].detach())
        step = dict(step=len(self.steps)+1, token_id=chosen, token=self.text(chosen),
                    raw_selected_probability=float(self.raw[chosen].exp()),
                    raw_selected_rank=1+int((self.raw>self.raw[chosen]).sum()),
                    processed_selected_probability=float(lp[chosen].exp()),
                    raw=self.raw_summary, processed=processed)
        self.steps.append(step)
        self.raw = None
        return scores  # No alteration of any generation score.


def worker(args):
    torch.set_num_threads(2)
    set_seed(42)
    rows = read(PILOT_SAMPLES)
    if args.smoke:
        rows = [next(s for s in rows if s['dataset']==ds) for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens']]
    else:
        rows = rows[args.shard::args.shards]
    output = relocate_path(OUT/args.model)
    output.mkdir(parents=True,exist_ok=True)
    rows = [r for r in rows if not (relocate_path(output/f"{r['sample_key']}.json")).exists()]
    if not rows:
        return
    mp = MODEL if args.model=='qwen3' else relocate_path(Path('/home/irteam/backbones/models/Qwen2.5-VL-7B-Instruct'))
    proc = AutoProcessor.from_pretrained(mp,padding_side='left',do_resize=False,trust_remote_code=True,local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(mp,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos,list) else [eos]
    allowed = [t['id'] for t in read(relocate_path(PILOT/args.model/'shard0/config.json'))['allowed_tokens']]
    capture = Capture(proc.tokenizer,allowed,eos)
    handle = model.register_forward_hook(capture.hook)
    write(relocate_path(OUT/'configs'/f'{args.model}_shard{args.shard}.json'),dict(model=str(mp),full_head=True,
        vocab_size=model.lm_head.out_features,top_k_recorded=30,normalization='FP32 softmax over full vocabulary; top30 is NOT renormalized',
        raw='model output logits before generation processors',processed='scores after generation processors; includes repetition penalty',
        repetition_penalty=model.generation_config.repetition_penalty,do_sample=False,max_new_tokens=512,
        dtype='bfloat16',attention='flash_attention_2',transformers=transformers.__version__,torch=torch.__version__,
        seed=42,manifest_sha256=sha(PILOT_SAMPLES),cuda_mps=bool(os.environ.get('CUDA_MPS_PIPE_DIRECTORY'))))
    loader = torch.utils.data.DataLoader(Prepared(rows,args.model),batch_size=None,collate_fn=identity,
        num_workers=2,prefetch_factor=2,pin_memory=True,multiprocessing_context='spawn')
    started = time.time()
    for idx,item in enumerate(loader):
        sample = item['sample']
        path = baseline_path(args.model,sample)
        baseline = read(path)
        if args.model=='qwen25':
            assert item['input_sha256']==baseline['input_ids_sha256']
        inputs = item['inputs'].to('cuda',non_blocking=True)
        n = inputs.input_ids.shape[1]
        capture.steps = []
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            generated = model.generate(**inputs,do_sample=False,temperature=None,top_p=None,top_k=None,
                max_new_tokens=512,logits_processor=LogitsProcessorList([capture]))[0,n:].tolist()
        torch.cuda.synchronize()
        assert generated == [s['token_id'] for s in capture.steps]
        assert generated == baseline['generated_ids'], f"Baseline token mismatch: {args.model}/{sample['sample_key']}"
        answer = proc.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert answer == baseline['answer']
        parsed = extract_time(answer)
        spans = [[round(s),round(e)] for s,e in (parsed or [[sample['duration']+10,sample['duration']+20]])]
        assert spans == baseline['timestamps']
        result = dict(model_key=args.model,sample=sample,answer=answer,generated_ids=generated,steps=capture.steps,
            timestamps=spans,parsed_timestamps=parsed,iou=iou(sample['span'][0],spans[0]),
            stopped_on_eos=generated[-1] in eos,baseline_tokens_identical=True,
            input_tokens=n,input_ids_sha256=item['input_sha256'],baseline_path=str(path),baseline_sha256=sha(path),
            elapsed_seconds=time.time()-start)
        write(relocate_path(output/f"{sample['sample_key']}.json"),result)
        print('COMPLETE',args.model,idx+1,len(rows),sample['sample_key'],len(generated),round(result['elapsed_seconds'],2),flush=True)
        del inputs,item
    handle.remove()
    write(relocate_path(OUT/'configs'/f'{args.model}_shard{args.shard}_complete.json'),dict(new_count=len(rows),wall_seconds=time.time()-started))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=['qwen25','qwen3'])
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=2)
    ap.add_argument('--smoke',action='store_true')
    worker(ap.parse_args())
