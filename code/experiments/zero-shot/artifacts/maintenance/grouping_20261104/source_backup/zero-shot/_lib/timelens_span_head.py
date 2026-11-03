"""Inference-only alphabet-restricted LM head against saved TimeLens zero-shot."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import time

ROOT = relocate_path(Path(__file__)).resolve().parents[1]
sys.path.insert(0, str(relocate_path(ROOT/'runtime/timelens_deps')))
sys.path.insert(0, str(relocate_path(ROOT/'vendor/TimeLens')))
os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import torch.nn.functional as F
from timelens_inputs import prepare_pair, MODEL, DATASETS, read, write, sha
from timelens.utils import extract_time, iou

from timelens_paths import QWEN3_ZERO, PILOT_SAMPLES, baseline_path
from timelens_store import save_prediction
BASE = QWEN3_ZERO
OUT = relocate_path(ROOT/'results/timelens_span_head')


class RestrictedHead(torch.nn.Module):
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


class Prepared:
    def __init__(self, rows, model_key):
        self.rows = rows
        self.model_key = model_key
        self.processor = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.processor is None:
            import decord
            from functools import partial
            from transformers import AutoProcessor
            torch.set_num_threads(1)
            decord.VideoReader = partial(decord.VideoReader, num_threads=1)
            mp = MODEL if self.model_key=='qwen3' else relocate_path(Path('/home/irteam/backbones/models/Qwen2.5-VL-7B-Instruct'))
            self.processor = AutoProcessor.from_pretrained(mp,padding_side='left',do_resize=False,trust_remote_code=True,local_files_only=True)
        row = self.rows[i]
        if self.model_key=='qwen25':
            from evaluation.utils import GroundingDataset
            from types import SimpleNamespace
            item=GroundingDataset([row],self.processor,SimpleNamespace(model_path='/home/irteam/backbones/models/Qwen2.5-VL-7B-Instruct',fps=2,min_tokens=64,total_tokens=14336))[0]
            inputs=item['inputs']
            return dict(sample=row,inputs=inputs,input_sha256=hashlib.sha256(inputs.input_ids.numpy().tobytes()).hexdigest(),metadata=dict(video_grid_thw=inputs.video_grid_thw.tolist(),sequence_length=inputs.input_ids.shape[1]))
        pair, info = prepare_pair(row, self.processor)
        saved = relocate_path(BASE/row['dataset']/row['sample_key'])
        old_info = read(relocate_path(saved/'metadata.json'))
        for k in ['prompt','video_grid_thw','timestamps','duration_video','decoder_metadata','sequence_length']:
            assert info[k] == old_info[k], (row['sample_key'], k)
        decoded = self.processor.tokenizer.decode(pair[0].input_ids[0], skip_special_tokens=False)
        assert decoded == (relocate_path(saved/'video_query/expanded_input.txt')).read_text(), row['sample_key']
        return dict(sample=row, inputs=pair[0], input_sha256=hashlib.sha256(pair[0].input_ids.numpy().tobytes()).hexdigest(), metadata=info)


def identity(value):
    return value


def worker(args):
    from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
    import transformers
    torch.set_num_threads(2)
    set_seed(42)
    full = read(relocate_path(BASE/'samples.json'))
    if args.smoke:
        rows = [next(r for r in full if r['dataset']==ds) for ds in DATASETS]
        dest = relocate_path(OUT/args.model/'smoke')
    else:
        rows = read(PILOT_SAMPLES)
        assert len(rows)==300 and all(sum(r['dataset']==d for r in rows)==100 for d in DATASETS)
        originals = {r['sample_key']:r for r in full}
        assert all(originals[r['sample_key']]==r for r in rows)
        rows = rows[args.shard::args.shards]
        dest = relocate_path(OUT/args.model/f'shard{args.shard}')
    dest.mkdir(parents=True,exist_ok=True)
    pp = relocate_path(dest/'predictions.jsonl')
    resume_paths = [pp] if args.smoke else list(result_glob((relocate_path(OUT/args.model)), 'shard*/predictions.jsonl'))
    done = {json.loads(s)['sample_key'] for p in resume_paths if p.exists() for s in p.read_text().splitlines()}
    rows = [r for r in rows if r['sample_key'] not in done]
    if not rows:
        return
    started = time.time()
    mp = MODEL if args.model=='qwen3' else relocate_path(Path('/home/irteam/backbones/models/Qwen2.5-VL-7B-Instruct'))
    proc = AutoProcessor.from_pretrained(mp,padding_side='left',do_resize=False,trust_remote_code=True,local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(mp,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos,list) else [eos]
    ids = sorted(set([i for i in range(len(proc.tokenizer)) if i not in proc.tokenizer.all_special_ids and re.fullmatch(r'[0-9.,\-\[\] :\n]+',proc.tokenizer.decode([i],clean_up_tokenization_spaces=False))] + eos))
    head = RestrictedHead(model.lm_head,ids)
    model.lm_head = head
    with torch.inference_mode():
        x = torch.randn(2,1,head.full_head.in_features,device='cuda',dtype=torch.bfloat16)
        actual = head(x)[...,ids]
        expected = head.full_head(x)[...,ids]
        torch.testing.assert_close(actual,expected,rtol=.02,atol=.02)
        error = float((actual-expected).abs().max())
    write(relocate_path(dest/'config.json'),dict(model=str(mp),repetition_penalty=model.generation_config.repetition_penalty,original_qwen3_config=read(relocate_path(BASE/'run_config.json')),order='video_query',transformers=transformers.__version__,torch=torch.__version__,allowed_count=len(ids),allowed_tokens=[dict(id=i,text=proc.tokenizer.decode([i])) for i in ids],full_vocabulary=head.full_head.out_features,projection_max_abs_error=error,intervention='Reduced weight-row projection; original token IDs retained; unchanged prompt; no training; no grammar or range constraints',baseline_manifest_sha256=sha(relocate_path(BASE/'samples.json')),manifest_sha256=None if args.smoke else sha(PILOT_SAMPLES),shard=args.shard,shards=args.shards))
    loader = torch.utils.data.DataLoader(Prepared(rows,args.model),batch_size=None,collate_fn=identity,num_workers=2,prefetch_factor=2,pin_memory=True,multiprocessing_context='spawn')
    checked = set()
    for i,item in enumerate(loader):
        row = item['sample']
        saved = relocate_path(BASE/row['dataset']/row['sample_key']/'video_query/answer.json')
        baseline = read(saved) if args.model=='qwen3' else None
        inputs = item['inputs'].to('cuda',non_blocking=True)
        n = inputs.input_ids.shape[1]
        kw = dict(do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512)
        with torch.inference_mode():
            if args.model=='qwen25' or row['dataset'] not in checked:
                head.enabled = False
                torch.cuda.synchronize()
                baseline_start=time.time()
                original = model.generate(**inputs,**kw)[0,n:].tolist()
                torch.cuda.synchronize()
                baseline_seconds=time.time()-baseline_start
                if args.model=='qwen25':
                    text=proc.tokenizer.decode(original,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    parsed_base=extract_time(text)
                    spans_base=[[round(s),round(e)] for s,e in (parsed_base or [[row['duration']+10,row['duration']+20]])]
                    baseline=dict(sample_key=row['sample_key'],dataset=row['dataset'],video_name=relocate_path(Path(row['video_path'])).name,query=row['query'],gt=row['span'][0],duration=row['duration'],order='video_query',answer=text,generated_ids=original,timestamps=spans_base,iou=iou(row['span'][0],spans_base[0]),parse_failed=not bool(parsed_base),truncated=original[-1] not in eos,input_ids_sha256=item['input_sha256'],elapsed_seconds=baseline_seconds,model_key=args.model)
                    saved=baseline_path(args.model,row)
                    save_prediction(args.model,baseline)
                audit = dict(sample_key=row['sample_key'],exact_tokens=original==baseline['generated_ids'],saved_answer=baseline['answer'],reproduced_answer=proc.tokenizer.decode(original,skip_special_tokens=True,clean_up_tokenization_spaces=False))
                with (relocate_path(dest/'baseline_validation.jsonl')).open('a') as f:
                    f.write(json.dumps(audit)+'\n')
                assert audit['exact_tokens'], audit
                print('BASELINE_REPRODUCED' if args.model=='qwen3' else 'BASELINE_GENERATED',row['sample_key'],flush=True)
                head.enabled = True
                checked.add(row['dataset'])
            torch.cuda.synchronize()
            start = time.time()
            torch.cuda.reset_peak_memory_stats()
            output = model.generate(**inputs,**kw)[0,n:].tolist()
            torch.cuda.synchronize()
            elapsed = time.time()-start
        assert set(output)<=set(ids)
        answer = proc.tokenizer.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        parsed = extract_time(answer)
        # Same fallback, integer rounding and first-span scoring as timelens_order.
        timestamps = [[round(s),round(e)] for s,e in (parsed or [[row['duration']+10,row['duration']+20]])]
        result = dict(model_key=args.model,sample_key=row['sample_key'],dataset=row['dataset'],video_name=relocate_path(Path(row['video_path'])).name,query=row['query'],gt=row['span'][0],duration=row['duration'],order='video_query',answer=answer,generated_ids=output,parsed_timestamps=parsed,timestamps=timestamps,iou=iou(row['span'][0],timestamps[0]),parse_failed=not bool(parsed),stopped_on_eos=output[-1] in eos,truncated=output[-1] not in eos,input_tokens=n,input_ids_sha256=item['input_sha256'],input_matches_baseline=True,elapsed_seconds=elapsed,peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,baseline_path=os.path.relpath(saved,ROOT),baseline_sha256=sha(saved))
        with pp.open('a') as f:
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
        print('COMPLETE',i+1,len(rows),row['sample_key'],round(elapsed,2),repr(answer[:120]),flush=True)
        del inputs,item
    write(relocate_path(dest/'complete.json'),dict(count=len(pp.read_text().splitlines()),wall_seconds=time.time()-started))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',choices=['qwen25','qwen3'],default='qwen3')
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=2)
    ap.add_argument('--smoke',action='store_true')
    worker(ap.parse_args())


if __name__=='__main__':
    main()
