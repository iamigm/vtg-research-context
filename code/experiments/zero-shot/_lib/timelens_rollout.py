"""Actual stochastic generation with the complete post-filter candidate set."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import argparse,hashlib,json,os,time,types,gzip,base64
import numpy as np
from timelens_span_head import ROOT,MODEL,Prepared,identity,read,write,sha
from timelens_paths import PILOT_SAMPLES,baseline_path
import torch
from transformers import AutoProcessor,AutoModelForImageTextToText,LogitsProcessor,LogitsProcessorList,set_seed
from timelens.utils import extract_time,iou
OUT=relocate_path(ROOT/'results/timelens_rollout')

class Presence(LogitsProcessor):
    def __init__(self,penalty):self.penalty=penalty;self.prompt_length=0
    def __call__(self,input_ids,scores):
        if self.penalty and input_ids.shape[1]>self.prompt_length:
            scores=scores.clone();ids=input_ids[0,self.prompt_length:].unique();scores[0,ids]-=self.penalty
        return scores

class ObserveBefore(LogitsProcessor):
    def __init__(self,capture):self.capture=capture
    def __call__(self,input_ids,scores):
        self.capture.before=torch.softmax(scores[0].float(),-1);return scores

class Capture(LogitsProcessor):
    def __init__(self,tokenizer):self.tokenizer=tokenizer;self.steps=[];self.cache={};self.raw=None;self.before=None
    def token(self,i):
        if i not in self.cache:self.cache[i]=self.tokenizer.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        return self.cache[i]
    def hook(self,module,args,output):
        assert self.raw is None
        self.raw=torch.softmax(output.logits[0,-1].detach().float(),-1)
    def __call__(self,input_ids,scores):
        assert self.raw is not None and self.before is not None
        values=scores[0].float();probs=torch.softmax(values,-1)
        ids=torch.where(torch.isfinite(values))[0];order=torch.argsort(probs[ids],descending=True,stable=True);ids=ids[order]
        entry=dict(step=len(self.steps)+1,candidate_count=len(ids),positive_probability_count=int((probs>0).sum()),retained_prefilter_mass=float(self.before[ids].sum()),retained_raw_mass=float(self.raw[ids].sum()))
        if len(ids)>1024:
            # Lossless full-vocabulary arrays, never a top-N approximation.
            arrays={};aliases={};seen={}
            for name,tensor in [('probability',probs),('raw_probability',self.raw),('prefilter_probability',self.before)]:
                binary=tensor.cpu().numpy().astype('<f4',copy=False).tobytes();digest=hashlib.sha256(binary).hexdigest()
                if digest in seen:aliases[name]=seen[digest]
                else:arrays[name]=base64.b64encode(gzip.compress(binary,compresslevel=1,mtime=0)).decode();seen[digest]=name
            entry['dense']=dict(encoding='gzip-base64-float32-le',vocab_size=len(probs),ids='all' if len(ids)==len(probs) else ids.tolist(),arrays=arrays,aliases=aliases)
        else:
            entry['candidates']=[dict(id=i,token=self.token(i),probability=p,raw_probability=r,prefilter_probability=b) for i,p,r,b in zip(ids.tolist(),probs[ids].tolist(),self.raw[ids].tolist(),self.before[ids].tolist())]
        self.steps.append(entry)
        self.raw=None;self.before=None
        return scores

def worker(args):
    config=read(relocate_path(OUT/'config.json'));settings=config['models'][args.model];rollouts=config['rollouts_per_variant']
    torch.set_num_threads(2)
    rows=read(PILOT_SAMPLES)[args.shard::args.shards]
    rows=[s for s in rows if any(not (relocate_path(OUT/'records'/args.model/s['sample_key']/f'{variant}_{r}.json')).exists() for variant in ['top_k','top_p'] for r in range(rollouts))]
    if not rows:return
    path=MODEL if args.model=='qwen3' else relocate_path(Path('/home1/irteam/backbones/models/Qwen2.5-VL-7B-Instruct'))
    proc=AutoProcessor.from_pretrained(path,padding_side='left',do_resize=False,trust_remote_code=True,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
    capture=Capture(proc.tokenizer);presence=Presence(settings['presence_penalty']);before=ObserveBefore(capture)
    original=model._get_logits_processor;enabled=True
    def processors(self,*a,**kw):
        pipeline=original(*a,**kw)
        # Repetition penalty is native; add presence before temperature and truncation.
        first=next((i for i,p in enumerate(pipeline) if type(p).__name__.endswith('LogitsWarper')),len(pipeline))
        if presence.penalty:pipeline.insert(first,presence)
        if enabled:
            cut=next((i for i,p in enumerate(pipeline) if type(p).__name__ in ['TopKLogitsWarper','TopPLogitsWarper']),len(pipeline))
            pipeline.insert(cut,before);pipeline.append(capture)
        return pipeline
    model._get_logits_processor=types.MethodType(processors,model)
    if args.shard==0:
        vocabulary=[]
        for idx in range(model.lm_head.out_features):
            try:vocabulary.append(capture.token(idx))
            except Exception:vocabulary.append(f'<unused:{idx}>')
        write(relocate_path(OUT/'vocab'/f'{args.model}.json'),vocabulary)
        write_result_text((relocate_path(OUT/'vocab'/f'{args.model}.js')), 'window.VOCABS=window.VOCABS||{};window.VOCABS['+json.dumps(args.model)+']='+json.dumps(vocabulary,ensure_ascii=False)+';\n')
    handle=model.register_forward_hook(capture.hook)
    loader=torch.utils.data.DataLoader(Prepared(rows,args.model),batch_size=None,collate_fn=identity,num_workers=1,prefetch_factor=1,pin_memory=True,multiprocessing_context='spawn')
    verified=set()
    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
    for item in loader:
        s=item['sample'];baseline=read(baseline_path(args.model,s));prior=read(relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300'/args.model/(s['sample_key']+'.json')))
        assert item['input_sha256']==prior['input_ids_sha256'],s['sample_key']
        inputs=item['inputs'].to('cuda');n=inputs.input_ids.shape[1];presence.prompt_length=n
        for variant in ['top_k','top_p']:
            params=dict(do_sample=True,temperature=settings['temperature'],repetition_penalty=settings['repetition_penalty'],top_k=settings['top_k'] if variant=='top_k' else 0,top_p=settings['top_p'] if variant=='top_p' else 1.0,max_new_tokens=512)
            for rollout in range(rollouts):
                dest=relocate_path(OUT/'records'/args.model/s['sample_key']/f'{variant}_{rollout}.json')
                if dest.exists():continue
                seed=int(hashlib.sha256(f"{s['sample_key']}:{rollout}:42".encode()).hexdigest()[:8],16)
                set_seed(seed);capture.steps=[];capture.raw=None;capture.before=None;start=time.time()
                with torch.inference_mode():generated=model.generate(**inputs,**params)[0,n:].tolist()
                assert len(generated)==len(capture.steps)
                for token,step in zip(generated,capture.steps):
                    if 'dense' in step:
                        dense=step['dense'];values=np.frombuffer(gzip.decompress(base64.b64decode(dense['arrays']['probability'])),dtype='<f4')
                        assert dense['ids']=='all' or token in dense['ids']
                        candidate=dict(token=capture.token(token),probability=float(values[token]))
                    else:candidate=next(c for c in step['candidates'] if c['id']==token)
                    assert candidate['probability']>0
                    step.update(token_id=token,token=candidate['token'],selected_probability=candidate['probability'])
                if variant not in verified:
                    handle.remove();enabled=False;set_seed(seed)
                    with torch.inference_mode():control=model.generate(**inputs,**params)[0,n:].tolist()
                    assert control==generated,'Capture altered sampled generation'
                    enabled=True;handle=model.register_forward_hook(capture.hook);verified.add(variant)
                    write(relocate_path(OUT/'validation'/f'{args.model}_{variant}_shard{args.shard}_instrumentation.json'),dict(passed=True,sample_key=s['sample_key'],seed=seed,generated_ids=generated,method='same seed, actual generation with vs without capture'))
                answer=proc.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                parsed=extract_time(answer);spans=[[round(a),round(b)] for a,b in (parsed or [[s['duration']+10,s['duration']+20]])]
                write(dest,dict(model_key=args.model,variant=variant,rollout=rollout,seed=seed,sample=s,settings=params,presence_penalty=settings['presence_penalty'],answer=answer,generated_ids=generated,steps=capture.steps,timestamps=spans,parsed_timestamps=parsed,iou=iou(s['span'][0],spans[0]),stopped_on_eos=generated[-1] in eos,input_ids_sha256=item['input_sha256'],baseline_path=str(baseline_path(args.model,s)),baseline_sha256=sha(baseline_path(args.model,s)),baseline=dict(answer=baseline['answer'],timestamps=baseline['timestamps'],iou=baseline['iou']),elapsed_seconds=time.time()-start))
                print(json.dumps(dict(event='rollout_complete',model=args.model,sample=s['sample_key'],variant=variant,rollout=rollout,steps=len(generated),elapsed_seconds=round(time.time()-start,2))),flush=True)
        del inputs,item;torch.cuda.empty_cache()
    handle.remove()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['qwen25','qwen3'],required=True);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=2);worker(ap.parse_args())
