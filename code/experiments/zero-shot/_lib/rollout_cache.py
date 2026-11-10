"""Shared single-token KV cache replay and rollback validation for rollout."""
import time

class Evaluator:
    """One mutable KV cache; crop/replay shared text prefixes without redoing video."""
    def __init__(self,model,inputs,torch):
        self.model=model;self.inputs=inputs;self.torch=torch;self.n=inputs.input_ids.shape[1]
        self.path=();self.logits={};self.calls=0;self.text_tokens=0
        t=time.monotonic()
        with torch.inference_mode():o=model(**inputs,use_cache=True,logits_to_keep=1)
        self.cache=o.past_key_values;self.logits[()]=o.logits[0,-1].float().cpu()
        self.prefill_seconds=time.monotonic()-t
        del o
    def get(self,path):
        path=tuple(path)
        if path in self.logits:return self.logits[path]
        common=0
        for a,b in zip(self.path,path):
            if a!=b:break
            common+=1
        # An ancestor's logits may have been evicted for fair policy costs.
        # Recompute its final token rather than querying an empty suffix.
        if common==len(path):common=max(0,len(path)-1)
        self.cache.crop(self.n+common)
        tokens=path[common:];assert tokens
        torch=self.torch
        # Always replay one text token per forward. Variable suffix batch shapes
        # otherwise introduce BF16 differences that depend on traversal order.
        for j,token in enumerate(tokens,common):
            with torch.inference_mode():
                o=self.model(input_ids=torch.tensor([[token]],device='cuda'),
                    attention_mask=torch.ones((1,self.n+j+1),device='cuda',dtype=self.inputs.attention_mask.dtype),
                    past_key_values=self.cache,cache_position=torch.tensor([self.n+j],device='cuda'),
                    use_cache=True,logits_to_keep=1)
            self.cache=o.past_key_values;self.path=path[:j+1]
            self.logits[self.path]=o.logits[0,-1].float().cpu();self.calls+=1;self.text_tokens+=1
            del o
        return self.logits[path]

def validate_cache(evaluator,inputs,anchor,tokenizer,torch):
    # Validate branching rollback, not only the original greedy trajectory.
    prefix=tuple(anchor)
    digit=tokenizer.encode('9',add_special_tokens=False)
    paths=[prefix+tuple(digit),prefix+tuple(tokenizer.encode('1',add_special_tokens=False))]
    audits=[]
    for path in paths:
        actual=evaluator.get(path)
        kw=dict(inputs);kw['input_ids']=torch.cat([inputs.input_ids,torch.tensor([path],device='cuda')],dim=1)
        kw['attention_mask']=torch.ones_like(kw['input_ids'])
        with torch.inference_mode():fresh=evaluator.model(**kw,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
        # Fresh prefill recalculates rope_deltas for the same video/text context.
        ap=torch.softmax(actual,-1);bp=torch.softmax(fresh,-1)
        audit=dict(prefix=list(path),top1_equal=int(actual.argmax())==int(fresh.argmax()),
            full_prefill_probability_l1=float((ap-bp).abs().sum()))
        reference=Evaluator(evaluator.model,inputs,torch)
        expected=reference.get(path)
        rp=torch.softmax(expected,-1)
        audit.update(rollback_top1_equal=int(actual.argmax())==int(expected.argmax()),
                     rollback_probability_l1=float((ap-rp).abs().sum()),
                     validation='rollback vs clean original multimodal prefill + identical single-token replay')
        assert audit['rollback_top1_equal'] and audit['rollback_probability_l1']<.001,audit
        del reference
        audits.append(audit)
    return audits
