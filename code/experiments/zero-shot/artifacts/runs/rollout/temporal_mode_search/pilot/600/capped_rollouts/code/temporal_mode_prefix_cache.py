"""Exact native-prefix KV reuse; generation probabilities and RNG are unchanged."""
import time
import os
from contextlib import contextmanager
import temporal_mode_capped_cost as accounting
from temporal_mode_capped_cost import *

BACKEND = 'prefix_kv_cache'
PREFIX_LIMIT = 4096
CACHE_FIELDS = ('prefix_logit_hits','prefix_kv_stores','prefix_kv_restores',
                'prefix_tokens_restored','prefix_cache_cuda_seconds','prefix_cache_enqueue_seconds')


class PrefixMeter(accounting.Meter):
    def __init__(self, torch):
        super().__init__(torch)
        self.events = []

    @contextmanager
    def segment(self, phase):
        with super().segment(phase) as record:
            self.events = []
            try:
                yield record
            finally:
                self.torch.cuda.synchronize()
                record['prefix_cache_cuda_seconds'] = sum(a.elapsed_time(b)/1000 for a,b in self.events)
                self.events = []

    @contextmanager
    def cache_operation(self):
        start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((start,end))
            self.count('prefix_cache_enqueue_seconds',time.perf_counter()-started)


class PrefixBackend(CostBackend):
    def __init__(self, model, inputs, torch, tokenizer, eos):
        self.cost_meter = PrefixMeter(torch)
        with self.cost_meter.segment('video_prefill') as record:
            Backend.__init__(self, MeasuredModel(model,self.cost_meter),inputs,torch,tokenizer,eos)
        self.baseline_cost = dict(schema=SCHEMA,execution_backend=BACKEND,video_prefill=record,
            execution_service='cuda_mps' if os.environ.get('CUDA_MPS_PIPE_DIRECTORY')=='/tmp/irteam_temporal_mode_mps' else 'cuda_context')
        assert all(type(layer).__name__=='DynamicLayer' for layer in self.cache.layers)
        self.video_kv = [(layer.keys,layer.values) for layer in self.cache.layers]
        self.prefix_deltas = {}
        self.prefix_logits = {():self.root}

    def _store(self, path):
        if len(self.prefix_deltas)>=PREFIX_LIMIT or path in self.prefix_logits or path[:-1] not in self.prefix_logits:
            return
        with self.cost_meter.cache_operation():
            # One compact token per layer, not a copy of the complete video KV.
            packed = self.torch.stack([tensor[..., -1:, :] for layer in self.cache.layers
                for tensor in (layer.keys,layer.values)])
        self.prefix_deltas[path] = packed
        self.prefix_logits[path] = self.last
        self.cost_meter.count('prefix_kv_stores')

    def _restore(self, path):
        with self.cost_meter.cache_operation():
            packed = self.torch.cat([self.prefix_deltas[path[:j]] for j in range(1,len(path)+1)],dim=-2)
            for i,layer in enumerate(self.cache.layers):
                layer.keys = self.torch.cat((self.video_kv[i][0],packed[2*i]),dim=-2)
                layer.values = self.torch.cat((self.video_kv[i][1],packed[2*i+1]),dim=-2)
        self.path = path
        self.last = self.prefix_logits[path]
        self.cost_meter.count('prefix_kv_restores')
        self.cost_meter.count('prefix_tokens_restored',len(path))

    def get(self, path):
        path = tuple(path)
        if self.cost_meter.current is None or self.cost_meter.current['phase']=='judge':
            return Backend.get(self,path)
        if path in self.prefix_logits:
            # Returning CPU logits need not change the currently resident GPU path.
            self.cost_meter.count('prefix_logit_hits')
            return self.prefix_logits[path]
        common = 0
        for a,b in zip(self.path,path):
            if a!=b:
                break
            common+=1
        if common==len(path):
            common-=1
        for length in range(len(path)-1,common,-1):
            ancestor=path[:length]
            if ancestor in self.prefix_logits:
                self._restore(ancestor)
                common=length
                break
        self.cache.crop(self.n+common)
        torch=self.torch
        for j in range(common,len(path)):
            with torch.inference_mode():
                output=self.model(input_ids=torch.tensor([[path[j]]],device='cuda'),
                    attention_mask=torch.ones((1,self.n+j+1),device='cuda',dtype=self.inputs.attention_mask.dtype),
                    past_key_values=self.cache,cache_position=torch.tensor([self.n+j],device='cuda'),
                    use_cache=True,logits_to_keep=1)
            self.cache=output.past_key_values
            self.last=output.logits[0,-1].float().cpu()
            self.path=path[:j+1]
            self.actual_tokens+=1
            self.actual_forwards+=1
            self._store(self.path)
        return self.last


def cache_sum(records):
    records=list(records)
    return {field:sum(record.get(field,0) for record in records) for field in CACHE_FIELDS}


class PrefixSearch(CostSearch):
    def sweep(self,save=None,checkpoints=None,judge_fn=judge):
        start=len(self.b.cost_meter.records)
        def measured_save(result):
            point=result['checkpoints'][str(max(map(int,result['checkpoints'])))]
            records=self.b.cost_meter.records[start:]
            cost=point['cost']
            cost['execution_backend']=BACKEND
            cost['execution_service']=self.b.baseline_cost['execution_service']
            cost['prefix_cache']=dict(limit=PREFIX_LIMIT,entries=len(self.b.prefix_deltas),
                gpu_bytes=sum(v.numel()*v.element_size() for v in self.b.prefix_deltas.values()),
                cpu_logit_bytes=sum(v.numel()*v.element_size() for v in self.b.prefix_logits.values()),
                phases={phase:cache_sum(r for r in records if r['phase']==phase) for phase in ('main','tail','judge')},
                sweep=cache_sum(records))
            if save:
                save(result)
        return super().sweep(save=measured_save,checkpoints=checkpoints,judge_fn=judge_fn)


class PrefixTraceWork(TraceWork):
    """Native operation count with the same bounded, prefix-closed KV cache."""
    def __init__(self,n):
        super().__init__(n)
        self.prefixes={()}

    def get(self,path,phase):
        path=tuple(path)
        if phase=='judge':
            return super().get(path,phase)
        if path in self.prefixes:
            return
        common=0
        for a,b in zip(self.path,path):
            if a!=b:
                break
            common+=1
        if common==len(path):
            common-=1
        for length in range(len(path)-1,common,-1):
            if path[:length] in self.prefixes:
                common=length
                break
        for j in range(common,len(path)):
            self.add(phase,1,self.n+j+1)
            prefix=path[:j+1]
            if len(self.prefixes)-1<PREFIX_LIMIT and prefix[:-1] in self.prefixes:
                self.prefixes.add(prefix)
        self.path=path


def prefix_standalone_work(data,route,checkpoint,n,dialog):
    work=PrefixTraceWork(n)
    work.greedy(data['baseline']['ids'])
    prefixes={digest([])[:24]:[]}
    for kind in ('main','tail'):
        prefixes.update({digest(task['prefix'])[:24]:task['prefix'] for task in checkpoint[kind]['tasks']})
    draws=route['draws'][:checkpoint['draw_cutoff']]
    work.draws([d for d in draws if d['kind']=='main'],data['sequences'],prefixes,'main')
    work.judge(checkpoint['selection'],data['choices'],dialog)
    work.draws([d for d in draws if d['kind']=='tail'],data['sequences'],prefixes,'tail')
    phases={phase:dict(work.records[phase]) for phase in ('video_prefill','greedy_decode','main','judge','tail')}
    return dict(schema=SCHEMA,execution_backend=BACKEND,prefix_limit=PREFIX_LIMIT,
        scope='trace-derived independent case with bounded native prefix KV reuse; excludes copy kernels',
        phases=phases,baseline=combine([phases['video_prefill'],phases['greedy_decode']]),
        operational=combine(phases[p] for p in ('video_prefill','greedy_decode','main','judge')),
        research=combine(phases.values()))
