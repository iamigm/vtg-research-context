"""Native Qwen3-VL decoder LRP and exact streamed target-row products.

The scalar derivative is d(greedy-token logit)/dA. R_A is independently
redistributed from a unit output relevance through actual decoder branches;
it is never substituted by activation times gradient. Native FlashAttention
is retained. Its attention probabilities are reconstructed in FP32, in
bounded query blocks, and checked against native pre-projection outputs.

Only two final N-vectors are returned. Every attention row, head and decoder
layer participates. A full N-by-N intermediate is neither retained nor saved.
Temporary native activations are disk-backed and owned by the caller's workdir.
Qwen extensions and the exclusion of vision-encoder LRP are explicit below.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import chefer_lrp_rules as rules

QUERY_BLOCK = 64
LINEAR_BLOCK = 256
REPLAY_PROBABILITY_TOLERANCE = .001
RECONSTRUCTION_RELATIVE_RMS_TOLERANCE = .03
SOFTMAX_ROW_TOLERANCE = 2e-6


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _finite(value, label):
    _require(bool(torch.isfinite(value).all()), f'{label}: nonfinite values')


def _summary(value):
    x = value.detach().double()
    return dict(signed_sum=float(x.sum()), absolute_sum=float(x.abs().sum()),
                positive_sum=float(x.clamp_min(0).sum()), l2=float(x.norm()))


class NativeCapture:
    """Capture each actual prefill/decode node once, including native GQA.

    Disk tensors preserve their native dtype, including BF16 as uint16 bits.
    Checkpoint recomputation cannot overwrite the original activation capture.
    Gradient hooks are attached to the original graph and each must fire once.
    """

    def __init__(self, model, length, prompt_length, workdir):
        self.model, self.length, self.prompt_length = model, length, prompt_length
        self.directory = Path(workdir) / 'native_capture'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.segment_offset = 0
        self.phase = 'forward'
        self.handles, self.tensor_handles = [], []
        self.arrays, self.dtypes, self.coverage = {}, {}, {}
        self.gradient_calls = {}
        self.deepstack_layers = []

    def begin_forward(self):
        self.segment_offset = 0

    def set_segment(self, offset):
        self.segment_offset = int(offset)

    def save(self, name, value, offset=None):
        if self.phase != 'forward':
            return
        self._write(name, value, self.segment_offset if offset is None else offset)

    def _write(self, name, value, offset):
        value = value.detach()
        # All call sites pass [sequence, flattened hidden/head channels].
        _require(value.ndim == 2, f'{name}: expected sequence/channel tensor')
        _require(offset + value.shape[0] <= self.length, f'{name}: sequence overflow')
        cpu = value.to('cpu').contiguous()
        array = cpu.view(torch.uint16).numpy() if cpu.dtype == torch.bfloat16 else cpu.numpy()
        if name not in self.arrays:
            path = self.directory / (name + '.npy')
            self.arrays[name] = np.lib.format.open_memmap(path, mode='w+', dtype=array.dtype,
                                                        shape=(self.length, array.shape[1]))
            self.dtypes[name] = cpu.dtype
            self.coverage[name] = np.zeros(self.length, dtype=np.uint8)
        _require(self.dtypes[name] == cpu.dtype, f'{name}: native dtype changed')
        sl = slice(offset, offset + value.shape[0])
        _require(not self.coverage[name][sl].any(), f'{name}: duplicate native capture')
        self.arrays[name][sl] = array
        self.coverage[name][sl] = 1

    def get(self, name, device, start=0, end=None, native=False):
        end = self.length if end is None else end
        data = torch.from_numpy(np.array(self.arrays[name][start:end], copy=True))
        if self.dtypes[name] == torch.bfloat16:
            data = data.view(torch.bfloat16)
        return data.to(device=device, dtype=self.dtypes[name] if native else torch.float32)

    def _output(self, name):
        def hook(module, args, output):
            if self.phase == 'forward':
                value = output[0] if isinstance(output, tuple) else output
                self.save(name, value.reshape(-1, value.shape[-1]) if value.ndim <= 3
                          else value.reshape(value.shape[1], -1))
        return hook

    def __enter__(self):
        language = self.model.model.language_model

        def embedding(module, args, output):
            # Frozen parameters remain frozen; only decoder input derivatives
            # are required. Vision features themselves are constant boundaries.
            return output.requires_grad_(True) if torch.is_grad_enabled() else output
        self.handles.append(language.embed_tokens.register_forward_hook(embedding))

        def language_before(module, args, kwargs):
            if self.phase != 'forward' or self.segment_offset != 0:
                return
            visual = kwargs.get('deepstack_visual_embeds')
            mask = kwargs.get('visual_pos_masks')
            if visual is not None:
                _require(mask is not None and mask.ndim == 2, 'Missing native DeepStack positions')
                self.visual_positions = mask[0].nonzero().flatten().cpu().numpy()
                for index, feature in enumerate(visual):
                    expanded = torch.zeros((self.prompt_length, feature.shape[-1]),
                                           device=feature.device, dtype=feature.dtype)
                    expanded[torch.as_tensor(self.visual_positions, device=feature.device)] = feature
                    self.save(f'deepstack_{index}', expanded, 0)
                    self.deepstack_layers.append(index)
        self.handles.append(language.register_forward_pre_hook(language_before, with_kwargs=True))
        self.handles.append(language.norm.register_forward_hook(self._output('final_norm')))

        for index, layer in enumerate(language.layers):
            prefix = f'layer_{index}_'
            def before(module, args, kwargs, name=prefix+'input'):
                if self.phase == 'forward':
                    hidden = kwargs.get('hidden_states', args[0] if args else None)
                    self.save(name, hidden.reshape(-1, hidden.shape[-1]))
            self.handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(self._output(prefix+'output')))
            self.handles.append(layer.input_layernorm.register_forward_hook(self._output(prefix+'attention_norm')))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(self._output(prefix+'mlp_norm')))
            attention = layer.self_attn
            for name, module in [('q', attention.q_norm), ('k', attention.k_norm), ('v', attention.v_proj),
                                 ('attention_branch', attention.o_proj), ('mlp_act', layer.mlp.act_fn),
                                 ('mlp_up', layer.mlp.up_proj), ('mlp_branch', layer.mlp.down_proj)]:
                self.handles.append(module.register_forward_hook(self._output(prefix+name)))

            def attention_before(module, args, kwargs, layer_index=index):
                if self.phase == 'forward' and layer_index == 0:
                    cosine, sine = kwargs['position_embeddings']
                    self.save('rope_cos', cosine.reshape(-1, cosine.shape[-1]))
                    self.save('rope_sin', sine.reshape(-1, sine.shape[-1]))
            self.handles.append(attention.register_forward_pre_hook(attention_before, with_kwargs=True))

            def projection_before(module, args, name=prefix+'attention_mix', layer_index=index):
                if self.phase != 'forward':
                    return
                value = args[0]; offset = self.segment_offset
                self.save(name, value.reshape(-1, value.shape[-1]))
                _require(value.requires_grad, f'Layer {layer_index}: missing native attention graph')
                def gradient_hook(gradient):
                    key = (layer_index, offset)
                    self.gradient_calls[key] = self.gradient_calls.get(key, 0) + 1
                    self._write(name+'_gradient', gradient.reshape(-1, gradient.shape[-1]), offset)
                self.tensor_handles.append(value.register_hook(gradient_hook))
            self.handles.append(attention.o_proj.register_forward_pre_hook(projection_before))
        return self

    def finish(self):
        expected_segments = [0, *range(self.prompt_length, self.length)]
        for layer in range(len(self.model.model.language_model.layers)):
            for offset in expected_segments:
                _require(self.gradient_calls.get((layer, offset)) == 1,
                         f'Layer {layer}, segment {offset}: native gradient hook count differs')
        for name, coverage in self.coverage.items():
            required = self.prompt_length if name.startswith('deepstack_') else self.length
            _require(bool(coverage[:required].all()), f'{name}: incomplete native capture')
            self.arrays[name].flush()
        return dict(fields=len(self.arrays), bytes=sum(array.nbytes for array in self.arrays.values()),
                    gradient_segments=len(self.gradient_calls), deepstack_layers=self.deepstack_layers,
                    preserves_native_dtype=True, checkpoint_recomputation_overwrites=False)

    def __exit__(self, *exception):
        for handle in self.handles + self.tensor_handles:
            handle.remove()
        self.handles.clear(); self.tensor_handles.clear()


def _segments(prompt_length, length):
    return [(0, prompt_length), *[(index, index+1) for index in range(prompt_length, length)]]


def _node_add(x, y, relevance, segments):
    left, right = torch.empty_like(relevance), torch.empty_like(relevance)
    for start, end in segments:
        left[start:end], right[start:end] = rules.add(x[start:end], y[start:end], relevance[start:end])
    return left, right


def _linear(x, module, relevance, block=LINEAR_BLOCK):
    result = torch.empty_like(x)
    weight = module.weight.detach().float()
    for start in range(0, x.shape[0], block):
        end = min(x.shape[0], start+block)
        result[start:end] = rules.linear(x[start:end], weight, relevance[start:end])
    return result


def _native_add(capture, left, right, device):
    # The forward residual sum is rounded once in the actual native dtype.
    a = capture.get(left, device, native=True)
    b = capture.get(right, device, native=True)
    return (a+b).float()


@torch.no_grad()
def _attention_reverse(layer, index, capture, relevance, rollout, chefer, device,
                       query_block=QUERY_BLOCK):
    """Exact all-row bilinear LRP, with bounded H x block x N temporaries."""
    prefix=f'layer_{index}_'; length=capture.length
    attention=layer.self_attn; heads=attention.config.num_attention_heads
    kv_heads=attention.config.num_key_value_heads; dimension=attention.head_dim
    repeats=heads//kv_heads
    q_native=capture.get(prefix+'q',device,native=True).reshape(length,heads,dimension).transpose(0,1)
    k_native=capture.get(prefix+'k',device,native=True).reshape(length,kv_heads,dimension).transpose(0,1)
    cosine_native=capture.get('rope_cos',device,native=True)[None]
    sine_native=capture.get('rope_sin',device,native=True)[None]
    # Preserve Qwen's BF16 multiplication/addition roundings before FP32 A.
    query=(q_native*cosine_native+rules.rotate_half(q_native)*sine_native).float()
    key=(k_native*cosine_native+rules.rotate_half(k_native)*sine_native).float().repeat_interleave(repeats,dim=0)
    value=capture.get(prefix+'v',device).reshape(length,kv_heads,dimension).transpose(0,1).repeat_interleave(repeats,dim=0)
    query_relevance=torch.zeros_like(query); key_relevance=torch.zeros_like(key); value_relevance=torch.zeros_like(value)
    mix=capture.get(prefix+'attention_mix',device)
    mix_relevance=_linear(mix,attention.o_proj,relevance).reshape(length,heads,dimension).transpose(0,1)
    native_gradient=capture.get(prefix+'attention_mix_gradient',device).reshape(length,heads,dimension).transpose(0,1)
    native_mix=mix.reshape(length,heads,dimension).transpose(0,1)
    next_rollout=torch.zeros_like(rollout); next_chefer=chefer.clone()
    difference_squared=reference_squared=0.; maximum_difference=0.; row_error=0.; max_future=0.
    attention_relevance_sum=0.; av_input_sum=float(mix_relevance.double().sum())
    av_expected_sum=qk_expected_sum=0.;av_expected_absolute=qk_expected_absolute=0.
    attention_relevance_absolute=0.
    for start in range(0,length,query_block):
        end=min(length,start+query_block); q=query[:,start:end]
        qk=torch.matmul(q,key.transpose(-2,-1))
        causal=torch.arange(length,device=device)[None,:] <= torch.arange(start,end,device=device)[:,None]
        scores=(qk*attention.scaling).masked_fill(~causal[None],-torch.inf)
        probability=torch.softmax(scores,dim=-1)
        _finite(probability,f'Layer {index} attention')
        row_error=max(row_error,float((probability.sum(-1)-1).abs().max()))
        reconstructed=torch.matmul(probability,value)
        delta=reconstructed-native_mix[:,start:end]
        difference_squared+=float(delta.double().square().sum())
        reference_squared+=float(native_mix[:,start:end].double().square().sum())
        maximum_difference=max(maximum_difference,float(delta.abs().max()))

        # LRP through A @ V: two independent input contributions, split half.
        scaled=rules.safe_divide(mix_relevance[:,start:end],reconstructed)
        rule_expected=scaled.double()*reconstructed.double()
        av_expected_sum+=float(rule_expected.sum());av_expected_absolute+=float(rule_expected.abs().sum())
        relevance_a=.5*probability*torch.matmul(scaled,value.transpose(-2,-1))
        value_relevance.add_(.5*value*torch.matmul(probability.transpose(-2,-1),scaled))
        attention_relevance_sum+=float(relevance_a.double().sum())
        attention_relevance_absolute+=float(relevance_a.double().abs().sum())
        # Softmax and fixed positive scale pass relevance to unscaled Q @ K.T.
        scaled_qk=rules.safe_divide(relevance_a,qk)
        rule_expected=scaled_qk.double()*qk.double()
        qk_expected_sum+=float(rule_expected.sum());qk_expected_absolute+=float(rule_expected.abs().sum())
        query_relevance[:,start:end]=.5*q*torch.matmul(scaled_qk,key)
        key_relevance.add_(.5*key*torch.matmul(scaled_qk.transpose(-2,-1),q))

        # Gradient and LRP relevance are independently computed quantities.
        gradient_a=torch.matmul(native_gradient[:,start:end],value.transpose(-2,-1))
        attention_mean=probability.mean(0)
        weighted=(gradient_a*relevance_a).clamp_min(0).mean(0)
        _finite(weighted,f'Layer {index} gradient times actual R_A')
        if (~causal).any():
            max_future=max(max_future,float(weighted.masked_select(~causal).abs().max()))
        denominator=1+attention_mean.sum(-1)
        next_rollout[start:end].add_(rollout[start:end]/denominator.double())
        next_rollout.add_(torch.matmul(rollout[start:end]/denominator.double(),attention_mean.double()))
        next_chefer.add_(torch.matmul(chefer[start:end],weighted.double()))
    relative_rms=math.sqrt(difference_squared/max(reference_squared,1e-30))
    _require(relative_rms<=RECONSTRUCTION_RELATIVE_RMS_TOLERANCE,
             f'Layer {index}: reconstructed attention relative RMS {relative_rms} exceeds {RECONSTRUCTION_RELATIVE_RMS_TOLERANCE}')
    _require(row_error<=SOFTMAX_ROW_TOLERANCE and max_future==0.,f'Layer {index}: causal/softmax probability gate failed')
    _finite(query_relevance,f'Layer {index} R_Q');_finite(key_relevance,f'Layer {index} R_K');_finite(value_relevance,f'Layer {index} R_V')
    q_relevance=rules.rope(q_native.float(),cosine_native.float(),sine_native.float(),query_relevance)
    k_relevance=rules.rope(k_native.float(),cosine_native.float(),sine_native.float(),rules.unrepeat_kv(key_relevance,kv_heads))
    v_relevance=rules.unrepeat_kv(value_relevance,kv_heads)
    normalized=capture.get(prefix+'attention_norm',device)
    source_relevance=rules.clone(normalized,[
        _linear(normalized,attention.q_proj,q_relevance.transpose(0,1).reshape(length,-1)),
        _linear(normalized,attention.k_proj,k_relevance.transpose(0,1).reshape(length,-1)),
        _linear(normalized,attention.v_proj,v_relevance.transpose(0,1).reshape(length,-1))])
    av_actual_sum=attention_relevance_sum+float(value_relevance.double().sum())
    qk_actual_sum=float(query_relevance.double().sum()+key_relevance.double().sum())
    av_absolute=attention_relevance_absolute+float(value_relevance.double().abs().sum())
    qk_absolute=float(query_relevance.double().abs().sum()+key_relevance.double().abs().sum())
    return source_relevance,next_rollout,next_chefer,dict(layer=index,rows=length,heads=heads,
        reconstruction_relative_rms=relative_rms,reconstruction_max_absolute_difference=maximum_difference,
        softmax_max_row_error=row_error,maximum_future_score=max_future,
        bilinear_output_relevance_sum=av_input_sum,attention_relevance_sum=attention_relevance_sum,
        value_relevance_sum=float(value_relevance.double().sum()),
        qk_output_relevance_sum=attention_relevance_sum,
        query_key_relevance_sum=qk_actual_sum,
        conservation_diagnostic=dict(guaranteed=False,renormalized=False,
            interpretation='Official epsilon/zero mask can lose relevance; numerical redistribution residual is separately reported, not a conservation guarantee',
            av=dict(incoming=av_input_sum,expected_after_official_rule=av_expected_sum,actual=av_actual_sum,
                stabilizer_signed_loss=av_input_sum-av_expected_sum,
                normalized_redistribution_residual=abs(av_actual_sum-av_expected_sum)/max(av_absolute,av_expected_absolute,1e-30)),
            qk=dict(incoming=attention_relevance_sum,expected_after_official_rule=qk_expected_sum,actual=qk_actual_sum,
                stabilizer_signed_loss=attention_relevance_sum-qk_expected_sum,
                normalized_redistribution_residual=abs(qk_actual_sum-qk_expected_sum)/max(qk_absolute,qk_expected_absolute,1e-30))))


@torch.no_grad()
def reverse_decoder(model,capture,target_token,progress=lambda **kw:None,query_block=QUERY_BLOCK):
    device=next(model.parameters()).device; length=capture.length; predictor=length-1
    layers=model.model.language_model.layers; segments=_segments(capture.prompt_length,length)
    hidden=capture.get('final_norm',device,start=predictor,end=length)
    seed=torch.ones((1,1),device=device,dtype=torch.float32)
    relevance=torch.zeros((length,hidden.shape[-1]),device=device,dtype=torch.float32)
    relevance[predictor:]=rules.linear(hidden,model.lm_head.weight[target_token:target_token+1].detach().float(),seed)
    rollout=torch.zeros(length,device=device,dtype=torch.float64);rollout[predictor]=1
    chefer=rollout.clone();checks=[];deepstack=[];layer_flows=[]
    for index in reversed(range(len(layers))):
        progress(stage='reverse_lrp',layer=index,decoder_layers=len(layers),context_tokens=length)
        layer=layers[index];prefix=f'layer_{index}_';flow=dict(layer=index,incoming=_summary(relevance))
        if index in capture.deepstack_layers:
            positions=torch.as_tensor(capture.visual_positions,device=device)
            output=capture.get(prefix+'output',device)
            side=capture.get(f'deepstack_{index}',device,end=capture.prompt_length)[positions]
            main,side_relevance=rules.add(output[positions],side,relevance[positions])
            relevance[positions]=main
            deepstack.append(dict(layer=index,positions=len(positions),relevance=_summary(side_relevance),
                position_signed_sum=side_relevance.double().sum(-1).cpu().tolist(),
                position_absolute_sum=side_relevance.double().abs().sum(-1).cpu().tolist(),
                destination='external visual feature boundary; no vision-encoder LRP'))
        x=capture.get(prefix+'input',device)
        residual=_native_add(capture,prefix+'input',prefix+'attention_branch',device)
        mlp_branch=capture.get(prefix+'mlp_branch',device)
        residual_relevance,mlp_relevance=_node_add(residual,mlp_branch,relevance,segments)
        activated=capture.get(prefix+'mlp_act',device,native=True)
        up=capture.get(prefix+'mlp_up',device,native=True)
        product=(activated*up).float()
        product_relevance=_linear(product,layer.mlp.down_proj,mlp_relevance)
        gate_relevance,up_relevance=rules.product(activated.float(),up.float(),product_relevance)
        normalized=capture.get(prefix+'mlp_norm',device)
        mlp_input_relevance=rules.clone(normalized,[
            _linear(normalized,layer.mlp.gate_proj,gate_relevance),
            _linear(normalized,layer.mlp.up_proj,up_relevance)])
        combined=rules.clone(residual,[residual_relevance,mlp_input_relevance])
        attention_branch=capture.get(prefix+'attention_branch',device)
        bypass,attention_relevance=_node_add(x,attention_branch,combined,segments)
        del residual,mlp_branch,activated,up,product,product_relevance,gate_relevance,up_relevance
        del normalized,mlp_input_relevance,combined,residual_relevance,mlp_relevance,attention_branch
        source_relevance,rollout,chefer,check=_attention_reverse(layer,index,capture,attention_relevance,rollout,chefer,device,query_block)
        relevance=rules.clone(x,[bypass,source_relevance]);_finite(relevance,f'Layer {index} input relevance')
        _finite(rollout,'Attention rollout');_finite(chefer,'Chefer C')
        flow['decoder_input']=_summary(relevance);layer_flows.append(flow);checks.append(check)
        del x,bypass,source_relevance,attention_relevance
    _require(bool((rollout>=0).all()) and bool((chefer>=0).all()),'Negative final map')
    _require(abs(float(rollout.sum())-1)<1e-6,'Normalized attention rollout does not conserve unit mass')
    return dict(maps=dict(attention_rollout=rollout.cpu().numpy(),chefer_grad_relevance=chefer.cpu().numpy()),
        attention_checks=checks,layer_relevance_flows=layer_flows,deepstack_side_relevance=deepstack,
        decoder_input_relevance=_summary(relevance),adaptations=dict(rules=rules.RULES,
            add_operation_scope='native prefill node and each single-token decode node separately',
            attention='native BF16 Q/K/V; reconstruct all causal A rows in FP32; derivative from native attention output gradient',
            seed='unit relevance at selected output logit; ordinary gradient of that same logit',
            softmax_scale='softmax and fixed positive attention scaling pass relevance',
            graph_boundary='decoder visual tokens and DeepStack side inputs; excludes vision-encoder/pixel LRP',
            relevance_arithmetic='FP32 redistribution using native captured activations; fixed-position RoPE recomputed in FP32 for its signed linear z-rule',
            row_product='exact reverse-layer vector contraction, mathematically equal to selected row of full matrix product',
            native_dtype_capture=True,all_rows_heads_layers=True))


def _reference(sample,step):
    original=_read(sample['source']['rollout'])
    greedy=original['result']['candidates'][0]['ids']
    _require(greedy==sample['greedy']['ids'],'Saved greedy IDs changed')
    trace=_read(sample['source_trace']['path']); nodes=trace['nodes']; chain=trace['candidates']['1']['nodes']
    top=[nodes[node]['top'][:30] for node in chain[:step+1]]
    _require(all(len(row)==30 for row in top),'Source lacks fixed top30')
    return greedy,np.array([[entry['id'] for entry in row] for row in top],dtype=np.int64),np.array([[entry['probability'] for entry in row] for row in top])


def analyze(model,processor,prompt,sample,workdir,progress=lambda **kw:None):
    """Analyze the saved greedy token at its original branch prefix.

    ``prompt`` is the untouched CPU BatchFeature returned by prepare(row,
    processor). The caller loads the original model via attribution load().
    Neither source tensors/records nor pretrained implementation are modified.
    Temporary module hooks and the existing checkpoint flag are restored.
    """
    from greedy_attribution_runtime import sequential_forward
    start=time.monotonic();device=next(model.parameters()).device
    _require(not model.training and not any(p.requires_grad for p in model.parameters()),'Expected eval model with frozen weights')
    _require(not torch.backends.cuda.matmul.allow_tf32,'Native/reconstruction gate requires TF32 disabled')
    hashes={key:_tensor_hash(value) for key,value in prompt.items()}
    _require(hashes==sample['input_hashes']==_read(sample['source']['input'])['hashes'],'Original input hashes differ')
    for name,path in sample['source'].items():
        _require(_hash(path)==sample['source_hashes'][name],f'Source {name} checksum changed')
    _require(_hash(sample['source_trace']['path'])==sample['source_trace']['sha256'],'Source top30 trace checksum changed')
    step=sample['branch']['step'];target=sample['branch']['greedy_token']
    ids,top_ids,expected=_reference(sample,step)
    _require(ids[:step]==sample['branch']['prefix_ids'] and ids[step]==target,'Target/prefix differs from saved greedy')
    gpu={key:value.to(device) for key,value in prompt.items()}
    prompt_length=int(gpu['input_ids'].shape[1]);length=prompt_length+step
    _require(gpu['input_ids'].shape[0]==1 and bool((gpu['attention_mask']==1).all()),'Expected unpadded batch-one source input')
    gpu['input_ids']=torch.cat([gpu['input_ids'],torch.tensor([ids[:step]],device=device)],dim=1)
    gpu['attention_mask']=torch.ones_like(gpu['input_ids'])
    previous_checkpoint=model._attribution_checkpoint;previous_captures=model._attribution_captures
    previous_offset=model._attribution_offset;model._attribution_checkpoint=True
    try:
        progress(stage='native_replay_validation',context_tokens=length)
        with torch.no_grad():
            baseline=sequential_forward(model,gpu,prompt_length,step+1).float()
        expected_ids=torch.tensor(ids[:step+1],device=device)
        _require(torch.equal(baseline.argmax(-1),expected_ids),'Native replay changed saved greedy argmax')
        selected=baseline.softmax(-1).gather(1,torch.as_tensor(top_ids,device=device)).cpu().numpy()
        probability_error=float(np.max(np.abs(selected-expected)))
        _require(probability_error<=REPLAY_PROBABILITY_TOLERANCE,f'Native top30 replay probability error {probability_error}')
        with NativeCapture(model,length,prompt_length,workdir) as capture:
            progress(stage='native_gradient_capture',context_tokens=length)
            logits=sequential_forward(model,gpu,prompt_length,step+1,capture=capture).float()
            graph_error=float((logits.detach()-baseline).abs().max())
            _require(graph_error==0.,f'Gradient-enabled native logits differ: {graph_error}')
            capture.phase='backward';logits[-1,target].backward();capture.phase='finished'
            capture_validation=capture.finish()
            del logits,baseline
            result=reverse_decoder(model,capture,target,progress)
        result.update(sample_key=sample['sample_key'],prompt_length=prompt_length,context_length=length,
            target=dict(step=step,token=target,predictor_position=length-1,objective='greedy_token_logit',relevance_seed=1),
            validation=dict(passed=True,input_hashes_match=True,source_checksums_match=True,
                native_greedy_argmax_match=True,source_top30_probability_max_error=probability_error,
                source_top30_probability_tolerance=REPLAY_PROBABILITY_TOLERANCE,
                gradient_native_logit_max_error=graph_error,capture=capture_validation,
                reconstruction_relative_rms_tolerance=RECONSTRUCTION_RELATIVE_RMS_TOLERANCE,
                softmax_row_tolerance=SOFTMAX_ROW_TOLERANCE,decoder_layers=len(model.model.language_model.layers),
                full_causal_rows=True,full_matrix_equivalent_target_row=True,
                lrp_conservation_guaranteed=False,
                gate_scope='Source hashes and saved greedy/top30 parity; exact gradient-on/native logits; complete native captures; attention reconstruction RMS, causal zeros and softmax row sums; finite nonnegative maps and unit-mass normalized baseline. LRP conservation is diagnostic only.'),
            elapsed_seconds=time.monotonic()-start)
        return result
    finally:
        model._attribution_checkpoint=previous_checkpoint;model._attribution_captures=previous_captures
        model._attribution_offset=previous_offset


def native_cpu_selftest():
    """Actual tiny multimodal Qwen, GQA, DeepStack and checkpointing, CPU only."""
    import tempfile
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from torch.utils.checkpoint import checkpoint
    from greedy_attribution_runtime import sequential_forward
    torch.manual_seed(13);torch.set_num_threads(1)
    config=Qwen3VLConfig(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        rope_scaling=dict(rope_type='default',mrope_section=[1,1,2],mrope_interleaved=True)),
        vision_config=dict(depth=3,hidden_size=16,intermediate_size=32,num_heads=2,patch_size=2,
            temporal_patch_size=2,spatial_merge_size=2,out_hidden_size=32,num_position_embeddings=16,
            deepstack_visual_indexes=[0,1,2]),image_token_id=61,video_token_id=59,
        vision_start_token_id=60,vision_end_token_id=62)
    model=Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    model.set_attn_implementation('eager');model._attribution_captures=[];model._attribution_offset=0
    for layer in list(model.model.language_model.layers)+list(model.model.visual.blocks):
        original=layer.forward
        def forward(*args,_original=original,**kwargs):
            captures=tuple(model._attribution_captures);offset=model._attribution_offset
            def call(*inner):
                old=[capture.segment_offset for capture in captures]
                try:
                    for capture in captures:capture.set_segment(offset)
                    return _original(*inner,**kwargs)
                finally:
                    for capture,previous in zip(captures,old):capture.set_segment(previous)
            return checkpoint(call,*args,use_reentrant=False) if model._attribution_checkpoint and torch.is_grad_enabled() else call(*args)
        layer.forward=forward
    prompt=dict(input_ids=torch.tensor([[10,60,61,62,11,12,13,14]]),
        attention_mask=torch.ones(1,8,dtype=torch.long),pixel_values=torch.randn(4,24),
        image_grid_thw=torch.tensor([[1,2,2]]))
    checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        model.to(dtype=dtype);previous=None
        for checkpointing in (False,True):
            model._attribution_checkpoint=checkpointing;model._attribution_captures=[]
            with torch.no_grad():baseline=sequential_forward(model,prompt,6,3)
            with tempfile.TemporaryDirectory(prefix='chefer_native_cpu_') as temporary:
                with NativeCapture(model,8,6,temporary) as capture:
                    logits=sequential_forward(model,prompt,6,3,capture=capture)
                    torch.testing.assert_close(logits,baseline,rtol=0,atol=0)
                    capture.phase='backward';logits[-1,3].backward();capture.phase='finished'
                    validation=capture.finish();result=reverse_decoder(model,capture,3,query_block=3)
                    _require(validation['deepstack_layers']==[0,1,2],'Tiny DeepStack capture missing')
                    if previous is not None:
                        for name,values in result['maps'].items():
                            np.testing.assert_allclose(values,previous[name],rtol=1e-7,atol=1e-9)
                    previous=result['maps']
                    checks.append(dict(dtype=str(dtype),checkpointing=checkpointing,
                        native_logit_parity=True,deepstack_layers=3,
                        attention_reconstruction_max_relative_rms=max(row['reconstruction_relative_rms'] for row in result['attention_checks'])))
    return dict(passed=True,device='cpu',native_qwen=True,gqa=True,deepstack=True,
                checkpoint_maps_equivalent=True,checks=checks)


def selftest():
    """CPU-only streamed-product and chunked bilinear equivalence checks."""
    torch.set_num_threads(1);generator=torch.Generator().manual_seed(43)
    n,h,d=11,4,6;maps=[];relevance_maps=[]
    for _ in range(4):
        q=torch.randn(h,n,d,generator=generator,dtype=torch.float64)
        k=torch.randn(h,n,d,generator=generator,dtype=torch.float64)
        v=torch.randn(h,n,d,generator=generator,dtype=torch.float64)
        scores=(q@k.transpose(-2,-1))/math.sqrt(d)
        scores.masked_fill_(torch.arange(n)[None,:]>torch.arange(n)[:,None],-torch.inf)
        a=scores.softmax(-1);r=torch.randn(h,n,d,generator=generator,dtype=torch.float64)
        ra,rv=rules.matmul(a,v,r)
        chunk_ra=[];chunk_rv=torch.zeros_like(v)
        for start in range(0,n,3):
            ar,vr=rules.matmul(a[:,start:start+3],v,r[:,start:start+3]);chunk_ra.append(ar);chunk_rv+=vr
        torch.testing.assert_close(torch.cat(chunk_ra,dim=1),ra);torch.testing.assert_close(chunk_rv,rv)
        grad=torch.randn(h,n,n,generator=generator,dtype=torch.float64)
        maps.append(a.mean(0));relevance_maps.append((grad*ra).clamp_min(0).mean(0))
    identity=torch.eye(n,dtype=torch.float64);full_a=identity.clone();full_c=identity.clone()
    for a,c in zip(maps,relevance_maps):
        aa=identity+a;aa=aa/aa.sum(-1,keepdim=True)
        full_a=aa@full_a;full_c=(identity+c)@full_c
    row_a=identity[-1].clone();row_c=row_a.clone()
    for a,c in reversed(list(zip(maps,relevance_maps))):
        next_a=torch.zeros_like(row_a);next_c=row_c.clone()
        for start in range(0,n,3):
            end=min(n,start+3);denominator=1+a[start:end].sum(-1)
            next_a[start:end]+=row_a[start:end]/denominator
            next_a+=(row_a[start:end]/denominator)@a[start:end]
            next_c+=row_c[start:end]@c[start:end]
        row_a,row_c=next_a,next_c
    torch.testing.assert_close(row_a,full_a[-1],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(row_c,full_c[-1],rtol=1e-12,atol=1e-12)
    return dict(passed=True,device='cpu',streamed_baseline_full_product=True,
                streamed_chefer_full_product=True,chunked_bilinear_full_equivalence=True,
                rules=rules.selftest(),native_qwen=native_cpu_selftest())


if __name__=='__main__':
    print(json.dumps(selftest(),ensure_ascii=False))
