"""Matched precision controls for Qwen3-VL's initial visual projection only.

The stored BF16 weights are upcast, never replaced by higher precision weights.
Both controls perform the same FP32 convolution and return its result to the
original weight dtype before the remaining vision/language model. This is not
a full-FP32 model comparison. Contexts must not overlap across threads.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

MODES = ('native', 'patch_fp32_bf16_input', 'patch_fp32_float_input')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _structure(model):
    core = model if type(model).__name__ == 'Qwen3VLModel' else getattr(model, 'model', None)
    if core is None or type(core).__name__ != 'Qwen3VLModel':
        raise TypeError('Expected Qwen3VLModel or its conditional-generation wrapper')
    visual = getattr(core, 'visual', None)
    patch = getattr(visual, 'patch_embed', None)
    if patch is None or type(patch).__name__ != 'Qwen3VLVisionPatchEmbed':
        raise TypeError('Expected Qwen3VLVisionPatchEmbed')
    if not isinstance(patch.proj, torch.nn.Conv3d):
        raise TypeError('Initial visual projection must be Conv3d')
    conv = patch.proj
    kernel = (patch.temporal_patch_size, patch.patch_size, patch.patch_size)
    if (conv.kernel_size != kernel or conv.stride != kernel
            or conv.padding != (0, 0, 0) or conv.dilation != (1, 1, 1)
            or conv.groups != 1 or conv.padding_mode != 'zeros'
            or conv.in_channels != patch.in_channels or conv.out_channels != patch.embed_dim):
        raise ValueError('Unexpected Qwen3-VL patch projection layout')
    if not callable(getattr(core, 'get_image_features', None)):
        raise TypeError('Missing image feature method')
    return core, visual, patch


def get_precision_source_metadata(model=None):
    """Record the actual imported implementation, versions and source hashes."""
    import transformers
    if model is None:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel, Qwen3VLVisionPatchEmbed)
        core_class, patch_class = Qwen3VLModel, Qwen3VLVisionPatchEmbed
    else:
        core, _, patch = _structure(model)
        core_class, patch_class = type(core), type(patch)
    def source(cls, method):
        fn = getattr(cls, method)
        path = Path(inspect.getsourcefile(fn)).resolve()
        return dict(path=str(path), sha256=_sha(path),
                    class_name=cls.__name__, method=method,
                    signature=str(inspect.signature(fn)),
                    method_source_sha256=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest())
    return dict(torch_version=torch.__version__, transformers_version=transformers.__version__,
                cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                helper_path=str(Path(__file__).resolve()), helper_sha256=_sha(__file__),
                image_features=source(core_class, 'get_image_features'),
                patch_forward=source(patch_class, 'forward'))


@contextmanager
def precision_mode(model, mode):
    """Temporarily apply a matched initial-projection precision control.

    Yields JSON-compatible metadata. Native mode changes no method/backend flag.
    BF16-input control quantizes to the original BF16 dtype then upcasts to FP32;
    float-input control preserves incoming FP32 values. Both return BF16 output.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown precision mode: {mode}')
    core, visual, patch = _structure(model)
    dtype = patch.proj.weight.dtype
    metadata = dict(mode=mode, original_weight_dtype=str(dtype),
                    scope='initial_visual_patch_projection_only',
                    full_fp32_model=False, weights='original stored weights, upcast only',
                    input_cast=('native' if mode == 'native' else
                                'original BF16 then FP32' if mode == MODES[1] else 'FP32'),
                    convolution_dtype='native' if mode == 'native' else 'torch.float32',
                    output_dtype=str(dtype))
    if mode == 'native':
        yield metadata
        return
    if dtype != torch.bfloat16 or visual.dtype != dtype:
        raise ValueError('Matched BF16-input control requires BF16 visual/patch weights')
    if getattr(core, '_oracle_precision_active', False):
        raise RuntimeError('Nested precision overrides are not supported')
    absent = object()
    old_image = core.__dict__.get('get_image_features', absent)
    old_patch = patch.__dict__.get('forward', absent)
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    metadata['tf32_before'] = dict(matmul=matmul_tf32, cudnn=cudnn_tf32)
    metadata['tf32_during'] = dict(matmul=False, cudnn=False)

    def image_features(this, pixel_values, image_grid_thw=None):
        # Deliberately omit only the original .type(self.visual.dtype) cast.
        image_embeds, deepstack = this.visual(pixel_values, grid_thw=image_grid_thw)
        sizes = (image_grid_thw.prod(-1) // this.visual.spatial_merge_size**2).tolist()
        return torch.split(image_embeds, sizes), deepstack

    def patch_forward(this, hidden_states):
        conv = this.proj
        hidden = hidden_states.view(-1, this.in_channels, this.temporal_patch_size,
                                    this.patch_size, this.patch_size)
        if mode == 'patch_fp32_bf16_input':
            hidden = hidden.to(dtype=dtype)
        hidden = hidden.float()
        # Disable autocast explicitly so external contexts cannot turn this back
        # into a BF16 convolution. The native path remains entirely untouched.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            result = F.conv3d(hidden, conv.weight.float(),
                              None if conv.bias is None else conv.bias.float(),
                              conv.stride, conv.padding, conv.dilation, conv.groups)
        return result.view(-1, this.embed_dim).to(dtype=dtype)

    try:
        core._oracle_precision_active = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        core.get_image_features = MethodType(image_features, core)
        patch.forward = MethodType(patch_forward, patch)
        yield metadata
    finally:
        if old_image is absent:
            core.__dict__.pop('get_image_features', None)
        else:
            core.get_image_features = old_image
        if old_patch is absent:
            patch.__dict__.pop('forward', None)
        else:
            patch.forward = old_patch
        core.__dict__.pop('_oracle_precision_active', None)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def precision_stats(original, perturbed, originaldtype=torch.bfloat16, chunk_size=1_048_576,
                    compute_hashes=False):
    """Chunked changes before/after the initial input dtype cast on input device.

    Units are normalized model input components, not raw RGB. All scalar
    accumulations use FP64. Optional hashes stream CPU chunks; they are disabled
    by default for repeated GPU probes. They cover contiguous tensor bytes in
    index order; shape and dtype are supplied separately. This measures input quantization,
    not the rounding/error of subsequent convolutions or model outputs.
    """
    if isinstance(originaldtype, str):
        originaldtype = getattr(torch, originaldtype.removeprefix('torch.'))
    if original.shape != perturbed.shape or original.numel() == 0 or original.device != perturbed.device:
        raise ValueError('Expected equal nonempty pixel tensor shapes')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    a, b = original.detach().reshape(-1), perturbed.detach().reshape(-1)
    n = a.numel()
    hashes = {name: hashlib.sha256() for name in ('original', 'perturbed', 'cast_original', 'cast_perturbed')} if compute_hashes else {}
    sums = torch.zeros((2, 3), dtype=torch.float64, device=a.device)
    finite = torch.ones((), dtype=torch.bool, device=a.device)
    def add(index, delta):
        sums[index, 0] += torch.count_nonzero(delta)
        sums[index, 1] += delta.square().sum()
        sums[index, 2] = torch.maximum(sums[index, 2], delta.abs().max())
    for start in range(0, n, chunk_size):
        x, y = a[start:start+chunk_size], b[start:start+chunk_size]
        finite &= torch.isfinite(x).all() & torch.isfinite(y).all()
        cx, cy = x.to(originaldtype), y.to(originaldtype)
        if compute_hashes:
            for name, tensor in (('original', x), ('perturbed', y), ('cast_original', cx), ('cast_perturbed', cy)):
                hashes[name].update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        xd, yd, cxd, cyd = x.double(), y.double(), cx.double(), cy.double()
        add(0, yd-xd); add(1, cyd-cxd)
    if not bool(finite):
        raise ValueError('Nonfinite input pixels')
    totals = sums.cpu().tolist()
    result = dict(shape=list(original.shape), components=n,
                  original_dtype=str(original.dtype), source_dtype=str(original.dtype), perturbed_dtype=str(perturbed.dtype),
                  cast_dtype=str(originaldtype), units='normalized input components',
                  accumulation=f'{original.device.type} FP64 in fixed chunks', chunk_size=chunk_size,
                  denominator='all packed input scalar components including repeated source frame slots')
    if compute_hashes:
        result['sha256']={name: digest.hexdigest() for name, digest in hashes.items()}
    for name, prefix, state in zip(('before_cast','after_cast'),('precast','postcast'),totals):
        count, squared, maximum = state
        result[name] = dict(changed_components=int(count), changed_fraction=count/n,
                            rms=math.sqrt(squared/n), max_abs=maximum)
        result[prefix+'_changed_fraction']=count/n
        result[prefix+'_linf']=maximum
        result[prefix+'_rms']=math.sqrt(squared/n)
    result['postcast_equal'] = result['input_pair_identical_after_cast'] = totals[1][0] == 0
    return result


def selftest():
    """Tiny CPU-only convolution/control test; no model loading or GPU work."""
    class Qwen3VLVisionPatchEmbed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels=1; self.temporal_patch_size=1; self.patch_size=1; self.embed_dim=1
            self.proj=torch.nn.Conv3d(1,1,1,stride=1,bias=True).to(torch.bfloat16)
            with torch.no_grad():
                self.proj.weight.fill_(1); self.proj.bias.fill_(-1)
        def forward(self, hidden_states):
            return self.proj(hidden_states.view(-1,1,1,1,1).to(self.proj.weight.dtype)).view(-1,1)
    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.patch_embed=Qwen3VLVisionPatchEmbed(); self.spatial_merge_size=1
        @property
        def dtype(self): return self.patch_embed.proj.weight.dtype
        def forward(self, pixels, grid_thw): return self.patch_embed(pixels), []
    class Qwen3VLModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.visual=Vision()
        def get_image_features(self, pixels, image_grid_thw=None):
            embeds, deep=self.visual(pixels.type(self.visual.dtype), grid_thw=image_grid_thw)
            return torch.split(embeds, image_grid_thw.prod(-1).tolist()), deep
    model=Qwen3VLModel(); patch=model.visual.patch_embed
    pixels=torch.tensor([[1.001],[1.002]],dtype=torch.float32); grid=torch.tensor([[1,1,2]])
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    original_method=model.get_image_features.__func__; original_patch=patch.forward.__func__
    weights=patch.proj.weight.detach().clone()
    with precision_mode(model,'native'):
        native=model.get_image_features(pixels,grid)[0][0]
        assert model.get_image_features.__func__ is original_method
        assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    with precision_mode(model,'patch_fp32_bf16_input'):
        cast=model.get_image_features(pixels,grid)[0][0]
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    with precision_mode(model,'patch_fp32_float_input'):
        preserved=model.get_image_features(pixels,grid)[0][0]
    expected=(pixels-1).to(torch.bfloat16)
    assert torch.equal(native,cast) and torch.count_nonzero(cast)==0
    assert torch.equal(preserved,expected) and torch.count_nonzero(preserved)==2
    try:
        with precision_mode(model,'patch_fp32_float_input'): raise RuntimeError('expected test exception')
    except RuntimeError as exc:
        assert str(exc)=='expected test exception'
    assert model.get_image_features.__func__ is original_method and patch.forward.__func__ is original_patch
    assert 'get_image_features' not in model.__dict__ and 'forward' not in patch.__dict__
    assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    assert torch.equal(weights,patch.proj.weight)
    stats=precision_stats(torch.ones_like(pixels),pixels,chunk_size=1,compute_hashes=True)
    assert stats['before_cast']['changed_components']==2 and stats['input_pair_identical_after_cast']
    assert stats['sha256']['cast_original']==stats['sha256']['cast_perturbed']
    assert stats['sha256']['original']!=stats['sha256']['perturbed']
    return dict(passed=True,cpu_only=True,native_untouched=True,matched_conv_cast_controls=True,
                restoration_on_normal_and_exception_exit=True,weights_unchanged=True,
                chunked_cast_stats_and_hashes=True)


if __name__ == '__main__':
    import json
    print(json.dumps(selftest(),indent=2))
