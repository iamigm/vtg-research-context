"""Explicit relevance propagation rules for the Qwen3-VL decoder adapter.

These are relevance redistribution operators, not activation-gradient scores.
The alpha=1 linear, Add, Clone, normalization/activation pass-through, and
half-split bilinear conventions follow Chefer et al.'s official implementation:
https://github.com/hila-chefer/Transformer-Explainability/blob/main/modules/layers_ours.py
https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_LRP.py

Qwen-specific extensions are explicit: RMSNorm and SiLU use pass-through;
SwiGLU uses the same half-split bilinear rule as attention; fixed RoPE uses a
signed linear z-rule; shared GQA aliases sum their relevance. These extensions
are not claimed to be a published, exact Chefer rule for Qwen3-VL.

All functions accept tensors on their existing device, build no autograd graph,
and leave inputs unchanged. Use float32 or float64; low precision can erase the
official 1e-9 stabilizer. Run this file for deterministic CPU-only self-tests.
"""
from __future__ import annotations

import json
from typing import Iterable

import torch
import torch.nn.functional as F

EPSILON = 1e-9
RULES = {
    "linear": "alpha=1 positive products: x+ w+ and x- w-; bias omitted as official",
    "add": "signed z-rule followed by GLOBAL absolute branch-sum renormalization",
    "bilinear": "signed z-rule to both inputs, each multiplied by one half",
    "clone": "sum alias relevance through official safe_divide and source activation",
    "normalization_activation": "pass-through; RMSNorm/SiLU are documented Qwen extensions",
    "softmax": "pass-through, as official Chefer implementation",
    "rope": "fixed signed linear z-rule, with cosine/sine held constant",
    "gqa": "sum relevance over repeated KV-head aliases",
    "zero_denominator": "official zero mask; relevance on an exactly zero contribution is lost",
}


def _floating(*tensors: torch.Tensor) -> None:
    if not all(isinstance(x, torch.Tensor) and x.dtype in (torch.float32, torch.float64) for x in tensors):
        raise TypeError("LRP rules require float32/float64 tensors; cast explicitly before redistribution")


@torch.no_grad()
def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Preserve the official 1e-9 denominator convention and exact-zero mask.

    The stabilizer is not sign-adjusted. In particular, small negative values
    near -1e-9 have the official implementation's asymmetric behavior.
    """
    _floating(numerator, denominator)
    stable = denominator.clamp_min(EPSILON) + denominator.clamp_max(EPSILON)
    stable = torch.where(stable == 0, torch.full_like(stable, EPSILON), stable)
    return (numerator / stable) * (denominator != 0).to(denominator.dtype)


@torch.no_grad()
def linear(x: torch.Tensor, weight: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    """Alpha=1 linear redistribution; ``weight`` is [output, input].

    Bias is deliberately absent, matching the official Linear.relprop rule.
    This rule can be chunked over independent batch/token rows.
    """
    _floating(x, weight, relevance)
    xp, xn = x.clamp_min(0), x.clamp_max(0)
    wp, wn = weight.clamp_min(0), weight.clamp_max(0)
    positive_contribution = F.linear(xp, wp) + F.linear(xn, wn)
    scaled = safe_divide(relevance, positive_contribution)
    return xp * torch.matmul(scaled, wp) + xn * torch.matmul(scaled, wn)


@torch.no_grad()
def linear_z(x: torch.Tensor, weight: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    """Signed fixed-linear z-rule; weight layout is [output, input]."""
    _floating(x, weight, relevance)
    scaled = safe_divide(relevance, F.linear(x, weight))
    return x * torch.matmul(scaled, weight)


@torch.no_grad()
def add(x: torch.Tensor, y: torch.Tensor, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Official Add rule, including whole-operation branch renormalization.

    IMPORTANT: do not call independently on token chunks. The two branch sums
    are global over the entire Add operation (all examples/tokens/channels).
    Exact cancellation/zero branches remain zero under the official rule.
    """
    _floating(x, y, relevance)
    scaled = safe_divide(relevance, x + y)
    rx = x * scaled.sum_to_size(x.shape)
    ry = y * scaled.sum_to_size(y.shape)
    sx, sy = rx.sum(), ry.sum()
    total = sx.abs() + sy.abs()
    target_x = safe_divide(sx.abs(), total) * relevance.sum()
    target_y = safe_divide(sy.abs(), total) * relevance.sum()
    return rx * safe_divide(target_x, sx), ry * safe_divide(target_y, sy)


@torch.no_grad()
def product(x: torch.Tensor, y: torch.Tensor, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise bilinear z-rule with one half assigned to each input.

    Broadcasting gradients are summed back to each input shape. Applying this
    rule to SwiGLU is a documented architectural extension.
    """
    _floating(x, y, relevance)
    scaled = safe_divide(relevance, x * y)
    rx = .5 * x * (scaled * y).sum_to_size(x.shape)
    ry = .5 * y * (scaled * x).sum_to_size(y.shape)
    return rx, ry


@torch.no_grad()
def matmul(a: torch.Tensor, b: torch.Tensor, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Matrix bilinear z-rule for ``a @ b``; both relevance branches get 1/2.

    Supports broadcast leading batch/head dimensions and matrix inputs only.
    For attention, a=A,b=V gives the actual LRP relevance R_A and R_V. They
    must not be confused with A times a target-score gradient.
    """
    _floating(a, b, relevance)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("matmul relevance expects two matrices or batched matrices")
    scaled = safe_divide(relevance, torch.matmul(a, b))
    ga = torch.matmul(scaled, b.transpose(-2, -1)).sum_to_size(a.shape)
    gb = torch.matmul(a.transpose(-2, -1), scaled).sum_to_size(b.shape)
    return .5 * a * ga, .5 * b * gb


def relpass(relevance: torch.Tensor) -> torch.Tensor:
    """Pass relevance unchanged through softmax/norm/activation conventions."""
    _floating(relevance)
    return relevance


@torch.no_grad()
def clone(x: torch.Tensor, relevances: Iterable[torch.Tensor]) -> torch.Tensor:
    """Official Clone merge, including its zero-activation convention."""
    _floating(x)
    scaled = torch.zeros_like(x)
    count = 0
    for relevance in relevances:
        _floating(relevance)
        scaled.add_(safe_divide(relevance, x))
        count += 1
    if not count:
        raise ValueError("Clone merge requires at least one relevance branch")
    return x * scaled


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen split-half rotation [-x_second, x_first]."""
    if x.shape[-1] % 2:
        raise ValueError("Rotary head dimension must be even")
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@torch.no_grad()
def rope(x: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    """Signed z-rule for y=x*cosine+rotate_half(x)*sine, fixed positions.

    The transpose of split-half rotation is its negative. Cosine and sine must
    already have the broadcasting axes used in the native Qwen operation.
    No relevance is assigned to fixed position encodings.
    """
    _floating(x, cosine, sine, relevance)
    output = x * cosine + rotate_half(x) * sine
    scaled = safe_divide(relevance, output)
    return x * (scaled * cosine - rotate_half(scaled * sine))


@torch.no_grad()
def unrepeat_kv(relevance: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """Sum contiguous repeated GQA aliases in [..., H, sequence, head_dim]."""
    _floating(relevance)
    if relevance.ndim < 3 or kv_heads < 1 or relevance.shape[-3] % kv_heads:
        raise ValueError("Invalid repeated-head relevance geometry")
    repeats = relevance.shape[-3] // kv_heads
    return relevance.reshape(*relevance.shape[:-3], kv_heads, repeats, *relevance.shape[-2:]).sum(dim=-3)


def selftest() -> dict:
    """Compare analytical rules against independent CPU autograd definitions."""
    torch.set_num_threads(1)
    generator = torch.Generator(device='cpu').manual_seed(1942)
    dtype = torch.float64
    checks = []

    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=dtype)

    def near(a, b, name, atol=1e-11):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=atol)
        checks.append(name)

    def simple_reference(inputs, relevance, operation, half=False):
        variables = [value.detach().clone().requires_grad_(True) for value in inputs]
        output = operation(*variables)
        gradient = torch.autograd.grad(output, variables, safe_divide(relevance, output))
        return [(value * grad * (.5 if half else 1)).detach() for value, grad in zip(variables, gradient)]

    x, w, r = rand(3, 5), rand(4, 5), rand(3, 4)
    xp, xn = x.clamp_min(0).requires_grad_(), x.clamp_max(0).requires_grad_()
    wp, wn = w.clamp_min(0), w.clamp_max(0)
    zp, zn = F.linear(xp, wp), F.linear(xn, wn)
    coefficient = safe_divide(r, zp + zn)
    expected = xp * torch.autograd.grad(zp, xp, coefficient)[0] + xn * torch.autograd.grad(zn, xn, coefficient)[0]
    near(linear(x, w, r), expected, 'alpha1_matches_autograd')
    positive = linear(torch.ones(2, 3, dtype=dtype), torch.ones(2, 3, dtype=dtype), torch.ones(2, 2, dtype=dtype))
    near(positive.sum(), torch.tensor(4., dtype=dtype), 'alpha1_conservation_with_stabilizer', 2e-9)
    near(linear(torch.zeros_like(x), w, r), torch.zeros_like(x), 'alpha1_zero_input')

    x, y, r = rand(2, 4), rand(2, 4), rand(2, 4)
    a, b = simple_reference([x, y], r, torch.add)
    denominator = a.sum().abs() + b.sum().abs()
    factor_a = safe_divide(a.sum().abs(), denominator) * r.sum()
    factor_b = safe_divide(b.sum().abs(), denominator) * r.sum()
    expected_a, expected_b = a * safe_divide(factor_a, a.sum()), b * safe_divide(factor_b, b.sum())
    actual_a, actual_b = add(x, y, r)
    near(actual_a, expected_a, 'add_global_branch_x')
    near(actual_b, expected_b, 'add_global_branch_y')
    near(actual_a.sum() + actual_b.sum(), r.sum(), 'add_conservation_with_stabilizer', 1e-8)
    zeros = add(x, -x, r)
    near(zeros[0], torch.zeros_like(x), 'add_exact_cancellation_x')
    near(zeros[1], torch.zeros_like(x), 'add_exact_cancellation_y')
    # A whole-operation normalization must not accidentally become per row.
    gx = torch.tensor([[1., 1.], [-1., -1.]], dtype=dtype)
    gy = torch.tensor([[1., 1.], [2., 2.]], dtype=dtype)
    gr = torch.ones_like(gx)
    globally, _ = add(gx, gy, gr)
    separately = [add(gx[i], gy[i], gr[i]) for i in range(2)]
    assert not torch.allclose(globally, torch.stack([pair[0] for pair in separately]))
    checks.append('add_is_global_not_rowwise')

    x, y, r = rand(2, 3), rand(1, 3), rand(2, 3)
    actual = product(x, y, r); expected = simple_reference([x, y], r, torch.mul, half=True)
    for index in range(2): near(actual[index], expected[index], f'product_broadcast_{index}')
    near(sum(value.sum() for value in actual), r.sum(), 'product_total_half_split', 1e-7)
    a, b, r = rand(2, 3, 4), rand(1, 4, 5), rand(2, 3, 5)
    actual = matmul(a, b, r); expected = simple_reference([a, b], r, torch.matmul, half=True)
    for index in range(2): near(actual[index], expected[index], f'matmul_broadcast_{index}')
    near(sum(value.sum() for value in actual), r.sum(), 'matmul_total_half_split', 1e-7)

    x, w, r = rand(2, 4), rand(5, 4), rand(2, 5)
    expected = simple_reference([x], r, lambda value: F.linear(value, w))[0]
    near(linear_z(x, w, r), expected, 'signed_linear_z')
    x, angle, r = rand(1, 2, 3, 6), rand(1, 1, 3, 6), rand(1, 2, 3, 6)
    cosine, sine = angle.cos(), angle.sin()
    expected = simple_reference([x], r, lambda value: value * cosine + rotate_half(value) * sine)[0]
    near(rope(x, cosine, sine, r), expected, 'fixed_rope_transpose_rule')
    near(rope(x, torch.ones_like(cosine), torch.zeros_like(sine), r), x * safe_divide(r, x), 'rope_identity')
    zero = torch.zeros_like(x)
    near(rope(zero, cosine, sine, r), zero, 'rope_zero_input')

    x, r1, r2 = rand(2, 3), rand(2, 3), rand(2, 3)
    expected = simple_reference([x], r1 + r2, lambda value: value)[0]
    near(clone(x, [r1, r2]), expected, 'clone_merge')
    near(clone(torch.zeros_like(x), [r1, r2]), torch.zeros_like(x), 'clone_zero_mask')
    repeated = rand(1, 8, 3, 4)
    merged = unrepeat_kv(repeated, 2)
    near(merged, repeated.reshape(1, 2, 4, 3, 4).sum(2), 'gqa_contiguous_alias_order')
    near(merged.sum(), repeated.sum(), 'gqa_relevance_conservation')
    assert relpass(r1) is r1
    checks.append('normalization_activation_softmax_pass')

    denominator = torch.tensor([0., 1., -1., 1e-10, -1e-10, -1e-9], dtype=dtype)
    divided = safe_divide(torch.ones_like(denominator), denominator)
    assert divided[0] == 0 and torch.isfinite(divided).all()
    near(divided[-1], torch.tensor(1e9, dtype=dtype), 'official_negative_epsilon_edge', 1e-6)
    checks.append('safe_divide_exact_zero_finite')
    return {'passed': True, 'device': 'cpu', 'dtype': str(dtype), 'checks': checks, 'rules': RULES,
            'conservation_limit': 'The official epsilon and exact-zero mask can lose relevance; no artificial renormalization is added except official Add.'}


if __name__ == '__main__':
    print(json.dumps(selftest(), ensure_ascii=False))
