"""Physical RGB-only sign directions for a fixed greedy-minus-oracle margin.

The packed layout is Qwen3-VL: temporal pair, merged y/x, within-block y/x,
channel, temporal member, patch y/x. Duplicate frame_ids represent one physical
RGB input: their gradients are summed and their perturbations remain identical.
No text, timestamp, position, or embedding intervention is performed here.
"""
from __future__ import annotations

import math
import torch


def _grid(grid):
    t, h, w = map(int, grid)
    if t < 1 or h < 2 or w < 2 or h % 2 or w % 2:
        raise ValueError('Expected positive time and even Qwen3-VL spatial grid')
    return t, h, w


def unpack_pixels(pixels, grid):
    """Packed [T*h*w,1536] -> [2*T,3,16*h,16*w], preserving dtype/device."""
    t, h, w = _grid(grid)
    if tuple(pixels.shape) != (t*h*w, 3*2*16*16):
        raise ValueError(f'Packed shape {tuple(pixels.shape)} does not match {grid}')
    return pixels.reshape(t,h//2,w//2,2,2,3,2,16,16).permute(
        0,6,5,1,3,7,2,4,8).reshape(2*t,3,16*h,16*w)


def pack_pixels(frames, grid):
    """Inverse of unpack_pixels; no resize, interpolation, or normalization."""
    t, h, w = _grid(grid)
    if tuple(frames.shape) != (2*t,3,16*h,16*w):
        raise ValueError(f'Frame shape {tuple(frames.shape)} does not match {grid}')
    return frames.reshape(t,2,3,h//2,2,16,w//2,2,16).permute(
        0,3,6,4,7,2,1,5,8).reshape(t*h*w,3*2*16*16)


def _normalization(tensor, mean, std):
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    mean = torch.as_tensor(mean, device=tensor.device, dtype=dtype).reshape(1,3,1,1)
    std = torch.as_tensor(std, device=tensor.device, dtype=dtype).reshape(1,3,1,1)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError('Processor mean/std must be finite, with positive std')
    return mean, std


@torch.no_grad()
def make_directions(pixels, gradient, grid, frame_ids, timestamps, mean, std):
    """Return {'sign': packed int8, 'supports': metadata, 'gradient_summary': ...}.

    Gradient is d(z_greedy-z_oracle)/d(normalized pixels). The returned sign is
    -sign(sum_aliases(gradient/std)) in physical RGB coordinates. Supports are
    nested and selected by physical-gradient L1, the local criterion appropriate
    to a fixed RGB L-infinity sign step. No gradient is recomputed here.
    """
    t,h,w = _grid(grid)
    x = unpack_pixels(pixels, grid)
    g = unpack_pixels(gradient, grid)
    if len(frame_ids) != 2*t or len(timestamps) != 2*t:
        raise ValueError('frame_ids and timestamps must cover every packed slot')
    if not bool(torch.isfinite(x).all() and torch.isfinite(g).all()):
        raise ValueError('Nonfinite pixel or gradient')
    mean_t, std_t = _normalization(x, mean, std)
    raw_gradient = g.to(std_t.dtype) / std_t
    aliases = {}
    for slot, ident in enumerate(frame_ids):
        ident = int(ident)
        aliases.setdefault(ident, []).append(slot)
    physical = {}
    sign_frames = torch.empty_like(x, dtype=torch.int8)
    for ident, slots in aliases.items():
        first = slots[0]
        for other in slots[1:]:
            if not torch.equal(x[first],x[other]) or timestamps[first] != timestamps[other]:
                raise ValueError(f'Duplicate frame_id {ident} has inconsistent input or timestamp')
        aggregate = raw_gradient[slots].sum(dim=0)
        physical[ident] = aggregate
        sign_frames[slots] = -aggregate.sign().to(torch.int8)
    reps = [slots[0] for slots in aliases.values()]
    frame_l1 = {ident: float(value.abs().sum(dtype=torch.float64)) for ident,value in physical.items()}
    frame_l2_squared = {ident: float(value.double().square().sum()) for ident,value in physical.items()}
    pair_ids = [list(dict.fromkeys(map(int,frame_ids[2*j:2*j+2]))) for j in range(t)]
    pair_l1 = [math.fsum(frame_l1[ident] for ident in ids) for ids in pair_ids]
    chosen_pair = max(range(t),key=lambda index:(pair_l1[index],-index))
    chosen_ids = pair_ids[chosen_pair]
    spatial_l1 = sum((physical[ident].abs().sum(dim=0) for ident in chosen_ids))
    cells = spatial_l1.reshape(h//2,32,w//2,32).sum(dim=(1,3),dtype=torch.float64)
    chosen_cell = int(cells.flatten().argmax())
    row,col = divmod(chosen_cell,w//2)
    bounds = [row*32,(row+1)*32,col*32,(col+1)*32]

    def support(ident,label,ids,spatial_bounds=None):
        slots = sorted(slot for source in ids for slot in aliases[source])
        y0,y1,x0,x1 = spatial_bounds or [0,h*16,0,w*16]
        local_l1 = math.fsum(float(physical[source][:,y0:y1,x0:x1].abs().sum(dtype=torch.float64)) for source in ids)
        return dict(id=ident,label=label,slots=slots,frame_ids=list(ids),
                    timestamps=[float(timestamps[slot]) for slot in slots],
                    spatial_bounds=spatial_bounds,all_physical_slots=reps,
                    physical_slots=[aliases[source][0] for source in ids],grid=[t,h,w],
                    source_frame_count=len(ids),input_slot_count=len(slots),
                    physical_gradient_l1=local_l1,
                    predicted_margin_slope_per_rgb_epsilon=-local_l1,
                    ranking='sum absolute raw-RGB gradient after summing duplicate frame slots')
    supports = [support('all_video','All video',list(aliases)),
                support('top_frame_pair','Top frame pair',chosen_ids),
                support('top_spatial_cell','Top spatial cell',chosen_ids,bounds)]
    for item in supports[1:]:
        item.update(pair=chosen_pair,pair_slots=[2*chosen_pair,2*chosen_pair+1])
    supports[2].update(row=row,col=col,cell_size=32)
    summary = dict(
        objective='greedy logit minus oracle logit',direction='negative sign of physical raw-RGB gradient',
        physical_gradient_l1=math.fsum(frame_l1.values()),
        physical_gradient_l2=math.sqrt(math.fsum(frame_l2_squared.values())),
        frame_gradient_l1=[dict(frame_id=ident,slots=aliases[ident],l1=frame_l1[ident],
                                l2=math.sqrt(frame_l2_squared[ident])) for ident in aliases],
        pair_gradient_l1=pair_l1,selected_pair=chosen_pair,selected_pair_cell_gradient_l1=cells.cpu().tolist(),
        unique_source_frames=len(aliases),input_slots=2*t,duplicate_slots=2*t-len(aliases),
        mean=mean_t.flatten().cpu().tolist(),std=std_t.flatten().cpu().tolist(),
        duplicate_policy='sum gradients by identical source frame_id; apply identical RGB change to every alias slot',
        limitation='One fixed measured gradient proposes directions; forward outcomes determine switching, not this first-order prediction.')
    return {'sign':pack_pixels(sign_frames,grid),'supports':supports,'gradient_summary':summary}


@torch.no_grad()
def apply(pixels, sign, grid, support, epsilon, mean, std):
    """Apply signed RGB L-infinity budget; negative epsilon is opposite control.

    Returns normalized packed pixels plus measured physical-RGB statistics.
    Outside-support and zero-direction normalized values are preserved exactly.
    Physical statistics count duplicate source frames once; separate input-slot
    RMS is supplied because the model contains all packed occurrences.
    """
    epsilon = float(epsilon)
    if not math.isfinite(epsilon):
        raise ValueError('epsilon must be finite')
    t,h,w = _grid(grid)
    x = unpack_pixels(pixels,grid)
    signs = unpack_pixels(sign,grid)
    if not bool(((signs == -1)|(signs == 0)|(signs == 1)).all()):
        raise ValueError('Direction must contain only -1,0,1')
    mean_t,std_t = _normalization(x,mean,std)
    raw = x.to(std_t.dtype)*std_t+mean_t
    tolerance = 8*torch.finfo(raw.dtype).eps
    if float(raw.min()) < -tolerance or float(raw.max()) > 1+tolerance:
        raise ValueError('Baseline pixels are outside physical RGB [0,1]')
    slots = list(map(int,support['slots']))
    if len(set(slots)) != len(slots) or any(i<0 or i>=2*t for i in slots):
        raise ValueError('Invalid support slots')
    y0,y1,x0,x1 = support.get('spatial_bounds') or [0,16*h,0,16*w]
    if not (0 <= y0 < y1 <=16*h and 0<=x0<x1<=16*w):
        raise ValueError('Invalid support spatial bounds')
    mask = torch.zeros((2*t,1,16*h,16*w),device=x.device,dtype=torch.bool)
    mask[slots,:,y0:y1,x0:x1] = True
    active = mask & (signs != 0)
    proposed = raw + epsilon*signs.to(raw.dtype)*mask
    clipped = proposed.clamp(0,1)
    changed_x = ((clipped-mean_t)/std_t).to(x.dtype)
    out = x.clone() if epsilon == 0 else torch.where(active,changed_x,x)
    actual_raw = out.to(raw.dtype)*std_t+mean_t
    delta = actual_raw-raw
    reps = list(map(int,support['all_physical_slots']))
    physical_delta = delta[reps]
    physical_mask = mask[reps].expand_as(physical_delta)
    clipped_mask = ((proposed<0)|(proposed>1)) & active & (epsilon != 0)
    physical_clipped = clipped_mask[reps]
    selected = physical_delta[physical_mask]
    full_n = physical_delta.numel(); support_n = selected.numel()
    rms = lambda value: float(value.double().square().mean().sqrt()) if value.numel() else 0.
    stats = dict(
        epsilon=epsilon,requested_rgb_linf=abs(epsilon),
        actual_rgb_linf=float(physical_delta.abs().max()),
        actual_rgb_linf_full=float(physical_delta.abs().max()),
        actual_rgb_linf_support=float(selected.abs().max()) if selected.numel() else 0.,
        actual_rgb_rms_full=rms(physical_delta),actual_rgb_rms_support=rms(selected),
        actual_rgb_l2=float(physical_delta.double().norm()),input_slot_rgb_rms=rms(delta),
        changed_fraction=float((physical_delta!=0).sum())/full_n,
        changed_fraction_full=float((physical_delta!=0).sum())/full_n,
        changed_fraction_support=float((selected!=0).sum())/support_n if support_n else 0.,
        clipped_fraction=float(physical_clipped.sum())/full_n,
        clipped_fraction_full=float(physical_clipped.sum())/full_n,
        clipped_fraction_support=float(physical_clipped.sum())/support_n if support_n else 0.,
        physical_components=full_n,support_physical_components=support_n,
        normalized_min=float(out.min()),normalized_max=float(out.max()),
        rgb_min=float(actual_raw.min()),rgb_max=float(actual_raw.max()),
        support_id=support['id'],outside_support_bit_equal=bool(torch.equal(out[~mask.expand_as(out)],x[~mask.expand_as(x)])),
        budget_roundoff_excess=max(0.,float(physical_delta.abs().max())-abs(epsilon)),
    )
    return pack_pixels(out,grid),stats


def selftest():
    grid=[2,4,6]; frames=torch.arange(4*3*64*96,dtype=torch.float32).reshape(4,3,64,96)
    assert torch.equal(unpack_pixels(pack_pixels(frames,grid),grid),frames)
    rgb=torch.full_like(frames,.5);rgb[0,0,0,0]=0.;rgb[0,0,0,1]=1.;rgb[3]=rgb[2]
    mean=[.2,.4,.6];std=[.3,.2,.1]
    mt=torch.tensor(mean).reshape(1,3,1,1);st=torch.tensor(std).reshape(1,3,1,1)
    normalized=(rgb-mt)/st
    raw_grad=torch.zeros_like(frames);raw_grad[0]=.01;raw_grad[2,:,32:64,64:96]=2.;raw_grad[3,:,32:64,64:96]=-1.
    # Opposing alias gradients must combine before taking a sign.
    normalized_grad=raw_grad*st
    pixels=pack_pixels(normalized,grid)
    bundle=make_directions(pixels,pack_pixels(normalized_grad,grid),grid,[10,11,12,12],[0.,.5,1.,1.],mean,std)
    assert bundle['sign'].dtype==torch.int8
    unpacked=unpack_pixels(bundle['sign'],grid)
    assert torch.equal(unpacked[2],unpacked[3]) and int(unpacked[2,0,40,70])==-1
    assert bundle['supports'][1]['pair']==1
    assert bundle['supports'][2]['spatial_bounds']==[32,64,64,96]
    for support in bundle['supports']:
        unchanged,zero=apply(pixels,bundle['sign'],grid,support,0.,mean,std)
        assert torch.equal(unchanged,pixels) and zero['actual_rgb_linf']==0
        for epsilon in [1/255,-1/255,1.]:
            updated,stats=apply(pixels,bundle['sign'],grid,support,epsilon,mean,std)
            image=unpack_pixels(updated,grid)
            assert torch.equal(image[2],image[3]) and stats['outside_support_bit_equal']
            assert stats['actual_rgb_linf']<=abs(epsilon)+1e-6
            assert stats['rgb_min']>=-1e-6 and stats['rgb_max']<=1+1e-6
    assert bundle['gradient_summary']['unique_source_frames']==3
    # Zero gradient yields a zero direction, even with a nonzero budget.
    empty=make_directions(pixels,torch.zeros_like(pixels),grid,[10,11,12,12],[0.,.5,1.,1.],mean,std)
    unchanged,_=apply(pixels,empty['sign'],grid,empty['supports'][0],1/255,mean,std)
    assert torch.equal(unchanged,pixels)
    return dict(passed=True,non_square_roundtrip=True,nonuniform_normalization=True,
                duplicate_gradient_sum_before_sign=True,duplicate_output_tied=True,
                nested_l1_support_selection=True,positive_negative_zero_budgets=True,
                physical_clipping=True,outside_support_bitwise_preserved=True,zero_gradient_no_change=True)
