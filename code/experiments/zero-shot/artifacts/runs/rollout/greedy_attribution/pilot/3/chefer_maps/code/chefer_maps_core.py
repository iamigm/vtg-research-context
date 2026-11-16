"""Frozen scope for decoder attention/LRP comparison on existing greedy inputs."""
from pathlib import Path
from greedy_attribution_core import ROOT, MODEL, read, write, sha, digest, stamp

RELATIVE = 'rollout/greedy_attribution/pilot/3/chefer_maps'
OUT = ROOT/'artifacts/runs'/RELATIVE
RESULTS = ROOT/'results'/RELATIVE
REFERENCE = ROOT.parent/'_shared/transformer_explainability'
REVISION = 'c3e578f76b954e8528afeaaee26de3f07e3fe559'
CONFIG = dict(scope='pilot', scope_queries=3, source_queries=900, analysis='chefer_maps',
    model=str(MODEL), source_revision=REVISION, target='original greedy branch token logit',
    target_step=5, relevance_seed='one-hot 1 at selected output token',
    candidate_top_k=30, candidates='original fixed top30; no candidate expansion',
    input_policy='unchanged original source input, prefix, frame IDs, timestamps and spatial resolution',
    measurement='36 decoder layers, all causal attention rows; exact target-row products',
    baseline='product RowNorm(I + mean_heads(A))',
    final_c='product (I + mean_heads(positive(gradient_logit_A * R_A)))',
    product_order='M_last @ ... @ M_first, represented by predictor row; no row normalization except baseline',
    lrp='official alpha=1 positive-product Linear, renormalized Add, half bilinear split',
    adaptations=['RMSNorm and SiLU relevance pass-through analogous to official LayerNorm/GELU',
                 'fixed RoPE linear redistribution; SwiGLU half bilinear split; GQA alias contributions summed',
                 'DeepStack additions split and side-input relevance recorded',
                 'decoder visual-token attribution only, not vision-encoder or raw-pixel LRP'],
    seed=42, min_gpu_free_mib=98304, min_host_headroom_gib=64, min_disk_free_gib=100)

def setup_selection():
    path=OUT/'metadata/selection.json'
    if path.exists(): return read(path)
    source=ROOT/'artifacts/runs/rollout/oracle_switch/pilot/3/metadata/selection.json'
    original=read(source); samples=[]
    for sample in original['samples']:
        sample=dict(sample); attribution=read(sample['source']['attribution']); info=attribution['input']
        t,h,w=info['grid']; timestamps=list(info['timestamps'])
        timestamps+=timestamps[-1:]*(2*t-len(timestamps))
        sample['display']=dict(video_url=sample['video_url'], grid=[t,h,w], timestamps=timestamps,
            input_hw=[h*16,w*16],target_token_text=sample['branch']['greedy_token_text'],
            prompt_length=info['prompt_length'],duration=info['duration'])
        samples.append(sample)
    result=dict(scope='pilot',scope_queries=3,source_queries=900,samples=samples,
        source_selection=dict(path=str(source),sha256=sha(source)),
        policy='Reuse the three previously inspected extreme cases; diagnostic comparison, not representative sample',
        target='greedy token at the saved first oracle divergence, fixed prefix; logit objective')
    write(path,result);write(OUT/'metadata/config.json',CONFIG)
    return result
