"""Prespecified visual perturbation controls on a hash-selected 60-query cohort."""
from oracle_switch_core import ROOT,MODEL,read,write,sha,stamp,digest,iou,CONFIG as BASE_CONFIG
FAMILY='rollout/oracle_switch_controls'
OUT=ROOT/'artifacts/runs'/FAMILY/'pilot/60'
RESULTS=ROOT/'results'/FAMILY/'pilot/60'
CONFIG=dict(BASE_CONFIG,
    family=FAMILY,queries=60,
    selection="20 lowest SHA256('oracle_switch_controls|42|'+sample_key) per dataset among 805 eligible source queries, excluding original three extreme queries",
    common_epsilon_255=[.25,1.],
    control_methods=['targeted','opposite','random','repeat_gradient','rounded8_targeted'],
    random_seeds=[42,43,44],
    controls_support='all_video; shared support and requested RGB L-infinity; report actual clipping and physical RGB RMS',
    primary='unforced greedy oracle branch reached and delta GT IoU at prespecified common budgets; random seeds are repeated measures within query',
    precision_subset='first two hash-selected queries per dataset, fixed before outcomes',
    precision_modes=['native','patch_fp32_bf16_input','patch_fp32_float_input'],
    precision_scope='matched first Conv3D projection in FP32 with existing BF16 weights upcast; optional BF16 input rounding; output and remainder stay BF16; not a full FP32 model',
    rounding_control='round targeted perturbed RGB to uint8 then run the original processor; unchanged quantized frames reuse exact original tensor',
    cast_measurement='normalized model input before/after native BF16 cast; denominator includes duplicate packed input slots',
    outcome_failures='baseline replay failures remain explicit in the selected 60; no replacement or silent exclusion',
)
