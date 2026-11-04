# Rollout reconsideration · pilot300

완료: 300 query, dataset별100. 3개 접근과 direct regeneration control을 같은 고정 cohort에서 평가했다. 생성 input hash는 기존 deterministic rollout과 일치하며 GT는 보고 단계에서만 사용했다.

[실험 설계](../600/SPEC.md)

## 지표와 비교 기준

mIoU는 300 query의 temporal IoU 평균, R@τ는 IoU≥τ query 비율이다. IoU=intersection duration/union duration이며 높을수록 좋다. Δ는 같은 query의 greedy 대비 차이, Δpp는 성공률 차이×100이다. failure는 파싱 실패·invalid interval·EOS 미도달 비율이고0점으로 포함한다. model seconds는 CUDA 동기화로 측정한 의사결정 model wall time이며 공통 video prefill과 사후 probability diagnostic을 제외한다. decoder tokens는 기존 proposal generation 비용과 새 review의 prompt·generated token·KV replay를 합산한다. probability diagnostics 비용은 별도 원자료에 보존한다. 기존 baseline은 추가 review time이0이다. 평가 호출 수를 맞춘 비교가 FLOPs 일치를 뜻하지 않는다.

## 전체 성능

| method | n | mIoU (Δ) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | model evaluations | decoder tokens (proposal+review) | review model seconds | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 300 | 0.4870 (+0.0000) | 0.6467 (+0.00) | 0.5133 (+0.00) | 0.3500 (+0.00) | 0.0 | 16.9 | 0.000 | 0.000 |
| max_probability_pool8 | 300 | 0.5012 (+0.0142) | 0.6700 (+2.33) | 0.5400 (+2.67) | 0.3633 (+1.33) | 0.0 | 85.8 | 0.000 | 0.000 |
| probability_weighted_medoid_pool8 | 300 | 0.5046 (+0.0176) | 0.6767 (+3.00) | 0.5400 (+2.67) | 0.3633 (+1.33) | 0.0 | 85.8 | 0.000 | 0.000 |
| direct_pool8 | 300 | 0.4957 (+0.0087) | 0.6667 (+2.00) | 0.5300 (+1.67) | 0.3567 (+0.67) | 1.0 | 375.8 | 0.822 | 0.000 |
| agent_pool8 | 300 | 0.4854 (-0.0016) | 0.6400 (-0.67) | 0.5133 (+0.00) | 0.3533 (+0.33) | 6.0 | 2529.0 | 6.717 | 0.000 |
| verifier_pool8 | 300 | 0.4922 (+0.0052) | 0.6533 (+0.67) | 0.5133 (+0.00) | 0.3533 (+0.33) | 7.0 | 646.4 | 1.089 | 0.000 |
| tts_pool8_select4 | 300 | 0.4907 (+0.0037) | 0.6567 (+1.00) | 0.5200 (+0.67) | 0.3533 (+0.33) | 7.0 | 780.0 | 1.093 | 0.000 |
| tts_pool8_select8 | 300 | 0.4907 (+0.0038) | 0.6567 (+1.00) | 0.5200 (+0.67) | 0.3533 (+0.33) | 14.7 | 1267.4 | 1.484 | 0.000 |
| direct_pool32 | 300 | 0.4853 (-0.0017) | 0.6433 (-0.33) | 0.5033 (-1.00) | 0.3433 (-0.67) | 1.0 | 619.6 | 0.827 | 0.000 |
| agent_pool32 | 300 | 0.4828 (-0.0041) | 0.6433 (-0.33) | 0.5067 (-0.67) | 0.3433 (-0.67) | 6.3 | 3224.6 | 7.087 | 0.000 |
| verifier_pool32 | 300 | 0.4907 (+0.0038) | 0.6533 (+0.67) | 0.5000 (-1.33) | 0.3500 (+0.00) | 14.9 | 1215.5 | 1.490 | 0.000 |
| tts_pool32_select4 | 300 | 0.4824 (-0.0045) | 0.6500 (+0.33) | 0.4967 (-1.67) | 0.3433 (-0.67) | 7.0 | 1028.3 | 1.110 | 0.000 |
| tts_pool32_select8 | 300 | 0.4782 (-0.0087) | 0.6467 (+0.00) | 0.4900 (-2.33) | 0.3367 (-1.33) | 14.9 | 1529.7 | 1.497 | 0.000 |
| max_probability_pool32 | 300 | 0.4997 (+0.0128) | 0.6700 (+2.33) | 0.5367 (+2.33) | 0.3600 (+1.00) | 0.0 | 300.0 | 0.000 | 0.000 |
| probability_weighted_medoid_pool32 | 300 | 0.5051 (+0.0181) | 0.6733 (+2.67) | 0.5367 (+2.33) | 0.3633 (+1.33) | 0.0 | 300.0 | 0.000 | 0.000 |
| oracle_pool8 | 300 | 0.6435 (+0.1565) | 0.8267 (+18.00) | 0.6933 (+18.00) | 0.5133 (+16.33) | 0.0 | 85.8 | 0.000 | 0.000 |
| oracle_pool32 | 300 | 0.7051 (+0.2182) | 0.8733 (+22.67) | 0.7500 (+23.67) | 0.6100 (+26.00) | 0.0 | 300.0 | 0.000 | 0.000 |

## Paired comparison · 95% bootstrap CI

video 단위 paired bootstrap의 방법 간 차이다. CI가0을 포함하면 순위 차이를 확정하지 않는다.

| comparison | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |
|---|---|---|
| verifier_pool8 minus agent_pool8 | +0.0068 [-0.0007, +0.0147] | +0.0000 [-0.0167, +0.0198] |
| tts_pool8_select4 minus agent_pool8 | +0.0053 [-0.0039, +0.0154] | +0.0067 [-0.0134, +0.0270] |
| tts_pool8_select4 minus verifier_pool8 | -0.0015 [-0.0092, +0.0062] | +0.0067 [-0.0100, +0.0234] |
| verifier_pool32 minus agent_pool32 | +0.0079 [-0.0073, +0.0207] | -0.0067 [-0.0367, +0.0233] |
| tts_pool32_select8 minus agent_pool32 | -0.0046 [-0.0170, +0.0083] | -0.0167 [-0.0438, +0.0100] |
| tts_pool32_select8 minus verifier_pool32 | -0.0125 [-0.0241, -0.0013] | -0.0100 [-0.0332, +0.0134] |
| tts_pool8_select8 minus tts_pool8_select4 | +0.0000 [-0.0027, +0.0032] | +0.0000 [+0.0000, +0.0000] |
| tts_pool32_select4 minus tts_pool8_select4 | -0.0083 [-0.0212, +0.0059] | -0.0233 [-0.0468, +0.0033] |
| tts_pool32_select8 minus tts_pool8_select8 | -0.0125 [-0.0261, +0.0013] | -0.0300 [-0.0537, -0.0034] |
| tts_pool32_select8 minus tts_pool32_select4 | -0.0042 [-0.0118, +0.0022] | -0.0067 [-0.0234, +0.0067] |

## Clustering sensitivity

semantic gold가 아닌 temporal proxy의 threshold 민감도다. 분모는 두 응답이 모두 valid인 query다.

| method | n | switch @.3 | switch @.5 | switch @.7 |
|---|---:|---:|---:|---:|
| greedy | 300 | 0.0000 | 0.0000 | 0.0000 |
| max_probability_pool8 | 300 | 0.1200 | 0.1500 | 0.1933 |
| probability_weighted_medoid_pool8 | 300 | 0.0933 | 0.1333 | 0.1867 |
| direct_pool8 | 300 | 0.0233 | 0.0433 | 0.0500 |
| agent_pool8 | 300 | 0.0133 | 0.0267 | 0.0500 |
| verifier_pool8 | 300 | 0.0200 | 0.0200 | 0.0300 |
| tts_pool8_select4 | 300 | 0.0200 | 0.0267 | 0.0567 |
| tts_pool8_select8 | 300 | 0.0167 | 0.0267 | 0.0567 |
| direct_pool32 | 300 | 0.0300 | 0.0533 | 0.0933 |
| agent_pool32 | 300 | 0.0367 | 0.0700 | 0.1100 |
| verifier_pool32 | 300 | 0.0367 | 0.0567 | 0.1067 |
| tts_pool32_select4 | 300 | 0.0133 | 0.0367 | 0.0800 |
| tts_pool32_select8 | 300 | 0.0233 | 0.0500 | 0.0967 |
| max_probability_pool32 | 300 | 0.1433 | 0.1833 | 0.2600 |
| probability_weighted_medoid_pool32 | 300 | 0.1433 | 0.2067 | 0.2967 |
| oracle_pool8 | 300 | 0.1633 | 0.2267 | 0.3533 |
| oracle_pool32 | 300 | 0.2467 | 0.3833 | 0.5400 |

## Model selection와 final regeneration

verifier/TTS가 실제로 선택한 기존 candidate의 IoU와 final regeneration의 IoU를 구분한다. agent의 inspection 대상은 선택 결과로 간주하지 않는다.

| method | n selected | selected mIoU | final mIoU | model order disagreement |
|---|---:|---:|---:|---:|
| verifier_pool8 | 300 | 0.4907 | 0.4922 | — |
| tts_pool8_select4 | 300 | 0.4941 | 0.4907 | 0.32589285714285715 |
| tts_pool8_select8 | 300 | 0.4904 | 0.4907 | 0.3821532492725509 |
| verifier_pool32 | 300 | 0.4947 | 0.4907 | — |
| tts_pool32_select4 | 300 | 0.4899 | 0.4824 | 0.26339285714285715 |
| tts_pool32_select8 | 300 | 0.4865 | 0.4782 | 0.32885261641862695 |

## event change와 boundary refinement

temporal clustering은 semantic event identity gold가 아닌 proxy다. switch는 valid 두 응답의 cluster 변경, event recovery는 greedy IoU<0.1에서 switch 후 IoU≥0.5가 된 경우다. same-cluster boundary 지표는 greedy IoU≥0.5인 paired-valid query에 조건부이며 분모를 별도로 표시한다. endpoint MAE는 두 endpoint의 절대오차 평균(seconds), 낮을수록 좋다.

| method | switch / paired-valid | beneficial / switch | event recovery / greedy IoU<.1 | harmful / greedy IoU≥.5 | same-cluster boundary n | MAE before → after | ΔIoU |
|---|---:|---:|---:|---:|---:|---|---:|
| greedy | 0/300 | 0/0 | 0/65 | 0/154 | 154 | 1.625 → 1.625 | +0.0000 |
| max_probability_pool8 | 45/300 | 18/45 | 7/65 | 9/154 | 148 | 1.642 → 1.768 | -0.0093 |
| probability_weighted_medoid_pool8 | 40/300 | 19/40 | 4/65 | 7/154 | 147 | 1.512 → 1.632 | -0.0091 |
| direct_pool8 | 13/300 | 9/13 | 2/65 | 3/154 | 153 | 1.632 → 1.889 | -0.0054 |
| agent_pool8 | 8/300 | 3/8 | 1/65 | 3/154 | 153 | 1.621 → 1.630 | -0.0045 |
| verifier_pool8 | 6/300 | 4/6 | 1/65 | 1/154 | 153 | 1.603 → 1.586 | +0.0017 |
| tts_pool8_select4 | 8/300 | 5/8 | 2/65 | 3/154 | 153 | 1.632 → 1.850 | -0.0027 |
| tts_pool8_select8 | 8/300 | 6/8 | 2/65 | 3/154 | 152 | 1.636 → 1.868 | -0.0045 |
| direct_pool32 | 16/300 | 8/16 | 3/65 | 8/154 | 150 | 1.624 → 1.602 | -0.0080 |
| agent_pool32 | 21/300 | 6/21 | 1/65 | 7/154 | 147 | 1.650 → 1.899 | -0.0091 |
| verifier_pool32 | 17/300 | 8/17 | 2/65 | 7/154 | 148 | 1.620 → 1.618 | -0.0032 |
| tts_pool32_select4 | 11/300 | 5/11 | 1/65 | 8/154 | 150 | 1.635 → 1.635 | -0.0112 |
| tts_pool32_select8 | 15/300 | 5/15 | 1/65 | 10/154 | 149 | 1.629 → 1.644 | -0.0118 |
| max_probability_pool32 | 55/300 | 23/55 | 7/65 | 12/154 | 144 | 1.482 → 1.635 | -0.0161 |
| probability_weighted_medoid_pool32 | 62/300 | 25/62 | 6/65 | 8/154 | 141 | 1.555 → 1.601 | -0.0153 |
| oracle_pool8 | 68/300 | 67/68 | 18/65 | 0/154 | 148 | 1.492 → 0.815 | +0.0750 |
| oracle_pool32 | 115/300 | 114/115 | 23/65 | 0/154 | 130 | 1.505 → 0.523 | +0.1001 |

## token distribution과 path probability

같은 token prefix를 review context에서 재평가했다. 아래 positive log-odds Δ는 원래 선택에 비해 대체 token을 더 선호하게 됐다는 뜻이다. GT에 가까운지와는 별도다. JS divergence는 전체 어휘 분포 차이이며0이면 같다.

| method | axis | n probes | mean JS divergence | mean alternative probability Δ | mean log-odds Δ | alternative full-path preference Δ |
|---|---|---:|---:|---:|---:|---:|
| direct_pool8 | event | 115 | 0.161526 | -0.107268 | -5.1826 | -6.1878 |
| direct_pool8 | boundary | 300 | 0.172956 | -0.193632 | -3.0104 | -3.4008 |
| agent_pool8 | event | 115 | 0.185679 | -0.134331 | -10.7522 | -11.2070 |
| agent_pool8 | boundary | 300 | 0.224171 | -0.198524 | -6.4142 | -6.6443 |
| verifier_pool8 | event | 115 | 0.180939 | -0.115648 | -6.2935 | -7.3398 |
| verifier_pool8 | boundary | 300 | 0.194990 | -0.208276 | -3.8342 | -4.2129 |
| tts_pool8_select4 | event | 115 | 0.164528 | -0.120421 | -5.2489 | -6.2948 |
| tts_pool8_select4 | boundary | 300 | 0.178140 | -0.197854 | -3.1167 | -3.5537 |
| tts_pool8_select8 | event | 115 | 0.163436 | -0.127181 | -5.0913 | -6.1146 |
| tts_pool8_select8 | boundary | 300 | 0.171862 | -0.189365 | -2.9033 | -3.3474 |
| direct_pool32 | event | 115 | 0.148259 | -0.064530 | -4.0793 | -5.4496 |
| direct_pool32 | boundary | 300 | 0.182004 | -0.164311 | -2.6588 | -2.8447 |
| agent_pool32 | event | 115 | 0.193715 | -0.080396 | -8.8033 | -9.9471 |
| agent_pool32 | boundary | 300 | 0.239613 | -0.183264 | -5.8229 | -6.1250 |
| verifier_pool32 | event | 115 | 0.175691 | -0.059851 | -5.4598 | -6.8792 |
| verifier_pool32 | boundary | 300 | 0.200419 | -0.165031 | -3.2208 | -3.3331 |
| tts_pool32_select4 | event | 115 | 0.158410 | -0.061582 | -4.3272 | -5.7245 |
| tts_pool32_select4 | boundary | 300 | 0.185405 | -0.166555 | -2.7462 | -3.0002 |
| tts_pool32_select8 | event | 115 | 0.156835 | -0.064993 | -4.3576 | -5.7462 |
| tts_pool32_select8 | boundary | 300 | 0.181907 | -0.167559 | -2.6646 | -2.9019 |

## 추가 분석

[dataset·video duration·GT ratio·교차표](STRATA.md). [video별 paired bootstrap 원자료](../../../../../artifacts/runs/rollout/reconsideration/pilot/300/metrics/paired_video_bootstrap.json). 동일 video query를 묶어400회 재표집했다.

선택만의 이득과 regeneration의 이득은 interactive detail에서 verifier/TTS selected candidate와 final answer를 비교한다. agent의 마지막 inspection 대상은 최종 선택 후보라고 간주하지 않는다. qvhighlights-timelens_00507은600 cohort 밖이므로 이 통계에서 제외했다.
