# Rollout reconsideration · pilot600

완료: 600 query, dataset별200. 3개 접근과 direct regeneration control을 같은 고정 cohort에서 평가했다. 생성 input hash는 기존 deterministic rollout과 일치하며 GT는 보고 단계에서만 사용했다.

[실험 설계](SPEC.md)

## 지표와 비교 기준

mIoU는 600 query의 temporal IoU 평균, R@τ는 IoU≥τ query 비율이다. IoU=intersection duration/union duration이며 높을수록 좋다. Δ는 같은 query의 greedy 대비 차이, Δpp는 성공률 차이×100이다. failure는 파싱 실패·invalid interval·EOS 미도달 비율이고0점으로 포함한다. model seconds는 CUDA 동기화로 측정한 의사결정 model wall time이며 공통 video prefill과 사후 probability diagnostic을 제외한다. decoder tokens는 기존 proposal generation 비용과 새 review의 prompt·generated token·KV replay를 합산한다. probability diagnostics 비용은 별도 원자료에 보존한다. 기존 baseline은 추가 review time이0이다. 평가 호출 수를 맞춘 비교가 FLOPs 일치를 뜻하지 않는다.

## 전체 성능

| method | n | mIoU (Δ) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | model evaluations | decoder tokens (proposal+review) | review model seconds | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 600 | 0.5157 (+0.0000) | 0.6783 (+0.00) | 0.5433 (+0.00) | 0.3817 (+0.00) | 0.0 | 16.8 | 0.000 | 0.000 |
| max_probability_pool8 | 600 | 0.5273 (+0.0117) | 0.6950 (+1.67) | 0.5633 (+2.00) | 0.4000 (+1.83) | 0.0 | 84.0 | 0.000 | 0.000 |
| probability_weighted_medoid_pool8 | 600 | 0.5319 (+0.0162) | 0.7000 (+2.17) | 0.5650 (+2.17) | 0.4033 (+2.17) | 0.0 | 84.0 | 0.000 | 0.000 |
| direct_pool8 | 600 | 0.5279 (+0.0122) | 0.6983 (+2.00) | 0.5600 (+1.67) | 0.3917 (+1.00) | 1.0 | 374.4 | 0.791 | 0.000 |
| agent_pool8 | 600 | 0.5200 (+0.0044) | 0.6783 (+0.00) | 0.5483 (+0.50) | 0.3917 (+1.00) | 6.0 | 2498.5 | 6.490 | 0.000 |
| verifier_pool8 | 600 | 0.5221 (+0.0064) | 0.6883 (+1.00) | 0.5483 (+0.50) | 0.3850 (+0.33) | 7.0 | 645.4 | 1.060 | 0.000 |
| tts_pool8_select4 | 600 | 0.5253 (+0.0096) | 0.6950 (+1.67) | 0.5567 (+1.33) | 0.3883 (+0.67) | 7.0 | 779.0 | 1.054 | 0.000 |
| tts_pool8_select8 | 600 | 0.5263 (+0.0106) | 0.6950 (+1.67) | 0.5567 (+1.33) | 0.3917 (+1.00) | 14.8 | 1266.6 | 1.443 | 0.000 |
| direct_pool32 | 600 | 0.5201 (+0.0045) | 0.6833 (+0.50) | 0.5483 (+0.50) | 0.3850 (+0.33) | 1.0 | 611.4 | 0.795 | 0.000 |
| agent_pool32 | 600 | 0.5115 (-0.0041) | 0.6767 (-0.17) | 0.5400 (-0.33) | 0.3717 (-1.00) | 6.5 | 3321.0 | 7.111 | 0.000 |
| verifier_pool32 | 600 | 0.5244 (+0.0087) | 0.6883 (+1.00) | 0.5483 (+0.50) | 0.3867 (+0.50) | 14.9 | 1207.7 | 1.438 | 0.000 |
| tts_pool32_select4 | 600 | 0.5193 (+0.0036) | 0.6900 (+1.17) | 0.5467 (+0.33) | 0.3783 (-0.33) | 7.0 | 1020.0 | 1.073 | 0.000 |
| tts_pool32_select8 | 600 | 0.5145 (-0.0012) | 0.6867 (+0.83) | 0.5367 (-0.67) | 0.3750 (-0.67) | 14.9 | 1520.3 | 1.449 | 0.000 |
| max_probability_pool32 | 600 | 0.5251 (+0.0094) | 0.6917 (+1.33) | 0.5583 (+1.50) | 0.4017 (+2.00) | 0.0 | 291.8 | 0.000 | 0.000 |
| probability_weighted_medoid_pool32 | 600 | 0.5331 (+0.0174) | 0.6967 (+1.83) | 0.5617 (+1.83) | 0.4083 (+2.67) | 0.0 | 291.8 | 0.000 | 0.000 |
| oracle_pool8 | 600 | 0.6697 (+0.1541) | 0.8350 (+15.67) | 0.7067 (+16.33) | 0.5533 (+17.17) | 0.0 | 84.0 | 0.000 | 0.000 |
| oracle_pool32 | 600 | 0.7295 (+0.2139) | 0.8883 (+21.00) | 0.7683 (+22.50) | 0.6417 (+26.00) | 0.0 | 291.8 | 0.000 | 0.000 |

## Paired comparison · 95% bootstrap CI

video 단위 paired bootstrap의 방법 간 차이다. CI가0을 포함하면 순위 차이를 확정하지 않는다.

| comparison | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |
|---|---|---|
| verifier_pool8 minus agent_pool8 | +0.0020 [-0.0060, +0.0106] | +0.0000 [-0.0134, +0.0134] |
| tts_pool8_select4 minus agent_pool8 | +0.0052 [-0.0029, +0.0134] | +0.0083 [-0.0067, +0.0218] |
| tts_pool8_select4 minus verifier_pool8 | +0.0032 [-0.0028, +0.0096] | +0.0083 [-0.0033, +0.0202] |
| verifier_pool32 minus agent_pool32 | +0.0128 [+0.0005, +0.0240] | +0.0083 [-0.0132, +0.0265] |
| tts_pool32_select8 minus agent_pool32 | +0.0030 [-0.0092, +0.0147] | -0.0033 [-0.0251, +0.0167] |
| tts_pool32_select8 minus verifier_pool32 | -0.0099 [-0.0176, -0.0016] | -0.0117 [-0.0268, +0.0034] |
| tts_pool8_select8 minus tts_pool8_select4 | +0.0010 [-0.0010, +0.0030] | +0.0000 [-0.0050, +0.0066] |
| tts_pool32_select4 minus tts_pool8_select4 | -0.0060 [-0.0136, +0.0018] | -0.0100 [-0.0264, +0.0050] |
| tts_pool32_select8 minus tts_pool8_select8 | -0.0118 [-0.0200, -0.0041] | -0.0200 [-0.0366, -0.0067] |
| tts_pool32_select8 minus tts_pool32_select4 | -0.0048 [-0.0090, -0.0008] | -0.0100 [-0.0216, +0.0000] |

## Clustering sensitivity

semantic gold가 아닌 temporal proxy의 threshold 민감도다. 분모는 두 응답이 모두 valid인 query다.

| method | n | switch @.3 | switch @.5 | switch @.7 |
|---|---:|---:|---:|---:|
| greedy | 600 | 0.0000 | 0.0000 | 0.0000 |
| max_probability_pool8 | 600 | 0.1067 | 0.1333 | 0.1833 |
| probability_weighted_medoid_pool8 | 600 | 0.0917 | 0.1250 | 0.1750 |
| direct_pool8 | 600 | 0.0200 | 0.0317 | 0.0400 |
| agent_pool8 | 600 | 0.0150 | 0.0300 | 0.0500 |
| verifier_pool8 | 600 | 0.0183 | 0.0183 | 0.0283 |
| tts_pool8_select4 | 600 | 0.0200 | 0.0250 | 0.0450 |
| tts_pool8_select8 | 600 | 0.0167 | 0.0267 | 0.0500 |
| direct_pool32 | 600 | 0.0350 | 0.0583 | 0.0983 |
| agent_pool32 | 600 | 0.0433 | 0.0667 | 0.1067 |
| verifier_pool32 | 600 | 0.0383 | 0.0567 | 0.0967 |
| tts_pool32_select4 | 600 | 0.0217 | 0.0417 | 0.0800 |
| tts_pool32_select8 | 600 | 0.0250 | 0.0467 | 0.0800 |
| max_probability_pool32 | 600 | 0.1283 | 0.1650 | 0.2483 |
| probability_weighted_medoid_pool32 | 600 | 0.1283 | 0.1800 | 0.2683 |
| oracle_pool8 | 600 | 0.1650 | 0.2333 | 0.3533 |
| oracle_pool32 | 600 | 0.2467 | 0.3783 | 0.5250 |

## Model selection와 final regeneration

verifier/TTS가 실제로 선택한 기존 candidate의 IoU와 final regeneration의 IoU를 구분한다. agent의 inspection 대상은 선택 결과로 간주하지 않는다.

| method | n selected | selected mIoU | final mIoU | model order disagreement |
|---|---:|---:|---:|---:|
| verifier_pool8 | 600 | 0.5204 | 0.5221 | — |
| tts_pool8_select4 | 600 | 0.5241 | 0.5253 | 0.30992196209587514 |
| tts_pool8_select8 | 600 | 0.5218 | 0.5263 | 0.37777777777777777 |
| verifier_pool32 | 600 | 0.5220 | 0.5244 | — |
| tts_pool32_select4 | 600 | 0.5229 | 0.5193 | 0.2408026755852843 |
| tts_pool32_select8 | 600 | 0.5216 | 0.5145 | 0.3143267848586488 |

## event change와 boundary refinement

temporal clustering은 semantic event identity gold가 아닌 proxy다. switch는 valid 두 응답의 cluster 변경, event recovery는 greedy IoU<0.1에서 switch 후 IoU≥0.5가 된 경우다. same-cluster boundary 지표는 greedy IoU≥0.5인 paired-valid query에 조건부이며 분모를 별도로 표시한다. endpoint MAE는 두 endpoint의 절대오차 평균(seconds), 낮을수록 좋다.

| method | switch / paired-valid | beneficial / switch | event recovery / greedy IoU<.1 | harmful / greedy IoU≥.5 | same-cluster boundary n | MAE before → after | ΔIoU |
|---|---:|---:|---:|---:|---:|---|---:|
| greedy | 0/600 | 0/0 | 0/117 | 0/326 | 326 | 1.904 → 1.904 | +0.0000 |
| max_probability_pool8 | 80/600 | 32/80 | 12/117 | 17/326 | 314 | 1.878 → 1.998 | -0.0079 |
| probability_weighted_medoid_pool8 | 75/600 | 33/75 | 11/117 | 16/326 | 313 | 1.822 → 1.908 | -0.0067 |
| direct_pool8 | 19/600 | 15/19 | 5/117 | 4/326 | 325 | 1.908 → 2.036 | -0.0033 |
| agent_pool8 | 18/600 | 10/18 | 2/117 | 6/326 | 322 | 1.849 → 1.842 | -0.0020 |
| verifier_pool8 | 11/600 | 8/11 | 3/117 | 2/326 | 324 | 1.900 → 1.898 | +0.0004 |
| tts_pool8_select4 | 15/600 | 12/15 | 5/117 | 4/326 | 325 | 1.908 → 2.015 | -0.0028 |
| tts_pool8_select8 | 16/600 | 14/16 | 5/117 | 3/326 | 324 | 1.911 → 2.021 | -0.0021 |
| direct_pool32 | 35/600 | 19/35 | 5/117 | 9/326 | 319 | 1.895 → 1.861 | -0.0057 |
| agent_pool32 | 40/600 | 15/40 | 4/117 | 16/326 | 310 | 1.877 → 2.004 | -0.0080 |
| verifier_pool32 | 34/600 | 18/34 | 4/117 | 7/326 | 319 | 1.892 → 1.905 | -0.0037 |
| tts_pool32_select4 | 25/600 | 15/25 | 4/117 | 8/326 | 319 | 1.905 → 1.921 | -0.0080 |
| tts_pool32_select8 | 28/600 | 12/28 | 4/117 | 13/326 | 318 | 1.902 → 1.926 | -0.0098 |
| max_probability_pool32 | 99/600 | 39/99 | 12/117 | 22/326 | 306 | 1.736 → 1.869 | -0.0113 |
| probability_weighted_medoid_pool32 | 108/600 | 42/108 | 11/117 | 16/326 | 302 | 1.871 → 1.922 | -0.0093 |
| oracle_pool8 | 140/600 | 139/140 | 34/117 | 0/326 | 315 | 1.855 → 0.958 | +0.0708 |
| oracle_pool32 | 227/600 | 225/227 | 44/117 | 0/326 | 279 | 1.790 → 0.651 | +0.0926 |

## token distribution과 path probability

같은 token prefix를 review context에서 재평가했다. 아래 positive log-odds Δ는 원래 선택에 비해 대체 token을 더 선호하게 됐다는 뜻이다. GT에 가까운지와는 별도다. JS divergence는 전체 어휘 분포 차이이며0이면 같다.

| method | axis | n probes | mean JS divergence | mean alternative probability Δ | mean log-odds Δ | alternative full-path preference Δ |
|---|---|---:|---:|---:|---:|---:|
| direct_pool8 | event | 232 | 0.148307 | -0.098355 | -4.2613 | -5.6385 |
| direct_pool8 | boundary | 600 | 0.171182 | -0.190106 | -3.0287 | -3.5277 |
| agent_pool8 | event | 232 | 0.185052 | -0.120058 | -9.5339 | -10.6666 |
| agent_pool8 | boundary | 600 | 0.222025 | -0.204674 | -6.4702 | -6.8078 |
| verifier_pool8 | event | 232 | 0.170542 | -0.115199 | -5.5399 | -6.9667 |
| verifier_pool8 | boundary | 600 | 0.193799 | -0.209924 | -3.8794 | -4.3622 |
| tts_pool8_select4 | event | 232 | 0.154367 | -0.109934 | -4.3545 | -5.7546 |
| tts_pool8_select4 | boundary | 600 | 0.177845 | -0.199205 | -3.1610 | -3.7066 |
| tts_pool8_select8 | event | 232 | 0.154700 | -0.113590 | -4.2602 | -5.6473 |
| tts_pool8_select8 | boundary | 600 | 0.171063 | -0.192045 | -2.9569 | -3.5001 |
| direct_pool32 | event | 232 | 0.147303 | -0.067569 | -3.3098 | -4.9159 |
| direct_pool32 | boundary | 600 | 0.180323 | -0.175755 | -2.7965 | -3.1611 |
| agent_pool32 | event | 232 | 0.198199 | -0.089615 | -8.0269 | -9.4916 |
| agent_pool32 | boundary | 600 | 0.237146 | -0.181196 | -6.2217 | -6.6884 |
| verifier_pool32 | event | 232 | 0.185812 | -0.077681 | -4.8114 | -6.4461 |
| verifier_pool32 | boundary | 600 | 0.200047 | -0.176613 | -3.3837 | -3.7329 |
| tts_pool32_select4 | event | 232 | 0.161269 | -0.067590 | -3.5544 | -5.1868 |
| tts_pool32_select4 | boundary | 600 | 0.184406 | -0.177438 | -2.9019 | -3.3311 |
| tts_pool32_select8 | event | 232 | 0.161793 | -0.071187 | -3.6460 | -5.2639 |
| tts_pool32_select8 | boundary | 600 | 0.182049 | -0.178150 | -2.8427 | -3.2962 |

## 추가 분석

[dataset·video duration·GT ratio·교차표](STRATA.md). [video별 paired bootstrap 원자료](../../../../../../artifacts/runs/rollout/deterministic/pilot/600/reconsideration/metrics/paired_video_bootstrap.json). 동일 video query를 묶어400회 재표집했다.

선택만의 이득과 regeneration의 이득은 interactive detail에서 verifier/TTS selected candidate와 final answer를 비교한다. agent의 마지막 inspection 대상은 최종 선택 후보라고 간주하지 않는다. qvhighlights-timelens_00507은600 cohort 밖이므로 이 통계에서 제외했다.

<!-- reconsideration-viewer:start -->
## Reconsideration 프로세스 탐색

[접근별 프로세스](viewer.html)는 deterministic rollout의 하위 분석이다. 기존 사례 유형·수치 필터와 같은 질문을 사용하며 원래 trajectory 문장·분기 지도·baseline 분포는 원본 뷰 링크에서 확인한다. 이 페이지는 그 화면을 복제하지 않는다.

agent는 실제 controller action과 inspector observation을 시간순으로 표시한다. 명시적 selected candidate가 없으므로 마지막 inspection을 최종 선택으로 간주하지 않는다. verifier는 후보별 event/경계 Yes 확률과 그 곱, 실제 선택 및 final regeneration을 연결한다. test-time scaling은 양쪽 presentation order의 pairwise 점수, round별 승자와 bye를 대진 구조로 표시한다. direct regeneration은 추가 평가 없이 같은 ledger로 생성한 두 pool 조건의 대조다. 각 접근 안에서 예산 조건을 비교한다.

pool은 기존 deterministic 후보의 앞8/32개다. 유효하고 중복되지 않은 구간에서 GT 없이 farthest-first로 다양성을 확보하며, ledger의 후보 목록은 최대8개다. verifier는 별도로 최대3/7개를 평가하고 TTS는 최대4/8개를 대진에 넣는다. 실제 목록·평가 수·원문은 저장된 기록을 사용한다. final answer는 선택 후보와 다를 수 있다. 사후 token distribution 진단은 기본 검토 과정과 분리한다. 원본 응답·영상·baseline token distribution는 복제하지 않는다.
<!-- reconsideration-viewer:end -->
