# Zero-shot entropy와 IoU의 관계 · pilot · 300 queries

이 보고서의 entropy와 후보 확률은 top-30 재정규화 기준이다. 각 step에서 M = Σ(top-30) pᵢ, qᵢ = pᵢ/M, H₃₀ = −Σ qᵢ ln qᵢ로 계산한다. 단위는 nat, 범위는 0–ln(30) ≈ 3.401197이며 높을수록 top-30 내부 분포가 고르다. raw와 processed는 각각 자신의 top-30을 사용한다. top-30 밖의 질량은 조건부 분포에서 제외한다. 정답 가능성은 기존 문법·반올림·IoU 판정을 뜻하며 모델의 최종 정답 확률과 다르다. 원본 응답·판정·전체 vocabulary 분포 계측값은 보존하고 보고서의 파생 지표만 재계산했다.


기존 top-30 실험의 600개 full-head greedy 응답을 재집계했다. GPU 추론을 다시 실행하지 않았다. 각 step의 저장된 raw top-30 확률을 재정규화하여 H₃₀를 계산했다. 단위는 nat.

분석 단위는 query다. 주 분석값은 **실제로 숫자 토큰을 생성한 단계들의 entropy 평균**이다. 숫자 토큰은 공백을 제외한 문자열이 숫자/소수와 일치하는 경우이며 소수부도 포함한다. 정형 문구 및 EOS를 포함한 전체 단계 평균은 별도 민감도 분석으로 둔다. 각 query에 같은 가중치를 부여해 긴 응답이 표본 수를 늘리지 않게 했다. 모델·벤치마크를 섞지 않고 각 100개씩 비교한다.

![IoU와 숫자 단계 entropy](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_iou/figures/entropy_vs_iou.png)

| 모델 | 벤치마크 | Spearman ρ [95% bootstrap CI] | IoU < .5: n / 평균 H | IoU ≥ .5: n / 평균 H | 성공−실패 ΔH [95% CI] |
|---|---|---|---|---|---|
| qwen25 | charades-timelens | -0.035 [-0.226, +0.160] | 58 / 1.494 | 42 / 1.490 | -0.004 [-0.112, +0.098] |
| qwen25 | activitynet-timelens | -0.015 [-0.201, +0.194] | 73 / 1.451 | 27 / 1.470 | +0.019 [-0.102, +0.142] |
| qwen25 | qvhighlights-timelens | -0.083 [-0.287, +0.114] | 76 / 1.420 | 24 / 1.448 | +0.028 [-0.078, +0.140] |
| qwen3 | charades-timelens | -0.230 [-0.405, -0.031] | 51 / 1.278 | 49 / 1.167 | -0.111 [-0.239, +0.014] |
| qwen3 | activitynet-timelens | -0.341 [-0.516, -0.153] | 57 / 1.311 | 43 / 1.020 | -0.292 [-0.441, -0.150] |
| qwen3 | qvhighlights-timelens | -0.352 [-0.518, -0.145] | 38 / 1.131 | 62 / 0.986 | -0.145 [-0.252, -0.037] |

ρ < 0이면 IoU가 높을수록 entropy가 낮은 경향이다. ΔH < 0이면 IoU≥0.5 집단에서 평균 entropy가 더 낮다. CI가 0을 포함하면 이 pilot에서 방향이 분명하다고 말하기 어렵다. CI는 query를 복원 추출한 비모수 percentile bootstrap이며 다중 비교 보정은 하지 않은 탐색적 분석이다.

6개 그룹 중 음의 상관 CI가 0을 제외한 그룹 3개, 양의 상관 CI가 0을 제외한 그룹 0개다. 나머지는 방향이 불확실하다. 그룹별 결과를 확인해야 하며 낮은 entropy를 높은 IoU의 보장으로 해석할 수 없다.

| 모델 | 벤치마크 | 전체 step ρ | 정수부 step ρ | 첫 숫자 step ρ | 보정 후 숫자 step ρ | 숫자 step 수와 IoU의 ρ |
|---|---|---:|---:|---:|---:|---:|
| qwen25 | charades-timelens | -0.112 | +0.058 | -0.029 | +0.016 | -0.129 |
| qwen25 | activitynet-timelens | -0.143 | -0.023 | -0.339 | +0.019 | -0.227 |
| qwen25 | qvhighlights-timelens | -0.313 | -0.059 | -0.286 | -0.063 | -0.177 |
| qwen3 | charades-timelens | -0.170 | -0.282 | -0.213 | -0.230 | +0.096 |
| qwen3 | activitynet-timelens | -0.409 | -0.413 | -0.392 | -0.341 | -0.229 |
| qwen3 | qvhighlights-timelens | -0.380 | -0.364 | -0.436 | -0.352 | +0.075 |

정수부 step은 답변에서 처음 두 숫자의 정수부 문자 위치와 겹치는 토큰으로 정했다. 첫 숫자 step은 두 시각 전체의 불확실성을 대표하지 않지만, 생성 길이 및 후속 숫자 선택의 영향을 줄여 확인하는 보조값이다. 소수부 자리와 토큰 길이가 평균 H에 영향을 줄 수 있어 주 분석과 같이 본다.

확률분포는 각 greedy 응답의 실제 이전 토큰열에 조건화되어 있다. 이 값은 영상 자체의 불확실성만을 측정하거나, 모든 가능한 rollout의 분포를 적분한 값이 아니다. 특히 확신 있게 틀린 답은 낮은 H·낮은 IoU로 나타날 수 있다. 정확도 calibration이나 인과관계를 주장하지 않는다.

Qwen2.5 기본 repetition penalty의 영향은 raw와 processed를 구분했다. 주 분석에는 processor 이전 raw 분포를 사용한다. GPU 수치 오차로 top-30 질량 합이 1을 미세하게 넘는 기록이 있을 수 있으나, 실제 top-30 확률 합으로 나눠 entropy를 재계산했다. 기존 Decord 디코딩 품질 한계도 그대로 유지된다.

재정규화 전 −Σ(top30) p ln p와 재정규화 후 H₃₀는 서로 다른 양이다. 이 보고서는 H₃₀를 사용한다.

[query별 CSV](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_iou/metrics/per_query.csv) · [집계 및 IoU 구간별 결과 JSON](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_iou/metrics/summary.json) · [벡터 그래프 SVG](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_iou/figures/entropy_vs_iou.svg)
