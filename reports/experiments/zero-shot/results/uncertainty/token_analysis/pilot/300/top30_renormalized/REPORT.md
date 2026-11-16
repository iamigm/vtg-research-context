# Top-30 renormalized entropy · pilot · 300 queries

기존 300 query × 2개 모델 = 600개 응답의 저장된 token distribution을 재집계했다. 원래 응답·IoU·전체 vocabulary entropy를 유지하며 GPU 추론은 실행하지 않았다.

## 지표 정의

각 생성 step에서 S는 해당 mode의 확률 상위 30개 token, M = Σ(i∈S) pᵢ다. qᵢ = pᵢ/M으로 재정규화하고 H₃₀ = −Σ(i∈S) qᵢ ln qᵢ를 계산했다. 단위는 nat이며 0 ≤ H₃₀ ≤ ln(30) ≈ 3.401197이다. 높을수록 top-30 내부 분포가 고르다. Hfull은 원자료의 전체 vocabulary entropy다. ΔH = Hfull − H₃₀이며 재정규화도 포함되므로 이 차이를 top-30 밖의 entropy 자체로 해석하지 않는다.

주 분석은 raw numeric step이다. 숫자 token 정의는 기존 entropy–IoU 분석과 같은 `\s*\d+(?:\.\d+)?\s*`이며 소수부도 포함한다. 먼저 query 내 해당 step의 평균을 구한 뒤 각 query에 같은 가중치를 부여해 모델·dataset별로 평균한다. 숫자 step이 없는 응답은 numeric 분석에서 제외하고 수를 명시한다. all은 EOS를 포함한 모든 생성 step이다. raw는 generation processor 적용 전, processed는 적용 후이며 각각 자신의 top-30을 재정규화한다.

Spearman ρ는 query별 평균 entropy와 IoU의 순위 상관이다. 음수면 IoU가 높을수록 entropy가 낮은 경향이다. 동률에는 평균 순위를 적용했다. 성공/실패는 IoU≥0.5 / IoU<0.5다. 이번 비교는 기술 통계이며 신뢰구간·유의성 검정을 수행하지 않았다.

## Raw numeric 결과

| 모델 | dataset | 유효/제외 query | Hfull (nat) | H₃₀ (nat) | ΔH (nat) | M | ρ full | ρ top-30 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen25 | activitynet-timelens | 100/0 | 1.456524 | 1.455992 | 0.000532 | 0.999973 | -0.015221 | -0.015221 |
| qwen25 | charades-timelens | 100/0 | 1.492465 | 1.491992 | 0.000473 | 0.999976 | -0.034288 | -0.034667 |
| qwen25 | qvhighlights-timelens | 100/0 | 1.427436 | 1.426882 | 0.000554 | 0.999971 | -0.082529 | -0.082691 |
| qwen3 | activitynet-timelens | 100/0 | 1.185864 | 1.185864 | 0.000001 | 1.000000 | -0.341284 | -0.341284 |
| qwen3 | charades-timelens | 100/0 | 1.223430 | 1.223429 | 0.000001 | 1.000000 | -0.230111 | -0.230111 |
| qwen3 | qvhighlights-timelens | 100/0 | 1.040699 | 1.040699 | 0.000000 | 1.000000 | -0.352462 | -0.352462 |

## 해석과 범위

H₃₀는 “다음 token이 top-30 안에 있다”는 조건부 entropy다. top-30 밖의 확률 질량은 제외되므로 M을 함께 확인해야 한다. 저장된 greedy trajectory에 대한 사후 분석이며 생성 정책은 변경하지 않는다. pilot과 full은 각각 집계하며 겹치는 query를 독립 표본으로 합산하지 않는다. 원자료의 FP32 오차 때문에 M이 1을 미세하게 넘을 수 있으며 clipping 없이 실제 합으로 나눴다.

## Raw/processed · numeric/all 보조 결과

| 모델 | dataset | mode | step 범위 | query 수 | Hfull | H₃₀ | 성공 H₃₀ | 실패 H₃₀ |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen25 | activitynet-timelens | raw | numeric | 100 | 1.456524 | 1.455992 | 1.469696 | 1.450924 |
| qwen25 | activitynet-timelens | raw | all | 100 | 0.492758 | 0.492572 | 0.479397 | 0.497445 |
| qwen25 | activitynet-timelens | processed | numeric | 100 | 1.472344 | 1.471735 | 1.491262 | 1.464512 |
| qwen25 | activitynet-timelens | processed | all | 100 | 0.498363 | 0.498136 | 0.484916 | 0.503026 |
| qwen25 | charades-timelens | raw | numeric | 100 | 1.492465 | 1.491992 | 1.489611 | 1.493716 |
| qwen25 | charades-timelens | raw | all | 100 | 0.453969 | 0.453820 | 0.444811 | 0.460343 |
| qwen25 | charades-timelens | processed | numeric | 100 | 1.535397 | 1.534832 | 1.545776 | 1.526907 |
| qwen25 | charades-timelens | processed | all | 100 | 0.466220 | 0.466022 | 0.461071 | 0.469607 |
| qwen25 | qvhighlights-timelens | raw | numeric | 100 | 1.427436 | 1.426882 | 1.448405 | 1.420086 |
| qwen25 | qvhighlights-timelens | raw | all | 100 | 0.490893 | 0.490702 | 0.470864 | 0.496966 |
| qwen25 | qvhighlights-timelens | processed | numeric | 100 | 1.420871 | 1.420227 | 1.450861 | 1.410553 |
| qwen25 | qvhighlights-timelens | processed | all | 100 | 0.488353 | 0.488121 | 0.471822 | 0.493268 |
| qwen3 | activitynet-timelens | raw | numeric | 100 | 1.185864 | 1.185864 | 1.019661 | 1.311244 |
| qwen3 | activitynet-timelens | raw | all | 100 | 0.404244 | 0.404243 | 0.338458 | 0.453871 |
| qwen3 | activitynet-timelens | processed | numeric | 100 | 1.185864 | 1.185864 | 1.019661 | 1.311244 |
| qwen3 | activitynet-timelens | processed | all | 100 | 0.404244 | 0.404243 | 0.338458 | 0.453871 |
| qwen3 | charades-timelens | raw | numeric | 100 | 1.223430 | 1.223429 | 1.167037 | 1.277610 |
| qwen3 | charades-timelens | raw | all | 100 | 0.364024 | 0.364023 | 0.340556 | 0.386571 |
| qwen3 | charades-timelens | processed | numeric | 100 | 1.223430 | 1.223429 | 1.167037 | 1.277610 |
| qwen3 | charades-timelens | processed | all | 100 | 0.364024 | 0.364023 | 0.340556 | 0.386571 |
| qwen3 | qvhighlights-timelens | raw | numeric | 100 | 1.040699 | 1.040699 | 0.985585 | 1.130623 |
| qwen3 | qvhighlights-timelens | raw | all | 100 | 0.361499 | 0.361499 | 0.338619 | 0.398831 |
| qwen3 | qvhighlights-timelens | processed | numeric | 100 | 1.040699 | 1.040699 | 0.985585 | 1.130623 |
| qwen3 | qvhighlights-timelens | processed | all | 100 | 0.361499 | 0.361499 | 0.338619 | 0.398831 |

전체 21,550개 분포에서 Σq=1, entropy 범위, H₃₀ = −Σ(pᵢ ln pᵢ)/M + ln M 항등식과 저장 mass 일치를 검증했다.

원래 top-30 ID·확률은 복제하지 않는다. sources가 가리키는 원본에서 qᵢ=pᵢ/M으로 재구성할 수 있다.

[query별 지표](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/top30_renormalized/metrics/per_query.csv)

[step별 지표 (gzip CSV)](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/top30_renormalized/metrics/per_step.csv.gz)

[집계 JSON](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/top30_renormalized/metrics/summary.json)

[원본 경로·SHA-256](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/top30_renormalized/metadata/sources.json)

[수치 검증](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/top30_renormalized/validation/numerical.json)
