# Top-30 renormalized entropy · full · 9,404 queries

기존 9,404 query × 2개 모델 = 18,808개 응답의 저장된 token distribution을 재집계했다. 원래 응답·IoU·전체 vocabulary entropy를 유지하며 GPU 추론은 실행하지 않았다.

## 지표 정의

각 생성 step에서 S는 해당 mode의 확률 상위 30개 token, M = Σ(i∈S) pᵢ다. qᵢ = pᵢ/M으로 재정규화하고 H₃₀ = −Σ(i∈S) qᵢ ln qᵢ를 계산했다. 단위는 nat이며 0 ≤ H₃₀ ≤ ln(30) ≈ 3.401197이다. 높을수록 top-30 내부 분포가 고르다. Hfull은 원자료의 전체 vocabulary entropy다. ΔH = Hfull − H₃₀이며 재정규화도 포함되므로 이 차이를 top-30 밖의 entropy 자체로 해석하지 않는다.

주 분석은 raw numeric step이다. 숫자 token 정의는 기존 entropy–IoU 분석과 같은 `\s*\d+(?:\.\d+)?\s*`이며 소수부도 포함한다. 먼저 query 내 해당 step의 평균을 구한 뒤 각 query에 같은 가중치를 부여해 모델·dataset별로 평균한다. 숫자 step이 없는 응답은 numeric 분석에서 제외하고 수를 명시한다. all은 EOS를 포함한 모든 생성 step이다. raw는 generation processor 적용 전, processed는 적용 후이며 각각 자신의 top-30을 재정규화한다.

Spearman ρ는 query별 평균 entropy와 IoU의 순위 상관이다. 음수면 IoU가 높을수록 entropy가 낮은 경향이다. 동률에는 평균 순위를 적용했다. 성공/실패는 IoU≥0.5 / IoU<0.5다. 이번 비교는 기술 통계이며 신뢰구간·유의성 검정을 수행하지 않았다.

## Raw numeric 결과

| 모델 | dataset | 유효/제외 query | Hfull (nat) | H₃₀ (nat) | ΔH (nat) | M | ρ full | ρ top-30 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen25 | activitynet-timelens | 4500/0 | 1.430330 | 1.429806 | 0.000524 | 0.999973 | -0.081825 | -0.081687 |
| qwen25 | charades-timelens | 3363/0 | 1.528036 | 1.527509 | 0.000527 | 0.999974 | -0.130662 | -0.130698 |
| qwen25 | qvhighlights-timelens | 1541/0 | 1.443317 | 1.442743 | 0.000574 | 0.999970 | -0.223092 | -0.223082 |
| qwen3 | activitynet-timelens | 4500/0 | 1.133774 | 1.133773 | 0.000001 | 1.000000 | -0.360556 | -0.360557 |
| qwen3 | charades-timelens | 3363/0 | 1.234211 | 1.234211 | 0.000001 | 1.000000 | -0.324956 | -0.324956 |
| qwen3 | qvhighlights-timelens | 1541/0 | 1.015727 | 1.015726 | 0.000000 | 1.000000 | -0.397660 | -0.397657 |

## 해석과 범위

H₃₀는 “다음 token이 top-30 안에 있다”는 조건부 entropy다. top-30 밖의 확률 질량은 제외되므로 M을 함께 확인해야 한다. 저장된 greedy trajectory에 대한 사후 분석이며 생성 정책은 변경하지 않는다. pilot과 full은 각각 집계하며 겹치는 query를 독립 표본으로 합산하지 않는다. 원자료의 FP32 오차 때문에 M이 1을 미세하게 넘을 수 있으며 clipping 없이 실제 합으로 나눴다.

## Raw/processed · numeric/all 보조 결과

| 모델 | dataset | mode | step 범위 | query 수 | Hfull | H₃₀ | 성공 H₃₀ | 실패 H₃₀ |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen25 | activitynet-timelens | raw | numeric | 4500 | 1.430330 | 1.429806 | 1.401185 | 1.443440 |
| qwen25 | activitynet-timelens | raw | all | 4500 | 0.482354 | 0.482170 | 0.456405 | 0.494445 |
| qwen25 | activitynet-timelens | processed | numeric | 4500 | 1.444562 | 1.443958 | 1.428284 | 1.451425 |
| qwen25 | activitynet-timelens | processed | all | 4500 | 0.487078 | 0.486848 | 0.463408 | 0.498014 |
| qwen25 | charades-timelens | raw | numeric | 3363 | 1.528036 | 1.527509 | 1.507765 | 1.539445 |
| qwen25 | charades-timelens | raw | all | 3363 | 0.461641 | 0.461474 | 0.456495 | 0.464485 |
| qwen25 | charades-timelens | processed | numeric | 3363 | 1.568841 | 1.568213 | 1.551510 | 1.578309 |
| qwen25 | charades-timelens | processed | all | 3363 | 0.473050 | 0.472831 | 0.468604 | 0.475386 |
| qwen25 | qvhighlights-timelens | raw | numeric | 1541 | 1.443317 | 1.442743 | 1.382856 | 1.468623 |
| qwen25 | qvhighlights-timelens | raw | all | 1541 | 0.487669 | 0.487470 | 0.458993 | 0.499776 |
| qwen25 | qvhighlights-timelens | processed | numeric | 1541 | 1.444974 | 1.444308 | 1.396526 | 1.464958 |
| qwen25 | qvhighlights-timelens | processed | all | 1541 | 0.488205 | 0.487958 | 0.462559 | 0.498935 |
| qwen3 | activitynet-timelens | raw | numeric | 4500 | 1.133774 | 1.133773 | 1.022369 | 1.251073 |
| qwen3 | activitynet-timelens | raw | all | 4500 | 0.382307 | 0.382307 | 0.337040 | 0.429969 |
| qwen3 | activitynet-timelens | processed | numeric | 4500 | 1.133774 | 1.133773 | 1.022369 | 1.251073 |
| qwen3 | activitynet-timelens | processed | all | 4500 | 0.382307 | 0.382307 | 0.337040 | 0.429969 |
| qwen3 | charades-timelens | raw | numeric | 3363 | 1.234211 | 1.234211 | 1.164109 | 1.315135 |
| qwen3 | charades-timelens | raw | all | 3363 | 0.374698 | 0.374697 | 0.352861 | 0.399905 |
| qwen3 | charades-timelens | processed | numeric | 3363 | 1.234211 | 1.234211 | 1.164109 | 1.315135 |
| qwen3 | charades-timelens | processed | all | 3363 | 0.374698 | 0.374697 | 0.352861 | 0.399905 |
| qwen3 | qvhighlights-timelens | raw | numeric | 1541 | 1.015727 | 1.015726 | 0.939982 | 1.152591 |
| qwen3 | qvhighlights-timelens | raw | all | 1541 | 0.347957 | 0.347957 | 0.320938 | 0.396778 |
| qwen3 | qvhighlights-timelens | processed | numeric | 1541 | 1.015727 | 1.015726 | 0.939982 | 1.152591 |
| qwen3 | qvhighlights-timelens | processed | all | 1541 | 0.347957 | 0.347957 | 0.320938 | 0.396778 |

전체 669,920개 분포에서 Σq=1, entropy 범위, H₃₀ = −Σ(pᵢ ln pᵢ)/M + ln M 항등식과 저장 mass 일치를 검증했다.

원래 top-30 ID·확률은 복제하지 않는다. sources가 가리키는 원본에서 qᵢ=pᵢ/M으로 재구성할 수 있다.

[query별 지표](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/top30_renormalized/metrics/per_query.csv)

[step별 지표 (gzip CSV)](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/top30_renormalized/metrics/per_step.csv.gz)

[집계 JSON](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/top30_renormalized/metrics/summary.json)

[원본 경로·SHA-256](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/top30_renormalized/metadata/sources.json)

[수치 검증](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/top30_renormalized/validation/numerical.json)
