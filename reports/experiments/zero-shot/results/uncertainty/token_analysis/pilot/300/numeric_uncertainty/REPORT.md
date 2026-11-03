# 숫자 생성과 나머지 토큰의 불확실성 · pilot300 (300질문)

기존 full-head greedy 600개 응답, 10,775개 step의 raw 전체 어휘 엔트로피를 재집계했다. GPU 재추론은 없다. 숫자 토큰은 공백 제거 후 문자열 전체가 숫자로 이루어진 토큰이다. 소수점은 숫자에 포함하지 않고 별도 분류했다.

주 통계는 query별 해당 유형 step의 평균을 구한 뒤 query 간 평균한 값이다. 같은 query의 숫자/나머지를 paired 비교했다. 95% CI는 query 단위 2,000회 bootstrap이다. CSV에는 step 가중 평균과 step 중앙값도 보존했다.

![숫자 대 나머지](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/figures/numeric_vs_other.png)

| 모델 | 벤치마크 | 숫자 H | 나머지 H | paired ΔH [95% CI] | 숫자의 step 비중 (%) | 숫자의 ΣH 비중 (%) |
|---|---|---:|---:|---|---:|---:|
| qwen25 | all | 1.4588 | 0.0028 | +1.4560 [+1.4266, +1.4878] | 33.22 | 99.61 |
| qwen25 | charades-timelens | 1.4925 | 0.0009 | +1.4916 [+1.4397, +1.5450] | 30.64 | 99.86 |
| qwen25 | activitynet-timelens | 1.4565 | 0.0066 | +1.4499 [+1.3868, +1.5101] | 33.81 | 99.11 |
| qwen25 | qvhighlights-timelens | 1.4274 | 0.0009 | +1.4266 [+1.3840, +1.4683] | 35.06 | 99.89 |
| qwen3 | all | 1.1500 | 0.0051 | +1.1449 [+1.1084, +1.1801] | 33.06 | 99.08 |
| qwen3 | charades-timelens | 1.2234 | 0.0026 | +1.2208 [+1.1603, +1.2853] | 29.93 | 99.50 |
| qwen3 | activitynet-timelens | 1.1859 | 0.0054 | +1.1804 [+1.1068, +1.2511] | 33.97 | 99.11 |
| qwen3 | qvhighlights-timelens | 1.0407 | 0.0074 | +1.0333 [+0.9787, +1.0867] | 35.06 | 98.67 |

ΣH 비중은 이 greedy 경로에서 관측한 step 엔트로피 합 중 숫자 step이 차지하는 비율이다. 가능한 모든 응답에 대해 평균한 sequence entropy나 IoU에 대한 설명력으로 해석하지 않는다.

![역할별 불확실성](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/figures/uncertainty_by_role.png)

| 모델 | 토큰 역할 | query N / step N | 평균 H [95% CI] |
|---|---|---|---|
| qwen25 | 백의 자리 | 55 / 107 | 0.3269 [0.2329, 0.4289] |
| qwen25 | 십의 자리 | 266 / 484 | 0.8507 [0.7903, 0.9104] |
| qwen25 | 일의 자리 | 300 / 600 | 1.6524 [1.6186, 1.6873] |
| qwen25 | 소수 첫째 자리 | 300 / 600 | 1.9590 [1.8957, 2.0190] |
| qwen25 | 소수점 | 300 / 600 | 0.0165 [0.0064, 0.0295] |
| qwen25 | 구분자 | 300 / 600 | 0.0000 [0.0000, 0.0000] |
| qwen25 | 공백 | 300 / 600 | 0.0000 [0.0000, 0.0000] |
| qwen25 | 일반 문구 | 300 / 1500 | 0.0001 [0.0000, 0.0002] |
| qwen25 | EOS | 300 / 300 | 0.0000 [0.0000, 0.0000] |
| qwen3 | 백의 자리 | 67 / 119 | 0.3786 [0.2850, 0.4846] |
| qwen3 | 십의 자리 | 253 / 447 | 0.5207 [0.4649, 0.5786] |
| qwen3 | 일의 자리 | 294 / 588 | 1.2328 [1.1788, 1.2843] |
| qwen3 | 소수 첫째 자리 | 293 / 586 | 1.7276 [1.6680, 1.7869] |
| qwen3 | 소수점 | 293 / 586 | 0.0244 [0.0103, 0.0405] |
| qwen3 | 구분자 | 300 / 616 | 0.0046 [0.0024, 0.0073] |
| qwen3 | 공백 | 300 / 600 | 0.0000 [0.0000, 0.0000] |
| qwen3 | 일반 문구 | 300 / 1502 | 0.0005 [0.0000, 0.0013] |
| qwen3 | EOS | 300 / 300 | 0.0000 [0.0000, 0.0000] |
| qwen3 | 기타 숫자(MM:SS 등) | 6 / 40 | 0.5746 [0.3929, 0.8311] |

역할은 반올림 전 생성된 시각 문자열에서 사후 지정했다. 백·십·일의 자리, 소수 첫째 자리를 나누었고 소수점은 별도다. 역할 분석에 포함되지 않은 MM:SS 등 6개 응답의 숫자도 전체 숫자/나머지 비교에는 포함하되 자릿수 표에서는 기타 숫자로 분리했다. 일반 문구, 공백, 구분자, EOS를 분리해 비교한다.

## 해석

- qwen25: 숫자 평균 H=1.4588, 나머지=0.0028 nat. 숫자 step은 33.2%지만 관측 ΣH의 99.61%를 차지한다. 자릿수별 집계에서 가장 큰 평균은 소수 첫째 자리의 1.9590 nat(N=300 query)다.
- qwen3: 숫자 평균 H=1.1500, 나머지=0.0051 nat. 숫자 step은 33.1%지만 관측 ΣH의 99.08%를 차지한다. 자릿수별 집계에서 가장 큰 평균은 소수 첫째 자리의 1.7276 nat(N=293 query)다.

이 결과는 “시간 숫자가 정형 문구보다 예측하기 어려운가”에 관한 것이다. 엔트로피가 큰 숫자가 반드시 틀리거나, 작은 숫자가 반드시 맞는다는 결론은 아니다. 평균 역할 표는 시작·종료와 자릿수 길이가 섞인 설명용 요약이다. 엄밀한 같은 조건 비교를 위해 모델·벤치마크·시작/종료·GT 자릿수·생성 자릿수·역할로 분리한 CSV를 함께 제공했다. 백의 자리처럼 표본이 적은 경우 N을 함께 보아야 한다.

[모든 step CSV](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/per_step.csv) · [역할별 통계 CSV](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/category_summary.csv) · [시작/종료별 역할 CSV](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/endpoint_role_summary.csv) · [GT·생성 자릿수 통제 CSV](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/digit_stratified_summary.csv) · [전체 정량 JSON](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/summary.json)
