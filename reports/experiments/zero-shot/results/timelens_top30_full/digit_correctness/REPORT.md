# 반올림 기준 자릿수별 불확실성

[샘플별 영상·토큰 HTML](report.html)

예측 endpoint를 기존 평가와 동일한 Python `round(float(value))`로 반올림한 결과가 GT와 같으면 모든 생성 step을 정답으로 처리한다. 다르면 각 prefix에서 반올림 정답으로 완성할 수 있는지 검사한다. 정수부에서 가능성이 사라지면 해당 숫자가 최초 오류다. 소수부도 동일하게 정답 가능성이 실제로 처음 사라지는 숫자 step을 최초 오류로 표시한다. 소수점으로 앞당기지 않는다. 정수 종료로 가능성이 사라지면 종료 단계가 최초 오류다. 후보 표기는 원자료와 동일하게 음이 아닌 초, 정수 또는 소수 한 자리, 불필요한 선행 0 없음으로 제한한다. 시작과 종료를 각각 판정한다. Python 반올림은 정확히 .5일 때 가까운 짝수로 처리한다.

예: GT 24에 대한 23.9는 모두 정답. 23.4는 23.까지 정답 가능, 숫자 4가 최초 오류다. 35.0은 첫 숫자 3, 25.0은 숫자 5에서 정답 가능성이 사라진다. 빨간 표시 first_error와 실제 가능성 상실 actual_feasibility_loss는 항상 일치한다. GT 문자 일치 여부는 참고 필드로만 남긴다.

전체 벤치마크 9,404개 × 두 모델의 greedy 생성 분포를 분석했다. 지원 표기 수와 제외 사유는 ../../../artifacts/runs/timelens_top30_full/digit_correctness/metrics/summary.json에 기록했다. 기존 pilot 분포는 재사용하고 나머지는 같은 조건으로 추론했다.

## endpoint 결과

| 모델 | 데이터셋 | 대상 | N | 반올림 정답 N | 최초 오류 숫자/소수점·종료 N |
|---|---|---|---:|---:|---|
| qwen25 | all | start | 9404 | 2410 | 6984/10 |
| qwen25 | all | end | 9404 | 826 | 8511/67 |
| qwen25 | charades-timelens | start | 3363 | 810 | 2551/2 |
| qwen25 | charades-timelens | end | 3363 | 361 | 3000/2 |
| qwen25 | activitynet-timelens | start | 4500 | 1137 | 3357/6 |
| qwen25 | activitynet-timelens | end | 4500 | 360 | 4095/45 |
| qwen25 | qvhighlights-timelens | start | 1541 | 463 | 1076/2 |
| qwen25 | qvhighlights-timelens | end | 1541 | 105 | 1416/20 |
| qwen3 | all | start | 9187 | 3413 | 5742/32 |
| qwen3 | all | end | 9187 | 1409 | 7709/69 |
| qwen3 | charades-timelens | start | 3363 | 1228 | 2126/9 |
| qwen3 | charades-timelens | end | 3363 | 429 | 2906/28 |
| qwen3 | activitynet-timelens | start | 4334 | 1520 | 2796/18 |
| qwen3 | activitynet-timelens | end | 4334 | 648 | 3657/29 |
| qwen3 | qvhighlights-timelens | start | 1490 | 665 | 820/5 |
| qwen3 | qvhighlights-timelens | end | 1490 | 332 | 1146/12 |

## 경우별 엔트로피

각 query 내부에서 해당 상태의 step 엔트로피를 평균한 뒤 query 간 평균한다. 95% CI는 query bootstrap 1,000회. 최초 오류 행은 소수점 또는 종료 토큰도 포함한다. 최종 정답/오답 endpoint 행은 모든 숫자 step으로 별도 집계하며 다른 행과 중복된다.

| 모델 | 대상 | 경우 | N query / N step | 평균 H [95% CI] |
|---|---|---|---|---|
| qwen25 | start | 첫 숫자 정답 | 6198 / 6198 | 0.954 [0.940, 0.969] |
| qwen25 | start | 첫 숫자 오답 | 3206 / 3206 | 1.584 [1.567, 1.601] |
| qwen25 | start | 이전까지 정답 → 정답 | 3652 / 4809 | 1.309 [1.283, 1.334] |
| qwen25 | start | 이전까지 정답 → 최초 오답 | 3778 / 3778 | 1.954 [1.939, 1.967] |
| qwen25 | start | 이전 오류 이후 숫자 | 5821 / 8455 | 1.993 [1.979, 2.007] |
| qwen25 | start | 이전까지 정답 가능 → 최초 오류 (서식) | 10 / 10 | 1.578 [1.376, 1.764] |
| qwen25 | start | all_correct_endpoints | 2410 / 5831 | 1.164 [1.140, 1.185] |
| qwen25 | start | not_all_correct_endpoints | 6994 / 20615 | 1.700 [1.691, 1.709] |
| qwen25 | end | 첫 숫자 정답 | 5079 / 5079 | 0.476 [0.459, 0.493] |
| qwen25 | end | 첫 숫자 오답 | 4325 / 4325 | 0.893 [0.871, 0.914] |
| qwen25 | end | 이전까지 정답 → 정답 | 1954 / 2858 | 1.217 [1.185, 1.251] |
| qwen25 | end | 이전까지 정답 → 최초 오답 | 4186 / 4186 | 1.433 [1.412, 1.455] |
| qwen25 | end | 이전 오류 이후 숫자 | 7882 / 11995 | 1.856 [1.842, 1.870] |
| qwen25 | end | 이전까지 정답 가능 → 최초 오류 (서식) | 67 / 67 | 0.468 [0.346, 0.591] |
| qwen25 | end | all_correct_endpoints | 826 / 2441 | 1.351 [1.318, 1.386] |
| qwen25 | end | not_all_correct_endpoints | 8578 / 26002 | 1.360 [1.350, 1.370] |
| qwen3 | start | 첫 숫자 정답 | 6583 / 6583 | 0.433 [0.421, 0.445] |
| qwen3 | start | 첫 숫자 오답 | 2604 / 2604 | 0.975 [0.955, 0.996] |
| qwen3 | start | 이전까지 정답 → 정답 | 4558 / 6541 | 0.979 [0.956, 1.001] |
| qwen3 | start | 이전까지 정답 → 최초 오답 | 3138 / 3138 | 1.571 [1.548, 1.595] |
| qwen3 | start | 이전 오류 이후 숫자 | 4441 / 6451 | 1.700 [1.676, 1.724] |
| qwen3 | start | 이전까지 정답 가능 → 최초 오류 (서식) | 32 / 32 | 1.296 [1.044, 1.561] |
| qwen3 | start | all_correct_endpoints | 3413 / 8694 | 0.778 [0.756, 0.799] |
| qwen3 | start | not_all_correct_endpoints | 5774 / 16623 | 1.260 [1.246, 1.272] |
| qwen3 | end | 첫 숫자 정답 | 5824 / 5824 | 0.342 [0.329, 0.354] |
| qwen3 | end | 첫 숫자 오답 | 3363 / 3363 | 0.855 [0.834, 0.880] |
| qwen3 | end | 이전까지 정답 → 정답 | 2472 / 4141 | 1.113 [1.084, 1.144] |
| qwen3 | end | 이전까지 정답 → 최초 오답 | 4346 / 4346 | 1.318 [1.301, 1.335] |
| qwen3 | end | 이전 오류 이후 숫자 | 7129 / 10286 | 1.750 [1.735, 1.763] |
| qwen3 | end | 이전까지 정답 가능 → 최초 오류 (서식) | 69 / 69 | 0.830 [0.633, 1.040] |
| qwen3 | end | all_correct_endpoints | 1409 / 4378 | 1.081 [1.057, 1.107] |
| qwen3 | end | not_all_correct_endpoints | 7778 / 23582 | 1.244 [1.234, 1.254] |

![시작](../../../artifacts/runs/timelens_top30_full/digit_correctness/figures/start_roles.png)

![종료](../../../artifacts/runs/timelens_top30_full/digit_correctness/figures/end_roles.png)

가로축은 구분선으로 나눈 자릿수·서식 범주이며 시간이 아니다. 칸 안 색상별 좌우 위치는 가독성을 위한 배치다. 파랑은 첫 숫자를 포함해 정답 가능성을 유지한 숫자, 빨강은 정답 가능성이 처음 사라진 step, 회색은 이전에 이미 가능성이 사라진 뒤의 숫자다. 생성 전 빈 문자열은 정답 가능 상태로 두므로 첫 숫자도 파랑 또는 빨강으로 표시한다. 첫 숫자별 정량 집계는 별도로 유지한다.

## 해석

오류 전 상태는 GT 문자 자체의 일치가 아니라 반올림 정답으로 완성 가능한 prefix를 뜻한다. 소수점이나 종료 구분자 자체가 가능한 완성 후보를 없애는 경우에는 그 서식 step이 최초 오류다. 첫 숫자도 동일한 가능성 기준으로 파랑 또는 빨강으로 표시한다. 반올림 endpoint 일치와 전체 구간 IoU는 다른 지표다. 엔트로피는 원래 생성 경로의 분포이며 인과 효과를 뜻하지 않는다.

[전체 step](../../../artifacts/runs/timelens_top30_full/digit_correctness/metrics/per_step.csv) · [endpoint](../../../artifacts/runs/timelens_top30_full/digit_correctness/metrics/per_endpoint.csv) · [경우별](../../../artifacts/runs/timelens_top30_full/digit_correctness/metrics/conditions.csv) · [자리별](../../../artifacts/runs/timelens_top30_full/digit_correctness/metrics/role_conditions.csv) · [조건부 상태 비율](../../../artifacts/runs/timelens_top30_full/digit_correctness/metrics/position_conditional_accuracy.csv) · [자릿수 통제](../../../artifacts/runs/timelens_top30_full/digit_correctness/metrics/digit_stratified.csv)

## 전체 벤치마크 분석 범위

전체 응답 18,808개 중 지원 표기 18,591개, 제외 217개. 제외 목록은 ../../../artifacts/runs/timelens_top30_full/digit_correctness/metrics/summary.json에 기록했다.

[정답 가능 상태의 세 축 분석](blue_analysis/REPORT.md)
