# 반올림 기준 자릿수별 불확실성

[샘플별 영상·토큰 HTML](index.html)

예측 endpoint를 기존 평가와 동일한 Python `round(float(value))`로 반올림한 결과가 GT와 같으면 모든 생성 step을 정답으로 처리한다. 다르면 각 prefix에서 반올림 정답으로 완성할 수 있는지 검사한다. 정수부에서 가능성이 사라지면 해당 숫자가 최초 오류다. 소수부도 동일하게 정답 가능성이 실제로 처음 사라지는 숫자 step을 최초 오류로 표시한다. 소수점으로 앞당기지 않는다. 정수 종료로 가능성이 사라지면 종료 단계가 최초 오류다. 후보 표기는 원자료와 동일하게 음이 아닌 초, 정수 또는 소수 한 자리, 불필요한 선행 0 없음으로 제한한다. 시작과 종료를 각각 판정한다. Python 반올림은 정확히 .5일 때 가까운 짝수로 처리한다.

예: GT 24에 대한 23.9는 모두 정답. 23.4는 23.까지 정답 가능, 숫자 4가 최초 오류다. 35.0은 첫 숫자 3, 25.0은 숫자 5에서 정답 가능성이 사라진다. 빨간 표시 first_error와 실제 가능성 상실 actual_feasibility_loss는 항상 일치한다. GT 문자 일치 여부는 참고 필드로만 남긴다.

기존 동일 pilot 300개 × 두 모델의 저장된 분포를 재사용했다. 초 단위 표기 594개 응답 / 1,188개 endpoint를 분석하고 MM:SS 6개는 제외했다. 새 추론 없이 raw 전체 어휘 엔트로피를 유지했다.

## endpoint 결과

| 모델 | 데이터셋 | 대상 | N | 반올림 정답 N | 최초 오류 숫자/소수점·종료 N |
|---|---|---|---:|---:|---|
| qwen25 | all | start | 300 | 67 | 233/0 |
| qwen25 | all | end | 300 | 25 | 275/0 |
| qwen25 | charades-timelens | start | 100 | 26 | 74/0 |
| qwen25 | charades-timelens | end | 100 | 11 | 89/0 |
| qwen25 | activitynet-timelens | start | 100 | 17 | 83/0 |
| qwen25 | activitynet-timelens | end | 100 | 7 | 93/0 |
| qwen25 | qvhighlights-timelens | start | 100 | 24 | 76/0 |
| qwen25 | qvhighlights-timelens | end | 100 | 7 | 93/0 |
| qwen3 | all | start | 294 | 107 | 187/0 |
| qwen3 | all | end | 294 | 43 | 248/3 |
| qwen3 | charades-timelens | start | 100 | 43 | 57/0 |
| qwen3 | charades-timelens | end | 100 | 11 | 87/2 |
| qwen3 | activitynet-timelens | start | 96 | 25 | 71/0 |
| qwen3 | activitynet-timelens | end | 96 | 9 | 87/0 |
| qwen3 | qvhighlights-timelens | start | 98 | 39 | 59/0 |
| qwen3 | qvhighlights-timelens | end | 98 | 23 | 74/1 |

## 경우별 엔트로피

각 query 내부에서 해당 상태의 step 엔트로피를 평균한 뒤 query 간 평균한다. 95% CI는 query bootstrap 1,000회. 최초 오류 행은 소수점 또는 종료 토큰도 포함한다. 최종 정답/오답 endpoint 행은 모든 숫자 step으로 별도 집계하며 다른 행과 중복된다.

| 모델 | 대상 | 경우 | N query / N step | 평균 H [95% CI] |
|---|---|---|---|---|
| qwen25 | start | 첫 숫자 정답 | 208 / 208 | 0.872 [0.794, 0.960] |
| qwen25 | start | 첫 숫자 오답 | 92 / 92 | 1.545 [1.444, 1.638] |
| qwen25 | start | 이전까지 정답 → 정답 | 113 / 152 | 1.403 [1.272, 1.536] |
| qwen25 | start | 이전까지 정답 → 최초 오답 | 141 / 141 | 1.957 [1.883, 2.033] |
| qwen25 | start | 이전 오류 이후 숫자 | 195 / 277 | 2.037 [1.962, 2.104] |
| qwen25 | start | all_correct_endpoints | 67 / 168 | 1.195 [1.069, 1.335] |
| qwen25 | start | not_all_correct_endpoints | 233 / 702 | 1.685 [1.639, 1.728] |
| qwen25 | end | 첫 숫자 정답 | 156 / 156 | 0.435 [0.347, 0.546] |
| qwen25 | end | 첫 숫자 오답 | 144 / 144 | 0.772 [0.659, 0.895] |
| qwen25 | end | 이전까지 정답 → 정답 | 58 / 84 | 1.076 [0.852, 1.266] |
| qwen25 | end | 이전까지 정답 → 최초 오답 | 131 / 131 | 1.334 [1.215, 1.440] |
| qwen25 | end | 이전 오류 이후 숫자 | 261 / 406 | 1.848 [1.774, 1.918] |
| qwen25 | end | all_correct_endpoints | 25 / 74 | 1.320 [1.137, 1.489] |
| qwen25 | end | not_all_correct_endpoints | 275 / 847 | 1.325 [1.268, 1.378] |
| qwen3 | start | 첫 숫자 정답 | 212 / 212 | 0.412 [0.356, 0.471] |
| qwen3 | start | 첫 숫자 오답 | 82 / 82 | 1.045 [0.937, 1.163] |
| qwen3 | start | 이전까지 정답 → 정답 | 143 / 221 | 1.131 [1.007, 1.254] |
| qwen3 | start | 이전까지 정답 → 최초 오답 | 105 / 105 | 1.450 [1.316, 1.584] |
| qwen3 | start | 이전 오류 이후 숫자 | 150 / 213 | 1.679 [1.541, 1.807] |
| qwen3 | start | all_correct_endpoints | 107 / 286 | 0.939 [0.824, 1.059] |
| qwen3 | start | not_all_correct_endpoints | 187 / 547 | 1.218 [1.144, 1.288] |
| qwen3 | end | 첫 숫자 정답 | 183 / 183 | 0.352 [0.285, 0.426] |
| qwen3 | end | 첫 숫자 오답 | 111 / 111 | 0.737 [0.624, 0.858] |
| qwen3 | end | 이전까지 정답 → 정답 | 75 / 137 | 1.041 [0.882, 1.196] |
| qwen3 | end | 이전까지 정답 → 최초 오답 | 137 / 137 | 1.372 [1.279, 1.460] |
| qwen3 | end | 이전 오류 이후 숫자 | 233 / 339 | 1.684 [1.600, 1.769] |
| qwen3 | end | 이전까지 정답 가능 → 최초 오류 (서식) | 3 / 3 | 0.771 [0.010, 2.237] |
| qwen3 | end | all_correct_endpoints | 43 / 144 | 0.972 [0.834, 1.111] |
| qwen3 | end | not_all_correct_endpoints | 251 / 763 | 1.223 [1.164, 1.280] |

![시작](start_roles.png)

![종료](end_roles.png)

가로축은 구분선으로 나눈 자릿수·서식 범주이며 시간이 아니다. 칸 안 색상별 좌우 위치는 가독성을 위한 배치다. 파랑은 첫 숫자를 포함해 정답 가능성을 유지한 숫자, 빨강은 정답 가능성이 처음 사라진 step, 회색은 이전에 이미 가능성이 사라진 뒤의 숫자다. 생성 전 빈 문자열은 정답 가능 상태로 두므로 첫 숫자도 파랑 또는 빨강으로 표시한다. 첫 숫자별 정량 집계는 별도로 유지한다.

## 해석

오류 전 상태는 GT 문자 자체의 일치가 아니라 반올림 정답으로 완성 가능한 prefix를 뜻한다. 소수점이나 종료 구분자 자체가 가능한 완성 후보를 없애는 경우에는 그 서식 step이 최초 오류다. 첫 숫자도 동일한 가능성 기준으로 파랑 또는 빨강으로 표시한다. 반올림 endpoint 일치와 전체 구간 IoU는 다른 지표다. 엔트로피는 원래 생성 경로의 분포이며 인과 효과를 뜻하지 않는다.

[전체 step](per_step.csv) · [endpoint](per_endpoint.csv) · [경우별](conditions.csv) · [자리별](role_conditions.csv) · [조건부 상태 비율](position_conditional_accuracy.csv) · [자릿수 통제](digit_stratified.csv)
