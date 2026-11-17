# Entropy 상승과 최초 오류 · full · 9,404 queries

**측정된 결과와 후속 실험 제안을 구분한다.** 이 appendix의 앞부분은 기존 greedy decoding 기록의 재집계다. 영상 증거를 바꿔 얻은 결과가 아니므로 영상 활용의 인과 효과는 아직 측정하지 않았다.

실제 query 수는 9,404, 고유 영상 수는 4,279다. 지원 표기의 18,591개 model×query 응답에서 37,182개 model×query×endpoint 경로와 182,440개 endpoint step을 읽었다. 두 모델의 반복 응답 수를 query 수로 세지 않으며, pilot과 full을 합산하지 않는다.

다른 저장 자료의 추가 확인: [다른 decoding 경로 · deterministic pilot 900](../../../../../rollout/deterministic/viewer.html?view=pilot%2F900%2FAPPENDIX_ENTROPY.md) · [고정 prefix 영상 perturbation · pilot 60](../../../../../rollout/oracle_switch_controls/viewer.html?view=pilot%2F60%2FAPPENDIX_ENTROPY.md)

## 먼저 읽을 결과

오류 때 entropy가 올라가는 사례는 많지만, 숫자 step에서는 정답 가능성을 유지할 때도 자주 올라간다. 특히 소수점 뒤 숫자는 이전 소수점의 entropy가 낮아 상승하기 쉽다. 따라서 상승 자체를 오류 신호로 바로 쓰기 전에 아래 오경보율을 확인해야 한다.

| 모델 | 현재 숫자 위치 | 오류 상승/오류 | 민감도 | safe 상승/safe | 오경보율 |
|---|---|---|---|---|---|
| qwen25 | integer | 13273/13626 | 97.4% | 15285/15708 | 97.3% |
| qwen25 | fractional | 1863/1869 | 99.7% | 3231/3236 | 99.8% |
| qwen3 | integer | 11255/11516 | 97.7% | 17317/18298 | 94.6% |
| qwen3 | fractional | 1934/1935 | 99.9% | 4780/4791 | 99.8% |

**표 해설.** 직전 decoding token 기준이며 모든 dataset·start/end의 해당 숫자 step을 합친 기술통계다. 소수부에서 민감도와 오경보율이 둘 다 높으면 소수점→숫자 전이의 정상적인 상승을 오류와 구분하지 못한다. 정수부도 동일하다. dataset별 수치, 경로 내부 비교, 통제 결과는 뒤 표에 따로 제시한다.

- 직전 decoding token 기준: Cpath 평균이 양수인 조건 12/12, 95% CI 전체가 0보다 큰 조건 12/12다(model×dataset×endpoint). role·직전 token 종류·숫자 순서 통제 후 β의 CI 전체가 양수인 조건은 식별 가능한 6개 model×dataset 중 5개, 전체가 음수인 조건은 0개다.
- 직전 정답 가능 숫자 기준: Cpath 평균이 양수인 조건 8/12, 95% CI 전체가 0보다 큰 조건 6/12다(model×dataset×endpoint). role·직전 token 종류·숫자 순서 통제 후 β의 CI 전체가 양수인 조건은 식별 가능한 4개 model×dataset 중 2개, 전체가 음수인 조건은 0개다.

같은 경로의 차이가 일부 조건에서 양수여도 상승 여부만으로 오류를 잘 분류한다는 뜻은 아니다. 위 숫자 step의 오경보율과 함께 판단한다.

단순한 “entropy 감소 방향의 decoding”은 목표로 충분하지 않다. 후속 실험의 성공 조건은 **관련 영상 증거가 들어왔을 때 정답 가능 후보의 확률이 늘고 오류가 줄어드는가**다. entropy 감소는 그와 함께 확인할 보조 지표이며, 관련 증거 때문에 기존의 잘못된 확신이 깨지면 오히려 entropy가 증가할 수도 있다.

## 1. 어떤 값이 오류의 신호인가

Hₖ는 **raw top-30 재정규화 entropy**다. 해당 step의 top-30 확률 pᵢ를 qᵢ=pᵢ/Σ₁³⁰pᵢ로 바꾸고 Hₖ=−Σ₁³⁰qᵢlnqᵢ를 계산한다. 단위는 nat, 범위는 0–ln30≈3.401이다. top-30 밖의 불확실성은 포함하지 않는다. entropy는 다음 token 선택 **전**의 분포이고 오류 판정은 실제 token을 고른 **후**의 판정이다.

오류는 허용된 표기법 안에서 반올림 endpoint 정답으로 완성할 수 없게 되는 최초 step이다. 최종 구간 IoU 오류, 영상 이해 오류, 정답 token과의 문자 불일치와 같지 않다. start/end 경로를 분리한다. risk step은 이전 prefix가 정답 가능성을 유지한 step(정답 유지 또는 최초 오류)이며 최초 오류 이후는 모두 제외한다.

두 변화량을 따로 계산한다. Δtokenₖ=Hₖ−Hₖ₋₁은 바로 앞 decoding token을 쓰며 endpoint 밖 token도 포함한다. Δnumericₖ=Hₖ−Hⱼ는 같은 endpoint 안의 직전 정답 가능 숫자 j를 쓰며 현재 step도 숫자여야 한다. 따라서 첫 숫자와 format 오류에는 Δnumeric이 없다. Δ>0은 상승, Δ<0은 감소, Δ=0은 둘 다 아니다.

가장 중요한 비교는 **같은 경로에서 오류 때의 변화가 평소 정답 유지 step의 변화보다 컸는가**다. Cpath=Δ(first error)−mean Δ(그 경로의 앞선 safe steps). 오류와 비교 가능한 safe 변화가 둘 다 있는 경로만 포함한다. 양수는 오류 순간에 상대적으로 더 많이 상승했다는 뜻이며, 오류 때 Δ 자체가 양수라는 뜻은 아니다. token 기준의 safe step에는 format도 포함되어 아래 role 통제를 함께 확인해야 한다.

CI는 model·dataset·endpoint 안에서 query 단위 percentile bootstrap 1,000회(seed 42)다. 두 endpoint를 합친 path-balanced 지표는 query 안에서 먼저 평균한다. 여러 query가 같은 영상을 공유할 수 있어 video 의존성까지 보정하지는 않는다. 단일 query는 CI를 제시하지 않는다. 다중 비교 보정 없는 탐색적 집계이며, 작은 표본 결과를 일반화하지 않는다.

## 2. 같은 경로의 평소 변화와 최초 오류 변화

| 모델 | dataset | endpoint | 비교 기준 | paired 경로 / query | Cpath 평균 [95% CI] nat | 상승 비율 차이 pp [95% CI] |
|---|---|---|---|---|---|---|
| qwen25 | activitynet-timelens | start | 직전 decoding token | 1830 / 1830 | 0.458 [0.402, 0.517] | 7.9 [6.8, 9.2] |
| qwen25 | activitynet-timelens | end | 직전 decoding token | 2076 / 2076 | 0.705 [0.656, 0.753] | -0.6 [-2.0, 0.8] |
| qwen25 | activitynet-timelens | start | 직전 정답 가능 숫자 | 676 / 676 | -0.111 [-0.189, -0.029] | 4.7 [1.7, 7.8] |
| qwen25 | activitynet-timelens | end | 직전 정답 가능 숫자 | 650 / 650 | 0.549 [0.481, 0.618] | 7.4 [4.8, 10.2] |
| qwen25 | charades-timelens | start | 직전 decoding token | 1264 / 1264 | 0.779 [0.710, 0.847] | 13.3 [11.6, 15.2] |
| qwen25 | charades-timelens | end | 직전 decoding token | 1437 / 1437 | 0.871 [0.810, 0.931] | 0.6 [-1.1, 2.3] |
| qwen25 | charades-timelens | start | 직전 정답 가능 숫자 | 263 / 263 | -0.702 [-0.799, -0.607] | -3.0 [-8.4, 2.3] |
| qwen25 | charades-timelens | end | 직전 정답 가능 숫자 | 203 / 203 | -0.542 [-0.631, -0.444] | -1.5 [-6.4, 3.9] |
| qwen25 | qvhighlights-timelens | start | 직전 decoding token | 694 / 694 | 0.833 [0.749, 0.914] | 9.0 [7.5, 10.6] |
| qwen25 | qvhighlights-timelens | end | 직전 decoding token | 740 / 740 | 0.577 [0.492, 0.660] | -1.5 [-3.5, 0.3] |
| qwen25 | qvhighlights-timelens | start | 직전 정답 가능 숫자 | 297 / 297 | 0.049 [-0.054, 0.159] | 7.6 [4.4, 10.9] |
| qwen25 | qvhighlights-timelens | end | 직전 정답 가능 숫자 | 242 / 242 | 0.679 [0.576, 0.771] | 2.9 [-0.8, 6.6] |
| qwen3 | activitynet-timelens | start | 직전 decoding token | 1462 / 1462 | 0.785 [0.724, 0.844] | 9.8 [8.0, 11.7] |
| qwen3 | activitynet-timelens | end | 직전 decoding token | 2081 / 2081 | 0.721 [0.682, 0.764] | 2.1 [0.9, 3.3] |
| qwen3 | activitynet-timelens | start | 직전 정답 가능 숫자 | 570 / 570 | 0.592 [0.531, 0.652] | 23.1 [19.3, 26.8] |
| qwen3 | activitynet-timelens | end | 직전 정답 가능 숫자 | 623 / 623 | 0.487 [0.423, 0.555] | 8.2 [4.8, 11.4] |
| qwen3 | charades-timelens | start | 직전 decoding token | 1259 / 1259 | 1.364 [1.310, 1.417] | 21.7 [20.2, 23.1] |
| qwen3 | charades-timelens | end | 직전 decoding token | 1568 / 1568 | 0.943 [0.899, 0.985] | 2.0 [0.9, 3.1] |
| qwen3 | charades-timelens | start | 직전 정답 가능 숫자 | 369 / 369 | 0.034 [-0.032, 0.107] | 4.3 [2.4, 6.8] |
| qwen3 | charades-timelens | end | 직전 정답 가능 숫자 | 135 / 135 | -0.134 [-0.277, 0.011] | 9.6 [4.4, 14.8] |
| qwen3 | qvhighlights-timelens | start | 직전 decoding token | 449 / 449 | 0.894 [0.796, 0.990] | 10.4 [7.4, 13.5] |
| qwen3 | qvhighlights-timelens | end | 직전 decoding token | 766 / 766 | 0.793 [0.725, 0.860] | 2.4 [0.4, 4.5] |
| qwen3 | qvhighlights-timelens | start | 직전 정답 가능 숫자 | 209 / 209 | 0.702 [0.620, 0.782] | 18.4 [13.6, 23.2] |
| qwen3 | qvhighlights-timelens | end | 직전 정답 가능 숫자 | 291 / 291 | 0.533 [0.433, 0.635] | 7.0 [2.2, 12.4] |

**표 해설.** 각 행은 같은 model·dataset·endpoint 안의 비교다. Cpath가 0보다 크면 그 sample에서 정답 가능성을 유지하던 step보다 최초 오류 step의 entropy 변화가 컸다. 상승 비율 차이는 1[오류 Δ>0]−그 경로의 safe step 상승 비율을 query별 평균한 값이다. 이전 safe 변화가 없는 첫 숫자 오류 등은 이 표에서 빠지므로 전체 오류의 탐지 성능으로 해석하면 안 된다.

![Within-path entropy 변화 대비](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/entropy_error/figures/within_path_delta_contrast.svg)

**그래프 해설.** 점은 위 표의 Cpath 평균, 오차 막대는 query bootstrap 95% CI다. 파랑(token)은 직전 decoding step, 주황(numeric)은 직전 정답 가능 숫자를 비교 대상으로 쓴다. 0선 위의 점은 오류 순간 변화가 같은 경로의 이전 변화보다 컸다는 뜻이다. 두 색이 다르면 소수점·문법 token과 숫자를 오가는 효과가 섞였는지 확인해야 한다.

## 3. 숫자 위치와 직전 token 종류를 통제해도 남는가

Δₚₖ = αpath + β·1[first error] + 현재 role + 직전 token 종류 + digit ordinal + ε로 비교한다. 각 경로의 평균을 빼서 αpath를 제거하고, 경로별 전체 가중치를 1로 맞춘 weighted least squares를 사용한다. 경로 안에 safe와 error 변화가 모두 있어야 한다. 95% CI는 query-cluster sandwich SE와 t 분포로 계산한다. 이전 token이 endpoint 밖이면 outside_endpoint로 묶고, 나머지는 숫자·소수점·공백·기타 서식으로 분류한다. β가 양수면 이 통제 아래 최초 오류 변화가 더 큰 연관성이 남았다는 뜻이다. 인과효과가 아니다.

| 모델 | dataset | 비교 기준 | query / 경로 / step | β [95% CI] nat |
|---|---|---|---|---|
| qwen25 | activitynet-timelens | 직전 decoding token | 2782 / 3906 / 10142 | 0.072 [0.014, 0.130] |
| qwen25 | activitynet-timelens | 직전 정답 가능 숫자 | 1071 / 1326 / 2824 | -0.079 [-0.206, 0.048] |
| qwen25 | charades-timelens | 직전 decoding token | 2077 / 2701 / 6670 | 0.070 [-0.042, 0.182] |
| qwen25 | charades-timelens | 직전 정답 가능 숫자 | 454 / 466 / 932 | — [—, —] |
| qwen25 | qvhighlights-timelens | 직전 decoding token | 1009 / 1434 / 3770 | 0.084 [0.004, 0.164] |
| qwen25 | qvhighlights-timelens | 직전 정답 가능 숫자 | 433 / 539 / 1167 | 0.062 [-0.097, 0.222] |
| qwen3 | activitynet-timelens | 직전 decoding token | 2624 / 3543 / 9169 | 0.217 [0.160, 0.274] |
| qwen3 | activitynet-timelens | 직전 정답 가능 숫자 | 1007 / 1193 / 2507 | 0.159 [0.013, 0.306] |
| qwen3 | charades-timelens | 직전 decoding token | 2140 / 2827 / 7048 | 0.133 [0.053, 0.213] |
| qwen3 | charades-timelens | 직전 정답 가능 숫자 | 480 / 504 / 1008 | — [—, —] |
| qwen3 | qvhighlights-timelens | 직전 decoding token | 952 / 1215 / 3320 | 0.235 [0.157, 0.314] |
| qwen3 | qvhighlights-timelens | 직전 정답 가능 숫자 | 432 / 500 / 1092 | 0.504 [0.303, 0.706] |

**표 해설.** 소수점은 보통 후보가 좁아 entropy가 낮으므로 소수점→숫자라는 이유만으로 Δtoken이 커질 수 있다. 이 표는 현재 숫자/format role과 직전 token 종류, 숫자 순서를 통제한 민감도 분석이다. 단순 paired 결과와 방향이 다르면 “entropy 상승=오류”라는 일반 규칙보다 전이 유형·자릿수의 영향에 주의해야 한다. 범주 통제로 모든 혼란 요인이 제거되는 것은 아니다. β가 —인 행은 통제변수와 오류 위치가 겹쳐 오류 계수를 별도로 식별할 수 없는 경우다. 효과가 0이라는 뜻이 아니다.

## 4. 상승하면 실제 오류인가: 정답 유지 step도 분모에 넣기

아래 지표는 사후에 오류인 경로만 고르지 않고 모든 eligible risk step에서 계산한다. 민감도=P(Δ>0 | 최초 오류), 오경보율=P(Δ>0 | safe), precision=P(최초 오류 | Δ>0), 비상승 오류율=P(최초 오류 | Δ≤0)이다. 이 표의 비율은 step 기준이며 긴 경로는 더 많은 step을 기여한다. 배포 성능 검증이나 새로운 threshold 학습은 아니다.

### 직전 decoding token 기준

| 모델 | dataset | 최초 오류 / safe step | 오류 상승 / safe 상승 | 민감도 | 오경보율 | precision | 비상승 오류율 |
|---|---|---|---|---|---|---|---|
| qwen25 | activitynet-timelens | 7503 / 13131 | 7283 / 9087 | 97.1% | 69.2% | 44.5% | 5.2% |
| qwen25 | charades-timelens | 5555 / 9167 | 5395 / 5920 | 97.1% | 64.6% | 47.7% | 4.7% |
| qwen25 | qvhighlights-timelens | 2514 / 4987 | 2459 / 3510 | 97.8% | 70.4% | 41.2% | 3.6% |
| qwen3 | activitynet-timelens | 6500 / 15880 | 6302 / 10284 | 97.0% | 64.8% | 38.0% | 3.4% |
| qwen3 | charades-timelens | 5069 / 11659 | 5003 / 7352 | 98.7% | 63.1% | 40.5% | 1.5% |
| qwen3 | qvhighlights-timelens | 1983 / 7098 | 1915 / 4484 | 96.6% | 63.2% | 29.9% | 2.5% |

**표 해설.** 민감도가 높아도 오경보율이 비슷하게 높으면 오류를 잘 구분하는 신호가 아니다. precision은 오류의 기본 비율과 함께 달라진다. 숫자 기준은 현재 숫자+이전 safe 숫자가 있는 step만 포함하고 token 기준은 format과 첫 숫자도 포함하므로 두 행 집합의 분모가 다르다.

| 모델 | dataset | 오류 상승률 query 평균 [95% CI] | safe 상승률 query 평균 [95% CI] |
|---|---|---|---|
| qwen25 | activitynet-timelens | 0.968 [0.964, 0.972] | 0.800 [0.794, 0.807] |
| qwen25 | charades-timelens | 0.970 [0.965, 0.975] | 0.755 [0.747, 0.763] |
| qwen25 | qvhighlights-timelens | 0.970 [0.961, 0.978] | 0.809 [0.799, 0.820] |
| qwen3 | activitynet-timelens | 0.969 [0.964, 0.973] | 0.757 [0.750, 0.763] |
| qwen3 | charades-timelens | 0.987 [0.984, 0.990] | 0.731 [0.723, 0.738] |
| qwen3 | qvhighlights-timelens | 0.965 [0.956, 0.973] | 0.736 [0.725, 0.746] |

**표 해설.** 비율을 먼저 endpoint 경로 안에서 평균하고 두 endpoint가 있으면 query 안에서 다시 평균해 긴 경로의 영향력을 줄였다. 수치는 0–1 비율이다. 오류 열과 safe 열은 자료가 있는 query 집합이 다를 수 있으므로 둘의 차이가 동일 경로 paired 효과는 아니다. 동일 경로 비교는 2절을 본다.

### 직전 정답 가능 숫자 기준

| 모델 | dataset | 최초 오류 / safe step | 오류 상승 / safe 상승 | 민감도 | 오경보율 | precision | 비상승 오류율 |
|---|---|---|---|---|---|---|---|
| qwen25 | activitynet-timelens | 3855 / 3927 | 3572 / 3124 | 92.7% | 79.6% | 53.3% | 26.1% |
| qwen25 | charades-timelens | 2697 / 2151 | 2427 / 1695 | 90.0% | 78.8% | 58.9% | 37.2% |
| qwen25 | qvhighlights-timelens | 1412 / 1589 | 1367 / 1284 | 96.8% | 80.8% | 51.6% | 12.9% |
| qwen3 | activitynet-timelens | 3496 / 5102 | 3235 / 3837 | 92.5% | 75.2% | 45.7% | 17.1% |
| qwen3 | charades-timelens | 2790 / 2971 | 2629 / 2515 | 94.2% | 84.7% | 51.1% | 26.1% |
| qwen3 | qvhighlights-timelens | 1198 / 2609 | 1112 / 1974 | 92.8% | 75.7% | 36.0% | 11.9% |

**표 해설.** 민감도가 높아도 오경보율이 비슷하게 높으면 오류를 잘 구분하는 신호가 아니다. precision은 오류의 기본 비율과 함께 달라진다. 숫자 기준은 현재 숫자+이전 safe 숫자가 있는 step만 포함하고 token 기준은 format과 첫 숫자도 포함하므로 두 행 집합의 분모가 다르다.

| 모델 | dataset | 오류 상승률 query 평균 [95% CI] | safe 상승률 query 평균 [95% CI] |
|---|---|---|---|
| qwen25 | activitynet-timelens | 0.922 [0.913, 0.931] | 0.732 [0.715, 0.750] |
| qwen25 | charades-timelens | 0.897 [0.883, 0.908] | 0.762 [0.741, 0.783] |
| qwen25 | qvhighlights-timelens | 0.961 [0.947, 0.971] | 0.739 [0.710, 0.767] |
| qwen3 | activitynet-timelens | 0.922 [0.912, 0.931] | 0.694 [0.677, 0.710] |
| qwen3 | charades-timelens | 0.934 [0.923, 0.944] | 0.818 [0.801, 0.833] |
| qwen3 | qvhighlights-timelens | 0.928 [0.911, 0.942] | 0.699 [0.675, 0.721] |

**표 해설.** 비율을 먼저 endpoint 경로 안에서 평균하고 두 endpoint가 있으면 query 안에서 다시 평균해 긴 경로의 영향력을 줄였다. 수치는 0–1 비율이다. 오류 열과 safe 열은 자료가 있는 query 집합이 다를 수 있으므로 둘의 차이가 동일 경로 paired 효과는 아니다. 동일 경로 비교는 2절을 본다.

### 현재 step의 정수부·소수부·서식을 나누면

| 모델 | dataset | 기준 | 현재 위치 | 오류 / safe | 민감도 | 오경보율 | precision |
|---|---|---|---|---|---|---|---|
| qwen25 | activitynet-timelens | token | integer | 6645 / 7833 | 97.5% | 96.9% | 46.0% |
| qwen25 | activitynet-timelens | token | fractional | 807 / 1497 | 99.3% | 99.7% | 34.9% |
| qwen25 | activitynet-timelens | token | format | 51 / 3801 | 2.0% | 0.0% | 100.0% |
| qwen25 | activitynet-timelens | numeric | integer | 3048 / 2430 | 94.6% | 90.2% | 56.8% |
| qwen25 | activitynet-timelens | numeric | fractional | 807 / 1497 | 85.3% | 62.3% | 42.4% |
| qwen25 | charades-timelens | token | integer | 4749 / 4852 | 96.7% | 97.9% | 49.2% |
| qwen25 | charades-timelens | token | fractional | 802 / 1171 | 100.0% | 99.9% | 40.7% |
| qwen25 | charades-timelens | token | format | 4 / 3144 | 0.0% | 0.0% | — |
| qwen25 | charades-timelens | numeric | integer | 1895 / 980 | 91.8% | 89.6% | 66.5% |
| qwen25 | charades-timelens | numeric | fractional | 802 / 1171 | 85.8% | 69.8% | 45.7% |
| qwen25 | qvhighlights-timelens | token | integer | 2232 / 3023 | 98.5% | 97.3% | 42.8% |
| qwen25 | qvhighlights-timelens | token | fractional | 260 / 568 | 100.0% | 100.0% | 31.4% |
| qwen25 | qvhighlights-timelens | token | format | 22 / 1396 | 0.0% | 0.1% | 0.0% |
| qwen25 | qvhighlights-timelens | numeric | integer | 1152 / 1021 | 97.1% | 92.1% | 54.3% |
| qwen25 | qvhighlights-timelens | numeric | fractional | 260 / 568 | 95.4% | 60.6% | 41.9% |
| qwen3 | activitynet-timelens | token | integer | 5696 / 8671 | 97.1% | 93.8% | 40.5% |
| qwen3 | activitynet-timelens | token | fractional | 757 / 2142 | 99.9% | 99.6% | 26.2% |
| qwen3 | activitynet-timelens | token | format | 47 / 5067 | 31.9% | 0.4% | 44.1% |
| qwen3 | activitynet-timelens | numeric | integer | 2739 / 2960 | 94.0% | 81.8% | 51.5% |
| qwen3 | activitynet-timelens | numeric | fractional | 757 / 2142 | 87.3% | 66.2% | 31.8% |
| qwen3 | charades-timelens | token | integer | 4142 / 5798 | 99.0% | 98.2% | 41.9% |
| qwen3 | charades-timelens | token | fractional | 890 / 1657 | 100.0% | 100.0% | 34.9% |
| qwen3 | charades-timelens | token | format | 37 / 4204 | 35.1% | 0.0% | 86.7% |
| qwen3 | charades-timelens | numeric | integer | 1900 / 1314 | 97.8% | 92.0% | 60.6% |
| qwen3 | charades-timelens | numeric | fractional | 890 / 1657 | 86.6% | 78.8% | 37.1% |
| qwen3 | qvhighlights-timelens | token | integer | 1678 / 3829 | 96.8% | 91.2% | 31.7% |
| qwen3 | qvhighlights-timelens | token | fractional | 288 / 992 | 100.0% | 99.7% | 22.6% |
| qwen3 | qvhighlights-timelens | token | format | 17 / 2277 | 17.6% | 0.1% | 60.0% |
| qwen3 | qvhighlights-timelens | numeric | integer | 910 / 1617 | 94.1% | 79.2% | 40.1% |
| qwen3 | qvhighlights-timelens | numeric | fractional | 288 / 992 | 88.9% | 69.9% | 27.0% |

**표 해설.** 여기의 위치는 현재 risk step의 role이며, viewer의 “미래 최초 오류가 어디인가” 필터와 다르다. tenths가 소수부 숫자, decimal_point와 경계 token 등은 format이다. 같은 위치의 정답 유지와 오류를 비교하므로 소수점 뒤 숫자가 원래 상승하기 쉬운 현상을 확인할 수 있다. 오류 수가 적은 format 행의 극단 비율은 표본 수와 함께 읽는다.

## 5. Viewer 특수 필터별 최초 오류의 상승·감소

이 절은 viewer와 같은 사후 선택 조건의 **기술통계**다. 예를 들어 “직전 token 대비 상승”을 고르면 그 상승 비율은 정의상 100%다. “최초 오류 위치”, “낮은 entropy 오류”, “유지 직후 다음 숫자 오류” 등은 정답/미래 결과를 알아야 정할 수 있어 online trigger나 오류 예측의 근거로 쓸 수 없다. 필터는 서로 겹치며 행을 합산하지 않는다. 모델 안에서 dataset을 합친 탐색용 집계이므로 dataset 분포에 민감하다.

### qwen25

| 조건 | query / endpoint 경로 | 최초 오류 수 | token 상승/감소/동일 (eligible) | numeric 상승/감소/동일 (eligible) | 오류 Δtoken 평균 nat |
|---|---|---|---|---|---|
| 전체 경로 | 9404 / 18808 | 15572 | 15137/435/0 (15572) | 7366/598/0 (7964) | 1.172 |
| 오류 전후 숫자 비교 가능 | 5801 / 7964 | 7964 | 7605/359/0 (7964) | 7366/598/0 (7964) | 1.180 |
| 정수부 최초 오류 | 8709 / 13626 | 13626 | 13273/353/0 (13626) | 5742/353/0 (6095) | 1.070 |
| 소수부 최초 오류 | 1774 / 1869 | 1869 | 1863/6/0 (1869) | 1624/245/0 (1869) | 2.013 |
| 소수점·서식 최초 오류 | 77 / 77 | 77 | 1/76/0 (77) | 0/0/0 (0) | -1.152 |
| 오류 없음 | 2993 / 3236 | 0 | 0/0/0 (0) | 0/0/0 (0) | — |
| 직전 숫자 대비 상승 | 5431 / 7366 | 7366 | 7366/0/0 (7366) | 7366/0/0 (7366) | 1.258 |
| 직전 숫자 대비 감소 | 562 / 598 | 598 | 239/359/0 (598) | 0/598/0 (598) | 0.216 |
| 직전 token 대비 상승 | 9021 / 15137 | 15137 | 15137/0/0 (15137) | 7366/239/0 (7605) | 1.217 |
| 직전 token 대비 감소 | 427 / 435 | 435 | 0/435/0 (435) | 0/359/0 (359) | -0.410 |
| 낮은 entropy 오류 (H≤0.5) | 2254 / 2306 | 2306 | 2153/153/0 (2306) | 429/197/0 (626) | 0.119 |
| 높은 entropy 유지 (H≥1.5) | 3880 / 4254 | 2262 | 1996/266/0 (2262) | 1852/345/0 (2197) | 1.313 |
| 유지 직후 다음 숫자 오류 | 5801 / 7964 | 7964 | 7605/359/0 (7964) | 7366/598/0 (7964) | 1.180 |
| 두 모델 판정 불일치 | 3548 / 4086 | 2866 | 2794/72/0 (2866) | 1647/102/0 (1749) | 1.321 |

**표 해설.** eligible은 해당 비교 대상이 존재하는 최초 오류 경로 수다. format 오류는 token 기준에 들어가지만 numeric 기준에는 들어가지 않는다. high-safe는 오류 여부와 무관하게 H≥1.5인 safe numeric step이 하나 이상인 경로, next-failure는 safe numeric step 다음 숫자에서 최초 오류가 난 경로다. 모델 불일치는 두 모델 모두 지원 표기인 경우의 선택 endpoint 최종 반올림 정답 판정 불일치다. 이 평균은 endpoint 경로 가중 평균이며 검증된 오류 탐지 임계값은 아니다.

### qwen3

| 조건 | query / endpoint 경로 | 최초 오류 수 | token 상승/감소/동일 (eligible) | numeric 상승/감소/동일 (eligible) | 오류 Δtoken 평균 nat |
|---|---|---|---|---|---|
| 전체 경로 | 9187 / 18374 | 13552 | 13220/332/0 (13552) | 6976/508/0 (7484) | 1.042 |
| 오류 전후 숫자 비교 가능 | 5630 / 7484 | 7484 | 7222/262/0 (7484) | 6976/508/0 (7484) | 1.172 |
| 정수부 최초 오류 | 7943 / 11516 | 11516 | 11255/261/0 (11516) | 5288/261/0 (5549) | 0.910 |
| 소수부 최초 오류 | 1831 / 1935 | 1935 | 1934/1/0 (1935) | 1688/247/0 (1935) | 1.917 |
| 소수점·서식 최초 오류 | 99 / 101 | 101 | 31/70/0 (101) | 0/0/0 (0) | -0.636 |
| 오류 없음 | 4181 / 4822 | 0 | 0/0/0 (0) | 0/0/0 (0) | — |
| 직전 숫자 대비 상승 | 5312 / 6976 | 6976 | 6976/0/0 (6976) | 6976/0/0 (6976) | 1.257 |
| 직전 숫자 대비 감소 | 501 / 508 | 508 | 246/262/0 (508) | 0/508/0 (508) | 0.016 |
| 직전 token 대비 상승 | 8447 / 13220 | 13220 | 13220/0/0 (13220) | 6976/246/0 (7222) | 1.080 |
| 직전 token 대비 감소 | 327 / 332 | 332 | 0/332/0 (332) | 0/262/0 (262) | -0.452 |
| 낮은 entropy 오류 (H≤0.5) | 2231 / 2418 | 2418 | 2273/145/0 (2418) | 338/302/0 (640) | 0.135 |
| 높은 entropy 유지 (H≥1.5) | 3331 / 3668 | 694 | 587/107/0 (694) | 527/99/0 (626) | 1.308 |
| 유지 직후 다음 숫자 오류 | 5630 / 7484 | 7484 | 7222/262/0 (7484) | 6976/508/0 (7484) | 1.172 |
| 두 모델 판정 불일치 | 3548 / 4086 | 1220 | 1205/15/0 (1220) | 797/13/0 (810) | 1.222 |

**표 해설.** eligible은 해당 비교 대상이 존재하는 최초 오류 경로 수다. format 오류는 token 기준에 들어가지만 numeric 기준에는 들어가지 않는다. high-safe는 오류 여부와 무관하게 H≥1.5인 safe numeric step이 하나 이상인 경로, next-failure는 safe numeric step 다음 숫자에서 최초 오류가 난 경로다. 모델 불일치는 두 모델 모두 지원 표기인 경우의 선택 endpoint 최종 반올림 정답 판정 불일치다. 이 평균은 endpoint 경로 가중 평균이며 검증된 오류 탐지 임계값은 아니다.

## 6. 현재 자료에서 말할 수 있는 것과 없는 것

entropy 상승과 오류의 연관성을 판단하려면 ① error-only 필터의 비율이 아니라 safe step을 포함한 오경보율, ② 동일 경로 paired 차이, ③ role·전이 종류를 통제한 결과를 함께 봐야 한다. 오류 순간에 entropy가 낮아지는 사례가 있다는 사실은 단순 entropy 최소화가 정답을 보장하지 않음을 뜻한다. 낮은 entropy는 영상 근거에 맞는 확신일 수도 있고 잘못된 답에 대한 확신일 수도 있다.

앞에서 집계한 greedy 기록은 영상 입력을 고정한 자료다. Hₖ−Hₖ₋₁에는 prefix 변화, 출력 문법, 자릿수, 새로 선택한 token의 영향이 섞이므로 이를 “영상 증거가 줄인 entropy”로 부를 수 없다. 별도로 저장된 영상 perturbation probe의 재집계는 아래 링크에서 구분해 확인한다. 자연스러운 관련 증거의 효과는 별도 대조 실험이 필요하다.

## 다른 저장된 decoding 자료에서의 확인

별도 실험에 저장된 분기 경로와 영상 perturbation probe도 재집계했다. 아래 appendix는 원래 실험 계열과 실제 query 수 아래 보존하며 이 보고서의 표본과 합산하지 않는다.

- [다른 decoding 경로 · deterministic pilot 900](../../../../../rollout/deterministic/viewer.html?view=pilot%2F900%2FAPPENDIX_ENTROPY.md)
- [고정 prefix 영상 perturbation · pilot 60](../../../../../rollout/oracle_switch_controls/viewer.html?view=pilot%2F60%2FAPPENDIX_ENTROPY.md)

deterministic 분기에서는 강제로 선택한 token과 이후 자유 생성 step을 구분한다. 영상 probe는 동일 prefix의 비교지만 GT 후보를 겨냥한 입력 perturbation이므로 자연스러운 증거 추가·제거와 구분한다. probe의 entropy는 원본에서 고정한 30개 후보를 사용하므로 각 조건의 native top-30 entropy와 같은 값으로 섞지 않는다.

## 영상 증거와 entropy를 연결하는 다음 실험 — 설계 제안, 미실행

목표를 **“entropy가 낮아지는 경로”보다 “영상 증거 때문에 정답 후보 쪽으로 확신이 모이는 경로”**로 잡는 것이 좋다. 앞의 표는 기존 응답에서 관측한 연관성이다. 아래는 그 연관성을 실제 영상 활용과 연결하기 위한 새 실험 설계이며, 아직 실행하거나 성능 개선을 확인한 결과가 아니다.

### 먼저 구분할 두 가지 변화

현재 viewer의 `Hₜ − Hₜ₋₁`은 서로 다른 prefix에서 계산한 분포의 차이다. 숫자 뒤에 소수점이 올 때와 소수점 뒤에 숫자가 올 때는 다음 token의 종류 자체가 달라진다. 같은 path로 제한해도 이 문법 효과는 남는다. 또한 `Hₜ`는 t번째 token을 고르기 **전** 분포의 entropy다. 그 분포에서 어떤 token을 선택했다고 이미 계산된 `Hₜ`가 낮아지는 것은 아니다. 선택은 다음 prefix와 `Hₜ₊₁`에 영향을 준다.

영상 효과를 보려면 query와 token prefix `y<ₜ`를 완전히 고정하고 영상 입력만 바꿔야 한다. `pₜ⁺ = p(token | query, y<ₜ, V⁺)`, `pₜ⁻ = p(token | query, y<ₜ, V⁻)`라 두자. `V⁺`는 증거가 있는 입력, `V⁻`는 해당 증거를 약화한 입력이다. 이때 `Gₜ = H(pₜ⁻) − H(pₜ⁺)`를 비교한다. 양수면 그 입력 개입에서 증거가 있는 쪽이 더 집중됐다는 뜻이다. 이는 **정해진 입력 개입에 대한 모델의 반응**이며, 영상의 참된 인과 구조나 정답임을 자동으로 입증하지 않는다.

`Gₜ`가 양수라고 정답인 것도, 음수라고 증거가 쓸모없는 것도 아니다. 잘못된 확신을 깨는 증거는 먼저 entropy를 높일 수 있다. 따라서 `Gₜ`와 정답 가능 후보의 probability mass 변화를 함께 본다. 서식 token의 entropy 감소는 숫자 판단 개선과 분리한다.

### 1단계: 같은 prefix에서 증거 제거와 복원을 비교한다

우선 기존 pilot 300개 query에서 저장된 prefix를 그대로 재생해 다음 token 분포만 다시 계산하는 실험을 제안한다. 아직 오류가 나지 않은 numeric prefix 전체를 주 분석 대상으로 삼고, 최초 오류 직전 prefix를 별도 표기한다. 오류 직전 사례만 고르면 어떤 경우에도 오류를 고른 표본이 되므로 정상 선택도 포함한다. 이미 틀린 prefix는 복구 가능성 연구로 분리한다. 기존 full 9,404개 결과와 pilot은 독립 표본으로 합산하지 않는다.

| 조건 | 입력을 어떻게 바꾸나 | 이 비교로 확인하는 것 |
|---|---|---|
| 원본 | 기존 frame IDs·timestamp·H×W·prompt·prefix 그대로 | 재실행 일치와 기준 분포 |
| 관련 증거 제거 | query와 관계된 구간의 frame 내용만 마스킹·약화. frame 수와 timestamp 위치는 유지 | 그 구간에 모델이 실제로 반응하는가 |
| 대조 구간 제거 | 같은 수·길이·처리 강도로 다른 구간을 제거. 여러 고정 seed 사용 | 어떤 영상 손상에도 생기는 효과와 관련 증거의 효과를 구분 |
| 시간 순서 교란 | frame 수·해상도·내용 집합을 유지하고 내용의 시간 순서를 교란. timestamp 슬롯 규칙을 고정·기록 | 정적 장면 정보보다 시간 순서에 민감한가. 인위적 불일치 영향도 포함 |
| query 기반 증거 보강 | coarse context는 남기고 일부 frame 예산을 query 관련 구간과 경계 주변에 배정 | 같은 예산에서 더 유용한 frame을 보여 주면 판단이 바뀌는가 |
| 무작위 보강 대조 | 같은 frame 예산과 같은 처리로 무작위 구간에 재배정 | 추가 계산·샘플링 변화 자체의 효과를 구분 |

**표 해설.** 핵심은 관련 구간을 지웠을 때와 대조 구간을 지웠을 때의 차이다. 화면이 손상되기만 해도 entropy가 오를 수 있으므로 원본 대 제거의 차이 하나로 결론 내리지 않는다. 보강 조건에서는 총 frame 수·각 frame 해상도·visual token 예산을 맞추고, 바뀐 frame timestamp와 전처리 시간을 기록한다. 순서 교란은 별도 진단이며 모든 오류를 시간 증거 부족으로 해석하는 근거가 되지 않는다.

실제 사용할 구간은 **GT를 보지 않는** query 기반 retrieval 또는 사전에 고정한 coarse prediction으로 선택한다. GT 구간을 제거·보강하는 조건은 “정확한 증거 위치를 안다면 효과가 있는가”를 보는 oracle 진단으로만 둔다. oracle 결과와 배포 가능한 선택기의 결과를 같은 성능으로 보고하지 않는다. 원본 전체 영상을 잘라 시간 좌표를 다시 매기면 답 자체가 바뀔 수 있으므로 원래 영상의 시간축과 timestamp 표기를 유지한다.

모든 조건에서 모델 revision, prompt, prefix token IDs, raw logits의 temperature(1), frame 수·실제 H×W, 시간축 병합 설정을 맞춘다. 입력 영상이 달라지면 visual encoding과 그에 의존하는 KV cache를 다시 계산한다. 원본 영상의 cache를 그대로 공유하면 영상 개입을 제대로 평가하지 못할 수 있다. 원본 재실행의 raw top-30과 저장값이 먼저 일치해야 한다.

### 무엇을 수치로 볼 것인가

| 지표 | 정의·분모 | 해석 |
|---|---|---|
| 증거에 따른 entropy 차이 | `Gₜ = H(V⁻, y<ₜ) − H(V⁺, y<ₜ)`; 같은 query·prefix 쌍 | 양수는 증거가 있는 조건에서 분포가 더 집중됨 |
| 관련 구간의 선택적 효과 | `Gₜ(관련 제거) − Gₜ(대조 제거)`; 같은 prefix·동일 손상량 | 양수면 관련 구간 제거의 entropy 효과가 더 큼. 손상 방식 의존성을 함께 보고 |
| 정답 가능 mass 변화 | `ΔSₜ = Σ_{c∈C_safe} pₜ⁺(c) − Σ_{c∈C_safe} pₜ⁻(c)` | 양수면 정답으로 완성할 수 있는 후보 쪽으로 이동. 최종 정답 확률은 아님 |
| 선택 교정·훼손률 | 동일 prefix에서 오답→가능 후보, 가능→오답으로 argmax가 바뀐 쌍의 수 / 평가 prefix 쌍 수 | 분포가 날카로워졌어도 잘못된 선택을 강화했는지 확인 |
| 최종 성능 | 별도의 실제 rollout에서 endpoint 정답률, mIoU, IoU≥0.3·0.5·0.7의 query 비율과 기준 대비 pp | 한 step의 개선이 최종 temporal grounding 개선으로 이어지는지 확인 |
| 비용·개입 비율 | query당 재인코딩 횟수, forward 수, wall time, peak memory, 개입 step 수 / 대상 numeric step 수 | 동일 계산 예산의 대조 정책과 비교 |

**표 해설.** 가장 설득력 있는 패턴은 관련 증거를 보여 줬을 때 entropy가 줄고(`Gₜ>0`), 정답 가능 mass가 늘며(`ΔSₜ>0`), 실제 rollout의 오류도 줄어드는 경우다. `Gₜ>0`인데 `ΔSₜ<0`이면 잘못된 확신을 강화한 반례다. 이 두 방향을 2×2로 나눠 query·prefix 수를 모두 적고, entropy가 높아지며 정답 후보가 살아나는 경우도 보존한다. 정답 가능 후보 집합은 같은 prefix와 같은 completion 규칙으로 평가하며, 판정하지 않은 후보를 오답으로 채우지 않는다.

기존 결과와 연결할 주 지표는 top-30 재정규화 `H₃₀`로 유지하되, 각 조건의 top-30 mass `M₃₀`와 후보 집합 변화를 함께 기록한다. 서로 다른 조건에서 각자 top-30을 뽑으면 비교 집합이 달라지므로 **공통 후보 집합**에서도 검사한다. 예를 들어 양쪽 top-30의 합집합 U를 만들고, 각 조건의 raw logits에서 U의 모든 token 확률을 직접 조회해 합이 1이 되게 정규화한다. U에 없는 token을 0으로 채우면 안 된다. 가능하면 같은 forward에서 full-vocabulary entropy도 보조 지표로 계산하되, 이를 기존 H₃₀와 같은 지표로 섞지 않는다. full-vocabulary safe mass가 불가능하면 U 내부에서 평가한 mass와 U 밖 mass·미판정 mass를 명시한다.

분석은 모델·dataset·start/end·numeric role별로 나누고, prefix를 평균한 뒤 query 또는 video 단위로 요약한다. 같은 video의 여러 query가 있으면 video 단위 paired bootstrap으로 95% CI를 계산한다. 검색·개입 정책과 임계값은 video가 겹치지 않는 개발 분할에서 정하고 평가 분할에서 고정한다. 기존 full 결과는 이미 열람했으므로 완전한 미열람 test라고 부르지 않는다. 특수 필터는 사례 해석용으로 함께 보고하되, 실제 최초 오류 위치나 정답 여부를 runtime trigger로 사용하지 않는다.

### 2단계: 영상에 의해 지지되는 후보를 선택하는 decoding

1단계에서 관련 증거의 효과가 보이면, 우선 numeric step에서만 query 기반 frame 재확인을 적용하는 작은 정책을 비교한다. entropy 상승·높은 entropy는 재확인 시점을 고르는 후보 신호다. 높은 entropy만으로 틀렸다고 판단하거나, 실제 최초 오류 annotation을 사용해 재확인을 시작하지 않는다. 같은 재확인 횟수의 무작위 trigger와 고정 주기 trigger를 대조군으로 둔다.

후보 c의 영상 의존 점수를 `Eₜ(c) = log p(c | V⁺, y<ₜ) − log p(c | V⁻, y<ₜ)`로 정의하고, 예시 선택 점수를 `scoreₜ(c) = log p(c | V⁺, y<ₜ) + αEₜ(c)`로 둘 수 있다. α≥0은 개발 분할에서 결정한다. 원본 또는 증거 입력에서 이미 그럴듯한 후보만 남기는 probability cutoff를 함께 두어, 두 조건 모두 확률이 극히 작은 후보의 큰 비율 차이가 선택을 지배하지 않게 한다. 계산에는 같은 prefix의 raw log-softmax 값을 사용하며, 두 조건의 잘린 top-30만 서로 빼지 않는다.

이는 증거가 사라질 때 지지를 잃는 후보를 우대하는 **검증할 정책 예시**다. `Eₜ(c)>0` 자체는 정답 보장이 아니므로, entropy 변화가 작거나 증거 개입이 불안정한 경우 원래 decoding으로 돌아가는 gate를 개발 분할에서 검증한다. α=0, 재확인만 적용, contrastive score만 적용, entropy만 낮추는 temperature 대조군, 재확인+score를 구분해 평가한다. temperature를 낮추는 것은 영상 증거를 더 활용하지 않아도 분포를 날카롭게 만들 수 있다. 수정된 score 분포의 entropy 감소와 **모델 raw 분포에서 관측한 entropy 감소**도 별도 열로 기록한다.

후보를 선택한 뒤 `Hₜ₊₁`이 가장 작은 branch를 고르는 방법은 첫 실험의 주목표로 삼지 않는 편이 좋다. 확신한 오답이나 소수점·종료 token 같은 쉬운 문법 branch가 유리해질 수 있기 때문이다. branch 비교를 하더라도 동일한 길이·numeric 의사결정 위치에서 영상 증거 점수와 정답 성능을 함께 평가해야 한다. 정상 후보를 오답으로 바꾸는 비율과 종료·서식 분포도 기록한다.

### 이 설계가 참고한 선행 연구

원본과 훼손된 시각 입력의 logits를 대비하는 발상은 [Visual Contrastive Decoding, CVPR 2024 공식 구현](https://github.com/DAMO-NLP-SG/VCD)에 근거한다. 해당 방법은 이미지 object hallucination을 다루며, 여기의 temporal grounding 숫자 정확도나 entropy 감소를 입증한 결과는 아니다.

[Video Contrastive Decoding, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ea6d17af54f827336fc8fed27ca0319d-Abstract-Conference.html)는 시간적으로 일관된 영상과 시간 구조를 훼손한 영상의 decoding을 대비하는 접근을 제안한다. 시간 순서 대조군을 두는 데 참고할 수 있다.

[SEASON, CVPR 2026 연구팀 소개](https://research.nvidia.com/labs/twn/publication/cvpr_2026_season/)는 token별 진단과 temporal·spatial negative를 이용한 adaptive contrastive decoding을 제안한다. 모든 token을 똑같이 처리하지 않는 gate 설계에 참고할 수 있으나, 현재 모델·dataset에서의 효과는 별도 검증해야 한다.

현재 단계에서 먼저 할 일은 **고정 prefix + 관련/대조 증거 제거 실험**이다. 이 실험에서 정답 후보를 지지하는 영상 효과가 확인되면 query 기반 재확인과 contrastive decoding으로 확장한다. entropy 감소만을 성공 기준으로 삼지는 않는다.
## 재현과 자료

생성기: `zero-shot/_lib/entropy_error_appendix.py`. bootstrap seed=42, repeats=1000. 모델 재추론 없이 기존 자료만 읽는다. 원본 step entropy와 top-30 자료를 182,440개 step에서 대조하고 viewer의 최초 오류 Δ와도 대조했다.

[집계 JSON](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/entropy_error/metrics/summary.json) · [출처와 SHA-256](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/entropy_error/metadata/sources.json) · [검증 기록](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/entropy_error/validation/checks.json)

- digit_steps: [per_step.csv](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/metrics/per_step.csv)
- all_steps: [per_step.csv.gz](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/top30_renormalized/metrics/per_step.csv.gz)
- paths: [per_path.json](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/within_path/metrics/per_path.json)
- catalog: [viewer_catalog.json](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/metrics/viewer_catalog.json)
- manifest: [manifest.js](../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/web/manifest.js)
