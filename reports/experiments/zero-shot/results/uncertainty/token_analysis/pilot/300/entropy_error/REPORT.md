# Entropy 상승과 최초 오류 · pilot · 300 queries

**측정된 결과와 후속 실험 제안을 구분한다.** 이 appendix의 앞부분은 기존 greedy decoding 기록의 재집계다. 영상 증거를 바꿔 얻은 결과가 아니므로 영상 활용의 인과 효과는 아직 측정하지 않았다.

실제 query 수는 300, 고유 영상 수는 292다. 지원 표기의 594개 model×query 응답에서 1,188개 model×query×endpoint 경로와 5,905개 endpoint step을 읽었다. 두 모델의 반복 응답 수를 query 수로 세지 않으며, pilot과 full을 합산하지 않는다.

다른 저장 자료의 추가 확인: [다른 decoding 경로 · deterministic pilot 900](../../../../../rollout/deterministic/viewer.html?view=pilot%2F900%2FAPPENDIX_ENTROPY.md) · [고정 prefix 영상 perturbation · pilot 60](../../../../../rollout/oracle_switch_controls/viewer.html?view=pilot%2F60%2FAPPENDIX_ENTROPY.md)

## 먼저 읽을 결과

오류 때 entropy가 올라가는 사례는 많지만, 숫자 step에서는 정답 가능성을 유지할 때도 자주 올라간다. 특히 소수점 뒤 숫자는 이전 소수점의 entropy가 낮아 상승하기 쉽다. 따라서 상승 자체를 오류 신호로 바로 쓰기 전에 아래 오경보율을 확인해야 한다.

| 모델 | 현재 숫자 위치 | 오류 상승/오류 | 민감도 | safe 상승/safe | 오경보율 |
|---|---|---|---|---|---|
| qwen25 | integer | 447/456 | 98.0% | 493/508 | 97.0% |
| qwen25 | fractional | 52/52 | 100.0% | 92/92 | 100.0% |
| qwen3 | integer | 370/382 | 96.9% | 565/603 | 93.7% |
| qwen3 | fractional | 53/53 | 100.0% | 150/150 | 100.0% |

**표 해설.** 직전 decoding token 기준이며 모든 dataset·start/end의 해당 숫자 step을 합친 기술통계다. 소수부에서 민감도와 오경보율이 둘 다 높으면 소수점→숫자 전이의 정상적인 상승을 오류와 구분하지 못한다. 정수부도 동일하다. dataset별 수치, 경로 내부 비교, 통제 결과는 뒤 표에 따로 제시한다.

- 직전 decoding token 기준: Cpath 평균이 양수인 조건 12/12, 95% CI 전체가 0보다 큰 조건 11/12다(model×dataset×endpoint). role·직전 token 종류·숫자 순서 통제 후 β의 CI 전체가 양수인 조건은 식별 가능한 6개 model×dataset 중 0개, 전체가 음수인 조건은 0개다.
- 직전 정답 가능 숫자 기준: Cpath 평균이 양수인 조건 7/12, 95% CI 전체가 0보다 큰 조건 5/12다(model×dataset×endpoint). role·직전 token 종류·숫자 순서 통제 후 β의 CI 전체가 양수인 조건은 식별 가능한 4개 model×dataset 중 1개, 전체가 음수인 조건은 0개다.

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
| qwen25 | activitynet-timelens | start | 직전 decoding token | 43 / 43 | 0.184 [-0.231, 0.596] | 7.9 [0.8, 14.7] |
| qwen25 | activitynet-timelens | end | 직전 decoding token | 46 / 46 | 0.565 [0.196, 0.856] | -5.1 [-15.9, 4.7] |
| qwen25 | activitynet-timelens | start | 직전 정답 가능 숫자 | 16 / 16 | -0.310 [-0.694, 0.070] | -6.2 [-25.0, 12.5] |
| qwen25 | activitynet-timelens | end | 직전 정답 가능 숫자 | 15 / 15 | 0.663 [0.206, 1.026] | 13.3 [0.0, 33.3] |
| qwen25 | charades-timelens | start | 직전 decoding token | 44 / 44 | 0.986 [0.739, 1.246] | 13.6 [8.0, 20.1] |
| qwen25 | charades-timelens | end | 직전 decoding token | 31 / 31 | 0.538 [0.081, 0.952] | -0.5 [-11.8, 9.7] |
| qwen25 | charades-timelens | start | 직전 정답 가능 숫자 | 6 / 6 | -0.769 [-1.148, -0.412] | 0.0 [0.0, 0.0] |
| qwen25 | charades-timelens | end | 직전 정답 가능 숫자 | 3 / 3 | -0.313 [-0.867, 0.478] | 33.3 [0.0, 100.0] |
| qwen25 | qvhighlights-timelens | start | 직전 decoding token | 54 / 54 | 0.882 [0.616, 1.130] | 10.8 [6.3, 15.9] |
| qwen25 | qvhighlights-timelens | end | 직전 decoding token | 54 / 54 | 0.725 [0.455, 0.987] | 1.2 [-4.2, 5.7] |
| qwen25 | qvhighlights-timelens | start | 직전 정답 가능 숫자 | 24 / 24 | 0.031 [-0.260, 0.335] | 12.5 [2.1, 27.1] |
| qwen25 | qvhighlights-timelens | end | 직전 정답 가능 숫자 | 15 / 15 | 0.990 [0.647, 1.265] | 0.0 [0.0, 0.0] |
| qwen3 | activitynet-timelens | start | 직전 decoding token | 36 / 36 | 0.454 [0.090, 0.813] | 1.4 [-12.0, 13.9] |
| qwen3 | activitynet-timelens | end | 직전 decoding token | 49 / 49 | 0.892 [0.678, 1.090] | 3.7 [-2.7, 9.5] |
| qwen3 | activitynet-timelens | start | 직전 정답 가능 숫자 | 10 / 10 | 0.791 [0.219, 1.387] | 50.0 [20.0, 80.0] |
| qwen3 | activitynet-timelens | end | 직전 정답 가능 숫자 | 10 / 10 | 0.666 [-0.032, 1.173] | 5.0 [-25.0, 35.0] |
| qwen3 | charades-timelens | start | 직전 decoding token | 35 / 35 | 1.282 [0.989, 1.568] | 21.4 [13.8, 29.0] |
| qwen3 | charades-timelens | end | 직전 decoding token | 38 / 38 | 0.685 [0.336, 0.981] | -3.1 [-11.8, 3.9] |
| qwen3 | charades-timelens | start | 직전 정답 가능 숫자 | 8 / 8 | -0.212 [-0.688, 0.451] | 12.5 [0.0, 37.5] |
| qwen3 | charades-timelens | end | 직전 정답 가능 숫자 | 1 / 1 | -0.436 [—, —] | 0.0 [—, —] |
| qwen3 | qvhighlights-timelens | start | 직전 decoding token | 34 / 34 | 1.000 [0.628, 1.368] | 3.7 [-11.3, 16.9] |
| qwen3 | qvhighlights-timelens | end | 직전 decoding token | 53 / 53 | 0.812 [0.517, 1.096] | 1.9 [-6.8, 9.9] |
| qwen3 | qvhighlights-timelens | start | 직전 정답 가능 숫자 | 18 / 18 | 0.762 [0.440, 1.054] | 13.9 [2.8, 27.8] |
| qwen3 | qvhighlights-timelens | end | 직전 정답 가능 숫자 | 20 / 20 | 0.501 [0.109, 0.898] | 17.5 [5.0, 35.0] |

**표 해설.** 각 행은 같은 model·dataset·endpoint 안의 비교다. Cpath가 0보다 크면 그 sample에서 정답 가능성을 유지하던 step보다 최초 오류 step의 entropy 변화가 컸다. 상승 비율 차이는 1[오류 Δ>0]−그 경로의 safe step 상승 비율을 query별 평균한 값이다. 이전 safe 변화가 없는 첫 숫자 오류 등은 이 표에서 빠지므로 전체 오류의 탐지 성능으로 해석하면 안 된다.

![Within-path entropy 변화 대비](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_error/figures/within_path_delta_contrast.svg)

**그래프 해설.** 점은 위 표의 Cpath 평균, 오차 막대는 query bootstrap 95% CI다. 파랑(token)은 직전 decoding step, 주황(numeric)은 직전 정답 가능 숫자를 비교 대상으로 쓴다. 0선 위의 점은 오류 순간 변화가 같은 경로의 이전 변화보다 컸다는 뜻이다. 두 색이 다르면 소수점·문법 token과 숫자를 오가는 효과가 섞였는지 확인해야 한다.

## 3. 숫자 위치와 직전 token 종류를 통제해도 남는가

Δₚₖ = αpath + β·1[first error] + 현재 role + 직전 token 종류 + digit ordinal + ε로 비교한다. 각 경로의 평균을 빼서 αpath를 제거하고, 경로별 전체 가중치를 1로 맞춘 weighted least squares를 사용한다. 경로 안에 safe와 error 변화가 모두 있어야 한다. 95% CI는 query-cluster sandwich SE와 t 분포로 계산한다. 이전 token이 endpoint 밖이면 outside_endpoint로 묶고, 나머지는 숫자·소수점·공백·기타 서식으로 분류한다. β가 양수면 이 통제 아래 최초 오류 변화가 더 큰 연관성이 남았다는 뜻이다. 인과효과가 아니다.

| 모델 | dataset | 비교 기준 | query / 경로 / step | β [95% CI] nat |
|---|---|---|---|---|
| qwen25 | activitynet-timelens | 직전 decoding token | 67 / 89 / 226 | 0.074 [-0.459, 0.607] |
| qwen25 | activitynet-timelens | 직전 정답 가능 숫자 | 25 / 31 / 63 | 1.284 [0.895, 1.673] |
| qwen25 | charades-timelens | 직전 decoding token | 60 / 75 / 177 | 0.450 [-0.236, 1.136] |
| qwen25 | charades-timelens | 직전 정답 가능 숫자 | 9 / 9 / 18 | — [—, —] |
| qwen25 | qvhighlights-timelens | 직전 decoding token | 73 / 108 / 279 | 0.094 [-0.159, 0.347] |
| qwen25 | qvhighlights-timelens | 직전 정답 가능 숫자 | 33 / 39 / 84 | -0.087 [-1.037, 0.863] |
| qwen3 | activitynet-timelens | 직전 decoding token | 62 / 85 / 203 | 0.403 [-0.081, 0.888] |
| qwen3 | activitynet-timelens | 직전 정답 가능 숫자 | 16 / 20 / 41 | 0.303 [-0.143, 0.748] |
| qwen3 | charades-timelens | 직전 decoding token | 56 / 73 / 174 | -0.142 [-0.846, 0.561] |
| qwen3 | charades-timelens | 직전 정답 가능 숫자 | 9 / 9 / 18 | — [—, —] |
| qwen3 | qvhighlights-timelens | 직전 decoding token | 68 / 87 / 244 | 0.280 [-0.022, 0.582] |
| qwen3 | qvhighlights-timelens | 직전 정답 가능 숫자 | 33 / 38 / 85 | 0.460 [-0.193, 1.113] |

**표 해설.** 소수점은 보통 후보가 좁아 entropy가 낮으므로 소수점→숫자라는 이유만으로 Δtoken이 커질 수 있다. 이 표는 현재 숫자/format role과 직전 token 종류, 숫자 순서를 통제한 민감도 분석이다. 단순 paired 결과와 방향이 다르면 “entropy 상승=오류”라는 일반 규칙보다 전이 유형·자릿수의 영향에 주의해야 한다. 범주 통제로 모든 혼란 요인이 제거되는 것은 아니다. β가 —인 행은 통제변수와 오류 위치가 겹쳐 오류 계수를 별도로 식별할 수 없는 경우다. 효과가 0이라는 뜻이 아니다.

## 4. 상승하면 실제 오류인가: 정답 유지 step도 분모에 넣기

아래 지표는 사후에 오류인 경로만 고르지 않고 모든 eligible risk step에서 계산한다. 민감도=P(Δ>0 | 최초 오류), 오경보율=P(Δ>0 | safe), precision=P(최초 오류 | Δ>0), 비상승 오류율=P(최초 오류 | Δ≤0)이다. 이 표의 비율은 step 기준이며 긴 경로는 더 많은 step을 기여한다. 배포 성능 검증이나 새로운 threshold 학습은 아니다.

### 직전 decoding token 기준

| 모델 | dataset | 최초 오류 / safe step | 오류 상승 / safe 상승 | 민감도 | 오경보율 | precision | 비상승 오류율 |
|---|---|---|---|---|---|---|---|
| qwen25 | activitynet-timelens | 176 / 249 | 170 / 180 | 96.6% | 72.3% | 48.6% | 8.0% |
| qwen25 | charades-timelens | 163 / 268 | 161 / 172 | 98.8% | 64.2% | 48.3% | 2.0% |
| qwen25 | qvhighlights-timelens | 169 / 319 | 168 / 233 | 99.4% | 73.0% | 41.9% | 1.1% |
| qwen3 | activitynet-timelens | 158 / 280 | 153 / 188 | 96.8% | 67.1% | 44.9% | 5.2% |
| qwen3 | charades-timelens | 146 / 342 | 144 / 213 | 98.6% | 62.3% | 40.3% | 1.5% |
| qwen3 | qvhighlights-timelens | 134 / 484 | 127 / 314 | 94.8% | 64.9% | 28.8% | 4.0% |

**표 해설.** 민감도가 높아도 오경보율이 비슷하게 높으면 오류를 잘 구분하는 신호가 아니다. precision은 오류의 기본 비율과 함께 달라진다. 숫자 기준은 현재 숫자+이전 safe 숫자가 있는 step만 포함하고 token 기준은 format과 첫 숫자도 포함하므로 두 행 집합의 분모가 다르다.

| 모델 | dataset | 오류 상승률 query 평균 [95% CI] | safe 상승률 query 평균 [95% CI] |
|---|---|---|---|
| qwen25 | activitynet-timelens | 0.969 [0.944, 0.990] | 0.844 [0.804, 0.886] |
| qwen25 | charades-timelens | 0.985 [0.960, 1.000] | 0.751 [0.704, 0.797] |
| qwen25 | qvhighlights-timelens | 0.995 [0.985, 1.000] | 0.829 [0.791, 0.866] |
| qwen3 | activitynet-timelens | 0.967 [0.935, 0.995] | 0.813 [0.763, 0.859] |
| qwen3 | charades-timelens | 0.984 [0.957, 1.000] | 0.721 [0.681, 0.764] |
| qwen3 | qvhighlights-timelens | 0.949 [0.904, 0.983] | 0.754 [0.719, 0.793] |

**표 해설.** 비율을 먼저 endpoint 경로 안에서 평균하고 두 endpoint가 있으면 query 안에서 다시 평균해 긴 경로의 영향력을 줄였다. 수치는 0–1 비율이다. 오류 열과 safe 열은 자료가 있는 query 집합이 다를 수 있으므로 둘의 차이가 동일 경로 paired 효과는 아니다. 동일 경로 비교는 2절을 본다.

### 직전 정답 가능 숫자 기준

| 모델 | dataset | 최초 오류 / safe step | 오류 상승 / safe 상승 | 민감도 | 오경보율 | precision | 비상승 오류율 |
|---|---|---|---|---|---|---|---|
| qwen25 | activitynet-timelens | 89 / 72 | 80 / 60 | 89.9% | 83.3% | 57.1% | 42.9% |
| qwen25 | charades-timelens | 75 / 64 | 70 / 49 | 93.3% | 76.6% | 58.8% | 25.0% |
| qwen25 | qvhighlights-timelens | 108 / 100 | 106 / 83 | 98.1% | 83.0% | 56.1% | 10.5% |
| qwen3 | activitynet-timelens | 85 / 81 | 78 / 61 | 91.8% | 75.3% | 56.1% | 25.9% |
| qwen3 | charades-timelens | 71 / 88 | 66 / 74 | 93.0% | 84.1% | 47.1% | 26.3% |
| qwen3 | qvhighlights-timelens | 86 / 189 | 79 / 153 | 91.9% | 81.0% | 34.1% | 16.3% |

**표 해설.** 민감도가 높아도 오경보율이 비슷하게 높으면 오류를 잘 구분하는 신호가 아니다. precision은 오류의 기본 비율과 함께 달라진다. 숫자 기준은 현재 숫자+이전 safe 숫자가 있는 step만 포함하고 token 기준은 format과 첫 숫자도 포함하므로 두 행 집합의 분모가 다르다.

| 모델 | dataset | 오류 상승률 query 평균 [95% CI] | safe 상승률 query 평균 [95% CI] |
|---|---|---|---|
| qwen25 | activitynet-timelens | 0.910 [0.843, 0.970] | 0.829 [0.722, 0.925] |
| qwen25 | charades-timelens | 0.917 [0.833, 0.983] | 0.756 [0.643, 0.863] |
| qwen25 | qvhighlights-timelens | 0.979 [0.945, 1.000] | 0.774 [0.675, 0.874] |
| qwen3 | activitynet-timelens | 0.911 [0.847, 0.968] | 0.695 [0.561, 0.805] |
| qwen3 | charades-timelens | 0.917 [0.833, 0.981] | 0.772 [0.670, 0.875] |
| qwen3 | qvhighlights-timelens | 0.918 [0.851, 0.970] | 0.799 [0.723, 0.864] |

**표 해설.** 비율을 먼저 endpoint 경로 안에서 평균하고 두 endpoint가 있으면 query 안에서 다시 평균해 긴 경로의 영향력을 줄였다. 수치는 0–1 비율이다. 오류 열과 safe 열은 자료가 있는 query 집합이 다를 수 있으므로 둘의 차이가 동일 경로 paired 효과는 아니다. 동일 경로 비교는 2절을 본다.

### 현재 step의 정수부·소수부·서식을 나누면

| 모델 | dataset | 기준 | 현재 위치 | 오류 / safe | 민감도 | 오경보율 | precision |
|---|---|---|---|---|---|---|---|
| qwen25 | activitynet-timelens | token | integer | 160 / 161 | 96.2% | 96.9% | 49.7% |
| qwen25 | activitynet-timelens | token | fractional | 16 / 24 | 100.0% | 100.0% | 40.0% |
| qwen25 | activitynet-timelens | token | format | 0 / 64 | — | 0.0% | — |
| qwen25 | activitynet-timelens | numeric | integer | 73 / 48 | 91.8% | 89.6% | 60.9% |
| qwen25 | activitynet-timelens | numeric | fractional | 16 / 24 | 81.2% | 70.8% | 43.3% |
| qwen25 | charades-timelens | token | integer | 145 / 139 | 98.6% | 97.1% | 51.4% |
| qwen25 | charades-timelens | token | fractional | 18 / 37 | 100.0% | 100.0% | 32.7% |
| qwen25 | charades-timelens | token | format | 0 / 92 | — | 0.0% | — |
| qwen25 | charades-timelens | numeric | integer | 57 / 27 | 96.5% | 85.2% | 70.5% |
| qwen25 | charades-timelens | numeric | fractional | 18 / 37 | 83.3% | 70.3% | 36.6% |
| qwen25 | qvhighlights-timelens | token | integer | 151 / 208 | 99.3% | 97.1% | 42.6% |
| qwen25 | qvhighlights-timelens | token | fractional | 18 / 31 | 100.0% | 100.0% | 36.7% |
| qwen25 | qvhighlights-timelens | token | format | 0 / 80 | — | 0.0% | — |
| qwen25 | qvhighlights-timelens | numeric | integer | 90 / 69 | 98.9% | 91.3% | 58.6% |
| qwen25 | qvhighlights-timelens | numeric | fractional | 18 / 31 | 94.4% | 64.5% | 45.9% |
| qwen3 | activitynet-timelens | token | integer | 146 / 166 | 96.6% | 92.8% | 47.8% |
| qwen3 | activitynet-timelens | token | fractional | 12 / 34 | 100.0% | 100.0% | 26.1% |
| qwen3 | activitynet-timelens | token | format | 0 / 80 | — | 0.0% | — |
| qwen3 | activitynet-timelens | numeric | integer | 73 / 47 | 93.2% | 74.5% | 66.0% |
| qwen3 | activitynet-timelens | numeric | fractional | 12 / 34 | 83.3% | 76.5% | 27.8% |
| qwen3 | charades-timelens | token | integer | 125 / 161 | 99.2% | 98.8% | 43.8% |
| qwen3 | charades-timelens | token | fractional | 19 / 54 | 100.0% | 100.0% | 26.0% |
| qwen3 | charades-timelens | token | format | 2 / 127 | 50.0% | 0.0% | 100.0% |
| qwen3 | charades-timelens | numeric | integer | 52 / 34 | 98.1% | 94.1% | 61.4% |
| qwen3 | charades-timelens | numeric | fractional | 19 / 54 | 78.9% | 77.8% | 26.3% |
| qwen3 | qvhighlights-timelens | token | integer | 111 / 276 | 94.6% | 91.3% | 29.4% |
| qwen3 | qvhighlights-timelens | token | fractional | 22 / 62 | 100.0% | 100.0% | 26.2% |
| qwen3 | qvhighlights-timelens | token | format | 1 / 146 | 0.0% | 0.0% | — |
| qwen3 | qvhighlights-timelens | numeric | integer | 64 / 127 | 90.6% | 81.1% | 36.0% |
| qwen3 | qvhighlights-timelens | numeric | fractional | 22 / 62 | 95.5% | 80.6% | 29.6% |

**표 해설.** 여기의 위치는 현재 risk step의 role이며, viewer의 “미래 최초 오류가 어디인가” 필터와 다르다. tenths가 소수부 숫자, decimal_point와 경계 token 등은 format이다. 같은 위치의 정답 유지와 오류를 비교하므로 소수점 뒤 숫자가 원래 상승하기 쉬운 현상을 확인할 수 있다. 오류 수가 적은 format 행의 극단 비율은 표본 수와 함께 읽는다.

## 5. Viewer 특수 필터별 최초 오류의 상승·감소

이 절은 viewer와 같은 사후 선택 조건의 **기술통계**다. 예를 들어 “직전 token 대비 상승”을 고르면 그 상승 비율은 정의상 100%다. “최초 오류 위치”, “낮은 entropy 오류”, “유지 직후 다음 숫자 오류” 등은 정답/미래 결과를 알아야 정할 수 있어 online trigger나 오류 예측의 근거로 쓸 수 없다. 필터는 서로 겹치며 행을 합산하지 않는다. 모델 안에서 dataset을 합친 탐색용 집계이므로 dataset 분포에 민감하다.

### qwen25

| 조건 | query / endpoint 경로 | 최초 오류 수 | token 상승/감소/동일 (eligible) | numeric 상승/감소/동일 (eligible) | 오류 Δtoken 평균 nat |
|---|---|---|---|---|---|
| 전체 경로 | 300 / 600 | 508 | 499/9/0 (508) | 256/16/0 (272) | 1.108 |
| 오류 전후 숫자 비교 가능 | 200 / 272 | 272 | 263/9/0 (272) | 256/16/0 (272) | 1.137 |
| 정수부 최초 오류 | 287 / 456 | 456 | 447/9/0 (456) | 211/9/0 (220) | 0.999 |
| 소수부 최초 오류 | 51 / 52 | 52 | 52/0/0 (52) | 45/7/0 (52) | 2.062 |
| 소수점·서식 최초 오류 | 0 / 0 | 0 | 0/0/0 (0) | 0/0/0 (0) | — |
| 오류 없음 | 88 / 92 | 0 | 0/0/0 (0) | 0/0/0 (0) | — |
| 직전 숫자 대비 상승 | 189 / 256 | 256 | 256/0/0 (256) | 256/0/0 (256) | 1.190 |
| 직전 숫자 대비 감소 | 14 / 16 | 16 | 7/9/0 (16) | 0/16/0 (16) | 0.293 |
| 직전 token 대비 상승 | 295 / 499 | 499 | 499/0/0 (499) | 256/7/0 (263) | 1.134 |
| 직전 token 대비 감소 | 9 / 9 | 9 | 0/9/0 (9) | 0/9/0 (9) | -0.360 |
| 낮은 entropy 오류 (H≤0.5) | 91 / 91 | 91 | 87/4/0 (91) | 18/5/0 (23) | 0.148 |
| 높은 entropy 유지 (H≥1.5) | 116 / 123 | 66 | 62/4/0 (66) | 56/10/0 (66) | 1.341 |
| 유지 직후 다음 숫자 오류 | 200 / 272 | 272 | 263/9/0 (272) | 256/16/0 (272) | 1.137 |
| 두 모델 판정 불일치 | 120 / 146 | 102 | 101/1/0 (102) | 64/2/0 (66) | 1.213 |

**표 해설.** eligible은 해당 비교 대상이 존재하는 최초 오류 경로 수다. format 오류는 token 기준에 들어가지만 numeric 기준에는 들어가지 않는다. high-safe는 오류 여부와 무관하게 H≥1.5인 safe numeric step이 하나 이상인 경로, next-failure는 safe numeric step 다음 숫자에서 최초 오류가 난 경로다. 모델 불일치는 두 모델 모두 지원 표기인 경우의 선택 endpoint 최종 반올림 정답 판정 불일치다. 이 평균은 endpoint 경로 가중 평균이며 검증된 오류 탐지 임계값은 아니다.

### qwen3

| 조건 | query / endpoint 경로 | 최초 오류 수 | token 상승/감소/동일 (eligible) | numeric 상승/감소/동일 (eligible) | 오류 Δtoken 평균 nat |
|---|---|---|---|---|---|
| 전체 경로 | 294 / 588 | 438 | 424/14/0 (438) | 223/19/0 (242) | 1.004 |
| 오류 전후 숫자 비교 가능 | 183 / 242 | 242 | 230/12/0 (242) | 223/19/0 (242) | 1.138 |
| 정수부 최초 오류 | 256 / 382 | 382 | 370/12/0 (382) | 177/12/0 (189) | 0.902 |
| 소수부 최초 오류 | 49 / 53 | 53 | 53/0/0 (53) | 46/7/0 (53) | 1.857 |
| 소수점·서식 최초 오류 | 3 / 3 | 3 | 1/2/0 (3) | 0/0/0 (0) | -1.063 |
| 오류 없음 | 131 / 150 | 0 | 0/0/0 (0) | 0/0/0 (0) | — |
| 직전 숫자 대비 상승 | 171 / 223 | 223 | 223/0/0 (223) | 223/0/0 (223) | 1.237 |
| 직전 숫자 대비 감소 | 19 / 19 | 19 | 7/12/0 (19) | 0/19/0 (19) | -0.031 |
| 직전 token 대비 상승 | 271 / 424 | 424 | 424/0/0 (424) | 223/7/0 (230) | 1.053 |
| 직전 token 대비 감소 | 14 / 14 | 14 | 0/14/0 (14) | 0/12/0 (12) | -0.477 |
| 낮은 entropy 오류 (H≤0.5) | 79 / 85 | 85 | 78/7/0 (85) | 8/11/0 (19) | 0.104 |
| 높은 entropy 유지 (H≥1.5) | 111 / 123 | 21 | 18/3/0 (21) | 17/2/0 (19) | 1.613 |
| 유지 직후 다음 숫자 오류 | 183 / 242 | 242 | 230/12/0 (242) | 223/19/0 (242) | 1.138 |
| 두 모델 판정 불일치 | 120 / 146 | 44 | 44/0/0 (44) | 29/0/0 (29) | 1.145 |

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

생성기: `zero-shot/_lib/entropy_error_appendix.py`. bootstrap seed=42, repeats=1000. 모델 재추론 없이 기존 자료만 읽는다. 원본 step entropy와 top-30 자료를 5,905개 step에서 대조하고 viewer의 최초 오류 Δ와도 대조했다.

[집계 JSON](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_error/metrics/summary.json) · [출처와 SHA-256](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_error/metadata/sources.json) · [검증 기록](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_error/validation/checks.json)

- digit_steps: [per_step.csv](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/metrics/per_step.csv)
- all_steps: [per_step.csv.gz](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/top30_renormalized/metrics/per_step.csv.gz)
- paths: [per_path.json](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/within_path/metrics/per_path.json)
- catalog: [viewer_catalog.json](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/metrics/viewer_catalog.json)
- manifest: [manifest.js](../../../../../../artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/web/manifest.js)
