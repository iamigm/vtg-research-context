# 최초 오류의 엔트로피와 GT 거리

전체 벤치마크의 지원 응답 18,591개에서 오답인 시작·종료 시각 29,124개를 분석했다. 끝까지 정답 가능성이 유지된 시각은 최초 오류가 없으므로 제외한다. 지원하지 않는 출력 형식 217개 응답은 기존 분석과 동일하게 제외한다.

- 세로축: 정답으로 완성할 가능성을 처음 잃은 생성 단계 직전의 **raw 전체 어휘 엔트로피**(nat). 기존 반올림 가능성 판정을 그대로 사용한다.
- 가로축: **최종 출력 시각과 GT 시각의 절대 차이**(초). 시작과 종료는 독립 비교하며 반대쪽 시각은 계산에 관여하지 않는다. 반올림 후 오차도 CSV에 별도로 저장했다.
- 미완성 숫자 prefix 자체와 GT를 뺀 거리가 아니다. 최초 오류 뒤에 생성된 숫자도 최종 오차에 영향을 준다.
- 점은 시각별 관측값. 빨간 선은 엔트로피 5분위별 오차 중앙값, 음영은 해당 오차의 25–75백분위다. 음영은 신뢰구간이 아니다.
- 숫자 오류는 파랑, 소수점·종료 등 서식 오류는 갈색. 가로축은 1초까지 선형, 그 위는 로그이며 패널별 범위가 다르다.

![시작 시각](../../../../artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/figures/start_entropy_vs_distance.png)

![종료 시각](../../../../artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/figures/end_entropy_vs_distance.png)

| 모델 | 벤치마크 | 시각 | 관측 수 | 오차 중앙값(초) | Spearman ρ | 95% CI |
|---|---|---|---:|---:|---:|---|
| qwen25 | charades-timelens | start | 2553 | 2.40 | -0.259 | [-0.300, -0.216] |
| qwen25 | activitynet-timelens | start | 3363 | 4.00 | -0.360 | [-0.389, -0.329] |
| qwen25 | qvhighlights-timelens | start | 1078 | 2.40 | -0.519 | [-0.565, -0.465] |
| qwen3 | charades-timelens | start | 2135 | 1.50 | -0.481 | [-0.521, -0.442] |
| qwen3 | activitynet-timelens | start | 2814 | 4.00 | -0.323 | [-0.359, -0.287] |
| qwen3 | qvhighlights-timelens | start | 825 | 3.00 | -0.457 | [-0.519, -0.389] |
| qwen25 | charades-timelens | end | 3002 | 2.80 | -0.307 | [-0.345, -0.268] |
| qwen25 | activitynet-timelens | end | 4140 | 5.80 | -0.481 | [-0.506, -0.451] |
| qwen25 | qvhighlights-timelens | end | 1436 | 9.70 | -0.419 | [-0.462, -0.373] |
| qwen3 | charades-timelens | end | 2934 | 2.00 | -0.182 | [-0.217, -0.145] |
| qwen3 | activitynet-timelens | end | 3686 | 3.20 | -0.278 | [-0.309, -0.247] |
| qwen3 | qvhighlights-timelens | end | 1158 | 2.00 | -0.333 | [-0.387, -0.275] |

## 해석

12개 그룹 모두 음의 상관을 보였다(ρ 약 −0.18~−0.52). 오답인 시각들 안에서는 최초 오류 단계의 엔트로피가 낮은 사례가 더 큰 최종 시간 오차를 보이는 경향이다. 오류 자릿수를 섞은 효과인지 후속 검토가 필요하다.

ρ가 양수이면 최초 오류 때 불확실성이 높은 사례에서 최종 시간 오차도 큰 경향, 음수이면 반대 경향을 뜻한다. 모델·벤치마크·시작·종료를 섞지 않고 해석한다. 신뢰구간은 각 패널의 시각 관측을 500회 복원추출한 percentile bootstrap이다.

오답 사례만 조건부로 분석하므로 오류 발생 여부를 예측하는 결과는 아니다. 최초 오류의 자릿수와 영상 길이, GT 시각 등의 교란이 있을 수 있다. 자릿수별 상관은 ../../../../artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metrics/role_statistics.csv에 함께 저장했으며, 본 그림의 관계를 인과관계로 해석하지 않는다.

[관측값 CSV](../../../../artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metrics/per_first_error.csv) · [그룹 통계](../../../../artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metrics/group_statistics.csv) · [최초 오류 자릿수별 통계](../../../../artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metrics/role_statistics.csv)
