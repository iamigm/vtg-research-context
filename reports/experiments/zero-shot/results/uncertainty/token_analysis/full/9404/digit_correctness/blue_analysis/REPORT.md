<!-- report-appendix:APPENDIX.md parts=8 -->
<!-- report-part:0:begin -->
# 정답 가능성을 유지한 숫자 단계: 세 축 분석 · 전체 full9404 (모델별 9,404질문)

전체 벤치마크 9,404개 × 두 모델의 지원 표기 응답 중 파란 숫자 **42033개**를 분석했다. 첫 숫자를 포함하며 기존 raw 전체 어휘 entropy를 그대로 사용한다. 기존 pilot 추론은 재사용하고 나머지는 같은 조건으로 새로 추론했다.

[영상과 후보별 확률 보기](report.html)

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## A와 B의 정의

후보 토큰을 현재 prefix에 붙였을 때, 기존과 동일한 비음수 정수 또는 소수 한 자리 문법에서 Python round(float(value)) 결과가 GT인 완성이 존재하면 정답 가능 후보다. 정수부 선행 0은 허용하지 않는다. 확실한 숫자·소수점·종료 후보만 판정하고 나머지 형식 또는 텍스트 토큰은 미확인으로 둔다. 이 가능성은 문법상 존재 여부이며 모델이 실제로 그 완성을 생성할 확률은 아니다.

A는 top-30에서 정답 가능이라고 판정한 후보들의 원확률 합이다. B는 정답 불가능이라고 판정한 후보들의 원확률 합이다. top-30 밖과 판정하지 못한 후보는 둘 다 제외한다. 재정규화하지 않으며 최종 응답 정답률을 의미하지 않는다.

예: 정답 가능 후보의 확률 합이 0.45, 정답 불가능 후보의 확률 합이 0.40이면 A=0.45, B=0.40으로 표시한다. 나머지 0.15는 미확인 확률이며 A나 B에 더하지 않는다. 따라서 A+B는 1보다 작을 수 있다.

표의 평균 A와 평균 B는 각 샘플 내부에서 해당 단계들의 값을 평균한 뒤 샘플 간 평균했다. 확률 그룹은 A≥2/3이면 정답 가능 후보 우세, B≥2/3이면 불가능 후보 우세, A≥1/3이고 B>1/3이면 양쪽 후보가 경쟁하는 것으로 분류한다. 나머지는 미확인 확률 때문에 그룹을 확정하지 않는다. 1/3과 2/3은 탐색적으로 정한 기준이다.

높은 entropy는 모델 × 데이터셋 × 시작 또는 종료 × 실제 출력 자릿수 내부 75백분위 이상이다. 동률 때문에 정확히 25%가 아닐 수 있다. 숫자 단계 10개 미만인 비교 집단은 별도 표시하며, 첫 숫자와 후속 숫자와 GT 자릿수와 생성 자릿수도 교차 표로 제공한다.

후속 결과는 동일한 시작 또는 종료 시각의 끝까지 유지 / 다음 숫자 최초 실패 / 더 나중 숫자 최초 실패 / 소수점·종료 최초 실패로 구분한다. 다른 시각의 실패와 혼합하지 않는다.

<!-- report-part:1:end -->
<!-- report-part:6:begin -->
## 관측 결과와 해석

- qwen25: 높은 entropy의 파란 단계 4743개. 정답 가능 우세 668개, 오류 후보 우세 1038개, 정답 가능 확률이 중간인 단계 3037개, 그룹 확정 불가 0개.
  - 시작 시각, 나머지: 서로 다른 샘플 5220개, 샘플 평균 후속 실패율 55.9%.
  - 시작 시각, 상위 entropy: 서로 다른 샘플 2105개, 샘플 평균 후속 실패율 61.0%.
  - 종료 시각, 나머지: 서로 다른 샘플 4055개, 샘플 평균 후속 실패율 81.7%.
  - 종료 시각, 상위 entropy: 서로 다른 샘플 1565개, 샘플 평균 후속 실패율 75.5%.
- qwen3: 높은 entropy의 파란 단계 5782개. 정답 가능 우세 1999개, 오류 후보 우세 207개, 정답 가능 확률이 중간인 단계 3574개, 그룹 확정 불가 2개.
  - 시작 시각, 나머지: 서로 다른 샘플 5646개, 샘플 평균 후속 실패율 43.0%.
  - 시작 시각, 상위 entropy: 서로 다른 샘플 2381개, 샘플 평균 후속 실패율 47.1%.
  - 종료 시각, 나머지: 서로 다른 샘플 4866개, 샘플 평균 후속 실패율 72.6%.
  - 종료 시각, 상위 entropy: 서로 다른 샘플 2021개, 샘플 평균 후속 실패율 61.3%.

정답 가능 질량이 높은 고entropy는 허용되는 후보 사이의 분산을, 낮은 경우는 오류 후보와의 경쟁을 구분하는 데 유용하다. 미확인 확률이 남아 있으면 A와 B만으로 우세를 단정하지 않는다. 단계별 비교는 첫 숫자와 후속 자리, 데이터셋, 자릿수 구성에 영향을 받으므로 통합 실패율을 예측 성능이나 인과 효과로 해석하지 않는다. 특히 소수 마지막 자리에는 이후 숫자가 없어 이후 실패 기회 자체가 적다. 같은 위치의 교차 표와 함께 보아야 한다.

observed_safe_entropy는 top-30 내 확인된 정답 가능 후보만 조건부 정규화한 entropy이며 전체 정답 후보 entropy가 아니다. 원래 전체 어휘 H와 혼동하지 않는다. 원분포의 정답 가능 후보와 불가능 후보 entropy 기여량도 CSV에 저장했다.

![정답 가능 질량과 불확실성](../../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/figures/safe_mass_vs_entropy.png)

<!-- report-part:6:end -->
<!-- report-part:7:begin -->
## 원자료

[단계별 수치](../../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/per_blue_step.csv) · [후보 축](../../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/candidate_axis.csv) · [후속 결과 축](../../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/future_axis.csv) · [위치 축](../../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/position_axis.csv) · [세 축 교차](../../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/three_axis_cross.csv) · [후보×후속](../../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/candidate_future.csv) · [자릿수 통제](../../../../../../../artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/length_control.csv)
<!-- report-part:7:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
