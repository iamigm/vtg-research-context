# Qwen3 input order에 따른 예측 일관성 · 전체 full9404 · pilot300 (질문 수)

동일한 영상–쿼리 9,404쌍에서 video → query와 query → video를 비교했다. 평가용 구간 완전 일치율은 17.79%, 두 예측의 평균 상호 IoU는 0.5248다. 정답 IoU 0.5 기준 성공 여부가 바뀐 비율은 28.31%다. 이 실행의 예측은 input order에 민감하다.

기존 Qwen3-VL-8B-Instruct 생성 결과만 CPU로 분석했다. 표본 단위는 영상–쿼리 쌍이며 전체 값은 샘플 수로 가중한 집계다. 두 순서의 샘플 ID, 영상, 쿼리, 정답 구간, 영상 길이가 모두 일치한다.

## 구간 일관성

평가용 timestamps는 원문 시간에 Python round를 적용한 정수 초다. 원문 구간은 기존 TimeLens extract_time으로 재파싱했다. 구간 일치와 응답 텍스트 일치는 별도 지표다. 상호 IoU는 두 예측 구간의 교집합 길이 / 합집합 길이이며, 양쪽 모두 파싱에 성공하고 양의 길이를 가진 쌍만 포함한다. 파싱 실패의 평가용 대체 구간은 상호 IoU에서 제외한다. 길이가 0 이하인 쌍은 제외하고 개수를 별도 표시한다. 영상 범위로 클리핑하지 않는다.

| 데이터셋 | N | 정수 구간 일치 % | 원문 구간 일치 % | 텍스트 일치 % | 상호 IoU 평균 | 상호 IoU 중앙값 | 상호 IoU ≥ .5 % | 유효 쌍 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| charades-timelens | 3363 | 17.69 | 8.30 | 8.24 | 0.6450 | 0.7143 | 73.88 | 3223 |
| activitynet-timelens | 4500 | 16.87 | 10.87 | 9.64 | 0.4375 | 0.3636 | 46.33 | 4125 |
| qvhighlights-timelens | 1541 | 20.70 | 12.26 | 10.58 | 0.5041 | 0.5351 | 51.82 | 1316 |
| all | 9404 | 17.79 | 10.18 | 9.29 | 0.5248 | 0.6000 | 57.41 | 8664 |

전체 생성 토큰 ID 완전 일치율: 9.29%. 원문 소수 구간의 상호 IoU 평균: 0.5037 (유효 9026쌍). 평가용 유효 구간의 무겹침: 1907 / 8664쌍.

## 시작 및 종료 시각 차이

양쪽 파싱에 성공한 9,336쌍의 정수 초 평가 구간 기준이다. 실패 시 대체 구간은 시간 차이 집계에서 제외한다. 허용 오차는 시작과 종료가 모두 조건을 만족해야 한다.

| 데이터셋 | 시작 차이 평균 / 중앙값 / P90 초 | 종료 차이 평균 / 중앙값 / P90 초 | 둘 다 1초 이내 % | 둘 다 2초 이내 % | 둘 다 5초 이내 % |
|---|---:|---:|---:|---:|---:|
| charades-timelens | 2.08 / 0.00 / 6.00 | 3.32 / 1.00 / 11.00 | 55.99 | 66.70 | 78.14 |
| activitynet-timelens | 37.85 / 1.00 / 122.00 | 45.48 / 6.00 / 134.90 | 33.09 | 37.20 | 44.00 |
| qvhighlights-timelens | 37.11 / 2.00 / 117.00 | 42.99 / 9.00 / 129.00 | 36.09 | 37.94 | 41.42 |
| all | 24.84 / 1.00 / 101.00 | 29.88 / 1.00 / 114.00 | 41.83 | 47.94 | 55.88 |

## 정답 기준 성공 여부의 변화

성공은 정답과의 IoU가 임계값 이상인 경우다. 아래 비율의 분모는 해당 데이터셋 전체 표본이다.

| 데이터셋 | 임계값 | 둘 다 성공 % | video 먼저만 성공 % | query 먼저만 성공 % | 둘 다 실패 % |
|---|---:|---:|---:|---:|---:|
| charades-timelens | 0.3 | 56.14 | 13.05 | 7.05 | 23.76 |
| charades-timelens | 0.5 | 40.98 | 12.61 | 8.71 | 37.70 |
| charades-timelens | 0.7 | 19.60 | 8.03 | 9.16 | 63.22 |
| activitynet-timelens | 0.3 | 35.82 | 26.38 | 5.67 | 32.13 |
| activitynet-timelens | 0.5 | 27.22 | 24.04 | 6.53 | 42.20 |
| activitynet-timelens | 0.7 | 17.31 | 17.16 | 6.13 | 59.40 |
| qvhighlights-timelens | 0.3 | 40.17 | 33.68 | 4.87 | 21.28 |
| qvhighlights-timelens | 0.5 | 33.35 | 31.02 | 5.91 | 29.72 |
| qvhighlights-timelens | 0.7 | 25.57 | 23.82 | 4.74 | 45.88 |
| all | 0.3 | 43.80 | 22.81 | 6.03 | 27.36 |
| all | 0.5 | 33.15 | 21.10 | 7.21 | 38.55 |
| all | 0.7 | 19.48 | 14.98 | 6.99 | 58.55 |

## 정답 IoU 변화

Δ는 query 먼저 − video 먼저다. 평균은 0–1 스케일이다.

| 데이터셋 | video 먼저 mIoU | query 먼저 mIoU | 평균 Δ | query 먼저 개선 % | 악화 % | 동일 % |
|---|---:|---:|---:|---:|---:|---:|
| charades-timelens | 0.4835 | 0.4605 | -0.0230 | 35.36 | 38.21 | 26.43 |
| activitynet-timelens | 0.4679 | 0.3224 | -0.1455 | 20.98 | 43.58 | 35.44 |
| qvhighlights-timelens | 0.5942 | 0.3693 | -0.2249 | 17.65 | 50.75 | 31.60 |
| all | 0.4942 | 0.3795 | -0.1147 | 25.57 | 42.83 | 31.59 |

## 파싱과 구간 품질

범위 초과는 저장된 annotation duration 기준이며, 반올림 또는 영상 metadata 차이도 포함할 수 있다. 자동으로 생성 오류라고 판정하지 않는다.

| 데이터셋 | 파싱 실패 V/Q | 잘림 V/Q | 길이 0 이하 V/Q | 영상 범위 초과 V/Q |
|---|---:|---:|---:|---:|
| charades-timelens | 0 / 0 | 0 / 0 | 0 / 140 | 377 / 171 |
| activitynet-timelens | 0 / 48 | 0 / 0 | 3 / 325 | 395 / 253 |
| qvhighlights-timelens | 0 / 20 | 0 / 0 | 0 / 205 | 4 / 21 |
| all | 0 / 68 | 0 / 0 | 3 / 670 | 776 / 445 |

## 해석과 재현

두 예측이 일치해도 둘 다 틀릴 수 있으므로 상호 IoU와 정답 기준 성공 여부를 함께 봐야 한다. 데이터셋마다 영상 길이가 달라 초 단위 차이를 직접 비교할 때 주의가 필요하다. 샘플별 CSV에는 영상 길이로 정규화한 최대 경계 차이도 포함했다. 이 분석은 저장된 한 번의 deterministic 생성 비교이며 반복 실행 변동성이나 attention 변화의 인과를 측정하지 않는다.

[전체 수치](../../../../../artifacts/runs/baseline/input_order/overview/consistency/metrics/summary.json) · [샘플별 비교](../../../../../artifacts/runs/baseline/input_order/overview/consistency/metrics/paired_samples.csv) · [검증](../../../../../artifacts/runs/baseline/input_order/overview/consistency/validation/validation.json) · [소스 SHA-256](../../../../../artifacts/runs/baseline/input_order/overview/consistency/metadata/source_manifest.json) · [재현 코드](../../../../../artifacts/runs/baseline/input_order/overview/consistency/code/analyze.py)

[정답 IoU 최대 악화 50개](../../../../../artifacts/runs/baseline/input_order/overview/consistency/metrics/largest_gt_degradation.csv) · [최대 개선 50개](../../../../../artifacts/runs/baseline/input_order/overview/consistency/metrics/largest_gt_improvement.csv) · [영상 길이 대비 최대 이동 50개](../../../../../artifacts/runs/baseline/input_order/overview/consistency/metrics/largest_normalized_shift.csv)

재현: `python experiments/zero-shot/results/input_order/consistency_v1/analyze.py`

## 인터랙티브 영상 비교

[영상 비교 뷰어](report.html)에서 데이터셋, 성공 여부, 무겹침, 파싱 실패 조건으로 샘플을 고르고 두 예측과 GT 구간을 재생할 수 있다. 원문 시간과 정수 초 표시를 전환할 수 있으며 기존 attention 그림도 연결한다.

실험 서버에서 `python experiments/zero-shot/results/input_order/consistency_v1/serve.py`로 실행한다. 접속 주소는 `http://localhost:8775/zero-shot/results/input_order/consistency_v1/report.html`이며 원격 접속 시 8775 포트 포워딩이 필요하다.

단계별 token distribution는 별도 video → query 실행에만 확인됐다. 이번 두 순서 비교 원본에는 생성 토큰 ID만 있고 후보 확률은 없다. 별도 전체 실행의 Qwen3 6,746건을 확인한 시점에 기존 video → query 응답과 토큰이 같은 것은 6,725건, 다른 것은 21건이었다. 저장 항목은 단계별 raw/processed top-30, 선택 토큰 확률·순위, 전체 어휘 entropy 등이며 전체 어휘 확률 벡터는 아니다. 실행이 진행 중이므로 수량은 계속 변한다. [확인 기록](../../../../../artifacts/runs/baseline/input_order/overview/consistency/validation/token_trace_audit.json)

## 뷰어 원본 대조 검증

9,404쌍의 영상 매핑·input order·구간을 원본과 대조했고, 18,808건의 정답 IoU를 독립 재계산해 일치를 확인했다. 전체 정답 성능은 파싱 실패 및 길이 0 구간을 제외하지 않는다. 상호 IoU의 유효 쌍 제한은 정답 성능 집계와 별개다.

기존 뷰어는 q→v 악화가 큰 순으로 정렬하여 첫 40개 모두 v→q IoU 1.0, q→v IoU 0.0이었다. 이 목록은 전체를 대표하지 않는다. 기본 정렬을 성능과 무관한 샘플 ID 순으로 변경하고, 전체 성능 표와 필터된 표본 mIoU 및 정렬 설명을 추가했다. [검증 결과](../../../../../artifacts/runs/baseline/input_order/overview/consistency/validation/viewer_source_audit.json)
