# Path mass constraints · pilot3

3개 데이터셋에서 고정 pilot300에 처음 등장하는 query를 하나씩 고른 진단이다. 일반 성능 향상을 결론내릴 표본이 아니다. 모델과 영상/query 입력은 기존 zero-shot과 동일한 Qwen3-VL-8B-Instruct·2 FPS이며 tensor hash를 대조했다. Sampling은 없다.

## 비교 조건과 종료 기준

두 조건 모두 매 step의 native top-p=0.99 집합과 native 확률≥1%의 교집합을 만든다. 정수부 숫자·소수점 기호 위치에서만 대안을 추적하며 일반 문구·소수 자릿수는 greedy로 한 경로만 이어간다. 소수 자릿수에서는 branch entropy도 측정하지 않는다. 최종 출력의 소수점은 보존한다. 재정규화하지 않고 정책상 제외한 확률 질량을 따로 저장한다.

- **mass_only**: 출력 문법을 제한하지 않고 확률 질량으로 탐색한다.
- **format_and_mass**: 원래 prompt의 `The event happens in <start> - <end> seconds` 형식, 양수/0의 초 단위 숫자·소수점 한 자리만 허용한다. 마지막 마침표는 선택이다. 이미 start≥영상길이이거나 완성된 clipped start≥end인 prefix를 종료한다. 낮은 확률 token을 강제로 추가하지 않는다.
- 두 조건 모두 native 확률 질량이 큰 frontier prefix부터 선택하고, 그 안에서는 greedy spine을 완성한 뒤 다음 frontier를 선택한다. 공유 video/query KV를 재사용한다.

V=완성된 유효 경로의 native 질량, U=미탐색 prefix의 native 질량 합+context limit 질량이다. `U/(V+U)≤ε`에서 ε=10%·5%·1% checkpoint를 기록한다. 한 번의 nested search에서 checkpoint를 얻으며 서로 독립 실행처럼 비용을 합산하지 않는다. 이 상한은 token 필터·숫자 분기 정책·해당 문법을 통과하는 최종 유효 분포에 대한 것이다. 원래 full model 분포 전체의99%를 보장하지 않는다.

**각 query·조건은 search wall 45초까지만 측정했다.** 시간 제한 전에 기준을 만족하지 못한 ε는 미도달이며, 완료된 ε 실험으로 계산하지 않는다. Model load·입력 준비·video prefill은45초 밖의 운영 비용이다. 미탐색 질량이 작아도 놓친 낮은 확률 후보에 높은 GT IoU가 있을 수 있다.

## 목표 질량까지의 비용

아래 평균은 실제 목표에 도달한 query만 포함한다. 도달 수와 함께 읽어야 한다. Search forward는 prefill 이후 실제 모델 호출 수이며 KV rollback에 필요한 replay도 포함한다.

| 조건 | ε | 도달 query /3 | 평균 search forwards | 평균 search wall s | 평균 고유 구간 수 |
|---|---:|---:|---:|---:|---:|
| mass_only | 10% | 3 /3 | 339.7 | 9.749 | 32.0 |
| mass_only | 5% | 3 /3 | 513.7 | 14.841 | 47.3 |
| mass_only | 1% | 2 /3 | 275.5 | 7.713 | 25.0 |
| format_and_mass | 10% | 3 /3 | 340.0 | 9.962 | 32.7 |
| format_and_mass | 5% | 3 /3 | 521.7 | 15.184 | 48.7 |
| format_and_mass | 1% | 2 /3 | 275.5 | 7.718 | 25.0 |

## ε 기준별 후보 품질

해당 ε에 실제 도달한 query만의 결과다. Δpp는 그와 같은 query들의 greedy 대비 차이이며 n이 다른 행의 평균을 직접 비교하지 않는다. Oracle은 GT로 사후 선택한 후보 상한이다.

| 조건 | ε | 도달 n | 평가 | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---|---:|---:|---|---:|---:|---:|---:|
| mass_only | 10% | 3 | native_mass | 0.8504 (-0.75) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.6667 (+0.00) |
| mass_only | 10% | 3 | candidate_oracle | 0.9994 (+14.15) | 1.0000 (+0.00) | 1.0000 (+0.00) | 1.0000 (+33.33) |
| mass_only | 5% | 3 | native_mass | 0.8504 (-0.75) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.6667 (+0.00) |
| mass_only | 5% | 3 | candidate_oracle | 0.9994 (+14.15) | 1.0000 (+0.00) | 1.0000 (+0.00) | 1.0000 (+33.33) |
| mass_only | 1% | 2 | native_mass | 0.7948 (-1.12) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.5000 (+0.00) |
| mass_only | 1% | 2 | candidate_oracle | 0.9991 (+19.31) | 1.0000 (+0.00) | 1.0000 (+0.00) | 1.0000 (+50.00) |
| format_and_mass | 10% | 3 | native_mass | 0.8504 (-0.75) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.6667 (+0.00) |
| format_and_mass | 10% | 3 | candidate_oracle | 0.9994 (+14.15) | 1.0000 (+0.00) | 1.0000 (+0.00) | 1.0000 (+33.33) |
| format_and_mass | 5% | 3 | native_mass | 0.8504 (-0.75) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.6667 (+0.00) |
| format_and_mass | 5% | 3 | candidate_oracle | 0.9994 (+14.15) | 1.0000 (+0.00) | 1.0000 (+0.00) | 1.0000 (+33.33) |
| format_and_mass | 1% | 2 | native_mass | 0.7948 (-1.12) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.5000 (+0.00) |
| format_and_mass | 1% | 2 | candidate_oracle | 0.9991 (+19.31) | 1.0000 (+0.00) | 1.0000 (+0.00) | 1.0000 (+50.00) |

## 시간 제한 시점 또는1% 목표 도달 시점의 실제 탐색량

U/(V+U)는 미탐색 유효 질량의 보수적 상한이다. 아래 interval entropy와 covariance는 탐색된 유효 구간에만 조건부인 통계이며 전체 분포의 정확한 값이 아니다. 서로 다른 sequence가 같은 clipped interval이면 질량을 합친다.

| Query | 조건 | 종료 사유 | Forwards | Search s | 구간 수 | 미탐색 상한 | V | 문법 제외 질량 | 정책 제외 질량 | 필터 제외 질량 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| charades-timelens_00027 | mass_only | mass_target_reached | 509 | 14.220 | 44 | 0.009 | 0.086 | 0.000 | 0.887 | 0.026 |
| charades-timelens_00027 | format_and_mass | mass_target_reached | 509 | 14.252 | 44 | 0.009 | 0.086 | 0.000 | 0.887 | 0.026 |
| activitynet-timelens_00054 | mass_only | mass_target_reached | 42 | 1.206 | 6 | 0.008 | 0.263 | 0.000 | 0.702 | 0.032 |
| activitynet-timelens_00054 | format_and_mass | mass_target_reached | 42 | 1.184 | 6 | 0.008 | 0.263 | 0.000 | 0.702 | 0.032 |
| qvhighlights-timelens_00014 | mass_only | diagnostic_time_limit | 1532 | 45.143 | 141 | 0.029 | 0.058 | 0.000 | 0.915 | 0.026 |
| qvhighlights-timelens_00014 | format_and_mass | diagnostic_time_limit | 1538 | 45.089 | 144 | 0.029 | 0.055 | 0.012 | 0.906 | 0.025 |

## 후보 구간의 GT 평가

IoU=예측/GT 교집합 길이÷합집합 길이,mIoU=query별 IoU 평균,R@τ=IoU≥τ 비율이다. 높을수록 좋다. 괄호 안 Δpp=100×(결과−같은 query의greedy). 최대 질량 후보는 native 경로 질량 합이 가장 큰 구간이며 verifier 선택 결과가 아니다. Oracle은 생성 후보 중 최고 GT IoU이며 실제 모델 성능이 아니다. GT는 탐색 이후 평가에만 사용한다.

| 조건 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---|---:|---:|---:|---:|---:|
| greedy | 3 | 0.8578 (+0.00) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.6667 (+0.00) |
| mass_only native_mass | 3 | 0.8504 (-0.75) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.6667 (+0.00) |
| mass_only candidate_oracle | 3 | 0.9994 (+14.15) | 1.0000 (+0.00) | 1.0000 (+0.00) | 1.0000 (+33.33) |
| format_and_mass native_mass | 3 | 0.8504 (-0.75) | 1.0000 (+0.00) | 1.0000 (+0.00) | 0.6667 (+0.00) |
| format_and_mass candidate_oracle | 3 | 0.9994 (+14.15) | 1.0000 (+0.00) | 1.0000 (+0.00) | 1.0000 (+33.33) |

## Query별 길이와 결과

n=1은 모두 작은 표본이다. 아래 개별 행은 dataset·영상 길이·GT 비율을 함께 표시하며 세 query를 benchmark 평균으로 해석하지 않는다.

| Query | Dataset | 영상 s | GT/영상 길이 | Greedy IoU | Mass-only 최고 질량 / oracle IoU | Format 최고 질량 / oracle IoU |
|---|---|---:|---:|---:|---:|---:|
| charades-timelens_00027 | charades-timelens | 35.30 | 0.1417 | 0.6250 | 0.6026 / 1.0000 | 0.6026 / 1.0000 |
| activitynet-timelens_00054 | activitynet-timelens | 54.00 | 0.9815 | 0.9870 | 0.9870 / 0.9981 | 0.9870 / 0.9981 |
| qvhighlights-timelens_00014 | qvhighlights-timelens | 150.02 | 0.1667 | 0.9615 | 0.9615 / 1.0000 | 0.9615 / 1.0000 |

## Greedy 대비 실측 연산 오버헤드

Native decoder inputs는 실제 모델에 넣은 token position 수다. Prefill token과 decode token은 attention 비용이 다르므로 같은 FLOPs로 해석하지 않는다. CUDA stream seconds는 GPU kernel active time과 같지 않다. 공유 prefill은 각 방법의 단독 비교 비용에 한 번씩 귀속하고 합산하지 않는다.

| Query | 조건 | search forwards / greedy decode forwards | search decoder inputs / greedy decode inputs | prefill 포함 inputs 배수 | CUDA stream s |
|---|---|---:|---:|---:|---:|
| charades-timelens_00027 | mass_only | 29.94 | 29.94 | 1.134 | 13.981 |
| charades-timelens_00027 | format_and_mass | 29.94 | 29.94 | 1.134 | 14.011 |
| activitynet-timelens_00054 | mass_only | 2.62 | 2.62 | 1.005 | 1.172 |
| activitynet-timelens_00054 | format_and_mass | 2.62 | 2.62 | 1.005 | 1.148 |
| qvhighlights-timelens_00014 | mass_only | 90.12 | 90.12 | 1.100 | 44.496 |
| qvhighlights-timelens_00014 | format_and_mass | 90.47 | 90.47 | 1.100 | 44.422 |

운영 실측: model load 3.939s; 입력 준비 1.824s; 공유 video/query prefill 1.878s; peak allocated 20.70GiB.

## 해석의 범위

문법 제약은 허용 분포 자체를 바꾸며 분포 밖 질량을 따로 보고한다. ε checkpoint는 해당 허용 분포의 탐색 충실도를 통제한다.45초 결과끼리 비교한 차이에는 문법과 탐색 순서의 영향이 함께 있을 수 있다. 제한 시간 내 기준에 도달하지 못했다는 사실도 결과다. Semantic embedding·독립 재시청·final answer 재생성은 이번 짧은 비교의 측정 범위가 아니다.

## 영상 길이 × GT 비율 교차 진단

각 칸은 n; greedy / mass_only 최대 질량 / format_and_mass 최대 질량 순서다. 조건별 진단 종료 시점의 값을 사용한다. n=1은 작은 표본이며 빈 칸은 —다. 모든 query의 개별 값은 위 표에 있다.

### mIoU

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
| [30, 60) | n=0; — | n=0; — | n=1 작은 표본; 0.625 / 0.603 / 0.603 | n=0; — | n=1 작은 표본; 0.987 / 0.987 / 0.987 |
| [60, 120) | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
| [120, 180) | n=0; — | n=0; — | n=1 작은 표본; 0.962 / 0.962 / 0.962 | n=0; — | n=0; — |
| ≥180 | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |

### R@.3

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
| [30, 60) | n=0; — | n=0; — | n=1 작은 표본; 1.000 / 1.000 / 1.000 | n=0; — | n=1 작은 표본; 1.000 / 1.000 / 1.000 |
| [60, 120) | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
| [120, 180) | n=0; — | n=0; — | n=1 작은 표본; 1.000 / 1.000 / 1.000 | n=0; — | n=0; — |
| ≥180 | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |

### R@.5

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
| [30, 60) | n=0; — | n=0; — | n=1 작은 표본; 1.000 / 1.000 / 1.000 | n=0; — | n=1 작은 표본; 1.000 / 1.000 / 1.000 |
| [60, 120) | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
| [120, 180) | n=0; — | n=0; — | n=1 작은 표본; 1.000 / 1.000 / 1.000 | n=0; — | n=0; — |
| ≥180 | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |

### R@.7

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
| [30, 60) | n=0; — | n=0; — | n=1 작은 표본; 0.000 / 0.000 / 0.000 | n=0; — | n=1 작은 표본; 1.000 / 1.000 / 1.000 |
| [60, 120) | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
| [120, 180) | n=0; — | n=0; — | n=1 작은 표본; 1.000 / 1.000 / 1.000 | n=0; — | n=0; — |
| ≥180 | n=0; — | n=0; — | n=0; — | n=0; — | n=0; — |
