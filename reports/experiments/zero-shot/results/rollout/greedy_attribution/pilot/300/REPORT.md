# Greedy attribution · pilot300

고정 pilot **300 query** 중 **100 query**의 계측을 완료했다. 전체 범위의 고유 영상은 292개이며 완료 query에 포함된 고유 영상은 96개다. Query 수·영상 수·backward target 수를 구분한다. 진행 중 부분집합의 수치이며 pilot 전체 결과로 해석하지 않는다.

## 목적과 현재 범위

Source greedy token열을 고정한 계측 replay에서 모든 선행 입력의 gradient를 계산하고, 현재 경쟁 분기와 지정한 입력 변화에 민감한 잠재 분기를 함께 추적한다. 원영상 prefill 이후 one-token decoding을 재현하고 differentiable KV로 모든 선행 입력까지 gradient를 연결한다. Source KV와 계측 replay probability를 함께 기록한다. 새 답변을 생성하거나 성능을 개선하는 실험이 아니다. 기존 deterministic continuation은 mode 후보의 사후 참조로만 사용한다.

[계측 정의·실행 조건·층화 결과](APPENDIX.md)

## 계측 coverage

| 항목 | 실제 수 | 분모·해석 |
|---|---:|---|
| 완료 query | 100 / 300 | 같은 고정 cohort의 완료 query 수 |
| Greedy step | 1717 | 완료 query의 출력 token 위치 합계 |
| Backward target | 2317 | Chosen token·endpoint 합계·선별 margin 목표의 합계 |
| 평균 계측 replay top30 mass | 1.000000 | step 내 합계를 query 평균 후 query 간 평균; 재정규화 없음 |
| 평균 계측 replay omitted mass | 6.498e-09 | 1−top30 mass; top30 밖 개별 분기는 미계측 |
| 상세 branch target | 400 | 다른 token과의 margin을 계측한 목표 수 |
| 현재 경쟁 candidate | 2587 | step×token 후보 수; query 수가 아님 |
| 잠재 경쟁 candidate | 1409 | 현재 경쟁과 중복 가능 |
| 잠재 경쟁만 해당 | 0 | 현재 경쟁 기준을 통과하지 않은 잠재 후보 |
| 분리된 구간을 관측한 branch target | 168 / 400 | 기존 continuation 기준 distinct mode candidate |
| 재현 오차에 민감한 branch target | 0 / 400 | replay margin에 비해 source와의 차이가 큰 목표 |
| 기존 continuation 미관측 branch target | 140 / 400 | mode를 unknown으로 유지 |

현재 경쟁은 source KV의 고정 top30 안에서 계측 replay native probability≥1% 또는 greedy와 logit gap≤log(100)인 후보이며, 잠재 경쟁은 지정한 source-group 변화 방향의 screening에서 선택된 후보다. 세부 threshold·후보 상한·source와 replay probability는 appendix와 실행 설정에 기록한다. Top30 밖은 미계측이며 이 수치는 full mode probability나 전체 입력 방향에 대한 coverage가 아니다.

## Source greedy 성능

IoU=예측·GT 교집합 길이÷합집합 길이, mIoU=query별 IoU 평균, R@τ=IoU≥τ인 query 수÷완료 query 수다. 유효 구간이 없으면 IoU=0으로 평가하며 높을수록 좋다. 같은 source greedy 답변을 유지하므로 Δpp=100×(현재−동일 source)=0.00이다.

| 조건 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---|---:|---:|---:|---:|---:|
| Source greedy · 답변 고정 | 100 | 0.4380 (+0.00) | 0.6100 (+0.00) | 0.4500 (+0.00) | 0.2800 (+0.00) |

## 입력 종류별 계측

아래 값은 chosen-token target에서 입력 group별 값을 먼저 query 내 평균한 뒤 query 간 평균한 값이다. 각 query를 동일 가중하며 endpoint 합계와 branch margin은 섞지 않는다. Gradient L2는 민감도이고 signed sum은 해당 group 표현을 미소 배율 확대하는 방향의 변화량이다. 절댓값 합은 위치별 signed 값의 절댓값 합으로 상쇄 정도를 보조한다. 세 값은 성공률이 아니며 크기가 클수록 좋은 지표도 아니다. 서로 다른 표현 단위의 raw 값과 그룹 크기에 영향을 받는다.

| 입력 group | 계측 query n | 평균 gradient L2 | 평균 signed sum | 평균 absolute sum |
|---|---:|---:|---:|---:|
| visual | 100 | 0.262467 | 0.000982 | 0.926423 |
| timestamp | 100 | 0.286005 | 0.010206 | 0.156974 |
| query | 100 | 0.184142 | 0.001029 | 0.011336 |
| instruction | 100 | 0.364258 | 0.006126 | 0.064855 |
| special | 100 | 0.662463 | 0.000442 | 0.021725 |
| generated | 100 | 0.187713 | 0.002485 | 0.029994 |

## 재현 검증과 실제 비용

| 항목 | 값 | 정의 |
|---|---:|---|
| 계측 실행 기준 통과 query | 100 / 100 | 입력 hash·causality·source argmax 일치·probability 오차·gradient forward 일치 |
| 최대 top30 absolute Δp | 2.384e-07 | 모든 완료 query·step·고정 top30의 max absolute(p_replay−p_source); 낮을수록 일치 |
| 동일 계측 replay gradient/no-grad 최대 logit 차이 | 0.000000 | Source KV와의 비교와 별도; 0을 요구 |
| Strict greedy argmax 일치 query | 100 / 100 | 모든 step의 argmax가 source greedy와 일치 |
| 평균 query wall s | 43.155 | 계측 query별 실측 평균; 대기·전체 작업 경과 시간과 구분 |
| Query wall 합 s | 4315.512 | 완료 query의 wall 실측 합 |
| 최대 GPU allocated GiB | 83.663 | query별 peak allocated 중 최댓값 |
| Scope·viewer 검증 | 완료 기록 대기 | 계측 완료와 후처리 검증 완료를 구분 |

각 query의 정확한 입력 hash·source KV와 계측 replay probability drift·future-position gradient 검증과 비용은 원자료에 남긴다. Source greedy argmax는 모든 step에서 정확히 일치해야 한다. Top30 probability 최대 절대 오차는 0.001 이하여야 하며, 허용 범위의 실제 차이도 그대로 기록한다.

## 해석 범위

이 결과는 source greedy token열을 고정한 계측 replay 계산점의 local sensitivity와 내부 신호 방향이다. 큰 gradient가 인과적 증거를 확정하지 않고, 작은 영상 gradient도 영상 근거가 없다는 증거가 아니다. 이전 생성 위치의 표현에 영상 정보가 들어 있을 수 있다. Layers의 점수를 합쳐 전체 증거량으로 해석하지 않는다. 잠재 경쟁 screening은 정한 변화 방향만 조사하므로 다른 방향의 급변을 배제하지 않는다. 실제 mode 전환은 변경 입력으로 이전 경로부터 다시 생성해야 확인할 수 있다.

원자료: [300 query source references](../../../../../artifacts/runs/rollout/greedy_attribution/pilot/300/metadata/source_records.json) · [동결 실행 설정](../../../../../artifacts/runs/rollout/greedy_attribution/pilot/300/metadata/config.json) · [계측 상태](../../../../../artifacts/runs/rollout/greedy_attribution/pilot/300/metadata/status.json)
