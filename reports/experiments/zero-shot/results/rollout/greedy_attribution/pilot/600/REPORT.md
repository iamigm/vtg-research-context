# Greedy attribution · pilot600

고정 pilot **600 query** 중 **600 query**의 계측을 완료했다. 전체 범위의 고유 영상은 592개이며 완료 query에 포함된 고유 영상은 592개다. Query 수·영상 수·backward target 수를 구분한다. 이 범위의 원자료 계측이 완료되었다. Viewer 검증의 완료 여부는 아래 검증 기록을 따른다.

## 목적과 현재 범위

Source greedy token열을 고정한 계측 replay에서 모든 선행 입력의 gradient를 계산하고, 현재 경쟁 분기와 지정한 입력 변화에 민감한 잠재 분기를 함께 추적한다. 원영상 prefill 이후 one-token decoding을 재현하고 differentiable KV로 모든 선행 입력까지 gradient를 연결한다. Source KV와 계측 replay probability를 함께 기록한다. 새 답변을 생성하거나 성능을 개선하는 실험이 아니다. 기존 deterministic continuation은 mode 후보의 사후 참조로만 사용한다.

[계측 정의·실행 조건·층화 결과](APPENDIX.md)

완료 query 구성: activitynet-timelens: 200 / 200, charades-timelens: 200 / 200, qvhighlights-timelens: 200 / 200. 진행 순서에 따라 dataset 구성이 치우칠 수 있으므로 현재 평균을 pilot 전체로 일반화하지 않는다.

## Viewer를 읽는 순서

먼저 한 query의 상세 계측 분기를 고른다. 기본 선택은 직접 conditional rollout에서 시간 구간이 달라진 분기, 그 외 직접 continuation이 있는 분기, 나머지 분기 순으로 정하고 같은 조건에서는 alternative의 source probability가 큰 것을 먼저 연다. 이는 읽기 순서이며 정답 여부나 gradient 크기에 따른 표본 선별이 아니다.

1. **어떤 선택인가**: 같은 prefix 다음의 greedy token과 alternative token을 비교한다. 표시 probability는 다음 token의 native probability이며 최종 시간 구간의 probability가 아니다.
2. **결과가 어디로 가는가**: alternative token 하나를 강제한 뒤 greedy로 이어간 direct conditional rollout만 직접 비교한다. 같은 prefix를 공유하지만 이후 추가 강제 분기를 거친 continuation은 참고로 구분한다. 미관측은 결과가 같다는 뜻이 아니다.
3. **어느 방향으로 변하는가**: margin m=z(greedy)−z(alternative)의 signed activation × gradient를 읽는다. 입력 group을 미소 배율 확대할 때 양수는 greedy 쪽으로, 음수는 alternative 쪽으로 격차를 움직인다. 입력의 자연스러운 수정 효과나 제거 효과를 뜻하지 않는다.
4. **어디를 자세히 볼 것인가**: 필요할 때 token distribution, 전체 입력 위치, frame 지도, attention·MLP·message 상세를 연다. 내부 layer 점수는 각 module 출력의 미소 배율 확대에 대한 목표 변화 방향이다. Attention과 MLP 모두 이미 섞인 영상·query·prefix 정보를 처리한다.

입력 group의 signed sum S=Σᵢ⟨xᵢ,∂m/∂xᵢ⟩와 absolute sum A=Σᵢ|⟨xᵢ,∂m/∂xᵢ⟩를 함께 본다. |S|가 A에 비해 작으면 위치별 양·음 신호가 상쇄된 것이다. S가 작다는 이유만으로 입력 연결이 약하다고 판단하지 않는다. 상쇄 비율 C=1−|S|/A (A>0)이며 클수록 위치별 부호가 더 상쇄된다. A=0일 때는 정의하지 않는다. A는 위치별 signed 값의 절댓값 합이며 gradient L2와 다른 값이다. Group별 단위·위치 수가 다르므로 raw norm이나 signed sum을 전체 증거의 기여율로 환산하지 않는다.

Viewer의 입력 방향 막대는 ΔL̂=εS, ε=0.03125인 gradient 1차 예측이다. Margin target에서는 Δm̂이고, chosen-token 또는 endpoint target에서는 해당 log probability 목표의 예측 변화량이다. 같은 차트의 최대 절댓값으로 막대 길이를 정하며 기여율로 정규화하지 않는다. 영상은 normalized pixels 전체의 signed sum을 사용해 vision encoder와 DeepStack 경로를 포함한다. Visual embedding의 별도 값은 raw group 통계에서 확인한다. 원래 계산점의 gradient 예측과 유한차분 screening의 추정은 서로 다른 근사이며 실제 유한 변화의 결과와 다를 수 있다.

Screen gap은 지정한 ±3.125% group scaling 방향의 중앙 유한차분으로 추정한 두 token의 격차다. 실측 최소 gap은 같은 유한 변화에서 다시 계산한 두 token의 최소 logit 격차다. 음수이면 그 비교에서 alternative가 greedy보다 높지만, 다른 token까지 포함한 top1이나 전체 rollout의 mode 전환을 확인한 것은 아니다. Top30 밖은 미계측이다.

## 영상 overlay 읽기

영상 위치 화면은 원래 영상 위에 저장된 spatial gradient 지도를 겹쳐 표시한다. Pixel 지도는 실제 sampled frame 하나의 16×16 입력 pixel patch별 집계이며, visual embedding 지도는 두 sampled frame과 공간상 2×2 patch를 합친 위치별 값이다. 크기는 모델용 resize 이후 입력 기준이다. 원래 영상은 입력 전체 영역으로 resize되었고 crop·spatial padding을 사용하지 않았으므로 지도는 원본 영상의 가시 영역 전체에 대응시키며 player의 letterbox 영역에는 표시하지 않는다. Resize 과정에서 입력과 원본의 종횡비가 달라질 수 있으므로 원본 전체의 정규화 좌표에 대응시키며 입력 H×W를 함께 표시한다.

재생 시 현재 영상 시각과 source timestamp 차이가 가장 작은 sampled frame을 고른다. Pixel 지도는 그 frame, visual embedding 지도는 그 frame이 속한 두-frame pair의 값을 사용한다. 시간 사이의 새 gradient를 계산하거나 보간하지 않는다. Visual pair를 직접 선택하면 첫 sampled frame으로 이동하며, pair의 첫째·둘째 실제 시각을 따로 확인할 수 있다. Pair 평균 시각이 실제 sampled frame이라고 가정하지 않는다. Temporal packing을 위해 같은 source frame이 두 input slot에 반복될 수 있다. 반복 slot의 gradient는 다를 수 있으므로 수동 선택에서 각각 유지하며, 영상의 같은 frame에 표시한다. 재생 중 timestamp가 같으면 먼저 나온 slot을 선택한다.

시간 버튼은 위에 frame 또는 pair index, 아래에 저장된 실제 source timestamp를 소수점 셋째 자리까지 초 단위로 표시한다. 2 FPS의 명목 간격으로 index를 나눠 시각을 만들지 않으며 visual pair는 두 실제 source timestamp를 함께 표시한다.

Overlay의 color scale은 선택한 target·표현·metric의 모든 frame에 공통인 최대 절댓값 M=maxₜ,ᵢ|vₜ,ᵢ|을 기준으로 한다. Gradient norm은 u=min(1,|v|/M)을 보라 → 파랑 → 청록 → 초록 → 노랑의 sequential palette로 표시한다. Signed activation × gradient는 u=(clip(v/M,−1,1)+1)/2인 diverging palette를 사용한다. 음수는 빨강·주황, 0은 밝은 회색, 양수는 하늘색·파랑이며 양끝으로 갈수록 절댓값이 크다. 두 palette는 9개의 등간격 color stop 사이에서 RGB를 선형 보간한다. M=0이면 모든 patch를 해당 palette의 0 색상으로 표시한다. Legend는 같은 함수로 생성하며 norm은 0·M/4·M/2·3M/4·M, signed는 −M·−M/2·0·M/2·M의 실제 값을 표기한다. 시간 버튼과 선행 위치 heatmap에도 같은 palette를 사용하되, 각각 frame 집계값과 전체 선행 위치 값의 최대 절댓값을 기준으로 한다. 모든 patch의 색 자체는 완전히 불투명하다. 투명도 T(%)에 따른 전체 overlay alpha는 1−T/100이다. T=0이면 0값 patch까지 heatmap이 영상을 완전히 가리고, T=100이면 원영상만 보인다. 투명도 조절은 화면 표시만 바꾸며 원래 수치를 바꾸지 않는다. 표현이나 target을 바꾸면 M도 다시 정해지므로 서로 다른 선택 간 색만 비교하지 않는다. Visual embedding은 이미 다른 위치의 정보를 섞은 표현이며, overlay 영역만의 독립적 인과 효과는 아니다.

## 계측 coverage

| 항목 | 실제 수 | 분모·해석 |
|---|---:|---|
| 완료 query | 600 / 600 | 같은 고정 cohort의 완료 query 수 |
| Greedy step | 10697 | 완료 query의 출력 token 위치 합계 |
| Backward target | 14297 | Chosen token·endpoint 합계·선별 margin 목표의 합계 |
| 평균 계측 replay top30 mass | 1.000000 | step 내 합계를 query 평균 후 query 간 평균; 재정규화 없음 |
| 평균 계측 replay omitted mass | 1.161e-08 | 1−top30 mass; top30 밖 개별 분기는 미계측 |
| 상세 branch target | 2400 | 다른 token과의 margin을 계측한 목표 수 |
| 현재 경쟁 candidate | 17060 | step×token 후보 수; query 수가 아님 |
| 잠재 경쟁 candidate | 7301 | 현재 경쟁과 중복 가능 |
| 잠재 경쟁만 해당 | 0 | 현재 경쟁 기준을 통과하지 않은 잠재 후보 |
| 분리된 구간을 관측한 branch target | 937 / 2400 | 기존 continuation 기준; 이후 추가 강제 분기 포함 가능, direct 전환 수와 다름 |
| Direct conditional rollout이 있는 branch target | 1618 / 2400 | 해당 token까지 강제한 후 greedy로 완성한 기존 경로 |
| Direct 구간 분리 branch target | 571 / 2400 | direct 경로에서 max endpoint 차이가 max(1 s, 영상 길이×0.02)를 초과; 사건 의미 차이를 확정하지 않음 |
| 재현 오차에 민감한 branch target | 0 / 2400 | replay margin에 비해 source와의 차이가 큰 목표 |
| 기존 continuation 미관측 branch target | 782 / 2400 | mode를 unknown으로 유지 |

현재 경쟁은 source KV의 고정 top30 안에서 계측 replay native probability≥1% 또는 greedy와 logit gap≤log(100)인 후보이며, 잠재 경쟁은 지정한 source-group 변화 방향의 screening에서 선택된 후보다. 세부 threshold·후보 상한·source와 replay probability는 appendix와 실행 설정에 기록한다. Top30 밖은 미계측이며 이 수치는 full mode probability나 전체 입력 방향에 대한 coverage가 아니다.

## Source greedy 성능

IoU=예측·GT 교집합 길이÷합집합 길이, mIoU=query별 IoU 평균, R@τ=IoU≥τ인 query 수÷완료 query 수다. 유효 구간이 없으면 IoU=0으로 평가하며 높을수록 좋다. 같은 source greedy 답변을 유지하므로 Δpp=100×(현재−동일 source)=0.00이다.

| 조건 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---|---:|---:|---:|---:|---:|
| Source greedy · 답변 고정 | 600 | 0.5157 (+0.00) | 0.6783 (+0.00) | 0.5433 (+0.00) | 0.3817 (+0.00) |

## 입력 종류별 계측

아래 값은 chosen-token target에서 입력 group별 값을 먼저 query 내 평균한 뒤 query 간 평균한 값이다. 각 query를 동일 가중하며 endpoint 합계와 branch margin은 섞지 않는다. Gradient L2는 민감도이고 signed sum은 해당 group 표현을 미소 배율 확대하는 방향의 변화량이다. 절댓값 합은 위치별 signed 값의 절댓값 합으로 상쇄 정도를 보조한다. 세 값은 성공률이 아니며 크기가 클수록 좋은 지표도 아니다. 서로 다른 표현 단위의 raw 값과 그룹 크기에 영향을 받는다.

| 입력 group | 계측 query n | 평균 gradient L2 | 평균 signed sum | 평균 absolute sum |
|---|---:|---:|---:|---:|
| visual | 600 | 0.494101 | -0.001055 | 2.034088 |
| timestamp | 600 | 0.507543 | 0.013202 | 0.383323 |
| query | 600 | 0.239176 | -0.000278 | 0.017134 |
| instruction | 600 | 0.565198 | 0.004557 | 0.094463 |
| special | 600 | 1.098283 | 0.000472 | 0.043912 |
| generated | 600 | 0.271202 | 0.001607 | 0.041036 |

## 재현 검증과 실제 비용

| 항목 | 값 | 정의 |
|---|---:|---|
| 계측 실행 기준 통과 query | 600 / 600 | 입력 hash·causality·source argmax 일치·probability 오차·gradient forward 일치 |
| 최대 top30 absolute Δp | 8.643e-07 | 모든 완료 query·step·고정 top30의 max absolute(p_replay−p_source); 낮을수록 일치 |
| 동일 계측 replay gradient/no-grad 최대 logit 차이 | 0.000000 | Source KV와의 비교와 별도; 0을 요구 |
| Strict greedy argmax 일치 query | 600 / 600 | 모든 step의 argmax가 source greedy와 일치 |
| 평균 query wall s | 102.345 | 계측 query별 실측 평균; 대기·전체 작업 경과 시간과 구분 |
| Query wall 합 s | 61406.734 | 완료 query의 wall 실측 합 |
| 최대 GPU allocated GiB | 83.706 | query별 peak allocated 중 최댓값 |
| Scope·viewer 검증 | 통과 | 계측 완료와 후처리 검증 완료를 구분 |

각 query의 정확한 입력 hash·source KV와 계측 replay probability drift·future-position gradient 검증과 비용은 원자료에 남긴다. Source greedy argmax는 모든 step에서 정확히 일치해야 한다. Top30 probability 최대 절대 오차는 0.001 이하여야 하며, 허용 범위의 실제 차이도 그대로 기록한다.

## 해석 범위

이 결과는 source greedy token열을 고정한 계측 replay 계산점의 local sensitivity와 내부 신호 방향이다. 큰 gradient가 인과적 증거를 확정하지 않고, 작은 영상 gradient도 영상 근거가 없다는 증거가 아니다. 이전 생성 위치의 표현에 영상 정보가 들어 있을 수 있다. Layers의 점수를 합쳐 전체 증거량으로 해석하지 않는다. 잠재 경쟁 screening은 정한 변화 방향만 조사하므로 다른 방향의 급변을 배제하지 않는다. 실제 mode 전환은 변경 입력으로 이전 경로부터 다시 생성해야 확인할 수 있다.

원자료: [600 query source references](../../../../../artifacts/runs/rollout/greedy_attribution/pilot/600/metadata/source_records.json) · [동결 실행 설정](../../../../../artifacts/runs/rollout/greedy_attribution/pilot/900/metadata/config.json) · [계측 상태](../../../../../artifacts/runs/rollout/greedy_attribution/pilot/900/metadata/status.json)

## Backward 반복 감사에서 확인한 한계

`activitynet-timelens_00261`의 `margin_12_20` 한 target에서 forward logits는 저장값과 일치했지만, 같은 graph의 backward 반복은 동일한 pixel gradient를 재현하지 못했다. 저장 pixel signed sum은 -0.050869, 원래 message hook을 포함한 두 반복값은 0.219944 · 0.105905였다. 두 반복의 gradient 상대 L2 차이 ‖g₂−g₁‖₂/‖g₁‖₂는 0.014233이며 낮을수록 잘 재현된다. 이 사례의 작게 상쇄된 pixel signed 방향을 안정적인 방향으로 해석하지 않는다. 원인을 특정 연산으로 확정하지 않았으며 다른 모든 target의 반복 안정성을 검사한 결과도 아니다.

이 반복 차이는 한 계산 안의 FP32 합산 순서 차이와 별개다. 저장 원자료의 shape·hash·합산 오차 상한 검증이 통과해도 gradient 반복 재현성까지 통과했다는 뜻은 아니다. 원래 계측과 audit 결과를 함께 보존하며, 후속 pilot에도 같은 계측 조건을 유지한다. [반복 감사 수치와 provenance](../../../../../artifacts/runs/rollout/greedy_attribution/pilot/900/validation/pixel_sum_audit.json)
