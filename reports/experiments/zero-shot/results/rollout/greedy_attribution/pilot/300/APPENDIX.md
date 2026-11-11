# Greedy attribution · 상세 정의와 층화 결과

이 표는 새 답변의 성능 개선을 측정하지 않는다. 고정된 source greedy 답변의 진단 표본 성능과 계측 coverage를 함께 기록한다. IoU=예측·GT 교집합 길이÷합집합 길이, mIoU=query별 IoU 평균, R@τ=IoU≥τ인 query 수÷평가 query 수다. 유효 구간이 없으면 IoU=0으로 평가한다. 네 지표 모두 높을수록 좋으며 동일 source greedy 대비 Δpp=0.00이다. n<10은 작은 표본, n=0은 빈 칸이다. 고정 cohort는 층화 진단 표본이며 전체 benchmark의 무작위 평균이 아니다.

## Dataset

| 범위 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | Backward targets |
|---|---:|---:|---:|---:|---:|---:|
| charades-timelens | 100 | 0.4380 (+0.00) | 0.6100 (+0.00) | 0.4500 (+0.00) | 0.2800 (+0.00) | 2317 |

## 영상 길이 s

| 범위 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | Backward targets |
|---|---:|---:|---:|---:|---:|---:|
| [0, 30) | 30 | 0.2981 (+0.00) | 0.4000 (+0.00) | 0.2667 (+0.00) | 0.1333 (+0.00) | 688 |
| [30, 60) | 70 | 0.4980 (+0.00) | 0.7000 (+0.00) | 0.5286 (+0.00) | 0.3429 (+0.00) | 1629 |
| [60, 120) · 빈 칸 | 0 | — | — | — | — | 0 |
| [120, 180) · 빈 칸 | 0 | — | — | — | — | 0 |
| ≥ 180 · 빈 칸 | 0 | — | — | — | — | 0 |

## GT/영상 길이 비율

| 범위 | n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | Backward targets |
|---|---:|---:|---:|---:|---:|---:|
| [0, 0.05) | 18 | 0.1253 (+0.00) | 0.0556 (+0.00) | 0.0000 (+0.00) | 0.0000 (+0.00) | 420 |
| [0.05, 0.1) | 26 | 0.3603 (+0.00) | 0.5769 (+0.00) | 0.3077 (+0.00) | 0.1154 (+0.00) | 599 |
| [0.1, 0.2) | 24 | 0.5259 (+0.00) | 0.8750 (+0.00) | 0.5417 (+0.00) | 0.3333 (+0.00) | 555 |
| [0.2, 0.4) | 21 | 0.6759 (+0.00) | 0.8571 (+0.00) | 0.8571 (+0.00) | 0.5714 (+0.00) | 492 |
| ≥ 0.4 | 11 | 0.4876 (+0.00) | 0.5455 (+0.00) | 0.5455 (+0.00) | 0.4545 (+0.00) | 251 |

## 영상 길이 × GT 비율 · mIoU

각 칸은 완료 query 수 n, 해당 범위의 source greedy 값과 동일 source 대비 Δpp다.

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥ 0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=7 작은 표본; 0.1104 (+0.00) | n=7 작은 표본; 0.1700 (+0.00) | n=5 작은 표본; 0.2947 (+0.00) | n=6 작은 표본; 0.6207 (+0.00) | n=5 작은 표본; 0.3563 (+0.00) |
| [30, 60) | n=11; 0.1347 (+0.00) | n=19; 0.4304 (+0.00) | n=19; 0.5867 (+0.00) | n=15; 0.6979 (+0.00) | n=6 작은 표본; 0.5969 (+0.00) |
| [60, 120) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [120, 180) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| ≥ 180 | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |

## 영상 길이 × GT 비율 · R@.3

각 칸은 완료 query 수 n, 해당 범위의 source greedy 값과 동일 source 대비 Δpp다.

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥ 0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=7 작은 표본; 0.0000 (+0.00) | n=7 작은 표본; 0.2857 (+0.00) | n=5 작은 표본; 0.6000 (+0.00) | n=6 작은 표본; 0.8333 (+0.00) | n=5 작은 표본; 0.4000 (+0.00) |
| [30, 60) | n=11; 0.0909 (+0.00) | n=19; 0.6842 (+0.00) | n=19; 0.9474 (+0.00) | n=15; 0.8667 (+0.00) | n=6 작은 표본; 0.6667 (+0.00) |
| [60, 120) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [120, 180) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| ≥ 180 | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |

## 영상 길이 × GT 비율 · R@.5

각 칸은 완료 query 수 n, 해당 범위의 source greedy 값과 동일 source 대비 Δpp다.

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥ 0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=7 작은 표본; 0.0000 (+0.00) | n=7 작은 표본; 0.0000 (+0.00) | n=5 작은 표본; 0.2000 (+0.00) | n=6 작은 표본; 0.8333 (+0.00) | n=5 작은 표본; 0.4000 (+0.00) |
| [30, 60) | n=11; 0.0000 (+0.00) | n=19; 0.4211 (+0.00) | n=19; 0.6316 (+0.00) | n=15; 0.8667 (+0.00) | n=6 작은 표본; 0.6667 (+0.00) |
| [60, 120) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [120, 180) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| ≥ 180 | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |

## 영상 길이 × GT 비율 · R@.7

각 칸은 완료 query 수 n, 해당 범위의 source greedy 값과 동일 source 대비 Δpp다.

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥ 0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=7 작은 표본; 0.0000 (+0.00) | n=7 작은 표본; 0.0000 (+0.00) | n=5 작은 표본; 0.0000 (+0.00) | n=6 작은 표본; 0.3333 (+0.00) | n=5 작은 표본; 0.4000 (+0.00) |
| [30, 60) | n=11; 0.0000 (+0.00) | n=19; 0.1579 (+0.00) | n=19; 0.4211 (+0.00) | n=15; 0.6667 (+0.00) | n=6 작은 표본; 0.5000 (+0.00) |
| [60, 120) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [120, 180) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| ≥ 180 | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |

## 목적과 계측 정의

원래 입력·frame sampling·spatial resolution·greedy prefix를 고정하고 모든 생성 token의 선택과 선행 입력의 연결을 계측한다. 영상·query·timestamp·instruction·special token·이미 생성된 prefix를 포함한다. GT는 사후 평가에만 사용한다.

- **Chosen log probability**: 각 step t의 Lₜ=log P(yₜ* | V,Q,y*<ₜ). Source greedy의 format·숫자·소수점·EOS를 고정한 계측 replay에서 측정하며 값이 클수록 해당 token의 replay probability가 높다.
- **Endpoint log probability**: 시작 또는 종료 숫자 token들의 Lₜ 합. 종료는 이미 생성된 시작 숫자를 조건으로 한다. 다른 길이의 token열 사이에서는 길이 효과를 포함한다.
- **Logit margin**: mₜ,k=zₜ(g)−zₜ(k)=log(Pₜ(g)/Pₜ(k)). 양수는 source greedy token g의 우세, 작을수록 두 선택이 가깝다. Margin gradient는 계측 replay 계산점에서 이 격차에 대한 민감도다. Source KV margin=log(p_source(g)/p_source(k))를 별도로 기록한다.
- **Input gradient norm**: 입력 위치 i의 ‖∂L/∂xᵢ‖₂. 위치 또는 표시한 입력 group에서의 L2 norm이며 값이 클수록 지정한 표현의 작은 변화에 민감하다. 픽셀·embedding의 단위가 달라 raw 값을 곧바로 의미적 기여 비율로 비교하지 않는다.
- **Signed activation × gradient**: ⟨xᵢ,∂L/∂xᵢ⟩. 해당 표현을 아주 조금 배율 확대하는 방향의 1차 변화량이다. 양수는 목표 증가, 음수는 목표 감소 방향이다. 그 신호의 제거 효과나 인과적 증거량은 아니다.
- **Top30 entropy contribution**: H₃₀=−Σᵥ∈top30 pᵥ log pᵥ, 단위 nats. 계측 replay의 native probability를 사용한 부분합이며 full entropy 또는 top30 내부 재정규화 entropy가 아니다. Top30 mass=Σᵥ∈top30 pᵥ, omitted mass=1−top30 mass다. Omitted mass의 개별 token은 계측하지 않는다.

## 현재 경쟁과 잠재 경쟁

현재 경쟁 candidate는 Source KV에서 고정한 top30 안에서 greedy 외 token 중 계측 replay native probability≥0.01 또는 logit gap≤4.605170인 token이다. 분기 집합 안에서 probability를 재정규화하지 않는다.

Screening은 입력 종류별 표현을 ±3.125% 배율 변경해 source KV에서 고정한 top30 token의 계측 replay logits만 계측한다. 영상은 normalized pixels, 텍스트는 입력 embedding에 대한 내부 진단이다. 변화 이후 top30을 다시 선택하지 않는다. 방향별 중앙 유한차분으로 greedy와 경쟁 token의 gap 감소량을 추정하고, 가장 크게 줄어드는 방향과 부호를 저장한다. Screen gap=계측 replay baseline logit gap−측정 방향의 최대 gap 감소 추정이며, 1 이하인 후보를 잠재 경쟁 대상으로 고려한다. 고정 top30에서 greedy 외 모든 candidate의 screening 수치를 보존하고 query당 최대 4개 분기에 상세 backward를 수행한다. 현재 경쟁 후보와 현재 경쟁 기준을 통과하지 않은 후보의 최소 screen gap을 각각 우선 선택한다. 나머지는 기존 continuation의 시간적 분리와 screen gap 순으로 채우며 가능한 한 다른 step을 조사한다. 따라서 상세 target에는 잠재 경쟁 threshold까지 도달하지 않은 낮은 probability probe도 포함될 수 있다. 낮은 probability를 곱해 민감도를 억제하지 않는다.

이 screening은 source KV의 고정 top30 안의 token과 지정한 유한 개 변화 방향에서의 진단이다. Top30 밖의 분기는 미계측이다. 자연스러운 query 수정·실제 영상 변화나 모든 입력 방향을 포괄하지 않으며, 낮은 sensitivity로도 분기를 영구 제외할 수 없다. 큰 변화의 비선형 급변이나 실제 top1 역전은 별도 재계산 확인이 필요하다.

## Temporal mode 후보

정확히 같은 token prefix와 분기 token을 공유하는 기존 deterministic continuation을 참조한다. Greedy 구간 대비 max(|Δstart|,|Δend|)>max(1s, 0.02×영상 길이)이면 distinct mode candidate로 표시한다. 이 기준은 구간의 시간적 분리를 판정하며 사건 의미의 차이를 확정하지 않는다. 기존 continuation이 없으면 unknown으로 유지한다. 한 분기의 관측 continuation이나 다음 token probability를 최종 mode probability로 해석하지 않는다.

## Source KV 재현과 gradient 계산

Source KV probability는 기존 greedy 실행의 incremental decoding 기록이다. 계측 replay는 같은 원영상 prefill 이후 source token을 한 번에 하나씩 입력하는 계산 순서를 유지한다. KV는 immutable differentiable chunks로 연결해 prefill부터 모든 생성 step까지 gradient가 이어지도록 한다. 기존에 분리 저장한 inference KV를 가져와 gradient graph를 대체하지 않는다. Source KV와 계측 replay probability를 나란히 보존하며 어느 쪽도 top30 내부에서 재정규화하지 않는다. Max top30 |Δp|=query의 모든 step과 고정 top30 token에 대한 |p_replay−p_source|의 최댓값으로 낮을수록 재현 오차가 작다. 이 값은 0.001 이하여야 하며, 모든 step의 전체-vocabulary argmax가 source greedy token과 정확히 일치해야 한다. 이 argmax 검사는 top30 밖 분기의 sensitivity를 계측하는 것이 아니다. 같은 계측 replay의 gradient-enabled forward와 no-grad forward 사이의 top30 logit 차이는 별도로 검사하며 0이어야 한다. Margin 차이 |m_replay−m_source|≥max(|m_replay|,10⁻⁶)이거나 두 margin의 부호가 다르면 재현 오차에 민감한 후보로 표시한다.

## 내부 계측 coverage

Layers 표는 각 decoder layer의 attention output, MLP output, residual input에서 선택한 prediction receiver의 signed activation × gradient다. Message score=Aᵢⱼ〈∂L/∂oᵢ,vⱼ〉는 o_proj 이전 head 좌표에서 메시지를 미소 강화하는 방향이다. Message 계측은 decoder layer 0·17·35의 prediction receivers에 한정하고 head·receiver·부호별 상위 source edge를 보존한다. Visual-to-visual 및 그 외 간접 receiver의 edge는 미계측이다. Attention row 재구성 오차를 원자료에 남긴다. Visual embedding gradient는 visual replacement 이후, DeepStack 추가 주입 이전의 language-model 입력에서 측정한다. Video pixel gradient는 vision encoder와 DeepStack 경로를 포함하므로 두 지도를 구분한다.

## 해석과 재현 범위

선행 입력 변화 이후에도 원래 prefix를 teacher-force하므로 현재 step에 대한 conditional 분석이다. 실제 입력 변화가 더 앞선 생성부터 바꾸는 효과는 이 기록으로 확정할 수 없다. Gradient가 작다고 영상 근거가 없다고 판단하지 않는다. 이전 생성 위치의 표현에도 영상·query 정보가 들어 있을 수 있고, MLP를 곧바로 language prior로 이름 붙이지 않는다. 층별·경로별 signed 점수를 모두 합쳐 전체 증거량을 만들지 않는다.

모델 파라미터는 고정하고 원영상 prefill부터 gradient 계산 그래프를 다시 구성한다. 기존 inference KV를 독립적인 입력 증거로 사용하지 않는다. 원자료에는 입력 tensor hash, source greedy argmax 일치 여부, source KV와 계측 replay probability drift, 같은 계측 replay 계산의 gradient/no-grad 일치, 연산 시간과 memory를 기록한다. 계측 replay는 모든 source greedy argmax 일치와 top30 probability 최대 절대 오차≤0.001를 요구한다.

참고: [DeepFool의 margin·국소 선형화](https://arxiv.org/html/1511.04599v3). 본 실험의 source-group screening과 attribution 계측은 해당 논문의 VTG 재현 실험이 아니다.

