# Oracle switch · pilot3

**측정 완료 3/3 query · 상태 `complete`**

선택된 기존 greedy·oracle 결과와 새 RGB perturbation 측정을 구분한다. 대기 중인 query의 곡선과 새 decoding 결과는 아직 없다.

## 선택된 사례

| Dataset | Query | Greedy span | Greedy IoU | Oracle span | Oracle IoU | ΔIoU | 최초 분기 | Source rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QVHighlights | qvhighlights-timelens_00229 | 45.00–55.00 s | 0.0000 | 5.00–10.00 s | 1.0000 | 1.0000 | Step 5: 4 → 5 | 4 |
| ActivityNet | activitynet-timelens_00812 | 1.00–10.00 s | 0.0000 | 75.00–79.00 s | 1.0000 | 1.0000 | Step 5: 1 → 7 | 2 |
| Charades | charades-timelens_00810 | 0.00–6.00 s | 0.0000 | 21.00–29.00 s | 1.0000 | 1.0000 | Step 5: 0 → 2 | 2 |

ΔIoU=저장된 oracle IoU−저장된 greedy IoU이며 새 perturbation의 효과가 아니다. Dataset별 1 query씩 총 3 query다. 완료 수는 실제 새 query 측정 완료 수이며 trial·방향·생성 응답 수와 구분한다.

## 측정 대상과 해석

이 pilot은 기존 deterministic rollout에서 dataset별 oracle−greedy IoU 차이가 큰 query를 한 개씩 고른 **총 3 query의 진단 실험**이다. Oracle은 저장된 후보 중 GT IoU가 가장 높은 결과다. GT를 사용해 고른 극단 사례이며 dataset 평균 성능이나 일반적인 전환 성공률을 추정하는 표본이 아니다.

원래 greedy prefix를 그대로 고정하고 최초 oracle 분기의 token g와 o를 비교한다. 원래 모델 입력으로 resize한 float RGB x∈[0,1]에서 m(x)=z_g−z_o의 gradient를 계산한다. 고정 방향 d=−sign(∇ₓm)에 변경 범위 mask M을 적용하여 x′(ε)=clip(x+εMd,0,1)로 바꾼다. Text token·text embedding·FPS·frame IDs·timestamp·spatial resolution·token 구성은 바꾸지 않는다. Pixel 변화에 따라 visual embedding은 달라질 수 있다. 같은 source frame이 반복 입력된 slot은 gradient를 합산해 방향을 정하고 동일한 perturbation을 적용한다.

변경 범위는 전체 영상, gradient L1 합이 가장 큰 frame pair, 그 pair에서 L1 합이 가장 큰 32×32 spatial cell이다. 변형은 resize 이후 모델 입력의 float RGB에 적용하며 압축된 원본 영상 자체를 편집한 것이 아니다. Processor의 실제 mean/std를 사용해 모델 입력으로 변환한다. ε₂₅₅=255ε는 8-bit intensity level 단위이며 소수도 허용한다. 추론 입력을 uint8로 양자화하지 않는다. Preview PNG만 표시를 위해 반올림될 수 있다.

초기 ε₂₅₅ grid는 0.25, 0.5, 1, 2, 4, 8, 16, 32다. 한 방향에서 처음 관측한 branch reached의 실패·성공 구간을 6회 추가 계측한다. 이 구간은 시험한 고정 gradient ray에서의 관측 경계이며, 미계측 구간의 단조성이나 전역 최소 adversarial distance를 보장하지 않는다. 전체 영상 방향에서 실패해도 국소 변경 범위의 방향은 별도로 조사한다.

| 지표 | 정의와 해석 |
|---|---|
| Paired margin | m=z_g−z_o. 음수이면 oracle token이 원래 greedy token보다 높다. 다른 token보다 높은지는 별도다. |
| Oracle margin | r=z_o−max_{v≠o}z_v. 양수이면 전체 vocabulary에서 oracle token이 단독 top1이다. 0은 동률이다. |
| Target top1 | 모델의 실제 전체 vocabulary argmax ID가 o인지 검사한다. 동률에서 o가 선택될 수 있으므로 strict target top1과 구분한다. |
| Prefix argmax match | 원래 prefix를 teacher-forcing으로 평가할 때 분기 이전 모든 step의 실제 argmax가 원래 token과 일치하는지 검사한다. |
| Branch reached | target top1 AND prefix argmax match. 이 조건을 만족하는 첫 관측 경계를 세분화하며 응답 처음부터 같은 greedy decoding으로 다시 확인한다. |
| Strict target top1 | Oracle margin>0. 모든 다른 token보다 o의 logit이 엄격히 높다. |
| Probability | 원래 고정 top30 ID의 native softmax probability. Top30 안에서 재정규화하지 않는다. 전체 분포는 저장하지 않고 전체 argmax ID 등 검증 scalar만 별도 기록한다. |
| RGB L∞ | maxᵢ|x′ᵢ−xᵢ|. Clipping 이후의 실제 최대 변경량이며 계획 ε보다 작을 수 있다. |
| RGB RMS | sqrt(meanᵢ((x′ᵢ−xᵢ)²)). 분모는 중복 input slot을 한 번만 센 전체 physical source frame의 RGB scalar 수이며 변경 범위 밖의 0 변경도 포함한다. |
| Changed fraction | 변경된 RGB scalar 수 / 중복 slot을 제외한 전체 physical source frame의 RGB scalar 수. |
| Temporal IoU | length(예측 구간∩GT) / length(예측 구간∪GT). 0–1이며 클수록 GT와 겹친다. |
| Oracle span IoU | 새 예측 구간과 저장된 oracle 구간의 temporal IoU. GT IoU와 다른 비교다. |

**Fixed-prefix probe**는 원래 prefix를 teacher-forcing으로 유지한 다음 token 평가이며, 응답을 생성하는 decoding 조건이 아니다. 결과표의 모든 행은 응답 처음부터 같은 greedy decoding을 사용한다. 조건은 입력 변경량의 차이다. 원래 prefix를 조건으로 한 token 역전, 실제 top1 전환, greedy 재생성의 prefix 도달 여부와 최종 구간·GT IoU를 각각 구분한다. Oracle branch reached와 oracle path equal도 서로 다른 검사다. Token 전환을 최종 mode 전환이나 정답 개선과 동일시하지 않는다.

Branch reached 경계 아래·위 입력 변경량에서 같은 greedy decoding 결과를 비교하고, 경계 위 변경량에서 반복 계측·반대 방향 control을 확인한다. 경계를 못 찾은 방향도 최대 시험 ε에서 같은 greedy decoding을 실행하며 입력 변경 조건을 시험 최대 변경량으로 표시한다. 미관측 결과는 실패로 세지 않는다. 전환을 못 찾은 방향은 해당 변경 범위·고정 방향·시험한 ε 범위에서의 음성 결과다. 자연스러운 영상 수정의 효과, 다른 방향의 불가능성, 전역 최소 perturbation을 주장하지 않는다.

## PNG 비교의 의미

원본 입력과 실제 변경 입력의 ×1 PNG는 float RGB를 8-bit 표시값으로 반올림한 것이다. 미세한 float 변경은 이 반올림 뒤 같아 보이거나 완전히 같은 PNG가 될 수 있다. PNG 일치 여부와 실제 float RGB의 최대 변경량을 함께 표시하며, PNG가 같다는 이유만으로 모델 입력이 같다고 판단하지 않는다.

확대 이미지는 x_display=clip(x+G(x′−x),0,1)로 변경 차이만 G배 확대해 원본 위에 표시한 것이다. **표시용 확대이며 모델에 넣은 영상이 아니다.** 표시 배율 G와 확대 후 clipping 비율을 따로 명시한다. 실제 ×1 이미지를 일부러 다르게 만들지 않으며 수치·응답·원자료는 변경하지 않는다.

## QVHighlights · qvhighlights-timelens_00229

A blonde woman in beige suite and white blouse is talking in front of a camera

저장된 oracle continuation과 최초 분기를 비교합니다.

원래 GT 5.00–10.00 s, greedy 45.00–55.00 s, oracle 5.00–10.00 s.

새 측정 상태: 완료. 기록된 방향 3개, trial 28개. Paired reversal 관측 1/3 방향, target top1 관측 1/3, branch reached 관측 1/3, strict target top1 관측 0/3.

아래 ε는 모두 ε₂₅₅ 단위다. 최소 관측 값은 저장된 trial 중의 최솟값이며 미계측 ε의 최솟값이 아니다.

동일 graph의 두 backward 반복: gradient 상대 L2 차이 0.0163, cosine 0.9999, sign 일치율 0.9934. 첫 gradient로 방향을 고정하며 실제 유한 변화의 forward 결과로 전환을 판단한다.

변형 전 free greedy token 재현: 전체 일치. 고정 top30 probability 최대 오차 0.0000.

| 변경 범위 | 첫 paired reversal ε | 첫 target top1 ε | 첫 strict top1 ε | 첫 branch reached ε | 관측 branch 경계 | 경계 아래 IoU | 경계 위 IoU | 시험 최대 IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 전체 영상 | 0.062 | 0.094 | — | 0.094 | (0.0898, 0.0938] | 0.8000 | 1.0000 | — |
| 선택한 frame pair | — | — | — | — | 미관측 · 최대 32.000 | — | — | 0.4762 |
| 선택한 spatial cell | — | — | — | — | 미관측 · 최대 32.000 | — | — | 0.0000 |

모든 행은 응답 처음부터 같은 greedy decoding이다. 입력 변경 조건은 경계 아래·위 또는 시험 최대 변경량을 구분한다.

| 변경 범위 | 입력 변경 조건 | ε₂₅₅ | 새 구간 | GT IoU | Oracle span IoU | 첫 token 차이 step | Oracle branch 도달 | Oracle path 일치 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 전체 영상 | 경계 아래 변경량 | 0.0898 | 6.00–10.00 s | 0.8000 | 0.8000 | 5 | 아니오 | 아니오 |
| 전체 영상 | 경계 위 변경량 | 0.0938 | 5.00–10.00 s | 1.0000 | 1.0000 | 5 | 예 | 아니오 |
| 선택한 frame pair | 시험 최대 변경량 | 32.0000 | 0.00–10.50 s | 0.4762 | 0.4762 | 5 | 아니오 | 아니오 |
| 선택한 spatial cell | 시험 최대 변경량 | 32.0000 | 45.00–55.00 s | 0.0000 | 0.0000 | None | 아니오 | 아니오 |

**전체 영상 변경 범위:** 입력 slot: 302개 · [0, 1, 2] … [299, 300, 301] · Source frame ID: 301개 · [0, 15, 30] … [4466, 4481, 4495] · 실제 timestamp: 302개 · 0.000–149.983 s · Spatial bounds: null.
8-bit 반올림 뒤 원본과 실제 변경 PNG가 동일합니다. Float RGB에는 아래 크기의 변경이 있습니다. 실제 float 최대 RGB 변경량: 0.093749 /255. 변경 차이 표시 배율: 341.339×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 8.371%.
Repeat high: margin -3.2500, oracle margin 0.0000, target top1=True.
Opposite control: ε₂₅₅=-0.0938, margin 6.1250, target top1=False.

**선택한 frame pair 변경 범위:** 입력 slot: [90, 91] · Source frame ID: [1349, 1364] · 실제 timestamp: 45.012 s, 45.512 s · Spatial bounds: null.
8-bit PNG에서도 원본과 실제 변경 입력의 차이가 남습니다. 실제 float 최대 RGB 변경량: 32.000006 /255. 변경 차이 표시 배율: 1.000×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 0.000%.

**선택한 spatial cell 변경 범위:** 입력 slot: [90, 91] · Source frame ID: [1349, 1364] · 실제 timestamp: 45.012 s, 45.512 s · Spatial bounds: [32, 64, 320, 352].
8-bit PNG에서도 원본과 실제 변경 입력의 차이가 남습니다. 실제 float 최대 RGB 변경량: 32.000006 /255. 변경 차이 표시 배율: 1.000×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 0.000%.

[Query 원자료](../../../../../artifacts/runs/rollout/oracle_switch/pilot/3/queries/qvhighlights-timelens_00229.json)

## ActivityNet · activitynet-timelens_00812

She shows off the contact in her eye

저장된 oracle continuation과 최초 분기를 비교합니다.

원래 GT 75.00–79.00 s, greedy 1.00–10.00 s, oracle 75.00–79.00 s.

새 측정 상태: 완료. 기록된 방향 3개, trial 28개. Paired reversal 관측 1/3 방향, target top1 관측 1/3, branch reached 관측 1/3, strict target top1 관측 1/3.

아래 ε는 모두 ε₂₅₅ 단위다. 최소 관측 값은 저장된 trial 중의 최솟값이며 미계측 ε의 최솟값이 아니다.

동일 graph의 두 backward 반복: gradient 상대 L2 차이 0.0193, cosine 0.9998, sign 일치율 0.9937. 첫 gradient로 방향을 고정하며 실제 유한 변화의 forward 결과로 전환을 판단한다.

변형 전 free greedy token 재현: 전체 일치. 고정 top30 probability 최대 오차 0.0000.

| 변경 범위 | 첫 paired reversal ε | 첫 target top1 ε | 첫 strict top1 ε | 첫 branch reached ε | 관측 branch 경계 | 경계 아래 IoU | 경계 위 IoU | 시험 최대 IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 전체 영상 | 0.008 | 0.008 | 0.008 | 0.008 | (0.0039, 0.0078] | 0.0000 | 1.0000 | — |
| 선택한 frame pair | — | — | — | — | 미관측 · 최대 32.000 | — | — | 0.0000 |
| 선택한 spatial cell | — | — | — | — | 미관측 · 최대 32.000 | — | — | 0.0000 |

모든 행은 응답 처음부터 같은 greedy decoding이다. 입력 변경 조건은 경계 아래·위 또는 시험 최대 변경량을 구분한다.

| 변경 범위 | 입력 변경 조건 | ε₂₅₅ | 새 구간 | GT IoU | Oracle span IoU | 첫 token 차이 step | Oracle branch 도달 | Oracle path 일치 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 전체 영상 | 경계 아래 변경량 | 0.0039 | 1.00–10.00 s | 0.0000 | 0.0000 | None | 아니오 | 아니오 |
| 전체 영상 | 경계 위 변경량 | 0.0078 | 75.00–79.00 s | 1.0000 | 1.0000 | 5 | 예 | 예 |
| 선택한 frame pair | 시험 최대 변경량 | 32.0000 | 1.00–10.00 s | 0.0000 | 0.0000 | None | 아니오 | 아니오 |
| 선택한 spatial cell | 시험 최대 변경량 | 32.0000 | 1.00–10.00 s | 0.0000 | 0.0000 | None | 아니오 | 아니오 |

**전체 영상 변경 범위:** 입력 slot: 160개 · [0, 1, 2] … [157, 158, 159] · Source frame ID: 160개 · [0, 15, 30] … [2355, 2370, 2385] · 실제 timestamp: 160개 · 0.000–79.500 s · Spatial bounds: null.
8-bit 반올림 뒤 원본과 실제 변경 PNG가 동일합니다. Float RGB에는 아래 크기의 변경이 있습니다. 실제 float 최대 RGB 변경량: 0.007812 /255. 변경 차이 표시 배율: 4096.063×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 15.946%.
Repeat high: margin -0.3750, oracle margin 0.3750, target top1=True.
Opposite control: ε₂₅₅=-0.0078, margin 1.3750, target top1=False.

**선택한 frame pair 변경 범위:** 입력 slot: [114, 115] · Source frame ID: [1710, 1725] · 실제 timestamp: 57.000 s, 57.500 s · Spatial bounds: null.
8-bit PNG에서도 원본과 실제 변경 입력의 차이가 남습니다. 실제 float 최대 RGB 변경량: 32.000006 /255. 변경 차이 표시 배율: 1.000×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 0.000%.

**선택한 spatial cell 변경 범위:** 입력 slot: [114, 115] · Source frame ID: [1710, 1725] · 실제 timestamp: 57.000 s, 57.500 s · Spatial bounds: [320, 352, 160, 192].
8-bit PNG에서도 원본과 실제 변경 입력의 차이가 남습니다. 실제 float 최대 RGB 변경량: 31.999998 /255. 변경 차이 표시 배율: 1.000×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 0.472%.

[Query 원자료](../../../../../artifacts/runs/rollout/oracle_switch/pilot/3/queries/activitynet-timelens_00812.json)

## Charades · charades-timelens_00810

A person is standing and eating some food

저장된 oracle은 여러 강제 분기(1 → 2 → 4)를 거친 경로입니다. 최초 분기만 바꾼 기존 결과는 22.00–29.00 s, GT IoU 0.8750입니다. 최초 oracle token 하나의 전환이 전체 oracle 경로 재현을 보장하지 않습니다.

원래 GT 21.00–29.00 s, greedy 0.00–6.00 s, oracle 21.00–29.00 s.

새 측정 상태: 완료. 기록된 방향 3개, trial 31개. Paired reversal 관측 2/3 방향, target top1 관측 2/3, branch reached 관측 2/3, strict target top1 관측 2/3.

아래 ε는 모두 ε₂₅₅ 단위다. 최소 관측 값은 저장된 trial 중의 최솟값이며 미계측 ε의 최솟값이 아니다.

동일 graph의 두 backward 반복: gradient 상대 L2 차이 0.0206, cosine 1.0000, sign 일치율 0.9943. 첫 gradient로 방향을 고정하며 실제 유한 변화의 forward 결과로 전환을 판단한다.

변형 전 free greedy token 재현: 전체 일치. 고정 top30 probability 최대 오차 0.0000.

| 변경 범위 | 첫 paired reversal ε | 첫 target top1 ε | 첫 strict top1 ε | 첫 branch reached ε | 관측 branch 경계 | 경계 아래 IoU | 경계 위 IoU | 시험 최대 IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 전체 영상 | 0.004 | 0.004 | 0.004 | 0.004 | (0.0000, 0.0039] | 0.0000 | 0.8750 | — |
| 선택한 frame pair | 0.695 | 0.695 | 0.695 | 0.695 | (0.6875, 0.6953] | 0.0000 | 0.8750 | — |
| 선택한 spatial cell | — | — | — | — | 미관측 · 최대 32.000 | — | — | 0.0000 |

모든 행은 응답 처음부터 같은 greedy decoding이다. 입력 변경 조건은 경계 아래·위 또는 시험 최대 변경량을 구분한다.

| 변경 범위 | 입력 변경 조건 | ε₂₅₅ | 새 구간 | GT IoU | Oracle span IoU | 첫 token 차이 step | Oracle branch 도달 | Oracle path 일치 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 전체 영상 | 경계 아래 변경량 | 0.0000 | 0.00–6.00 s | 0.0000 | 0.0000 | None | 아니오 | 아니오 |
| 전체 영상 | 경계 위 변경량 | 0.0039 | 22.00–29.00 s | 0.8750 | 0.8750 | 5 | 예 | 아니오 |
| 선택한 frame pair | 경계 아래 변경량 | 0.6875 | 0.00–6.00 s | 0.0000 | 0.0000 | None | 아니오 | 아니오 |
| 선택한 frame pair | 경계 위 변경량 | 0.6953 | 22.00–29.00 s | 0.8750 | 0.8750 | 5 | 예 | 아니오 |
| 선택한 spatial cell | 시험 최대 변경량 | 32.0000 | 0.00–6.00 s | 0.0000 | 0.0000 | None | 아니오 | 아니오 |

**전체 영상 변경 범위:** 입력 slot: 60개 · [0, 1, 2] … [57, 58, 59] · Source frame ID: 59개 · [0, 7, 15] … [420, 427, 435] · 실제 timestamp: 60개 · 0.000–29.029 s · Spatial bounds: null.
8-bit 반올림 뒤 원본과 실제 변경 PNG가 동일합니다. Float RGB에는 아래 크기의 변경이 있습니다. 실제 float 최대 RGB 변경량: 0.003906 /255. 변경 차이 표시 배율: 8192.125×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 21.231%.
Repeat high: margin -0.8750, oracle margin 0.8750, target top1=True.
Opposite control: ε₂₅₅=-0.0039, margin 1.8750, target top1=False.

**선택한 frame pair 변경 범위:** 입력 slot: [52, 53] · Source frame ID: [390, 397] · 실제 timestamp: 26.026 s, 26.493 s · Spatial bounds: null.
8-bit PNG에서도 원본과 실제 변경 입력의 차이가 남습니다. 실제 float 최대 RGB 변경량: 0.695317 /255. 변경 차이 표시 배율: 46.022×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 21.231%.
Repeat high: margin -0.2500, oracle margin 0.2500, target top1=True.
Opposite control: ε₂₅₅=-0.6953, margin 1.0000, target top1=False.

**선택한 spatial cell 변경 범위:** 입력 slot: [52, 53] · Source frame ID: [390, 397] · 실제 timestamp: 26.026 s, 26.493 s · Spatial bounds: [256, 288, 0, 32].
8-bit PNG에서도 원본과 실제 변경 입력의 차이가 남습니다. 실제 float 최대 RGB 변경량: 31.999998 /255. 변경 차이 표시 배율: 1.000×. 표시용 확대이며 모델에 넣은 영상이 아닙니다. 확대 표시 frame의 RGB scalar 중 clipping 비율: 0.001%.

[Query 원자료](../../../../../artifacts/runs/rollout/oracle_switch/pilot/3/queries/charades-timelens_00810.json)

## 원자료와 재현

- [선택 규칙·source hashes](../../../../../artifacts/runs/rollout/oracle_switch/pilot/3/metadata/selection.json)
- [실행 설정](../../../../../artifacts/runs/rollout/oracle_switch/pilot/3/metadata/config.json)
- [현재 실행 상태](../../../../../artifacts/runs/rollout/oracle_switch/pilot/3/metadata/status.json)
- [PNG 비교·표시 배율·원자료 대조](../../../../../artifacts/runs/rollout/oracle_switch/pilot/3/analyses/preview_manifest.json)

새 데이터는 원본 greedy·oracle source를 덮어쓰지 않는다. 입력 hash·실제 preprocessing·baseline 재현·support와 중복 source frame 처리·실제 clipping 이후 거리·runtime cost는 query 원자료에 보존한다. 이 보고서의 곡선은 측정점만 나타내며 점 사이 연결선은 미계측 결과를 뜻하지 않는다.
