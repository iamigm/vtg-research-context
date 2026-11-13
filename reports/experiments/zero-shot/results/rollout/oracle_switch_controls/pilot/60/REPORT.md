# Visual perturbation controls · pilot60

**Control 확정 60/60 query · 이 중 baseline 재현 실패 0 · 상태 `complete`**

같은 RGB 변경 예산에서 targeted 방향의 효과를 반대·random·반복 gradient·8-bit roundtrip과 비교한다. 기존 극단 3사례는 이 60개의 집계에 포함하지 않는다.

## 현재 관측의 해석

선정 60 query의 처리가 확정됐다(Charades 20/20, ActivityNet 20/20, QVHighlights 20/20; baseline 재현 실패 0). Branch token 도달과 저장된 oracle 전체 경로 재현·GT IoU 개선은 별개다.

ε₂₅₅=0.25에서 모집단 가중 관측 branch 도달률은 targeted 49.9% / random 1.1%다. 선정 query 기준 paired targeted−random 차이는 0.4882(95% cluster bootstrap [0.3550, 0.6154])이며, 관측 ΔGT IoU는 targeted 0.0675 / random 0.0087다. 전환률 차이와 성능 차이를 각각 읽어야 한다.

ε₂₅₅=1에서 모집단 가중 관측 branch 도달률은 targeted 56.8% / random 2.2%다. 선정 query 기준 paired targeted−random 차이는 0.5459(95% cluster bootstrap [0.4118, 0.6768])이며, 관측 ΔGT IoU는 targeted 0.0199 / random 0.0085다. 전환률 차이와 성능 차이를 각각 읽어야 한다.

ε₂₅₅=0.25의 8-bit roundtrip 후 normalized tensor가 원본과 완전히 같은 경우는 관측 60/60 query다. 동일한 경우 모델에 남은 입력 변경이 없음을 뜻하며, 양자화 전 변경이나 다른 변경량의 효과까지 없다고 뜻하지 않는다.

사전 지정 precision 진단의 첫 projection FP32 · BF16 입력에서 native baseline prefix·선택 token 일치는 관측 4/6 query다. 두 예산의 실제 branch 결과가 native와 같은 비교 11/12, 다른 비교 1/12다.

사전 지정 precision 진단의 첫 projection FP32 · FP32 입력에서 native baseline prefix·선택 token 일치는 관측 4/6 query다. 두 예산의 실제 branch 결과가 native와 같은 비교 11/12, 다른 비교 1/12다.

Precision 진단은 사전 지정6 query의 첫 projection만 바꾼 비교이며 전체 FP32 모델 실험이 아니다. 각 mode의 자체 baseline 기준 Δoracle margin과 baseline 일치 여부를 함께 읽는다. 방향·양자화·첫 계산 정밀도의 영향을 좁혀 보는 증거이며 한 원인을 확정하지 않는다.

## 모집단과 표본

기존 fixed pilot900 중 저장된 oracle의 GT IoU가 greedy보다 높고, 최초 oracle 분기 token이 원래 top30에 있는 query에서 기존 극단 사례 3개를 제외했다. 이 조건부 모집단은 **805 query**(Charades 277, ActivityNet 266, QVHighlights 262)다. Dataset별로 SHA256(`oracle_switch_controls|42|`+sample_key)가 작은 20개씩, 총 60개를 사전에 고정했다. 기존 3개를 60개에 합산하지 않는다. 결과는 이 조건부 모집단을 대상으로 하며 전체 benchmark·oracle 개선이 없는 query·다른 모델·자연스러운 영상 수정으로 일반화하지 않는다.

## 같은 변경량에서 비교하는 조건

주 비교 예산은 ε₂₅₅=0.25, 1이며 전체 영상의 같은 변경 범위를 사용한다. 모든 결과 행은 변경된 입력에서 **응답 처음부터 같은 greedy decoding**으로 생성한다. Targeted gradient, 반대 방향, random seed42·43·44, 반복 gradient, targeted 후 8-bit roundtrip을 비교한다. 고정 prefix의 다음 token을 평가하는 probe는 별도 진단이며 decoding 방식이 아니다. 요청 RGB L∞를 맞춰도 clipping 뒤 실제 RGB L∞·RMS가 같다고 가정하지 않는다.

Random의 세 seed는 같은 query의 반복 측정이다. 주 표에는 세 seed의 결과를 query 안에서 평균한 뒤 query를 집계한다. 세 seed가 모두 완료되어야 그 query의 random 조건을 평가 가능으로 센다. 180개의 독립 query로 취급하지 않는다. Query 상세에서 각 seed를 따로 확인할 수 있다.

모델에 넣는 resized floating RGB x∈[0,1]에서 m=z_g−z_oracle의 첫 gradient를 고정하고 targeted 방향은 x′=clip(x−ε·sign(∂m/∂x),0,1)로 만든다. ε₂₅₅=255ε이며 추론 입력을 uint8로 반올림하지 않는다. 같은 source frame의 반복 입력 slot은 같은 변경을 공유한다. 실제 RGB L∞=max|x′−x|, RMS=√mean((x′−x)²)이며 원자료의 physical RGB 분모와 packed normalized input 분모를 구분한다. Cast 전후 변경 비율은 반복 slot을 포함한 전체 packed normalized scalar 중 값이 달라진 비율이다.

## 변경 범위와 관측 경계

Native scan은 전체 영상, gradient 크기로 선택한 frame pair, 선택한 spatial cell에서 각각 하나의 고정 gradient 방향을 시험한다. ε₂₅₅∈{0.25,0.5,1,2,4,8,16,32}에서 처음 관측된 branch 도달 구간을6회 좁히며 upper에서 반복 probe와 실제 greedy 응답을 확인한다. 표의 검증 upper 중앙값은 **검증된 사례만의 중앙값**이다. 검사 완료 중 경계가 없거나 검증되지 않은 경우와 아직 미관측인 경우를 따로 센다. 최대32/255까지 경계를 못 찾았다고 해당 branch가 불가능한 것은 아니다. 이 값은 선택한 gradient ray의 탐색 결과이며 전역 최소 변경량이 아니다.

원인 후보는 네 축으로 읽는다. ①같은 예산의 targeted−random 차이는 시험한 방향의 특이성, ②전체 영상과 pair/cell의 경계는 변경 범위와 누적 효과, ③8-bit roundtrip은 실제 양자화 이후의 효과 유지, ④첫 projection precision은 입력 반올림과 첫 계산 정밀도의 영향을 각각 조사한다. 범위마다 gradient와 clipping도 달라지므로 범위 비교만으로 누적 효과를 유일 원인으로 확정하지 않는다. 이 통제들은 원인 후보를 좁히며 하나의 설명을 자동으로 확정하지 않는다.

## 지표·분모와 불확실성

- **Branch reached**: 실제 응답 전체를 생성한 `free.oracle_branch_reached`. 원래 prefix를 그대로 따르고 해당 위치에서 oracle token을 선택했는지 검사한다. 고정 prefix probe의 top1 전환만으로 성공을 세지 않는다. **분기 token 도달은 저장된 oracle의 전체 경로 재현이나 GT IoU 개선을 뜻하지 않는다.** 이후 token이 달라져 IoU가 낮아질 수 있으므로 실제 생성 구간·GT IoU·ΔIoU를 함께 확인한다.
- **확인 전환률**: 확인된 branch reached 수 / 선정 query 수. Random은 query별 세 seed 성공 비율을 합한다. Baseline 재현 실패는 확인 성공 0으로 포함하되 실패 수를 별도 표시한다. 아직 대기 중인 query를 실패로 분류하지 않는다. 진행 중 값은 현재까지 확인된 전환의 하한이며 완료된 성공률이 아니다.
- **관측 전환률**: 실제 decoding을 평가할 수 있었던 query의 전환률. 실패와 대기를 분모에서 제외하므로 확인 전환률과 함께 읽는다.
- **GT IoU**: length(예측 구간∩GT)/length(예측 구간∪GT). **ΔIoU**는 같은 query의 실제 native baseline greedy IoU를 뺀 값이다. 생성된 응답의 parse 실패가 원자료에서 IoU0으로 정의되어 있으면 그 관측값을 사용한다. 추론 자체가 실패해 응답이 없으면 IoU0을 임의로 채우지 않으며 관측 평균·관측 n·미관측 실패 수를 별도로 표시한다.
- **모집단 가중값**: Σ_d (N_d/805)×dataset d의 평균, N_d∈{277,266,262}. Dataset별 표본 수20의 단순 pooled 평균과 다르다. 확인 전환률의 진행 중 하한은 각 dataset의 선정20 query를 분모로 계산한다. 가중 관측 평균은 세 dataset 모두에 해당 지표의 관측이 있을 때만 표시한다. 가중 관측 IoU는 관측된 응답 평균이며 미관측 실패를 포함한 805개 전체의 결과를 복원하지 않는다.
- **95% Wilson 구간**: binary 조건의 dataset별 확인 전환에 사용한다. n=선정 query 수, p=확인 성공/n, z=1.959964이며 중심=(p+z²/(2n))/(1+z²/n), 반폭=z√(p(1−p)/n+z²/(4n²))/(1+z²/n)이다. 실패를 확인 성공0으로 둔 보수적 프로토콜 지표의 구간이다. 대기 query가 있으면 구간을 계산하지 않는다.
- **Bootstrap 구간**: random query 평균, 모집단 가중 전환률, paired 차이에 사용한다. Dataset별 video cluster를 복원추출하며 같은 영상의 query와 query 안의 모든 seed를 함께 유지한다. 1,000회, seed42, percentile2.5–97.5%다. Targeted−random 및 targeted−opposite는 같은 query 안에서 차이를 구해 재표집한다. IoU 차이는 두 조건 모두 관측된 query를 사용하고 그 n을 명시한다. 미완료 비교의 95% 구간은 표시하지 않는다. 관측값이 모두 같아 empirical bootstrap 구간의 폭이0이어도 모집단 빈도가 확정된 것은 아니다.

## 수치 정밀도 진단

Dataset별 hash 순서 첫 두 query, 총 6개를 결과 전에 지정했다. Native, 첫 Conv3D projection만 FP32로 계산하되 입력을 BF16으로 반올림한 조건, 같은 projection에 FP32 입력을 유지한 조건을 비교한다. 기존 BF16 가중치를 FP32로 올려 사용하며 projection 출력과 이후 모델은 BF16 그대로다. **전체 FP32 모델 실험이 아니다.**

각 mode마다 자체 변형 전 baseline을 구한다. Δoracle margin = r_mode(x′)−r_mode(x), r=z_oracle−max_{v≠oracle}z_v이며 자체 baseline 기준이다. Native와 baseline prefix·선택 token이 일치하는지도 별도 표시한다. Baseline이 다른 mode의 원래 margin을 그대로 같은 기준인 것처럼 비교하지 않는다. 이 6개는 진단 사례이며 정밀도 결과를 805개 전체의 빈도로 해석하지 않는다.

실제 RGB 변경량, native BF16 cast 전후 normalized input 차이, 8-bit roundtrip 뒤 원본 tensor와의 일치를 함께 기록한다. 이 비교는 원인 후보를 좁히기 위한 통제다. 한 조건의 성공이나 실패만으로 의미 이해·수치 정밀도·특정 layer를 유일한 원인으로 단정하지 않는다.

## 변경 범위별 native scan · 선정60 query

| 변경 범위 | 검증 경계 도달/검사 query | 검증 upper ε₂₅₅ 중앙값 | 검사 중 경계 미확인 | 미관측 query |
| --- | --- | --- | --- | --- |
| 전체 영상 | 43/60 | 0.0625 | 17 | 0 |
| 선택 frame pair | 9/60 | 2.4688 | 51 | 0 |
| 선택 spatial cell | 2/60 | 6.2949 | 58 | 0 |

## 공통 예산 ε₂₅₅=0.25

### 모집단 가중 요약

| 입력 변경 조건 | 확정/선정 query | 확인 전환률 | 95% 구간 | 관측 전환률 | 관측 GT IoU | 관측 ΔIoU | IoU 관측 n | 실패 / 대기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Targeted gradient | 60/60 | 49.9% | 36.7%–63.2% | 49.9% | 0.6077 | 0.0675 | 60 | 0 / 0 |
| 반대 방향 | 60/60 | 0.0% | 0.0%–0.0% | 0.0% | 0.4929 | -0.0473 | 60 | 0 / 0 |
| Random · 3 seed 평균 | 60/60 | 1.1% | 0.0%–2.8% | 1.1% | 0.5489 | 0.0087 | 60 | 0 / 0 |
| 반복 gradient | 60/60 | 46.6% | 33.4%–58.4% | 46.6% | 0.6075 | 0.0673 | 60 | 0 / 0 |
| Targeted 후 8-bit roundtrip | 60/60 | 0.0% | 0.0%–0.0% | 0.0% | 0.5402 | 0.0000 | 60 | 0 / 0 |

### Charades · 선정20 query

| 입력 변경 조건 | 확정/선정 query | 확인 전환률 | 95% 구간 | 관측 전환률 | 관측 GT IoU | 관측 ΔIoU | IoU 관측 n | 실패 / 대기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Targeted gradient | 20/20 | 45.0% | 25.8%–65.8% | 45.0% | 0.5473 | 0.0108 | 20 | 0 / 0 |
| 반대 방향 | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.4240 | -0.1126 | 20 | 0 / 0 |
| Random · 3 seed 평균 | 20/20 | 1.7% | 0.0%–5.0% | 1.7% | 0.5451 | 0.0085 | 20 | 0 / 0 |
| 반복 gradient | 20/20 | 40.0% | 21.9%–61.3% | 40.0% | 0.5454 | 0.0088 | 20 | 0 / 0 |
| Targeted 후 8-bit roundtrip | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.5365 | 0.0000 | 20 | 0 / 0 |

### ActivityNet · 선정20 query

| 입력 변경 조건 | 확정/선정 query | 확인 전환률 | 95% 구간 | 관측 전환률 | 관측 GT IoU | 관측 ΔIoU | IoU 관측 n | 실패 / 대기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Targeted gradient | 20/20 | 55.0% | 34.2%–74.2% | 55.0% | 0.5479 | 0.1531 | 20 | 0 / 0 |
| 반대 방향 | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.4047 | 0.0100 | 20 | 0 / 0 |
| Random · 3 seed 평균 | 20/20 | 0.0% | 0.0%–0.0% | 0.0% | 0.4038 | 0.0091 | 20 | 0 / 0 |
| 반복 gradient | 20/20 | 50.0% | 29.9%–70.1% | 50.0% | 0.5458 | 0.1510 | 20 | 0 / 0 |
| Targeted 후 8-bit roundtrip | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.3948 | 0.0000 | 20 | 0 / 0 |

### QVHighlights · 선정20 query

| 입력 변경 조건 | 확정/선정 query | 확인 전환률 | 95% 구간 | 관측 전환률 | 관측 GT IoU | 관측 ΔIoU | IoU 관측 n | 실패 / 대기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Targeted gradient | 20/20 | 50.0% | 29.9%–70.1% | 50.0% | 0.7322 | 0.0405 | 20 | 0 / 0 |
| 반대 방향 | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.6553 | -0.0363 | 20 | 0 / 0 |
| Random · 3 seed 평균 | 20/20 | 1.7% | 0.0%–5.0% | 1.7% | 0.7001 | 0.0085 | 20 | 0 / 0 |
| 반복 gradient | 20/20 | 50.0% | 29.9%–70.1% | 50.0% | 0.7357 | 0.0441 | 20 | 0 / 0 |
| Targeted 후 8-bit roundtrip | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.6916 | 0.0000 | 20 | 0 / 0 |

### 같은 query의 paired 차이

| 범위 | 비교 | 확정 pair/선정 | 전환률 차이 | 95% 구간 | 관측 IoU 차이 | IoU pair n | 95% 구간 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 모집단 가중 | Targeted − Random · 3 seed 평균 | 60/60 | 0.4882 | (0.355, 0.6154) | 0.0588 | 60 | (-0.0296, 0.145) |
| 모집단 가중 | Targeted − 반대 방향 | 60/60 | 0.4993 | (0.3665, 0.6319) | 0.1148 | 60 | (0.0183, 0.2091) |
| Charades | Targeted − Random · 3 seed 평균 | 20/20 | 0.4333 | (0.2, 0.65) | 0.0022 | 20 | (-0.1018, 0.0966) |
| Charades | Targeted − 반대 방향 | 20/20 | 0.4500 | (0.2, 0.65) | 0.1233 | 20 | (-0.011, 0.2511) |
| ActivityNet | Targeted − Random · 3 seed 평균 | 20/20 | 0.5500 | (0.35, 0.75) | 0.1440 | 20 | (-0.0175, 0.2997) |
| ActivityNet | Targeted − 반대 방향 | 20/20 | 0.5500 | (0.35, 0.75) | 0.1432 | 20 | (-0.0215, 0.2982) |
| QVHighlights | Targeted − Random · 3 seed 평균 | 20/20 | 0.4833 | (0.2333, 0.7) | 0.0320 | 20 | (-0.1767, 0.2182) |
| QVHighlights | Targeted − 반대 방향 | 20/20 | 0.5000 | (0.25, 0.7) | 0.0769 | 20 | (-0.1413, 0.2761) |

## 공통 예산 ε₂₅₅=1

### 모집단 가중 요약

| 입력 변경 조건 | 확정/선정 query | 확인 전환률 | 95% 구간 | 관측 전환률 | 관측 GT IoU | 관측 ΔIoU | IoU 관측 n | 실패 / 대기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Targeted gradient | 60/60 | 56.8% | 44.8%–70.1% | 56.8% | 0.5601 | 0.0199 | 60 | 0 / 0 |
| 반대 방향 | 60/60 | 0.0% | 0.0%–0.0% | 0.0% | 0.4671 | -0.0731 | 60 | 0 / 0 |
| Random · 3 seed 평균 | 60/60 | 2.2% | 0.0%–6.0% | 2.2% | 0.5487 | 0.0085 | 60 | 0 / 0 |
| 반복 gradient | 60/60 | 53.5% | 40.4%–66.8% | 53.5% | 0.5741 | 0.0339 | 60 | 0 / 0 |
| Targeted 후 8-bit roundtrip | 60/60 | 58.4% | 45.2%–70.3% | 58.4% | 0.5639 | 0.0237 | 60 | 0 / 0 |

### Charades · 선정20 query

| 입력 변경 조건 | 확정/선정 query | 확인 전환률 | 95% 구간 | 관측 전환률 | 관측 GT IoU | 관측 ΔIoU | IoU 관측 n | 실패 / 대기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Targeted gradient | 20/20 | 65.0% | 43.3%–81.9% | 65.0% | 0.4956 | -0.0409 | 20 | 0 / 0 |
| 반대 방향 | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.3724 | -0.1642 | 20 | 0 / 0 |
| Random · 3 seed 평균 | 20/20 | 1.7% | 0.0%–5.0% | 1.7% | 0.5394 | 0.0028 | 20 | 0 / 0 |
| 반복 gradient | 20/20 | 65.0% | 43.3%–81.9% | 65.0% | 0.5114 | -0.0251 | 20 | 0 / 0 |
| Targeted 후 8-bit roundtrip | 20/20 | 65.0% | 43.3%–81.9% | 65.0% | 0.5028 | -0.0337 | 20 | 0 / 0 |

### ActivityNet · 선정20 query

| 입력 변경 조건 | 확정/선정 query | 확인 전환률 | 95% 구간 | 관측 전환률 | 관측 GT IoU | 관측 ΔIoU | IoU 관측 n | 실패 / 대기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Targeted gradient | 20/20 | 50.0% | 29.9%–70.1% | 50.0% | 0.5110 | 0.1163 | 20 | 0 / 0 |
| 반대 방향 | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.3812 | -0.0136 | 20 | 0 / 0 |
| Random · 3 seed 평균 | 20/20 | 0.0% | 0.0%–0.0% | 0.0% | 0.3761 | -0.0187 | 20 | 0 / 0 |
| 반복 gradient | 20/20 | 45.0% | 25.8%–65.8% | 45.0% | 0.5104 | 0.1156 | 20 | 0 / 0 |
| Targeted 후 8-bit roundtrip | 20/20 | 50.0% | 29.9%–70.1% | 50.0% | 0.5106 | 0.1159 | 20 | 0 / 0 |

### QVHighlights · 선정20 query

| 입력 변경 조건 | 확정/선정 query | 확인 전환률 | 95% 구간 | 관측 전환률 | 관측 GT IoU | 관측 ΔIoU | IoU 관측 n | 실패 / 대기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Targeted gradient | 20/20 | 55.0% | 34.2%–74.2% | 55.0% | 0.6779 | -0.0137 | 20 | 0 / 0 |
| 반대 방향 | 20/20 | 0.0% | 0.0%–16.1% | 0.0% | 0.6545 | -0.0371 | 20 | 0 / 0 |
| Random · 3 seed 평균 | 20/20 | 5.0% | 0.0%–15.0% | 5.0% | 0.7337 | 0.0421 | 20 | 0 / 0 |
| 반복 gradient | 20/20 | 50.0% | 29.9%–70.1% | 50.0% | 0.7051 | 0.0135 | 20 | 0 / 0 |
| Targeted 후 8-bit roundtrip | 20/20 | 60.0% | 38.7%–78.1% | 60.0% | 0.6826 | -0.0090 | 20 | 0 / 0 |

### 같은 query의 paired 차이

| 범위 | 비교 | 확정 pair/선정 | 전환률 차이 | 95% 구간 | 관측 IoU 차이 | IoU pair n | 95% 구간 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 모집단 가중 | Targeted − Random · 3 seed 평균 | 60/60 | 0.5459 | (0.4118, 0.6768) | 0.0114 | 60 | (-0.0856, 0.1042) |
| 모집단 가중 | Targeted − 반대 방향 | 60/60 | 0.5679 | (0.4478, 0.7007) | 0.0930 | 60 | (-0.0176, 0.2002) |
| Charades | Targeted − Random · 3 seed 평균 | 20/20 | 0.6333 | (0.3833, 0.85) | -0.0437 | 20 | (-0.1964, 0.1049) |
| Charades | Targeted − 반대 방향 | 20/20 | 0.6500 | (0.4, 0.85) | 0.1233 | 20 | (-0.0475, 0.3025) |
| ActivityNet | Targeted − Random · 3 seed 평균 | 20/20 | 0.5000 | (0.3, 0.7) | 0.1349 | 20 | (-0.035, 0.2951) |
| ActivityNet | Targeted − 반대 방향 | 20/20 | 0.5000 | (0.3, 0.7) | 0.1299 | 20 | (-0.0642, 0.3095) |
| QVHighlights | Targeted − Random · 3 seed 평균 | 20/20 | 0.5000 | (0.3, 0.7) | -0.0558 | 20 | (-0.2324, 0.1082) |
| QVHighlights | Targeted − 반대 방향 | 20/20 | 0.5500 | (0.35, 0.75) | 0.0235 | 20 | (-0.1827, 0.22) |

## 사전 지정 정밀도 진단 · 6 query

지정 목록: charades-timelens_01558, charades-timelens_01249, activitynet-timelens_03033, activitynet-timelens_04182, qvhighlights-timelens_00561, qvhighlights-timelens_00199

| Query | 계산 조건 | ε₂₅₅ | Native baseline 일치 | 자체 baseline r | 변경 후 r | 자체 기준 Δr | 실제 branch 도달 | GT IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| charades-timelens_01558 | Native BF16 | 0.25 | 예 | -2.0000 | -0.2500 | 1.7500 | 아니오 | 1.0000 |
| charades-timelens_01558 | Native BF16 | 1.00 | 예 | -2.0000 | 0.0000 | 2.0000 | 예 | 0.7895 |
| charades-timelens_01558 | 첫 projection FP32 · BF16 입력 | 0.25 | 예 | -1.8750 | -0.3750 | 1.5000 | 아니오 | 0.7500 |
| charades-timelens_01558 | 첫 projection FP32 · BF16 입력 | 1.00 | 예 | -1.8750 | -0.2500 | 1.6250 | 아니오 | 0.7500 |
| charades-timelens_01558 | 첫 projection FP32 · FP32 입력 | 0.25 | 예 | -1.8750 | -0.2500 | 1.6250 | 아니오 | 1.0000 |
| charades-timelens_01558 | 첫 projection FP32 · FP32 입력 | 1.00 | 예 | -1.8750 | -0.1250 | 1.7500 | 아니오 | 0.7500 |
| charades-timelens_01249 | Native BF16 | 0.25 | 예 | -1.5000 | 0.7500 | 2.2500 | 예 | 0.0000 |
| charades-timelens_01249 | Native BF16 | 1.00 | 예 | -1.5000 | 1.5000 | 3.0000 | 예 | 0.0000 |
| charades-timelens_01249 | 첫 projection FP32 · BF16 입력 | 0.25 | 예 | -1.6250 | 0.8750 | 2.5000 | 예 | 0.0000 |
| charades-timelens_01249 | 첫 projection FP32 · BF16 입력 | 1.00 | 예 | -1.6250 | 1.2500 | 2.8750 | 예 | 0.0000 |
| charades-timelens_01249 | 첫 projection FP32 · FP32 입력 | 0.25 | 예 | -1.5000 | 0.7500 | 2.2500 | 예 | 0.0000 |
| charades-timelens_01249 | 첫 projection FP32 · FP32 입력 | 1.00 | 예 | -1.5000 | 1.5000 | 3.0000 | 예 | 0.0000 |
| activitynet-timelens_03033 | Native BF16 | 0.25 | 예 | -0.2500 | 0.0000 | 0.2500 | 예 | 0.8571 |
| activitynet-timelens_03033 | Native BF16 | 1.00 | 예 | -0.2500 | 0.1250 | 0.3750 | 예 | 0.8571 |
| activitynet-timelens_03033 | 첫 projection FP32 · BF16 입력 | 0.25 | 예 | -0.3750 | 0.0000 | 0.3750 | 예 | 0.8571 |
| activitynet-timelens_03033 | 첫 projection FP32 · BF16 입력 | 1.00 | 예 | -0.3750 | 0.1250 | 0.5000 | 예 | 0.8571 |
| activitynet-timelens_03033 | 첫 projection FP32 · FP32 입력 | 0.25 | 예 | -0.2500 | 0.0000 | 0.2500 | 예 | 0.8571 |
| activitynet-timelens_03033 | 첫 projection FP32 · FP32 입력 | 1.00 | 예 | -0.2500 | 0.1250 | 0.3750 | 예 | 0.8571 |
| activitynet-timelens_04182 | Native BF16 | 0.25 | 예 | -0.1250 | 0.2500 | 0.3750 | 예 | 0.4105 |
| activitynet-timelens_04182 | Native BF16 | 1.00 | 예 | -0.1250 | 0.2500 | 0.3750 | 아니오 | 0.0000 |
| activitynet-timelens_04182 | 첫 projection FP32 · BF16 입력 | 0.25 | 아니오 | -0.1250 | 0.1250 | 0.2500 | 예 | 0.4105 |
| activitynet-timelens_04182 | 첫 projection FP32 · BF16 입력 | 1.00 | 아니오 | -0.1250 | 0.2500 | 0.3750 | 아니오 | 0.0000 |
| activitynet-timelens_04182 | 첫 projection FP32 · FP32 입력 | 0.25 | 아니오 | -0.1250 | 0.1250 | 0.2500 | 예 | 0.4105 |
| activitynet-timelens_04182 | 첫 projection FP32 · FP32 입력 | 1.00 | 아니오 | -0.1250 | 0.2500 | 0.3750 | 아니오 | 0.0000 |
| qvhighlights-timelens_00561 | Native BF16 | 0.25 | 예 | -2.7500 | -1.3750 | 1.3750 | 아니오 | 0.8824 |
| qvhighlights-timelens_00561 | Native BF16 | 1.00 | 예 | -2.7500 | -1.3750 | 1.3750 | 아니오 | 0.9375 |
| qvhighlights-timelens_00561 | 첫 projection FP32 · BF16 입력 | 0.25 | 아니오 | -2.6250 | -1.3750 | 1.2500 | 아니오 | 0.9375 |
| qvhighlights-timelens_00561 | 첫 projection FP32 · BF16 입력 | 1.00 | 아니오 | -2.6250 | -1.3750 | 1.2500 | 아니오 | 0.9375 |
| qvhighlights-timelens_00561 | 첫 projection FP32 · FP32 입력 | 0.25 | 아니오 | -2.7500 | -1.3750 | 1.3750 | 아니오 | 0.9375 |
| qvhighlights-timelens_00561 | 첫 projection FP32 · FP32 입력 | 1.00 | 아니오 | -2.7500 | -1.3750 | 1.3750 | 아니오 | 0.9375 |
| qvhighlights-timelens_00199 | Native BF16 | 0.25 | 예 | -2.1250 | -2.1250 | 0.0000 | 아니오 | 0.9297 |
| qvhighlights-timelens_00199 | Native BF16 | 1.00 | 예 | -2.1250 | -3.1250 | -1.0000 | 아니오 | 0.9297 |
| qvhighlights-timelens_00199 | 첫 projection FP32 · BF16 입력 | 0.25 | 예 | -2.1250 | -2.1250 | 0.0000 | 아니오 | 0.9198 |
| qvhighlights-timelens_00199 | 첫 projection FP32 · BF16 입력 | 1.00 | 예 | -2.1250 | -3.1250 | -1.0000 | 아니오 | 0.9297 |
| qvhighlights-timelens_00199 | 첫 projection FP32 · FP32 입력 | 0.25 | 예 | -2.1250 | -2.1250 | 0.0000 | 아니오 | 0.9202 |
| qvhighlights-timelens_00199 | 첫 projection FP32 · FP32 입력 | 1.00 | 예 | -2.1250 | -3.0000 | -0.8750 | 아니오 | 0.9351 |

## 원자료와 상태

- [표본·805개 조건부 모집단·hash 선택](../../../../../artifacts/runs/rollout/oracle_switch_controls/pilot/60/metadata/selection.json)
- [실행 설정](../../../../../artifacts/runs/rollout/oracle_switch_controls/pilot/60/metadata/config.json)
- [진행·실패 상태](../../../../../artifacts/runs/rollout/oracle_switch_controls/pilot/60/metadata/status.json)
- [기존 3사례와 PNG 비교](../../../oracle_switch/viewer.html)

Query별 원본 참조·control 기록·정밀도 원자료는 결과 화면에서 선택해 연다. 기존 3개 query를 복제하거나 60개와 독립 합산하지 않는다.
