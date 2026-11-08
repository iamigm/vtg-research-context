# Interval distribution · pilot300

전체300 query 중 decoding0개,독립 검증0개 완료. Sampling과 고정 탐색 예산 없이 허용 경로를 추적한다. 완료하지 않은 query를 제외하고 진행 중 결과임을 명시한다.

## 정확도와 최종 답 변화

IoU는 예측/GT 교집합 길이÷합집합 길이,mIoU는 query별 IoU 평균,R@τ는IoU≥τ 비율이다. 높을수록 좋다. Δpp=100×(같은 query의 결과−zero-shot). GT는 사후 채점에만 사용한다. Oracle은 실제 후보 중 사후 최고 IoU이며 모델 성능이 아니다. 확률 질량이 가장 큰 interval은 제안 분포 기준의 단일 답이다.

| 방법 | 공통 query | mIoU | Δpp vs zero-shot | R@.3 | R@.5 | R@.7 | 답 변경 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero_shot | 0 | — | — | — | — | — | — |
| native_mass | 0 | — | — | — | — | — | — |
| event_only | 0 | — | — | — | — | — | 0 |
| event_onset_offset | 0 | — | — | — | — | — | 0 |
| candidate_oracle | 0 | — | — | — | — | — | — |
| verification_pool_oracle | 0 | — | — | — | — | — | — |

## Dataset와 구간 길이별 정확도

아래 모든 값은 해당 칸에 완료된 query만의 값이다. n<10은 작은 표본, n=0은 빈 칸이다. 고정 cohort는 층화 진단 표본이며 benchmark 전체 분포의 무작위 평균이 아니다. Δpp는 같은 칸의 zero-shot 대비 차이다.

### dataset

| 범위 | n | 방법 | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---|---:|---|---:|---:|---:|---:|
### 영상 길이 s

| 범위 | n | 방법 | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---|---:|---|---:|---:|---:|---:|
| [0, 30) · 빈 칸 | 0 | zero_shot | — | — | — | — |
| [0, 30) · 빈 칸 | 0 | native_mass | — | — | — | — |
| [0, 30) · 빈 칸 | 0 | event_only | — | — | — | — |
| [0, 30) · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| [0, 30) · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
| [30, 60) · 빈 칸 | 0 | zero_shot | — | — | — | — |
| [30, 60) · 빈 칸 | 0 | native_mass | — | — | — | — |
| [30, 60) · 빈 칸 | 0 | event_only | — | — | — | — |
| [30, 60) · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| [30, 60) · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
| [60, 120) · 빈 칸 | 0 | zero_shot | — | — | — | — |
| [60, 120) · 빈 칸 | 0 | native_mass | — | — | — | — |
| [60, 120) · 빈 칸 | 0 | event_only | — | — | — | — |
| [60, 120) · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| [60, 120) · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
| [120, 180) · 빈 칸 | 0 | zero_shot | — | — | — | — |
| [120, 180) · 빈 칸 | 0 | native_mass | — | — | — | — |
| [120, 180) · 빈 칸 | 0 | event_only | — | — | — | — |
| [120, 180) · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| [120, 180) · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
| ≥180 · 빈 칸 | 0 | zero_shot | — | — | — | — |
| ≥180 · 빈 칸 | 0 | native_mass | — | — | — | — |
| ≥180 · 빈 칸 | 0 | event_only | — | — | — | — |
| ≥180 · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| ≥180 · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
### GT/영상 길이 비율

| 범위 | n | 방법 | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---|---:|---|---:|---:|---:|---:|
| [0, 0.05) · 빈 칸 | 0 | zero_shot | — | — | — | — |
| [0, 0.05) · 빈 칸 | 0 | native_mass | — | — | — | — |
| [0, 0.05) · 빈 칸 | 0 | event_only | — | — | — | — |
| [0, 0.05) · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| [0, 0.05) · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
| [0.05, 0.1) · 빈 칸 | 0 | zero_shot | — | — | — | — |
| [0.05, 0.1) · 빈 칸 | 0 | native_mass | — | — | — | — |
| [0.05, 0.1) · 빈 칸 | 0 | event_only | — | — | — | — |
| [0.05, 0.1) · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| [0.05, 0.1) · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
| [0.1, 0.2) · 빈 칸 | 0 | zero_shot | — | — | — | — |
| [0.1, 0.2) · 빈 칸 | 0 | native_mass | — | — | — | — |
| [0.1, 0.2) · 빈 칸 | 0 | event_only | — | — | — | — |
| [0.1, 0.2) · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| [0.1, 0.2) · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
| [0.2, 0.4) · 빈 칸 | 0 | zero_shot | — | — | — | — |
| [0.2, 0.4) · 빈 칸 | 0 | native_mass | — | — | — | — |
| [0.2, 0.4) · 빈 칸 | 0 | event_only | — | — | — | — |
| [0.2, 0.4) · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| [0.2, 0.4) · 빈 칸 | 0 | candidate_oracle | — | — | — | — |
| ≥0.4 · 빈 칸 | 0 | zero_shot | — | — | — | — |
| ≥0.4 · 빈 칸 | 0 | native_mass | — | — | — | — |
| ≥0.4 · 빈 칸 | 0 | event_only | — | — | — | — |
| ≥0.4 · 빈 칸 | 0 | event_onset_offset | — | — | — | — |
| ≥0.4 · 빈 칸 | 0 | candidate_oracle | — | — | — | — |

### 영상 길이 × GT 비율 · miou

각 칸은 n; zero-shot / native_mass / event_only / event_onset_offset / candidate_oracle 순서이며, 각 값 뒤 괄호는 zero-shot 대비 Δpp다. 비율 마지막 칸은[0.4,1]을 포함한다.

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [30, 60) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [60, 120) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [120, 180) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| ≥180 | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |

### 영상 길이 × GT 비율 · r3

각 칸은 n; zero-shot / native_mass / event_only / event_onset_offset / candidate_oracle 순서이며, 각 값 뒤 괄호는 zero-shot 대비 Δpp다. 비율 마지막 칸은[0.4,1]을 포함한다.

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [30, 60) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [60, 120) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [120, 180) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| ≥180 | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |

### 영상 길이 × GT 비율 · r5

각 칸은 n; zero-shot / native_mass / event_only / event_onset_offset / candidate_oracle 순서이며, 각 값 뒤 괄호는 zero-shot 대비 Δpp다. 비율 마지막 칸은[0.4,1]을 포함한다.

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [30, 60) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [60, 120) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [120, 180) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| ≥180 | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |

### 영상 길이 × GT 비율 · r7

각 칸은 n; zero-shot / native_mass / event_only / event_onset_offset / candidate_oracle 순서이며, 각 값 뒤 괄호는 zero-shot 대비 Δpp다. 비율 마지막 칸은[0.4,1]을 포함한다.

| 영상 길이 s | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.4) | ≥0.4 |
|---|---|---|---|---|---|
| [0, 30) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [30, 60) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [60, 120) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| [120, 180) | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |
| ≥180 | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — | n=0 빈 칸; — |

## 다시 본 뒤 실제로 달라진 답

구간 좌표가 baseline과 달라진 query를 답 변경으로 센다. 개선/악화는 같은 query의 IoU가 baseline보다 커짐/작아짐이며, 동일 IoU의 좌표 변경도 있을 수 있다.

| 검증 방식 | 답 변경 | IoU 개선 | IoU 악화 | IoU 동일 |
|---|---:|---:|---:|---:|
| event_only | 0 | 0 | 0 | 0 |
| event_onset_offset | 0 | 0 | 0 | 0 |

## Decoding 분포

Interval entropy는 exact clipped 구간의 조건부 분포에서−Σp log₂p,분산은 실제 시작·끝 시간의 covariance다. 필터 제외 질량과 무효·미완성 질량을 별도로 남긴다.

| Query | Nodes | 유효 interval | Entropy bits | 시작 variance s² | 끝 variance s² | 원래 유효 질량 |
|---|---:|---:|---:|---:|---:|---:|

## 방법별 연산 오버헤드

아래는 현재 cohort의 실측 누적 비용이다. 공유 영상 입력의 재사용을 포함한 운영 비용이며 독립 실행시간으로 해석하지 않는다. Instruct 세 pooling은 한 번의 encoding을 공유한다.

| 단계 | Native forwards | Decoder inputs | CUDA stream s | Wall s |
|---|---:|---:|---:|---:|
| decoding (prefill wall 별도) | 0 | 0 | 0.000 | 0.000 |
| decoding video/query prefill | 0 | 0 | — | 0.000 |
| decoding input preparation | 0 | 0 | — | 0.000 |
| instruct semantic encoding | 0 | 0 | 0.000 | 0.000 |
| instruct input preparation | 0 | 0 | — | 0.000 |
| embedding semantic encoding | 0 | 0 | 0.000 | 0.000 |
| embedding input preparation | 0 | 0 | — | 0.000 |
| independent verifier event_only | 0 | 0 | 0.000 | 0.000 |
| independent verifier event_onset_offset | 0 | 0 | 0.000 | 0.000 |
| model loading (native forward 아님) | 0 | 0 | — | 0.000 |

동일한 decoding 완료 query 0개에서 greedy 기준은 0 decoder inputs / 0 native forwards다. 경로 추적은 0 inputs / 0 forwards로, 각각 —배 / —배다.

Native greedy 기준의 독립 decoder inputs는 동일 영상/query prefill 입력 수 + 첫 token 이후 EOS 예측까지의 추가 greedy token forward 수다. 모델 간 decoder inputs는 FLOPs가 같다는 뜻이 아니다.

| 추가 검증 조건 | 대응 query | 추가 inputs / greedy inputs | 추가 forwards / greedy forwards | 입력 준비 wall s | peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| event_only | 0 | — | — | 0.000 | 0.000 |
| event_onset_offset | 0 | — | — | 0.000 | 0.000 |

두 검증 조건은 공유 prefill을 각각 한 번 포함한 비교값이므로 서로 합산하지 않는다. Event-only 비용은 실제 실행한 공통 prefill+event suffix이고 독립 실행을 별도로 반복 측정한 wall time은 아니다. Semantic은 위 공유 encoding 비용과 pooling 추가 시간을 함께 비교한다.


## Semantic 통계

각 조건의 unit vector 가중 평균μ에서 dispersion=1−||μ||²,그룹 사이 평균 cosine=μ_A·μ_B다. 원래 구간 확률로 가중하고 영상 입력이 없는 질량은 coverage로 따로 표시한다. 전체 vector는 영구 저장하지 않는다.

| 조건 | Query | 평균 semantic coverage | 평균 group dispersion |
|---|---:|---:|---:|
| instruct_mean | 0 | — | — |
| instruct_temporal_four | 0 | — | — |
| instruct_readout | 0 | — | — |
| embedding_readout | 0 | — | — |

[실행 명세와 지표 정의](SPEC.md)

고정300 ⊂ 900 cohort이며 중복 query를 독립 합산하지 않는다. 첫300의 보고서·viewer·검증이 통과한 뒤에만 나머지600을 실행한다.
