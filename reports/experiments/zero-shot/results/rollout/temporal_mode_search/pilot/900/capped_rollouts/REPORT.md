<!-- report-appendix:APPENDIX.md parts=118 -->
<!-- report-part:0:begin -->
# Temporal mode search · capped rollouts · pilot900

상태: 완료.900 query 중 900 query의 전체 조건 완료. 세 데이터셋별300 query,3방법×2seed×4예산 checkpoint를 독립 query로 합산하지 않는다.

동일 source stream에서 main/tail 각각16·32·64·128회,총32·64·128·256회로 비교한다. 중복·무효 응답을 포함한 시도 수가 예산이며 unique sequence·interval·mode는 별도 집계한다. Main 초기화는최대8회이며 새 prefix마다8회를 요구하지 않는다.

IoU=교집합/합집합,mIoU=query 평균 IoU,R@τ=IoU≥τ 비율(τ=.3/.5/.7). 높을수록 좋으며 실패는0점이다. 같은 query의 seed42·43을 먼저 평균한다. 진행 중에는 두 seed가 완료된 checkpoint만 표에 포함한다. Δ는 동일 clipping의 greedy greedy 대비다. Main oracle/pool oracle은 실제 main 후보/shortlist의 사후 최고 IoU이며 diagnostic oracle은 tail까지 포함한 비운영 상한이다.

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## 예산 비교 · greedy 대비

greedy 기준은 동일 모델·동일 영상 입력·원래 query prompt에서 얻은 greedy 단일 응답이다. 추가 rollout이나 후보 판단 없이 생성했으며 deterministic 원본과 token IDs가 일치함을 실행 시 검증한다. 탐색 결과와 동일하게 [0,D] clipping·유효성 검사·IoU 평가를 적용한다. 모든 비교는 해당 행과 동일한 query 집합을 사용하고 seed42·43을 query 안에서 먼저 평균한다. 각 성능 cell은 greedy → 결과 (Δpp)이며 Δpp=100×(결과−greedy)이다. mIoU와 R@.3/.5/.7은0–1 값이고 양의 Δpp는 개선,음수는 악화다. 진행 중에는 조건별 완료 query 집합이 다를 수 있으므로 행별 query 수와 greedy 기준값을 함께 읽는다.

| 총 rollout 상한 | Main/Tail 상한 | 방법 | query / seed 응답 | mIoU: greedy → 결과 (Δpp) | R@.3: greedy → 결과 (Δpp) | R@.5: greedy → 결과 (Δpp) | R@.7: greedy → 결과 (Δpp) | main oracle |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 16/16 | probability | 900/1800 | 0.5195 → 0.5262 (+0.66pp) | 0.6744 → 0.6922 (+1.78pp) | 0.5456 → 0.5494 (+0.39pp) | 0.4022 → 0.3911 (-1.11pp) | 0.7131 |
| 32 | 16/16 | token_entropy | 900/1800 | 0.5195 → 0.5289 (+0.94pp) | 0.6744 → 0.6928 (+1.83pp) | 0.5456 → 0.5506 (+0.50pp) | 0.4022 → 0.3983 (-0.39pp) | 0.7098 |
| 32 | 16/16 | temporal_mode | 900/1800 | 0.5195 → 0.5266 (+0.71pp) | 0.6744 → 0.6922 (+1.78pp) | 0.5456 → 0.5500 (+0.44pp) | 0.4022 → 0.3917 (-1.06pp) | 0.7132 |
| 64 | 32/32 | probability | 900/1800 | 0.5195 → 0.5255 (+0.60pp) | 0.6744 → 0.6939 (+1.94pp) | 0.5456 → 0.5411 (-0.44pp) | 0.4022 → 0.3950 (-0.72pp) | 0.7366 |
| 64 | 32/32 | token_entropy | 900/1800 | 0.5195 → 0.5243 (+0.48pp) | 0.6744 → 0.6967 (+2.22pp) | 0.5456 → 0.5422 (-0.33pp) | 0.4022 → 0.3911 (-1.11pp) | 0.7397 |
| 64 | 32/32 | temporal_mode | 900/1800 | 0.5195 → 0.5246 (+0.51pp) | 0.6744 → 0.6933 (+1.89pp) | 0.5456 → 0.5378 (-0.78pp) | 0.4022 → 0.3950 (-0.72pp) | 0.7379 |
| 128 | 64/64 | probability | 900/1800 | 0.5195 → 0.5218 (+0.22pp) | 0.6744 → 0.6867 (+1.22pp) | 0.5456 → 0.5389 (-0.67pp) | 0.4022 → 0.3989 (-0.33pp) | 0.7591 |
| 128 | 64/64 | token_entropy | 900/1800 | 0.5195 → 0.5244 (+0.49pp) | 0.6744 → 0.6867 (+1.22pp) | 0.5456 → 0.5400 (-0.56pp) | 0.4022 → 0.3972 (-0.50pp) | 0.7632 |
| 128 | 64/64 | temporal_mode | 900/1800 | 0.5195 → 0.5219 (+0.24pp) | 0.6744 → 0.6833 (+0.89pp) | 0.5456 → 0.5394 (-0.61pp) | 0.4022 → 0.3950 (-0.72pp) | 0.7635 |
| 256 | 128/128 | probability | 900/1800 | 0.5195 → 0.5229 (+0.34pp) | 0.6744 → 0.6867 (+1.22pp) | 0.5456 → 0.5411 (-0.44pp) | 0.4022 → 0.3939 (-0.83pp) | 0.7765 |
| 256 | 128/128 | token_entropy | 900/1800 | 0.5195 → 0.5273 (+0.78pp) | 0.6744 → 0.6950 (+2.06pp) | 0.5456 → 0.5422 (-0.33pp) | 0.4022 → 0.3972 (-0.50pp) | 0.7801 |
| 256 | 128/128 | temporal_mode | 900/1800 | 0.5195 → 0.5265 (+0.70pp) | 0.6744 → 0.6906 (+1.61pp) | 0.5456 → 0.5450 (-0.06pp) | 0.4022 → 0.4017 (-0.06pp) | 0.7826 |

<!-- report-part:1:end -->
<!-- report-part:2:begin -->
## 연산 비용과 성능 대비 비용

비용의 분모는 query·method·seed·예산 한 경우다. seed42·43을 query 안에서 먼저 평균한다. 진행 중 비교는 3방법×4예산×2seed가 모두 완료된 같은 query만 사용한다. 다음 두 비용을 구분한다.

**Independent trace work**는 저장된 token ID·분기 prefix·judge prompt를 따라 fresh video KV와 빈 distribution/prompt cache에서 greedy → main K회 → 해당 K의 judge → tail K회만 실행하는 경우의 연산 횟수를 재구성한다. 다른 method·seed·작은 예산 judge의 cache를 빌려 쓰지 않는다. 영상 prefill은 한 번 포함한다. 이는 시간 실측이나 FLOPs 추정이 아니며 실제 native forward 계측과 대조한다. Tail이 중간 main에 영향을 주지 않는 고정 draw stream을 사용한다.

- `decoder inputs`: native decoder forward에 실제 투입하는 token position 수의 합. 재생한 prefix, judge text prefill·근거 생성, 영상 prefill 뒤의 multimodal decoder positions를 포함한다. sampling 횟수나 출력 token 수와 다르다.
- `forwards`: native model forward 호출 수. 하나의 긴 judge text prefill도1회이므로 token 수와 함께 해석한다.
- `attention pairs`: 각 forward의 causal query-key position 쌍 `q(k−q)+q(q+1)/2`의 합. q는 새 input 길이,k는 전체 KV 길이다. layer/head 곱을 생략한 context 길이 workload proxy이며 정확한 FLOPs가 아니다. vision encoder의 별도 연산은 포함하지 않는다.
- `operational`: video prefill+greedy+main+judge. `research total`: operational+진단용 tail. Tail을 후보로 쓰지 않아도 이번 실험의 총비용에서 제외하지 않는다.
- `greedy ×`: 같은 query의 video prefill+greedy 대비 decoder inputs 또는 forward 호출 수 배율. 원래 입력의 hash와 greedy ID가 동일하다. 입력 prefill과 추가 text 비용도 분리한다.

**Observed execution**은 지금 사용하는 worker3개·공유 cache의 nested sweep에서 각 main/tail/judge 구간의 wall seconds,native forward CUDA stream seconds,forward 수,decoder inputs,process별 peak allocated/reserved memory를 기록한다. CUDA events를 각 forward 전후에 배치하며 GPU 동시 실행·queue 대기의 영향을 받을 수 있다. CUDA stream seconds는 GPU 전체 사용시간이나 정확한 kernel active time이 아니다. wall−forward wall은 Python/cache/mode grouping 등 host-side 잔여시간의 근사다. peak memory는 worker별 model/KV를 포함하고 다른 worker의 점유량과 합산한 장치 전체 peak가 아니다.

Observed main/tail은 해당 K까지의 누적,judge는 해당 K의 구간이다. `sweep`은 이전 모든 작은 예산 judge를 포함한 phase 합계이고,`sweep_elapsed_seconds`는 이전 checkpoint 저장과 phase 밖 Python 처리까지 포함한 경과시간(현재 checkpoint 저장 직전)이다. 독립 K 실행시간으로 부르지 않는다. method 실행 순서·이미 저장된 cache·재개에 따라 관측 비용이 달라지므로 method 공정 비교는 independent trace work를 사용한다. 재개 실행의 재탐색·실패 비용은 누적 실행 로그와 raw actual_cost에 보존한다. 계측 도입 이전 checkpoint의 실측은 결측으로 남기며0초로 처리하지 않는다.

효율 지표는 greedy 대비 `ΔmIoU(pp)=100×mean(IoU−IoU_greedy)`,추가1,000 decoder inputs당 향상 `ΔmIoU(pp)/(mean(extra decoder inputs)/1000)`다. 음수는 비용을 써도 성능이 악화됐다는 뜻이다. `marginal Δ`와 `추가 비용`은 같은 query에서 직전 절반 예산 대비 값이다. 같은 query 집합에서 더 낮은 비용·같거나 높은 mIoU의 다른 조건이 있으면 dominated로 표시한다. 평균 개선만으로 연구적 기여를 단정하지 않으며 비용 배율,실측 시간,seed·dataset별 결과와 불확실성을 함께 본다.


<!-- report-part:2:end -->
<!-- report-part:5:begin -->
### Independent trace work · query당 평균

| 총 rollout | 방법 | mIoU / Δpp | Main inputs | Tail inputs | Judge inputs | Prefill / greedy inputs | Operational inputs / greedy× | Research inputs / greedy× | Research forwards / greedy× | Attention pairs | 향상 pp / 추가1k inputs | Marginal Δpp / inputs | Dominated |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | probability | 0.5262 / +0.66 | 118.9 | 113.2 | 5697.5 | 10225.3 / 16.8 | 16058.6 / 2.04× | 16171.8 / 2.06× | 493.2 / 27.64× | 133849791 | +0.112 | +0.66 / +5929.6 | False |
| 64 | probability | 0.5255 / +0.60 | 209.8 | 194.8 | 7052.0 | 10225.3 / 16.8 | 17504.0 / 2.30× | 17698.7 / 2.33× | 721.5 / 40.37× | 149142516 | +0.080 | -0.07 / +1527.0 | True |
| 128 | probability | 0.5218 / +0.22 | 347.9 | 341.1 | 7782.8 | 10225.3 / 16.8 | 18372.8 / 2.45× | 18713.9 / 2.51× | 1037.8 / 57.96× | 160052607 | +0.026 | -0.37 / +1015.1 | True |
| 256 | probability | 0.5229 / +0.34 | 554.1 | 606.1 | 8145.4 | 10225.3 / 16.8 | 18941.7 / 2.54× | 19547.8 / 2.63× | 1524.7 / 84.94× | 169785935 | +0.036 | +0.11 / +833.9 | True |
| 32 | token_entropy | 0.5289 / +0.94 | 121.6 | 113.2 | 5702.5 | 10225.3 / 16.8 | 16066.3 / 2.04× | 16179.5 / 2.06× | 496.4 / 27.82× | 133665247 | +0.159 | +0.94 / +5937.3 | False |
| 64 | token_entropy | 0.5243 / +0.48 | 221.6 | 195.1 | 7212.3 | 10225.3 / 16.8 | 17676.0 / 2.33× | 17871.2 / 2.36× | 740.1 / 41.42× | 150987164 | +0.063 | -0.46 / +1691.7 | True |
| 128 | token_entropy | 0.5244 / +0.49 | 369.2 | 341.5 | 7883.8 | 10225.3 / 16.8 | 18495.1 / 2.47× | 18836.6 / 2.53× | 1063.9 / 59.42× | 161534724 | +0.057 | +0.01 / +965.5 | True |
| 256 | token_entropy | 0.5273 / +0.78 | 590.6 | 606.3 | 8182.0 | 10225.3 / 16.8 | 19014.7 / 2.55× | 19621.1 / 2.64× | 1563.4 / 87.09× | 170663863 | +0.083 | +0.28 / +784.4 | True |
| 32 | temporal_mode | 0.5266 / +0.71 | 119.3 | 113.2 | 5710.4 | 10225.3 / 16.8 | 16071.8 / 2.04× | 16185.0 / 2.06× | 494.1 / 27.69× | 134006555 | +0.119 | +0.71 / +5942.9 | True |
| 64 | temporal_mode | 0.5246 / +0.51 | 213.8 | 194.8 | 7114.9 | 10225.3 / 16.8 | 17570.8 / 2.31× | 17765.7 / 2.34× | 728.1 / 40.75× | 149915125 | +0.068 | -0.20 / +1580.7 | True |
| 128 | temporal_mode | 0.5219 / +0.24 | 367.6 | 341.2 | 7934.8 | 10225.3 / 16.8 | 18544.6 / 2.48× | 18885.8 / 2.53× | 1063.7 / 59.43× | 162113711 | +0.028 | -0.27 / +1120.1 | True |
| 256 | temporal_mode | 0.5265 / +0.70 | 614.5 | 606.3 | 8235.8 | 10225.3 / 16.8 | 19092.4 / 2.56× | 19698.7 / 2.65× | 1589.4 / 88.56× | 171650093 | +0.074 | +0.46 / +812.9 | True |

<!-- report-part:5:end -->
<!-- report-part:15:begin -->
## 실제 시도·중복·적응 선택

| 총 상한 | 방법 | Main 시도 | Tail 시도 | Main unique sequence | Tail unique sequence | Main adaptive 선택 |
|---:|---|---:|---:|---:|---:|---:|
| 32 | probability | 16.00 | 16.00 | 15.41 | 15.46 | 8.00 |
| 32 | token_entropy | 16.00 | 16.00 | 15.58 | 15.45 | 8.00 |
| 32 | temporal_mode | 16.00 | 16.00 | 15.48 | 15.46 | 8.00 |
| 64 | probability | 32.00 | 32.00 | 29.19 | 30.01 | 22.05 |
| 64 | token_entropy | 32.00 | 32.00 | 30.14 | 30.02 | 22.06 |
| 64 | temporal_mode | 32.00 | 32.00 | 29.77 | 30.02 | 22.06 |
| 128 | probability | 64.00 | 64.00 | 53.11 | 57.71 | 50.24 |
| 128 | token_entropy | 64.00 | 64.00 | 55.45 | 57.74 | 50.32 |
| 128 | temporal_mode | 64.00 | 64.00 | 55.91 | 57.77 | 50.31 |
| 256 | probability | 128.00 | 128.00 | 94.29 | 109.88 | 106.88 |
| 256 | token_entropy | 128.00 | 128.00 | 99.76 | 110.00 | 107.04 |
| 256 | temporal_mode | 128.00 | 128.00 | 104.03 | 110.06 | 107.05 |

<!-- report-part:15:end -->
<!-- report-part:117:begin -->
## 사례 탐색과 mode 비교

사례 필터는 현재 선택한 방법·seed·예산의 query별 결과를 사용한다. 비교 기준은 같은 입력의 greedy 단일 응답이다. IoU는 GT와 예측 interval의 교집합 길이 / 합집합 길이이며 범위는 0–1이고 높을수록 좋다. ΔIoU = 해당 값 − greedy IoU이며 0.01은 1pp다. Main oracle과 judge 후보 oracle은 각각 발견한 모든 main interval과 judge에 전달한 대표 interval의 최고 IoU다. Diagnostic oracle은 tail까지 포함한 최고 IoU다. Oracle은 사후 상한이며 GT는 생성·선택에 사용하지 않는다.

| 특수 기준 | query별 조건 |
|---|---|
| 회복 | greedy IoU < 0.5, 최종 선택 IoU ≥ 0.5 |
| 개선 / 악화 | 최종 선택의 greedy 대비 ΔIoU ≥ 0.01 / ≤ −0.01 |
| 후보 발견 후 선택 실패 | greedy IoU < 0.5, main oracle ≥ 0.5, 최종 선택 IoU < 0.5 |
| 탐색 후에도 미회복 | greedy IoU < 0.5, main oracle < 0.5 |
| 여러 mode | main mode 수 ≥ 2 |
| Tail에만 회복 후보 | greedy IoU < 0.5, main oracle < 0.5, diagnostic oracle ≥ 0.5 |
| 최종 선택 무효 | 유효한 최종 interval이 없음 |

수치 범위는 greedy IoU, greedy 대비 ΔIoU, main mode 수에 적용하며 양 끝을 포함한다. 모든 조건을 동시에 적용한다. Query 수는 현재 cohort의 고유 query 수이며 seed 응답 수나 영상 수가 아니다. 표본별 목록은 canonical 응답을 복제하지 않고 기존 query 지표를 조회한다.

Mode 비교는 저장된 checkpoint의 소속을 그대로 보여준다. 새 interval은 기존 mode의 **모든 interval**에 대해 IoU ≥ 0.5이고 시작·끝 차이가 각각 max(1 s, 0.1 × 두 interval 중 짧은 길이) 이내이면 통합한다. 해당 mode가 여러 개면 endpoint 거리와 mode ID 순서로 결정한다. 조건을 만족하는 mode가 없으면 새 mode다. Mode ID는 해당 방법·seed·예산 안에서만 유효하며 실제 사건 수나 정답 확률을 뜻하지 않는다. 무효 경로는 mode에 넣지 않는다. 같은 sequence의 반복은 관측 횟수로 묶고 각 고유 응답의 mode, interval, IoU와 greedy 대비 ΔIoU를 표시한다. 공통 시간축의 얇은 막대는 각 고유 interval이며 mode 전체를 하나의 합친 interval로 해석하지 않는다.
<!-- case-finder:end -->
<!-- report-part:117:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
