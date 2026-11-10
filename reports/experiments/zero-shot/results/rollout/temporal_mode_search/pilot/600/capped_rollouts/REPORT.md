<!-- report-appendix:APPENDIX.md parts=118 -->
<!-- report-part:0:begin -->
# Temporal mode search · capped rollouts · pilot600

상태: 완료.600 query 중 600 query의 전체 조건 완료. 세 데이터셋별200 query,3방법×2seed×4예산 checkpoint를 독립 query로 합산하지 않는다.

동일 source stream에서 main/tail 각각16·32·64·128회,총32·64·128·256회로 비교한다. 중복·무효 응답을 포함한 시도 수가 예산이며 unique sequence·interval·mode는 별도 집계한다. Main 초기화는최대8회이며 새 prefix마다8회를 요구하지 않는다.

IoU=교집합/합집합,mIoU=query 평균 IoU,R@τ=IoU≥τ 비율(τ=.3/.5/.7). 높을수록 좋으며 실패는0점이다. 같은 query의 seed42·43을 먼저 평균한다. 진행 중에는 두 seed가 완료된 checkpoint만 표에 포함한다. Δ는 동일 clipping의 greedy greedy 대비다. Main oracle/pool oracle은 실제 main 후보/shortlist의 사후 최고 IoU이며 diagnostic oracle은 tail까지 포함한 비운영 상한이다.

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## 예산 비교 · greedy 대비

greedy 기준은 동일 모델·동일 영상 입력·원래 query prompt에서 얻은 greedy 단일 응답이다. 추가 rollout이나 후보 판단 없이 생성했으며 deterministic 원본과 token IDs가 일치함을 실행 시 검증한다. 탐색 결과와 동일하게 [0,D] clipping·유효성 검사·IoU 평가를 적용한다. 모든 비교는 해당 행과 동일한 query 집합을 사용하고 seed42·43을 query 안에서 먼저 평균한다. 각 성능 cell은 greedy → 결과 (Δpp)이며 Δpp=100×(결과−greedy)이다. mIoU와 R@.3/.5/.7은0–1 값이고 양의 Δpp는 개선,음수는 악화다. 진행 중에는 조건별 완료 query 집합이 다를 수 있으므로 행별 query 수와 greedy 기준값을 함께 읽는다.

| 총 rollout 상한 | Main/Tail 상한 | 방법 | query / seed 응답 | mIoU: greedy → 결과 (Δpp) | R@.3: greedy → 결과 (Δpp) | R@.5: greedy → 결과 (Δpp) | R@.7: greedy → 결과 (Δpp) | main oracle |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 16/16 | probability | 600/1200 | 0.5164 → 0.5259 (+0.95pp) | 0.6767 → 0.6967 (+2.00pp) | 0.5433 → 0.5450 (+0.17pp) | 0.3867 → 0.3875 (+0.08pp) | 0.7096 |
| 32 | 16/16 | token_entropy | 600/1200 | 0.5164 → 0.5298 (+1.34pp) | 0.6767 → 0.6983 (+2.17pp) | 0.5433 → 0.5467 (+0.33pp) | 0.3867 → 0.3992 (+1.25pp) | 0.7072 |
| 32 | 16/16 | temporal_mode | 600/1200 | 0.5164 → 0.5263 (+0.99pp) | 0.6767 → 0.6967 (+2.00pp) | 0.5433 → 0.5458 (+0.25pp) | 0.3867 → 0.3883 (+0.17pp) | 0.7097 |
| 64 | 32/32 | probability | 600/1200 | 0.5164 → 0.5250 (+0.87pp) | 0.6767 → 0.7000 (+2.33pp) | 0.5433 → 0.5375 (-0.58pp) | 0.3867 → 0.3942 (+0.75pp) | 0.7310 |
| 64 | 32/32 | token_entropy | 600/1200 | 0.5164 → 0.5238 (+0.74pp) | 0.6767 → 0.7042 (+2.75pp) | 0.5433 → 0.5342 (-0.92pp) | 0.3867 → 0.3917 (+0.50pp) | 0.7344 |
| 64 | 32/32 | temporal_mode | 600/1200 | 0.5164 → 0.5229 (+0.65pp) | 0.6767 → 0.6992 (+2.25pp) | 0.5433 → 0.5300 (-1.33pp) | 0.3867 → 0.3925 (+0.58pp) | 0.7330 |
| 128 | 64/64 | probability | 600/1200 | 0.5164 → 0.5195 (+0.31pp) | 0.6767 → 0.6917 (+1.50pp) | 0.5433 → 0.5292 (-1.42pp) | 0.3867 → 0.3967 (+1.00pp) | 0.7561 |
| 128 | 64/64 | token_entropy | 600/1200 | 0.5164 → 0.5247 (+0.84pp) | 0.6767 → 0.6892 (+1.25pp) | 0.5433 → 0.5342 (-0.92pp) | 0.3867 → 0.3967 (+1.00pp) | 0.7599 |
| 128 | 64/64 | temporal_mode | 600/1200 | 0.5164 → 0.5213 (+0.49pp) | 0.6767 → 0.6900 (+1.33pp) | 0.5433 → 0.5308 (-1.25pp) | 0.3867 → 0.3942 (+0.75pp) | 0.7618 |
| 256 | 128/128 | probability | 600/1200 | 0.5164 → 0.5224 (+0.60pp) | 0.6767 → 0.6967 (+2.00pp) | 0.5433 → 0.5342 (-0.92pp) | 0.3867 → 0.3933 (+0.67pp) | 0.7737 |
| 256 | 128/128 | token_entropy | 600/1200 | 0.5164 → 0.5269 (+1.05pp) | 0.6767 → 0.7025 (+2.58pp) | 0.5433 → 0.5367 (-0.67pp) | 0.3867 → 0.3975 (+1.08pp) | 0.7773 |
| 256 | 128/128 | temporal_mode | 600/1200 | 0.5164 → 0.5267 (+1.03pp) | 0.6767 → 0.7000 (+2.33pp) | 0.5433 → 0.5408 (-0.25pp) | 0.3867 → 0.4000 (+1.33pp) | 0.7807 |

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
| 32 | probability | 0.5259 / +0.95 | 118.4 | 113.0 | 5669.1 | 10364.6 / 16.8 | 16168.8 / 2.02× | 16281.8 / 2.04× | 491.5 / 27.52× | 135919479 | +0.161 | +0.95 / +5900.4 | True |
| 64 | probability | 0.5250 / +0.87 | 208.3 | 194.5 | 7049.9 | 10364.6 / 16.8 | 17639.6 / 2.29× | 17834.0 / 2.32× | 719.6 / 40.24× | 151938467 | +0.116 | -0.08 / +1552.2 | True |
| 128 | probability | 0.5195 / +0.31 | 344.7 | 340.3 | 7768.9 | 10364.6 / 16.8 | 18495.0 / 2.43× | 18835.3 / 2.48× | 1033.3 / 57.66× | 162977862 | +0.037 | -0.55 / +1001.3 | True |
| 256 | probability | 0.5224 / +0.60 | 547.6 | 603.6 | 8147.1 | 10364.6 / 16.8 | 19076.1 / 2.52× | 19679.7 / 2.61× | 1515.8 / 84.36× | 172975233 | +0.065 | +0.29 / +844.4 | True |
| 32 | token_entropy | 0.5298 / +1.34 | 121.0 | 112.9 | 5663.9 | 10364.6 / 16.8 | 16166.3 / 2.03× | 16279.2 / 2.04× | 494.0 / 27.66× | 135694313 | +0.228 | +1.34 / +5897.8 | False |
| 64 | token_entropy | 0.5238 / +0.74 | 220.3 | 194.9 | 7212.9 | 10364.6 / 16.8 | 17814.6 / 2.32× | 18009.4 / 2.35× | 738.5 / 41.30× | 153969221 | +0.097 | -0.60 / +1730.2 | True |
| 128 | token_entropy | 0.5247 / +0.84 | 366.2 | 340.7 | 7881.8 | 10364.6 / 16.8 | 18629.4 / 2.45× | 18970.1 / 2.51× | 1059.9 / 59.15× | 164692766 | +0.097 | +0.09 / +960.7 | True |
| 256 | token_entropy | 0.5269 / +1.05 | 584.6 | 603.9 | 8175.2 | 10364.6 / 16.8 | 19141.2 / 2.52× | 19745.1 / 2.62× | 1554.7 / 86.51× | 173806762 | +0.113 | +0.22 / +775.0 | True |
| 32 | temporal_mode | 0.5263 / +0.99 | 118.7 | 113.0 | 5680.2 | 10364.6 / 16.8 | 16180.3 / 2.02× | 16293.3 / 2.04× | 492.3 / 27.57× | 136072493 | +0.168 | +0.99 / +5911.9 | True |
| 64 | temporal_mode | 0.5229 / +0.65 | 212.5 | 194.5 | 7111.8 | 10364.6 / 16.8 | 17705.7 / 2.30× | 17900.2 / 2.33× | 726.2 / 40.61× | 152806246 | +0.087 | -0.34 / +1606.9 | True |
| 128 | temporal_mode | 0.5213 / +0.49 | 365.1 | 340.5 | 7931.5 | 10364.6 / 16.8 | 18678.0 / 2.46× | 19018.5 / 2.51× | 1060.3 / 59.19× | 165292485 | +0.057 | -0.16 / +1118.3 | True |
| 256 | temporal_mode | 0.5267 / +1.03 | 609.6 | 603.9 | 8239.2 | 10364.6 / 16.8 | 19230.2 / 2.54× | 19834.1 / 2.63× | 1582.3 / 88.08× | 174891677 | +0.109 | +0.53 / +815.6 | True |

<!-- report-part:5:end -->
<!-- report-part:15:begin -->
## 실제 시도·중복·적응 선택

| 총 상한 | 방법 | Main 시도 | Tail 시도 | Main unique sequence | Tail unique sequence | Main adaptive 선택 |
|---:|---|---:|---:|---:|---:|---:|
| 32 | probability | 16.00 | 16.00 | 15.41 | 15.46 | 8.00 |
| 32 | token_entropy | 16.00 | 16.00 | 15.57 | 15.44 | 8.00 |
| 32 | temporal_mode | 16.00 | 16.00 | 15.48 | 15.46 | 8.00 |
| 64 | probability | 32.00 | 32.00 | 29.19 | 29.96 | 22.05 |
| 64 | token_entropy | 32.00 | 32.00 | 30.16 | 29.98 | 22.07 |
| 64 | temporal_mode | 32.00 | 32.00 | 29.78 | 29.98 | 22.06 |
| 128 | probability | 64.00 | 64.00 | 52.87 | 57.51 | 50.26 |
| 128 | token_entropy | 64.00 | 64.00 | 55.30 | 57.55 | 50.33 |
| 128 | temporal_mode | 64.00 | 64.00 | 55.77 | 57.58 | 50.32 |
| 256 | probability | 128.00 | 128.00 | 93.47 | 109.23 | 106.90 |
| 256 | token_entropy | 128.00 | 128.00 | 99.04 | 109.39 | 107.05 |
| 256 | temporal_mode | 128.00 | 128.00 | 103.55 | 109.46 | 107.06 |

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
