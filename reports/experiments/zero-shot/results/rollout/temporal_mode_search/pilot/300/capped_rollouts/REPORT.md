<!-- report-appendix:APPENDIX.md parts=118 -->
<!-- report-part:0:begin -->
# Temporal mode search · capped rollouts · pilot300

상태: 완료.300 query 중 300 query의 전체 조건 완료. 세 데이터셋별100 query,3방법×2seed×4예산 checkpoint를 독립 query로 합산하지 않는다.

동일 source stream에서 main/tail 각각16·32·64·128회,총32·64·128·256회로 비교한다. 중복·무효 응답을 포함한 시도 수가 예산이며 unique sequence·interval·mode는 별도 집계한다. Main 초기화는최대8회이며 새 prefix마다8회를 요구하지 않는다.

IoU=교집합/합집합,mIoU=query 평균 IoU,R@τ=IoU≥τ 비율(τ=.3/.5/.7). 높을수록 좋으며 실패는0점이다. 같은 query의 seed42·43을 먼저 평균한다. 진행 중에는 두 seed가 완료된 checkpoint만 표에 포함한다. Δ는 동일 clipping의 greedy greedy 대비다. Main oracle/pool oracle은 실제 main 후보/shortlist의 사후 최고 IoU이며 diagnostic oracle은 tail까지 포함한 비운영 상한이다.

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## 예산 비교 · greedy 대비

greedy 기준은 동일 모델·동일 영상 입력·원래 query prompt에서 얻은 greedy 단일 응답이다. 추가 rollout이나 후보 판단 없이 생성했으며 deterministic 원본과 token IDs가 일치함을 실행 시 검증한다. 탐색 결과와 동일하게 [0,D] clipping·유효성 검사·IoU 평가를 적용한다. 모든 비교는 해당 행과 동일한 query 집합을 사용하고 seed42·43을 query 안에서 먼저 평균한다. 각 성능 cell은 greedy → 결과 (Δpp)이며 Δpp=100×(결과−greedy)이다. mIoU와 R@.3/.5/.7은0–1 값이고 양의 Δpp는 개선,음수는 악화다. 진행 중에는 조건별 완료 query 집합이 다를 수 있으므로 행별 query 수와 greedy 기준값을 함께 읽는다.

| 총 rollout 상한 | Main/Tail 상한 | 방법 | query / seed 응답 | mIoU: greedy → 결과 (Δpp) | R@.3: greedy → 결과 (Δpp) | R@.5: greedy → 결과 (Δpp) | R@.7: greedy → 결과 (Δpp) | main oracle |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 16/16 | probability | 300/600 | 0.4875 → 0.4877 (+0.02pp) | 0.6433 → 0.6550 (+1.17pp) | 0.5133 → 0.5067 (-0.67pp) | 0.3567 → 0.3483 (-0.83pp) | 0.6791 |
| 32 | 16/16 | token_entropy | 300/600 | 0.4875 → 0.4929 (+0.54pp) | 0.6433 → 0.6583 (+1.50pp) | 0.5133 → 0.5133 (+0.00pp) | 0.3567 → 0.3667 (+1.00pp) | 0.6754 |
| 32 | 16/16 | temporal_mode | 300/600 | 0.4875 → 0.4887 (+0.12pp) | 0.6433 → 0.6550 (+1.17pp) | 0.5133 → 0.5083 (-0.50pp) | 0.3567 → 0.3500 (-0.67pp) | 0.6791 |
| 64 | 32/32 | probability | 300/600 | 0.4875 → 0.4883 (+0.07pp) | 0.6433 → 0.6617 (+1.83pp) | 0.5133 → 0.5067 (-0.67pp) | 0.3567 → 0.3550 (-0.17pp) | 0.6996 |
| 64 | 32/32 | token_entropy | 300/600 | 0.4875 → 0.4826 (-0.49pp) | 0.6433 → 0.6617 (+1.83pp) | 0.5133 → 0.4950 (-1.83pp) | 0.3567 → 0.3417 (-1.50pp) | 0.7051 |
| 64 | 32/32 | temporal_mode | 300/600 | 0.4875 → 0.4849 (-0.26pp) | 0.6433 → 0.6583 (+1.50pp) | 0.5133 → 0.4950 (-1.83pp) | 0.3567 → 0.3550 (-0.17pp) | 0.7014 |
| 128 | 64/64 | probability | 300/600 | 0.4875 → 0.4791 (-0.84pp) | 0.6433 → 0.6433 (+0.00pp) | 0.5133 → 0.4850 (-2.83pp) | 0.3567 → 0.3533 (-0.33pp) | 0.7281 |
| 128 | 64/64 | token_entropy | 300/600 | 0.4875 → 0.4785 (-0.90pp) | 0.6433 → 0.6367 (-0.67pp) | 0.5133 → 0.4867 (-2.67pp) | 0.3567 → 0.3383 (-1.83pp) | 0.7330 |
| 128 | 64/64 | temporal_mode | 300/600 | 0.4875 → 0.4831 (-0.45pp) | 0.6433 → 0.6467 (+0.33pp) | 0.5133 → 0.4917 (-2.17pp) | 0.3567 → 0.3467 (-1.00pp) | 0.7360 |
| 256 | 128/128 | probability | 300/600 | 0.4875 → 0.4835 (-0.40pp) | 0.6433 → 0.6517 (+0.83pp) | 0.5133 → 0.4950 (-1.83pp) | 0.3567 → 0.3533 (-0.33pp) | 0.7490 |
| 256 | 128/128 | token_entropy | 300/600 | 0.4875 → 0.4868 (-0.07pp) | 0.6433 → 0.6600 (+1.67pp) | 0.5133 → 0.5017 (-1.17pp) | 0.3567 → 0.3500 (-0.67pp) | 0.7538 |
| 256 | 128/128 | temporal_mode | 300/600 | 0.4875 → 0.4876 (+0.00pp) | 0.6433 → 0.6583 (+1.50pp) | 0.5133 → 0.4967 (-1.67pp) | 0.3567 → 0.3600 (+0.33pp) | 0.7569 |

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
| 32 | probability | 0.4877 / +0.02 | 123.0 | 115.7 | 5813.7 | 10735.7 / 16.9 | 16689.4 / 1.97× | 16805.0 / 1.98× | 504.0 / 28.08× | 142114096 | +0.003 | +0.02 / +6052.4 | False |
| 64 | probability | 0.4883 / +0.07 | 217.6 | 198.3 | 7271.9 | 10735.7 / 16.9 | 18242.1 / 2.21× | 18440.4 / 2.23× | 740.2 / 41.18× | 160305916 | +0.009 | +0.05 / +1635.4 | True |
| 128 | probability | 0.4791 / -0.84 | 360.8 | 346.0 | 8014.2 | 10735.7 / 16.9 | 19127.6 / 2.34× | 19473.6 / 2.39× | 1064.0 / 59.09× | 171989108 | -0.096 | -0.91 / +1033.2 | True |
| 256 | probability | 0.4835 / -0.40 | 573.1 | 613.6 | 8356.1 | 10735.7 / 16.9 | 19681.9 / 2.41× | 20295.5 / 2.49× | 1559.2 / 86.39× | 181961824 | -0.042 | +0.44 / +821.9 | True |
| 32 | token_entropy | 0.4929 / +0.54 | 125.6 | 115.6 | 5823.6 | 10735.7 / 16.9 | 16701.9 / 1.97× | 16817.4 / 1.99× | 507.2 / 28.26× | 142172206 | +0.088 | +0.54 / +6064.8 | False |
| 64 | token_entropy | 0.4826 / -0.49 | 230.3 | 198.6 | 7412.1 | 10735.7 / 16.9 | 18395.1 / 2.23× | 18593.7 / 2.25× | 759.3 / 42.25× | 162208943 | -0.062 | -1.02 / +1776.3 | True |
| 128 | token_entropy | 0.4785 / -0.90 | 383.5 | 346.5 | 8080.5 | 10735.7 / 16.9 | 19216.7 / 2.35× | 19563.3 / 2.40× | 1090.3 / 60.56× | 173204175 | -0.103 | -0.41 / +969.5 | True |
| 256 | token_entropy | 0.4868 / -0.07 | 610.8 | 614.1 | 8372.8 | 10735.7 / 16.9 | 19736.3 / 2.41× | 20350.4 / 2.50× | 1598.3 / 88.54× | 182615889 | -0.008 | +0.83 / +787.2 | True |
| 32 | temporal_mode | 0.4887 / +0.12 | 123.2 | 115.7 | 5827.0 | 10735.7 / 16.9 | 16702.9 / 1.97× | 16818.5 / 1.98× | 504.7 / 28.13× | 142307936 | +0.020 | +0.12 / +6065.9 | True |
| 64 | temporal_mode | 0.4849 / -0.26 | 221.5 | 198.3 | 7312.4 | 10735.7 / 16.9 | 18286.5 / 2.21× | 18484.8 / 2.24× | 746.3 / 41.53× | 160875942 | -0.034 | -0.38 / +1666.3 | True |
| 128 | temporal_mode | 0.4831 / -0.45 | 380.6 | 346.2 | 8136.1 | 10735.7 / 16.9 | 19269.4 / 2.35× | 19615.6 / 2.40× | 1089.3 / 60.52× | 173907890 | -0.051 | -0.19 / +1130.8 | True |
| 256 | temporal_mode | 0.4876 / +0.00 | 636.3 | 614.2 | 8445.1 | 10735.7 / 16.9 | 19834.0 / 2.42× | 20448.2 / 2.51× | 1626.8 / 90.16× | 183916881 | +0.000 | +0.45 / +832.6 | True |

<!-- report-part:5:end -->
<!-- report-part:15:begin -->
## 실제 시도·중복·적응 선택

| 총 상한 | 방법 | Main 시도 | Tail 시도 | Main unique sequence | Tail unique sequence | Main adaptive 선택 |
|---:|---|---:|---:|---:|---:|---:|
| 32 | probability | 16.00 | 16.00 | 15.51 | 15.49 | 8.00 |
| 32 | token_entropy | 16.00 | 16.00 | 15.60 | 15.48 | 8.00 |
| 32 | temporal_mode | 16.00 | 16.00 | 15.54 | 15.49 | 8.00 |
| 64 | probability | 32.00 | 32.00 | 29.63 | 30.15 | 22.06 |
| 64 | token_entropy | 32.00 | 32.00 | 30.47 | 30.16 | 22.07 |
| 64 | temporal_mode | 32.00 | 32.00 | 30.09 | 30.16 | 22.06 |
| 128 | probability | 64.00 | 64.00 | 54.43 | 58.28 | 50.23 |
| 128 | token_entropy | 64.00 | 64.00 | 56.73 | 58.33 | 50.26 |
| 128 | temporal_mode | 64.00 | 64.00 | 57.09 | 58.36 | 50.25 |
| 256 | probability | 128.00 | 128.00 | 97.03 | 111.31 | 106.75 |
| 256 | token_entropy | 128.00 | 128.00 | 102.47 | 111.50 | 106.84 |
| 256 | temporal_mode | 128.00 | 128.00 | 106.89 | 111.58 | 106.87 |

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
