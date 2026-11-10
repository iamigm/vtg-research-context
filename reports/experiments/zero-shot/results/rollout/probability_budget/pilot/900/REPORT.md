<!-- report-appendix:APPENDIX.md parts=15 -->
<!-- report-part:0:begin -->
# Probability budget · pilot900

900 query 중 후보 생성 세 방법을 모두 완료한 query 900개. 고유 영상 892개. 이 수치는 후보 생성 단계의 완료 수다. 독립 crop 검증·semantic 측정·최종 답변 생성의 완료 여부는 End-to-end completion 하위 분석에서 확인한다. 조건 반복 수와 query 수를 구분한다. 완료 전 수치는 진행 중 공통 부분집합이며 전체 성능으로 해석하지 않는다.

[방법·예산별 추가 연산](APPENDIX_COMPUTE.md)

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## 후보 품질과 greedy 대비 차이

IoU=예측·GT 교집합 길이÷합집합 길이,mIoU=query별 IoU 평균,R@τ=IoU≥τ 비율이다. 높을수록 좋다. 괄호 안Δpp=100×(결과−같은 query의 greedy). greedy는 이 zero-shot 실험의 기준 답변을 뜻한다. 동일 입력의 native generation이며 clipping·평가 정책도 같다. GT는 탐색 후 평가에만 사용한다.

최대 native-mass 후보는 관측된 sequence 질량을 exact interval별로 합친 뒤 가장 큰 interval을 고른 진단값이다. 독립 verifier가 선택한 final answer가 아니다. Candidate oracle은 생성 후보 중 사후 최고 GT IoU이며 실제 selector 성능이 아니다.

| 예산 | 방법·평가 | 공통 n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---:|---|---:|---:|---:|---:|---:|
| — | greedy | 900 | 0.5195 (+0.00) | 0.6744 (+0.00) | 0.5456 (+0.00) | 0.4022 (+0.00) |
| 512 | levin · native_mass | 900 | 0.5264 (+0.69) | 0.6911 (+1.67) | 0.5567 (+1.11) | 0.4056 (+0.33) |
| 512 | levin · candidate_oracle | 900 | 0.7737 (+25.42) | 0.9011 (+22.67) | 0.8267 (+28.11) | 0.6978 (+29.56) |
| 512 | native_mass · native_mass | 900 | 0.5273 (+0.78) | 0.6922 (+1.78) | 0.5567 (+1.11) | 0.4056 (+0.33) |
| 512 | native_mass · candidate_oracle | 900 | 0.8117 (+29.22) | 0.9389 (+26.44) | 0.8611 (+31.56) | 0.7433 (+34.11) |
| 512 | numeric_novelty · native_mass | 900 | 0.5273 (+0.78) | 0.6922 (+1.78) | 0.5567 (+1.11) | 0.4056 (+0.33) |
| 512 | numeric_novelty · candidate_oracle | 900 | 0.8117 (+29.22) | 0.9389 (+26.44) | 0.8611 (+31.56) | 0.7433 (+34.11) |
| 1024 | levin · native_mass | 900 | 0.5267 (+0.72) | 0.6911 (+1.67) | 0.5567 (+1.11) | 0.4056 (+0.33) |
| 1024 | levin · candidate_oracle | 900 | 0.8015 (+28.20) | 0.9244 (+25.00) | 0.8556 (+31.00) | 0.7311 (+32.89) |
| 1024 | native_mass · native_mass | 900 | 0.5272 (+0.76) | 0.6922 (+1.78) | 0.5578 (+1.22) | 0.4056 (+0.33) |
| 1024 | native_mass · candidate_oracle | 900 | 0.8299 (+31.03) | 0.9544 (+28.00) | 0.8789 (+33.33) | 0.7667 (+36.44) |
| 1024 | numeric_novelty · native_mass | 900 | 0.5272 (+0.76) | 0.6922 (+1.78) | 0.5578 (+1.22) | 0.4056 (+0.33) |
| 1024 | numeric_novelty · candidate_oracle | 900 | 0.8299 (+31.03) | 0.9544 (+28.00) | 0.8789 (+33.33) | 0.7667 (+36.44) |
| 2048 | levin · native_mass | 900 | 0.5275 (+0.80) | 0.6922 (+1.78) | 0.5578 (+1.22) | 0.4056 (+0.33) |
| 2048 | levin · candidate_oracle | 900 | 0.8246 (+30.50) | 0.9444 (+27.00) | 0.8811 (+33.56) | 0.7578 (+35.56) |
| 2048 | native_mass · native_mass | 900 | 0.5275 (+0.80) | 0.6922 (+1.78) | 0.5578 (+1.22) | 0.4056 (+0.33) |
| 2048 | native_mass · candidate_oracle | 900 | 0.8418 (+32.23) | 0.9600 (+28.56) | 0.8944 (+34.89) | 0.7822 (+38.00) |
| 2048 | numeric_novelty · native_mass | 900 | 0.5275 (+0.80) | 0.6922 (+1.78) | 0.5578 (+1.22) | 0.4056 (+0.33) |
| 2048 | numeric_novelty · candidate_oracle | 900 | 0.8417 (+32.21) | 0.9600 (+28.56) | 0.8944 (+34.89) | 0.7822 (+38.00) |

<!-- report-part:1:end -->
<!-- report-part:2:begin -->
## 실제 연산 비용과1% 달성률

비용은 같은 공통 query의 누적 실측이다. 예산 checkpoint끼리 합산하지 않는다. Decoder inputs는 실제 모델 입력 position 수이며 replay도 포함한다. Prefill·KV 복사·reward 계산은 아래와 별도/함께 표시하므로 forward 수를 FLOPs나 wall time과 동일시하지 않는다.

| 예산 | 방법 | n | 평균 decoder inputs | 평균 search wall s | CUDA decode 합 s | KV 저장·복원 합 s | Reward 계산 합 s | 1% 도달 | 평균 미탐색 상한 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | levin | 900 | 338.2 | 14.683 | 11151.349 | 1276.152 | 0.000 | 493 / 900 | 0.142 |
| 512 | native_mass | 900 | 344.0 | 13.679 | 11176.440 | 406.473 | 0.000 | 482 / 900 | 0.159 |
| 512 | numeric_novelty | 900 | 344.0 | 13.715 | 11199.566 | 401.176 | 8.611 | 482 / 900 | 0.159 |
| 1024 | levin | 900 | 530.2 | 23.358 | 17614.940 | 2166.942 | 0.000 | 616 / 900 | 0.083 |
| 1024 | native_mass | 900 | 540.3 | 21.558 | 17624.930 | 673.370 | 0.000 | 613 / 900 | 0.090 |
| 1024 | numeric_novelty | 900 | 540.3 | 21.608 | 17649.049 | 665.573 | 19.013 | 613 / 900 | 0.090 |
| 2048 | levin | 900 | 794.5 | 35.332 | 26493.924 | 3449.864 | 0.000 | 705 / 900 | 0.049 |
| 2048 | native_mass | 900 | 810.0 | 32.575 | 26605.771 | 1079.017 | 0.000 | 704 / 900 | 0.051 |
| 2048 | numeric_novelty | 900 | 810.1 | 32.564 | 26568.174 | 1058.354 | 41.603 | 704 / 900 | 0.051 |

<!-- report-part:2:end -->
<!-- report-part:3:begin -->
## Greedy 대비 오버헤드

배수는같은 query들의 합계 비율이다. Decode 배수는 prefill 이후의 실제 입력 position 비율,전체 inputs 배수는 각 방법에 shared prefill을 한 번 귀속한 값이다. 서로 다른 context 길이에서 같은 입력 position 수가 같은 FLOPs라는 뜻은 아니다.

| 예산 | 방법 | Decode inputs 배수 | Prefill 포함 inputs 배수 | Search+prefill wall 배수 |
|---:|---|---:|---:|---:|
| 512 | levin | 20.125 | 1.031 | 11.450 |
| 512 | native_mass | 20.467 | 1.032 | 10.704 |
| 512 | numeric_novelty | 20.468 | 1.032 | 10.731 |
| 1024 | levin | 31.549 | 1.050 | 17.886 |
| 1024 | native_mass | 32.147 | 1.051 | 16.550 |
| 1024 | numeric_novelty | 32.150 | 1.051 | 16.587 |
| 2048 | levin | 47.273 | 1.076 | 26.769 |
| 2048 | native_mass | 48.196 | 1.077 | 24.724 |
| 2048 | numeric_novelty | 48.199 | 1.077 | 24.716 |

공유 입력 준비 713.420s; video/query prefill 673.748s; greedy decode 539.301s; peak allocated 28.04GiB.

<!-- report-part:3:end -->
<!-- report-part:14:begin -->
## 해석 범위

현재 얻은 interval entropy·variance는 관측 유효 분포에 조건부다. 종료 시 남은 질량·context limit·정책 제외 질량을 함께 확인한다. Candidate oracle 향상이 최대 native-mass 답의 성능 향상을 의미하지 않는다. Novelty는 시간적 차이이며 사건 semantic이나 정답 여부를 판별하지 않는다.

[실행 명세와 지표 정의](SPEC.md)
<!-- report-part:14:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
