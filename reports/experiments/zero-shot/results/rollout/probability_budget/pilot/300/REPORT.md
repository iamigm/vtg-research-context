<!-- report-appendix:APPENDIX.md parts=15 -->
<!-- report-part:0:begin -->
# Probability budget · pilot300

300 query 중 후보 생성 세 방법을 모두 완료한 query 300개. 고유 영상 292개. 이 수치는 후보 생성 단계의 완료 수다. 독립 crop 검증·semantic 측정·최종 답변 생성의 완료 여부는 End-to-end completion 하위 분석에서 확인한다. 조건 반복 수와 query 수를 구분한다. 완료 전 수치는 진행 중 공통 부분집합이며 전체 성능으로 해석하지 않는다.

[방법·예산별 추가 연산](APPENDIX_COMPUTE.md)

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## 후보 품질과 greedy 대비 차이

IoU=예측·GT 교집합 길이÷합집합 길이,mIoU=query별 IoU 평균,R@τ=IoU≥τ 비율이다. 높을수록 좋다. 괄호 안Δpp=100×(결과−같은 query의 greedy). greedy는 이 zero-shot 실험의 기준 답변을 뜻한다. 동일 입력의 native generation이며 clipping·평가 정책도 같다. GT는 탐색 후 평가에만 사용한다.

최대 native-mass 후보는 관측된 sequence 질량을 exact interval별로 합친 뒤 가장 큰 interval을 고른 진단값이다. 독립 verifier가 선택한 final answer가 아니다. Candidate oracle은 생성 후보 중 사후 최고 GT IoU이며 실제 selector 성능이 아니다.

| 예산 | 방법·평가 | 공통 n | mIoU (Δpp) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |
|---:|---|---:|---:|---:|---:|---:|
| — | greedy | 300 | 0.4875 (+0.00) | 0.6433 (+0.00) | 0.5133 (+0.00) | 0.3567 (+0.00) |
| 512 | levin · native_mass | 300 | 0.4985 (+1.10) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 512 | levin · candidate_oracle | 300 | 0.7515 (+26.40) | 0.8933 (+25.00) | 0.8167 (+30.33) | 0.6667 (+31.00) |
| 512 | native_mass · native_mass | 300 | 0.4985 (+1.10) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 512 | native_mass · candidate_oracle | 300 | 0.7904 (+30.29) | 0.9367 (+29.33) | 0.8600 (+34.67) | 0.7133 (+35.67) |
| 512 | numeric_novelty · native_mass | 300 | 0.4985 (+1.10) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 512 | numeric_novelty · candidate_oracle | 300 | 0.7904 (+30.29) | 0.9367 (+29.33) | 0.8600 (+34.67) | 0.7133 (+35.67) |
| 1024 | levin · native_mass | 300 | 0.4986 (+1.11) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 1024 | levin · candidate_oracle | 300 | 0.7837 (+29.62) | 0.9200 (+27.67) | 0.8533 (+34.00) | 0.7067 (+35.00) |
| 1024 | native_mass · native_mass | 300 | 0.4986 (+1.11) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 1024 | native_mass · candidate_oracle | 300 | 0.8120 (+32.44) | 0.9467 (+30.33) | 0.8767 (+36.33) | 0.7467 (+39.00) |
| 1024 | numeric_novelty · native_mass | 300 | 0.4986 (+1.11) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 1024 | numeric_novelty · candidate_oracle | 300 | 0.8120 (+32.44) | 0.9467 (+30.33) | 0.8767 (+36.33) | 0.7467 (+39.00) |
| 2048 | levin · native_mass | 300 | 0.4986 (+1.11) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 2048 | levin · candidate_oracle | 300 | 0.8040 (+31.65) | 0.9300 (+28.67) | 0.8733 (+36.00) | 0.7367 (+38.00) |
| 2048 | native_mass · native_mass | 300 | 0.4986 (+1.11) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 2048 | native_mass · candidate_oracle | 300 | 0.8252 (+33.77) | 0.9533 (+31.00) | 0.8900 (+37.67) | 0.7667 (+41.00) |
| 2048 | numeric_novelty · native_mass | 300 | 0.4986 (+1.11) | 0.6700 (+2.67) | 0.5333 (+2.00) | 0.3567 (+0.00) |
| 2048 | numeric_novelty · candidate_oracle | 300 | 0.8251 (+33.76) | 0.9533 (+31.00) | 0.8900 (+37.67) | 0.7667 (+41.00) |

<!-- report-part:1:end -->
<!-- report-part:2:begin -->
## 실제 연산 비용과1% 달성률

비용은 같은 공통 query의 누적 실측이다. 예산 checkpoint끼리 합산하지 않는다. Decoder inputs는 실제 모델 입력 position 수이며 replay도 포함한다. Prefill·KV 복사·reward 계산은 아래와 별도/함께 표시하므로 forward 수를 FLOPs나 wall time과 동일시하지 않는다.

| 예산 | 방법 | n | 평균 decoder inputs | 평균 search wall s | CUDA decode 합 s | KV 저장·복원 합 s | Reward 계산 합 s | 1% 도달 | 평균 미탐색 상한 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | levin | 300 | 344.1 | 12.326 | 3125.696 | 390.166 | 0.000 | 160 / 300 | 0.134 |
| 512 | native_mass | 300 | 349.2 | 11.435 | 3130.170 | 118.544 | 0.000 | 159 / 300 | 0.153 |
| 512 | numeric_novelty | 300 | 349.2 | 11.569 | 3161.425 | 121.474 | 2.646 | 159 / 300 | 0.153 |
| 1024 | levin | 300 | 540.8 | 19.667 | 4954.804 | 657.380 | 0.000 | 205 / 300 | 0.072 |
| 1024 | native_mass | 300 | 550.6 | 18.122 | 4963.912 | 199.495 | 0.000 | 204 / 300 | 0.081 |
| 1024 | numeric_novelty | 300 | 550.6 | 18.350 | 5017.134 | 203.042 | 6.071 | 204 / 300 | 0.081 |
| 2048 | levin | 300 | 802.8 | 29.489 | 7379.546 | 1037.959 | 0.000 | 239 / 300 | 0.039 |
| 2048 | native_mass | 300 | 815.4 | 27.132 | 7425.822 | 317.562 | 0.000 | 238 / 300 | 0.042 |
| 2048 | numeric_novelty | 300 | 815.4 | 27.432 | 7493.112 | 320.489 | 13.083 | 238 / 300 | 0.042 |

<!-- report-part:2:end -->
<!-- report-part:3:begin -->
## Greedy 대비 오버헤드

배수는같은 query들의 합계 비율이다. Decode 배수는 prefill 이후의 실제 입력 position 비율,전체 inputs 배수는 각 방법에 shared prefill을 한 번 귀속한 값이다. 서로 다른 context 길이에서 같은 입력 position 수가 같은 FLOPs라는 뜻은 아니다.

| 예산 | 방법 | Decode inputs 배수 | Prefill 포함 inputs 배수 | Search+prefill wall 배수 |
|---:|---|---:|---:|---:|
| 512 | levin | 20.326 | 1.030 | 10.951 |
| 512 | native_mass | 20.624 | 1.031 | 10.200 |
| 512 | numeric_novelty | 20.625 | 1.031 | 10.313 |
| 1024 | levin | 31.945 | 1.049 | 17.132 |
| 1024 | native_mass | 32.524 | 1.050 | 15.831 |
| 1024 | numeric_novelty | 32.525 | 1.050 | 16.022 |
| 2048 | levin | 47.418 | 1.073 | 25.401 |
| 2048 | native_mass | 48.166 | 1.074 | 23.417 |
| 2048 | numeric_novelty | 48.165 | 1.074 | 23.669 |

공유 입력 준비 269.194s; video/query prefill 203.899s; greedy decode 152.402s; peak allocated 27.80GiB.

<!-- report-part:3:end -->
<!-- report-part:14:begin -->
## 해석 범위

현재 얻은 interval entropy·variance는 관측 유효 분포에 조건부다. 종료 시 남은 질량·context limit·정책 제외 질량을 함께 확인한다. Candidate oracle 향상이 최대 native-mass 답의 성능 향상을 의미하지 않는다. Novelty는 시간적 차이이며 사건 semantic이나 정답 여부를 판별하지 않는다.

[실행 명세와 지표 정의](SPEC.md)
<!-- report-part:14:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
