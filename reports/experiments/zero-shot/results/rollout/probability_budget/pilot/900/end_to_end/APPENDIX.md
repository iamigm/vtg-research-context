# Appendix · End-to-end completion · pilot900

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:3:begin -->
## 독립 구간 검증·semantic의 추가 비용

서로 다른 숫자 구간 수와 실제 고유 crop 수를 구분한다. Batch validation의 반복 비용도 아래 합계에 포함한다. 통계용 semantic 계산을 final 선택 비용으로 숨기지 않는다.

구간 합 65935개, 고유 crop 합 63140개.

| 단계 | 실제 forward 합 | 실제 입력 token 합 | Padded token 합 | CUDA 합 s | Model wall 합 s |
|---|---:|---:|---:|---:|---:|
| verifier | 63141 | 194150054 | 194150534 | 14853.863 | 15319.517 |
| instruct | 14521 | 157025194 | 160388650 | 10139.652 | 10619.125 |
| embedding | 14522 | 157088335 | 160451458 | 10088.124 | 10568.291 |

완료 query의 후속 처리 wall 합은 65409.398s (18.169시간), query당 평균은 119.360s다. 이는 세 방법×세 예산과 semantic 진단을 공유 실행한 전체 작업량이며 단일 조건의 단독 latency가 아니다. 이 중 semantic 모델 forward wall 합은 21187.416s (5.885시간), 고유 final generation wall 합은 5331.436s (1.481시간)다. CPU 입력 준비·공유 prefill 등은 각각의 model forward wall과 구분한다. 중단 전에 저장되지 못한 작업의 비용은 아래 재개 기록의 제한을 따른다.


<!-- report-part:3:end -->
<!-- report-part:4:begin -->
## 시간 그룹과 semantic 차이

Cosine은 같은 모델의 벡터끼리 비교한다. 1에 가까울수록 유사하며, 높은 유사도가 같은 사건의 확정 근거는 아니다. 아래 within−between은 시간 그룹 내부 mean cosine에서 그룹 사이 mean cosine을 뺀 값이다. 양수는 이 embedding이 시간 그룹의 차이와 같은 방향으로 구분한다는 진단이며 GT 검증 점수가 아니다. 그룹이 하나뿐이면 between은 정의되지 않는다.

| 방법 | 예산 | Pooling | 관측 pair 수 합 | 평균 cosine (유효query n) | 평균 within−between (유효query n) | 평균 시간 그룹 수 |
|---|---:|---|---:|---:|---:|---:|
| levin | 512 | instruct_visual_mean | 388346 | 0.9834 (n=541) | 0.0395 (n=338) | 2.61 |
| levin | 512 | instruct_last | 388346 | 0.9907 (n=541) | 0.0161 (n=338) | 2.61 |
| levin | 512 | embedding_last | 388346 | 0.9077 (n=541) | 0.1896 (n=338) | 2.61 |
| levin | 1024 | instruct_visual_mean | 1322428 | 0.9829 (n=545) | 0.0373 (n=357) | 3.25 |
| levin | 1024 | instruct_last | 1322428 | 0.9903 (n=545) | 0.0154 (n=357) | 3.25 |
| levin | 1024 | embedding_last | 1322428 | 0.9030 (n=545) | 0.1854 (n=357) | 3.25 |
| levin | 2048 | instruct_visual_mean | 4211175 | 0.9826 (n=547) | 0.0362 (n=360) | 3.89 |
| levin | 2048 | instruct_last | 4211175 | 0.9901 (n=547) | 0.0151 (n=360) | 3.89 |
| levin | 2048 | embedding_last | 4211175 | 0.9000 (n=547) | 0.1829 (n=360) | 3.89 |
| native_mass | 512 | instruct_visual_mean | 788096 | 0.9827 (n=548) | 0.0370 (n=372) | 4.08 |
| native_mass | 512 | instruct_last | 788096 | 0.9904 (n=548) | 0.0152 (n=372) | 4.08 |
| native_mass | 512 | embedding_last | 788096 | 0.9029 (n=548) | 0.1847 (n=372) | 4.08 |
| native_mass | 1024 | instruct_visual_mean | 2462021 | 0.9826 (n=548) | 0.0365 (n=371) | 4.79 |
| native_mass | 1024 | instruct_last | 2462021 | 0.9901 (n=548) | 0.0151 (n=371) | 4.79 |
| native_mass | 1024 | embedding_last | 2462021 | 0.9008 (n=548) | 0.1837 (n=371) | 4.79 |
| native_mass | 2048 | instruct_visual_mean | 7365803 | 0.9825 (n=548) | 0.0356 (n=371) | 5.59 |
| native_mass | 2048 | instruct_last | 7365803 | 0.9900 (n=548) | 0.0148 (n=371) | 5.59 |
| native_mass | 2048 | embedding_last | 7365803 | 0.8992 (n=548) | 0.1815 (n=371) | 5.59 |
| numeric_novelty | 512 | instruct_visual_mean | 788497 | 0.9827 (n=548) | 0.0369 (n=372) | 4.08 |
| numeric_novelty | 512 | instruct_last | 788497 | 0.9904 (n=548) | 0.0152 (n=372) | 4.08 |
| numeric_novelty | 512 | embedding_last | 788497 | 0.9029 (n=548) | 0.1846 (n=372) | 4.08 |
| numeric_novelty | 1024 | instruct_visual_mean | 2461706 | 0.9826 (n=548) | 0.0365 (n=371) | 4.77 |
| numeric_novelty | 1024 | instruct_last | 2461706 | 0.9901 (n=548) | 0.0151 (n=371) | 4.77 |
| numeric_novelty | 1024 | embedding_last | 2461706 | 0.9008 (n=548) | 0.1838 (n=371) | 4.77 |
| numeric_novelty | 2048 | instruct_visual_mean | 7362666 | 0.9825 (n=548) | 0.0356 (n=371) | 5.60 |
| numeric_novelty | 2048 | instruct_last | 7362666 | 0.9900 (n=548) | 0.0148 (n=371) | 5.60 |
| numeric_novelty | 2048 | embedding_last | 7362666 | 0.8991 (n=548) | 0.1815 (n=371) | 5.60 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## 중단·재개 기록

완료된 147개 query를 보존하고 activitynet-timelens_01908부터 재개했다. 후보1개인 조건의 빈 pair mask 처리 오류를 수정했으며 과거 다중 후보 통계는 그대로다. 해당 query에서 저장되지 못했던 semantic 계산은 다시 실행했다. 위 비용 합은 checkpoint에 남은 계산 기준이며, 중단 전에 저장되지 못한 semantic 작업의 CUDA 비용까지 복원한 수치가 아니다. 전체 worker 경과 시간에는 중단 전 작업 시간도 포함한다. 이후부터 모델별 scalar checkpoint를 저장해 재개 시 완료된 모델을 반복하지 않는다.


<!-- report-part:5:end -->
