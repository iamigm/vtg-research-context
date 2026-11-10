# Appendix · End-to-end completion · pilot300

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:3:begin -->
## 독립 구간 검증·semantic의 추가 비용

서로 다른 숫자 구간 수와 실제 고유 crop 수를 구분한다. Batch validation의 반복 비용도 아래 합계에 포함한다. 통계용 semantic 계산을 final 선택 비용으로 숨기지 않는다.

구간 합 37938개, 고유 crop 합 36341개.

| 단계 | 실제 forward 합 | 실제 입력 token 합 | Padded token 합 | CUDA 합 s | Model wall 합 s |
|---|---:|---:|---:|---:|---:|
| verifier | 36342 | 94143275 | 94143755 | 8513.977 | 8772.690 |
| instruct | 7524 | 75726320 | 77572454 | 5856.566 | 6090.899 |
| embedding | 7524 | 75762662 | 77608796 | 5818.958 | 6053.733 |

완료 query의 후속 처리 wall 합은 35791.850s (9.942시간), query당 평균은 119.306s다. 이는 세 방법×세 예산과 semantic 진단을 공유 실행한 전체 작업량이며 단일 조건의 단독 latency가 아니다. 이 중 semantic 모델 forward wall 합은 12144.632s (3.374시간), 고유 final generation wall 합은 3599.357s (1.000시간)다. CPU 입력 준비·공유 prefill 등은 각각의 model forward wall과 구분한다. 중단 전에 저장되지 못한 작업의 비용은 아래 재개 기록의 제한을 따른다.


<!-- report-part:3:end -->
<!-- report-part:4:begin -->
## 시간 그룹과 semantic 차이

Cosine은 같은 모델의 벡터끼리 비교한다. 1에 가까울수록 유사하며, 높은 유사도가 같은 사건의 확정 근거는 아니다. 아래 within−between은 시간 그룹 내부 mean cosine에서 그룹 사이 mean cosine을 뺀 값이다. 양수는 이 embedding이 시간 그룹의 차이와 같은 방향으로 구분한다는 진단이며 GT 검증 점수가 아니다. 그룹이 하나뿐이면 between은 정의되지 않는다.

| 방법 | 예산 | Pooling | 관측 pair 수 합 | 평균 cosine (유효query n) | 평균 within−between (유효query n) | 평균 시간 그룹 수 |
|---|---:|---|---:|---:|---:|---:|
| levin | 512 | instruct_visual_mean | 221510 | 0.9829 (n=298) | 0.0402 (n=188) | 2.61 |
| levin | 512 | instruct_last | 221510 | 0.9897 (n=298) | 0.0178 (n=188) | 2.61 |
| levin | 512 | embedding_last | 221510 | 0.9026 (n=298) | 0.2040 (n=188) | 2.61 |
| levin | 1024 | instruct_visual_mean | 807016 | 0.9822 (n=300) | 0.0392 (n=197) | 3.21 |
| levin | 1024 | instruct_last | 807016 | 0.9892 (n=300) | 0.0175 (n=197) | 3.21 |
| levin | 1024 | embedding_last | 807016 | 0.8964 (n=300) | 0.2052 (n=197) | 3.21 |
| levin | 2048 | instruct_visual_mean | 2581558 | 0.9820 (n=300) | 0.0375 (n=198) | 3.75 |
| levin | 2048 | instruct_last | 2581558 | 0.9890 (n=300) | 0.0169 (n=198) | 3.75 |
| levin | 2048 | embedding_last | 2581558 | 0.8936 (n=300) | 0.2006 (n=198) | 3.75 |
| native_mass | 512 | instruct_visual_mean | 446143 | 0.9823 (n=300) | 0.0381 (n=206) | 4.00 |
| native_mass | 512 | instruct_last | 446143 | 0.9894 (n=300) | 0.0171 (n=206) | 4.00 |
| native_mass | 512 | embedding_last | 446143 | 0.8973 (n=300) | 0.2014 (n=206) | 4.00 |
| native_mass | 1024 | instruct_visual_mean | 1456353 | 0.9821 (n=300) | 0.0377 (n=206) | 4.63 |
| native_mass | 1024 | instruct_last | 1456353 | 0.9891 (n=300) | 0.0169 (n=206) | 4.63 |
| native_mass | 1024 | embedding_last | 1456353 | 0.8950 (n=300) | 0.2003 (n=206) | 4.63 |
| native_mass | 2048 | instruct_visual_mean | 4288812 | 0.9819 (n=300) | 0.0367 (n=206) | 5.26 |
| native_mass | 2048 | instruct_last | 4288812 | 0.9890 (n=300) | 0.0166 (n=206) | 5.26 |
| native_mass | 2048 | embedding_last | 4288812 | 0.8925 (n=300) | 0.1978 (n=206) | 5.26 |
| numeric_novelty | 512 | instruct_visual_mean | 446364 | 0.9823 (n=300) | 0.0381 (n=206) | 4.00 |
| numeric_novelty | 512 | instruct_last | 446364 | 0.9894 (n=300) | 0.0171 (n=206) | 4.00 |
| numeric_novelty | 512 | embedding_last | 446364 | 0.8974 (n=300) | 0.2013 (n=206) | 4.00 |
| numeric_novelty | 1024 | instruct_visual_mean | 1456119 | 0.9821 (n=300) | 0.0377 (n=206) | 4.62 |
| numeric_novelty | 1024 | instruct_last | 1456119 | 0.9891 (n=300) | 0.0169 (n=206) | 4.62 |
| numeric_novelty | 1024 | embedding_last | 1456119 | 0.8950 (n=300) | 0.2004 (n=206) | 4.62 |
| numeric_novelty | 2048 | instruct_visual_mean | 4285093 | 0.9819 (n=300) | 0.0367 (n=206) | 5.27 |
| numeric_novelty | 2048 | instruct_last | 4285093 | 0.9890 (n=300) | 0.0166 (n=206) | 5.27 |
| numeric_novelty | 2048 | embedding_last | 4285093 | 0.8925 (n=300) | 0.1978 (n=206) | 5.27 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## 중단·재개 기록

완료된 147개 query를 보존하고 activitynet-timelens_01908부터 재개했다. 후보1개인 조건의 빈 pair mask 처리 오류를 수정했으며 과거 다중 후보 통계는 그대로다. 해당 query에서 저장되지 못했던 semantic 계산은 다시 실행했다. 위 비용 합은 checkpoint에 남은 계산 기준이며, 중단 전에 저장되지 못한 semantic 작업의 CUDA 비용까지 복원한 수치가 아니다. 전체 worker 경과 시간에는 중단 전 작업 시간도 포함한다. 이후부터 모델별 scalar checkpoint를 저장해 재개 시 완료된 모델을 반복하지 않는다.


<!-- report-part:5:end -->
