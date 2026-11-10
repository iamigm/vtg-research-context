# Appendix · End-to-end completion · pilot900

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:3:begin -->
## 독립 구간 검증·semantic의 추가 비용

서로 다른 숫자 구간 수와 실제 고유 crop 수를 구분한다. Batch validation의 반복 비용도 아래 합계에 포함한다. 통계용 semantic 계산을 final 선택 비용으로 숨기지 않는다.

구간 합 76902개, 고유 crop 합 73487개.

| 단계 | 실제 forward 합 | 실제 입력 token 합 | Padded token 합 | CUDA 합 s | Model wall 합 s |
|---|---:|---:|---:|---:|---:|
| verifier | 73488 | 236771358 | 236771838 | 17512.640 | 18066.520 |
| instruct | 17507 | 191774985 | 195645274 | 11929.976 | 12515.391 |
| embedding | 17508 | 191848473 | 195718429 | 11878.105 | 12464.532 |

완료 query의 후속 처리 wall 합은 77759.460s (21.600시간), query당 평균은 128.105s다. 이는 세 방법×세 예산과 semantic 진단을 공유 실행한 전체 작업량이며 단일 조건의 단독 latency가 아니다. 이 중 semantic 모델 forward wall 합은 24979.924s (6.939시간), 고유 final generation wall 합은 5966.751s (1.657시간)다. CPU 입력 준비·공유 prefill 등은 각각의 model forward wall과 구분한다. 중단 전에 저장되지 못한 작업의 비용은 아래 재개 기록의 제한을 따른다.


<!-- report-part:3:end -->
<!-- report-part:4:begin -->
## 시간 그룹과 semantic 차이

Cosine은 같은 모델의 벡터끼리 비교한다. 1에 가까울수록 유사하며, 높은 유사도가 같은 사건의 확정 근거는 아니다. 아래 within−between은 시간 그룹 내부 mean cosine에서 그룹 사이 mean cosine을 뺀 값이다. 양수는 이 embedding이 시간 그룹의 차이와 같은 방향으로 구분한다는 진단이며 GT 검증 점수가 아니다. 그룹이 하나뿐이면 between은 정의되지 않는다.

| 방법 | 예산 | Pooling | 관측 pair 수 합 | 평균 cosine (유효query n) | 평균 within−between (유효query n) | 평균 시간 그룹 수 |
|---|---:|---|---:|---:|---:|---:|
| levin | 512 | instruct_visual_mean | 433686 | 0.9830 (n=600) | 0.0402 (n=389) | 2.74 |
| levin | 512 | instruct_last | 433686 | 0.9902 (n=600) | 0.0173 (n=389) | 2.74 |
| levin | 512 | embedding_last | 433686 | 0.9057 (n=600) | 0.1963 (n=389) | 2.74 |
| levin | 1024 | instruct_visual_mean | 1549387 | 0.9825 (n=604) | 0.0381 (n=407) | 3.45 |
| levin | 1024 | instruct_last | 1549387 | 0.9898 (n=604) | 0.0165 (n=407) | 3.45 |
| levin | 1024 | embedding_last | 1549387 | 0.9010 (n=604) | 0.1914 (n=407) | 3.45 |
| levin | 2048 | instruct_visual_mean | 5011526 | 0.9823 (n=606) | 0.0370 (n=409) | 4.22 |
| levin | 2048 | instruct_last | 5011526 | 0.9896 (n=606) | 0.0162 (n=409) | 4.22 |
| levin | 2048 | embedding_last | 5011526 | 0.8981 (n=606) | 0.1889 (n=409) | 4.22 |
| native_mass | 512 | instruct_visual_mean | 892820 | 0.9823 (n=607) | 0.0379 (n=422) | 4.28 |
| native_mass | 512 | instruct_last | 892820 | 0.9899 (n=607) | 0.0164 (n=422) | 4.28 |
| native_mass | 512 | embedding_last | 892820 | 0.9011 (n=607) | 0.1915 (n=422) | 4.28 |
| native_mass | 1024 | instruct_visual_mean | 2902894 | 0.9822 (n=607) | 0.0372 (n=422) | 5.12 |
| native_mass | 1024 | instruct_last | 2902894 | 0.9896 (n=607) | 0.0162 (n=422) | 5.12 |
| native_mass | 1024 | embedding_last | 2902894 | 0.8989 (n=607) | 0.1896 (n=422) | 5.12 |
| native_mass | 2048 | instruct_visual_mean | 8867110 | 0.9821 (n=607) | 0.0364 (n=421) | 6.05 |
| native_mass | 2048 | instruct_last | 8867110 | 0.9895 (n=607) | 0.0159 (n=421) | 6.05 |
| native_mass | 2048 | embedding_last | 8867110 | 0.8973 (n=607) | 0.1876 (n=421) | 6.05 |
| numeric_novelty | 512 | instruct_visual_mean | 893209 | 0.9823 (n=607) | 0.0378 (n=422) | 4.29 |
| numeric_novelty | 512 | instruct_last | 893209 | 0.9899 (n=607) | 0.0164 (n=422) | 4.29 |
| numeric_novelty | 512 | embedding_last | 893209 | 0.9011 (n=607) | 0.1914 (n=422) | 4.29 |
| numeric_novelty | 1024 | instruct_visual_mean | 2902851 | 0.9822 (n=607) | 0.0372 (n=422) | 5.11 |
| numeric_novelty | 1024 | instruct_last | 2902851 | 0.9896 (n=607) | 0.0162 (n=422) | 5.11 |
| numeric_novelty | 1024 | embedding_last | 2902851 | 0.8989 (n=607) | 0.1897 (n=422) | 5.11 |
| numeric_novelty | 2048 | instruct_visual_mean | 8864001 | 0.9821 (n=607) | 0.0364 (n=421) | 6.06 |
| numeric_novelty | 2048 | instruct_last | 8864001 | 0.9895 (n=607) | 0.0159 (n=421) | 6.06 |
| numeric_novelty | 2048 | embedding_last | 8864001 | 0.8972 (n=607) | 0.1876 (n=421) | 6.06 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## 중단·재개 기록

완료된 147개 query를 보존하고 activitynet-timelens_01908부터 재개했다. 후보1개인 조건의 빈 pair mask 처리 오류를 수정했으며 과거 다중 후보 통계는 그대로다. 해당 query에서 저장되지 못했던 semantic 계산은 다시 실행했다. 위 비용 합은 checkpoint에 남은 계산 기준이며, 중단 전에 저장되지 못한 semantic 작업의 CUDA 비용까지 복원한 수치가 아니다. 전체 worker 경과 시간에는 중단 전 작업 시간도 포함한다. 이후부터 모델별 scalar checkpoint를 저장해 재개 시 완료된 모델을 반복하지 않는다.


<!-- report-part:5:end -->
