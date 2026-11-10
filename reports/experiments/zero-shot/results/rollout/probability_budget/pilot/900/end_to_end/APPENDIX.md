# Appendix · End-to-end completion · pilot900

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:3:begin -->
## 독립 구간 검증·semantic의 추가 비용

서로 다른 숫자 구간 수와 실제 고유 crop 수를 구분한다. Batch validation의 반복 비용도 아래 합계에 포함한다. 통계용 semantic 계산을 final 선택 비용으로 숨기지 않는다.

구간 합 68342개, 고유 crop 합 65417개.

| 단계 | 실제 forward 합 | 실제 입력 token 합 | Padded token 합 | CUDA 합 s | Model wall 합 s |
|---|---:|---:|---:|---:|---:|
| verifier | 65418 | 204241728 | 204242208 | 15493.574 | 15981.228 |
| instruct | 15239 | 165271696 | 168749551 | 10570.773 | 11075.360 |
| embedding | 15240 | 165337114 | 168814636 | 10519.353 | 11024.681 |

완료 query의 후속 처리 wall 합은 68404.138s (19.001시간), query당 평균은 122.369s다. 이는 세 방법×세 예산과 semantic 진단을 공유 실행한 전체 작업량이며 단일 조건의 단독 latency가 아니다. 이 중 semantic 모델 forward wall 합은 22100.042s (6.139시간), 고유 final generation wall 합은 5476.124s (1.521시간)다. CPU 입력 준비·공유 prefill 등은 각각의 model forward wall과 구분한다. 중단 전에 저장되지 못한 작업의 비용은 아래 재개 기록의 제한을 따른다.


<!-- report-part:3:end -->
<!-- report-part:4:begin -->
## 시간 그룹과 semantic 차이

Cosine은 같은 모델의 벡터끼리 비교한다. 1에 가까울수록 유사하며, 높은 유사도가 같은 사건의 확정 근거는 아니다. 아래 within−between은 시간 그룹 내부 mean cosine에서 그룹 사이 mean cosine을 뺀 값이다. 양수는 이 embedding이 시간 그룹의 차이와 같은 방향으로 구분한다는 진단이며 GT 검증 점수가 아니다. 그룹이 하나뿐이면 between은 정의되지 않는다.

| 방법 | 예산 | Pooling | 관측 pair 수 합 | 평균 cosine (유효query n) | 평균 within−between (유효query n) | 평균 시간 그룹 수 |
|---|---:|---|---:|---:|---:|---:|
| levin | 512 | instruct_visual_mean | 397641 | 0.9832 (n=552) | 0.0395 (n=348) | 2.64 |
| levin | 512 | instruct_last | 397641 | 0.9906 (n=552) | 0.0163 (n=348) | 2.64 |
| levin | 512 | embedding_last | 397641 | 0.9064 (n=552) | 0.1921 (n=348) | 2.64 |
| levin | 1024 | instruct_visual_mean | 1375332 | 0.9827 (n=556) | 0.0373 (n=366) | 3.32 |
| levin | 1024 | instruct_last | 1375332 | 0.9902 (n=556) | 0.0156 (n=366) | 3.32 |
| levin | 1024 | embedding_last | 1375332 | 0.9017 (n=556) | 0.1873 (n=366) | 3.32 |
| levin | 2048 | instruct_visual_mean | 4365811 | 0.9825 (n=558) | 0.0362 (n=369) | 4.01 |
| levin | 2048 | instruct_last | 4365811 | 0.9899 (n=558) | 0.0152 (n=369) | 4.01 |
| levin | 2048 | embedding_last | 4365811 | 0.8988 (n=558) | 0.1848 (n=369) | 4.01 |
| native_mass | 512 | instruct_visual_mean | 811665 | 0.9826 (n=559) | 0.0370 (n=381) | 4.17 |
| native_mass | 512 | instruct_last | 811665 | 0.9902 (n=559) | 0.0154 (n=381) | 4.17 |
| native_mass | 512 | embedding_last | 811665 | 0.9017 (n=559) | 0.1866 (n=381) | 4.17 |
| native_mass | 1024 | instruct_visual_mean | 2557733 | 0.9824 (n=559) | 0.0364 (n=381) | 4.90 |
| native_mass | 1024 | instruct_last | 2557733 | 0.9900 (n=559) | 0.0152 (n=381) | 4.90 |
| native_mass | 1024 | embedding_last | 2557733 | 0.8995 (n=559) | 0.1854 (n=381) | 4.90 |
| native_mass | 2048 | instruct_visual_mean | 7670439 | 0.9823 (n=559) | 0.0355 (n=381) | 5.74 |
| native_mass | 2048 | instruct_last | 7670439 | 0.9899 (n=559) | 0.0149 (n=381) | 5.74 |
| native_mass | 2048 | embedding_last | 7670439 | 0.8979 (n=559) | 0.1832 (n=381) | 5.74 |
| numeric_novelty | 512 | instruct_visual_mean | 812002 | 0.9826 (n=559) | 0.0370 (n=381) | 4.17 |
| numeric_novelty | 512 | instruct_last | 812002 | 0.9902 (n=559) | 0.0154 (n=381) | 4.17 |
| numeric_novelty | 512 | embedding_last | 812002 | 0.9017 (n=559) | 0.1866 (n=381) | 4.17 |
| numeric_novelty | 1024 | instruct_visual_mean | 2557418 | 0.9824 (n=559) | 0.0364 (n=381) | 4.89 |
| numeric_novelty | 1024 | instruct_last | 2557418 | 0.9900 (n=559) | 0.0152 (n=381) | 4.89 |
| numeric_novelty | 1024 | embedding_last | 2557418 | 0.8995 (n=559) | 0.1855 (n=381) | 4.89 |
| numeric_novelty | 2048 | instruct_visual_mean | 7667302 | 0.9823 (n=559) | 0.0355 (n=381) | 5.75 |
| numeric_novelty | 2048 | instruct_last | 7667302 | 0.9899 (n=559) | 0.0149 (n=381) | 5.75 |
| numeric_novelty | 2048 | embedding_last | 7667302 | 0.8979 (n=559) | 0.1831 (n=381) | 5.75 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## 중단·재개 기록

완료된 147개 query를 보존하고 activitynet-timelens_01908부터 재개했다. 후보1개인 조건의 빈 pair mask 처리 오류를 수정했으며 과거 다중 후보 통계는 그대로다. 해당 query에서 저장되지 못했던 semantic 계산은 다시 실행했다. 위 비용 합은 checkpoint에 남은 계산 기준이며, 중단 전에 저장되지 못한 semantic 작업의 CUDA 비용까지 복원한 수치가 아니다. 전체 worker 경과 시간에는 중단 전 작업 시간도 포함한다. 이후부터 모델별 scalar checkpoint를 저장해 재개 시 완료된 모델을 반복하지 않는다.


<!-- report-part:5:end -->
