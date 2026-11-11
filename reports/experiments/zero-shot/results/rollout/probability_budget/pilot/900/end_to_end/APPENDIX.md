# Appendix · End-to-end completion · pilot900

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:3:begin -->
## 독립 구간 검증·semantic의 추가 비용

서로 다른 숫자 구간 수와 실제 고유 crop 수를 구분한다. Batch validation의 반복 비용도 아래 합계에 포함한다. 통계용 semantic 계산을 final 선택 비용으로 숨기지 않는다.

구간 합 114253개, 고유 crop 합 108952개.

| 단계 | 실제 forward 합 | 실제 입력 token 합 | Padded token 합 | CUDA 합 s | Model wall 합 s |
|---|---:|---:|---:|---:|---:|
| verifier | 108953 | 368188257 | 368188737 | 25956.714 | 26796.475 |
| instruct | 26578 | 298625723 | 304438798 | 17616.370 | 18540.587 |
| embedding | 26579 | 298734676 | 304548084 | 17555.893 | 18481.410 |

완료 query의 후속 처리 wall 합은 117729.949s (32.703시간), query당 평균은 130.811s다. 이는 세 방법×세 예산과 semantic 진단을 공유 실행한 전체 작업량이며 단일 조건의 단독 latency가 아니다. 이 중 semantic 모델 forward wall 합은 37021.997s (10.284시간), 고유 final generation wall 합은 8316.838s (2.310시간)다. CPU 입력 준비·공유 prefill 등은 각각의 model forward wall과 구분한다. 중단 전에 저장되지 못한 작업의 비용은 아래 재개 기록의 제한을 따른다.


<!-- report-part:3:end -->
<!-- report-part:4:begin -->
## 시간 그룹과 semantic 차이

Cosine은 같은 모델의 벡터끼리 비교한다. 1에 가까울수록 유사하며, 높은 유사도가 같은 사건의 확정 근거는 아니다. 아래 within−between은 시간 그룹 내부 mean cosine에서 그룹 사이 mean cosine을 뺀 값이다. 양수는 이 embedding이 시간 그룹의 차이와 같은 방향으로 구분한다는 진단이며 GT 검증 점수가 아니다. 그룹이 하나뿐이면 between은 정의되지 않는다.

| 방법 | 예산 | Pooling | 관측 pair 수 합 | 평균 cosine (유효query n) | 평균 within−between (유효query n) | 평균 시간 그룹 수 |
|---|---:|---|---:|---:|---:|---:|
| levin | 512 | instruct_visual_mean | 648497 | 0.9838 (n=887) | 0.0387 (n=580) | 2.73 |
| levin | 512 | instruct_last | 648497 | 0.9905 (n=887) | 0.0165 (n=580) | 2.73 |
| levin | 512 | embedding_last | 648497 | 0.9080 (n=887) | 0.1912 (n=580) | 2.73 |
| levin | 1024 | instruct_visual_mean | 2295339 | 0.9831 (n=895) | 0.0370 (n=608) | 3.43 |
| levin | 1024 | instruct_last | 2295339 | 0.9900 (n=895) | 0.0159 (n=608) | 3.43 |
| levin | 1024 | embedding_last | 2295339 | 0.9033 (n=895) | 0.1855 (n=608) | 3.43 |
| levin | 2048 | instruct_visual_mean | 7362110 | 0.9829 (n=898) | 0.0356 (n=613) | 4.15 |
| levin | 2048 | instruct_last | 7362110 | 0.9898 (n=898) | 0.0155 (n=613) | 4.15 |
| levin | 2048 | embedding_last | 7362110 | 0.9007 (n=898) | 0.1825 (n=613) | 4.15 |
| native_mass | 512 | instruct_visual_mean | 1337050 | 0.9831 (n=899) | 0.0365 (n=632) | 4.29 |
| native_mass | 512 | instruct_last | 1337050 | 0.9901 (n=899) | 0.0159 (n=632) | 4.29 |
| native_mass | 512 | embedding_last | 1337050 | 0.9036 (n=899) | 0.1859 (n=632) | 4.29 |
| native_mass | 1024 | instruct_visual_mean | 4323355 | 0.9829 (n=899) | 0.0359 (n=633) | 5.10 |
| native_mass | 1024 | instruct_last | 4323355 | 0.9899 (n=899) | 0.0157 (n=633) | 5.10 |
| native_mass | 1024 | embedding_last | 4323355 | 0.9012 (n=899) | 0.1843 (n=633) | 5.10 |
| native_mass | 2048 | instruct_visual_mean | 13227762 | 0.9828 (n=899) | 0.0352 (n=632) | 6.02 |
| native_mass | 2048 | instruct_last | 13227762 | 0.9898 (n=899) | 0.0154 (n=632) | 6.02 |
| native_mass | 2048 | embedding_last | 13227762 | 0.8997 (n=899) | 0.1823 (n=632) | 6.02 |
| numeric_novelty | 512 | instruct_visual_mean | 1337861 | 0.9831 (n=899) | 0.0365 (n=632) | 4.29 |
| numeric_novelty | 512 | instruct_last | 1337861 | 0.9901 (n=899) | 0.0158 (n=632) | 4.29 |
| numeric_novelty | 512 | embedding_last | 1337861 | 0.9035 (n=899) | 0.1859 (n=632) | 4.29 |
| numeric_novelty | 1024 | instruct_visual_mean | 4323313 | 0.9829 (n=899) | 0.0359 (n=633) | 5.09 |
| numeric_novelty | 1024 | instruct_last | 4323313 | 0.9899 (n=899) | 0.0157 (n=633) | 5.09 |
| numeric_novelty | 1024 | embedding_last | 4323313 | 0.9012 (n=899) | 0.1843 (n=633) | 5.09 |
| numeric_novelty | 2048 | instruct_visual_mean | 13224274 | 0.9828 (n=899) | 0.0352 (n=632) | 6.03 |
| numeric_novelty | 2048 | instruct_last | 13224274 | 0.9898 (n=899) | 0.0154 (n=632) | 6.03 |
| numeric_novelty | 2048 | embedding_last | 13224274 | 0.8997 (n=899) | 0.1822 (n=632) | 6.03 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## 중단·재개 기록

완료된 147개 query를 보존하고 activitynet-timelens_01908부터 재개했다. 후보1개인 조건의 빈 pair mask 처리 오류를 수정했으며 과거 다중 후보 통계는 그대로다. 해당 query에서 저장되지 못했던 semantic 계산은 다시 실행했다. 위 비용 합은 checkpoint에 남은 계산 기준이며, 중단 전에 저장되지 못한 semantic 작업의 CUDA 비용까지 복원한 수치가 아니다. 전체 worker 경과 시간에는 중단 전 작업 시간도 포함한다. 이후부터 모델별 scalar checkpoint를 저장해 재개 시 완료된 모델을 반복하지 않는다.


<!-- report-part:5:end -->
