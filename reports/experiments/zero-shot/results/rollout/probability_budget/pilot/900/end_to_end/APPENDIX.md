# Appendix · End-to-end completion · pilot900

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:3:begin -->
## 독립 구간 검증·semantic의 추가 비용

서로 다른 숫자 구간 수와 실제 고유 crop 수를 구분한다. Batch validation의 반복 비용도 아래 합계에 포함한다. 통계용 semantic 계산을 final 선택 비용으로 숨기지 않는다.

구간 합 110108개, 고유 crop 합 104975개.

| 단계 | 실제 forward 합 | 실제 입력 token 합 | Padded token 합 | CUDA 합 s | Model wall 합 s |
|---|---:|---:|---:|---:|---:|
| verifier | 104976 | 350178802 | 350179282 | 24834.360 | 25634.154 |
| instruct | 25327 | 283906781 | 289500629 | 16848.979 | 17728.806 |
| embedding | 25328 | 284011757 | 289605938 | 16788.703 | 17669.723 |

완료 query의 후속 처리 wall 합은 112354.275s (31.210시간), query당 평균은 129.291s다. 이는 세 방법×세 예산과 semantic 진단을 공유 실행한 전체 작업량이며 단일 조건의 단독 latency가 아니다. 이 중 semantic 모델 forward wall 합은 35398.529s (9.833시간), 고유 final generation wall 합은 8067.069s (2.241시간)다. CPU 입력 준비·공유 prefill 등은 각각의 model forward wall과 구분한다. 중단 전에 저장되지 못한 작업의 비용은 아래 재개 기록의 제한을 따른다.


<!-- report-part:3:end -->
<!-- report-part:4:begin -->
## 시간 그룹과 semantic 차이

Cosine은 같은 모델의 벡터끼리 비교한다. 1에 가까울수록 유사하며, 높은 유사도가 같은 사건의 확정 근거는 아니다. 아래 within−between은 시간 그룹 내부 mean cosine에서 그룹 사이 mean cosine을 뺀 값이다. 양수는 이 embedding이 시간 그룹의 차이와 같은 방향으로 구분한다는 진단이며 GT 검증 점수가 아니다. 그룹이 하나뿐이면 between은 정의되지 않는다.

| 방법 | 예산 | Pooling | 관측 pair 수 합 | 평균 cosine (유효query n) | 평균 within−between (유효query n) | 평균 시간 그룹 수 |
|---|---:|---|---:|---:|---:|---:|
| levin | 512 | instruct_visual_mean | 626790 | 0.9838 (n=856) | 0.0389 (n=558) | 2.72 |
| levin | 512 | instruct_last | 626790 | 0.9905 (n=856) | 0.0165 (n=558) | 2.72 |
| levin | 512 | embedding_last | 626790 | 0.9074 (n=856) | 0.1910 (n=558) | 2.72 |
| levin | 1024 | instruct_visual_mean | 2212711 | 0.9830 (n=864) | 0.0371 (n=586) | 3.42 |
| levin | 1024 | instruct_last | 2212711 | 0.9901 (n=864) | 0.0158 (n=586) | 3.42 |
| levin | 1024 | embedding_last | 2212711 | 0.9028 (n=864) | 0.1852 (n=586) | 3.42 |
| levin | 2048 | instruct_visual_mean | 7086513 | 0.9829 (n=867) | 0.0356 (n=591) | 4.14 |
| levin | 2048 | instruct_last | 7086513 | 0.9899 (n=867) | 0.0154 (n=591) | 4.14 |
| levin | 2048 | embedding_last | 7086513 | 0.9001 (n=867) | 0.1822 (n=591) | 4.14 |
| native_mass | 512 | instruct_visual_mean | 1294766 | 0.9831 (n=868) | 0.0365 (n=610) | 4.27 |
| native_mass | 512 | instruct_last | 1294766 | 0.9902 (n=868) | 0.0158 (n=610) | 4.27 |
| native_mass | 512 | embedding_last | 1294766 | 0.9030 (n=868) | 0.1856 (n=610) | 4.27 |
| native_mass | 1024 | instruct_visual_mean | 4174095 | 0.9829 (n=868) | 0.0360 (n=610) | 5.08 |
| native_mass | 1024 | instruct_last | 4174095 | 0.9899 (n=868) | 0.0156 (n=610) | 5.08 |
| native_mass | 1024 | embedding_last | 4174095 | 0.9007 (n=868) | 0.1841 (n=610) | 5.08 |
| native_mass | 2048 | instruct_visual_mean | 12709109 | 0.9828 (n=868) | 0.0353 (n=609) | 5.97 |
| native_mass | 2048 | instruct_last | 12709109 | 0.9898 (n=868) | 0.0154 (n=609) | 5.97 |
| native_mass | 2048 | embedding_last | 12709109 | 0.8992 (n=868) | 0.1821 (n=609) | 5.97 |
| numeric_novelty | 512 | instruct_visual_mean | 1295667 | 0.9831 (n=868) | 0.0365 (n=610) | 4.27 |
| numeric_novelty | 512 | instruct_last | 1295667 | 0.9902 (n=868) | 0.0158 (n=610) | 4.27 |
| numeric_novelty | 512 | embedding_last | 1295667 | 0.9030 (n=868) | 0.1855 (n=610) | 4.27 |
| numeric_novelty | 1024 | instruct_visual_mean | 4174216 | 0.9829 (n=868) | 0.0360 (n=610) | 5.07 |
| numeric_novelty | 1024 | instruct_last | 4174216 | 0.9899 (n=868) | 0.0156 (n=610) | 5.07 |
| numeric_novelty | 1024 | embedding_last | 4174216 | 0.9007 (n=868) | 0.1842 (n=610) | 5.07 |
| numeric_novelty | 2048 | instruct_visual_mean | 12705241 | 0.9828 (n=868) | 0.0353 (n=609) | 5.98 |
| numeric_novelty | 2048 | instruct_last | 12705241 | 0.9898 (n=868) | 0.0154 (n=609) | 5.98 |
| numeric_novelty | 2048 | embedding_last | 12705241 | 0.8992 (n=868) | 0.1821 (n=609) | 5.98 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## 중단·재개 기록

완료된 147개 query를 보존하고 activitynet-timelens_01908부터 재개했다. 후보1개인 조건의 빈 pair mask 처리 오류를 수정했으며 과거 다중 후보 통계는 그대로다. 해당 query에서 저장되지 못했던 semantic 계산은 다시 실행했다. 위 비용 합은 checkpoint에 남은 계산 기준이며, 중단 전에 저장되지 못한 semantic 작업의 CUDA 비용까지 복원한 수치가 아니다. 전체 worker 경과 시간에는 중단 전 작업 시간도 포함한다. 이후부터 모델별 scalar checkpoint를 저장해 재개 시 완료된 모델을 반복하지 않는다.


<!-- report-part:5:end -->
