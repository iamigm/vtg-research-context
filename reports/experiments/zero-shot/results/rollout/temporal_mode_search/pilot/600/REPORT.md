<!-- report-appendix:APPENDIX.md parts=26 -->
<!-- report-part:0:begin -->
# Temporal mode search · pilot600 · 결과 보고서

> 이 token-budget 조건은 초기 조사에서 방법별 우선순위가 충분히 작동하지 않아 중단했습니다. 원자료를 보존하며 [capped_rollouts 조건](capped_rollouts/REPORT.md)과 합산하지 않습니다.

상태: 중단·미완료. 고정600 query 중 2 query 완료. 세 데이터셋별200 query,3조건×2seed의 반복 선택은 query 수와 구분한다.

절대 하한0.9%,상대 계수0.01,누적 main 질량90%,예측을[0,D]로 clipping한다. main65,536/tail131,072/judge32,768 decoder input tokens이며 기존 제안 대비 분기 예산을 두 배로 늘렸다.

IoU=교집합/합집합,mIoU=전체 query의 평균 IoU,R@τ=IoU≥τ 비율(τ=.3/.5/.7). 높을수록 좋다. 실패는0점이며 각 query 안에서2seed를 평균한다. ΔmIoU는 동일 clipping greedy 대비 차이다. main oracle은 실제 main 후보의 GT 최고 IoU,pool oracle은 shortlist의 최고 IoU,diagnostic oracle은 main+tail의 사후 상한이며 실제 선택 성능이 아니다.

| 조건 | query / 반복 응답 | mIoU | ΔmIoU | R@.3 | R@.5 | R@.7 | main oracle | diagnostic oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| probability | 2 / 4 | 0.9170 | +0.1110 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| token_entropy | 2 / 4 | 0.9170 | +0.1110 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| temporal_mode | 2 / 4 | 0.9170 | +0.1110 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

<!-- report-part:0:end -->
<!-- report-part:8:begin -->
## 비용과 관측 mode

| 조건 | main tokens | tail tokens | judge tokens | 총 tokens | main mode | tail-only mode |
|---|---:|---:|---:|---:|---:|---:|
| probability | 65535.5 | 131062.0 | 5287.2 | 201902.2 | 10.75 | 18.75 |
| token_entropy | 65535.5 | 131062.0 | 5287.2 | 201902.2 | 10.75 | 18.75 |
| temporal_mode | 65535.5 | 131062.0 | 5287.2 | 201902.2 | 10.75 | 18.75 |

Token 값은 query·seed당 평균 logical 비용이다. 실제 cache execution은 원자료에 별도 보존한다. Mode 수는 관측 시간 가설 수이며 모든 실제 사건에 대한 coverage가 아니다. Tail은 후보에 제안하지 않았다.

<!-- report-part:8:end -->
<!-- report-part:25:begin -->
## 재현과 검증

[실행 명세](SPEC.md). [상세 결과](detail.html). 원자료는 동일 상대 경로의 artifacts/runs/rollout/temporal_mode_search/pilot/600에 저장한다. 관측 mass는 전체 posterior가 아닌 lower bound이며 MI는 추정치다. 미탐색·미달 continuation 수와 실패 상태를 원자료에 표시한다.

현재 active seed:42·43. 기존 seed44는 원자료에 남아 있으나 모든 평균·성능·비용·실패율에서 제외한다. 실제 실행 분모는600 query 및 최대3,600 반복 선택이다.
<!-- report-part:25:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
