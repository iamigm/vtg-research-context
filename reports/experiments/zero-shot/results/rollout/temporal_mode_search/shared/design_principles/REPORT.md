<!-- report-appendix:APPENDIX.md parts=28 -->
<!-- report-part:0:begin -->
# Temporal mode search: 설계 원리와 기대 효과

이 실험은 **모델이 이미 고려하고 있는 시간 구간의 대안들을 찾아, 영상을 다시 보고 그중 하나를 고르게 하면 greedy보다 정확해지는가**를 확인한다. 동시에 **그 개선에 드는 추가 연산이 감당할 만한가**를 평가한다.

예를 들어 “사람이 컵을 집어 드는 순간”을 찾는 질문에 모델이 `10–15초`를 답했다고 하자. 실제로는 처음 컵을 만지는 `9–12초`와 나중에 다시 집는 `40–44초` 사이에서 망설였을 수 있다. 이 실험은 정답 구간인 GT를 보지 않고, 모델의 token distribution과 이어지는 응답을 조사해 이런 대안을 찾는다. 예시의 숫자와 분포는 설명용이며 실제 측정값이 아니다.

설명 대상은 **capped_rollouts 조건**이다. 고정 pilot300·600·900, seed42·43, 세 가지 탐색 방법과 네 가지 rollout 예산을 사용한다. 아래의 기대 효과는 검증할 가설이며, 이미 입증된 성능으로 읽지 않는다.

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## 1. 무엇을 바꾸고, 무엇을 고정하는가

greedy 기준은 Qwen3-VL-8B-Instruct가 원래 영상과 질문을 보고 greedy로 생성한 단일 응답이다. 탐색 실험도 같은 모델과 영상 입력에서 출발한다. 모델을 추가로 학습하지 않는다. 현재 입력은2 FPS,목표 프레임 면적100,352 pixels,`video_query` 순서이며,기준 실행의 frame IDs·timestamp·H×W·tensor hash를 유지한다.

| 설계 원칙 | 쉽게 말하면 | 필요한 이유 |
|---|---|---|
| 모델의 확률 지지 안에서 탐색 | 모델이 어느 정도 가능하다고 본 token만 따라간다. | 임의의 숫자 나열과 모델의 불확실성 탐색을 구분한다. |
| GT 없이 후보 생성·선택 | 정답은 마지막 채점 때만 연다. | 실제 추론에서도 쓸 수 있는 방법을 평가한다. |
| 응답 표현과 시간 가설을 구분 | 다른 문장이 같은 구간을 가리키면 같은 후보로 관리한다. | 표현만 다양한 것을 사건 탐색 성과로 세지 않는다. |
| Tail은 조사만 수행 | Tail에서 발견한 구간은 최종 후보로 주지 않는다. | 주력 탐색 바깥에 무엇이 남았는지 별도로 진단한다. |
| 같은 입력과 예산으로 비교 | 프레임·해상도·질문을 유지하고 탐색 우선순위만 비교한다. | 입력 개선이나 연산 증가를 방법 자체의 효과와 혼동하지 않는다. |
| 정확도와 비용을 함께 평가 | 성능 향상 옆에 추가 연산을 적는다. | 오래 계산했다는 사실만으로 연구적 가치를 주장하지 않는다. |

첫 숫자를 0–9로 모두 바꾸는 forcing과, “다른 구간을 여러 개 제안하라”는 후보 생성 prompt는 사용하지 않는다. 다만 **발견한 후보를 판단하도록 질문하는 단계는 있다.** 후보 생성과 후보 판단은 역할이 다르다.

<!-- report-part:1:end -->
<!-- report-part:2:begin -->
## 2. 전체 과정: 대안을 찾고, 묶고, 비교한다

한 query는 다음 과정을 거친다.

1. **기준 응답 확인:** 같은 입력의 greedy greedy 응답을 재현하고 원본과 token IDs가 일치하는지 확인한다.
2. **Main 탐색:** 확률적으로 응답을 생성하면서, 지나온 prefix에서 다른 유력 token을 선택하면 어디로 이어지는지 조사한다.
3. **Mode 구성:** 완성된 유효 interval을 비슷한 시간 구간끼리 묶는다.
4. **모델 판단:** main의 대표 interval들을 영상 근거로 비교하고 최종 interval 하나를 고른다.
5. **Tail 진단:** 그 checkpoint까지 main이 방문한 prefix에서 tail로 들어가는 경로를 조사한다. 결과는 후보 선택에 전달하지 않는다.
6. **사후 평가:** 선택을 모두 마친 뒤 GT로 정확도를 채점하고 탐색 비용을 집계한다.

작은 예산과 큰 예산은 별도 응답 묶음이 아니다. 같은 탐색을 이어 가며 중간 checkpoint를 저장한다. 각 checkpoint의 main 후보와 판단은 해당 시점의 결과로 고정한다.

<!-- report-part:2:end -->
<!-- report-part:21:begin -->
## 10. 예산과 비교 단위

| 총 rollout 상한 / method / seed | Main 상한 | Tail 상한 |
|---:|---:|---:|
| 32 | 16 | 16 |
| 64 | 32 | 32 |
| 128 | 64 | 64 |
| 256 | 128 | 128 |

Rollout1회는 응답 하나를 완성하거나 유효성 실패로 끝나는 생성 시도다. 중복도 포함한다. Judge 호출은 이 횟수에 포함하지 않고 비용으로 따로 기록한다. 최대 길이512 tokens와 main/tail/judge token guard도 유지하므로 실제 rollout 수가 상한보다 작을 수 있다.

한 query에서 세 방법과 두 seed를 모두 최대 예산까지 수행하면 rollout 시도는 최대1,536회다. 작은 예산은 같은 stream의 checkpoint이므로 네 예산의 상한을 독립 실행처럼 더하지 않는다. 다만 여러 checkpoint에서 실제로 수행한 judge 계산은 전체 실험 비용에 남는다.

Pilot300 ⊂ 600 ⊂ 900은 고정 cohort의 정확한 prefix다. Pilot900은 세 데이터셋별300 query이며, seed나 방법에 따른 반복 응답을 query 수에 더하지 않는다. 두 seed는 query 안에서 먼저 평균한다. 이 cohort는 층화 진단 표본이며 전체 benchmark의 무작위 표본 평균으로 일반화하지 않는다.

<!-- report-part:21:end -->
<!-- report-part:26:begin -->
## 12. 이 실험이 답하려는 질문

좋은 결과는 mode 수가 많아지는 것만으로 정해지지 않는다. **GT 없이 찾은 대안이 더 좋은 답으로 이어지고, 그 추가 비용까지 설득력 있어야 한다.**

현재 설계가 확인하려는 것은 세 가지다. 모델의 token uncertainty 중 시간 가설을 바꾸는 부분을 탐색에 활용할 수 있는가, 그렇게 얻은 후보를 같은 모델이 영상 근거로 더 잘 선택할 수 있는가, 그리고 단순히 더 많이 sampling하는 것보다 효율적인가다.

Mode의 의미적 정확성, 제한된 탐색에서 놓친 분기, 같은 모델을 judge로 쓰는 한계, 두 seed와 층화 pilot의 일반화 범위는 결과를 해석할 때 함께 남겨 둔다. 이 설명 문서는 설계의 이유를 다루며, 최신 성능과 완료 여부는 아래 누적 보고서를 따른다.

<!-- report-part:26:end -->
<!-- report-part:27:begin -->
## 실행 명세·결과·구현

- [Pilot300·600·900 누적 비교](../../pilot/overview/capped_rollouts/REPORT.md)
- [현재 실행 명세: pilot900](../../pilot/900/capped_rollouts/SPEC.md)
- [현재 결과 보고서: pilot900](../../pilot/900/capped_rollouts/REPORT.md)
- [계열 대표 viewer](../../viewer.html)

설명은 다음 동결 구현과 설정을 확인해 작성했다. 실제 식과 예외 처리는 구현을 기준으로 한다.

- [동결 설정](../../../../../artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/config.json)
- [탐색 우선순위와 checkpoint](../../../../../artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_core.py)
- [확률 필터·mode·MI·judge](../../../../../artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_core.py)
<!-- report-part:27:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
