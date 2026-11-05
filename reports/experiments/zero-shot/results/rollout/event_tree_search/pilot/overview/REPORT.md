# Pilot overview · rollout/event_tree_search/pilot

동일 실험 계열에서 표본이 추가되면300·600·900개 단계를 함께 탐색한다. 아래에 등록된 범위만 현재 자료가 있다.

| 표본 범위 | 보고서·사양 | 고정 코호트 |
|---|---|---|
| pilot300 · 300질문 | [REPORT.md](../300/REPORT.md) | [고정300개](../../../../../artifacts/common/cohorts/nested_pilot_v1/pilot_300/samples.json) |
| pilot600 · 600질문 | [REPORT.md](../600/REPORT.md) | [고정600개](../../../../../artifacts/common/cohorts/nested_pilot_v1/pilot_600/samples.json) |
| pilot900 · 900질문 | [REPORT.md](../900/REPORT.md) | [고정900개](../../../../../artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json) |

300·600·900개를 포함하는 계열의 공통 코호트는300 ⊂ 600 ⊂ 900이며 작은 코호트는 큰 코호트의 정확한 prefix다. 겹치는 질문을 독립 표본처럼 더하지 않는다. 확장 실행의 추가분은 별도 결과로 노출하지 않는다. 이 코호트는 층화 진단 표본으로 전체 벤치마크 평균이 아니다.

새600/900개 결과는 같은 계열의 해당 숫자 폴더와 artifacts/runs의 대응 경로에 등록한다. 전체 실험 목록을 재생성하면 이 통합 목록도 갱신된다. 조건·입력·생성 정책이 바뀌면 별도 버전 계열로 둔다.

[전체 실험 목록](../../../../../../index.html)
