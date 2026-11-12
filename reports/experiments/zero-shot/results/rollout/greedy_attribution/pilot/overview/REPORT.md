# Greedy attribution · 누적 pilot 비교

300 ⊂ 600 ⊂ 900의 고정 prefix cohort다. 각 범위는 같은 누적 원자료를 참조하며 중첩 query를 독립 표본처럼 합산하지 않는다. 완료 query 수와 계획 query 수를 구분한다. 상세 정의·한계·층화 결과는 각 범위 보고서에 있다.

| 범위 | 완료 query | Greedy step | Backward target | 잠재 경쟁만 해당 | mIoU |
|---|---:|---:|---:|---:|---:|
| [pilot300](../300/REPORT.md) | 300 / 300 | 5379 | 7179 | 0 | 0.4870 |
| [pilot600](../600/REPORT.md) | 550 / 600 | 9785 | 13085 | 0 | 0.5077 |
