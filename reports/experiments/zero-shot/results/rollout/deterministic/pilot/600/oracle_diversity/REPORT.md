<!-- report-appendix:APPENDIX.md parts=14 -->
<!-- report-part:0:begin -->
# Oracle interval overlap · pilot600

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## Measurement — 무엇을 비교했는가

고정 pilot600 query의 기존 deterministic rollout을 CPU로 분석했다. 모델은 Qwen3-VL-8B-Instruct, 원래 2 FPS·spatial target 100,352 pixels 입력이다. 새로운 GPU 추론·prompt 변경은 하지 않았다. pilot300·600·900은 canonical900 원자료의 고정 prefix subset이다.

**구간이 실제로 겹치는지만 측정한다.** 두 interval의 교집합 길이가 양수이면 overlapping, 0이면 disjoint다. 끝점만 닿는 것도 disjoint에 포함한다. IoU 0.1·0.5 cutoff나 center distance·semantic identity 판정은 사용하지 않는다.

interval a=[s_a,e_a], b=[s_b,e_b]의 교집합 길이는 max(0, min(e_a,e_b)−max(s_a,s_b))다. 이 값이 양수인지로 아래 모든 overlap 비율을 계산한다. **예측과 oracle의 overlap**과 **예측과 GT의 IoU**는 다른 값이다.

<!-- report-part:1:end -->
<!-- report-part:4:begin -->
## Oracle performance

<!-- clipping-exception:begin -->

**범위 초과 예측:** 예측 start/end를 `[0, 영상 길이]`로 clipping하고 길이0이면 IoU=0으로 처리하는 예외를 사후 확인했다. GT는 유지한다. 후보 clipping 후 oracle을 다시 고르면 mIoU는 0.7295→0.7302, greedy와 비겹침인 oracle은 49→46 query다. clipping 후 길이0인 응답 때문에 4 query는 구간 비교가 불가능하며, 기존 표는 clipping 전 기준이다.

<!-- clipping-exception:end -->

| K | greedy mIoU | oracle mIoU (Δ) | oracle R@.3 (Δpp) | oracle R@.5 (Δpp) | oracle R@.7 (Δpp) | 개선 query |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.5157 | 0.5157 (+0.0000) | 67.83% (+0.00) | 54.33% (+0.00) | 38.17% (+0.00) | 0/600 |
| 4 | 0.5157 | 0.6348 (+0.1191) | 79.83% (+12.00) | 67.67% (+13.33) | 52.00% (+13.83) | 415/600 |
| 8 | 0.5157 | 0.6697 (+0.1541) | 83.50% (+15.67) | 70.67% (+16.33) | 55.33% (+17.17) | 485/600 |
| 16 | 0.5157 | 0.7011 (+0.1854) | 85.83% (+18.00) | 74.50% (+20.17) | 60.33% (+22.17) | 517/600 |
| 32 | 0.5157 | 0.7295 (+0.2139) | 88.83% (+21.00) | 76.83% (+22.50) | 64.17% (+26.00) | 543/600 |

<!-- report-part:4:end -->
<!-- report-part:7:begin -->
## Greedy → oracle · K=32

| interval relation | 전체 query | 개선 query 중 | ΔIoU>0.01 중 | 개선 & oracle GT IoU≥.5 | ΔmIoU 기여 | 전체 gain 중 비율 |
|---|---:|---:|---:|---:|---:|---:|
| Identical interval · greedy와 같은 구간 | 57/600 (9.50%) | 0/543 (0.00%) | 0/514 (0.00%) | 0 | 0.0000 | 0.00% |
| Overlapping intervals · 구간이 바뀌었지만 겹침 | 494/600 (82.33%) | 494/543 (90.98%) | 465/514 (90.47%) | 398 | 0.1685 | 78.80% |
| Disjoint intervals · 전혀 겹치지 않음 | 49/600 (8.17%) | 49/543 (9.02%) | 49/514 (9.53%) | 28 | 0.0454 | 21.20% |
| Invalid greedy · 비교할 유효 구간 없음 | 0/600 (0.00%) | 0/543 (0.00%) | 0/514 (0.00%) | 0 | 0.0000 | 0.00% |
| Invalid oracle · 유효 후보 없음 | 0/600 (0.00%) | 0/543 (0.00%) | 0/514 (0.00%) | 0 | 0.0000 | 0.00% |

Oracle GT IoU≥.5: 461/600. duration 초과 oracle: 32/600.
최대 GT IoU에 동률인 경로가 2개 이상인 query는 77/600; oracle IoU≥.5에서는 42/461다. 이 subset에서 동률 선택에 따라 위 interval relation이 달라질 수 있는 query는 0개다. 표는 작은 index tie-break를 고정한 결과다.

<!-- report-part:7:end -->
<!-- report-part:13:begin -->
## Reproducibility

[조건별 성능](STRATA.md) · [summary](../../../../../../artifacts/runs/rollout/deterministic/pilot/600/oracle_diversity/metrics/summary.json) · [query별 파생값](../../../../../../artifacts/runs/rollout/deterministic/pilot/900/oracle_diversity/metrics/per_query.json) · [완성 분기별 비교](../../../../../../artifacts/runs/rollout/deterministic/pilot/900/oracle_diversity/metrics/branches.csv)

query별 파생값·source SHA-256·oracle의 parent chain은 canonical900 analysis에 한 번만 저장한다. 작은 cohort는 같은 배열의 앞부분을 참조한다. 원래 응답·GT·frame·token distribution은 복제하거나 변경하지 않았다.

재생성: `python experiments/zero-shot/_lib/deterministic_oracle_diversity.py`.
<!-- report-part:13:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
