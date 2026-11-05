# Oracle interval overlap · pilot600

## Measurement — 무엇을 비교했는가

고정 pilot600 query의 기존 deterministic rollout을 CPU로 분석했다. 모델은 Qwen3-VL-8B-Instruct, 원래 2 FPS·spatial target 100,352 pixels 입력이다. 새로운 GPU 추론·prompt 변경은 하지 않았다. pilot300·600·900은 canonical900 원자료의 고정 prefix subset이다.

**구간이 실제로 겹치는지만 측정한다.** 두 interval의 교집합 길이가 양수이면 overlapping, 0이면 disjoint다. 끝점만 닿는 것도 disjoint에 포함한다. IoU 0.1·0.5 cutoff나 center distance·semantic identity 판정은 사용하지 않는다.

interval a=[s_a,e_a], b=[s_b,e_b]의 교집합 길이는 max(0, min(e_a,e_b)−max(s_a,s_b))다. 이 값이 양수인지로 아래 모든 overlap 비율을 계산한다. **예측과 oracle의 overlap**과 **예측과 GT의 IoU**는 다른 값이다.

### Oracle와 greedy의 비교

각 query에서 앞선 K개 경로 중 GT IoU 최대 후보가 oracle이다(K=1/4/8/16/32). 동률이면 작은 candidate index를 선택한다. EOS 도달·parsing 성공·0≤start<end를 유효 조건으로 쓰고 duration 초과는 기존 평가와 같이 제외하지 않는다. 유효 후보가 없으면 IoU=0이다.

- Identical interval: oracle와 greedy의 start/end가 정확히 같다.
- Overlapping intervals: interval은 달라졌지만 교집합 길이가 양수다.
- Disjoint intervals: 유효한 두 interval의 교집합 길이가 0이다.
- Invalid greedy/oracle: 유효 interval이 없어 비교할 수 없다.

여기서는 겹치는 변화와 겹치지 않는 이동만 구분하며, action의 실제 의미를 판정하지 않는다. 개선 query는 oracle GT IoU−greedy GT IoU>10⁻¹²인 경우다. 아주 작은 개선의 영향을 확인하도록 ΔIoU>0.01도 병기한다. oracle IoU≥0.5 subset도 따로 표시한다.

mIoU gain contribution은 해당 범주의 (oracle IoU−greedy IoU) 합 / 전체 query 수다. 모든 범주를 더하면 전체 ΔmIoU다. gain share는 해당 범주의 gain 합 / 전체 gain 합이다.

### 전체 분기 중 oracle와 겹치는 비율

**분기**는 실제 EOS까지 생성한 non-greedy 완성 경로다. 미완성 queue prefix는 interval을 알 수 없어 분모에 넣지 않는다. 같은 interval로 이어지는 서로 다른 token 경로는 각각 센다.

| 집계 대상 | 포함·제외 기준 |
|---|---|
| Other candidates | K개에서 oracle 자신 한 경로만 제외한다. greedy는 포함한다. |
| All branches | non-greedy 경로 전체다. oracle가 non-greedy라면 그 경로도 포함한다. |
| Other branches | non-greedy 경로에서 oracle 자신도 제외한다. 주 분기 유사 비율은 이 집계다. |

- **Macro fraction**: query마다 (oracle와 겹치는 경로 수 / 해당 경로 수)를 구한 뒤 query를 동일 가중 평균한다. 분모 0인 query는 평균에서 제외하고 N을 표시한다.
- **Pooled fraction**: 전체 query의 겹치는 경로 수 합 / 전체 경로 수 합이다. macro와 구분한다.
- **Singleton**: oracle 자신을 뺀 나머지 모든 후보 중 oracle와 겹치는 것이 하나도 없는 query다. greedy도 이 비교에 포함한다. 이것이 후보 집합 안에서 oracle가 얼마나 혼자 떨어져 있는지를 나타낸다.
- **Unique interval fraction**: 동일 start/end를 합치고 oracle interval을 제외한 뒤 overlap 비율을 query별 평균한다. 중복 token 경로의 영향을 확인하는 보조 지표다.

invalid 경로는 전체 경로 수 분모에는 포함하되 overlap이나 disjoint로 분류하지 않고 별도 수를 표시한다. deterministic 경로 빈도는 sampling probability가 아니다. oracle 자신의 자명한 일치를 제외한 결과와 포함한 결과를 모두 제공한다.

성능 지표 mIoU는 query별 GT IoU의 동일 가중 평균이며 높을수록 좋다. R@τ는 GT IoU≥τ query 비율(τ=.3/.5/.7)이며 표에서는 %다. Δ는 같은 query의 greedy 대비이며 R의 변화는 pp다. 실패는0점이다.

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

## Greedy → oracle · K=8

| interval relation | 전체 query | 개선 query 중 | ΔIoU>0.01 중 | 개선 & oracle GT IoU≥.5 | ΔmIoU 기여 | 전체 gain 중 비율 |
|---|---:|---:|---:|---:|---:|---:|
| Identical interval · greedy와 같은 구간 | 115/600 (19.17%) | 0/485 (0.00%) | 0/445 (0.00%) | 0 | 0.0000 | 0.00% |
| Overlapping intervals · 구간이 바뀌었지만 겹침 | 440/600 (73.33%) | 440/485 (90.72%) | 400/445 (89.89%) | 325 | 0.1186 | 77.00% |
| Disjoint intervals · 전혀 겹치지 않음 | 45/600 (7.50%) | 45/485 (9.28%) | 45/445 (10.11%) | 20 | 0.0354 | 23.00% |
| Invalid greedy · 비교할 유효 구간 없음 | 0/600 (0.00%) | 0/485 (0.00%) | 0/445 (0.00%) | 0 | 0.0000 | 0.00% |
| Invalid oracle · 유효 후보 없음 | 0/600 (0.00%) | 0/485 (0.00%) | 0/445 (0.00%) | 0 | 0.0000 | 0.00% |

Oracle GT IoU≥.5: 424/600. duration 초과 oracle: 42/600.
최대 GT IoU에 동률인 경로가 2개 이상인 query는 64/600; oracle IoU≥.5에서는 28/424다. 이 subset에서 동률 선택에 따라 위 interval relation이 달라질 수 있는 query는 5개다. 표는 작은 index tie-break를 고정한 결과다.

## 다른 후보·분기와 oracle의 overlap · K=8

| 대상 | 겹치는 경로 / 전체 경로 | 겹치지 않는 유효 경로 | invalid | macro overlap fraction (N) | 평균 겹치는 경로 수/query |
|---|---:|---:|---:|---:|---:|
| Other candidates · greedy 포함, oracle 제외 | 3703/4174 (88.72%) | 468 | 3 | 88.79% (600) | 6.17 |
| All branches · oracle 포함 | 3748/4174 (89.79%) | 423 | 3 | 89.86% (600) | 6.25 |
| Other branches · greedy·oracle 제외 | 3263/3689 (88.45%) | 423 | 3 | 88.41% (598) | 5.44 |

다른 후보가 oracle와 하나도 겹치지 않는 singleton: 15/600 (2.50%). oracle GT IoU≥.5인 경우만 보면 4/424 (0.94%). 동일 interval 경로를 합친 unique interval fraction은 88.71%다.

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

## 다른 후보·분기와 oracle의 overlap · K=32

| 대상 | 겹치는 경로 / 전체 경로 | 겹치지 않는 유효 경로 | invalid | macro overlap fraction (N) | 평균 겹치는 경로 수/query |
|---|---:|---:|---:|---:|---:|
| Other candidates · greedy 포함, oracle 제외 | 15688/17700 (88.63%) | 2003 | 9 | 89.18% (600) | 26.15 |
| All branches · oracle 포함 | 15737/17700 (88.91%) | 1954 | 9 | 89.45% (600) | 26.23 |
| Other branches · greedy·oracle 제외 | 15194/17157 (88.56%) | 1954 | 9 | 89.09% (598) | 25.32 |

다른 후보가 oracle와 하나도 겹치지 않는 singleton: 0/600 (0.00%). oracle GT IoU≥.5인 경우만 보면 0/461 (0.00%). 동일 interval 경로를 합친 unique interval fraction은 89.11%다.

## Greedy와 겹치지 않는 oracle만 따로 본 결과 · K=32

전체 평균은 greedy 주변 후보가 많은 query의 영향을 받으므로 disjoint subset도 따로 집계한다. 다른 분기에서는 greedy와 oracle 자신을 제외한다.

| subset | query 수 | 다른 분기 overlap macro | pooled 겹치는 수/분기 수 | 다른 겹치는 후보 수 평균 | singleton |
|---|---:|---:|---:|---:|---:|
| 모든 query | 600 | 89.09% | 15194/17157 (88.56%) | 26.15 | 0/600 (0.00%) |
| greedy와 disjoint | 49 | 31.29% | 460/1470 (31.29%) | 9.39 | 0/49 (0.00%) |
| greedy와 disjoint & oracle GT IoU≥.5 | 28 | 34.29% | 288/840 (34.29%) | 10.29 | 0/28 (0.00%) |

## Overlap by budget

| K | greedy와 disjoint인 oracle | 다른 분기 overlap macro | 다른 분기 overlap pooled | oracle GT IoU≥.5 중 singleton |
|---|---:|---:|---:|---:|
| 4 | 33/600 (5.50%) | 87.76% | 87.74% | 7/406 (1.72%) |
| 8 | 45/600 (7.50%) | 88.41% | 88.45% | 4/424 (0.94%) |
| 16 | 48/600 (8.00%) | 88.95% | 88.81% | 3/447 (0.67%) |
| 32 | 49/600 (8.17%) | 89.09% | 88.56% | 0/461 (0.00%) |

## Dataset · K=32

| dataset | N | identical | overlapping change | disjoint | 개선 중 disjoint | 다른 분기 overlap macro |
|---|---:|---:|---:|---:|---:|---:|
| activitynet-timelens | 200 | 21 | 155 | 24 | 24/179 (13.41%) | 83.32% |
| charades-timelens | 200 | 18 | 171 | 11 | 11/182 (6.04%) | 93.37% |
| qvhighlights-timelens | 200 | 18 | 168 | 14 | 14/182 (7.69%) | 90.52% |

## Cases · K=32

아래는 큰 개선이 있었던 query의 원자료 참조이며 대표성을 보장하는 표본은 아니다.

| sample | relation | greedy GT IoU → oracle GT IoU | oracle candidate | other overlapping candidates |
|---|---|---|---:|---:|
| [activitynet-timelens_00812](../viewer.html#preset=all&sample=activitynet-timelens_00812&candidate=2&panel=distributions) | Disjoint intervals · 전혀 겹치지 않음 | 0.0000 → 1.0000 | 2 | 17 |
| [activitynet-timelens_01906](../viewer.html#preset=all&sample=activitynet-timelens_01906&candidate=2&panel=distributions) | Disjoint intervals · 전혀 겹치지 않음 | 0.0000 → 1.0000 | 2 | 13 |
| [activitynet-timelens_03505](../viewer.html#preset=all&sample=activitynet-timelens_03505&candidate=30&panel=distributions) | Disjoint intervals · 전혀 겹치지 않음 | 0.0000 → 1.0000 | 30 | 14 |
| [qvhighlights-timelens_00229](../viewer.html#preset=all&sample=qvhighlights-timelens_00229&candidate=8&panel=distributions) | Disjoint intervals · 전혀 겹치지 않음 | 0.0000 → 1.0000 | 8 | 15 |
| [activitynet-timelens_01723](../viewer.html#preset=all&sample=activitynet-timelens_01723&candidate=6&panel=distributions) | Overlapping intervals · 구간이 바뀌었지만 겹침 | 0.0123 → 0.9972 | 6 | 31 |
| [activitynet-timelens_00313](../viewer.html#preset=all&sample=activitynet-timelens_00313&candidate=15&panel=distributions) | Overlapping intervals · 구간이 바뀌었지만 겹침 | 0.0455 → 1.0000 | 15 | 26 |
| [activitynet-timelens_04235](../viewer.html#preset=all&sample=activitynet-timelens_04235&candidate=15&panel=distributions) | Overlapping intervals · 구간이 바뀌었지만 겹침 | 0.0457 → 0.9982 | 15 | 31 |
| [charades-timelens_02575](../viewer.html#preset=all&sample=charades-timelens_02575&candidate=29&panel=distributions) | Overlapping intervals · 구간이 바뀌었지만 겹침 | 0.0357 → 0.9756 | 29 | 31 |

## Reproducibility

[조건별 성능](STRATA.md) · [summary](../../../../../../artifacts/runs/rollout/deterministic/pilot/600/oracle_diversity/metrics/summary.json) · [query별 파생값](../../../../../../artifacts/runs/rollout/deterministic/pilot/900/oracle_diversity/metrics/per_query.json) · [완성 분기별 비교](../../../../../../artifacts/runs/rollout/deterministic/pilot/900/oracle_diversity/metrics/branches.csv)

query별 파생값·source SHA-256·oracle의 parent chain은 canonical900 analysis에 한 번만 저장한다. 작은 cohort는 같은 배열의 앞부분을 참조한다. 원래 응답·GT·frame·token distribution은 복제하거나 변경하지 않았다.

재생성: `python experiments/zero-shot/_lib/deterministic_oracle_diversity.py`.
