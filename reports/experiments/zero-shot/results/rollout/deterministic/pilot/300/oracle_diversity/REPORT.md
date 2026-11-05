# Oracle interval overlap · pilot300

## Measurement — 무엇을 비교했는가

고정 pilot300 query의 기존 deterministic rollout을 CPU로 분석했다. 모델은 Qwen3-VL-8B-Instruct, 원래 2 FPS·spatial target 100,352 pixels 입력이다. 새로운 GPU 추론·prompt 변경은 하지 않았다. pilot300·600·900은 canonical900 원자료의 고정 prefix subset이다.

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

**범위 초과 예측:** 예측 start/end를 `[0, 영상 길이]`로 clipping하고 길이0이면 IoU=0으로 처리하는 예외를 사후 확인했다. GT는 유지한다. 후보 clipping 후 oracle을 다시 고르면 mIoU는 0.7051→0.7060, greedy와 비겹침인 oracle은 27→26 query다. clipping 후 길이0인 응답 때문에 1 query는 구간 비교가 불가능하며, 기존 표는 clipping 전 기준이다.

<!-- clipping-exception:end -->

| K | greedy mIoU | oracle mIoU (Δ) | oracle R@.3 (Δpp) | oracle R@.5 (Δpp) | oracle R@.7 (Δpp) | 개선 query |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.4870 | 0.4870 (+0.0000) | 64.67% (+0.00) | 51.33% (+0.00) | 35.00% (+0.00) | 0/300 |
| 4 | 0.4870 | 0.6074 (+0.1205) | 78.33% (+13.67) | 65.67% (+14.33) | 48.67% (+13.67) | 208/300 |
| 8 | 0.4870 | 0.6435 (+0.1565) | 82.67% (+18.00) | 69.33% (+18.00) | 51.33% (+16.33) | 239/300 |
| 16 | 0.4870 | 0.6730 (+0.1860) | 84.33% (+19.67) | 73.00% (+21.67) | 56.67% (+21.67) | 255/300 |
| 32 | 0.4870 | 0.7051 (+0.2182) | 87.33% (+22.67) | 75.00% (+23.67) | 61.00% (+26.00) | 270/300 |

## Greedy → oracle · K=8

| interval relation | 전체 query | 개선 query 중 | ΔIoU>0.01 중 | 개선 & oracle GT IoU≥.5 | ΔmIoU 기여 | 전체 gain 중 비율 |
|---|---:|---:|---:|---:|---:|---:|
| Identical interval · greedy와 같은 구간 | 61/300 (20.33%) | 0/239 (0.00%) | 0/221 (0.00%) | 0 | 0.0000 | 0.00% |
| Overlapping intervals · 구간이 바뀌었지만 겹침 | 215/300 (71.67%) | 215/239 (89.96%) | 197/221 (89.14%) | 154 | 0.1185 | 75.69% |
| Disjoint intervals · 전혀 겹치지 않음 | 24/300 (8.00%) | 24/239 (10.04%) | 24/221 (10.86%) | 13 | 0.0381 | 24.31% |
| Invalid greedy · 비교할 유효 구간 없음 | 0/300 (0.00%) | 0/239 (0.00%) | 0/221 (0.00%) | 0 | 0.0000 | 0.00% |
| Invalid oracle · 유효 후보 없음 | 0/300 (0.00%) | 0/239 (0.00%) | 0/221 (0.00%) | 0 | 0.0000 | 0.00% |

Oracle GT IoU≥.5: 208/300. duration 초과 oracle: 21/300.
최대 GT IoU에 동률인 경로가 2개 이상인 query는 36/300; oracle IoU≥.5에서는 15/208다. 이 subset에서 동률 선택에 따라 위 interval relation이 달라질 수 있는 query는 2개다. 표는 작은 index tie-break를 고정한 결과다.

## 다른 후보·분기와 oracle의 overlap · K=8

| 대상 | 겹치는 경로 / 전체 경로 | 겹치지 않는 유효 경로 | invalid | macro overlap fraction (N) | 평균 겹치는 경로 수/query |
|---|---:|---:|---:|---:|---:|
| Other candidates · greedy 포함, oracle 제외 | 1806/2083 (86.70%) | 275 | 2 | 86.81% (300) | 6.02 |
| All branches · oracle 포함 | 1830/2083 (87.85%) | 251 | 2 | 87.95% (300) | 6.10 |
| Other branches · greedy·oracle 제외 | 1591/1844 (86.28%) | 251 | 2 | 86.27% (298) | 5.30 |

다른 후보가 oracle와 하나도 겹치지 않는 singleton: 8/300 (2.67%). oracle GT IoU≥.5인 경우만 보면 2/208 (0.96%). 동일 interval 경로를 합친 unique interval fraction은 86.68%다.

## Greedy → oracle · K=32

| interval relation | 전체 query | 개선 query 중 | ΔIoU>0.01 중 | 개선 & oracle GT IoU≥.5 | ΔmIoU 기여 | 전체 gain 중 비율 |
|---|---:|---:|---:|---:|---:|---:|
| Identical interval · greedy와 같은 구간 | 30/300 (10.00%) | 0/270 (0.00%) | 0/257 (0.00%) | 0 | 0.0000 | 0.00% |
| Overlapping intervals · 구간이 바뀌었지만 겹침 | 243/300 (81.00%) | 243/270 (90.00%) | 230/257 (89.49%) | 192 | 0.1689 | 77.42% |
| Disjoint intervals · 전혀 겹치지 않음 | 27/300 (9.00%) | 27/270 (10.00%) | 27/257 (10.51%) | 16 | 0.0493 | 22.58% |
| Invalid greedy · 비교할 유효 구간 없음 | 0/300 (0.00%) | 0/270 (0.00%) | 0/257 (0.00%) | 0 | 0.0000 | 0.00% |
| Invalid oracle · 유효 후보 없음 | 0/300 (0.00%) | 0/270 (0.00%) | 0/257 (0.00%) | 0 | 0.0000 | 0.00% |

Oracle GT IoU≥.5: 225/300. duration 초과 oracle: 15/300.
최대 GT IoU에 동률인 경로가 2개 이상인 query는 49/300; oracle IoU≥.5에서는 28/225다. 이 subset에서 동률 선택에 따라 위 interval relation이 달라질 수 있는 query는 0개다. 표는 작은 index tie-break를 고정한 결과다.

## 다른 후보·분기와 oracle의 overlap · K=32

| 대상 | 겹치는 경로 / 전체 경로 | 겹치지 않는 유효 경로 | invalid | macro overlap fraction (N) | 평균 겹치는 경로 수/query |
|---|---:|---:|---:|---:|---:|
| Other candidates · greedy 포함, oracle 제외 | 7656/8868 (86.33%) | 1208 | 4 | 86.97% (300) | 25.52 |
| All branches · oracle 포함 | 7683/8868 (86.64%) | 1181 | 4 | 87.26% (300) | 25.61 |
| Other branches · greedy·oracle 제외 | 7413/8598 (86.22%) | 1181 | 4 | 86.79% (298) | 24.71 |

다른 후보가 oracle와 하나도 겹치지 않는 singleton: 0/300 (0.00%). oracle GT IoU≥.5인 경우만 보면 0/225 (0.00%). 동일 interval 경로를 합친 unique interval fraction은 86.83%다.

## Greedy와 겹치지 않는 oracle만 따로 본 결과 · K=32

전체 평균은 greedy 주변 후보가 많은 query의 영향을 받으므로 disjoint subset도 따로 집계한다. 다른 분기에서는 greedy와 oracle 자신을 제외한다.

| subset | query 수 | 다른 분기 overlap macro | pooled 겹치는 수/분기 수 | 다른 겹치는 후보 수 평균 | singleton |
|---|---:|---:|---:|---:|---:|
| 모든 query | 300 | 86.79% | 7413/8598 (86.22%) | 25.52 | 0/300 (0.00%) |
| greedy와 disjoint | 27 | 27.28% | 221/810 (27.28%) | 8.19 | 0/27 (0.00%) |
| greedy와 disjoint & oracle GT IoU≥.5 | 16 | 31.46% | 151/480 (31.46%) | 9.44 | 0/16 (0.00%) |

## Overlap by budget

| K | greedy와 disjoint인 oracle | 다른 분기 overlap macro | 다른 분기 overlap pooled | oracle GT IoU≥.5 중 singleton |
|---|---:|---:|---:|---:|
| 4 | 18/300 (6.00%) | 84.34% | 83.99% | 5/197 (2.54%) |
| 8 | 24/300 (8.00%) | 86.27% | 86.28% | 2/208 (0.96%) |
| 16 | 25/300 (8.33%) | 86.91% | 86.76% | 1/219 (0.46%) |
| 32 | 27/300 (9.00%) | 86.79% | 86.22% | 0/225 (0.00%) |

## Dataset · K=32

| dataset | N | identical | overlapping change | disjoint | 개선 중 disjoint | 다른 분기 overlap macro |
|---|---:|---:|---:|---:|---:|---:|
| activitynet-timelens | 100 | 10 | 77 | 13 | 13/90 (14.44%) | 79.27% |
| charades-timelens | 100 | 10 | 85 | 5 | 5/90 (5.56%) | 92.37% |
| qvhighlights-timelens | 100 | 10 | 81 | 9 | 9/90 (10.00%) | 88.57% |

## Cases · K=32

아래는 큰 개선이 있었던 query의 원자료 참조이며 대표성을 보장하는 표본은 아니다.

| sample | relation | greedy GT IoU → oracle GT IoU | oracle candidate | other overlapping candidates |
|---|---|---|---:|---:|
| [activitynet-timelens_01906](../viewer.html#preset=all&sample=activitynet-timelens_01906&candidate=2&panel=distributions) | Disjoint intervals · 전혀 겹치지 않음 | 0.0000 → 1.0000 | 2 | 13 |
| [qvhighlights-timelens_00229](../viewer.html#preset=all&sample=qvhighlights-timelens_00229&candidate=8&panel=distributions) | Disjoint intervals · 전혀 겹치지 않음 | 0.0000 → 1.0000 | 8 | 15 |
| [qvhighlights-timelens_01519](../viewer.html#preset=all&sample=qvhighlights-timelens_01519&candidate=20&panel=distributions) | Disjoint intervals · 전혀 겹치지 않음 | 0.0000 → 1.0000 | 20 | 11 |
| [qvhighlights-timelens_00778](../viewer.html#preset=all&sample=qvhighlights-timelens_00778&candidate=28&panel=distributions) | Disjoint intervals · 전혀 겹치지 않음 | 0.0000 → 0.9692 | 28 | 27 |
| [charades-timelens_02575](../viewer.html#preset=all&sample=charades-timelens_02575&candidate=29&panel=distributions) | Overlapping intervals · 구간이 바뀌었지만 겹침 | 0.0357 → 0.9756 | 29 | 31 |
| [charades-timelens_02886](../viewer.html#preset=all&sample=charades-timelens_02886&candidate=26&panel=distributions) | Overlapping intervals · 구간이 바뀌었지만 겹침 | 0.0600 → 0.9700 | 26 | 31 |
| [qvhighlights-timelens_01310](../viewer.html#preset=all&sample=qvhighlights-timelens_01310&candidate=16&panel=distributions) | Overlapping intervals · 구간이 바뀌었지만 겹침 | 0.0959 → 0.9863 | 16 | 28 |
| [qvhighlights-timelens_00665](../viewer.html#preset=all&sample=qvhighlights-timelens_00665&candidate=10&panel=distributions) | Overlapping intervals · 구간이 바뀌었지만 겹침 | 0.1453 → 0.9933 | 10 | 31 |

## Reproducibility

[조건별 성능](STRATA.md) · [summary](../../../../../../artifacts/runs/rollout/deterministic/pilot/300/oracle_diversity/metrics/summary.json) · [query별 파생값](../../../../../../artifacts/runs/rollout/deterministic/pilot/900/oracle_diversity/metrics/per_query.json) · [완성 분기별 비교](../../../../../../artifacts/runs/rollout/deterministic/pilot/900/oracle_diversity/metrics/branches.csv)

query별 파생값·source SHA-256·oracle의 parent chain은 canonical900 analysis에 한 번만 저장한다. 작은 cohort는 같은 배열의 앞부분을 참조한다. 원래 응답·GT·frame·token distribution은 복제하거나 변경하지 않았다.

재생성: `python experiments/zero-shot/_lib/deterministic_oracle_diversity.py`.
