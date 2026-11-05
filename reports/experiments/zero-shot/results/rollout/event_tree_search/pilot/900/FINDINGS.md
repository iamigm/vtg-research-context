# Failure analysis · pilot900

기존 고정 900 query의 완료된 원자료를 CPU로 분석했다. 새로운 GPU 추론·prompt 변경·응답 수정은 하지 않았다. 결과의 분모는 query 또는 명시한 조건 내 candidate/비교 수이며, 같은 query의 여러 방법을 독립 query로 합산하지 않는다.

## 확인된 운영 오류

GPU worker 두 개는900개 추론을 완료했지만 attached worker를 처리하는 supervisor가 `returncode` 속성을 참조해 최종 보고가 중단됐다. `poll()` 반환값으로 처리하도록 고치고 정상/실패 종료 회귀 검증을 수행했다. 추론 원자료와 search 정책은 변경하지 않았다.

## 1. Duration 기준의 불일치

기존 greedy mIoU 0.5193. 새 후보와 같은 duration admissibility(0≤start<end≤duration+.001)를 기존 greedy에도 사후 적용해 실패를0점 처리하면 0.4638다. 이는 정책 영향을 분리하는 진단이며 원래 benchmark 점수를 대체하지 않는다.
기존 greedy가 이 기준에서 제외되는 query는 79/900. 영상 길이 초과가≤.05초 5개, ≤.1초 14개, ≤.5초 74개, ≤1초 75개다(누적 수). 작은 초과는 timestamp 반올림 규모와 맞지만 반올림이 원인이라는 직접 검증은 아니다.

이 불일치는 baseline 대비 손실의 일부를 만든다. 다만 출력 단계의 기준을 바꾸면 당시 수행하지 않은 평가·탐색도 달라지므로, 원자료만으로 전체 알고리즘의 수정 후 성능을 재구성할 수 없다. 새 policy로 재실행해야 한다.

## 2. Proposal 자체와 gate

| 단계 | oracle mIoU | R@.5 정보를 잃은 query |
|---|---:|---:|
| 기존8 | 0.6717 | — |
| 모든 신규 proposal, positive interval | 0.7204 | — |
| duration 기준 통과 | 0.6983 | 25 |
| model gate 통과 | 0.6523 | 45 |

여기서 oracle은 생성된 후보 중 GT IoU가 가장 높은 값이다. 각 단계에서 실제로 존재한 후보만 보므로 원래 존재하지 않은 경로의 반사실적 성능이 아니다. R@.5 정보 손실은 직전 단계에서는 oracle≥.5인데 해당 단계에서<.5가 된 query 수다.
Gate 통과 후보가 없는 query 38/900; 그중 duration-valid 후보 자체가 없는 경우 4, valid 후보는 있지만 model gate가 모두 제외한 경우 34다.
Leading digit 경로는 9000개 중 8448개가 duration-valid였다. Semantic proposal의 unique interval은 query당 평균 1.46개다.
Semantic 요청에 완성된 JSON array로 응답한 query는 389/900다. JSON 후보는 없지만 통상적인 timestamp 응답이 나온 495개는 proposal로 사용되지 않았다. 그중 duration-valid 473개, 기존 신규 proposal oracle<.5에서 그 버려진 응답이 IoU≥.5인 경우 3개다. 이는 응답 형식 예외 처리의 한계다.

| 후보 구성(continuous IoU, duration cut 이전) | oracle mIoU | R@.5 |
|---|---:|---:|
| old8 | 0.6717 | 70.78% |
| leading | 0.6594 | 72.11% |
| semantic | 0.2670 | 28.22% |
| old8_leading | 0.7153 | 77.11% |
| old8_semantic | 0.6817 | 72.00% |

## 3. 어느 단계에서 좋은 후보를 잃었는가

각 method의 실제 생성된 후보에서 최대 IoU를 구하고, duration → critic/blocked → 최종 비교 pool → 실제 선택 순서로 평균을 계산했다. 각 차이는 해당 기록에서의 oracle 손실 분해이며 gate를 없애 재탐색했을 때의 인과효과는 아니다. commit 대조군의 pool 손실에는 사건 고정 제약도 포함된다.

| method | all oracle | valid oracle | eligible oracle | pool oracle | selected mIoU | pool≥.5인데 최종<.5 |
|---|---:|---:|---:|---:|---:|---:|
| proposal_only_token_seed | 0.6717 | 0.6370 | 0.5817 | 0.5558 | 0.4664 | 94/525 |
| proposal_only_wide | 0.7204 | 0.6983 | 0.6523 | 0.5325 | 0.3953 | 149/509 |
| nucleus_frontier@4096 | 0.7234 | 0.7063 | 0.6527 | 0.5126 | 0.4033 | 117/487 |
| nucleus_frontier@8192 | 0.7246 | 0.7075 | 0.6524 | 0.5114 | 0.4045 | 111/485 |
| value_best_first@8192 | 0.7268 | 0.7103 | 0.6568 | 0.4241 | 0.3772 | 46/389 |
| puct@8192 | 0.7293 | 0.7126 | 0.6564 | 0.4976 | 0.4001 | 110/471 |
| submodular_coverage@8192 | 0.7285 | 0.7114 | 0.6557 | 0.4878 | 0.3961 | 108/469 |
| token_seed_best_first@8192 | 0.6736 | 0.6396 | 0.5814 | 0.4700 | 0.4350 | 43/450 |

## 4. 최종 pool의 동일 구간 중복

현재 final selector는 context/token 경로가 다르면 같은 interval도 별도 후보로 받아들인다. Soft IoU diversity penalty만 적용하므로 동일한 interval이 최대4개의 비교 자리를 차지할 수 있다. 아래 dedup은 정확히 같은 interval을 하나로 묶고, 기존 visual value와 동일한 selection rule로4개를 다시 구성한 CPU 진단이다. 새 pool의 oracle은 실제 model selection 성능이 아니라 상한이며, 새로운 pairwise inference를 수행하지 않았다.

| method | 중복 pool / nonempty | unique=1 pool / nonempty | 평균 unique/pool size | 원래 pool oracle | dedup pool oracle |
|---|---:|---:|---:|---:|---:|
| proposal_only_wide | 0/862 | 32/862 | 3.81/3.81 | 0.5325 | 0.5325 |
| reward_balanced@4096 | 328/858 | 64/858 | 3.34/3.96 | 0.5282 | 0.5472 |
| reward_balanced@8192 | 532/858 | 103/858 | 2.92/3.97 | 0.5116 | 0.5530 |
| nucleus_frontier@8192 | 376/858 | 153/858 | 3.04/3.97 | 0.5114 | 0.5380 |
| value_best_first@8192 | 734/858 | 425/858 | 1.87/3.97 | 0.4241 | 0.5413 |
| puct@8192 | 633/858 | 137/858 | 2.66/3.97 | 0.4976 | 0.5527 |
| submodular_coverage@8192 | 665/858 | 180/858 | 2.43/3.97 | 0.4878 | 0.5591 |
| token_seed_best_first@8192 | 782/807 | 408/807 | 1.64/4.00 | 0.4700 | 0.5520 |

## 5. Visual score와 pairwise 선택의 진단

초기 valid 후보의 query 내부에서 IoU 차이≥.1인 65288쌍 중 visual value 순위가 GT IoU 순위와 맞는 비율은 57.64%다(tie=.5). 사건이 포함된다는 판단과구간이 정확하다는 판단은 다른 목표이므로, 이 수치를 사건 판별의 semantic accuracy로 해석하지 않는다.
여러 방법·예산에서 재사용한 동일 model comparison을 query별로 중복 제거하면 10963쌍이다. Presentation order를 바꿨을 때 판단이 불일치한 비율 68.32%. GT IoU 차이≥.1인 4892쌍에서 더 높은 IoU 후보를 고른 비율 60.30%다.
길이 차이≥20% 및 IoU 차이≥.1인 비교에서 긴 구간을 고른 경우 2652/4127, 긴 구간을 골랐지만 그 구간의 IoU가 더 낮은 경우 772/4127다. 이는 길이 편향에 대한 관찰이며 원인 확정은 아니다.

| model criterion | unique calls | median raw option mass | raw mass<.01 |
|---|---:|---:|---:|
| event_presence | 15539 | 1.0000 | 0.00% |
| event_contradiction | 15537 | 1.0000 | 0.00% |
| boundary_fit | 8294 | 1.0000 | 0.00% |
| final_pairwise | 17392 | 1.0000 | 0.00% |

## 6. 탐색과 refinement가 실제로 한 일

동일 parent interval을 다시 생성한 비율은 generated-valid node가 분모다. 반복 interval은 해당 query·method의 초기 후보 및 앞선 node와 같은 구간을 뜻한다. 서로 다른 token/context 경로일 수 있으므로 동일 model evaluation과 동일시하지 않는다. cut-off 폐기는 token cap에서 생성은 끝났지만 후속 assessment/prior가 끝나지 않아 atomic tree commit을 하지 못한 경우다.

| method | mean expansions | final이 초기 coarse 후보 | parent와 같은 생성 | 반복 interval | cut-off 폐기 query | 폐기 token/search token |
|---|---:|---:|---:|---:|---:|---:|
| value_best_first@4096 | 6.33 | 20.05% | 71.99% | 91.53% | 294/900 | 2.36% |
| value_best_first@8192 | 11.98 | 12.35% | 73.38% | 94.92% | 235/900 | 1.12% |
| nucleus_frontier@4096 | 6.21 | 51.28% | 79.30% | 93.49% | 343/900 | 3.06% |
| nucleus_frontier@8192 | 11.87 | 48.60% | 79.04% | 96.23% | 214/900 | 1.07% |
| puct@4096 | 6.68 | 11.31% | 68.07% | 89.52% | 460/900 | 3.93% |
| puct@8192 | 12.66 | 9.44% | 68.23% | 93.64% | 392/900 | 1.70% |
| neural_controller@4096 | 2.64 | 46.74% | 79.21% | 92.95% | 273/900 | 10.64% |
| neural_controller@8192 | 4.07 | 27.27% | 69.29% | 89.94% | 228/900 | 4.11% |
| submodular_coverage@4096 | 6.53 | 14.57% | 68.52% | 89.67% | 462/900 | 4.18% |
| submodular_coverage@8192 | 12.19 | 12.12% | 70.90% | 93.77% | 390/900 | 1.83% |
| self_refine@4096 | 6.24 | 26.34% | 77.88% | 92.52% | 85/900 | 0.94% |
| self_refine@8192 | 11.89 | 24.83% | 78.17% | 95.84% | 163/900 | 0.76% |

## 7. 방법 간 차이가 작아지는 이유의 단서

Nucleus와 typicality 최종 interval은4096 cap에서 900/900,8192 cap에서 900/900 query가 같았다. 초기 eligible finite-list prior의 top1 probability≥.95는 228/862, median top1=0.6284368924657872. 두 방식의 초기 .95 support 집합이 동일한 경우 862/862, nucleus support 크기 평균 3.01다. 이번 구현은 support 안의 방문 순서를 같은 q/(1+visits)로 정하므로 support가 같으면 이 단계에서 차이를 만들지 못한다. 이는 token-level nucleus와 typical sampling 일반의 동등성을 뜻하지 않는다.
원래 leading digit의 stepwise probability와 같은 경로를 batched teacher forcing한 probability 차이는 평균 0.001715, 최대 0.0769341였다. 입력 hash·greedy replay와 함께 implementation drift를 점검한 진단이며 batch/stepwise 수치 차이를 완전히 배제한 검증은 아니다.

## 8. GT 기준 회복과 손상

회복 분모는 기존 greedy IoU<.1, 손상 분모는 기존 greedy IoU≥.5다. 전체 query의 성능과 함께 해석한다.

| method | recovery to IoU≥.5 | harmful to IoU<.5 | final length>2×GT / valid | median length/GT |
|---|---:|---:|---:|---:|
| proposal_only_wide | 24/174 | 194/491 | 226/862 | 1.1428571428571428 |
| nucleus_frontier@8192 | 21/174 | 169/491 | 209/858 | 1.1111111111111112 |
| value_best_first@8192 | 15/174 | 196/491 | 198/858 | 1.0749198717948718 |
| puct@8192 | 23/174 | 190/491 | 211/858 | 1.108125 |
| submodular_coverage@8192 | 22/174 | 190/491 | 210/858 | 1.0916083916083914 |
| self_refine@8192 | 16/174 | 201/491 | 198/858 | 1.080128205128205 |
| commit_then_refine@8192 | 21/174 | 223/491 | 192/833 | 1.0416666666666667 |
| token_seed_best_first@8192 | 19/174 | 143/491 | 173/807 | 1.0625 |

## 핵심 사례: 정수 prefix의 내부 대안을 놓침

qvhighlights-timelens_00507의 GT는 [115, 146]다. start의 첫 token으로1을 선택한 직후 P('.')=0.4031, P('4')=0.3139, P('1')=0.1680, P('2')=0.0481였다. 이번 leading-digit proposal은 가장 높은 decimal point를 따라 [1.0, 10.0]로 완성했다. 그 뒤 정수 자리의1·4·2 대안을 체계적으로 열지 않았다.
기존32개에는 이미 candidate#14=[116.0, 145.0], IoU=0.9355가 있었다. 이번 seed는 첫8개만 사용했고 해당 좋은 경로를 초기 pool에 포함하지 않았다. 새 semantic proposal 원문은 `The event happens in 147.0 - 150.0 seconds.`였고 요청한 JSON array가 아니었다. 모든 방법이 최종 유효 응답을 내지 못했다.

따라서 이번 실행의 event/action tree 탐색 결과로 숫자 prefix tree 내부의 재검토·확장 가설을 기각할 수 없다. 첫 자리 뒤를 greedy로 완성한 설계가 사용자가 지적한 중요한 내부 분기를 놓쳤다. 이 구현 한계와 candidate 선택 문제를 먼저 수정해야 한다.

## 9. 실패 원인 추정과 해석 한계

확인된 사실은 위 수치와 코드 경로다. 다음은 그 사실로부터의 추정이다.

- Strict duration gate와 benchmark baseline의 admissibility 불일치가 일부 손실을 만든다. endpoint를 임의 clipping해 성능을 올리지 말고, 허용오차 정책을 모든 조건에 맞춘 재실험으로 확인해야 한다.
- Final comparison pool에서 같은 interval을 별도 후보로 유지한 것이 유효한 선택 폭을 줄였다. token/context 경로는 보존하되 interval만 제시하는 최종 비교에서는 exact interval을 중복 제거하는 대조가 필요하다.
- 초기 coarse 후보의 boundary probability는 실제로 평가하지 않고 .5로 둔다. 같은 interval을 refinement에서 다시 평가해 boundary probability가1에 가까워지면 다른 값이 같아도 value의 boundary multiplier는 .75에서1로 올라간다. 이런 깊이별 평가 기준 차이가 반복 생성의 점수를 높이고 초기 다른 사건 후보를 밀어낼 가능성이 있다.
- Boundary child에서 확인한 contradiction을 event root 전체에 전파한다. 이 때문에 원래 GT에 가까운 parent·siblings까지 함께 제거될 수 있다. 원래 사건이 잘못됐다는 판단과 특정 refinement가 잘못됐다는 판단을 분리하는 대조가 필요하다.
- Visual value와 pairwise judgement가 localization IoU에 충분히 정렬되지 않았다면, 더 강한 탐색은 잘못된 proxy를 더 잘 최적화할 수 있다. 사건 포함 여부와 정확한 사건·경계를 나눠 검증할 필요가 있다.
- Same full-video input을 모든 검토에 사용했다. 시간 구간별 evidence를 새로 분리해서 보는 실험을 하지 않았으므로, 영상 정보 부족/FPS가 원인이라고 확정할 수 없다.
- Candidate-level tree/beam/UCT/PUCT/typicality adaptations의 결과이며 원 논문의 token-level 또는 학습된 value-model 성능 전체로 일반화할 수 없다. 이번 초기 numeric expansion은 leading digit 뒤를 greedy로 완성하며 둘째·셋째 integer token의 대안을 체계적으로 열지 않는다.
- Budget cut-off의 atomic rollback과 반복 boundary 생성은 실제 계산의 일부를 후보 증가로 연결하지 못한다. 재실험에서는 생성된 후보와 완료된 critic을 단계별로 checkpoint하는 비교가 필요하다.
- 수학적 coverage의 approximation guarantee는 고정 F에 관한 것이다. F의 kernel·quality가 GT localization 목표와 다르면 그 목적함수의 최적화 성공과 실제 성능은 다를 수 있다.

## 사례

### duration

```json
{
  "key": "activitynet-timelens_02810",
  "query": "A man talks to the camera and demonstrates how to clean paint brushes before applying paint with one to a painting he is standing in front of",
  "dataset": "activitynet-timelens",
  "duration": 154.154,
  "gt": [
    0,
    154
  ],
  "greedy": [
    0.0,
    154.2
  ],
  "greedy_iou": 0.9987029831387809,
  "excess": 0.04599999999999227
}
```

```json
{
  "key": "activitynet-timelens_03752",
  "query": "Several women doing aerobic exercise together",
  "dataset": "activitynet-timelens",
  "duration": 203.53666666666666,
  "gt": [
    0,
    203
  ],
  "greedy": [
    0.0,
    203.6
  ],
  "greedy_iou": 0.9970530451866405,
  "excess": 0.06333333333333258
}
```

```json
{
  "key": "activitynet-timelens_00011",
  "query": "Two people are arm wrestling on a chair",
  "dataset": "activitynet-timelens",
  "duration": 51.233333333333334,
  "gt": [
    0,
    50
  ],
  "greedy": [
    0.0,
    51.3
  ],
  "greedy_iou": 0.9746588693957116,
  "excess": 0.06666666666666288
}
```

```json
{
  "key": "charades-timelens_01682",
  "query": "A person is holding another person",
  "dataset": "charades-timelens",
  "duration": 31.266666666666666,
  "gt": [
    23,
    31
  ],
  "greedy": [
    23.7,
    31.3
  ],
  "greedy_iou": 0.8795180722891567,
  "excess": 0.03333333333333499
}
```

```json
{
  "key": "charades-timelens_01923",
  "query": "A person is laughing",
  "dataset": "charades-timelens",
  "duration": 30.912988143881037,
  "gt": [
    14,
    29
  ],
  "greedy": [
    15.0,
    31.0
  ],
  "greedy_iou": 0.8235294117647058,
  "excess": 0.08701185611896278
}
```

### duplicate_pool

```json
{
  "key": "qvhighlights-timelens_00010",
  "query": "A kid in blue hoodie is sitting with a paper in front of him and then he starts writing on the paper",
  "dataset": "qvhighlights-timelens",
  "duration": 150.0,
  "gt": [
    0,
    142
  ],
  "greedy": [
    0.0,
    14.5
  ],
  "greedy_iou": 0.10211267605633803,
  "method": "nucleus_frontier@8192",
  "final": [
    0.0,
    15.0
  ],
  "final_iou": 0.1056338028169014,
  "original_pool": [
    {
      "id": 18,
      "span": [
        0.0,
        15.0
      ],
      "iou": 0.1056338028169014,
      "value": 0.8003354082491606
    },
    {
      "id": 19,
      "span": [
        0.0,
        15.0
      ],
      "iou": 0.1056338028169014,
      "value": 0.8003354082491606
    },
    {
      "id": 20,
      "span": [
        0.0,
        15.0
      ],
      "iou": 0.1056338028169014,
      "value": 0.8003354082491606
    },
    {
      "id": 21,
      "span": [
        0.0,
        15.0
      ],
      "iou": 0.1056338028169014,
      "value": 0.8003354082491606
    }
  ],
  "dedup_pool": [
    {
      "id": 18,
      "span": [
        0.0,
        15.0
      ],
      "iou": 0.1056338028169014,
      "value": 0.8003354082491606
    },
    {
      "id": 27,
      "span": [
        0.0,
        150.0
      ],
      "iou": 0.9466666666666667,
      "value": 0.6287617794006632
    },
    {
      "id": 5,
      "span": [
        0.0,
        10.0
      ],
      "iou": 0.07042253521126761,
      "value": 0.5952071631572305
    },
    {
      "id": 8,
      "span": [
        0.0,
        23.0
      ],
      "iou": 0.1619718309859155,
      "value": 0.5902210278696581
    }
  ],
  "dedup_pool_oracle": 0.9466666666666667
}
```

```json
{
  "key": "qvhighlights-timelens_00957",
  "query": "Woman in blue top and black skinny jeans poses for the camera",
  "dataset": "qvhighlights-timelens",
  "duration": 150.0,
  "gt": [
    96,
    105
  ],
  "greedy": [
    137.0,
    143.0
  ],
  "greedy_iou": 0.0,
  "method": "nucleus_frontier@8192",
  "final": [
    75.0,
    96.0
  ],
  "final_iou": 0.0,
  "original_pool": [
    {
      "id": 14,
      "span": [
        67.0,
        85.0
      ],
      "iou": 0.0,
      "value": 0.34478737630259615
    },
    {
      "id": 17,
      "span": [
        75.0,
        96.0
      ],
      "iou": 0.0,
      "value": 0.3312029095260308
    },
    {
      "id": 19,
      "span": [
        75.0,
        96.0
      ],
      "iou": 0.0,
      "value": 0.3312029095260308
    },
    {
      "id": 20,
      "span": [
        75.0,
        96.0
      ],
      "iou": 0.0,
      "value": 0.3312029095260308
    }
  ],
  "dedup_pool": [
    {
      "id": 14,
      "span": [
        67.0,
        85.0
      ],
      "iou": 0.0,
      "value": 0.34478737630259615
    },
    {
      "id": 17,
      "span": [
        75.0,
        96.0
      ],
      "iou": 0.0,
      "value": 0.3312029095260308
    },
    {
      "id": 18,
      "span": [
        97.0,
        106.0
      ],
      "iou": 0.8,
      "value": 0.1817035072630747
    },
    {
      "id": 6,
      "span": [
        106.0,
        119.0
      ],
      "iou": 0.0,
      "value": 0.16540354479989827
    }
  ],
  "dedup_pool_oracle": 0.8
}
```

```json
{
  "key": "qvhighlights-timelens_00312",
  "query": "A man in a black T-shirt and a cap is standing and talking besides a highway",
  "dataset": "qvhighlights-timelens",
  "duration": 150.0,
  "gt": [
    0,
    150
  ],
  "greedy": [
    0.0,
    149.5
  ],
  "greedy_iou": 0.9966666666666667,
  "method": "nucleus_frontier@8192",
  "final": [
    0.0,
    30.0
  ],
  "final_iou": 0.2,
  "original_pool": [
    {
      "id": 18,
      "span": [
        0.0,
        30.0
      ],
      "iou": 0.2,
      "value": 0.9000223985185185
    },
    {
      "id": 19,
      "span": [
        0.0,
        30.0
      ],
      "iou": 0.2,
      "value": 0.9000223985185185
    },
    {
      "id": 20,
      "span": [
        0.0,
        30.0
      ],
      "iou": 0.2,
      "value": 0.9000223985185185
    },
    {
      "id": 21,
      "span": [
        0.0,
        30.0
      ],
      "iou": 0.2,
      "value": 0.9000223985185185
    }
  ],
  "dedup_pool": [
    {
      "id": 18,
      "span": [
        0.0,
        30.0
      ],
      "iou": 0.2,
      "value": 0.9000223985185185
    },
    {
      "id": 2,
      "span": [
        0.0,
        150.0
      ],
      "iou": 1.0,
      "value": 0.6697573791517605
    },
    {
      "id": 5,
      "span": [
        0.0,
        35.0
      ],
      "iou": 0.23333333333333334,
      "value": 0.6726681885775385
    },
    {
      "id": 3,
      "span": [
        0.0,
        29.0
      ],
      "iou": 0.19333333333333333,
      "value": 0.6795139946880547
    }
  ],
  "dedup_pool_oracle": 1.0
}
```

```json
{
  "key": "qvhighlights-timelens_01049",
  "query": "Family opens Christmas presents together",
  "dataset": "qvhighlights-timelens",
  "duration": 150.0,
  "gt": [
    0,
    150
  ],
  "greedy": [
    0.0,
    150.3
  ],
  "greedy_iou": 0.998003992015968,
  "method": "nucleus_frontier@8192",
  "final": [
    0.0,
    23.0
  ],
  "final_iou": 0.15333333333333332,
  "original_pool": [
    {
      "id": 2,
      "span": [
        0.0,
        23.0
      ],
      "iou": 0.15333333333333332,
      "value": 0.3547939806345176
    },
    {
      "id": 16,
      "span": [
        0.0,
        23.0
      ],
      "iou": 0.15333333333333332,
      "value": 0.3547939806345176
    },
    {
      "id": 17,
      "span": [
        0.0,
        23.0
      ],
      "iou": 0.15333333333333332,
      "value": 0.3547939806345176
    },
    {
      "id": 18,
      "span": [
        0.0,
        23.0
      ],
      "iou": 0.15333333333333332,
      "value": 0.3547939806345176
    }
  ],
  "dedup_pool": [
    {
      "id": 2,
      "span": [
        0.0,
        23.0
      ],
      "iou": 0.15333333333333332,
      "value": 0.3547939806345176
    },
    {
      "id": 15,
      "span": [
        9.0,
        149.3
      ],
      "iou": 0.9353333333333335,
      "value": 0.08926797316812575
    }
  ],
  "dedup_pool_oracle": 0.9353333333333335
}
```

### selection

```json
{
  "key": "qvhighlights-timelens_00156",
  "query": "A man in a striped shirt wearing sunglasses sits in the car with his arms crossed, talking",
  "dataset": "qvhighlights-timelens",
  "duration": 150.0,
  "gt": [
    98,
    108
  ],
  "greedy": [
    138.0,
    148.0
  ],
  "greedy_iou": 0.0,
  "method": "nucleus_frontier@8192",
  "final": [
    138.7,
    148.7
  ],
  "final_iou": 0.0,
  "pool": [
    {
      "span": [
        98.0,
        108.0
      ],
      "iou": 1.0,
      "value": 0.3279835950633775
    },
    {
      "span": [
        137.9,
        148.0
      ],
      "iou": 0.0,
      "value": 0.3181973764055184
    },
    {
      "span": [
        87.0,
        107.0
      ],
      "iou": 0.42857142857142855,
      "value": 0.26244900656907943
    },
    {
      "span": [
        138.7,
        148.7
      ],
      "iou": 0.0,
      "value": 0.30408143217865247
    }
  ],
  "oracle_pool": 1.0
}
```

```json
{
  "key": "activitynet-timelens_01085",
  "query": "A man in blue clothes stands nearby watching another man lift a barbell",
  "dataset": "activitynet-timelens",
  "duration": 88.79,
  "gt": [
    0,
    29
  ],
  "greedy": [
    0.0,
    28.5
  ],
  "greedy_iou": 0.9827586206896551,
  "method": "nucleus_frontier@8192",
  "final": [
    28.0,
    88.79
  ],
  "final_iou": 0.011262529564140105,
  "pool": [
    {
      "span": [
        0.0,
        1.0
      ],
      "iou": 0.034482758620689655,
      "value": 0.393152504404457
    },
    {
      "span": [
        0.0,
        29.0
      ],
      "iou": 1.0,
      "value": 0.2915472146425106
    },
    {
      "span": [
        28.0,
        88.79
      ],
      "iou": 0.011262529564140105,
      "value": 0.2538191675263655
    },
    {
      "span": [
        9.0,
        28.0
      ],
      "iou": 0.6551724137931034,
      "value": 0.25172346747168917
    }
  ],
  "oracle_pool": 1.0
}
```

```json
{
  "key": "activitynet-timelens_04304",
  "query": "She clips her hair back and brushes her bangs",
  "dataset": "activitynet-timelens",
  "duration": 218.75186666666667,
  "gt": [
    84,
    131
  ],
  "greedy": [
    128.0,
    170.0
  ],
  "greedy_iou": 0.03488372093023256,
  "method": "nucleus_frontier@8192",
  "final": [
    200.0,
    216.0
  ],
  "final_iou": 0.0,
  "pool": [
    {
      "span": [
        87.0,
        132.0
      ],
      "iou": 0.9166666666666666,
      "value": 0.2590592433013162
    },
    {
      "span": [
        200.0,
        216.0
      ],
      "iou": 0.0,
      "value": 0.20409216612401007
    },
    {
      "span": [
        135.0,
        170.0
      ],
      "iou": 0.0,
      "value": 0.19672147785249625
    },
    {
      "span": [
        77.0,
        132.0
      ],
      "iou": 0.8545454545454545,
      "value": 0.24940254063643763
    }
  ],
  "oracle_pool": 0.9166666666666666
}
```

```json
{
  "key": "activitynet-timelens_00910",
  "query": "He gets to the other side and goes back across",
  "dataset": "activitynet-timelens",
  "duration": 81.467,
  "gt": [
    21,
    45
  ],
  "greedy": [
    31.0,
    45.0
  ],
  "greedy_iou": 0.5833333333333334,
  "method": "nucleus_frontier@8192",
  "final": [
    50.0,
    81.3
  ],
  "final_iou": 0.0,
  "pool": [
    {
      "span": [
        7.0,
        22.0
      ],
      "iou": 0.02631578947368421,
      "value": 0.2729025146097626
    },
    {
      "span": [
        50.0,
        81.3
      ],
      "iou": 0.0,
      "value": 0.22601097809991172
    },
    {
      "span": [
        18.0,
        45.0
      ],
      "iou": 0.8888888888888888,
      "value": 0.23031421193090879
    },
    {
      "span": [
        9.0,
        22.0
      ],
      "iou": 0.027777777777777776,
      "value": 0.26908355473817847
    }
  ],
  "oracle_pool": 0.8888888888888888
}
```

### prune

```json
{
  "key": "activitynet-timelens_01396",
  "query": "A boy is wearing boxing gloves practicing boxing",
  "dataset": "activitynet-timelens",
  "duration": 94.12736666666667,
  "gt": [
    61,
    87
  ],
  "greedy": [
    62.0,
    87.0
  ],
  "greedy_iou": 0.9615384615384616,
  "method": "nucleus_frontier@8192",
  "pruned": [
    59.8,
    87.5
  ],
  "pruned_iou": 0.9386281588447652,
  "final": null,
  "final_iou": 0.0,
  "assessment": {
    "event": 0.9999724643056137,
    "conflict": 0.9770226281664712,
    "boundary": 0.9998601779423897,
    "value": 0.0,
    "eligible": false,
    "blocked": true,
    "reason": "model_confirmed_contradiction",
    "calls": [
      "9b06eb4e3f76889b6c726efca4c10e239be18013432c7c8a994c26c3147dbfb9",
      "0c1cf7ddea1053ff3f9b7aa0cb931ab5df424d87a7d19a09bb2034055a6f5f92",
      "ab720f1ebbdb1a9a43fbe53fda7f76006149d5b3be137df95320884a392d07af",
      "76c1838395b92e2c7c987291c0bd3f79c921935d5de1dacbe4ad1ed4877ed2ac"
    ]
  }
}
```

```json
{
  "key": "activitynet-timelens_01396",
  "query": "A boy is wearing boxing gloves practicing boxing",
  "dataset": "activitynet-timelens",
  "duration": 94.12736666666667,
  "gt": [
    61,
    87
  ],
  "greedy": [
    62.0,
    87.0
  ],
  "greedy_iou": 0.9615384615384616,
  "method": "nucleus_frontier@8192",
  "pruned": [
    59.0,
    87.0
  ],
  "pruned_iou": 0.9285714285714286,
  "final": null,
  "final_iou": 0.0,
  "assessment": {
    "event": 0.9999724643056137,
    "conflict": 0.9770226281664712,
    "boundary": 0.9998601779423897,
    "value": 0.0,
    "eligible": false,
    "blocked": true,
    "reason": "model_confirmed_contradiction",
    "calls": [
      "9b06eb4e3f76889b6c726efca4c10e239be18013432c7c8a994c26c3147dbfb9",
      "0c1cf7ddea1053ff3f9b7aa0cb931ab5df424d87a7d19a09bb2034055a6f5f92",
      "ab720f1ebbdb1a9a43fbe53fda7f76006149d5b3be137df95320884a392d07af",
      "76c1838395b92e2c7c987291c0bd3f79c921935d5de1dacbe4ad1ed4877ed2ac"
    ]
  }
}
```

```json
{
  "key": "qvhighlights-timelens_00020",
  "query": "Different types of meat products are being cut, shaped and prepared",
  "dataset": "qvhighlights-timelens",
  "duration": 150.0,
  "gt": [
    31,
    51
  ],
  "greedy": [
    32.0,
    53.0
  ],
  "greedy_iou": 0.8636363636363636,
  "method": "nucleus_frontier@8192",
  "pruned": [
    29.0,
    52.0
  ],
  "pruned_iou": 0.8695652173913043,
  "final": [
    50.0,
    53.0
  ],
  "final_iou": 0.045454545454545456,
  "assessment": {
    "event": 0.9991959140377139,
    "conflict": 0.9740426448743533,
    "boundary": 0.9980732650230317,
    "value": 0.0,
    "eligible": false,
    "blocked": true,
    "reason": "model_confirmed_contradiction",
    "calls": [
      "8c71b63aa732087edd9fc827c1c37c85b9dd5a695ec81ca1dce7b725e3554e7e",
      "aad814ebd637ab4f886416838b6f696e205577b14dd58d3e68664cdcf2a5e1c6",
      "e7c214ba00aa62cdb771f1d74ba5bd99430fe5c76daf2d1287c794d9d91dcdf5",
      "71c22f6da23fb4187cffd7f3ae226102bd311a21be0f18ac6c5dbe14b79a069f"
    ]
  }
}
```

### new_candidate_unused

```json
{
  "key": "charades-timelens_00420",
  "query": "A woman lies down on the bed",
  "dataset": "charades-timelens",
  "duration": 31.77689136191466,
  "gt": [
    0,
    3
  ],
  "greedy": [
    0.0,
    4.2
  ],
  "greedy_iou": 0.7142857142857143,
  "method": "nucleus_frontier@8192",
  "final": [
    3.0,
    8.0
  ],
  "final_iou": 0.0,
  "root": [
    3.0,
    8.0
  ],
  "best_proposals": [
    {
      "span": [
        0.0,
        4.2
      ],
      "source": "token_seed",
      "iou": 0.7142857142857143,
      "assessment": {
        "event": 0.9890130548090553,
        "conflict": 0.43782349911420193,
        "boundary": 0.5,
        "value": 0.41699992386219653,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "ebc47a84e2962ed65d7e2bd0c83a2acac0abf2e3babdd7b25a6c2f6ceeaf38f6",
          "8a9c8d13427e016ef259fe8ace47de354cfd2dfde0a784ed828a18956deeb776"
        ]
      }
    },
    {
      "span": [
        0.0,
        5.2
      ],
      "source": "token_seed",
      "iou": 0.5769230769230769,
      "assessment": {
        "event": 0.9875683510791508,
        "conflict": 0.4378234991142019,
        "boundary": 0.5,
        "value": 0.4163907899964258,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "b3c31af031bc9035d66e1e898aa66ab066f8fdeea46c2459b4d1fadc5f2c19cb",
          "ac56d8667b81ee2e71c54dd60aac7d68e2c8d0853ca37947a44c048fea8229ec"
        ]
      }
    },
    {
      "span": [
        0.0,
        6.0
      ],
      "source": "token_seed",
      "iou": 0.5,
      "assessment": {
        "event": 0.9890130585939225,
        "conflict": 0.32082130731838393,
        "boundary": 0.5,
        "value": 0.5037874471356502,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "62645172bf1156176ea3d9e9dc13a8de0fc6d9ab95064c87f69a3d322ea65009",
          "9c79d96cb3a1b0b37cb53657c61e727d834b052440f921a75dd63601ed584994"
        ]
      }
    },
    {
      "span": [
        0.0,
        7.0
      ],
      "source": "token_seed",
      "iou": 0.42857142857142855,
      "assessment": {
        "event": 0.9914225128043136,
        "conflict": 0.34864513533394575,
        "boundary": 0.5,
        "value": 0.4843259074908999,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "e9fe18261bc699146a75dff5c8213b03d1820bb4b3fd858f8090b47ac739ba3a",
          "a7a8f973d74bc84ac8f70bb2e773fd5e3898ea0cd99a4392fd7f41b448f661cd"
        ]
      }
    }
  ]
}
```

```json
{
  "key": "charades-timelens_00915",
  "query": "A person opened the door",
  "dataset": "charades-timelens",
  "duration": 26.966666666666665,
  "gt": [
    2,
    5
  ],
  "greedy": [
    2.7,
    7.2
  ],
  "greedy_iou": 0.44230769230769224,
  "method": "nucleus_frontier@8192",
  "final": [
    6.0,
    11.0
  ],
  "final_iou": 0.0,
  "root": [
    6.0,
    11.0
  ],
  "best_proposals": [
    {
      "span": [
        1.8,
        6.0
      ],
      "source": "token_seed",
      "iou": 0.7142857142857143,
      "assessment": {
        "event": 0.9859363707872744,
        "conflict": 0.679178699175393,
        "boundary": 0.5,
        "value": 0.23723204175469909,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "0917f3d00a6dff0cc4a41d0f60419cc9004e38fa3800d3b52d2301562058b8e7",
          "190989ca1ca2f517b353da4b0b23208308b8256d40b54b1a611b4f4965e7ea96"
        ]
      }
    },
    {
      "span": [
        2.5,
        5.5
      ],
      "source": "semantic_proposal",
      "iou": 0.7142857142857143,
      "assessment": {
        "event": 0.9959298631863726,
        "conflict": 0.817574471748733,
        "boundary": 0.5,
        "value": 0.13626227354473958,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "ebce1f0d9d129347e8d0199ac2cba86b0ba386648ce0c002790ef58c47349861",
          "37ce83e8c30d70830cbb83e0dcdea2699c00c0cd68f4912c3cb5bd19dd4a86a5",
          "7da2ce707adb868d1c8d1624e9e82be04b9777cde00ede5e2ff2b3f8f8ebf90e"
        ]
      }
    },
    {
      "span": [
        1.8,
        6.2
      ],
      "source": "leading_digit",
      "iou": 0.6818181818181818,
      "assessment": {
        "event": 0.9875683509305094,
        "conflict": 0.7981867729389246,
        "boundary": 0.5,
        "value": 0.14947826688350302,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "e925a6c7d1c5a23ddb63377e17965daff2c0bfa9900464c98fefb4cb74f78246",
          "59c9188eb67e1c4a171ffaab3881722be8d471b5a5ba7b64d865f0b7845dbe0c"
        ]
      }
    },
    {
      "span": [
        0.0,
        5.0
      ],
      "source": "token_seed",
      "iou": 0.6,
      "assessment": {
        "event": 0.9669140232797719,
        "conflict": 0.5621765008857981,
        "boundary": 0.5,
        "value": 0.3175032607612055,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "c29b0e09945cb63df11cb1b85760d34fb05024d9af848b291d719257dbe8b1cb",
          "cfd757aa6d3d365a443a70e0765acd749ccb31298782b3258cf565cd3258005e"
        ]
      }
    }
  ]
}
```

```json
{
  "key": "charades-timelens_01495",
  "query": "A person takes the towel off their shoulder",
  "dataset": "charades-timelens",
  "duration": 32.69933333333333,
  "gt": [
    0,
    2
  ],
  "greedy": [
    0.0,
    4.0
  ],
  "greedy_iou": 0.5,
  "method": "nucleus_frontier@8192",
  "final": [
    5.0,
    10.0
  ],
  "final_iou": 0.0,
  "root": [
    5.0,
    10.0
  ],
  "best_proposals": [
    {
      "span": [
        0.0,
        2.0
      ],
      "source": "token_seed",
      "iou": 1.0,
      "assessment": {
        "event": 0.9241418158002492,
        "conflict": 0.7057850340255386,
        "boundary": 0.5,
        "value": 0.2039222646684355,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "4cf4bf9fca3c7317fbd4a8d4cb9bf56ae72b7891532ab65b2c3f71d882387d71",
          "376bf02f1c5e7a6ec505aca1ce720083478a51ab09c4b8548abc05f29d5bf87d"
        ]
      }
    },
    {
      "span": [
        0.0,
        3.0
      ],
      "source": "token_seed",
      "iou": 0.6666666666666666,
      "assessment": {
        "event": 0.9046505395995927,
        "conflict": 0.679178699175393,
        "boundary": 0.5,
        "value": 0.21767337217951802,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "e3fcab8f2e1b1e2ce566e848d32e195cc46376ded7ed7b9bca0e5f06b0443b54",
          "dfc295387ce845a01ae5c031ce5e9de23c5706a03b5ab58516e00ac335399eda"
        ]
      }
    },
    {
      "span": [
        0.0,
        4.0
      ],
      "source": "token_seed",
      "iou": 0.5,
      "assessment": {
        "event": 0.9149009491921664,
        "conflict": 0.7310585669110202,
        "boundary": 0.5,
        "value": 0.1845410793076568,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "dbf6f853d51ae741e4e53c9ac5f59651889a1257d16837a0cee6eecbd21c6b49",
          "725706b9e59bdf0cbac2b21960e7a10bc4a4352b0cf3abe2a8197c6d20636a41"
        ]
      }
    },
    {
      "span": [
        0.0,
        4.2
      ],
      "source": "token_seed",
      "iou": 0.47619047619047616,
      "assessment": {
        "event": 0.9399133540338004,
        "conflict": 0.7310585786300049,
        "boundary": 0.5,
        "value": 0.18958622504886727,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "5551569298aadc3e247829331c64daa053a512f29b146fa1f68c72220f342c1d",
          "2366d02a648a9d5d7997ed159f9b25060fa823a8cd5e7195b67da30a5407253d"
        ]
      }
    }
  ]
}
```

```json
{
  "key": "charades-timelens_01719",
  "query": "person starts laughing",
  "dataset": "charades-timelens",
  "duration": 30.597233333333335,
  "gt": [
    4,
    9
  ],
  "greedy": [
    5.5,
    11.0
  ],
  "greedy_iou": 0.5,
  "method": "nucleus_frontier@8192",
  "final": [
    14.0,
    20.0
  ],
  "final_iou": 0.0,
  "root": [
    14.0,
    20.0
  ],
  "best_proposals": [
    {
      "span": [
        3.0,
        10.0
      ],
      "source": "leading_digit",
      "iou": 0.7142857142857143,
      "assessment": {
        "event": 0.9324533097989076,
        "conflict": 0.6224593382055109,
        "boundary": 0.5,
        "value": 0.264029279755456,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "bc2ba016ae936acdd529f8ad3390a8c37ed231b92a21b9ae5f5ff2afd8dc45da",
          "10a0227b9581d5619afce7556b2969a627e9af452dd3c179ce1095e3f87e4af9"
        ]
      }
    },
    {
      "span": [
        4.5,
        10.5
      ],
      "source": "leading_digit",
      "iou": 0.6923076923076923,
      "assessment": {
        "event": 0.9740426434615237,
        "conflict": 0.7310585844894971,
        "boundary": 0.5,
        "value": 0.1964703054751007,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "be938930b7a0bc91efa9f069495d5da42d9927be0ceb1582ebbfe2a08f028d0f",
          "bc9199f6712b8411ebb3c5da76ef67335f77bb50d38759ab175c267a05f582a8"
        ]
      }
    },
    {
      "span": [
        5.0,
        10.0
      ],
      "source": "token_seed",
      "iou": 0.6666666666666666,
      "assessment": {
        "event": 0.9875683509305094,
        "conflict": 0.7772998714925403,
        "boundary": 0.5,
        "value": 0.16494869899659337,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "e91c4c993c123716813deb2c855f65659ae76ddca95bf7ca9325c712e7767dd3",
          "83f99167ff483d069f91a44d85d965b5a2be56cab54b98dfff49c9264624ed7d"
        ]
      }
    },
    {
      "span": [
        2.0,
        10.0
      ],
      "source": "leading_digit",
      "iou": 0.625,
      "assessment": {
        "event": 0.8670357649558006,
        "conflict": 0.5312093733737563,
        "boundary": 0.5,
        "value": 0.3048436796707457,
        "eligible": true,
        "blocked": false,
        "reason": null,
        "calls": [
          "ff72aa78203284c5d0efae064ac8ee9914f268fe643577619dfa6ebf894ccbfa",
          "a5433fb49c4c284f07a61a80af591e6481ef87bff81f3dbb0f8b06920662f95a"
        ]
      }
    }
  ]
}
```

### requested_reference

```json
{
  "key": "qvhighlights-timelens_00507",
  "query": "A man handles a two headed snake",
  "dataset": "qvhighlights-timelens",
  "duration": 150.0,
  "gt": [
    115,
    146
  ],
  "greedy": [
    0.0,
    1.3
  ],
  "greedy_iou": 0.0,
  "proposals": [
    {
      "span": [
        0.0,
        1.3
      ],
      "source": "token_seed",
      "provenance": [
        {
          "source": "token_seed",
          "call": "baseline:1",
          "evidence": ""
        },
        {
          "source": "leading_digit",
          "call": "3d70535f7614deccd78e62dabfc208081ffcc5e15548e39d7152752a3a4a895d",
          "evidence": ""
        }
      ],
      "iou": 0.0,
      "assessment": {
        "event": 0.11920293297380238,
        "conflict": 0.095349472611171,
        "boundary": 0.5,
        "value": 0.0,
        "eligible": false,
        "blocked": false,
        "reason": null,
        "calls": [
          "2d1ce6188330b9b0df28c80c22cf1cf84ab934fde23e5a977eb68b0035f790d1",
          "592dac166cefa76a571307268b44e65b9760278a00c88ed660cfb750765a9da0"
        ]
      }
    },
    {
      "span": [
        0.0,
        2.0
      ],
      "source": "token_seed",
      "provenance": [
        {
          "source": "token_seed",
          "call": "baseline:2",
          "evidence": ""
        }
      ],
      "iou": 0.0,
      "assessment": {
        "event": 0.07585818002124356,
        "conflict": 0.06008665690650098,
        "boundary": 0.5,
        "value": 0.0,
        "eligible": false,
        "blocked": false,
        "reason": null,
        "calls": [
          "a6687e36aa8e24c7e58287636ae8bfaf0fb3aba87c8b1c6bd7c90ec252d52699",
          "56dc1b4e22c2c31c8d9d3ed1035b9095ab6439e482d5d60262e9239339cd9726"
        ]
      }
    },
    {
      "span": [
        0.0,
        3.0
      ],
      "source": "token_seed",
      "provenance": [
        {
          "source": "token_seed",
          "call": "baseline:3",
          "evidence": ""
        }
      ],
      "iou": 0.0,
      "assessment": {
        "event": 0.05340332904654947,
        "conflict": 0.0474258782264522,
        "boundary": 0.5,
        "value": 0.0,
        "eligible": false,
        "blocked": false,
        "reason": null,
        "calls": [
          "aab943338ddec857cd64381bcf341532e43bb059a9b4eb7badffd5a992206e95",
          "8f51cf8222fa194d9ea901ebf4056d52d3d7956c2e44f0ff654b4ce2ae7fc5e8"
        ]
      }
    },
    {
      "span": [
        1.0,
        10.0
      ],
      "source": "token_seed",
      "provenance": [
        {
          "source": "token_seed",
          "call": "baseline:4",
          "evidence": ""
        },
        {
          "source": "leading_digit",
          "call": "9a33b936af2fc9640f944bfbdf0a3bcdd5640ac252523291cf918f5d4b36247a",
          "evidence": ""
        }
      ],
      "iou": 0.0,
      "assessment": {
        "event": 0.047425871494605094,
        "conflict": 0.04208772551257712,
        "boundary": 0.5,
        "value": 0.0,
        "eligible": false,
        "blocked": false,
        "reason": null,
        "calls": [
          "4cabb86b6ad1924bf1dd940ce96b4992d75c1cfc72642729373ea5ef4a06bb75",
          "d1e369115874e76a765f604ecbc3db005b6af59ce7f46356158544e06d53c8e2"
        ]
      }
    },
    {
      "span": [
        0.0,
        4.0
      ],
      "source": "token_seed",
      "provenance": [
        {
          "source": "token_seed",
          "call": "baseline:5",
          "evidence": ""
        }
      ],
      "iou": 0.0,
      "assessment": {
        "event": 0.03732688814730781,
        "conflict": 0.04742586846527418,
        "boundary": 0.5,
        "value": 0.0,
        "eligible": false,
        "blocked": false,
        "reason": null,
        "calls": [
          "1654f404b8f055d29f867b6ac60477733f970b37fdcb1465b5180f11af24be66",
          "f54092cec9024561f2d92338ca204d8d905b939a17ebf08d76645c1337877f6b"
        ]
      }
    },
    {
      "span": [
        2.0,
        10.0
      ],
      "source": "token_seed",
      "provenance": [
        {
          "source": "token_seed",
          "call": "baseline:6",
          "evidence": ""
        },
        {
          "source": "leading_digit",
          "call": "d2b1719d516c682700c1baa108e2045feec39e2d19a638a986d72f71934e6973",
          "evidence": ""
        }
      ],
      "iou": 0.0,
      "assessment": {
        "event": 0.033085982202364056,
        "conflict": 0.04208772371029592,
        "boundary": 0.5,
        "value": 0.0,
        "eligible": false,
        "blocked": false,
        "reason": null,
        "calls": [
          "677f1600cb5bc7d3e8b524224ee35a7da337d0c69afc4d84fba3c738c79984b2",
          "353f84975668fd30229dc7578efb3b01c74971340cb3e25dc579a62278eb4374"
        ]
      }
    },
    {
      "span": [
        147.0,
        150.3
      ],
      "source": "token_seed",
      "provenance": [
        {
          "source": "token_seed",
          "call": "baseline:7",
          "evidence": ""
        }
      ],
      "iou": 0.0,
      "assessment": {
        "event": 0.0,
        "conflict": 1.0,
        "boundary": 0.0,
        "value": 0.0,
        "eligible": false,
        "blocked": true,
        "reason": "invalid_or_out_of_duration",
        "calls": []
      }
    },
    {
      "span": [
        0.0,
        1.5
      ],
      "source": "token_seed",
      "provenance": [
        {
          "source": "token_seed",
          "call": "baseline:8",
          "evidence": ""
        }
      ],
      "iou": 0.0,
      "assessment": {
        "event": 0.07585817270885642,
        "conflict": 0.07585817688736327,
        "boundary": 0.5,
        "value": 0.0,
        "eligible": false,
        "blocked": false,
        "reason": null,
        "calls": [
          "a34e51b3f98363dc7bcc4242847c55738dac6400c5aaedae12233d6c9e41e483",
          "b837491cd2371539da97e08f6fb1e01fba5f88bd59332a1106c4c0d0b273b339"
        ]
      }
    }
  ],
  "leading_digit_paths": [
    {
      "digit": "0",
      "span": [
        0.0,
        1.3
      ],
      "answer": "The event happens in 0.0 - 1.3 seconds seconds."
    },
    {
      "digit": "1",
      "span": [
        1.0,
        10.0
      ],
      "answer": "The event happens in 1.0 - 10.0 seconds."
    },
    {
      "digit": "2",
      "span": [
        2.0,
        10.0
      ],
      "answer": "The event happens in 2.0 - 10.0 seconds."
    },
    {
      "digit": "3",
      "span": [
        3.0,
        20.0
      ],
      "answer": "The event happens in 3.0 - 20.0 seconds."
    },
    {
      "digit": "4",
      "span": [
        4.0,
        20.0
      ],
      "answer": "The event happens in 4.0 - 20.0 seconds."
    },
    {
      "digit": "5",
      "span": [
        5.0,
        19.0
      ],
      "answer": "The event happens in 5.0 - 19.0 seconds."
    },
    {
      "digit": "6",
      "span": [
        6.0,
        19.0
      ],
      "answer": "The event happens in 6.0 - 19.0 seconds."
    },
    {
      "digit": "7",
      "span": [
        7.8,
        19.5
      ],
      "answer": "The event happens in 7.8 - 19.5 seconds."
    },
    {
      "digit": "8",
      "span": [
        80.0,
        95.0
      ],
      "answer": "The event happens in 80.0 - 95.0 seconds."
    },
    {
      "digit": "9",
      "span": [
        9.0,
        15.0
      ],
      "answer": "The event happens in 9.0 - 15.0 seconds."
    }
  ],
  "after_start_one": {
    "step": 0,
    "id": 13,
    "token": ".",
    "probability": 0.4030792500791484,
    "cumulative_log_probability": -0.9086220860481262,
    "top": [
      {
        "id": 13,
        "token": ".",
        "probability": 0.4030792500791484
      },
      {
        "id": 19,
        "token": "4",
        "probability": 0.31391845431247284
      },
      {
        "id": 16,
        "token": "1",
        "probability": 0.1680284402937676
      },
      {
        "id": 17,
        "token": "2",
        "probability": 0.04814095415310046
      },
      {
        "id": 15,
        "token": "0",
        "probability": 0.017710063088894043
      },
      {
        "id": 18,
        "token": "3",
        "probability": 0.015629075820526764
      },
      {
        "id": 20,
        "token": "5",
        "probability": 0.008365641450126766
      },
      {
        "id": 24,
        "token": "9",
        "probability": 0.007382652667870174
      }
    ],
    "numeric_probabilities": {
      "0": 0.017710063088894043,
      "1": 0.1680284402937676,
      "2": 0.04814095415310046,
      "3": 0.015629075820526764,
      "4": 0.31391845431247284,
      "5": 0.008365641450126766,
      "6": 0.00447780519307253,
      "7": 0.006515168112253328,
      "8": 0.005749615678881488,
      "9": 0.007382652667870174,
      ".": 0.4030792500791484,
      ":": 0.0009991333898437442
    }
  },
  "old32_best": {
    "index": 14,
    "span": [
      116.0,
      145.0
    ],
    "iou": 0.9354838709677419,
    "branch": {
      "position": 6,
      "roles": [
        "start.seconds.decimal_separator"
      ],
      "original_token": 13,
      "forced_token": 16,
      "raw_probability": 0.16953766954759197,
      "parent": 4,
      "forced_prefix_ids": [
        785,
        1538,
        8573,
        304,
        220,
        16,
        16
      ],
      "target_roles": [
        "start.seconds.10^1"
      ]
    }
  },
  "semantic_output": "The event happens in 147.0 - 150.0 seconds.",
  "methods": {
    "likelihood_best_first": null,
    "value_best_first": null,
    "value_beam": null,
    "diverse_beam": null,
    "uct": null,
    "puct": null,
    "neural_controller": null,
    "self_refine": null,
    "submodular_coverage": null,
    "reward_balanced": null,
    "sequential_halving": null,
    "nucleus_frontier": null,
    "typicality_frontier": null,
    "commit_then_refine": null,
    "token_seed_best_first": null
  }
}
```
