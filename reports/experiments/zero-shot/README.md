# Zero-shot 평가와 생성 실험

[Temporal mode search · 누적 pilot300·600·900](results/rollout/temporal_mode_search/viewer.html): 확률적 분기3방법·seed42/43·총 rollout32/64/128/256을 비교한다. 기존600 응답을 보존하고 고정900으로 확장 중이며, pilot300·600은 같은 원자료의 고정 부분집합이다. [누적 보고서](results/rollout/temporal_mode_search/pilot/overview/capped_rollouts/REPORT.md)에서 zero-shot 대비 성능과 independent trace work,main/tail/judge의 실제 CUDA·wall 시간 및 memory를 비교한다. [설계 원리와 기대 효과](results/rollout/temporal_mode_search/shared/design_principles/REPORT.md)에서 방법과 예외 처리를 쉽게 설명한다.

[Event proposal and tree search · pilot900](results/rollout/event_tree_search/viewer.html): 넓은 모델 기반 사건 proposal, boundary refinement, contradiction backtracking을15개 정책·두 search token 예산에서 비교한다. 설계·수식·문헌은 [SPEC.md](results/rollout/event_tree_search/pilot/900/SPEC.md), 실행 상태는 대응 artifacts의 metadata/status.json을 참조한다.

[Branch feedback 후속 실험 설계](results/rollout/deterministic/shared/reconsideration/NEXT_EXPERIMENT.md): 재검토 후 실제 conditional rollout을 새로 수행하는 pilot600 설계. 현재 실행 결과와 구분한다.

[전체 실험 목록](../index.html)에서 아래 주제별로 탐색한다. 같은 질문의 선행·후속 실험을 함께 묶고, pilot/full 및 실행 버전은 구분한다. 실행 ID는 유지한다. `<주제>/<실험 계열>/full/` 또는 `pilot/`로 실제 폴더를 나누고, `results/`와 `artifacts/runs/`의 상대 경로를 맞췄다. 전체는 full9404, pilot은 질문 수를 명시한다. 저장·재사용 방법은 [산출물 저장소 안내](artifacts/README.md)를 따른다.


## 주제 → 실험 계열 → 전체·pilot

```text
results/                         artifacts/runs/도 같은 상대 경로
├── uncertainty/
│   ├── token_analysis/
│   │   ├── full/9404/
│   │   └── pilot/{300,600,900,overview}/
│   └── span_head/pilot/{300,overview}/
├── sampling/
│   ├── fps_spatial_control/pilot/{300,overview}/
│   └── spatial_allocation/{budget_comparison,allocation_sweep}/pilot/
├── baseline/
│   ├── thinking_vs_instruct/pilot/{300,overview}/
│   ├── input_order/overview/     전체 full9404·pilot300 비교
│   └── all_zeroshot/overview/    전체 full9404·pilot300 집계
└── operations/<진단 계열>/{full,pilot,shared}/
```

없는 단계의 결과는 만들지 않는다. pilot300 ⊂ pilot600 ⊂ pilot900을 유지하고, 각 실험 계열의 `pilot/overview/`는 등록된 단계의 보고서·동결 코호트를 참조한다. 원자료를 복제하거나 겹치는 질문을 합산하지 않는다. 조건이 다른 버전은 별도 계열로 둔다. 새 단계 등록 후 공통 `build_index.py`가 누적 목록을 자동 갱신한다. Rollout도 같은 구조로 통합했다.

## 1. 기준 성능·입력 조건

질문: 모델과 input order를 바꾸면 기준 성능과 예측 일관성이 어떻게 달라지는가?

| 실험 흐름 | 자료 |
|---|---|
| 기존 실행 전체 full9404·pilot300 통합 집계: 실행 버전·FPS·input order | [Zero-shot evaluation](results/baseline/all_zeroshot/overview/REPORT.md) |
| input order와 모델별 성능 → 같은 질문의 예측 일관성 | [Input-order consistency](results/baseline/input_order/overview/REPORT.md) · [일관성 분석](results/baseline/input_order/overview/consistency/REPORT.md) · [영상 비교](results/baseline/input_order/viewer.html?view=overview%2Fconsistency%2Freport.html) |
| Thinking vs Instruct 동일 pilot300 비교 | [Thinking vs Instruct / pilot300](results/baseline/thinking_vs_instruct/pilot/300/REPORT.md) |

## 2. 토큰 불확실성·오류 진단

질문: 숫자를 생성하는 어느 단계에서 정답 가능성을 잃으며, 토큰 확률과 entropy가 이를 얼마나 설명하는가?

| 실험 흐름 | 자료 |
|---|---|
| Full-head top-30 분포 계측: pilot300 | [timelens_top30 보고서](results/uncertainty/token_analysis/pilot/300/REPORT.md) · [토큰 뷰어](results/uncertainty/token_analysis/viewer.html?view=pilot%2F300%2Freport.html) |
| entropy와 IoU → prefix 정답 가능성 → 숫자/기타 토큰 비교 | [entropy_iou](results/uncertainty/token_analysis/pilot/300/entropy_iou/REPORT.md) · [prefix_entropy](results/uncertainty/token_analysis/pilot/300/prefix_entropy/REPORT.md) · [numeric_uncertainty](results/uncertainty/token_analysis/pilot/300/numeric_uncertainty/REPORT.md) · [통합 해석](results/uncertainty/token_analysis/pilot/300/integrated/REPORT.md) · [통합 뷰어](results/uncertainty/token_analysis/viewer.html?view=pilot%2F300%2Fintegrated%2Freport.html) |
| 자릿수별 정답 가능성: pilot300 → 전체 full9404 확장 | [pilot300](results/uncertainty/token_analysis/pilot/300/digit_correctness/REPORT.md) · [전체 full9404](results/uncertainty/token_analysis/full/9404/digit_correctness/REPORT.md) |
| 정답 가능 상태의 세 축 분석: pilot300 → 전체 full9404 | [pilot300 blue_analysis](results/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/REPORT.md) · [전체 full9404 blue_analysis](results/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/REPORT.md) |
| 최초 오류 entropy와 최종 GT 거리 | [first_error_distance](results/uncertainty/token_analysis/full/9404/digit_correctness/first_error_distance/REPORT.md) |
| 숫자·구간 전용 head로 생성 제약 비교: pilot300 | [timelens_span_head 보고서](results/uncertainty/span_head/pilot/300/REPORT.md) |

pilot과 full은 같은 분석의 확장이다. prefix의 IoU 가능성 판정과 endpoint 반올림 정답 가능성 판정은 기준이 다르므로 각 보고서 정의를 따른다.

## 3. 영상 샘플링·공간 통제

질문: spatial resolution를 통제한 상태에서 프레임의 수와 시간 배정을 바꾸면 성능이 어떻게 달라지는가?

| 실험 흐름 | 자료 |
|---|---|
| 선행 공간 예산 비교: Uniform/GT전체8배/Random | [Budget comparison / pilot900](results/sampling/spatial_allocation/budget_comparison/pilot/900/REPORT.md) · [pilot300](results/sampling/spatial_allocation/budget_comparison/pilot/300/REPORT.md) · [pilot600](results/sampling/spatial_allocation/budget_comparison/pilot/600/REPORT.md) |
| 후속 전체·경계 배정 sweep: 공간 low/high × A/B | [Allocation sweep / pilot900](results/sampling/spatial_allocation/allocation_sweep/pilot/900/REPORT.md) · [사양](results/sampling/spatial_allocation/allocation_sweep/pilot/900/SPEC.md) · [현재 상태](artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/metadata/status.json) |
| 별도 FPS 효과 비교: pilot300 · 고정 해상도 FPS2/3/4/6/8 | [fps_spatial_control_pilot300_v1 사양](results/sampling/fps_spatial_control/pilot/300/SPEC.md) · [이전 공간 변화 감사](results/sampling/fps_spatial_control/pilot/300/SPATIAL_AUDIT.md) · [현재 상태](artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json) |

GT를 활용한 프레임 배정은 정답 정보를 사용하는 진단이며 순수 zero-shot 성능과 구분한다. 선행 실행의 Random42/43과 후속 실행의 활성 seed42 범위도 구분한다. 공간 비통제 폐기 결과는 포함하지 않는다.

## Rollout

- [Probability budget](results/rollout/probability_budget/viewer.html): LevinTS·native mass·numeric novelty 후보 생성 비교. Query×방법별512·1,024·2,048 decoder 입력 상한,미탐색 질량1% 조기 종료. 후보 생성의 pilot300 완료와 전체 후속 처리의 완료를 구분한다. 같은 viewer의 [End-to-end completion · pilot300](results/rollout/probability_budget/viewer.html?view=pilot%2F300%2Fend_to_end%2Fdetail.html)에서 모든 후보의 독립 crop 검증, 두 모델의 query-free semantic 측정, 검증 전·후 실제 최종 답변을 비교한다. 후속 계산은 pilot300의 보고서·viewer 검증 후 나머지600으로 진행한다.
- [Deterministic rollout](results/rollout/deterministic/viewer.html): 고정 pilot300·600·900을 같은900개 원자료에서 참조한다. token distribution과 branch search도 같은 뷰에 통합했다.
- [Stochastic rollout](results/rollout/stochastic/viewer.html): pilot300.

확장 실행 부분의 별도 집계·보고서·출처는 제거했다. 실제 원본 응답900개는 `artifacts/runs/rollout/deterministic/pilot/900/rollouts/`에 한 번만 저장한다. HTML은 각 실험 계열의 대표 viewer에서 평가 범위와 분석을 전환한다.

## 운영·입력 검증 자료

연구 결과를 뒷받침하는 운영 기록으로 별도 분류한다.

- [전체 추론 병목](results/operations/inference_performance/full/9404/REPORT.md) · [파이프라인 검토](results/operations/inference_performance/full/9404/PIPELINE_REVIEW.md) · [캐시 조사](results/operations/inference_performance/full/9404/CACHE_REPORT.md)
- [H.264 MMCO 디코딩 진단](results/operations/mmco_diagnosis/pilot/300/REPORT.md)
- [선택 프레임 캐시 이관](results/operations/selected_frame_cache_migration/shared/REPORT.md)

HTML은 통합 서버8780에서 제공한다. 입력 전처리와 Python 환경은 `../_shared/timelens/`를 공유한다. [Attention 실험](../attention/README.md)은 독립 프로젝트다.

## Deterministic rollout → reconsideration · pilot600

저장된 deterministic trajectory를 이용하는 하위 분석으로, 고정600 query에서 agent, verifier, model-based test-time scaling을 비교한다. [계열 통합 뷰](results/rollout/deterministic/viewer.html?view=pilot/600/reconsideration/viewer.html)의 Reconsideration 분석에서 deterministic 사례 유형·분기 경로와 검토·선택·final answer를 함께 탐색한다. [실험 설계](results/rollout/deterministic/pilot/600/reconsideration/SPEC.md).
