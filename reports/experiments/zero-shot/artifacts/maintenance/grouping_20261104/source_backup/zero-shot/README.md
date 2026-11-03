# Zero-shot 평가와 생성 실험

[전체 실험 목록](../index.html)에서 아래 주제별로 탐색한다. 같은 질문의 선행·후속 실험을 함께 묶고, pilot/full 및 실행 버전은 구분한다. 실행 ID와 원자료 경로는 유지한다. 저장·재사용 방법은 [산출물 저장소 안내](artifacts/README.md)를 따른다.

## 1. 기준 성능·입력 조건

질문: 모델과 입력 순서를 바꾸면 기준 성능과 예측 일관성이 어떻게 달라지는가?

| 실험 흐름 | 자료 |
|---|---|
| 기존 실행 전체 집계: 실행 버전·full/pilot·FPS·입력 순서 | [all_zeroshot_v1 보고서](results/all_zeroshot_v1/REPORT.md) |
| 입력 순서와 모델별 성능 → 같은 질문의 예측 일관성 | [input_order 보고서](results/input_order/REPORT.md) · [일관성 분석](results/input_order/consistency_v1/REPORT.md) · [영상 비교](results/input_order/consistency_v1/report.html) |
| Thinking vs Instruct 동일 pilot 비교 | [thinking_vs_instruct_v1 보고서](results/thinking_vs_instruct_v1/REPORT.md) |

## 2. 토큰 불확실성·오류 진단

질문: 숫자를 생성하는 어느 단계에서 정답 가능성을 잃으며, 토큰 확률과 엔트로피가 이를 얼마나 설명하는가?

| 실험 흐름 | 자료 |
|---|---|
| Full-head top-30 분포 계측 | [timelens_top30 보고서](results/timelens_top30/REPORT.md) · [토큰 뷰어](results/timelens_top30/report.html) |
| 엔트로피와 IoU → prefix 정답 가능성 → 숫자/기타 토큰 비교 | [entropy_iou](results/timelens_top30/entropy_iou/REPORT.md) · [prefix_entropy](results/timelens_top30/prefix_entropy/REPORT.md) · [numeric_uncertainty](results/timelens_top30/numeric_uncertainty/REPORT.md) · [통합 해석](results/timelens_top30/integrated/REPORT.md) · [통합 뷰어](results/timelens_top30/integrated/report.html) |
| 자릿수별 정답 가능성: pilot → 전체 벤치마크 확장 | [pilot](results/timelens_top30/digit_correctness/REPORT.md) · [full](results/timelens_top30_full/digit_correctness/REPORT.md) |
| 정답 가능 상태의 세 축 분석: pilot → full | [pilot blue_analysis](results/timelens_top30/digit_correctness/blue_analysis/REPORT.md) · [full blue_analysis](results/timelens_top30_full/digit_correctness/blue_analysis/REPORT.md) |
| 최초 오류 엔트로피와 최종 GT 거리 | [first_error_distance](results/timelens_top30_full/digit_correctness/first_error_distance/REPORT.md) |
| 숫자·구간 전용 head로 생성 제약 비교 | [timelens_span_head 보고서](results/timelens_span_head/REPORT.md) |

pilot과 full은 같은 분석의 확장이다. prefix의 IoU 가능성 판정과 endpoint 반올림 정답 가능성 판정은 기준이 다르므로 각 보고서 정의를 따른다.

## 3. 영상 샘플링·공간 통제

질문: 공간 해상도를 통제한 상태에서 프레임의 수와 시간 배정을 바꾸면 성능이 어떻게 달라지는가?

| 실험 흐름 | 자료 |
|---|---|
| 선행 공간 예산 비교: Uniform/GT전체8배/Random | [spatial_budget_pilot900_v1 보고서](results/spatial_budget_pilot900_v1/REPORT.md) · [pilot300](results/spatial_budget_pilot900_v1/PILOT_300.md) · [pilot600](results/spatial_budget_pilot900_v1/PILOT_600.md) |
| 후속 전체·경계 배정 sweep: 공간 low/high × A/B | [spatial_sampling_pilot900_v2 보고서](results/spatial_sampling_pilot900_v2/REPORT.md) · [사양](results/spatial_sampling_pilot900_v2/SPEC.md) · [현재 상태](artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json) |
| 별도 FPS 효과 비교: 고정 해상도 FPS2/3/4/6/8 | [fps_spatial_control_pilot300_v1 사양](results/fps_spatial_control_pilot300_v1/SPEC.md) · [이전 공간 변화 감사](results/fps_spatial_control_pilot300_v1/SPATIAL_AUDIT.md) · [현재 상태](artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json) |

GT를 활용한 프레임 배정은 정답 정보를 사용하는 진단이며 순수 zero-shot 성능과 구분한다. 선행 실행의 Random42/43과 후속 실행의 활성 seed42 범위도 구분한다. 공간 비통제 폐기 결과는 포함하지 않는다.

## Rollout · 이번 정리 제외

진행 중이므로 이번 재분류에서 제외한다. 기존 링크만 유지한다.

| 기존 실험 | 자료 |
|---|---|
| 결정적 rollout 900개 샘플별 분기 탐색 | [개선 분기 HTML](results/deterministic_rollout_pilot900_v2/viewer.html) · [종료 소수점 사례](results/deterministic_rollout_pilot900_v2/END_DECIMAL_CASES.md) · [900개 정량 보고서](results/deterministic_rollout_pilot900_v2/REPORT.md) |
| 결정적 rollout 나머지600개 → 중첩600/900 통합 | [사양](results/deterministic_rollout_additional600_v2/SPEC.md) · [상태](artifacts/runs/deterministic_rollout_additional600_v2/metadata/status.json) |
| 고정 pilot300 결정적 rollout 성능·계산량 | [사양](results/deterministic_rollout_pilot300_v2/SPEC.md) · [상태](artifacts/runs/deterministic_rollout_pilot300_v2/metadata/status.json) |
| 시작 prefix 조건부 종료 탐색 예산 | [결과 해석](results/conditional_rollout_budget_pilot30_v1/ANALYSIS.md) · [원보고서](results/conditional_rollout_budget_pilot30_v1/REPORT.md) · [후속 설계](results/conditional_rollout_budget_pilot30_v1/NEXT_EXPERIMENT.md) |
| stochastic rollout | [설정·결과](results/timelens_rollout/REPORT.md) |

## 운영·입력 검증 자료

연구 결과를 뒷받침하는 운영 기록으로 별도 분류한다.

- [전체 추론 병목](results/timelens_top30_full/performance/REPORT.md) · [파이프라인 검토](results/timelens_top30_full/performance/PIPELINE_REVIEW.md) · [캐시 조사](results/timelens_top30_full/performance/CACHE_REPORT.md)
- [H.264 MMCO 디코딩 진단](results/timelens_top30/mmco_diagnosis/REPORT.md)
- [선택 프레임 캐시 이관](results/selected_frame_cache_migration_v1/REPORT.md)

HTML은 통합 서버8780에서 제공한다. 입력 전처리와 Python 환경은 `../_shared/timelens/`를 공유한다. [Attention 실험](../attention/README.md)은 독립 프로젝트다.
