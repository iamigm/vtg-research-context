<!-- report-appendix:APPENDIX.md parts=15 -->
<!-- report-part:0:begin -->
# 세 세팅의 zero-shot failure 분석 · pilot200 (200질문)

**Qwen2.5 기본 / Qwen2.5 + TimeLens-7B식 입력 / Qwen3 기본**을 비교한다. 데이터셋별 같은 100 query이며 전수평가는 아니다. Qwen3의 별도 입력 변환 실험은 삭제했다.

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## 정확한 입력 세팅

| 항목 | Qwen2.5 기본 | Qwen2.5 + TimeLens-7B식 입력 | Qwen3 기본 |
|---|---|---|---|
| 시각 블록 | 서로 다른 두 프레임 | 선택한 한 프레임 복제 | 서로 다른 두 프레임 |
| 서로 다른 프레임 빈도 | 2 FPS | 약 1 FPS | 2 FPS |
| 절대 시간 표현 | native temporal MRoPE | `12.3s:` 텍스트 | `<12.3 seconds>` 텍스트 |
| 텍스트 timestamp 기준 | 해당 없음 | 선택한 프레임 시간 | 두 프레임 시간의 평균 |
| modality | video | 독립 image | video |
| 추가 timestamp 설명 | 없음 | 있음 | 없음 |

모두 원래 Instruct 가중치이며 추가 학습은 없다. TimeLens 학습 모델을 평가한 결과가 아니다. Qwen3 native에는 이미 textual timestamp가 있으며 공식 TimeLens-8B 입력 경로와 같은 방식이다. **Qwen2.5 변환은 프레임 복제·modality·시간 표현·지시문을 함께 바꾸므로 단일 요인의 효과가 아니다.**

[프롬프트 전문·전처리·공통 조건](../../../../../README.md).

<!-- report-part:1:end -->
<!-- report-part:2:begin -->
## 성능

| 세팅 | 데이터셋 | n | R@1@0.3 | R@1@0.5 | R@1@0.7 | mIoU |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5 native | charades-timelens | 100 | 62.0 | 39.0 | 21.0 | 41.2 |
| Qwen2.5 native | qvhighlights-timelens | 100 | 47.0 | 36.0 | 18.0 | 37.2 |
| Qwen2.5 + TimeLens-7B input | charades-timelens | 100 | 40.0 | 20.0 | 10.0 | 26.0 |
| Qwen2.5 + TimeLens-7B input | qvhighlights-timelens | 100 | 21.0 | 8.0 | 3.0 | 14.8 |
| Qwen3 native (text timestamps) | charades-timelens | 100 | 68.0 | 53.0 | 25.0 | 47.5 |
| Qwen3 native (text timestamps) | qvhighlights-timelens | 100 | 76.0 | 68.0 | 56.0 | 63.6 |

단위 %. 공식 parser의 첫 구간과 정수 초 반올림 규칙을 적용했다.

<!-- report-part:2:end -->
<!-- report-part:3:begin -->
## Qwen2.5 입력 변환의 영향

| 데이터셋 | 실패→성공 | 성공→실패 | R@1@0.5 변화 | 95% paired bootstrap 구간 | 무겹침 실패 변화 |
|---|---:|---:|---:|---|---|
| charades-timelens | 2 | 21 | -19.0%p | [-28.0, -10.8]%p | 11 → 35 |
| qvhighlights-timelens | 0 | 28 | -28.0%p | [-37.0, -19.0]%p | 14 → 44 |

변환 뒤 Qwen2.5의 잘못된 시점 선택과 영상 범위 초과가 늘었다. 입력만 바꾸면 성능이 개선된다는 근거가 아니다. Bootstrap은 영상 단위 2,000회이며 표본 내 탐색적 추정이다. Qwen3 native는 별도 기준 모델로 함께 표시하며 Qwen2.5 변환 효과의 paired 계산에는 포함하지 않는다.

[Qwen2.5 paired 결과](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/metrics/baseline_paired_cases.csv) · [실패 유형 전환](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/figures/failure_transitions.png) · [GT 길이별 분석](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/metadata/stratified_analysis.json) · [프레임 보존 점검](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/metadata/sampling_analysis.json).

<!-- report-part:3:end -->
<!-- report-part:14:begin -->
## 검증과 산출물

- Qwen2.5 변환 200개에서 baseline과 시각 토큰 수·공간 grid·temporal group 수 및 실제 frame index/FPS 정합성을 검증했다. [결과](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/validation/paired_input_validation.json).
- 공식 evaluator와 Qwen2.5 변환 결과의 IoU 차이는 0이다. [결과](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/validation/validation.json). 기존 두 native 모델의 검증은 baseline 디렉터리에 보존되어 있다.
- 기본 입력 두 모델 및 Qwen2.5 변환의 원시 예측은 수정하지 않았다. [보존 hash와 삭제 내역](../../../../../artifacts/common/configs/removed_qwen3_variant.json).

[세 세팅 전체 query 표](report.html) · [세 세팅 CSV](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/metrics/setting_case_metrics.csv) · [성능 비교 그림](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/figures/summary.png) · [입력 세팅 전문](../../../../../README.md).
<!-- report-part:14:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
