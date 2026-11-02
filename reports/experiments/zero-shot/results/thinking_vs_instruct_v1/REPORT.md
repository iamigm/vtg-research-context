# Qwen3-VL-8B Thinking vs Instruct — 동일 pilot 비교

기존 결과를 사용한 300개 샘플 × 2개 입력 순서의 대응 비교다. GPU 추론은 추가로 실행하지 않았다.

600개 모두 프롬프트·영상 grid·샘플링 timestamp·decoder metadata가 같음을 확인했다. 확장 입력은 Thinking 전용 assistant 접미부 `<think>\n` 외에는 정확히 일치한다.

현재 설정에서 영상→질문의 전체 mIoU는 거의 같지만, 질문→영상에서는 Thinking이 크게 높다. Instruct의 입력 순서 변경에 따른 전체 mIoU 하락은 약 12.34pp이고 Thinking은 약 0.71pp다. 이는 Thinking 실행이 입력 순서 변경에 덜 민감한 패턴이지만, 모델과 decoding 설정이 동시에 달라 원인을 분리할 수 없다.

## 비교 조건

| 항목 | Instruct | Thinking |
|---|---|---|
| 모델 | Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Thinking |
| 생성 | greedy | sampling: T=1, top_p=.95, top_k=20, seed=42 |
| 최대 생성 토큰 | 512 | 16,384 |
| 전처리 | 2 fps, min_tokens=64, total_tokens=14336 | 동일 |

따라서 순수한 checkpoint 효과나 reasoning 유무의 인과 효과를 분리한 실험은 아니다. 현재 두 실행 설정의 실제 성능·출력 비용을 비교한다.

## 성능

| Dataset | 순서 | N | Instruct mIoU(%) | Thinking mIoU(%) | 차이(pp) | 차이 95% CI(pp) |
|---|---|---:|---:|---:|---:|---|
| charades-timelens | video_query | 100 | 46.90 | 42.36 | -4.54 | [-8.67, -0.24] |
| charades-timelens | query_video | 100 | 47.12 | 44.04 | -3.08 | [-9.15, +3.02] |
| activitynet-timelens | video_query | 100 | 40.63 | 42.15 | +1.52 | [-2.34, +5.35] |
| activitynet-timelens | query_video | 100 | 27.46 | 38.98 | +11.52 | [+5.34, +17.54] |
| qvhighlights-timelens | video_query | 100 | 57.13 | 61.35 | +4.22 | [-1.69, +10.54] |
| qvhighlights-timelens | query_video | 100 | 33.06 | 60.72 | +27.66 | [+19.05, +36.24] |
| all | video_query | 300 | 48.22 | 48.62 | +0.40 | [-2.34, +3.21] |
| all | query_video | 300 | 35.88 | 47.91 | +12.03 | [+7.73, +16.33] |

최종 답변의 원문을 같은 extract_time으로 파싱하고 첫 예측 구간 대 단일 GT의 IoU를 계산했다. 반올림·영상 길이 clipping을 하지 않았으며 파싱 실패는 0점으로 포함했다. 과거 Instruct 보고서는 반올림한 timestamp로 채점했기 때문에 수치가 조금 다를 수 있다. ../../artifacts/runs/thinking_vs_instruct_v1/metrics/summary.json에는 동일 반올림 규칙을 적용한 두 모델 수치도 제공한다. CI는 샘플 쌍 bootstrap 20,000회이며 seed 변동성을 포함하지 않는다.

## 생성량

| 순서 | Instruct 평균 생성 토큰 | Thinking 평균 생성 토큰 | 배율 |
|---|---:|---:|---:|
| video_query | 17.9 | 410.0 | 22.8× |
| query_video | 17.1 | 1451.8 | 84.8× |

생성량은 EOS와 경계 태그를 포함한 실제 generated_ids 길이다. Thinking에는 reasoning이 포함된다. 이 비율은 GPU 시간 배율을 의미하지 않는다.

## 검증 및 해석 범위

- 동일 cohort/order 600쌍의 원본·GT·입력 일치 확인. Thinking 재채점은 원래 저장된 IoU와 일치한다.
- 원본 결과를 변경하지 않고 파생 JSON/보고서와 참조 파일 hash만 저장했다.
- 기존 Instruct attention은 선택 layer/토큰 집계 방식이 달라 이번 비교에서 attention 수치를 직접 대조하지 않았다.

- Greedy Instruct vs sampled Thinking: checkpoint and decoding-policy effects are confounded.
- Token budgets differ: 512 vs 16384.
- One rollout per model/order; bootstrap does not account for generation-seed variation.
- No model runtime or attention-pattern comparison: capture protocols differ.
