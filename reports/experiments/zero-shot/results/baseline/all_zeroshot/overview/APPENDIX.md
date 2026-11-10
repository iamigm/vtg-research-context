# Appendix · 기존 zero-shot 전체 집계 · 전체 full9404 · pilot300 (질문 수)

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:4:begin -->
## 별도 비교군: 제한 head

모든 성능은 %. 본문은 반올림 없는 동일 parser 재채점 값이다. 원래 반올림 규칙 및 저장 점수는 CSV·JSON에서 함께 제공한다. `all`은 query 수에 따른 가중 평균이다.

| Run | 모델 | FPS | input order | Dataset | N / 목표 | mIoU | R@.3 | R@.5 | R@.7 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| restricted_head_526 | Qwen2.5-VL-7B-Instruct | 2 | video_query | activitynet-timelens | 100/100 | 34.94 | 47.00 | 35.00 | 20.00 |
| restricted_head_526 | Qwen2.5-VL-7B-Instruct | 2 | video_query | all | 300/300 | 39.81 | 56.33 | 39.00 | 22.33 |
| restricted_head_526 | Qwen2.5-VL-7B-Instruct | 2 | video_query | charades-timelens | 100/100 | 39.45 | 59.00 | 38.00 | 17.00 |
| restricted_head_526 | Qwen2.5-VL-7B-Instruct | 2 | video_query | qvhighlights-timelens | 100/100 | 45.05 | 63.00 | 44.00 | 30.00 |
| restricted_head_526 | Qwen3-VL-8B-Instruct | 2 | video_query | activitynet-timelens | 100/100 | 36.93 | 51.00 | 34.00 | 25.00 |
| restricted_head_526 | Qwen3-VL-8B-Instruct | 2 | video_query | all | 300/300 | 43.24 | 59.00 | 42.67 | 29.67 |
| restricted_head_526 | Qwen3-VL-8B-Instruct | 2 | video_query | charades-timelens | 100/100 | 37.25 | 55.00 | 36.00 | 24.00 |
| restricted_head_526 | Qwen3-VL-8B-Instruct | 2 | video_query | qvhighlights-timelens | 100/100 | 55.54 | 71.00 | 58.00 | 40.00 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## 별도 비교군: stochastic rollout

기존 검증된 ../../../../artifacts/runs/baseline/all_zeroshot/overview/metrics/summary.json과 ../../../../artifacts/runs/baseline/all_zeroshot/overview/validation/validation.json을 사용했다. 이 섹션은 원래 반올림 채점 값이며 raw 재채점이 아니다. 각 query 5회 생성, 벤치마크별 100 query / 500 rollout이다. Oracle any-of-5는 GT를 본 상한이며 실제 추론 성능이 아니다.

| 모델 | 방식 | Dataset | Query | Rollout | mIoU % | R@.5 % | Oracle any5 R@.5 % |
|---|---|---|---:|---:|---:|---:|---:|
| qwen25 | top_k | charades-timelens | 100 | 500 | 32.98 | 30.00 | 56.00 |
| qwen25 | top_k | activitynet-timelens | 100 | 500 | 20.98 | 17.40 | 38.00 |
| qwen25 | top_k | qvhighlights-timelens | 100 | 500 | 23.88 | 16.40 | 43.00 |
| qwen25 | top_p | charades-timelens | 100 | 500 | 32.98 | 30.00 | 56.00 |
| qwen25 | top_p | activitynet-timelens | 100 | 500 | 20.98 | 17.40 | 38.00 |
| qwen25 | top_p | qvhighlights-timelens | 100 | 500 | 23.88 | 16.40 | 43.00 |
| qwen3 | top_k | charades-timelens | 100 | 500 | 43.63 | 47.80 | 63.00 |
| qwen3 | top_k | activitynet-timelens | 100 | 500 | 38.72 | 40.40 | 57.00 |
| qwen3 | top_k | qvhighlights-timelens | 100 | 500 | 53.24 | 57.40 | 73.00 |
| qwen3 | top_p | charades-timelens | 100 | 500 | 44.49 | 49.00 | 59.00 |
| qwen3 | top_p | activitynet-timelens | 100 | 500 | 39.42 | 40.40 | 51.00 |
| qwen3 | top_p | qvhighlights-timelens | 100 | 500 | 54.92 | 59.60 | 71.00 |

<!-- report-part:5:end -->
