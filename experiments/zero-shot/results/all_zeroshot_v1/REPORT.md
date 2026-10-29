# 기존 zero-shot 전체 집계

집계 시각: 2026-10-28T11:56:59.734714+09:00. GPU 추론 없이 기존 응답 39,205개를 재채점하고, 별도 sampling rollout 6,000개는 기존 검증된 통계를 통합했다. 8fps는 집계 시점에 저장된 완료분만 포함한다.

동일 실행의 response·top30 trace·attention 사본은 한 번만 집계했다. pilot은 full의 부분집합이며, 서로 다른 실행 버전과 rollout을 합쳐 하나의 성능으로 계산하지 않았다.

## 전체 실행 목록

| Run | 응답 수 | 근거 |
|---|---:|---|
| full_benchmark_v1 | 18,808 | raw responses, re-scored |
| legacy_qwen25_pilot | 300 | raw responses, re-scored |
| legacy_qwen3 | 18,808 | raw responses, re-scored |
| pilot_8fps_v1 | 89 | raw responses, re-scored |
| restricted_head_526 | 600 | raw responses, re-scored |
| thinking_pilot_2fps | 600 | raw responses, re-scored |
| stochastic_rollout | 6,000 | existing validated summary; 6000 response files counted, candidate traces not reloaded |

## 전체 벤치마크

모든 성능은 %. 본문은 반올림 없는 동일 parser 재채점 값이다. 원래 반올림 규칙 및 저장 점수는 CSV·JSON에서 함께 제공한다. `all`은 query 수에 따른 가중 평균이다.

| Run | 모델 | FPS | 입력 순서 | Dataset | N / 목표 | mIoU | R@.3 | R@.5 | R@.7 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| full_benchmark_v1 | Qwen2.5-VL-7B-Instruct | 2 | video_query | activitynet-timelens | 4500/4500 | 32.07 | 44.93 | 29.80 | 16.22 |
| full_benchmark_v1 | Qwen2.5-VL-7B-Instruct | 2 | video_query | all | 9404/9404 | 34.88 | 50.40 | 30.90 | 16.37 |
| full_benchmark_v1 | Qwen2.5-VL-7B-Instruct | 2 | video_query | charades-timelens | 3363/3363 | 39.47 | 60.54 | 33.96 | 16.77 |
| full_benchmark_v1 | Qwen2.5-VL-7B-Instruct | 2 | video_query | qvhighlights-timelens | 1541/1541 | 33.02 | 44.26 | 27.45 | 15.90 |
| full_benchmark_v1 | Qwen3-VL-8B-Instruct | 2 | video_query | activitynet-timelens | 4500/4500 | 46.88 | 62.60 | 50.82 | 34.76 |
| full_benchmark_v1 | Qwen3-VL-8B-Instruct | 2 | video_query | all | 9404/9404 | 49.41 | 66.96 | 52.95 | 34.49 |
| full_benchmark_v1 | Qwen3-VL-8B-Instruct | 2 | video_query | charades-timelens | 3363/3363 | 48.17 | 69.58 | 50.67 | 27.21 |
| full_benchmark_v1 | Qwen3-VL-8B-Instruct | 2 | video_query | qvhighlights-timelens | 1541/1541 | 59.53 | 73.98 | 64.11 | 49.58 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | query_video | activitynet-timelens | 4500/4500 | 32.27 | 41.78 | 33.51 | 23.42 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | query_video | all | 9404/9404 | 37.89 | 50.00 | 40.00 | 26.36 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | query_video | charades-timelens | 3363/3363 | 45.90 | 63.25 | 49.12 | 28.55 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | query_video | qvhighlights-timelens | 1541/1541 | 36.82 | 45.10 | 39.07 | 30.18 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | video_query | activitynet-timelens | 4500/4500 | 46.86 | 62.62 | 50.80 | 34.71 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | video_query | all | 9404/9404 | 49.40 | 66.97 | 52.93 | 34.46 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | video_query | charades-timelens | 3363/3363 | 48.17 | 69.58 | 50.67 | 27.21 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | video_query | qvhighlights-timelens | 1541/1541 | 59.53 | 73.98 | 64.11 | 49.58 |

## 동일 pilot 300개

모든 성능은 %. 본문은 반올림 없는 동일 parser 재채점 값이다. 원래 반올림 규칙 및 저장 점수는 CSV·JSON에서 함께 제공한다. `all`은 query 수에 따른 가중 평균이다.

| Run | 모델 | FPS | 입력 순서 | Dataset | N / 목표 | mIoU | R@.3 | R@.5 | R@.7 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| full_benchmark_v1 | Qwen2.5-VL-7B-Instruct | 2 | video_query | activitynet-timelens | 100/100 | 29.74 | 41.00 | 26.00 | 14.00 |
| full_benchmark_v1 | Qwen2.5-VL-7B-Instruct | 2 | video_query | all | 300/300 | 33.60 | 49.67 | 29.00 | 13.33 |
| full_benchmark_v1 | Qwen2.5-VL-7B-Instruct | 2 | video_query | charades-timelens | 100/100 | 40.32 | 60.00 | 37.00 | 18.00 |
| full_benchmark_v1 | Qwen2.5-VL-7B-Instruct | 2 | video_query | qvhighlights-timelens | 100/100 | 30.74 | 48.00 | 24.00 | 8.00 |
| full_benchmark_v1 | Qwen3-VL-8B-Instruct | 2 | video_query | activitynet-timelens | 100/100 | 40.63 | 55.00 | 42.00 | 27.00 |
| full_benchmark_v1 | Qwen3-VL-8B-Instruct | 2 | video_query | all | 300/300 | 48.22 | 63.67 | 50.00 | 34.33 |
| full_benchmark_v1 | Qwen3-VL-8B-Instruct | 2 | video_query | charades-timelens | 100/100 | 46.90 | 63.00 | 48.00 | 31.00 |
| full_benchmark_v1 | Qwen3-VL-8B-Instruct | 2 | video_query | qvhighlights-timelens | 100/100 | 57.13 | 73.00 | 60.00 | 45.00 |
| legacy_qwen25_pilot | Qwen2.5-VL-7B-Instruct | 2 | video_query | activitynet-timelens | 100/100 | 29.74 | 41.00 | 26.00 | 14.00 |
| legacy_qwen25_pilot | Qwen2.5-VL-7B-Instruct | 2 | video_query | all | 300/300 | 33.60 | 49.67 | 29.00 | 13.33 |
| legacy_qwen25_pilot | Qwen2.5-VL-7B-Instruct | 2 | video_query | charades-timelens | 100/100 | 40.32 | 60.00 | 37.00 | 18.00 |
| legacy_qwen25_pilot | Qwen2.5-VL-7B-Instruct | 2 | video_query | qvhighlights-timelens | 100/100 | 30.74 | 48.00 | 24.00 | 8.00 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | query_video | activitynet-timelens | 100/100 | 27.46 | 34.00 | 30.00 | 22.00 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | query_video | all | 300/300 | 35.88 | 46.33 | 38.33 | 27.00 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | query_video | charades-timelens | 100/100 | 47.12 | 60.00 | 51.00 | 35.00 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | query_video | qvhighlights-timelens | 100/100 | 33.06 | 45.00 | 34.00 | 24.00 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | video_query | activitynet-timelens | 100/100 | 40.63 | 55.00 | 42.00 | 27.00 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | video_query | all | 300/300 | 48.22 | 63.67 | 50.00 | 34.33 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | video_query | charades-timelens | 100/100 | 46.90 | 63.00 | 48.00 | 31.00 |
| legacy_qwen3 | Qwen3-VL-8B-Instruct | 2 | video_query | qvhighlights-timelens | 100/100 | 57.13 | 73.00 | 60.00 | 45.00 |
| thinking_pilot_2fps | Qwen3-VL-8B-Thinking | 2 | query_video | activitynet-timelens | 100/100 | 38.98 | 52.00 | 42.00 | 25.00 |
| thinking_pilot_2fps | Qwen3-VL-8B-Thinking | 2 | query_video | all | 300/300 | 47.91 | 65.67 | 52.33 | 32.67 |
| thinking_pilot_2fps | Qwen3-VL-8B-Thinking | 2 | query_video | charades-timelens | 100/100 | 44.04 | 66.00 | 50.00 | 23.00 |
| thinking_pilot_2fps | Qwen3-VL-8B-Thinking | 2 | query_video | qvhighlights-timelens | 100/100 | 60.72 | 79.00 | 65.00 | 50.00 |
| thinking_pilot_2fps | Qwen3-VL-8B-Thinking | 2 | video_query | activitynet-timelens | 100/100 | 42.15 | 57.00 | 44.00 | 29.00 |
| thinking_pilot_2fps | Qwen3-VL-8B-Thinking | 2 | video_query | all | 300/300 | 48.62 | 65.33 | 51.33 | 36.00 |
| thinking_pilot_2fps | Qwen3-VL-8B-Thinking | 2 | video_query | charades-timelens | 100/100 | 42.36 | 64.00 | 47.00 | 24.00 |
| thinking_pilot_2fps | Qwen3-VL-8B-Thinking | 2 | video_query | qvhighlights-timelens | 100/100 | 61.35 | 75.00 | 63.00 | 55.00 |

## 8fps 진행 중인 결과

모든 성능은 %. 본문은 반올림 없는 동일 parser 재채점 값이다. 원래 반올림 규칙 및 저장 점수는 CSV·JSON에서 함께 제공한다. `all`은 query 수에 따른 가중 평균이다.

| Run | 모델 | FPS | 입력 순서 | Dataset | N / 목표 | mIoU | R@.3 | R@.5 | R@.7 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| pilot_8fps_v1 | Qwen3-VL-8B-Instruct | 8 | query_video | all | 44/300 | 44.91 | 61.36 | 52.27 | 27.27 |
| pilot_8fps_v1 | Qwen3-VL-8B-Instruct | 8 | query_video | charades-timelens | 44/100 | 44.91 | 61.36 | 52.27 | 27.27 |
| pilot_8fps_v1 | Qwen3-VL-8B-Instruct | 8 | video_query | all | 45/300 | 49.48 | 71.11 | 55.56 | 35.56 |
| pilot_8fps_v1 | Qwen3-VL-8B-Instruct | 8 | video_query | charades-timelens | 45/100 | 49.48 | 71.11 | 55.56 | 35.56 |

부분 완료 행은 선택된 일부 샘플의 결과이므로 완료된 300개 결과와 직접 비교하면 안 된다. Instruct greedy(max512)와 Thinking sampling(T=1,p=.95,k=20,seed42,max16384)은 생성 정책과 예산이 달라 checkpoint만의 효과가 아니다. 모든 실행의 요청 FPS는 2 또는 8이며, 실제 프레임 sampling은 원본 fps 및 기존 최대 768프레임 제한을 따른다. min_tokens=64,total_tokens=14336을 유지한다.

## 별도 비교군: 제한 head

모든 성능은 %. 본문은 반올림 없는 동일 parser 재채점 값이다. 원래 반올림 규칙 및 저장 점수는 CSV·JSON에서 함께 제공한다. `all`은 query 수에 따른 가중 평균이다.

| Run | 모델 | FPS | 입력 순서 | Dataset | N / 목표 | mIoU | R@.3 | R@.5 | R@.7 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| restricted_head_526 | Qwen2.5-VL-7B-Instruct | 2 | video_query | activitynet-timelens | 100/100 | 34.94 | 47.00 | 35.00 | 20.00 |
| restricted_head_526 | Qwen2.5-VL-7B-Instruct | 2 | video_query | all | 300/300 | 39.81 | 56.33 | 39.00 | 22.33 |
| restricted_head_526 | Qwen2.5-VL-7B-Instruct | 2 | video_query | charades-timelens | 100/100 | 39.45 | 59.00 | 38.00 | 17.00 |
| restricted_head_526 | Qwen2.5-VL-7B-Instruct | 2 | video_query | qvhighlights-timelens | 100/100 | 45.05 | 63.00 | 44.00 | 30.00 |
| restricted_head_526 | Qwen3-VL-8B-Instruct | 2 | video_query | activitynet-timelens | 100/100 | 36.93 | 51.00 | 34.00 | 25.00 |
| restricted_head_526 | Qwen3-VL-8B-Instruct | 2 | video_query | all | 300/300 | 43.24 | 59.00 | 42.67 | 29.67 |
| restricted_head_526 | Qwen3-VL-8B-Instruct | 2 | video_query | charades-timelens | 100/100 | 37.25 | 55.00 | 36.00 | 24.00 |
| restricted_head_526 | Qwen3-VL-8B-Instruct | 2 | video_query | qvhighlights-timelens | 100/100 | 55.54 | 71.00 | 58.00 | 40.00 |

## 별도 비교군: stochastic rollout

기존 검증된 summary.json과 validation.json을 사용했다. 이 섹션은 원래 반올림 채점 값이며 raw 재채점이 아니다. 각 query 5회 생성, 벤치마크별 100 query / 500 rollout이다. Oracle any-of-5는 GT를 본 상한이며 실제 추론 성능이 아니다.

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

## 해석 및 중복 처리

- Qwen2.5-VL-7B-Instruct: 기존 baseline과 full_benchmark_v1의 대응 300개 중 raw IoU가 달라진 응답은 0개다. 실행 버전을 분리해 보존했다. 동일 IoU는 동일 생성 토큰을 뜻하지 않는다.
- Qwen3-VL-8B-Instruct: 기존 baseline과 full_benchmark_v1의 대응 9,404개 중 raw IoU가 달라진 응답은 25개다. 실행 버전을 분리해 보존했다. 동일 IoU는 동일 생성 토큰을 뜻하지 않는다.
- 2fps pilot video_query: Instruct 48.22%, Thinking 48.62%, 차이 +0.40pp mIoU.
- 2fps pilot query_video: Instruct 35.88%, Thinking 47.91%, 차이 +12.03pp mIoU.

원본 예측과 파생 측정은 변경하지 않았다. smoke 검사·중단된 old cohort·성능과 무관한 entropy/attention 파생값은 별도 성능 실행으로 세지 않았다. top30 pilot은 full_benchmark_v1에서 재사용한 동일 응답이므로 중복 합산하지 않았다.

[지표 CSV](metrics.csv) · [샘플별 재채점 CSV](sample_metrics.csv) · [전체 JSON](summary.json) · [출처 SHA-256](sources.json) · [검증](validation.json) · [rollout 원 보고서](../timelens_rollout/REPORT.md)
