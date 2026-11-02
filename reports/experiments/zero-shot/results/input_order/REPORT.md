# Zero-shot 성능 비교

PNG 정량 표: [전체 벤치마크](../../artifacts/runs/legacy_quantitative/figures/full_benchmark.png) · [공통 pilot300](../../artifacts/runs/legacy_quantitative/figures/pilot300.png). 기존 predictions 원본과 12개 지표 행을 대조해 검증했다. 재생성 코드: [`zeroshot_metrics_png.py`](../../_lib/zeroshot_metrics_png.py).

공용 저장소의 원본 예측만으로 재집계했다. Attention 측정값이나 NPZ를 읽지 않는다. Qwen3 전체 9,404개 입력 순서 비교와 두 모델의 동일 pilot 300개를 구분한다.

| 모델 | 표본 | 순서 | 벤치마크 | N | mIoU (%) | R@1 .3 (%) | R@1 .5 (%) | R@1 .7 (%) |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | full_9404 | video_query | charades-timelens | 3363 | 48.35 | 69.19 | 53.58 | 27.62 |
| qwen3 | full_9404 | video_query | activitynet-timelens | 4500 | 46.79 | 62.20 | 51.27 | 34.47 |
| qwen3 | full_9404 | video_query | qvhighlights-timelens | 1541 | 59.42 | 73.85 | 64.37 | 49.38 |
| qwen3 | full_9404 | query_video | charades-timelens | 3363 | 46.05 | 63.19 | 49.69 | 28.75 |
| qwen3 | full_9404 | query_video | activitynet-timelens | 4500 | 32.24 | 41.49 | 33.76 | 23.44 |
| qwen3 | full_9404 | query_video | qvhighlights-timelens | 1541 | 36.93 | 45.04 | 39.26 | 30.30 |
| qwen25 | pilot_300 | video_query | charades-timelens | 100 | 40.65 | 61.00 | 42.00 | 17.00 |
| qwen25 | pilot_300 | video_query | activitynet-timelens | 100 | 29.96 | 41.00 | 27.00 | 15.00 |
| qwen25 | pilot_300 | video_query | qvhighlights-timelens | 100 | 30.83 | 50.00 | 24.00 | 8.00 |
| qwen3 | pilot_300 | video_query | charades-timelens | 100 | 47.53 | 65.00 | 49.00 | 31.00 |
| qwen3 | pilot_300 | video_query | activitynet-timelens | 100 | 40.77 | 55.00 | 43.00 | 28.00 |
| qwen3 | pilot_300 | video_query | qvhighlights-timelens | 100 | 57.13 | 73.00 | 62.00 | 45.00 |

Qwen3는 역순에서 mIoU와 R@1 .3/.5가 세 벤치마크 모두 낮아졌다. Charades의 R@1 .7은 역순이 1.13%p 높다. 모델 간 비교는 동일 pilot_300 행을 사용해야 한다.

[실패 예측 참조](../../artifacts/runs/input_order/metrics/invalid_prediction_refs.csv) · [집계 CSV](../../artifacts/runs/input_order/metrics/benchmark_metrics.csv) · [전체 지표](../../artifacts/runs/input_order/metrics/summary.json) · [순수 예측과 재사용 방법](../../artifacts/common/predictions/legacy/README.md)

[Span-head 실험](../timelens_span_head/REPORT.md) · [Top-30 실험](../timelens_top30/REPORT.md) · [별도 attention 분석](../../../attention/results/timelens_order/final_report.md)

순수 예측 캐시는 기존 full-head 생성 결과에서 측정 필드만 분리했다. Qwen3 원본 생성에는 출력을 변경하지 않는 attention 계측이 있었고, Qwen2.5는 full-head 기준선 실행에서 얻었다. 새 추론을 실행한 결과는 아니다. 디코더에 관한 후속 진단은 [기록](../timelens_top30/mmco_diagnosis/REPORT.md)을 참고한다.
