# TimeLens answer → video attention

전체 9,404쌍의 attention 측정 결과다. 원본 예측은 공용 pure zero-shot 저장소에서 읽고, 이 폴더에는 attention 배열·측정값·검증과 비교 결과만 둔다.

수치쌍은 **video → query / query → video**, 단위는 %다.

| 벤치마크 | 영상 attention 총량 | 영상 내 GT 비중 | 전체 key 중 GT 비중 | 초반 10% peak 빈도 |
|---|---:|---:|---:|---:|
| charades-timelens | 7.65 / 13.35 | 21.11 / 19.71 | 1.59 / 2.60 | 99.14 / 87.12 |
| activitynet-timelens | 7.04 / 13.21 | 21.51 / 18.44 | 1.48 / 2.40 | 97.87 / 88.58 |
| qvhighlights-timelens | 7.15 / 13.08 | 27.13 / 22.53 | 1.90 / 2.90 | 99.03 / 95.98 |

역순에서는 영상 attention 총량은 증가하지만 영상 내부 GT 비중은 감소한다. 초반 peak 빈도도 낮아지지만 여전히 높다. 서로 다른 생성 답변의 관측값이므로 이 변화가 성능 차이의 원인이라고 단정하지 않는다.

Consumed answer 행(EOS 제외), L8–21, head·token 평균이다. 실제 post-RoPE·KV-cache Q/K로 FP32 softmax를 재구성하며 FlashAttention 출력은 변경하지 않았다.

[지표 CSV](dataset_metrics.csv) · [전체 집계](final_results.json) · [그래프](final_comparison.png) · [검증](final_aggregation_validation.json)

[순수 예측 저장소](../../../zero-shot/predictions/README.md) · [Zero-shot 성능 비교](../../../zero-shot/results/input_order/REPORT.md)
