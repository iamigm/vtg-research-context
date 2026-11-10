<!-- report-appendix:APPENDIX.md parts=4 -->
<!-- report-part:0:begin -->
# spatial resolution 통제 zero-shot FPS sweep · pilot · 300 queries

전체 완료: 6,000개 응답. 기존 고정 pilot300 × 2모델 × 2input order × FPS 2/3/4/6/8.

mIoU=평균(예측 구간∩GT 구간 길이 / 예측 구간∪GT 구간 길이). R@τ=100×IoU≥τ인 질문 수/N. 네 지표 모두 높을수록 좋다. 표의 점수는 %, 변화는 같은 모델·input order·질문의 2fps 대비 pp다. 최종 답변의 첫 구간을 채점하고 파싱 실패는 0점, 반올림·clipping은 하지 않는다. all은 질문 수 가중 평균이다.

영상별 목표100,352픽셀 ±10% 안에서 32배수 H/W를 정하고 종횡비 오차를 최소화했다. 같은 영상은 모든 FPS·모델·순서에서 정확히 같은 H/W·resize를 사용했다. 원본 RGB에서 직접 resize했으며 실제 픽셀hash를 검증했다. 전체 영상 토큰 예산과 768프레임 상한을 적용하지 않았다. FPS가 증가하면 프레임 수·시각 토큰·연산량이 증가한다.

샘플링은 원본 PTS에서 k/FPS에 가장 가까운 프레임(동률은 앞 프레임)을 선택한다. 중복은 제거하고 시간 패치 묶음 때문에 마지막 홀수 프레임만 한 번 복제한다. GT·기존 예측·점수는 계획과 모델 입력에 전달하지 않았다. 학습·few-shot 예시·추가 rollout은 없다. 원본 FPS가 낮으면 실제 unique FPS도 목표보다 낮다.

Instruct는 greedy/max512, Thinking은 T=1,top_p=.95,top_k=20,seed42/max16384다. 같은 모델 안에서는 모든 FPS에 같은 생성 설정과 seed를 적용했다. 모델 간 차이는 생성 정책도 포함하며 FPS 효과는 모델·순서별로 해석한다. 단일 seed로 seed 변동성을 측정하지 않았다.

<!-- report-part:0:end -->
<!-- report-part:2:begin -->

<!-- report-part:2:end -->
<!-- report-part:3:begin -->
## 실제 입력과 실패

| 모델 | 순서 | FPS | 평균 면적 | 평균 unique frames | 평균 실효 FPS | 평균 visual tokens | 파싱 실패 | EOS 미도달 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| instruct | video_query | 2 | 94702.9 | 210.3 | 2.009 | 9708.1 | 0 | 0 |
| instruct | video_query | 3 | 94702.9 | 315.1 | 3.008 | 14532.4 | 0 | 0 |
| instruct | video_query | 4 | 94702.9 | 420.0 | 4.008 | 19364.5 | 0 | 0 |
| instruct | video_query | 6 | 94702.9 | 629.7 | 6.007 | 29017.1 | 0 | 0 |
| instruct | video_query | 8 | 94702.9 | 835.8 | 7.905 | 38496.6 | 0 | 0 |
| instruct | query_video | 2 | 94702.9 | 210.3 | 2.009 | 9708.1 | 1 | 0 |
| instruct | query_video | 3 | 94702.9 | 315.1 | 3.008 | 14532.4 | 2 | 0 |
| instruct | query_video | 4 | 94702.9 | 420.0 | 4.008 | 19364.5 | 5 | 0 |
| instruct | query_video | 6 | 94702.9 | 629.7 | 6.007 | 29017.1 | 5 | 0 |
| instruct | query_video | 8 | 94702.9 | 835.8 | 7.905 | 38496.6 | 5 | 0 |
| thinking | video_query | 2 | 94702.9 | 210.3 | 2.009 | 9708.1 | 1 | 1 |
| thinking | video_query | 3 | 94702.9 | 315.1 | 3.008 | 14532.4 | 0 | 0 |
| thinking | video_query | 4 | 94702.9 | 420.0 | 4.008 | 19364.5 | 0 | 0 |
| thinking | video_query | 6 | 94702.9 | 629.7 | 6.007 | 29017.1 | 0 | 0 |
| thinking | video_query | 8 | 94702.9 | 835.8 | 7.905 | 38496.6 | 0 | 0 |
| thinking | query_video | 2 | 94702.9 | 210.3 | 2.009 | 9708.1 | 3 | 2 |
| thinking | query_video | 3 | 94702.9 | 315.1 | 3.008 | 14532.4 | 3 | 2 |
| thinking | query_video | 4 | 94702.9 | 420.0 | 4.008 | 19364.5 | 3 | 2 |
| thinking | query_video | 6 | 94702.9 | 629.7 | 6.007 | 29017.1 | 1 | 1 |
| thinking | query_video | 8 | 94702.9 | 835.8 | 7.905 | 38496.6 | 2 | 1 |

[영상 길이·GT 비율·두 축 교차표의 네 지표와 2fps 대비 변화](AXES.md). 고정 pilot300만 평가했으며 600/900 확장 결과는 없다. 이 코호트는 기존 진단 표본이며 벤치마크 전체의 무작위 평균으로 해석하지 않는다.

[질문별 CSV](../../../../../artifacts/runs/sampling/fps_spatial_control/pilot/300/metrics/per_sample.csv) · [요약 CSV](../../../../../artifacts/runs/sampling/fps_spatial_control/pilot/300/metrics/summary.csv) · [검증](../../../../../artifacts/runs/sampling/fps_spatial_control/pilot/300/validation/final.json) · [전체 실험 목록](../../../../../../index.html)
<!-- report-part:3:end -->
## 전체 query의 FPS 비교

| 모델 | 순서 | 데이터셋 | FPS | N | mIoU | Δpp | R@.3 | Δpp | R@.5 | Δpp | R@.7 | Δpp |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| instruct | query_video | all | 2 | 300 | 31.83 | +0.00 | 40.67 | +0.00 | 34.67 | +0.00 | 22.33 | +0.00 |
| instruct | query_video | all | 3 | 300 | 33.72 | +1.89 | 44.00 | +3.33 | 36.33 | +1.67 | 22.67 | +0.33 |
| instruct | query_video | all | 4 | 300 | 36.65 | +4.82 | 47.67 | +7.00 | 38.67 | +4.00 | 26.00 | +3.67 |
| instruct | query_video | all | 6 | 300 | 34.80 | +2.98 | 45.00 | +4.33 | 37.33 | +2.67 | 23.33 | +1.00 |
| instruct | query_video | all | 8 | 300 | 35.10 | +3.27 | 46.67 | +6.00 | 36.67 | +2.00 | 22.67 | +0.33 |
| instruct | video_query | all | 2 | 300 | 48.70 | +0.00 | 64.67 | +0.00 | 51.33 | +0.00 | 35.00 | +0.00 |
| instruct | video_query | all | 3 | 300 | 49.01 | +0.32 | 63.67 | -1.00 | 51.33 | +0.00 | 33.67 | -1.33 |
| instruct | video_query | all | 4 | 300 | 48.21 | -0.49 | 63.67 | -1.00 | 48.33 | -3.00 | 35.00 | +0.00 |
| instruct | video_query | all | 6 | 300 | 47.35 | -1.34 | 64.00 | -0.67 | 49.33 | -2.00 | 33.00 | -2.00 |
| instruct | video_query | all | 8 | 300 | 47.86 | -0.83 | 65.00 | +0.33 | 50.67 | -0.67 | 33.00 | -2.00 |
| thinking | query_video | all | 2 | 300 | 47.65 | +0.00 | 66.00 | +0.00 | 51.00 | +0.00 | 34.00 | +0.00 |
| thinking | query_video | all | 3 | 300 | 46.56 | -1.10 | 63.33 | -2.67 | 49.00 | -2.00 | 34.33 | +0.33 |
| thinking | query_video | all | 4 | 300 | 45.61 | -2.05 | 61.67 | -4.33 | 48.67 | -2.33 | 34.00 | +0.00 |
| thinking | query_video | all | 6 | 300 | 47.40 | -0.25 | 63.67 | -2.33 | 50.33 | -0.67 | 35.67 | +1.67 |
| thinking | query_video | all | 8 | 300 | 48.35 | +0.70 | 66.33 | +0.33 | 51.67 | +0.67 | 36.67 | +2.67 |
| thinking | video_query | all | 2 | 300 | 51.12 | +0.00 | 68.67 | +0.00 | 56.00 | +0.00 | 36.67 | +0.00 |
| thinking | video_query | all | 3 | 300 | 49.60 | -1.52 | 67.33 | -1.33 | 52.33 | -3.67 | 36.33 | -0.33 |
| thinking | video_query | all | 4 | 300 | 49.81 | -1.31 | 68.00 | -0.67 | 52.67 | -3.33 | 36.33 | -0.33 |
| thinking | video_query | all | 6 | 300 | 51.41 | +0.29 | 67.67 | -1.00 | 55.00 | -1.00 | 38.33 | +1.67 |
| thinking | video_query | all | 8 | 300 | 51.94 | +0.82 | 71.33 | +2.67 | 55.67 | -0.33 | 37.33 | +0.67 |

## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
