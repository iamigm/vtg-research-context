# Appendix · spatial resolution 통제 zero-shot FPS sweep · pilot · 300 queries

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:1:begin -->
| 모델 | 순서 | 데이터셋 | FPS | N | mIoU | Δpp | R@.3 | Δpp | R@.5 | Δpp | R@.7 | Δpp |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| instruct | query_video | activitynet-timelens | 2 | 100 | 26.10 | +0.00 | 33.00 | +0.00 | 30.00 | +0.00 | 19.00 | +0.00 |
| instruct | query_video | activitynet-timelens | 3 | 100 | 25.77 | -0.33 | 33.00 | +0.00 | 26.00 | -4.00 | 18.00 | -1.00 |
| instruct | query_video | activitynet-timelens | 4 | 100 | 26.97 | +0.86 | 35.00 | +2.00 | 28.00 | -2.00 | 20.00 | +1.00 |
| instruct | query_video | activitynet-timelens | 6 | 100 | 29.19 | +3.09 | 37.00 | +4.00 | 31.00 | +1.00 | 21.00 | +2.00 |
| instruct | query_video | activitynet-timelens | 8 | 100 | 28.44 | +2.34 | 38.00 | +5.00 | 30.00 | +0.00 | 20.00 | +1.00 |
| instruct | query_video | all | 2 | 300 | 31.83 | +0.00 | 40.67 | +0.00 | 34.67 | +0.00 | 22.33 | +0.00 |
| instruct | query_video | all | 3 | 300 | 33.72 | +1.89 | 44.00 | +3.33 | 36.33 | +1.67 | 22.67 | +0.33 |
| instruct | query_video | all | 4 | 300 | 36.65 | +4.82 | 47.67 | +7.00 | 38.67 | +4.00 | 26.00 | +3.67 |
| instruct | query_video | all | 6 | 300 | 34.80 | +2.98 | 45.00 | +4.33 | 37.33 | +2.67 | 23.33 | +1.00 |
| instruct | query_video | all | 8 | 300 | 35.10 | +3.27 | 46.67 | +6.00 | 36.67 | +2.00 | 22.67 | +0.33 |
| instruct | query_video | charades-timelens | 2 | 100 | 40.00 | +0.00 | 50.00 | +0.00 | 42.00 | +0.00 | 26.00 | +0.00 |
| instruct | query_video | charades-timelens | 3 | 100 | 38.49 | -1.51 | 52.00 | +2.00 | 43.00 | +1.00 | 22.00 | -4.00 |
| instruct | query_video | charades-timelens | 4 | 100 | 40.76 | +0.76 | 57.00 | +7.00 | 44.00 | +2.00 | 22.00 | -4.00 |
| instruct | query_video | charades-timelens | 6 | 100 | 38.40 | -1.60 | 51.00 | +1.00 | 43.00 | +1.00 | 20.00 | -6.00 |
| instruct | query_video | charades-timelens | 8 | 100 | 39.49 | -0.51 | 53.00 | +3.00 | 42.00 | +0.00 | 21.00 | -5.00 |
| instruct | query_video | qvhighlights-timelens | 2 | 100 | 29.38 | +0.00 | 39.00 | +0.00 | 32.00 | +0.00 | 22.00 | +0.00 |
| instruct | query_video | qvhighlights-timelens | 3 | 100 | 36.89 | +7.51 | 47.00 | +8.00 | 40.00 | +8.00 | 28.00 | +6.00 |
| instruct | query_video | qvhighlights-timelens | 4 | 100 | 42.22 | +12.84 | 51.00 | +12.00 | 44.00 | +12.00 | 36.00 | +14.00 |
| instruct | query_video | qvhighlights-timelens | 6 | 100 | 36.82 | +7.43 | 47.00 | +8.00 | 38.00 | +6.00 | 29.00 | +7.00 |
| instruct | query_video | qvhighlights-timelens | 8 | 100 | 37.36 | +7.98 | 49.00 | +10.00 | 38.00 | +6.00 | 27.00 | +5.00 |
| instruct | video_query | activitynet-timelens | 2 | 100 | 43.55 | +0.00 | 58.00 | +0.00 | 47.00 | +0.00 | 29.00 | +0.00 |
| instruct | video_query | activitynet-timelens | 3 | 100 | 44.57 | +1.02 | 58.00 | +0.00 | 44.00 | -3.00 | 30.00 | +1.00 |
| instruct | video_query | activitynet-timelens | 4 | 100 | 45.75 | +2.20 | 58.00 | +0.00 | 44.00 | -3.00 | 35.00 | +6.00 |
| instruct | video_query | activitynet-timelens | 6 | 100 | 42.99 | -0.56 | 60.00 | +2.00 | 42.00 | -5.00 | 28.00 | -1.00 |
| instruct | video_query | activitynet-timelens | 8 | 100 | 45.53 | +1.98 | 62.00 | +4.00 | 46.00 | -1.00 | 32.00 | +3.00 |
| instruct | video_query | all | 2 | 300 | 48.70 | +0.00 | 64.67 | +0.00 | 51.33 | +0.00 | 35.00 | +0.00 |
| instruct | video_query | all | 3 | 300 | 49.01 | +0.32 | 63.67 | -1.00 | 51.33 | +0.00 | 33.67 | -1.33 |
| instruct | video_query | all | 4 | 300 | 48.21 | -0.49 | 63.67 | -1.00 | 48.33 | -3.00 | 35.00 | +0.00 |
| instruct | video_query | all | 6 | 300 | 47.35 | -1.34 | 64.00 | -0.67 | 49.33 | -2.00 | 33.00 | -2.00 |
| instruct | video_query | all | 8 | 300 | 47.86 | -0.83 | 65.00 | +0.33 | 50.67 | -0.67 | 33.00 | -2.00 |
| instruct | video_query | charades-timelens | 2 | 100 | 43.80 | +0.00 | 61.00 | +0.00 | 45.00 | +0.00 | 28.00 | +0.00 |
| instruct | video_query | charades-timelens | 3 | 100 | 43.60 | -0.20 | 58.00 | -3.00 | 48.00 | +3.00 | 26.00 | -2.00 |
| instruct | video_query | charades-timelens | 4 | 100 | 42.94 | -0.86 | 59.00 | -2.00 | 45.00 | +0.00 | 28.00 | +0.00 |
| instruct | video_query | charades-timelens | 6 | 100 | 42.53 | -1.27 | 59.00 | -2.00 | 47.00 | +2.00 | 28.00 | +0.00 |
| instruct | video_query | charades-timelens | 8 | 100 | 43.87 | +0.07 | 60.00 | -1.00 | 49.00 | +4.00 | 26.00 | -2.00 |
| instruct | video_query | qvhighlights-timelens | 2 | 100 | 58.74 | +0.00 | 75.00 | +0.00 | 62.00 | +0.00 | 48.00 | +0.00 |
| instruct | video_query | qvhighlights-timelens | 3 | 100 | 58.88 | +0.14 | 75.00 | +0.00 | 62.00 | +0.00 | 45.00 | -3.00 |
| instruct | video_query | qvhighlights-timelens | 4 | 100 | 55.93 | -2.81 | 74.00 | -1.00 | 56.00 | -6.00 | 42.00 | -6.00 |
| instruct | video_query | qvhighlights-timelens | 6 | 100 | 56.55 | -2.20 | 73.00 | -2.00 | 59.00 | -3.00 | 43.00 | -5.00 |
| instruct | video_query | qvhighlights-timelens | 8 | 100 | 54.18 | -4.56 | 73.00 | -2.00 | 57.00 | -5.00 | 41.00 | -7.00 |
| thinking | query_video | activitynet-timelens | 2 | 100 | 39.63 | +0.00 | 55.00 | +0.00 | 41.00 | +0.00 | 26.00 | +0.00 |
| thinking | query_video | activitynet-timelens | 3 | 100 | 40.53 | +0.90 | 53.00 | -2.00 | 43.00 | +2.00 | 30.00 | +4.00 |
| thinking | query_video | activitynet-timelens | 4 | 100 | 35.45 | -4.18 | 47.00 | -8.00 | 34.00 | -7.00 | 27.00 | +1.00 |
| thinking | query_video | activitynet-timelens | 6 | 100 | 41.80 | +2.17 | 54.00 | -1.00 | 40.00 | -1.00 | 35.00 | +9.00 |
| thinking | query_video | activitynet-timelens | 8 | 100 | 42.56 | +2.93 | 57.00 | +2.00 | 45.00 | +4.00 | 30.00 | +4.00 |
| thinking | query_video | all | 2 | 300 | 47.65 | +0.00 | 66.00 | +0.00 | 51.00 | +0.00 | 34.00 | +0.00 |
| thinking | query_video | all | 3 | 300 | 46.56 | -1.10 | 63.33 | -2.67 | 49.00 | -2.00 | 34.33 | +0.33 |
| thinking | query_video | all | 4 | 300 | 45.61 | -2.05 | 61.67 | -4.33 | 48.67 | -2.33 | 34.00 | +0.00 |
| thinking | query_video | all | 6 | 300 | 47.40 | -0.25 | 63.67 | -2.33 | 50.33 | -0.67 | 35.67 | +1.67 |
| thinking | query_video | all | 8 | 300 | 48.35 | +0.70 | 66.33 | +0.33 | 51.67 | +0.67 | 36.67 | +2.67 |
| thinking | query_video | charades-timelens | 2 | 100 | 43.37 | +0.00 | 65.00 | +0.00 | 48.00 | +0.00 | 28.00 | +0.00 |
| thinking | query_video | charades-timelens | 3 | 100 | 38.91 | -4.46 | 57.00 | -8.00 | 39.00 | -9.00 | 25.00 | -3.00 |
| thinking | query_video | charades-timelens | 4 | 100 | 42.13 | -1.24 | 61.00 | -4.00 | 49.00 | +1.00 | 25.00 | -3.00 |
| thinking | query_video | charades-timelens | 6 | 100 | 41.95 | -1.42 | 61.00 | -4.00 | 45.00 | -3.00 | 26.00 | -2.00 |
| thinking | query_video | charades-timelens | 8 | 100 | 42.47 | -0.90 | 64.00 | -1.00 | 44.00 | -4.00 | 27.00 | -1.00 |
| thinking | query_video | qvhighlights-timelens | 2 | 100 | 59.96 | +0.00 | 78.00 | +0.00 | 64.00 | +0.00 | 48.00 | +0.00 |
| thinking | query_video | qvhighlights-timelens | 3 | 100 | 60.22 | +0.27 | 80.00 | +2.00 | 65.00 | +1.00 | 48.00 | +0.00 |
| thinking | query_video | qvhighlights-timelens | 4 | 100 | 59.24 | -0.72 | 77.00 | -1.00 | 63.00 | -1.00 | 50.00 | +2.00 |
| thinking | query_video | qvhighlights-timelens | 6 | 100 | 58.46 | -1.50 | 76.00 | -2.00 | 66.00 | +2.00 | 46.00 | -2.00 |
| thinking | query_video | qvhighlights-timelens | 8 | 100 | 60.02 | +0.06 | 78.00 | +0.00 | 66.00 | +2.00 | 53.00 | +5.00 |
| thinking | video_query | activitynet-timelens | 2 | 100 | 44.71 | +0.00 | 62.00 | +0.00 | 47.00 | +0.00 | 32.00 | +0.00 |
| thinking | video_query | activitynet-timelens | 3 | 100 | 42.29 | -2.42 | 57.00 | -5.00 | 43.00 | -4.00 | 30.00 | -2.00 |
| thinking | video_query | activitynet-timelens | 4 | 100 | 42.42 | -2.29 | 59.00 | -3.00 | 43.00 | -4.00 | 28.00 | -4.00 |
| thinking | video_query | activitynet-timelens | 6 | 100 | 44.21 | -0.50 | 54.00 | -8.00 | 44.00 | -3.00 | 34.00 | +2.00 |
| thinking | video_query | activitynet-timelens | 8 | 100 | 48.41 | +3.70 | 65.00 | +3.00 | 51.00 | +4.00 | 35.00 | +3.00 |
| thinking | video_query | all | 2 | 300 | 51.12 | +0.00 | 68.67 | +0.00 | 56.00 | +0.00 | 36.67 | +0.00 |
| thinking | video_query | all | 3 | 300 | 49.60 | -1.52 | 67.33 | -1.33 | 52.33 | -3.67 | 36.33 | -0.33 |
| thinking | video_query | all | 4 | 300 | 49.81 | -1.31 | 68.00 | -0.67 | 52.67 | -3.33 | 36.33 | -0.33 |
| thinking | video_query | all | 6 | 300 | 51.41 | +0.29 | 67.67 | -1.00 | 55.00 | -1.00 | 38.33 | +1.67 |
| thinking | video_query | all | 8 | 300 | 51.94 | +0.82 | 71.33 | +2.67 | 55.67 | -0.33 | 37.33 | +0.67 |
| thinking | video_query | charades-timelens | 2 | 100 | 46.14 | +0.00 | 66.00 | +0.00 | 53.00 | +0.00 | 25.00 | +0.00 |
| thinking | video_query | charades-timelens | 3 | 100 | 43.80 | -2.34 | 66.00 | +0.00 | 45.00 | -8.00 | 24.00 | -1.00 |
| thinking | video_query | charades-timelens | 4 | 100 | 43.01 | -3.13 | 64.00 | -2.00 | 43.00 | -10.00 | 23.00 | -2.00 |
| thinking | video_query | charades-timelens | 6 | 100 | 46.09 | -0.05 | 68.00 | +2.00 | 48.00 | -5.00 | 25.00 | +0.00 |
| thinking | video_query | charades-timelens | 8 | 100 | 43.90 | -2.24 | 67.00 | +1.00 | 45.00 | -8.00 | 24.00 | -1.00 |
| thinking | video_query | qvhighlights-timelens | 2 | 100 | 62.50 | +0.00 | 78.00 | +0.00 | 68.00 | +0.00 | 53.00 | +0.00 |
| thinking | video_query | qvhighlights-timelens | 3 | 100 | 62.69 | +0.20 | 79.00 | +1.00 | 69.00 | +1.00 | 55.00 | +2.00 |
| thinking | video_query | qvhighlights-timelens | 4 | 100 | 63.99 | +1.49 | 81.00 | +3.00 | 72.00 | +4.00 | 58.00 | +5.00 |
| thinking | video_query | qvhighlights-timelens | 6 | 100 | 63.92 | +1.42 | 81.00 | +3.00 | 73.00 | +5.00 | 56.00 | +3.00 |
| thinking | video_query | qvhighlights-timelens | 8 | 100 | 63.51 | +1.01 | 82.00 | +4.00 | 71.00 | +3.00 | 53.00 | +0.00 |
<!-- report-part:1:end -->
