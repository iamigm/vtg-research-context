# Appendix · Pilot 300 · 16fps 유사도와 Instruct 생성 비용 비교 · pilot300 (300질문)

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

## 샘플당 시간과 연산량

<!-- report-part:3:begin -->
### 실제 실행 구간의 전체 경과 시간

아래에는 모델별 마지막 실행 구간의 파일 저장·진행 기록을 포함한 wall time도 제시한다. Loading과 warmup은 별도다. 재개한 경우 측정 sample 수를 명시하므로 전체 300개 측정 시간과 혼동하지 않는다.

| Model | Measured samples | Stage wall seconds | Stage wall / sample | Model loading seconds |
|---|---:|---:|---:|---:|
| clip | 300 | 2169.210 | 7.231 | 2.748 |
| qwen | 300 | 5306.027 | 17.687 | 3.079 |
| instruct | 300 | 4287.027 | 14.290 | 3.094 |

<!-- report-part:3:end -->
<!-- report-part:8:begin -->
## 인접 frame embedding 유사도

`a_i = cos(f_(i-1), f_i)` (i=1,…,N−1). Query와 무관한 frame–frame cosine이다. 저장된 이미지 embedding을 사용하며 모델 추론은 추가로 수행하지 않았다.

그래프의 점은 두 frame 중 나중 timestamp t_i에 표시한다. Hover에는 이전·현재 timestamp와 시간 간격을 표시하고, 클릭하면 두 frame을 함께 보여준다. 첫 frame에는 이전 frame이 없어 점이 없다. 연결선은 인접 측정값을 이은 표시이며 사이 시점에서 계산한 값이 아니다.

아래 평균은 각 video–query 샘플에서 인접 pair cosine을 평균한 뒤 샘플에 동일 가중치를 부여한다. 같은 영상의 다른 query는 동일한 frame–frame 곡선을 공유한다. FPS가 다르면 인접 pair의 시간 간격도 달라진다. 기존 frame–query 지표 및 시간·FLOPs 측정은 유지한다.

| Dataset | Model | N | Mean adjacent cosine | Mean pair count |
|---|---|---:|---:|---:|
| all | clip | 300 | 0.974351 | 1644.1 |
| all | qwen | 300 | 0.956399 | 1644.1 |
| activitynet-timelens | clip | 100 | 0.977384 | 2115.3 |
| activitynet-timelens | qwen | 100 | 0.958761 | 2115.3 |
| charades-timelens | clip | 100 | 0.977450 | 421.5 |
| charades-timelens | qwen | 100 | 0.953684 | 421.5 |
| qvhighlights-timelens | clip | 100 | 0.968218 | 2395.5 |
| qvhighlights-timelens | qwen | 100 | 0.956753 | 2395.5 |
<!-- report-part:8:end -->
