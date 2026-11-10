# Appendix · Frame–query similarity: full 9,404 · 전체 full9404 (9,404질문)

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

## GT mass·uniform baseline·lift

<!-- report-part:5:begin -->
### Temperature sensitivity (all 9,404)

| τ | Model | Mean GT mass | Mean uniform baseline | Mean lift |
|---:|---|---:|---:|---:|
| 0.01 | clip | 0.4046 | 0.1890 | 5.7380 |
| 0.01 | qwen | 0.5130 | 0.1890 | 9.0267 |
| 0.03 | clip | 0.2949 | 0.1890 | 2.7920 |
| 0.03 | qwen | 0.4604 | 0.1890 | 7.4101 |
| 0.07 | clip | 0.2318 | 0.1890 | 1.5414 |
| 0.07 | qwen | 0.3536 | 0.1890 | 4.2219 |
| 0.1 | clip | 0.2178 | 0.1890 | 1.3394 |
| 0.1 | qwen | 0.3059 | 0.1890 | 2.9760 |
| 0.3 | clip | 0.1979 | 0.1890 | 1.0951 |
| 0.3 | qwen | 0.2227 | 0.1890 | 1.4010 |
| 1.0 | clip | 0.1916 | 0.1890 | 1.0269 |
| 1.0 | qwen | 0.1982 | 0.1890 | 1.0979 |

<!-- report-part:5:end -->
<!-- report-part:6:begin -->
## GT 비율별 순위 성능

| GT fraction | Model | N / valid | mAP | MRR |
|---|---|---:|---:|---:|
| GT<10% | clip | 5179 / 5179 | 0.3052 | 0.4113 |
| GT<10% | qwen | 5179 / 5179 | 0.3813 | 0.5010 |
| 10%<=GT<30% | clip | 2474 / 2474 | 0.4958 | 0.6399 |
| 10%<=GT<30% | qwen | 2474 / 2474 | 0.5511 | 0.6938 |
| GT>=30% | clip | 1751 / 1751 | 0.8179 | 0.8837 |
| GT>=30% | qwen | 1751 / 1751 | 0.8341 | 0.8930 |

<!-- report-part:6:end -->
<!-- report-part:8:begin -->
## 재실행

공유 Python: `/home/irteam/experiments/_shared/timelens/.venv/bin/python`

```bash
cd /home/irteam/experiments/similarity
export SIMILARITY_RUN=full_9404_v1
export SIMILARITY_MANIFEST=/home/irteam/experiments/zero-shot/artifacts/common/predictions/legacy/qwen3/samples.json
PY=/home/irteam/experiments/_shared/timelens/.venv/bin/python
$PY _lib/download_models.py
$PY _lib/prepare.py
$PY _lib/infer.py --model clip --batch-size 256
$PY _lib/infer.py --model qwen --batch-size 128
$PY _lib/report.py
$PY _lib/validate.py
```

Embedding 캐시가 있으면 재사용한다. 입력·설정을 바꾸는 실험은 별도 output version으로 분리해야 한다.

<!-- report-part:8:end -->
<!-- report-part:9:begin -->
## 인접 frame embedding 유사도

`a_i = cos(f_(i-1), f_i)` (i=1,…,N−1). Query와 무관한 frame–frame cosine이다. 저장된 이미지 embedding을 사용하며 모델 추론은 추가로 수행하지 않았다.

그래프의 점은 두 frame 중 나중 timestamp t_i에 표시한다. Hover에는 이전·현재 timestamp와 시간 간격을 표시하고, 클릭하면 두 frame을 함께 보여준다. 첫 frame에는 이전 frame이 없어 점이 없다. 연결선은 인접 측정값을 이은 표시이며 사이 시점에서 계산한 값이 아니다.

아래 평균은 각 video–query 샘플에서 인접 pair cosine을 평균한 뒤 샘플에 동일 가중치를 부여한다. 같은 영상의 다른 query는 동일한 frame–frame 곡선을 공유한다. FPS가 다르면 인접 pair의 시간 간격도 달라진다. 기존 frame–query 지표 및 시간·FLOPs 측정은 유지한다.

| Dataset | Model | N | Mean adjacent cosine | Mean pair count |
|---|---|---:|---:|---:|
| all | clip | 9404 | 0.933985 | 203.4 |
| all | qwen | 9404 | 0.893098 | 203.4 |
| activitynet-timelens | clip | 4500 | 0.933011 | 278.5 |
| activitynet-timelens | qwen | 4500 | 0.891741 | 278.5 |
| charades-timelens | clip | 3363 | 0.944960 | 59.2 |
| charades-timelens | qwen | 3363 | 0.903893 | 59.2 |
| qvhighlights-timelens | clip | 1541 | 0.912877 | 298.4 |
| qvhighlights-timelens | qwen | 1541 | 0.873503 | 298.4 |
<!-- report-part:9:end -->
