# Appendix · Frame–query similarity: pilot 300 · pilot300 (300질문)

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

## GT mass·uniform baseline·lift

<!-- report-part:5:begin -->
### Temperature sensitivity (all 300)

| τ | Model | Mean GT mass | Mean uniform baseline | Mean lift |
|---:|---|---:|---:|---:|
| 0.01 | clip | 0.4322 | 0.1767 | 5.4788 |
| 0.01 | qwen | 0.5398 | 0.1767 | 9.0599 |
| 0.03 | clip | 0.3156 | 0.1767 | 2.8785 |
| 0.03 | qwen | 0.4938 | 0.1767 | 7.6253 |
| 0.07 | clip | 0.2354 | 0.1767 | 1.6365 |
| 0.07 | qwen | 0.3942 | 0.1767 | 4.6683 |
| 0.1 | clip | 0.2164 | 0.1767 | 1.4091 |
| 0.1 | qwen | 0.3411 | 0.1767 | 3.4291 |
| 0.3 | clip | 0.1889 | 0.1767 | 1.1175 |
| 0.3 | qwen | 0.2262 | 0.1767 | 1.5274 |
| 1.0 | clip | 0.1802 | 0.1767 | 1.0334 |
| 1.0 | qwen | 0.1900 | 0.1767 | 1.1280 |

<!-- report-part:5:end -->
<!-- report-part:6:begin -->
## GT 비율별 순위 성능

| GT fraction | Model | N / valid | mAP | MRR |
|---|---|---:|---:|---:|
| GT<10% | clip | 158 / 158 | 0.2992 | 0.3709 |
| GT<10% | qwen | 158 / 158 | 0.3650 | 0.4695 |
| 10%<=GT<30% | clip | 89 / 89 | 0.5686 | 0.6870 |
| 10%<=GT<30% | qwen | 89 / 89 | 0.6669 | 0.7891 |
| GT>=30% | clip | 53 / 53 | 0.7912 | 0.8834 |
| GT>=30% | qwen | 53 / 53 | 0.8417 | 0.9244 |

<!-- report-part:6:end -->
<!-- report-part:8:begin -->
## 재실행

공유 Python: `/home/irteam/experiments/_shared/timelens/.venv/bin/python`

```bash
cd /home/irteam/experiments/similarity
export SIMILARITY_RUN=pilot_300_v1
export SIMILARITY_MANIFEST=/home/irteam/experiments/zero-shot/artifacts/common/predictions/legacy/cohorts/pilot_300/samples.json
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
| all | clip | 300 | 0.927792 | 209.1 |
| all | qwen | 300 | 0.885944 | 209.1 |
| activitynet-timelens | clip | 100 | 0.933057 | 266.7 |
| activitynet-timelens | qwen | 100 | 0.889876 | 266.7 |
| charades-timelens | clip | 100 | 0.945476 | 61.9 |
| charades-timelens | qwen | 100 | 0.902259 | 61.9 |
| qvhighlights-timelens | clip | 100 | 0.904842 | 298.7 |
| qvhighlights-timelens | qwen | 100 | 0.865696 | 298.7 |
<!-- report-part:9:end -->
