# Frame–query similarity: full 9,404

완료: 2026-10-30T10:20:49.509605+09:00

[샘플별 interactive HTML](report.html) · [전체 샘플 지표](../../artifacts/runs/full_9404_v1/sample_metrics.json) · [CSV](../../artifacts/runs/full_9404_v1/sample_metrics.csv) · [요약 JSON](../../artifacts/runs/full_9404_v1/summary.json)

## 설정

- 고정 manifest의 전체 9,404쌍을 사용했다. Dataset별 개수: {'charades-timelens': 3363, 'activitynet-timelens': 4500, 'qvhighlights-timelens': 1541}. 재추출하지 않았다.
- GT와 무관한 2fps sampling. 원본 frame index를 round(k × source_fps / 2)로 선택하고 실제 frame PTS 시작 시간을 저장했다. 중복 index는 제거했다.
- 동일한 RGB JPEG 입력(최대 변 512px, quality=95, chroma subsampling 없음)을 두 모델에 제공했다. 각 모델 고유 resize/crop은 유지한다.
- CLIP: openai/clip-vit-large-patch14, FP16 inference. 공식 projected image/text features, raw query, 최대 77 text tokens.
- Qwen: Qwen/Qwen3-VL-Embedding-8B, BF16 inference, SDPA. 공식 chat template와 기본 instruction `Represent the user's input.` 사용. 마지막 유효 token의 final hidden state를 추출한다. 이미지 max_pixels=262144, min_pixels=4096, text truncation 비활성화.
- Frame과 query는 독립적으로 인코딩한다. 전체 영상이나 frame+query 공동 입력, 생성, task fine-tuning은 사용하지 않는다.
- 두 모델 모두 FP32로 변환해 L2 정규화하고 내적으로 cosine을 계산한다. CLIP learned logit scale은 적용하지 않는다.
- 원본 frame, query/video embedding, FP32 raw score, timestamp, config, 모델 revision과 weight SHA256을 보관했다.

| Backbone | Revision | Local path |
|---|---|---|
| openai/clip-vit-large-patch14 | `32bd64288804d66eefd0ccbe215aa642df71cc41` | `/home/irteam/backbones/models/clip-vit-large-patch14` |
| Qwen/Qwen3-VL-Embedding-8B | `2c4565515e0f265c6511776e7193b22c0968ddc7` | `/home/irteam/backbones/models/Qwen3-VL-Embedding-8B` |

## 지표 정의

- Positive: frame PTS가 GT 구간 합집합 `[start,end)` 안에 있음. AP는 query 내부 frame 순위를 평가하고 sklearn average_precision_score로 계산한다. PR curve의 사다리꼴 면적과 다르다. 정확한 score tie는 AP에서 같은 threshold로 묶는다.
- mAP: video–query 쌍별 AP의 동일 가중 평균. Micro pooling하지 않는다.
- First GT rank: 내림차순 순위 중 첫 positive 위치(1부터). 낮을수록 좋다. 정확한 score tie에서는 timestamp 순서를 유지한다.
- MRR (Mean Reciprocal Rank): 쿼리별 첫 GT frame 순위 r_q의 역수를 동일 가중 평균한 값, MRR = (1/Q) × Σ_q (1/r_q). Q는 GT sampled frame이 있는 유효 video–query 쌍 수다. 높을수록 좋고, 모든 쿼리의 1위 frame이 GT 안이면 1이다. 예를 들어 첫 GT 순위가 1·2·10위이면 MRR = (1 + 1/2 + 1/10)/3 ≈ 0.533이다. 상위 순위 차이에 민감하며, GT frame 전체의 순위를 평가하는 AP와 달리 GT frame 하나를 얼마나 빨리 찾는지 평가한다.
- First GT percentile: (rank−1)/(N−1), 낮을수록 좋다. 서로 다른 frame 수의 영향을 확인하는 보조값이다.
- p_i = softmax(cosine_i / τ). GT mass = Σ_GT p_i. Uniform baseline = N_GT/N. Lift = mass/baseline. 보고되는 mean lift는 샘플별 lift의 평균이며 평균 mass를 평균 baseline으로 나눈 값이 아니다.
- GT sampled frame이 0이면 AP/rank/lift는 정의되지 않아 집계에서 제외하고 개수를 명시한다. GT 질량 자체는 0으로 저장한다. 전체 manifest를 확인한다.

## 순위 성능

| Dataset | Model | N / valid | mAP ↑ | MRR ↑ | Median first GT rank ↓ | Mean first GT rank ↓ | Mean first GT percentile ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| all | clip | 9404 / 9404 | 0.4508 | 0.5594 | 2.0 | 12.13 | 0.0731 |
| all | qwen | 9404 / 9404 | 0.5103 | 0.6247 | 1.0 | 9.61 | 0.0599 |
| activitynet-timelens | clip | 4500 / 4500 | 0.4104 | 0.5210 | 2.0 | 17.89 | 0.0660 |
| activitynet-timelens | qwen | 4500 / 4500 | 0.4877 | 0.6064 | 1.0 | 13.67 | 0.0496 |
| charades-timelens | clip | 3363 / 3363 | 0.4212 | 0.5295 | 2.0 | 7.31 | 0.1087 |
| charades-timelens | qwen | 3363 / 3363 | 0.4509 | 0.5683 | 2.0 | 6.57 | 0.0959 |
| qvhighlights-timelens | clip | 1541 / 1541 | 0.6334 | 0.7367 | 1.0 | 5.81 | 0.0161 |
| qvhighlights-timelens | qwen | 1541 / 1541 | 0.7058 | 0.8012 | 1.0 | 4.41 | 0.0114 |

## GT mass·uniform baseline·lift

Temperature는 calibration된 확률이 아니다. 같은 τ라도 모델별 score scale 차이가 남는다. 주 지표는 순위 기반 mAP이며, 아래 τ=.07은 시각화 기본값으로 사전 고정했다.

| Dataset | Model | Mean GT mass | Mean uniform baseline | Mean lift |
|---|---|---:|---:|---:|
| all | clip | 0.2318 | 0.1890 | 1.5414 |
| all | qwen | 0.3536 | 0.1890 | 4.2219 |
| activitynet-timelens | clip | 0.2132 | 0.1825 | 1.5883 |
| activitynet-timelens | qwen | 0.3210 | 0.1825 | 4.9178 |
| charades-timelens | clip | 0.2215 | 0.1884 | 1.2039 |
| charades-timelens | qwen | 0.3006 | 0.1884 | 1.7926 |
| qvhighlights-timelens | clip | 0.3087 | 0.2092 | 2.1409 |
| qvhighlights-timelens | qwen | 0.5648 | 0.2092 | 7.4911 |

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

## GT 비율별 순위 성능

| GT fraction | Model | N / valid | mAP | MRR |
|---|---|---:|---:|---:|
| GT<10% | clip | 5179 / 5179 | 0.3052 | 0.4113 |
| GT<10% | qwen | 5179 / 5179 | 0.3813 | 0.5010 |
| 10%<=GT<30% | clip | 2474 / 2474 | 0.4958 | 0.6399 |
| 10%<=GT<30% | qwen | 2474 / 2474 | 0.5511 | 0.6938 |
| GT>=30% | clip | 1751 / 1751 | 0.8179 | 0.8837 |
| GT>=30% | qwen | 1751 / 1751 | 0.8341 | 0.8930 |

## 검증 및 한계

- [검증 결과](../../artifacts/runs/full_9404_v1/validation.json): 9,404쌍 × 두 모델의 timestamps·score·embedding norm·독립 metric 재계산과 mass 합계 확인.
- 이미지 한 장씩의 retrieval 평가로, 동작 순서나 시간적 맥락을 이용하는 video encoder 성능은 측정하지 않는다.
- 긴 GT는 첫 positive를 찾기 쉽고 AP baseline도 높다. Macro 평균이나 lift가 GT 길이 효과를 완전히 제거하지는 않는다.
- CLIP과 Qwen은 모델 크기, 학습 데이터, 이미지 전처리와 text template가 다르다. 결과는 이 checkpoint와 입력 설정 사이의 비교다.
- Softmax는 주어진 영상 안에서만 정규화되며 query relevance가 없는 영상에서도 합이 1이다. GT mass를 절대적인 정답 확률로 해석하지 않는다.

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
