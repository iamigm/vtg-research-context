# TimeLens stochastic rollout: top-k vs top-p

동일 pilot 300개 × 두 모델 × 두 sampling 방식 × 5회 = 6,000개 응답, 107,105개 생성 단계. 기존 zero-shot과 같은 입력 hash를 확인했다.

**두 실험은 top-k만 적용(p=1) / top-p만 적용(k=0)하는 분리 실험이다.** Qwen3 k=20/p=0.8, Qwen2.5 라이브러리 기본 k=50/p=1.0을 사용한다. 모델 카드에서 k와 p를 함께 쓰는 설정 자체의 재현은 아니다. 매 단계에는 truncation 후 유한 score를 가진 후보 전체를 저장했다. 고정 top-30 기록은 하지 않는다.

Qwen3 [모델 카드](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)는 일반 태스크에 temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0, presence_penalty=1.5를 제시한다. Presence penalty는 이미 생성된 토큰에 한 번씩 차감한다. 최대 길이는 zero-shot과 비교하도록 512토큰을 유지했다.

사용자 지시에 따라 Qwen2.5는 Transformers 라이브러리 sampling 기본값 temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0을 명시적으로 적용한다. 모델 배포 파일의 temperature=0.000001/repetition_penalty=1.05와 다르다. Top-p only에서는 top_k=0으로 꺼 p=1.0 전체 어휘 sampling을 수행한다. Qwen2.5 모델 카드가 권장하는 수치라고 주장하지 않는다.

설정 원문과 retrieval 시각/해시는 `../../../../../artifacts/runs/rollout/stochastic/pilot/300/sources/`에 보존했다. 실제 설정은 [../../../../../artifacts/runs/rollout/stochastic/pilot/300/metadata/config.json](../../../../../artifacts/runs/rollout/stochastic/pilot/300/metadata/config.json)에 기록했다. 각 sample/rollout의 seed는 sample ID·rollout 번호·42에서 SHA-256으로 결정하며 두 방식과 모델 사이에 동일하게 대응시켰다. 동일 seed는 서로 다른 분포에서 동일 응답을 보장하지 않는다.

| 모델 | 방식 | 벤치마크 | baseline mIoU (%) | rollout 평균 mIoU (%) | baseline R@.5 (%) | rollout 평균 R@.5 (%) | oracle any-of-5 R@.5 (%) | 평균 고유 응답 수 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen25 | top_k | charades-timelens | 40.65 | 32.98 | 42.00 | 30.00 | 56.00 | 4.91 |
| qwen25 | top_k | activitynet-timelens | 29.96 | 20.98 | 27.00 | 17.40 | 38.00 | 4.95 |
| qwen25 | top_k | qvhighlights-timelens | 30.83 | 23.88 | 24.00 | 16.40 | 43.00 | 5.00 |
| qwen25 | top_p | charades-timelens | 40.65 | 32.98 | 42.00 | 30.00 | 56.00 | 4.91 |
| qwen25 | top_p | activitynet-timelens | 29.96 | 20.98 | 27.00 | 17.40 | 38.00 | 4.95 |
| qwen25 | top_p | qvhighlights-timelens | 30.83 | 23.88 | 24.00 | 16.40 | 43.00 | 5.00 |
| qwen3 | top_k | charades-timelens | 47.53 | 43.63 | 49.00 | 47.80 | 63.00 | 4.74 |
| qwen3 | top_k | activitynet-timelens | 40.77 | 38.72 | 43.00 | 40.40 | 57.00 | 4.75 |
| qwen3 | top_k | qvhighlights-timelens | 57.13 | 53.24 | 62.00 | 57.40 | 73.00 | 4.77 |
| qwen3 | top_p | charades-timelens | 47.53 | 44.49 | 49.00 | 49.00 | 59.00 | 4.62 |
| qwen3 | top_p | activitynet-timelens | 40.77 | 39.42 | 43.00 | 40.40 | 51.00 | 4.52 |
| qwen3 | top_p | qvhighlights-timelens | 57.13 | 54.92 | 62.00 | 59.60 | 71.00 | 4.59 |

평균 지표는 query마다 rollout 5개를 동일 가중 평균한 후 100개 query를 집계한 값이다. 500개 rollout을 독립 query로 해석하지 않는다. oracle any-of-5는 GT를 보고 가장 좋은 답을 고르는 상한이며 실제 배포 성능이 아니다.

| 모델 | 방식 | 벤치마크 | 평균 후보 수 | 양의 확률 후보가 1개인 step 비율 (%) |
|---|---|---|---:|---:|
| qwen25 | top_k | charades-timelens | 51.20 | 0.00 |
| qwen25 | top_k | activitynet-timelens | 51.04 | 0.00 |
| qwen25 | top_k | qvhighlights-timelens | 51.02 | 0.00 |
| qwen25 | top_p | charades-timelens | 152064.00 | 0.00 |
| qwen25 | top_p | activitynet-timelens | 152064.00 | 0.00 |
| qwen25 | top_p | qvhighlights-timelens | 152064.00 | 0.00 |
| qwen3 | top_k | charades-timelens | 20.20 | 0.00 |
| qwen3 | top_k | activitynet-timelens | 20.20 | 0.00 |
| qwen3 | top_k | qvhighlights-timelens | 20.18 | 0.00 |
| qwen3 | top_p | charades-timelens | 1.61 | 81.13 |
| qwen3 | top_p | activitynet-timelens | 1.62 | 79.19 |
| qwen3 | top_p | qvhighlights-timelens | 1.53 | 80.90 |

후보 수 통계는 응답 안에서 step 평균을 내고 응답 간 평균했다. top-k 경계의 logit 동률은 Hugging Face 구현에 따라 k개보다 많이 남을 수 있다. HTML은 해당 후보도 모두 표시한다. top-p는 temperature·penalty 적용 후 분포에 적용하며 다른 top-k 제한을 사용하지 않는다.

sampling 확률은 필터 후 분포를 재정규화한 실제 추출 확률이다. 필터 전 확률과 raw LM-head 확률을 함께 표시해 구분한다. 유한 score라도 FP32 softmax underflow로 확률 0이 될 수 있어 유한 후보 수와 양의 확률 후보 수를 따로 기록했다. 토큰 확률을 구간 정답 확률로 해석하면 안 된다.

- qwen25 top_k: 세 벤치마크 평균 mIoU 차이 -7.87 pp, 평균 고유 응답 4.95/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.
- qwen25 top_p: 세 벤치마크 평균 mIoU 차이 -7.87 pp, 평균 고유 응답 4.95/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.
- qwen3 top_k: 세 벤치마크 평균 mIoU 차이 -3.28 pp, 평균 고유 응답 4.75/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.
- qwen3 top_p: 세 벤치마크 평균 mIoU 차이 -2.20 pp, 평균 고유 응답 4.58/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.

기존 decoder와 프레임 전처리를 유지했다. 앞선 MMCO 진단에서 sparse Decord와 순차 Decord 사이에 9개 영상의 표본 픽셀 차이가 있었으므로 입력 디코딩 품질에 관한 기존 한계도 유지된다.

[영상·생성 단계별 전체 후보 집합 HTML](report.html) · [정량 JSON](../../../../../artifacts/runs/rollout/stochastic/pilot/300/metrics/summary.json) · [진행 로그](../../../../../artifacts/runs/rollout/stochastic/pilot/300/logs/progress.log)
