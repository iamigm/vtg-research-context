# TimeLens pilot: full-head autoregressive top-30

기존 pilot 100개 × 3개 벤치마크 × 2개 모델 = 600개 응답, 총 10,775개 생성 step을 기록했다. EOS step도 포함한다.

기존 `experiments/zero-shot/predictions/cohorts/pilot_300/samples.json`의 seed 42 표본을 그대로 사용했다. 별도의 원본 TimeLens pilot manifest는 발견되지 않아 앞선 실험에서 정한 표본이며, 이번 실험에서 재추출하지 않았다.

모든 600개 응답에서 생성 token ID 전체가 저장된 기존 zero-shot full-head 결과와 정확히 일치했다. 따라서 아래 grounding 지표도 기존 full-head와 동일하다. 숫자 제한 head를 적용하지 않았다.

| 모델 | 벤치마크 | N | R@0.3 (%) | R@0.5 (%) | R@0.7 (%) | mIoU (%) | 평균 생성 토큰 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen25 | charades-timelens | 100 | 61.00 | 42.00 | 17.00 | 40.65 | 17.30 |
| qwen25 | activitynet-timelens | 100 | 41.00 | 27.00 | 15.00 | 29.96 | 18.13 |
| qwen25 | qvhighlights-timelens | 100 | 50.00 | 24.00 | 8.00 | 30.83 | 18.48 |
| qwen3 | charades-timelens | 100 | 65.00 | 49.00 | 31.00 | 47.53 | 17.14 |
| qwen3 | activitynet-timelens | 100 | 55.00 | 43.00 | 28.00 | 40.77 | 18.22 |
| qwen3 | qvhighlights-timelens | 100 | 73.00 | 62.00 | 45.00 | 57.13 | 18.48 |

확률은 BF16 모델의 logits를 FP32로 변환하여 **전체 LM-head 어휘**에 softmax를 적용한 값이다. Top-30 안에서 재정규화하지 않았다. HTML 기본 표시는 generation processor 이전 raw 분포이며, 보정 후 분포도 선택할 수 있다. Qwen2.5의 repetition penalty는 1.05, Qwen3는 1.0이다. Greedy decoding이므로 이 분포는 실제 무작위 샘플링을 의미하지 않는다.

| 모델 | 벤치마크 | 모든 step 평균 선택 확률 (%) | 숫자 step 평균 선택 확률 (%) | 평균 top-30 질량 (%) | 평균 엔트로피 (nat) | raw 최대 확률보다 낮은 토큰 선택 |
|---|---|---:|---:|---:|---:|---:|
| qwen25 | charades-timelens | 82.98 | 44.49 | 100.00 | 0.455 | 79 |
| qwen25 | activitynet-timelens | 81.33 | 45.09 | 100.00 | 0.495 | 76 |
| qwen25 | qvhighlights-timelens | 81.01 | 45.86 | 100.00 | 0.492 | 72 |
| qwen3 | charades-timelens | 86.12 | 53.83 | 100.00 | 0.365 | 0 |
| qwen3 | activitynet-timelens | 85.32 | 57.11 | 100.00 | 0.407 | 0 |
| qwen3 | qvhighlights-timelens | 87.18 | 63.85 | 100.00 | 0.361 | 0 |

동률 토큰의 top-k 표시 순서는 greedy argmax의 선택 순서와 다를 수 있다. 선택 확률이 최대 확률과 같은지 검증하며 동률은 불일치로 세지 않는다. 숫자 step은 토큰 문자열이 공백을 제외하고 숫자 또는 소수인 경우로 정의했다. 각 step에 동일한 가중치를 주므로 긴 응답의 영향이 더 크다. 자연어 정형 문구의 높은 확률이 전체 평균을 높일 수 있어 숫자 step을 별도로 제시했다. 토큰 확률은 영상 구간 정답 확률이나 calibration 지표가 아니다.

- charades-timelens: Qwen3의 mIoU는 Qwen2.5보다 +6.88 percentage points, R@0.5는 +7.00 percentage points 차이였다.
- activitynet-timelens: Qwen3의 mIoU는 Qwen2.5보다 +10.81 percentage points, R@0.5는 +16.00 percentage points 차이였다.
- qvhighlights-timelens: Qwen3의 mIoU는 Qwen2.5보다 +26.31 percentage points, R@0.5는 +38.00 percentage points 차이였다.

동일한 입력·프롬프트·프레임 전처리와 greedy 설정(max_new_tokens=512)을 재사용했다. 공식 시간 파서 및 반올림 후 첫 구간 IoU를 사용했다. GPU는 H200에서 CUDA MPS로 두 모델의 4개 shard를 병렬 실행했다. 확률 기록은 logits를 변경하지 않는 forward hook과 processor로 수행했다. 기록 비용이 포함되므로 실행 시간은 모델 순수 추론 속도 비교용이 아니다.

[영상 및 step별 top-30 탐색](report.html) · [정량 JSON](../../artifacts/runs/timelens_top30/metrics/summary.json) · [step별 CSV](../../artifacts/runs/timelens_top30/metrics/step_metrics.csv)

HTML은 로컬 HTTP 서버로 열고 모델별 원본 JSON을 직접 읽는다. 별도의 응답 복제본을 만들지 않는다. 영상은 브라우저용 무음 H.264 사본이며 모델 입력 원본은 변경하지 않았다. GT/예측 구간 버튼으로 영상 시각을 이동할 수 있다. 생성 step 번호는 영상 시각이 아니다.

디코더 진단: 292개 고유 영상 중 55개에서 sparse Decord MMCO 경고가 재현됐고, 그중 9개는 순차 Decord와 표본 RGB hash가 달랐다. 기존 baseline과 동일 생성 토큰이라는 사실은 입력 영상의 디코딩 정확성까지 보증하지 않는다. 원본 입력/decoder를 그대로 유지한 비교 결과이며, 자세한 범위와 증거는 [디코더 진단 보고서](mmco_diagnosis/REPORT.md)를 참고한다.
