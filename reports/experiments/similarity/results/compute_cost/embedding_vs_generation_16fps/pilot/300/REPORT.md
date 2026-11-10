<!-- report-appendix:APPENDIX.md parts=9 -->
<!-- report-part:0:begin -->
# Pilot 300 · 16fps 유사도와 Instruct 생성 비용 비교 · pilot300 (300질문)

완료: 2026-10-30T14:08:18.425470+09:00

[샘플별 HTML](report.html) · [샘플별 비용 CSV](../../../../../artifacts/runs/compute_cost/embedding_vs_generation_16fps/pilot/300/measurements/per_sample.csv) · [요약 JSON](../../../../../artifacts/runs/compute_cost/embedding_vs_generation_16fps/pilot/300/measurements/summary.json) · [검증](../../../../../artifacts/runs/compute_cost/embedding_vs_generation_16fps/pilot/300/measurements/validation.json)

<!-- report-part:0:end -->
<!-- report-part:1:begin -->
## 비교 조건

- 고정 pilot 300쌍: dataset별 100개, 고유 영상 292개. 세 모델은 같은 source frame index와 실제 PTS를 사용한다.
- 16fps 목표로 nearest source index를 선택하며 중복 index는 제거한다. 원본 fps가 16보다 낮으면 실제 unique frame rate는 그보다 낮다. 영상 전체를 사용하며 768-frame 상한이나 GT 기반 sampling은 없다.
- CLIP ViT-L/14 FP16, Qwen3-VL-Embedding-8B BF16, Qwen3-VL-8B-Instruct BF16. SDPA. 모델은 순차 실행하고 sample 간 작업을 겹치지 않는다. Similarity 내부 frame batch만 CPU prefetch한다. GPU 경쟁 작업이 감지되면 중단한다.
- 유사도: 독립 frame/query embedding을 FP32 L2 정규화하고 cosine을 계산한다. CLIP batch256, Qwen batch128; 공통 JPEG 최대 변512, Qwen max_pixels262144.
- Instruct: video→query 입력, TimeLens grounding prompt, greedy, KV cache, 최대512 output tokens, 마지막 위치의 vocabulary logits만 계산. 같은16fps frame을 video temporal patch2로 묶고 마지막 홀수 frame만 복제한다.
- Instruct의 기존 min_tokens64·total_tokens14336 전처리 정책을 유지한다. total_tokens는 soft spatial budget이다. min_pixels가 우선하므로 긴 영상의 실제 visual token 수는14336을 넘을 수 있다. 실제 해상도·frame index·visual/input token 수를 샘플별 prediction에 보관했다.
- 네이티브 전처리와 시간 결합 구조가 다르므로 같은 frame 수가 같은 model token 수나 spatial resolution를 뜻하지 않는다. 이 실험은 각 파이프라인의 비용 비교다.

<!-- report-part:1:end -->
<!-- report-part:2:begin -->
## 샘플당 시간과 연산량

| Dataset | Model | N | Mean GPU forward s | Median s | P95 s | Cached-frame wall s | Estimated TFLOPs/sample | Mean frames | Mean input tokens | Mean output tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | clip | 300 | 0.7251 | 0.9193 | 1.3690 | 7.0240 | 266.549 | 1645.1 | 12.1 | 0.0 |
| all | qwen | 300 | 15.0923 | 18.8913 | 28.7317 | 16.7870 | 4890.454 | 1645.1 | 281675.0 | 0.0 |
| all | instruct | 300 | 7.8314 | 9.2279 | 17.6330 | 14.1863 | 2332.353 | 1645.1 | 57921.0 | 18.1 |
| activitynet-timelens | clip | 100 | 0.9315 | 0.9259 | 1.5452 | 9.3125 | 342.890 | 2116.3 | 13.6 | 0.0 |
| activitynet-timelens | qwen | 100 | 20.0905 | 19.2189 | 34.1555 | 22.0990 | 6523.247 | 2116.3 | 375045.1 | 0.0 |
| activitynet-timelens | instruct | 100 | 10.5740 | 9.3343 | 21.9142 | 18.9418 | 3164.418 | 2116.3 | 74047.0 | 18.3 |
| charades-timelens | clip | 100 | 0.1911 | 0.2089 | 0.2935 | 1.9570 | 68.457 | 422.5 | 9.5 | 0.0 |
| charades-timelens | qwen | 100 | 3.9007 | 4.0641 | 6.7096 | 4.9881 | 1267.976 | 422.5 | 72991.9 | 0.0 |
| charades-timelens | instruct | 100 | 1.4145 | 1.3828 | 1.9328 | 3.1651 | 370.917 | 422.5 | 16618.6 | 17.1 |
| qvhighlights-timelens | clip | 100 | 1.0528 | 1.0545 | 1.0578 | 9.8026 | 388.298 | 2396.5 | 13.2 | 0.0 |
| qvhighlights-timelens | qwen | 100 | 21.2858 | 21.3284 | 21.5547 | 23.2738 | 6880.140 | 2396.5 | 396988.0 | 0.0 |
| qvhighlights-timelens | instruct | 100 | 11.5056 | 11.5117 | 11.6835 | 20.4519 | 3461.724 | 2396.5 | 83097.5 | 18.8 |

<!-- report-part:2:end -->
<!-- report-part:4:begin -->
## Instruct 대비 유사도 계산의 시간 차이

동일 query 쌍에서 `Instruct − similarity`를 계산했다. 양수면 생성이 더 오래 걸린다. 비율은 `Instruct / similarity`다.

| Similarity model | Mean paired GPU Δ seconds | Mean paired wall Δ seconds | Mean paired GPU ratio | Ratio of mean GPU times | Ratio of mean FLOPs |
|---|---:|---:|---:|---:|---:|
| clip | 7.1062 | 7.1623 | 9.794 | 10.800 | 8.750 |
| qwen | -7.2610 | -2.6007 | 0.473 | 0.519 | 0.477 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## 측정 정의

- GPU forward seconds: CUDA synchronize 전후 perf_counter로 재는 model 호출 wall time. Similarity는 query와 모든 frame batch 호출의 합, generation은 prefill+전체 autoregressive decode다. H2D·CPU 전처리·cosine·파일 저장은 제외한다. 모델 loading과 warmup도 제외한다.
- CUDA event seconds도 원자료에 별도 기록했다. Generation은 첫 forward의 prefill과 이후 decode 시간을 구분했다.
- Cached-frame wall: 공통 JPEG 읽기부터 전처리·H2D·모델·cosine 또는 답변 decode까지 실제 경과 시간. 결과 파일 쓰기는 제외한다. Similarity에서 CPU batch 준비와 GPU 계산이 겹치므로 CPU work와 GPU 시간을 단순 합산하면 안 된다.
- Raw 영상 decode·resize·JPEG 저장은 공통 준비 단계에서 영상당 한 번 측정하고 각 query에 해당 video의 값을 연결했다. 병렬 준비에서 측정한 비용이며 cached-frame wall과 별개다. Literal raw-video end-to-end latency로 오인하지 않는다.
- 샘플당 입력 비용은 캐시 적중을 가정하지 않는다. 같은 영상에 다른 query가 있어도 매 query에서 frame embedding을 새로 계산했다. embedding 파일만 재사용 가능하게 영상당 한 벌 저장했다.

<!-- report-part:5:end -->
<!-- report-part:6:begin -->
## FLOPs 정의와 계산식

- 추정 forward matmul/conv FLOPs다. Multiply-add=2 FLOPs, 1 TFLOP=10^12. 장치의 hardware counter로 측정한 명령 수가 아니다. Activation·normalization·softmax·embedding lookup·memory 이동·전처리 FLOPs는 제외했다.
- Linear(in,out,T)=2×T×in×out. CLIP transformer는 attention projection과 2-layer MLP, Qwen은 GQA projection과 3-projection gated MLP를 실제 config로 계산했다.
- Attention=4×hidden_size×query-key pair 수×layer 수. Language causal prefill은 S(S+1)/2 pair, cached decode j번째는 S+j pair다. Vision은 frame/temporal-patch별 full attention pair를 사용한다. 실제 fused kernel의 tile·padding 실행량은 다를 수 있다.
- Vision patch convolution, spatial merger, Qwen deepstack merger 3개, CLIP projection, Instruct 실제 생성 token별 LM head, similarity cosine 내적을 포함했다. Generation은 S개 prefill token과 G−1개 cached decode token을 계산했다.
- Similarity input tokens는 독립된 query/frame 시퀀스의 합이다. 하나의 긴 시퀀스 길이가 아니다. Generation input tokens는 단일 video–query 시퀀스 길이다.

<!-- report-part:6:end -->
<!-- report-part:7:begin -->
## 정량 성능

| Dataset | Model | mAP | MRR | Median first GT rank |
|---|---|---:|---:|---:|
| all | clip | 0.4538 | 0.5246 | 2.0 |
| all | qwen | 0.5223 | 0.6330 | 1.0 |
| activitynet-timelens | clip | 0.3530 | 0.4138 | 6.0 |
| activitynet-timelens | qwen | 0.4547 | 0.5779 | 1.5 |
| charades-timelens | clip | 0.4231 | 0.5180 | 2.0 |
| charades-timelens | qwen | 0.4408 | 0.5753 | 2.0 |
| qvhighlights-timelens | clip | 0.5854 | 0.6419 | 1.0 |
| qvhighlights-timelens | qwen | 0.6712 | 0.7459 | 1.0 |

AP는 GT frame을 positive로 한 query 내부 retrieval AP, mAP는 query별 동일 가중 평균이다. First GT rank는 첫 positive의 1-based 순위(낮을수록 좋음), MRR은 1/rank의 평균(높을수록 좋음)이다. GT는 PTS 기준 [start,end) 합집합이다. GT mass=Σ_GT softmax(cosine/τ), uniform=N_GT/N, lift=mass/uniform이며 τ=.01/.03/.07/.1/.3/1.0을 저장한다.

| Dataset | Instruct mIoU | R@1 IoU≥.5 | Parse failures | Truncated |
|---|---:|---:|---:|---:|
| all | 0.4569 | 0.4900 | 0 | 0 |
| activitynet-timelens | 0.4156 | 0.4300 | 0 | 0 |
| charades-timelens | 0.4062 | 0.4500 | 0 | 0 |
| qvhighlights-timelens | 0.5488 | 0.5900 | 0 | 0 |

<!-- report-part:7:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
