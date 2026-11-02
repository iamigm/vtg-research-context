# 저장 구조

Qwen3-VL-8B-Thinking, pilot manifest 300개, video_query/query_video 각각 한 번 실행한다. 입력 프롬프트·전처리는 기존 TimeLens prepare_pair를 사용한다. 샘플당 두 순서는 같은 영상 프레임을 재사용하고 입력 token block의 순서만 바뀌는지 검사한다.

- `../../artifacts/runs/thinking_pilot_orders_v1/metadata/config.json`: 모델, 입력·생성 설정, head 압축, layer, 소스 및 manifest hash.
- `../../artifacts/runs/thinking_pilot_orders_v1/code/source.py`, `../../artifacts/runs/thinking_pilot_orders_v1/inputs/samples.json`: 실행 코드 및 고정 cohort 사본.
- `../../artifacts/runs/thinking_pilot_orders_v1/metadata/status.json`, `worker_*.json`, `../../artifacts/runs/thinking_pilot_orders_v1/logs/`: 진행률, 단계, GPU 사용, 완료 수 및 대략적인 ETA.
- `REPORT.md`, `../../artifacts/runs/thinking_pilot_orders_v1/metrics/summary.json`: 검증을 통과한 기록만 자동 집계. reasoning 길이·종료 여부·IoU·실제 저장량을 benchmark/order별로 기록.
- `../../artifacts/runs/thinking_pilot_orders_v1/samples/<dataset>/<sample_key>/<order>/`: 각 실행 결과.

각 결과 폴더에는 reasoning.txt, answer.txt, output_raw.txt, generation.json, input.json, tokens.json, result.json, attention.h5, validation.json, complete.json을 저장한다. 완료 표시는 모든 데이터의 검증 및 SHA-256 생성 이후에만 기록한다. 생성은 완료했지만 attention 저장이 실패하면 generation.json의 원본 생성 IDs를 재사용한다.

## Attention 배열

`attention.h5`는 FP32 배열을 LZF로 무손실 압축한다. 상세 key 축의 정보는 줄이지 않는다.

- `layer_00`, `layer_07`, `layer_14`, `layer_21`, `layer_28`, `layer_35`: `[1, G, N+G-1]`. 채널 1개는 query head 32개의 산술평균. 레이어 평균은 하지 않는다.
- `category_mass`: `[36, G, 32, 7]`. 모든 레이어·개별 head를 유지한다.
- `key_lengths[r] = N+r`, `query_positions[r] = N+r-1`. 행 r은 generated_ids[r]를 예측하는 query다. N은 입력 토큰 수, G는 경계 태그와 EOS를 포함한 전체 생성 수다.
- 미래 key는 0으로 채우며 분석에서는 key_lengths 이후를 제외해야 한다. 미래 0을 유효한 관측으로 평균내지 않는다.
- `generated_ids`, `key_category_ids`: 원본 토큰 및 범주 매핑. 입력 token IDs와 생성 token role은 JSON에도 보존한다.

범주는 video, input_query(사용자 event 문장), input_other(지시문·timestamp·특수 토큰 등), reasoning, answer, generated_boundary, current_position의 7개다. 각 행의 현재 query 위치는 원래 범주에서 빼고 current_position으로 옮겨서 합이 1이 되게 한다. 종료 태그를 reasoning/answer로 섞지 않는다. 영상 토큰만 따로 재정규화하지 않는다.

## 계산 및 검증

FlashAttention2로 원본 응답을 생성하고, 동일한 입력과 생성 IDs로 한 토큰씩 replay한다. 실제 post-RoPE/post-cache Q/K를 FP32로 올려 전체 causal key에 대해 softmax를 계산한다. 모델 attention 출력은 교체하지 않는다. K/V head 8개가 query head 32개에 올바르게 대응하는지 독립 reference로 확인한다.

처음 paired sample의 첫 5토큰은 실제 live generation의 attention과 replay를 비교한다. 이전에 시도한 4-token replay는 개별 head 범주 합계에서 최대 0.02272 차이가 나 본 실험에서 제외했다. 검증 로그에 과거 실패가 남아 있을 수 있으며 현재 상태와 complete.json이 최종 판정이다.

각 파일에서 행 합계, 유한·비음수 값, 미래 key 0, 모든 layer/query의 단 한 번 기록, 상세 head 평균과 범주 요약 평균의 일치 여부를 검증한다. 원본 생성은 sampling이므로 순서 변경에 따라 trace 길이와 답변 자체도 달라질 수 있다.

## Live 전환

전환 시점의 184개 완료 기록을 보존했다. `../../artifacts/runs/thinking_pilot_orders_v1/data/migration_to_live/transition.json`은 기존 complete.json 경로·SHA-256 및 전환 시 미완료 generation 목록을 기록한다. 이전 소스·config·상태도 같은 폴더에 보존한다. 신규 생성은 `thinking_pilot_live.py`로 실행하면서 attention을 즉시 추출한다. 저장된 generation만 있는 항목은 같은 token IDs를 replay해 복구한다. 신규 result.json/validation.json의 measurement_method로 live/replay_saved_generation을 구분한다. 기존 결과에 필드가 없으면 전환 이전 replay 기록이다. 최대 생성 길이·seed·레이어·head 평균·key 범위·정밀도는 유지했다.

Live 추출은 </think>를 관측하며 key 범주를 갱신하고 종료 후 배열을 실제 생성 길이에 맞춘다. 첫 샘플 전체를 live로 재실행하여 원본 token IDs, attention, EOS 조기 종료 시 저장 형태를 비교한다. 검증 결과는 `../../artifacts/runs/thinking_pilot_orders_v1/data/migration_to_live/validation.json`에 기록한다.
