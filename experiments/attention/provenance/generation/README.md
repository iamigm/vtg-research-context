# Input query vs actually generated answer attention: 100 samples per dataset

이전 `mtla_query_prefill_temporal_attention_100`의 **같은 300개 샘플**에서 실제 답변을 생성하고,
입력 query와 답변 토큰의 영상 attention을 비교한다. 원본 `/home/irteam/vtg_works/MTLA`는
이 실험의 `MTLA/`로 다시 복사했으며 원본과 이전 실험 결과는 변경하지 않았다.

## 결과

- [분석 결과](FINDINGS.md)
- [Head별 분포 뷰어](results/head_analysis/index.html) / [초반 편향 분석](results/head_analysis/REPORT.md)
- [36개 레이어 × 32개 head PDF](results/head_analysis/all_layers_heads.pdf) / [head별 순위 CSV](results/head_analysis/rankings.csv)
- [300개 query·답변·비교 그림 검색](results/index.html)
- [요약 그림](results/summary.png) / [레이어별 비교](results/layer_comparison.png)
- [Charades-STA 100개 비교 곡선](results/Charades-STA_overview.pdf) / [상세](results/Charades-STA_detail.pdf)
- [QVHighlights 100개 비교 곡선](results/QVHighlights_overview.pdf) / [상세](results/QVHighlights_detail.pdf)
- [ActivityNet-Captions 100개 비교 곡선](results/ActivityNet-Captions_overview.pdf) / [상세](results/ActivityNet-Captions_detail.pdf)
- [전체 상세 PDF](results/comparison_report.pdf)
- [샘플별 수치 CSV](results/sample_metrics.csv) / [JSON](results/sample_metrics.json)

개별 그림은 가로축이 **초 단위 시간**이고 세로축은 **attention score**다.
GT는 초록, 모델의 실제 답변에서 파싱한 구간은 빨강으로 별도 트랙에 표시한다.
원래 attention score와 영상 안에서 다시 정규화한 분포를 모두 제공한다.
Pearson correlation은 계산하지 않는다.

## 입력과 생성 조건

| 데이터셋 | split | 샘플 | FPS | min/max pixels | 프롬프트 |
|---|---|---:|---:|---|---|
| Charades-STA | test | 100 | 2 | 4096 / 131072 | MTLA single-span prompt 그대로 |
| QVHighlights | val | 100 | 1 | 4096 / 65536 | MTLA multi-span prompt 그대로 |
| ActivityNet-Captions | val_1 | 100 | 1 | 4096 / 65536 | Charades single-span prompt + QVHighlights 전처리 |

이전 샘플 목록을 복사해 사용한다. 재추첨하거나 생성/파싱 실패를 이유로 샘플을 교체하지 않는다.
프롬프트는 영상 timestamp/patch 토큰 → 위치 추정 지시문 → `Event: query` 순서의 원래 Qwen chat template이다.
실제 `<prefill>`, `<query>` special token을 추가하지 않는다. GT나 정답 답변은 모델에 넣지 않는다.

모델은 `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`다.
**샘플당 답변 한 개**, greedy decoding(`do_sample=False`, `num_beams=1`), `max_new_tokens=128`,
repetition penalty 1.0, BF16, eval mode로 실행한다. MTLA 영상 데이터셋의 rollout 0 greedy 기준과
128-token 설정을 따른다. 생성 엔진은 동일 attention을 직접 읽기 위해 HF eager를 사용하며 vLLM은 사용하지 않는다.
16개 rollout 생성·NMS·confidence re-ranking 실험은 수행하지 않는다.

모델의 raw 답변은 `answer.txt`, 실제 생성 token ID/토큰 문자열과 파싱한 구간은 `answer.json`에 저장한다.
단일 구간 지시문은 실제로 `s - e seconds` 또는 설명 문장으로 답할 수 있다. QVHighlights는 `[s, e]`와
`[MM:SS, MM:SS]` 형식이 나올 수 있다. MTLA의 원래 `_parse_spans_with_offsets` 파서를 그대로 사용한다.
이벤트 부재 답변, 유효 구간 없음, 128-token 제한에 도달한 답변도 삭제하거나 GT에 맞게 수정하지 않는다.
제한 도달 시 완료된 구간만 파싱하며 답변 전체 attention에는 마지막 미완성 부분도 포함된다.

## 실제 생성에서 attention을 읽는 방법

모델의 `generate()` 실행 중 원래 `eager_attention_forward`의 결과를 읽는다.
Vision과 text 모두 eager이고, 긴 prefill(5,500 tokens 초과)만 독립적인 head를 4개씩 나누어
같은 함수로 계산한다. 같은 head의 전체 query×key 행렬, causal mask, softmax 축과 GQA 그룹은 유지한다.
이 방식은 이전 전체 eager 실험과 수치가 동일함을 검증한 방법이다.

1. **Prefill:** 원래 입력 query 행과 assistant prefix의 마지막 행을 저장한다.
2. **Autoregressive decoding:** KV cache를 사용한 실제 각 생성 forward에서 새로 처리되는 행을 저장한다.
3. **EOS:** attention 비교에서는 EOS를 제외한다. 마지막 답변 토큰은 EOS를 예측하는 forward에서 처리되므로
   그 행까지 자연스럽게 확보된다.
4. **128-token 제한:** 마지막으로 출력된 토큰은 아직 forward 입력으로 처리되지 않았으므로 기존 cache에
   그 토큰만 한 번 더 넣어 attention을 확보한다. 그 forward의 다음 토큰을 답변에 추가하지 않는다.

입력 query와 답변의 attention 모두 **같은 실제 생성 실행**에서 읽는다. 정답을 teacher forcing한 결과가 아니다.
이전 실험의 query 배열과 이번 prefill query 배열이 모든 샘플에서 비트 단위로 같은지 확인한다.

## 답변 토큰의 두 가지 위치 관례

prompt 길이를 `P`, 답변 토큰을 `r_0, ..., r_(N-1)`이라 하자.

- **토큰 처리 행(주 비교, MTLA 관례):** `r_j`를 입력으로 읽는 `P+j` 위치의 attention.
  그 행은 다음 토큰 `r_(j+1)`을 예측한다.
- **토큰 생성 직전 행(보조 비교):** `r_j` 자체를 선택할 때 사용한 `P+j-1` 위치의 attention.
  첫 답변 토큰 `r_0`의 경우 assistant prefix 마지막 행이다.

이 두 가지는 한 칸 차이가 있으므로 섞어서 해석하지 않는다. 기본 `comparison.png`는 토큰 처리 행,
별도 `predictive.png`는 생성 직전 행을 사용한다. 둘 다 NPZ에 저장하고 따로 집계한다.

## 비교 대상과 집계

- **query:** `Event:` 뒤의 입력 이벤트 문장 토큰 전체. 지시문/assistant prefix 제외.
- **answer:** 실제 생성된 답변의 전체 토큰. EOS 제외, 구두점과 설명 문장 포함.
- **coordinates:** 원래 MTLA 파서가 찾은 유효 구간 안의 숫자를 포함하는 토큰만 선택.
  실제 생성 token ID의 decode prefix로 문자 범위를 대응시키며 재토큰화로 토큰 위치를 바꾸지 않는다.
- **first_coordinate:** 각 파싱 구간의 첫 숫자 토큰. MTLA 영상 실험의 `first` 관례를 확인하는 보조값.

각 행의 softmax는 영상뿐 아니라 **모든 causal key**를 포함한다. 그 후 영상 열을 선택하고 공간 토큰을
시간별로 합한다. query/답변 각각의 선택 토큰·32개 head·레이어 8–21(0-based)을 평균한다.
답변은 query 및 앞선 답변 토큰도 key로 볼 수 있으므로, query보다 영상 attention 총량이 작다는 사실만으로
시각 정보의 인과적 사용량이 줄었다고 단정하지 않는다.

## 수치와 GT 처리

시간 좌표는 실제 sampled frame indices/FPS에서 Qwen processor가 만든 timestamp다.
time-bin 경계는 이 timestamp의 중점, 양 끝은 0/실제 영상 길이다.
영상 내부 조건부 attention은 주 곡선을 총합 1로 정규화한 것이다.
GT 질량은 각 bin 내부를 균일하다고 두고 GT와의 시간 중첩 비율을 적용해 합한다.
복수 GT와 예측 구간은 질량 계산에서 합집합 처리한다.

요약량:

- 영상 attention 총량: 영상 토큰에 보낸 원래 softmax 질량의 합.
- GT peak: 시간별 최고점이 GT 안에 있는지.
- GT mass: 영상 내부 정규화 분포 중 GT에 할당된 질량.
- GT mass gain: 같은 샘플의 `답변 GT mass − query GT mass`.
- Total variation: 영상 내부 query/답변 분포의 `0.5 × sum(abs(p-q))`.
- 예측 구간의 attention 질량 및 예측 길이를 감안한 초과량도 기록한다.
- 예측–GT tIoU: 파싱된 답변과 원본 GT의 교집합/합집합 길이 비율.
  Charades/ActivityNet은 MTLA처럼 첫 파싱 구간을 선택하고, QVHighlights는 모든 구간을 사용한다.
  QVHighlights의 ‘어느 쌍이라도 tIoU≥0.5’는 탐색적 요약이며 공식 mAP가 아니다.

GT가 실제 영상 길이를 넘는 경우 원본 GT를 보존하고 그림에 모두 표시한다.
Attention 질량 계산에만 `[0, video duration]`과의 교집합을 사용한다.
모델 답변의 raw 시간도 보존하며 GT에 맞게 보정하지 않는다.

## 저장 배열

샘플별 `attention.npz`:

- `query_temporal`: `[36,32,N_query,T]`.
- `answer_temporal`: `[36,32,N_answer,T]`, 답변 토큰 처리 행.
- `predictive_temporal`: `[36,32,N_answer,T]`, 해당 답변 토큰 생성 직전 행.
- `{query,answer,predictive}_key_mass`: 위와 같은 앞 3개 축, 마지막 축은
  `[video, input_query, generated_answer, other_prefix]` 확률 질량.
- `sampled_timestamps`, `mtla_uniform_bin_edges`.

검증에는 attention 행 합(BF16 허용 오차 0.0041), future attention 0, 시간별 합과 영상 총량의 일치,
실제 decode 위치의 연속성, 입력 text/토큰/프레임 좌표의 동일성, 이전 query 배열의 비트 일치가 포함된다.

## 재실행

기존 실험의 `.venv`, `model`, PDF 도구 `python_deps`를 symlink로 재사용한다.
원본 MTLA commit과 파일 hash는 `source_provenance.json`, `source_sha256.json`에 있다.

```bash
cd /home/irteam/experiments/mtla_query_vs_generated_answer_attention_100
.venv/bin/python run_generation.py
.venv/bin/python plot_comparison.py
.venv/bin/python analyze_comparison.py
.venv/bin/python verify_results.py
```

생성 중 별도 프로세스로 `plot_comparison.py --watch`를 실행하면 완료한 샘플부터 그림을 만든다.
생성이 완료된 샘플은 재생성하지 않는다. 원문 답변, NPZ, 검증 파일과 생성 설정을 함께 보존한다.

## 벤치마크별 샘플 폴더

샘플은 `results/<dataset>/<sample_key>/`에 저장한다. 예: `results/Charades-STA/Charades-STA_01/`.
`Charades-STA`, `QVHighlights`, `ActivityNet-Captions` 폴더 아래에 각 샘플의 배열·그림·수치를 모았다. 전체 요약과 `index.html`은 `results/`에 있다.
