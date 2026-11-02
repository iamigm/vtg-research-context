# MTLA query → video prefill attention: 100 samples per dataset

기존 9개 실험을 데이터셋별 100개, 총 300개로 확대한 실험이다.
기존 결과는 `../mtla_query_prefill_temporal_attention/`에 그대로 보존했다.
원본 `/home/irteam/vtg_works/MTLA`를 이 실험의 `MTLA/`로 다시 복사했으며 원본을 수정하지 않았다.

## 결과

- [분석 결과](FINDINGS.md)
- [검색 가능한 전체 결과](results/index.html)
- [데이터셋 요약 그림](results/summary.png) / [PDF](results/summary.pdf)
- [Charades-STA 100개 곡선](results/Charades-STA_overview.pdf) / [상세](results/Charades-STA_detail.pdf)
- [QVHighlights 100개 곡선](results/QVHighlights_overview.pdf) / [상세](results/QVHighlights_detail.pdf)
- [ActivityNet-Captions 100개 곡선](results/ActivityNet-Captions_overview.pdf) / [상세](results/ActivityNet-Captions_detail.pdf)
- [전체 300개 상세 PDF](results/attention_report.pdf)
- [샘플별 수치 CSV](results/sample_metrics.csv) / [JSON](results/sample_metrics.json)
- [데이터셋별 요약 JSON](results/summary.json)

각 샘플의 그림은 가로축이 **초 단위 시간**, 세로축이 **attention score**이며,
GT `[s, e]`는 같은 시간축의 별도 초록색 막대에 표시한다.
Pearson correlation은 계산하지 않았다.

## 데이터와 프로토콜

| 데이터셋 | split | 개수 | FPS | min/max pixels | 입력 지시문 |
|---|---|---:|---:|---|---|
| Charades-STA | test | 100 | 2 | 4096 / 131072 | MTLA Charades single-span prompt |
| QVHighlights | val | 100 | 1 | 4096 / 65536 | MTLA QVHighlights multi-span prompt |
| ActivityNet-Captions | val_1 | 100 | 1 | 4096 / 65536 | Charades single-span prompt, QVHighlights 영상 전처리 |

각 데이터셋에서 query 목록을 `random.Random(20260929)`로 섞고, 같은 영상이 다시 나오면 제외해
서로 다른 영상 100개를 선택했다. 영상/GT 존재 외에 길이·GT 위치·attention 결과에 따른 필터는 없다.
기존 실험의 각 데이터셋 첫 3개가 이번 100개에 포함된다. 고정한 샘플은 `results/samples.json`에 있다.
정규화된 경로 annotation으로 로딩하고, 공식 원본 QV JSONL / Charades TXT / ActivityNet JSON과
video ID·query·GT가 같은지 300개 모두 확인한다.

ActivityNet은 MTLA가 직접 지원하지 않는 추가 데이터셋이며, 공식 `val_1`의 문장별 GT를 사용한다.
QVHighlights의 시간은 원래 제공된 clip 기준이다. 전체 영상에 대한 YouTube offset을 더하지 않는다.

## 모델 입력과 attention

백본은 `Qwen/Qwen3-VL-8B-Instruct`의 revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`다.
`<prefill>`, `<query>`는 개념상 구분이며 literal special token으로 추가하지 않는다.
원래 chat template의 `user [영상 timestamp/patch 토큰, MTLA 지시문, Event: query]` 순서를 유지한다.
관측 query는 `Event:` 뒤의 **입력 이벤트 문장**만이다. GT·정답 timestamp는 모델에 입력하지 않고,
응답 생성이나 학습도 하지 않는다. MTLA의 생성 좌표 토큰 기반 confidence 평가와 구분되는 실험이다.

주 attention score는 다음과 같다.

1. 모든 causal key에 대해 계산된 원래 attention에서 입력 query 행을 선택한다.
2. 각 temporal bin에 속한 영상 공간 토큰들의 attention을 합한다.
3. 모든 query 토큰·32개 head 및 레이어 8–21(0-based)의 평균을 취한다.

파란 곡선이 기본 레이어 8–21, 주황 곡선이 전체 36개 레이어 평균이다.
주 곡선은 영상 내부로 다시 정규화하지 않은 **확률 질량 / temporal bin**이다.
query-token 및 layer 히트맵은 각 행의 영상 attention 합이 1이 되도록 정규화한 조건부 분포다.
FPS가 다르면 bin 길이도 다르므로 데이터셋 간 원래 곡선 높이를 직접 비교하지 않는다.

시간축은 실제 frame indices/FPS로 Qwen processor가 계산한 temporal patch 대표 timestamp다.
히트맵 bin 경계는 이 timestamp들의 중점이며 첫/끝 경계는 0/실제 영상 길이다.
원래 MTLA의 `duration / T` 균등 시간 bin 매핑 그림도 `attention_mtla_time.png`에 별도로 저장한다.

## 메모리 처리와 수치 검증

Transformers 4.57.6, qwen-vl-utils 0.0.14, BF16 eval mode, vision/text 모두 원래 HF eager attention이다.
5,500 tokens 이하에서는 전체 eager 연산을 그대로 호출한다. 더 긴 입력은 독립적인 head를 4개씩 나누어
같은 eager 함수를 호출한다. 각 head의 전체 query×key 행렬, causal mask, softmax 정규화 축, GQA 그룹은 유지한다.
FPS·pixel budget·영상 길이를 줄이지 않는다. 모델의 출력 attention vector도 그대로 다음 레이어에 전달한다.

기존 실험의 긴 ActivityNet 샘플 두 개에서 전체 eager와 head 분할 결과의 저장 배열이
**비트 단위로 동일함**을 확인했다(`results/chunk_validation.json`).
시험했던 query-row 분할 방식은 BF16 수치 차이가 있어 최종 결과에 사용하지 않았고,
시험 출력은 `diagnostics/query_chunk_trial/`로 분리했다.

모든 샘플에서 토큰 round-trip, query/영상 위치, grid 토큰 수, 미래 attention 0,
attention 합, finite/nonnegative 값, 시간 합과 영상 총량의 일치를 확인한다.
원래 FP32 softmax 출력을 BF16으로 변환할 때 발생하는 반올림을 감안해 행 합 오차 한도는 0.0041이다.
추출 검증 결과는 각 샘플의 `validation.json`과 전체 `results/final_validation.json`에 기록한다.

## GT가 실제 영상 길이를 벗어나는 경우

샘플 선정 후 발견된 annotation의 범위를 이유로 샘플을 바꾸거나 원본 GT를 고치지 않는다.
그림에는 **원본 GT를 끝까지 표시**하고, 실제 영상 이후에는 회색 영역을 표시한다.
수치 요약에서 attention을 정의할 수 있는 구간은 실제 영상 안이므로 GT와 `[0, duration]`의 교집합을 사용한다.
원본과 유효 구간을 `sample_metrics.json`의 `original_gt_windows` / `effective_gt_windows`에 모두 저장한다.
GT 전체가 영상 밖이라면 수치 요약의 유효 GT 개수에서 제외하고 별도로 보고한다.

## 보조 요약량의 정의

요약량은 입력 attention과 GT의 시각적 관계를 보조하는 기술 통계이며, MR 모델의 예측 성능이나
MTLA re-ranking 성능 지표가 아니다. 별도의 통계적 상관계수나 유의성 검정은 계산하지 않는다.

- **Peak in GT:** 레이어 8–21 주 곡선의 최고점 timestamp가 GT 합집합 안에 있는지.
- **Uniform-bin reference:** 샘플링된 bin 중 GT 안에 있는 timestamp 비율. 균등하게 bin을 선택했을 때의 기대값.
- **Always-first-bin reference:** 항상 첫 temporal bin을 선택했을 때 GT 안에 있는 비율.
- **GT attention mass:** 주 곡선을 영상 내 총합 1로 정규화하고 GT에 할당된 질량을 적분한 값.
  bin 내부는 균일하다고 두고 GT와 bin의 겹치는 시간 비율만큼 합한다. 복수 GT의 중복은 합집합 처리한다.
- **GT mass excess:** `GT attention mass − GT duration / video duration`.
  균등한 시간당 attention보다 GT에 더 집중하는 정도다. 0보다 크면 더 많은 질량이 GT 안에 있다.
- **Enrichment:** `GT attention mass / (GT duration / video duration)`.
- **First-bin / first-10%-peak:** 전체 최고점이 첫 bin / 영상 초반 10% 안에 있는 샘플 비율.
- **Peak excluding first:** 첫 bin을 제외한 나머지 곡선의 최고점이 GT 안에 있는지.
  사후 집계만 변경한 보조값이며 별도의 모델 실험이 아니다.

`relative_time_summary.png`는 서로 길이가 다른 영상을 `[0,1]`의 상대 시간에 맞춘 **추가 요약 그림**이다.
개별 샘플의 원래 초 단위 곡선은 항상 별도로 제공한다. 상대 시간별 GT occupancy도 함께 표시해
단순한 영상 초반 집중과 GT 위치 편향을 구분해 볼 수 있게 한다.

## 저장 구조와 재실행

`results/<dataset>/<dataset>_<number>/`에 attention 원자료(`attention.npz`), 시간별 CSV, 실제 입력 text,
원본 annotation/전처리/토큰 위치 metadata, 검증 JSON, PNG/PDF를 저장한다.
NPZ의 `temporal_attention` 형상은 `[36 layers, 32 heads, query tokens, temporal bins]`다.
공간 토큰은 합산하되 head/query/layer 축은 보존한다.

기존 실험의 `.venv`와 `model/`은 symlink로 재사용한다. PDF 병합용 `pypdf`만 이 디렉토리의
`python_deps/`에 추가했다. 기존 결과나 원본 MTLA 코드는 변경하지 않는다.

```bash
cd /home/irteam/experiments/mtla_query_prefill_temporal_attention_100
.venv/bin/python run_attention.py --samples-per-dataset 100 --seed 20260929
.venv/bin/python plot_attention.py
.venv/bin/python analyze_results.py
.venv/bin/python write_report.py
.venv/bin/python verify_artifacts.py
```

추론과 그림 렌더링을 동시에 진행할 때는 별도 프로세스로 `plot_attention.py --watch`를 실행할 수 있다.
완료한 샘플은 재사용하며, 기존 결과와 다른 샘플 목록으로 덮어쓰기는 금지한다.
추론 로그는 `logs/run.log`, 그림 로그는 `logs/plot.log`다.

## 벤치마크별 샘플 폴더

샘플은 `results/<dataset>/<sample_key>/`에 저장한다. 예: `results/Charades-STA/Charades-STA_01/`.
`Charades-STA`, `QVHighlights`, `ActivityNet-Captions` 폴더 아래에 각 샘플의 배열·그림·수치를 모았다. 전체 요약과 `index.html`은 `results/`에 있다.
