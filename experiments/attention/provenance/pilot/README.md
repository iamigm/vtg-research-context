# MTLA backbone: input query → video prefill temporal attention

## 실험 목적과 입력 정의

Qwen3-VL-8B-Instruct에 **영상이 앞에, 이벤트 query 문장이 뒤에** 오는 MTLA 원래 입력을 넣고,
입력 query가 앞선 영상 토큰에 보내는 self-attention 분포와 GT 시간 구간을 시각적으로 비교한다.
`<prefill>`, `<query>`는 역할을 설명하는 표기이며 실제 문자열 또는 새로운 special token을 추가하지 않았다.
모델의 원래 chat template은 `user [영상 timestamp/patch 토큰, MTLA 지시문, Event: query]` 다음
assistant 시작 prefix를 붙인다. query는 `Event:` 뒤의 이벤트 문장만이며, 지시문·assistant prefix는
평균 대상에서 제외한다. 실제 확장 입력은 각 샘플의 `expanded_input.txt`에 보존한다.

GT나 정답 timestamp는 모델에 입력하지 않는다. 응답 생성이나 학습도 하지 않는다.
이는 MTLA의 **생성 응답 좌표 토큰 기반 confidence/re-ranking 재현 실험이 아니라**,
동일 백본·전처리·질의 프롬프트로 수행하는 입력 query attention 분석이다.
별도 KV-cache 단계 없이 한 번의 causal forward에서 query→앞선 prefill attention을 읽는다.
Pearson correlation은 계산하지 않는다.

## 결과 보기

- [9개 샘플 개요](results/overview.png) / [개요 PDF](results/overview.pdf)
- [전체 상세 PDF](results/attention_report.pdf)
- [HTML 결과 모음](results/index.html)
- [샘플 목록과 GT](results/samples.json)
- [실행 설정](results/run_config.json)
- [관찰 결과](FINDINGS.md)

주 그래프의 가로축은 실제 디코딩 프레임으로부터 Qwen processor가 계산한 시간(초)이다.
파란색은 레이어 8–21 평균, 주황색은 전체 36개 레이어 평균이며 레이어 번호는 MTLA 코드와 같은 0-based다.
회색 점선은 같은 영상 attention 총량을 시간 bin에 균등 분배했을 때의 참고선이다.
초록색 GT 막대와 `[s, e]`는 별도 가로축 트랙에 표시한다. 복수 GT는 전부 표시한다.

세로축은 **전체 causal key에 대해 softmax한 원래 attention에서, 해당 temporal bin의 공간 토큰을 합하고
query 토큰·head·레이어를 평균한 확률 질량**이다. 영상만으로 softmax하지 않으며, 주 곡선을 임의로
최댓값 1 또는 총합 1로 정규화하지 않는다. 각 bin은 Qwen의 temporal patch(기본 2개 샘플링 프레임)에 해당한다.
곡선은 샘플 위치 사이를 선으로 연결한 표시이며, 프레임 사이의 attention을 추가 추정한 결과가 아니다.
세로축 범위는 샘플마다 다르며 bin 길이도 데이터셋의 FPS에 따라 다르다.

상세 그림의 query-token 및 layer 히트맵은 **각 행의 영상 attention 합을 1로 정규화한 조건부 분포**다.
주 곡선의 원래 확률 질량과 구분해야 한다. 히트맵 셀 경계는 실제 timestamp의 중점으로 표시한다.
MTLA의 `duration / T` 균등 시간 bin 매핑을 사용한 별도 그림 `attention_mtla_time.png`도 제공한다.
실제 timestamp와 MTLA 균등 bin 경계는 CSV/NPZ에 모두 기록한다.

## 데이터 및 MTLA 프로토콜

원본 코드는 `/home/irteam/vtg_works/MTLA`에서 이 디렉토리의 `MTLA/`로 복사했다.
원본은 수정하지 않았으며 72개 파일의 SHA-256을 `source_sha256.json`에 저장했다.
실험 스크립트는 복사본 외부에 추가했고, 복사된 원래 파일도 수정하지 않았다.

| 데이터셋 | split | 샘플 | FPS | min/max pixels | prompt |
|---|---|---:|---:|---|---|
| Charades-STA | test | 3 | 2 | 4096 / 131072 | MTLA Charades single-span prompt 그대로, 끝 마침표 제거 포함 |
| QVHighlights | val | 3 | 1 | 4096 / 65536 | MTLA QVHighlights multi-span prompt 그대로 |
| ActivityNet-Captions | val_1 | 3 | 1 | 4096 / 65536 | 문장별 단일 GT에 Charades single-span prompt, 영상 전처리는 QVHighlights와 동일 |

ActivityNet은 이 MTLA 저장소가 직접 지원하지 않는 추가 벤치마크다. 공식 `val_1`의 문장별 timestamps를
사용하며, 여러 문장을 paragraph로 합치거나 split 이름을 변경하지 않는다.
QVHighlights GT는 제공된 clip 내부의 초 단위이며 YouTube 원본 offset을 더하지 않는다.
정규화된 `annotations_with_paths` 파일로 경로를 읽되, 최종 선택 샘플의 query/video/GT가 공식 원본
QV JSONL, Charades TXT, ActivityNet JSON과 일치하는지 별도로 검증했다.

샘플은 데이터셋별 query 목록을 Python `random.Random(20260929)`로 섞은 후 서로 다른 영상에서
3개씩 선택했다. 영상 존재/GT 존재 외에 길이, GT 위치, attention 결과에 따른 필터는 없다.
총 9개에 대한 탐색적 시각화이며 벤치마크 전체의 성능이나 통계적 상관을 주장하지 않는다.

## attention 추출과 검증

Transformers 4.57.6, qwen-vl-utils 0.0.14, BF16, eval mode, vision/text 모두 `eager`를 사용했다.
복사한 MTLA의 모델 adapter에서 video message를 만들고, 원래 config에서 FPS/pixel budget을 읽는다.
영상 디코더는 `decord`로 고정했다. 원래 Qwen 모델의 `eager_attention_forward`를 그대로 호출하고,
실제 softmax attention에서 query 행/영상 열을 읽는다. 모델의 forward 출력은 바꾸지 않는다.
언어모델 출력 projection(logits)은 필요하지 않아 `model.model`까지만 실행한다.

- decode→tokenize round-trip이 입력 ID와 정확히 같은지 확인 후 query 문자 offset을 토큰 위치에 대응.
- 영상 토큰 개수가 `T × H × W / spatial_merge_size²`와 일치하고 모든 영상이 query보다 앞에 있는지 확인.
- 전체 eager attention 행과 선택 행 계산의 일치, GQA, 절대 위치 causal mask를 작은 독립 입력에서 검증.
- 실제 모든 레이어의 attention row sum, 미래 토큰 attention 0, finite/nonnegative 값을 검증.
- BF16 softmax 반올림에 따른 row sum 오차 허용값은 0.0041(약 2^-8)이며 실제 오차도 저장.
- 시간별 공간 합이 전체 영상 attention mass와 일치하는지 확인.
- 공식 annotation 대응 및 원본/복사 파일 SHA-256 불변성 확인.

샘플별 `validation.json`, `results/capture_unit_validation.json`,
`results/provenance_validation.json`에 검증 기록을 저장한다.

## 저장 배열

각 `results/<dataset>/<dataset>_<number>/attention.npz`:

- `temporal_attention`: `[36 layers, 32 heads, query tokens, temporal bins]`, float32.
  원래 BF16 attention을 float32로 올린 후 공간 토큰을 합했다.
- `key_group_mass`: `[layers, heads, query tokens, 4]`.
  마지막 축은 video / other-prefill / query / post-query 순서다.
- `sampled_timestamps`: Qwen processor가 사용한 temporal patch 대표 timestamp.
- `mtla_uniform_bin_edges`: 원래 MTLA 시간 매핑용 `linspace(0, duration, T+1)`.

`temporal_attention.csv`에는 주 곡선, 전체 레이어 곡선, 영상 내 합이 1인 조건부 곡선이 있다.
공간 위치별 원래 거대 attention 행렬은 저장하지 않으며, 시간 분포를 재집계할 수 있는 head/query/layer 축은 보존한다.
`metadata.json`에는 GT, 전처리, 실제 frame indices/FPS, 입력 토큰 위치, 원본 annotation hash가 있다.

## 재실행

이 실험의 `.venv`는 `/opt/conda/envs/DiT`의 PyTorch 2.6.0+cu124 / torchvision 0.21.0을 읽기 전용으로
사용하는 `--system-site-packages` 환경이다. 추가 패키지는 실험 내부에만 설치했다.
MTLA requirements의 torchvision 0.25.0 대신 기존 torch와 호환되는 0.21.0을 사용하며,
프레임 디코딩은 decord, attention은 고정한 Transformers 원본 구현으로 실행한다.
전체 버전은 `environment.freeze.txt`를 참고한다. vLLM은 응답 생성이 없으므로 필요하지 않다.

```bash
cd /home/irteam/experiments/mtla_query_prefill_temporal_attention
.venv/bin/python run_attention.py --seed 20260929 --samples-per-dataset 3
.venv/bin/python plot_attention.py
```

이미 저장된 attention 샘플은 건너뛴다. 다른 seed/샘플 수를 시험할 때는 기존 결과를 별도로 보존하고
새 실험 디렉토리에서 실행한다. `--prepare-only`는 모델 추론 없이 영상/토큰 매핑을 점검한다.

모델은 `model/`에 로컬 보관하며 다운로드 revision은
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`다. MTLA commit은 `source_provenance.json`에 기록했다.
`logs/run.log`가 최종 eager 실행 로그다.

## 벤치마크별 샘플 폴더

샘플은 `results/<dataset>/<sample_key>/`에 저장한다. 예: `results/Charades-STA/Charades-STA_01/`.
`Charades-STA`, `QVHighlights`, `ActivityNet-Captions` 폴더 아래에 각 샘플의 배열·그림·수치를 모았다. 전체 요약과 `index.html`은 `results/`에 있다.
