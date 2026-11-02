# MTLA attention 통합 실험

Zero-shot 원본은 **[`results/zeroshot`](results/zeroshot/REPORT.md)** 한 곳에서 관리한다.
`qwen3/`는 전체 9,404 query의 두 순서 결과, `qwen25/predictions/`는 같은 pilot 300개의
Qwen2.5 원본이며, `cohorts/pilot_300/`의 표본을 span-head와 top-30 실험이 공유한다.
후속 실험 폴더에는 각 실험의 추가 측정 결과만 두고, 공통 기준선은 이 경로를 참조한다.
모델별·표본별 zero-shot 집계는 `.venv/bin/python run.py zeroshot report`로 재생성한다.

실험은 백그라운드 실행 후 채팅을 종료하며, 진행 로그에 완료 수/전체 수, 진행률, 경과 시간, 예상 남은 시간을 기록한다.
현재 full-head top-30 실험의 [진행 로그](results/timelens_top30/logs/progress.log)와 [진행 상태 JSON](results/timelens_top30/progress.json)은 15초마다 갱신된다. ETA는 추론 완료 기준이며 최종 보고서 생성·검증 시간은 제외한다.

정량 결과와 해석은 Markdown 보고서로 제공한다. HTML은 실제 영상 재생 등 상호작용이 필요한 경우에만 작성한다.

추가 **span 전용 LM head pilot**은 [정량 결과·해석 보고서](results/timelens_span_head/REPORT.md),
[고정 샘플](results/zeroshot/cohorts/pilot_300/samples.json),
[설정·샘플 선정](results/zeroshot/cohorts/pilot_300/selection.json),
[샘플별 CSV](results/timelens_span_head/paired.csv)에 저장한다.
Charades/ActivityNet/QVHighlights-TimeLens에서 각각 100개를 기존 9,404개 결과 중
seed 42로 추출했다. 별도의 기존 TimeLens pilot manifest는 찾지 못했으며,
이 고정 목록으로 두 비교군을 맞춘다. 일반 벤치마크의 기존 300개 목록과는 다르다.
Qwen3는 기존 `video_query` zero-shot 결과와 비교하고, Qwen2.5는 같은 300개에서
기본 head zero-shot을 새로 실행해 제한 head와 비교한다.
기존 LM head의 숫자·온점·쉼표·하이픈·대괄호·콜론·공백/줄바꿈·EOS에 해당하는
526개 행만 projection한다. 입력과 프롬프트, BF16/FlashAttention 2/Transformers 4.57.1,
greedy 512토큰 조건은 유지하며 추가 학습이나 문법·범위 제약을 적용하지 않는다.
Qwen2.5는 모델 기본 repetition penalty 1.05, Qwen3는 1.0을 각 비교군에 동일 적용한다.
기존 Qwen3 입력 문자열·프레임 metadata 일치와 대표 샘플의 생성 토큰 재현을 검사한다.
최종 집계는 공식 evaluator CLI와도 대조한다. 완료 여부는 보고서와 `summary.json`에 표시한다.

```bash
# 각 모델의 shard 0, 1은 독립 실행 가능 (H200에서 총 4개 작업 동시 실행)
.venv/bin/python run.py span-head run --model qwen3 --shard 0
.venv/bin/python run.py span-head run --model qwen25 --shard 0
.venv/bin/python run.py span-head report
```

추가 실험 **TimeLens 전체 9,404개 query의 입력 순서 비교**는 2026-10-16 09:25 KST에 완료됐다.
공식 zero-shot 설정으로 `video → query`와 `query → video`를 실행하고
답변의 `answer → video attention`을 비교한다.
[최종 성능·attention 집계](results/zeroshot/REPORT.md) ·
[벤치마크별 CSV](results/zeroshot/qwen3/dataset_metrics.csv) ·
[개별 결과](results/zeroshot/qwen3/index.html) · [현재 상태](results/zeroshot/qwen3/status.json) ·
[실행 로그](results/zeroshot/qwen3/run.log)

```bash
.venv/bin/python run.py timelens status
# 저장된 결과에서 최종 집계·그래프·공식 지표 일치 검증을 재생성
.venv/bin/python run.py timelens report
# 검증이 완료된 작업을 채팅과 독립적인 백그라운드에서 시작/재개
.venv/bin/python run.py timelens start
```

Charades-TimeLens 3,363개, ActivityNet-TimeLens 4,500개, QVHighlights-TimeLens 1,541개를
두 순서로 실행한다. FPS 2, min_tokens 64, total_tokens 14336, BF16, FlashAttention 2,
greedy 생성 최대 512토큰, seed 42이며 공식 파싱·반올림·첫 구간 IoU 평가를 유지한다.
`vendor/TimeLens`는 공식 commit `5d2110fb3faf74f4b0fc5957cd7ab73bea4763dc`다.
공식 버전의 Transformers 4.57.1/FlashAttention 2.7.4.post1 등은
`runtime/timelens_deps`에서 이 실험에만 사용한다.

주 attention은 EOS를 제외한 답변 토큰을 소비하는 consumed 행의 영상 attention이다.
predictive 행도 함께 저장한다. 실제 FlashAttention forward의 출력은 그대로 반환하고,
post-RoPE·KV-cache Q/K를 FP32로 올려 전체 causal key에 대한 softmax를 별도로 계산한다.
이는 FlashAttention이 직접 반환한 attention 배열이 아닌 **live Q/K 재구성값**이다.
head 평균 후 레이어별·토큰별 시간 분포를 NPZ에 보존하고, 주 요약은 레이어 8–21을 평균한다.
두 순서는 같은 디코딩 프레임을 재사용하며 최종 입력이 정확한 블록 순열인지 검증한다.
각 벤치마크 1개씩 양 순서에서 계측 유무에 따른 생성 토큰 동일성을 검사한 뒤 시작한다.
실행 중 결과는 10쌍마다 갱신하고, 완료 표시가 있는 샘플은 재개 시 건너뛴다.
CPU 전처리는 별도 작업자 4개와 prefetch 2로 GPU 추론과 겹쳐 수행한다.
컨테이너 CPU 한도(12코어)에 맞춰 작업자별 CPU affinity, 디코더 및 PyTorch 스레드 2개,
BLAS 스레드 1개로 제한한다.
pinned memory에서 비동기로 입력을 전송하며 추론 batch size는 1을 유지한다.
`execution_policy.json`에 실행 구성을, `prefetch_validation.json`에 세 벤치마크의
직렬·병렬 전처리 입력 텐서 및 메타데이터 동일성 검증을 기록한다.
로그의 `PREFETCH wait_seconds`와 `PAIR_COMPLETE elapsed_seconds`로 전처리 대기와 처리 속도를 확인한다.

아래는 보존된 기존 실험이다. Qwen3-VL-8B-Instruct의 영상 attention을 동일한 300개 입력에서 비교한다.
Charades-STA test, QVHighlights val, ActivityNet-Captions val_1 각각 100개이며,
모든 분석이 하나의 샘플 목록을 공유한다.

- [통합 결과 뷰어](results/index.html): 데이터셋·query·답변·샘플 ID 검색과 상세 그림
- [통합 실험 보고서](results/REPORT.md)
- [샘플별 통합 CSV](results/sample_metrics.csv) / [JSON](results/sample_metrics.json)
- [통합 요약](results/summary.json) / [공통 프로토콜](protocol.json)
- [검증 결과](results/validation.json)

## 실행

공개 실행 스크립트는 `run.py` 하나다. `_lib/`는 공통 입력 처리와 분석 함수의 내부 구현이다.
모델, MTLA 소스 복사본, Python 환경, 브라우저 검증 도구는 각각 한 벌만 유지한다.

```bash
cd /home/irteam/experiments/mtla_attention

# 기존 측정값을 합쳐 통합 결과를 다시 작성한다. GPU를 사용하지 않는다.
.venv/bin/python run.py report

# 원본 배열, 수치, 그림, HTML 링크, 브라우저 동작까지 검증한다.
.venv/bin/python run.py verify

# 저장된 배열에서 모든 지표·그림을 다시 계산한다.
.venv/bin/python run.py analyze

# 특정 분석만 갱신한다.
.venv/bin/python run.py analyze --stage dynamics

# 누락된 추출을 이어서 실행한다. 완료된 샘플은 모델 로딩 전에 건너뛴다.
.venv/bin/python run.py run
.venv/bin/python run.py run --stage generation --only Charades-STA_01
```

`run --stage`는 `all`, `query`, `generation`, `hidden`, `spatial`을 지원한다.
전체 실행 순서는 query → generation → hidden → spatial이다.
`--only`와 `--limit`은 공통 300개 목록에서 실행할 샘플만 제한하며, 목록을 변경하지 않는다.
generation에는 해당 query 결과가, hidden에는 해당 generation 결과가 필요하다.
spatial은 hidden 분석에서 선정된 25개 사례의 query 교체 대조를 수행한다.
추출 후에는 `analyze`로 파생 결과를 갱신한다.

`analyze --stage`는 `all`, `query`, `generation`, `distribution`, `heads`, `dynamics`,
`hidden`, `spatial`을 지원한다. 분석·전체 검증은 보존된 300개 완성 결과를 기준으로 한다.
`verify --layout-only`는 데이터 해시와 경로만 빠르게 확인한다.

## 분석과 결과 구조

| 분석 옵션 | 결과 위치 | 측정 대상 |
|---|---|---|
| query | `results/query/` | 입력 이벤트 문장 → 앞선 영상 토큰 |
| generation | `results/generation/` | query, 생성 답변, 좌표 토큰, predictive/consumed 비교 |
| distribution / heads | `results/generation/{distribution_analysis,head_analysis}/` | 분포 형태, 레이어·head별 분석 |
| dynamics | `results/dynamics/` | 같은 생성 기록의 토큰별 변화, 4,139개 토큰 |
| hidden | `results/hidden/` | decoder output cosine과 attention 비교 |
| spatial | `results/spatial/` | 초반 편향 25개 사례의 공간 분포·query 교체 대조 |

주 분석 레이어는 0-based 8–21이다. attention의 raw mass는 전체 causal key의 softmax에서
영상 열을 선택해 합산한 값이다. 영상 내부 조건부 분포, consumed 행, predictive 행,
hidden cosine의 softmax 분포는 정의를 구분해 저장한다. GT는 입력으로 주지 않는다.
시간축은 실제 디코딩 timestamp를 사용하며 GT 구간은 질량 적분 시 영상 범위와 교집합을 취한다.
세부 설정과 원래 관측치는 각 분석의 `run_config.json`, `metadata.json`, `validation.json`에 있다.

## 통합 내역

기존 attention 실험 5개를 이 폴더로 합쳤다. 9개 탐색 실험은 같은 300개 실험의 부분집합이며
NPZ의 모든 배열이 bitwise 동일함을 확인하고 중복 결과·스크립트를 제거했다.
복제된 입력 처리와 지표 구현, 중복 MTLA 복사본, 과거 실행 로그와 임시 진단 결과도 정리했다.
300개 실험의 고유 측정값, 생성 답변, 그림, 공간 대조 결과는 보존했다.

`provenance/migration.json`에는 이전 경로, 코드 해시, 이동한 데이터의 SHA-256,
9개 탐색 샘플의 동일성 검증을 기록했다. `provenance/`의 과거 기록에 등장하는 이전 경로는
실행 의존성이 아니라 당시 이력이다. 현재 실행은 이 폴더 안의 공통 경로만 사용한다.
