# 실험 산출물 저장소

## Temporal mode search · pilot600

현재 활성 조건은 `runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/`이다. Main/tail 각각16·32·64·128개의 rollout 시도 checkpoint를 한 번의 누적 stream으로 비교한다. 총 예산은32·64·128·256회이며 query당 최대1,536개의 탐색 시도와 별도 greedy baseline1개다. 중복 시도·unique sequence·mode를 구분하고,초기 공통 조사를최대8회로 제한한 뒤 방법별 적응 선택을 실행한다. Seed42·43와 GPU worker3개는 유지한다. 이전 token-budget 원자료·checkpoint는 상위600폴더에 보존하지만 새 조건의 집계에는 포함하지 않는다. 활성 상태와 로그는 capped_rollouts의 `metadata/status.json`과 `logs/progress.log`다.

`runs/rollout/temporal_mode_search/pilot/600/`은 확률 하한·clipping·main/tail 격리를 적용한 새 stochastic rollout 실험의 단일 원자료 저장소다. `metadata/config.json`과 code snapshot으로 설정을 동결하고, 입력은 기존 deterministic의 tensor hash·frame IDs·PTS·H×W와 대조해 재사용한다. `responses/`의 query별 원자료와 조건별 atomic checkpoint를 사용하며 tail은 진단 전용이다. 상태는 `metadata/status.json`, 진행 로그는 `logs/progress.log`다. `temporal_mode_search.pilot600.responses` registry는 최종 데이터·뷰어 검증까지 `building` 상태를 유지한다.

현재 실행 범위는 `metadata/execution_scope.json`과 `metadata/effective_config.json`을 따른다. Active seed는42·43,영상 hash로 분할한 GPU worker는3개이며600 query×3조건×2seed=3,600개 선택이 목표다. 최초 설정의 seed44 원자료는 보존하지만 신규 생성·집계·뷰어 선택에서 제외한다. 최초 생성 코드8개는 유지하고 병렬 runtime snapshot과 hash는 `metadata/parallel_manifest.json`에 기록한다. Worker별 로그는 `logs/worker0.log`부터 `worker2.log`이며 공통 진행 로그와 자동 후처리는 supervisor 하나가 담당한다.

실행별 자료는 `runs/<주제>/<실험 계열>/<full|pilot>/<범위>/`(results와 동일 경로), 실행 간 재사용 자료는 `common/`에 둔다. 보고서는 별도의 `../results/<주제>/<실험 계열>/<full|pilot>/<범위>/`에 두며 그 안에 `artifacts/`를 만들지 않는다.

재사용 자료의 기준 목록은 `common/registry/*.json`이다. 파일을 새로 저장할 때 기존 분류에 등록하고, 모델·입력·생성 설정 또는 판정 기준이 바뀌면 새 버전 ID를 부여한다. 같은 ID로 다른 경로를 덮어쓰지 않는다.

## 분류

| 분류 | 저장 내용 | 주의사항 |
|---|---|---|
| `predictions` | 원문 응답, 생성 토큰 ID, 예측 구간, GT, IoU, 입력 해시 | 모델·순서·실행 버전을 명시해 선택 |
| `token_traces` | 매 생성 단계의 raw·processed top-30, 전체 어휘 entropy | top-30은 전체 분포가 아님. 누락 확률을 재정규화하지 않음 |
| `manifests` | 전체·pilot 샘플 목록과 선정 기록 | 모델 비교 시 동일 샘플 사용 |
| `inputs` | 입력 전처리 설정, 확장 프롬프트, 영상 시각 metadata | 입력 재현·동일성 검사에 사용 |
| `media` | 브라우저 재생용 영상과 변환 기록 | 추론 입력 영상과 혼동하지 않음 |
| `rollouts` | 샘플링 응답, seed, 후보 분포, 생성 설정 | deterministic zero-shot 기준선과 분리 |
| `analyses` | CSV 통계, 판정 라벨, MD, HTML | 원자료·분석 코드·판정 기준을 함께 참조 |

## 현재 전체 zero-shot 결과

`full_benchmark_v1`은 Qwen2.5와 Qwen3의 공식 input order `video_query`, full-head greedy 출력이다. 세 벤치마크 9,404개씩 총 18,808개 응답이 대상이다. 먼저 저장된 pilot 600개 응답을 재사용하고 나머지를 같은 조건으로 생성한다. 원래 zero-shot 실행과 이번 계측 실행은 별도 버전으로 유지한다. 기존 기준선과 토큰이 달라지는 사례도 덮어쓰지 않고 기록한다.

- 응답 원본: `common/predictions/full_benchmark_v1/<model>/<dataset>/<sample_key>.json`
- token distribution: `common/token_traces/full_benchmark_v1/<model>/<dataset>/<sample_key>.json`
- 입력 목록: `common/manifests/full_benchmark_v1.json`
- 실행 설정: `common/metadata/full_benchmark_v1/` — pilot과 전체 실행 구분
- 검색·무결성 인덱스: `common/index.sqlite`
- 저장 진행 상태: [common/status.json](common/status.json)
- 저장 진행 로그: [common/sync.log](common/sync.log)

응답 JSON에는 확률 배열을 넣지 않는다. `source_trace`와 해시로 token distribution에 연결한다. 분포 JSON은 추론이 원자적으로 저장한 완성 파일에 별도 hard link를 만들므로 분석 디렉터리의 원래 경로를 삭제해도 보관소 경로는 남는다. 파일 내용의 제자리 수정은 금지한다. 수정된 결과는 새 버전으로 저장한다. 이는 디스크 장애에 대비한 외부 백업은 아니다.

인덱스에는 실제 저장을 끝낸 응답만 들어간다. 실행 중에는 부분 집합이므로 전체 비교 전에 `common/status.json`의 `completed`, `total`, 모델별·데이터셋별 개수를 확인한다. `registry`의 `building`과 `complete`도 구분한다. 범위 밖 응답 형식은 원본 저장에서 제외하지 않으며, 특정 분석에서만 제외 사유를 기록한다.

## 읽기 — GPU나 모델 로드 불필요

```python
import sys
sys.path.insert(0, '/home1/irteam/experiments/zero-shot/_lib')
from artifact_store import iter_records, read_record

for entry in iter_records(run_id='full_benchmark_v1', model='qwen25',
                          dataset='charades-timelens'):
    response = read_record(entry, kind='response', verify=True)
    trace = read_record(entry, kind='trace', verify=True)
    print(response['sample_key'], response['answer'], trace['steps'][0]['raw'])
```

```bash
cd /home1/irteam/experiments/zero-shot
python _lib/artifact_store.py list
python _lib/artifact_store.py verify
# 동기화가 중단되었을 때 재개. 중복 프로세스는 lock으로 차단한다.
nohup python -u _lib/artifact_store.py sync >> artifacts/sync.log 2>&1 &
```

## 새로운 재사용 출력 추가

1. 일곱 분류 중 하나를 선택한다. 원본 응답과 파생 분석을 섞지 않는다.
2. 안정적인 ID와 버전을 정하고, 설정·표본 목록·생성 코드·의존 자료를 기록한다.
3. 기존 자료는 복제하지 않고 경로를 등록한다. 새 원본은 별도 보관 경로를 사용한다.
4. 아래 명령으로 등록한다. 분석 설명은 해당 보고서에만 쓰고 이 문서에 복사하지 않는다.

```bash
python _lib/artifact_store.py register \
  --id my_analysis.v1 --kind analyses \
  --path results/my_analysis \
  --description '정답 가능성에 대한 새 분석' \
  --producer _lib/my_analysis.py --cohort full_benchmark_v1 \
  --depends-on full_benchmark_v1.token_traces \
  --state complete
```

`--path`는 실제 존재하는 경로여야 한다. 처음에는 `building`으로 등록하고 완료·검증 후 같은 ID와 경로로 `complete`를 등록할 수 있다. `analyses`가 원본 token distribution를 수정해서는 안 된다.

## 과거 실행과 분석 문서

기존 Qwen3 기본 순서·역순, Qwen2.5 pilot 기준선은 `legacy.*` ID로 등록되어 있다. 기존 loader는 유지하며, 물리 경로는 `common/predictions/legacy/`로 통일했다. 현재 전체 실행의 Qwen2.5 결과는 위의 새 API로 읽는다. 제한 head 출력은 `analyses`, stochastic 생성은 `rollouts`로 구분한다. 과거 문서의 고정 표본 수는 해당 실행 당시의 범위다.

실험 보고서 목록은 [상위 README](../README.md)에만 유지한다.

## 중복 데이터 방지

확률 뷰어는 HTTP 서버에서 원본 JSON을 직접 읽는다. 동일 응답을 JavaScript 파일로 다시 저장하지 않는다. 공통 영상·폰트·보고서 자산은 필요한 뷰어 경로에서 hard link로 공유한다. 기존 실행 경로는 호환성 참조이며 별도 데이터 사본을 만들지 않는다. 정리 내역은 [common/cleanup.json](common/cleanup.json)에 기록한다.

### Nested diagnostic pilot cohorts (300/600/900)

Frozen samples and per-stage selection records: `common/cohorts/nested_pilot_v1/pilot_<N>/`.
The original300 is preserved as the exact prefix of600 and900. The expansion balances available duration×GT-fraction cells without reading predictions. Hashes/source references/exclusions are in `common/cohorts/nested_pilot_v1/manifest.json`. Registry IDs: `nested_pilot_v1.300.cohort`, `.600.cohort`, `.900.cohort`. Selection policy and execution constraints are in project `AGENTS.md`; do not resample these fixed cohorts.

### Selected-frame shared cache migration (scheduled after pilot900)

`_lib/wait_for_cache_cleanup.py` waits until both pilot900 runs and the integrated report are complete. It then migrates referenced RGB observations into `common/selected_frames_v1/frames/`, indexed by cache_key. Full timestamp/pixel-hash metadata stays at its original path. Old array paths are resolved through `_lib/shared_selected_frames.py`, with a per-cache migration journal under `runs/operations/selected_frame_cache_migration/shared/metadata/`. Retired array hashes are preserved there; old array files are removed only after all required RGB hashes validate. Missing requested frames are decoded and verified on demand. Unknown/unreferenced cache groups are retained, not guessed safe to remove.

## 공간 통제 전체 sweep (2026-11-02)

`spatial_sampling_pilot900_v2`가 현재 전체·경계 비교 실행이다. `common/spatial_sampling_sources_v2/`에 독립 코호트 records와 source-frame metadata를 보관하며, 새 RGB는 `common/selected_frames_v1/frames/`에서 cache key로 재사용한다. `runs/sampling/spatial_allocation/allocation_sweep/pilot/900/`에 plans/inputs/responses/metrics/logs를 저장한다. 공간 비통제 이전 실험10개는 사용자 요청으로 삭제됐고, 목록은 `maintenance/spatial_reset_20261102.json`이다. 기존 공간 통제 `spatial_budget_pilot900_v1`의7200응답만 입력 일치 시 재사용한다.

## conditional rollout 예산 진단

`runs/rollout/conditional_budget/pilot/30/`의 `common/metadata/`는 동결 설정·코호트·상태, `inputs/`는 기존 FPS 입력 hash와 PTS, `trees/`는 시작 prefix와 조건부 종료 경로, `metrics/`는 예산 곡선·분포 차이, `validation/`는 질량 보존·cache 검증, `logs/`는 detached 실행 로그, `code/`는 코드 snapshot이다. 원래 raw softmax 질량을 재정규화하지 않는다. 재실행은 `_lib/conditional_rollout_budget.py`이며 완료 질문은 재사용한다. 기존 프레임은 FPS 실행의 경로를 참조하고 복제하지 않는다.

## Pilot300 deterministic rollout

`runs/rollout/deterministic/pilot/300/`에 고정300개 전체의 새 실행을 보존한다. `rollouts/`는 강제 prefix·자릿수·완성 답·raw 경로확률·실제 계산량, `inputs/`는 고정 FPS 입력hash/PTS, `common/metadata/`는 동결 설정·코호트·상태, `metrics/`는 선택기별 성능/비용·bootstrap·자리별 변화, `validation/`는 캐시·합성 분기·보고서 검증, `logs/`는 실행/heartbeat, `code/`는 코드 snapshot이다. 프레임은 FPS 실행의 기존 캐시를 참조하고 복제하지 않는다. 재실행은 `_lib/deterministic_rollout300.py`, 보고서만 재생성은 동일 명령의 `--report`다. 완료 질문은 재사용하며 기존 v1 결과를 덮어쓰지 않는다.

## Deterministic rollout source store

원본900개 응답과 입력은 `runs/rollout/deterministic/pilot/900/{rollouts,inputs}/`에 통합했다. pilot300·600·900은 각각의 고정 cohort와 source_records.json으로 같은 원자료를 참조한다. 확장 부분만의 집계·보고서·로그·실행 출처는 보관하지 않는다. 보고서 재생성은 `_lib/deterministic_rollout900.py --report`, 계열 뷰와 전체 목록 갱신은 공통 `build_index.py`를 사용한다.

## deterministic rollout 샘플별 HTML 뷰어

`runs/rollout/deterministic/pilot/900/web/`는900개 샘플의 축약 표시 자료,26,560개 분기 목록,JS/CSS다. 원래 응답은 source_records.json의 참조로 검증하며 복제하지 않는다. 영상은 기존 full.media 캐시, 한국어 폰트는 `_lib/entropy_viewer/NotoSansKR.ttf`를 재사용한다. 종료 소수점84질문/243분기와 최초greedy1초대·백초대GT2질문의 목록은 `metrics/end_decimal_cases.json/csv`에 보존한다. 생성은 `_lib/deterministic_rollout_viewer.py`, 브라우저 검증은 `_lib/deterministic_rollout_viewer_check.py`다. 통합 서버8780을 사용하며 보고서를 재생성해도 HTML과 사례 링크를 갱신한다. 검증은 `validation/viewer_data.json`, `viewer_browser.json`이고 preview는 results의 desktop/mobile 최신본만 유지한다.

## Rollout token distribution와 cumulative path probability

`runs/rollout/deterministic/pilot/900/token_traces/<sample>/index.json`은 저장된 각 후보의 generation step을 정확한 prefix 분포에 연결한다. 같은 prefix는 샘플 내에서 한 번만 저장한다. `<prefix-hash>.f32.gz`는 전체 어휘 순서의 little-endian FP32 확률이며, `vocab.json`이 토큰 ID를 설명한다. index의 top30/observed는 표시용 부분집합으로 재정규화하지 않는다. `prefix_log_probability`는 해당 prefix까지 generation step 확률의 로그 합이고 전체 응답은 EOS를 포함한다. 재측정은 `_lib/rollout_token_distributions.py`, 현재2분할 운영은 `_lib/rollout_distribution_parallel.py`다. 입력 hash·greedy suffix·강제 토큰 확률·전체 응답 확률 곱을 원자료와 대조한다. 원래 응답/점수는 수정하지 않는다. 상태는 `common/metadata/distribution_status.json`, 로그는 `logs/distribution_capture.log`와 worker별 로그다. HTML은 완료된 샘플부터 자동 표시하며 `_lib/rollout_distribution_viewer_check.py`로 분포와 누적곱·전체 어휘 검색을 검증한다.

분포 측정 완료 후 `_lib/rollout_distribution_integrate.py`가 원래900개 REPORT.md와 viewer.html에 검증 결과를 통합한다. 실행 중인 측정에는 `--wait` 후처리를 연결했다. 별도 분포 실험 run이나 페이지를 생성하지 않으며 DISTRIBUTIONS.md가 임시로 생기면 기존 REPORT.md로 통합 후 제거한다. 원자료는 기존 run의 token_traces category에 유지한다. 통합 상태는 기존 common/metadata/status.json의 token_distributions와 common/metadata/distribution_integration.json, 로그는 logs/distribution_integration.log다.

고정 중첩pilot300/600/900별 분포 준비 여부를 `_lib/rollout_pilot_html_watch.py`가 확인한다. 각 고정 코호트의 분포가 모두 준비되면 해당 기존 results/run의 viewer.html을 게시하고 통합 목록을 갱신한다. 분포와 샘플 payload는900개 실행의 token_traces·web/samples를 공유하고 복제하지 않는다. 진행은900개 실행 common/metadata/pilot_html_progress.json, 로그는 logs/pilot_html.log다.

## 2026-11-04 최종 분류와 누적 pilot

`runs/<주제>/<실험 계열>/full/<질문수>/` 또는 `runs/<주제>/<실험 계열>/pilot/<300|600|900|overview>/`를 results와 동일하게 사용한다. 주제는 baseline/uncertainty/sampling/operations다. 버전이 다른 공간 배정은 spatial_allocation 아래 budget_v1/sweep_v2로 구분한다. 전체·pilot 혼합 비교는 해당 실험 계열의 overview에 두고 두 범위를 명시한다. 운영 공유 자료는 shared에 둔다.

`pilot_catalog.py`가 각 계열의 pilot/overview/metadata/catalog.json과 표본별 common/metadata/cohort_reference.json을 생성한다. 없는 단계의 결과는 만들지 않으며 원자료·응답을 복제하지 않는다. 공간 예산 pilot300/600은 기존900 응답의 부분집합이다. run_id와 원본 응답·입력 JSON 및 설정 hash는 유지한다. 동결 metadata의 과거 경로는 공통 result_layout.relocate_path로 읽고, 재생성 코드는 새 위치에 쓴다.

rollout도 같은 계층으로 통합했다. 기존 FPS 임시 링크는 제거됐고 공통 path resolver가 고정 입력 metadata의 경로를 해석한다.

뷰어의 기본 흐름은 샘플 선택 → 정답·분기 전후 구간 비교 → 바꾼 토큰과 조건화된 다음 token distribution다. 목록은 분기별 중복 행을 샘플당 카드 하나로 묶으며 각 카드의 대표 변경은 현재 정렬·필터에 따른다. greedy 응답과 분기 직전 부모가 다르면 구분해서 표시한다. 최종 후보 선택기는 접힌 보조 분석이며, 선택기 설정을 바꿔도 확인 중인 분기는 유지한다. 기본 분포는 실제 분기 위치에 자동 정렬되고 두 경로의 전체 응답 확률(EOS 포함)을 함께 표시한다. 분포 위의 디코딩 토큰 버튼으로 위치를 선택하며 분포·후속 경로가 함께 갱신된다. 검색·데이터셋·샘플 정렬은 나란히 표시하고 변화 유형 선택과 독립적으로 유지한다. 초기 화면은 모든 샘플, ID순이다.

Rollout UI는 공통 prefix/분기 이후 경로의 토큰별 확률과 누적곱을 가로 두 줄로 표시한다. top-k/top-p는 원래 전체 어휘 확률 기준이며, 상위30개로 부족하면 FP32 전체 배열을 읽는다. 경로 토큰의 추가 표시를 별도 표시하고 검색 시 전체 어휘를 사용한다. 수치 필터는 greedy/부모/분기 IoU와 두 기준 ΔIoU, 분기 토큰 순번, 대체 토큰 확률의 min/max다. 영상·응답 원문 검토를 하나로 합쳤다. 응답 쌍 선택과 디코딩 위치 선택을 구분한다.

## Deterministic rollout / reconsideration

`runs/rollout/deterministic/pilot/600/reconsideration/`에 새 model review와 final answer(`responses/`), review-conditioned FP32 token distribution(`token_traces/`), input hash 참조(`inputs/`), metrics·validation·logs·source snapshots를 저장한다. 원래 후보·영상·frame arrays·baseline distribution은 `runs/rollout/deterministic/pilot/900/`를 참조하고 복제하지 않는다. config/cohort/provenance/source_manifest는 metadata에 동결한다. 실행은 `_lib/rollout_reconsideration.py`, 보고·검증·HTML은 `_lib/rollout_reconsideration_report.py`, `_lib/rollout_reconsideration_check.py`다. 두 worker의 완료 후 supervisor가 보고서를 자동 생성하며 고정 첫300 query가 준비되면300 보고서를 먼저 생성한다.600 결과와300 결과는 같은 responses를 참조한다. supervisor와 worker의 진행·오류는 `metadata/status.json`, `metadata/worker*.json`, `logs/`에서 확인한다.

Reconsideration 뷰는 같은 cohort manifest와 사례 유형·필터를 재사용하고, trajectory·분기·baseline 분포는 원본 뷰 링크로 연결한다. 본문은 agent action–observation, verifier 점수 순위, TTS 대진과 direct regeneration 대조를 각각의 형태로 보여준다.300은600의 responses를 참조한다. frozen config와 과거 source snapshot의 이전 family 값은 원래 실행 출처이며 보존한다. 현재 저장 위치와 재생성 코드는 deterministic 하위 경로를 사용한다. 이동 검증은 `maintenance/reconsideration_nested.json`에 있다.
# Event proposal and tree search

`runs/rollout/event_tree_search/pilot/900/`에900 query의 원자료를 한 번 저장한다. `responses/`의 query JSON은 모델 기반 proposal, event/boundary tree, 선택·backtracking, context IDs, raw token probability·path probability, model-call 참조 및 비용을 보존한다. `inputs/`는 deterministic rollout의 frame/input hash와 원자료 참조를 보존하며 영상·기존 token trace를 복제하지 않는다. `metadata/`는 동결 설정·cohort/source hash·실행 상태, `logs/`는 통합 실행 로그, `validation/`은 검증, `metrics/`는 파생 지표, `web/`는 조회 자료다. pilot300·600은 같은900 응답의 고정 prefix를 참조하고 MD·viewer만 생성한다. 재실행은 동일 config hash·input hash일 때 checkpoint와 exact model-operation cache를 재사용한다. 새 생성 정책으로 기존 응답을 덮어쓰지 않는다.

## Reconsideration canonical storage

Deterministic rollout의 reconsideration 응답·입력·token distribution·실행 로그는 `artifacts/runs/rollout/deterministic/pilot/900/reconsideration/`에 한 번 저장한다. pilot300·600·900은 각 scope의 `metadata/source_records.json`으로 같은 원자료를 참조한다. 기존600 응답의 byte hash와 과학적 설정 동등성은 canonical900의 `metadata/reuse.json`에 기록한다. 원래600 응답 내부 config hash는 보존하며, legacy config와900 config의 scope 외 과학적 설정이 같을 때만 재사용한다.

## Temporal mode search · pilot600 · capped rollouts

`runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/`는3방법×seed42/43×main/tail 각각16·32·64·128회 checkpoint를 저장한다. `responses/`의 기존 token IDs·probability·선택은 유지하고 native forward 계측을 추가했다. `metadata/manifest.json`은 생성 정책의 원본 snapshot,`metadata/cost_manifest.json`은 계측 확장 코드·원본 manifest hash다. 계측 전 checkpoint의 시간은 결측으로 남긴다.

`metrics/cost_per_query.json`은 기존 경로로 재구성한 independent-case decoder inputs·forward·attention pairs와 checkpoint별 실측,`metrics/costs.json`은 모든 방법·예산의 공통 query 비교다. 현재 independent trace work는 bounded prefix cache를 적용한 native forward 기준이고 이전 single-KV 연산은 `single_kv_reference`로 보존한다. 실제 시간은 기존 replay와 MPS+prefix cache를 분리한다. Cache 저장·복원 CUDA 시간과 GPU/CPU memory도 따로 기록한다. attention pairs는 context workload proxy이며 FLOPs가 아니다. Tail을 제외한 operational 비용과 tail 포함 research 비용을 모두 보존한다. `responses/*.json`의 `checkpoint.cost`는 main/tail/judge·누적 sweep CUDA stream 시간·wall 시간·memory이고,`measured_actual_cost`는 재개 실행을 포함한 측정 구간의 누적 실제 비용이다.

재개 진입점은 동결된 `code/temporal_mode_prefix_cache_runtime.py`다. 입력·모델·확률·RNG·rollout 상한을 유지하고 최대4,096개 exact prefix의 KV/CPU logits를 재사용한다. `metadata/prefix_cache_manifest.json`은 원본 generation·cost manifest의 hash와 최적화 코드 snapshot을 연결한다. worker3개는 전용 `/tmp/irteam_temporal_mode_mps` pipe의 NVIDIA MPS를 사용하며 supervisor가 시작하고 정상 완료 후 종료한다. 실행 전후 raw token distribution·draw·mode·judge의 일치를 검증했다. 현재 사용하는 Python은 프로젝트 `.venv/bin/python`이고 프로젝트 `_lib`를 PYTHONPATH에 둔다. 보고서·HTML 재생성은 같은 code의 `temporal_mode_prefix_cache_report.py --initialize`,검증은 `temporal_mode_capped_cost_check.py`와 기존 capped 검증을 사용한다. GPU smoke는 입력 hash·greedy·draw stream·선택 보존과 재구성 counter의 native forward 일치를 검사한다. 완료 후 보고서·검증·공통 index 갱신은 supervisor가 자동 수행한다. 상태는 `metadata/status.json`,통합 로그는 `logs/progress.log`,worker 로그는 `logs/worker0.log`–`worker2.log`다.
