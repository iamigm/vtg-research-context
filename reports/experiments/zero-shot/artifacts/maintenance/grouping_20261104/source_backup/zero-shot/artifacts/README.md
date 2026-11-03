# 실험 산출물 저장소

실행별 자료는 `runs/<run>/`, 실행 간 재사용 자료는 `common/`에 둔다. 보고서는 별도의 `../results/<run>/`에 두며 그 안에 `artifacts/`를 만들지 않는다.

재사용 자료의 기준 목록은 `common/registry/*.json`이다. 파일을 새로 저장할 때 기존 분류에 등록하고, 모델·입력·생성 설정 또는 판정 기준이 바뀌면 새 버전 ID를 부여한다. 같은 ID로 다른 경로를 덮어쓰지 않는다.

## 분류

| 분류 | 저장 내용 | 주의사항 |
|---|---|---|
| `predictions` | 원문 응답, 생성 토큰 ID, 예측 구간, GT, IoU, 입력 해시 | 모델·순서·실행 버전을 명시해 선택 |
| `token_traces` | 매 생성 단계의 raw·processed top-30, 전체 어휘 엔트로피 | top-30은 전체 분포가 아님. 누락 확률을 재정규화하지 않음 |
| `manifests` | 전체·pilot 샘플 목록과 선정 기록 | 모델 비교 시 동일 샘플 사용 |
| `inputs` | 입력 전처리 설정, 확장 프롬프트, 영상 시각 metadata | 입력 재현·동일성 검사에 사용 |
| `media` | 브라우저 재생용 영상과 변환 기록 | 추론 입력 영상과 혼동하지 않음 |
| `rollouts` | 샘플링 응답, seed, 후보 분포, 생성 설정 | deterministic zero-shot 기준선과 분리 |
| `analyses` | CSV 통계, 판정 라벨, MD, HTML | 원자료·분석 코드·판정 기준을 함께 참조 |

## 현재 전체 zero-shot 결과

`full_benchmark_v1`은 Qwen2.5와 Qwen3의 공식 입력 순서 `video_query`, full-head greedy 출력이다. 세 벤치마크 9,404개씩 총 18,808개 응답이 대상이다. 먼저 저장된 pilot 600개 응답을 재사용하고 나머지를 같은 조건으로 생성한다. 원래 zero-shot 실행과 이번 계측 실행은 별도 버전으로 유지한다. 기존 기준선과 토큰이 달라지는 사례도 덮어쓰지 않고 기록한다.

- 응답 원본: `common/predictions/full_benchmark_v1/<model>/<dataset>/<sample_key>.json`
- 토큰 분포: `common/token_traces/full_benchmark_v1/<model>/<dataset>/<sample_key>.json`
- 입력 목록: `common/manifests/full_benchmark_v1.json`
- 실행 설정: `common/metadata/full_benchmark_v1/` — pilot과 전체 실행 구분
- 검색·무결성 인덱스: `common/index.sqlite`
- 저장 진행 상태: [common/status.json](common/status.json)
- 저장 진행 로그: [common/sync.log](common/sync.log)

응답 JSON에는 확률 배열을 넣지 않는다. `source_trace`와 해시로 토큰 분포에 연결한다. 분포 JSON은 추론이 원자적으로 저장한 완성 파일에 별도 hard link를 만들므로 분석 디렉터리의 원래 경로를 삭제해도 보관소 경로는 남는다. 파일 내용의 제자리 수정은 금지한다. 수정된 결과는 새 버전으로 저장한다. 이는 디스크 장애에 대비한 외부 백업은 아니다.

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

`--path`는 실제 존재하는 경로여야 한다. 처음에는 `building`으로 등록하고 완료·검증 후 같은 ID와 경로로 `complete`를 등록할 수 있다. `analyses`가 원본 토큰 분포를 수정해서는 안 된다.

## 과거 실행과 분석 문서

기존 Qwen3 기본 순서·역순, Qwen2.5 pilot 기준선은 `legacy.*` ID로 등록되어 있다. 기존 loader는 유지하며, 물리 경로는 `common/predictions/legacy/`로 통일했다. 현재 전체 실행의 Qwen2.5 결과는 위의 새 API로 읽는다. 제한 head 출력은 `analyses`, stochastic 생성은 `rollouts`로 구분한다. 과거 문서의 고정 표본 수는 해당 실행 당시의 범위다.

실험 보고서 목록은 [상위 README](../README.md)에만 유지한다.

## 중복 데이터 방지

확률 뷰어는 HTTP 서버에서 원본 JSON을 직접 읽는다. 동일 응답을 JavaScript 파일로 다시 저장하지 않는다. 공통 영상·폰트·보고서 자산은 필요한 뷰어 경로에서 hard link로 공유한다. 기존 실행 경로는 호환성 참조이며 별도 데이터 사본을 만들지 않는다. 정리 내역은 [common/cleanup.json](common/cleanup.json)에 기록한다.

### Nested diagnostic pilot cohorts (300/600/900)

Frozen samples and per-stage selection records: `common/cohorts/nested_pilot_v1/pilot_<N>/`.
The original300 is preserved as the exact prefix of600 and900. The expansion balances available duration×GT-fraction cells without reading predictions. Hashes/source references/exclusions are in `common/cohorts/nested_pilot_v1/manifest.json`. Registry IDs: `nested_pilot_v1.300.cohort`, `.600.cohort`, `.900.cohort`. Selection policy and execution constraints are in project `AGENTS.md`; do not resample these fixed cohorts.

### Selected-frame shared cache migration (scheduled after pilot900)

`_lib/wait_for_cache_cleanup.py` waits until both pilot900 runs and the integrated report are complete. It then migrates referenced RGB observations into `common/selected_frames_v1/frames/`, indexed by cache_key. Full timestamp/pixel-hash metadata stays at its original path. Old array paths are resolved through `_lib/shared_selected_frames.py`, with a per-cache migration journal under `runs/selected_frame_cache_migration_v1/metadata/`. Retired array hashes are preserved there; old array files are removed only after all required RGB hashes validate. Missing requested frames are decoded and verified on demand. Unknown/unreferenced cache groups are retained, not guessed safe to remove.

## 공간 통제 전체 sweep (2026-11-02)

`spatial_sampling_pilot900_v2`가 현재 전체·경계 비교 실행이다. `common/spatial_sampling_sources_v2/`에 독립 코호트 records와 source-frame metadata를 보관하며, 새 RGB는 `common/selected_frames_v1/frames/`에서 cache key로 재사용한다. `runs/spatial_sampling_pilot900_v2/`에 plans/inputs/responses/metrics/logs를 저장한다. 공간 비통제 이전 실험10개는 사용자 요청으로 삭제됐고, 목록은 `maintenance/spatial_reset_20261102.json`이다. 기존 공간 통제 `spatial_budget_pilot900_v1`의7200응답만 입력 일치 시 재사용한다.

## 조건부 rollout 예산 진단

`runs/conditional_rollout_budget_pilot30_v1/`의 `metadata/`는 동결 설정·코호트·상태, `inputs/`는 기존 FPS 입력 hash와 PTS, `trees/`는 시작 prefix와 조건부 종료 경로, `metrics/`는 예산 곡선·분포 차이, `validation/`는 질량 보존·cache 검증, `logs/`는 detached 실행 로그, `code/`는 코드 snapshot이다. 원래 raw softmax 질량을 재정규화하지 않는다. 재실행은 `_lib/conditional_rollout_budget.py`이며 완료 질문은 재사용한다. 기존 프레임은 FPS 실행의 경로를 참조하고 복제하지 않는다.

## Pilot300 결정적 완성 rollout

`runs/deterministic_rollout_pilot300_v2/`에 고정300개 전체의 새 실행을 보존한다. `rollouts/`는 강제 prefix·자릿수·완성 답·raw 경로확률·실제 계산량, `inputs/`는 고정 FPS 입력hash/PTS, `metadata/`는 동결 설정·코호트·상태, `metrics/`는 선택기별 성능/비용·bootstrap·자리별 변화, `validation/`는 캐시·합성 분기·보고서 검증, `logs/`는 실행/heartbeat, `code/`는 코드 snapshot이다. 프레임은 FPS 실행의 기존 캐시를 참조하고 복제하지 않는다. 재실행은 `_lib/deterministic_rollout300.py`, 보고서만 재생성은 동일 명령의 `--report`다. 완료 질문은 재사용하며 기존 v1 결과를 덮어쓰지 않는다.

## 결정적 rollout 추가600개와 통합

`runs/deterministic_rollout_additional600_v2/`는 고정pilot900[300:900]의 새 입력·응답을 보존한다. `media/`는 동일 FPS 코드로 필요한2 FPS만 저장한 영상별 픽셀hash/PTS/RGB, `rollouts/`는 새600개 응답이며 나머지 category는 기존300개 실행과 같다. `metadata/original300.json`의 기존 응답 hash를 검증하고 재추론하지 않는다. `runs/deterministic_rollout_pilot600_v2/`와 `runs/deterministic_rollout_pilot900_v2/`에는 통합 지표·검증·원자료 참조 manifest만 두며 응답을 복제하지 않는다. 실행/재개는 `_lib/deterministic_rollout900.py`, 보고서 재생성은 같은 명령의 `--report`다.

결정적 rollout 보고서의 추가 진단은 각 실행 `metrics/recovery_and_selection.csv`, `boundary_errors.csv`, `branch_parent_comparisons.csv`, `branch_role_summary.csv`, `start_end_coupling.csv`, `marginal_efficiency.csv`, `additional_diagnostics.json`에 보존한다. 재생성은 기존 `--report` 진입점을 사용하며 `_lib/deterministic_rollout_diagnostics.py`가 정의·분모·보고서 표를 함께 생성한다. 현재 보고 코드 snapshot은 `code/reporting/`, hash는 `metadata/reporting_code.json`에 별도 저장해 원래 추론 snapshot을 보존한다.

## 결정적 rollout 샘플별 HTML 뷰어

`runs/deterministic_rollout_pilot900_v2/web/`는900개 샘플의 축약 표시 자료,26,560개 분기 목록,JS/CSS다. 원래 응답은 source_records.json의 참조로 검증하며 복제하지 않는다. 영상은 기존 full.media 캐시, 한국어 폰트는 `_lib/entropy_viewer/NotoSansKR.ttf`를 재사용한다. 종료 소수점84질문/243분기와 최초greedy1초대·백초대GT2질문의 목록은 `metrics/end_decimal_cases.json/csv`에 보존한다. 생성은 `_lib/deterministic_rollout_viewer.py`, 브라우저 검증은 `_lib/deterministic_rollout_viewer_check.py`다. 통합 서버8780을 사용하며 보고서를 재생성해도 HTML과 사례 링크를 갱신한다. 검증은 `validation/viewer_data.json`, `viewer_browser.json`이고 preview는 results의 desktop/mobile 최신본만 유지한다.

## Rollout 토큰 분포와 누적 경로 확률

`runs/deterministic_rollout_pilot900_v2/token_traces/<sample>/index.json`은 저장된 각 후보의 generation step을 정확한 prefix 분포에 연결한다. 같은 prefix는 샘플 내에서 한 번만 저장한다. `<prefix-hash>.f32.gz`는 전체 어휘 순서의 little-endian FP32 확률이며, `vocab.json`이 토큰 ID를 설명한다. index의 top30/observed는 표시용 부분집합으로 재정규화하지 않는다. `prefix_log_probability`는 해당 prefix까지 generation step 확률의 로그 합이고 전체 응답은 EOS를 포함한다. 재측정은 `_lib/rollout_token_distributions.py`, 현재2분할 운영은 `_lib/rollout_distribution_parallel.py`다. 입력 hash·greedy suffix·강제 토큰 확률·전체 응답 확률 곱을 원자료와 대조한다. 원래 응답/점수는 수정하지 않는다. 상태는 `metadata/distribution_status.json`, 로그는 `logs/distribution_capture.log`와 worker별 로그다. HTML은 완료된 샘플부터 자동 표시하며 `_lib/rollout_distribution_viewer_check.py`로 분포와 누적곱·전체 어휘 검색을 검증한다.

분포 측정 완료 후 `_lib/rollout_distribution_integrate.py`가 원래900개 REPORT.md와 viewer.html에 검증 결과를 통합한다. 실행 중인 측정에는 `--wait` 후처리를 연결했다. 별도 분포 실험 run이나 페이지를 생성하지 않으며 DISTRIBUTIONS.md가 임시로 생기면 기존 REPORT.md로 통합 후 제거한다. 원자료는 기존 run의 token_traces category에 유지한다. 통합 상태는 기존 metadata/status.json의 token_distributions와 metadata/distribution_integration.json, 로그는 logs/distribution_integration.log다.

고정 중첩pilot300/600/900별 분포 준비 여부를 `_lib/rollout_pilot_html_watch.py`가 확인한다. 각 고정 코호트의 분포가 모두 준비되면 해당 기존 results/run의 viewer.html을 게시하고 통합 목록을 갱신한다. 분포와 샘플 payload는900개 실행의 token_traces·web/samples를 공유하고 복제하지 않는다. 진행은900개 실행 metadata/pilot_html_progress.json, 로그는 logs/pilot_html.log다.
