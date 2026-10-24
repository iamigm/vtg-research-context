# 실험 산출물 저장소

재사용 자료의 기준 목록은 `registry/*.json`이다. 파일을 새로 저장할 때 기존 분류에 등록하고, 모델·입력·생성 설정 또는 판정 기준이 바뀌면 새 버전 ID를 부여한다. 같은 ID로 다른 경로를 덮어쓰지 않는다.

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

- 응답 원본: `predictions/full_benchmark_v1/<model>/<dataset>/<sample_key>.json`
- 토큰 분포: `token_traces/full_benchmark_v1/<model>/<dataset>/<sample_key>.json`
- 입력 목록: `manifests/full_benchmark_v1.json`
- 실행 설정: `metadata/full_benchmark_v1/` — pilot과 전체 실행 구분
- 검색·무결성 인덱스: `index.sqlite`
- 저장 진행 상태: [status.json](status.json)
- 저장 진행 로그: [sync.log](sync.log)

응답 JSON에는 확률 배열을 넣지 않는다. `source_trace`와 해시로 토큰 분포에 연결한다. 분포 JSON은 추론이 원자적으로 저장한 완성 파일에 별도 hard link를 만들므로 분석 디렉터리의 원래 경로를 삭제해도 보관소 경로는 남는다. 파일 내용의 제자리 수정은 금지한다. 수정된 결과는 새 버전으로 저장한다. 이는 디스크 장애에 대비한 외부 백업은 아니다.

인덱스에는 실제 저장을 끝낸 응답만 들어간다. 실행 중에는 부분 집합이므로 전체 비교 전에 `status.json`의 `completed`, `total`, 모델별·데이터셋별 개수를 확인한다. `registry`의 `building`과 `complete`도 구분한다. 범위 밖 응답 형식은 원본 저장에서 제외하지 않으며, 특정 분석에서만 제외 사유를 기록한다.

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

기존 Qwen3 기본 순서·역순, Qwen2.5 pilot 기준선은 `legacy.*` ID로 등록되어 있다. 기존 경로 및 loader는 호환성을 위해 유지한다. 현재 전체 실행의 Qwen2.5 결과는 위의 새 API로 읽는다. 제한 head 출력은 `analyses`, stochastic 생성은 `rollouts`로 구분한다. 과거 문서의 고정 표본 수는 해당 실행 당시의 범위다.

실험 보고서 목록은 [상위 README](../README.md)에만 유지한다.

## 중복 데이터 방지

확률 뷰어는 HTTP 서버에서 원본 JSON을 직접 읽는다. 동일 응답을 JavaScript 파일로 다시 저장하지 않는다. 공통 영상·폰트·보고서 자산은 필요한 뷰어 경로에서 hard link로 공유한다. 기존 실행 경로는 호환성 참조이며 별도 데이터 사본을 만들지 않는다. 정리 내역은 [cleanup.json](cleanup.json)에 기록한다.
