# 산출물 저장 기준

이 프로젝트의 산출물 저장소는 이 폴더 한 곳이다.

- `runs/<주제>/<실험 계열>/<full|pilot>/<N>/`: 같은 이름의 `../results/<주제>/<실험 계열>/<full|pilot>/<N>/` 보고서에 대응하는 수치·원자료·배열·영상·자산·로그.
- `common/`: 여러 실행이 공유하는 입력·설정·출처. 필요한 경우에만 생성한다.

실행별 폴더에 다시 `artifacts/`를 만들지 않는다. [전체 저장 지침](../../AGENTS.md)을 따른다.

## 완료된 embedding에서 CPU 후처리 재개

모든 영상 embedding이 있고 점수 계산이나 보고서 단계만 중단됐다면 GPU 추론을 재실행하지 않는다. `SIMILARITY_RUN`·`SIMILARITY_MANIFEST`를 해당 실행으로 설정하고 공유 Python으로 `_lib/run_pipeline.py --postprocess-only`를 실행한다. `runs/<주제>/<실험 계열>/<full|pilot>/<N>/logs/pipeline.log`에 출력을 append한다.

점수 계산은 query embedding을 한 번만 압축 해제하고, 같은 영상의 query들을 모아서 처리한다. 기존 score 파일은 보존하고 누락된 score만 생성한다. 실행 감시는 embedding 수뿐 아니라 `completed_samples`도 추적한다. 정량 검증에서도 query embedding 배열을 한 번만 읽는다.

## 16fps pilot 비용 측정

`runs/compute_cost/embedding_vs_generation_16fps/pilot/300/`의 자료 종류는 `config/`(고정 cohort·조건), `inputs/`(공통 frame·timestamp·token IDs), `predictions/`(모델별 답변·시간·FLOPs), `embeddings/`, `scores/`, `measurements/`(요약·샘플 CSV·검증), `media/`, `common/logs/`다. 사람이 여는 보고서·HTML·최신 PNG는 `../results/compute_cost/embedding_vs_generation_16fps/pilot/300/`에 생성한다.

공유 Python으로 `_lib/benchmark_16fps.py`를 실행하면 완료된 샘플 JSON을 재사용해 재개한다. 기본 단계는 `prepare,clip,qwen,instruct,report`이며, 예를 들어 후처리만 재실행하려면 `--stages report`를 사용한다. 출력은 `runs/compute_cost/embedding_vs_generation_16fps/pilot/300/logs/benchmark.log`에 append한다. 다른 GPU 작업을 종료한 뒤 재개해야 성능 측정 조건을 유지할 수 있다.

## 기존 결과에 인접 frame cosine 통합

별도 run을 생성하지 않는다. 기존 score NPZ에 `adjacent_scores`, `adjacent_start_timestamps`, `adjacent_end_timestamps`와 원본 embedding fingerprint를 추가하고, 기존 MD·CSV·HTML에 설명·통계·그래프를 반영한다. Frame–query score·timestamp는 유지한다. 보고서 생성기도 이 값을 재사용한다.

`_lib/integrate_adjacent.py`는 2fps pilot→16fps pilot→full 순서로 기존 embedding을 CUDA에서 batch 계산하고 CPU ZIP 읽기를 병렬화한다. 같은 영상의 다른 query는 동일한 인접 cosine 배열을 공유한다. 마지막에 각 뷰어의 frame pair 선택·영상 seek·점 개수를 검증하고 통합 목록을 갱신한다. 공통 진행 로그는 `common/logs/adjacent_integration.log`, 실행별 상태는 `runs/<주제>/<실험 계열>/<full|pilot>/<N>/logs/adjacent_status.json`에 둔다.

주제 → 실험 계열 → full/pilot → 실제 질문 수 순서를 사용한다. 표본이 없는 조사·공유 자료는 shared에 두고, 전수 메타데이터의 다중 출처는 full/all_sources에서 각각의 모수를 구분한다. 상세 기준은 프로젝트 AGENTS.md와 상위 공통 AGENTS.md를 따른다.
