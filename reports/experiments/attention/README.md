# Attention 실험

Attention 분석 코드와 측정 결과를 모은 독립 실험 폴더다. 기존 MTLA 기반 300개 분석과 TimeLens 9,404쌍의 answer → video 분석을 보관한다.

- [Thinking pilot 300 × 입력 순서 2종 최종 요약](results/input_order/timelens_thinking/pilot/300/FINAL_REPORT.md): 전체 생성 토큰의 attention, 선택한 6개 layer의 head 평균 및 전체 36개 layer의 head별 범주 합계. [실행 상태](artifacts/runs/input_order/timelens_thinking/pilot/300/metadata/status.json).
- [기존 300개 분석](results/token_attention/mtla/pilot/300/REPORT.md): query / generation / dynamics / hidden / spatial.
- [TimeLens answer → video 분석](results/input_order/timelens_instruct/full/9404/final_report.md): 입력 순서별 attention 총량, GT 집중도, 시간 분포.
- [별도 zero-shot 실험](../zero-shot/README.md): 입력 순서별 성능, span-head, top-30.
- [공용 순수 zero-shot 예측](../zero-shot/artifacts/common/predictions/legacy/README.md): 측정 필드를 제외한 재사용 가능한 원본 예측.

TimeLens `artifacts/runs/input_order/timelens_instruct/full/9404/`의 샘플별 `measurement.json`은 공용 예측 파일의 경로와 SHA-256을 참조한다. `attention.npz`, attention 통계·검증·그림은 이 실험에만 둔다. 응답·생성 토큰·예측 구간 원본과 입력 metadata·확장 프롬프트는 공용 저장소에서 읽는다.

```bash
cd /home1/irteam/experiments/attention
.venv/bin/python run.py report
.venv/bin/python run.py timelens report
.venv/bin/python run.py timelens status
# 기존 300개 attention 분석을 재계산할 때
.venv/bin/python run.py analyze --stage generation
```

원래 데이터와 원본 배열을 보존했다. TimeLens 측정은 EOS 제외 consumed answer 행, L8–21의 head·token 평균이며 live post-RoPE·KV-cache Q/K의 FP32 재구성값이다. 순서별 생성 답변도 달라지므로 attention과 성능 변화 사이의 인과를 단정하지 않는다.

Python 환경·TimeLens vendor·입력 전처리는 `../_shared/timelens/`에서 공유한다. MTLA 코드는 이 attention 실험에만 속한다. Zero-shot 코드에서 attention 계측 모듈을 import하지 않는다.

TimeLens 성능 지표와 실패 예측 목록은 `../zero-shot/results/baseline/input_order/overview/`에서 관리한다. 이 폴더의 TimeLens 집계·그래프에는 attention 지표만 저장한다. 기존 `results/token_attention/mtla/pilot/300/generation/`의 300개 답변은 TimeLens와 다른 프롬프트·전처리로 생성한 결과이며 `answer.json` 한 벌을 유지한다.

설정은 `artifacts/common/configs/`, 출처·환경·이전 문서는 `artifacts/common/provenance/`, MTLA 소스는 `_lib/vendor/MTLA/`에 보관한다. 공통 실행 자원은 [별도 _shared](../_shared/README.md)에 둔다.

보고서는 `results/<주제>/<실험 계열>/<full|pilot>/<질문 수>/`, 실행 자료는 `artifacts/runs/`의 동일 상대 경로에 연결한다. `results/` 아래에는 `artifacts/`를 만들지 않는다.

기존 MTLA의 query/generation/dynamics/hidden/spatial은 `token_attention/mtla/pilot/300/`에 통합했다. TimeLens Instruct는 `input_order/timelens_instruct/full/9404/`, Thinking은 `input_order/timelens_thinking/pilot/300/`이다. 세 계열은 프롬프트·입력·모델 조건을 구분한다.
