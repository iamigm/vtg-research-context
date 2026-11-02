# 공유 TimeLens 실행 자원

attention·zero-shot·similarity가 같은 버전의 환경과 입력 처리를 사용하도록 Python 환경, 의존성, TimeLens vendor 및 브라우저 도구를 한 벌만 보관한다. 각 실험은 이 공통 위치를 사용하며, attention·zero-shot의 `.venv`만 실행용 링크로 유지한다.

- `_lib/timelens_paths.py`: 실험 및 공용 예측 경로.
- `_lib/timelens_inputs.py`: 공식 TimeLens 입력 전처리. Attention 계측을 포함하지 않는다.
- `_lib/timelens_store.py`: 순수 예측 읽기 API. 기본 입력 순서는 `video_query`다.

실험 코드: `../../attention`, `../../zero-shot`. 순수 예측: `../../zero-shot/artifacts/predictions/legacy`.
