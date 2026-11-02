# 공통 실행 자원

실험 결과와 구분한 공통 자원이다. [TimeLens 실행 환경과 전처리](../_shared/timelens/README.md)를 attention·zero-shot·similarity가 함께 사용한다.

- `../_shared/timelens/.venv/`: Python 가상환경
- `../_shared/timelens/_lib/`: 입력 전처리·경로·예측 읽기 코드
- `../_shared/timelens/vendor/`, `../_shared/runtime/`, `../_shared/python_deps/`: 공통 외부 구현과 런타임 의존성
- `../_shared/timelens/browser_*`: 결과 뷰어 검증용 브라우저와 의존성

모델 가중치는 `/home/irteam/backbones/models/`, 모델 출처·revision·검증 기록은 `/home/irteam/backbones/manifests/`에 보관한다. 이 폴더에 실험별 결과를 저장하지 않는다.
