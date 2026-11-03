# 기존 pilot900 중 나머지600개 결정적 rollout

기존 고정 pilot900의 `[300:900]`을 순서 그대로 실행한다. 데이터셋별200개, 총600질문·600영상이다. 기존300개와 질문·영상이 겹치지 않음을 확인했다. 기존 코호트 파일과 기존300개 응답·점수는 수정하지 않는다.

## 유지하는 설정

- 로컬 Qwen3-VL-8B-Instruct, 기존 공식 TimeLens 프롬프트, video_query, 2 FPS, 목표100,352픽셀 고정 H/W. 기존 입력 코드의 SHA-256이 원래300개 실행 snapshot과 같은지 검사했다.
- 시작·종료 숫자의 백/십/일/소수 자리, 분:초 예외의 각 필드와 자릿수를 기록한다. 출력 형식·프롬프트를 바꾸지 않고 예외는 평가에서만 초로 환산한다.
- 질문당 greedy 포함 최대32개의 서로 다른 완성 토큰 경로. 후보1/4/8/16/32 시점의 성능과 비용을 저장한다. 대안 토큰을 강제하고 나머지는 정확한 전체 prefix 조건에서 EOS까지 greedy로 완성한다. 무작위 sampling은 없다.
- 위치당 최대4개 숫자/소수점/콜론 토큰, 최소 raw 확률0.0001, 누적 raw 확률목표0.95. 경로확률 우선순위·중복 방지·최대512응답토큰을 유지한다. 생성 함수 소스를 원래300개 snapshot과 비교해 동일함을 검사했다.
- 선택 기준은 전체 응답 확률1위, 비가중 temporal medoid, 확률가중 temporal medoid, GT를 사용하는 oracle 상한이다. 실제 선택기는 GT를 사용하지 않는다.

## 입력 준비와 운영

추가600개 영상에는 기존 FPS 입력 캐시가 없어, 동일한 소스 RGB hash·PTS·디코더·resize·공간 grid·시간 계획 코드로 필요한2 FPS 프레임만 만든다. 불필요한3/4/6/8 FPS 프레임은 만들지 않는다. 영상별 계획과 픽셀 hash를 보존한다.

전처리는 spawn worker1개, 영상마다 process 교체, 최대2영상 prefetch로 GPU 추론과 병행한다. 추론은 기존처럼 모델 프로세스1개·batch1·Torch2 threads이며 다른 GPU 작업은 중단하지 않는다. GPU 여유60GiB 이상에서 시작한다. CPU quota12코어·메모리160GiB 한도를 확인했다.

상태·진행 로그는20초마다 기록한다. 완료 질문은 재실행 시 재사용한다. 채팅 종료 후 detached process가 계속 실행한다. 환경 전체 재시작 이후 자동 복구까지 보장하지 않는다.

## 자동 보고서와 검증

- **추가600개:** 이 실행의 별도 REPORT.md와 STRATA.md.
- **중첩600개:** 원래300개 + 추가범위의 앞300개. `results/deterministic_rollout_pilot600_v2/`.
- **전체900개:** 원래300개 + 추가600개. `results/deterministic_rollout_pilot900_v2/`.
- 기존300개 원자료는 hash가 바뀌지 않았는지 검사하고 참조만 한다. 통합 보고서용 원자료를 복제하거나 기존 응답을 덮어쓰지 않는다.
- 모든 보고서는 mIoU·IoU≥.3/.5/.7 성공률과 greedy 대비 변화(pp), decoder 계산량, 영상 단위 paired bootstrap95% CI, 영상 길이·GT비율·두축 교차표를 제공한다.
- 주 평가는 원래 연속 시간 구간의 IoU(교집합/합집합)이며 높을수록 좋다. clipping·반올림은 하지 않는다. 파싱 실패·역전·EOS 미도달은0점이며 영상 길이 초과는 별도 표시한다. 후보 미달 질문도 분모에 유지한다.
- 데이터셋별 첫 질문에서 cache rollback 검증을 수행한다. 일반화한 보고서 생성기가 기존300개의7,500개 질문별 평가 행을 정확히 재현하는지 CPU 검증을 통과했다.

고정900개는 층화 진단 표본이며 원래 벤치마크 무작위 평균으로 일반화하지 않는다. 사건 구분·같은 사건 경계 보정 여부는 영상 내용 검토 없이 단정하지 않는다.

- [상태](../../artifacts/runs/deterministic_rollout_additional600_v2/metadata/status.json)
- [실행 로그](../../artifacts/runs/deterministic_rollout_additional600_v2/logs/runner.log)
- [실행 진입점](../../_lib/deterministic_rollout900.py)
- [기존300개 보고서](../deterministic_rollout_pilot300_v2/REPORT.md)
