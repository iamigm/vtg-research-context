# 공간 통제 전체 sweep — 실행 중

pilot900, 두 공간 예산, 전체·경계2·4·8배, Random42/43, A/B 두 배정 방식입니다. 완료 후 IoU 기준별 성능과 각 축·두 축 교차표를 자동 생성합니다. [실험 명세](SPEC.md)

## 현재 실행 로그

현재는 두 추론 작업자가 영상을 나눠 처리합니다. 아래 두 로그가 각각 갱신됩니다.

- [작업자 0 로그](../../artifacts/runs/spatial_sampling_pilot900_v2/logs/worker0.log)
- [작업자 1 로그](../../artifacts/runs/spatial_sampling_pilot900_v2/logs/worker1.log)
- [전체 진행 상태 JSON](../../artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json)
- [메모리 사용 기록](../../artifacts/runs/spatial_sampling_pilot900_v2/logs/resources.jsonl)

`parallel_supervisor.log`는 감독 프로세스 오류 출력용이므로 정상 실행 중 비어 있을 수 있습니다. 메모리 제한으로 단일 추론에 복귀하면 `singleton_fallback.log`에 기록합니다.

## 종료된 실행 로그

이전 `runner.log`와 단일 실행 감독 로그는 `logs/history/`로 옮겼습니다. 파일명에 종료 시각(2026-11-02 18:50)을 표시했습니다. 현재 진행 확인용 로그가 아닙니다. 별도의 통합 텍스트 로그는 만들지 않았습니다.
