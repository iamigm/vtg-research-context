# 영상 전처리 캐시와 입력 공급 개선

같은 영상을 여러 질의에서 사용할 때 디코딩·resize 결과를 재사용한다. 원본 영상, 프레임 선택, 전처리 계산과 생성 설정은 변경하지 않았다. 입력 공급은 준비 완료 순서로 변경했다. 결과는 기존처럼 sample_key로 저장한다.

- 캐시 위치: `cache/timelens_video_v1`. 전체 저장된 tensor 파일 상한 8GiB, 최근 사용 순서로 제거한다. 쓰기 중 임시 파일은 별도다. 여러 번 남은 영상만 캐시한다.
- 모델, 영상 경로·크기·수정 시각, 전체 영상 옵션, 함수 인자, reader, 구현 파일 해시와 torch 버전을 키로 구분한다. 프로세스 간 잠금과 원자적 파일 교체를 사용한다.
- 기존 완료 파일은 보존하고 미완료 샘플만 재개했다. 전환 시 아직 저장되지 않은 작업만 다시 준비할 수 있다.
- 각 응답에 CPU 준비 시간, 캐시 적중 여부, 입력 대기 시간, 준비 후 경과 시간, GPU 전송·설정 시간을 추가 기록한다.

## 검증

모델 2종 × 벤치마크 3종에서 각각 1개, 총 6개 입력을 캐시 없이 준비한 결과와 최초 캐시 저장, 캐시 재사용 결과를 비교했다. 모든 입력 tensor, 입력 ID 및 메타데이터가 정확히 일치했다. 생성 연산·확률 계측은 변경하지 않았으며 전체 GPU 응답 재생 검증은 하지 않았다.

| 모델 | 벤치마크 | 최초 디코딩·resize·캐시 저장(초) | 캐시 읽기(초) |
|---|---|---:|---:|
| qwen25 | charades-timelens | 3.446 | 0.134 |
| qwen25 | activitynet-timelens | 5.229 | 0.113 |
| qwen25 | qvhighlights-timelens | 3.840 | 0.144 |
| qwen3 | charades-timelens | 3.693 | 0.149 |
| qwen3 | activitynet-timelens | 5.731 | 0.145 |
| qwen3 | qvhighlights-timelens | 3.628 | 0.151 |

위 시간은 영상 처리 함수 구간이며 모델 추론을 포함하지 않는다. 최초 처리에는 저장 비용, 반복 처리에는 파일 캐시 효과가 포함된다. 전체 벤치마크의 가속 비율로 해석하지 않는다.

별도 지연 입력 테스트에서 순서 보장 해제 시 느린 선행 입력보다 준비된 후속 입력을 먼저 받았고, 샘플 누락·중복 없이 모든 tensor가 일치했다. 실제 실험 처리량·GPU 사용률·캐시 적중률은 백그라운드 모니터가 이 보고서에 추가한다.

[입력 검증](../../../artifacts/runs/timelens_top30_full/performance/validation/cache_validation.json) · [순서 검증](../../../artifacts/runs/timelens_top30_full/performance/validation/ready_order_validation.json) · [재개 기록](../../../artifacts/runs/timelens_top30_full/performance/metadata/cache_tuning.json) · [진행률](../../../artifacts/runs/timelens_top30_full/metadata/status.json)

## 적용 후 관측

전후 입력이 달라 통제된 속도 비교가 아니다. 캐시 적중률과 입력 대기를 포함한 실행 중 관측이다.

```json
{
  "gpu_average_percent": 65.11764705882354,
  "elapsed_seconds": 243.18488097190857,
  "models": {
    "qwen25": {
      "samples": 67,
      "seconds_per_response": 2.9501377705371743,
      "mean_input_wait_seconds": 1.1149998192082753,
      "cache_hits": 43,
      "cache_lookups": 65
    },
    "qwen3": {
      "samples": 64,
      "seconds_per_response": 3.354571671712966,
      "mean_input_wait_seconds": 1.2113303807046678,
      "cache_hits": 42,
      "cache_lookups": 62
    }
  }
}
```
