# 본 실험의 live 추출 속도 측정

측정 시각: 2026-10-25T19:38:05+0900, 완료 snapshot 571/600.

Charades의 동일 입력 순서끼리 비교했다. 토큰당 시간 = 합산 worker 소요 시간 / 합산 생성 토큰 수. 동일 샘플을 재실행한 paired 비교는 아니므로 측정된 개선폭을 알고리즘만의 인과 효과로 단정하지 않는다.

| 입력 순서 | replay n | live n | replay ms/token | live ms/token | 속도비 | 시간 감소 |
|---|---:|---:|---:|---:|---:|---:|
| video_query | 92 | 7 | 147.92 | 124.66 | 1.187× | 15.7% |
| query_video | 90 | 9 | 144.60 | 117.00 | 1.236× | 19.1% |

## 해석

본 실험의 토큰당 작업 시간은 약 16–19% 줄었다. 고정 입력의 짧은 벤치마크에서 얻은 1.55–1.71배보다 개선폭이 작다. GPU 동시 작업 수, 입력·출력 길이, 데이터셋 및 저장 비용 차이가 영향을 줄 수 있다. 측정은 그 원인별 기여를 분리하지 않는다.

단위 시간당 완료 샘플 수는 전환 전후 데이터셋/trace 길이가 달라 직접적인 가속 지표로 쓰지 않는다.
- supervisor.log: 109.8회/시간 (182회, 1.66시간 구간)
- supervisor_live.log: 109.1회/시간 (387회, 3.55시간 구간)

## 제한

- Different samples before/after; not a paired causal estimate.
- Same-dataset/order comparison restricted to Charades, only 7 and 9 live records.
- Mean input/output lengths differ; tokens/sec does not remove all sequence-length effects.
- Excludes saved-generation recovery and first reused-generation/probe timing.
- Worker time includes generation/capture, input transfer, text serialization and array validation; excludes preloading, final checksum and model loading.
- Overall wall throughput mixes datasets and trace lengths; 4 workers may become fewer near completion.
