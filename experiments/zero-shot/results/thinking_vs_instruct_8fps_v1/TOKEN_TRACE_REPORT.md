# 8fps token trace·불확실성·endpoint 집계

pilot 300개 × Instruct/Thinking × 두 입력 순서의 목표 1,200개 중 1,200개 응답을 검증했다. Rollout은 수행하지 않았다. 기존 zero-shot과 같은 raw/processed top30·전체 어휘 엔트로피·span alphabet mass·EOS mass·선택 토큰 확률/순위를 저장했다. Thinking은 reasoning과 최종 answer를 분리 집계했다.

processed는 temperature/top-k/top-p를 포함한 실제 sampling 직전 분포다. greedy에는 원래 generation processor만 적용한다. 이미 완료된 응답은 동일 prefix를 강제로 재생해 raw/processed 분포를 측정했고, 강제 토큰 선택은 분포를 기록한 뒤 적용한다. replay와 live는 trace metadata로 구분한다.

| Model | Order | Dataset | N | mIoU % | Answer raw H | Reasoning raw H | Answer top30 mass |
|---|---|---|---:|---:|---:|---:|---:|
| instruct | query_video | activitynet-timelens | 100 | 28.77 | 0.3105 | NA | 1.0000 |
| instruct | query_video | all | 300 | 36.96 | 0.2829 | NA | 1.0000 |
| instruct | query_video | charades-timelens | 100 | 40.30 | 0.2706 | NA | 1.0000 |
| instruct | query_video | qvhighlights-timelens | 100 | 41.81 | 0.2677 | NA | 1.0000 |
| instruct | video_query | activitynet-timelens | 100 | 41.42 | 0.4437 | NA | 1.0000 |
| instruct | video_query | all | 300 | 48.49 | 0.4038 | NA | 1.0000 |
| instruct | video_query | charades-timelens | 100 | 42.81 | 0.3837 | NA | 1.0000 |
| instruct | video_query | qvhighlights-timelens | 100 | 61.23 | 0.3840 | NA | 1.0000 |
| thinking | query_video | activitynet-timelens | 100 | 38.94 | 0.0498 | 0.4961 | 1.0000 |
| thinking | query_video | all | 300 | 46.43 | 0.0479 | 0.4699 | 1.0000 |
| thinking | query_video | charades-timelens | 100 | 40.09 | 0.0551 | 0.4432 | 1.0000 |
| thinking | query_video | qvhighlights-timelens | 100 | 60.27 | 0.0389 | 0.4705 | 1.0000 |
| thinking | video_query | activitynet-timelens | 100 | 41.25 | 0.0341 | 0.3720 | 1.0000 |
| thinking | video_query | all | 300 | 49.24 | 0.0383 | 0.3404 | 1.0000 |
| thinking | video_query | charades-timelens | 100 | 43.58 | 0.0491 | 0.3154 | 1.0000 |
| thinking | video_query | qvhighlights-timelens | 100 | 62.91 | 0.0318 | 0.3339 | 1.0000 |

blue_candidate_mass.csv는 기존과 동일한 top30 정답 가능/불가능/미확인 확률 질량과 하한·상한을 저장한다. top30을 재정규화하지 않는다. digit_per_step.csv와 digit_conditions.csv는 기존 rounded-prefix 첫 정답 가능성 상실 판정을 그대로 재사용했다. per_endpoint.csv는 반대쪽 endpoint를 GT로 고정한 IoU와 최초 IoU .5 가능성 상실 entropy를 제공한다. 지원 문법 밖의 응답은 trace_excluded.json에 명시했으며 원본 응답이나 성능 집계에서는 제외하지 않았다.

[응답별 trace 지표](token_trace_per_response.csv) · [집계 지표](token_trace_metrics.csv) · [token trace summary](token_trace_summary.json) · [검증](trace_validation.json) · [성능 비교](REPORT.md)
