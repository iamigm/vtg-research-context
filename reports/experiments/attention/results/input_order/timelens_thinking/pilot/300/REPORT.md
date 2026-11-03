# Qwen3-VL-8B-Thinking pilot input-order attention · pilot300 (300질문)

검증 완료: 600/600 실행. 미완료 결과를 전체 결과로 해석하지 않는다.

300개 고정 pilot × video_query/query_video. seed=42, temperature=1, top_p=.95, top_k=20, 최대 16,384 생성 토큰. FPS=2, min_tokens=64, total_tokens=14336.

상세: L0/7/14/21/28/35, 모든 key, FP32. Head 압축 방식은 ../../../../../artifacts/runs/input_order/timelens_thinking/pilot/300/metadata/config.json 참조. 전체 36 layers × 32 heads에는 7개 범주의 attention mass를 저장한다. 각 생성 토큰을 예측하는 직전 위치의 attention이며 종료 태그와 EOS 예측 행도 포함한다.

전환 전 기록은 replay, 전환 후 신규 생성은 live 추출이다. 저장된 generation만 있는 미완료 항목은 replay로 복구한다. 각 신규 결과의 measurement_method와 migration_to_live 목록으로 구분한다. Attention은 post-RoPE Q/K의 FP32 softmax다. 실제 생성 실행의 attention과 비트 단위 동일함을 주장하지 않는다. 미래 key는 0이며 key_lengths로 유효 구간을 구분한다. 구간/head 평균은 key 축을 재정규화하지 않는다. 입력 순서에 따라 reasoning과 답변도 달라질 수 있어 attention 변화의 인과를 단정하지 않는다.

| Dataset/order | n | reasoning mean | median | p95 | max | mIoU | truncated | GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| activitynet-timelens/video_query | 100 | 394.8 | 170.5 | 1544.3 | 2496 | 0.4215 | 0 | 14.28 |
| charades-timelens/video_query | 100 | 379.9 | 133.5 | 1625.8 | 4629 | 0.4236 | 0 | 10.54 |
| qvhighlights-timelens/video_query | 100 | 400.9 | 114.5 | 709.1 | 16384 | 0.6135 | 1 | 18.59 |
| activitynet-timelens/query_video | 100 | 1910.7 | 1440.0 | 5109.4 | 15787 | 0.3898 | 0 | 74.55 |
| charades-timelens/query_video | 100 | 1374.3 | 952.0 | 3718.7 | 6171 | 0.4404 | 0 | 41.08 |
| qvhighlights-timelens/query_video | 100 | 1010.6 | 646.0 | 3201.0 | 6558 | 0.6072 | 0 | 40.02 |

category_mass: [layer, generated token, head, category]. layer_XX: [head channel, generated token, key token]. tokens.json/input.json에 위치·토큰·범주 매핑을 저장한다. mean이면 유일한 채널은 32개 head의 산술평균이다. 전체 레이어 요약은 head별로 보존한다.

표본 1개가 전체 길이 분포를 대표하지 않으므로 benchmark/order별 p95와 max, 종료 여부를 함께 확인한다.
