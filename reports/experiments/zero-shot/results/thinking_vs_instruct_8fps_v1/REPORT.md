# Qwen3-VL-8B Instruct / Thinking — 8fps pilot zero-shot

작성: 2026-10-28T18:25:51.087655+09:00. **전체 완료: 1,200/1,200개 응답, trace 1,200개.**

pilot 300개(Charades·ActivityNet·QVHighlights 각 100개)를 모델 2종과 입력 순서 2종으로 평가한다. 영상→질문(video_query), 질문→영상(query_video)을 모두 포함한다. Rollout은 제외했다.

## 조건 및 평가

| 항목 | Instruct | Thinking |
|---|---|---|
| Checkpoint | Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Thinking |
| 생성 | greedy | sampling T=1, top_p=.95, top_k=20, seed=42 |
| 최대 생성 토큰 | 512 | 16,384 |
| 영상 입력 | 8fps 요청, min_tokens=64, total_tokens=14,336 | 동일 |
| 정밀도·backend | BF16·FlashAttention2 | 동일 |

기존 2fps와 동일 cohort·프롬프트·모델별 생성 설정을 사용한다. 실제 frame sampling은 원본 fps와 최대 768프레임 제한을 따른다. 따라서 긴 영상의 실효 fps는 8보다 낮을 수 있고, 고정 visual token 예산에서 공간 해상도도 달라질 수 있다.

공통 TimeLens extract_time으로 최종 답변의 첫 구간을 파싱하고 단일 GT와 IoU를 계산했다. 파싱 실패는 0점, clipping·반올림은 적용하지 않는다. CSV에는 정수 반올림 민감도 결과도 함께 저장했다. 두 모델은 decoding 정책과 생성 예산이 달라 checkpoint 단독 효과로 해석할 수 없다.

## 성능과 동일 샘플의 2fps 비교

모든 지표는 %. 2fps 값은 해당 8fps 완료분과 정확히 같은 sample ID·입력 순서의 재채점 값이다. 미완료 행은 일부 샘플 결과로, 완성된 다른 행과 직접 비교하지 않는다. all은 query 수 가중 평균이다.


### Instruct: 2fps vs 8fps

각 수치는 **2fps → 8fps**, 단위는 %다. R@.3/.5/.7은 첫 예측 구간 IoU가 각각 .3/.5/.7 이상인 샘플 비율이다.

| 입력 순서 | 벤치마크 | N / 목표 | 상태 | R@.3 (2→8fps) | R@.5 (2→8fps) | R@.7 (2→8fps) | mIoU (2→8fps) | ΔmIoU pp |
|---|---|---:|---|---:|---:|---:|---:|---:|
| video_query | charades-timelens | 100/100 | 완료 | 63.00 → 57.00 | 48.00 → 45.00 | 31.00 → 26.00 | 46.90 → 42.81 | -4.09 |
| video_query | activitynet-timelens | 100/100 | 완료 | 55.00 → 56.00 | 42.00 → 39.00 | 27.00 → 28.00 | 40.63 → 41.42 | +0.79 |
| video_query | qvhighlights-timelens | 100/100 | 완료 | 73.00 → 76.00 | 60.00 → 66.00 | 45.00 → 48.00 | 57.13 → 61.23 | +4.10 |
| video_query | all | 300/300 | 완료 | 63.67 → 63.00 | 50.00 → 50.00 | 34.33 → 34.00 | 48.22 → 48.49 | +0.27 |
| query_video | charades-timelens | 100/100 | 완료 | 60.00 → 53.00 | 51.00 → 45.00 | 35.00 → 24.00 | 47.12 → 40.30 | -6.82 |
| query_video | activitynet-timelens | 100/100 | 완료 | 34.00 → 39.00 | 30.00 → 29.00 | 22.00 → 19.00 | 27.46 → 28.77 | +1.30 |
| query_video | qvhighlights-timelens | 100/100 | 완료 | 45.00 → 54.00 | 34.00 → 45.00 | 24.00 → 33.00 | 33.06 → 41.81 | +8.75 |
| query_video | all | 300/300 | 완료 | 46.33 → 48.67 | 38.33 → 39.67 | 27.00 → 25.33 | 35.88 → 36.96 | +1.08 |

### Thinking: 2fps vs 8fps

각 수치는 **2fps → 8fps**, 단위는 %다. R@.3/.5/.7은 첫 예측 구간 IoU가 각각 .3/.5/.7 이상인 샘플 비율이다.

| 입력 순서 | 벤치마크 | N / 목표 | 상태 | R@.3 (2→8fps) | R@.5 (2→8fps) | R@.7 (2→8fps) | mIoU (2→8fps) | ΔmIoU pp |
|---|---|---:|---|---:|---:|---:|---:|---:|
| video_query | charades-timelens | 100/100 | 완료 | 64.00 → 64.00 | 47.00 → 48.00 | 24.00 → 25.00 | 42.36 → 43.58 | +1.22 |
| video_query | activitynet-timelens | 100/100 | 완료 | 57.00 → 54.00 | 44.00 → 41.00 | 29.00 → 30.00 | 42.15 → 41.25 | -0.91 |
| video_query | qvhighlights-timelens | 100/100 | 완료 | 75.00 → 82.00 | 63.00 → 70.00 | 55.00 → 54.00 | 61.35 → 62.91 | +1.56 |
| video_query | all | 300/300 | 완료 | 65.33 → 66.67 | 51.33 → 53.00 | 36.00 → 36.33 | 48.62 → 49.24 | +0.62 |
| query_video | charades-timelens | 100/100 | 완료 | 66.00 → 58.00 | 50.00 → 45.00 | 23.00 → 23.00 | 44.04 → 40.09 | -3.95 |
| query_video | activitynet-timelens | 100/100 | 완료 | 52.00 → 52.00 | 42.00 → 37.00 | 25.00 → 29.00 | 38.98 → 38.94 | -0.04 |
| query_video | qvhighlights-timelens | 100/100 | 완료 | 79.00 → 81.00 | 65.00 → 64.00 | 50.00 → 49.00 | 60.72 → 60.27 | -0.45 |
| query_video | all | 300/300 | 완료 | 65.67 → 63.67 | 52.33 → 48.67 | 32.67 → 33.67 | 47.91 → 46.43 | -1.48 |

## 생성량과 실패

| 모델 | 순서 | N | 평균 생성 토큰 | 파싱 실패 | EOS 미도달 |
|---|---|---:|---:|---:|---:|
| instruct | query_video | 300 | 18.1 | 6 | 0 |
| instruct | video_query | 300 | 18.0 | 0 | 0 |
| thinking | query_video | 300 | 1416.7 | 2 | 1 |
| thinking | video_query | 300 | 317.1 | 0 | 0 |

## 결과 해석

- instruct / query_video: 완료된 전체 300개에서 8fps의 mIoU는 36.96%이며, 동일 샘플 2fps 대비 +1.08pp다.
- instruct / video_query: 완료된 전체 300개에서 8fps의 mIoU는 48.49%이며, 동일 샘플 2fps 대비 +0.27pp다.
- thinking / query_video: 완료된 전체 300개에서 8fps의 mIoU는 46.43%이며, 동일 샘플 2fps 대비 -1.48pp다.
- thinking / video_query: 완료된 전체 300개에서 8fps의 mIoU는 49.24%이며, 동일 샘플 2fps 대비 +0.62pp다.

단일 seed 결과이며 seed 변동성이나 신뢰구간은 평가하지 않았다. FPS 증가의 효과는 시간·공간 sampling 변화와 함께 해석해야 한다.

## Token trace와 불확실성 분석

raw/processed top-30, 전체 어휘 FP32 entropy(nats), 선택 토큰 확률·순위, span alphabet/EOS mass를 저장한다. top-30은 재정규화하지 않는다. processed는 sampling에서 temperature·top-k·top-p까지 적용한 선택 직전 분포이며 raw와 구분한다. Thinking의 reasoning·answer·경계 토큰도 분리한다.

새 응답은 생성과 동시에 수집한다. trace 수집 전에 생성된 응답만 저장된 생성 토큰과 입력 hash를 유지한 fixed-prefix replay로 보완했다. 각 trace에 측정 방법을 명시했으며 attention 행렬 추출을 위한 추가 replay는 없다.

Trace 분석: 1,200개 응답 / 530,958개 생성 단계. 지원 문법 등으로 인한 분석 제외 기록 41건은 ../../artifacts/runs/thinking_vs_instruct_8fps_v1/metadata/trace_excluded.json에 명시했다. 제외는 전체 성능 채점과 원본 trace 보존에 영향을 주지 않는다.

| 모델 | 순서 | N | Answer raw H | Reasoning raw H | Answer top30 mass |
|---|---|---:|---:|---:|---:|
| instruct | query_video | 300 | 0.2829 | NA | 1.0000 |
| instruct | video_query | 300 | 0.4038 | NA | 1.0000 |
| thinking | query_video | 300 | 0.0479 | 0.4699 | 1.0000 |
| thinking | video_query | 300 | 0.0383 | 0.3404 | 1.0000 |

[토큰·불확실성 상세 보고서](TOKEN_TRACE_REPORT.md)에서 기존과 같은 rounded-prefix 자릿수별 최초 오류, GT로 반대 endpoint를 고정한 정답 가능성, top30 정답 가능/불가능/미확인 후보 확률 질량을 확인할 수 있다.

## 속도 개선과 검증

GPU 통계를 응답 단위로 CPU에 전송하고, processor가 없는 greedy의 동일 raw/processed 분포를 재사용했다. 같은 샘플의 두 입력 순서에서는 비텍스트 입력 동일성 검사 후 영상 특징을 재사용한다. 샘플·모델이 바뀌면 cache를 초기화한다. 단일 8fps 입력에서 기존 수집기 대비 약 3.6%, vision cache로 추가 약 9.3% 가속을 측정했다. 생성 ID와 모든 trace 값의 일치를 검증했으며, 이는 전체 cohort 가속률을 의미하지 않는다. [측정 원자료·범위](performance/REPORT.md).

완료분 응답의 sample·GT·저장 IoU를 재검증하고 2fps 출처 hash를 기록했다. trace 검증 범위와 입력 hash·생성 ID·확률 합·top30 정렬·entropy 검사는 [../../artifacts/runs/thinking_vs_instruct_8fps_v1/validation/trace_validation.json](../../artifacts/runs/thinking_vs_instruct_8fps_v1/validation/trace_validation.json)에 기록했다.

## 저장 자료

- [성능 지표 CSV](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/performance_metrics.csv) · [동일 샘플 2fps/8fps CSV](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/paired_2fps_8fps.csv) · [성능 JSON·미완료 목록](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/performance_summary.json)
- [Token trace 집계](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/token_trace_metrics.csv) · [응답별 지표](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/token_trace_per_response.csv) · [digit 조건 집계](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/digit_conditions.csv) · [endpoint 지표](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/per_endpoint.csv) · [후보 확률 질량](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/blue_candidate_mass.csv)
- [입력·생성 설정](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metadata/config.json) · [응답 출처 hash](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metadata/report_sources.json) · [trace 출처 hash](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metadata/trace_index.json) · [진행 상태](../../artifacts/runs/thinking_vs_instruct_8fps_v1/metadata/status.json)
