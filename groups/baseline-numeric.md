# Zero-shot 기준선과 숫자 불확실성

전체9,404 query/model과 pilot300, 제한 head·full-head top30·자릿수·endpoint·rollout의 계보. Full 18,808응답과 rollout6,000응답은 별도 실행이며 부분집합을 합산하지 않는다.

## 실행별 보고서·관련 코드

### zero-shot/all_zeroshot_v1

[REPORT.md](../reports/experiments/zero-shot/results/all_zeroshot_v1/REPORT.md)

현재 관련 코드 76개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/conditional_rollout_budget_pilot30_v1

[SPEC.md](../reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/SPEC.md)

현재 관련 코드 76개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/legacy_quantitative

[README.md](../reports/experiments/zero-shot/results/legacy_quantitative/README.md)

현재 관련 코드 76개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/timelens_rollout

[REPORT.md](../reports/experiments/zero-shot/results/timelens_rollout/REPORT.md)

현재 관련 코드 76개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/timelens_span_head

[REPORT.md](../reports/experiments/zero-shot/results/timelens_span_head/REPORT.md)

현재 관련 코드 79개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/timelens_top30

[REPORT.md](../reports/experiments/zero-shot/results/timelens_top30/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30/digit_correctness/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30/digit_correctness/blue_analysis/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30/entropy_iou/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30/integrated/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30/mmco_diagnosis/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30/numeric_uncertainty/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30/prefix_entropy/REPORT.md)

현재 관련 코드 76개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/timelens_top30_full

[REPORT.md](../reports/experiments/zero-shot/results/timelens_top30_full/digit_correctness/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30_full/digit_correctness/blue_analysis/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30_full/digit_correctness/first_error_distance/REPORT.md) · [CACHE_REPORT.md](../reports/experiments/zero-shot/results/timelens_top30_full/performance/CACHE_REPORT.md) · [PIPELINE_REVIEW.md](../reports/experiments/zero-shot/results/timelens_top30_full/performance/PIPELINE_REVIEW.md) · [REPORT.md](../reports/experiments/zero-shot/results/timelens_top30_full/performance/REPORT.md)

현재 관련 코드 76개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

## 관련 실험 대화

- [같은 pilot 100개 샘플에 대해 lm head에서 숫자와 온점, 반점 등 span prediction에 필요한 것만 분류하게 해봐. GPU 최대로 써서 빠르게 결과 뽑아](../conversations/01a153a1-eddc-75b1-a442-3b3ed0ebf708.md) — active
- [Save benchmark result PNGs](../conversations/01a1715e-ffa1-7fe1-9ef4-2f1dc8e65b58.md) — deleted
## 코드 백업

아래는 현재 구현의 연결이다. 역사적 실행 revision의 증명은 아니다. 공용 dependency는 [shared](shared.md)를 참조한다.

- [zero-shot/run.py](../code/experiments/zero-shot/run.py)
- [zero-shot/_lib/aggregate_all_zeroshot.py](../code/experiments/zero-shot/_lib/aggregate_all_zeroshot.py)
- [zero-shot/_lib/artifact_store.py](../code/experiments/zero-shot/_lib/artifact_store.py)
- [zero-shot/_lib/cleanup_duplicate_artifacts.py](../code/experiments/zero-shot/_lib/cleanup_duplicate_artifacts.py)
- [zero-shot/_lib/conditional_rollout_budget.py](../code/experiments/zero-shot/_lib/conditional_rollout_budget.py)
- [zero-shot/_lib/temporal_pilot300_run.py](../code/experiments/zero-shot/_lib/temporal_pilot300_run.py)
- [zero-shot/_lib/timelens_blue_analysis.py](../code/experiments/zero-shot/_lib/timelens_blue_analysis.py)
- [zero-shot/_lib/timelens_blue_analysis_finish.py](../code/experiments/zero-shot/_lib/timelens_blue_analysis_finish.py)
- [zero-shot/_lib/timelens_digit_correctness.py](../code/experiments/zero-shot/_lib/timelens_digit_correctness.py)
- [zero-shot/_lib/timelens_digit_correctness_browser.py](../code/experiments/zero-shot/_lib/timelens_digit_correctness_browser.py)
- [zero-shot/_lib/timelens_digit_correctness_finish.py](../code/experiments/zero-shot/_lib/timelens_digit_correctness_finish.py)
- [zero-shot/_lib/timelens_digit_correctness_replot.py](../code/experiments/zero-shot/_lib/timelens_digit_correctness_replot.py)
- [zero-shot/_lib/timelens_digit_correctness_serve.py](../code/experiments/zero-shot/_lib/timelens_digit_correctness_serve.py)
- [zero-shot/_lib/timelens_digit_role_entropy.py](../code/experiments/zero-shot/_lib/timelens_digit_role_entropy.py)
- [zero-shot/_lib/timelens_endpoint_gt_entropy.py](../code/experiments/zero-shot/_lib/timelens_endpoint_gt_entropy.py)
- [zero-shot/_lib/timelens_entropy_integrated.py](../code/experiments/zero-shot/_lib/timelens_entropy_integrated.py)
- [zero-shot/_lib/timelens_entropy_integrated_browser.py](../code/experiments/zero-shot/_lib/timelens_entropy_integrated_browser.py)
- [zero-shot/_lib/timelens_entropy_integrated_serve.py](../code/experiments/zero-shot/_lib/timelens_entropy_integrated_serve.py)
- [zero-shot/_lib/timelens_entropy_iou.py](../code/experiments/zero-shot/_lib/timelens_entropy_iou.py)
- [zero-shot/_lib/timelens_first_error_distance.py](../code/experiments/zero-shot/_lib/timelens_first_error_distance.py)
- [zero-shot/_lib/timelens_full_blue_impl.py](../code/experiments/zero-shot/_lib/timelens_full_blue_impl.py)
- [zero-shot/_lib/timelens_full_digit_browser.py](../code/experiments/zero-shot/_lib/timelens_full_digit_browser.py)
- [zero-shot/_lib/timelens_full_digit_finish.py](../code/experiments/zero-shot/_lib/timelens_full_digit_finish.py)
- [zero-shot/_lib/timelens_full_digit_impl.py](../code/experiments/zero-shot/_lib/timelens_full_digit_impl.py)
- [zero-shot/_lib/timelens_full_digit_media.py](../code/experiments/zero-shot/_lib/timelens_full_digit_media.py)
- [zero-shot/_lib/timelens_full_digit_prepare.py](../code/experiments/zero-shot/_lib/timelens_full_digit_prepare.py)
- [zero-shot/_lib/timelens_full_digit_serve.py](../code/experiments/zero-shot/_lib/timelens_full_digit_serve.py)
- [zero-shot/_lib/timelens_mmco_diagnose.py](../code/experiments/zero-shot/_lib/timelens_mmco_diagnose.py)
- [zero-shot/_lib/timelens_numeric_uncertainty.py](../code/experiments/zero-shot/_lib/timelens_numeric_uncertainty.py)
- [zero-shot/_lib/timelens_prefix_entropy.py](../code/experiments/zero-shot/_lib/timelens_prefix_entropy.py)
- [zero-shot/_lib/timelens_profile_preparation.py](../code/experiments/zero-shot/_lib/timelens_profile_preparation.py)
- [zero-shot/_lib/timelens_rollout.py](../code/experiments/zero-shot/_lib/timelens_rollout.py)
- [zero-shot/_lib/timelens_rollout_browser.py](../code/experiments/zero-shot/_lib/timelens_rollout_browser.py)
- [zero-shot/_lib/timelens_rollout_manage.py](../code/experiments/zero-shot/_lib/timelens_rollout_manage.py)
- [zero-shot/_lib/timelens_rollout_report.py](../code/experiments/zero-shot/_lib/timelens_rollout_report.py)
- [zero-shot/_lib/timelens_rollout_serve.py](../code/experiments/zero-shot/_lib/timelens_rollout_serve.py)
- [zero-shot/_lib/timelens_span_head.py](../code/experiments/zero-shot/_lib/timelens_span_head.py)
- [zero-shot/_lib/timelens_span_markdown.py](../code/experiments/zero-shot/_lib/timelens_span_markdown.py)
- [zero-shot/_lib/timelens_span_monitor.py](../code/experiments/zero-shot/_lib/timelens_span_monitor.py)
- [zero-shot/_lib/timelens_span_report.py](../code/experiments/zero-shot/_lib/timelens_span_report.py)
- [zero-shot/_lib/timelens_top30.py](../code/experiments/zero-shot/_lib/timelens_top30.py)
- [zero-shot/_lib/timelens_top30_browser.py](../code/experiments/zero-shot/_lib/timelens_top30_browser.py)
- [zero-shot/_lib/timelens_top30_finish.py](../code/experiments/zero-shot/_lib/timelens_top30_finish.py)
- [zero-shot/_lib/timelens_top30_full_worker.py](../code/experiments/zero-shot/_lib/timelens_top30_full_worker.py)
- [zero-shot/_lib/timelens_top30_media.py](../code/experiments/zero-shot/_lib/timelens_top30_media.py)
- [zero-shot/_lib/timelens_top30_progress.py](../code/experiments/zero-shot/_lib/timelens_top30_progress.py)
- [zero-shot/_lib/timelens_top30_report.py](../code/experiments/zero-shot/_lib/timelens_top30_report.py)
- [zero-shot/_lib/timelens_top30_serve.py](../code/experiments/zero-shot/_lib/timelens_top30_serve.py)
- [zero-shot/_lib/timelens_validate_ready_order.py](../code/experiments/zero-shot/_lib/timelens_validate_ready_order.py)
- [zero-shot/_lib/timelens_validate_video_cache.py](../code/experiments/zero-shot/_lib/timelens_validate_video_cache.py)
- [zero-shot/_lib/timelens_verify_cached_pipeline.py](../code/experiments/zero-shot/_lib/timelens_verify_cached_pipeline.py)
- [zero-shot/_lib/timelens_verify_throughput.py](../code/experiments/zero-shot/_lib/timelens_verify_throughput.py)
- [zero-shot/_lib/timelens_video_cache.py](../code/experiments/zero-shot/_lib/timelens_video_cache.py)
- [zero-shot/_lib/digit_correctness_viewer/style.css](../code/experiments/zero-shot/_lib/digit_correctness_viewer/style.css)
- [zero-shot/_lib/digit_correctness_viewer/viewer.js](../code/experiments/zero-shot/_lib/digit_correctness_viewer/viewer.js)
- [zero-shot/_lib/entropy_viewer/style.css](../code/experiments/zero-shot/_lib/entropy_viewer/style.css)
- [zero-shot/_lib/entropy_viewer/viewer.js](../code/experiments/zero-shot/_lib/entropy_viewer/viewer.js)
- [zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/code/conditional_rollout_budget.py](../code/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/code/conditional_rollout_budget.py)
- [zero-shot/artifacts/runs/timelens_rollout/web/manifest.js](../code/experiments/zero-shot/artifacts/runs/timelens_rollout/web/manifest.js)
- [zero-shot/artifacts/runs/timelens_rollout/web/viewer.js](../code/experiments/zero-shot/artifacts/runs/timelens_rollout/web/viewer.js)
- [zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/monitor.py](../code/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/monitor.py)
- [zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/report.py](../code/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/report.py)
- [zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/run.py](../code/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/run.py)
- [zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/manifest.js](../code/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/manifest.js)
- [zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/style.css](../code/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/style.css)
- [zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/viewer.js](../code/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/viewer.js)
- [zero-shot/artifacts/runs/timelens_top30/integrated/web/manifest.js](../code/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/web/manifest.js)
- [zero-shot/artifacts/runs/timelens_top30/integrated/web/style.css](../code/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/web/style.css)
- [zero-shot/artifacts/runs/timelens_top30/integrated/web/viewer.js](../code/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/web/viewer.js)
- [zero-shot/artifacts/runs/timelens_top30/web/manifest.js](../code/experiments/zero-shot/artifacts/runs/timelens_top30/web/manifest.js)
- [zero-shot/artifacts/runs/timelens_top30/web/viewer.js](../code/experiments/zero-shot/artifacts/runs/timelens_top30/web/viewer.js)
- [zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/web/style.css](../code/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/web/style.css)
- [zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/web/viewer.js](../code/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/web/viewer.js)
- [zero-shot/artifacts/runs/timelens_top30_full/performance/code/observe.py](../code/experiments/zero-shot/artifacts/runs/timelens_top30_full/performance/code/observe.py)
- [zero-shot/artifacts/runs/timelens_top30_full/performance/code/observe_after.py](../code/experiments/zero-shot/artifacts/runs/timelens_top30_full/performance/code/observe_after.py)
- [zero-shot/artifacts/runs/timelens_top30_full/performance/code/worker_before_cache.py](../code/experiments/zero-shot/artifacts/runs/timelens_top30_full/performance/code/worker_before_cache.py)
