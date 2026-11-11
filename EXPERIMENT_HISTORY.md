# VTG 실험 변경 이력

## 2026-11-02T19:12:52.797388+09:00

원본 파일 직접 관찰: 22개 실행, 335개 코드·의존성 파일. 이전 commit: a0824288d3fab917db04d016eeb32d32e170f3c1.

추가 503 / 변경 0 / 현재 원본에서 사라짐 84. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

초기 실험 중심 전환. 관련 실험의 계보·차이·수치는 EXPERIMENTS.md와 groups/에 정리했다. 과거 채팅은 현재 트리에서 제외하고 이전 Git commit에 보존했다.

## 2026-11-02T19:27:19.660097+09:00

원본 파일 직접 관찰: 23개 실행, 341개 코드·의존성 파일. 이전 commit: 54fe905696f584f59e74a60e9ea50efb094f3e12.

추가 108 / 변경 4 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/fps_spatial_control.py
- code/experiments/zero-shot/_lib/fps_spatial_report.py
- code/experiments/zero-shot/_lib/fps_spatial_smoke.py
- code/experiments/zero-shot/_lib/fps_spatial_supervisor.py
- evidence/experiments/zero-shot/artifacts/common/cohorts/nested_pilot_v1/pilot_300/samples.json
- evidence/experiments/zero-shot/artifacts/common/cohorts/nested_pilot_v1/pilot_600/samples.json
- evidence/experiments/zero-shot/artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json
- evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metrics/metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metrics/rollout_metrics.csv
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_report.py
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/metrics/largest_gt_degradation.csv
- evidence/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/metrics/largest_gt_improvement.csv
- evidence/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/metrics/largest_normalized_shift.csv
- evidence/experiments/zero-shot/artifacts/runs/input_order/metrics/benchmark_metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/input_order/metrics/invalid_prediction_refs.csv
- evidence/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/metrics/compute_cost.csv
- evidence/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/metrics/metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/metrics/spatial_inputs.csv
- evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/digit_conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/paired_2fps_8fps.csv
- evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/per_endpoint.csv
- evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/performance_metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/token_trace_metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/token_trace_per_response.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/metrics/paired.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/candidate_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/candidate_future.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/future_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/length_control.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/per_blue_step.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/position_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/three_axis_cross.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/digit_stratified.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/per_endpoint.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/position_conditional_accuracy.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/role_conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/entropy_iou/metrics/per_query.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/prefix_entropy/summary.json
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/numeric_uncertainty/metrics/category_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/numeric_uncertainty/metrics/digit_stratified_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/numeric_uncertainty/metrics/endpoint_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/prefix_entropy/metrics/excluded.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30/prefix_entropy/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/candidate_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/candidate_future.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/future_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/length_control.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/position_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/three_axis_cross.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metrics/group_statistics.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metrics/role_statistics.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/digit_stratified.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/position_conditional_accuracy.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/role_conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/summary.json
- reports/experiments/zero-shot/results/fps_spatial_control_pilot300_v1/SPATIAL_AUDIT.md
- reports/experiments/zero-shot/results/fps_spatial_control_pilot300_v1/SPEC.md
- evidence/experiments/attention/artifacts/runs/dynamics/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/generation/distribution_analysis/metrics/sample_metrics.csv
- evidence/experiments/attention/artifacts/runs/generation/distribution_analysis/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/generation/metrics/sample_metrics.csv
- evidence/experiments/attention/artifacts/runs/hidden/metrics/entropy_matched_metrics.csv
- evidence/experiments/attention/artifacts/runs/hidden/metrics/query_token_metrics.csv
- evidence/experiments/attention/artifacts/runs/hidden/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/hidden/metrics/video_token_metrics.csv
- evidence/experiments/attention/artifacts/runs/overview/metrics/sample_metrics.csv
- evidence/experiments/attention/artifacts/runs/query/metrics/sample_metrics.csv
- evidence/experiments/attention/artifacts/runs/spatial/metrics/counterfactual_metrics.csv
- evidence/experiments/attention/artifacts/runs/spatial/metrics/spatial_metrics.csv
- evidence/experiments/attention/artifacts/runs/timelens_order/metrics/dataset_metrics.csv
- evidence/experiments/attention/artifacts/runs/timelens_order/metrics/paired_metrics.csv
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/baseline_paired_cases.csv
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/case_metrics.csv
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/setting_case_metrics.csv
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/stratified_analysis.csv
- evidence/experiments/similarity/artifacts/runs/pilot_300_16fps_cost_v1/measurements/per_sample.csv
- conversations/01a0cdf4-f9ee-7d22-a760-a253660fb045.md
- conversations/01a0e7a6-4380-7cf2-ab28-04fd79163847.md
- conversations/01a0eb98-2716-7e21-8366-1f182062f967.md
- conversations/01a0ebf8-47fb-7fd0-b80a-36c612c5f275.md
- conversations/01a0eca8-0b26-78f2-bbae-59a2a32c457e.md
- conversations/01a0f09d-8388-75d0-b859-814e032a9039.md
- conversations/01a0f0a4-087b-7912-88d8-2b2d7053a830.md
- conversations/01a11511-6b4d-7120-abec-81d084deaba4.md
- conversations/01a11543-617e-7c51-89d2-4878a1b84d65.md
- conversations/01a11a8b-1fbb-7991-94dc-3a3ba2db00a4.md
- conversations/01a11b61-a962-76e0-9337-5892b74d66e1.md
- conversations/01a13f33-09f0-7042-985f-7d112c123f6a.md
- conversations/01a13f3f-a47e-7c80-9c2e-0be40df386d7.md
- conversations/01a153a1-eddc-75b1-a442-3b3ed0ebf708.md
- conversations/01a1689a-0c59-7920-8e01-475d6b29c0c5.md
- conversations/01a1715b-f50f-7c52-83e4-6d235e825199.md
- conversations/01a1715e-ffa1-7fe1-9ef4-2f1dc8e65b58.md
- conversations/01a1806c-2311-77a0-888a-04546017331f.md
- conversations/01a186f0-efb7-7550-b57f-7340962f10e3.md
- conversations/01a18c45-0978-7563-9999-067f29e4426e.md
- conversations/01a1902b-c9f9-7243-aa08-3f6204b2e32e.md
- conversations/01a192da-87cd-7582-8f38-d022b7d79a99.md
- conversations/01a19350-4a1d-7572-b26c-ae796896677a.md
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md
- code/experiments/zero-shot/_lib/aggregate_all_zeroshot.py
- code/experiments/zero-shot/_lib/spatial_sweep_supervisor.py
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json
- reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/REPORT.md

## 2026-11-02T19:38:41.162177+09:00

원본 파일 직접 관찰: 22개 실행, 336개 코드·의존성 파일. 이전 commit: 75d856ce627c183ef8c8c85932eef249a5d0d3b5.

추가 3 / 변경 12 / 현재 원본에서 사라짐 19. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_supervisor.py
- context/specs/01a18c45-0978-7563-9999-067f29e4426e/Pasted text.txt
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/aggregate_all_zeroshot.py
- code/experiments/zero-shot/_lib/fps_spatial_control.py
- code/experiments/zero-shot/_lib/fps_spatial_supervisor.py
- evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metrics/metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_control.py
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json
- reports/experiments/zero-shot/results/all_zeroshot_v1/REPORT.md
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md
- 원본 부재: code/experiments/zero-shot/_lib/pilot_8fps.py
- 원본 부재: code/experiments/zero-shot/_lib/pilot_8fps_traces.py
- 원본 부재: code/experiments/zero-shot/_lib/report_pilot_8fps.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/code/response_only_runner.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/digit_conditions.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/paired_2fps_8fps.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/per_endpoint.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/performance_metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/token_trace_metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/metrics/token_trace_per_response.csv
- 원본 부재: code/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/before_pilot_8fps.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/before_pilot_8fps_traces.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/benchmark.py
- 원본 부재: reports/experiments/zero-shot/results/thinking_vs_instruct_8fps_v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/thinking_vs_instruct_8fps_v1/TOKEN_TRACE_REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/thinking_vs_instruct_8fps_v1/performance/REPORT.md

## 2026-11-02T19:40:10.390500+09:00

원본 파일 직접 관찰: 22개 실행, 336개 코드·의존성 파일. 이전 commit: 6d58ec1949cbc4451ce471295965528c0ef2c59b.

추가 0 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json

## 2026-11-02T20:39:42.001749+09:00

원본 파일 직접 관찰: 22개 실행, 337개 코드·의존성 파일. 이전 commit: 1e109494917692926a4ce28d52305c07d7f9eb9a.

추가 1 / 변경 7 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/spatial_sweep_report_s42.py
- code/experiments/zero-shot/_lib/spatial_sweep_parallel_supervisor.py
- code/experiments/zero-shot/_lib/spatial_sweep_runtime.py
- code/experiments/zero-shot/_lib/spatial_sweep_supervisor.py
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json
- reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/REPORT.md
- conversations/01a18c45-0978-7563-9999-067f29e4426e.md

## 2026-11-02T21:00:00.002254+09:00

원본 파일 직접 관찰: 22개 실행, 337개 코드·의존성 파일. 이전 commit: 616256d96894c1421c6688c8de10afb36d0ff401.

추가 0 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json

## 2026-11-03T00:00:00.002294+09:00

원본 파일 직접 관찰: 22개 실행, 337개 코드·의존성 파일. 이전 commit: 433781a989cb9ce235687f78763f89e05117dbb1.

추가 0 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json

## 2026-11-03T00:33:13.358641+09:00

원본 파일 직접 관찰: 22개 실행, 337개 코드·의존성 파일. 이전 commit: 5d6a6964f1828f68fced28ed446fb8d37f9fca5b.

추가 0 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json
- conversations/01a18c45-0978-7563-9999-067f29e4426e.md

## 2026-11-03T09:00:00.001468+09:00

원본 파일 직접 관찰: 22개 실행, 337개 코드·의존성 파일. 이전 commit: 190adc6190f94d59c1d9e1570ac2882962d36c97.

추가 6 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metrics/axes.csv
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metrics/metrics.csv
- reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/A_high.md
- reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/A_low.md
- reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/B_high.md
- reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/B_low.md
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json
- reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/REPORT.md

## 2026-11-03T15:34:46.211247+09:00

원본 파일 직접 관찰: 22개 실행, 349개 코드·의존성 파일. 이전 commit: b3fa11451c879463b0a1ca4b5b98cd7301f5c78f.

추가 13 / 변경 10 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/initial/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/initial/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/initial/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/initial/fps_spatial_supervisor.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_supervisor.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_supervisor.py
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/recovery_20261103/status.json
- code/experiments/zero-shot/_lib/fps_spatial_control.py
- code/experiments/zero-shot/_lib/fps_spatial_report.py
- code/experiments/zero-shot/_lib/fps_spatial_supervisor.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_supervisor.py
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- reports/experiments/zero-shot/results/fps_spatial_control_pilot300_v1/SPEC.md
- conversations/01a18c45-0978-7563-9999-067f29e4426e.md
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-03T17:05:38.610079+09:00

원본 파일 직접 관찰: 23개 실행, 352개 코드·의존성 파일. 이전 commit: 5c2746ba8d8d81aa848cfb896433f950e3f0918f.

추가 6 / 변경 4 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/code/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/code/fps_spatial_control.py
- evidence/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/metadata/status.json
- reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/SPEC.md
- reports/experiments/zero-shot/README.md
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-03T18:00:00.000434+09:00

원본 파일 직접 관찰: 23개 실행, 354개 코드·의존성 파일. 이전 commit: a72e6b6c7b500f41843f4b64a7953f98c380ef88.

추가 7 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/conditional_rollout_analysis.py
- code/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/code/conditional_rollout_analysis.py
- evidence/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/metrics/budget.csv
- reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/ANALYSIS.md
- reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/NEXT_EXPERIMENT.md
- reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/REPORT.md
- reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/STRATA.md
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/conditional_rollout_budget.py
- evidence/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/metadata/status.json

## 2026-11-03T21:00:00.003212+09:00

원본 파일 직접 관찰: 25개 실행, 366개 코드·의존성 파일. 이전 commit: cdadd5da870c0522a3e19690578938dd592df53f.

추가 20 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/deterministic_rollout300.py
- code/experiments/zero-shot/_lib/deterministic_rollout300_report.py
- code/experiments/zero-shot/_lib/deterministic_rollout900.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/deterministic_rollout300.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/deterministic_rollout900.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/fps_spatial_control.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/deterministic_rollout300.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/fps_spatial_control.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metadata/status.json
- reports/experiments/zero-shot/results/deterministic_rollout_additional600_v2/SPEC.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/SPEC.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/STRATA.md
- reports/experiments/zero-shot/README.md
- reports/experiments/zero-shot/artifacts/README.md

## 2026-11-04T00:00:00.000955+09:00

원본 파일 직접 관찰: 27개 실행, 372개 코드·의존성 파일. 이전 commit: 11675f61a6b3600b799fd7c66cd121cc7f5c7e52.

추가 14 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/deterministic_rollout300.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/deterministic_rollout900.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/deterministic_rollout300.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/deterministic_rollout900.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metadata/status.json
- reports/experiments/zero-shot/results/deterministic_rollout_additional600_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_additional600_v2/STRATA.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/STRATA.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/STRATA.md
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metadata/status.json

## 2026-11-04T00:33:36.262216+09:00

원본 파일 직접 관찰: 27개 실행, 381개 코드·의존성 파일. 이전 commit: 1acdefd3b0181bbb209a3891f5d8bcf16941d45d.

추가 29 / 변경 13 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/deterministic_rollout_diagnostics.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/reporting/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/reporting/deterministic_rollout_diagnostics.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/boundary_errors.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/marginal_efficiency.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/recovery_and_selection.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_diagnostics.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/boundary_errors.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/marginal_efficiency.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/recovery_and_selection.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_diagnostics.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/boundary_errors.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/marginal_efficiency.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/recovery_and_selection.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_diagnostics.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/boundary_errors.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/marginal_efficiency.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/recovery_and_selection.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/start_end_coupling.csv
- code/experiments/zero-shot/_lib/deterministic_rollout300_report.py
- reports/experiments/zero-shot/artifacts/README.md
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/deterministic_rollout300_report.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/deterministic_rollout300_report.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- reports/experiments/zero-shot/results/deterministic_rollout_additional600_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/STRATA.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/REPORT.md
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-04T01:47:13.883220+09:00

원본 파일 직접 관찰: 27개 실행, 402개 코드·의존성 파일. 이전 commit: 665859e7bd6d0703580279f4216221d1333483c5.

추가 23 / 변경 24 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.css
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.js
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.py
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer_check.py
- code/experiments/zero-shot/_lib/rollout_distribution_integrate.py
- code/experiments/zero-shot/_lib/rollout_distribution_parallel.py
- code/experiments/zero-shot/_lib/rollout_distribution_viewer.js
- code/experiments/zero-shot/_lib/rollout_distribution_viewer_check.py
- code/experiments/zero-shot/_lib/rollout_token_distributions.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/distribution_capture/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/distribution_capture/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/distribution_capture/rollout_token_distributions.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/rollout_distribution_parallel.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/rollout_distribution_viewer.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/rollout_distribution_viewer_check.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/end_decimal_cases.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/viewer.js
- reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/END_DECIMAL_CASES.md
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/deterministic_rollout300_report.py
- code/experiments/zero-shot/_lib/deterministic_rollout_diagnostics.py
- reports/experiments/zero-shot/artifacts/README.md
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/reporting/deterministic_rollout_diagnostics.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_diagnostics.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_diagnostics.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_diagnostics.py
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/start_end_coupling.csv
- evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- reports/experiments/zero-shot/results/deterministic_rollout_additional600_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/REPORT.md
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-04T03:02:31.147942+09:00

원본 파일 직접 관찰: 23개 실행, 525개 코드·의존성 파일. 이전 commit: ead6ef5119a1400c1cd18a3aa4583621ecec9db2.

추가 405 / 변경 141 / 현재 원본에서 사라짐 241. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/rollout_pilot_html_watch.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/_lib/build_index.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/_lib/check_comparison.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/_lib/check_result_viewers.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/_lib/organize_results.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/_lib/unify_artifact_hierarchy.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/_lib/update_hierarchy_consumers.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/_lib/validate_result_layout.py
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/attention/README.md
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/attention/_lib/timelens_report.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/benchmark_16fps.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/integrate_adjacent.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/prepare.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/prepare_media.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/report_16fps.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/run_pipeline.py
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/README.md
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/run.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/adaptive_temporal_sampling.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/aggregate_all_zeroshot.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/artifact_store.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/boundary_oracle_analysis.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/boundary_oracle_inputs.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/cleanup_duplicate_artifacts.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/compact_selected_frames.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/compare_thinking_instruct.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/deterministic_rollout300.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/deterministic_rollout900.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/fps_spatial_supervisor.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_actual_density.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_dataset_diagnostics.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_density_combined.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_density_pilot.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_density_prefetch.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_density_queue.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_density_sweep.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_fraction_comparison.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_threshold_report.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/gt_video_fraction_axes.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/pilot900_actual.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/pilot900_extent.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/pilot900_inputs.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/pilot900_overview.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/pilot900_prefetch.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/pilot900_report_tables.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/pilot900_supervisor.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/rollout_distribution_integrate.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/rollout_pilot_html_watch.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/rollout_token_distributions.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/spatial_budget_pilot900.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/spatial_budget_report.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/spatial_budget_supervisor.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/spatial_sweep900.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/spatial_sweep_parallel_supervisor.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/spatial_sweep_supervisor.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_blue_analysis.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_blue_analysis_finish.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_digit_correctness.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_digit_correctness_browser.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_digit_correctness_finish.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_digit_correctness_serve.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_digit_role_entropy.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_endpoint_gt_entropy.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_entropy_integrated.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_entropy_integrated_browser.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_entropy_integrated_serve.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_entropy_iou.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_first_error_distance.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_full_blue_impl.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_full_digit_browser.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_full_digit_finish.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_full_digit_impl.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_full_digit_media.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_full_digit_prepare.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_full_digit_serve.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_mmco_diagnose.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_numeric_uncertainty.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_prefix_entropy.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_profile_preparation.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_rollout.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_rollout_report.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_span_head.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_span_markdown.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_span_monitor.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_span_report.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_top30.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_top30_browser.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_top30_finish.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_top30_full_worker.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_top30_media.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_top30_progress.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_top30_report.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_top30_serve.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_validate_video_cache.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_verify_cached_pipeline.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/timelens_verify_throughput.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/wait_for_cache_cleanup.py
- code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/zeroshot_report.py
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/artifacts/README.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/all_zeroshot_v1/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/fps_spatial_control_pilot300_v1/SPATIAL_AUDIT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/fps_spatial_control_pilot300_v1/SPEC.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/input_order/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/input_order/consistency_v1/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/spatial_budget_pilot900_v1/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/thinking_vs_instruct_v1/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_span_head/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30/digit_correctness/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30/digit_correctness/blue_analysis/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30/entropy_iou/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30/integrated/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30/mmco_diagnosis/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30/numeric_uncertainty/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30/prefix_entropy/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30_full/digit_correctness/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30_full/digit_correctness/blue_analysis/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30_full/digit_correctness/first_error_distance/REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30_full/performance/CACHE_REPORT.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30_full/performance/PIPELINE_REVIEW.md
- reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/timelens_top30_full/performance/REPORT.md
- evidence/experiments/zero-shot/artifacts/runs/baseline/all_zeroshot/overview/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/baseline/all_zeroshot/overview/metrics/metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/all_zeroshot/overview/metrics/rollout_metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/all_zeroshot/overview/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/code/analyze.py
- code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/code/build_viewer.py
- code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/code/serve.py
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metrics/largest_gt_degradation.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metrics/largest_gt_improvement.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metrics/largest_normalized_shift.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/metrics/benchmark_metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/metrics/invalid_prediction_refs.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/baseline/thinking_vs_instruct/pilot/300/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/rollout_distribution_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/end_decimal_cases.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/rollout_distribution_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/end_decimal_cases.csv
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/operations/inference_performance/full/9404/code/observe.py
- code/experiments/zero-shot/artifacts/runs/operations/inference_performance/full/9404/code/observe_after.py
- code/experiments/zero-shot/artifacts/runs/operations/inference_performance/full/9404/code/worker_before_cache.py
- evidence/experiments/zero-shot/artifacts/runs/operations/mmco_diagnosis/pilot/300/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/v1/code/boundary_oracle_inputs.py
- code/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/v1/code/pilot900_inputs.py
- evidence/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/v1/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/fps_spatial_supervisor.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/initial/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/initial/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/initial/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/initial/fps_spatial_supervisor.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_supervisor.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_supervisor.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/recovery_20261103/status.json
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/gt_density_batch.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/pilot900_inputs.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/shared_selected_frames.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/spatial_budget_before_overlap.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/spatial_budget_pilot900.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/spatial_budget_report.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metrics/compute_cost.csv
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metrics/metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metrics/spatial_inputs.csv
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/gt_density_batch.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/memory_fix_spatial_sweep_runtime.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/memory_fix_spatial_sweep_supervisor.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/pilot900_inputs.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/runtime_before_memory_fix.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/spatial_sweep900.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/spatial_sweep_inputs.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/spatial_sweep_report.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/metrics/axes.csv
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/metrics/metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/metrics/paired.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/protocol.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/reports/summary.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/results/qwen25/shard0/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/results/qwen25/shard1/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/results/qwen3/shard0/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/results/qwen3/shard1/config.json
- code/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/scripts/monitor.py
- code/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/scripts/report.py
- code/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort/scripts/run.py
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/samples/qwen25/shard0/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/samples/qwen25/shard1/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/samples/qwen3/shard0/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/samples/qwen3/shard1/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/samples/qwen3/shard2/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/samples/qwen3/shard3/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/span_head/pilot/300/samples/qwen3/smoke/config.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/candidate_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/candidate_future.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/future_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/length_control.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/position_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/metrics/three_axis_cross.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/first_error_distance/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/first_error_distance/metrics/group_statistics.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/first_error_distance/metrics/role_statistics.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/metrics/conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/metrics/digit_stratified.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/metrics/position_conditional_accuracy.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/metrics/role_conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/web/style.css
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/web/viewer.js
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metrics/candidate_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metrics/candidate_future.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metrics/future_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metrics/length_control.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metrics/per_blue_step.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metrics/position_axis.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/metrics/three_axis_cross.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/metrics/conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/metrics/digit_stratified.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/metrics/per_endpoint.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/metrics/position_conditional_accuracy.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/metrics/role_conditions.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/web/manifest.js
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/web/style.css
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/web/viewer.js
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_iou/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_iou/metrics/per_query.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/entropy_iou/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/metadata/status.json
- reports/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/digit_role_entropy/REPORT.md
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/digit_role_entropy/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/digit_role_entropy/summary.json
- reports/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/endpoint_gt_entropy/REPORT.md
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/endpoint_gt_entropy/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/endpoint_gt_entropy/summary.json
- reports/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/prefix_entropy/REPORT.md
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/prefix_entropy/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/prefix_entropy/summary.json
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/web/manifest.js
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/web/style.css
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/web/viewer.js
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/category_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/digit_stratified_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/endpoint_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/numeric_uncertainty/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/prefix_entropy/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/prefix_entropy/metrics/excluded.csv
- evidence/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/prefix_entropy/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/web/manifest.js
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/web/viewer.js
- reports/experiments/zero-shot/results/baseline/all_zeroshot/overview/REPORT.md
- reports/experiments/zero-shot/results/baseline/input_order/overview/REPORT.md
- reports/experiments/zero-shot/results/baseline/input_order/overview/consistency_v1/REPORT.md
- reports/experiments/zero-shot/results/baseline/thinking_vs_instruct/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/baseline/thinking_vs_instruct/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/END_DECIMAL_CASES.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/END_DECIMAL_CASES.md
- reports/experiments/zero-shot/results/operations/inference_performance/full/9404/CACHE_REPORT.md
- reports/experiments/zero-shot/results/operations/inference_performance/full/9404/PIPELINE_REVIEW.md
- reports/experiments/zero-shot/results/operations/inference_performance/full/9404/REPORT.md
- reports/experiments/zero-shot/results/operations/mmco_diagnosis/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/operations/mmco_diagnosis/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/operations/selected_frame_cache_migration/shared/v1/REPORT.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/300/SPATIAL_AUDIT.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/300/SPEC.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/900/SPEC.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/A_high.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/A_low.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/B_high.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/B_low.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/SPEC.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/span_head/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/span_head/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/digit_correctness/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/digit_correctness/first_error_distance/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/digit_correctness/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/entropy_iou/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/integrated/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/numeric_uncertainty/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/prefix_entropy/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/overview/REPORT.md
- evidence/experiments/attention/artifacts/runs/input_order/timelens_instruct/full/9404/metadata/status.json
- evidence/experiments/attention/artifacts/runs/input_order/timelens_instruct/full/9404/metrics/dataset_metrics.csv
- evidence/experiments/attention/artifacts/runs/input_order/timelens_instruct/full/9404/metrics/paired_metrics.csv
- evidence/experiments/attention/artifacts/runs/input_order/timelens_instruct/full/9404/metrics/summary.json
- code/experiments/attention/artifacts/runs/input_order/timelens_thinking/pilot/300/code/source.py
- evidence/experiments/attention/artifacts/runs/input_order/timelens_thinking/pilot/300/metadata/config.json
- evidence/experiments/attention/artifacts/runs/input_order/timelens_thinking/pilot/300/metadata/status.json
- evidence/experiments/attention/artifacts/runs/input_order/timelens_thinking/pilot/300/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/dynamics/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/generation/distribution_analysis/metrics/sample_metrics.csv
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/generation/distribution_analysis/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/generation/metrics/sample_metrics.csv
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/generation/metrics/summary.json
- code/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/hidden/fonts/LICENSE
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/hidden/metrics/entropy_matched_metrics.csv
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/hidden/metrics/query_token_metrics.csv
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/hidden/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/hidden/metrics/video_token_metrics.csv
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/metrics/sample_metrics.csv
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/query/metrics/sample_metrics.csv
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/query/metrics/summary.json
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/spatial/metrics/counterfactual_metrics.csv
- evidence/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/spatial/metrics/spatial_metrics.csv
- reports/experiments/attention/results/input_order/timelens_instruct/full/9404/final_report.md
- reports/experiments/attention/results/input_order/timelens_thinking/pilot/300/FINAL_REPORT.md
- reports/experiments/attention/results/input_order/timelens_thinking/pilot/300/REPORT.md
- reports/experiments/attention/results/input_order/timelens_thinking/pilot/300/SCHEMA.md
- reports/experiments/attention/results/input_order/timelens_thinking/pilot/overview/REPORT.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/REPORT.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/dynamics/FINDINGS.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/FINDINGS.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/distribution_analysis/REPORT.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/head_analysis/REPORT.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/query/FINDINGS.md
- reports/experiments/attention/results/token_attention/mtla/pilot/overview/REPORT.md
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/failure_modes/input_settings/pilot/200/metrics/baseline_paired_cases.csv
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/failure_modes/input_settings/pilot/200/metrics/case_metrics.csv
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/failure_modes/input_settings/pilot/200/metrics/setting_case_metrics.csv
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/failure_modes/input_settings/pilot/200/metrics/stratified_analysis.csv
- evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/failure_modes/input_settings/pilot/200/metrics/summary.json
- reports/experiments/analysis/failure_mode_timestamp/failure_cases/results/failure_modes/input_settings/pilot/200/REPORT.md
- reports/experiments/analysis/failure_mode_timestamp/failure_cases/results/failure_modes/input_settings/pilot/overview/REPORT.md
- evidence/experiments/similarity/artifacts/runs/compute_cost/embedding_vs_generation_16fps/pilot/300/config/config.json
- evidence/experiments/similarity/artifacts/runs/compute_cost/embedding_vs_generation_16fps/pilot/300/measurements/per_sample.csv
- evidence/experiments/similarity/artifacts/runs/compute_cost/embedding_vs_generation_16fps/pilot/300/measurements/summary.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/full/9404/summary.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/full/9404/clip/config.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/full/9404/clip/status.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/full/9404/qwen/config.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/full/9404/qwen/status.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/pilot/300/summary.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/pilot/300/clip/config.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/pilot/300/clip/status.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/pilot/300/qwen/config.json
- evidence/experiments/similarity/artifacts/runs/frame_query/cosine_2fps/pilot/300/qwen/status.json
- reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/300/REPORT.md
- reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/overview/REPORT.md
- reports/experiments/similarity/results/frame_query/cosine_2fps/full/9404/REPORT.md
- reports/experiments/similarity/results/frame_query/cosine_2fps/pilot/300/REPORT.md
- reports/experiments/similarity/results/frame_query/cosine_2fps/pilot/overview/REPORT.md
- code/experiments/_lib/cleanup_zero_shot_group_links.py
- code/experiments/_lib/finalize_zero_shot_scopes.py
- code/experiments/_lib/group_pilot_families.py
- code/experiments/_lib/group_project_experiments.py
- code/experiments/_lib/group_topic_first.py
- code/experiments/_lib/group_zero_shot.py
- code/experiments/_lib/pilot_catalog.py
- code/experiments/_lib/project_pilot_catalog.py
- code/experiments/_lib/repair_grouped_consumers.py
- code/experiments/_lib/validate_project_grouping.py
- code/experiments/_lib/validate_zero_shot_grouping.py
- conversations/01a1a254-4ff9-79f0-9647-3d3934565691.md
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/run.py
- code/experiments/zero-shot/_lib/adaptive_temporal_report.py
- code/experiments/zero-shot/_lib/adaptive_temporal_sampling.py
- code/experiments/zero-shot/_lib/aggregate_all_zeroshot.py
- code/experiments/zero-shot/_lib/artifact_store.py
- code/experiments/zero-shot/_lib/boundary_oracle_analysis.py
- code/experiments/zero-shot/_lib/boundary_oracle_inputs.py
- code/experiments/zero-shot/_lib/boundary_oracle_report.py
- code/experiments/zero-shot/_lib/cleanup_duplicate_artifacts.py
- code/experiments/zero-shot/_lib/compact_selected_frames.py
- code/experiments/zero-shot/_lib/compare_thinking_instruct.py
- code/experiments/zero-shot/_lib/conditional_rollout_budget.py
- code/experiments/zero-shot/_lib/deterministic_rollout300.py
- code/experiments/zero-shot/_lib/deterministic_rollout300_report.py
- code/experiments/zero-shot/_lib/deterministic_rollout900.py
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.css
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.js
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.py
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer_check.py
- code/experiments/zero-shot/_lib/fps_spatial_control.py
- code/experiments/zero-shot/_lib/fps_spatial_report.py
- code/experiments/zero-shot/_lib/fps_spatial_supervisor.py
- code/experiments/zero-shot/_lib/gt_actual_density.py
- code/experiments/zero-shot/_lib/gt_actual_density_report.py
- code/experiments/zero-shot/_lib/gt_dataset_diagnostics.py
- code/experiments/zero-shot/_lib/gt_density_combined.py
- code/experiments/zero-shot/_lib/gt_density_pilot.py
- code/experiments/zero-shot/_lib/gt_density_prefetch.py
- code/experiments/zero-shot/_lib/gt_density_queue.py
- code/experiments/zero-shot/_lib/gt_density_sweep.py
- code/experiments/zero-shot/_lib/gt_density_sweep_report.py
- code/experiments/zero-shot/_lib/gt_fraction_comparison.py
- code/experiments/zero-shot/_lib/gt_threshold_report.py
- code/experiments/zero-shot/_lib/gt_video_fraction_axes.py
- code/experiments/zero-shot/_lib/pilot900_actual.py
- code/experiments/zero-shot/_lib/pilot900_density_report.py
- code/experiments/zero-shot/_lib/pilot900_extent.py
- code/experiments/zero-shot/_lib/pilot900_inputs.py
- code/experiments/zero-shot/_lib/pilot900_overview.py
- code/experiments/zero-shot/_lib/pilot900_prefetch.py
- code/experiments/zero-shot/_lib/pilot900_report_tables.py
- code/experiments/zero-shot/_lib/pilot900_supervisor.py
- code/experiments/zero-shot/_lib/rollout_distribution_integrate.py
- code/experiments/zero-shot/_lib/rollout_distribution_viewer.js
- code/experiments/zero-shot/_lib/rollout_distribution_viewer_check.py
- code/experiments/zero-shot/_lib/rollout_token_distributions.py
- code/experiments/zero-shot/_lib/shared_selected_frames.py
- code/experiments/zero-shot/_lib/spatial_budget_pilot900.py
- code/experiments/zero-shot/_lib/spatial_budget_report.py
- code/experiments/zero-shot/_lib/spatial_budget_supervisor.py
- code/experiments/zero-shot/_lib/spatial_sweep900.py
- code/experiments/zero-shot/_lib/spatial_sweep_parallel_supervisor.py
- code/experiments/zero-shot/_lib/spatial_sweep_report.py
- code/experiments/zero-shot/_lib/spatial_sweep_report_s42.py
- code/experiments/zero-shot/_lib/spatial_sweep_supervisor.py
- code/experiments/zero-shot/_lib/temporal_pilot300_run.py
- code/experiments/zero-shot/_lib/timelens_blue_analysis.py
- code/experiments/zero-shot/_lib/timelens_blue_analysis_finish.py
- code/experiments/zero-shot/_lib/timelens_digit_correctness.py
- code/experiments/zero-shot/_lib/timelens_digit_correctness_browser.py
- code/experiments/zero-shot/_lib/timelens_digit_correctness_finish.py
- code/experiments/zero-shot/_lib/timelens_digit_correctness_serve.py
- code/experiments/zero-shot/_lib/timelens_digit_role_entropy.py
- code/experiments/zero-shot/_lib/timelens_endpoint_gt_entropy.py
- code/experiments/zero-shot/_lib/timelens_entropy_integrated.py
- code/experiments/zero-shot/_lib/timelens_entropy_integrated_browser.py
- code/experiments/zero-shot/_lib/timelens_entropy_integrated_serve.py
- code/experiments/zero-shot/_lib/timelens_entropy_iou.py
- code/experiments/zero-shot/_lib/timelens_first_error_distance.py
- code/experiments/zero-shot/_lib/timelens_full_blue_impl.py
- code/experiments/zero-shot/_lib/timelens_full_digit_browser.py
- code/experiments/zero-shot/_lib/timelens_full_digit_finish.py
- code/experiments/zero-shot/_lib/timelens_full_digit_impl.py
- code/experiments/zero-shot/_lib/timelens_full_digit_media.py
- code/experiments/zero-shot/_lib/timelens_full_digit_prepare.py
- code/experiments/zero-shot/_lib/timelens_full_digit_serve.py
- code/experiments/zero-shot/_lib/timelens_mmco_diagnose.py
- code/experiments/zero-shot/_lib/timelens_numeric_uncertainty.py
- code/experiments/zero-shot/_lib/timelens_prefix_entropy.py
- code/experiments/zero-shot/_lib/timelens_profile_preparation.py
- code/experiments/zero-shot/_lib/timelens_rollout.py
- code/experiments/zero-shot/_lib/timelens_rollout_report.py
- code/experiments/zero-shot/_lib/timelens_span_head.py
- code/experiments/zero-shot/_lib/timelens_span_markdown.py
- code/experiments/zero-shot/_lib/timelens_span_monitor.py
- code/experiments/zero-shot/_lib/timelens_span_report.py
- code/experiments/zero-shot/_lib/timelens_top30.py
- code/experiments/zero-shot/_lib/timelens_top30_browser.py
- code/experiments/zero-shot/_lib/timelens_top30_finish.py
- code/experiments/zero-shot/_lib/timelens_top30_full_worker.py
- code/experiments/zero-shot/_lib/timelens_top30_media.py
- code/experiments/zero-shot/_lib/timelens_top30_progress.py
- code/experiments/zero-shot/_lib/timelens_top30_report.py
- code/experiments/zero-shot/_lib/timelens_top30_serve.py
- code/experiments/zero-shot/_lib/timelens_validate_video_cache.py
- code/experiments/zero-shot/_lib/timelens_verify_cached_pipeline.py
- code/experiments/zero-shot/_lib/timelens_verify_throughput.py
- code/experiments/zero-shot/_lib/wait_for_cache_cleanup.py
- code/experiments/zero-shot/_lib/zeroshot_report.py
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/rollout_distribution_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/viewer.js
- reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/REPORT.md
- reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/REPORT.md
- reports/experiments/attention/README.md
- code/experiments/attention/_lib/dynamics_browser.py
- code/experiments/attention/_lib/dynamics_plot.py
- code/experiments/attention/_lib/spatial_viewer.py
- code/experiments/attention/_lib/timelens_report.py
- reports/experiments/attention/artifacts/README.md
- reports/experiments/analysis/failure_mode_timestamp/README.md
- reports/experiments/analysis/failure_mode_timestamp/failure_cases/README.md
- reports/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/README.md
- reports/experiments/similarity/README.md
- code/experiments/similarity/_lib/benchmark_16fps.py
- code/experiments/similarity/_lib/integrate_adjacent.py
- code/experiments/similarity/_lib/prepare.py
- code/experiments/similarity/_lib/prepare_media.py
- code/experiments/similarity/_lib/report.py
- code/experiments/similarity/_lib/report_16fps.py
- code/experiments/similarity/_lib/run_pipeline.py
- reports/experiments/similarity/artifacts/README.md
- code/experiments/_lib/build_index.py
- code/experiments/_lib/check_comparison.py
- code/experiments/_lib/check_index.py
- code/experiments/_lib/check_result_viewers.py
- code/experiments/_lib/organize_results.py
- code/experiments/_lib/result_layout.py
- code/experiments/_lib/unify_artifact_hierarchy.py
- code/experiments/_lib/update_hierarchy_consumers.py
- code/experiments/_lib/validate_result_layout.py
- 원본 부재: code/experiments/zero-shot/_lib/zeroshot_metrics_png.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metrics/metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metrics/rollout_metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/all_zeroshot_v1/metrics/summary.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_control.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_smoke.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_supervisor.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/initial/fps_spatial_control.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/initial/fps_spatial_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/initial/fps_spatial_smoke.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/initial/fps_spatial_supervisor.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_control.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_smoke.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/0f15c7b68db7490f8359aaeddd30c0c6593a2107e365cae57519d7430e98e659/fps_spatial_supervisor.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_control.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_smoke.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/versions/543c552cf193d7b1a385cd7926d52e8eed037f7b7e23f7ed39241e8eb66e7bcc/fps_spatial_supervisor.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/metadata/recovery_20261103/status.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/code/analyze.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/code/build_viewer.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/code/serve.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/metrics/largest_gt_degradation.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/metrics/largest_gt_improvement.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/metrics/largest_normalized_shift.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/input_order/metrics/benchmark_metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/input_order/metrics/invalid_prediction_refs.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/input_order/metrics/summary.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/selected_frame_cache_migration_v1/code/boundary_oracle_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/selected_frame_cache_migration_v1/code/pilot900_inputs.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/selected_frame_cache_migration_v1/metadata/status.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/gt_density_batch.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/pilot900_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/shared_selected_frames.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_before_overlap.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_pilot900.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_report.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/metrics/compute_cost.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/metrics/metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/metrics/spatial_inputs.csv
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/gt_density_batch.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/memory_fix_spatial_sweep_runtime.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/memory_fix_spatial_sweep_supervisor.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/pilot900_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/runtime_before_memory_fix.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep900.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep_report.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metrics/axes.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/metrics/metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_v1/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/metrics/paired.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/protocol.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/reports/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/results/qwen25/shard0/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/results/qwen25/shard1/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/results/qwen3/shard0/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/results/qwen3/shard1/config.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/monitor.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_span_head/provenance/aborted_old_cohort/scripts/run.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/samples/qwen25/shard0/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/samples/qwen25/shard1/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/samples/qwen3/shard0/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/samples/qwen3/shard1/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/samples/qwen3/shard2/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/samples/qwen3/shard3/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_span_head/samples/qwen3/smoke/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/candidate_axis.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/candidate_future.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/future_axis.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/length_control.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/per_blue_step.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/position_axis.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/blue_analysis/metrics/three_axis_cross.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/conditions.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/digit_stratified.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/per_endpoint.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/position_conditional_accuracy.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/role_conditions.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/metrics/summary.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/manifest.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/style.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30/digit_correctness/web/viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/entropy_iou/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/entropy_iou/metrics/per_query.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/entropy_iou/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/metadata/status.json
- 원본 부재: reports/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/digit_role_entropy/REPORT.md
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/digit_role_entropy/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/digit_role_entropy/summary.json
- 원본 부재: reports/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/endpoint_gt_entropy/REPORT.md
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/endpoint_gt_entropy/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/endpoint_gt_entropy/summary.json
- 원본 부재: reports/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/prefix_entropy/REPORT.md
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/prefix_entropy/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/sources/prefix_entropy/summary.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/web/manifest.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/web/style.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30/integrated/web/viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/mmco_diagnosis/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/numeric_uncertainty/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/numeric_uncertainty/metrics/category_summary.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/numeric_uncertainty/metrics/digit_stratified_summary.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/numeric_uncertainty/metrics/endpoint_role_summary.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/numeric_uncertainty/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/prefix_entropy/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/prefix_entropy/metrics/excluded.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30/prefix_entropy/metrics/summary.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30/web/manifest.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30/web/viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/candidate_axis.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/candidate_future.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/future_axis.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/length_control.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/position_axis.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/summary.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/blue_analysis/metrics/three_axis_cross.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metrics/group_statistics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/first_error_distance/metrics/role_statistics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/conditions.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/digit_stratified.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/position_conditional_accuracy.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/role_conditions.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/metrics/summary.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/web/style.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30_full/digit_correctness/web/viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_top30_full/metadata/status.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30_full/performance/code/observe.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30_full/performance/code/observe_after.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_top30_full/performance/code/worker_before_cache.py
- 원본 부재: reports/experiments/zero-shot/results/all_zeroshot_v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/fps_spatial_control_pilot300_v1/SPATIAL_AUDIT.md
- 원본 부재: reports/experiments/zero-shot/results/fps_spatial_control_pilot300_v1/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/input_order/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/input_order/consistency_v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/legacy_quantitative/README.md
- 원본 부재: reports/experiments/zero-shot/results/selected_frame_cache_migration_v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_budget_pilot900_v1/PILOT_300.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_budget_pilot900_v1/PILOT_600.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_budget_pilot900_v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_budget_pilot900_v1/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/A_high.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/A_low.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/B_high.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/B_low.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/thinking_vs_instruct_v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_span_head/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30/digit_correctness/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30/digit_correctness/blue_analysis/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30/entropy_iou/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30/integrated/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30/mmco_diagnosis/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30/numeric_uncertainty/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30/prefix_entropy/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30_full/digit_correctness/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30_full/digit_correctness/blue_analysis/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30_full/digit_correctness/first_error_distance/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30_full/performance/CACHE_REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30_full/performance/PIPELINE_REVIEW.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_top30_full/performance/REPORT.md
- 원본 부재: evidence/experiments/attention/artifacts/runs/dynamics/metrics/summary.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/generation/distribution_analysis/metrics/sample_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/generation/distribution_analysis/metrics/summary.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/generation/metrics/sample_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/generation/metrics/summary.json
- 원본 부재: code/experiments/attention/artifacts/runs/hidden/fonts/LICENSE
- 원본 부재: evidence/experiments/attention/artifacts/runs/hidden/metrics/entropy_matched_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/hidden/metrics/query_token_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/hidden/metrics/summary.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/hidden/metrics/video_token_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/overview/metrics/sample_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/overview/metrics/summary.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/query/metrics/sample_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/query/metrics/summary.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/spatial/metrics/counterfactual_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/spatial/metrics/spatial_metrics.csv
- 원본 부재: code/experiments/attention/artifacts/runs/thinking_pilot_orders_v1/code/source.py
- 원본 부재: evidence/experiments/attention/artifacts/runs/thinking_pilot_orders_v1/metadata/config.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/thinking_pilot_orders_v1/metadata/status.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/thinking_pilot_orders_v1/metrics/summary.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/timelens_order/metadata/status.json
- 원본 부재: evidence/experiments/attention/artifacts/runs/timelens_order/metrics/dataset_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/timelens_order/metrics/paired_metrics.csv
- 원본 부재: evidence/experiments/attention/artifacts/runs/timelens_order/metrics/summary.json
- 원본 부재: reports/experiments/attention/results/dynamics/FINDINGS.md
- 원본 부재: reports/experiments/attention/results/generation/FINDINGS.md
- 원본 부재: reports/experiments/attention/results/generation/distribution_analysis/REPORT.md
- 원본 부재: reports/experiments/attention/results/generation/head_analysis/REPORT.md
- 원본 부재: reports/experiments/attention/results/overview/REPORT.md
- 원본 부재: reports/experiments/attention/results/query/FINDINGS.md
- 원본 부재: reports/experiments/attention/results/thinking_pilot_orders_v1/FINAL_REPORT.md
- 원본 부재: reports/experiments/attention/results/thinking_pilot_orders_v1/REPORT.md
- 원본 부재: reports/experiments/attention/results/thinking_pilot_orders_v1/SCHEMA.md
- 원본 부재: reports/experiments/attention/results/timelens_order/final_report.md
- 원본 부재: evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/baseline_paired_cases.csv
- 원본 부재: evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/case_metrics.csv
- 원본 부재: evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/setting_case_metrics.csv
- 원본 부재: evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/stratified_analysis.csv
- 원본 부재: evidence/experiments/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/metrics/summary.json
- 원본 부재: reports/experiments/analysis/failure_mode_timestamp/failure_cases/results/pilot_100/REPORT.md
- 원본 부재: evidence/experiments/similarity/artifacts/runs/full_9404_v1/summary.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/full_9404_v1/clip/config.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/full_9404_v1/clip/status.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/full_9404_v1/qwen/config.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/full_9404_v1/qwen/status.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/pilot_300_16fps_cost_v1/config/config.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/pilot_300_16fps_cost_v1/measurements/per_sample.csv
- 원본 부재: evidence/experiments/similarity/artifacts/runs/pilot_300_16fps_cost_v1/measurements/summary.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/pilot_300_v1/summary.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/pilot_300_v1/clip/config.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/pilot_300_v1/clip/status.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/pilot_300_v1/qwen/config.json
- 원본 부재: evidence/experiments/similarity/artifacts/runs/pilot_300_v1/qwen/status.json
- 원본 부재: reports/experiments/similarity/results/full_9404_v1/REPORT.md
- 원본 부재: reports/experiments/similarity/results/pilot_300_16fps_cost_v1/REPORT.md
- 원본 부재: reports/experiments/similarity/results/pilot_300_v1/REPORT.md

## 2026-11-04T04:19:55.812632+09:00

원본 파일 직접 관찰: 18개 실행, 533개 코드·의존성 파일. 이전 commit: 70891e4c33b94ae1a35998377eae162eb9853544.

추가 159 / 변경 78 / 현재 원본에서 사라짐 158. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/analyze.py
- code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/build_viewer.py
- code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/serve.py
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/metrics/largest_gt_degradation.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/metrics/largest_gt_improvement.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/metrics/largest_normalized_shift.csv
- evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/code/boundary_oracle_inputs.py
- code/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/code/pilot900_inputs.py
- evidence/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/code/conditional_rollout_analysis.py
- code/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/code/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/code/fps_spatial_control.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/metrics/budget.csv
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/deterministic_rollout300.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_diagnostics.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/rollout_distribution_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metrics/boundary_errors.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metrics/end_decimal_cases.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metrics/marginal_efficiency.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metrics/recovery_and_selection.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/deterministic_rollout300.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/deterministic_rollout900.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_diagnostics.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/rollout_distribution_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/metrics/boundary_errors.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/metrics/end_decimal_cases.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/metrics/marginal_efficiency.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/metrics/recovery_and_selection.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/deterministic_rollout300.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/deterministic_rollout900.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/distribution_capture/conditional_rollout_budget.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/distribution_capture/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/distribution_capture/rollout_token_distributions.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout300_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_diagnostics.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/rollout_distribution_integrate.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/rollout_distribution_parallel.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/rollout_distribution_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/rollout_distribution_viewer_check.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/metrics/boundary_errors.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/metrics/branch_role_summary.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/metrics/end_decimal_cases.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/metrics/marginal_efficiency.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/metrics/recovery_and_selection.csv
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/metrics/start_end_coupling.csv
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/stochastic/pilot/300/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/stochastic/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/stochastic/pilot/300/metrics/summary.json
- reports/experiments/zero-shot/artifacts/runs/rollout/stochastic/pilot/300/sources/Qwen2.5-VL-7B-Instruct_README.md
- reports/experiments/zero-shot/artifacts/runs/rollout/stochastic/pilot/300/sources/Qwen3-VL-8B-Instruct_README.md
- code/experiments/zero-shot/artifacts/runs/rollout/stochastic/pilot/300/web/manifest.js
- code/experiments/zero-shot/artifacts/runs/rollout/stochastic/pilot/300/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/support/adaptive_temporal_sampling.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/cf730e867c3033a02d6ba92e542f39c6e29facf3d4bbb6691afea7482d32cd19/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/cf730e867c3033a02d6ba92e542f39c6e29facf3d4bbb6691afea7482d32cd19/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/cf730e867c3033a02d6ba92e542f39c6e29facf3d4bbb6691afea7482d32cd19/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/cf730e867c3033a02d6ba92e542f39c6e29facf3d4bbb6691afea7482d32cd19/fps_spatial_supervisor.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/code/gt_density_batch.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/code/memory_fix_spatial_sweep_runtime.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/code/memory_fix_spatial_sweep_supervisor.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/code/pilot900_inputs.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/code/runtime_before_memory_fix.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/code/spatial_sweep900.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/code/spatial_sweep_inputs.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/code/spatial_sweep_report.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/metrics/axes.csv
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/allocation_sweep/pilot/900/metrics/metrics.csv
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/code/gt_density_batch.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/code/pilot900_inputs.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/code/shared_selected_frames.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/code/spatial_budget_before_overlap.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/code/spatial_budget_pilot900.py
- code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/code/spatial_budget_report.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/metrics/compute_cost.csv
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/metrics/metrics.csv
- evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_comparison/pilot/900/metrics/spatial_inputs.csv
- reports/experiments/zero-shot/results/baseline/input_order/overview/consistency/REPORT.md
- reports/experiments/zero-shot/results/operations/selected_frame_cache_migration/shared/REPORT.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/ANALYSIS.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/NEXT_EXPERIMENT.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/REPORT.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/SPEC.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/STRATA.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/END_DECIMAL_CASES.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/SPEC.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/END_DECIMAL_CASES.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/END_DECIMAL_CASES.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/rollout/stochastic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/stochastic/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/900/A_high.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/900/A_low.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/900/B_high.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/900/B_low.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/900/SPEC.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/900/SPEC.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/overview/REPORT.md
- code/experiments/_lib/check_family_views.py
- code/experiments/_lib/family_catalog.py
- code/experiments/_lib/family_view.css
- code/experiments/_lib/family_view.js
- code/experiments/_lib/finalize_result_aliases.py
- code/experiments/_lib/merge_rollout_expansion.py
- code/experiments/_lib/organize_completed_rollouts.py
- code/experiments/_lib/remove_result_version_labels.py
- code/experiments/_lib/result_terms.py
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/adaptive_temporal_sampling.py
- code/experiments/zero-shot/_lib/conditional_rollout_budget.py
- code/experiments/zero-shot/_lib/deterministic_rollout300.py
- code/experiments/zero-shot/_lib/deterministic_rollout300_report.py
- code/experiments/zero-shot/_lib/deterministic_rollout900.py
- code/experiments/zero-shot/_lib/deterministic_rollout_diagnostics.py
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.css
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.js
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.py
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer_check.py
- code/experiments/zero-shot/_lib/fps_spatial_supervisor.py
- code/experiments/zero-shot/_lib/rollout_distribution_integrate.py
- code/experiments/zero-shot/_lib/rollout_distribution_parallel.py
- code/experiments/zero-shot/_lib/rollout_distribution_viewer.js
- code/experiments/zero-shot/_lib/rollout_distribution_viewer_check.py
- code/experiments/zero-shot/_lib/rollout_pilot_html_watch.py
- code/experiments/zero-shot/_lib/rollout_token_distributions.py
- code/experiments/zero-shot/_lib/spatial_budget_supervisor.py
- code/experiments/zero-shot/_lib/spatial_sweep900.py
- code/experiments/zero-shot/_lib/spatial_sweep_parallel_supervisor.py
- code/experiments/zero-shot/_lib/spatial_sweep_supervisor.py
- code/experiments/zero-shot/_lib/timelens_rollout_serve.py
- reports/experiments/zero-shot/artifacts/README.md
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/fps_spatial_supervisor.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/full/9404/digit_correctness/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/digit_correctness/web/viewer.js
- reports/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/sources/prefix_entropy/REPORT.md
- code/experiments/zero-shot/artifacts/runs/uncertainty/token_analysis/pilot/300/integrated/web/viewer.js
- reports/experiments/zero-shot/results/baseline/all_zeroshot/overview/REPORT.md
- reports/experiments/zero-shot/results/baseline/input_order/overview/REPORT.md
- reports/experiments/zero-shot/results/baseline/thinking_vs_instruct/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/baseline/thinking_vs_instruct/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/operations/inference_performance/full/9404/PIPELINE_REVIEW.md
- reports/experiments/zero-shot/results/operations/mmco_diagnosis/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/300/SPATIAL_AUDIT.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/300/SPEC.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/span_head/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/digit_correctness/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/digit_correctness/first_error_distance/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/digit_correctness/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/entropy_iou/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/integrated/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/numeric_uncertainty/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/prefix_entropy/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/overview/REPORT.md
- reports/experiments/attention/README.md
- reports/experiments/attention/results/input_order/timelens_thinking/pilot/300/FINAL_REPORT.md
- reports/experiments/attention/results/input_order/timelens_thinking/pilot/300/REPORT.md
- reports/experiments/attention/results/input_order/timelens_thinking/pilot/overview/REPORT.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/dynamics/FINDINGS.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/FINDINGS.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/distribution_analysis/REPORT.md
- reports/experiments/attention/results/token_attention/mtla/pilot/overview/REPORT.md
- reports/experiments/analysis/failure_mode_timestamp/failure_cases/README.md
- reports/experiments/analysis/failure_mode_timestamp/failure_cases/results/failure_modes/input_settings/pilot/overview/REPORT.md
- reports/experiments/similarity/README.md
- reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/300/REPORT.md
- reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/overview/REPORT.md
- reports/experiments/similarity/results/frame_query/cosine_2fps/pilot/overview/REPORT.md
- code/experiments/_lib/build_index.py
- code/experiments/_lib/check_comparison.py
- code/experiments/_lib/check_index.py
- code/experiments/_lib/check_result_viewers.py
- code/experiments/_lib/index_compare.js
- code/experiments/_lib/pilot_catalog.py
- code/experiments/_lib/project_pilot_catalog.py
- code/experiments/_lib/result_layout.py
- code/experiments/_lib/serve_index.py
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md
- conversations/01a1a254-4ff9-79f0-9647-3d3934565691.md
- 원본 부재: code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/code/analyze.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/code/build_viewer.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/code/serve.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metrics/largest_gt_degradation.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metrics/largest_gt_improvement.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metrics/largest_normalized_shift.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency_v1/metrics/summary.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/code/conditional_rollout_analysis.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/code/conditional_rollout_budget.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/code/fps_spatial_control.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/conditional_rollout_budget_pilot30_v1/metrics/budget.csv
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/conditional_rollout_budget.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/deterministic_rollout300.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/deterministic_rollout300_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/deterministic_rollout900.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/fps_spatial_control.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/reporting/deterministic_rollout300_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/code/reporting/deterministic_rollout_diagnostics.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/boundary_errors.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/branch_role_summary.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/marginal_efficiency.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/recovery_and_selection.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_additional600_v2/metrics/start_end_coupling.csv
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/conditional_rollout_budget.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/deterministic_rollout300.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/deterministic_rollout300_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/fps_spatial_control.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout300_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_diagnostics.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_viewer.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_viewer.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/deterministic_rollout_viewer.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/code/reporting/rollout_distribution_viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/boundary_errors.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/branch_role_summary.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/end_decimal_cases.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/marginal_efficiency.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/recovery_and_selection.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/metrics/start_end_coupling.csv
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/web/distributions.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/web/viewer.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot300_v2/web/viewer.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/deterministic_rollout300.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/deterministic_rollout300_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/deterministic_rollout900.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout300_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_diagnostics.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_viewer.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_viewer.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/deterministic_rollout_viewer.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/code/reporting/rollout_distribution_viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/boundary_errors.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/branch_role_summary.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/end_decimal_cases.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/marginal_efficiency.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/recovery_and_selection.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/metrics/start_end_coupling.csv
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/web/distributions.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/web/viewer.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot600_v2/web/viewer.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/deterministic_rollout300.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/deterministic_rollout300_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/deterministic_rollout900.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/distribution_capture/conditional_rollout_budget.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/distribution_capture/fps_spatial_control.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/distribution_capture/rollout_token_distributions.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout300_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_diagnostics.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/deterministic_rollout_viewer.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/rollout_distribution_parallel.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/rollout_distribution_viewer.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/code/reporting/rollout_distribution_viewer_check.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/boundary_errors.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/branch_role_summary.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/end_decimal_cases.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/marginal_efficiency.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/recovery_and_selection.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/metrics/start_end_coupling.csv
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/distributions.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/viewer.css
- 원본 부재: code/experiments/zero-shot/artifacts/runs/deterministic_rollout_pilot900_v2/web/viewer.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/v1/code/boundary_oracle_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/v1/code/pilot900_inputs.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/operations/selected_frame_cache_migration/shared/v1/metadata/status.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/gt_density_batch.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/pilot900_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/shared_selected_frames.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/spatial_budget_before_overlap.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/spatial_budget_pilot900.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/code/spatial_budget_report.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metrics/compute_cost.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metrics/metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/budget_v1/pilot/900/metrics/spatial_inputs.csv
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/gt_density_batch.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/memory_fix_spatial_sweep_runtime.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/memory_fix_spatial_sweep_supervisor.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/pilot900_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/runtime_before_memory_fix.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/spatial_sweep900.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/spatial_sweep_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/code/spatial_sweep_report.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/metrics/axes.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/sampling/spatial_allocation/sweep_v2/pilot/900/metrics/metrics.csv
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_rollout/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_rollout/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/timelens_rollout/metrics/summary.json
- 원본 부재: reports/experiments/zero-shot/artifacts/runs/timelens_rollout/sources/Qwen2.5-VL-7B-Instruct_README.md
- 원본 부재: reports/experiments/zero-shot/artifacts/runs/timelens_rollout/sources/Qwen3-VL-8B-Instruct_README.md
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_rollout/web/manifest.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/timelens_rollout/web/viewer.js
- 원본 부재: reports/experiments/zero-shot/results/baseline/input_order/overview/consistency_v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/ANALYSIS.md
- 원본 부재: reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/NEXT_EXPERIMENT.md
- 원본 부재: reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/conditional_rollout_budget_pilot30_v1/STRATA.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_additional600_v2/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_additional600_v2/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_additional600_v2/STRATA.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/END_DECIMAL_CASES.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot300_v2/STRATA.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/END_DECIMAL_CASES.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot600_v2/STRATA.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/END_DECIMAL_CASES.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/deterministic_rollout_pilot900_v2/STRATA.md
- 원본 부재: reports/experiments/zero-shot/results/operations/selected_frame_cache_migration/shared/v1/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/300/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/600/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/900/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/900/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_v1/pilot/overview/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/A_high.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/A_low.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/B_high.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/B_low.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/900/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/sampling/spatial_allocation/sweep_v2/pilot/overview/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/timelens_rollout/REPORT.md

## 2026-11-04T09:00:00.000308+09:00

원본 파일 직접 관찰: 18개 실행, 533개 코드·의존성 파일. 이전 commit: fcd17923d1266581bfdd85a36875aa9a8efa71eb.

추가 0 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json

## 2026-11-04T11:35:13.945507+09:00

원본 파일 직접 관찰: 18개 실행, 533개 코드·의존성 파일. 이전 commit: cfba42a94e723a276179e90a80d8bd50ca9dba5c.

추가 0 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-04T12:00:00.001012+09:00

원본 파일 직접 관찰: 18개 실행, 548개 코드·의존성 파일. 이전 commit: affcd32e6223eb649b22e083645fcf83d516d1b3.

추가 21 / 변경 39 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/rollout_reconsideration.py
- code/experiments/zero-shot/_lib/rollout_reconsideration_check.py
- code/experiments/zero-shot/_lib/rollout_reconsideration_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/rollout_distribution_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/rollout_distribution_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/code/rollout_reconsideration.py
- code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/code/rollout_reconsideration_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/code/rollout_reconsideration_report.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/metadata/status.json
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/600/SPEC.md
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/600/STRATA.md
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/overview/REPORT.md
- code/experiments/_lib/check_index_preferences.py
- code/experiments/_lib/experiment_theme.css
- code/experiments/_lib/index_catalog.css
- code/experiments/_lib/index_catalog.js
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.css
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.js
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.py
- code/experiments/zero-shot/_lib/rollout_distribution_viewer.js
- code/experiments/zero-shot/_lib/rollout_distribution_viewer_check.py
- reports/experiments/zero-shot/artifacts/README.md
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/rollout_distribution_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/rollout_distribution_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/rollout_distribution_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/rollout_distribution_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/viewer.js
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/REPORT.md
- reports/experiments/_lib/README.md
- code/experiments/_lib/build_index.py
- code/experiments/_lib/check_comparison.py
- code/experiments/_lib/check_family_views.py
- code/experiments/_lib/family_catalog.py
- code/experiments/_lib/family_view.css
- code/experiments/_lib/index_compare.js

## 2026-11-04T13:35:23.935355+09:00

원본 파일 직접 관찰: 18개 실행, 549개 코드·의존성 파일. 이전 commit: f408d42cd6bf9d10b63dda880a3b31e2376c7b1c.

추가 4 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/300/code/reporting/rollout_reconsideration_report.py
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/300/STRATA.md
- conversations/01a1a468-0ed4-7750-978e-18011c5d8260.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/metadata/status.json
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/overview/REPORT.md

## 2026-11-04T15:00:00.002351+09:00

원본 파일 직접 관찰: 18개 실행, 550개 코드·의존성 파일. 이전 commit: d8a407618466cc2767bcb4cd2fcad55f6a9b0af6.

추가 1 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/code/reporting/rollout_reconsideration_report.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/metadata/status.json
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/reconsideration/pilot/600/STRATA.md

## 2026-11-04T15:30:43.191821+09:00

원본 파일 직접 관찰: 18개 실행, 556개 코드·의존성 파일. 이전 commit: 1eca87e7a0c5f84b789693904cfdd0f9088e1003.

추가 7 / 변경 17 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/a9520c1d4f0e6cfe8672e01ee86097bca6e3be4769356590f673b0053e83051f/fps_spatial_control.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/a9520c1d4f0e6cfe8672e01ee86097bca6e3be4769356590f673b0053e83051f/fps_spatial_report.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/a9520c1d4f0e6cfe8672e01ee86097bca6e3be4769356590f673b0053e83051f/fps_spatial_smoke.py
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/versions/a9520c1d4f0e6cfe8672e01ee86097bca6e3be4769356590f673b0053e83051f/fps_spatial_supervisor.py
- reports/experiments/zero-shot/results/rollout/reconsideration/shared/NEXT_EXPERIMENT.md
- code/experiments/_lib/check_view_controls.py
- code/experiments/_lib/history_controls.js
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.css
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/code/fps_spatial_supervisor.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- reports/experiments/_lib/README.md
- code/experiments/_lib/build_index.py
- code/experiments/_lib/check_comparison.py
- code/experiments/_lib/check_family_views.py
- code/experiments/_lib/experiment_theme.css
- code/experiments/_lib/family_catalog.py
- code/experiments/_lib/family_view.css
- code/experiments/_lib/family_view.js
- code/experiments/_lib/index_catalog.css
- code/experiments/_lib/index_compare.css
- code/experiments/_lib/index_compare.js
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-04T17:49:36.419495+09:00

원본 파일 직접 관찰: 18개 실행, 573개 코드·의존성 파일. 이전 commit: eb49a98c49cba6f4b1da0fe1e5a06a65cccacc20.

추가 31 / 변경 50 / 현재 원본에서 사라짐 14. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/rollout_reconsideration_viewer.css
- code/experiments/zero-shot/_lib/rollout_reconsideration_viewer.js
- code/experiments/zero-shot/_lib/rollout_reconsideration_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/code/reporting/rollout_reconsideration_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/code/reporting/rollout_reconsideration_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/code/reporting/rollout_reconsideration_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/code/reporting/rollout_reconsideration_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/web/reconsideration.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/web/reconsideration.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/rollout_reconsideration.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/rollout_reconsideration_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/rollout_reconsideration_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/reporting/rollout_reconsideration_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/reporting/rollout_reconsideration_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/reporting/rollout_reconsideration_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/reporting/rollout_reconsideration_viewer_check.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/web/reconsideration.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/web/reconsideration.js
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/reconsideration/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/SPEC.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/overview/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/shared/reconsideration/NEXT_EXPERIMENT.md
- code/experiments/_lib/check_bookmark_refresh.py
- code/experiments/_lib/family_bookmarks.js
- code/experiments/_lib/family_view_test_utils.py
- code/experiments/_lib/nest_rollout_reconsideration.py
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.css
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.js
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.py
- code/experiments/zero-shot/_lib/deterministic_rollout_viewer_check.py
- code/experiments/zero-shot/_lib/rollout_distribution_viewer.js
- code/experiments/zero-shot/_lib/rollout_distribution_viewer_check.py
- code/experiments/zero-shot/_lib/rollout_reconsideration.py
- code/experiments/zero-shot/_lib/rollout_reconsideration_report.py
- reports/experiments/zero-shot/artifacts/README.md
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/rollout_distribution_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/rollout_distribution_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/rollout_distribution_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/distributions.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/viewer.js
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/REPORT.md
- reports/experiments/_lib/README.md
- code/experiments/_lib/build_index.py
- code/experiments/_lib/check_comparison.py
- code/experiments/_lib/check_family_views.py
- code/experiments/_lib/check_view_controls.py
- code/experiments/_lib/experiment_theme.css
- code/experiments/_lib/family_catalog.py
- code/experiments/_lib/family_view.css
- code/experiments/_lib/family_view.js
- code/experiments/_lib/history_controls.js
- code/experiments/_lib/index_catalog.css
- code/experiments/_lib/index_catalog.js
- code/experiments/_lib/index_compare.js
- code/experiments/_lib/pilot_catalog.py
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/300/code/reporting/rollout_reconsideration_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/code/rollout_reconsideration.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/code/rollout_reconsideration_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/code/rollout_reconsideration_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/code/reporting/rollout_reconsideration_report.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/reconsideration/pilot/600/metadata/status.json
- 원본 부재: reports/experiments/zero-shot/results/rollout/reconsideration/pilot/300/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/reconsideration/pilot/300/STRATA.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/reconsideration/pilot/600/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/reconsideration/pilot/600/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/reconsideration/pilot/600/STRATA.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/reconsideration/pilot/overview/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/reconsideration/shared/NEXT_EXPERIMENT.md

## 2026-11-04T18:00:00.003322+09:00

원본 파일 직접 관찰: 18개 실행, 573개 코드·의존성 파일. 이전 commit: 8deb279fbd3caff8650d19b90e14e058b13205b0.

추가 0 / 변경 16 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/deterministic_rollout_viewer.js
- code/experiments/zero-shot/_lib/rollout_reconsideration_viewer.js
- code/experiments/zero-shot/_lib/rollout_reconsideration_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/code/reporting/rollout_reconsideration_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/code/reporting/rollout_reconsideration_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/web/reconsideration.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/reporting/rollout_reconsideration_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/reporting/rollout_reconsideration_viewer_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/web/reconsideration.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/code/reporting/deterministic_rollout_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/web/viewer.js
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json

## 2026-11-04T20:25:10.080244+09:00

원본 파일 직접 관찰: 18개 실행, 573개 코드·의존성 파일. 이전 commit: bfd9f016e05840b58f697b850452c8be245a2215.

추가 1 / 변경 4 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- conversations/01a1a53c-e6a1-7181-b338-d38607a7b02b.md
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- code/experiments/_lib/check_view_controls.py
- code/experiments/_lib/family_view.js
- code/experiments/_lib/serve_index.py

## 2026-11-04T21:00:00.004471+09:00

원본 파일 직접 관찰: 18개 실행, 573개 코드·의존성 파일. 이전 commit: 0d324b533f8fe89edd7484a43c8d530bc3e38ad6.

추가 0 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json

## 2026-11-05T00:00:00.001308+09:00

원본 파일 직접 관찰: 18개 실행, 573개 코드·의존성 파일. 이전 commit: 93f91296a51f40a7687928710e3c310112a7eccb.

추가 0 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json

## 2026-11-05T02:08:03.461123+09:00

원본 파일 직접 관찰: 18개 실행, 575개 코드·의존성 파일. 이전 commit: 5953f6c089781c50ba7d383db6aca47582ab8704.

추가 2 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/event_tree_core.py
- code/experiments/zero-shot/_lib/event_tree_runtime.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-05T09:00:00.000323+09:00

원본 파일 직접 관찰: 18개 실행, 583개 코드·의존성 파일. 이전 commit: 8ba9a3b1342d4e35a6fbdd5eb27dc1600fa9956c.

추가 19 / 변경 5 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/event_tree_check.py
- code/experiments/zero-shot/_lib/event_tree_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/300/code/reporting/event_tree_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/600/code/reporting/event_tree_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/code/event_tree_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/code/event_tree_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/code/event_tree_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/code/event_tree_runtime.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/metadata/status.json
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/300/STRATA.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/600/STRATA.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/SPEC.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/overview/REPORT.md
- evidence/experiments/_shared/browser_runtime/manifest.json
- reports/experiments/_shared/browser_runtime/system/usr/share/doc/fonts-dejavu-core/README.md
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/event_tree_core.py
- code/experiments/zero-shot/_lib/event_tree_runtime.py
- reports/experiments/zero-shot/artifacts/README.md
- code/experiments/_lib/family_catalog.py

## 2026-11-05T12:00:00.001224+09:00

원본 파일 직접 관찰: 18개 실행, 583개 코드·의존성 파일. 이전 commit: b87ffdcf2c7b49ef62e0df11e814dd49804f9a78.

추가 0 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/metadata/status.json

## 2026-11-05T13:27:56.072984+09:00

원본 파일 직접 관찰: 18개 실행, 586개 코드·의존성 파일. 이전 commit: 86fd29c2f18e958605705c2e49124223c3287c9c.

추가 5 / 변경 9 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/event_tree_failure_analysis.py
- code/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/code/reporting/event_tree_failure_analysis.py
- code/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/code/reporting/event_tree_report.py
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/FINDINGS.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/STRATA.md
- code/experiments/zero-shot/_lib/event_tree_check.py
- code/experiments/zero-shot/_lib/event_tree_report.py
- code/experiments/zero-shot/_lib/event_tree_runtime.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/event_tree_search/pilot/900/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/REPORT.md
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-05T15:00:00.002367+09:00

원본 파일 직접 관찰: 18개 실행, 586개 코드·의존성 파일. 이전 commit: d686698b35d4bdf5f5d2022b6de4f3784913811c.

추가 0 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/event_tree_report.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json

## 2026-11-05T17:09:24.870572+09:00

원본 파일 직접 관찰: 18개 실행, 591개 코드·의존성 파일. 이전 commit: 936ac6da4661160b15d450447fb51ededad1f21a.

추가 12 / 변경 27 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/check_fps_family_view.py
- code/experiments/zero-shot/_lib/rollout_method_docs.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metrics/summary.csv
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metrics/summary.json
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/300/AXES.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/300/REPORT.md
- reports/experiments/_shared/math_renderer/README.md
- code/experiments/_shared/math_renderer/katex/LICENSE
- code/experiments/_shared/math_renderer/katex/katex.min.css
- code/experiments/_shared/math_renderer/katex/katex.min.js
- evidence/experiments/_shared/math_renderer/katex/manifest.json
- evidence/backbones/manifests/qwen2_vl_download.json
- code/experiments/zero-shot/_lib/conditional_rollout_budget.py
- code/experiments/zero-shot/_lib/deterministic_rollout300_report.py
- code/experiments/zero-shot/_lib/event_tree_report.py
- code/experiments/zero-shot/_lib/rollout_reconsideration_report.py
- code/experiments/zero-shot/_lib/timelens_rollout_report.py
- evidence/experiments/zero-shot/artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/REPORT.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/SPEC.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/SPEC.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/SPEC.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/SPEC.md
- reports/experiments/zero-shot/results/rollout/stochastic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/overview/REPORT.md
- code/experiments/_lib/family_catalog.py
- code/experiments/_lib/family_view.css
- code/experiments/_lib/family_view.js
- code/experiments/_lib/result_layout.py
- evidence/backbones/manifests/registry.json
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-05T18:00:00.000942+09:00

원본 파일 직접 관찰: 18개 실행, 592개 코드·의존성 파일. 이전 commit: d441acf70452a9f275b788d47c6bce4918d7f0fb.

추가 10 / 변경 4 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/deterministic_oracle_diversity.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/oracle_diversity/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/oracle_diversity/metrics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/oracle_diversity/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/oracle_diversity/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/oracle_diversity/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/oracle_diversity/STRATA.md
- code/experiments/zero-shot/_lib/deterministic_rollout300_report.py
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/REPORT.md

## 2026-11-05T20:12:48.368276+09:00

원본 파일 직접 관찰: 18개 실행, 606개 코드·의존성 파일. 이전 commit: 57c4edb0c562cb722b8f1dcc51a0517927ac6035.

추가 19 / 변경 19 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/rollout_reconsideration_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/rollout_method_docs.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/rollout_reconsideration.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/rollout_reconsideration_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/rollout_reconsideration_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/rollout_reconsideration_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/rollout_reconsideration_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/rollout_reconsideration_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/reporting/rollout_reconsideration_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/reporting/rollout_reconsideration_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/reporting/rollout_reconsideration_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/code/reporting/rollout_reconsideration_viewer_check.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/web/reconsideration.css
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/reconsideration/web/reconsideration.js
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/reconsideration/SPEC.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/reconsideration/STRATA.md
- code/experiments/zero-shot/_lib/deterministic_oracle_diversity.py
- code/experiments/zero-shot/_lib/rollout_method_docs.py
- code/experiments/zero-shot/_lib/rollout_reconsideration.py
- code/experiments/zero-shot/_lib/rollout_reconsideration_check.py
- code/experiments/zero-shot/_lib/rollout_reconsideration_report.py
- reports/experiments/zero-shot/artifacts/README.md
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/reconsideration/code/reporting/rollout_reconsideration_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/reconsideration/code/reporting/rollout_reconsideration_report.py
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/oracle_diversity/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/oracle_diversity/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/SPEC.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/oracle_diversity/STRATA.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/overview/reconsideration/REPORT.md
- conversations/01a19bb6-3014-7f63-b648-622dd1487d58.md

## 2026-11-06T00:00:00.001284+09:00

원본 파일 직접 관찰: 18개 실행, 607개 코드·의존성 파일. 이전 commit: 9e9f2683e5e8624096932246653eae6cdb2ac4e0.

추가 4 / 변경 12 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/rollout_boundary_clipping.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/300/metrics/boundary_clipping/summary.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/600/metrics/boundary_clipping/summary.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/deterministic/pilot/900/metrics/boundary_clipping/summary.json
- code/experiments/zero-shot/_lib/deterministic_oracle_diversity.py
- code/experiments/zero-shot/_lib/deterministic_rollout300_report.py
- code/experiments/zero-shot/_lib/rollout_reconsideration_report.py
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/reconsideration/REPORT.md

## 2026-11-06T17:01:35.503378+09:00

원본 파일 직접 관찰: 18개 실행, 607개 코드·의존성 파일. 이전 commit: 9d27d72af46f5db4a6d9dc77b8c0283618e436a9.

추가 1 / 변경 0 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-06T17:43:00.401668+09:00

원본 파일 직접 관찰: 18개 실행, 622개 코드·의존성 파일. 이전 commit: 3e6a16962a9c90bc3678d026aa8c24549cdbc75c.

추가 22 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_check.py
- code/experiments/zero-shot/_lib/temporal_mode_core.py
- code/experiments/zero-shot/_lib/temporal_mode_report.py
- code/experiments/zero-shot/_lib/temporal_mode_runtime.py
- code/experiments/zero-shot/_lib/temporal_mode_smoke.py
- code/experiments/zero-shot/_lib/temporal_mode_view_check.py
- code/experiments/zero-shot/_lib/temporal_mode_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_smoke.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/SPEC.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/overview/REPORT.md
- reports/experiments/zero-shot/artifacts/README.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-06T18:00:00.002343+09:00

원본 파일 직접 관찰: 18개 실행, 622개 코드·의존성 파일. 이전 commit: a66f1dc8700f50eb2a2e01c89cb02cf9fc173209.

추가 0 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metadata/status.json

## 2026-11-06T18:31:16.917350+09:00

원본 파일 직접 관찰: 18개 실행, 631개 코드·의존성 파일. 이전 commit: 4d039f0182c7a8ae9822eadf26438f85bcdd63e3.

추가 9 / 변경 7 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_parallel.py
- code/experiments/zero-shot/_lib/temporal_mode_parallel_check.py
- code/experiments/zero-shot/_lib/temporal_mode_scope.py
- code/experiments/zero-shot/_lib/temporal_mode_scope_report.py
- code/experiments/zero-shot/_lib/temporal_mode_switch.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_parallel.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_parallel_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_scope.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_scope_report.py
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/SPEC.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-06T19:13:34.881447+09:00

원본 파일 직접 관찰: 18개 실행, 631개 코드·의존성 파일. 이전 commit: c17b20b1e4f96ee18099bfa6cc4f3de14a003012.

추가 0 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metadata/status.json
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-06T20:06:07.268898+09:00

원본 파일 직접 관찰: 18개 실행, 644개 코드·의존성 파일. 이전 commit: 90582dad5cef8cb6bca223b39bd9951d423b0db0.

추가 19 / 변경 5 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_capped_check.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_core.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_report.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_runtime.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_smoke.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_smoke.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_view_check.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/SPEC.md
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/metadata/status.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/REPORT.md
- code/experiments/_lib/family_catalog.py
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-06T20:47:31.863588+09:00

원본 파일 직접 관찰: 18개 실행, 654개 코드·의존성 파일. 이전 commit: 49bc8024cdccc7061910975ebc7735b9447ec748.

추가 10 / 변경 8 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_capped_cost.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_cost_check.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_cost_report.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_cost_runtime.py
- code/experiments/zero-shot/_lib/temporal_mode_capped_cost_smoke.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost_smoke.py
- reports/experiments/zero-shot/README.md
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/SPEC.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-06T21:00:00.001291+09:00

원본 파일 직접 관찰: 18개 실행, 654개 코드·의존성 파일. 이전 commit: cc766456e491109588c2afc75266f945a9db88ac.

추가 0 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json

## 2026-11-06T23:59:43.302798+09:00

원본 파일 직접 관찰: 18개 실행, 666개 코드·의존성 파일. 이전 commit: d343d7a380553f3db8e25025be803cac62e35184.

추가 12 / 변경 8 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_mps_benchmark.py
- code/experiments/zero-shot/_lib/temporal_mode_mps_check.py
- code/experiments/zero-shot/_lib/temporal_mode_prefix_cache.py
- code/experiments/zero-shot/_lib/temporal_mode_prefix_cache_check.py
- code/experiments/zero-shot/_lib/temporal_mode_prefix_cache_report.py
- code/experiments/zero-shot/_lib/temporal_mode_prefix_cache_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_mps_benchmark.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_mps_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache_runtime.py
- reports/experiments/zero-shot/README.md
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/SPEC.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-07T00:00:00.001136+09:00

원본 파일 직접 관찰: 18개 실행, 666개 코드·의존성 파일. 이전 commit: 822d24fb77f326de833d79cf8d3c0a4d8e32c3fc.

추가 0 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json

## 2026-11-07T00:57:11.530197+09:00

원본 파일 직접 관찰: 18개 실행, 666개 코드·의존성 파일. 이전 commit: 2cdb9edcbede45d4ea394797c1d1d0ba30d38422.

추가 0 / 변경 4 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-07T03:29:20.132248+09:00

원본 파일 직접 관찰: 18개 실행, 666개 코드·의존성 파일. 이전 commit: f539663f26cd34906285450551099dee0d15ebd3.

추가 0 / 변경 7 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_prefix_cache_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache_report.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/SPEC.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-07T04:16:10.873379+09:00

원본 파일 직접 관찰: 18개 실행, 671개 코드·의존성 파일. 이전 commit: 7754427eecee565d2ddc4e49e052ea20eb20bcbe.

추가 5 / 변경 6 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_ground_truth.js
- code/experiments/zero-shot/_lib/temporal_mode_ground_truth_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_ground_truth_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/_lib/temporal_mode_prefix_cache_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache_report.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-07T05:09:50.155183+09:00

원본 파일 직접 관찰: 18개 실행, 683개 코드·의존성 파일. 이전 commit: e9540ca157637755f8d7e69097e4be65313bd540.

추가 12 / 변경 6 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_continue_pilot.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_check.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_report.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_runtime.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_scope.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_continue_pilot.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_scope.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_view_check.py
- reports/experiments/zero-shot/README.md
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-07T09:00:00.001103+09:00

원본 파일 직접 관찰: 18개 실행, 685개 코드·의존성 파일. 이전 commit: 9e0dd474bab7020b31f68905e69ccaca7bdab503.

추가 33 / 변경 4 / 현재 원본에서 사라짐 27. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_cost.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_cost_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_cost_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_cost_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_cost_smoke.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_smoke.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_continue_pilot.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_scope.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_ground_truth_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_mps_benchmark.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_mps_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_prefix_cache.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_prefix_cache_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_prefix_cache_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_prefix_cache_runtime.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/SPEC.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_core.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost_runtime.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_cost_smoke.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_runtime.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_smoke.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_capped_view_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_continue_pilot.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_runtime.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_scope.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_cumulative_view_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_ground_truth.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_ground_truth_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_mps_benchmark.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_mps_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/code/temporal_mode_prefix_cache_runtime.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/manifest.json

## 2026-11-07T17:53:08.765130+09:00

원본 파일 직접 관찰: 18개 실행, 687개 코드·의존성 파일. 이전 commit: 3ab89e3c5c3f3de9e4bd5e77ef8fdbefe3503bb9.

추가 7 / 변경 21 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/metrics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/300/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/300/capped_rollouts/SPEC.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/overview/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/temporal_mode_continue_pilot.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_check.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_report.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_runtime.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_scope.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_view_check.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_continue_pilot.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_scope.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_view_check.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/SPEC.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/SPEC.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-07T18:00:00.000660+09:00

원본 파일 직접 관찰: 18개 실행, 687개 코드·의존성 파일. 이전 commit: 4142575564b73cce0955c211cdb1951dc3f58d3b.

추가 0 / 변경 1 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/status.json

## 2026-11-07T18:57:08.497879+09:00

원본 파일 직접 관찰: 18개 실행, 687개 코드·의존성 파일. 이전 commit: 911a3ca45996bdf6ed106469c38044e74f790d96.

추가 1 / 변경 8 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- reports/experiments/zero-shot/results/rollout/temporal_mode_search/shared/design_principles/REPORT.md
- reports/experiments/zero-shot/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/overview/capped_rollouts/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-07T21:00:00.000270+09:00

원본 파일 직접 관찰: 18개 실행, 687개 코드·의존성 파일. 이전 commit: be07719c5e39da564e13a2a4551f40fc05c5c53b.

추가 0 / 변경 6 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/overview/capped_rollouts/REPORT.md

## 2026-11-08T00:00:00.002815+09:00

원본 파일 직접 관찰: 18개 실행, 687개 코드·의존성 파일. 이전 commit: 9c23b9d301ffa53a6c8cfdda81c382894ab39ef8.

추가 0 / 변경 6 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metrics/summary.json
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/overview/capped_rollouts/REPORT.md

## 2026-11-08T02:32:02.050208+09:00

원본 파일 직접 관찰: 18개 실행, 698개 코드·의존성 파일. 이전 commit: 07b6c8ce9ef50657788b9b5f2f822c76ebc1f40a.

추가 11 / 변경 16 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_case_presentation.py
- code/experiments/zero-shot/_lib/temporal_mode_case_view_check.py
- code/experiments/zero-shot/_lib/temporal_mode_case_viewer.css
- code/experiments/zero-shot/_lib/temporal_mode_case_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/web/case_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/case_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_case_presentation.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_case_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_case_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_case_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/web/case_viewer.css
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_report.py
- code/experiments/zero-shot/_lib/temporal_mode_ground_truth.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_cumulative_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/300/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/shared/design_principles/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-08T05:14:15.909194+09:00

원본 파일 직접 관찰: 18개 실행, 698개 코드·의존성 파일. 이전 commit: 81ceabdc6a8770ad416e849d6d778e7affed76ea.

추가 0 / 변경 27 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/temporal_mode_case_presentation.py
- code/experiments/zero-shot/_lib/temporal_mode_case_view_check.py
- code/experiments/zero-shot/_lib/temporal_mode_case_viewer.css
- code/experiments/zero-shot/_lib/temporal_mode_case_viewer.js
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_report.py
- code/experiments/zero-shot/_lib/temporal_mode_cumulative_view_check.py
- code/experiments/zero-shot/_lib/temporal_mode_ground_truth.js
- code/experiments/zero-shot/_lib/temporal_mode_prefix_cache_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/web/case_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/300/capped_rollouts/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/case_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts/web/viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/web/case_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/web/ground_truth.js
- code/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/web/viewer.js
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/300/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/300/capped_rollouts/SPEC.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/SPEC.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/SPEC.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/overview/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/shared/design_principles/REPORT.md
- code/experiments/_lib/build_index.py
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-09T02:42:29.182581+09:00

원본 파일 직접 관찰: 18개 실행, 719개 코드·의존성 파일. 이전 commit: 487d4b6d308cbfd66eb7ba9fccf91beaf2254003.

추가 29 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/interval_distribution_check.py
- code/experiments/zero-shot/_lib/interval_distribution_core.py
- code/experiments/zero-shot/_lib/interval_distribution_launch.py
- code/experiments/zero-shot/_lib/interval_distribution_prepare.py
- code/experiments/zero-shot/_lib/interval_distribution_report.py
- code/experiments/zero-shot/_lib/interval_distribution_runtime.py
- code/experiments/zero-shot/_lib/interval_distribution_semantic.py
- code/experiments/zero-shot/_lib/interval_distribution_verify.py
- code/experiments/zero-shot/_lib/interval_distribution_view_check.py
- code/experiments/zero-shot/_lib/interval_distribution_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/300/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_launch.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_prepare.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_semantic.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_verify.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/web/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/web/viewer.js
- reports/experiments/zero-shot/results/rollout/interval_distribution/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/interval_distribution/pilot/300/SPEC.md
- reports/experiments/zero-shot/results/rollout/interval_distribution/pilot/overview/REPORT.md
- reports/experiments/zero-shot/README.md
- reports/experiments/zero-shot/artifacts/README.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-09T06:06:24.528536+09:00

원본 파일 직접 관찰: 18개 실행, 732개 코드·의존성 파일. 이전 commit: e7ae862df4d3804e26f144354adf405de15f961f.

추가 46 / 변경 3 / 현재 원본에서 사라짐 29. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/path_mass_core.py
- code/experiments/zero-shot/_lib/path_mass_inputs.py
- code/experiments/zero-shot/_lib/path_mass_report.py
- code/experiments/zero-shot/_lib/path_mass_runtime.py
- code/experiments/zero-shot/_lib/path_mass_validate.py
- code/experiments/zero-shot/_lib/path_mass_viewer.js
- code/experiments/zero-shot/_lib/probability_budget_backend.py
- code/experiments/zero-shot/_lib/probability_budget_checks.py
- code/experiments/zero-shot/_lib/probability_budget_core.py
- code/experiments/zero-shot/_lib/probability_budget_inputs.py
- code/experiments/zero-shot/_lib/probability_budget_manage.py
- code/experiments/zero-shot/_lib/probability_budget_report.py
- code/experiments/zero-shot/_lib/probability_budget_runtime.py
- code/experiments/zero-shot/_lib/probability_budget_search.py
- code/experiments/zero-shot/_lib/probability_budget_view_check.py
- code/experiments/zero-shot/_lib/probability_budget_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_inputs.py
- code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_validate.py
- code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/web/viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_backend.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_checks.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_inputs.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_manage.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_search.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/web/viewer.js
- reports/experiments/zero-shot/results/rollout/path_mass_constraints/pilot/3/REPORT.md
- reports/experiments/zero-shot/results/rollout/path_mass_constraints/pilot/overview/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/SPEC.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/overview/REPORT.md
- reports/experiments/zero-shot/README.md
- reports/experiments/zero-shot/artifacts/README.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_check.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_core.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_launch.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_prepare.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_report.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_runtime.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_semantic.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_verify.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_view_check.py
- 원본 부재: code/experiments/zero-shot/_lib/interval_distribution_viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/300/metadata/status.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_core.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_launch.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_prepare.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_runtime.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_semantic.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_verify.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_view_check.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/code/interval_distribution_viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/metadata/manifest.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/web/summary.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/interval_distribution/pilot/900/web/viewer.js
- 원본 부재: reports/experiments/zero-shot/results/rollout/interval_distribution/pilot/300/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/interval_distribution/pilot/300/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/interval_distribution/pilot/overview/REPORT.md

## 2026-11-09T09:00:00.002266+09:00

원본 파일 직접 관찰: 18개 실행, 732개 코드·의존성 파일. 이전 commit: ec0bfd2cde688906c654c6b73a1274ce27dec2a6.

추가 0 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md

## 2026-11-09T11:30:51.211277+09:00

원본 파일 직접 관찰: 18개 실행, 740개 코드·의존성 파일. 이전 commit: 358f3d97d61e9288bee2c5490849b1d38802a951.

추가 8 / 변경 16 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/probability_budget_capture.py
- code/experiments/zero-shot/_lib/probability_budget_recording_handoff.py
- code/experiments/zero-shot/_lib/probability_budget_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_capture.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_recording_handoff.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/validation/search_before_recording.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/web/viewer.css
- code/experiments/zero-shot/_lib/probability_budget_report.py
- code/experiments/zero-shot/_lib/probability_budget_runtime.py
- code/experiments/zero-shot/_lib/probability_budget_search.py
- code/experiments/zero-shot/_lib/probability_budget_view_check.py
- code/experiments/zero-shot/_lib/probability_budget_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_search.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/web/viewer.js
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-09T12:00:00.002277+09:00

원본 파일 직접 관찰: 18개 실행, 740개 코드·의존성 파일. 이전 commit: bed7c977f1fc0a4b973d1ddee530ea2510b5f525.

추가 0 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md

## 2026-11-09T15:00:00.001307+09:00

원본 파일 직접 관찰: 18개 실행, 740개 코드·의존성 파일. 이전 commit: 206978979ef24134d721979889ca58ffbbbc84f4.

추가 2 / 변경 4 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/SPEC.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/overview/REPORT.md

## 2026-11-09T18:00:00.001301+09:00

원본 파일 직접 관찰: 18개 실행, 740개 코드·의존성 파일. 이전 commit: 6ca23fbad0a7d0acf0bc33031502b252b0e40247.

추가 0 / 변경 5 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/overview/REPORT.md

## 2026-11-09T21:00:00.000876+09:00

원본 파일 직접 관찰: 18개 실행, 730개 코드·의존성 파일. 이전 commit: fc331427a0788ab41bb8136fdb6e9805daa4f42c.

추가 4 / 변경 17 / 현재 원본에서 사라짐 18. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/probability_completion_core.py
- code/experiments/zero-shot/_lib/probability_completion_models.py
- code/experiments/zero-shot/_lib/probability_completion_runtime.py
- conversations/01a1b790-f2cb-7ce0-a157-efb9097a01e1.md
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/probability_budget_report.py
- code/experiments/zero-shot/_lib/probability_budget_viewer.js
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/web/viewer.js
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/SPEC.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/SPEC.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/overview/REPORT.md
- code/experiments/_lib/build_index.py
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md
- 원본 부재: code/experiments/zero-shot/_lib/path_mass_core.py
- 원본 부재: code/experiments/zero-shot/_lib/path_mass_inputs.py
- 원본 부재: code/experiments/zero-shot/_lib/path_mass_report.py
- 원본 부재: code/experiments/zero-shot/_lib/path_mass_runtime.py
- 원본 부재: code/experiments/zero-shot/_lib/path_mass_validate.py
- 원본 부재: code/experiments/zero-shot/_lib/path_mass_viewer.js
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_core.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_inputs.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_report.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_runtime.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_validate.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/code/path_mass_viewer.js
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/metadata/manifest.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/metadata/status.json
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/path_mass_constraints/pilot/3/web/viewer.js
- 원본 부재: reports/experiments/zero-shot/results/rollout/path_mass_constraints/pilot/3/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/path_mass_constraints/pilot/overview/REPORT.md

## 2026-11-10T00:00:00.000316+09:00

원본 파일 직접 관찰: 18개 실행, 745개 코드·의존성 파일. 이전 commit: 8bceb0326735478aab4d706183a0f16721f30c5b.

추가 20 / 변경 18 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/probability_completion_manage.py
- code/experiments/zero-shot/_lib/probability_completion_report.py
- code/experiments/zero-shot/_lib/probability_completion_view_check.py
- code/experiments/zero-shot/_lib/probability_completion_viewer.css
- code/experiments/zero-shot/_lib/probability_completion_viewer.js
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_manage.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_models.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_view_check.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/config.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/web/viewer.css
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/web/viewer.js
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/end_to_end/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/end_to_end/SPEC.md
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/probability_budget_report.py
- code/experiments/zero-shot/_lib/probability_budget_viewer.js
- code/experiments/zero-shot/_lib/probability_completion_models.py
- code/experiments/zero-shot/_lib/probability_completion_runtime.py
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/web/viewer.js
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/SPEC.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/SPEC.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/overview/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-10T00:46:34.383952+09:00

원본 파일 직접 관찰: 18개 실행, 745개 코드·의존성 파일. 이전 commit: c7e48ad62d7f502ea3982e219ebc672759d3268f.

추가 0 / 변경 20 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/probability_completion_core.py
- code/experiments/zero-shot/_lib/probability_completion_models.py
- code/experiments/zero-shot/_lib/probability_completion_report.py
- code/experiments/zero-shot/_lib/probability_completion_runtime.py
- code/experiments/zero-shot/_lib/probability_completion_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_core.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_models.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_runtime.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/web/viewer.js
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/end_to_end/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/end_to_end/SPEC.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/overview/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-10T09:00:00.004296+09:00

원본 파일 직접 관찰: 18개 실행, 745개 코드·의존성 파일. 이전 commit: a7f8f22b13160d9c30f5dc98adec0a44ae3d0024.

추가 2 / 변경 6 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/SPEC.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/end_to_end/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/overview/REPORT.md

## 2026-11-10T12:00:00.001288+09:00

원본 파일 직접 관찰: 18개 실행, 745개 코드·의존성 파일. 이전 commit: b42acfc5de78c0fd553c9fa8b8e77657c66a4b0c.

추가 0 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md

## 2026-11-10T13:14:24.927025+09:00

원본 파일 직접 관찰: 18개 실행, 745개 코드·의존성 파일. 이전 commit: 0fd3195d636fd833de0f74572b6483d66839dffe.

추가 0 / 변경 7 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/probability_completion_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_report.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/end_to_end/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-10T15:00:00.004726+09:00

원본 파일 직접 관찰: 18개 실행, 745개 코드·의존성 파일. 이전 commit: 84b5c0b0b58f7d8f449b1a695e7b3653efdd7a65.

추가 0 / 변경 2 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md

## 2026-11-10T17:08:47.486511+09:00

원본 파일 직접 관찰: 18개 실행, 749개 코드·의존성 파일. 이전 commit: ea23ed6a36b6076db0856fd541d7970662b0aa2c.

추가 45 / 변경 49 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/probability_compute_appendix.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/300/compute_cost/statistics/summary.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/compute_cost/statistics/summary.json
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_compute_appendix.py
- reports/experiments/zero-shot/results/baseline/all_zeroshot/overview/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/oracle_diversity/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/reconsideration/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/oracle_diversity/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/oracle_diversity/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/reconsideration/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/300/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/600/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/APPENDIX_COMPUTE.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/end_to_end/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/APPENDIX_COMPUTE.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/stochastic/pilot/300/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/300/capped_rollouts/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/shared/design_principles/APPENDIX.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/300/APPENDIX.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/900/APPENDIX.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/300/APPENDIX.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/600/APPENDIX.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/900/APPENDIX.md
- reports/experiments/zero-shot/results/uncertainty/span_head/pilot/300/APPENDIX.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/APPENDIX.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/APPENDIX.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/head_analysis/APPENDIX.md
- reports/experiments/analysis/failure_mode_timestamp/failure_cases/results/failure_modes/input_settings/pilot/200/APPENDIX.md
- reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/300/APPENDIX.md
- reports/experiments/similarity/results/frame_query/cosine_2fps/full/9404/APPENDIX.md
- reports/experiments/similarity/results/frame_query/cosine_2fps/pilot/300/APPENDIX.md
- code/experiments/_lib/check_report_appendices.py
- code/experiments/_lib/report_appendices.py
- code/experiments/zero-shot/_lib/probability_budget_report.py
- code/experiments/zero-shot/_lib/probability_completion_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/code/probability_budget_report.py
- code/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/code/probability_completion_report.py
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/manifest.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/metadata/manifest.json
- reports/experiments/zero-shot/results/baseline/all_zeroshot/overview/REPORT.md
- reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/300/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/600/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/oracle_diversity/REPORT.md
- reports/experiments/zero-shot/results/rollout/deterministic/pilot/900/reconsideration/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/event_tree_search/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/300/end_to_end/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md
- reports/experiments/zero-shot/results/rollout/stochastic/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/300/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/pilot/900/capped_rollouts/REPORT.md
- reports/experiments/zero-shot/results/rollout/temporal_mode_search/shared/design_principles/REPORT.md
- reports/experiments/zero-shot/results/sampling/fps_spatial_control/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/allocation_sweep/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/600/REPORT.md
- reports/experiments/zero-shot/results/sampling/spatial_allocation/budget_comparison/pilot/900/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/span_head/pilot/300/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/digit_correctness/blue_analysis/REPORT.md
- reports/experiments/zero-shot/results/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis/REPORT.md
- reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/head_analysis/REPORT.md
- reports/experiments/analysis/failure_mode_timestamp/failure_cases/results/failure_modes/input_settings/pilot/200/REPORT.md
- reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/300/REPORT.md
- reports/experiments/similarity/results/frame_query/cosine_2fps/full/9404/REPORT.md
- reports/experiments/similarity/results/frame_query/cosine_2fps/pilot/300/REPORT.md
- reports/experiments/_lib/README.md
- code/experiments/_lib/build_index.py
- code/experiments/_lib/family_catalog.py
- code/experiments/_lib/family_view.js
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md

## 2026-11-10T18:00:00.004562+09:00

원본 파일 직접 관찰: 18개 실행, 749개 코드·의존성 파일. 이전 commit: 0f5331fec5dd50b54da022ca1d9f74dcc37f33e2.

추가 0 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md

## 2026-11-10T19:41:58.329071+09:00

원본 파일 직접 관찰: 18개 실행, 744개 코드·의존성 파일. 이전 commit: 1600526368cafe4f9f9e572a5eaab1810380642c.

추가 1 / 변경 14 / 현재 원본에서 사라짐 16. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- code/experiments/zero-shot/_lib/rollout_cache.py
- reports/experiments/zero-shot/README.md
- code/experiments/zero-shot/_lib/deterministic_rollout300.py
- code/experiments/zero-shot/_lib/rollout_method_docs.py
- code/experiments/zero-shot/_lib/rollout_reconsideration.py
- code/experiments/zero-shot/_lib/rollout_token_distributions.py
- reports/experiments/zero-shot/artifacts/README.md
- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md
- code/experiments/_lib/build_index.py
- code/experiments/_lib/family_catalog.py
- code/experiments/_lib/organize_completed_rollouts.py
- code/experiments/_lib/report_appendices.py
- conversations/01a1af6d-1894-79e1-9d9c-9a38a6397610.md
- 원본 부재: code/experiments/zero-shot/_lib/conditional_rollout_analysis.py
- 원본 부재: code/experiments/zero-shot/_lib/conditional_rollout_budget.py
- 원본 부재: code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/conditional_rollout_budget.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/code/conditional_rollout_analysis.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/code/conditional_rollout_budget.py
- 원본 부재: code/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/code/fps_spatial_control.py
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/metadata/config.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/metadata/status.json
- 원본 부재: evidence/experiments/zero-shot/artifacts/runs/rollout/conditional_budget/pilot/30/metrics/budget.csv
- 원본 부재: reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/ANALYSIS.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/APPENDIX.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/NEXT_EXPERIMENT.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/REPORT.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/SPEC.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/30/STRATA.md
- 원본 부재: reports/experiments/zero-shot/results/rollout/conditional_budget/pilot/overview/REPORT.md

## 2026-11-10T21:00:00.002321+09:00

원본 파일 직접 관찰: 18개 실행, 744개 코드·의존성 파일. 이전 commit: 13c0b3367b790be5e049709e6c940f71ef3f6c0a.

추가 0 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md

## 2026-11-11T00:00:00.004269+09:00

원본 파일 직접 관찰: 18개 실행, 744개 코드·의존성 파일. 이전 commit: 266ddc72da4fcbe92f0103e9b7378b2174b52c1c.

추가 0 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md

## 2026-11-11T09:00:00.004628+09:00

원본 파일 직접 관찰: 18개 실행, 744개 코드·의존성 파일. 이전 commit: e167228bb56f02cc3cde2d972bf19e5cf5e8349d.

추가 0 / 변경 3 / 현재 원본에서 사라짐 0. 원본 보고서의 실험 시각과 이번 수집 시각은 구분한다.

- evidence/experiments/zero-shot/artifacts/runs/rollout/probability_budget/pilot/900/end_to_end/metadata/status.json
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/APPENDIX.md
- reports/experiments/zero-shot/results/rollout/probability_budget/pilot/900/end_to_end/REPORT.md
