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
