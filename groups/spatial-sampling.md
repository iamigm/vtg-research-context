# 공간 해상도 통제와 Temporal Sampling

Nested300⊂600⊂900, 데이터셋별300 query. 공간통제v1은 두공간목표50,176/100,352픽셀 ×4조건=7,200논리응답. 후속v2는 A/B ×두공간목표 ×19조건 ×900=68,400응답 목표. 완료 전 명세·임시보고서를 최종결과로 간주하지 않는다. 11월2일 폐기된 공간 비통제10실험의 결과는 제외한다. 남은 과거 구현은 보존하되 재실행·결과복원하지 않는다. GT oracle 결과와 실제 추론 알고리즘을 구분한다.

## 실행별 보고서·관련 코드

### zero-shot/fps_spatial_control_pilot300_v1

[SPATIAL_AUDIT.md](../reports/experiments/zero-shot/results/fps_spatial_control_pilot300_v1/SPATIAL_AUDIT.md) · [SPEC.md](../reports/experiments/zero-shot/results/fps_spatial_control_pilot300_v1/SPEC.md)

현재 관련 코드 72개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/selected_frame_cache_migration_v1

[REPORT.md](../reports/experiments/zero-shot/results/selected_frame_cache_migration_v1/REPORT.md)

현재 관련 코드 72개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/spatial_budget_pilot900_v1

[PILOT_300.md](../reports/experiments/zero-shot/results/spatial_budget_pilot900_v1/PILOT_300.md) · [PILOT_600.md](../reports/experiments/zero-shot/results/spatial_budget_pilot900_v1/PILOT_600.md) · [REPORT.md](../reports/experiments/zero-shot/results/spatial_budget_pilot900_v1/REPORT.md) · [SPEC.md](../reports/experiments/zero-shot/results/spatial_budget_pilot900_v1/SPEC.md)

현재 관련 코드 72개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/spatial_sampling_pilot900_v2

[REPORT.md](../reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/REPORT.md) · [SPEC.md](../reports/experiments/zero-shot/results/spatial_sampling_pilot900_v2/SPEC.md)

현재 관련 코드 72개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

## 관련 실험 대화

- [# Files pasted by the user:](../conversations/01a18c45-0978-7563-9999-067f29e4426e.md) — active
- [주요 데이터셋 메타데이터 통계 정리](../conversations/01a192da-87cd-7582-8f38-d022b7d79a99.md) — deleted
- [Charades-STA, ActivityNet, QVHilights 를 사용하는 기존 코드를 참고하여 fps, 이미지 해상도 등을 어떻게 처리하는지 봐봐. 웬만하면 최신, 검증된 학회 논문에서 공개한 코드로 조사해](../conversations/01a19350-4a1d-7572-b26c-ae796896677a.md) — archived
## 코드 백업

아래는 현재 구현의 연결이다. 역사적 실행 revision의 증명은 아니다. 공용 dependency는 [shared](shared.md)를 참조한다.

- [zero-shot/_lib/adaptive_temporal_finalize.py](../code/experiments/zero-shot/_lib/adaptive_temporal_finalize.py)
- [zero-shot/_lib/adaptive_temporal_prepare.py](../code/experiments/zero-shot/_lib/adaptive_temporal_prepare.py)
- [zero-shot/_lib/adaptive_temporal_provenance.py](../code/experiments/zero-shot/_lib/adaptive_temporal_provenance.py)
- [zero-shot/_lib/adaptive_temporal_report.py](../code/experiments/zero-shot/_lib/adaptive_temporal_report.py)
- [zero-shot/_lib/adaptive_temporal_sampling.py](../code/experiments/zero-shot/_lib/adaptive_temporal_sampling.py)
- [zero-shot/_lib/adaptive_temporal_validate.py](../code/experiments/zero-shot/_lib/adaptive_temporal_validate.py)
- [zero-shot/_lib/boundary_oracle_analysis.py](../code/experiments/zero-shot/_lib/boundary_oracle_analysis.py)
- [zero-shot/_lib/boundary_oracle_inputs.py](../code/experiments/zero-shot/_lib/boundary_oracle_inputs.py)
- [zero-shot/_lib/boundary_oracle_report.py](../code/experiments/zero-shot/_lib/boundary_oracle_report.py)
- [zero-shot/_lib/boundary_oracle_sampling.py](../code/experiments/zero-shot/_lib/boundary_oracle_sampling.py)
- [zero-shot/_lib/build_nested_pilots.py](../code/experiments/zero-shot/_lib/build_nested_pilots.py)
- [zero-shot/_lib/compact_selected_frames.py](../code/experiments/zero-shot/_lib/compact_selected_frames.py)
- [zero-shot/_lib/fps_spatial_control.py](../code/experiments/zero-shot/_lib/fps_spatial_control.py)
- [zero-shot/_lib/fps_spatial_report.py](../code/experiments/zero-shot/_lib/fps_spatial_report.py)
- [zero-shot/_lib/fps_spatial_smoke.py](../code/experiments/zero-shot/_lib/fps_spatial_smoke.py)
- [zero-shot/_lib/fps_spatial_supervisor.py](../code/experiments/zero-shot/_lib/fps_spatial_supervisor.py)
- [zero-shot/_lib/gt_actual_density.py](../code/experiments/zero-shot/_lib/gt_actual_density.py)
- [zero-shot/_lib/gt_actual_density_report.py](../code/experiments/zero-shot/_lib/gt_actual_density_report.py)
- [zero-shot/_lib/gt_dataset_diagnostics.py](../code/experiments/zero-shot/_lib/gt_dataset_diagnostics.py)
- [zero-shot/_lib/gt_density_batch.py](../code/experiments/zero-shot/_lib/gt_density_batch.py)
- [zero-shot/_lib/gt_density_combined.py](../code/experiments/zero-shot/_lib/gt_density_combined.py)
- [zero-shot/_lib/gt_density_pilot.py](../code/experiments/zero-shot/_lib/gt_density_pilot.py)
- [zero-shot/_lib/gt_density_prefetch.py](../code/experiments/zero-shot/_lib/gt_density_prefetch.py)
- [zero-shot/_lib/gt_density_queue.py](../code/experiments/zero-shot/_lib/gt_density_queue.py)
- [zero-shot/_lib/gt_density_sweep.py](../code/experiments/zero-shot/_lib/gt_density_sweep.py)
- [zero-shot/_lib/gt_density_sweep_report.py](../code/experiments/zero-shot/_lib/gt_density_sweep_report.py)
- [zero-shot/_lib/gt_fraction_comparison.py](../code/experiments/zero-shot/_lib/gt_fraction_comparison.py)
- [zero-shot/_lib/gt_threshold_report.py](../code/experiments/zero-shot/_lib/gt_threshold_report.py)
- [zero-shot/_lib/gt_video_fraction_axes.py](../code/experiments/zero-shot/_lib/gt_video_fraction_axes.py)
- [zero-shot/_lib/pilot900_actual.py](../code/experiments/zero-shot/_lib/pilot900_actual.py)
- [zero-shot/_lib/pilot900_density_report.py](../code/experiments/zero-shot/_lib/pilot900_density_report.py)
- [zero-shot/_lib/pilot900_extent.py](../code/experiments/zero-shot/_lib/pilot900_extent.py)
- [zero-shot/_lib/pilot900_inputs.py](../code/experiments/zero-shot/_lib/pilot900_inputs.py)
- [zero-shot/_lib/pilot900_overview.py](../code/experiments/zero-shot/_lib/pilot900_overview.py)
- [zero-shot/_lib/pilot900_prefetch.py](../code/experiments/zero-shot/_lib/pilot900_prefetch.py)
- [zero-shot/_lib/pilot900_report_tables.py](../code/experiments/zero-shot/_lib/pilot900_report_tables.py)
- [zero-shot/_lib/pilot900_supervisor.py](../code/experiments/zero-shot/_lib/pilot900_supervisor.py)
- [zero-shot/_lib/shared_selected_frames.py](../code/experiments/zero-shot/_lib/shared_selected_frames.py)
- [zero-shot/_lib/spatial_budget_pilot900.py](../code/experiments/zero-shot/_lib/spatial_budget_pilot900.py)
- [zero-shot/_lib/spatial_budget_recovery_check.py](../code/experiments/zero-shot/_lib/spatial_budget_recovery_check.py)
- [zero-shot/_lib/spatial_budget_report.py](../code/experiments/zero-shot/_lib/spatial_budget_report.py)
- [zero-shot/_lib/spatial_budget_runtime.py](../code/experiments/zero-shot/_lib/spatial_budget_runtime.py)
- [zero-shot/_lib/spatial_budget_supervisor.py](../code/experiments/zero-shot/_lib/spatial_budget_supervisor.py)
- [zero-shot/_lib/spatial_sweep900.py](../code/experiments/zero-shot/_lib/spatial_sweep900.py)
- [zero-shot/_lib/spatial_sweep_inputs.py](../code/experiments/zero-shot/_lib/spatial_sweep_inputs.py)
- [zero-shot/_lib/spatial_sweep_parallel_supervisor.py](../code/experiments/zero-shot/_lib/spatial_sweep_parallel_supervisor.py)
- [zero-shot/_lib/spatial_sweep_parallel_worker.py](../code/experiments/zero-shot/_lib/spatial_sweep_parallel_worker.py)
- [zero-shot/_lib/spatial_sweep_report.py](../code/experiments/zero-shot/_lib/spatial_sweep_report.py)
- [zero-shot/_lib/spatial_sweep_report_s42.py](../code/experiments/zero-shot/_lib/spatial_sweep_report_s42.py)
- [zero-shot/_lib/spatial_sweep_runtime.py](../code/experiments/zero-shot/_lib/spatial_sweep_runtime.py)
- [zero-shot/_lib/spatial_sweep_supervisor.py](../code/experiments/zero-shot/_lib/spatial_sweep_supervisor.py)
- [zero-shot/_lib/wait_for_cache_cleanup.py](../code/experiments/zero-shot/_lib/wait_for_cache_cleanup.py)
- [zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_control.py](../code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_control.py)
- [zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_report.py](../code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_report.py)
- [zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_smoke.py](../code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_smoke.py)
- [zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_supervisor.py](../code/experiments/zero-shot/artifacts/runs/fps_spatial_control_pilot300_v1/code/fps_spatial_supervisor.py)
- [zero-shot/artifacts/runs/selected_frame_cache_migration_v1/code/boundary_oracle_inputs.py](../code/experiments/zero-shot/artifacts/runs/selected_frame_cache_migration_v1/code/boundary_oracle_inputs.py)
- [zero-shot/artifacts/runs/selected_frame_cache_migration_v1/code/pilot900_inputs.py](../code/experiments/zero-shot/artifacts/runs/selected_frame_cache_migration_v1/code/pilot900_inputs.py)
- [zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/gt_density_batch.py](../code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/gt_density_batch.py)
- [zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/pilot900_inputs.py](../code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/pilot900_inputs.py)
- [zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/shared_selected_frames.py](../code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/shared_selected_frames.py)
- [zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_before_overlap.py](../code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_before_overlap.py)
- [zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_pilot900.py](../code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_pilot900.py)
- [zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_report.py](../code/experiments/zero-shot/artifacts/runs/spatial_budget_pilot900_v1/code/spatial_budget_report.py)
- [zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/gt_density_batch.py](../code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/gt_density_batch.py)
- [zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/memory_fix_spatial_sweep_runtime.py](../code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/memory_fix_spatial_sweep_runtime.py)
- [zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/memory_fix_spatial_sweep_supervisor.py](../code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/memory_fix_spatial_sweep_supervisor.py)
- [zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/pilot900_inputs.py](../code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/pilot900_inputs.py)
- [zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/runtime_before_memory_fix.py](../code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/runtime_before_memory_fix.py)
- [zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep900.py](../code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep900.py)
- [zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep_inputs.py](../code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep_inputs.py)
- [zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep_report.py](../code/experiments/zero-shot/artifacts/runs/spatial_sampling_pilot900_v2/code/spatial_sweep_report.py)
