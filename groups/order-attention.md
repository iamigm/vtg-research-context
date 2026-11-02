# 입력 순서와 Attention

영상→질문 / 질문→영상 성능·예측 일관성과 query/generation/hidden/spatial/dynamics attention. MTLA pilot과 TimeLens 전체평가를 구분하며 계측량과 성능의 인과를 단정하지 않는다.

## 실행별 보고서·관련 코드

### attention/dynamics

[FINDINGS.md](../reports/experiments/attention/results/dynamics/FINDINGS.md)

현재 관련 코드 68개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### attention/generation

[FINDINGS.md](../reports/experiments/attention/results/generation/FINDINGS.md) · [REPORT.md](../reports/experiments/attention/results/generation/distribution_analysis/REPORT.md) · [REPORT.md](../reports/experiments/attention/results/generation/head_analysis/REPORT.md)

현재 관련 코드 68개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### attention/overview

[REPORT.md](../reports/experiments/attention/results/overview/REPORT.md)

현재 관련 코드 68개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### attention/query

[FINDINGS.md](../reports/experiments/attention/results/query/FINDINGS.md)

현재 관련 코드 68개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### attention/timelens_order

[final_report.md](../reports/experiments/attention/results/timelens_order/final_report.md)

현재 관련 코드 68개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/input_order

[REPORT.md](../reports/experiments/zero-shot/results/input_order/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/input_order/consistency_v1/REPORT.md)

현재 관련 코드 73개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

## 코드 백업

아래는 현재 구현의 연결이다. 역사적 실행 revision의 증명은 아니다. 공용 dependency는 [shared](shared.md)를 참조한다.

- [zero-shot/_lib/zeroshot_metrics_png.py](../code/experiments/zero-shot/_lib/zeroshot_metrics_png.py)
- [zero-shot/_lib/zeroshot_report.py](../code/experiments/zero-shot/_lib/zeroshot_report.py)
- [zero-shot/artifacts/runs/input_order/consistency_v1/code/analyze.py](../code/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/code/analyze.py)
- [zero-shot/artifacts/runs/input_order/consistency_v1/code/build_viewer.py](../code/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/code/build_viewer.py)
- [zero-shot/artifacts/runs/input_order/consistency_v1/code/serve.py](../code/experiments/zero-shot/artifacts/runs/input_order/consistency_v1/code/serve.py)
- [attention/run.py](../code/experiments/attention/run.py)
- [attention/_lib/comparison.py](../code/experiments/attention/_lib/comparison.py)
- [attention/_lib/distribution.py](../code/experiments/attention/_lib/distribution.py)
- [attention/_lib/dynamics.py](../code/experiments/attention/_lib/dynamics.py)
- [attention/_lib/dynamics_browser.py](../code/experiments/attention/_lib/dynamics_browser.py)
- [attention/_lib/dynamics_plot.py](../code/experiments/attention/_lib/dynamics_plot.py)
- [attention/_lib/dynamics_report.py](../code/experiments/attention/_lib/dynamics_report.py)
- [attention/_lib/dynamics_verify.py](../code/experiments/attention/_lib/dynamics_verify.py)
- [attention/_lib/generation.py](../code/experiments/attention/_lib/generation.py)
- [attention/_lib/generation_analysis.py](../code/experiments/attention/_lib/generation_analysis.py)
- [attention/_lib/generation_plot.py](../code/experiments/attention/_lib/generation_plot.py)
- [attention/_lib/generation_verify.py](../code/experiments/attention/_lib/generation_verify.py)
- [attention/_lib/heads.py](../code/experiments/attention/_lib/heads.py)
- [attention/_lib/hidden_analysis.py](../code/experiments/attention/_lib/hidden_analysis.py)
- [attention/_lib/hidden_browser.py](../code/experiments/attention/_lib/hidden_browser.py)
- [attention/_lib/hidden_capture.py](../code/experiments/attention/_lib/hidden_capture.py)
- [attention/_lib/hidden_verify.py](../code/experiments/attention/_lib/hidden_verify.py)
- [attention/_lib/input_pipeline.py](../code/experiments/attention/_lib/input_pipeline.py)
- [attention/_lib/measure_live_speed.py](../code/experiments/attention/_lib/measure_live_speed.py)
- [attention/_lib/paths.py](../code/experiments/attention/_lib/paths.py)
- [attention/_lib/query_analysis.py](../code/experiments/attention/_lib/query_analysis.py)
- [attention/_lib/query_plot.py](../code/experiments/attention/_lib/query_plot.py)
- [attention/_lib/query_report.py](../code/experiments/attention/_lib/query_report.py)
- [attention/_lib/query_verify.py](../code/experiments/attention/_lib/query_verify.py)
- [attention/_lib/report_browser.py](../code/experiments/attention/_lib/report_browser.py)
- [attention/_lib/spatial_analysis.py](../code/experiments/attention/_lib/spatial_analysis.py)
- [attention/_lib/spatial_counterfactual.py](../code/experiments/attention/_lib/spatial_counterfactual.py)
- [attention/_lib/spatial_verify.py](../code/experiments/attention/_lib/spatial_verify.py)
- [attention/_lib/spatial_viewer.py](../code/experiments/attention/_lib/spatial_viewer.py)
- [attention/_lib/timelens_measurements.py](../code/experiments/attention/_lib/timelens_measurements.py)
- [attention/_lib/timelens_order.py](../code/experiments/attention/_lib/timelens_order.py)
- [attention/_lib/timelens_report.py](../code/experiments/attention/_lib/timelens_report.py)
- [attention/_lib/vendor/MTLA/LICENSE](../code/experiments/attention/_lib/vendor/MTLA/LICENSE)
- [attention/_lib/vendor/MTLA/evaluate.py](../code/experiments/attention/_lib/vendor/MTLA/evaluate.py)
- [attention/_lib/vendor/MTLA/extract.py](../code/experiments/attention/_lib/vendor/MTLA/extract.py)
- [attention/_lib/vendor/MTLA/gen_strategies.py](../code/experiments/attention/_lib/vendor/MTLA/gen_strategies.py)
- [attention/_lib/vendor/MTLA/generate.py](../code/experiments/attention/_lib/vendor/MTLA/generate.py)
- [attention/_lib/vendor/MTLA/mypy.ini](../code/experiments/attention/_lib/vendor/MTLA/mypy.ini)
- [attention/_lib/vendor/MTLA/pyproject.toml](../code/experiments/attention/_lib/vendor/MTLA/pyproject.toml)
- [attention/_lib/vendor/MTLA/requirements.txt](../code/experiments/attention/_lib/vendor/MTLA/requirements.txt)
- [attention/_lib/vendor/MTLA/configs/charades_qwen3vl.yaml](../code/experiments/attention/_lib/vendor/MTLA/configs/charades_qwen3vl.yaml)
- [attention/_lib/vendor/MTLA/configs/coco_internvl.yaml](../code/experiments/attention/_lib/vendor/MTLA/configs/coco_internvl.yaml)
- [attention/_lib/vendor/MTLA/configs/coco_qwen3vl.yaml](../code/experiments/attention/_lib/vendor/MTLA/configs/coco_qwen3vl.yaml)
- [attention/_lib/vendor/MTLA/configs/qvhighlights_qwen3vl.yaml](../code/experiments/attention/_lib/vendor/MTLA/configs/qvhighlights_qwen3vl.yaml)
- [attention/_lib/vendor/MTLA/mtla/__init__.py](../code/experiments/attention/_lib/vendor/MTLA/mtla/__init__.py)
- [attention/_lib/vendor/MTLA/mtla/config.py](../code/experiments/attention/_lib/vendor/MTLA/mtla/config.py)
- [attention/_lib/vendor/MTLA/mtla/metrics.py](../code/experiments/attention/_lib/vendor/MTLA/mtla/metrics.py)
- [attention/_lib/vendor/MTLA/mtla/mtla_attn.py](../code/experiments/attention/_lib/vendor/MTLA/mtla/mtla_attn.py)
- [attention/_lib/vendor/MTLA/mtla/registry.py](../code/experiments/attention/_lib/vendor/MTLA/mtla/registry.py)
- [attention/_lib/vendor/MTLA/mtla/types.py](../code/experiments/attention/_lib/vendor/MTLA/mtla/types.py)
- [attention/_lib/vendor/MTLA/mtla/utils.py](../code/experiments/attention/_lib/vendor/MTLA/mtla/utils.py)
- [attention/_lib/vendor/MTLA/mtla/voting.py](../code/experiments/attention/_lib/vendor/MTLA/mtla/voting.py)
- [attention/_lib/vendor/MTLA/scripts/figure_pertoken.py](../code/experiments/attention/_lib/vendor/MTLA/scripts/figure_pertoken.py)
- [attention/_lib/vendor/MTLA/scripts/prep_utils.py](../code/experiments/attention/_lib/vendor/MTLA/scripts/prep_utils.py)
- [attention/_lib/vendor/MTLA/scripts/prepare_charades.py](../code/experiments/attention/_lib/vendor/MTLA/scripts/prepare_charades.py)
- [attention/_lib/vendor/MTLA/scripts/prepare_coco.py](../code/experiments/attention/_lib/vendor/MTLA/scripts/prepare_coco.py)
- [attention/_lib/vendor/MTLA/scripts/prepare_qvhighlights.py](../code/experiments/attention/_lib/vendor/MTLA/scripts/prepare_qvhighlights.py)
- [attention/_lib/vendor/MTLA/third_party/moment_detr_eval/LICENSE](../code/experiments/attention/_lib/vendor/MTLA/third_party/moment_detr_eval/LICENSE)
- [attention/_lib/vendor/MTLA/third_party/moment_detr_eval/__init__.py](../code/experiments/attention/_lib/vendor/MTLA/third_party/moment_detr_eval/__init__.py)
- [attention/_lib/vendor/MTLA/third_party/moment_detr_eval/eval.py](../code/experiments/attention/_lib/vendor/MTLA/third_party/moment_detr_eval/eval.py)
- [attention/_lib/vendor/MTLA/third_party/moment_detr_eval/utils.py](../code/experiments/attention/_lib/vendor/MTLA/third_party/moment_detr_eval/utils.py)
- [attention/artifacts/common/provenance/history/separation_before/run.py](../code/experiments/attention/artifacts/common/provenance/history/separation_before/run.py)
- [attention/artifacts/runs/hidden/fonts/LICENSE](../code/experiments/attention/artifacts/runs/hidden/fonts/LICENSE)
