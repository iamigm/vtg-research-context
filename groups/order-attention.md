# 입력 순서와 Attention

영상→질문 / 질문→영상 성능·예측 일관성과 query/generation/hidden/spatial/dynamics attention. MTLA pilot과 TimeLens 전체평가를 구분하며 계측량과 성능의 인과를 단정하지 않는다.

## 실행별 보고서·관련 코드

### attention/input_order

[final_report.md](../reports/experiments/attention/results/input_order/timelens_instruct/full/9404/final_report.md) · [FINAL_REPORT.md](../reports/experiments/attention/results/input_order/timelens_thinking/pilot/300/FINAL_REPORT.md) · [REPORT.md](../reports/experiments/attention/results/input_order/timelens_thinking/pilot/300/REPORT.md) · [SCHEMA.md](../reports/experiments/attention/results/input_order/timelens_thinking/pilot/300/SCHEMA.md) · [REPORT.md](../reports/experiments/attention/results/input_order/timelens_thinking/pilot/overview/REPORT.md)

현재 관련 코드 71개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### attention/token_attention

[REPORT.md](../reports/experiments/attention/results/token_attention/mtla/pilot/300/REPORT.md) · [FINDINGS.md](../reports/experiments/attention/results/token_attention/mtla/pilot/300/dynamics/FINDINGS.md) · [FINDINGS.md](../reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/FINDINGS.md) · [REPORT.md](../reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/distribution_analysis/REPORT.md) · [REPORT.md](../reports/experiments/attention/results/token_attention/mtla/pilot/300/generation/head_analysis/REPORT.md) · [FINDINGS.md](../reports/experiments/attention/results/token_attention/mtla/pilot/300/query/FINDINGS.md) · [REPORT.md](../reports/experiments/attention/results/token_attention/mtla/pilot/overview/REPORT.md)

현재 관련 코드 69개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/input_order

[REPORT.md](../reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/input_order/REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/results/input_order/consistency_v1/REPORT.md)

현재 관련 코드 74개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

## 관련 실험 대화

- [./experiments에 이 실험을 대표하는 이름으로 디렉토리를 하나 만들고 거기서 실험을 진행해.](../conversations/01a0eb98-2716-7e21-8366-1f182062f967.md) — deleted
- [MTLA moment retrieval 구현 확인](../conversations/01a0ebf8-47fb-7fd0-b80a-36c612c5f275.md) — deleted
- [query attention, generated answer attention이 그리는 분포의 차이를 계산해봐. 얼마나 다른지 경향성은 유지하는지 등](../conversations/01a0eca8-0b26-78f2-bbae-59a2a32c457e.md) — deleted
- [GT attention mass 실험 확인](../conversations/01a0f0a4-087b-7912-88d8-2b2d7053a830.md) — deleted
- [기존 attention 관련 실험 하나의 폴더 아래에 정리해봐. 세팅만 살짝 다르고 실험의 목적은 동일하니까 필요 없는 건 없애고 하나의 스크립트 및 실험 결과로 통합해버려](../conversations/01a13f33-09f0-7042-985f-7d112c123f6a.md) — deleted
- [이전에 쿼리; 비디오, 비디오; 쿼리 두 입력의 배치 순서를 바꿔본 적이 있잖아. 이때의 같은 샘플로 수행했는지, 몇 개의 샘플로 수행했는지, 저장해놓은 산출물 종류 알려줘.](../conversations/01a1689a-0c59-7920-8e01-475d6b29c0c5.md) — deleted
## 코드 백업

아래는 현재 구현의 연결이다. 역사적 실행 revision의 증명은 아니다. 공용 dependency는 [shared](shared.md)를 참조한다.

- [zero-shot/_lib/zeroshot_report.py](../code/experiments/zero-shot/_lib/zeroshot_report.py)
- [zero-shot/artifacts/maintenance/grouping_20261104/source_backup/attention/_lib/timelens_report.py](../code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/attention/_lib/timelens_report.py)
- [zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/zeroshot_report.py](../code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/zero-shot/_lib/zeroshot_report.py)
- [zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/analyze.py](../code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/analyze.py)
- [zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/build_viewer.py](../code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/build_viewer.py)
- [zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/serve.py](../code/experiments/zero-shot/artifacts/runs/baseline/input_order/overview/consistency/code/serve.py)
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
- [attention/artifacts/runs/token_attention/mtla/pilot/300/hidden/fonts/LICENSE](../code/experiments/attention/artifacts/runs/token_attention/mtla/pilot/300/hidden/fonts/LICENSE)
