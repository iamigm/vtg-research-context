# Frame–Query 유사도와 비용

CLIP ViT-L/14와 Qwen3-VL-Embedding-8B. 2fps pilot300 query/292 videos →full9,404 query/4,279 videos →16fps pilot300 비용비교. Full mAP .4508/.5103, MRR .5594/.6247. 16fps GPU forward CLIP .7251s/Qwen15.0923s/Instruct7.8314s. 네이티브 전처리·해상도·구조가 달라 파이프라인 비용비교다. 인접frame cosine은 저장embedding의 파생분석이다.

## 실행별 보고서·관련 코드

### similarity/compute_cost

[APPENDIX.md](../reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/300/APPENDIX.md) · [REPORT.md](../reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/300/REPORT.md) · [REPORT.md](../reports/experiments/similarity/results/compute_cost/embedding_vs_generation_16fps/pilot/overview/REPORT.md)

현재 관련 코드 49개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### similarity/frame_query

[APPENDIX.md](../reports/experiments/similarity/results/frame_query/cosine_2fps/full/9404/APPENDIX.md) · [REPORT.md](../reports/experiments/similarity/results/frame_query/cosine_2fps/full/9404/REPORT.md) · [APPENDIX.md](../reports/experiments/similarity/results/frame_query/cosine_2fps/pilot/300/APPENDIX.md) · [REPORT.md](../reports/experiments/similarity/results/frame_query/cosine_2fps/pilot/300/REPORT.md) · [REPORT.md](../reports/experiments/similarity/results/frame_query/cosine_2fps/pilot/overview/REPORT.md)

현재 관련 코드 50개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

## 관련 실험 대화

- [MR 데이터셋 임베딩 temporal signal 실험](../conversations/01a0e7a6-4380-7cf2-ab28-04fd79163847.md) — deleted
- [Compare vector similarity and scores](../conversations/01a0f09d-8388-75d0-b859-814e032a9039.md) — deleted
- [experiments/similarity 하나 만들어서 2개 실험을 돌리자.](../conversations/01a186f0-efb7-7550-b57f-7340962f10e3.md) — active
## 코드 백업

아래는 현재 구현의 연결이다. 역사적 실행 revision의 증명은 아니다. 공용 dependency는 [shared](shared.md)를 참조한다.

- [zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/benchmark_16fps.py](../code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/benchmark_16fps.py)
- [zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/integrate_adjacent.py](../code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/integrate_adjacent.py)
- [zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/prepare.py](../code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/prepare.py)
- [zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/prepare_media.py](../code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/prepare_media.py)
- [zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/report_16fps.py](../code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/report_16fps.py)
- [zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/run_pipeline.py](../code/experiments/zero-shot/artifacts/maintenance/grouping_20261104/source_backup/similarity/_lib/run_pipeline.py)
- [similarity/_lib/adjacent_similarity.py](../code/experiments/similarity/_lib/adjacent_similarity.py)
- [similarity/_lib/benchmark_16fps.py](../code/experiments/similarity/_lib/benchmark_16fps.py)
- [similarity/_lib/check_viewer.py](../code/experiments/similarity/_lib/check_viewer.py)
- [similarity/_lib/compute_cost.py](../code/experiments/similarity/_lib/compute_cost.py)
- [similarity/_lib/download_models.py](../code/experiments/similarity/_lib/download_models.py)
- [similarity/_lib/finalize.py](../code/experiments/similarity/_lib/finalize.py)
- [similarity/_lib/frame_loader.py](../code/experiments/similarity/_lib/frame_loader.py)
- [similarity/_lib/infer.py](../code/experiments/similarity/_lib/infer.py)
- [similarity/_lib/integrate_adjacent.py](../code/experiments/similarity/_lib/integrate_adjacent.py)
- [similarity/_lib/metrics.py](../code/experiments/similarity/_lib/metrics.py)
- [similarity/_lib/prepare.py](../code/experiments/similarity/_lib/prepare.py)
- [similarity/_lib/prepare_media.py](../code/experiments/similarity/_lib/prepare_media.py)
- [similarity/_lib/report.py](../code/experiments/similarity/_lib/report.py)
- [similarity/_lib/report_16fps.py](../code/experiments/similarity/_lib/report_16fps.py)
- [similarity/_lib/run_pipeline.py](../code/experiments/similarity/_lib/run_pipeline.py)
- [similarity/_lib/score.py](../code/experiments/similarity/_lib/score.py)
- [similarity/_lib/validate.py](../code/experiments/similarity/_lib/validate.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/LICENSE](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/LICENSE)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/main.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/main.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/pyproject.toml](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/pyproject.toml)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/uv.lock](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/uv.lock)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/setup_environment.sh](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/setup_environment.sh)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/eval_embedding.sh](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/eval_embedding.sh)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/eval_reranker.sh](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/eval_reranker.sh)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/image.yaml](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/image.yaml)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/image_retrieval.yaml](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/image_retrieval.yaml)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/video.yaml](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/video.yaml)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/video_retrieval.yaml](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/video_retrieval.yaml)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/visdoc.yaml](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/visdoc.yaml)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/visdoc_retrieval.yaml](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/visdoc_retrieval.yaml)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/arguments.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/arguments.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/constant.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/constant.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/eval_embedding.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/eval_embedding.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/eval_reranker.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/eval_reranker.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/gather_results.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/gather_results.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/models.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/models.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/report_score_v2.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/report_score_v2.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/basic_utils.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/basic_utils.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/dataset_utils.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/dataset_utils.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/eval_utils/metrics.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/eval_utils/metrics.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/vision_utils/video_transforms.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/vision_utils/video_transforms.py)
- [similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/vision_utils/vision_utils.py](../code/experiments/similarity/_lib/vendor/Qwen3-VL-Embedding/src/evaluation/mmeb_v2/utils/vision_utils/vision_utils.py)
- [similarity/artifacts/common/provenance/project_root_migration/infer.py](../code/experiments/similarity/artifacts/common/provenance/project_root_migration/infer.py)
