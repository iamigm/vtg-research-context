# 입력 변환과 실패 유형

Charades·QVHighlights 각100 query. Qwen2.5 native / TimeLens-7B식 입력 / Qwen3 native. Qwen2.5 변환 R@IoU.5 −19/−28pp. Frame복제·fps·modality·시간표현·지시문이 함께 바뀌므로 단일원인 효과로 해석하지 않는다.

## 실행별 보고서·관련 코드

### analysis/failure_mode_timestamp/failure_cases/pilot_100

[REPORT.md](../reports/experiments/analysis/failure_mode_timestamp/failure_cases/results/pilot_100/REPORT.md)

현재 관련 코드 24개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

## 코드 백업

아래는 현재 구현의 연결이다. 역사적 실행 revision의 증명은 아니다. 공용 dependency는 [shared](shared.md)를 참조한다.

- [analysis/failure_mode_timestamp/failure_cases/_lib/analyze.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/analyze.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/build_video_gallery.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/build_video_gallery.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/check_inputs.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/check_inputs.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/compare_baseline.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/compare_baseline.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/duration_analysis.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/duration_analysis.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/finish.sh](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/finish.sh)
- [analysis/failure_mode_timestamp/failure_cases/_lib/metrics.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/metrics.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/record_artifact_hashes.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/record_artifact_hashes.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/render_cases.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/render_cases.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/render_comparison.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/render_comparison.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/render_details.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/render_details.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/run_all.sh](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/run_all.sh)
- [analysis/failure_mode_timestamp/failure_cases/_lib/run_inference.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/run_inference.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/sampling_analysis.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/sampling_analysis.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/serve_gallery.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/serve_gallery.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/stratified_analysis.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/stratified_analysis.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/timestamp_inputs.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/timestamp_inputs.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/validate_conversion.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/validate_conversion.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/validate_paired_inputs.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/validate_paired_inputs.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/validate_results.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/validate_results.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/validate_video_gallery.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/validate_video_gallery.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/write_report.py](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/write_report.py)
- [analysis/failure_mode_timestamp/failure_cases/_lib/video_gallery/viewer.css](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/video_gallery/viewer.css)
- [analysis/failure_mode_timestamp/failure_cases/_lib/video_gallery/viewer.js](../code/experiments/analysis/failure_mode_timestamp/failure_cases/_lib/video_gallery/viewer.js)
