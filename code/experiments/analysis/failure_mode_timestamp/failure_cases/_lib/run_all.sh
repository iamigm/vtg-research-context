#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
.venv/bin/python _lib/run_inference.py --model qwen25 > artifacts/logs/qwen25.log 2>&1
.venv/bin/python _lib/analyze.py > artifacts/logs/summary.json
.venv/bin/python _lib/validate_results.py > artifacts/logs/validation.json
.venv/bin/python _lib/validate_paired_inputs.py > artifacts/logs/paired_input_validation.json
.venv/bin/python _lib/duration_analysis.py > artifacts/logs/duration.json
.venv/bin/python _lib/compare_baseline.py > artifacts/logs/comparison.json
.venv/bin/python _lib/stratified_analysis.py > artifacts/logs/stratified_analysis.json
.venv/bin/python _lib/sampling_analysis.py > artifacts/logs/sampling_analysis.json
.venv/bin/python _lib/render_comparison.py > artifacts/logs/render_comparison.log 2>&1
.venv/bin/python _lib/render_details.py
.venv/bin/python _lib/write_report.py
.venv/bin/python _lib/record_artifact_hashes.py
