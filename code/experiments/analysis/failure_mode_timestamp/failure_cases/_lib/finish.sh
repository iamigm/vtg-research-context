#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
.venv/bin/python _lib/analyze.py
.venv/bin/python _lib/validate_results.py
.venv/bin/python _lib/validate_paired_inputs.py
.venv/bin/python _lib/duration_analysis.py
.venv/bin/python _lib/compare_baseline.py
.venv/bin/python _lib/stratified_analysis.py > artifacts/logs/stratified_analysis.json
.venv/bin/python _lib/sampling_analysis.py
.venv/bin/python _lib/render_comparison.py
.venv/bin/python _lib/render_details.py
.venv/bin/python _lib/write_report.py
.venv/bin/python _lib/record_artifact_hashes.py
