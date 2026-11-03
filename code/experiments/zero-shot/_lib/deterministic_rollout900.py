"""Regenerate cumulative deterministic rollout reports from the unified900 records."""
from result_layout import relocate_path
from pathlib import Path
import json
from deterministic_rollout300 import ROOT,read,write,sha
from deterministic_rollout300_report import report as base_report

OUT=relocate_path(ROOT/'artifacts/runs/deterministic_rollout_pilot900_v2')
REPORT=relocate_path(ROOT/'results/deterministic_rollout_pilot900_v2')

def reports():
 import deterministic_rollout300_report as report_module
 cohort=read(OUT/'metadata/cohort.json')
 assert len(cohort)==900
 store=OUT/'rollouts'
 records={row['sample_key']:read(store/(row['sample_key']+'.json')) for row in cohort}
 assert all(records[row['sample_key']]['sample']==row for row in cohort)
 for n in [300,600,900]:
  run=f'deterministic_rollout_pilot{n}_v2'
  out=relocate_path(ROOT/'artifacts/runs'/run);dest=relocate_path(ROOT/'results'/run)
  assert read(out/'metadata/cohort.json')==cohort[:n]
  refs=[dict(path=str(store/(row['sample_key']+'.json')),sha256=sha(store/(row['sample_key']+'.json'))) for row in cohort[:n]]
  write(out/'metadata/source_records.json',refs)
  report_module.OUT=out;report_module.REPORT=dest
  rows=sorted(cohort[:n],key=lambda row:row['sample_key']) if n==300 else cohort[:n]
  base_report(records=[records[row['sample_key']] for row in rows],cohort_label=f'pilot{n}')
 write(OUT/'validation/integration.json',dict(passed=True,queries=900,nested_cohorts=[300,600,900],single_source_store=str(store),records_copied=False))

if __name__=='__main__':reports()
