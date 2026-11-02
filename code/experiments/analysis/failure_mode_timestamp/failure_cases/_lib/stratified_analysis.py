"""Paired results by predeclared GT fraction bins reused from baseline analysis."""
from result_layout import relocate_path, write_result_text
import collections
import csv
import json
from pathlib import Path
import numpy as np

ROOT = relocate_path(Path(__file__)).resolve().parents[1]
DEST = relocate_path(ROOT / 'reports/pilot_100')
samples = {r['sample_id']: r for r in json.loads((relocate_path(ROOT / 'data/samples_100.json')).read_text())}
pairs = json.loads((relocate_path(DEST / 'baseline_paired_cases.json')).read_text())
groups = collections.defaultdict(list)
for pair in pairs:
    groups[pair['model_key']+'/'+pair['dataset']].append(pair)
rows = []
for key, group in groups.items():
    for label, lo, hi in [('GT <=10% of video', 0, .1), ('GT 10-30% of video', .1, .3), ('GT >30% of video', .3, float('inf'))]:
        selected = [p for p in group if lo < (p['gt'][1]-p['gt'][0])/samples[p['sample_id']]['duration'] <= hi]
        if not selected:
            continue
        rows.append({'group': key, 'bin': label, 'n': len(selected),
                     'baseline_R1_05': 100*float(np.mean([p['baseline_iou'] >= .5 for p in selected])),
                     'timestamp_R1_05': 100*float(np.mean([p['timestamp_iou'] >= .5 for p in selected])),
                     'baseline_mIoU': 100*float(np.mean([p['baseline_iou'] for p in selected])),
                     'timestamp_mIoU': 100*float(np.mean([p['timestamp_iou'] for p in selected])),
                     'recovered': sum(p['outcome'] == 'recovered' for p in selected),
                     'regressed': sum(p['outcome'] == 'regressed' for p in selected)})
result = {'complete': len(pairs) == 200,
          'scope': 'Exploratory paired subgroup results. GT fraction cutoffs are reused from the baseline analysis; no significance claim or causal attribution.',
          'rows': rows}
write_result_text((relocate_path(DEST / 'stratified_analysis.json')), json.dumps(result, indent=2))
with (relocate_path(DEST / 'stratified_analysis.csv')).open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(result, indent=2))
