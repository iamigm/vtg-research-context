"""Frame-point coverage of GT; this is not a measure of visible action content."""
from result_layout import relocate_path, write_result_text
import collections
import json
from pathlib import Path

ROOT = relocate_path(Path(__file__)).resolve().parents[1]
DEST = relocate_path(ROOT / 'reports/pilot_100')
pairs = json.loads((relocate_path(DEST / 'baseline_paired_cases.json')).read_text())
lookup = {}
for model in ['qwen25']:
    path = relocate_path(ROOT / 'results/pilot_100' / model / 'predictions.jsonl')
    if path.exists():
        for row in map(json.loads, path.read_text().splitlines()):
            lookup[(model, row['sample_id'])] = row
rows = []
groups = collections.defaultdict(list)
for pair in pairs:
    source = lookup[(pair['model_key'], pair['sample_id'])]
    audit = source['input_audit']
    start, end = pair['gt']
    native = [idx / audit['decoded_fps'] for idx in audit['sampled_frame_indices_2fps']]
    converted = audit['timestamps_seconds']
    native_count = sum(start <= t <= end for t in native)
    converted_count = sum(start <= t <= end for t in converted)
    record = {k: pair[k] for k in ['model_key', 'dataset', 'sample_id', 'gt', 'outcome']}
    record.update(native_frames_inside_gt=native_count,
                  selected_frames_inside_gt=converted_count,
                  lost_all_frame_points=native_count > 0 and converted_count == 0,
                  no_selected_frame_point=converted_count == 0)
    rows.append(record)
    groups[pair['model_key'] + '/' + pair['dataset']].append(record)
summary = {}
for key, records in groups.items():
    regressions = [r for r in records if r['outcome'] == 'regressed']
    summary[key] = {
        'n': len(records),
        'no_native_frame_point': sum(r['native_frames_inside_gt'] == 0 for r in records),
        'no_selected_frame_point': sum(r['no_selected_frame_point'] for r in records),
        'lost_all_frame_points': sum(r['lost_all_frame_points'] for r in records),
        'regressions': len(regressions),
        'regressions_lost_all_frame_points': sum(r['lost_all_frame_points'] for r in regressions),
        'regressions_with_selected_frame_inside_gt': sum(r['selected_frames_inside_gt'] > 0 for r in regressions),
    }
result = {'complete': len(rows) == 200, 'groups': summary,
          'definition': 'Count decoded frame timestamps inside the closed GT interval, using original frame indices / FPS, before display rounding.',
          'limitation': 'A frame point inside GT does not establish that the relevant action or object is visually discernible. Motion loss and boundary sampling remain possible even with retained points.',
          'cases': rows}
write_result_text((relocate_path(DEST / 'sampling_analysis.json')), json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({'complete': result['complete'], 'groups': summary}, indent=2))
