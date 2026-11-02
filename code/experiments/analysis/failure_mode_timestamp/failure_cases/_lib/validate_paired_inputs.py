"""Check equal visual token budgets and temporal-group counts against baseline."""
from result_layout import relocate_path, write_result_text
import argparse,json
from pathlib import Path
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT.parents[1]/'failure_mode_baseline/failure_cases')
p=argparse.ArgumentParser();p.add_argument('--allow-incomplete',action='store_true');args=p.parse_args();results={}
for model in ['qwen25']:
 bp=relocate_path(BASE/'results/pilot_100'/model/'predictions.jsonl');tp=relocate_path(ROOT/'results/pilot_100'/model/'predictions.jsonl')
 if not tp.exists():results[model]={'n':0,'complete':False};continue
 bs={r['sample_id']:r for r in map(json.loads,bp.read_text().splitlines())};rows=list(map(json.loads,tp.read_text().splitlines()));assert len(rows)==len({r['sample_id'] for r in rows})
 for r in rows:
  b=bs[r['sample_id']];a=r['input_audit'];t,h,w=b['video_grid_thw'][0]
  assert a['image_groups']==t
  assert a['visual_tokens']==t*h*w//4
  assert a['image_grid_thw']==[[1,h,w]]*t
  assert a['selected_frame_indices']==a['sampled_frame_indices_2fps'][::2]
  assert a['timestamps_seconds']==[idx/a['decoded_fps'] for idx in a['selected_frame_indices']]
  assert a['displayed_timestamps']==[f'{x:.1f}s:' for x in a['timestamps_seconds']]
  assert len(a['sampled_frame_indices_2fps'])==2*t
 results[model]={'n':len(rows),'complete':len(rows)==len(bs),'same_visual_token_budget_all_rows':True,'same_spatial_grid_all_rows':True,'same_number_of_temporal_groups_all_rows':True,'timestamp_frame_alignment_all_rows':True}
report={'complete':all(v['complete'] for v in results.values()),'models':results};dest=relocate_path(ROOT/'reports/pilot_100');dest.mkdir(parents=True,exist_ok=True);write_result_text((relocate_path(dest/'paired_input_validation.json')), json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not report['complete'] and not args.allow_incomplete:raise SystemExit('Incomplete')
