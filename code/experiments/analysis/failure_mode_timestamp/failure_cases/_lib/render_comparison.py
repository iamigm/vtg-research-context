"""Evidence gallery: Qwen2.5 native / TimeLens-7B input and Qwen3 native on the same videos."""
from result_layout import relocate_path, write_result_text
import html,json
from pathlib import Path
from metrics import measure
from render_cases import render
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT.parents[1]/'failure_mode_baseline/failure_cases');DEST=relocate_path(ROOT/'reports/pilot_100');OUT=relocate_path(DEST/'cases');OUT.mkdir(exist_ok=True)
pairs=json.loads((relocate_path(DEST/'baseline_paired_cases.json')).read_text());ids=set(json.loads((relocate_path(ROOT/'configs/gallery_sample_ids.json')).read_text()));reasons={sid:['Fixed representative video cohort; three retained settings'] for sid in ids}
for m in ['qwen25']:
 for ds in ['charades-timelens','qvhighlights-timelens']:
  for outcome in ['recovered','regressed']:
   ranked=sorted([p for p in pairs if p['model_key']==m and p['dataset']==ds and p['outcome']==outcome],key=lambda p:(-abs(p['delta_iou']),p['sample_id']))
   for r in ranked[:2]:ids.add(r['sample_id']);reasons.setdefault(r['sample_id'],[]).append(f'{m}/{ds}/{outcome}: top absolute IoU change')
review=relocate_path(DEST/'manual_review.json')
if review.exists():
 for r in json.loads(review.read_text()):ids.add(r['sample_id']);reasons.setdefault(r['sample_id'],[]).append('manual review')
all_rows=[]
for root,variant in [(BASE,'baseline'),(ROOT,'timestamp')]:
 for m in (['qwen25','qwen3'] if variant=='baseline' else ['qwen25']):
  for r in map(json.loads,(relocate_path(root/'results/pilot_100'/m/'predictions.jsonl')).read_text().splitlines()):
   r.update(measure(r));r.update(model_key=m,variant=variant,display_label=('Qwen2.5' if m=='qwen25' else 'Qwen3')+' '+('native' if variant=='baseline' else 'TimeLens-7B input'));all_rows.append(r)
selected=[]
for sid in sorted(ids):
 rows=[r for r in all_rows if r['sample_id']==sid];assert len(rows)==3
 meta=render(sid,rows,OUT);meta['selection_reasons']=reasons[sid]
 selected.append(meta);print(sid,flush=True)
write_result_text((relocate_path(OUT/'selection.json')), json.dumps({'selection':'Fixed representative video cohort plus Qwen2.5 recovered/regressed examples; three retained settings. Descriptive examples, not frequency estimates.','cases':selected},indent=2))
entries=[]
for r in selected:
 key=r['sample_id'].replace('/','__');entries.append('<article><h2>'+html.escape(r['sample_id'])+'</h2><p>'+html.escape(r['query'])+'</p><img loading="lazy" src="'+key+'.jpg" width="1100"><p><a href="'+key+'.pdf">PDF</a> · <a href="'+key+'.json">Metadata</a></p></article>')
write_result_text((relocate_path(OUT/'contact_sheets.html')), '<!doctype html><meta charset="utf-8"><title>Baseline vs timestamp cases</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}article{border-bottom:1px solid #aaa}</style><h1>Native baseline vs TimeLens-style timestamp conversion</h1><p>Qwen2.5 native / TimeLens-7B input and Qwen3 native. Frames below are for visual review and are not necessarily the model input frames. <a href="../REPORT.md">Report</a> · <a href="../baseline_paired_cases.csv">Paired CSV</a></p>'+''.join(entries))

from build_video_gallery import build
build()
