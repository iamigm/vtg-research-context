from result_layout import relocate_path, write_result_text
from pathlib import Path
import csv,json
OUT=Path('/home/irteam/experiments/zero-shot/results/input_order/consistency_v1/build_viewer.py').resolve().parent
EXP=OUT.parents[3]
media=json.loads((relocate_path(EXP/'zero-shot/results/timelens_top30_full/media_manifest.json')).read_text())
rows=[]; missing=[]
for r in csv.DictReader((relocate_path(OUT/'paired_samples.csv')).open()):
 d={k:r[k] for k in ['dataset','sample_key','video','query']}
 for k in ['duration','pair_iou','raw_pair_iou','iou_video_query','iou_query_video','iou_delta','normalized_max_endpoint_diff']:
  d[k]=float(r[k]) if r[k] else None
 for k in ['gt','video_query','query_video','raw_video_query','raw_query_video']: d[k]=json.loads(r[k]) if r[k] else None
 for k in ['rounded_equal','raw_equal','parse_failed_vq','parse_failed_qv','nonpositive_vq','nonpositive_qv']: d[k]=r[k]=='True'
 source='/data/video/MR/TimeLens/TimeLens-Bench/videos/'+r['dataset'].replace('-timelens','')+'/'+r['video']
 m=media.get(source)
 if m and (relocate_path(EXP/'zero-shot/results/timelens_top30_full'/m['media'])).exists(): d['media']='../../timelens_top30_full/'+m['media']
 else: missing.append(r['sample_key']);d['media']=None
 rows.append(d)
write_result_text((relocate_path(OUT/'viewer_data.json')), json.dumps(rows,ensure_ascii=False,separators=(',',':')))
write_result_text((relocate_path(OUT/'viewer_validation.json')), json.dumps(dict(samples=len(rows),media_available=len(rows)-len(missing),missing_media=missing),indent=2))
print('samples',len(rows),'media available',len(rows)-len(missing),'missing',len(missing))
