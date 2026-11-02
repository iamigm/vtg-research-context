from result_layout import relocate_path, write_result_text
from pathlib import Path
import json,os,time,hashlib
import timelens_top30_media as media
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/timelens_top30_full');OLD=relocate_path(ROOT/'results/timelens_top30');media.OUT=BASE;(relocate_path(BASE/'media')).mkdir(exist_ok=True)
rows=json.loads((relocate_path(BASE/'samples.json')).read_text());sources=list(dict.fromkeys(s['video_path'] for s in rows));results={};start=time.time()
for i,src in enumerate(sources):
    key=hashlib.sha256(src.encode()).hexdigest()[:20]
    for suffix in ['.mp4','.json']:
        old=relocate_path(OLD/'media'/(key+suffix));new=relocate_path(BASE/'media'/(key+suffix))
        if old.exists() and not new.exists():os.link(old,new)
    results[src]=media.prepare(src)
    if (i+1)%25==0 or i+1==len(sources):
        state=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),completed=i+1,total=len(sources),percent=100*(i+1)/len(sources),elapsed_seconds=time.time()-start,eta_seconds=(time.time()-start)*(len(sources)-i-1)/(i+1));write_result_text((relocate_path(BASE/'media_status.json')), json.dumps(state));print(json.dumps(state),flush=True)
write_result_text((relocate_path(BASE/'media_manifest.json')), json.dumps(results,ensure_ascii=False,indent=2));write_result_text((relocate_path(BASE/'media_complete.json')), json.dumps(dict(complete=True,unique_videos=len(sources))))
