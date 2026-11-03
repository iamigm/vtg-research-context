from result_layout import relocate_path, write_result_text
"""Reuse browser-ready H.264 videos from the existing TimeLens media cache."""
import json
import os
from pathlib import Path
from prepare import OUT, VIEW, video_key, write

def prepare_media(rows):
    folder=OUT/'media'
    folder.mkdir(exist_ok=True)
    records={}
    for source in dict.fromkeys(s['video_path'] for s in rows):
        key=video_key(source)
        cached=relocate_path(Path('/home/irteam/experiments/zero-shot/results/uncertainty/token_analysis/full/9404/media'))
        info=json.loads((cached/f'{key}.json').read_text())
        assert info['source']==source and info['codec']=='h264'
        target=folder/f'{key}.mp4'
        if not target.exists(): os.link(cached/target.name,target)
        assert target.stat().st_size==info['size']
        records[source]=dict(info,media=os.path.relpath(target,VIEW),storage='hard link to existing browser-ready media')
    write(OUT/'media_manifest.json',records)
    print('Browser video media ready:',len(records),flush=True)
    return records

if __name__=='__main__':prepare_media(json.loads((OUT/'samples.json').read_text()))
