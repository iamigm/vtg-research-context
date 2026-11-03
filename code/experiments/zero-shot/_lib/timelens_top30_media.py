"""Prepare portable browser-playable, silent videos without changing inference inputs."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess

from timelens_paths import PILOT_SAMPLES
ROOT=relocate_path(Path(__file__)).resolve().parents[1]
OUT=relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300')
FFMPEG='/home1/irteam/envs/dit/bin/ffmpeg'
FFPROBE='/home1/irteam/envs/dit/bin/ffprobe'


def prepare(source):
    source=relocate_path(Path(source))
    key=hashlib.sha256(str(source).encode()).hexdigest()[:20]
    dest=relocate_path(OUT/'media'/f'{key}.mp4')
    meta=relocate_path(OUT/'media'/f'{key}.json')
    if dest.exists() and meta.exists():return json.loads(meta.read_text())
    probe=json.loads(subprocess.check_output([FFPROBE,'-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,duration,nb_frames:format=duration','-of','json',str(source)]))
    stream=probe['streams'][0]
    codec=stream['codec_name']
    copy=codec=='h264'
    temp=dest.with_suffix('.tmp.mp4')
    video=['-c:v','copy'] if copy else ['-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p']
    subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y','-i',str(source),'-map','0:v:0',*video,'-an','-movflags','+faststart',str(temp)],check=True)
    temp.replace(dest)
    after=json.loads(subprocess.check_output([FFPROBE,'-v','error','-select_streams','v:0','-show_entries','stream=codec_name,duration,nb_frames:format=duration','-of','json',str(dest)]))
    duration=float(probe['format']['duration']);new_duration=float(after['format']['duration'])
    # Container duration can include a longer audio track, which we intentionally drop.
    # Check the visual stream's frame count instead of comparing audio/container tails.
    before_frames=stream.get('nb_frames'); after_frames=after['streams'][0].get('nb_frames')
    if copy and before_frames and after_frames and before_frames!='N/A' and after_frames!='N/A':
        assert before_frames==after_frames,(str(source),before_frames,after_frames)
    result=dict(source=str(source),media='media/'+dest.name,source_codec=codec,codec=after['streams'][0]['codec_name'],
        conversion='stream copy; no audio; faststart' if copy else 'H.264 transcode; no audio; no temporal resampling',
        duration=new_duration,source_duration=duration,size=dest.stat().st_size,source_size=source.stat().st_size)
    write_result_text(meta, json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    (relocate_path(OUT/'media')).mkdir(parents=True,exist_ok=True)
    paths=sorted({r['video_path'] for r in json.loads(PILOT_SAMPLES.read_text())})
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i,record in enumerate(pool.map(prepare,paths)):
            records.append(record)
            print('MEDIA',i+1,len(paths),record['media'],flush=True)
    write_result_text((relocate_path(OUT/'media_manifest.json')), json.dumps({r['source']:r for r in records},indent=2)+'\n')
