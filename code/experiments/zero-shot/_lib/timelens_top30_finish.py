"""Detached completion supervisor: media -> await inference -> validated report."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import os,json,time,subprocess,sys,traceback
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300')
def status(state,**kw):
    p=relocate_path(OUT/'status.json');tmp=p.with_suffix('.tmp');write_result_text(tmp, json.dumps(dict(state=state,updated_epoch=time.time(),**kw),indent=2)+'\n');tmp.replace(p)
def alive(pid):
    try:return relocate_path(Path(f'/proc/{pid}/stat')).read_text().split()[2]!='Z'
    except FileNotFoundError:return False
try:
    status('running',stage='media and inference')
    subprocess.Popen([sys.executable,str(relocate_path(ROOT/'_lib/timelens_top30_progress.py'))],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    with (relocate_path(OUT/'logs/media.log')).open('a') as log:subprocess.run([sys.executable,str(relocate_path(ROOT/'_lib/timelens_top30_media.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=json.loads((relocate_path(OUT/'execution.json')).read_text())['jobs']
    while any(alive(j['pid']) for j in jobs):
        counts={m:len(list(result_glob((relocate_path(OUT/m)), '*.json'))) for m in ['qwen25','qwen3']};status('running',stage='inference',completed=counts);time.sleep(15)
    counts={m:len(list(result_glob((relocate_path(OUT/m)), '*.json'))) for m in ['qwen25','qwen3']}
    assert counts=={'qwen25':300,'qwen3':300},counts
    status('running',stage='report validation',completed=counts)
    subprocess.run([sys.executable,str(relocate_path(ROOT/'_lib/timelens_top30_report.py'))],check=True)
    subprocess.run([sys.executable,str(relocate_path(ROOT/'_lib/timelens_top30_browser.py'))],check=True)
    status('complete',completed=counts,html='report.html',markdown='REPORT.md')
except Exception:
    status('failed',error=traceback.format_exc());raise
finally:
    jobs=json.loads((relocate_path(OUT/'execution.json')).read_text())['jobs']
    if not any(alive(j['pid']) for j in jobs):
        env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY='/tmp/mtla_top30_mps');subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,check=False)
