"""Remove temporary FPS access only after current rollout and postprocessing exit."""
import datetime,json,os,time,traceback
from pathlib import Path
ROOT=Path('/home/irteam/experiments');P=ROOT/'zero-shot';S=P/'artifacts/maintenance/grouping_20261104'
def log(message):
 print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),message,flush=True)
def identity(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
 except FileNotFoundError:return None
state=S/'cleanup_status.json'
try:
 watched=json.loads(state.read_text())['watched'] if state.exists() else {}
 if not watched:
  names={'rollout_distribution_parallel.py','rollout_token_distributions.py','rollout_distribution_integrate.py'}
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:args=(proc/'cmdline').read_bytes().decode().split('\0')
   except (OSError,UnicodeError):continue
   if any(Path(a).name in names for a in args):watched[proc.name]=identity(proc.name)
 def write(stage,**kw):state.write_text(json.dumps({'stage':stage,'watched':watched,**kw},indent=2)+'\n')
 write('waiting_for_rollout');log('Watching current rollout and integration PIDs: '+','.join(watched))
 while any(identity(pid)==start for pid,start in watched.items()):time.sleep(20)
 from result_layout import relocate_path
 link=P/'artifacts/runs/fps_spatial_control_pilot300_v1'
 if link.is_symlink():link.unlink();log('Removed temporary FPS link')
 # Validate that immutable metadata resolves without the compatibility link.
 out=relocate_path(link);assert out.is_dir()
 cache=next((out/'media').glob('*/metadata.json'))
 obj=json.loads(cache.read_text());array=relocate_path(Path(obj['array']));assert array.is_file()
 assert not link.exists()
 write('complete',validated_array=str(array));log('PASS: canonical FPS cache accessible after link removal')
except Exception:
 log(traceback.format_exc());raise
