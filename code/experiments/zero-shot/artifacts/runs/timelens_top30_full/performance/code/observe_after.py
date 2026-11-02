from result_layout import relocate_path, write_result_text
from pathlib import Path
import time,subprocess,json,os
base=relocate_path(Path('/home1/irteam/experiments/zero-shot/results/timelens_top30_full'));out=relocate_path(base/'performance');execution=json.loads((relocate_path(base/'execution.json')).read_text());parents=[w['pid'] for w in execution['workers']];pids=list(parents)
for p in relocate_path(Path('/proc')).iterdir():
 if p.name.isdigit():
  try:
   a=(relocate_path(p/'stat')).read_text().rsplit(')',1)[1].split()
   if int(a[1]) in parents and 'pt_data_worker' in (relocate_path(p/'comm')).read_text():pids.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
def proc(pid):
 try:
  p=relocate_path(relocate_path(Path('/proc'))/str(pid));a=(relocate_path(p/'stat')).read_text().rsplit(')',1)[1].split();io=dict(x.split(': ') for x in (relocate_path(p/'io')).read_text().splitlines());return dict(ticks=int(a[11])+int(a[12]),state=a[0],wchan=(relocate_path(p/'wchan')).read_text(),read_bytes=int(io['read_bytes']))
 except FileNotFoundError:return None
rows=[]
for i in range(60):
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used','--format=csv,noheader,nounits'],text=True).strip();rows.append(dict(time=time.time(),gpu=gpu,processes={p:proc(p) for p in pids},cpu=dict(x.split() for x in relocate_path(Path('/sys/fs/cgroup/cpu.stat')).read_text().splitlines()),io_pressure=relocate_path(Path('/sys/fs/cgroup/io.pressure')).read_text()));time.sleep(.5)
write_result_text((relocate_path(out/'observation_after.json')), json.dumps(rows,indent=2));print('60 post-tuning observations saved',flush=True)
