from result_layout import relocate_path, write_result_text
import json,time,subprocess,os
from pathlib import Path
root=relocate_path(Path('/home1/irteam/experiments/zero-shot/results/timelens_top30_full'));out=relocate_path(root/'performance');pids=[3298186,3298187,3298553,3298704,3298517,3298662];rows=[]
def stats(pid):
 try:
  text=relocate_path(Path(f'/proc/{pid}/stat')).read_text().split();io=dict(line.split(': ') for line in relocate_path(Path(f'/proc/{pid}/io')).read_text().splitlines());return dict(cpu_ticks=int(text[13])+int(text[14]),state=text[2],wchan=relocate_path(Path(f'/proc/{pid}/wchan')).read_text(),read_bytes=int(io['read_bytes']),rchar=int(io['rchar']))
 except FileNotFoundError:return None
for i in range(45):
 g=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
 rows.append(dict(time=time.time(),gpu=g,processes={p:stats(p) for p in pids},cgroup=relocate_path(Path('/sys/fs/cgroup/cpu.stat')).read_text()));time.sleep(1)
write_result_text((relocate_path(out/'observation.json')), json.dumps(rows,indent=2));print('45 observations saved',flush=True)
