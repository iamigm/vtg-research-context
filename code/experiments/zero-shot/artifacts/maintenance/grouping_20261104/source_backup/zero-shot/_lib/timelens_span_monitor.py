from result_layout import relocate_path, write_result_text
import json, subprocess, time, os
from pathlib import Path
root=relocate_path(relocate_path(Path(__file__)).resolve().parents[1]/'results/timelens_span_head')
jobs=json.loads((relocate_path(root/'execution.json')).read_text())['jobs']
with (relocate_path(root/'logs/gpu.csv')).open('a') as f:
 if f.tell()==0:f.write('time,utilization_percent,memory_mib,power_watts\n')
 while True:
  active=False
  for job in jobs:
   try:
    state=relocate_path(relocate_path(Path('/proc'))/str(job['pid'])/'stat')
    active |= state.exists() and state.read_text().split()[2]!='Z'
   except FileNotFoundError: pass
  if not active:break
  out=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
  f.write(str(time.time())+','+out+'\n');f.flush();time.sleep(2)
