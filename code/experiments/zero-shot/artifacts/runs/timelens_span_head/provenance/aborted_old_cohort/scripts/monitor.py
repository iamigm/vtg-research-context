import json, subprocess, time, os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
jobs=json.loads((root/'execution.json').read_text())['jobs']
with (root/'logs/gpu.csv').open('w') as f:
 f.write('time,utilization_percent,memory_mib,power_watts\n')
 while True:
  active=False
  for job in jobs:
   try:
    state=Path('/proc')/str(job['pid'])/'stat'
    active |= state.exists() and state.read_text().split()[2]!='Z'
   except FileNotFoundError: pass
  if not active:break
  out=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
  f.write(str(time.time())+','+out+'\n');f.flush();time.sleep(2)
