"""Freeze, launch and supervise the one canonical cumulative experiment."""
import argparse,fcntl,os,py_compile,shutil,subprocess,sys,time
from pathlib import Path
from probability_budget_core import *

def prepare():
 for name in ('metadata','logs','queries','search','validation','code','web'):(OUT/name).mkdir(parents=True,exist_ok=True)
 with (OUT/'metadata/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if list((OUT/'queries').glob('*.json')):raise RuntimeError('Run already has query records; resume frozen runtime without refreezing')
  files=[]
  for source in (ROOT/'_lib').glob('probability_budget_*'):
   if not source.is_file():continue
   if source.suffix=='.py':py_compile.compile(str(source),doraise=True)
   target=OUT/'code'/source.name;shutil.copyfile(source,target);files.append(dict(source=str(source),snapshot=str(target),sha256=sha(target)))
  for source in [ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_core.py',ROOT/'_lib/fps_spatial_control.py',ROOT/'_lib/shared_selected_frames.py',ROOT/'_lib/adaptive_temporal_sampling.py']:
   files.append(dict(source=str(source),snapshot=str(source),sha256=sha(source)))
  shutil.copyfile(ROOT/'_lib/probability_budget_viewer.js',OUT/'web/viewer.js');write(OUT/'metadata/config.json',CONFIG);write(OUT/'metadata/manifest.json',dict(config=CONFIG,config_hash=digest(CONFIG),cohort=str(COHORT),cohort_sha256=sha(COHORT),files=files,timestamp=stamp()));write(OUT/'metadata/status.json',dict(state='preparing',stage='preflight',active_queries=300,completed=0,total=900,timestamp=stamp()))
  index=read(MODEL/'model.safetensors.index.json');weights=sorted(set(index['weight_map'].values()));assert all((MODEL/f).exists() for f in weights);write(OUT/'metadata/model.json',dict(model=str(MODEL),config_sha256=sha(MODEL/'config.json'),index_sha256=sha(MODEL/'model.safetensors.index.json'),tokenizer_sha256=sha(MODEL/'tokenizer.json'),weight_files=[dict(name=f,bytes=(MODEL/f).stat().st_size) for f in weights],backbone_registry_sha256=sha(Path('/home/irteam/backbones/manifests/registry.json'))))
  rows=read(COHORT)
  for n in (300,600,900):assert read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json')==rows[:n]
  for r in rows:assert (SOURCE/'inputs'/f'{r["sample_key"]}.json').exists() and (SOURCE/'rollouts'/f'{r["sample_key"]}.json').exists()
  write(OUT/'validation/cohort.json',dict(passed=True,queries=900,nested_prefixes=[300,600,900],source_inputs_present=True,timestamp=stamp()))
  from probability_budget_report import report
  report();print('PREPARED',OUT)

def launch():
 prior=OUT/'metadata/launcher.json'
 if prior.exists():
  info=read(prior)
  try:
   cmd=Path(f'/proc/{info["pid"]}/cmdline').read_bytes()
   if b'probability_budget_manage.py' in cmd:raise RuntimeError(f'Already running PID {info["pid"]}')
  except FileNotFoundError:pass
 smoke=read(OUT/'validation/gpu_smoke.json');assert smoke['passed'] and smoke['config_hash']==digest(CONFIG)
 for name,value in smoke['tested_source_hashes'].items():assert sha(OUT/'code'/name)==value,name
 env=dict(os.environ,PYTHONPATH=str(ROOT/'_lib'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with (OUT/'logs/supervisor.log').open('ab',buffering=0) as stream:
  child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(OUT/'code/probability_budget_manage.py'),'supervise'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 write(prior,dict(pid=child.pid,python=str(ROOT/'.venv/bin/python'),entry=str(OUT/'code/probability_budget_runtime.py'),log=str(OUT/'logs/supervisor.log'),detached=True,timestamp=stamp()));print('STARTED',child.pid,OUT/'logs/supervisor.log',flush=True)

def supervise():
 with (OUT/'metadata/supervisor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  attempts=[]
  for attempt in range(1,4):
   start=time.monotonic();worker=subprocess.Popen([sys.executable,'-u',str(OUT/'code/probability_budget_runtime.py')],cwd=ROOT,stdin=subprocess.DEVNULL);code=worker.wait();state=read(OUT/'metadata/status.json');attempts.append(dict(attempt=attempt,exit_code=code,wall_seconds=time.monotonic()-start,last_status=state,timestamp=stamp()));write(OUT/'metadata/supervision.json',dict(attempts=attempts))
   if code==0:return
   error=state.get('error','');retry=code in (-9,137) or 'OutOfMemory' in error or 'CUDA out of memory' in error
   if not retry or attempt==3:raise RuntimeError(f'Worker stopped, exit={code}; metadata/status.json and supervision.json contain the reason')
   print('RETRY_OWN_WORKER',attempt,code,flush=True);time.sleep(10)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','supervise']);a=p.parse_args();globals()[a.action]()
