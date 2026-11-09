"""Freeze and detach the complete 300-first pipeline, with automatic publication."""
import argparse,fcntl,os,py_compile,subprocess,sys,time
from probability_completion_core import *
def prepare():
 for name in ('metadata','logs','code','queries','web','validation'):(OUT/name).mkdir(parents=True,exist_ok=True)
 with (OUT/'metadata/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if list((OUT/'queries').glob('*.json')):raise RuntimeError('Use the existing frozen run; query records already exist')
  for row in read(COHORT)[:300]:assert read(PARENT/'queries'/f'{row["sample_key"]}.json')['complete']
  files=[]
  for source in (ROOT/'_lib').glob('probability_completion_*'):
   if not source.is_file():continue
   if source.suffix=='.py':py_compile.compile(str(source),doraise=True)
   target=OUT/'code'/source.name;target.write_bytes(source.read_bytes());files.append(dict(source=str(source),snapshot=str(target),sha256=sha(target)))
  for source in [PARENT/'code/probability_budget_core.py',PARENT/'code/probability_budget_inputs.py',EMBED_SOURCE]:files.append(dict(source=str(source),snapshot=str(source),sha256=sha(source)))
  write(OUT/'metadata/manifest.json',dict(settings=SETTINGS,settings_hash=digest(SETTINGS),cohort=str(COHORT),cohort_sha256=sha(COHORT),files=files,references=REFERENCES,timestamp=stamp()));write(OUT/'metadata/config.json',SETTINGS);write(OUT/'metadata/status.json',dict(state='prepared',stage='prepared',active_queries=300,completed=0,total=900,timestamp=stamp()))
  models=[]
  for name in (SETTINGS['verifier'],SETTINGS['generator'],SETTINGS['embedding']):
   path=BACKBONES/name;index=read(path/'model.safetensors.index.json');weights=sorted(set(index['weight_map'].values()));assert all((path/f).exists() for f in weights);models.append(dict(name=name,path=str(path),config_sha256=sha(path/'config.json'),tokenizer_sha256=sha(path/'tokenizer.json'),index_sha256=sha(path/'model.safetensors.index.json'),weight_files=[dict(name=f,bytes=(path/f).stat().st_size) for f in weights]))
  write(OUT/'metadata/models.json',dict(models=models,registry_sha256=sha(Path('/home/irteam/backbones/manifests/registry.json'))));print('COMPLETION_PREPARED',OUT)

def launch():
 prior=OUT/'metadata/launcher.json'
 if prior.exists():
  pid=read(prior)['pid']
  try:
   if b'probability_completion_manage.py' in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError('Already running')
  except FileNotFoundError:pass
 env=dict(os.environ,PYTHONPATH=str(PARENT/'code')+':'+str(ROOT/'_lib'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with (OUT/'logs/supervisor.log').open('ab',buffering=0) as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(OUT/'code/probability_completion_manage.py'),'supervise'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 write(prior,dict(pid=child.pid,log=str(OUT/'logs/supervisor.log'),detached=True,timestamp=stamp()));print('COMPLETION_STARTED',child.pid)

def supervise():
 with (OUT/'metadata/supervisor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for attempt in range(3):
   worker=subprocess.Popen([sys.executable,'-u',str(OUT/'code/probability_completion_runtime.py')],cwd=ROOT,stdin=subprocess.DEVNULL);code=worker.wait()
   if code==0:return
   error=read(OUT/'metadata/status.json').get('error','')
   if attempt==2 or not(code in (-9,137) or 'OutOfMemory' in error):raise RuntimeError(f'Completion worker failed: {code}; metadata/status.json')
   print('RETRY',attempt+1,code,flush=True);time.sleep(10)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','supervise']);a=p.parse_args();globals()[a.action]()
