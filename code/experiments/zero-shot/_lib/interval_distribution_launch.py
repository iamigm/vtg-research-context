"""Start the frozen experiment in its own session; stdout and errors persist."""
import os,subprocess,sys
from interval_distribution_core import ROOT,OUT,stamp,write,read

def main():
 p=OUT/'metadata/launcher.json'
 if p.exists():
  previous=read(p)
  try:
   command=open(f'/proc/{previous["pid"]}/cmdline','rb').read()
   if b'interval_distribution_runtime.py' in command:raise RuntimeError(f'Experiment already running: PID {previous["pid"]}')
  except FileNotFoundError:pass
 entry=OUT/'code/interval_distribution_runtime.py';assert entry.is_file()
 env=dict(os.environ,PYTHONPATH=str(ROOT/'_lib'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 with (OUT/'logs/supervisor.log').open('ab',buffering=0) as stream:
  process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(entry)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 write(p,dict(pid=process.pid,entry=str(entry),python=str(ROOT/'.venv/bin/python'),log=str(OUT/'logs/supervisor.log'),timestamp=stamp(),detached=True))
 print(f'STARTED PID={process.pid} LOG={OUT}/logs/supervisor.log',flush=True)

if __name__=='__main__':main()
