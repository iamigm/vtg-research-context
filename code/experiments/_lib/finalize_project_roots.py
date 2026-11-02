"""Remove temporary project aliases once live similarity processes have exited."""
import json,os,time,subprocess,sys,traceback,fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent.parent;STATE=ROOT/'artifacts/maintenance'
def now():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def save(data):
 p=STATE/'project_roots_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(timestamp=now(),**data),indent=2)+'\n');tmp.replace(p)
def active():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   args=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]
   if not args or (p/'stat').read_text().split(') ',1)[1].startswith('Z'):continue
   normalized=[a.replace('/home1/irteam/','/home/irteam/') for a in args]
   # Python workers may keep the old environment prefix throughout their lifetime.
   old_env=normalized[0].startswith('/home/irteam/experiments/shared/')
   similarity=any(a.startswith('/home/irteam/experiments/similarity/_lib/') and a.endswith('.py') for a in normalized[1:])
   if old_env or similarity:found.append(dict(pid=int(p.name),command=args))
  except (OSError,UnicodeError):continue
 return found
def main():
 lock=(STATE/'project_roots_finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 aliases=json.loads((STATE/'project_roots.json').read_text())['temporary_aliases'];last=None
 while True:
  processes=active();pids=sorted(x['pid'] for x in processes)
  save(dict(stage='waiting_for_live_experiment' if pids else 'removing_aliases',active_processes=processes,temporary_aliases=aliases))
  if pids!=last:print(now(),'waiting for active processes',pids,flush=True);last=pids
  if not pids:break
  time.sleep(20)
 removed=[]
 for name,target in aliases.items():
  p=Path(name)
  if p.is_symlink() and p.resolve()==Path(target).resolve():p.unlink();removed.append(name)
  elif p.exists():raise RuntimeError('Alias was replaced; refusing to remove '+name)
 bridge=ROOT/'similarity/artifacts/common/logs'
 if bridge.is_dir() and not any(bridge.iterdir()):bridge.rmdir()
 print(now(),'removed aliases',removed,flush=True)
 # Revalidate preserved science and report links after the old paths no longer exist.
 result=subprocess.run([sys.executable,str(ROOT/'_lib/validate_result_layout.py')],check=True)
 verification=json.loads((STATE/'validation.json').read_text())
 if verification['errors'] or verification['broken_html_refs']:raise RuntimeError('Post-removal validation found errors')
 doc=ROOT/'README.md'
 text=doc.read_text()
 start=text.find('실행 중인 similarity 작업의 기존 경로를 위해')
 if start>=0:
  end=text.find('\n\n',start)
  if end<0:end=len(text)
  text=text[:start]+'기존 경로의 임시 호환 링크는 실행 작업 종료 후 제거했다. [자동 정리 상태](artifacts/maintenance/project_roots_status.json) · [자동 정리 로그](artifacts/maintenance/project_roots_finalize.log).'+text[end:]
  doc.write_text(text)
 save(dict(stage='complete',removed_aliases=removed,validation_passed=True))
 print(now(),'project-root cleanup complete; post-removal validation passed',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  save(dict(stage='failed',error=traceback.format_exc()));raise
