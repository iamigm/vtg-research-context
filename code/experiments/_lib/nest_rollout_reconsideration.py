"""Nest the completed reconsideration analysis under its deterministic trajectories."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
P=Path('/home/irteam/experiments/zero-shot')
sys.path.insert(0,str(P/'_lib'))
from result_layout import ROOT as LAYOUT_ROOT

def main():
 old=P/'artifacts/runs/rollout/reconsideration'
 record=P/'artifacts/maintenance/reconsideration_nested.json'
 if not old.exists():
  assert record.exists();print('Already nested');return
 # Use exact per-file hashes for all experiment data whose bytes must remain unchanged.
 frozen={}
 for f in old.rglob('*'):
  if f.is_file() and ('responses' in f.parts or 'token_traces' in f.parts or 'logs' in f.parts or f.suffix=='.csv'):
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
   frozen[str(f)]=dict(sha256=h.hexdigest(),bytes=f.stat().st_size)
 print('Recorded immutable outputs:',len(frozen),flush=True)
 moves={}
 for tree in ('results','artifacts/runs'):
  for scope in ('300','600','overview'):
   src=P/tree/'rollout/reconsideration/pilot'/scope;dst=P/tree/'rollout/deterministic/pilot'/scope/'reconsideration'
   if src.exists():moves[str(src)]=str(dst)
  src=P/tree/'rollout/reconsideration/shared';dst=P/tree/'rollout/deterministic/shared/reconsideration'
  if src.exists():moves[str(src)]=str(dst)
 layout_path=LAYOUT_ROOT/'artifacts/maintenance/layout.json';layout=json.loads(layout_path.read_text());layout['moves'].update(moves)
 layout_path.write_text(json.dumps(layout,ensure_ascii=False,indent=2)+'\n')
 from result_layout import relocate_path,rewrite_document
 for src,dst in moves.items():
  assert not Path(dst).exists(),dst
  Path(dst).parent.mkdir(parents=True,exist_ok=True);shutil.move(src,dst)
 # Rewrite relative links from their original document base, and absolute derived references.
 def old_file(f):
  value=str(f)
  for src,dst in sorted(moves.items(),key=lambda kv:len(kv[1]),reverse=True):
   if value==dst or value.startswith(dst+'/'):return Path(src+value[len(dst):])
  return f
 for f in (P/'results').rglob('*'):
  if f.is_file() and f.suffix in ('.md','.html') and 'data-family-shell' not in f.read_text()[:2000]:
   text=f.read_text();new=rewrite_document(text,old_file(f),f)
   if new!=text:f.write_text(new)
 def convert(value):
  if isinstance(value,str):
   for src,dst in sorted(moves.items(),key=lambda kv:len(kv[0]),reverse=True):value=value.replace(src,dst)
   return value
  if isinstance(value,list):return [convert(v) for v in value]
  if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
  return value
 for scope in ('300','600','overview'):
  root=P/'artifacts/runs/rollout/deterministic/pilot'/scope/'reconsideration'
  for category in ('web','metrics','metadata'):
   for f in (root/category).glob('*.json'):
    if f.name in ('config.json','source_manifest.json','provenance.json'):continue
    value=json.loads(f.read_text());new=convert(value)
    if new!=value:f.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n')
 for f in (P/'artifacts/common/registry').glob('*.json'):
  value=json.loads(f.read_text());new=convert(value)
  if new!=value:f.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n')
 for name in ('README.md','artifacts/README.md'):
  f=P/name;text=f.read_text()
  for src,dst in sorted(moves.items(),key=lambda kv:len(kv[0]),reverse=True):
   text=text.replace(str(Path(src).relative_to(P)),str(Path(dst).relative_to(P)))
  text=text.replace('results/rollout/reconsideration/viewer.html','results/rollout/deterministic/viewer.html?view=pilot/600/reconsideration/viewer.html')
  f.write_text(text)
 # Remove only obsolete generated navigation and empty directories; no compatibility links.
 for root in (P/'results/rollout/reconsideration',old):
  for f in root.rglob('*'):
   if f.is_file():
    assert f.name in ('viewer.html','view_manifest.json'),f
    f.unlink()
  for d in sorted(root.rglob('*'),key=lambda x:len(x.parts),reverse=True):
   if d.is_dir():d.rmdir()
  root.rmdir()
 checked=0
 for src,expected in frozen.items():
  f=relocate_path(Path(src));h=hashlib.sha256()
  with f.open('rb') as stream:
   for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
  assert f.stat().st_size==expected['bytes'] and h.hexdigest()==expected['sha256'],f
  checked+=1
 record.parent.mkdir(parents=True,exist_ok=True)
 record.write_text(json.dumps(dict(passed=True,moves=moves,immutable_files_checked=checked,immutable_bytes=sum(x['bytes'] for x in frozen.values()),responses_traces_logs_csv_preserved=True,compatibility_links=False),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(passed=True,immutable_files_checked=checked,moves=len(moves))),flush=True)
if __name__=='__main__':main()
