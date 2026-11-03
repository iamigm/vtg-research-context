"""Move completed rollout runs into research families; retain immutable records."""
import ast,hashlib,json,os,re
from pathlib import Path
from result_layout import ROOT,relocate_path,rewrite_document,rewrite_reference,layout,rules
P=ROOT/'zero-shot';STATE=ROOT/'artifacts/maintenance/rollout_views_20261104'
RUNS={
 'timelens_rollout':'rollout/stochastic/pilot/300',
 'conditional_rollout_budget_pilot30_v1':'rollout/conditional_budget/pilot/30',
 'deterministic_rollout_pilot300_v2':'rollout/deterministic/v2/pilot/300',
 'deterministic_rollout_pilot600_v2':'rollout/deterministic/v2/pilot/600',
 'deterministic_rollout_pilot900_v2':'rollout/deterministic/v2/pilot/900',
 'deterministic_rollout_additional600_v2':'rollout/deterministic/v2/pilot/900/additional600',
}
MOVES={str(P/base/a):str(P/base/b) for base in ['results','artifacts/runs'] for a,b in RUNS.items()}
def patch_code(s):
 tree=ast.parse(s);parents={c:n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)};lines=s.splitlines(keepends=True);edits=[]
 def pos(n,end=False):
  ln=n.end_lineno if end else n.lineno;col=n.end_col_offset if end else n.col_offset
  return sum(map(len,lines[:ln-1]))+len(lines[ln-1].encode()[:col].decode())
 for n in ast.walk(tree):
  parent=parents.get(n)
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Path':
   if isinstance(parent,ast.Call) and isinstance(parent.func,ast.Name) and parent.func.id=='relocate_path':continue
   edits.extend([(pos(n),pos(n),'relocate_path('),(pos(n,True),pos(n,True),')')])
 for a,b,t in sorted(edits,reverse=True):s=s[:a]+t+s[b:]
 if edits and not re.search(r'from result_layout import .*relocate_path',s):
  tree=ast.parse(s);at=0
  for i,n in enumerate(tree.body):
   if (i==0 and isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) or isinstance(n,ast.ImportFrom) and n.module=='__future__':at=sum(map(len,s.splitlines(keepends=True)[:n.end_lineno]))
   else:break
  s=s[:at]+'from result_layout import relocate_path\n'+s[at:]
 ast.parse(s);return s

def apply():
 if (STATE/'migration.json').exists():raise SystemExit('Already moved')
 STATE.mkdir(parents=True,exist_ok=True)
 for run in RUNS:
  o=json.loads((P/'artifacts/runs'/run/'metadata/status.json').read_text());assert o['state']=='complete',run
 distribution=json.loads((P/'artifacts/runs/deterministic_rollout_pilot900_v2/metadata/distribution_integration.json').read_text());assert distribution['state']=='complete'
 lfile=ROOT/'artifacts/maintenance/layout.json';obj=json.loads(lfile.read_text());obj['moves'].update(MOVES)
 obj['presentation_moves'].update({a:b for a,b in MOVES.items() if '/results/' in a});obj['document_roots'] += [a for a in MOVES if '/results/' in a]
 lfile.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');layout.cache_clear();rules.cache_clear()
 for v in MOVES.values():assert relocate_path(Path(v))==Path(v)
 preserved={};hashes={};scripts={};json_pages={};symlinks=[]
 for run in RUNS:
  report=P/'results'/run
  for html in report.glob('*.html'):
   for src in re.findall(r'<script[^>]*src=[\'\"]([^\'\"]+)',html.read_text()):
    if src.startswith(('http:','https:','/')):continue
    f=Path(os.path.abspath(html.parent/src.split('?')[0]))
    if f.exists():scripts[f]=(html,relocate_path(html))
  for base in [report,P/'artifacts/runs'/run]:
   for folder,dirs,files in os.walk(base,followlinks=False):
    for name in files:
     f=Path(folder,name)
     if f.is_symlink():symlinks.append((str(f),str((f.parent/os.readlink(f)).resolve())));continue
     st=f.stat();preserved[str(relocate_path(f))]=[st.st_ino,st.st_size]
     if f.suffix=='.csv' or (f.suffix=='.json' and any(x in f.parts for x in ['rollouts','trees'])):
      hashes[str(relocate_path(f))]=hashlib.sha256(f.read_bytes()).hexdigest()
   # Browser payload paths resolve against this run's HTML, not against their JSON file.
  art=P/'artifacts/runs'/run
  for f in (art/'web').rglob('*.json'):
   json_pages[f]=(report,relocate_path(report))
 changed=[];json_updates=[]
 def edit(f,text):
  before=f.read_text()
  if before==text:return
  backup=STATE/'source_backup'/f.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True);backup.write_text(before)
  f.write_text(text);changed.append(str(relocate_path(f)))
 for run in RUNS:
  for base in [P/'results'/run,P/'artifacts/runs'/run]:
   for folder,dirs,files in os.walk(base):
    dirs[:]=[d for d in dirs if d not in {'code','token_traces','rollouts','trees','inputs','media','logs'}]
    for name in files:
     f=Path(folder,name)
     if f.is_symlink() or f.suffix not in {'.md','.html','.js','.css'}:continue
     old,new=scripts.get(f,(f,relocate_path(f))) if f.suffix=='.js' else (f,relocate_path(f))
     edit(f,rewrite_document(f.read_text(),old,new))
 from update_result_json_paths import numeric_tree
 for f,(old,new) in json_pages.items():
  before=json.loads(f.read_text())
  def update(x):
   if isinstance(x,dict):return {k:update(v) for k,v in x.items()}
   if isinstance(x,list):return [update(v) for v in x]
   return rewrite_reference(x,old,new) if isinstance(x,str) and '/' in x and not re.search(r'\s',x) else x
  after=update(before);assert numeric_tree(before)==numeric_tree(after)
  if before!=after:edit(f,json.dumps(after,ensure_ascii=False,indent=2)+'\n');json_updates.append(str(relocate_path(f)))
 for f in (P/'_lib').glob('*.py'):
  if 'rollout' not in f.name or f.is_symlink():continue
  text=patch_code(f.read_text())
  if f.name=='deterministic_rollout_viewer.py':text=text.replace("REPORT=relocate_path(ROOT/'results'/OUT.name)","REPORT=relocate_path(ROOT/'results'/f'deterministic_rollout_pilot{pilot_size}_v2')")
  if f.name=='rollout_pilot_html_watch.py':text=text.replace("report=relocate_path(ROOT/'results'/out.name)","report=relocate_path(ROOT/'results'/f'deterministic_rollout_pilot{n}_v2')")
  ast.parse(text);edit(f,text)
 for f in [*P.glob('*.md'),P/'artifacts/README.md']:
  edit(f,rewrite_document(f.read_text(),f,f))
 for f in (P/'artifacts/common/registry').glob('*.json'):
  o=json.loads(f.read_text());value=o.get('path')
  if value:o['path']=os.path.relpath(relocate_path(P/value),P)
  if o!=json.loads(f.read_text()):edit(f,json.dumps(o,ensure_ascii=False,indent=2)+'\n')
 # The900 container is placed before its additional600 source partition.
 for a,b in sorted(MOVES.items(),key=lambda x:len(x[1])):
  src,dst=Path(a),Path(b);assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
 for old,target in symlinks:
  link=relocate_path(Path(old));dest=relocate_path(Path(target))
  if link.is_symlink():link.unlink();link.symlink_to(os.path.relpath(dest,link.parent))
 (STATE/'migration.json').write_text(json.dumps(dict(moves=MOVES,preserved=preserved,hashes=hashes,changed=changed,json_updates=json_updates),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'moved_roots':len(MOVES),'preserved_files':len(preserved),'response_metric_hashes':len(hashes),'updated':len(changed)}))
if __name__=='__main__':apply()
