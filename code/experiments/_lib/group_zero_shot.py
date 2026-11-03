"""Group zero-shot results and matching artifacts, preserving rollout runs."""
import ast, hashlib, json, os, re, sys
from pathlib import Path
ROOT=Path('/home/irteam/experiments'); PROJECT=ROOT/'zero-shot'
STATE=PROJECT/'artifacts/maintenance/grouping_20261104'
RUNS={
 'all_zeroshot_v1':'baseline/all_zeroshot_v1',
 'input_order':'baseline/input_order',
 'thinking_vs_instruct_v1':'baseline/thinking_vs_instruct_v1',
 'timelens_span_head':'uncertainty/timelens_span_head',
 'timelens_top30':'uncertainty/token_analysis/timelens_top30',
 'timelens_top30_full':'uncertainty/token_analysis/timelens_top30_full',
 'spatial_budget_pilot900_v1':'sampling/spatial_allocation/spatial_budget_pilot900_v1',
 'spatial_sampling_pilot900_v2':'sampling/spatial_allocation/spatial_sampling_pilot900_v2',
 'fps_spatial_control_pilot300_v1':'sampling/fps_spatial_control_pilot300_v1',
 'selected_frame_cache_migration_v1':'operations/selected_frame_cache_migration_v1',
 'timelens_top30/mmco_diagnosis':'operations/mmco_diagnosis',
 'timelens_top30_full/performance':'operations/inference_performance',
}
MOVES={str(PROJECT/base/old):str(PROJECT/base/new) for base in ['results','artifacts/runs'] for old,new in RUNS.items()}

def canon(p):return str(p).replace('/home1/irteam/','/home/irteam/')
def route(p):
 s=canon(p)
 for old,new in sorted(MOVES.items(),key=lambda x:len(x[0]),reverse=True):
  if s==old or s.startswith(old+'/'):return Path(new+s[len(old):])
 return Path(s)
def replace_paths(s):
 for base in ['results','artifacts/runs']:
  for old,new in sorted(RUNS.items(),key=lambda x:len(x[0]),reverse=True):
   s=re.sub(re.escape(base+'/'+old)+r'(?=/|[\s\'"`)>\]}]|$)',lambda m:base+'/'+new,s)
 return s

def patch_source(s):
 s=replace_paths(s)
 tree=ast.parse(s);parents={c:n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
 lines=s.splitlines(keepends=True)
 def pos(n,end=False):
  ln=n.end_lineno if end else n.lineno;col=n.end_col_offset if end else n.col_offset
  return sum(map(len,lines[:ln-1]))+len(lines[ln-1].encode()[:col].decode())
 edits=[];need=False
 for n in ast.walk(tree):
  parent=parents.get(n)
  if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div):
   if isinstance(parent,ast.BinOp) and isinstance(parent.op,ast.Div):continue
   if isinstance(parent,ast.Call) and isinstance(parent.func,ast.Name) and parent.func.id=='relocate_path':continue
   constants=[c.value for c in ast.walk(n) if isinstance(c,ast.Constant) and isinstance(c.value,str)]
   if not any(c=='results' or c=='artifacts/runs' or c.startswith(('results/','artifacts/runs/')) for c in constants):continue
   edits += [(pos(n),pos(n),'relocate_path('),(pos(n,True),pos(n,True),')')];need=True
 for a,b,v in sorted(edits,reverse=True):s=s[:a]+v+s[b:]
 if need and not re.search(r'from result_layout import .*relocate_path',s):
  tree=ast.parse(s);at=0
  for i,n in enumerate(tree.body):
   if (i==0 and isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) or (isinstance(n,ast.ImportFrom) and n.module=='__future__'):
    at=sum(map(len,s.splitlines(keepends=True)[:n.end_lineno]))
   else:break
  s=s[:at]+'from result_layout import relocate_path\n'+s[at:]
 ast.parse(s);return s

def apply():
 if STATE.exists():raise SystemExit('Migration already started; inspect grouping journal')
 STATE.mkdir(parents=True)
 # Add persistent routing for frozen metadata and historical code snapshots.
 lp=ROOT/'artifacts/maintenance/layout.json';layout=json.loads(lp.read_text())
 layout['moves'].update(MOVES)
 layout.setdefault('presentation_moves',{}).update({a:b for a,b in MOVES.items() if '/results/' in a})
 layout['document_roots']=list(dict.fromkeys(layout.get('document_roots',layout['roots'])+[a for a in MOVES if '/results/' in a]))
 lp.write_text(json.dumps(layout,ensure_ascii=False,indent=2)+'\n')
 from result_layout import rewrite_document,layout as get_layout,rules
 get_layout.cache_clear();rules.cache_clear()
 changed=[];preserved=[];symlinks=[]
 def edit(f,s):
  before=f.read_text()
  if before==s:return
  dest=route(f)
  backup=STATE/'source_backup'/f.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True);backup.write_text(before)
  f.write_text(s);changed.append(str(dest))
 # Presentation files and web bundles need relative links rebased at their new home.
 for old in [PROJECT/'results',PROJECT/'artifacts/runs']:
  for base,dirs,files in os.walk(old):
   dirs[:]=[d for d in dirs if 'rollout' not in d and d not in {'code','token_traces','samples','responses','inputs','frames','logs'}]
   for name in files:
    f=Path(base,name)
    if f.is_symlink():symlinks.append((str(f),os.path.abspath(f.parent/os.readlink(f))));continue
    if f.suffix in {'.md','.html','.js','.css'}:
     edit(f,rewrite_document(f.read_text(),f,route(f)))
    elif f.name=='media_manifest.json':
     before=json.loads(f.read_text())
     # Media paths resolve from the result page, not from the manifest folder.
     rel=f.relative_to(PROJECT/'artifacts/runs');run='timelens_top30_full' if str(rel).startswith('timelens_top30_full/') else 'timelens_top30'
     oldbase=PROJECT/'results'/run;newbase=route(oldbase)
     from result_layout import rewrite_reference
     def rewrite(x):
      if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
      if isinstance(x,list):return [rewrite(v) for v in x]
      return rewrite_reference(x,oldbase,newbase) if isinstance(x,str) and '/' in x and not re.search(r'\s',x) else x
     after=rewrite(before)
     if after!=before:edit(f,json.dumps(after,ensure_ascii=False,indent=2)+'\n')
 # Numeric/raw records are moved byte-for-byte. Record inode/size for every file.
 for run in RUNS:
  for root in [PROJECT/'results'/run,PROJECT/'artifacts/runs'/run]:
   if root.exists():
    for base,dirs,files in os.walk(root):
     for name in files:
      f=Path(base,name)
      if not f.is_symlink():
       st=f.stat();preserved.append([str(route(f)),st.st_ino,st.st_size])
 # Update native sources, cross-project consumers, registry and docs; omit frozen snapshots.
 for project in ['zero-shot','attention','similarity','_lib']:
  base=ROOT/project
  candidates=list((base/'_lib').rglob('*.py')) if project!='_lib' else list(base.glob('*.py'))
  if project=='zero-shot':candidates += [PROJECT/'run.py']
  for f in candidates:
   if f.is_symlink() or f.name in {'group_zero_shot.py','result_layout.py'}:continue
   s=f.read_text();new=patch_source(s)
   if new!=s:edit(f,new)
  if project!='_lib':
   for f in base.glob('*.md'):edit(f,rewrite_document(f.read_text(),f,f))
 for f in [PROJECT/'artifacts/README.md',PROJECT/'artifacts/common/predictions/legacy/README.md']:
  edit(f,rewrite_document(f.read_text(),f,f))
 for f in (PROJECT/'artifacts/common/registry').glob('*.json'):
  edit(f,replace_paths(f.read_text()))
 # Nested roots first: operational diagnostics are pulled out before moving their parent.
 for old,new in sorted(MOVES.items(),key=lambda x:len(x[0]),reverse=True):
  src,dst=Path(old),Path(new)
  if not src.exists():continue
  assert not dst.exists(),dst
  dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
  if src==PROJECT/'artifacts/runs/fps_spatial_control_pilot300_v1':src.symlink_to(os.path.relpath(dst,src.parent))
 # Resolve symlinks relative to their relocated containers.
 for old,target in symlinks:
  f=route(old)
  if f.is_symlink():f.unlink();f.symlink_to(os.path.relpath(route(target),f.parent))
 (STATE/'migration.json').write_text(json.dumps({'moves':MOVES,'changed':changed,'preserved':preserved,'temporary_links':[str(PROJECT/'artifacts/runs/fps_spatial_control_pilot300_v1')]},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'moves':len(MOVES),'changed':len(changed),'preserved_files':len(preserved)}))
if __name__=='__main__':apply()
