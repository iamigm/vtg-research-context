"""Apply the reviewed layout, update consumers, and record preservation evidence."""
from organize_results import *
from result_layout import relocate_path, rewrite_document, original_path
import sys,tokenize,io,collections
D=json.loads((STATE/'layout.json').read_text());MOVES=D['moves']
def files_all():
 for base,ds,fs in os.walk(ROOT):
  ds[:]=[d for d in ds if d not in SKIP and not (Path(base)==ROOT and d=='similarity') and not Path(base,d).is_symlink() and not (Path(base)==ROOT/'artifacts')]
  for f in fs:yield Path(base,f)
def position(src,node,end=False):
 lines=src.splitlines(keepends=True);ln=node.end_lineno if end else node.lineno;col=node.end_col_offset if end else node.col_offset
 return sum(len(x) for x in lines[:ln-1])+len(lines[ln-1].encode()[:col].decode())
def patch_code(src,old,new):
 tree=ast.parse(src);parents={c:n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)};edits=[]
 def ispath(n):
  if isinstance(n,ast.Name):return n.id.lower() in {'root','out','r','results','query','generation','hidden','dynamics','spatial','base','dest','output','folder','directory','zeroshot','experiment','resources','report','data','logs','cfgdir','here','p','path'} or n.id.isupper()
  if isinstance(n,ast.Constant):return isinstance(n.value,str)
  if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div):return ispath(n.left) or ispath(n.right)
  if isinstance(n,ast.Call):return isinstance(n.func,ast.Name) and n.func.id=='Path' or isinstance(n.func,ast.Attribute) and n.func.attr=='resolve'
  return isinstance(n,(ast.Attribute,ast.Subscript)) and ispath(n.value)
 for n in ast.walk(tree):
  if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div) and ispath(n):
   par=parents.get(n)
   if isinstance(par,ast.BinOp) and isinstance(par.op,ast.Div):continue
   edits.extend([(position(src,n),position(src,n),'relocate_path('),(position(src,n,True),position(src,n,True),')')])
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Path':
   edits.extend([(position(src,n),position(src,n),'relocate_path('),(position(src,n,True),position(src,n,True),')')])
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text' and n.args:
   obj=n.func.value;start=position(src,n);endobj=position(src,obj,True)
   # Keep original argument formatting and any routed path expression.
   edits.extend([(start,start,'write_result_text('),(position(src,n.func,True)-len('.write_text'),position(src,n.func,True)+1,', ')])
 if old!=new:
  for n in ast.walk(tree):
   if isinstance(n,ast.Name) and n.id=='__file__':edits.append((position(src,n),position(src,n,True),repr(str(old))))
 if not edits:return src
 # Replacements before insertions at the same offset preserve expression boundaries.
 for a,b,t in sorted(edits,key=lambda x:(x[0],x[1]),reverse=True):src=src[:a]+t+src[b:]
 tree=ast.parse(src)
 at=len(src.splitlines(keepends=True)[0]) if src.startswith('#!') else 0
 for i,n in enumerate(tree.body):
  if i==0 and isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) or isinstance(n,ast.ImportFrom) and n.module=='__future__':at=position(src,n,True)+1
  else:break
 src=src[:at]+'from result_layout import relocate_path, write_result_text\n'+src[at:]
 ast.parse(src)
 return src
if __name__=='__main__':
 if (STATE/'changed_files.json').exists():raise SystemExit('Layout already applied; use validate_result_layout.py')
 files=list(files_all());identity=[];hashes={};before_links=[]
 for p in files:
  if p.is_symlink():continue
  try:st=p.stat()
  except FileNotFoundError:continue
  identity.append((str(p),st.st_ino,st.st_size,st.st_mtime_ns))
  if p.suffix in {'.csv','.npz','.npy'} or (p.suffix=='.json' and any(w in p.name for w in ['summary','metrics','paired','result'])):
   if st.st_size<25_000_000:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 (STATE/'before_identity.json').write_text(json.dumps(identity)+'\n');(STATE/'before_metrics_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
 # Preserve relative symlink destinations before moving their parents.
 links=[]
 for p in files:
  if p.is_symlink():links.append((p,os.path.abspath(p.parent/os.readlink(p))))
 # Rewrite consumers before moving their containers.
 changed=[]
 for old in files:
  if old.is_symlink() or old.name in {'AGENTS.md'} or 'provenance' in old.parts or 'migration_to_live' in old.parts or old.is_relative_to(ROOT/'_lib'):continue
  new=relocate_path(old)
  if old.suffix not in {'.py','.html','.md','.js','.css','.sh'}:continue
  try:src=old.read_text()
  except (UnicodeError,OSError):continue
  if old.suffix=='.py':
   dst=patch_code(src,old,new)
  else:dst=rewrite_document(src,old,new)
  if src!=dst:
   backup=STATE/'source_backup'/old.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True);backup.write_text(src)
   old.write_text(dst);changed.append(str(new))
 # Moves are disjoint except nested independent report roots, which are retained.
 for old,new in sorted(MOVES.items(),key=lambda x:len(x[0]),reverse=True):
  a=Path(old);b=Path(new);b.parent.mkdir(parents=True,exist_ok=True)
  if b.exists() or b.is_symlink():raise RuntimeError(f'Collision {b}')
  a.rename(b)
 for old,target in links:
  new=relocate_path(old);dst=relocate_path(Path(target))
  if new.is_symlink():new.unlink();new.symlink_to(os.path.relpath(dst,new.parent))
 # Modules import the same routing helper via a local symlink; no PYTHONPATH setup required.
 for f in changed:
  p=Path(f)
  if p.suffix=='.py':
   link=p.parent/'result_layout.py'
   if not link.exists():link.symlink_to(os.path.relpath(ROOT/'_lib/result_layout.py',link.parent))
 for f in D['deleted_previews']:Path(f).unlink()
 (STATE/'changed_files.json').write_text(json.dumps(changed,indent=2)+'\n')
 print(json.dumps(dict(moved=len(MOVES),updated_consumers=len(changed),preserved_numeric_hashes=len(hashes),tracked_files=len(identity)),indent=2))
