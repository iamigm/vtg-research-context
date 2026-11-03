"""Apply theme/family/scope organization to the remaining experiment projects."""
import ast,hashlib,json,os,re,sys
from pathlib import Path
from result_layout import ROOT,canonical,layout,rules,relocate_path,rewrite_document,rewrite_reference
STATE=ROOT/'artifacts/maintenance/project_grouping_20261104'
PROJECT_RUNS={
 'attention':{
  'overview':'token_attention/mtla/pilot/300',
  'query':'token_attention/mtla/pilot/300/query',
  'generation':'token_attention/mtla/pilot/300/generation',
  'dynamics':'token_attention/mtla/pilot/300/dynamics',
  'hidden':'token_attention/mtla/pilot/300/hidden',
  'spatial':'token_attention/mtla/pilot/300/spatial',
  'timelens_order':'input_order/timelens_instruct/full/9404',
  'thinking_pilot_orders_v1':'input_order/timelens_thinking/pilot/300',
 },
 'similarity':{
  'pilot_300_v1':'frame_query/cosine_2fps/pilot/300',
  'full_9404_v1':'frame_query/cosine_2fps/full/9404',
  'pilot_300_16fps_cost_v1':'compute_cost/embedding_vs_generation_16fps/pilot/300',
 },
 'analysis/failure_mode_timestamp/failure_cases':{'pilot_100':'failure_modes/input_settings/pilot/200'},
 'dataset_metadata':{'20261101':'datasets/source_statistics/full/all_sources/20261101'},
 'audio_caption_resources_20260930':{'resource_survey':'resources/audio_caption/shared/20260930'},
}
MOVES={str(ROOT/project/base/a):str(ROOT/project/base/b) for project,runs in PROJECT_RUNS.items() for base in ['results','artifacts/runs'] for a,b in runs.items()}

def patch_source(source):
 # Existing consumers keep historical run IDs and use routing for dynamic paths.
 tree=ast.parse(source);parents={c:n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
 lines=source.splitlines(keepends=True)
 def pos(n,end=False):
  ln=n.end_lineno if end else n.lineno;col=n.end_col_offset if end else n.col_offset
  return sum(map(len,lines[:ln-1]))+len(lines[ln-1].encode()[:col].decode())
 edits=[];needed=set()
 for n in ast.walk(tree):
  parent=parents.get(n)
  if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div):
   if isinstance(parent,ast.BinOp) and isinstance(parent.op,ast.Div):continue
   if isinstance(parent,ast.Call) and isinstance(parent.func,ast.Name) and parent.func.id=='relocate_path':continue
   strings=[x.value for x in ast.walk(n) if isinstance(x,ast.Constant) and isinstance(x.value,str)]
   if any(x in ('results','artifacts/runs') or x.startswith(('results/','artifacts/runs/')) for x in strings):
    edits += [(pos(n),pos(n),'relocate_path('),(pos(n,True),pos(n,True),')')];needed.add('relocate_path')
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text' and n.args:
   expression=ast.get_source_segment(source,n.func.value)
   # Dynamic MD/HTML renderers need the same link rebasing on regeneration.
   if not re.search(r'\b(VIEW|REPORT|RESULT)\b',expression):continue
   edits.append((pos(n),pos(n.func,True)+1,'write_result_text('+expression+', '));needed.add('write_result_text')
 # Avoid overlapping edits for a path inside a rewritten write_text receiver.
 replacements=[e for e in edits if e[1]>e[0]]
 edits=[e for e in edits if not(e[0]==e[1] and any(a<=e[0]<=b for a,b,_ in replacements))]
 for a,b,t in sorted(edits,reverse=True):source=source[:a]+t+source[b:]
 imports={a.name for n in ast.walk(ast.parse(source)) if isinstance(n,ast.ImportFrom) and n.module=='result_layout' for a in n.names}
 missing=needed-imports
 if missing:
  tree=ast.parse(source);at=0
  for i,n in enumerate(tree.body):
   if (i==0 and isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) or (isinstance(n,ast.ImportFrom) and n.module=='__future__'):
    at=sum(map(len,source.splitlines(keepends=True)[:n.end_lineno]))
   else:break
  source=source[:at]+'from result_layout import '+', '.join(sorted(missing))+'\n'+source[at:]
 ast.parse(source);return source

def no_conflicting_aliases(obj):
 finals=list(MOVES.values());removed={old:next(target for target in finals if target.startswith(old+'/')) for old in list(obj['moves']) if any(target.startswith(old+'/') for target in finals)}
 for old in removed:obj['moves'].pop(old)
 for k,v in list(obj['moves'].items()):
  for old,new in sorted(removed.items(),key=lambda x:len(x[0]),reverse=True):
   if v==old or v.startswith(old+'/'):obj['moves'][k]=new+v[len(old):];break
 return removed

def apply():
 if STATE.exists():raise SystemExit('Migration journal exists; inspect before rerunning')
 STATE.mkdir(parents=True)
 file=ROOT/'artifacts/maintenance/layout.json';obj=json.loads(file.read_text());retired=no_conflicting_aliases(obj);obj['moves'].update(MOVES)
 obj['presentation_moves'].update({a:b for a,b in MOVES.items() if '/results/' in a});obj['document_roots'] += [a for a in MOVES if '/results/' in a]
 file.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');layout.cache_clear();rules.cache_clear()
 # Fail before modifying files if any historical routing prefix repeats.
 for new in MOVES.values():assert relocate_path(Path(new))==Path(new),new
 preserved={};numeric_hashes={};scripts={};json_bases={};symlinks=[]
 for project in PROJECT_RUNS:
  p=ROOT/project
  for base in [p/'results',p/'artifacts/runs']:
   for folder,dirs,files in os.walk(base,followlinks=False):
    for name in files:
     f=Path(folder,name)
     if f.is_symlink():symlinks.append((str(f),canonical((f.parent/os.readlink(f)).resolve())));continue
     st=f.stat();preserved[str(relocate_path(f))]=[st.st_ino,st.st_size]
     if f.suffix in {'.csv','.npz','.npy'} and st.st_size<30000000:numeric_hashes[str(relocate_path(f))]=hashlib.sha256(f.read_bytes()).hexdigest()
  for html in (p/'results').rglob('*.html'):
   for src in re.findall(r'<script[^>]*src=[\'\"]([^\'\"]+)',html.read_text()):
    if src.startswith(('https:','http:','/')):continue
    js=Path(os.path.abspath(html.parent/src.split('?')[0]))
    if js.exists():scripts[js]=(html,relocate_path(html))
  # Presentation JSON values use their owning HTML directory as URL base.
  for name in ['manifest.json','viewer_data.json','media_manifest.json']:
   for f in (p/'artifacts/runs').rglob(name):
    rel=f.relative_to(p/'artifacts/runs');run=rel.parts[0]
    if run not in PROJECT_RUNS[project]:continue
    # Restrict to known browser bundles; keep immutable raw source metadata intact.
    if not any(part in rel.parts for part in ['web','metadata']):continue
    if name=='manifest.json' and 'web' not in rel.parts:continue
    parent=rel.parent
    if parent.name in ('web','metadata'):parent=parent.parent
    old=p/'results'/parent
    if not any(old.glob('*.html')):old=p/'results'/run
    json_bases[f]=(old,relocate_path(old))
 changed=[];json_updates=[]
 def edit(f,text):
  before=f.read_text()
  if before==text:return
  backup=STATE/'source_backup'/f.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True);backup.write_text(before)
  f.write_text(text);changed.append(str(relocate_path(f)))
 for project in PROJECT_RUNS:
  p=ROOT/project
  for base in [p/'results',p/'artifacts/runs']:
   for folder,dirs,files in os.walk(base):
    dirs[:]=[d for d in dirs if d not in {'code','samples','arrays','frames','media','inputs','logs','sources','provenance','token_traces','responses','vendor'}]
    for name in files:
     f=Path(folder,name)
     if f.is_symlink() or f.suffix not in {'.md','.html','.js','.css'}:continue
     old,new=scripts.get(f,(f,relocate_path(f))) if f.suffix=='.js' else (f,relocate_path(f))
     edit(f,rewrite_document(f.read_text(),old,new))
  sources=list((p/'_lib').glob('*.py'))+list(p.glob('*.py'))
  for f in sources:
   if f.is_symlink() or f.name=='timestamp_inputs.py':continue
   edit(f,patch_source(f.read_text()))
  for f in [*p.glob('*.md'),p/'artifacts/README.md']:
   if f.exists():edit(f,rewrite_document(f.read_text(),f,f))
  helper=p/'_lib/result_layout.py'
  if not helper.exists():helper.symlink_to(os.path.relpath(ROOT/'_lib/result_layout.py',helper.parent))
 # Cross-project links include pure zero-shot reports and root guides.
 for p in [ROOT,ROOT/'zero-shot',ROOT/'analysis',ROOT/'analysis/failure_mode_timestamp']:
  files=list(p.glob('*.md'))
  if p.name=='zero-shot':files += [f for f in (p/'results').rglob('*.md') if 'rollout' not in str(f)]
  for f in files:edit(f,rewrite_document(f.read_text(),f,f))
 from update_result_json_paths import numeric_tree
 for f,(old,new) in json_bases.items():
  before=json.loads(f.read_text())
  def update(x):
   if isinstance(x,dict):return {k:update(v) for k,v in x.items()}
   if isinstance(x,list):return [update(v) for v in x]
   return rewrite_reference(x,old,new) if isinstance(x,str) and '/' in x and not re.search(r'\s',x) else x
  after=update(before)
  assert numeric_tree(before)==numeric_tree(after),f
  if after!=before:edit(f,json.dumps(after,ensure_ascii=False,indent=2)+'\n');json_updates.append(str(relocate_path(f)))
 # Registry entries are mutable path catalogs; hashes refer to preserved raw content.
 for f in (ROOT/'zero-shot/artifacts/common/registry').glob('*.json'):
  text=f.read_text()
  for old,new in sorted(MOVES.items(),key=lambda x:len(x[0]),reverse=True):text=text.replace(old,new).replace(old.removeprefix(str(ROOT/'zero-shot')+'/'),new.removeprefix(str(ROOT/'zero-shot')+'/')) if old.startswith(str(ROOT/'zero-shot')+'/') else text.replace(old,new)
  edit(f,text)
 # Place the integrated pilot300 root before its query/generation subanalyses.
 for old,new in sorted(MOVES.items(),key=lambda x:len(x[1])):
  src,dst=Path(old),Path(new)
  if not src.exists():continue
  assert not dst.exists(),dst
  dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
 for old,target in symlinks:
  link=relocate_path(Path(old));target=relocate_path(Path(target))
  if link.is_symlink():link.unlink();link.symlink_to(os.path.relpath(target,link.parent))
 # Remove only empty containers outside active rollout projects.
 for project in PROJECT_RUNS:
  for base in [ROOT/project/'results',ROOT/project/'artifacts/runs']:
   for folder,dirs,files in os.walk(base,topdown=False):
    f=Path(folder)
    if f!=base and not f.is_symlink():
     try:f.rmdir()
     except OSError:pass
 (STATE/'migration.json').write_text(json.dumps(dict(moves=MOVES,retired_aliases=retired,preserved=preserved,numeric_hashes=numeric_hashes,changed=changed,json_updates=json_updates),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'moved_roots':len(MOVES),'preserved_files':len(preserved),'metric_array_hashes':len(numeric_hashes),'updated_files':len(changed)}))
if __name__=='__main__':apply()
