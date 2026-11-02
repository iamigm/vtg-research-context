"""One artifact store per project: common resources and run-scoped data."""
import os,json,hashlib,re,ast,signal
from pathlib import Path
ROOT=Path('/home/irteam/experiments');STATE=ROOT/'artifacts/maintenance'
if (STATE/'artifact_hierarchy.json').exists():raise SystemExit('Already applied')
pid=json.loads((STATE/'project_roots_finalize_process.json').read_text())['pid']
try:os.kill(pid,signal.SIGTERM)
except ProcessLookupError:pass
D=json.loads((STATE/'layout.json').read_text());old_roots=D['roots'][:]
projects=[ROOT/x for x in ['attention','zero-shot','similarity','analysis/failure_mode_timestamp/failure_cases','audio_caption_resources_20260930']]
project_for=lambda p:max((r for r in projects if p.is_relative_to(r)),key=lambda r:len(str(r)))
moves={};operations=[];common_ops=[];presentation={};artifact_owners={}
def add(src,dst,kind='rename'):
 if src==dst:return
 moves[str(src)]=str(dst);operations.append((src,dst,kind))
# Cross-run resources belong in common/. Storage README stays at the artifact root.
for project in projects:
 art=project/'artifacts'
 if project.name.startswith('audio_caption'):continue
 if art.exists():
  for p in list(art.iterdir()):
   if p.name in {'README.md','common','runs'}:continue
   add(p,art/'common'/p.name)
# Every result run has one peer artifact directory; nested analyses preserve the run path.
active=ROOT/'similarity/results/full_9404_v1'
rs=list(dict.fromkeys([Path(r) for r in old_roots]+[active]))
for root in rs:
 project=project_for(root)
 if project.name.startswith('audio_caption'):
  name='resource_survey';dest=project/'results'/name;presentation[str(root)]=str(dest)
  for p in list(root.iterdir()):
   if p.is_file() and p.name in {'report.html','preview.png'}:add(p,dest/p.name)
  for p in list((project/'artifacts').iterdir()):
   if p.name in {'README.md','runs','common'}:continue
   add(p,project/'artifacts/runs'/name/p.name)
  artifact_owners[str(project/'artifacts')]=dict(old_view=str(root),new_view=str(dest),project=str(project))
  continue
 if root==ROOT/'attention/results':
  name='overview';dest=root/'overview';presentation[str(root)]=str(dest)
  for p in list(root.iterdir()):
   if p.is_file() and (p.suffix=='.md' or p.suffix=='.png' or p.name=='report.html'):add(p,dest/p.name)
 elif root.is_relative_to(project/'results'):
  name=str(root.relative_to(project/'results'));dest=root
 else:
  name='legacy_quantitative';dest=project/'results'/name;presentation[str(root)]=str(dest)
  for p in list(root.iterdir()):
   if p.is_file():add(p,dest/p.name)
 oldart=root/'artifacts';newart=project/'artifacts/runs'/name
 if oldart.is_dir() and not oldart.is_symlink():
  artifact_owners[str(oldart)]=dict(old_view=str(root),new_view=str(dest),project=str(project))
  # Collapse wrapper/category/category produced by the previous organization.
  for category in list(oldart.iterdir()):
   if category.is_dir() and not category.is_symlink():
    children=list(category.iterdir())
    if len(children)==1 and children[0].is_dir() and children[0].name==category.name:
     moves[str(children[0])]=str(newart/category.name)
  add(oldart,newart,'artifact')
# Resolve nested moves to their final destination before editing consumers.
D['moves'].update(moves);D['presentation_moves']=presentation
D['document_roots']=list(dict.fromkeys(D.get('document_roots',[])+old_roots+[str(active)]))
(STATE/'layout.json').write_text(json.dumps(D,indent=2)+'\n')
from result_layout import relocate_path,layout,rules,rewrite_document,rewrite_reference
layout.cache_clear();rules.cache_clear()
# Capture writable document assets only, excluding raw samples and vendor sources.
text_files=set();json_files={}
for oldart,owner in artifact_owners.items():
 art=Path(oldart)
 for base,dirs,fs in os.walk(art):
  dirs[:]=[d for d in dirs if d not in {'samples','arrays','frames','media','token_traces','records','qwen25','qwen3','clip','qwen','vendor','sources','provenance','__pycache__','migration_to_live'}]
  for f in fs:
   p=Path(base,f)
   if p.is_symlink():continue
   if p.suffix in {'.html','.md','.js','.css'}:text_files.add(p)
   if p.suffix=='.json' and p.name in {'media_manifest.json','viewer_data.json','source_manifest.json','report_sources.json','sources.json','source_hashes.json'}:json_files[p]=owner
for root in rs:
 text_files.update(p for p in root.glob('*') if p.is_file() and not p.is_symlink() and p.suffix in {'.html','.md','.js','.css'})
for project in projects:
 text_files.update(p for p in project.glob('*.md') if p.name!='AGENTS.md')
# Also update viewer bundles within per-sample data (small HTML/JS assets only).
for base in [ROOT/'analysis/failure_mode_timestamp/failure_cases/results/pilot_100/artifacts/data/cases']:
 if base.exists():text_files.update(p for p in base.rglob('*') if p.is_file() and p.suffix in {'.html','.js','.css','.md'})
text_edits=[]
for old in text_files:
 try:src=old.read_text()
 except (OSError,UnicodeError):continue
 dst=relocate_path(old);text=rewrite_document(src,old,dst)
 if src!=text:old.write_text(text);text_edits.append(str(dst))
# Path-bearing JSON documents keep every number and structural value unchanged.
def skeleton(x):
 if isinstance(x,dict):return {k:skeleton(v) for k,v in x.items()}
 if isinstance(x,list):return [skeleton(v) for v in x]
 return '<text>' if isinstance(x,str) else x
def semantic(x):return hashlib.sha256(json.dumps(skeleton(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def json_paths(x,owner):
 if isinstance(x,dict):return {k:json_paths(v,owner) for k,v in x.items()}
 if isinstance(x,list):return [json_paths(v,owner) for v in x]
 if isinstance(x,str) and '/' in x and not re.search(r'\s',x):
  for oldbase,newbase in [(Path(owner['old_view']),Path(owner['new_view'])),(Path(owner['project']),Path(owner['project']))]:
   out=rewrite_reference(x,oldbase,newbase)
   if out!=x:return out
 return x
json_edits=[]
for p,owner in json_files.items():
 before=json.loads(p.read_text());after=json_paths(before,owner)
 if before!=after:
  assert skeleton(before)==skeleton(after)
  json_edits.append(dict(old=str(p),path=str(relocate_path(p)),numeric_tree_sha256=semantic(before),before_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  p.write_text(json.dumps(after,ensure_ascii=False,indent=2)+'\n')
# Snapshot moved symlinks as logical destinations, without following dependency trees.
symlinks=[]
for src,dst,kind in operations:
 if src.is_dir() and not src.is_symlink():
  for base,dirs,fs in os.walk(src):
   dirs[:]=[d for d in dirs if d not in {'.git','vendor','__pycache__'} and not Path(base,d).is_symlink()]
   for f in fs:
    p=Path(base,f)
    if p.is_symlink():symlinks.append((p,Path(os.path.abspath(p.parent/os.readlink(p)))))
def merge(src,dst):
 if src.is_dir() and not src.is_symlink() and dst.is_dir():
  for p in list(src.iterdir()):merge(p,dst/p.name)
  src.rmdir()
 else:
  if dst.exists() or dst.is_symlink():raise RuntimeError('Collision '+str(dst))
  dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
# First move deepest artifact sources, then presentation files and common resources.
for src,dst,kind in sorted(operations,key=lambda t:len(str(t[0])),reverse=True):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if kind=='artifact':
  # Move collapsed categories directly; the rest retain their category names.
  dst.mkdir(exist_ok=True)
  for category in list(src.iterdir()):
   nested=category/category.name
   if str(nested) in moves:
    merge(nested,dst/category.name);category.rmdir()
   else:merge(category,dst/category.name)
  src.rmdir()
 else:merge(src,dst)
for old,target in symlinks:
 p=relocate_path(old);dest=relocate_path(target)
 if p.is_symlink():p.unlink();p.symlink_to(os.path.relpath(dest,p.parent))
# Only the actively written similarity output needs an in-place compatibility link.
alias=active/'artifacts';target=ROOT/'similarity/artifacts/runs/full_9404_v1';alias.symlink_to(os.path.relpath(target,alias.parent))
project_state=json.loads((STATE/'project_roots.json').read_text())
for old in list(project_state['temporary_aliases']):
 p=Path(old);target=relocate_path(p)
 if p.is_symlink():p.unlink();p.symlink_to(os.path.relpath(target,p.parent))
 project_state['temporary_aliases'][old]=str(target)
project_state['temporary_aliases'][str(alias)]=str(ROOT/'similarity/artifacts/runs/full_9404_v1')
(STATE/'project_roots.json').write_text(json.dumps(project_state,indent=2)+'\n')
# Canonical run roots plus category mappings must work when composed incrementally.
original=D['moves'].copy();aliases={}
for old,new in original.items():
 current=old;seen=set()
 while current not in seen:
  seen.add(current);probe=os.path.dirname(current)
  while probe and probe!='/':
   if probe in original:current=original[probe]+current[len(probe):];break
   probe=os.path.dirname(probe)
  else:break
 if current!=old:
  final=str(relocate_path(Path(new)))
  if current!=final:aliases[current]=final
D['moves'].update(aliases)
D['roots']=list(dict.fromkeys(presentation.get(r,r) for r in old_roots+[str(active)]))
D['previews']={presentation.get(k.rsplit('/',1)[0],k.rsplit('/',1)[0])+'/'+k.rsplit('/',1)[1]:str(relocate_path(Path(v))) for k,v in D['previews'].items()}
(STATE/'layout.json').write_text(json.dumps(D,indent=2)+'\n');layout.cache_clear();rules.cache_clear()
changed=json.loads((STATE/'changed_files.json').read_text());changed=[str(relocate_path(Path(p))) for p in changed];changed+=text_edits+[j['path'] for j in json_edits]
(STATE/'changed_files.json').write_text(json.dumps(sorted(set(changed)),indent=2)+'\n')
record=dict(moves=moves,presentation_moves=presentation,artifact_owners=artifact_owners,json_path_updates=json_edits,text_files_updated=len(text_edits),temporary_alias=str(alias))
(STATE/'artifact_hierarchy.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(artifact_stores=len(projects),run_roots=len(D['roots']),moves=len(moves),text_files_updated=len(text_edits),json_path_updates=len(json_edits)),indent=2))
