"""Replace result-directory version suffixes with meaningful research conditions."""
import ast,json,os,re
from pathlib import Path
from result_layout import ROOT,layout,rules,relocate_path,rewrite_document,rewrite_reference
P=ROOT/'zero-shot';STATE=ROOT/'artifacts/maintenance/rollout_views_20261104'
RENAMES={
 'rollout/deterministic/v2':'rollout/deterministic',
 'sampling/spatial_allocation/budget_v1':'sampling/spatial_allocation/budget_comparison',
 'sampling/spatial_allocation/sweep_v2':'sampling/spatial_allocation/allocation_sweep',
 'baseline/input_order/overview/consistency_v1':'baseline/input_order/overview/consistency',
 'operations/selected_frame_cache_migration/shared/v1':'operations/selected_frame_cache_migration/shared',
}
MOVES={str(P/base/a):str(P/base/b) for base in ['results','artifacts/runs'] for a,b in RENAMES.items()}
def apply():
 if (STATE/'version_label_moves.json').exists():raise SystemExit('Already applied')
 f=ROOT/'artifacts/maintenance/layout.json';obj=json.loads(f.read_text());obj['moves'].update(MOVES)
 obj['presentation_moves'].update({a:b for a,b in MOVES.items() if '/results/' in a});obj['document_roots'] += [a for a in MOVES if '/results/' in a]
 f.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');layout.cache_clear();rules.cache_clear()
 for new in MOVES.values():assert relocate_path(Path(new))==Path(new),new
 scripts={};json_bases={};changed=[];links=[]
 for old in [P/'results'/a for a in RENAMES]:
  for page in old.rglob('*.html'):
   for src in re.findall(r'<script[^>]*src=[\'\"]([^\'\"]+)',page.read_text()):
    if src.startswith(('http:','https:','/')):continue
    js=Path(os.path.abspath(page.parent/src.split('?')[0]))
    if js.exists():scripts[js]=(page,relocate_path(page))
 for a in RENAMES:
  art=P/'artifacts/runs'/a;report=P/'results'/a
  for f in art.rglob('*.json'):
   rel=f.relative_to(art)
   if 'web' in rel.parts:
    before_web=Path(*rel.parts[:rel.parts.index('web')]);owner=report/before_web
    if not list(owner.glob('*.html')):owner=report
    json_bases[f]=(owner,relocate_path(owner))
   elif f.name=='viewer_data.json' and 'metadata' in rel.parts:
    i=rel.parts.index('metadata');owner=report/Path(*rel.parts[:i]);json_bases[f]=(owner,relocate_path(owner))
  for base in [report,art]:
   for folder,dirs,files in os.walk(base):
    dirs[:]=[d for d in dirs if d not in {'code','rollouts','trees','token_traces','inputs','samples','media','logs'}]
    for name in files:
     f=Path(folder,name)
     if f.is_symlink():links.append((str(f),str((f.parent/os.readlink(f)).resolve())));continue
     if f.suffix not in {'.md','.html','.js','.css'}:continue
     before=f.read_text();old,new=scripts.get(f,(f,relocate_path(f))) if f.suffix=='.js' else (f,relocate_path(f));after=rewrite_document(before,old,new)
     if after!=before:f.write_text(after);changed.append(str(relocate_path(f)))
 from update_result_json_paths import numeric_tree
 for f,(old,new) in json_bases.items():
  before=json.loads(f.read_text())
  def update(x):
   if isinstance(x,dict):return {k:update(v) for k,v in x.items()}
   if isinstance(x,list):return [update(v) for v in x]
   return rewrite_reference(x,old,new) if isinstance(x,str) and '/' in x and not re.search(r'\s',x) else x
  after=update(before);assert numeric_tree(before)==numeric_tree(after)
  if after!=before:f.write_text(json.dumps(after,ensure_ascii=False,indent=2)+'\n');changed.append(str(relocate_path(f)))
 # Native code keeps immutable run identities, but concrete folder strings and routing use new locations.
 for project in ['zero-shot','attention','similarity','_lib']:
  base=ROOT/project;files=list((base/'_lib').glob('*.py')) if project!='_lib' else [base/'pilot_catalog.py']
  for f in files:
   if f.is_symlink():continue
   before=f.read_text();after=before
   for a,b in RENAMES.items():after=after.replace('/'+a,'/'+b).replace("'"+a,"'"+b)
   if f.name=='pilot_catalog.py':after=after.replace("family.parent.name=='budget_v1'", "family.parent.name=='budget_comparison'")
   ast.parse(after)
   if after!=before:f.write_text(after);changed.append(str(f))
 for f in [*P.glob('*.md'),P/'artifacts/README.md']:
  before=f.read_text();after=rewrite_document(before,f,f)
  if after!=before:f.write_text(after);changed.append(str(f))
 for f in (P/'artifacts/common/registry').glob('*.json'):
  obj=json.loads(f.read_text());v=obj.get('path')
  if v:
   obj['path']=os.path.relpath(relocate_path(P/v),P);f.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 # Merge only the explicitly renamed directory; never overwrite another output.
 def move(src,dst):
  if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst);return
  assert src.is_dir() and dst.is_dir(),(src,dst)
  for child in list(src.iterdir()):move(child,dst/child.name)
  src.rmdir()
 for a,b in MOVES.items():move(Path(a),Path(b))
 for a,target in links:
  f=relocate_path(Path(a));dst=relocate_path(Path(target))
  if f.is_symlink():f.unlink();f.symlink_to(os.path.relpath(dst,f.parent))
 (STATE/'version_label_moves.json').write_text(json.dumps(dict(moves=MOVES,changed=changed),ensure_ascii=False,indent=2)+'\n')
 print('Removed version directories:',len(MOVES))
if __name__=='__main__':apply()
