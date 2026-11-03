"""Validate moved data, result surfaces, Python consumers and local HTML links."""
from organize_results import *
from result_layout import canonical, relocate_path, original_path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
D=json.loads((STATE/'layout.json').read_text())
from update_result_json_paths import digest
updates={x['old']:x for x in json.loads((STATE/'json_path_updates.json').read_text())}
removed={str(relocate_path(Path(p))) for p in json.loads((STATE/'removed_disposable.json').read_text())}
errors=[];counts={};hashes=json.loads((STATE/'before_metrics_sha256.json').read_text())
for old,digest in hashes.items():
 p=relocate_path(Path(old))
 if old in updates:
  from update_result_json_paths import digest as semantic_digest
  if semantic_digest(json.loads(p.read_text()))!=updates[old]['numeric_tree_sha256']:errors.append(['numeric_json',old])
 elif not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append(['numeric_hash',old,str(p)])
hierarchy=json.loads((STATE/'artifact_hierarchy.json').read_text()) if (STATE/'artifact_hierarchy.json').exists() else {}
from update_result_json_paths import digest as semantic_digest
for edit in hierarchy.get('json_path_updates',[]):
 p=relocate_path(Path(edit['path']))
 if semantic_digest(json.loads(p.read_text()))!=edit['numeric_tree_sha256']:errors.append(['hierarchy_json_numeric',str(p)])
counts['hierarchy_json_numeric_structures_verified']=len(hierarchy.get('json_path_updates',[]))
counts['numeric_files_verified']=len(hashes)
counts['numeric_sha256_verified']=len(set(hashes)-set(updates))
counts['numeric_json_semantics_verified']=len(set(hashes)&set(updates))
ident=json.loads((STATE/'before_identity.json').read_text());changed={canonical(p) for p in json.loads((STATE/'changed_files.json').read_text())}
for old,ino,size,mtime in ident:
 p=relocate_path(Path(old))
 if str(p) in changed or old in updates or old in D['deleted_previews'] or Path(old).is_relative_to(ROOT/'_lib') or Path(old).is_relative_to(ROOT/'shared') or old in hashes or str(p) in D['previews'].values() or str(p) in removed:continue
 if not p.exists():errors.append(['missing',old,str(p)]);continue
 st=p.stat()
 if (st.st_ino,st.st_size,st.st_mtime_ns)!=(ino,size,mtime):errors.append(['identity_changed',old])
counts['file_identities_checked']=len(ident)
for r in D['roots']:
 for p in Path(r).iterdir():
  if p.is_dir():continue
  if p.suffix!='.md' and p.suffix!='.html' and str(p) not in D['previews'].values():errors.append(['result_surface',str(p)])
counts['result_roots']=len(D['roots'])
class Links(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,tag,attrs):
  for key,val in attrs:
   if key in {'href','src','poster'} and val:self.refs.append(val)
broken=[];html_count=0;refs=0
html_roots=list(D['roots'])+[str(relocate_path(p/'artifacts/runs')) for p in [ROOT/'attention',ROOT/'zero-shot',ROOT/'similarity',ROOT/'analysis/failure_mode_timestamp/failure_cases',ROOT/'audio_caption_resources_20260930']]
for r in html_roots:
 for p in Path(r).rglob('*.html'):
  if p.is_symlink() or any(x in p.parts for x in ['source_backup','vendor','sources','provenance']):continue
  parser=Links()
  try:parser.feed(p.read_text())
  except UnicodeError:continue
  html_count+=1
  for value in parser.refs:
   u=urlsplit(value)
   if u.scheme or u.netloc or not u.path or u.path.startswith('/api/'):continue
   if '{' in u.path:continue
   refs+=1;target=p.parent/unquote(u.path)
   if not target.exists():broken.append([str(p),value])
counts.update(html_pages_checked=html_count,local_html_refs_checked=refs)
# Deduplicate pages visited via both parent and nested report roots.
broken=sorted(set(tuple(x) for x in broken));counts['broken_html_refs']=len(broken)
for f in changed:
 p=Path(f)
 if p.suffix=='.py':
  try:ast.parse(p.read_text(),filename=str(p))
  except SyntaxError as e:errors.append(['syntax',str(p),str(e)])
counts['python_sources_parsed']=sum(Path(f).suffix=='.py' for f in changed)
# Markdown navigation and all owned symlinks are checked after the final moves.
mdrefs=0
mdfiles={p for r in D['roots'] for p in Path(r).glob('*.md')}|{ROOT/'README.md',ROOT/'attention/README.md',ROOT/'zero-shot/README.md',ROOT/'CLEANUP_REPORT.md'}
mdfiles.update(ROOT.glob('*/README.md'))
mdfiles.update((ROOT/'analysis').glob('*/README.md'))
mdfiles.add(ROOT/'analysis/failure_mode_timestamp/failure_cases/README.md')
mdfiles.add(ROOT/'zero-shot/artifacts/predictions/legacy/README.md')
for p in mdfiles:
 if not p.exists():continue
 for ref in re.findall(r'\]\(([^\s)]+)\)',p.read_text()):
  u=urlsplit(ref)
  if u.scheme or not u.path:continue
  mdrefs+=1
  if not (p.parent/unquote(u.path)).exists():errors.append(['markdown_link',str(p),ref])
counts['markdown_refs_checked']=mdrefs
symlinks=0
for family in ['attention','zero-shot','analysis','audio_caption_resources_20260930']:
 for base,dirs,files in os.walk(ROOT/family):
  dirs[:]=[d for d in dirs if d not in SKIP and not Path(base,d).is_symlink()]
  for name in files:
   p=Path(base,name)
   if p.is_symlink():
    symlinks+=1
    if not p.exists():errors.append(['broken_symlink',str(p)])
counts['symlinks_checked']=symlinks
allowed={'attention':{'README.md','AGENTS.md','run.py','_lib','results','artifacts','.venv'},'zero-shot':{'README.md','AGENTS.md','run.py','_lib','results','artifacts','.venv','predictions'},'similarity':{'README.md','_lib','results','artifacts','logs','vendor'},'analysis/failure_mode_timestamp/failure_cases':{'README.md','_lib','results','artifacts','.venv'}}
for family,names in allowed.items():
 for p in (ROOT/family).iterdir():
  if p.name not in names:errors.append(['project_surface',str(p)])
counts['project_roots_checked']=len(allowed)
nested=[];live_aliases=[]
for family in ['attention','zero-shot','similarity','analysis/failure_mode_timestamp/failure_cases','audio_caption_resources_20260930']:
 for p in (relocate_path(ROOT/family/'results')).rglob('artifacts'):
  if p.is_symlink():live_aliases.append(str(p))
  elif p.is_dir():nested.append(str(p))
for p in nested:errors.append(['nested_artifact_directory',p])
counts['physical_artifact_dirs_below_results']=len(nested)
counts['temporary_artifact_aliases']=len(live_aliases)
result=dict(counts=counts,errors=errors,broken_html_refs=broken)
(STATE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(counts=counts,errors=errors[:15],broken_html_refs=broken[:20]),indent=2))
