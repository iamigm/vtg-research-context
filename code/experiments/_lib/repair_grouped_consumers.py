"""Rebase browser bundles and keep frozen input paths readable after grouping."""
import ast,json,os,re
from pathlib import Path
from group_zero_shot import ROOT,PROJECT,STATE
from result_layout import relocate_path,rewrite_document
journal=json.loads((STATE/'migration.json').read_text())
moves=journal['moves']
def prior(path):
 s=str(path).replace('/home1/irteam/','/home/irteam/');seen=set()
 while s not in seen:
  seen.add(s)
  for old,new in sorted(moves.items(),key=lambda x:len(x[1]),reverse=True):
   if s==new or s.startswith(new+'/'):s=old+s[len(new):];break
  else:return Path(s)
 raise ValueError(path)
# Scripts execute with page-relative fetch/video URLs. Rebase from their owning HTML.
fixed=[]
for html in (PROJECT/'results').rglob('*.html'):
 if any('rollout' in part for part in html.parts):continue
 for src in re.findall(r'<script[^>]*src=[\'\"]([^\'\"]+)',html.read_text()):
  if src.startswith(('https:','http:','/')):continue
  js=Path(str((html.parent/src.split('?')[0]).resolve()).replace('/home1/irteam/','/home/irteam/'))
  if not js.exists() or not js.is_relative_to(PROJECT):continue
  old=prior(js);backup=STATE/'source_backup'/old.relative_to(ROOT)
  content=backup.read_text() if backup.exists() else js.read_text()
  after=rewrite_document(content,prior(html),html)
  if after!=js.read_text():js.write_text(after);fixed.append(str(js))
# Frozen input metadata is kept byte-identical; resolve paths when loading it.
f=PROJECT/'_lib/adaptive_temporal_sampling.py';s=f.read_text()
for old,new in [
 ("with Path(p).open('rb')", "with relocate_path(Path(p)).open('rb')"),
 ("json.loads(Path(p).read_text())", "json.loads(relocate_path(Path(p)).read_text())"),
 ("np.load(cache['path'],", "np.load(relocate_path(Path(cache['path'])),")]:s=s.replace(old,new)
if 'from result_layout import' not in s:s='from result_layout import relocate_path\n'+s
ast.parse(s);f.write_text(s)
f=PROJECT/'_lib/fps_spatial_control.py';s=f.read_text().replace("np.load(cache['array'],", "np.load(relocate_path(Path(cache['array'])),")
ast.parse(s);f.write_text(s)
# Newly written reports use the same historical-relative template routing as existing producers.
for f in (PROJECT/'_lib').glob('*.py'):
 if 'rollout' in f.name or f.is_symlink():continue
 s=f.read_text();tree=ast.parse(s);lines=s.splitlines(keepends=True)
 def pos(n,end=False):
  ln=n.end_lineno if end else n.lineno;col=n.end_col_offset if end else n.col_offset
  return sum(map(len,lines[:ln-1]))+len(lines[ln-1].encode()[:col].decode())
 edits=[]
 for n in ast.walk(tree):
  if not (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text'):continue
  obj=n.func.value;expression=ast.get_source_segment(s,obj)
  if not re.search(r'\b(REPORT|VIEW)\b',expression):continue
  edits.append((pos(n),pos(n.func,True)+1,'write_result_text('+expression+', '))
 if not edits:continue
 for a,b,t in sorted(edits,reverse=True):s=s[:a]+t+s[b:]
 if not re.search(r'from result_layout import .*write_result_text',s):
  tree=ast.parse(s);at=0
  if isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):at=sum(map(len,s.splitlines(keepends=True)[:tree.body[0].end_lineno]))
  s=s[:at]+'from result_layout import write_result_text\n'+s[at:]
 ast.parse(s);f.write_text(s)
print('Rebased page-owned JS bundles:',len(fixed))
