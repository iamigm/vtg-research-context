"""Shared result-path routing and link rewriting for organized experiment outputs.

Only paths recorded in the migration manifest are routed. Scientific values and
unrelated paths are returned unchanged; no global pathlib behavior is changed.
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, unquote, quote
ROOT=Path(str(Path(__file__).resolve().parent.parent).replace('/home1/irteam/','/home/irteam/'))
@lru_cache(maxsize=1)
def layout():
 return json.loads((ROOT/'artifacts/maintenance/layout.json').read_text())
def canonical(path):
 return str(path).replace('/home1/irteam/','/home/irteam/')
@lru_cache(maxsize=1)
def rules():
 return sorted(layout()['moves'].items(),key=lambda x:len(x[0]),reverse=True)
def relocate_path(path):
 if not isinstance(path,Path):return path
 s=canonical(os.path.abspath(path));initial=s;moves=layout()['moves'];seen=set();applied=set()
 while s not in seen:
  seen.add(s);probe=s
  while probe and probe!='/':
   if probe in moves:
    if probe in applied:raise ValueError('Repeated migration prefix: '+probe)
    applied.add(probe)
    s=moves[probe]+s[len(probe):];break
   probe=probe.rsplit('/',1)[0]
  else:return Path(s) if s!=initial else path
 raise ValueError('Cyclic result migration: '+initial)
def original_path(path):
 s=canonical(os.path.abspath(path));seen=set();applied=set()
 reverse=sorted(rules(),key=lambda x:len(x[1]),reverse=True)
 while s not in seen:
  seen.add(s)
  for old,new in reverse:
   if (old,new) in applied:continue
   if s==new or s.startswith(new+'/'):
    applied.add((old,new))
    s=old+s[len(new):];break
  else:return Path(s)
 raise ValueError('Cyclic reverse migration: '+str(path))
def rewrite_reference(value,old_base,new_base):
 if not value or value.startswith(('http:','https:','data:','mailto:','javascript:','#','//')):return value
 old_base=Path(canonical(old_base));new_base=Path(canonical(new_base))
 parts=urlsplit(value)
 if parts.scheme or not parts.path:return value
 raw=unquote(parts.path)
 absolute=raw.startswith('/')
 src=Path(raw) if absolute else Path(os.path.normpath(str(old_base/raw)))
 dst=relocate_path(src)
 # Never change plain words or unresolvable non-path text.
 changed=canonical(src)!=canonical(dst)
 if not changed and old_base==new_base and '/home/irteam/' not in raw:return value
 if not changed and not src.exists():return value
 target=str(dst) if absolute else os.path.relpath(dst,new_base)
 if raw.endswith('/') and not target.endswith('/'):target+='/'
 return target+('?' + parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
def rewrite_document(text,old_file,new_file):
 if not isinstance(text,str):return text
 old_file=Path(canonical(old_file));new_file=Path(canonical(new_file))
 old_base=old_file.parent;new_base=new_file.parent
 # JavaScript URLs resolve against the owning HTML page; CSS uses its asset folder.
 if Path(old_file).suffix == '.js':
  candidates=[Path(r) for r in layout().get('document_roots',layout()['roots']) if str(old_file).startswith(r+'/')]
  if candidates:
   old_base=max(candidates,key=lambda p:len(str(p)));new_base=Path(layout().get('presentation_moves',{}).get(str(old_base),str(old_base)))
 def rep(m):
  token=m.group(0)
  if '/' not in token and not re.search(r'\.[A-Za-z]',token):return token
  try:return rewrite_reference(token,old_base,new_base)
  except (ValueError,OSError):return token
 return re.sub(r'[A-Za-z0-9_~./%{}:@+\-]+(?:\?[A-Za-z0-9_=&.%+\-]+)?(?:#[A-Za-z0-9_\-]+)?',rep,text)
def write_result_text(path,text,*args,**kwargs):
 path=relocate_path(path)
 from result_terms import normalize_report,normalize_markup
 if path.suffix=='.md':text=normalize_report(text)
 if path.suffix=='.html':text=normalize_markup(text)
 if path.suffix in {'.md','.html','.js','.css'}:text=rewrite_document(text,original_path(path),path)
 if path.suffix == '.md' and '/results/' in canonical(path):
  from build_index import scope_label
  scope=scope_label(path)
  if scope and text.startswith('# '):
   first,sep,rest=text.partition('\n')
   if scope not in first:text=first+' · '+scope+sep+rest
 if path.suffix == '.html' and 'results' in path.parts:
  from build_index import add_navigation
  text=add_navigation(path,text)
 path.parent.mkdir(parents=True,exist_ok=True)
 return path.write_text(text,*args,**kwargs)

def result_glob(path,pattern):
 """Glob a logical result root across its categorized artifact directories."""
 import fnmatch
 path=relocate_path(path)
 seen=set()
 for p in path.glob(pattern):
  seen.add(str(p));yield p
 first,sep,rest=pattern.partition('/')
 base=canonical(os.path.abspath(path))
 for old,new in rules():
  parent,name=old.rsplit('/',1)
  if parent!=base or not fnmatch.fnmatchcase(name,first):continue
  target=Path(new)
  candidates=target.glob(rest) if sep else [target]
  for p in candidates:
   if p.exists() and str(p) not in seen:seen.add(str(p));yield p

def copy_result_file(source,destination,*args,**kwargs):
 """Copy an asset and relocate links that resolve from its result page."""
 import shutil
 destination=relocate_path(destination)
 destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.suffix in {'.html','.md','.js','.css'}:
  text=Path(source).read_text()
  write_result_text(destination,text)
  return destination
 return shutil.copyfile(source,destination,*args,**kwargs)
