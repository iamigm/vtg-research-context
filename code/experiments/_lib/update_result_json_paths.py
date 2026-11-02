"""Rewrite result-level JSON path values while proving numbers are unchanged."""
import json,re,hashlib
from pathlib import Path
from result_layout import ROOT,layout,relocate_path,rewrite_reference
STATE=ROOT/'artifacts/maintenance'
def numeric_tree(x):
 if isinstance(x,dict):return {k:numeric_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [numeric_tree(v) for v in x]
 return '<text>' if isinstance(x,str) else x
def digest(x):return hashlib.sha256(json.dumps(numeric_tree(x),sort_keys=True,separators=(',',':'),allow_nan=True).encode()).hexdigest()
def rewrite(x,base):
 if isinstance(x,dict):return {k:rewrite(v,base) for k,v in x.items()}
 if isinstance(x,list):return [rewrite(v,base) for v in x]
 if isinstance(x,str) and '/' in x and not re.search(r'\s',x):
  return rewrite_reference(x,base,base)
 return x
if __name__=='__main__':
 changes=[]
 for old,new in layout()['moves'].items():
  p=Path(new)
  if p.suffix!='.json' or not p.is_file():continue
  try:before=json.loads(p.read_text());after=rewrite(before,Path(old).parent)
  except (ValueError,UnicodeError):continue
  if before==after:continue
  assert numeric_tree(before)==numeric_tree(after)
  b=p.read_bytes();p.write_text(json.dumps(after,ensure_ascii=False,indent=2)+'\n')
  changes.append(dict(old=old,path=str(p),before_sha256=hashlib.sha256(b).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),numeric_tree_sha256=digest(before)))
 (STATE/'json_path_updates.json').write_text(json.dumps(changes,indent=2)+'\n');print('JSON path consumers updated:',len(changes))
