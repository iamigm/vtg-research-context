"""Resolve historical child resources consistently after their presentation parent moves."""
import json
from pathlib import Path
from result_layout import ROOT,layout,rules,relocate_path

def finalize():
 f=ROOT/'artifacts/maintenance/layout.json';obj=json.loads(f.read_text());added={}
 # Derive canonical child-resource aliases from the recorded moves; never create files or copies.
 for old in list(obj['moves']):
  target=relocate_path(Path(old))
  if not target.exists():continue
  for parent in Path(old).parents:
   key=str(parent)
   if key not in obj['moves']:continue
   new_parent=relocate_path(parent)
   candidate=new_parent/Path(old).relative_to(parent)
   c=str(candidate);t=str(target)
   if c==old or c==t or c in obj['moves'] or c in added or t.startswith(c+'/'):continue
   if candidate.exists():continue
   added[c]=t
 obj['moves'].update(added);f.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');layout.cache_clear();rules.cache_clear()
 for old,new in added.items():assert relocate_path(Path(old))==Path(new)
 print('Canonical child-resource aliases:',len(added))
if __name__=='__main__':finalize()
