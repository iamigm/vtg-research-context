#!/usr/bin/env python3
"""Organize result trees, preserving an auditable path and file identity manifest."""
from result_layout import relocate_path
import os, json, re, hashlib, ast, shutil
from pathlib import Path
ROOT=Path('/home/irteam/experiments')
STATE=ROOT/'artifacts/maintenance'
SKIP={'.git','.venv','vendor','runtime','python_deps','browser_cache','browser_deps','browser_libs','__pycache__','MTLA','sources'}
def walk(root):
 for base,ds,fs in os.walk(root):
  ds[:]=[d for d in ds if d not in SKIP and not Path(base,d).is_symlink()]
  for f in fs:yield Path(base,f)
def roots():
 out=[ROOT/'attention/results']
 for family in ['attention','zero-shot','similarity']:
  r=relocate_path(ROOT/family/'results')
  out.extend(p for p in r.iterdir() if p.is_dir() and p.name!='artifacts')
 # Nested independent analyses retain their own report and artifact root.
 for p in list(walk(ROOT/'attention/results'))+list(walk(ROOT/'zero-shot/results')):
  if p.suffix=='.md' and p.name in {'REPORT.md','PIPELINE_REVIEW.md','CACHE_REPORT.md'} and not any(x in p.parts for x in ['provenance','migration_to_live','sources','artifacts']):out.append(p.parent)
 out.extend([ROOT/'analysis/failure_mode_timestamp/failure_cases/reports/pilot_100',ROOT/'audio_caption_resources_20260930'])
 return sorted(set(out),key=lambda p:(len(p.parts),str(p)))
def category(p):
 n=p.name.lower()
 if p.is_dir() or p.is_symlink() and p.resolve().is_dir():
  if n in {'logs','media','frames','token_traces','configs','provenance','fonts','papers','sources','vendor'}:return n
  if n in {'qwen25','qwen3','samples','records','charades-sta','qvhighlights','activitynet-captions'} or 'timelens' in n or re.search(r'_\d+$',n):return 'samples'
  if n=='overview_pages':return 'figures'
  return 'data'
 if p.suffix in {'.log','.lock','.pid'}:return 'logs'
 if p.suffix in {'.py','.sh'}:return 'code'
 if p.suffix in {'.csv','.npz','.npy','.pt','.safetensors'}:return 'metrics' if p.suffix=='.csv' else 'arrays'
 if p.suffix in {'.png','.jpg','.jpeg','.svg','.pdf'}:return 'figures'
 if p.suffix in {'.js','.css','.ttf','.woff','.woff2'} or n=='ofl.txt':return 'web'
 if 'sample' in n or 'cohort' in n:return 'inputs'
 if 'valid' in n or 'audit' in n or 'sha256' in n:return 'validation'
 if 'summary' in n or 'metrics' in n or 'paired' in n or 'results' in n:return 'metrics'
 return 'metadata'
def plan():
 if (STATE/'changed_files.json').exists():raise SystemExit('Layout already applied; use validate_result_layout.py')
 rs=roots();moves={};deleted=[];preview={}
 for r in rs:
  ps=[p for p in r.iterdir() if p.is_file() and p.suffix=='.png' and ('preview' in p.name or p.name.startswith('viewer_'))]
  groups={}
  for p in ps:
   k='mobile' if 'mobile' in p.name else 'desktop'
   groups.setdefault(k,[]).append(p)
  for k,ls in groups.items():
   keep=max(ls,key=lambda p:p.stat().st_mtime_ns);preview[str(r)+'/'+k]=str(keep)
   deleted.extend(str(p) for p in ls if p!=keep)
  for p in r.iterdir():
   if p.name=='artifacts' or p in rs or str(p) in deleted:continue
   if p.is_file() and (p.suffix=='.md' or p.name=='report.html' or str(p) in preview.values()):continue
   # Development/runtime dependencies remain outside result artifacts only at project roots.
   if r.name.startswith('audio_caption') and (p.suffix=='.py' or p.name in {'__pycache__'}):continue
   if r.name.startswith('audio_caption') and p.name=='report.html':moves[str(p)]=str(r/'report.html');continue
   moves[str(p)]=str(r/'artifacts'/category(p)/p.name)
 for p in ROOT.glob('*.json'):moves[str(p)]=str(ROOT/'artifacts/metadata'/p.name)
 data=dict(roots=[str(r) for r in rs],moves=moves,deleted_previews=deleted,previews=preview)
 (STATE/'layout.json').write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps(dict(result_roots=len(rs),moves=len(moves),old_previews=len(deleted)),indent=2))
if __name__=='__main__':plan()
