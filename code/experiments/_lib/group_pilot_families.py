"""Put full/pilot apart and make nested pilot cohorts extend one family."""
import ast,json,os,re
from pathlib import Path
from group_zero_shot import ROOT,PROJECT as P,STATE as S
from result_layout import relocate_path,rewrite_document,rewrite_reference,layout,rules
RENAME={
 'baseline/all_zeroshot_full9404_pilot300_v1':'overview/baseline/all_zeroshot_full9404_pilot300_v1',
 'baseline/input_order_full9404_pilot300':'overview/baseline/input_order_full9404_pilot300',
 'baseline/thinking_vs_instruct_pilot300_v1':'pilot/baseline/thinking_vs_instruct/300',
 'uncertainty/timelens_span_head_pilot300':'pilot/uncertainty/span_head/300',
 'uncertainty/token_analysis/pilot300':'pilot/uncertainty/token_analysis/300',
 'uncertainty/token_analysis/full9404':'full/uncertainty/token_analysis/9404',
 'sampling/fps_spatial_control_pilot300_v1':'pilot/sampling/fps_spatial_control/300',
 'sampling/spatial_allocation/spatial_budget_pilot900_v1':'pilot/sampling/spatial_allocation/budget_v1/900',
 'sampling/spatial_allocation/spatial_sampling_pilot900_v2':'pilot/sampling/spatial_allocation/sweep_v2/900',
}
MOVES={str(P/base/a):str(P/base/b) for base in ['results','artifacts/runs'] for a,b in RENAME.items()}
for n in [300,600]:MOVES[str(P/f'results/sampling/spatial_allocation/spatial_budget_pilot900_v1/PILOT_{n}.md')]=str(P/f'results/pilot/sampling/spatial_allocation/budget_v1/{n}/REPORT.md')

def replace(s):
 for base in ['results','artifacts/runs']:
  for a,b in sorted(RENAME.items(),key=lambda x:len(x[0]),reverse=True):s=re.sub(re.escape(base+'/'+a)+r'(?=/|[\s\'"`)>\]}]|$)',lambda m:base+'/'+b,s)
 return s

def apply():
 journal=json.loads((S/'migration.json').read_text());assert not journal.get('pilot_families_applied')
 lp=ROOT/'artifacts/maintenance/layout.json';obj=json.loads(lp.read_text());obj['moves'].update(MOVES)
 obj['presentation_moves'].update({a:b for a,b in MOVES.items() if '/results/' in a});obj['document_roots'] += [a for a in MOVES if '/results/' in a]
 lp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');layout.cache_clear();rules.cache_clear()
 # Capture page ownership before rebasing source URLs.
 scripts={}
 for html in (P/'results').rglob('*.html'):
  if 'rollout' in str(html):continue
  for src in re.findall(r'<script[^>]*src=[\'\"]([^\'\"]+)',html.read_text()):
   if src.startswith(('https:','http:','/')):continue
   js=Path(os.path.abspath(html.parent/src.split('?')[0]))
   if js.exists():scripts[js]=(html,relocate_path(html))
 changed=[]
 for base in [P/'results',P/'artifacts/runs']:
  for folder,dirs,files in os.walk(base):
   dirs[:]=[d for d in dirs if 'rollout' not in d and d not in {'code','token_traces','samples','responses','inputs','frames','logs','metadata','metrics','validation','media','rollouts','trees'}]
   for name in files:
    f=Path(folder,name)
    if f.is_symlink() or f.suffix not in {'.md','.html','.js','.css'}:continue
    before=f.read_text()
    old,new=scripts.get(f,(f,relocate_path(f))) if f.suffix=='.js' else (f,relocate_path(f))
    after=rewrite_document(before,old,new)
    if after!=before:f.write_text(after);changed.append(str(relocate_path(f)))
 # Main video manifests resolve from the cohort's result folder.
 for a,b in RENAME.items():
  f=P/'artifacts/runs'/a/'metadata/media_manifest.json'
  if not f.exists():continue
  before=json.loads(f.read_text())
  def update(x):
   if isinstance(x,dict):return {k:update(v) for k,v in x.items()}
   if isinstance(x,list):return [update(v) for v in x]
   if isinstance(x,str) and '/' in x and not re.search(r'\s',x):return rewrite_reference(x,P/'results'/a,P/'results'/b)
   return x
  f.write_text(json.dumps(update(before),ensure_ascii=False,indent=2)+'\n')
 for project in ['zero-shot','attention','similarity','_lib']:
  base=ROOT/project
  files=list((base/'_lib').rglob('*.py')) if project!='_lib' else list(base.glob('*.py'))
  if project=='_lib':files=[f for f in files if f.name not in {'group_zero_shot.py','finalize_zero_shot_scopes.py','group_pilot_families.py','repair_grouped_consumers.py','validate_zero_shot_grouping.py'}]
  else:files += list(base.glob('*.md'))
  for f in files:
   if f.is_symlink():continue
   before=f.read_text();after=replace(before) if f.suffix=='.py' else rewrite_document(before,f,f)
   if f.suffix=='.py':ast.parse(after)
   if before!=after:f.write_text(after);changed.append(str(f))
 for f in [P/'artifacts/README.md',*(P/'artifacts/common/registry').glob('*.json')]:
  before=f.read_text();after=replace(before) if f.suffix=='.json' else rewrite_document(before,f,f)
  if before!=after:f.write_text(after);changed.append(str(f))
 for a,b in sorted(MOVES.items(),key=lambda x:len(x[0]),reverse=True):
  src,dst=Path(a),Path(b)
  if not src.exists():continue
  assert not dst.exists(),dst
  dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
 # Remove emptied grouping containers, never rollout folders or temporary dependency links.
 for base in [P/'results',P/'artifacts/runs']:
  for folder,dirs,files in os.walk(base,topdown=False):
   f=Path(folder)
   if f!=base and not f.is_symlink():
    try:f.rmdir()
    except OSError:pass
 journal['moves'].update(MOVES);journal['changed']+=changed
 journal['preserved']=[[str(relocate_path(Path(p))),ino,size] for p,ino,size in journal['preserved']]
 journal['pilot_families_applied']=True
 (S/'migration.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2)+'\n')
 print('Full/pilot family migration complete:',len(MOVES),'moves')
if __name__=='__main__':apply()
