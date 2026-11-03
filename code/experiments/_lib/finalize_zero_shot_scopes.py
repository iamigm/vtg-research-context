"""Make cohort scope explicit in grouped zero-shot paths and references."""
import ast,json,os,re
from pathlib import Path
from group_zero_shot import ROOT,PROJECT,STATE,patch_source
import group_zero_shot as grouping
RENAME={
 'baseline/all_zeroshot_v1':'baseline/all_zeroshot_full9404_pilot300_v1',
 'baseline/input_order':'baseline/input_order_full9404_pilot300',
 'baseline/thinking_vs_instruct_v1':'baseline/thinking_vs_instruct_pilot300_v1',
 'uncertainty/timelens_span_head':'uncertainty/timelens_span_head_pilot300',
 'uncertainty/token_analysis/timelens_top30':'uncertainty/token_analysis/pilot300',
 'uncertainty/token_analysis/timelens_top30_full':'uncertainty/token_analysis/full9404',
 'operations/mmco_diagnosis':'operations/mmco_diagnosis_pilot300',
 'operations/inference_performance':'operations/inference_performance_full9404',
}
MOVES={str(PROJECT/base/a):str(PROJECT/base/b) for base in ['results','artifacts/runs'] for a,b in RENAME.items()}
def replace(s):
 for a,b in sorted(MOVES.items(),key=lambda x:len(x[0]),reverse=True):s=s.replace(a,b).replace(a.replace('/home/','/home1/'),b.replace('/home/','/home1/'))
 for base in ['results','artifacts/runs']:
  for a,b in sorted(RENAME.items(),key=lambda x:len(x[0]),reverse=True):s=re.sub(re.escape(base+'/'+a)+r'(?=/|[\s\'"`)>\]}]|$)',lambda m:base+'/'+b,s)
 return s
lp=ROOT/'artifacts/maintenance/layout.json';layout=json.loads(lp.read_text());layout['moves'].update(MOVES)
layout['presentation_moves'].update({a:b for a,b in MOVES.items() if '/results/' in a});layout['document_roots'] += [a for a in MOVES if '/results/' in a]
lp.write_text(json.dumps(layout,ensure_ascii=False,indent=2)+'\n')
from result_layout import relocate_path,rewrite_document,layout as l,rules
l.cache_clear();rules.cache_clear()
changed=[]
# No rollout data/report/code content is changed here; its external URLs use server routing.
for project in ['zero-shot','attention','similarity']:
 p=ROOT/project
 for base,dirs,files in os.walk(p):
  dirs[:]=[d for d in dirs if d not in {'common','code','token_traces','samples','responses','inputs','frames','logs','source_backup','__pycache__','.venv','media','metadata','metrics','validation','rollouts','trees'} and 'rollout' not in d]
  for name in files:
   f=Path(base,name)
   if f.is_symlink() or f.suffix not in {'.py','.md','.html','.js','.css'}:continue
   before=f.read_text()
   if f.suffix=='.py':after=replace(before);ast.parse(after)
   else:after=rewrite_document(before,f,relocate_path(f))
   if before!=after:f.write_text(after);changed.append(str(relocate_path(f)))
for f in (PROJECT/'artifacts/common/registry').glob('*.json'):
 before=f.read_text();after=replace(before)
 if before!=after:f.write_text(after);changed.append(str(f))
# Web/media manifests contain path strings; raw input and response JSON stay byte-identical.
for f in (PROJECT/'artifacts/runs').rglob('media_manifest.json'):
 if 'rollout' in str(f):continue
 before=f.read_text();after=replace(before)
 if before!=after:f.write_text(after);changed.append(str(relocate_path(f)))
for a,b in sorted(MOVES.items(),key=lambda x:len(x[0]),reverse=True):
 src,dst=Path(a),Path(b)
 if src.exists():dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
journal=json.loads((STATE/'migration.json').read_text());journal['moves'].update(MOVES);journal['changed']+=changed
journal['preserved']=[[str(relocate_path(Path(p))),ino,size] for p,ino,size in journal['preserved']]
(STATE/'migration.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2)+'\n')
print('Explicit full/pilot scope paths:',len(MOVES))
