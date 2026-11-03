"""One900-query source store;300/600 are fixed cohort views, not new copies."""
import hashlib,json,os
from pathlib import Path
from result_layout import ROOT,layout,rules,relocate_path,rewrite_document,rewrite_reference
P=ROOT/'zero-shot';BASE=P/'artifacts/runs/rollout/deterministic/pilot';OUT=BASE/'900';EXTRA=OUT/'additional600';S=ROOT/'artifacts/maintenance/rollout_views_20261104'
MOVES={}
for source in [BASE/'300',EXTRA]:
 for category in ['rollouts','inputs']:
  if (source/category).exists():MOVES[str(source/category)]=str(OUT/category)
for category in ['media','logs','code','metrics','metadata','validation']:
 target=OUT/category if category=='media' else OUT/category/'source_batches/expansion'
 MOVES[str(EXTRA/category)]=str(target)
MOVES[str(P/'results/rollout/deterministic/pilot/900/additional600')]=str(OUT/'provenance/expansion_reports')

def apply():
 if (S/'unified_sources.json').exists():raise SystemExit('Already merged')
 records={}
 for source in [BASE/'300/rollouts',EXTRA/'rollouts']:
  for f in source.glob('*.json'):records[f.stem]=hashlib.sha256(f.read_bytes()).hexdigest()
 frozen=json.loads((OUT/'metadata/cohort.json').read_text());assert len(records)==900 and set(records)=={r['sample_key'] for r in frozen}
 path=ROOT/'artifacts/maintenance/layout.json';obj=json.loads(path.read_text());obj['moves'].update(MOVES);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');layout.cache_clear();rules.cache_clear()
 for value in MOVES.values():assert relocate_path(Path(value))==Path(value)
 report=P/'results/rollout/deterministic/pilot/900/additional600'
 for f in report.glob('*.md'):f.write_text(rewrite_document(f.read_text(),f,relocate_path(f)))
 def merge(src,dst):
  if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst);return
  assert src.is_dir() and dst.is_dir(),(src,dst)
  for f in list(src.iterdir()):merge(f,dst/f.name)
  src.rmdir()
 for a,b in MOVES.items():
  if Path(a).exists():merge(Path(a),Path(b))
 if EXTRA.exists():EXTRA.rmdir()
 assert len(list((OUT/'rollouts').glob('*.json')))==900
 for key,sha in records.items():assert hashlib.sha256((OUT/'rollouts'/(key+'.json')).read_bytes()).hexdigest()==sha
 # Reference manifests and browser payloads contain paths, never duplicate answers.
 for n in [300,600,900]:
  out=BASE/str(n);source=out/'metadata/source_records.json'
  refs=[dict(path=str(OUT/'rollouts'/(r['sample_key']+'.json')),sha256=records[r['sample_key']]) for r in frozen[:n]]
  source.write_text(json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
  (out/'metadata/source_store.json').write_text(json.dumps(dict(queries=n,source_store=str(OUT/'rollouts'),cohort=str(out/'metadata/cohort.json'),records_copied=False),indent=2)+'\n')
 for f in (OUT/'web/samples').glob('*.json'):
  obj=json.loads(f.read_text());ref=obj.get('raw_record')
  if ref:obj['raw_record']=rewrite_reference(ref,P/'results/rollout/deterministic/pilot/900',P/'results/rollout/deterministic/pilot/900');f.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 for f in (P/'artifacts/common/registry').glob('*.json'):
  obj=json.loads(f.read_text());value=obj.get('path')
  if value:obj['path']=os.path.relpath(relocate_path(P/value),P)
  f.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 (S/'unified_sources.json').write_text(json.dumps(dict(queries=900,moves=MOVES,response_sha256=records,cohorts=[300,600,900],copies=0),ensure_ascii=False,indent=2)+'\n')
 print('Unified900 responses and inputs; SHA256 preserved; no additional600 experiment')
if __name__=='__main__':apply()
