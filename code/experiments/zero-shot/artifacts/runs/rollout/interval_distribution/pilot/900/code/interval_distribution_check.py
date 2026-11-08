"""Raw invariants and cumulative gate validation."""
import json,math,sqlite3
from pathlib import Path
from interval_distribution_core import *

def validate(count=900):
 rows=read(COHORT)[:count];assert read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{count}/samples.json')==rows
 for row in rows:
  key=row['sample_key'];query=read(OUT/'queries'/f'{key}.json');assert query['config_hash']==digest(CONFIG) and query['complete']
  db=connect(OUT/'trees'/f'{key}.sqlite')
  for status,accepted in db.execute('SELECT status,accepted FROM nodes'):
   assert status not in ('pending','context_limit')
   if accepted:
    edges=json.loads(accepted);assert all(e['p']>=.01 for e in edges)
    assert all(edges[i]['p']>=edges[i+1]['p'] for i in range(len(edges)-1))
  db.close();total=sum(query['mass'].values());assert abs(total-1)<2e-4,(key,total)
  if query['intervals']:assert abs(sum(x['conditional_weight'] for x in query['intervals'])-1)<1e-8
  for phase in ('instruct','embedding'):
   sem=read(OUT/'semantic'/phase/f'{key}.json');assert not sem['vectors_persisted']
   for p in sem['pools'].values():
    if p:assert 0<=p['embedding_coverage_mass']<=1+1e-8
  v=read(OUT/'verification'/f'{key}.json');assert v['complete'] and v['model']==str(VERIFIER)
  spans=[r['span'] for r in v['candidates']]
  for f in v['finals'].values():assert f['span'] is None or f['span'] in spans
 write(OUT/'validation'/f'raw_pilot{count}.json',dict(passed=True,queries=count,threshold_verified=True,mass_conservation=True,no_permanent_vectors=True,timestamp=stamp()));print('RAW_TRACE_VALIDATED',count)

def cpu_smoke():
 import tempfile
 p=np.array([.97,.015,.009,.006]);edges,h=admitted(p);assert [e['token'] for e in edges]==[0,1]
 p=np.array([.99,.01]);assert [e['token'] for e in admitted(p)[0]]==[0]
 p=np.array([.98,.01,.01]);assert all(e['p']>=.01 for e in admitted(p)[0])
 with tempfile.TemporaryDirectory() as td:
  db=connect(Path(td)/'tree.sqlite');db.execute("INSERT INTO nodes(id,p,logp,status,depth,accepted,entropy) VALUES(0,1,0,'expanded',0,?,1)",(json.dumps([dict(token=1,p=.5),dict(token=2,p=.5)]),))
  for ident,span in [(1,[10.,15.]),(2,[30.,35.])]:db.execute("INSERT INTO nodes(id,parent,p,logp,status,depth,span) VALUES(?,0,.5,?,'valid',1,?)",(ident,math.log(.5),json.dumps(span)))
  db.commit();db.close();summary,intervals=summarize_tree(Path(td)/'tree.sqlite');assert abs(summary['root']['entropy']-1)<1e-12;assert summary['root']['covariance']==[[100.,100.],[100.,100.]];assert summary['root']['expected_residual_interval_entropy']==0
 groups,assign=grouped([dict(id='a',span=[10.,15.],conditional_weight=.5),dict(id='b',span=[30.,35.],conditional_weight=.5)]);assert len(groups)==2
 print('CPU_SMOKE_PASSED')
if __name__=='__main__':
 import sys
 cpu_smoke() if '--cpu-smoke' in sys.argv else validate()
