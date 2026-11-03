"""Publish fixed nested pilot300/600 HTML as soon as each cohort is ready."""
from result_layout import relocate_path
import collections,fcntl,json,sys,time,traceback
from pathlib import Path
from deterministic_rollout300 import ROOT,read,write,stamp

OUT=relocate_path(ROOT/'artifacts/runs/deterministic_rollout_pilot900_v2')
TRACES=OUT/'token_traces'

def record(state,**kw):
 value=dict(state=state,timestamp=stamp(),**kw)
 write(OUT/'metadata/pilot_html_progress.json',value)
 with (OUT/'logs/pilot_html.log').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
 print(json.dumps(value,ensure_ascii=False),flush=True)

def publish(n,cohort):
 from deterministic_rollout_viewer import build
 out=relocate_path(ROOT/'artifacts/runs'/f'deterministic_rollout_pilot{n}_v2')
 report=relocate_path(ROOT/'results'/out.name)
 assert read(out/'metadata/cohort.json')==cohort
 prefixes=size=0
 for sample in cohort:
  index=read(TRACES/sample['sample_key']/'index.json');assert index['validation']['passed']
  for node in index['nodes'].values():
   assert (TRACES/sample['sample_key']/node['file']).stat().st_size==node['bytes']
   prefixes+=1;size+=node['bytes']
 write(out/'validation/pilot_distribution_ready.json',dict(passed=True,queries=n,prefixes=prefixes,
  compressed_bytes=size,shared_source=str(TRACES),fixed_cohort_order=True,timestamp=stamp()))
 build(pilot_size=n)
 manifest=read(out/'web/manifest.json')
 assert [s['key'] for s in manifest['samples']]==[s['sample_key'] for s in cohort]
 assert len(manifest['samples'])==n and (report/'viewer.html').exists()
 assert not (out/'token_traces').exists(),'Do not duplicate distribution files'
 write(out/'validation/pilot_html.json',dict(passed=True,queries=n,distributions_ready=n,
  shared_token_traces=True,shared_sample_payloads=True,html=str(report/'viewer.html'),timestamp=stamp()))
 state=read(out/'metadata/status.json');state['token_distributions']=dict(state='complete',completed=n,total=n,timestamp=stamp())
 write(out/'metadata/status.json',state)

def main():
 with (OUT/'metadata/pilot_html_watch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cohorts={n:read(ROOT/'artifacts/common/cohorts/nested_pilot_v1'/f'pilot_{n}/samples.json') for n in [300,600,900]}
  assert cohorts[900][:300]==cohorts[300] and cohorts[900][:600]==cohorts[600]
  try:
   while True:
    done={p.parent.name for p in TRACES.glob('*/index.json')}
    ready={str(n):sum(s['sample_key'] in done for s in cohort) for n,cohort in cohorts.items()}
    published=[]
    for n in [300,600]:
     validation=relocate_path(ROOT/'artifacts/runs'/f'deterministic_rollout_pilot{n}_v2/validation/pilot_html.json')
     if ready[str(n)]==n and not validation.exists():
      record('running',stage=f'publishing_pilot{n}',ready=ready,pilot=n)
      publish(n,cohorts[n])
     if validation.exists():published.append(n)
    integrated=OUT/'validation/distribution_integration.json'
    if integrated.exists() and read(integrated).get('passed'):published.append(900)
    record('complete' if len(published)==3 else 'waiting',stage='complete' if len(published)==3 else 'waiting_for_fixed_cohorts',
       total_captured=len(done),ready=ready,published=published)
    if len(published)==3:break
    capture=read(OUT/'metadata/distribution_status.json')
    if capture['state']=='failed':raise RuntimeError('Distribution capture failed; see distribution_status.json')
    time.sleep(20)
  except BaseException:record('failed',stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
