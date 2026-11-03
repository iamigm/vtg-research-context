"""Run only frozen pilot900[300:], reuse first300, report extra600 and nested600/900."""
from result_layout import relocate_path
import ast,collections,fcntl,hashlib,json,multiprocessing,sys,time,traceback
from pathlib import Path
import deterministic_rollout300 as m
from deterministic_rollout300_report import report as base_report

ROOT=m.ROOT
RUN='deterministic_rollout_additional600_v2'
OUT=relocate_path(ROOT/'artifacts/runs'/RUN)
REPORT=relocate_path(ROOT/'results'/RUN)
ORIGINAL=relocate_path(ROOT/'artifacts/runs/deterministic_rollout_pilot300_v2')
NESTED=ROOT/'artifacts/common/cohorts/nested_pilot_v1'
PARENTS=ROOT/'artifacts/common/spatial_sampling_sources_v2/source_frames'
POOL=None
PENDING={}
WAITING=[]

def prepare_video(group):
 import fps_spatial_control as source
 # Same preparation implementation/grid/decoder/resize; retain only 2 FPS frames.
 source.OUT=OUT;source.FPS=[2]
 source.prepare_video(group)
 return str(OUT/'media'/group['key']/'metadata.json')

def immutable(path,value):
 if path.exists():assert m.read(path)==value, f'Frozen identity changed: {path}'
 else:m.write(path,value)

def function_sources(path,names):
 text=path.read_text();tree=ast.parse(text)
 return {node.name:ast.get_source_segment(text,node) for node in tree.body
         if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names}

def setup():
 original=m.read(NESTED/'pilot_300/samples.json')
 six=m.read(NESTED/'pilot_600/samples.json');nine=m.read(NESTED/'pilot_900/samples.json')
 assert nine[:300]==original and nine[:600]==six
 rows=nine[300:];assert len(rows)==600
 assert collections.Counter(x['dataset'] for x in rows)==dict.fromkeys(['charades-timelens','activitynet-timelens','qvhighlights-timelens'],200)
 assert not {x['sample_key'] for x in original}&{x['sample_key'] for x in rows}
 assert not {x['video_path'] for x in original}&{x['video_path'] for x in rows}
 for category in ['metadata','logs','code','inputs','rollouts','metrics','validation','media']:(OUT/category).mkdir(parents=True,exist_ok=True)
 REPORT.mkdir(parents=True,exist_ok=True)
 baseline_config=m.read(ORIGINAL/'metadata/config.json')
 for key,value in baseline_config.items():
  if key not in ['run','queries','cohort']:assert m.CONFIG[key]==value,(key,m.CONFIG[key],value)
 names=['complete','propose','infer','parse','time_matches','seconds','sites']
 assert function_sources(Path(m.__file__),names)==function_sources(ORIGINAL/'code/deterministic_rollout300.py',names)
 assert m.sha(Path(__file__).with_name('fps_spatial_control.py'))==m.sha(ORIGINAL/'code/fps_spatial_control.py'),'Input implementation changed'
 immutable(OUT/'metadata/cohort.json',rows)
 immutable(OUT/'metadata/config.json',dict(m.CONFIG,run=RUN,queries=600,cohort=str(NESTED/'pilot_900/samples.json'),selection='exact frozen pilot900[300:900]'))
 original_records=[]
 for row in original:
  p=ORIGINAL/'rollouts'/(row['sample_key']+'.json')
  assert m.read(p)['sample']==row
  original_records.append(dict(path=str(p),sha256=m.sha(p)))
 immutable(OUT/'metadata/original300.json',original_records)
 parents={m.read(p)['video_path']:str(p) for p in PARENTS.glob('*.json')}
 groups=[];seen=set()
 for row in rows:
  video=row['video_path']
  if video in seen:continue
  seen.add(video);key=hashlib.sha256(video.encode()).hexdigest()
  assert video in parents
  groups.append(dict(video=video,key=key,parent_metadata=parents[video]))
 immutable(OUT/'metadata/groups.json',groups)
 files=[]
 for name in ['deterministic_rollout900.py','deterministic_rollout300.py','deterministic_rollout300_report.py','conditional_rollout_budget.py','fps_spatial_control.py']:
  p=Path(__file__).with_name(name);(OUT/'code'/name).write_bytes(p.read_bytes())
  files.append(dict(path=str(p),sha256=m.sha(p)))
 m.write(OUT/'metadata/provenance.json',dict(files=files,original_cohort_sha256=m.sha(NESTED/'pilot_300/samples.json'),
    full_cohort_sha256=m.sha(NESTED/'pilot_900/samples.json'),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),frame_workers=1,
    worker_video_limit=1,prefetch_videos=2,spatial_temporal_policy='same FPS preparation, only required 2 FPS frames stored',
    generation_functions_identical=True))
 m.write(OUT/'validation/preflight.json',dict(passed=True,original_queries=300,additional_queries=600,
    per_dataset=200,disjoint_sample_ids=True,disjoint_video_paths=True,nested_prefixes_preserved=True,
    same_generation_functions=True,same_inference_settings=True))
 return rows,groups

def fill():
 while WAITING and len(PENDING)<2:
  group=WAITING.pop(0);path=OUT/'media'/group['key']/'metadata.json'
  PENDING[group['video']]=(str(path) if path.exists() else POOL.apply_async(prepare_video,(group,)))

def input_cache(row,source):
 video=row['video_path'];key=hashlib.sha256(video.encode()).hexdigest()
 legacy=source.OUT/'media'/key/'metadata.json'
 if legacy.exists():return m.read(legacy),legacy
 path=OUT/'media'/key/'metadata.json'
 if video in PENDING:
  task=PENDING.pop(video);path=Path(task if isinstance(task,str) else task.get())
 elif not path.exists():
  group=next(x for x in WAITING if x['video']==video);WAITING.remove(group)
  path=Path(POOL.apply_async(prepare_video,(group,)).get())
 fill()
 return m.read(path),path

def reports():
 import deterministic_rollout300_report as r
 frozen=m.read(NESTED/'pilot_900/samples.json')
 records={}
 for item in m.read(OUT/'metadata/original300.json'):
  assert m.sha(item['path'])==item['sha256'],'Original result changed'
  record=m.read(item['path']);records[record['sample']['sample_key']]=record
 additional=[m.read(OUT/'rollouts'/(row['sample_key']+'.json')) for row in frozen[300:]]
 assert len(additional)==600
 records.update({x['sample']['sample_key']:x for x in additional})
 r.OUT=OUT;r.REPORT=REPORT
 base_report(records=additional,cohort_label='추가600개 (고정900 중 기존300 제외)')
 for n in [600,900]:
  run=f'deterministic_rollout_pilot{n}_v2';out=relocate_path(ROOT/'artifacts/runs'/run);dest=relocate_path(ROOT/'results'/run)
  for category in ['metadata','metrics','validation','code']:(out/category).mkdir(parents=True,exist_ok=True)
  dest.mkdir(parents=True,exist_ok=True)
  immutable(out/'metadata/cohort.json',frozen[:n])
  refs=[]
  for row in frozen[:n]:
   record=records[row['sample_key']];assert record['sample']==row
   parent=ORIGINAL if row in frozen[:300] else OUT
   path=parent/'rollouts'/(row['sample_key']+'.json');refs.append(dict(path=str(path),sha256=m.sha(path)))
  m.write(out/'metadata/source_records.json',refs)
  for name in ['deterministic_rollout900.py','deterministic_rollout300.py','deterministic_rollout300_report.py']:
   p=Path(__file__).with_name(name);(out/'code'/name).write_bytes(p.read_bytes())
  r.OUT=out;r.REPORT=dest
  base_report(records=[records[row['sample_key']] for row in frozen[:n]],cohort_label=f'고정 중첩 pilot{n}')
  m.write(out/'metadata/status.json',dict(state='complete',stage='complete',completed=n,total=n,timestamp=m.stamp(),inference='original300 reused; only additional600 inferred'))
  p=dest/'REPORT.md';p.write_text(p.read_text()+'\n[기존300 보고서](../deterministic_rollout_pilot300_v2/REPORT.md) · [추가600 보고서](../deterministic_rollout_additional600_v2/REPORT.md)\n')
 m.write(OUT/'validation/integration.json',dict(passed=True,original300_unchanged=True,additional600=600,
    nested600=True,nested900=True,records_copied=False))

def main():
 global POOL,WAITING
 if '--report' in sys.argv:reports();return
 rows,groups=setup();m.OUT=OUT;m.REPORT=REPORT;m.TOTAL=600;m.input_cache=input_cache
 import deterministic_rollout300_report as report_module
 report_module.report=reports
 with (OUT/'metadata/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  p=m.Progress()
  try:
   m.write(OUT/'validation/cpu.json',m.selftest())
   remaining={row['video_path'] for row in rows if not (OUT/'rollouts'/(row['sample_key']+'.json')).exists()}
   WAITING=[g for g in groups if g['video'] in remaining]
   POOL=multiprocessing.get_context('spawn').Pool(1,maxtasksperchild=1);fill()
   m.run(rows,p)
  except BaseException:p.update(state='failed',stage='failed',error=traceback.format_exc());raise
  finally:
   p.stop.set()
   if POOL is not None:POOL.terminate();POOL.join()

if __name__=='__main__':main()
