"""Remove validated HTML copies of canonical generation records; leave originals intact."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
from collections import defaultdict
import json,hashlib,os,shutil,time
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'artifacts');start=time.time();audit={'removed_trace_files':0,'removed_bytes':0,'deleted_superseded':[],'deduplicated_files':[]}
def report(state):
 d=dict(audit,state=state,elapsed_seconds=time.time()-start);write_result_text((relocate_path(OUT/'cleanup.json')), json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k not in ['deduplicated_files','deleted_superseded']},ensure_ascii=False),flush=True)
for exp in ['timelens_rollout','timelens_top30']:
 base=relocate_path(ROOT/'results'/exp);traces=relocate_path(base/'traces')
 if not traces.exists():continue
 for i,path in enumerate(sorted(traces.rglob('*.js'))):
  parts=path.relative_to(traces).parts
  if exp=='timelens_rollout':
   model,variant,sample,roll=parts;original=relocate_path(base/'records'/model/sample/(variant+'_'+relocate_path(Path(roll)).stem+'.json'))
  else:model,name=parts;original=relocate_path(base/model/(relocate_path(Path(name)).stem+'.json'))
  text=path.read_text();assert text.startswith('window.receiveTrace(') and text.endswith(');\n'),path
  copied=json.loads(text.partition(',')[2][:-3]);canonical=json.loads(original.read_text());assert copied==canonical,('Nonidentical trace, retained',path)
  audit['removed_bytes']+=path.stat().st_size;path.unlink();audit['removed_trace_files']+=1
  if (i+1)%100==0:report('removing_verified_copies')
 for directory in sorted(traces.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if directory.is_dir():directory.rmdir()
 traces.rmdir()
# This temporary snapshot contains the superseded strict-string classification,
# not any unique model response or probability distribution.
archive=relocate_path(ROOT/'results/timelens_top30/digit_correctness/strict_archive_20261023_140206')
if archive.exists():
 audit['deleted_superseded'].append(str(archive.relative_to(ROOT)));audit['removed_bytes']+=sum(p.stat().st_size for p in archive.rglob('*') if p.is_file());shutil.rmtree(archive)
# Exact duplicate immutable report assets retain their required viewer paths
# as hard links, avoiding server restrictions on out-of-root symlinks.
prefix=relocate_path(ROOT/'results/timelens_top30/prefix_entropy');copied=relocate_path(ROOT/'results/timelens_top30/integrated/sources/prefix_entropy')
for p in copied.iterdir():
 original=relocate_path(prefix/p.name)
 if p.is_file() and original.is_file():
  if hashlib.sha256(p.read_bytes()).digest()!=hashlib.sha256(original.read_bytes()).digest():continue
  if p.is_symlink() or p.stat().st_ino!=original.stat().st_ino:
   size=0 if p.is_symlink() else p.stat().st_size;p.unlink();os.link(original,p);audit['removed_bytes']+=size;audit['deduplicated_files'].append(str(p.relative_to(ROOT)))
# Fonts and other static figure copies with identical names and bytes.
groups=defaultdict(list)
for p in (relocate_path(ROOT/'results/timelens_top30')).rglob('*'):
 if p.is_file() and not p.is_symlink() and p.suffix in ['.ttf','.png']:
  groups[(p.name,p.stat().st_size)].append(p)
for paths in groups.values():
 if len(paths)<2:continue
 canonical=paths[0]
 for p in paths[1:]:
  if p.stat().st_ino==canonical.stat().st_ino:continue
  if hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(canonical.read_bytes()).digest():
   size=p.stat().st_size;p.unlink();os.link(canonical,p);audit['removed_bytes']+=size;audit['deduplicated_files'].append(str(p.relative_to(ROOT)))
report('complete')
