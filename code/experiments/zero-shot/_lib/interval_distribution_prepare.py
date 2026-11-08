"""Freeze configuration and code before a new run; never call while running."""
import sys,json,hashlib,shutil,py_compile,os
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir());sys.path.insert(0,str(root/'_lib'))
from interval_distribution_core import *
import fcntl
lock=(OUT/'metadata/run.lock');lock.parent.mkdir(parents=True,exist_ok=True)
locked=lock.open('a');fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (OUT/'queries').exists() and any((OUT/'queries').glob('*.json')):raise RuntimeError('Run already contains completed queries; use frozen runtime to resume instead of re-freezing')
for name in ('metadata','logs','code','trees','queries','inputs','semantic/instruct','semantic/embedding','verification','web','work','validation'):(OUT/name).mkdir(parents=True,exist_ok=True)
names=[p.name for p in (root/'_lib').glob('interval_distribution_*.py')]+['interval_distribution_viewer.js']
files=[]
for name in names:
 source=root/'_lib'/name;dest=OUT/'code'/name
 if name.endswith('.py'):py_compile.compile(str(source),doraise=True)
 shutil.copyfile(source,dest);files.append(dict(source=str(source),snapshot=str(dest),sha256=sha(dest)))
shutil.copyfile(root/'_lib/interval_distribution_viewer.js',OUT/'web/viewer.js')
write(OUT/'metadata/config.json',CONFIG)
base=root/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_core.py'
files.append(dict(source=str(base),snapshot=str(base),sha256=sha(base)))
wrapper=EMBED/'scripts/qwen3_vl_embedding.py';files.append(dict(source=str(wrapper),snapshot=str(wrapper),sha256=sha(wrapper)))
write(OUT/'metadata/manifest.json',dict(config=CONFIG,config_hash=digest(CONFIG),cohort=str(COHORT),cohort_sha256=sha(COHORT),files=files,timestamp=stamp()))
models={}
for model in (MODEL,EMBED,VERIFIER):
 index=read(model/'model.safetensors.index.json');weights=sorted(set(index['weight_map'].values()));assert all((model/w).exists() for w in weights)
 models[str(model)]=dict(config_sha256=sha(model/'config.json'),tokenizer_sha256=sha(model/'tokenizer.json'),index_sha256=sha(model/'model.safetensors.index.json'),weight_files=[dict(name=w,bytes=(model/w).stat().st_size) for w in weights])
write(OUT/'metadata/models.json',models)
write(OUT/'metadata/status.json',dict(state='preparing',stage='gpu_smoke',active_queries=300,completed=0,total=900,timestamp=stamp()))
write(OUT/'metadata/source_references.json',dict(VideoMind=dict(commit='6721d2564eeb9ea8ab6c0ab8eaa39bd231ee8927',code='https://github.com/yeliudev/VideoMind/blob/6721d2564eeb9ea8ab6c0ab8eaa39bd231ee8927/videomind/dataset/wrappers/verifying.py',adaptation='crop each predicted segment and independently verify visible event'),VideoHV=dict(commit='ddc160bd05e2d64922ca717d1834500fa7d8f492',code='https://github.com/Haorane/VideoHV-Agent/blob/ddc160bd05e2d64922ca717d1834500fa7d8f492/VideoHV-Agent/video_hv/verifier.py',adaptation='structured evidence support/contradiction; no replication claim')))
from interval_distribution_report import report
report()
print(OUT)
