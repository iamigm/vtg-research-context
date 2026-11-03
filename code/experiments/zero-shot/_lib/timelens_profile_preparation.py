from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import json,time,cProfile,pstats,io
from timelens_span_head import Prepared
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404');OUT=relocate_path(BASE/'performance')
results=[]
for model in ['qwen25','qwen3']:
 files=sorted(result_glob((relocate_path(BASE/model)), '*.json'),key=lambda p:p.stat().st_mtime)[-3:];records=[json.loads(p.read_text()) for p in files];loader=Prepared([r['sample'] for r in records],model)
 start=time.time();warm=loader[0];del warm;warm_seconds=time.time()-start
 profile=cProfile.Profile();profile.enable()
 for i in [1,2]:
  start=time.time();item=loader[i];duration=time.time()-start;assert item['input_sha256']==records[i]['input_ids_sha256'];results.append(dict(model=model,sample_key=records[i]['sample']['sample_key'],prepare_seconds=duration,input_hash_identical=True,input_tokens=item['inputs'].input_ids.shape[1],warmup_seconds=warm_seconds));del item
 profile.disable();stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(25);write_result_text((relocate_path(OUT/(model+'_cpu_profile.txt'))), stream.getvalue());write_result_text((relocate_path(OUT/'preparation.json')), json.dumps(results,indent=2));print(json.dumps(results[-2:]),flush=True)
