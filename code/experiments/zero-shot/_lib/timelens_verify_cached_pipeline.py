"""Detached progress and throughput observation; excludes prior completed samples."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import json,time,subprocess,statistics
ROOT=relocate_path(Path(__file__)).resolve().parents[1]; BASE=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404'); OUT=relocate_path(BASE/'performance')
start=time.time(); gpu=[]; models=['qwen25','qwen3']
while True:
    groups={m:[] for m in models}
    for m in models:
        for p in result_glob((relocate_path(BASE/m)), '*.json'):
            if p.stat().st_mtime<start: continue
            r=json.loads(p.read_text())
            if r.get('pipeline',{}).get('version')=='video-cache-v1-ready-order': groups[m].append((p.stat().st_mtime,r))
        groups[m].sort(key=lambda x:x[0])
    gpu.append(float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]))
    done={m:len(v) for m,v in groups.items()}; elapsed=time.time()-start
    status=dict(stage='cached pipeline observation',timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),completed=done,target_per_model=60,elapsed_seconds=elapsed)
    write_result_text((relocate_path(OUT/'cached_pipeline_status.json')), json.dumps(status,indent=2)); print(json.dumps(status),flush=True)
    if min(done.values())>=60 or elapsed>1200: break
    time.sleep(15)
result=dict(gpu_average_percent=statistics.mean(gpu),elapsed_seconds=elapsed,models={})
for m,items in groups.items():
    if len(items)<2: continue
    rows=[r for _,r in items]; events=[e for r in rows for e in r['pipeline']['preparation']['cache']]
    result['models'][m]=dict(samples=len(items),seconds_per_response=(items[-1][0]-items[0][0])/(len(items)-1),
        mean_input_wait_seconds=statistics.mean(r['pipeline']['input_wait_seconds'] for r in rows[1:]),
        cache_hits=sum(e['hit'] for e in events),cache_lookups=len(events))
write_result_text((relocate_path(OUT/'cached_pipeline_result.json')), json.dumps(result,indent=2))
with (relocate_path(OUT/'CACHE_REPORT.md')).open('a') as f:
    f.write('\n## 적용 후 관측\n\n전후 입력이 달라 통제된 속도 비교가 아니다. 캐시 적중률과 입력 대기를 포함한 실행 중 관측이다.\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
print(json.dumps(dict(state='complete',**result)),flush=True)
