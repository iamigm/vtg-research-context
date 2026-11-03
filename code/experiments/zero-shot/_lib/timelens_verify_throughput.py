from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import json,time,subprocess,statistics
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404');OUT=relocate_path(BASE/'performance');START=time.time();gpu=[];initial={m:{p.name for p in result_glob((relocate_path(BASE/m)), '*.json')} for m in ['qwen25','qwen3']}
while True:
    groups={m:sorted([p for p in result_glob((relocate_path(BASE/m)), '*.json') if p.name not in initial[m]],key=lambda p:p.stat().st_mtime) for m in initial}
    line=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();gpu.append(float(line.splitlines()[0]));elapsed=time.time()-START
    progress=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage='post-tuning measurement',completed={m:len(v) for m,v in groups.items()},target_per_model=60,elapsed_seconds=elapsed)
    write_result_text((relocate_path(OUT/'throughput_status.json')), json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
    if min(map(len,groups.values()))>=60 and elapsed>=120:break
    if elapsed>1200:break
    time.sleep(15)
result=dict(gpu_average_percent=statistics.mean(gpu),elapsed_seconds=elapsed,models={})
for model,paths in groups.items():
    if len(paths)<2:continue
    data=[json.loads(p.read_text()) for p in paths];wall=paths[-1].stat().st_mtime-paths[0].stat().st_mtime;count=len(paths)-1;generation=sum(r['elapsed_seconds'] for r in data[1:]);baseline=594.5/119 if model=='qwen25' else 773/119
    result['models'][model]=dict(samples=count,wall_seconds=wall,seconds_per_response=wall/count,baseline_seconds_per_response=baseline,observed_speed_ratio=baseline/(wall/count),generation_fraction=relocate_path(generation/wall),median_video_duration=statistics.median(r['sample']['duration'] for r in data),median_input_tokens=statistics.median(r['input_tokens'] for r in data))
write_result_text((relocate_path(OUT/'throughput_after.json')), json.dumps(result,indent=2));tuning=json.loads((relocate_path(OUT/'tuning.json')).read_text());tuning['post_restart_validation']='measured';tuning['measured_models']=list(result['models']);write_result_text((relocate_path(OUT/'tuning.json')), json.dumps(tuning,indent=2))
with (relocate_path(OUT/'REPORT.md')).open('a') as f:
    f.write('\n## 변경 후 관측\n\n전후 샘플이 달라 영상 길이와 디코딩 난이도 차이가 포함된다. 아래 비율은 동일 입력의 통제 실험이 아닌 처리량 관측치다.\n\n| 모델 | 측정 응답 수 | 변경 전 초/응답 | 변경 후 초/응답 | 관측 속도 비율 |\n|---|---:|---:|---:|---:|\n')
    for m,v in result['models'].items():f.write(f"| {m} | {v['samples']} | {v['baseline_seconds_per_response']:.2f} | {v['seconds_per_response']:.2f} | {v['observed_speed_ratio']:.2f} |\n")
    f.write(f"\n변경 후 측정 구간 GPU 평균 사용률: {result['gpu_average_percent']:.1f}%.\n")
print(json.dumps(result),flush=True)
