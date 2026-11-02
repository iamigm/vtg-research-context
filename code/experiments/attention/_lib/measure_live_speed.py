"""Read-only production comparison with an explicit cohort-confounding caveat."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import json,statistics,time
ROOT=relocate_path(Path(__file__)).resolve().parents[1]
RUN=relocate_path(ROOT/'results/thinking_pilot_orders_v1');OUT=relocate_path(ROOT/'results/thinking_live_benchmark_v1')
rows=[]
for p in result_glob(RUN, 'samples/*/*/*/result.json'):
 if (relocate_path(p.parent/'complete.json')).exists():rows.append(json.loads(p.read_text()))
tables=[]
for order in ['video_query','query_video']:
 record={'order':order}
 for method in ['replay','live']:
  subset=[r for r in rows if r['dataset']=='charades-timelens' and r['order']==order and r.get('measurement_method','replay')==method and (method!='replay' or (r.get('generation_seconds') or 0)>1)]
  seconds=sum(r['total_seconds'] for r in subset);tokens=sum(r['generated_tokens'] for r in subset)
  record[method]={'n':len(subset),'worker_seconds':seconds,'generated_tokens':tokens,'ms_per_token':1000*seconds/tokens,'tokens_per_second':tokens/seconds,'mean_input_tokens':statistics.mean(r['input_tokens'] for r in subset),'mean_generated_tokens':tokens/len(subset),'sample_keys':[r['sample_key'] for r in subset]}
 record['speedup']=record['replay']['ms_per_token']/record['live']['ms_per_token'];record['time_reduction_pct']=100*(1-1/record['speedup']);tables.append(record)
throughput={}
for name in ['supervisor.log','supervisor_live.log']:
 logs=[]
 for line in (relocate_path(RUN/'logs'/name)).read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if r.get('stage')=='running':logs.append(r)
 a,b=logs[0],logs[-1];seconds=b['elapsed_seconds']-a['elapsed_seconds'];done=b['completed']-a['completed']
 throughput[name]={'completed_between_observations':done,'wall_seconds':seconds,'records_per_hour':done/seconds*3600}
result={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'completed_snapshot':len(rows),'comparison':tables,'overall_throughput':throughput,'caveats':['Different samples before/after; not a paired causal estimate.','Same-dataset/order comparison restricted to Charades, only 7 and 9 live records.','Mean input/output lengths differ; tokens/sec does not remove all sequence-length effects.','Excludes saved-generation recovery and first reused-generation/probe timing.','Worker time includes generation/capture, input transfer, text serialization and array validation; excludes preloading, final checksum and model loading.','Overall wall throughput mixes datasets and trace lengths; 4 workers may become fewer near completion.']}
write_result_text((relocate_path(OUT/'production_comparison.json')), json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# 본 실험의 live 추출 속도 측정','',f"측정 시각: {result['timestamp']}, 완료 snapshot {len(rows)}/600.",'', 'Charades의 동일 입력 순서끼리 비교했다. 토큰당 시간 = 합산 worker 소요 시간 / 합산 생성 토큰 수. 동일 샘플을 재실행한 paired 비교는 아니므로 측정된 개선폭을 알고리즘만의 인과 효과로 단정하지 않는다.','', '| 입력 순서 | replay n | live n | replay ms/token | live ms/token | 속도비 | 시간 감소 |','|---|---:|---:|---:|---:|---:|---:|']
for t in tables:lines.append(f"| {t['order']} | {t['replay']['n']} | {t['live']['n']} | {t['replay']['ms_per_token']:.2f} | {t['live']['ms_per_token']:.2f} | {t['speedup']:.3f}× | {t['time_reduction_pct']:.1f}% |")
lines+=['','## 해석','', '본 실험의 토큰당 작업 시간은 약 16–19% 줄었다. 고정 입력의 짧은 벤치마크에서 얻은 1.55–1.71배보다 개선폭이 작다. GPU 동시 작업 수, 입력·출력 길이, 데이터셋 및 저장 비용 차이가 영향을 줄 수 있다. 측정은 그 원인별 기여를 분리하지 않는다.','', '단위 시간당 완료 샘플 수는 전환 전후 데이터셋/trace 길이가 달라 직접적인 가속 지표로 쓰지 않는다.']
for k,v in throughput.items():lines.append(f"- {k}: {v['records_per_hour']:.1f}회/시간 ({v['completed_between_observations']}회, {v['wall_seconds']/3600:.2f}시간 구간)")
lines+=['','## 제한','']+['- '+x for x in result['caveats']]
write_result_text((relocate_path(OUT/'PRODUCTION_COMPARISON.md')), '\n'.join(lines)+'\n')
print(json.dumps({'snapshot':len(rows),'speedups':[t['speedup'] for t in tables],'reductions':[t['time_reduction_pct'] for t in tables],'throughput':throughput},indent=2))
