from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import json,collections,datetime
import numpy as np
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/thinking_pilot_orders_v1')
read=lambda p:json.loads(p.read_text())
rows=[];errors=[];validations=[]
for p in sorted(result_glob(OUT, 'samples/*/*/*/result.json')):
 r=read(p);v=read(relocate_path(p.parent/'validation.json'));c=read(relocate_path(p.parent/'complete.json'))
 assert v['passed'] and v['saved_file']['passed'] and c['passed']
 assert (relocate_path(p.parent/'attention.h5')).stat().st_size==r['attention_bytes']
 rows.append(r);validations.append(v)
manifest=read(relocate_path(OUT/'samples.json'));expected={(s['sample_key'],o) for s in manifest for o in ['video_query','query_video']}
assert len(rows)==600 and {(r['sample_key'],r['order']) for r in rows}==expected
bykey={(r['sample_key'],r['order']):r for r in rows}
rng=np.random.default_rng(20261026)
def group(rs):
 lengths=np.array([r['reasoning_tokens'] for r in rs]);ious=np.array([r['iou'] for r in rs])
 return dict(n=len(rs),miou=float(ious.mean()),r05=float((ious>=.5).mean()),r07=float((ious>=.7).mean()),reasoning_mean=float(lengths.mean()),reasoning_median=float(np.median(lengths)),reasoning_p95=float(np.percentile(lengths,95)),reasoning_max=int(lengths.max()),answer_mean=float(np.mean([r['answer_tokens'] for r in rs])),truncated=sum(not r['eos_reached'] for r in rs),unclosed_thinking=sum(not r['thinking_closed'] for r in rs),parse_failures=sum(not r['predicted_spans'] for r in rs),attention_gib=sum(r['attention_bytes'] for r in rs)/2**30)
def comparison(samples):
 d=np.array([bykey[(s['sample_key'],'query_video')]['iou']-bykey[(s['sample_key'],'video_query')]['iou'] for s in samples])
 bs=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(1)
 return dict(delta_miou=float(d.mean()),paired_bootstrap_95ci=np.quantile(bs,[.025,.975]).tolist(),improved=int((d>1e-10).sum()),worse=int((d< -1e-10).sum()),tied=int((abs(d)<=1e-10).sum()))
groups={};comparisons={}
for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens','all']:
 for order in ['video_query','query_video']:groups[ds+'/'+order]=group([r for r in rows if (ds=='all' or r['dataset']==ds) and r['order']==order])
 comparisons[ds]=comparison([s for s in manifest if ds=='all' or s['dataset']==ds])
status=read(relocate_path(OUT/'status.json'));prev=read(relocate_path(OUT/'migration_to_live/execution_replay.json'))
end=datetime.datetime.strptime(status['timestamp'],'%Y-%m-%dT%H:%M:%S%z').timestamp()
methods=dict(collections.Counter(r.get('measurement_method','replay') for r in rows))
result=dict(completed=600,samples=300,groups=groups,paired_comparisons=comparisons,method_counts=methods,completion_timestamp=status['timestamp'],wall_elapsed_seconds=end-prev['started_epoch'],validation=dict(all_records_passed=True,unique_complete_manifest_pairs=True,file_sizes_match=True,full_attention_checksums_recomputed_in_this_summary=False,max_category_sum_error=max(v['category_sum_max_error'] for v in validations)),truncated_cases=[dict(sample_key=r['sample_key'],order=r['order'],tokens=r['generated_tokens']) for r in rows if not r['eos_reached']],parse_failure_cases=[dict(sample_key=r['sample_key'],order=r['order']) for r in rows if not r['predicted_spans']],limitations=['Single seed=42 and one rollout per order; no repeated-seed uncertainty.', 'Scores reuse the current parser and first predicted span versus single GT; not an official leaderboard evaluation.', 'Paired bootstrap is over the fixed pilot pairs; it does not account for sampling-seed variability.', 'Attention patterns have been captured and validated, but this summary does not infer visual grounding or causal reliance from them.'])
write_result_text((relocate_path(OUT/'FINAL_SUMMARY.json')), json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# Thinking pilot 300 × 입력 순서 2종 — 최종 요약','',f"600/600회 완료. 종료: {status['timestamp']}. 최초 정상 supervisor 시작부터 약 {result['wall_elapsed_seconds']/3600:.2f}시간(전환·검증·공유 GPU 벤치마크 포함).",'', 'Qwen3-VL-8B-Thinking. 각 데이터셋 100개를 동일한 두 입력 순서로 비교했다. seed=42, temperature=1, top_p=.95, top_k=20, 최대 16,384 생성 토큰.','', '## 성능과 reasoning 길이','', '| Dataset | 영상→질문 mIoU(%) | 질문→영상 mIoU(%) | 변화(pp) | 영상→질문 reasoning 평균 | 질문→영상 reasoning 평균 |','|---|---:|---:|---:|---:|---:|']
for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens','all']:
 a=groups[ds+'/video_query'];b=groups[ds+'/query_video'];lines.append(f"| {ds} | {a['miou']*100:.2f} | {b['miou']*100:.2f} | {(b['miou']-a['miou'])*100:+.2f} | {a['reasoning_mean']:.1f} | {b['reasoning_mean']:.1f} |")
a=groups['all/video_query'];b=groups['all/query_video'];ci=comparisons['all']['paired_bootstrap_95ci']
lines+=['',f"질문→영상의 reasoning은 평균 {b['reasoning_mean']/a['reasoning_mean']:.2f}배 길지만, 전체 mIoU는 {(b['miou']-a['miou'])*100:+.2f}pp다. paired bootstrap 95% CI는 [{ci[0]*100:.2f}, {ci[1]*100:.2f}]pp로 입력 순서의 확실한 성능 우열을 주장하기 어렵다. 긴 reasoning이 더 좋은 답을 보장하지 않는다는 관측이며 인과 결론은 아니다.",'', 'mIoU는 저장된 최종 답변에서 파싱한 첫 구간과 manifest의 단일 GT의 IoU 평균이다. 파싱 실패는 0점으로 포함했다. 공식 leaderboard 평가와는 구분한다.','', '## 종료 및 저장 검증','', '- 600개 기록 모두 기존 저장 검증 통과. 이번 요약에서는 cohort/order 중복·누락, 완료 표시, validation 결과 및 attention 파일 크기를 다시 확인했다. 전체 약 199 GiB의 attention SHA-256을 다시 읽어 검증하지는 않았다.', '- QVHighlights 영상→질문 1건(qvhighlights-timelens_00147)은 reasoning 16,384토큰 한도에 도달해 최종 답변이 없다. 나머지 599건은 EOS 및 thinking 종료가 확인됐다.', '- 최종 구간 파싱 실패는 총 2건: 위 길이 제한 사례와 Charades 영상→질문 charades-timelens_01482.','',f"- Attention 실제 저장량: 영상→질문 {a['attention_gib']:.2f} GiB, 질문→영상 {b['attention_gib']:.2f} GiB, 합계 {a['attention_gib']+b['attention_gib']:.2f} GiB. HDF5 FP32 + LZF 기준이며 기타 JSON/텍스트·벤치마크 파일은 제외한다.",f'- 측정 방식: {methods}. 전환 이전 결과를 덮어쓰지 않았다.','', '## Attention 산출물','', '- 상세 분포: L0·7·14·21·28·35 각각의 32개 head 평균. 영상·질문·이전 생성 토큰을 포함한 모든 key를 보존.', '- 보조 요약: 전체 36개 레이어 × 32개 head × 7개 key 범주의 attention mass.', '- Query는 모든 생성 토큰을 예측하는 직전 위치. 종료 태그와 EOS 행도 포함하며 tokens.json으로 구분한다.', '- 현재 보고서는 결과·길이·저장 검증 요약이다. 어느 레이어/head가 영상 근거를 더 참조했는지에 대한 attention 내용 분석은 별도 작업이다.','', '## 실행 속도','', '별도 고정 입력 벤치마크에서는 live 추출이 1.55–1.71배 빨랐다. 본 실험의 동일 데이터셋·입력 순서별 비대응 표본 비교에서는 약 1.19–1.24배였다. 표본 길이와 동시 GPU 부하가 달라 두 수치를 혼동하지 않는다. 속도 측정 보고서 원문은 현재 결과 트리에 남아 있지 않다.','', '## 해석 범위','']+['- '+x for x in result['limitations']]
write_result_text((relocate_path(OUT/'FINAL_REPORT.md')), '\n'.join(lines)+'\n')
print(json.dumps({'overall_video_query':a,'overall_query_video':b,'paired':comparisons,'methods':methods,'storage_gib':a['attention_gib']+b['attention_gib'],'wall_hours':result['wall_elapsed_seconds']/3600},indent=2))
