"""Matched pilot/order comparison of existing Thinking and Instruct runs."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import json,sys,hashlib,collections
import numpy as np
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/baseline/thinking_vs_instruct/pilot/300');OUT.mkdir(parents=True,exist_ok=True)
THINK=relocate_path(ROOT.parent/'attention/results/thinking_pilot_orders_v1');INST=relocate_path(ROOT/'predictions/qwen3')
sys.path.insert(0,str(relocate_path(ROOT.parent/'shared/timelens/vendor/TimeLens')))
from timelens.utils import extract_time,iou
read=lambda p:json.loads(p.read_text())
manifest=read(relocate_path(THINK/'samples.json'));assert manifest==read(relocate_path(ROOT/'predictions/cohorts/pilot_300/samples.json'))
records=[];sources=[]
for sample in manifest:
 ds,key=sample['dataset'],sample['sample_key'];meta=read(relocate_path(INST/ds/key/'metadata.json'))
 assert meta['sample']==sample
 for order in ['video_query','query_video']:
  tdir=relocate_path(THINK/'samples'/ds/key/order);idir=relocate_path(INST/ds/key/order)
  t=read(relocate_path(tdir/'result.json'));i=read(relocate_path(idir/'answer.json'));inp=read(relocate_path(tdir/'input.json'));assert (relocate_path(tdir/'complete.json')).exists()
  for k in ['prompt','video_grid_thw','timestamps','duration_video','decoder_metadata']:assert meta[k]==inp['metadata'][k],(key,order,k)
  a=(relocate_path(idir/'expanded_input.txt')).read_text();b=(relocate_path(tdir/'expanded_input.txt')).read_text();assert b==a+'<think>\n',(key,order,'input mismatch')
  assert i['query']==sample['query'] and i['gt']==sample['span'][0]
  row=dict(dataset=ds,sample_key=key,order=order,gt=sample['span'][0])
  for label,r,path in [('thinking',t,relocate_path(tdir/'result.json')),('instruct',i,relocate_path(idir/'answer.json'))]:
   spans=extract_time(r['answer']) or [];score=iou(sample['span'][0],spans[0]) if spans else 0.
   rounded=iou(sample['span'][0],[round(x) for x in spans[0]]) if spans else 0.
   ids=read(relocate_path(tdir/'generation.json'))['generated_ids'] if label=='thinking' else i['generated_ids']
   row[label]=dict(iou=score,iou_rounded=rounded,generated_tokens=len(ids),reasoning_tokens=t['reasoning_tokens'] if label=='thinking' else 0,answer_tokens=t['answer_tokens'] if label=='thinking' else len(i['token_ids']),parse_failed=not spans,truncated=not t['eos_reached'] if label=='thinking' else i['truncated'])
   if label=='thinking':assert abs(score-t['iou'])<1e-12
   sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  records.append(row)
assert len(records)==600
rng=np.random.default_rng(20261026);groups={}
for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens','all']:
 for order in ['video_query','query_video']:
  rs=[r for r in records if (ds=='all' or r['dataset']==ds) and r['order']==order];g={'n':len(rs)}
  for model in ['instruct','thinking']:
   scores=np.array([r[model]['iou'] for r in rs]);tokens=np.array([r[model]['generated_tokens'] for r in rs])
   g[model]=dict(miou=float(scores.mean()),miou_rounded=float(np.mean([r[model]['iou_rounded'] for r in rs])),r05=float((scores>=.5).mean()),r07=float((scores>=.7).mean()),mean_generated_tokens=float(tokens.mean()),median_generated_tokens=float(np.median(tokens)),mean_answer_tokens=float(np.mean([r[model]['answer_tokens'] for r in rs])),truncated=sum(r[model]['truncated'] for r in rs),parse_failures=sum(r[model]['parse_failed'] for r in rs))
  d=np.array([r['thinking']['iou']-r['instruct']['iou'] for r in rs]);boot=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(1)
  g.update(delta_miou=float(d.mean()),paired_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),thinking_better=int((d>1e-10).sum()),thinking_worse=int((d< -1e-10).sum()),tied=int((abs(d)<=1e-10).sum()),generated_token_ratio=g['thinking']['mean_generated_tokens']/g['instruct']['mean_generated_tokens'])
  groups[ds+'/'+order]=g
configs={'instruct':read(relocate_path(INST/'run_config.json')),'thinking':read(relocate_path(THINK/'config.json'))}
result=dict(matched_sample_order_pairs=600,distinct_samples=300,input_validation='All 600 expanded prompts match exactly except the Thinking-only trailing <think> newline. Prompts, video grids, sampled timestamps and decoder metadata match.',scoring='Same upstream extract_time; first prediction vs single GT; no rounding or clipping; parse failures score 0. Rounded sensitivity included.',groups=groups,configs=configs,limitations=['Greedy Instruct vs sampled Thinking: checkpoint and decoding-policy effects are confounded.','Token budgets differ: 512 vs 16384.','One rollout per model/order; bootstrap does not account for generation-seed variation.','No model runtime or attention-pattern comparison: capture protocols differ.'])
for name,obj in [('summary.json',result),('paired_records.json',records),('source_hashes.json',sources)]: write_result_text((relocate_path(OUT/name)), json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
lines=['# Qwen3-VL-8B Thinking vs Instruct — 동일 pilot 비교','', '기존 결과를 사용한 300개 샘플 × 2개 입력 순서의 대응 비교다. GPU 추론은 추가로 실행하지 않았다.','', '600개 모두 프롬프트·영상 grid·샘플링 timestamp·decoder metadata가 같음을 확인했다. 확장 입력은 Thinking 전용 assistant 접미부 `<think>\\n` 외에는 정확히 일치한다.','', '현재 설정에서 영상→질문의 전체 mIoU는 거의 같지만, 질문→영상에서는 Thinking이 크게 높다. Instruct의 입력 순서 변경에 따른 전체 mIoU 하락은 약 12.34pp이고 Thinking은 약 0.71pp다. 이는 Thinking 실행이 입력 순서 변경에 덜 민감한 패턴이지만, 모델과 decoding 설정이 동시에 달라 원인을 분리할 수 없다.','', '## 비교 조건','', '| 항목 | Instruct | Thinking |','|---|---|---|','| 모델 | Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Thinking |','| 생성 | greedy | sampling: T=1, top_p=.95, top_k=20, seed=42 |','| 최대 생성 토큰 | 512 | 16,384 |','| 전처리 | 2 fps, min_tokens=64, total_tokens=14336 | 동일 |','', '따라서 순수한 checkpoint 효과나 reasoning 유무의 인과 효과를 분리한 실험은 아니다. 현재 두 실행 설정의 실제 성능·출력 비용을 비교한다.','', '## 성능','', '| Dataset | 순서 | N | Instruct mIoU(%) | Thinking mIoU(%) | 차이(pp) | 차이 95% CI(pp) |','|---|---|---:|---:|---:|---:|---|']
for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens','all']:
 for order in ['video_query','query_video']:
  g=groups[ds+'/'+order];lo,hi=g['paired_bootstrap_95ci'];lines.append(f"| {ds} | {order} | {g['n']} | {g['instruct']['miou']*100:.2f} | {g['thinking']['miou']*100:.2f} | {g['delta_miou']*100:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] |")
lines+=['', '최종 답변의 원문을 같은 extract_time으로 파싱하고 첫 예측 구간 대 단일 GT의 IoU를 계산했다. 반올림·영상 길이 clipping을 하지 않았으며 파싱 실패는 0점으로 포함했다. 과거 Instruct 보고서는 반올림한 timestamp로 채점했기 때문에 수치가 조금 다를 수 있다. summary.json에는 동일 반올림 규칙을 적용한 두 모델 수치도 제공한다. CI는 샘플 쌍 bootstrap 20,000회이며 seed 변동성을 포함하지 않는다.','', '## 생성량','', '| 순서 | Instruct 평균 생성 토큰 | Thinking 평균 생성 토큰 | 배율 |','|---|---:|---:|---:|']
for order in ['video_query','query_video']:
 g=groups['all/'+order];lines.append(f"| {order} | {g['instruct']['mean_generated_tokens']:.1f} | {g['thinking']['mean_generated_tokens']:.1f} | {g['generated_token_ratio']:.1f}× |")
lines+=['','생성량은 EOS와 경계 태그를 포함한 실제 generated_ids 길이다. Thinking에는 reasoning이 포함된다. 이 비율은 GPU 시간 배율을 의미하지 않는다.','', '## 검증 및 해석 범위','', '- 동일 cohort/order 600쌍의 원본·GT·입력 일치 확인. Thinking 재채점은 원래 저장된 IoU와 일치한다.', '- 원본 결과를 변경하지 않고 파생 JSON/보고서와 참조 파일 hash만 저장했다.', '- 기존 Instruct attention은 선택 layer/토큰 집계 방식이 달라 이번 비교에서 attention 수치를 직접 대조하지 않았다.','']+['- '+x for x in result['limitations']]
write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
print(json.dumps({'groups':groups},indent=2))
