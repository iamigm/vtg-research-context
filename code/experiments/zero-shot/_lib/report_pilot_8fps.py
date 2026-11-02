"""Write a complete or clearly marked partial 8fps report from saved responses."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
from collections import defaultdict,Counter
import sys,json,csv,hashlib,statistics,datetime
from zoneinfo import ZoneInfo
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/thinking_vs_instruct_8fps_v1')
sys.path[:0]=[str(relocate_path(ROOT/'_lib')),str(relocate_path(ROOT/'runtime/timelens_deps')),str(relocate_path(ROOT/'vendor/TimeLens'))]
from timelens_inputs import read,write,sha
from timelens.utils import extract_time,iou
from artifact_store import register

def score(answer,gt,rounded=False):
 p=extract_time(answer) or []
 return iou(gt,[round(v) for v in p[0]] if rounded else p[0]) if p else 0.
def build_report():
 rows=[];groups=defaultdict(list);sources=[];methods=Counter();missing=[];samples=read(relocate_path(OUT/'samples.json'))
 for sample in samples:
  for m in ['instruct','thinking']:
   for order in ['video_query','query_video']:
    p=relocate_path(OUT/'samples'/m/sample['dataset']/sample['sample_key']/order/'result.json');tp=relocate_path(OUT/'token_traces'/m/sample['dataset']/sample['sample_key']/order/'trace.json')
    if not p.exists():missing.append(dict(model=m,dataset=sample['dataset'],sample_key=sample['sample_key'],order=order));continue
    r=read(p);assert r['sample']==sample;gt=sample['span'][0];v=score(r['answer'],gt);assert abs(v-r['iou'])<1e-12
    bp=(relocate_path(ROOT/'predictions/qwen3'/sample['dataset']/sample['sample_key']/order/'answer.json')) if m=='instruct' else relocate_path(ROOT.parent/'attention/results/thinking_pilot_orders_v1/samples'/sample['dataset']/sample['sample_key']/order/'result.json')
    b=read(bp);source=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),baseline_path=str(bp),baseline_sha256=sha(bp));sources.append(source)
    row=dict(model=m,dataset=sample['dataset'],sample_key=sample['sample_key'],order=order,iou_8fps=v,iou_2fps=score(b['answer'],gt),iou_8fps_rounded=score(r['answer'],gt,True),iou_2fps_rounded=score(b['answer'],gt,True),tokens=r['generated_tokens'],parse_failed=not bool(extract_time(r['answer'])),truncated=not r['eos_reached'],trace_saved=tp.exists())
    rows.append(row)
    for ds in [sample['dataset'],'all']:groups[(m,order,ds)].append(row)
 metrics=[]
 for (m,order,ds),rs in sorted(groups.items()):
  x=dict(model=m,order=order,dataset=ds,n=len(rs),expected=300 if ds=='all' else 100,complete=len(rs)==(300 if ds=='all' else 100),mean_generated_tokens=statistics.mean(r['tokens'] for r in rs),parse_failures=sum(r['parse_failed'] for r in rs),truncated=sum(r['truncated'] for r in rs),traces=sum(r['trace_saved'] for r in rs))
  for fps in [2,8]:
   x[f'miou_{fps}fps']=statistics.mean(r[f'iou_{fps}fps'] for r in rs);x[f'miou_{fps}fps_rounded']=statistics.mean(r[f'iou_{fps}fps_rounded'] for r in rs)
   for th in [.3,.5,.7]:x[f'r{int(th*10):02d}_{fps}fps']=statistics.mean(r[f'iou_{fps}fps']>=th for r in rs)
  x['delta_miou_pp']=100*(x['miou_8fps']-x['miou_2fps']);metrics.append(x)
 def csvfile(name,rs):
  if rs:
   with (relocate_path(OUT/name)).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 csvfile('performance_metrics.csv',metrics);csvfile('paired_2fps_8fps.csv',rows)
 summary=dict(timestamp=datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),complete=len(rows)==1200,responses=len(rows),expected=1200,traces=sum(r['trace_saved'] for r in rows),metrics=metrics,missing=missing,scoring='first predicted interval vs single GT; parser failures zero; no rounding/clipping; rounded sensitivity separate',rollouts=False)
 write(relocate_path(OUT/'performance_summary.json'),summary);write(relocate_path(OUT/'report_sources.json'),sources)
 lines=['# Qwen3-VL-8B Instruct / Thinking — 8fps pilot zero-shot','',f"작성: {summary['timestamp']}. **{'전체 완료' if summary['complete'] else '부분 완료'}: {len(rows):,}/1,200개 응답, trace {summary['traces']:,}개.**",'', 'pilot 300개(Charades·ActivityNet·QVHighlights 각 100개)를 모델 2종과 입력 순서 2종으로 평가한다. 영상→질문(video_query), 질문→영상(query_video)을 모두 포함한다. Rollout은 제외했다.','']
 if not summary['complete']:lines+=['현재 보고서는 저장된 완료 응답만 집계한 중간 결과다. 1,033개 시점에 진행 상태 기록용 임시 파일의 스레드 간 충돌로 중단된 이력이 있으며, 기록 함수를 수정해 남은 샘플을 재개했다. 미완료 목록은 performance_summary.json에 보존한다. 완료 후 보고서를 자동 갱신한다.','']
 lines+=['## 조건 및 평가','', '| 항목 | Instruct | Thinking |','|---|---|---|','| Checkpoint | Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Thinking |','| 생성 | greedy | sampling T=1, top_p=.95, top_k=20, seed=42 |','| 최대 생성 토큰 | 512 | 16,384 |','| 영상 입력 | 8fps 요청, min_tokens=64, total_tokens=14,336 | 동일 |','| 정밀도·backend | BF16·FlashAttention2 | 동일 |','', '기존 2fps와 동일 cohort·프롬프트·모델별 생성 설정을 사용한다. 실제 frame sampling은 원본 fps와 최대 768프레임 제한을 따른다. 따라서 긴 영상의 실효 fps는 8보다 낮을 수 있고, 고정 visual token 예산에서 공간 해상도도 달라질 수 있다.','', '공통 TimeLens extract_time으로 최종 답변의 첫 구간을 파싱하고 단일 GT와 IoU를 계산했다. 파싱 실패는 0점, clipping·반올림은 적용하지 않는다. CSV에는 정수 반올림 민감도 결과도 함께 저장했다. 두 모델은 decoding 정책과 생성 예산이 달라 checkpoint 단독 효과로 해석할 수 없다.','', '## 성능과 동일 샘플의 2fps 비교','', '모든 지표는 %. 2fps 값은 해당 8fps 완료분과 정확히 같은 sample ID·입력 순서의 재채점 값이다. 미완료 행은 일부 샘플 결과로, 완성된 다른 행과 직접 비교하지 않는다. all은 query 수 가중 평균이다.','', ]
 for model,label in [('instruct','Instruct: 2fps vs 8fps'),('thinking','Thinking: 2fps vs 8fps')]:
  lines+=['',f'### {label}','','각 수치는 **2fps → 8fps**, 단위는 %다. R@.3/.5/.7은 첫 예측 구간 IoU가 각각 .3/.5/.7 이상인 샘플 비율이다.','', '| 입력 순서 | 벤치마크 | N / 목표 | 상태 | R@.3 (2→8fps) | R@.5 (2→8fps) | R@.7 (2→8fps) | mIoU (2→8fps) | ΔmIoU pp |','|---|---|---:|---|---:|---:|---:|---:|---:|']
  for order in ['video_query','query_video']:
   for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens','all']:
    g=next((g for g in metrics if g['model']==model and g['order']==order and g['dataset']==ds),None)
    if g is None:continue
    pairs=['{:.2f} → {:.2f}'.format(100*g[k+'_2fps'],100*g[k+'_8fps']) for k in ['r03','r05','r07','miou']]
    state='완료' if g['complete'] else '**부분 완료**'
    lines.append(f"| {g['order']} | {g['dataset']} | {g['n']}/{g['expected']} | {state} | "+' | '.join(pairs)+f" | {g['delta_miou_pp']:+.2f} |")
 lines+=['','## 생성량과 실패','', '| 모델 | 순서 | N | 평균 생성 토큰 | 파싱 실패 | EOS 미도달 |','|---|---|---:|---:|---:|---:|']
 for g in metrics:
  if g['dataset']=='all':lines.append(f"| {g['model']} | {g['order']} | {g['n']} | {g['mean_generated_tokens']:.1f} | {g['parse_failures']} | {g['truncated']} |")
 lines+=['','## 결과 해석','']
 for g in metrics:
  if g['dataset']=='all':lines.append(f"- {g['model']} / {g['order']}: {'완료된 전체 300개' if g['complete'] else '현재 완료된 일부 샘플'}에서 8fps의 mIoU는 {100*g['miou_8fps']:.2f}%이며, 동일 샘플 2fps 대비 {g['delta_miou_pp']:+.2f}pp다.")
 lines+=['','단일 seed 결과이며 seed 변동성이나 신뢰구간은 평가하지 않았다. FPS 증가의 효과는 시간·공간 sampling 변화와 함께 해석해야 한다.','', '## Token trace와 불확실성 분석','', 'raw/processed top-30, 전체 어휘 FP32 entropy(nats), 선택 토큰 확률·순위, span alphabet/EOS mass를 저장한다. top-30은 재정규화하지 않는다. processed는 sampling에서 temperature·top-k·top-p까지 적용한 선택 직전 분포이며 raw와 구분한다. Thinking의 reasoning·answer·경계 토큰도 분리한다.','', '새 응답은 생성과 동시에 수집한다. trace 수집 전에 생성된 응답만 저장된 생성 토큰과 입력 hash를 유지한 fixed-prefix replay로 보완했다. 각 trace에 측정 방법을 명시했으며 attention 행렬 추출을 위한 추가 replay는 없다.','']
 if (relocate_path(OUT/'token_trace_summary.json')).exists():
  t=read(relocate_path(OUT/'token_trace_summary.json'));lines +=[f"Trace 분석: {t['responses']:,}개 응답 / {t['steps']:,}개 생성 단계. 지원 문법 등으로 인한 분석 제외 기록 {t['excluded']:,}건은 trace_excluded.json에 명시했다. 제외는 전체 성능 채점과 원본 trace 보존에 영향을 주지 않는다.",'', '| 모델 | 순서 | N | Answer raw H | Reasoning raw H | Answer top30 mass |','|---|---|---:|---:|---:|---:|']
  for g in t['metrics']:
   if g['dataset']=='all':
    fmt=lambda v:'NA' if v is None else f'{v:.4f}'
    lines.append(f"| {g['model']} | {g['order']} | {g['n']} | {fmt(g['answer_raw_entropy_nats'])} | {fmt(g['reasoning_raw_entropy_nats'])} | {fmt(g['answer_raw_top30_mass'])} |")
 lines+=['','[토큰·불확실성 상세 보고서](TOKEN_TRACE_REPORT.md)에서 기존과 같은 rounded-prefix 자릿수별 최초 오류, GT로 반대 endpoint를 고정한 정답 가능성, top30 정답 가능/불가능/미확인 후보 확률 질량을 확인할 수 있다.','', '## 속도 개선과 검증','', 'GPU 통계를 응답 단위로 CPU에 전송하고, processor가 없는 greedy의 동일 raw/processed 분포를 재사용했다. 같은 샘플의 두 입력 순서에서는 비텍스트 입력 동일성 검사 후 영상 특징을 재사용한다. 샘플·모델이 바뀌면 cache를 초기화한다. 단일 8fps 입력에서 기존 수집기 대비 약 3.6%, vision cache로 추가 약 9.3% 가속을 측정했다. 생성 ID와 모든 trace 값의 일치를 검증했으며, 이는 전체 cohort 가속률을 의미하지 않는다. [측정 원자료·범위](performance/REPORT.md).','', '완료분 응답의 sample·GT·저장 IoU를 재검증하고 2fps 출처 hash를 기록했다. trace 검증 범위와 입력 hash·생성 ID·확률 합·top30 정렬·entropy 검사는 [trace_validation.json](trace_validation.json)에 기록했다.','', '## 저장 자료','', '- [성능 지표 CSV](performance_metrics.csv) · [동일 샘플 2fps/8fps CSV](paired_2fps_8fps.csv) · [성능 JSON·미완료 목록](performance_summary.json)','- [Token trace 집계](token_trace_metrics.csv) · [응답별 지표](token_trace_per_response.csv) · [digit 조건 집계](digit_conditions.csv) · [endpoint 지표](per_endpoint.csv) · [후보 확률 질량](blue_candidate_mass.csv)','- [입력·생성 설정](config.json) · [응답 출처 hash](report_sources.json) · [trace 출처 hash](trace_index.json) · [진행 상태](status.json)']
 write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
 register('pilot8fps.v1.analyses','analyses',OUT,'8fps pilot performance, matched 2fps comparison and token trace report; no rollouts','_lib/report_pilot_8fps.py','pilot_300','complete' if summary['complete'] else 'building',['pilot_300.cohort','pilot8fps.v1.token_traces'])
 print(json.dumps(dict(report=str(relocate_path(OUT/'REPORT.md')),responses=len(rows),complete=summary['complete'])),flush=True)
 return summary
if __name__=='__main__':
 build_report()
 if '--with-traces' in sys.argv:
  from pilot_8fps_traces import report
  report(ROOT,OUT,expected=None);build_report()
