"""Response-only pilot at 8 fps; detached, resumable and automatically validated."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import os,sys,json,time,threading,hashlib,traceback,gc
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/home/irteam/experiments/zero-shot/results/thinking_vs_instruct_8fps_v1/performance/before_pilot_8fps.py').resolve().parents[1]
sys.path[:0]=[str(relocate_path(ROOT/'runtime/timelens_deps')),str(relocate_path(ROOT/'vendor/TimeLens')),str(relocate_path(ROOT/'_lib'))]
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['FORCE_QWENVL_VIDEO_READER']='decord'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
OUT=relocate_path(ROOT/'results/thinking_vs_instruct_8fps_v1')
MANIFEST=relocate_path(ROOT/'predictions/cohorts/pilot_300/samples.json')
ORDERS=['video_query','query_video'];MODELS=['instruct','thinking']
from timelens_inputs import prepare_pair,read,write,sha

def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def dest(m,s,o):return relocate_path(OUT/'samples'/m/s['dataset']/s['sample_key']/o)

class Prepared:
 def __init__(self,rows,mp):self.rows=rows;self.mp=mp;self.proc=None
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  import torch
  if self.proc is None:
   import decord
   from functools import partial
   from transformers import AutoProcessor
   torch.set_num_threads(1);decord.VideoReader=partial(decord.VideoReader,num_threads=1)
   self.proc=AutoProcessor.from_pretrained(self.mp,padding_side='left',do_resize=False,local_files_only=True)
  pair,info=prepare_pair(self.rows[i],self.proc,fps=8)
  return self.rows[i],pair,info

def identity(x):return x

def aggregate(rows):
 from timelens.utils import extract_time,iou
 import numpy as np
 records=[]
 for s in rows:
  for o in ORDERS:
   rs={m:read(relocate_path(dest(m,s,o)/'result.json')) for m in MODELS}
   a=read(relocate_path(dest('instruct',s,o)/'input.json'));b=read(relocate_path(dest('thinking',s,o)/'input.json'))
   for k in ['prompt','video_grid_thw','timestamps','duration_video','decoder_metadata']:assert a['metadata'][k]==b['metadata'][k],(s['sample_key'],o,k)
   ta=(relocate_path(dest('instruct',s,o)/'expanded_input.txt')).read_text();tb=(relocate_path(dest('thinking',s,o)/'expanded_input.txt')).read_text()
   assert tb==ta+'<think>\n'
   for m,r in rs.items():
    assert r['sample']==s and r['fps']==8 and r['order']==o
    spans=extract_time(r['answer']) or [];score=iou(s['span'][0],spans[0]) if spans else 0.
    assert abs(score-r['iou'])<1e-12
    assert r['input_sha256']==hashlib.sha256(bytes_array(a['input_ids'] if m=='instruct' else b['input_ids'])).hexdigest()
   records.append(dict(dataset=s['dataset'],sample_key=s['sample_key'],order=o,**rs))
 groups={}
 for ds in sorted({s['dataset'] for s in rows})+['all']:
  for o in ORDERS:
   rr=[r for r in records if r['order']==o and (ds=='all' or r['dataset']==ds)]
   groups[ds+'/'+o]={'n':len(rr)}
   for m in MODELS:
    vals=np.array([r[m]['iou'] for r in rr]);groups[ds+'/'+o][m]=dict(miou=float(vals.mean()),r03=float((vals>=.3).mean()),r05=float((vals>=.5).mean()),r07=float((vals>=.7).mean()),parse_failures=sum(not r[m]['predicted_spans'] for r in rr),truncated=sum(not r[m]['eos_reached'] for r in rr),mean_generated_tokens=float(np.mean([r[m]['generated_tokens'] for r in rr])))
 write(relocate_path(OUT/'summary.json'),dict(groups=groups,validation_passed=True,matched_pairs=len(records),distinct_samples=len(rows)))
 write(relocate_path(OUT/'validation.json'),dict(passed=True,records=1200,matched_pairs=600,cohort_sha256=sha(MANIFEST),checks=['cohort and GT','input IDs hash','matched video metadata and prompt','Thinking-only assistant prefix','independent answer re-scoring']))
 lines=['# Qwen3-VL-8B Instruct / Thinking: 8fps pilot zero-shot','',f'완료: {stamp()}. 기존 pilot 300개(벤치마크별 100개), 입력 순서 2개, 모델 2개로 총 1,200회 평가했다.','', '8fps 요청, min_tokens=64, total_tokens=14336, BF16, FlashAttention2. 원본 프레임 수·영상 길이·전처리 상한에 따라 실제 sampling 간격은 달라질 수 있으며 각 input.json에 timestamps와 decoder metadata를 저장했다. Instruct: greedy, 최대 512 tokens. Thinking: T=1, top_p=.95, top_k=20, seed=42, 최대 16384 tokens. 기존 2fps 실행의 설정을 유지했다. 두 모델은 decoding 정책과 생성 예산이 달라 checkpoint만의 효과로 해석할 수 없다.','', '최종 답변에서 첫 구간을 추출해 단일 GT와 비교한다. 반올림 및 clipping 없이 채점하며 파싱 실패는 0점이다.','', '| Dataset | Order | N | Instruct mIoU % | Thinking mIoU % | Δ pp | Instruct R@.5 % | Thinking R@.5 % |','|---|---|---:|---:|---:|---:|---:|---:|']
 for key,g in groups.items():
  ds,o=key.split('/');a,b=g['instruct'],g['thinking'];lines.append(f"| {ds} | {o} | {g['n']} | {100*a['miou']:.2f} | {100*b['miou']:.2f} | {100*(b['miou']-a['miou']):+.2f} | {100*a['r05']:.2f} | {100*b['r05']:.2f} |")
 lines+=['','## 생성량 및 실패','', '| Order | Model | Mean tokens | Parse failures | Truncated |','|---|---|---:|---:|---:|']
 for o in ORDERS:
  g=groups['all/'+o]
  for m in MODELS:
   v=g[m];lines.append(f"| {o} | {m} | {v['mean_generated_tokens']:.1f} | {v['parse_failures']} | {v['truncated']} |")
 lines+=['','## 해석','']
 for o in ORDERS:
  g=groups['all/'+o];delta=100*(g['thinking']['miou']-g['instruct']['miou']);lines.append(f'{o}: Thinking의 mIoU 차이는 {delta:+.2f}pp다. 단일 seed에서 측정한 설정 간 차이다.')
 old=read(relocate_path(ROOT/'results/thinking_vs_instruct_v1/summary.json'))['groups'];lines+=['','## 기존 2fps 대비 mIoU 변화','', '| Order | Model | 2fps % | 8fps % | Δ pp |','|---|---|---:|---:|---:|']
 for o in ORDERS:
  for m in MODELS:
   a=old['all/'+o][m]['miou'];b=groups['all/'+o][m]['miou'];lines.append(f'| {o} | {m} | {100*a:.2f} | {100*b:.2f} | {100*(b-a):+.2f} |')
 lines+=['','동일 cohort와 생성 정책을 사용했으나 시간 sampling 증가 시 고정 visual token 예산 아래 공간 해상도도 달라질 수 있다. 검증 결과는 validation.json, 상세 통계는 summary.json을 참조한다.']
 write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')

def tracepath(m,s,o):return relocate_path(OUT/'token_traces'/m/s['dataset']/s['sample_key']/o/'trace.json')

def bytes_array(ids):
 import numpy as np
 return np.asarray(ids,dtype=np.int64).tobytes()

def main():
 import torch
 from transformers import AutoProcessor,AutoModelForImageTextToText
 from timelens.utils import extract_time,iou
 from pilot_8fps_traces import TraceCapture,report as trace_report
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
 rows=read(MANIFEST);assert len(rows)==300
 assert all(sum(s['dataset']==d for s in rows)==100 for d in {s['dataset'] for s in rows})
 write(relocate_path(OUT/'samples.json'),rows)
 config=dict(fps=8,min_tokens=64,total_tokens=14336,models=MODELS,orders=ORDERS,samples=300,total=1200,manifest_sha256=sha(MANIFEST),seed=42,instruct=dict(do_sample=False,max_new_tokens=512),thinking=dict(do_sample=True,temperature=1.,top_p=.95,top_k=20,max_new_tokens=16384),source_sha256=sha(Path('/home/irteam/experiments/zero-shot/results/thinking_vs_instruct_8fps_v1/performance/before_pilot_8fps.py')),attention_capture=False,token_trace_capture=True,top_k_recorded=30,processed_distribution='after temperature and top-k/top-p; before token selection',completed_response_trace='fixed saved prefix replay; original response preserved',rollouts=False)
 write(relocate_path(OUT/'config.json'),config)
 state=dict(stage='starting',completed=0,total=1200,counts={},started_at=stamp());start=time.time();lock=threading.Lock();stop=threading.Event()
 for m in MODELS:
  for s in rows:
   for o in ORDERS:
    if (relocate_path(dest(m,s,o)/'result.json')).exists() and tracepath(m,s,o).exists():state['completed']+=1;key=m+'/'+s['dataset'];state['counts'][key]=state['counts'].get(key,0)+1
 def update(**kw):
  with lock:state.update(kw)
 def emit():
  with lock:
   ss=dict(state);ss['counts']=dict(state['counts'])
  elapsed=time.time()-start;n=ss['completed'];ss.update(timestamp=stamp(),elapsed_seconds=elapsed,percentage=100*n/1200,eta_seconds=(elapsed*(1200-n)/max(n-initial,1) if n>initial else None),eta_scope='remaining inference and final aggregation; estimated from completed records in this process')
  write(relocate_path(OUT/'status.json'),ss)
  with (relocate_path(OUT/'progress.log')).open('a') as f:f.write(json.dumps(ss,ensure_ascii=False)+'\n')
 def heartbeat():
  while not stop.wait(20):emit()
 initial=state['completed'];emit();thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
 try:
  for m in MODELS:
   mp=relocate_path(relocate_path(Path('/home/irteam/backbones/models'))/('Qwen3-VL-8B-Instruct' if m=='instruct' else 'Qwen3-VL-8B-Thinking'))
   todo=[s for s in rows if any(not (relocate_path(dest(m,s,o)/'result.json')).exists() or not tracepath(m,s,o).exists() for o in ORDERS)]
   if not todo:continue
   update(stage='loading_model',model=m);emit()
   proc=AutoProcessor.from_pretrained(mp,padding_side='left',do_resize=False,local_files_only=True)
   model=AutoModelForImageTextToText.from_pretrained(mp,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
   allowed=[t['id'] for t in read(relocate_path(ROOT/'results/timelens_span_head/qwen3/shard0/config.json'))['allowed_tokens']]
   eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
   capture=TraceCapture(model,proc.tokenizer,allowed,eos);capture.model_label=m
   loader=torch.utils.data.DataLoader(Prepared(todo,str(mp)),batch_size=None,collate_fn=identity,num_workers=2,prefetch_factor=1,multiprocessing_context='spawn')
   update(stage='preparing_inputs')
   for sample,pair,info in loader:
    for o,cpu in zip(ORDERS,pair):
     d=dest(m,sample,o)
     if (relocate_path(d/'result.json')).exists() and tracepath(m,sample,o).exists():continue
     update(stage='generation',model=m,dataset=sample['dataset'],sample_key=sample['sample_key'],order=o)
     d.mkdir(parents=True,exist_ok=True);ids=cpu.input_ids[0].tolist();h=hashlib.sha256(bytes_array(ids)).hexdigest()
     existing=read(relocate_path(d/'result.json')) if (relocate_path(d/'result.json')).exists() else None
     if existing:assert existing['input_sha256']==h,'Replay input differs from saved generation'
     write(relocate_path(d/'input.json'),dict(sample=sample,order=o,input_ids=ids,metadata=info,fps=8))
     write_result_text((relocate_path(d/'expanded_input.txt')), proc.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False))
     inputs=cpu.to('cuda');torch.manual_seed(42);t=time.time();n=len(ids)
     kw=dict(do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512) if m=='instruct' else dict(do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,max_new_tokens=16384)
     capture.reset(existing['generated_ids'] if existing else None)
     if existing:kw['max_new_tokens']=len(existing['generated_ids'])
     with torch.inference_mode():output=model.generate(**inputs,**kw,use_cache=True)
     generated=output[0,n:].tolist()
     if existing:assert generated==existing['generated_ids'],'Fixed prefix replay changed token IDs'
     steps=capture.finish(generated)
     raw=proc.tokenizer.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
     answer=proc.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False) if m=='instruct' else (raw.split('</think>',1)[1].replace('<|im_end|>','').replace('<|endoftext|>','').strip() if '</think>' in raw else '')
     spans=extract_time(answer) or [];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
     write_result_text((relocate_path(d/'output_raw.txt')), raw);write_result_text((relocate_path(d/'answer.txt')), answer)
     result=dict(model=m,sample=sample,order=o,fps=8,answer=answer,generated_ids=generated,generated_tokens=len(generated),input_sha256=h,input_tokens=n,predicted_spans=spans,iou=iou(sample['span'][0],spans[0]) if spans else 0.,eos_reached=generated[-1] in eos,generation_seconds=time.time()-t,completed_at=stamp())
     if existing:result=existing
     else:write(relocate_path(d/'result.json'),result)
     write(tracepath(m,sample,o),dict(schema_version=1,model=m,sample=sample,order=o,fps=8,input_sha256=h,generated_ids=generated,steps=steps,answer=result['answer'],iou=result['iou'],measurement_method='fixed_saved_prefix_replay' if existing else 'live_generation',normalization='FP32 softmax over entire vocabulary including padded output rows; top30 NOT renormalized',processed='actual generation processors including temperature/top-k/top-p; before optional replay forcing',response_source=str(relocate_path(d/'result.json')),response_sha256=sha(relocate_path(d/'result.json'))))
     with lock:
      state['completed']+=1;key=m+'/'+sample['dataset'];state['counts'][key]=state['counts'].get(key,0)+1
     emit();del inputs,output
    del pair
   capture.remove();del capture,model,proc,loader;gc.collect();torch.cuda.empty_cache()
  update(stage='aggregation_and_validation');emit();aggregate(rows);trace_report(ROOT,OUT)
  update(stage='complete',validation_passed=True,report=str(relocate_path(OUT/'REPORT.md')));emit()
 except BaseException:
  update(stage='failed',error=traceback.format_exc());emit();raise
 finally:stop.set();thread.join()

if __name__=='__main__':main()
