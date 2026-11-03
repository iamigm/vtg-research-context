"""Pilot300: unique forced-token branches, greedy completion, accuracy and cost."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,collections,datetime,fcntl,hashlib,heapq,json,math,re,subprocess,sys,threading,time,traceback
from pathlib import Path
from conditional_rollout_budget import Evaluator,validate_cache

ROOT=Path(__file__).resolve().parents[1]
RUN='deterministic_rollout_pilot300_v2'
OUT=ROOT/'artifacts/runs'/RUN
REPORT=ROOT/'results'/RUN
MODEL=Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct')
COHORT=ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_300/samples.json'
BUDGETS=[1,4,8,16,32]
CONFIG=dict(run=RUN,cohort=str(COHORT),queries=300,model=str(MODEL),fps=2,
 target_pixels=100352,order='video_query',prompt='unchanged official TimeLens GROUNDER_PROMPT',
 candidates=32,budgets=BUDGETS,max_new_tokens=512,
 decoding='deterministic forced alternative token then unconstrained greedy to EOS',
 branch_sites='numeric characters, decimal points, colons of first start/end expressions; retain digit roles',
 alternatives_per_site=4,minimum_raw_probability=.0001,top_mass_target=.95,
 priority='raw full-prefix probability upper bound; observed completions are not certified global top-k',
 conditioning='full exact generated prefix; start alternatives change conditional end distribution',
 time_formats='seconds, MM:SS, HH:MM:SS; parsing only, no output grammar or prompt intervention',
 selectors=['max_probability','medoid','probability_weighted_medoid','oracle'],
 primary_evaluation='continuous seconds, positive interval, no rounding or clipping; out-of-duration flagged separately',
 generation_uses_gt=False,seed='none; deterministic inference; bootstrap42 only')
NUMBER=re.compile(r'\d+(?::\d{1,2}){0,2}(?:\.\d+)?')

def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def time_matches(text):
 # Prefer an explicit interval pair; fallback supports otherwise exceptional prose.
 pair=re.search(r'('+NUMBER.pattern+r')\s*(?:-|–|—|\bto\b)\s*('+NUMBER.pattern+r')',text)
 if pair:return [dict(text=pair.group(i),start=pair.start(i),end=pair.end(i)) for i in [1,2]]
 return [dict(text=m.group(),start=m.start(),end=m.end()) for m in list(NUMBER.finditer(text))[:2]]

def seconds(text):
 parts=text.split(':')
 if len(parts)>1 and any(not 0<=float(x)<60 for x in parts[1:]):raise ValueError('Invalid clock fields')
 n=0.
 for x in parts:n=n*60+float(x)
 return n

def parse(text):
 matches=time_matches(text)
 if len(matches)!=2:return None
 try:s,e=[seconds(m['text']) for m in matches]
 except ValueError:return None
 return dict(start=s,end=e,positive=0<=s<e,
             format='clock' if any(':' in m['text'] for m in matches) else 'seconds')

def sites(ids,tokenizer):
 pieces=[tokenizer.decode([x],clean_up_tokenization_spaces=False) for x in ids]
 text=''.join(pieces);matches=time_matches(text);positions={}
 for endpoint,m in zip(['start','end'],matches):
  value=m['text'];parts=value.split(':');cursor=m['start']
  units=['hours','minutes','seconds'][-len(parts):]
  for part,unit in zip(parts,units):
   integer=part.split('.')[0]
   for j,char in enumerate(part):
    if char=='.':role=f'{endpoint}.{unit}.decimal_separator'
    elif j<len(integer):role=f'{endpoint}.{unit}.10^{len(integer)-j-1}'
    else:role=f'{endpoint}.{unit}.10^-{j-len(integer)}'
    positions[cursor+j]=role
   cursor+=len(part)
   if cursor<m['end']:positions[cursor]=f'{endpoint}.clock_separator';cursor+=1
 result=[];offset=0
 for j,piece in enumerate(pieces):
  roles=sorted({positions[k] for k in range(offset,offset+len(piece)) if k in positions})
  if roles:result.append((j,roles))
  offset+=len(piece)
 return result

class Progress:
 def __init__(self):
  self.t=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event()
  self.s=dict(state='running',stage='initializing',completed=0,total=300,counts={},eta_seconds=None,
    eta_scope='remaining inference; unknown before first query; excludes reporting',model='qwen3-instruct')
  threading.Thread(target=self.heartbeat,daemon=True).start()
 def update(self,**kw):
  with self.lock:
   self.s.update(kw);self.s.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.t,
      percentage=100*self.s['completed']/300)
   write(OUT/'metadata/status.json',self.s)
   line=json.dumps(self.s,ensure_ascii=False)
   with (OUT/'logs/progress.log').open('a') as f:f.write(line+'\n')
   print(line,flush=True)
 def heartbeat(self):
  while not self.stop.wait(20):self.update()

def setup():
 rows=read(COHORT);assert len(rows)==300
 assert collections.Counter(x['dataset'] for x in rows)==dict.fromkeys(['charades-timelens','activitynet-timelens','qvhighlights-timelens'],100)
 for d in ['metadata','logs','code','inputs','rollouts','metrics','validation']:(OUT/d).mkdir(parents=True,exist_ok=True)
 REPORT.mkdir(parents=True,exist_ok=True)
 for name,value in [('config.json',CONFIG),('cohort.json',rows)]:
  p=OUT/'metadata'/name
  if p.exists():assert read(p)==value
  else:write(p,value)
 manifest=[]
 for name in ['deterministic_rollout300.py','deterministic_rollout300_report.py','conditional_rollout_budget.py','fps_spatial_control.py']:
  p=Path(__file__).with_name(name);(OUT/'code'/name).write_bytes(p.read_bytes());manifest.append(dict(path=str(p),sha256=sha(p)))
 write(OUT/'metadata/provenance.json',dict(files=manifest,cohort_sha256=sha(COHORT),
  model_manifest_sha256=sha('/home/irteam/backbones/manifests/registry.json'),
  cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip()))
 return rows

def complete(ev,tokenizer,eos,prefix,prefix_logp):
 ids=list(prefix);logp=prefix_logp
 while len(ids)<CONFIG['max_new_tokens'] and (not ids or ids[-1] not in eos):
  lp=ev.torch.log_softmax(ev.get(ids),-1)
  token=int(lp.argmax());logp+=float(lp[token]);ids.append(token)
 answer=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 return dict(ids=ids,answer=answer,log_probability=logp,parsed=parse(answer),
             stopped_on_eos=bool(ids and ids[-1] in eos))

def propose(ev,tokenizer,candidate,frozen,allowed,queue,seen,parent):
 prefixlogp=0.;selected=[]
 site_map=dict(sites(candidate['ids'],tokenizer))
 for j,token in enumerate(candidate['ids']):
  lp=ev.torch.log_softmax(ev.get(candidate['ids'][:j]),-1)
  if j>=frozen and j in site_map:
   ranked=sorted(allowed,key=lambda i:(-float(lp[i]),i));kept=0.;count=0;alternatives=[]
   for alt in ranked:
    p=math.exp(float(lp[alt]))
    if p<CONFIG['minimum_raw_probability']:break
    if count>=CONFIG['alternatives_per_site'] or kept>=CONFIG['top_mass_target']:break
    kept+=p;count+=1
    if alt==token:continue
    path=tuple(candidate['ids'][:j]+[alt])
    if path in seen:continue
    seen.add(path);score=prefixlogp+float(lp[alt])
    info=dict(position=j,roles=site_map[j],original_token=token,forced_token=alt,
              raw_probability=p,parent=parent,forced_prefix_ids=list(path))
    heapq.heappush(queue,(-score,path,info));alternatives.append(info)
   selected.append(dict(position=j,roles=site_map[j],retained_numeric_token_probability=kept,
                        omitted_raw_probability=max(0,1-kept),alternatives=alternatives))
  prefixlogp+=float(lp[token])
 return selected

def infer(ev,tokenizer,eos,progress):
 allowed=[i for i in range(len(tokenizer)) if i not in tokenizer.all_special_ids and
          re.fullmatch(r'[0-9.:]+',tokenizer.decode([i],clean_up_tokenization_spaces=False))]
 queue=[];seen=set();completed=set();records=[];start=time.monotonic()
 baseline=complete(ev,tokenizer,eos,[],0.)
 first_site=sites(baseline['ids'],tokenizer)
 anchor=first_site[0][0] if first_site else len(baseline['ids'])
 current=baseline;frozen=anchor;branch=None
 for index in range(32):
  if index:
   if not queue:break
   score,prefix,branch=heapq.heappop(queue);frozen=len(prefix)
   current=complete(ev,tokenizer,eos,prefix,-score)
   branch['target_roles']=dict(sites(current['ids'],tokenizer)).get(branch['position'],[])
  identity=tuple(current['ids']);assert identity not in completed,'Duplicate completion'
  completed.add(identity)
  current.update(index=index+1,branch=branch,decoder_tokens=ev.text_tokens,
                 forward_calls=ev.calls,elapsed_seconds=time.monotonic()-start)
  # Only extend suffixes after the last forced decision. Tasks partition paths
  # by their first alternative decision, preventing duplicate completions.
  current['branch_sites']=propose(ev,tokenizer,current,frozen,allowed,queue,seen,index+1) if index<31 else []
  current.update(decoder_tokens=ev.text_tokens,forward_calls=ev.calls,
                 elapsed_seconds=time.monotonic()-start)
  records.append(current)
  if index+1 in BUDGETS:progress.update(stage='rollout',candidates=index+1,
        decoder_tokens=ev.text_tokens,latest_answer=current['answer'][:160])
 return dict(candidates=records,queue_remaining=len(queue),exhausted=not queue,
             decoder_tokens=ev.text_tokens,prefill_seconds=ev.prefill_seconds)

def run(rows,p):
 while True:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
  if free>=60*1024:break
  p.update(state='waiting',stage='waiting_for_gpu',free_mib=free);time.sleep(20)
 import fps_spatial_control as source
 import torch
 from transformers import AutoProcessor,AutoModelForImageTextToText
 torch.set_num_threads(2);p.update(state='running',stage='loading_model')
 proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
 model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,
    dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
 eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
 counts=collections.Counter();done=0;fresh=0;t=time.monotonic();checked=set()
 for row in rows:
  dest=OUT/'rollouts'/(row['sample_key']+'.json')
  if dest.exists():
   old=read(dest);assert old['sample']==row;done+=1;counts[row['dataset']]+=1;continue
  p.update(stage='preparing_input',sample_key=row['sample_key'],candidates=0)
  key=hashlib.sha256(row['video_path'].encode()).hexdigest()
  cache=read(source.OUT/'media'/key/'metadata.json')
  public={k:row[k] for k in ['sample_key','dataset','video_path','query','duration']}
  cpu,detail=source.prepare_input(public,cache,2,'video_query',proc)
  write(OUT/'inputs'/(row['sample_key']+'.json'),detail)
  inputs=cpu.to('cuda');ev=Evaluator(model,inputs,torch)
  if row['dataset'] not in checked:
   probe=complete(ev,proc.tokenizer,eos,[],0.)
   ss=sites(probe['ids'],proc.tokenizer);anchor=probe['ids'][:ss[0][0]] if ss else []
   audits=validate_cache(ev,inputs,anchor,proc.tokenizer,torch)
   write(OUT/'validation'/(row['dataset']+'_cache.json'),dict(passed=True,audits=audits))
   checked.add(row['dataset']);del ev;torch.cuda.empty_cache();ev=Evaluator(model,inputs,torch)
  result=infer(ev,proc.tokenizer,eos,p)
  oldpath=ROOT/'artifacts/runs/conditional_rollout_budget_pilot30_v1/trees'/(row['sample_key']+'.json')
  if oldpath.exists():
   old=read(oldpath);assert old['input_hashes']==detail['hashes']
   assert old['greedy_ids']==result['candidates'][0]['ids'],'Greedy baseline changed'
  write(dest,dict(sample=row,input_hashes=detail['hashes'],result=result))
  done+=1;fresh+=1;counts[row['dataset']]+=1
  p.update(stage='generation',completed=done,counts=dict(counts),
           eta_seconds=(time.monotonic()-t)/fresh*(300-done))
  del ev,inputs,cpu;torch.cuda.empty_cache()
 p.update(stage='reporting')
 from deterministic_rollout300_report import report
 report()
 p.update(state='complete',stage='complete',completed=300,eta_seconds=0)

def selftest():
 assert seconds('2:11')==131 and seconds('1:02:03.5')==3723.5
 assert parse('The event happens in 2:11 - 2:14 seconds.')['end']==134
 assert parse('The event happens in 120.0 - 135.0 seconds.')['start']==120
 assert parse('The event happens in 2:71 - 2:74 seconds.') is None
 class T:
  def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
 text='The event happens in 120.0 - 135.0 seconds.'
 roles=dict(sites([ord(c) for c in text],T()))
 assert 'start.seconds.10^2' in roles[text.index('120')]
 clock='The event happens in 2:11 - 2:14 seconds.'
 roles=dict(sites([ord(c) for c in clock],T()))
 assert 'start.minutes.10^0' in roles[clock.index('2:11')]
 assert 'start.seconds.10^1' in roles[clock.index('2:11')+2]
 return dict(passed=True,clock_and_seconds=True,digit_roles=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--report',action='store_true');a=ap.parse_args()
 if a.selftest:print(json.dumps(selftest()));return
 if a.report:
  from deterministic_rollout300_report import report
  report();return
 rows=setup()
 with (OUT/'metadata/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=Progress()
  try:write(OUT/'validation/cpu.json',selftest());run(rows,p)
  except BaseException:p.update(state='failed',stage='failed',error=traceback.format_exc());raise
  finally:p.stop.set()

if __name__=='__main__':main()
