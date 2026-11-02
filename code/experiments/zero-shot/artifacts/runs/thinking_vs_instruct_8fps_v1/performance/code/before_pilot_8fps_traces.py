"""Unmodified raw/final generation distributions, with explicit fixed-prefix replay."""
from result_layout import relocate_path, write_result_text, result_glob
import json,hashlib,statistics,re,math,csv
from pathlib import Path
import torch
from transformers import LogitsProcessor

class TraceCapture(LogitsProcessor):
 def __init__(self,model,tokenizer,allowed,eos):
  self.model=model;self.tok=tokenizer;self.allowed=torch.tensor(allowed,device='cuda');self.eos=torch.tensor(eos,device='cuda');self.cache={};self.original=model._get_logits_processor
  def processors(*args,**kwargs):
   items=self.original(*args,**kwargs);items.append(self)
   if self.force is not None:items.append(Force(self))
   return items
  model._get_logits_processor=processors
  self.handle=model.register_forward_hook(self.hook);self.reset()
 def reset(self,force=None):self.steps=[];self.raw=None;self.pending=None;self.force=force
 def text(self,i):
  if i not in self.cache:self.cache[i]=self.tok.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
  return self.cache[i]
 def summarize(self,logits):
  lp=logits.float().log_softmax(-1);p=lp.exp();values,ids=lp.topk(30);probs=values.exp().tolist();ids=ids.tolist();mass=float(p.sum());assert abs(mass-1)<2e-5
  h=-(p*torch.where(torch.isfinite(lp),lp,0)).sum()
  return lp,dict(top30=[dict(id=i,token=self.text(i),probability=v) for i,v in zip(ids,probs)],top30_mass=sum(probs),entropy_nats=float(h),span_alphabet_mass=float(p[self.allowed].sum()),eos_mass=float(p[self.eos].sum()),probability_sum=mass)
 def hook(self,module,args,output):
  assert self.raw is None
  self.raw=output.logits[0,-1].detach()
 def finish_token(self,chosen):
  if self.pending is None:return
  raw,processed,rs,ps=self.pending
  self.steps.append(dict(step=len(self.steps)+1,token_id=chosen,token=self.text(chosen),raw_selected_probability=float(raw[chosen].exp()),raw_selected_rank=1+int((raw>raw[chosen]).sum()),processed_selected_probability=float(processed[chosen].exp()),processed_selected_rank=1+int((processed>processed[chosen]).sum()),raw=rs,processed=ps))
  self.pending=None
 def __call__(self,input_ids,scores):
  if self.pending is not None:self.finish_token(int(input_ids[0,-1]))
  assert self.raw is not None
  raw,rs=self.summarize(self.raw);processed,ps=self.summarize(scores[0]);self.pending=(raw,processed,rs,ps);self.raw=None
  return scores
 def finish(self,generated):
  self.finish_token(generated[-1]);assert generated==[s['token_id'] for s in self.steps]
  role='reasoning' if self.model_label=='thinking' else 'answer'
  for s in self.steps:
   tok=s['token'];s['role']='boundary' if s['token_id'] in self.eos.tolist() or tok in ['</think>','<think>'] else role
   if tok=='</think>':role='answer'
  return self.steps
 def remove(self):self.handle.remove();self.model._get_logits_processor=self.original

class Force(LogitsProcessor):
 def __init__(self,capture):self.capture=capture
 def __call__(self,input_ids,scores):
  token=self.capture.force[len(self.capture.steps)];forced=torch.full_like(scores,float('-inf'));forced[:,token]=0.;return forced

def report(root,out):
 """Validate complete traces and save comparable uncertainty/digit summaries."""
 import numpy as np
 from timelens_full_digit_impl import classify
 from timelens_endpoint_gt_entropy import best_iou
 from timelens_full_blue_impl import dist_info
 from timelens.utils import extract_time,iou
 from artifact_store import register
 from timelens_inputs import write,read,sha
 grouped={};per_response=[];per_digit=[];blue=[];endpoints=[];excluded=[];inventory=[];records=[]
 for p in sorted(result_glob((relocate_path(out/'token_traces')), '*/*/*/*/trace.json')):
  r=read(p);d=relocate_path(out/'samples'/r['model']/r['sample']['dataset']/r['sample']['sample_key']/r['order']);a=read(relocate_path(d/'result.json'));assert r['generated_ids']==a['generated_ids']
  assert r['input_sha256']==a['input_sha256'];assert len(r['steps'])==len(r['generated_ids'])
  for t in r['steps']:
   assert t['token_id']==r['generated_ids'][t['step']-1]
   for mode in ['raw','processed']:
    ss=t[mode];assert abs(ss['probability_sum']-1)<2e-5 and 0<=ss['top30_mass']<=1+2e-5 and math.isfinite(ss['entropy_nats'])
    assert len(ss['top30'])==30 and all(ss['top30'][i]['probability']>=ss['top30'][i+1]['probability'] for i in range(29))
  sample=r['sample'];base=dict(model=r['model'],order=r['order'],dataset=sample['dataset'],sample_key=sample['sample_key']);gt=sample['span'][0]
  parsed=extract_time(a['answer']) or [];raw_iou=iou(gt,parsed[0]) if parsed else 0.;rounded_iou=iou(gt,[round(v) for v in parsed[0]]) if parsed else 0.
  q=dict(base,iou_raw=raw_iou,iou_rounded=rounded_iou,generated_tokens=len(r['steps']),trace_method=r['measurement_method'])
  for role in ['all','answer','reasoning']:
   ss=[t for t in r['steps'] if role=='all' or t['role']==role];q[role+'_tokens']=len(ss)
   for mode in ['raw','processed']:
    for field in ['entropy_nats','top30_mass','span_alphabet_mass','eos_mass']:
     q[role+'_'+mode+'_'+field]=float(np.mean([t[mode][field] for t in ss])) if ss else None
  per_response.append(q);records.append((r,a));inventory.append(dict(path=str(p.relative_to(out)),sha256=sha(p),steps=len(r['steps'])))
  text=''.join(t['token'] for t in r['steps']);closed=text.find('</think>');answer_begin=closed+len('</think>') if r['model']=='thinking' and closed>=0 else (len(text) if r['model']=='thinking' else 0)
  answertext=text[answer_begin:];match=re.search(r'(?<![\d:])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d:])',answertext)
  if ':' in answertext or not match or any(len(v.partition('.')[2])>1 or not re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9])?',v) for v in match.groups()):
   excluded.append(dict(base,reason='outside canonical seconds grammar with at most one decimal',answer=a['answer']));continue
  cap=math.ceil(max(sample['duration'],gt[0]+(gt[1]-gt[0])/.3,*map(float,match.groups())))+2
  labels=[];vals=[]
  for k in range(cap*10+1):
   labels.append(f'{k//10}.{k%10}');vals.append(round(float(labels[-1])))
   if k%10==0:labels.append(str(k//10));vals.append(k//10)
  vals=np.asarray(vals)
  for n,ep in [(1,'start'),(2,'end')]:
   begin,end=match.span(n);begin+=answer_begin;end+=answer_begin;number=match.group(n);ss=[];offset=0;previous,_=best_iou(gt,np.unique(vals),ep)
   for t in r['steps']:
    limit=offset+len(t['token'])
    if limit>begin and offset<=end:
     prefix=text[begin:min(limit,end)];closed=limit>end
     vv=np.asarray([round(float(number))]) if closed else np.unique(vals[[label.startswith(prefix) for label in labels]])
     best,_=best_iou(gt,vv,ep)
     row=dict(base,endpoint=ep,gt=gt[n-1],ar_step=t['step'],token=t['token'],prefix=prefix,closed=closed,entropy=t['raw']['entropy_nats'],processed_entropy=t['processed']['entropy_nats'],endpoint_oracle_before=previous,endpoint_oracle_after=best,first_loss_iou50=previous>=.5 and best<.5)
     ss.append(row);previous=best
    offset=limit
   if not ss or not ss[-1]['closed']:
    excluded.append(dict(base,reason='missing endpoint closing token',answer=a['answer']));continue
   if float(gt[n-1]).is_integer():
    target,digits,correct=classify(number,gt[n-1],ss)
    per_digit.extend(digits)
    offset_before=''
    for digit in digits:
     if digit['numeric'] and digit['previous_correct']:
      t=r['steps'][digit['ar_step']-1];di,items=dist_info(t['raw'],offset_before,int(gt[n-1]),ep)
      blue.append(dict(base,endpoint=ep,ar_step=digit['ar_step'],role=digit['role'],condition=digit['condition'],entropy=digit['entropy'],**di))
     offset_before=digit['prefix']
   else:correct=round(float(number))==gt[n-1]
   endpoints.append(dict(base,endpoint=ep,prediction=number,gt=gt[n-1],rounded_correct=correct,fixed_gt_endpoint_iou=previous,first_loss_entropy=next((s['entropy'] for s in ss if s['first_loss_iou50']),None)))
 for q in per_response:
  for ds in [q['dataset'],'all']:grouped.setdefault((q['model'],q['order'],ds),[]).append(q)
 metrics=[]
 for (m,o,ds),qs in sorted(grouped.items()):
  g=dict(model=m,order=o,dataset=ds,n=len(qs))
  for k in qs[0]:
   if k in ['model','order','dataset','sample_key','trace_method']:continue
   vv=[q[k] for q in qs if q[k] is not None];g[k]=float(np.mean(vv)) if vv else None
  for threshold in [.3,.5,.7]:g[f'r{int(threshold*10):02d}']=float(np.mean([q['iou_raw']>=threshold for q in qs]))
  metrics.append(g)
 def csvfile(name,rs):
  if rs:
   with (relocate_path(out/name)).open('w') as f:w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rs))));w.writeheader();w.writerows(rs)
 csvfile('token_trace_per_response.csv',per_response);csvfile('token_trace_metrics.csv',metrics);csvfile('digit_per_step.csv',per_digit);csvfile('per_endpoint.csv',endpoints)
 cond=[];buckets={}
 for s in per_digit:buckets.setdefault((s['model'],s['order'],s['dataset'],s['endpoint'],s['condition'],s['role']),[]).append(s)
 for key,ss in sorted(buckets.items()):
  by={}
  for s in ss:by.setdefault(s['sample_key'],[]).append(s['entropy'])
  cond.append(dict(zip(['model','order','dataset','endpoint','condition','role'],key),n_queries=len(by),n_steps=len(ss),mean_entropy=float(np.mean([np.mean(v) for v in by.values()])),step_mean=float(np.mean([s['entropy'] for s in ss]))))
 csvfile('digit_conditions.csv',cond);csvfile('blue_candidate_mass.csv',blue);write(relocate_path(out/'trace_excluded.json'),excluded)
 write(relocate_path(out/'token_trace_summary.json'),dict(responses=len(per_response),steps=sum(q['generated_tokens'] for q in per_response),metrics=metrics,excluded=len(excluded),conditions=cond,normalization='full-vocabulary FP32 softmax; no top30 renormalization',averaging='within response then across responses; reasoning/answer separate',rollouts=False));write(relocate_path(out/'trace_index.json'),inventory)
 assert len(per_response)==1200
 write(relocate_path(out/'trace_validation.json'),dict(passed=True,responses=1200,generated_ids_match=True,input_hash_match=True,finite_entropies=True,probability_sum_valid=True,top30_sorted=True,selected_token_aligned=True,rollouts=False))
 lines=['# 8fps token trace·불확실성·endpoint 집계','', 'pilot 300개 × Instruct/Thinking × 두 입력 순서, 1,200개 응답을 검증했다. Rollout은 수행하지 않았다. 기존 zero-shot과 같은 raw/processed top30·전체 어휘 엔트로피·span alphabet mass·EOS mass·선택 토큰 확률/순위를 저장했다. Thinking은 reasoning과 최종 answer를 분리 집계했다.','', 'processed는 temperature/top-k/top-p를 포함한 실제 sampling 직전 분포다. greedy에는 원래 generation processor만 적용한다. 이미 완료된 응답은 동일 prefix를 강제로 재생해 raw/processed 분포를 측정했고, 강제 토큰 선택은 분포를 기록한 뒤 적용한다. replay와 live는 trace metadata로 구분한다.','', '| Model | Order | Dataset | N | mIoU % | Answer raw H | Reasoning raw H | Answer top30 mass |','|---|---|---|---:|---:|---:|---:|---:|']
 for g in metrics:
  fmt=lambda x:'NA' if x is None else f'{x:.4f}'
  lines.append(f"| {g['model']} | {g['order']} | {g['dataset']} | {g['n']} | {100*g['iou_raw']:.2f} | {fmt(g['answer_raw_entropy_nats'])} | {fmt(g['reasoning_raw_entropy_nats'])} | {fmt(g['answer_raw_top30_mass'])} |")
 lines+=['', 'blue_candidate_mass.csv는 기존과 동일한 top30 정답 가능/불가능/미확인 확률 질량과 하한·상한을 저장한다. top30을 재정규화하지 않는다. digit_per_step.csv와 digit_conditions.csv는 기존 rounded-prefix 첫 정답 가능성 상실 판정을 그대로 재사용했다. per_endpoint.csv는 반대쪽 endpoint를 GT로 고정한 IoU와 최초 IoU .5 가능성 상실 entropy를 제공한다. 지원 문법 밖의 응답은 trace_excluded.json에 명시했으며 원본 응답이나 성능 집계에서는 제외하지 않았다.','', '[응답별 trace 지표](token_trace_per_response.csv) · [집계 지표](token_trace_metrics.csv) · [token trace summary](token_trace_summary.json) · [검증](trace_validation.json) · [성능 비교](REPORT.md)']
 write_result_text((relocate_path(out/'TOKEN_TRACE_REPORT.md')), '\n'.join(lines)+'\n')
 for kind,path in [('token_traces',relocate_path(out/'token_traces')),('predictions',relocate_path(out/'samples')),('analyses',out)]:register('pilot8fps.v1.'+kind,kind,path,'8fps pilot '+kind+'; no rollouts','_lib/pilot_8fps.py','pilot_300','complete',['pilot_300.cohort'])
