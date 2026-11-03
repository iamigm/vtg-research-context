"""Construct endpoint traces for full-cohort digit analysis without the old IoU oracle."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import json,re,time,hashlib,os
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404')
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);write_result_text(path, json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def span_iou(gt,pred):
    inter=max(0,min(gt[1],pred[1])-max(gt[0],pred[0]));union=max(gt[1],pred[1])-min(gt[0],pred[0]);return inter/union if union>0 else 0.
def extract(raw):
    sample=raw['sample'];answer=raw['answer'];match=re.search(r'(?<![\d:])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d:])',answer)
    if ':' in answer or not match or any(not re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9])?',v) for v in match.groups()):return None
    text=''.join(t['token'] for t in raw['steps']);assert text.startswith(answer)
    predicted=[round(float(v)) for v in match.groups()];assert predicted==raw['timestamps'][0];gt=sample['span'][0];trajectories=[]
    for n,endpoint in [(1,'start'),(2,'end')]:
        a,b=match.span(n);offset=0;states=[]
        for t in raw['steps']:
            limit=offset+len(t['token'])
            if limit>a and offset<=b:
                prefix=text[a:min(limit,b)];closed=limit>b;intro=text[max(a,offset):min(b,limit)]
                states.append(dict(model=raw['model_key'],dataset=sample['dataset'],sample_key=sample['sample_key'],endpoint=endpoint,endpoint_step=len(states)+1,ar_step=t['step'],token=t['token'],numeric=bool(re.search(r'\d',intro)),closed=closed,prefix=prefix,entropy=t['raw']['entropy_nats'],processed_entropy=t['processed']['entropy_nats'],joint_iou=raw['iou']))
            offset=limit
        if not states or not states[-1]['closed']:return None
        ep_pred=[predicted[0],gt[1]] if n==1 else [gt[0],predicted[1]]
        trajectories.append(dict(model=raw['model_key'],dataset=sample['dataset'],sample_key=sample['sample_key'],endpoint=endpoint,gt=gt,answer=answer,predicted_timestamps=predicted,joint_iou=raw['iou'],endpoint_iou=span_iou(gt,ep_pred),steps=states))
    return trajectories

def main():
    rows=json.loads((relocate_path(BASE/'samples.json')).read_text());traces=[];excluded=[];audit=[]
    for model in ['qwen25','qwen3']:
        for i,sample in enumerate(rows):
            raw=json.loads((relocate_path(BASE/model/(sample['sample_key']+'.json'))).read_text());result=extract(raw)
            if result:traces.extend(result)
            else:excluded.append(dict(model=model,dataset=sample['dataset'],sample_key=sample['sample_key'],answer=raw['answer'],reason='outside canonical nonnegative seconds with at most one decimal'))
            if raw.get('baseline_tokens_identical') is False:audit.append(dict(model=model,sample_key=sample['sample_key'],reason='greedy tokens differ from saved zero-shot; actual new trace retained'))
            if (i+1)%250==0:print(json.dumps(dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage='endpoint traces',model=model,completed=i+1,total=len(rows),percent=100*(i+1)/len(rows))),flush=True)
    write(relocate_path(BASE/'endpoint_gt_entropy/trajectories.json'),traces);write(relocate_path(BASE/'endpoint_gt_entropy/summary.json'),dict(excluded=excluded));write(relocate_path(BASE/'baseline_audit.json'),dict(mismatches=audit,count=len(audit)))
if __name__=='__main__':main()
