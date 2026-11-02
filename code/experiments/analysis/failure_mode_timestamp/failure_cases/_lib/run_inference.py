"""Paired timestamp-conversion experiment, unchanged Qwen weights and query cohort."""
from result_layout import relocate_path, write_result_text
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT.parents[1]/'failure_mode_baseline/failure_cases')
os.environ.setdefault('QWEN_VL_VIDEO_READER','decord');os.environ.setdefault('OMP_NUM_THREADS','4')
sys.path.insert(0,str(relocate_path(ROOT/'vendor/TimeLens')))
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor,set_seed
from timelens.utils import extract_time
from timestamp_inputs import TimestampDataset,OFFICIAL_SOURCE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['qwen25']);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--memory-gb',type=float,default=28);a=ap.parse_args()
    torch.set_num_threads(4);set_seed(42)
    torch.cuda.set_per_process_memory_fraction(a.memory_gb*1024**3/torch.cuda.get_device_properties(0).total_memory)
    name='Qwen2.5-VL-7B-Instruct';mp='/home/irteam/backbones/models/'+name
    manifest=relocate_path(ROOT/'data/samples_100.json');rows=json.loads(manifest.read_text());baseline_manifest=relocate_path(BASE/'data/samples_100.json')
    assert manifest.read_bytes()==baseline_manifest.read_bytes(),'Cohort changed'
    cfg={'model':name,'model_path':mp,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'baseline_root':str(BASE),'fps_decode':2,'effective_frame_sampling':'even-index frames from 2fps samples; each duplicated (~1fps unique images)','min_tokens':64,'total_tokens':14336,'max_new_tokens':512,'do_sample':False,'seed':42,'dtype':'bfloat16','attention':'sdpa','timestamp_format':'{seconds:.1f}s: before each image','additional_prompt':'TimeLens timestamp interpretation prefix','no_finetuning':True,'official_processor_sha256':hashlib.sha256(OFFICIAL_SOURCE.read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256((relocate_path(ROOT/'scripts/timestamp_inputs.py')).read_bytes()).hexdigest(),'qwen3_extension':False}
    out=relocate_path(ROOT/'results/pilot_100'/a.model);out.mkdir(parents=True,exist_ok=True);cp=relocate_path(out/'config.json')
    if cp.exists():assert json.loads(cp.read_text())==cfg,'Incompatible resume'
    else:write_result_text(cp, json.dumps(cfg,indent=2))
    pp=relocate_path(out/'predictions.jsonl');done=set()
    if pp.exists():done={json.loads(s)['sample_id'] for s in pp.read_text().splitlines()}
    todo=[r for r in rows if r['sample_id'] not in done]
    if not todo:print('Already complete',flush=True);return
    print('Loading',name,'pending',len(todo),flush=True)
    model=AutoModelForImageTextToText.from_pretrained(mp,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True).eval()
    proc=AutoProcessor.from_pretrained(mp,padding_side='left',do_resize=False,local_files_only=True)
    ds=TimestampDataset(todo,proc,a.model)
    loader=torch.utils.data.DataLoader(ds,batch_size=1,num_workers=a.workers,collate_fn=lambda b:b[0],prefetch_factor=2 if a.workers else None)
    for i,item in enumerate(loader):
        row=item['anno'];start=time.time()
        try:
            with torch.inference_mode():
                inputs=item['inputs'].to('cuda');n=inputs.input_ids.shape[1]
                output=model.generate(**inputs,do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512)
                answer=proc.batch_decode(output[:,n:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                parsed=extract_time(answer);rounded=[[round(s),round(e)] for s,e in parsed]
                r={**row,'model':name,'prompt':item['prompt'],'answer':answer,'parsed_timestamps':parsed,'timestamps':rounded,'parse_failed':not bool(parsed),'input_tokens':int(n),'output_tokens':int(output.shape[1]-n),'input_audit':item['input_audit'],'elapsed_seconds':time.time()-start,'timing_scope':'generation with CPU prefetch','peak_allocated_gb':torch.cuda.max_memory_allocated()/1024**3}
            with pp.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            print(f'{i+1}/{len(todo)} {row["sample_id"]} {time.time()-start:.1f}s {answer}',flush=True)
            del inputs,output,item;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        except Exception:
            with (relocate_path(out/'errors.jsonl')).open('a') as f:f.write(json.dumps({'sample_id':row['sample_id'],'error':traceback.format_exc()})+'\n')
            raise
    write_result_text((relocate_path(out/'complete.json')), json.dumps({'count':len(rows),'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}))
if __name__=='__main__':main()
