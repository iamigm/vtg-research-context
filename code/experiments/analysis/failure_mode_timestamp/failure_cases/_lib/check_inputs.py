from result_layout import relocate_path, write_result_text
import json,os,sys
from pathlib import Path
os.environ['QWEN_VL_VIDEO_READER']='decord'
import torch
from transformers import AutoProcessor
from timestamp_inputs import TimestampDataset
ROOT=relocate_path(Path(__file__)).resolve().parents[1];torch.set_num_threads(3)
rows=json.loads((relocate_path(ROOT/'data/samples_100.json')).read_text());rows=[rows[0],rows[100]]
report={}
for key,name in [('qwen25','Qwen2.5-VL-7B-Instruct')]:
    proc=AutoProcessor.from_pretrained('/home/irteam/backbones/models/'+name,padding_side='left',do_resize=False,local_files_only=True)
    ds=TimestampDataset(rows,proc,key);report[key]=[]
    for i in range(len(ds)):
        x=ds[i];a=x['input_audit'];report[key].append({'sample_id':x['anno']['sample_id'],**a})
        print(key,x['anno']['dataset'],x['inputs'].pixel_values.shape,x['inputs'].image_grid_thw.shape,a['input_tokens'],a['displayed_timestamps'][:3],flush=True)
write_result_text((relocate_path(ROOT/'configs/input_smoke_validation.json')), json.dumps(report,indent=2))
