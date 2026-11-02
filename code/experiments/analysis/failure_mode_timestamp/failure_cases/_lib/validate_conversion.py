"""Check image conversion against each backbone's own independent-image processor."""
from result_layout import relocate_path, write_result_text
import json,os
from pathlib import Path
os.environ['QWEN_VL_VIDEO_READER']='decord'
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
ROOT=relocate_path(Path(__file__)).resolve().parents[1];torch.set_num_threads(3)
row=json.loads((relocate_path(ROOT/'data/samples_100.json')).read_text())[0];results={}
for key,name,factor in [('qwen25','Qwen2.5-VL-7B-Instruct',28)]:
    p=AutoProcessor.from_pretrained('/home/irteam/backbones/models/'+name,do_resize=False,local_files_only=True)
    msg=[{'role':'user','content':[{'type':'video','video':row['video_path'],'fps':2,'min_pixels':64*factor**2,'total_pixels':14336*factor**2}]}]
    _,vs=process_vision_info(msg,image_patch_size=factor//2,return_video_metadata=True)
    v=vs[0][0][:6].clone();selected=[v[i].clone() for i in [0,2,4]];v[1::2]=v[::2]
    dup=p.video_processor(videos=[v],do_sample_frames=False,do_resize=False,return_tensors='pt')
    direct=p.image_processor(images=selected,do_resize=False,return_tensors='pt')
    error=float((dup['pixel_values_videos']-direct['pixel_values']).abs().max())
    assert error==0,(key,error)
    h,w=dup['video_grid_thw'][0,1:].tolist();expected=torch.tensor([[1,h,w]]*3)
    assert torch.equal(direct['image_grid_thw'],expected)
    results[key]={'three_selected_frames':'[0,2,4] from the 2fps video samples','pixel_values_bitwise_equal_to_native_multi_image_processor':True,'image_grid_equal':True,'max_absolute_pixel_difference':error}
write_result_text((relocate_path(ROOT/'configs/conversion_validation.json')), json.dumps(results,indent=2));print(json.dumps(results,indent=2))
