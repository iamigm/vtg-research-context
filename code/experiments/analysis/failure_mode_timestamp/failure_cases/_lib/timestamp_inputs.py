"""TimeLens-7B input conversion on unchanged Qwen weights; explicit Qwen3 extension."""
from result_layout import relocate_path, write_result_text
import copy,hashlib,importlib.util,json,re,sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers.feature_extraction_utils import BatchFeature
from qwen_vl_utils import process_vision_info
ROOT=relocate_path(Path(__file__)).resolve().parents[1]
sys.path.insert(0,str(relocate_path(ROOT/'vendor/TimeLens')))
from evaluation.utils import GROUNDER_PROMPT_TEXT_TIMESTAMP
OFFICIAL_SOURCE=relocate_path(Path('/home/irteam/backbones/processors/TimeLens-7B/processing_timelens.py'))
spec=importlib.util.spec_from_file_location('timelens_pinned_processor',OFFICIAL_SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TimestampDataset(Dataset):
    def __init__(self,rows,base_processor,model_key):
        self.rows=rows;self.base=base_processor;self.model_key=model_key
        self.processor=module.TimeLensProcessor(image_processor=base_processor.image_processor,tokenizer=base_processor.tokenizer,video_processor=base_processor.video_processor,chat_template=base_processor.chat_template) if model_key=='qwen25' else base_processor
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];factor=28 if self.model_key=='qwen25' else 32
        prompt=GROUNDER_PROMPT_TEXT_TIMESTAMP.format(r['query'])
        messages=[{'role':'user','content':[{'type':'video','video':r['video_path'],'min_pixels':64*factor**2,'total_pixels':14336*factor**2,'fps':2},{'type':'text','text':prompt}]}]
        text=self.base.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        _,videos=process_vision_info(messages,image_patch_size=factor//2,return_video_metadata=True)
        tensor,metadata=videos[0];assert tensor.shape[0]%2==0
        frame_indices=list(metadata['frames_indices']);selected=frame_indices[::2]
        timestamps=[idx/metadata['fps'] for idx in selected]
        if self.model_key=='qwen25':
            inputs=self.processor(text=[text],videos=videos,padding=True,return_tensors='pt')
        else:
            # Same frame duplication / image-token conversion as TimeLensProcessor,
            # using Qwen3's native pixel normalization and 16-pixel patch processor.
            tensor[1::2]=tensor[::2]
            vp=self.base.video_processor(videos=[tensor],do_sample_frames=False,do_resize=False,return_tensors='pt')
            grid=vp['video_grid_thw'];assert int(grid[0,0])==len(timestamps)
            n=int(grid[0,1]*grid[0,2])//self.base.video_processor.merge_size**2
            visual=''.join(f'{t:.1f}s: '+self.base.vision_start_token+self.base.image_token*n+self.base.vision_end_token for t in timestamps)
            text=text.replace(self.base.vision_start_token+self.base.video_token+self.base.vision_end_token,visual,1)
            tokenized=self.base.tokenizer([text],padding=True,return_tensors='pt')
            image_grid=torch.tensor([[1,int(grid[0,1]),int(grid[0,2])] for _ in timestamps],dtype=torch.long)
            inputs=BatchFeature(data={**tokenized,'pixel_values':vp['pixel_values_videos'],'image_grid_thw':image_grid})
        decoded=self.base.tokenizer.decode(inputs.input_ids[0],skip_special_tokens=False)
        image_grid=inputs.image_grid_thw
        assert image_grid.shape[0]==len(timestamps)
        assert torch.all(image_grid[:,0]==1)
        n_images=int((inputs.input_ids==self.base.image_token_id).sum())
        expected=int(image_grid.prod(-1).sum())//self.base.image_processor.merge_size**2
        assert n_images==expected,(n_images,expected)
        assert int((inputs.input_ids==self.base.video_token_id).sum())==0
        assert 'pixel_values_videos' not in inputs and 'second_per_grid_ts' not in inputs
        # Pin each displayed timestamp to the actual decoded frame index.
        observed=re.findall(r'(\d+\.\d)s: <\|vision_start\|>',decoded)
        assert observed==[f'{t:.1f}' for t in timestamps],(observed,timestamps)
        compact=re.sub(r'(?:<\|image_pad\|>)+',lambda m:f'[image patches × {m.group().count("<|image_pad|>")}]',decoded)
        meta={'decoded_fps':float(metadata['fps']),'sampled_frame_indices_2fps':frame_indices,'selected_frame_indices':selected,'timestamps_seconds':timestamps,'displayed_timestamps':[f'{t:.1f}s:' for t in timestamps],'sampled_frames':len(frame_indices),'unique_selected_frames':len(set(selected)),'image_groups':len(timestamps),'visual_tokens':n_images,'input_tokens':int(inputs.input_ids.shape[1]),'image_grid_thw':image_grid.tolist(),'input_ids_sha256':hashlib.sha256(inputs.input_ids.numpy().tobytes()).hexdigest(),'rendered_prompt_compact':compact,'validation':'image-only modality, t=1 groups, visual token count, timestamp/frame alignment checked','transformation':'TimeLens-7B official processor' if self.model_key=='qwen25' else 'TimeLens-7B-style transformation adapted to Qwen3; not official TimeLens-8B preprocessing'}
        return {'inputs':inputs,'anno':r,'prompt':prompt,'input_audit':meta}
