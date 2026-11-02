"""CPU-prefetched, cross-video batches using the unchanged backbone processors."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from prepare import ROOT,OUT,video_key

class PreparedBatches(Dataset):
    def __init__(self,model,paths,batch_size):
        self.model=model;self.processor=None;self.formatter=None
        self.pending=[];self.metadata={};batch=[]
        for path in paths:
            key=video_key(path)
            if (OUT/model/'video_embeddings'/f'{key}.npz').exists():continue
            meta=json.loads((OUT/'frames'/key/'metadata.json').read_text())
            self.metadata[key]=meta
            for i in range(meta['count']):
                batch.append((key,i))
                if len(batch)==batch_size:self.pending.append(batch);batch=[]
        if batch:self.pending.append(batch)
    def __len__(self):return len(self.pending)
    def __getitem__(self,index):
        torch.set_num_threads(1)
        if self.processor is None:
            if self.model=='clip':
                from transformers import CLIPProcessor
                self.processor=CLIPProcessor.from_pretrained('/home/irteam/backbones/models/clip-vit-large-patch14',local_files_only=True)
            else:
                sys.path.insert(0,str(ROOT/'_lib/vendor/Qwen3-VL-Embedding'))
                from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
                from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
                self.processor=Qwen3VLProcessor.from_pretrained('/home/irteam/backbones/models/Qwen3-VL-Embedding-8B',padding_side='right',local_files_only=True)
                # Only formatting state is needed; workers never construct weights.
                self.formatter=object.__new__(Qwen3VLEmbedder)
                self.formatter.min_pixels=4096;self.formatter.max_pixels=262144
                self.formatter.default_instruction="Represent the user's input."
        refs=self.pending[index];images=[]
        for key,i in refs:
            with Image.open(OUT/'frames'/key/f'{i:05d}.jpg') as im:images.append(im.convert('RGB'))
        if self.model=='clip':
            inputs=self.processor(images=images,return_tensors='pt')
        else:
            from qwen_vl_utils import process_vision_info
            conversations=[self.formatter.format_model_input(image=im) for im in images]
            text=self.processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=False)
            vision,videos=process_vision_info(conversations,image_patch_size=16)
            assert videos is None and len(vision)==len(images)
            inputs=self.processor(text=text,images=vision,return_tensors='pt',padding=True,truncation=False,do_resize=False)
            assert inputs['input_ids'].shape[1]<=8192
            assert inputs['image_grid_thw'].shape[0]==len(images)
        for im in images:im.close()
        return dict(inputs=dict(inputs),refs=refs)

def prefetched(model,paths,batch_size,workers=4):
    dataset=PreparedBatches(model,paths,batch_size)
    loader=DataLoader(dataset,batch_size=None,num_workers=workers,
        multiprocessing_context='spawn' if workers else None,
        prefetch_factor=2 if workers else None,pin_memory=True)
    return dataset,loader
