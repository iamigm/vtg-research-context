"""Exact original inputs; no output files are written by this adapter."""
import os,re,sys
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from types import MethodType
from probability_budget_core import *
import fps_spatial_control as inputs_module
sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
from temporal_mode_core import parse_clip
sys.path.insert(0,str(Path(__file__).resolve().parent))
def source_frames(row):
 from result_layout import relocate_path
 from shared_selected_frames import load_selected
 prior=read(SOURCE/'inputs'/f'{row["sample_key"]}.json');cache=read(relocate_path(Path(prior['cache_metadata'])));plan=cache['plans']['2'];ids=plan['frame_ids'];adapter=dict(cache_key=digest(dict(source=cache['key'],hw=cache['hw'],kind='fps_selected')),path=cache['array'],ids=cache['ids'])
 frames=load_selected(adapter,ids);loc={i:j for j,i in enumerate(cache['ids'])}
 assert all(hashlib.sha256(frame.tobytes()).hexdigest()==cache['rgb_hashes'][loc[ident]] for frame,ident in zip(frames,ids))
 return prior,cache,frames

def processor_input(processor,frames,ids,pts,sourcefps,N,query=None,direct_text=None):
 import torch
 if query is not None or direct_text is not None:
  from evaluation.utils import GROUNDER_PROMPT
  messages=[dict(role='user',content=[dict(type='video',video='frames'),dict(type='text',text=direct_text if direct_text is not None else GROUNDER_PROMPT.format(query))])]
 else:messages=[dict(role='system',content=[dict(type='text',text=CONFIG['semantic_instruction'])]),dict(role='user',content=[dict(type='video',video='frames')])]
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 if query is None and direct_text is None:text+='<|endoftext|>'
 mapping=dict(zip(ids,pts))
 def timestamps(self,indices,video_fps,merge_size=2):
  indices=list(indices);indices+=indices[-1:]*((-len(indices))%merge_size)
  return [(mapping[indices[j]]+mapping[indices[j+merge_size-1]])/2 for j in range(0,len(indices),merge_size)]
 original=processor._calculate_timestamps;processor._calculate_timestamps=MethodType(timestamps,processor)
 try:return processor(text=[text],videos=[torch.from_numpy(np.asarray(frames)).float()],video_metadata=[dict(fps=sourcefps,frames_indices=list(ids),total_num_frames=N,video_backend='decord')],do_resize=False,do_sample_frames=False,padding=True,return_tensors='pt')
 finally:processor._calculate_timestamps=original

def numeric_role(text):
 for endpoint,anchor in [('start','The event happens in '),('end',' - ')]:
  if anchor not in text:continue
  tail=text.rsplit(anchor,1)[1]
  if re.fullmatch(r'[0-9]*',tail):return endpoint+'.integer_or_decimal'
  if re.fullmatch(r'[0-9]+\.[0-9]*',tail):return endpoint+'.fraction'
 return 'format'

def choose_edges(edges,tokenizer,text):
 role=numeric_role(text)
 if not edges:return [],role
 if not role.endswith('integer_or_decimal'):return edges[:1],role
 top=edges[0];selected=[];keys=set()
 for edge in edges:
  token_text=tokenizer.decode([edge['token']],clean_up_tokenization_spaces=False)
  if edge is not top and not re.fullmatch(r'[0-9.]+',token_text):continue
  # If an atomic token crosses the decimal point, fractional-only alternatives
  # share one key; keep the most likely native token rather than splitting it.
  projected=token_text.split('.',1)[0]+('.' if '.' in token_text else '')
  key=projected if re.fullmatch(r'[0-9.]+',token_text) else ('other',edge['token'])
  if key in keys:continue
  keys.add(key);selected.append(edge)
 return selected,role
