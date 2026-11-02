"""One saved response replay after rebuilding the runtime; no experiment outputs overwritten."""
import copy,sys
import spatial_budget_pilot900 as m
import torch,transformers,torchvision,flash_attn
from transformers import AutoProcessor,AutoModelForImageTextToText
assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.57.1'
job=next(j for j in m.read(m.OUT/'metadata/jobs.json') if j['condition']=='uniform' and j['level']=='high')
key=job['sample_key'];saved=m.read(m.OUT/'responses'/job['level']/job['condition']/(key+'.json'));previous=m.read(m.OUT/'inputs'/job['level']/job['condition']/(key+'.json'))
r=next(r for r in m.bridge.records() if r['sample']['sample_key']==key);r.update(F=job['F'],B=job['B'],grid=job['grid'])
mp=m.read(m.OUT/'metadata/prepared.json');cache=m.read(mp[job['video']]['metadata'][job['level']]);plan=m.read(m.OUT/'plans'/job['condition']/(key+'.json'))
torch.set_num_threads(2);processor=AutoProcessor.from_pretrained(m.old.MODEL,padding_side='left',do_resize=False,local_files_only=True)
cpu,detail=m.bridge.make_input(r,cache,plan,processor);assert detail['hashes']==previous['hashes'],'Recovered processor changed input'
model=AutoModelForImageTextToText.from_pretrained(m.old.MODEL,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
result=m.old.generate_fresh(cpu,detail,saved['identity'],model,processor,m.GEN,0)
passed=result['generated_ids']==saved['generated_ids']
m.write(m.OUT/'validation/recovered_runtime.json',dict(passed=passed,timestamp=m.stamp(),sample_key=key,input_hashes_equal=True,generated_ids_equal=passed,torch=torch.__version__,torchvision=torchvision.__version__,transformers=transformers.__version__,flash_attn=flash_attn.__version__,python=sys.version,scope='one saved high-resolution Uniform; not universal reproducibility proof'))
assert passed,'Recovered runtime generated IDs differ'
print('Recovered runtime: exact input and generated IDs match',flush=True)
