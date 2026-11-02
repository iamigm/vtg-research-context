from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Change only event query, reusing the exact video tensor, for 25 early-biased cases."""
import argparse
import json,copy,time
import numpy as np
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from hidden_capture import HiddenCapture,SOURCE,encode_sample,write_json
from spatial_analysis import OUT,RESULTS

CONTROLS=['A dog catches a frisbee in a park.','A violinist plays a violin on a stage.']


def replace_query(inputs,info,expanded,query,proc):
    start=expanded.rindex(info['prompt'])+len(info['prompt'])-len(info['query'])
    changed=expanded[:start]+query+expanded[start+len(info['query']):]
    enc=proc.tokenizer(changed,add_special_tokens=False,return_offsets_mapping=True)
    ids=enc['input_ids']
    qpos=[i for i,(s,e) in enumerate(enc['offset_mapping']) if s<start+len(query) and e>start]
    vpos=[i for i,token in enumerate(ids) if token==proc.tokenizer.convert_tokens_to_ids('<|video_pad|>')]
    assert vpos==info['video_positions']
    # Prefix up to the start of the event text is exactly the original prompt.
    before=min(qpos[0],info['query_positions'][0])
    assert ids[:before]==inputs['input_ids'][0,:before].tolist()
    result=dict(inputs)
    result['input_ids']=torch.tensor([ids],device='cuda:0')
    result['attention_mask']=torch.ones_like(result['input_ids'])
    newinfo=copy.deepcopy(info)
    newinfo.update(query=query,query_positions=qpos,sequence_length=len(ids),
                   query_tokens=[proc.tokenizer.decode([ids[i]]) for i in qpos])
    return result,newinfo


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    cases=json.loads((relocate_path(OUT/'strong_samples.json')).read_text())
    cases=[s for s in cases if not args.only or s['sample_key'] in args.only]
    manifest={s['sample_key']:s for s in json.loads((relocate_path(RESULTS/'samples.json')).read_text())}
    pending=[s for s in cases if not (relocate_path(OUT/s['sample_key']/'counterfactual_validation.json')).exists()]
    if not pending:return
    proc=AutoProcessor.from_pretrained(relocate_path(SOURCE/'model'),local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(relocate_path(SOURCE/'model'),local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager',device_map='cuda:0').eval()
    capture=HiddenCapture(model)
    try:
        for case in pending:
            start=time.time();key=case['sample_key'];folder=relocate_path(OUT/key)
            inputs,info,expanded=encode_sample(manifest[key],proc)
            saved=json.loads((relocate_path(RESULTS/case['dataset']/key/'metadata.json')).read_text())
            for field in ['video_positions','query_positions','sampled_timestamps','video_grid_thw']:
                assert info[field]==saved[field]
            inputs=inputs.to('cuda:0');attentions=[];cosines=[];counts=[]
            for i,query in enumerate([info['query']]+CONTROLS):
                current,meta=(inputs,info) if i==0 else replace_query(inputs,info,expanded,query,proc)
                capture.configure(meta,'cuda:0')
                with torch.inference_mode():
                    out=model.model(**current,use_cache=False,return_dict=True)
                del out
                a=np.stack(capture.token_attention)[8:22].astype(float).mean(0)
                c=np.stack(capture.token_cosine)[8:22].astype(float).mean(0)
                if i==0:
                    prior=np.load(relocate_path(GENERATION/case['dataset']/key/'attention.npz'))['query_temporal']
                    assert np.array_equal(prior,np.stack(capture.temporal))
                    old=np.load(relocate_path(RESULTS/case['dataset']/key/'distributions.npz'))
                    np.testing.assert_allclose(a,old['attention_video_token'][8:22].astype(float).mean(0),atol=1e-8,rtol=2e-6)
                T,H,W=info['video_grid_thw'];attentions.append(a.reshape(T,H//2,W//2));cosines.append(c.reshape(T,H//2,W//2))
                counts.append(len(meta['query_positions']))
                print(key,i,query,flush=True)
            np.savez_compressed(relocate_path(folder/'counterfactual.npz'),attention=np.stack(attentions),cosine=np.stack(cosines))
            write_json(relocate_path(folder/'counterfactual_validation.json'),dict(original_attention_bit_identical=True,
                video_tensor_reused=True,prefix_before_event_identical=True,video_positions_identical=True,
                queries=[info['query']]+CONTROLS,query_token_counts=counts,elapsed_seconds=time.time()-start))
            del inputs,current
            torch.cuda.empty_cache()
    finally:capture.restore()


if __name__=='__main__':main()
