"""Small non-scored checks, kept separate from the 6,000 production responses."""
import os, sys, time, gc, traceback
from pathlib import Path
from fps_spatial_control import *

def main():
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    samples,groups=setup();byvideo={g['video']:g for g in groups}
    short=min((s for s in samples if s['dataset']=='charades-timelens'),key=lambda s:s['duration'])
    longest=max(samples,key=lambda s:s['duration'])
    chosen=[short,longest];prepared={};checks=[];pixels={};ids={};torch.set_num_threads(2)
    for sample in chosen:
        print('Preparing smoke video',sample['sample_key'],sample['duration'],flush=True)
        prepared[sample['sample_key']]=prepare_video(byvideo[sample['video_path']])
    for label,path in MODELS.items():
        print('Smoke model',label,flush=True)
        proc=AutoProcessor.from_pretrained(path,padding_side='left',do_resize=False,local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True).eval()
        for sample in chosen:
            public={k:sample[k] for k in ['sample_key','dataset','video_path','query','duration']};cache=prepared[sample['sample_key']]
            fs=FPS if sample is short else [8]
            for fps in fs:
                orders=ORDERS if sample is short else ['video_query']
                for order in orders:
                    cpu,detail=prepare_input(public,cache,fps,order,proc)
                    need(detail['input_tokens']+CONFIG[label]['max_new_tokens']<model.config.text_config.max_position_embeddings,'Context overflow')
                    key=(sample['sample_key'],fps);sig={k:v for k,v in detail['hashes'].items() if 'pixel' in k or 'grid' in k}
                    if key in pixels:need(pixels[key]==sig,'Vision differs across order/model')
                    pixels[key]=sig;need(detail['hw']==cache['hw'],'FPS changed spatial grid')
                    # Both endpoint rates, orders and model variants exercise model.generate.
                    generate=(fps in [2,8]) and (sample is short or label=='instruct')
                    if generate:
                        torch.cuda.synchronize();start=time.perf_counter();torch.manual_seed(42)
                        kw=dict(CONFIG[label],max_new_tokens=2)
                        gpu=cpu.to('cuda')
                        with torch.inference_mode():out=model.generate(**gpu,**kw)
                        torch.cuda.synchronize();generated=out[0,cpu.input_ids.shape[1]:].tolist()
                        need(len(generated)>0,'Empty smoke generation');del out,gpu
                    checks.append(dict(sample_key=sample['sample_key'],model=label,fps=fps,order=order,hw=detail['hw'],grid=detail['grid'],input_tokens=detail['input_tokens'],
                                       unique_frames=detail['plan']['unique_frames'],padding_frames=detail['plan']['padding_frames'],generation_checked=generate,
                                       generation_seconds=time.perf_counter()-start if generate else None))
                    print('Smoke probe',checks[-1],flush=True);del cpu
        del model,proc;gc.collect();torch.cuda.empty_cache()
    keys=[{s['sample_key'] for s in samples if int(byvideo[s['video_path']]['key'],16)%2==r} for r in range(2)]
    need(not keys[0]&keys[1] and len(keys[0]|keys[1])==300,'Worker partition incomplete or overlapping')
    write(OUT/'validation/smoke.json',dict(passed=True,completed_at=stamp(),probes=checks,source_rgb_hash_verified=True,fixed_grid_across_fps=True,
                                         pixel_hashes_match_across_models_orders=True,context_checked_at_longest_8fps=True,partition_counts=[len(k) for k in keys],
                                         max_new_tokens=2,production_responses_written=0,production_generation_config_unchanged=True,
                                         model_manifest_sha256=sha(MODEL_ROOT.parent/'manifests/registry.json')))
    print('SMOKE PASSED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'validation/smoke.json',dict(passed=False,error=traceback.format_exc(),timestamp=stamp()));raise
