"""CPU-only display previews reconstructed from saved visual interventions.

Actual inference pixels remain floating point. Existing original/perturbed
uint8 PNGs are verified and reused. The new amplified view is display-only:
clip(original_RGB + gain * actual_float_delta, 0, 1), then PNG quantization.
No model inference, query mutation, or scientific-array rewrite is performed.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import gc
import math
from pathlib import Path
from oracle_switch_core import OUT,ROOT,MODEL,read,write,sha,stamp


def quantize(rgb):
    import numpy as np
    return (rgb.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)


def amplified_view(original, altered, target_shift=32/255, max_gain=10000.):
    """Return display tensor and explicit frame-local gain/clipping metadata."""
    import torch
    delta=altered-original
    maximum=float(delta.abs().max())
    gain=max(1.,min(max_gain,target_shift/maximum)) if maximum else 1.
    proposed=original+gain*delta
    image=proposed.clamp(0,1)
    clipped=(proposed<0)|(proposed>1)
    changed=delta!=0
    changed_count=int(changed.sum())
    stats=dict(amplification_gain=gain,actual_delta_max_255=maximum*255,
        desired_display_shift_255=target_shift*255,gain_cap=max_gain,
        gain_cap_reached=bool(maximum and target_shift/maximum>max_gain),
        max_display_shift_255=float((image-original).abs().max())*255,
        display_clipped_fraction=float(clipped.float().mean()),
        display_clipped_fraction_changed=float((clipped&changed).sum())/changed_count if changed_count else 0.,
        display_clipped_fraction_denominator='all RGB scalar components of the displayed frame (3*H*W)',
        display_clipped_fraction_changed_denominator='displayed frame RGB scalars with nonzero actual float perturbation',
        actual_float_changed_fraction=float(changed.float().mean()),
        display_formula='clip(original_RGB + amplification_gain * actual_float_RGB_delta,0,1)',
        amplified_is_model_input=False,actual_view_gain=1.)
    return image,stats


def selftest():
    import numpy as np
    import torch
    original=torch.full((3,2,3),.5)
    altered=original.clone();altered[0,0,0]+=.00001
    image,stats=amplified_view(original,altered)
    assert np.array_equal(quantize(original),quantize(altered))
    assert not np.array_equal(quantize(original),quantize(image))
    assert stats['amplification_gain']>1 and stats['amplified_is_model_input'] is False
    same,zero=amplified_view(original,original)
    assert torch.equal(same,original) and zero['amplification_gain']==1.
    original[0,0,1]=.999;altered=original.clone();altered[0,0,1]=1.
    _,clip=amplified_view(original,altered)
    assert clip['display_clipped_fraction']>0 and clip['display_clipped_fraction_changed']==1.
    return dict(passed=True,sub_uint8_change_preserved_in_amplified_view=True,
                zero_delta_unchanged=True,display_clipping_recorded=True)


def generate(base=OUT):
    import numpy as np
    import torch
    from PIL import Image
    from probability_budget_runtime import prepare
    from transformers import AutoProcessor
    from oracle_switch_visual import apply,unpack_pixels

    base=Path(base);torch.set_num_threads(1)
    selection_path=base/'metadata/selection.json';selection=read(selection_path)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
    protected={str(selection_path):sha(selection_path),str(base/'metadata/config.json'):sha(base/'metadata/config.json')}
    samples={};directions_count=identical_count=0
    for sample in selection['samples']:
        key=sample['sample_key'];query_path=base/'queries'/f'{key}.json'
        query=read(query_path)
        assert query['complete'],(key,'Incomplete experiment query')
        protected[str(query_path)]=sha(query_path)
        sign_path=Path(query['gradient']['direction_file'])
        assert sha(sign_path)==query['gradient']['direction_sha256']
        protected[str(sign_path)]=sha(sign_path)
        print(f'CPU preview reconstruction: {key}',flush=True)
        cpu,info=prepare(sample['row'],processor)
        assert info['hashes']==query['input']['hashes']==sample['input_hashes']
        pixels=cpu.pixel_values_videos
        assert pixels.device.type=='cpu'
        grid=query['input']['grid'];mean=query['input']['image_mean'];std=query['input']['image_std']
        actual_processor=processor.video_processor
        assert list(map(float,actual_processor.image_mean))==mean
        assert list(map(float,actual_processor.image_std))==std
        with np.load(sign_path,allow_pickle=False) as arrays:
            sign=torch.from_numpy(arrays['direction_sign'].copy())
        assert sign.dtype==torch.int8 and sign.shape==pixels.shape
        mean_t=torch.tensor(mean).view(1,3,1,1);std_t=torch.tensor(std).view(1,3,1,1)
        sample_result=dict(query_path=str(query_path),query_sha256=protected[str(query_path)],
            direction_array_path=str(sign_path),direction_array_sha256=protected[str(sign_path)],
            input_hashes_verified=True,directions={})
        for direction in query['directions']:
            ident=direction['id'];outcome_name='free_high' if 'free_high' in direction else 'free_limit'
            outcome=direction[outcome_name];epsilon=float(outcome['epsilon'])
            changed,stats=apply(pixels,sign,grid,direction['support'],epsilon,mean,std)
            differences={}
            for name,value in stats.items():
                saved=outcome['perturbation'].get(name)
                if saved is None:continue
                if isinstance(value,bool) or isinstance(value,str):assert value==saved,(key,ident,name,value,saved)
                elif isinstance(value,(int,float)):
                    difference=abs(float(value)-float(saved));differences[name]=difference
                    assert math.isclose(float(value),float(saved),rel_tol=1e-8,abs_tol=1e-12),(key,ident,name,value,saved)
            media=direction['media'];slot=int(media['slot'])
            original=(unpack_pixels(pixels,grid)[slot:slot+1].float()*std_t+mean_t)[0].clamp(0,1)
            altered=(unpack_pixels(changed,grid)[slot:slot+1].float()*std_t+mean_t)[0].clamp(0,1)
            original_u8,altered_u8=quantize(original),quantize(altered)
            for label,expected in [('original',original_u8),('perturbed',altered_u8)]:
                path=Path(media[label]);protected[str(path)]=sha(path)
                with Image.open(path) as image:actual=np.asarray(image.convert('RGB'))
                assert np.array_equal(actual,expected),(key,ident,label,'Existing PNG differs from reconstructed float inputs')
            maximum=float((altered-original).abs().max())
            assert math.isclose(media['difference_gain'],1/maximum if maximum else 1.,rel_tol=1e-12)
            display,display_stats=amplified_view(original,altered)
            output=base/'media'/key/f'{ident}_amplified.png'
            Image.fromarray(quantize(display)).save(output)
            same=bool(np.array_equal(original_u8,altered_u8));identical_count+=int(same);directions_count+=1
            entry=dict(original=str(Path(media['original'])),perturbed=str(Path(media['perturbed'])),
                amplified=str(output),amplified_sha256=sha(output),
                difference=media.get('difference'),difference_gain=media['difference_gain'],
                slot=slot,timestamp=float(query['input']['timestamps'][slot]),
                source_frame_id=int(query['input']['frame_ids'][slot]),input_hw=query['input']['hw'],
                spatial_bounds=direction['support'].get('spatial_bounds'),
                actual_epsilon=epsilon,actual_epsilon_255=epsilon*255,outcome=outcome_name,
                png_identical=same,actual_png_changed_fraction=float(np.mean(original_u8!=altered_u8)),
                original_sha256=sha(media['original']),perturbed_sha256=sha(media['perturbed']),
                reconstruction=dict(input_hashes_match=True,saved_scalar_stats_match=True,
                    scalar_comparison_relative_tolerance=1e-8,scalar_comparison_absolute_tolerance=1e-12,
                    maximum_scalar_absolute_difference=max(differences.values(),default=0.),
                    original_png_pixel_exact=True,perturbed_png_pixel_exact=True),
                **display_stats)
            sample_result['directions'][ident]=entry
            print(f'  {ident}: PNG identical={same}, actual max={maximum*255:.8g}/255, display gain={display_stats["amplification_gain"]:.6g}',flush=True)
            del changed,stats,original,altered,display
            gc.collect()
        samples[key]=sample_result
        del cpu,pixels,sign
        gc.collect()
    assert all(sha(path)==digest for path,digest in protected.items()),'Protected scientific/actual-preview file changed'
    manifest=dict(timestamp=stamp(),scope='pilot',queries=len(samples),directions=directions_count,
        samples=samples,policy=dict(actual='Existing original and actual perturbed PNGs retained at x1; floating model inputs are never quantized for reconstruction.',
            amplified='Display-only clip(original_RGB + gain*actual_float_delta,0,1); not submitted to the model.',
            gain='max(1,min(10000,(32/255)/max(abs(actual_float_delta)))) in the displayed frame; gain=1 for zero delta',
            png='PNG views round RGB*255 to uint8; distinct float inputs may have identical PNG pixels.'),
        validation=dict(passed=True,cpu_only=True,model_inference=False,original_queries_arrays_and_actual_pngs_unchanged=True,
            png_identical_directions=identical_count,protected_files_sha256=protected,selftest=selftest()),
        provenance=dict(runner=str(Path(__file__).resolve()),runner_sha256=sha(__file__),
            visual_helper_sha256=sha(ROOT/'_lib/oracle_switch_visual.py'),
            input_adapter_sha256=sha(ROOT/'_lib/probability_budget_inputs.py'),
            video_processor_sha256=sha(MODEL/'video_preprocessor_config.json')))
    write(base/'analyses/preview_manifest.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--store',type=Path,default=OUT)
    args=parser.parse_args();result=generate(args.store)
    print(f"Verified {result['queries']} queries / {result['directions']} directions; identical actual PNG pairs: {result['validation']['png_identical_directions']}",flush=True)
