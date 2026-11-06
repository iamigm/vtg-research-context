"""Paired native single-token throughput, three clients, same model and logits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT=Path('/home1/irteam/experiments/zero-shot')
OUT=PROJECT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/capped_rollouts'
PIPE=Path('/tmp/irteam_temporal_mode_mps')


def client(mode,rank):
    import fps_spatial_control
    import torch
    from transformers import AutoModelForImageTextToText
    torch.set_num_threads(2)
    model=AutoModelForImageTextToText.from_pretrained('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct',
        local_files_only=True,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
    n=8192
    with torch.inference_mode():
        output=model(input_ids=torch.full((1,n),15,device='cuda'),attention_mask=torch.ones((1,n),device='cuda'),
            use_cache=True,logits_to_keep=1)
    cache=output.past_key_values
    inputs=dict(input_ids=torch.tensor([[20]],device='cuda'),attention_mask=torch.ones((1,n+1),device='cuda'),
        cache_position=torch.tensor([n],device='cuda'),use_cache=True,logits_to_keep=1)
    folder=OUT/'validation/mps_benchmark'
    folder.mkdir(exist_ok=True)
    (folder/f'{mode}_{rank}.ready').write_text('ready')
    while not (folder/f'{mode}.go').exists():
        time.sleep(.05)
    torch.cuda.synchronize()
    started=time.perf_counter()
    hashes=[]
    for _ in range(80):
        cache.crop(n)
        with torch.inference_mode():
            output=model(**inputs,past_key_values=cache)
        cache=output.past_key_values
        logits=output.logits[0,-1].float().cpu()
        hashes.append(hashlib.sha256(logits.numpy().tobytes()).hexdigest())
    elapsed=time.perf_counter()-started
    (folder/f'{mode}_{rank}.json').write_text(json.dumps(dict(mode=mode,rank=rank,forwards=80,
        wall_seconds=elapsed,logit_hashes=hashes)))


def run():
    folder=OUT/'validation/mps_benchmark'
    folder.mkdir(exist_ok=True)
    PIPE.mkdir(mode=0o700,exist_ok=True)
    env=os.environ.copy()
    env['CUDA_MPS_PIPE_DIRECTORY']=str(PIPE)
    env['CUDA_MPS_LOG_DIRECTORY']=str(OUT/'logs/mps')
    Path(env['CUDA_MPS_LOG_DIRECTORY']).mkdir(exist_ok=True)
    subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
    summary={}
    try:
        for mode in ('native','mps'):
            run_env=env.copy()
            if mode=='native':
                run_env['CUDA_MPS_PIPE_DIRECTORY']=str(PIPE/'disabled')
            for path in folder.glob(mode+'*'):
                path.unlink()
            processes=[]
            for rank in range(3):
                log=(OUT/'logs'/f'mps_benchmark_{mode}_{rank}.log').open('w')
                processes.append(subprocess.Popen([sys.executable,__file__,'--client',mode,str(rank)],
                    env=run_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT))
                log.close()
            deadline=time.monotonic()+180
            while len(list(folder.glob(mode+'_*.ready')))<3:
                assert all(p.poll() is None for p in processes),'benchmark client failed'
                assert time.monotonic()<deadline,'benchmark initialization timeout'
                time.sleep(.1)
            (folder/f'{mode}.go').write_text('go')
            for process in processes:
                assert process.wait(timeout=120)==0
            rows=[json.loads((folder/f'{mode}_{rank}.json').read_text()) for rank in range(3)]
            summary[mode]=dict(clients=rows,tokens_per_second=240/max(row['wall_seconds'] for row in rows))
            print(mode,summary[mode]['tokens_per_second'],flush=True)
        assert summary['native']['clients'][0]['logit_hashes']==summary['mps']['clients'][0]['logit_hashes']
        summary.update(passed=True,logits_bitwise_equal=True,speedup=summary['mps']['tokens_per_second']/summary['native']['tokens_per_second'],
            context_tokens=8192,clients=3,scope='synthetic repeated native decode; not pilot completion speed')
        (OUT/'validation/mps_benchmark.json').write_text(json.dumps(summary)+'\n')
        print('MPS_THROUGHPUT',summary['speedup'],flush=True)
    finally:
        subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,check=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--client',nargs=2);args=parser.parse_args()
    client(args.client[0],int(args.client[1])) if args.client else run()
