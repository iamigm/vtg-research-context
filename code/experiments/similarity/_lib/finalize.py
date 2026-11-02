"""Finish artifacts automatically after both resumable embedding runs complete."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from prepare import ROOT,OUT,write

def main():
    while True:
        statuses={}
        for m in ['clip','qwen']:
            p=OUT/m/'status.json'
            statuses[m]=json.loads(p.read_text()) if p.exists() else {}
        if all(v.get('stage')=='complete' for v in statuses.values()):break
        time.sleep(5)
    write(OUT/'execution.json',dict(
        final_batch_sizes=dict(clip=256,qwen=128),
        batch_history=dict(clip=[256],qwen=[128]),
        note='Resumable independent frame/query embeddings; compatible pilot video caches reused when available.',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'_lib').glob('*')) if p.is_file()},
        model_statuses=statuses))
    for script in ['report.py','validate.py','check_viewer.py']:
        print('RUN',script,flush=True)
        subprocess.run([sys.executable,str(ROOT/'_lib'/script)],check=True)
    execution=json.loads((OUT/'execution.json').read_text())
    execution.update(complete=True,samples=len(json.loads((OUT/'samples.json').read_text())),models=2,
                     quantitative_validation=True,browser_validation=True)
    write(OUT/'execution.json',execution)
    print('ALL COMPLETE',flush=True)

if __name__=='__main__':main()
