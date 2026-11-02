#!/usr/bin/env python3
"""Zero-shot evaluation and decoding experiments; pure predictions are shared."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parent/'_lib'))
from result_layout import relocate_path, write_result_text
from pathlib import Path
import argparse,json,os,subprocess,sys

ROOT=relocate_path(Path(__file__)).resolve().parent
PYTHON=relocate_path(ROOT/'.venv/bin/python')


def invoke(module,extra=()):
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(relocate_path(ROOT/'_lib')),str(relocate_path(ROOT/'python_deps')),env.get('PYTHONPATH','')])
    env.setdefault('MPLBACKEND','Agg')
    subprocess.run([str(PYTHON),str(relocate_path(ROOT/'_lib'/f'{module}.py')),*extra],cwd=ROOT,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command')
    sub.add_parser('report',help='Validate and aggregate pure zero-shot predictions; no inference')
    span=sub.add_parser('span-head')
    span.add_argument('action',choices=['report','run','smoke'])
    span.add_argument('--model',choices=['qwen3','qwen25'],default='qwen3')
    span.add_argument('--shard',type=int,default=0);span.add_argument('--shards',type=int,default=2)
    top=sub.add_parser('top30')
    top.add_argument('action',choices=['report','run','smoke','serve'])
    top.add_argument('--model',choices=['qwen3','qwen25'],default='qwen3')
    top.add_argument('--shard',type=int,default=0);top.add_argument('--shards',type=int,default=2)
    top.add_argument('--port',type=int,default=8770)
    args=parser.parse_args()
    if args.command in (None,'report'):invoke('zeroshot_report');return
    if args.command=='span-head' and args.action=='report':invoke('timelens_span_report');return
    if args.command=='top30' and args.action=='report':invoke('timelens_top30_report');return
    if args.command=='top30' and args.action=='serve':
        out=relocate_path(ROOT/'results/timelens_top30');record=relocate_path(out/'server.json')
        if record.exists():
            old=json.loads(record.read_text());cmd=relocate_path(Path(f'/proc/{old["pid"]}/cmdline'))
            if cmd.exists() and b'timelens_top30_serve.py' in cmd.read_bytes():
                print(json.dumps(old));return
        with (relocate_path(out/'logs/server.log')).open('a') as stream:
            proc=subprocess.Popen([str(PYTHON),'-u',str(relocate_path(ROOT/'_lib/timelens_top30_serve.py')),'--port',str(args.port)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        data=dict(pid=proc.pid,port=args.port,url=f'http://127.0.0.1:{args.port}/',log='logs/server.log')
        write_result_text(record, json.dumps(data,indent=2)+'\n');print(json.dumps(data));return
    extra=['--model',args.model,'--shard',str(args.shard),'--shards',str(args.shards)]
    if args.action=='smoke':extra+=['--smoke']
    invoke('timelens_span_head' if args.command=='span-head' else 'timelens_top30',extra)


if __name__=='__main__':main()
