"""CPU-only reusable-output catalog and immutable zero-shot capture store."""
from result_layout import canonical, relocate_path, write_result_text, copy_result_file, result_glob
from pathlib import Path
from collections import Counter
import argparse,contextlib,datetime,fcntl,hashlib,json,os,sqlite3,time,traceback
ROOT=Path(canonical(Path(__file__).resolve().parents[1]));STORE=relocate_path(ROOT/'artifacts/common');RUN='full_benchmark_v1'
KINDS=('predictions','token_traces','manifests','inputs','media','rollouts','analyses')
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');write_result_text(tmp, json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path):
    try:return str(relocate_path(Path(path)).absolute().relative_to(ROOT))
    except ValueError:return str(relocate_path(Path(path)).absolute())
def register(resource_id,kind,path,description,producer='',cohort='',state='available',depends_on=()):
    if kind not in KINDS:raise ValueError(kind)
    if not resource_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in resource_id):raise ValueError('Use a stable alphanumeric resource ID')
    p=relocate_path(Path(path));p=p if p.is_absolute() else relocate_path(ROOT/p)
    if not p.exists():raise FileNotFoundError(p)
    target=relocate_path(STORE/'registry'/f'{resource_id}.json');old=json.loads(target.read_text()) if target.exists() else {}
    if old and (old['path']!=relative(p) or old['kind']!=kind):raise ValueError('Resource identity is immutable; register a new version ID')
    atomic(target,dict(schema_version=1,id=resource_id,kind=kind,path=relative(p),description=description,producer=producer,cohort=cohort,state=state,depends_on=list(depends_on),created_at=old.get('created_at',now()),updated_at=now()))
def db():
    STORE.mkdir(exist_ok=True);conn=sqlite3.connect(relocate_path(STORE/'index.sqlite'),timeout=30);conn.execute('pragma journal_mode=WAL');conn.execute('CREATE TABLE IF NOT EXISTS records (run_id TEXT, model TEXT, dataset TEXT, sample_key TEXT, response_path TEXT, trace_path TEXT, response_sha256 TEXT, trace_sha256 TEXT, source_path TEXT, reused_pilot INTEGER, PRIMARY KEY(run_id,model,sample_key))');return conn

def iter_records(run_id=RUN,model=None,dataset=None):
    conn=db();conn.row_factory=sqlite3.Row;where=['run_id=?'];args=[run_id]
    for col,val in [('model',model),('dataset',dataset)]:
        if val is not None:where.append(col+'=?');args.append(val)
    try:
        for row in conn.execute('SELECT * FROM records WHERE '+' AND '.join(where)+' ORDER BY model,dataset,sample_key',args):yield dict(row)
    finally:conn.close()
def read_record(record,kind='response',verify=False):
    if kind not in ('response','trace'):raise ValueError(kind)
    p=relocate_path(ROOT/record[kind+'_path'])
    if verify and digest(p)!=record[kind+'_sha256']:raise ValueError('Stored checksum mismatch: '+str(p))
    return json.loads(p.read_text())
def immutable_json(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Refusing to replace immutable record '+str(path))
    else:atomic(path,value)
def sync(conn):
    source=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404');known={(m,k) for m,k in conn.execute('SELECT model,sample_key FROM records WHERE run_id=?',(RUN,))};added=0
    for model in ['qwen25','qwen3']:
        for path in sorted(result_glob((source/model), '*.json')):
            if (model,path.stem) in known:continue
            raw=json.loads(path.read_text());sample=raw['sample'];key=sample['sample_key'];ds=sample['dataset'];assert key==path.stem
            assert raw['generated_ids']==[s['token_id'] for s in raw['steps']]
            trace=relocate_path(STORE/'token_traces'/RUN/model/ds/path.name);trace.parent.mkdir(parents=True,exist_ok=True)
            if not trace.exists():
                # Source files are atomically published complete records and never edited.
                # A hard link keeps an independent directory entry without duplicating bytes.
                try:os.link(path,trace)
                except OSError:
                    import shutil
                    copy_result_file(path,trace)
            trace_hash=digest(trace)
            if trace_hash!=digest(path):raise ValueError('Trace differs from source '+str(path))
            pilot=relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300'/model/path.name);reused=pilot.exists() and digest(pilot)==trace_hash
            response=relocate_path(STORE/'predictions'/RUN/model/ds/path.name)
            value={k:v for k,v in raw.items() if k not in ['steps','elapsed_seconds']}
            value.update(schema_version=1,run_id=RUN,dataset=ds,sample_key=key,order='video_query',source_trace=relative(trace),source_trace_sha256=trace_hash,reused_pilot=reused)
            immutable_json(response,value)
            conn.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',(RUN,model,ds,key,relative(response),relative(trace),digest(response),trace_hash,relative(path),int(reused)));conn.commit();added+=1
    configdest=relocate_path(STORE/'metadata'/RUN)
    for origin,prefix in [(source,'full_run'),(relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300'),'pilot_run')]:
        paths=list(result_glob((relocate_path(origin/'configs')), '*.json'))
        if (relocate_path(origin/'run_config.json')).exists():paths.append(relocate_path(origin/'run_config.json'))
        for p in paths:
            immutable_json(relocate_path(configdest/prefix/p.name),json.loads(p.read_text()))
    manifest=relocate_path(source/'samples.json');immutable_json(relocate_path(STORE/'manifests'/'full_benchmark_v1.json'),json.loads(manifest.read_text()))
    counts={}
    for m,d,n in conn.execute('SELECT model,dataset,count(*) FROM records WHERE run_id=? GROUP BY model,dataset',(RUN,)):counts.setdefault(m,{})[d]=n
    total=sum(sum(v.values()) for v in counts.values());target=2*len(json.loads(manifest.read_text()));complete=total==target
    state=dict(timestamp=now(),state='complete' if complete else 'syncing',completed=total,total=target,percent=100*total/target,added=added,counts=counts,sha256_verified_on_ingest=True)
    atomic(relocate_path(STORE/'status.json'),state)
    for kind,desc in [('predictions','Full-head greedy responses, separate from probability traces and derived analyses'),('token_traces','Raw and processed top-30 plus full-vocabulary entropy at every generation step')]:
        register(f'{RUN}.{kind}',kind,relocate_path(STORE/kind/RUN),desc,'timelens_top30_full_worker.py','full_benchmark_v1','complete' if complete else 'building',['full_benchmark_v1.cohort'])
    upstream=relocate_path(source/'status.json');up=json.loads(upstream.read_text()) if upstream.exists() else {}
    for resource_id,folder in [('analysis.full.digit_correctness',relocate_path(source/'digit_correctness')),('analysis.full.blue_analysis',relocate_path(source/'digit_correctness/blue_analysis'))]:
        if folder.exists():register(resource_id,'analyses',folder,'Full benchmark '+folder.name,'timelens_full_digit_finish.py','full_benchmark_v1','complete' if up.get('state')=='complete' else 'building',[RUN+'.token_traces'])
    print(json.dumps(state,ensure_ascii=False),flush=True);return complete

def watch(once=False):
    STORE.mkdir(exist_ok=True)
    with (relocate_path(STORE/'.sync.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with contextlib.closing(db()) as conn:
            while True:
                complete=sync(conn)
                if once:break
                p=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404/status.json');up=json.loads(p.read_text()) if p.exists() else {}
                if complete and up.get('state')=='complete':break
                if up.get('state')=='failed':
                    status=json.loads((relocate_path(STORE/'status.json')).read_text());status.update(state='upstream_failed',error=up.get('error'));atomic(relocate_path(STORE/'status.json'),status);raise RuntimeError('Inference supervisor failed; captured records remain reusable')
                time.sleep(15)
def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('register');p.add_argument('--id',required=True);p.add_argument('--kind',required=True,choices=KINDS);p.add_argument('--path',required=True);p.add_argument('--description',required=True);p.add_argument('--producer',default='');p.add_argument('--cohort',default='');p.add_argument('--state',default='available');p.add_argument('--depends-on',action='append',default=[])
    p=sub.add_parser('sync');p.add_argument('--once',action='store_true')
    sub.add_parser('list')
    p=sub.add_parser('verify');p.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if args.command=='register':register(args.id,args.kind,args.path,args.description,args.producer,args.cohort,args.state,args.depends_on)
    elif args.command=='sync':watch(args.once)
    elif args.command=='list':
        for p in sorted(result_glob((relocate_path(STORE/'registry')), '*.json')):
            r=json.loads(p.read_text());print(r['id'],r['kind'],r['state'],r['path'],sep='\t')
    else:
        n=0
        for r in iter_records():
            answer=read_record(r,verify=True);trace=read_record(r,'trace',verify=True);assert 'steps' not in answer;assert all(answer[k]==trace[k] for k in ['answer','generated_ids','timestamps','iou','input_ids_sha256']);n+=1
            if args.limit and n>=args.limit:break
        print(json.dumps(dict(verified=n,passed=True)))
if __name__=='__main__':
    try:main()
    except Exception:
        if len(os.sys.argv)>1 and os.sys.argv[1]=='sync':atomic(relocate_path(STORE/'sync_error.json'),dict(timestamp=now(),error=traceback.format_exc()))
        raise
