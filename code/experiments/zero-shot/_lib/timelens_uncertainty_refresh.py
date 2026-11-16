"""Rebuild all Token uncertainty reports on conditional top-30 metrics, CPU only."""
import argparse
import ast
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

from result_layout import relocate_path
from artifact_store import register
from token_uncertainty_metrics import entropy_top30, selected_top30, DEFINITION

ROOT = Path('/home/irteam/experiments/zero-shot')
LIB = ROOT/'_lib'
RESULT = ROOT/'results/uncertainty/token_analysis'
ART = ROOT/'artifacts/runs/uncertainty/token_analysis'
JOB = ART/'full/9404/top30_renormalized'
STATUS = JOB/'metadata/report_refresh_status.json'
PYTHON = ROOT/'.venv/bin/python'
STAGES = ['top30_report','entropy_iou','prefix_entropy','endpoint_gt_entropy','digit_role_entropy',
          'numeric_uncertainty','entropy_integrated','digit_correctness','blue_analysis',
          'full_digit_prepare','full_digit_impl','full_blue_impl','first_error_distance']


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def state(stage, completed, status='running', **more):
    previous=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    if 'finished' not in more:
        more['finished']=previous.get('finished',[])
    if status=='failed':
        more['failed_stage']=previous.get('stage')
    value=dict(state=status,stage=stage,completed_stages=completed,total_stages=len(STAGES)+3,
               timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),scope='pilot/300 and full/9404',**more)
    save(STATUS,value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def decision_hash(scope):
    path=relocate_path(RESULT/scope/'digit_correctness/per_step.csv')
    h=hashlib.sha256();n=0
    # Hash semantic labels, ordering and identifiers, excluding numerical uncertainty.
    with path.open() as f:
        for row in csv.DictReader(f):
            labels={k:v for k,v in row.items() if 'entropy' not in k and 'probability' not in k}
            h.update(json.dumps(labels,sort_keys=True).encode());n+=1
    ep=relocate_path(RESULT/scope/'digit_correctness/per_endpoint.csv')
    return dict(steps=n,labels_sha256=h.hexdigest(),endpoints_sha256=hashlib.sha256(ep.read_bytes()).hexdigest())


def preflight():
    for name in STAGES:
        ast.parse((LIB/f'timelens_{name}.py').read_text())
    import timelens_blue_analysis as blue
    import timelens_full_blue_impl as full_blue
    blue.validate();full_blue.validate()
    assert abs(entropy_top30({'top30':[{'id':i,'probability':.01} for i in range(30)]})-math.log(30))<1e-12
    assert entropy_top30({'top30':[{'id':i,'probability':.2 if i==0 else 0} for i in range(30)]})==0
    source=json.loads((ART/'pilot/300/top30_renormalized/metadata/sources.json').read_text())
    record=json.loads((ROOT/source['records'][0]['path']).read_text())
    import timelens_full_digit_prepare as prepare
    ep=prepare.extract(record)
    if ep:
        for endpoint in ep:
            for row in endpoint['steps']:
                t=record['steps'][row['ar_step']-1]
                assert abs(row['entropy']-entropy_top30(t['raw']))<1e-12
    return {'passed':True,'analytic_entropy_checks':True,'candidate_mass_fixtures':True,'actual_trace_checked':True}


def run_command(args, stage, done):
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TZ='Asia/Seoul')
    log=JOB/'logs'/f'report_refresh_{stage}.log'
    with log.open('a') as stream:
        process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=ROOT.parent.parent)
        started=time.monotonic()
        while True:
            state(stage,done,child_pid=process.pid,stage_elapsed_seconds=time.monotonic()-started,log=str(log))
            try:
                code=process.wait(timeout=20)
                if code:raise RuntimeError(f'{stage} exited {code}; see {log}')
                return
            except subprocess.TimeoutExpired:
                pass


def publish_source_reports():
    """Present auxiliary MD under results while retaining one copy of source data."""
    for name in ('endpoint_gt_entropy','digit_role_entropy'):
        source=ART/'pilot/300/integrated/sources'/name/'REPORT.md'
        target=RESULT/'pilot/300/integrated'/name/'REPORT.md'
        if not source.exists() and target.exists():continue
        text=source.read_text()
        def link(match):
            dest=match[1]
            if '://' in dest or dest.startswith('#'):return match[0]
            absolute=(source.parent/dest).resolve()
            return ']('+os.path.relpath(absolute,target.parent)+')'
        text=re.sub(r'\]\(([^)]+)\)',link,text)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text)
        source.unlink()
    duplicate=ART/'pilot/300/integrated/sources/prefix_entropy/REPORT.md'
    if duplicate.exists():duplicate.unlink()


def numerical_validation(before):
    checks=[]
    for scope in ('pilot/300','full/9404'):
        after=decision_hash(scope)
        assert before[scope]==after,(scope,before[scope],after)
        expected={}
        with gzip.open(ART/scope/'top30_renormalized/metrics/per_step.csv.gz','rt') as f:
            for row in csv.DictReader(f):
                expected[(row['model'],row['sample_key'],int(row['step']),row['mode'])]=float(row['top30_entropy_nats'])
        names=['digit_correctness/per_step.csv','digit_correctness/blue_analysis/per_blue_step.csv']
        if scope=='pilot/300':
            names+=['prefix_entropy/per_step.csv','numeric_uncertainty/per_step.csv',
                    'integrated/sources/endpoint_gt_entropy/per_step.csv',
                    'integrated/sources/digit_role_entropy/per_step_roles.csv']
        for name in names:
            path=relocate_path(RESULT/scope/name)
            n=0;max_error=0.
            with path.open() as f:
                for row in csv.DictReader(f):
                    step=int(row.get('ar_step') or row['step'])
                    key=(row['model'],row['sample_key'],step,'raw')
                    error=abs(float(row['entropy'])-expected[key]);max_error=max(max_error,error)
                    assert error<1e-10,(path,key,error)
                    if 'processed_entropy' in row:
                        assert abs(float(row['processed_entropy'])-expected[key[:3]+('processed',)])<1e-10
                    if 'safe_lower' in row:
                        total=sum(float(row[k]) for k in ('safe_lower','unsafe_mass','unresolved_mass'))
                        assert abs(total-1)<1e-10 and float(row['tail_mass'])==0
                    n+=1
            checks.append(dict(scope=scope,path=str(path),rows=n,max_entropy_error=max_error))
        if scope=='full/9404':
            path=relocate_path(RESULT/scope/'digit_correctness/first_error_distance/per_first_error.csv')
            with path.open() as f:
                for row in csv.DictReader(f):
                    key=(row['model'],row['sample_key'],int(row['first_error_step']),'raw')
                    assert abs(float(row['entropy_nats'])-expected[key])<1e-10
        # Original hashes were captured by the earlier independent entropy comparison.
        sources=json.loads((ART/scope/'top30_renormalized/metadata/sources.json').read_text())
        for record in sources['records']:
            assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
    save(JOB/'validation/report_refresh_numerical.json',dict(passed=True,checks=checks,
        correctness_labels_unchanged=True,endpoint_metrics_unchanged=True,original_trace_hashes_verified=True))


def browser_validation(scopes=('pilot/300','full/9404')):
    from verify_token_viewer import check
    return check(scopes)


def run(resume=False):
    lock=(JOB/'metadata/report_refresh.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed=[]
    if resume and STATUS.exists():completed=json.loads(STATUS.read_text()).get('finished',[])
    snapshot=JOB/'metadata/report_refresh_before.json'
    try:
        save(JOB/'validation/report_refresh_preflight.json',preflight())
        if not resume or not snapshot.exists():
            save(snapshot,{scope:decision_hash(scope) for scope in ('pilot/300','full/9404')})
        before=json.loads(snapshot.read_text())
        for name in STAGES:
            if name in completed:continue
            run_command([str(PYTHON),'-u',str(LIB/f'timelens_{name}.py')],name,len(completed))
            completed.append(name)
            state(name,len(completed),finished=completed)
        publish_source_reports()
        state('numerical_validation',len(STAGES),finished=completed)
        numerical_validation(before)
        run_command([str(PYTHON),str(ROOT.parent/'_lib/build_index.py')],'index',len(STAGES)+1)
        run_command([str(PYTHON),str(ROOT.parent/'_lib/refresh_result_servers.py')],'server',len(STAGES)+1)
        state('browser_validation',len(STAGES)+2,finished=completed)
        browser_validation()
        reports=[]
        for report in RESULT.rglob('REPORT.md'):
            if 'top30_renormalized' in report.parts or 'overview' in report.parts:continue
            text=report.read_text()
            assert 'top-30 재정규화' in text,report
            with urlopen('http://127.0.0.1:8780/'+str(report.relative_to(ROOT.parent)),timeout=10) as response:
                assert response.status==200
            reports.append(str(report))
        for scope in ('pilot/300','full/9404'):
            register('analysis.'+scope.replace('/','')+'.token_uncertainty_top30','analyses',ART/scope,
                'All Token uncertainty reports use top-30 renormalized entropy and probabilities',
                Path(__file__).name,scope,'complete')
        save(JOB/'validation/report_refresh_publication.json',dict(passed=True,reports=reports))
        state('complete',len(STAGES)+3,'complete',finished=completed,reports=reports)
    except Exception:
        state('failed',len(completed),'failed',finished=completed,error=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['launch','run','preflight'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.command=='preflight':print(json.dumps(preflight()))
    elif args.command=='run':run(args.resume)
    else:
        (JOB/'logs').mkdir(parents=True,exist_ok=True)
        with (JOB/'logs/report_refresh.log').open('a') as stream:
            process=subprocess.Popen([str(PYTHON),'-u',str(Path(__file__).resolve()),'run']+(['--resume'] if args.resume else []),
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,TZ='Asia/Seoul'))
        print(json.dumps(dict(pid=process.pid,status=str(STATUS),log=str(JOB/'logs/report_refresh.log'))))
