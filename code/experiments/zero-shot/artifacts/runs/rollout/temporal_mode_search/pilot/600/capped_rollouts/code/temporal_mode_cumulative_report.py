"""Regenerate pilot600/pilot900 from one canonical response store."""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
import temporal_mode_cumulative_scope as scope


def prepare(count):
    core=scope.configure(count,reporting=True)
    import temporal_mode_prefix_cache_report as report
    modules=[report,report.original,report.original.original]
    for module in modules:
        module.read=scope.canonical_read
    # These constants are only the cohort completion guards in the original MD
    # reporter. No inference code object is modified.
    old=report._base_report
    constants=tuple(count if type(value) is int and value==600 else value for value in old.__code__.co_consts)
    report._base_report=types.FunctionType(old.__code__.replace(co_consts=constants),old.__globals__,old.__name__,old.__defaults__,old.__closure__)
    tables=report.tables
    def canonical_calibration(summary):
        saved=report.OUT
        report.OUT=scope.CANONICAL
        try:
            return tables(summary)
        finally:
            report.OUT=saved
    report.original.tables=canonical_calibration
    return core,report


def scoped_text(text,count):
    return (text.replace('pilot600',f'pilot{count}')
        .replace('600 query 중',f'{count} query 중')
        .replace('600 query×3방법×2seed=3,600 route',f'{count} query×3방법×2seed={count*6:,} route')
        .replace('14,400개 예산별 선택',f'{count*24:,}개 예산별 선택')
        .replace('독립14,400 query',f'독립{count*24:,} query')
        .replace('세 데이터셋별200 query',f'세 데이터셋별{count//3} query')
        .replace('세 데이터셋별200 query',f'세 데이터셋별{count//3} query'))


def install_view(count):
    result=scope.RESULT
    html_path=result/'detail.html'
    html=html_path.read_text()
    html=scoped_text(html,count)
    raw=os.path.relpath(scope.CANONICAL,result)
    tag='<script>window.TEMPORAL_RAW_STORE='+json.dumps(raw)+';</script>'
    html=re.sub(r'<script>window.TEMPORAL_RAW_STORE=.*?</script>','',html)
    html=html.replace('</head>',tag+'</head>')
    html=html.replace('data-family-default','') if count!=900 else html
    if count==900 and 'data-family-default' not in html:
        html=html.replace('<html lang="ko"','<html lang="ko" data-family-default')
    html_path.write_text(html)
    js_path=scope.OUT/'web/viewer.js'
    js=js_path.read_text()
    for suffix in ('.json','.partial.json'):
        old=f"new URL('responses/'+key+'{suffix}',store)"
        new=f"new URL('responses/'+key+'{suffix}',new URL(window.TEMPORAL_RAW_STORE+'/',location.href))"
        js=js.replace(old,new)
    js=js.replace('pilot600',f'pilot{count}').replace('/600 query',f'/{count} query').replace('??3600',f'??{count*6}').replace('??14400',f'??{count*24}')
    js=js.replace('${summary.common_queries}/600',f'${{summary.common_queries}}/{count}')
    js_path.write_text(js)


def refresh(count,initialize=False):
    core,report=prepare(count)
    scope.OUT.mkdir(parents=True,exist_ok=True)
    scope.RESULT.mkdir(parents=True,exist_ok=True)
    for name in ('metrics','web','metadata','validation'):
        (scope.OUT/name).mkdir(exist_ok=True)
    if initialize or not (scope.RESULT/'detail.html').exists():
        report.initialize()
    else:
        report.report()
    for name in ('REPORT.md','SPEC.md'):
        path=scope.RESULT/name
        text=scoped_text(path.read_text(),count)
        text=re.sub(r'(비교 공통 query: \*\*\d+)/600',rf'\g<1>/{count}',text)
        text=re.sub(r'(공통 query: \*\*\d+)/600',rf'\g<1>/{count}',text)
        text+='\n원자료는 누적 pilot900에 한 번 저장한다. pilot600과 pilot900은 고정 cohort 부분집합이며 겹치는 query를 독립 합산하지 않는다.\n'
        path.write_text(text)
    summary=core.read(scope.OUT/'metrics/summary.json')
    summary.update(target_queries=count,complete=summary['queries_completed']==count)
    core.write(scope.OUT/'metrics/summary.json',summary)
    rows=core.read(scope.COHORT)
    complete=core.read(scope.OUT/'web/completed.json')
    if count!=900:
        status=dict(state='complete' if len(complete)==count else 'running',stage='canonical900_subset',
            completed=len(complete),total=count,completed_conditions=len(complete)*6,total_conditions=count*6,
            completed_budget_selections=len(complete)*24,total_budget_selections=count*24,
            seeds=[42,43],counts=dict(collections.Counter(row['dataset'] for row in rows if row['sample_key'] in set(complete))),
            canonical_source=str(scope.CANONICAL),timestamp=core.stamp())
        core.write(scope.OUT/'metadata/status.json',status)
    core.write(scope.OUT/'metadata/source_records.json',dict(canonical_source=str(scope.CANONICAL),
        queries=count,cohort=str(scope.COHORT),cohort_sha256=core.sha(scope.COHORT),
        records=[dict(sample_key=row['sample_key'],response=str(scope.CANONICAL/'responses'/f'{row["sample_key"]}.json')) for row in rows]))
    install_view(count)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',type=int,choices=[600,900]);parser.add_argument('--all',action='store_true');parser.add_argument('--initialize',action='store_true');parser.add_argument('--viewer-only',action='store_true')
    args=parser.parse_args()
    if args.viewer_only:
        prepare(args.scope or 900)
        install_view(args.scope or 900)
    elif args.all:
        for count in (600,900):
            subprocess.run([sys.executable,__file__,'--scope',str(count),*(['--initialize'] if args.initialize else [])],check=True)
        subprocess.run([sys.executable,str(scope.ROOT.parent/'_lib/build_index.py')],check=True)
    else:
        refresh(args.scope or 900,args.initialize)
