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
    # Complete response files are immutable. Reuse derived query rows only when
    # their source file predates the aggregate; calculate new/partial records normally.
    def cached_rows(name):
        path=scope.CANONICAL/'metrics'/name
        if not path.exists():
            return [],set()
        data=scope.json_read(path)
        grouped=collections.defaultdict(list)
        for row in data:
            grouped[row['key']].append(row)
        valid=set()
        for key,values in grouped.items():
            response=scope.CANONICAL/'responses'/f'{key}.json'
            if len(values)==24 and response.exists() and response.stat().st_mtime_ns<=path.stat().st_mtime_ns:
                valid.add(key)
        return data,valid
    base=report.original.original
    collect=base.collect
    def collect_cached(cohort):
        data,valid=cached_rows('per_query.json')
        keys={r['sample_key'] for r in cohort}
        rows,done=collect([r for r in cohort if r['sample_key'] not in valid])
        return [r for r in data if r['key'] in keys & valid]+rows,[r['sample_key'] for r in cohort if r['sample_key'] in valid]+done
    base.collect=collect_cached
    cost_module=report.original
    cost_read=cost_module.read
    cost_rows=cost_module.cost_rows
    def costs_cached():
        data,valid=cached_rows('cost_per_query.json')
        cohort=scope.json_read(scope.COHORT)
        keys={r['sample_key'] for r in cohort}
        def filtered_read(path):
            if Path(path)==scope.COHORT:
                return [r for r in cohort if r['sample_key'] not in valid]
            return cost_read(path)
        cost_module.read=filtered_read
        try:
            fresh=cost_rows()
        finally:
            cost_module.read=cost_read
        return [r for r in data if r['key'] in keys & valid]+fresh
    cost_module.cost_rows=costs_cached
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
        text+='\n원자료는 누적 pilot900에 한 번 저장한다. pilot300·600·900은 고정 cohort 부분집합이며 겹치는 query를 독립 합산하지 않는다.\n'
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


def overview_report():
    overview=scope.ROOT/'results/rollout/temporal_mode_search/pilot/overview/capped_rollouts'
    overview.mkdir(parents=True,exist_ok=True)
    lines=['# Temporal mode search · 누적 pilot300·600·900','','고정 cohort는300 ⊂ 600 ⊂ 900이다. 동일 query를 독립 합산하지 않는다. 응답은 pilot900에 한 번 저장한다.','','| Query | 완료 | 보고서 |','|---:|---:|---|']
    for count in (300,600,900):
        info=scope.json_read(scope.FAMILY/str(count)/'capped_rollouts/metrics/summary.json')
        lines.append(f'| {count} | {info["queries_completed"]}/{count} | [pilot{count}](../../{count}/capped_rollouts/REPORT.md) |')
    lines+=['','mIoU는 query별 IoU 평균,Δpp는 같은 query의 zero-shot 대비100×평균 차이다. 높을수록 좋다. seed42·43을 query 안에서 평균한다. 진행 중 pilot900의 값은 완료된 부분집합이며 최종900 성능이 아니다.','','| Query | 총 rollout | 방법 | 평가 n | mIoU | Zero-shot | Δpp |','|---:|---:|---|---:|---:|---:|---:|']
    for count in (300,600,900):
        for g in scope.json_read(scope.FAMILY/str(count)/'capped_rollouts/metrics/zero_shot_comparison.json')['groups']:
            if g['queries']:
                lines.append(f'| {count} | {g["total_rollout_budget"]} | {g["method"]} | {g["queries"]} | {g["iou"]:.4f} | {g["zero_shot"]["iou"]:.4f} | {g["delta_pp"]["iou"]:+.2f} |')
    (overview/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',type=int,choices=[300,600,900]);parser.add_argument('--all',action='store_true');parser.add_argument('--initialize',action='store_true');parser.add_argument('--viewer-only',action='store_true');parser.add_argument('--overview-only',action='store_true')
    args=parser.parse_args()
    if args.overview_only:
        overview_report()
    elif args.viewer_only:
        prepare(args.scope or 900)
        install_view(args.scope or 900)
    elif args.all:
        for count in (900,300,600):
            subprocess.run([sys.executable,__file__,'--scope',str(count),*(['--initialize'] if args.initialize else [])],check=True)
        overview_report()
        subprocess.run([sys.executable,str(scope.ROOT.parent/'_lib/build_index.py')],check=True)
    else:
        refresh(args.scope or 900,args.initialize)
