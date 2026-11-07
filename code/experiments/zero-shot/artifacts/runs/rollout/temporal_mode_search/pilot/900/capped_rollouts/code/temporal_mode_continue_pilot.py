"""Detached handoff from the running pilot600 to one cumulative pilot900 store."""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
import temporal_mode_cumulative_scope as scope

FILES=['temporal_mode_cumulative_scope.py','temporal_mode_cumulative_runtime.py',
    'temporal_mode_cumulative_report.py','temporal_mode_cumulative_check.py',
    'temporal_mode_cumulative_view_check.py','temporal_mode_continue_pilot.py']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False)+'\n');temporary.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:
        return False


def migrate():
    from temporal_mode_capped_core import digest,stamp,read,SOURCE
    old=scope.OLD;new=scope.CANONICAL
    plan=read(old/'metadata/cumulative_plan.json')
    assert sha(old/'metadata/config.json')==plan['source_config_sha256']
    for entry in plan['files']:
        assert sha(entry['snapshot'])==entry['sha256']
    origin=read(old/'metadata/config.json');config=dict(origin,queries=900)
    rows=read(scope.COHORT_ROOT/'pilot_900/samples.json')
    assert read(scope.COHORT_ROOT/'pilot_600/samples.json')==rows[:600]
    assert read(old/'metadata/status.json')['state']=='complete'
    assert read(old/'validation/final.json')['passed']
    paths=sorted((old/'responses').glob('*.json'))
    assert len(paths)==600 and not any(p.name.endswith('.partial.json') for p in paths)
    records=[]
    for row in rows[:600]:
        path=old/'responses'/f'{row["sample_key"]}.json'
        inputs=read(old/'inputs'/f'{row["sample_key"]}.json')
        assert inputs['hashes']==read(SOURCE/'inputs'/f'{row["sample_key"]}.json')['hashes']
        records.append(dict(name=path.name,sha256=sha(path)))
    origin_manifests={name:(old/'metadata'/name).read_text() for name in ('manifest.json','cost_manifest.json','prefix_cache_manifest.json')}
    previous_status=read(old/'metadata/status.json')
    new.parent.mkdir(parents=True,exist_ok=True)
    assert not new.exists()
    old.rename(new)
    sys.path.insert(0,str(new/'code'))
    for name in ('web','metrics'):
        shutil.copytree(new/name,old/name)
    write(old/'metadata/status.json',dict(previous_status,canonical_source=str(new)))
    write(new/'metadata/reuse.json',dict(source_queries=600,canonical_queries=900,
        source_config=origin,source_config_hash=digest(origin),canonical_config_hash=digest(config),
        scope_only_config_changes=['queries'],source_cohort_sha256=sha(scope.COHORT_ROOT/'pilot_600/samples.json'),
        canonical_cohort_sha256=sha(scope.COHORT_ROOT/'pilot_900/samples.json'),
        source_manifests=origin_manifests,responses=records,timestamp=stamp()))
    write(new/'metadata/config.json',config)
    def relocate(value):
        if isinstance(value,str):return value.replace(str(old),str(new))
        if isinstance(value,list):return [relocate(x) for x in value]
        if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
        return value
    for name in ('manifest.json','cost_manifest.json','prefix_cache_manifest.json'):
        manifest=relocate(json.loads(origin_manifests[name]))
        if name=='manifest.json':
            manifest.update(config_hash=digest(config),queries=900,cohort=str(scope.COHORT_ROOT/'pilot_900/samples.json'),
                cohort_sha256=sha(scope.COHORT_ROOT/'pilot_900/samples.json'),unique_videos=len({r['video_path'] for r in rows}))
        else:
            manifest['generation_manifest_sha256']=sha(new/'metadata/manifest.json')
            if name=='prefix_cache_manifest.json':manifest['cost_manifest_sha256']=sha(new/'metadata/cost_manifest.json')
        write(new/'metadata'/name,manifest)
    entries=[]
    for name in FILES:
        source=Path(__file__).with_name(name);target=new/'code'/name
        if not source.exists():
            source=target
        if source.resolve()!=target.resolve():
            shutil.copyfile(source,target)
        entries.append(dict(source=str(source),snapshot=str(target),sha256=sha(target)))
    write(new/'metadata/cumulative_manifest.json',dict(queries=900,source_queries=600,
        cohort_sha256=sha(scope.COHORT_ROOT/'pilot_900/samples.json'),files=entries,timestamp=stamp()))
    from temporal_mode_cumulative_report import prepare,install_view,scoped_text
    core,report=prepare(900)
    result=scope.ROOT/'results/rollout/temporal_mode_search/pilot/900/capped_rollouts'
    result.mkdir(parents=True,exist_ok=True)
    old_result=scope.ROOT/'results/rollout/temporal_mode_search/pilot/600/capped_rollouts'
    html=(old_result/'detail.html').read_text().replace('pilot/600/capped_rollouts','pilot/900/capped_rollouts').replace('pilot600','pilot900')
    (result/'detail.html').write_text(html)
    (result/'REPORT.md').write_text('# Temporal mode search · capped rollouts · pilot900\n\n상태: 실행 중·미완료. 고정900 query 중 기존600 query를 재사용한다. 세 데이터셋별300 query,seed42·43,총 rollout32·64·128·256을 유지한다. 상세 집계는 자동 갱신 중이다.\n\n[기존 pilot600 보고서](../../600/capped_rollouts/REPORT.md)\n')
    (result/'SPEC.md').write_text(scoped_text((old_result/'SPEC.md').read_text(),900)+'\n\n고정900 cohort로 범위만 확장한다. 기존600의 원본 response/config hash는 보존하며 원자료는 누적900에 통합한다.\n')
    core.write(new/'web/samples.json',report.original.original.common.sample_catalog(rows))
    core.write(new/'metadata/status.json',dict(previous_status,state='running',stage='cumulative_handoff',completed=600,total=900,
        percentage=600/9,total_conditions=5400,total_budget_selections=21600,timestamp=stamp()))
    install_view(900)
    subprocess.run([sys.executable,str(new/'code/temporal_mode_cumulative_report.py'),'--scope','600','--viewer-only'],check=True)
    for count in (600,900):
        selected=rows[:count]
        write(scope.FAMILY/str(count)/'capped_rollouts/metadata/source_records.json',dict(queries=count,
            canonical_source=str(new),cohort=str(scope.COHORT_ROOT/f'pilot_{count}/samples.json'),
            records=[dict(sample_key=r['sample_key'],response=str(new/'responses'/f'{r["sample_key"]}.json')) for r in selected]))
    overview=scope.ROOT/'results/rollout/temporal_mode_search/pilot/overview'
    overview.mkdir(parents=True,exist_ok=True)
    (overview/'REPORT.md').write_text('# Temporal mode search · 누적 pilot\n\n고정600 ⊂ 900 cohort다. 응답은 pilot900에 한 번 저장하고 각 범위를 부분집합으로 평가한다. 중복 query를 독립 합산하지 않는다.\n\n| 범위 | Query | 보고서 |\n|---|---:|---|\n| pilot | 600 | [결과](../600/capped_rollouts/REPORT.md) |\n| pilot | 900 | [결과](../900/capped_rollouts/REPORT.md) |\n')
    subprocess.run([sys.executable,str(scope.ROOT.parent/'_lib/build_index.py')],check=True)
    subprocess.run([sys.executable,str(scope.ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
    from artifact_store import register
    for count in (600,900):
        register(f'temporal_mode_search.capped_rollouts.pilot{count}.responses','rollouts',new/'responses',
            f'Cumulative pilot{count}; shared canonical900; generation unchanged.',producer='temporal_mode_cumulative_runtime.py',
            cohort=str(scope.COHORT_ROOT/f'pilot_{count}/samples.json'),state='complete' if count==600 else 'building')
    print(json.dumps(dict(state='running',stage='cumulative900_start',queries=900,reused_queries=600,timestamp=stamp())),flush=True)
    os.execv(sys.executable,[sys.executable,'-u',str(new/'code/temporal_mode_cumulative_runtime.py')])


def run():
    from temporal_mode_capped_core import stamp
    if scope.CANONICAL.exists():
        launcher=scope.json_read(scope.CANONICAL/'metadata/launcher.json')
        if alive(launcher['supervisor_pid']):
            print('Cumulative900 is already running',flush=True)
            return
        state=scope.json_read(scope.CANONICAL/'metadata/execution_scope.json')
        if state.get('watcher_pid') and alive(state['watcher_pid']):
            print('Cumulative900 handoff is already active',flush=True)
            return
        os.execv(sys.executable,[sys.executable,'-u',str(scope.CANONICAL/'code/temporal_mode_cumulative_runtime.py')])
    result=scope.preflight()
    write(scope.OLD/'validation/cumulative_preflight.json',result)
    with (scope.OLD/'metadata/cumulative_continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            status=scope.json_read(scope.OLD/'metadata/status.json')
            launcher=scope.json_read(scope.OLD/'metadata/launcher.json')
            state=dict(state='waiting_for_pilot600',target_queries=900,active_queries=600,
                completed=status.get('completed',0),total=600,watcher_pid=os.getpid(),
                source_supervisor_pid=launcher['supervisor_pid'],timestamp=stamp())
            write(scope.OLD/'metadata/execution_scope.json',state)
            print(json.dumps(state),flush=True)
            if status['state']=='failed':
                raise RuntimeError('pilot600 failed; preserving its data instead of extending an invalid run')
            if status['state']=='complete' and not alive(launcher['supervisor_pid']):
                migrate()
                return
            time.sleep(20)


if __name__=='__main__':
    try:
        run()
    except BaseException:
        target=scope.CANONICAL if scope.CANONICAL.exists() else scope.OLD
        write(target/'metadata/execution_scope.json',dict(state='failed',target_queries=900,
            error=traceback.format_exc(),watcher_pid=os.getpid()))
        raise
