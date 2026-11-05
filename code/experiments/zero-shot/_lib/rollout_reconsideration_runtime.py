"""Integrate the completed600 prefix, run missing900 queries, and publish automatically."""
import argparse
import ast
import collections
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

import rollout_reconsideration as m

OLD = m.ROOT/'artifacts/runs'/m.FAMILY/'pilot/600'/m.ANALYSIS


def status(stage, **values):
    done=sum((m.OUT/'responses'/(s['sample_key']+'.json')).exists() for s in m.read(m.COHORT))
    value=dict(state='running',stage=stage,completed=done,total=900,percentage=100*done/900,timestamp=m.stamp(),**values)
    m.write(m.OUT/'metadata/status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def scientific_functions(path):
    tree=ast.parse(path.read_text())
    names={'Session','interval','iou','unique_candidates','diverse','clusters','select_probes','ledger_text','final_answer','run_variant','capture_probes'}
    return {node.name:ast.dump(node,include_attributes=False)for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names}


def references():
    cohort=m.read(m.COHORT)
    for n in (300,600,900):
        subset=cohort[:n]
        entries=[dict(sample_key=s['sample_key'],path=str(m.OUT/'responses'/(s['sample_key']+'.json')),sha256=m.sha(m.OUT/'responses'/(s['sample_key']+'.json')))for s in subset if (m.OUT/'responses'/(s['sample_key']+'.json')).exists()]
        dest=m.ROOT/'artifacts/runs'/m.FAMILY/'pilot'/str(n)/m.ANALYSIS/'metadata'
        m.write(dest/'source_records.json',dict(queries=n,completed=len(entries),cohort=str(m.ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json'),records=entries))
        m.write(dest/'cohort_reference.json',dict(path=str(m.COHORT),queries=n,indices=f'0:{n}',sha256=m.sha(m.COHORT),responses=str(m.OUT/'responses')))


def verify_reuse():
    reused=m.read(m.OUT/'metadata/reuse.json')
    for key,digest in reused['response_sha256'].items():
        assert m.sha(m.OUT/'responses'/(key+'.json'))==digest,key
    return len(reused['response_sha256'])


def prepare():
    cohort=m.read(m.COHORT)
    assert m.read(m.ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_600/samples.json')==cohort[:600]
    accepted=m.accepted_config_hashes()
    old_code=OLD/'code/rollout_reconsideration.py'
    assert scientific_functions(old_code)==scientific_functions(Path(m.__file__)),'Scientific inference functions changed'
    for category in ('metadata','validation'):(m.OUT/category).mkdir(parents=True,exist_ok=True)
    reuse=m.OUT/'metadata/reuse.json'
    if not reuse.exists():
        response_dir=OLD/'responses' if (OLD/'responses').exists() else m.OUT/'responses'
        hashes={}
        for s in cohort[:600]:
            p=response_dir/(s['sample_key']+'.json');r=m.read(p)
            assert r['config_hash'] in accepted
            assert r['sample']['sample_key']==s['sample_key']
            assert set(r['variants'])=={v[0]for v in m.VARIANTS}
            hashes[s['sample_key']]=m.sha(p)
        m.write(reuse,dict(queries=600,response_sha256=hashes,
            accepted_config_hashes=sorted(accepted),scientific_functions_identical=True,
            frozen_config_reference=str(OLD/'metadata/config.json'),
            scope='canonical900; original600 responses retained byte-for-byte; fixed prefixes use references'))
    # Move raw payloads; no symlinks or second copy of response/probability data.
    for category in ('responses','token_traces','inputs','logs','environment'):
        src=OLD/category;dst=m.OUT/category
        if src.exists():
            if not dst.exists():src.rename(dst)
            else:
                for p in src.iterdir():
                    assert not (dst/p.name).exists(),str(dst/p.name)
                    p.rename(dst/p.name)
                src.rmdir()
    assert verify_reuse()==600
    m.setup()
    references()
    # Preserve completed300/600 reports and retarget their existing interactive views.
    from rollout_reconsideration_report import build_view
    for n in (300,600):build_view(n)
    from rollout_method_docs import formalize
    (m.REPORT/'SPEC.md').write_text(formalize('', 'reconsideration',900,spec=True))
    if not (m.REPORT/'REPORT.md').exists():
        (m.REPORT/'REPORT.md').write_text('# Rollout reconsideration · pilot900\n\n상태: running. 목표900 query 중 기존600 query의 완료 결과를 재사용하고 나머지300 query를 수행한다. 완료된 pilot900 결과가 아니다.\n\n모델·입력·prompt·후보 pool·10개 조건·token distribution 진단은 기존 실행과 동일하다. 완료 후 성능·비용·paired bootstrap·viewer·검증을 자동 통합한다.\n\n[실험 설계](SPEC.md) · [누적 실행 상태](../../../../../../artifacts/runs/rollout/deterministic/pilot/900/reconsideration/metadata/status.json)\n')
    # Update old manifest links together with the raw directories.
    for path in (m.ROOT/'artifacts/common/registry').glob('*.json'):
        original=path.read_text();updated=original
        for category in ('responses','token_traces','inputs','logs','environment'):
            updated=updated.replace(str(OLD.relative_to(m.ROOT)/category),str(m.OUT.relative_to(m.ROOT)/category))
        if updated!=original:path.write_text(updated)
    overview=m.ROOT/'results'/m.FAMILY/'pilot/overview/reconsideration/REPORT.md'
    if overview.exists():
        text=overview.read_text()
        note='pilot900은 기존600 query 결과를 재사용하여 진행 중이다. [상태·설계](../../900/reconsideration/REPORT.md). 완료 전에는900 query 성능을 제시하지 않는다.'
        if note not in text:overview.write_text(text.rstrip()+'\n\n'+note+'\n')
    m.write(m.OUT/'validation/reuse.json',dict(passed=True,reused_queries=600,scientific_functions_identical=True,response_hashes_identical=True,canonical_storage_only=True))
    status('prepared',reused_completed=600,remaining=300)
    subprocess.run([sys.executable,str(m.ROOT.parent/'_lib/build_index.py')],check=True)
    print('PREPARE COMPLETE:600 preserved; missing300 ready.',flush=True)


def browser_check(n=900):
    root=m.ROOT.parent;runtime=root/'_shared/browser_runtime'
    chrome=Path('/home1/irteam/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell')
    env=dict(os.environ,LD_LIBRARY_PATH=str(runtime/'system/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH',''),FONTCONFIG_FILE=str(runtime/'fonts.conf'))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    key='activitynet-timelens_01723'
    url=f'http://127.0.0.1:8780/zero-shot/results/rollout/deterministic/pilot/{n}/reconsideration/viewer.html#sample='+key+'&review=agent_pool8'
    checks=[]
    for name,size in [('desktop','1440,1100'),('mobile','390,844')]:
        preview=m.ROOT/f'results/rollout/deterministic/pilot/{n}/reconsideration/preview_{name}.png'
        args=[str(chrome),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-proxy-server','--hide-scrollbars','--virtual-time-budget=8000','--window-size='+size,'--screenshot='+str(preview),'--dump-dom',url]
        result=subprocess.run(args,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
        assert result.returncode==0,result.stderr[-2000:]
        assert 'data-outcome="agent_pool8"' in result.stdout,'Review UI did not render'
        assert key in result.stdout
        checks.append(dict(viewport=name,review_rendered=True,sample=key))
    m.write(m.OUT/'validation/viewer_browser.json',dict(passed=True,queries=n,checks=checks))


def finish():
    stage={'value':'validating'};stop=threading.Event()
    def heartbeat():
        while not stop.wait(20):status(stage['value'])
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        assert verify_reuse()==600
        cohort=m.read(m.COHORT);accepted=m.accepted_config_hashes()
        for s in cohort:
            p=m.OUT/'responses'/(s['sample_key']+'.json');r=m.read(p)
            assert r['config_hash'] in accepted and set(r['variants'])=={v[0]for v in m.VARIANTS}
        references()
        from rollout_reconsideration_report import report
        for n in (300,600,900):
            stage['value']='reporting_pilot'+str(n);status(stage['value']);report(n,refresh=False)
        from rollout_reconsideration_report import bootstrap,summarize
        overview=m.ROOT/'results'/m.FAMILY/'pilot/overview/reconsideration/REPORT.md'
        lines=['# Rollout reconsideration · cumulative pilot300/600/900','','고정 prefix cohort이며 겹치는 query를 독립 합산하지 않는다. 각 범위는 canonical900 응답을 참조한다.','','| scope | query 수 | report |','|---|---:|---|']
        for n in (300,600,900):lines.append(f'| pilot{n} | {n} | [결과](../../{n}/reconsideration/REPORT.md) |')
        overview.write_text('\n'.join(lines)+'\n')
        subprocess.run([sys.executable,str(m.ROOT.parent/'_lib/build_index.py')],check=True)
        stage['value']='validating_viewer';status(stage['value'])
        browser_check()
        for n in (300,600,900):
            path=m.ROOT/f'results/rollout/deterministic/pilot/{n}/reconsideration/REPORT.md'
            for ref in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                parsed=urllib.parse.urlsplit(ref)
                if not parsed.scheme and parsed.path:assert (path.parent/urllib.parse.unquote(parsed.path)).resolve().exists(),ref
        for path in ('index.html','zero-shot/results/rollout/deterministic/viewer.html','zero-shot/results/rollout/deterministic/pilot/900/reconsideration/viewer.html'):
            with urllib.request.urlopen('http://127.0.0.1:8780/'+path,timeout=15) as response:assert response.status==200
        from artifact_store import register
        for label,kind,directory in [('responses','rollouts','responses'),('token_traces','token_traces','token_traces'),('metrics','analyses','metrics')]:
            register('deterministic.reconsideration.pilot900.'+label,kind,m.OUT/directory,
                'Canonical fixed900 reconsideration;300/600 prefixes reference the same stored outputs.',producer='rollout_reconsideration_runtime.py',cohort=str(m.COHORT))
        assert verify_reuse()==600
        # Remove stale launch state from smaller scopes; execution logs are unified in900.
        for n in (300,600):
            metadata=m.ROOT/f'artifacts/runs/rollout/deterministic/pilot/{n}/reconsideration/metadata'
            for name in ('launcher.json','supervisor_launcher.json','run.lock','worker0.json','worker1.json'):(metadata/name).unlink(missing_ok=True)
        m.write(m.OUT/'validation/completion.json',dict(passed=True,queries=900,variants=len(m.VARIANTS),reused600_unchanged=True,probability_vectors_verified_in_reports=True,scopes=[300,600,900],viewer=True,links=True))
    finally:stop.set()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:m.supervise()
