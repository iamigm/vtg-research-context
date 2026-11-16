"""CPU-only conditional top-30 entropy; saved source traces remain immutable."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen
from artifact_store import register
from result_layout import relocate_path

ROOT = Path('/home/irteam/experiments/zero-shot')
FAMILY = Path('uncertainty/token_analysis')
NAME = 'top30_renormalized'
ART = ROOT / 'artifacts/runs' / FAMILY
RESULT = ROOT / 'results' / FAMILY
COMMON = ROOT / 'artifacts/common'
JOB = ART / 'full/9404' / NAME


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def normalized(p):
    mass = math.fsum(p)
    assert mass > 0 and all(math.isfinite(v) and v >= 0 for v in p)
    q = [v / mass for v in p]
    return mass, q, -math.fsum(v * math.log(v) for v in q if v)


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in order[i:j]:
            result[k] = (i + j - 1) / 2
        i = j
    return result


def rho(x, y):
    a, b = ranks(x), ranks(y)
    return statistics.correlation(a, b) if len(set(a)) > 1 and len(set(b)) > 1 else None


def mean(values):
    return statistics.fmean(values) if values else None


def fmt(value):
    return 'NA' if value is None else f'{value:.6f}'


def analyze(scope, count):
    out = ART / scope / str(count) / NAME
    report = RESULT / scope / str(count) / NAME / 'REPORT.md'
    manifest = (COMMON / 'manifests/full_benchmark_v1.json' if scope == 'full' else
                COMMON / 'predictions/legacy/cohorts/pilot_300/samples.json')
    manifest_bytes = manifest.read_bytes()
    samples = json.loads(manifest_bytes)
    assert len(samples) == count == len({s['sample_key'] for s in samples})
    rows, sources = [], []
    errors = dict(normalized_sum=0., entropy_identity=0., stored_mass=0.)
    distributions = 0
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    step_path = out / 'metrics/per_step.csv.gz'
    started = last = time.monotonic()
    with gzip.open(step_path, 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','dataset','sample_key','step','mode','numeric','top30_mass',
                         'full_entropy_nats','top30_entropy_nats','full_minus_top30_nats'])
        for model in ('qwen25', 'qwen3'):
            for sample in samples:
                key, ds = sample['sample_key'], sample['dataset']
                path = (COMMON / 'token_traces/full_benchmark_v1' / model / ds / (key + '.json')
                        if scope == 'full' else relocate_path(RESULT / scope / str(count) / model / (key + '.json')))
                content = path.read_bytes()
                trace = json.loads(content)
                assert trace['sample'] == sample and trace['model_key'] == model
                assert trace['generated_ids'] == [t['token_id'] for t in trace['steps']]
                sources.append(dict(model=model,sample_key=key,path=str(path.relative_to(ROOT)),
                                    sha256=hashlib.sha256(content).hexdigest()))
                item = dict(model=model,dataset=ds,sample_key=key,iou=trace['iou'],
                            all_steps=len(trace['steps']),numeric_steps=0)
                buckets = {mode: {'all': [], 'numeric': []} for mode in ('raw','processed')}
                for t in trace['steps']:
                    numeric = bool(re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',t['token']))
                    item['numeric_steps'] += int(numeric)
                    for mode in buckets:
                        d = t[mode]
                        assert len(d['top30']) == len({v['id'] for v in d['top30']}) == 30
                        p = [v['probability'] for v in d['top30']]
                        mass, q, entropy = normalized(p)
                        full = d['entropy_nats']
                        assert math.isfinite(full) and full >= -1e-5
                        assert -1e-12 <= entropy <= math.log(30) + 1e-12
                        partial = -math.fsum(v * math.log(v) for v in p if v)
                        checks = dict(normalized_sum=abs(math.fsum(q)-1),
                            entropy_identity=abs(entropy-(partial/mass+math.log(mass))),
                            stored_mass=abs(mass-d['top30_mass']))
                        assert max(checks.values()) < 1e-10
                        for name,value in checks.items():
                            errors[name] = max(errors[name],value)
                        distributions += 1
                        value = (full,entropy,mass)
                        buckets[mode]['all'].append(value)
                        if numeric:
                            buckets[mode]['numeric'].append(value)
                        writer.writerow([model,ds,key,t['step'],mode,int(numeric),mass,full,entropy,full-entropy])
                for mode in buckets:
                    for subset in ('all','numeric'):
                        for i,metric in enumerate(('full_entropy','top30_entropy','top30_mass')):
                            item[f'{mode}_{subset}_{metric}'] = mean([v[i] for v in buckets[mode][subset]])
                rows.append(item)
                now = time.monotonic()
                if now-last >= 20 or len(rows) == 2*count:
                    state = dict(state='running',stage='analysis',scope=scope,queries=count,
                        completed_responses=len(rows),total_responses=2*count,percent=100*len(rows)/(2*count),
                        elapsed_seconds=now-started,eta_analysis_seconds=(now-started)*(2*count-len(rows))/len(rows))
                    save(JOB/'metadata/status.json',state)
                    print(json.dumps(state),flush=True)
                    last = now
    with (out/'metrics/per_query.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for model,ds in sorted({(r['model'],r['dataset']) for r in rows}):
        base = [r for r in rows if (r['model'],r['dataset']) == (model,ds)]
        for mode in ('raw','processed'):
            for subset in ('numeric','all'):
                prefix = f'{mode}_{subset}_'
                rs = [r for r in base if r[prefix+'top30_entropy'] is not None]
                old = [r[prefix+'full_entropy'] for r in rs]
                new = [r[prefix+'top30_entropy'] for r in rs]
                ious = [r['iou'] for r in rs]
                groups.append(dict(model=model,dataset=ds,mode=mode,subset=subset,
                    eligible_queries=len(rs),excluded_queries=len(base)-len(rs),
                    mean_full_entropy=mean(old),mean_top30_entropy=mean(new),
                    mean_full_minus_top30=mean([a-b for a,b in zip(old,new)]),
                    mean_top30_mass=mean([r[prefix+'top30_mass'] for r in rs]),
                    spearman_iou_full=rho(ious,old),spearman_iou_top30=rho(ious,new),
                    success_top30_entropy=mean([v for v,iou in zip(new,ious) if iou>=.5]),
                    failure_top30_entropy=mean([v for v,iou in zip(new,ious) if iou<.5])))
    save(out/'metrics/summary.json',dict(scope=scope,queries=count,responses=len(rows),groups=groups))
    save(out/'metadata/sources.json',dict(manifest=str(manifest.relative_to(ROOT)),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),records=sources,
        producer=Path(__file__).name,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    save(out/'validation/numerical.json',dict(passed=True,queries=count,responses=len(rows),
        distributions=distributions,max_errors=errors,source_traces_read_only=True,
        entropy_bounds_nats=[0,math.log(30)]))
    lines = [f'# Top-30 renormalized entropy · {scope} · {count:,} queries','',
        f'기존 {count:,} query × 2개 모델 = {len(rows):,}개 응답의 저장된 token distribution을 재집계했다. '
        '원래 응답·IoU·전체 vocabulary entropy를 유지하며 GPU 추론은 실행하지 않았다.','',
        '## 지표 정의','',
        '각 생성 step에서 S는 해당 mode의 확률 상위 30개 token, M = Σ(i∈S) pᵢ다. '
        'qᵢ = pᵢ/M으로 재정규화하고 H₃₀ = −Σ(i∈S) qᵢ ln qᵢ를 계산했다. '
        '단위는 nat이며 0 ≤ H₃₀ ≤ ln(30) ≈ 3.401197이다. 높을수록 top-30 내부 분포가 고르다. '
        'Hfull은 원자료의 전체 vocabulary entropy다. ΔH = Hfull − H₃₀이며 재정규화도 포함되므로 '
        '이 차이를 top-30 밖의 entropy 자체로 해석하지 않는다.','',
        '주 분석은 raw numeric step이다. 숫자 token 정의는 기존 entropy–IoU 분석과 같은 '
        '`\\s*\\d+(?:\\.\\d+)?\\s*`이며 소수부도 포함한다. 먼저 query 내 해당 step의 평균을 구한 뒤 '
        '각 query에 같은 가중치를 부여해 모델·dataset별로 평균한다. 숫자 step이 없는 응답은 numeric 분석에서 제외하고 수를 명시한다. '
        'all은 EOS를 포함한 모든 생성 step이다. raw는 generation processor 적용 전, processed는 적용 후이며 '
        '각각 자신의 top-30을 재정규화한다.','',
        'Spearman ρ는 query별 평균 entropy와 IoU의 순위 상관이다. 음수면 IoU가 높을수록 entropy가 낮은 경향이다. '
        '동률에는 평균 순위를 적용했다. 성공/실패는 IoU≥0.5 / IoU<0.5다. '
        '이번 비교는 기술 통계이며 신뢰구간·유의성 검정을 수행하지 않았다.','',
        '## Raw numeric 결과','',
        '| 모델 | dataset | 유효/제외 query | Hfull (nat) | H₃₀ (nat) | ΔH (nat) | M | ρ full | ρ top-30 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['mode']=='raw' and g['subset']=='numeric':
            lines.append(f"| {g['model']} | {g['dataset']} | {g['eligible_queries']}/{g['excluded_queries']} | "+
                ' | '.join(fmt(g[k]) for k in ('mean_full_entropy','mean_top30_entropy','mean_full_minus_top30',
                    'mean_top30_mass','spearman_iou_full','spearman_iou_top30'))+' |')
    lines += ['','## 해석과 범위','',
        'H₃₀는 “다음 token이 top-30 안에 있다”는 조건부 entropy다. top-30 밖의 확률 질량은 제외되므로 '
        'M을 함께 확인해야 한다. 저장된 greedy trajectory에 대한 사후 분석이며 생성 정책은 변경하지 않는다. '
        'pilot과 full은 각각 집계하며 겹치는 query를 독립 표본으로 합산하지 않는다. '
        '원자료의 FP32 오차 때문에 M이 1을 미세하게 넘을 수 있으며 clipping 없이 실제 합으로 나눴다.','',
        '## Raw/processed · numeric/all 보조 결과','',
        '| 모델 | dataset | mode | step 범위 | query 수 | Hfull | H₃₀ | 성공 H₃₀ | 실패 H₃₀ |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['model']} | {g['dataset']} | {g['mode']} | {g['subset']} | {g['eligible_queries']} | "+
            ' | '.join(fmt(g[k]) for k in ('mean_full_entropy','mean_top30_entropy',
                'success_top30_entropy','failure_top30_entropy'))+' |')
    lines += ['',f'전체 {distributions:,}개 분포에서 Σq=1, entropy 범위, '
        'H₃₀ = −Σ(pᵢ ln pᵢ)/M + ln M 항등식과 저장 mass 일치를 검증했다.','',
        '원래 top-30 ID·확률은 복제하지 않는다. sources가 가리키는 원본에서 qᵢ=pᵢ/M으로 재구성할 수 있다.']
    for label,p in [('query별 지표',out/'metrics/per_query.csv'),('step별 지표 (gzip CSV)',step_path),
        ('집계 JSON',out/'metrics/summary.json'),('원본 경로·SHA-256',out/'metadata/sources.json'),
        ('수치 검증',out/'validation/numerical.json')]:
        lines += ['',f'[{label}]({os.path.relpath(p,report.parent)})']
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text('\n'.join(lines)+'\n')
    register(f'analysis.{scope}.top30_renormalized','analyses',out,
        f'{scope} {count} queries: top-30 conditional entropy from immutable traces',
        Path(__file__).name,f'{scope}{count}','complete')
    return dict(scope=scope,queries=count,responses=len(rows),report=str(report))


def run():
    try:
        assert abs(normalized([1.]*30)[2]-math.log(30))<1e-12
        assert normalized([.3]+[0.]*29)[2]==0
        assert abs(normalized([.2,.1])[2]-(math.log(3)-2*math.log(2)/3))<1e-12
        assert abs(rho([1,2,2,4],[4,2,2,1])+1)<1e-12
        scopes = [analyze('pilot',300),analyze('full',9404)]
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
        subprocess.run([sys.executable,str(ROOT.parent/'_lib/refresh_result_servers.py')],check=True)
        manifest = json.loads((ART/'overview/web/view_manifest.json').read_text())
        assert json.dumps(manifest).count(NAME)>=2
        for s in scopes:
            path = Path(s['report'])
            with urlopen('http://127.0.0.1:8780/'+str(path.relative_to(ROOT.parent)),timeout=15) as response:
                assert response.status==200 and 'H₃₀' in response.read().decode()
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                assert (path.parent/target).exists(),target
        save(JOB/'validation/publication.json',dict(passed=True,scopes=scopes,
            index_http_status=200,reports_http_status=200,manifest_entries_present=True))
        save(JOB/'metadata/status.json',dict(state='complete',scopes=scopes))
        print('COMPLETE',json.dumps(scopes),flush=True)
    except Exception:
        save(JOB/'metadata/status.json',dict(state='failed',error=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['launch','run'])
    args = parser.parse_args()
    if args.command=='run':
        run()
    else:
        log = JOB/'logs/run.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            process = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=process.pid,log=str(log),status=str(JOB/'metadata/status.json'))))
