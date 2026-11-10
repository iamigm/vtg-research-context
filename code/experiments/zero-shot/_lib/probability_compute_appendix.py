"""Reproduce query-level compute distributions from saved production records only."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())
PARENT = ROOT/'artifacts/runs/rollout/probability_budget/pilot/900'
COHORT = ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'
METHODS = ('levin', 'native_mass', 'numeric_novelty')
BUDGETS = (512, 1024, 2048)

def read(path):
    return json.loads(path.read_text())

def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)

def stats(pairs):
    assert pairs and all(math.isfinite(v) for _, v in pairs)
    values = [v for _, v in pairs]
    ordered = sorted(values)
    def quantile(p):
        at = (len(ordered)-1)*p
        lo = int(at)
        return ordered[lo]+(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(at-lo)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), variance=statistics.pvariance(values),
                std=statistics.pstdev(values), p90=quantile(.9), p95=quantile(.95),
                total=sum(values), first_key=pairs[0][0], first=values[0],
                min_key=min(pairs,key=lambda p:p[1])[0], max_key=max(pairs,key=lambda p:p[1])[0])

def fmt(x):
    return f'{x:,.3f}' if abs(x)<1e6 else f'{x:,.6g}'

def build():
    cohort = read(COHORT)
    parent = {r['sample_key']:read(PARENT/'queries'/f'{r["sample_key"]}.json') for r in cohort}
    child = {}
    for r in cohort:
        path = PARENT/'end_to_end/queries'/f'{r["sample_key"]}.json'
        if path.exists():
            d = read(path)
            if d.get('complete'): child[r['sample_key']] = d
    for count in (300,900):
        rows = cohort[:count]
        if not all(parent[r['sample_key']].get('complete') for r in rows): continue
        complete = all(r['sample_key'] in child for r in rows)
        result = ROOT/f'results/rollout/probability_budget/pilot/{count}'
        artifact = ROOT/f'artifacts/runs/rollout/probability_budget/pilot/{count}/compute_cost'
        raw_rows, summaries = [], []
        def record(group,method,budget,metric,unit,pairs):
            s = stats(pairs)
            summaries.append(dict(group=group,method=method,budget=budget,metric=metric,unit=unit,**s))
            raw_rows.extend(dict(sample_key=k,group=group,method=method,budget=budget,metric=metric,unit=unit,value=v) for k,v in pairs)
            return s
        def table(title, group, metrics, getters, conditions):
            lines.extend(['',f'## {title}',''])
            for metric,unit in metrics:
                lines.extend([f'### {metric} · {unit}','',
                    '| 方法・단계 | 예산 | n | 평균 | 중앙값 | 최소 | 최대 | 분산 | 표준편차 | P90 | P95 |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
                for method,budget in conditions:
                    pairs=[(r['sample_key'],float(getters[metric](r,method,budget))) for r in rows]
                    s=record(group,method,budget,metric,unit,pairs)
                    lines.append(f'| {method} | {budget or "—"} | {s["n"]} | '+' | '.join(fmt(s[k]) for k in ('mean','median','minimum','maximum','variance','std','p90','p95'))+' |')
        def pc(r): return parent[r['sample_key']]
        def cc(r): return child[r['sample_key']]
        def sc(r,m,b): return pc(r)['methods'][m]['snapshots'][str(b)]['cost']
        conditions=[(m,b) for b in BUDGETS for m in METHODS]
        lines=[f'# Compute cost · pilot{count}', '',
            f'고정 cohort {count} query의 샘플별 추가 연산 분포다. 후보 탐색은 {count}/{count} 완료. '+
            (f'후속처리도 {count}/{count} 완료하여 포함한다.' if complete else '후속처리는 전체 cohort가 완료되기 전이므로 이 appendix에서 제외한다.'), '',
            '이 문서는 같은 분석의 appendix이며 별도 실험이 아니다. pilot300은 pilot900의 정확한 prefix다. 두 범위를 독립 표본처럼 합산하지 않는다. '+
            '모든 값은 저장된 실측을 집계하며 새로운 추론이나 시간 추정을 하지 않는다.', '',
            '## 비용을 읽는 방법', '',
            '- 한 행의 n은 조건 반복 수가 아닌 query 수다. 각 query를 같은 가중치로 평균한다.',
            '- 평균 μ=Σx/n, 중앙값=정렬한 가운데 값(짝수이면 두 값 평균), 분산=Σ(x−μ)²/n인 population variance, 표준편차=√분산이다. 분산 단위는 표에 적힌 단위의 제곱이다.',
            '- 최소·최대는 실제 관측값이다. P90·P95는 정렬 위치 (n−1)p의 선형 보간 quantile이다. 비용 지표는 대체로 작을수록 좋으며, 분산은 샘플별 비용의 불균일함을 뜻한다.',
            '- 최초 값은 cohort 첫 query의 값이며 실행 startup의 독립 측정이 아니다. 최소·최대 query와 함께 문서 끝에 제시한다.',
            '- decoder inputs는 실제 모델에 넣은 position 수로 prefix replay를 포함한다. Forward 호출 수·token 수·CUDA event 시간·wall time은 서로 다른 지표이며 FLOPs로 환산하지 않는다.',
            '- 512·1024·2048은 한 탐색의 누적 checkpoint다. checkpoint 세 개를 합산하지 않는다. 서로 다른 방법은 별도로 실행했다.',
            '- search는 greedy 답변을 얻은 뒤 후보를 확장하는 추가 비용이다. shared video/query prefill은 search에 포함하지 않고 한 번만 센다. 같은 GPU의 동시 작업, CPU 준비, KV 관리 때문에 token 비율과 wall 비율은 다를 수 있다.', '',
            '## 핵심 비교', '',
            '아래 추가 wall은 후보 탐색만의 시간이다. 오른쪽 배수는 `(shared prefill + greedy + search) / (shared prefill + greedy)`의 합계 비율로, 기존 답변을 유지하고 후보를 더 만드는 실제 비교다. '+
            '아래의 샘플별 배수 평균과 달리 각 query의 분모 크기로 가중된 값이다.', '',
            '| 방법 | 예산 | 추가 search wall 평균 s | 중앙값 s | P95 s | 추가 decoder inputs 평균 | 전체 model wall 배수 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        baseline_sum=sum(pc(r)['input']['prefill_wall_seconds']+pc(r)['greedy']['cost']['wall_seconds'] for r in rows)
        for m,b in conditions:
            w=stats([(r['sample_key'],sc(r,m,b)['wall_seconds']) for r in rows])
            tokens=statistics.mean(sc(r,m,b)['decoder_inputs'] for r in rows)
            lines.append(f'| {m} | {b} | {fmt(w["mean"])} | {fmt(w["median"])} | {fmt(w["p95"])} | {fmt(tokens)} | {fmt(1+w["total"]/baseline_sum)}× |')
        getters={
            'shared prefill inputs':lambda r,m,b:pc(r)['input']['input_tokens'],
            'shared prefill wall':lambda r,m,b:pc(r)['input']['prefill_wall_seconds'],
            'input preparation wall':lambda r,m,b:pc(r)['input']['input_prepare_wall_seconds'],
            'greedy decoder inputs':lambda r,m,b:pc(r)['greedy']['cost']['decoder_inputs'],
            'greedy decode wall':lambda r,m,b:pc(r)['greedy']['cost']['wall_seconds'],
        }
        table('공유 입력과 greedy 기준 비용','baseline',[(k,'tokens' if 'inputs' in k else 's') for k in getters],getters,[('shared / greedy',None)])
        search_fields={'decoder inputs':('decoder_inputs','tokens'),'replay inputs':('replay_inputs','tokens'),
            'native forwards':('native_forwards','calls'),'search wall':('wall_seconds','s'),
            'decode CUDA':('decode_cuda_seconds','s'),'KV cache wall':('kv_cache_wall_seconds','s'),
            'reward wall':('reward_wall_seconds','s'),'scheduler wall':('scheduler_wall_seconds','s'),
            'checkpoint KV suffix cache':('kv_suffix_cache_bytes','GiB')}
        getters={label:(lambda r,m,b,k=field:sc(r,m,b)[k]/(2**30 if k.endswith('_bytes') else 1)) for label,(field,unit) in search_fields.items()}
        getters['KV store+restore CUDA']=lambda r,m,b:sc(r,m,b)['kv_store_cuda_seconds']+sc(r,m,b)['kv_restore_cuda_seconds']
        table('방법·예산별 후보 탐색의 추가 비용','search',[(label,unit) for label,(field,unit) in search_fields.items()]+[('KV store+restore CUDA','s')],getters,conditions)
        lines.extend(['','Checkpoint KV suffix cache는 해당 시점의 text KV 보유량이며 GPU peak memory나 전체 video KV 크기가 아니다. search wall에는 decode·KV·CPU scheduler 등이 함께 들어간다. 이 항목들을 search wall에 다시 더하지 않는다.'])
        def ratio(r,m,b):
            q=pc(r);base=q['input']['prefill_wall_seconds']+q['greedy']['cost']['wall_seconds']
            return 1+sc(r,m,b)['wall_seconds']/base
        table('query별 추가 비용과 greedy 대비 배수','ratio',
            [('search extra / greedy decode inputs','×'),('search+greedy+prefill / greedy+prefill wall','×'),('search replacement minus greedy wall','s')],
            {'search extra / greedy decode inputs':lambda r,m,b:sc(r,m,b)['decoder_inputs']/pc(r)['greedy']['cost']['decoder_inputs'],
             'search+greedy+prefill / greedy+prefill wall':ratio,
             'search replacement minus greedy wall':lambda r,m,b:sc(r,m,b)['wall_seconds']-pc(r)['greedy']['cost']['wall_seconds']},conditions)
        lines.extend(['','Replacement minus greedy는 greedy를 탐색으로 대체한다고 가정한 비용 차이이며 실제 실행에서 이미 계산한 greedy 비용을 환급한 값이 아니다. 음수이면 그 query에서 탐색 wall이 측정된 greedy wall보다 작았다는 뜻이다.'])
        if complete:
            def cropids(r,m,b):
                d=cc(r)
                return sorted({d['crop_assignment'][a['id']] for a in d['conditions'][f'{m}:{b}']['members']})
            def vcost(r,m,b,field):
                d=cc(r)
                return sum(d['verifications'][i]['cost'][field] for i in cropids(r,m,b))
            def gen(r,m,b,variant):
                d=cc(r);return d['generations'][d['conditions'][f'{m}:{b}'][variant]]['cost']
            getters={'candidate crops':lambda r,m,b:len(cropids(r,m,b)),
                     'verifier inputs':lambda r,m,b:vcost(r,m,b,'input_tokens'),
                     'verifier allocated model wall':lambda r,m,b:vcost(r,m,b,'amortized_wall_seconds')}
            for variant in ('proposal_only','reviewed'):
                for field in ('input_tokens','forward_calls','cuda_seconds','wall_seconds'):
                    getters[f'{variant} {field}']=lambda r,m,b,v=variant,k=field:gen(r,m,b,v)[k]
            getters['reviewed search+verify+final model wall']=lambda r,m,b:sc(r,m,b)['wall_seconds']+vcost(r,m,b,'amortized_wall_seconds')+cc(r)['phase_costs']['final_prefill']['wall_seconds']+gen(r,m,b,'reviewed')['wall_seconds']
            units=lambda k:'tokens' if 'inputs' in k or 'input_tokens' in k else 'calls' if 'forward_calls' in k else 'crops' if k=='candidate crops' else 's'
            lines.extend(['','## 후속처리 조건별 귀속 원칙','',
                '아래 verifier는 그 조건에서 제안한 고유 crop만 센다. 같은 crop은 한 query·조건 안에서 한 번 센다. 원래 후보에 없고 greedy 참조용으로만 추가한 crop은 제외한다. '+
                '입력 token은 해당 crop의 실측이고 시간은 batch 시간을 token 비율로 나눈 귀속값이다. 독립 실행 latency가 아니다. 최초 검증의 재실행·버린 batch는 조건별로 임의 배분하지 않고 다음 공유 실행 표에 포함한다.', '',
                'final generation은 해당 prompt의 실제 측정값을 각 조건에 참조한다. 같은 prompt를 공유한 조건을 더하면 중복 계산이므로 조건 행끼리 합산하지 않는다. '+
                'reviewed search+verify+final model wall은 후보 탐색+귀속 verifier wall+full-video final prefill 한 번+reviewed generation wall이다. '+
                '기존 greedy/prefill·CPU crop 준비·semantic 진단·검증 재실행은 제외하며, 이 합은 측정 단계의 회계 합계다. end-to-end 단독 latency로 해석하지 않는다.'])
            table('방법·예산별 검증과 최종 답변 생성','followup_attribution',[(k,units(k)) for k in getters],getters,conditions)
            stages=('verifier','instruct','embedding')
            def phase(r,kind,field):
                p=cc(r)['phase_costs'][kind]
                return sum(b[field] for b in p['batches']+p['validation'])
            getters={f'{kind} {field}':(lambda r,m,b,k=kind,f=field:phase(r,k,f)) for kind in stages for field in ('input_tokens','padded_tokens','forward_calls','cuda_seconds','wall_seconds')}
            for kind in stages:
                getters[f'{kind} CPU prepare']=lambda r,m,b,k=kind:cc(r)['phase_costs'][k]['cpu_prepare_seconds']
            getters.update({
                'all-condition exact intervals':lambda r,m,b:len(cc(r)['spans']),
                'all-condition unique crops':lambda r,m,b:len(cc(r)['crops']),
                'all-condition unique final prompts':lambda r,m,b:len(cc(r)['generations']),
                'all-condition unique final wall':lambda r,m,b:sum(g['cost']['wall_seconds'] for g in cc(r)['generations'].values()),
                'statistics wall':lambda r,m,b:cc(r)['phase_costs']['statistics_wall_seconds'],
                'final shared prefill wall':lambda r,m,b:cc(r)['phase_costs']['final_prefill']['wall_seconds'],
                'successful invocation postprocess wall':lambda r,m,b:cc(r)['postprocess_wall_seconds'],
            })
            def shared_unit(k):
                return 'tokens' if 'tokens' in k else 'calls' if 'forward_calls' in k else 'count' if any(s in k for s in ('intervals','crops','prompts')) else 's'
            lines.extend(['','## 共有 실행 총비용의 범위','',
                '다음 표는 세 방법×세 예산과 두 final variant의 후보 합집합을 실제 공유 실행한 query별 비용이다. 개별 방법·예산 비용이 아니며 조건별 표에 더하지 않는다. '+
                'Instruct visual mean과 last-token pooling은 같은 forward에서 얻으므로 Instruct 비용은 한 번만 센다. Embedding은 별도 모델 비용이다. '+
                'Semantic 계산은 진단 전용이며 final answer를 선택하는 데 필요하지 않다. Batches+validation은 최초 검증과 기록된 버린 batch까지 포함한다.'])
            table('실제 공유 후속처리 전체 작업량','shared_followup',[(k,shared_unit(k)) for k in getters],getters,[('all conditions shared',None)])
            lines.extend(['','중단된 query에서 checkpoint 전에 폐기된 semantic 작업은 CUDA 통계가 저장되지 않아 복원할 수 없다. successful invocation postprocess wall은 완료를 기록한 호출의 wall이며 과거 실패 호출·모델 초기 로딩·보고서 생성 시간을 포함하지 않는다. 모델별 checkpoint에서 재사용한 작업은 phase 비용에 남지만 호출 wall에는 다시 포함되지 않을 수 있다. 따라서 phase 합과 이 wall을 동일하다고 가정하지 않는다.'])
        lines.extend(['','## 최초 값과 극단값의 원자료 위치','',
            f'최초 query는 `{rows[0]["sample_key"]}`다. 모든 metric의 최초 값, 최소·최대 query ID 및 소수점 반올림 전 통계는 아래 JSON에 보존한다. 주요 wall의 위치는 다음과 같다.','',
            '| 단계 | 방법 | 예산 | 최초 값 s | 최소 query | 최대 query |', '|---|---|---:|---:|---|---|'])
        for s in summaries:
            if s['metric'] not in ('search wall','reviewed search+verify+final model wall','successful invocation postprocess wall'):continue
            lines.append(f'| {s["group"]} | {s["method"]} | {s["budget"] or "—"} | {fmt(s["first"])} | `{s["min_key"]}` | `{s["max_key"]}` |')
        out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(raw_rows[0]));writer.writeheader();writer.writerows(raw_rows)
        save(artifact/'statistics/query_costs.csv',out.getvalue())
        sources=[dict(sample_key=r['sample_key'],proposal_sha256=hashlib.sha256((PARENT/'queries'/f'{r["sample_key"]}.json').read_bytes()).hexdigest(),
            followup_sha256=hashlib.sha256((PARENT/'end_to_end/queries'/f'{r["sample_key"]}.json').read_bytes()).hexdigest() if complete else None) for r in rows]
        payload=dict(queries=count,followup_complete=complete,cohort_sha256=hashlib.sha256(COHORT.read_bytes()).hexdigest(),
                     aggregation='population variance ddof=0; equal-weight queries; linear quantiles; cumulative checkpoints',
                     sources=sources,statistics=summaries)
        save(artifact/'statistics/summary.json',json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
        csvlink=os.path.relpath(artifact/'statistics/query_costs.csv',result)
        jsonlink=os.path.relpath(artifact/'statistics/summary.json',result)
        lines.extend(['',f'[query별 scalar 비용 CSV]({csvlink}) · [전체 통계·원자료 SHA-256 JSON]({jsonlink})','',
            '재생성: `zero-shot/.venv/bin/python zero-shot/_lib/probability_compute_appendix.py` (experiments 폴더 기준). 원래 예측·탐색·cohort를 변경하지 않는다.'])
        save(result/'APPENDIX_COMPUTE.md','\n'.join(lines).replace('方法・단계','방법·단계').replace('共有 실행','공유 실행')+'\n')
    print('COMPUTE_APPENDIX_UPDATED',flush=True)

if __name__=='__main__':
    build()
