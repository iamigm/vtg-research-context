"""Keep old and optimized execution timings separate; compare identical trace work."""
import temporal_mode_capped_cost_report as original
from temporal_mode_prefix_cache import *

_work=original.standalone_work
_summary=original.summarize
_definitions=original.definitions
_tables=original.tables
_rows=original.cost_rows


def work(*args,**kwargs):
    result=prefix_standalone_work(*args,**kwargs)
    result['single_kv_reference']=_work(*args,**kwargs)
    return result


def cost_rows():
    rows=_rows()
    path=OUT/'metadata/execution_interruptions.json'
    interrupted={entry['sample_key'] for entry in read(path)['affected']} if path.exists() else set()
    for row in rows:
        if row['key'] in interrupted and row['observed'] and row['observed'].get('execution_backend','single_kv_replay')!=BACKEND:
            row['observed']=None
            row['observed_exclusion']='administrative interruption during MPS benchmark; original measurement retained in responses'
    return rows


def definitions():
    return _definitions()+'''

### Exact prefix KV cache

생성의 input·token distribution·RNG·draw stream·mode·judge prompt는 유지한다. 한 query 안에서 이미 계산한 prefix의 CPU logits와 token별 KV를 재사용한다. 영상 KV는 한 번만 별도 참조하고 text KV는 token당 한 개를 저장하며 최대4,096 prefix로 제한한다. cache가 가득 차면 추가 prefix는 기존 native replay를 사용한다. judge text prefill과 관측 근거 생성은 기존 경로를 유지한다.

현재 independent trace work는 이 prefix cache를 fresh 상태로 시작했을 때의 native forward 횟수다. 과거 query에도 같은 trace의 최적화된 연산 횟수를 재구성한다. 기존 single-KV replay 횟수는 `single_kv_reference`로 함께 보존하며 같은 query·method·seed·예산에서 비교한다. cache 저장·복원 kernel은 decoder forward가 아니므로 decoder inputs/attention pairs에는 더하지 않고 별도 CUDA 시간과 memory로 실측한다.

실측은 `single_kv_replay`와 `prefix_kv_cache`를 구분한다. 두 구현의 서로 다른 query 평균으로 가속률을 계산하지 않는다. 기본 실측 표는 prefix cache의 모든24개 checkpoint가 완료된 공통 query만 사용한다. 기존 구현의 실측은 아래 별도 표와 원자료에 남긴다. CUDA forward 시간에 cache copy 시간이 포함되지 않으므로 cache CUDA 시간도 함께 읽는다. 본문 peak memory에는 prefix cache를 포함한다. 같은 입력으로 두 구현을 실행한 calibration의 가속률은 제한된3 query 결과이며 전체 pilot 가속률을 보장하지 않는다.

worker3개의 CUDA 실행은 이 실험 전용 pipe의 NVIDIA MPS를 사용한다. 서로 다른 process의 kernel 실행을 겹칠 수 있는 실행 서비스이며 모델·입력·dtype·rollout 정책은 바꾸지 않는다. [NVIDIA MPS 설명](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html). 동일3 client·8,192 context·client당80 native single-token forward의 microbenchmark는 실험 전체 가속률과 구분한다. MPS 전후 전체 logits byte hash와 세 데이터셋의 기존 분포·경로·judge 기록을 대조한다. 기존 kernel replay 비용과 새 MPS+prefix cache 실측은 합산하지 않는다.

paired microbenchmark 중 일시 정지된 기존 worker의3 query는 예측·trace를 유지하되 기존 구현의 wall/CUDA 평균에서 제외한다. 중단 시간과 query는 metadata/execution_interruptions.json에 기록하고 원래 시간 계측은 responses에 보존한다. 행정적 대기 시간을 탐색 연산 시간으로 해석하지 않는다.
'''


def summarize(rows):
    result=_summary(rows)
    protocols={}
    for backend in ('single_kv_replay',BACKEND):
        selected=[row for row in rows if row['observed'] and row['observed'].get('execution_backend','single_kv_replay')==backend]
        protocols[backend]=_summary(selected)
    accelerated=protocols[BACKEND]
    accelerated_groups={(g['method'],g['budget']):g for g in accelerated['groups']}
    common=set(result['common_keys'])
    accelerated_common=set(accelerated['common_keys'])
    for group in result['groups']:
        measured=accelerated_groups[group['method'],group['budget']]
        group['observed']=measured.get('observed',{})
        group['measured_queries']=measured.get('measured_queries',0)
        subset=[row for row in rows if row['key'] in common and row['method']==group['method'] and row['budget']==group['budget']]
        if subset:
            group['single_kv_reference']={field:original.avg(subset,lambda r:r['standalone']['single_kv_reference']['research'][field])
                for field in ('forwards','decoder_inputs','attention_pairs')}
            group['native_forward_reduction']=1-group['research']['forwards']/max(1,group['single_kv_reference']['forwards'])
        subset=[row for row in subset if row['key'] in accelerated_common]
        if subset:
            obs=group['observed']
            obs['prefix_cache_cuda_seconds']={phase:original.avg(subset,lambda r:r['observed']['prefix_cache']['phases'][phase]['prefix_cache_cuda_seconds'])
                for phase in ('main','tail','judge')}
            obs['prefix_cache_cuda_seconds']['sweep']=original.avg(subset,lambda r:r['observed']['prefix_cache']['sweep']['prefix_cache_cuda_seconds'])
            obs['prefix_entries']=original.avg(subset,lambda r:r['observed']['prefix_cache']['entries'])
            obs['prefix_gpu_mib']=original.avg(subset,lambda r:r['observed']['prefix_cache']['gpu_bytes']/2**20)
            obs['prefix_cpu_logits_mib']=original.avg(subset,lambda r:r['observed']['prefix_cache']['cpu_logit_bytes']/2**20)
    result['execution_backend']=BACKEND
    result['observed_protocols']=protocols
    result['measured_common_queries']=accelerated['measured_common_queries']
    result['measured_common_keys']=accelerated['measured_common_keys']
    return result


def tables(summary):
    text=_tables(summary)
    lines=['### Prefix cache 비용과 구현별 실측','',
        '| 총 rollout | 방법 | 동일 trace native forward: replay → prefix cache | 감소율 | Prefix cache 공통 n | Cache CUDA s | Prefix KV MiB / CPU logits MiB |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for group in summary['groups']:
        if not group['queries']:
            continue
        obs=group.get('observed',{})
        cache_seconds=obs.get('prefix_cache_cuda_seconds',{}).get('sweep')
        cache=f'{cache_seconds:.4f}' if cache_seconds is not None else '—'
        memory=f'{obs["prefix_gpu_mib"]:.1f} / {obs["prefix_cpu_logits_mib"]:.1f}' if 'prefix_gpu_mib' in obs else '—'
        lines.append(f'| {group["total_rollout_budget"]} | {group["method"]} | '
            f'{group["single_kv_reference"]["forwards"]:.1f} → {group["research"]["forwards"]:.1f} | '
            f'{100*group["native_forward_reduction"]:.1f}% | {group["measured_queries"]} | {cache} | {memory} |')
    old=summary['observed_protocols']['single_kv_replay']
    lines+=['',f'기존 single-KV replay의 실측 공통 query는{old["common_queries"]}개이며 prefix cache 표와 합산하지 않는다.','',
        '| 총 rollout | 방법 | 기존 공통 n | 기존 sweep native CUDA / wall s | 기존 native forwards |',
        '|---:|---|---:|---:|---:|']
    for group in old['groups']:
        obs=group.get('observed')
        if obs:
            lines.append(f'| {group["total_rollout_budget"]} | {group["method"]} | {group["queries"]} | '
                f'{obs["sweep"]["cuda_forward_seconds"]:.3f} / {obs["sweep"]["wall_seconds"]:.3f} | {obs["sweep"]["forwards"]:.1f} |')
    calibration=OUT/'validation/prefix_cache_gpu.json'
    if calibration.exists():
        lines+=['','### 같은 입력의 paired calibration','',
            '| Query | Dataset | Native forwards: replay → prefix cache | Wall speedup | 분포·경로·judge 일치 |',
            '|---|---|---:|---:|---|']
        for record in read(calibration)['cases']:
            lines.append(f'| {record["sample_key"]} | {record["dataset"]} | '
                f'{record["native"]["sweep"]["forwards"]} → {record["optimized"]["sweep"]["forwards"]} | '
                f'{record["wall_speedup"]:.2f}× | {record["passed"]} |')
    benchmark=OUT/'validation/mps_benchmark.json'
    if benchmark.exists():
        test=read(benchmark)
        lines+=['','### MPS paired microbenchmark','',
            f'동일 native decode,context{test["context_tokens"]:,} tokens,client{test["clients"]}개. '
            f'기존{test["native"]["tokens_per_second"]:.2f} forwards/s → MPS{test["mps"]["tokens_per_second"]:.2f} forwards/s '
            f'({test["speedup"]:.2f}×). 전체 logits hash 일치={test["logits_bitwise_equal"]}. '
            '영상 처리와 main/tail scheduling을 포함한 전체 pilot 처리량이 아닌 제한된 microbenchmark다.']
    return text+'\n'+'\n'.join(lines)+'\n\n'


original.standalone_work=work
original.cost_rows=cost_rows
original.summarize=summarize
original.definitions=definitions
original.tables=tables


def report():
    original.report()


def initialize():
    original.initialize()
    path=RESULT/'detail.html'
    html=path.read_text().replace('<h2>성능과 연산 비용</h2>',
        '<h2>성능과 연산 비용</h2><p class="muted">Independent trace work는 최대4,096개의 exact prefix KV cache를 적용한 native forward 기준입니다. Cache 저장·복원 비용은 별도 CUDA 시간·memory로 측정하고 보고서에 함께 표시합니다. 기존 single-KV 구현의 실제 시간과 구분합니다.</p>')
    path.write_text(html)
    script=OUT/'web/viewer.js'
    script.write_text(script.read_text()+'''\nconst renderBeforePrefixCost=render;render=function(){renderBeforePrefixCost();const cost=data?.results?.[$('method').value+':'+$('seed').value]?.checkpoints?.[$('budget').value]?.cost;const output=document.getElementById('case-cost');if(output&&cost?.prefix_cache){output.textContent+=` Prefix cache: ${cost.prefix_cache.entries}/${cost.prefix_cache.limit}, KV ${(cost.prefix_cache.gpu_bytes/2**20).toFixed(1)}MiB, 저장·복원 CUDA ${cost.prefix_cache.sweep.prefix_cache_cuda_seconds.toFixed(4)}s.`;}};\n''')


if __name__=='__main__':
    import sys
    initialize() if '--initialize' in sys.argv else report()
