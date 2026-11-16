"""Within-query distinct-interval overlap; CPU-only appendix from frozen leaves."""
import argparse, hashlib, json, os, sys
from textwrap import dedent
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[2]/'_shared/timelens/.venv/lib/python3.10/site-packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from probability_budget_core import ROOT, OUT, COHORT, METHODS, BUDGETS, read, write, sha, stamp

SAMPLE='charades-timelens_00810'
COLORS=['#2368b5','#da7900','#7954a1']

def matrix(intervals):
    a=np.asarray(intervals,dtype=float).reshape(-1,2)
    lo=np.maximum(a[:,None,0],a[None,:,0]);hi=np.minimum(a[:,None,1],a[None,:,1])
    intersection=np.maximum(0,hi-lo)
    union=(a[:,1]-a[:,0])[:,None]+(a[:,1]-a[:,0])[None,:]-intersection
    return intersection/union

def statistics(intervals):
    n=len(intervals);x=matrix(intervals);v=x[np.triu_indices(n,1)]
    return dict(intervals=n,pairs=len(v),mean=float(v.mean()) if len(v) else None,
        median=float(np.median(v)) if len(v) else None,
        zero_fraction=float((v==0).mean()) if len(v) else None,
        half_fraction=float((v>=.5).mean()) if len(v) else None,
        near_duplicate_fraction=float((v>=.9).mean()) if len(v) else None)

def selfcheck():
    x=matrix([[0,2],[1,3],[4,5]])
    np.testing.assert_allclose(x,[[1,1/3,0],[1/3,1,0],[0,0,1]])
    assert statistics([[0,2]])['mean'] is None
    assert statistics([])['pairs']==0
    assert statistics([[0,2],[1,3],[4,5]])['pairs']==3

def collect():
    rows=read(COHORT);stats=[];examples={};sources=[]
    for index,row in enumerate(rows):
        key=row['sample_key'];query=read(OUT/'queries'/f'{key}.json')
        assert query['complete']
        for method in METHODS:
            path=OUT/'search'/key/f'{method}.json';raw_bytes=path.read_bytes();raw=json.loads(raw_bytes)
            sources.append(dict(sample_key=key,method=method,path=str(path),sha256=hashlib.sha256(raw_bytes).hexdigest()))
            previous=set()
            for budget in BUDGETS:
                snapshot=raw['snapshots'][str(budget)]
                leaves=raw['leaves'][:snapshot['leaf_count']]
                # Exact clipped interval atoms, matching interval_distribution.
                intervals=sorted({tuple(leaf['span']) for leaf in leaves if leaf['valid'] and leaf['logp']>-740})
                assert all(0<=a<b<=query['input']['duration']+1e-5 for a,b in intervals)
                assert previous.issubset(intervals);previous=set(intervals)
                assert len(intervals)==snapshot['unique_intervals']==query['methods'][method]['snapshots'][str(budget)]['unique_intervals']
                result=dict(sample_key=key,cohort_index=index,dataset=row['dataset'],method=method,budget=budget,
                    completed_paths=len(leaves),valid_paths=sum(bool(leaf['valid']) for leaf in leaves),**statistics(intervals))
                stats.append(result)
                if key==SAMPLE:examples[method,budget]=intervals
        if (index+1)%100==0:print('OVERLAP',index+1,len(rows),flush=True)
    write(OUT/'analyses/interval_overlap.json',dict(scope='pilot',queries=900,definition='unordered distinct-interval pairs within each query; equal query weighting in aggregate',rows=stats))
    write(OUT/'validation/interval_overlap.json',dict(passed=True,queries=900,conditions=9,source_files=len(sources),
        exact_unique_counts=True,cumulative_candidate_subset=True,no_self_pairs=True,no_cross_query_pairs=True,
        cohort_sha256=sha(COHORT),sources=sources,timestamp=stamp(),
        dependencies=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
        implementation_sha256=sha(__file__)))
    return rows,stats,examples

def figures(artifact,stats,examples,row):
    media=artifact/'media';media.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,layout='constrained')
    for ax,method in zip(axes,METHODS):
        for budget,color in zip(BUDGETS,COLORS):
            v=np.sort([x['mean'] for x in stats if x['method']==method and x['budget']==budget and x['mean'] is not None])
            ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'{budget} (n={len(v)})',color=color)
        ax.set(title=method,xlabel='Mean pairwise temporal IoU per query',xlim=(0,1),ylim=(0,1));ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Fraction of evaluable queries');fig.suptitle('Within-query overlap | each query has equal weight')
    fig.savefig(media/'interval_overlap_distribution.svg');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,11),layout='constrained',sharex=True)
    for i,method in enumerate(METHODS):
        for j,budget in enumerate(BUDGETS):
            a=examples[method,budget];ax=axes[i,j]
            lines=[[(lo,k),(hi,k)] for k,(lo,hi) in enumerate(a)]
            ax.add_collection(LineCollection(lines,colors=COLORS[j],linewidths=1.1,alpha=.8))
            ax.set(xlim=(0,row['duration']),ylim=(len(a),-1),title=f'{method} | {budget} | K={len(a)}')
            if i==2:ax.set_xlabel('Video time (seconds)')
            if j==0:ax.set_ylabel('Interval index, sorted by (start, end)')
    fig.suptitle(f'{SAMPLE} | all distinct intervals, no subsampling')
    fig.savefig(media/'interval_overlap_timeline.svg');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained')
    palette=plt.colormaps['viridis'].copy();palette.set_bad('#d8dadd')
    for i,method in enumerate(METHODS):
        for j,budget in enumerate(BUDGETS):
            a=examples[method,budget];m=matrix(a);np.fill_diagonal(m,np.nan)
            im=axes[i,j].imshow(m,vmin=0,vmax=1,cmap=palette,interpolation='nearest')
            axes[i,j].set(title=f'{method} | {budget} | mean={statistics(a)["mean"]:.3f}',xlabel='Interval index',ylabel='Interval index')
    fig.colorbar(im,ax=axes,label='Temporal IoU (diagonal excluded)',shrink=.65)
    fig.suptitle(f'{SAMPLE} | pairwise overlap | same ordering as timeline')
    fig.savefig(media/'interval_overlap_heatmap.svg');plt.close(fig)

def number(x):return '—' if x is None else f'{x:.3f}'

def appendix(count,rows,all_stats,examples,render_figures=True):
    stats=[x for x in all_stats if x['cohort_index']<count]
    result=ROOT/f'results/rollout/probability_budget/pilot/{count}'
    artifact=ROOT/f'artifacts/runs/rollout/probability_budget/pilot/{count}'
    example=dict(next(r for r in rows if r['sample_key']==SAMPLE))
    example['duration']=read(OUT/'queries'/f'{SAMPLE}.json')['input']['duration']
    example_in_scope=any(r['sample_key']==SAMPLE for r in rows[:count])
    if render_figures:figures(artifact,stats,examples,example)
    def link(name):return os.path.relpath(artifact/'media'/name,result)
    def block(text):return dedent(text).strip().splitlines()
    def aggregate(method,budget,key):
        values=[x[key] for x in stats if x['method']==method and x['budget']==budget and x[key] is not None]
        return float(np.mean(values)) if values else None
    example_stats={(method,budget):statistics(examples[method,budget]) for method in METHODS for budget in BUDGETS}
    levin=[example_stats['levin',b] for b in BUDGETS]
    lines=block(f"""
        # 후보 구간의 중복 · pilot{count}

        **후보를 많이 만들었을 때, 서로 다른 시간대를 찾은 것인지 비슷한 구간을 반복한 것인지 살펴본다.** 같은 영상·질문에 대한 후보 100개가 모두 10–20초 주변에 몰려 있다면, 후보 수만으로는 탐색의 폭을 알기 어렵다. 이 appendix는 각 query 안에서 서로 다른 후보 구간을 두 개씩 비교해 그 겹침을 측정한다.

        분석 범위는 pilot{count}의 {count} query × 세 방법 × 512·1024·2048 예산이다. 먼저 한 query의 구간 배치와 heatmap을 읽고, 뒤에서 전체 query의 분포와 요약 수치를 확인한다. 시간적 다양성을 보는 자료이며, 정답을 더 잘 찾았는지는 본 보고서의 candidate oracle과 함께 판단한다.

        ## 1. 후보들은 영상의 어디를 가리키는가

        사례: `{SAMPLE}` · Query: **{example['query']}**

        앞선 영상 분석에 사용한 고정 사례로, 겹침 정도를 기준으로 새로 고르지 않았다. 현재 pilot{count} cohort 포함: {'예' if example_in_scope else '아니오 — 사례 그림만 참조하며 전체 집계에는 포함하지 않음'}.

        **가로선 하나가 후보 구간 하나다.** 선의 왼쪽은 시작, 오른쪽은 끝이며, 가로축은 영상의 실제 시간(초)이다. 각 패널에 모든 후보를 표시했다. 세 열은 예산 512·1024·2048, 세 행은 탐색 방법이다. 예산은 사용할 수 있는 decoder 계산량의 상한이며 후보 개수가 아니다.

        ![한 query의 모든 후보를 실제 시간축에 배치한 그림]({link('interval_overlap_timeline.svg')})

        선들의 시작과 끝이 비슷한 위치에 겹쳐 있으면 같은 시간대 주변의 후보가 많다는 뜻이다. 선들이 여러 시간대로 나뉘면 서로 다른 위치를 탐색한 것이다. 세 열의 시간축은 동일하므로 예산이 늘면서 어느 위치의 후보가 추가되는지 비교할 수 있다.

        세로축은 `(start,end)` 순서로 정렬한 후보 번호다. 확률 순위가 아니며 패널마다 다시 정렬하므로, 서로 다른 예산에서 같은 번호가 같은 후보를 뜻하지는 않는다. 패널의 K는 완전히 같은 구간을 합치고 남은 고유 구간 수다.

        이 사례의 levin은 K가 **{' → '.join(str(x['intervals']) for x in levin)}개**다. 1024와 2048의 후보 집합은 같다. 예산은 상한이므로 조기 종료한 탐색에 더 큰 상한을 주어도 후보가 추가되지 않을 수 있다. 예산과 방법의 세부 의미는 마지막 절에서 설명한다.

        ## 2. 어느 후보들이 서로 비슷한가

        두 구간의 겹침은 **temporal IoU = 겹치는 시간의 길이 ÷ 두 구간의 합집합 길이**로 나타낸다. 0이면 겹치는 시간이 없고, 1이면 완전히 같은 구간이다. 아래는 계산을 이해하기 위한 가상 예시다.

        | 후보 A | 후보 B | temporal IoU | 읽는 법 |
        |---|---|---:|---|
        | 0–10초 | 0–10초 | 1.00 | 완전히 같다. 실제 분석에서는 먼저 하나로 합친다. |
        | 0–10초 | 2–12초 | 0.67 | 8초가 겹치고 합집합은 12초다. |
        | 0–10초 | 8–18초 | 0.11 | 경계 부근에서 조금 겹친다. |
        | 0–10초 | 20–30초 | 0.00 | 겹치는 시간이 없다. |

        **Heatmap의 한 칸은 후보 두 개의 IoU다.** 행과 열 모두 위 구간 배치의 후보 번호를 사용한다. 노란색에 가까울수록 많이 겹치고, 보라색에 가까울수록 적게 겹친다. 모든 패널의 색상 범위는 0–1로 같다. 회색 대각선은 자기 자신과의 비교이므로 제외한 칸이며, IoU가 0이라는 뜻이 아니다.

        ![같은 query 안의 후보 두 개씩을 비교한 temporal IoU heatmap]({link('interval_overlap_heatmap.svg')})

        **밝은 사각형 덩어리는 서로 비슷한 시간대를 가리키는 후보 묶음이다.** 두 밝은 덩어리 사이가 어두우면 그 두 묶음은 서로 덜 겹친다. 현재 사례에서도 여러 밝은 덩어리와 그 사이의 어두운 영역이 보인다. 따라서 전체 평균만 보기보다, 구간 배치와 함께 어떤 시간대의 후보가 반복되는지 살펴보는 것이 유용하다. 이 패턴은 시간적 묶음을 보여주며 서로 다른 사건이나 decoding mode를 직접 판정한 결과는 아니다.

        ## 3. 이 사례의 겹침을 숫자로 읽기

        K개 후보에서 서로 다른 두 개를 고르는 pair 수는 `K(K−1)/2`다. 예를 들어 이 사례의 levin·512는 {levin[0]['intervals']}개 후보로 {levin[0]['pairs']:,}개 pair를 비교한다. A–B와 B–A는 한 번만 세고, 자기 비교는 제외한다.

        | 방법 | 예산 | 구간 K | 서로 다른 pair | 평균 IoU | 중앙 IoU | IoU≥0.9 비율 |
        |---|---:|---:|---:|---:|---:|---:|
    """)
    for method in METHODS:
        for budget in BUDGETS:
            x=example_stats[method,budget]
            lines.append(f'| {method} | {budget} | {x["intervals"]} | {x["pairs"]:,} | {number(x["mean"])} | {number(x["median"])} | {number(x["near_duplicate_fraction"])} |')
    lines+=['']+block(f"""
        levin·512의 평균 IoU **{number(levin[0]['mean'])}**은 모든 pair의 겹침 점수를 평균한 값이다. 후보의 {levin[0]['mean']*100:.1f}%가 중복이라는 뜻은 아니다. 거의 같은 구간을 가리키는 pair의 비율을 보려면 IoU≥0.9 열을 읽는다. 이 사례에서는 **{levin[0]['near_duplicate_fraction']*100:.1f}%**다.

        | 지표 | 계산과 해석 |
        |---|---|
        | 평균 pairwise IoU | 한 query 안의 모든 서로 다른 pair에 대한 IoU 평균. 높을수록 평균적인 시간 겹침이 크다. |
        | 중앙 IoU | pair별 IoU를 정렬했을 때 가운데 값. pair가 짝수 개면 가운데 두 값의 평균이다. |
        | IoU≥0.5 비율 | IoU가 0.5 이상인 pair 수 ÷ 전체 pair 수. 상당히 겹치는 pair가 얼마나 흔한지 본다. |
        | IoU≥0.9 비율 | IoU가 0.9 이상인 pair 수 ÷ 전체 pair 수. 거의 같은 시간 구간을 제안한 정도를 본다. |
        | IoU=0 비율 | 겹치는 시간이 없는 pair 수 ÷ 전체 pair 수. 높을수록 서로 떨어진 후보 쌍이 많다. |

        **평균이 낮아도 비슷한 후보 묶음이 남아 있을 수 있다.** 후보가 영상 앞부분과 뒷부분에 나뉘면, 각 묶음 내부의 IoU는 높아도 묶음 사이의 IoU는 낮다. 전체 평균은 이 둘을 함께 섞는다. 그래서 평균 하나로 모든 후보가 고르게 흩어졌다고 해석하지 않고 heatmap의 덩어리를 함께 확인한다.

        ## 4. 여러 query에서도 같은 경향인가

        앞의 두 그림은 한 사례를 보여준다. 아래 그림은 pilot{count}에서 **각 query의 평균 pairwise IoU를 하나씩 구한 뒤** 그 값의 분포를 그린 것이다. 서로 다른 query의 구간끼리 직접 비교하지 않는다.

        ![Query 내부 평균 pairwise IoU의 누적 분포]({link('interval_overlap_distribution.svg')})

        가로축은 query 내부 평균 IoU, 세로축은 그 값 이하인 query의 누적 비율이다. 예를 들어 곡선이 `(0.5, 0.8)`을 지난다면 query의 80%에서 평균 IoU가 0.5 이하라는 뜻이다. 이는 읽는 법을 위한 가상 좌표이며 측정 결과를 인용한 값이 아니다. 같은 세로축 높이에서 왼쪽에 있는 곡선일수록 후보 간 겹침이 낮은 query가 많다. 곡선이 교차하면 모든 구간에서 한 예산의 겹침이 더 낮다고 단정할 수 없다.

        패널은 방법, 곡선 색은 예산을 구분한다. 범례의 n은 후보가 두 개 이상 있어 IoU를 계산할 수 있는 query 수다. K<2인 query는 겹침을 0으로 넣지 않고 분포와 겹침 평균에서 제외한다.

        아래 표에서 **고유 구간 합계·평균 구간 수는 전체 {count} query 기준**이다. 평균 pairwise IoU와 세 비율은 평가 가능한 query 내부에서 먼저 계산한 뒤 query마다 동일 가중으로 평균했다. 후보가 많은 query의 수많은 pair가 전체 결과를 지배하지 않도록 한 것이다. 비율은 0–1로 표기하며 0.10은 10%를 뜻한다.

        | 방법 | 예산 | query | 고유 구간 합계 | 평균 구간 수 | 평가 가능 / K<2 | 평균 pairwise IoU | 평균 IoU≥0.5 비율 | 평균 IoU≥0.9 비율 | 평균 IoU=0 비율 |
        |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    """)
    for method in METHODS:
        for budget in BUDGETS:
            group=[x for x in stats if x['method']==method and x['budget']==budget];valid=[x for x in group if x['pairs']]
            means=[aggregate(method,budget,k) for k in ('mean','half_fraction','near_duplicate_fraction','zero_fraction')]
            lines.append(f'| {method} | {budget} | {len(group)} | {sum(x["intervals"] for x in group):,} | {np.mean([x["intervals"] for x in group]):.2f} | {len(valid)} / {len(group)-len(valid)} | '+' | '.join(number(x) for x in means)+' |')
    lines+=['','**현재 범위의 결과를 읽으면:**','']
    for method in METHODS:
        values=' → '.join(number(aggregate(method,b,'mean')) for b in BUDGETS)
        lines.append(f'- {method}: 예산 512 → 1024 → 2048에서 평균 pairwise IoU는 **{values}**다.')
    lines+=['']+block("""
        예산을 늘릴수록 후보 수는 늘고 전체 평균 겹침은 낮아지는 경향이다. 추가 계산으로 시간적으로 덜 겹치는 후보도 확보했음을 시사한다. 다만 예산에 따라 평가 가능한 query 수도 달라질 수 있으므로, 이 요약만으로 모든 개별 query의 겹침이 줄었다고 해석하지 않는다.

        겹침이 낮다는 이유만으로 더 좋은 후보라고 볼 수는 없다. 정답 주변에 모인 후보는 겹침이 높아도 유용할 수 있고, 엉뚱한 시간대로 퍼진 후보는 겹침이 낮아도 도움이 안 될 수 있다. **시간적 다양성은 이 appendix로, 정답 포함 여부는 본 보고서의 candidate oracle로 함께 확인한다.** 고정 cohort는 층화 진단 표본이므로 이 평균을 전체 벤치마크 모집단의 추정치로 일반화하지 않는다.

        ## 5. 방법과 예산은 무엇이 다른가

        세 방법은 같은 모델에서 어떤 decoding 분기에 계산을 먼저 배분할지가 다르다. 현재 실행은 정수부 숫자와 소수점 기호에서 대안을 만들고, 그 밖의 위치는 greedy로 이어간다. 무작위 sampling을 반복한 결과가 아니다.

        | 방법 | 계산을 배분하는 기준 |
        |---|---|
        | levin | 모델이 부여한 경로 확률이 높고, 현재까지 생성한 답변 token 수가 적은 prefix를 우선한다. 정확히는 log(depth)−log(path probability)가 작은 순서다. |
        | native_mass | 첫 numeric 분기에서 나눈 묶음 중 미탐색 모델 확률 질량이 큰 곳에 더 많은 계산을 배분한다. |
        | numeric_novelty | native_mass에 더해, 다른 묶음의 완성 구간과 덜 겹치는 구간을 내놓는 묶음에 작은 보너스를 준다. 목표 배분 가중치는 U×(1+0.1q), q는 0–1의 temporal novelty다. |

        numeric_novelty의 목표 가중치 보너스는 최대 10%이며 실제 계산량이 정확히 그 비율로 배분된다는 뜻은 아니다. 시간적 다양성을 강제로 확보하는 방식도 아니다. 이 분석에서 native_mass와 수치가 비슷하므로, 현재 설정의 novelty 보정이 큰 차이를 만들었다고 해석할 근거는 없다. 탐색 중 계산 배분에는 모델 확률을 쓰지만, 이 appendix의 후보 간 겹침은 확률로 가중하지 않는다.

        **512·1024·2048은 query×방법별 실제 decoder 입력 position의 누적 상한**이다. 후보 개수나 영상 token 수가 아니며 prefix를 다시 계산한 비용도 포함한다. 공유 video/query prefill과 greedy reference 비용은 별도다. 세 예산은 한 search의 누적 checkpoint여서, 작은 예산의 후보는 큰 예산의 후보 집합에 포함된다.

        이번 구현은 허용된 분기 안에서 남은 미탐색 질량 U의 비율 U/(V+U)이 1% 이하이면 일찍 종료할 수 있다. V는 완성된 유효 경로의 질량이다. 따라서 예산을 두 배로 늘려도 후보가 두 배가 되거나 반드시 추가되는 것은 아니다. 세부 분기 정책과 질량 정의는 [실행 명세](SPEC.md)에 있다.

        ## 집계 정의와 재현

        각 checkpoint까지 EOS로 완성한 유효 후보를 사용한다. 실제 영상 길이 안으로 clipping된 `(start,end)`가 완전히 같으면 하나로 합치고, `start<end`인 구간만 비교한다. 기존 interval 집계와 일치하도록 native 질량 계산에서 제외한 `logp≤−740` 경로는 제외한다. 고유 후보 구간은 모두 같은 가중치를 갖는다.

        구간 i=[sᵢ,eᵢ], j=[sⱼ,eⱼ]에 대해 `I=max(0,min(eᵢ,eⱼ)−max(sᵢ,sⱼ))`, `IoU=I/((eᵢ−sᵢ)+(eⱼ−sⱼ)−I)`다. K≥2인 query의 평균은 `Σᵢ<ⱼ IoU(i,j) / (K(K−1)/2)`이며, 임계값 비율도 같은 pair 수를 분모로 쓴다. 전체 요약은 해당 예산·방법에서 K≥2인 query의 지표를 합해 그 query 수로 나눈다.

        Heatmap은 대칭 행렬 전체를 보여주지만 통계에는 위쪽 삼각형만 쓴다. 자기 비교와 순서를 뒤집은 중복 pair를 제외하며, 다른 query 사이의 구간 비교나 서로 다른 예산의 후보 집합 교집합 비율은 이 지표에 포함하지 않는다.
    """)
    lines+=['']+block(f"""
        [Query별 지표와 실제 분모]({os.path.relpath(OUT/'analyses/interval_overlap.json',result)}) · [출처 hash·집계 검증]({os.path.relpath(OUT/'validation/interval_overlap.json',result)}) · [핵심 보고서](REPORT.md)

        `experiments/zero-shot/.venv/bin/python experiments/zero-shot/_lib/probability_budget_overlap.py`로 재생성한다. 기존 shared TimeLens의 plotting dependency를 재사용한다. 기존 search 원자료를 읽는 CPU 분석이며 모델을 다시 실행하지 않는다. pilot300은 같은 canonical900 원자료의 고정 부분집합이다.
    """)
    result.mkdir(parents=True,exist_ok=True)
    (result/'APPENDIX_OVERLAP.md').write_text('\n'.join(lines),encoding='utf-8')

def generate():
    selfcheck();rows,stats,examples=collect()
    for count in (300,900):appendix(count,rows,stats,examples)
    print('OVERLAP_APPENDICES_COMPLETE',flush=True)

def check_viewer():
    shared=ROOT.parent/'_shared/timelens'
    sys.path.insert(0,str(shared/'browser_deps'))
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
    os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
    from playwright.sync_api import sync_playwright
    base='http://127.0.0.1:8780';errors=[];checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1500,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
        for count in (300,900):
            url=base+f'/zero-shot/results/rollout/probability_budget/viewer.html?view=pilot%2F{count}%2FAPPENDIX_OVERLAP.md'
            response=page.goto(url,wait_until='networkidle');assert response.status==200
            page.wait_for_selector('#report h1');assert f'pilot{count}' in page.locator('#report h1').inner_text()
            assert page.locator('#report img').count()==3
            page.wait_for_function("[...document.querySelectorAll('#report img')].every(i=>i.complete&&i.naturalWidth>0)")
            assert page.locator('#report table').count()==5
            for image in page.locator('#report img').all():assert page.request.get(image.get_attribute('src')).status==200
            assert page.locator('#reference-list a').filter(has_text='후보 구간').count()==1
            checks.append(dict(scope=count,images=3,tables=5,appendix_navigation=True))
        folder=ROOT/'results/rollout/probability_budget/pilot/900'
        page.screenshot(path=str(folder/'preview_overlap_desktop.png'),full_page=True)
        page.set_viewport_size(dict(width=390,height=844));page.screenshot(path=str(folder/'preview_overlap_mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2')
        assert page.request.get(base+'/index.html').status==200
        browser.close()
    assert not errors,errors
    write(OUT/'validation/interval_overlap_viewer.json',dict(passed=True,scopes=checks,page_errors=errors,timestamp=stamp()))
    print('OVERLAP_BROWSER_PASSED')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-viewer',action='store_true');args=parser.parse_args()
    check_viewer() if args.check_viewer else generate()
