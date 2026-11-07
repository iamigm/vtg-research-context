"""Presentation only: query tree and aggregate report figures; no inference changes."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())

def case_index(run):
    """Compact derived lookup; response/token records stay in their canonical store."""
    fields=('key','method','seed','budget','baseline','iou','main_oracle',
            'pool_oracle','diagnostic_oracle','main_modes','failed')
    rows=json.loads((run/'metrics/per_query.json').read_text())
    index=dict(fields=fields,rows=[[row[field] for field in fields] for row in rows])
    (run/'web/cases.json').write_text(json.dumps(index,ensure_ascii=False,separators=(',',':'))+'\n')

def install(count):
    result=ROOT/f'results/rollout/temporal_mode_search/pilot/{count}/capped_rollouts'
    run=ROOT/f'artifacts/runs/rollout/temporal_mode_search/pilot/{count}/capped_rollouts'
    html=(result/'detail.html').read_text()
    finder=Path(__file__).with_name('temporal_mode_case_finder.html').read_text().strip()
    html=re.sub(r'<aside class="panel(?: case-finder)?".*?</aside>',finder,html,count=1,flags=re.S)
    case_index(run)
    for title in ('Rollout 예산별 mIoU','성능과 연산 비용'):
        html=re.sub(r'<section><h2>'+title+r'</h2>.*?</section>','',html,flags=re.S)
    for name in ('temporal_mode_case_viewer.js','temporal_mode_case_viewer.css'):
        dest=run/'web'/('viewer.js' if name.endswith('.js') else 'case_viewer.css')
        dest.write_text(Path(__file__).with_name(name).read_text())
        version=hashlib.sha256(dest.read_bytes()).hexdigest()[:12]
        url=os.path.relpath(dest,result)+'?sha256='+version
        if name.endswith('.js'):
            html=re.sub(r'<script src="[^"]*/viewer.js(?:\?[^"]*)?"></script>',f'<script src="{url}"></script>',html)
        else:
            html=re.sub(r'<link id="case-view-style"[^>]*>','',html)
            html=html.replace('</head>',f'<link id="case-view-style" rel="stylesheet" href="{url}"></head>')
    ground_truth=run/'web/ground_truth.js'
    ground_truth.write_text(Path(__file__).with_name('temporal_mode_ground_truth.js').read_text())
    version=hashlib.sha256(ground_truth.read_bytes()).hexdigest()[:12]
    tag=f'<script id="ground-truth-script" src="{os.path.relpath(ground_truth,result)}?sha256={version}"></script>'
    html=re.sub(r'<script id="ground-truth-script"[^>]*></script>','',html)
    html=html.replace('</body>',tag+'</body>')
    html=re.sub(r'<script>window.TEMPORAL_EXPERIMENT_ROOT=.*?</script>','',html)
    media_root=os.path.relpath(ROOT.parent,result)
    html=html.replace('</head>',f'<script>window.TEMPORAL_EXPERIMENT_ROOT={json.dumps(media_root)};</script></head>')
    html=re.sub(r'<script>window.TEMPORAL_QUERIES=.*?</script>','',html)
    html=html.replace('</head>',f'<script>window.TEMPORAL_QUERIES={count};</script></head>')
    (result/'detail.html').write_text(html)
    definitions='''## 사례 탐색과 mode 비교

사례 필터는 현재 선택한 방법·seed·예산의 query별 결과를 사용한다. 비교 기준은 같은 입력의 greedy 단일 응답이다. IoU는 GT와 예측 interval의 교집합 길이 / 합집합 길이이며 범위는 0–1이고 높을수록 좋다. ΔIoU = 해당 값 − greedy IoU이며 0.01은 1pp다. Main oracle과 judge 후보 oracle은 각각 발견한 모든 main interval과 judge에 전달한 대표 interval의 최고 IoU다. Diagnostic oracle은 tail까지 포함한 최고 IoU다. Oracle은 사후 상한이며 GT는 생성·선택에 사용하지 않는다.

| 특수 기준 | query별 조건 |
|---|---|
| 회복 | greedy IoU < 0.5, 최종 선택 IoU ≥ 0.5 |
| 개선 / 악화 | 최종 선택의 greedy 대비 ΔIoU ≥ 0.01 / ≤ −0.01 |
| 후보 발견 후 선택 실패 | greedy IoU < 0.5, main oracle ≥ 0.5, 최종 선택 IoU < 0.5 |
| 탐색 후에도 미회복 | greedy IoU < 0.5, main oracle < 0.5 |
| 여러 mode | main mode 수 ≥ 2 |
| Tail에만 회복 후보 | greedy IoU < 0.5, main oracle < 0.5, diagnostic oracle ≥ 0.5 |
| 최종 선택 무효 | 유효한 최종 interval이 없음 |

수치 범위는 greedy IoU, greedy 대비 ΔIoU, main mode 수에 적용하며 양 끝을 포함한다. 모든 조건을 동시에 적용한다. Query 수는 현재 cohort의 고유 query 수이며 seed 응답 수나 영상 수가 아니다. 표본별 목록은 canonical 응답을 복제하지 않고 기존 query 지표를 조회한다.

Mode 비교는 저장된 checkpoint의 소속을 그대로 보여준다. 새 interval은 기존 mode의 **모든 interval**에 대해 IoU ≥ 0.5이고 시작·끝 차이가 각각 max(1 s, 0.1 × 두 interval 중 짧은 길이) 이내이면 통합한다. 해당 mode가 여러 개면 endpoint 거리와 mode ID 순서로 결정한다. 조건을 만족하는 mode가 없으면 새 mode다. Mode ID는 해당 방법·seed·예산 안에서만 유효하며 실제 사건 수나 정답 확률을 뜻하지 않는다. 무효 경로는 mode에 넣지 않는다. 같은 sequence의 반복은 관측 횟수로 묶고 각 고유 응답의 mode, interval, IoU와 greedy 대비 ΔIoU를 표시한다. 공통 시간축의 얇은 막대는 각 고유 interval이며 mode 전체를 하나의 합친 interval로 해석하지 않는다.
'''
    path=result/'REPORT.md'
    text=re.sub(r'\n<!-- case-finder:start -->.*?<!-- case-finder:end -->\n?', '',path.read_text(),flags=re.S)
    path.write_text(text.rstrip()+'\n\n<!-- case-finder:start -->\n'+definitions+'<!-- case-finder:end -->\n')


def report(count):
    result=ROOT/f'results/rollout/temporal_mode_search/pilot/{count}/capped_rollouts'
    run=ROOT/f'artifacts/runs/rollout/temporal_mode_search/pilot/{count}/capped_rollouts'
    rows=json.loads((run/'metrics/per_query.json').read_text())
    groups=[]
    fields=('iou','main_oracle','pool_oracle','diagnostic_oracle','baseline')
    for method in ('probability','token_entropy','temporal_mode'):
        for budget in (16,32,64,128):
            selected=[r for r in rows if r['method']==method and r['budget']==budget]
            if not selected:continue
            keys={r['key'] for r in selected}
            assert len(selected)==2*len(keys)
            g=dict(method=method,budget=budget,queries=len(keys))
            g.update({f:sum(r[f] for r in selected)/len(selected) for f in fields});groups.append(g)
    lines=['## Rollout 예산별 mIoU · 최종 선택과 oracle','','전체 cohort의 집계다. 같은 query의 seed42·43을 먼저 평균한다. mIoU는 GT와의 temporal IoU를 query별로 평균한 값이며 높을수록 좋다. Main oracle은 발견한 main interval 전체,judge 후보 oracle은 전달한 대표 interval,diagnostic oracle은 main+tail의 사후 최고 IoU다. GT는 후보 생성·선택에 사용하지 않는다. Oracle은 실제 모델 성능이 아니다.','','**탐색과 선택을 구분한다.** Main oracle − judge 후보 oracle은 shortlist에서 잃은 기회,judge 후보 oracle − 최종 선택 mIoU는 모델의 선택에서 잃은 기회다. Tail은 diagnostic oracle에만 포함하고 최종 선택 후보로 쓰지 않는다.','','| 총 rollout | 방법 | Query | 최종 선택 mIoU | Main oracle | Judge 후보 oracle | 선택 손실 | Diagnostic oracle (tail 포함) |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f'| {2*g["budget"]} | {g["method"]} | {g["queries"]} | {g["iou"]:.4f} | {g["main_oracle"]:.4f} | {g["pool_oracle"]:.4f} | {g["pool_oracle"]-g["iou"]:.4f} | {g["diagnostic_oracle"]:.4f} |')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True,constrained_layout=True)
    for number,(ax,method) in enumerate(zip(axes,('probability','token_entropy','temporal_mode')),1):
        subset=[g for g in groups if g['method']==method]
        for field,label,color,style in [('iou','Selected','#2469ac','-'),('main_oracle','Main oracle','#25805a','--'),('pool_oracle','Judge pool oracle','#b17a26',':')]:
            ax.plot([2*g['budget'] for g in subset],[g[field] for g in subset],marker='o',label=label,color=color,linestyle=style)
        ax.axhline(subset[0]['baseline'],color='#6b7280',label='greedy',linestyle='-.')
        ax.set_title(f'{number}. {method}');ax.legend(fontsize=7,loc='lower right');ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256],['32','64','128','256']);ax.set_xlabel('Total rollout cap');ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean temporal IoU')
    dest=run/'web/budget_oracle.png';fig.savefig(dest,dpi=160);plt.close(fig)
    lines += [
        '', '### 그래프1·2·3: 더 많이 탐색하면 좋은 후보와 좋은 답이 함께 늘어나는가?', '',
        '아래 그림에는 왼쪽부터 세 그래프가 있다. **같은 평가를 서로 다른 탐색 방법에 각각 적용한 것**이다. 각 그래프 안의 색은 평가값을 구분하며 회색 선은 greedy 기준이다.', '',
        '| 그래프 | 제목 | 무엇을 우선 탐색하는 방법인가? |',
        '|---|---|---|',
        '| 1 · 왼쪽 | probability | 모델이 그 prefix와 token을 선택할 확률이 높고,조사 비용이 낮은 분기 |',
        '| 2 · 가운데 | token_entropy | 위 조건에 더해,다음 token 확률이 여러 대안으로 흩어져 있는 분기 |',
        '| 3 · 오른쪽 | temporal_mode | 위 조건에 더해,유효한 새 시간 가설을 발견할 전망이 있는 분기 |', '',
        '**가로축은 조사 횟수의 상한이다.** 32·64·128·256은 query·방법·seed 한 경우의 main+tail 총 rollout 상한이다. 예를 들어256이면 main128회와 진단용 tail128회다. 서로 다른 후보256개라는 뜻은 아니며 중복·무효 생성도 시도 수에 포함한다. Judge 호출은 이 횟수에 넣지 않는다. 오른쪽 한 칸마다 상한이 두 배가 되므로 칸 간격이 같아도32→64와128→256의 증가량은 다르다.', '',
        '**세로축은 최종 interval이 GT와 얼마나 겹치는지의 평균(mIoU)이다.** 1에 가까울수록 좋다. 0.5는 전체 query에서 interval 겹침 점수의 평균이0.5라는 뜻이며,정답률50%라는 뜻이 아니다. 세 그래프는 같은0–1 눈금을 사용한다. 각 query의 seed42·43을 먼저 평균하고 query들을 평균한다.', '',
        '| 선의 색·모양 | 그림의 범례 | 쉽게 읽는 뜻 | 실제 모델 성능인가? |',
        '|---|---|---|---|',
        '| 파란 실선 | Selected | 모델이 마지막에 실제로 고른 답의 평균 IoU | **그렇다. 성능 개선 여부는 이 선을 본다.** |',
        '| 초록 파선 | Main oracle | 찾은 모든 main interval 중 GT에 가장 잘 맞는 것을 사후에 골랐다면 얻는 평균 | 아니다. 탐색으로 확보한 기회의 상한이다. |',
        '| 황갈색 점선 | Judge pool oracle | 모델에게 실제 전달한 대표 interval 중 GT에 가장 잘 맞는 것을 사후에 골랐다면 얻는 평균 | 아니다. Shortlist 안에 남은 기회의 상한이다. |',
        '| 회색 일점쇄선 | greedy | 동일 입력의 greedy 단일 응답 mIoU | 모든 성능 비교의 기준이다. |', '',
        f'![그래프1·2·3: 예산별 최종 선택과 oracle]({os.path.relpath(dest,result)})', '',
        '**읽는 순서는 초록 → 황갈색 → 파랑이다.** 먼저 좋은 후보를 발견했는지,그 후보가 judge에 전달됐는지,judge가 실제로 골랐는지를 순서대로 본다. 같은 가로 위치에서 초록과 황갈색 사이의 틈은 shortlist에서 잃은 기회이고,황갈색과 파랑 사이의 틈은 최종 선택에서 잃은 기회다. 이 값은 query마다 최고점을 구한 뒤 평균한 것이며,모든 query에 공통인 하나의 정답 후보가 있다는 뜻은 아니다.', '',
        '**초록만 오르고 파랑이 그대로라면 “좋은 대안은 더 찾았지만 실제 답은 좋아지지 않았다”고 읽는다.** 후보가 늘어도 상위6 mode와 대표 interval이 바뀔 수 있어 황갈색 선은 내려갈 수 있고,모델의 선택도 바뀌므로 파란 선 역시 내려갈 수 있다. 선을 연결한 것은 네 예산의 변화를 보기 위한 것이며 중간 예산을 모두 실험한 것은 아니다. Tail을 포함한 diagnostic oracle은 표에만 있고,이 세 선에는 포함하지 않는다.', '',
        '#### 이 pilot의 숫자로 읽어 보기', ''
    ]
    for number,method in enumerate(('probability','token_entropy','temporal_mode'),1):
        subset=sorted([g for g in groups if g['method']==method],key=lambda g:g['budget'])
        if not subset:continue
        first,last=subset[0],subset[-1]
        delta=100*(last['iou']-last['baseline'])
        verdict='개선' if delta>0 else '악화' if delta<0 else '동일'
        lines.append(f'- **그래프{number} · {method}:** greedy 기준 mIoU는{last["baseline"]:.4f}다. 총 rollout 상한{2*first["budget"]}에서 실제 선택 mIoU는{first["iou"]:.4f}(greedy 대비{100*(first["iou"]-first["baseline"]):+.2f}pp), 상한{2*last["budget"]}에서는{last["iou"]:.4f}(greedy 대비{delta:+.2f}pp, {verdict})다. 최대 예산의 main oracle은{last["main_oracle"]:.4f}, judge 후보 oracle은{last["pool_oracle"]:.4f}다.')
    lines += ['', '모든 성능 변화는 **같은 query의 greedy 대비**다. Δpp=100×(선택 mIoU−greedy mIoU)이며 양수일수록 좋다. 예를 들어0.01 차이는1pp다.', '',
        '### 그래프4: 정확도를 조금 높이는 데 계산은 얼마나 더 필요한가?', '',
        '앞의 세 그래프는 가로축을 생성 시도 수로 맞췄다. 네 번째 그래프는 **같은 시도 수라도 응답 길이와 후보 비교량에 따라 계산이 다를 수 있으므로,실제 처리할 token position 수를 기준으로 다시 배치한 그림**이다.', '']
    costs=json.loads((run/'metrics/costs.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for method,color in [('probability','#3974b9'),('token_entropy','#a66723'),('temporal_mode','#22806a')]:
        points=sorted([g for g in costs['groups'] if g['method']==method and g['queries']],key=lambda g:g['budget'])
        ax.plot([g['research']['decoder_inputs'] for g in points],[g['iou'] for g in points],marker='o',label=method,color=color)
        for g in points:ax.annotate(str(g['total_rollout_budget']),(g['research']['decoder_inputs'],g['iou']),xytext=(4,5),textcoords='offset points',fontsize=8)
    if costs['groups']:
        b=costs['groups'][0]['baseline'];ax.scatter([b['decoder_inputs']],[b['iou']],label='greedy',color='#666')
    ax.set_title('4. Accuracy vs computation');ax.set_xlabel('Independent research decoder inputs (tail included)');ax.set_ylabel('Mean temporal IoU');ax.grid(alpha=.2);ax.legend(fontsize=8)
    dest=run/'web/performance_cost.png';fig.savefig(dest,dpi=160);plt.close(fig)
    lines += [
        f'![그래프4: 전체 샘플의 성능과 연산 비용]({os.path.relpath(dest,result)})', '',
        '**가로축은 query·방법·seed·예산 한 경우를 독립 실행할 때 필요한 decoder inputs의 query 평균이다.** 숫자가 클수록 모델에 더 많은 token position을 처리시킨다는 뜻이다. 출력 문장의 token 수나 GPU seconds가 아니다. 영상 prefill,greedy,main 탐색,judge 비교,진단용 tail까지 포함한 research total이다. 저장된 trace를 빈 cache에서 실행하는 기준으로 재구성하며,다른 방법·seed·작은 예산에서 계산한 cache를 공짜로 빌려 쓰지 않는다. Vision encoder의 모든 연산까지 나타내는 FLOPs 측정은 아니다.', '',
        '**세로축은 모델이 실제 고른 답의 평균 IoU다.** 앞의 세 그래프의 파란 Selected 선과 같은 종류의 점수다. Oracle 선은 없다. 네 번째 그림의 세로축은 값 차이를 보기 위해 자동으로 범위를 잡으므로,0부터1까지 고정된 앞의 세 그래프와 선의 기울기나 높이를 눈대중으로 직접 비교하지 않는다.', '',
        '**여기서는 색의 뜻이 바뀐다.** 앞의 세 그래프는 색으로 평가값을 구분했지만,그래프4는 색으로 탐색 방법을 구분한다.', '',
        '| 색·표시 | 의미 |', '|---|---|',
        '| 파란 선 | probability 방법의 실제 최종 성능과 계산량 |',
        '| 황갈색 선 | token_entropy 방법의 실제 최종 성능과 계산량 |',
        '| 초록 선 | temporal_mode 방법의 실제 최종 성능과 계산량 |',
        '| 회색 점 greedy | 추가 탐색·후보 judge 없이 greedy 답 하나를 생성한 기준 |',
        '| 각 점 옆의32·64·128·256 | 그 점을 얻을 때의 main+tail 총 rollout 상한. Query 수나 계산량 값 자체가 아니다. |', '',
        '**같은 높이라면 왼쪽 점이 더 적은 계산으로 같은 성능을 낸다. 같은 가로 위치라면 위쪽 점이 더 정확하다.** 한 점이 다른 점보다 오른쪽에만 있고 위로 올라가지 않았다면,이 비용 지표상 계산만 늘고 정확도는 좋아지지 않은 것이다. 선이 거의 수평으로 오른쪽으로 뻗으면 추가 계산의 이득이 작다는 뜻이다. greedy 회색 점보다 위에 있는지와,그만큼 오른쪽으로 얼마나 이동했는지를 함께 본다. Tail은 답에 사용하지 않았어도 이번 연구에 쓴 계산이므로 비용에 포함한다.', '',
        '#### 이 pilot의 비용·정확도 예시', ''
    ]
    points=sorted([g for g in costs['groups'] if g['method']=='temporal_mode' and g['queries']],key=lambda g:g['budget'])
    if points:
        first,last=points[0],points[-1]
        ratio=last['research']['decoder_inputs']/last['baseline']['decoder_inputs']
        lines += [f'temporal_mode의 greedy 기준은 평균 decoder inputs {last["baseline"]["decoder_inputs"]:,.0f}, mIoU {last["baseline"]["iou"]:.4f}다. 총 rollout 상한{last["total_rollout_budget"]}의 평균 decoder inputs는{last["research"]["decoder_inputs"]:,.0f}로 greedy의 약{ratio:.2f}배다. 실제 선택 mIoU는{last["iou"]:.4f}, greedy 대비{100*(last["iou"]-last["baseline"]["iou"]):+.2f}pp다. 비용과 성능 모두 같은 greedy를 기준으로 비교한다.']
    lines += ['', '그래프4의 공통 비교 query 수는'+str(costs['common_queries'])+'개다. 같은 query 집합에서 모든 방법·예산·두 seed가 완료된 경우를 비교한다. 이 그림에는 error bar가 없으므로 작은 차이가 통계적으로 확실하다고 결론 내리지 않는다. 실제 wall/CUDA 시간,forward 호출 수,memory는 이 보고서의 연산 비용 표와 함께 읽는다. Operational 비용은 tail을 제외한 video prefill+greedy+main+judge이고,research total은 tail까지 포함한다. Viewer에는 선택한 query의 값만 표시한다.', '']
    path=result/'REPORT.md';text=path.read_text();text=re.sub(r'<!-- case-presentation:start -->.*?<!-- case-presentation:end -->\n?','',text,flags=re.S)
    block='<!-- case-presentation:start -->\n'+'\n'.join(lines)+'\n<!-- case-presentation:end -->\n'
    text=text.replace('## 실제 시도·중복·적응 선택',block+'\n## 실제 시도·중복·적응 선택')
    path.write_text(text)

if __name__=='__main__':
    for count in (300,600,900):install(count);report(count)
