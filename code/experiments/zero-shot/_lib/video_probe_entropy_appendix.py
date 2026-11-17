"""Entropy on the original fixed top-30 support of saved visual probes; no inference."""
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/runs/rollout/oracle_switch_controls/pilot/60'
RESULT=ROOT/'results/rollout/oracle_switch_controls/pilot/60'
METHODS=('targeted','opposite','random','repeat_gradient','rounded8_targeted')
FIELDS=('gain','target_delta','target_q_delta','mass','mass_delta','h_decreased','target_top1','branch_reached','prefix_match','argmax_changed','outside_support','iou_delta','iou_improved','free_target_branch','free_earlier_divergence','baseline_iou','probe_iou','r03','r05','r07','r03_delta','r05_delta','r07_delta')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def f(v,n=3):return '—' if v is None else f'{v:.{n}f}'
def ci(v,n=3):return f"{f(v['mean'],n)} [{f(v['ci95'][0],n)}, {f(v['ci95'][1],n)}]"
def pct(v):return f'{100*v:.1f}%'


def entropy(probe,ids):
    assert [r['id'] for r in probe['top30']]==ids
    ps=np.array([r['probability'] for r in probe['top30']],float)
    assert len(ps)==30 and np.all(np.isfinite(ps)) and np.all(ps>=0)
    mass=float(ps.sum());assert mass>0
    q=ps/mass
    logits=np.array([r['logit'] for r in probe['top30']]);lq=np.exp(logits-logits.max());lq/=lq.sum()
    assert np.max(np.abs(lq-q))<2e-6
    value=float(-np.sum(q[q>0]*np.log(q[q>0])))
    assert 0<=value<=math.log(30)+1e-10
    return value,mass


def record(sample,baseline,basefree,probe,free,**tags):
    ids=[r['id'] for r in baseline['top30']]
    bh,bm=entropy(baseline,ids);hh,mass=entropy(probe,ids)
    gain=bh-hh
    assert free is not None and 'iou' in free
    return dict(sample_key=sample['sample_key'],dataset=sample['dataset'],video=sample['row']['video_path'],
                baseline_h=bh,probe_h=hh,gain=gain,baseline_mass=bm,mass=mass,mass_delta=mass-bm,
                target_delta=probe['oracle_p']-baseline['oracle_p'],
                target_q_delta=probe['oracle_p']/mass-baseline['oracle_p']/bm,
                h_decreased=float(gain>0),target_top1=float(probe['target_top1']),
                branch_reached=float(probe['branch_reached']),prefix_match=float(probe['prefix_argmax_match']),
                argmax_changed=float(probe['top1_id']!=baseline['top1_id']),
                outside_support=float(not probe['top1_in_original_top30']),
                baseline_iou=basefree['iou'],probe_iou=free['iou'],iou_delta=free['iou']-basefree['iou'],
                iou_improved=float(free['iou']>basefree['iou']),free_target_branch=float(free['oracle_branch_reached']),
                free_earlier_divergence=float(free['first_divergence_step'] is not None and free['first_divergence_step']<sample['branch']['step']),
                **{'r0'+str(int(t*10)):float(free['iou']>=t) for t in (.3,.5,.7)},
                **{'r0'+str(int(t*10))+'_delta':float(free['iou']>=t)-float(basefree['iou']>=t) for t in (.3,.5,.7)},**tags)


def query_means(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['sample_key']].append(row)
    result=[]
    for key,rr in sorted(groups.items()):
        result.append(dict(sample_key=key,dataset=rr[0]['dataset'],video=rr[0]['video'],
                           **{field:float(np.mean([r[field] for r in rr])) for field in FIELDS}))
    return result


def cluster_stats(rows,fields=FIELDS):
    """Average repeats in query; stratified video-cluster percentile bootstrap."""
    rr=query_means(rows);n=len(rr)
    if not rr:return dict(queries=0,trials=0,videos=0,metrics={})
    matrix=np.array([[r[k] for k in fields] for r in rr]);draws=[];rng=np.random.default_rng(42)
    strata=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rr):strata[r['dataset']][r['video']].append(i)
    clusters=[[np.array(v,int) for v in by.values()] for by in strata.values()]
    for _ in range(1000):
        values=[]
        for group in clusters:
            ix=np.concatenate([group[j] for j in rng.integers(0,len(group),size=len(group))])
            # Hold the observed query weight of each dataset fixed.
            size=sum(len(g) for g in group)
            values.append(matrix[ix].mean(axis=0)*size/n)
        draws.append(np.sum(values,axis=0))
    interval=np.quantile(draws,[.025,.975],axis=0)
    return dict(queries=n,trials=len(rows),videos=len({r['video'] for r in rr}),
                metrics={k:dict(mean=float(matrix[:,i].mean()),ci95=[float(interval[0,i]),float(interval[1,i])]) for i,k in enumerate(fields)})


def table(headers,rows,caption):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
            *['| '+' | '.join(map(str,row))+' |' for row in rows],'','**표 해설.** '+caption,'']


def render(summary):
    RESULT.mkdir(parents=True,exist_ok=True)
    primary=summary['primary'];paired=summary['paired'];boundaries=summary['boundaries'];quadrants=summary['targeted_groups']
    t=next(r for r in primary if r['epsilon_255']==.25 and r['method']=='targeted')
    o=next(r for r in primary if r['epsilon_255']==.25 and r['method']=='opposite')
    count=lambda row,key:round(row['metrics'][key]['mean']*row['queries'])
    lines=['# 영상 입력 개입과 entropy · pilot · 60 queries','',
      '**기존에 추출한 분포를 다시 계산한 결과다. 새 모델 추론은 하지 않았다.** 같은 query·고정 prefix에서 영상 RGB만 바뀐 probe를 이용한다. 주 분석은 사전에 정한 동일 변경 예산의 60개 query 전체이며, 성공한 전환 경계만 고른 결과는 뒤에 따로 둔다.','',
      '## 이 결과를 어떻게 읽는가','',
      '여기서 묻는 것은 “영상 입력이 바뀌어 더 좋은 답을 내면 entropy도 내려가는가”다. 이 실험은 GT로 선택한 oracle 후보 token을 향해 RGB gradient 방향으로 입력을 바꾼 진단이다. 자연스러운 증거 frame 추가·제거 실험이나 GT 없이 쓸 수 있는 decoding 정책의 성능은 아니다. 이전 Token uncertainty appendix의 관측 greedy 경로와 데이터 조건·정답 정의가 다르므로 합산하지 않는다.','',
      '## 먼저 읽을 결과','',
      f'고정 예산 ε₂₅₅=0.25에서 targeted 방향은 {t["queries"]}개 중 {count(t,"h_decreased")}개에서 H를 낮추고 {count(t,"iou_improved")}개에서 IoU를 높였다. 반대 방향은 {count(o,"h_decreased")}개에서 H를 낮췄지만 IoU 개선은 {count(o,"iou_improved")}개뿐이었다. 평균 IoU 변화는 targeted {t["metrics"]["iou_delta"]["mean"]:+.3f}, opposite {o["metrics"]["iou_delta"]["mean"]:+.3f}이었다. 즉 **더 자주 entropy를 낮춘 방향이 오히려 답을 악화시킬 수 있었다.** 이 수치는 같은 변경 예산의 사후 관측 결과이며 GT 없는 자연 증거 정책의 성능은 아니다.','',
      '## 자료와 지표 정의','',
      f'Qwen3-VL-8B-Instruct의 pilot 60개 query(데이터셋별 20개, 고유 video {summary["videos"]}개)를 사용했다. 원래 pilot900 중 조건을 만족한 805개에서 hash로 고른 집합이며, 별도로 고른 극단 사례 3개는 제외했다. 각 query에서 고정 예산 ε₂₅₅=0.25·1.0과 5개 방법을 비교한다. random은 seed42·43·44의 반복이며 전체 fixed-budget probe/free-decoding 쌍은 {summary["trial_count"]}개다.','',
      '원본 prefix에서 확률이 높았던 token 30개를 고정 집합 S라 두고, 각 개입 조건에서 저장된 확률을 qᵢ=pᵢ/Σⱼ∈S pⱼ로 재정규화한다. **Hfixed30=−Σᵢ∈S qᵢlnqᵢ**이며 단위는 nat이다. 이 값은 원본 top-30 재정규화 entropy를 같은 support 위에서 비교한 값이다. 개입 후 새로 뽑은 native top-30 entropy와 같지 않다. 집합 S의 원래 probability mass M=Σᵢ∈S pᵢ와 native argmax가 S 밖으로 나간 비율도 표시한다.','',
      '**G=Hbaseline−Hperturbed**로 정의해 양수는 entropy 감소, 음수는 증가다. Δp_target은 GT로 고른 oracle 후보 token의 원래 probability 변화이고, Δq_target은 S 안에서 재정규화한 share 변화다. target은 한 token이며 정답 가능 후보 전체의 mass가 아니다. 같은 목표 token을 선택해도 뒤쪽 답이 달라질 수 있어 실제 unforced greedy decoding의 ΔIoU와 별도로 본다.','',
      '각 query의 random seed 반복은 먼저 평균하고 query를 같은 가중치로 평균했다. 95% CI는 dataset별 video cluster bootstrap 1,000회(seed42)로 계산하며 같은 video의 query를 함께 재표집한다. 데이터셋의 관측 query 가중치는 고정한다. 차이 CI도 동일 query 안의 paired 차이를 재표집한다. 다중 비교 보정 없는 진단이며 60개 선정 표본 밖으로 일반화하지 않는다.','',
      '## 1. 같은 변경 예산에서 entropy와 답이 함께 좋아지는가','']
    rows=[]
    for r in primary:
        m=r['metrics'];rows.append([r['epsilon_255'],r['method'],f"{r['queries']} / {r['trials']}",ci(m['gain']),ci(m['target_delta']),pct(m['h_decreased']['mean']),ci(m['iou_delta']),pct(m['iou_improved']['mean'])])
    lines+=table(['ε₂₅₅','방법','query / trial','G 평균 [95% CI] nat','Δp_target [95% CI]','H 감소 비율','ΔIoU [95% CI]','IoU 개선 비율'],rows,
      'targeted는 원래 greedy token보다 oracle token을 지지하도록 RGB를 바꾼 방향이고 opposite는 반대 방향이다. random은 같은 budget의 무작위 방향, repeat_gradient는 별도 gradient 반복, rounded8_targeted는 변경 뒤 8-bit 영상으로 되돌린 조건이다. G>0이더라도 ΔIoU가 양수라는 보장은 없다. random 비율은 query 안의 세 seed 비율을 평균한 값이다.')
    baseline=t['metrics']
    performance=[['—','original baseline',f"{100*baseline['baseline_iou']['mean']:.1f} (+0.0)",*[f"{100*(baseline[k]['mean']-baseline[k+'_delta']['mean']):.1f} (+0.0)" for k in ('r03','r05','r07')]]]
    for r in primary:
        m=r['metrics']
        performance.append([r['epsilon_255'],r['method'],f"{100*m['probe_iou']['mean']:.1f} ({100*m['iou_delta']['mean']:+.1f})",*[f"{100*m[k]['mean']:.1f} ({100*m[k+'_delta']['mean']:+.1f})" for k in ('r03','r05','r07')]])
    lines+=table(['ε₂₅₅','방법','mIoU % (Δpp)','R@.3 % (Δpp)','R@.5 % (Δpp)','R@.7 % (Δpp)'],performance,
      'IoU는 예측·GT 시간 구간의 교집합 길이/합집합 길이, mIoU는 query별 IoU 평균, R@τ는 IoU≥τ인 query 비율이다. 값은 100을 곱한 %이며 괄호는 원본 free decoding 대비 percentage-point 변화다. random은 seed를 query 안에서 먼저 평균한다. 모두 높을수록 좋다. 이 표의 최종 성능은 fixed-prefix target 선택과 구분한다.')
    lines+=['## 2. Entropy 감소와 target 선택을 구분하기','']
    lines+=table(['ε₂₅₅','방법','native argmax 변경','target=argmax','앞 prefix 유지','고정-prefix branch 도달','실제 free branch 도달','free의 더 이른 분기','S 밖 argmax'],[[r['epsilon_255'],r['method'],*[pct(r['metrics'][k]['mean']) for k in ('argmax_changed','target_top1','prefix_match','branch_reached','free_target_branch','free_earlier_divergence','outside_support')]] for r in primary],
      '고정-prefix probe는 원래 prefix를 입력으로 주었을 때의 분포다. 실제 free decoding은 이전 token부터 다시 생성하므로 prefix가 먼저 바뀔 수 있다. 더 이른 분기는 최초 생성 차이 위치가 probe 위치보다 앞선 경우다. 따라서 fixed-prefix H와 최종 IoU의 연관성이 반드시 해당 branch를 통한 효과라는 뜻은 아니다. branch 도달은 target이 argmax이고 앞 prefix도 유지됐다는 조건이며, free decoding의 실제 도달과 따로 기록했다. S 밖 argmax가 있으면 고정 30개 안의 집중도만으로 전체 분포를 설명할 수 없다.')
    lines+=table(['ε₂₅₅','방법','S mass 평균 [95% CI]','관측 S mass 최소','Δq_target [95% CI]'],[[r['epsilon_255'],r['method'],ci(r['metrics']['mass'],6),f(r['min_mass'],6),ci(r['metrics']['target_q_delta'])] for r in primary],
      'mass가 1에 가까우면 이 조건에서 S 밖에 남은 확률이 작다는 뜻이다. 이는 모든 영상 개입에서 고정 S가 충분하다는 일반 보장이 아니다. 원래 확률 합의 미세한 1 초과는 저장된 float32 반올림 범위다. target share와 raw target probability는 다른 분모를 가진다.')
    lines+=['## 3. Targeted에서 H가 내려간 query가 실제로 개선됐는가','']
    lines+=table(['ε₂₅₅','H 방향','query 수','target=argmax 수','IoU 개선 수','ΔIoU 평균','G 평균 nat'],[[r['epsilon_255'],r['group'],r['queries'],r['target_top1'],r['iou_improved'],f(r['mean_iou_delta']),f(r['mean_gain'])] for r in quadrants],
      '같은 fixed-budget targeted 조건의 60개 query를 G>0(감소), G=0(동일), G<0(증가)로 나눈 기술통계다. H 방향을 본 뒤 나눈 집계이므로 entropy를 낮추는 정책의 인과 효과가 아니다. IoU가 좋아지면서 H가 증가한 query는 “정답 방향으로 바뀌려면 반드시 entropy가 감소해야 한다”는 주장에 대한 반례다. target은 반드시 완전한 정답 token이라는 뜻도 아니다.')
    lines+=['![고정 예산별 entropy 변화와 실제 IoU 변화]('+os.path.relpath(ART/'figures/entropy_gain_iou.svg',RESULT)+')','',
      '**그래프 해설.** 각 점은 targeted 조건의 query 하나다. 가로축 G>0은 entropy 감소, 세로축 ΔIoU>0은 실제 답 개선이다. 왼쪽 위는 entropy가 증가했지만 답은 좋아진 경우, 오른쪽 아래는 entropy가 감소했지만 답은 나빠진 경우다. 점의 분포는 같은 예산 아래 관측된 관계이며 인과 회귀선이나 성능 보장을 뜻하지 않는다.','',
      '## 4. Matched 대조 방향과의 차이','']
    lines+=table(['ε₂₅₅','비교','query 수','G 차이 [95% CI] nat','Δp_target 차이 [95% CI]','ΔIoU 차이 [95% CI]'],[[r['epsilon_255'],r['comparison'],r['queries'],ci(r['metrics']['gain']),ci(r['metrics']['target_delta']),ci(r['metrics']['iou_delta'])] for r in paired],
      '같은 query·고정 ε에서 targeted 값에서 opposite 또는 random query 평균을 뺀다. G 차이가 양수면 targeted에서 더 많이 집중됐다는 뜻이다. ΔIoU 차이도 양수인지 함께 봐야 한다. 두 조건 모두 실제 영상 입력에 개입하지만 방향은 GT oracle 목표에 의존하므로 자연 증거 활용의 효과로 해석하지 않는다.')
    lines+=['## 5. 변경 범위별 adaptive 경계는 보조 진단','']
    lines+=table(['변경 범위','free decoding 위치','query 수','G 평균 [95% CI] nat','H 감소 비율','ΔIoU 평균 [95% CI]','IoU 개선 비율'],[[r['support'],r['position'],r['queries'],ci(r['metrics']['gain']),pct(r['metrics']['h_decreased']['mean']),ci(r['metrics']['iou_delta']),pct(r['metrics']['iou_improved']['mean'])] for r in boundaries],
      'boundary high는 oracle 전환이 확인된 상단 ε, low는 바로 아래 ε, limit은 최대 탐색 ε에서도 전환 경계가 없던 경우다. probe와 free decoding의 실제 ε를 대조해 연결했다. high만 보면 성공 사례를 고른 편향이 있으며 support별 ε·query 분모도 다르다. 따라서 1절의 고정 예산 60개 전체 비교가 주 결과다. 같은 query의 low/high를 독립 표본으로 합산하지 않는다.')
    lines+=['## 6. 현재 해석과 다른 저장 자료','',
      '이 자료는 영상 입력 변화에 따른 fixed-prefix 분포 변화를 실제로 측정해 두었다는 점에서 greedy 경로의 연속 step 비교보다 직접적이다. 그러나 RGB perturbation으로 GT oracle을 겨냥한 결과이므로, 의미 있는 영상 증거를 더 보여 줘서 정답을 얻었다고 부를 수 없다. 자연 증거 개입 설계에서는 관련 구간 제거와 동일 예산의 대조 구간 제거를 따로 비교하고, entropy와 정답 성능을 함께 평가해야 한다.','',
      'H 감소만을 decoding 목표로 삼으면 잘못된 확신을 강화할 수 있다. 반대로 원래의 잘못된 확신을 깨고 후보를 다시 경쟁시키면 H가 증가하면서 답이 좋아질 수 있다. 따라서 성공 기준은 영상 개입이 실제 선택과 최종 구간을 어떻게 바꾸는지이며, H는 그 과정을 설명하는 보조 지표다.','']
    lines+=table(['기존 저장 자료','저장 범위','이 appendix와 합치지 않은 이유'],[
      ['deterministic rollout','pilot900의 여러 prefix·분기와 전체 vocabulary 분포','같은 영상에서 prefix가 달라지는 비교. 별도 경로 분석으로 연결'],
      ['stochastic rollout','pilot300, 2모델×2sampling방법×5회=6,000응답','post-filter 후보만 저장; Qwen3 top-k는20개여서 native raw top30 전체를 복원할 수 없음'],
      ['probability budget','pilot900, 탐색 prefix별 raw top30와3방법의 tree','방문한 prefix가 탐색 정책에 의해 선택됨; 각 경로의 중복 prefix를 독립 표본으로 합산하면 안 됨'],
      ['reconsideration','pilot900×10조건의 root/child probe, full-vocabulary float32 분포','review와 candidate ledger가 context에 추가됨. 영상만 바꾼 효과와 구분 필요'],
      ['oracle switch 극단 사례','pilot3, dataset별 oracle 개선 최대 사례','성과로 고른 별도 진단 집합이며 이60개와 합산하지 않음']],
      '자료의 존재와 분석 가능성을 확인한 inventory다. 위 자료의 entropy 수치를 이 표에서 계산했다고 주장하지 않으며, 서로 다른 입력·sampling·선정 조건을 하나의 평균으로 합치지 않는다.')
    lines+=['## 재현과 검증','',
      '`_lib/video_probe_entropy_appendix.py`를 실행하면 저장된 probe만 읽어 이 appendix를 재생성한다. 원본 top30 ID 집합 일치, 확률 재정규화와 logit softmax 일치, baseline 연결, free decoding ε 연결 및 입력 SHA-256 불변을 검증한다. fixed-budget trial은 정확히 840개이며 random seed는 독립 query로 세지 않는다.','',
      '[집계 JSON]('+os.path.relpath(ART/'metrics/entropy_summary.json',RESULT)+') · [출처 SHA-256]('+os.path.relpath(ART/'metadata/entropy_sources.json',RESULT)+') · [검증 기록]('+os.path.relpath(ART/'validation/entropy_checks.json',RESULT)+')','',
      '[기존 핵심 보고서](REPORT.md)','']
    (RESULT/'APPENDIX_ENTROPY.md').write_text('\n'.join(lines)+'\n')


def build():
    selection=ART/'metadata/selection.json';selected=json.loads(selection.read_text())['samples']
    files=[selection,ART/'metadata/config.json'];rows=[];boundary_rows=[]
    for sample in selected:
        key=sample['sample_key'];qpath=ART/'queries'/f'{key}.json';cpath=ART/'analyses/controls'/f'{key}.json';files.extend([qpath,cpath])
        query=json.loads(qpath.read_text());controls=json.loads(cpath.read_text());assert query['complete'] and controls['complete']
        assert controls['baseline']==query['baseline'] and controls['baseline_free']==query['baseline_free']
        assert len(controls['trials'])==14
        expected={(method,seed,eps) for eps in (.25,1.) for method in METHODS for seed in ((42,43,44) if method=='random' else (None,))}
        assert {(t['method'],t['seed'],t['epsilon_255']) for t in controls['trials']}==expected
        for trial in controls['trials']:
            rows.append(record(sample,controls['baseline'],controls['baseline_free'],trial['probe'],trial['free'],method=trial['method'],epsilon_255=trial['epsilon_255'],seed=trial['seed']))
        for direction in query['directions']:
            for label,keyfree in [('low','free_low'),('high','free_high'),('limit','free_limit')]:
                free=direction.get(keyfree)
                if free is None:continue
                epsilon=free['epsilon_255']
                candidates=[t for t in direction['trials'] if abs(t['epsilon_255']-epsilon)<1e-12 and t['phase']!='opposite_control']
                assert candidates,(sample['sample_key'],direction['id'],label,epsilon)
                probe=direction['repeat_high'] if label=='high' else candidates[-1]
                assert abs(probe['epsilon_255']-epsilon)<1e-12
                boundary_rows.append(record(sample,query['baseline'],query['baseline_free'],probe,free,support=direction['id'],position=label,epsilon_255=epsilon))
    assert len(selected)==60 and len(rows)==840
    before={str(p.relative_to(ROOT)):digest(p) for p in files}
    primary=[];paired=[];quadrants=[]
    for eps in (.25,1.):
        for method in METHODS:
            rr=[r for r in rows if r['method']==method and r['epsilon_255']==eps]
            primary.append(dict(method=method,epsilon_255=eps,min_mass=min(r['mass'] for r in rr),**cluster_stats(rr)))
        targeted=[r for r in rows if r['method']=='targeted' and r['epsilon_255']==eps]
        for label,predicate in [('감소',lambda r:r['gain']>0),('동일',lambda r:r['gain']==0),('증가',lambda r:r['gain']<0)]:
            rr=[r for r in targeted if predicate(r)]
            quadrants.append(dict(epsilon_255=eps,group=label,queries=len(rr),target_top1=sum(int(r['target_top1']) for r in rr),iou_improved=sum(int(r['iou_improved']) for r in rr),mean_iou_delta=float(np.mean([r['iou_delta'] for r in rr])) if rr else None,mean_gain=float(np.mean([r['gain'] for r in rr])) if rr else None))
        for method in ('opposite','random'):
            other={r['sample_key']:r for r in query_means([r for r in rows if r['method']==method and r['epsilon_255']==eps])}
            diff=[dict(r,**{k:r[k]-other[r['sample_key']][k] for k in FIELDS}) for r in targeted]
            paired.append(dict(epsilon_255=eps,comparison='targeted − '+method,**cluster_stats(diff)))
    boundaries=[]
    for support in ('all_video','top_frame_pair','top_spatial_cell'):
        for position in ('low','high','limit'):
            rr=[r for r in boundary_rows if r['support']==support and r['position']==position]
            if rr:boundaries.append(dict(support=support,position=position,**cluster_stats(rr)))
    summary=dict(queries=60,videos=len({s['row']['video_path'] for s in selected}),trial_count=len(rows),primary=primary,paired=paired,targeted_groups=quadrants,boundaries=boundaries,bootstrap=dict(seed=42,repeats=1000,unit='video_cluster_stratified_dataset',random='average_within_query'))
    save(ART/'metrics/entropy_summary.json',summary)
    save(ART/'metrics/entropy_per_probe.json',dict(fixed_budget=rows,adaptive_boundary=boundary_rows))
    assert all(digest(ROOT/p)==value for p,value in before.items())
    save(ART/'metadata/entropy_sources.json',dict(sources=before,generator_sha256=digest(Path(__file__)),entropy='original_fixed_top30_renormalized',no_new_inference=True))
    save(ART/'validation/entropy_checks.json',dict(passed=True,queries=60,fixed_budget_pairs=840,source_unchanged=True,fixed_support_ids=True,probability_logit_agreement=True,free_probe_epsilon_matched=True,baseline_match=True,random_repeats_within_query=True))
    out=ART/'figures';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,eps in zip(axes,(.25,1.)):
        for ds,color in [('activitynet-timelens','#247ba0'),('charades-timelens','#c75a38'),('qvhighlights-timelens','#55885b')]:
            rr=[r for r in rows if r['method']=='targeted' and r['epsilon_255']==eps and r['dataset']==ds]
            ax.scatter([r['gain'] for r in rr],[r['iou_delta'] for r in rr],s=32,alpha=.7,label=ds.split('-')[0],c=color)
        ax.axhline(0,color='#777',linewidth=.7);ax.axvline(0,color='#777',linewidth=.7);ax.set_title(f'Targeted: epsilon x 255 = {eps}');ax.set_xlabel('H baseline - H perturbed (nat)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Free-decoding IoU change');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'entropy_gain_iou.svg');plt.close(fig)
    render(summary)
    for row in primary:
        if row['method'] in ('targeted','opposite','random'):print(row['epsilon_255'],row['method'],{k:row['metrics'][k]['mean'] for k in ('gain','h_decreased','target_top1','iou_delta','iou_improved')},flush=True)
    print('Saved '+str(RESULT/'APPENDIX_ENTROPY.md'),flush=True)
    return summary


if __name__=='__main__':build()
