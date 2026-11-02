from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Paired distribution shape analysis using the saved plotted temporal curves.

Run: .venv/bin/python analyze_distribution.py
Writes only results/distribution_analysis/. No model inference is required.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = GENERATION
OUT = relocate_path(RESULTS / 'distribution_analysis')
DATASETS = ['Charades-STA', 'QVHighlights', 'ActivityNet-Captions']
SEED = 20260929


def compare(qraw, araw, t, duration):
    assert len(t) >= 2 and np.all(np.diff(t) > 0)
    assert np.all(np.isfinite(qraw)) and np.all(np.isfinite(araw))
    assert np.all(qraw >= 0) and np.all(araw >= 0)
    q, a = qraw / qraw.sum(), araw / araw.sum()
    k = max(1, int(np.ceil(.2 * len(t))))
    # Stable sorting supplies a reproducible tie rule for top-k; Spearman averages ties.
    qi = np.argsort(-q, kind='stable')[:k]
    ai = np.argsort(-a, kind='stable')[:k]
    dq, da = np.diff(q), np.diff(a)
    valid = (dq != 0) & (da != 0)
    chance_slope = ((dq[valid] > 0).mean() * (da[valid] > 0).mean()
                    + (dq[valid] < 0).mean() * (da[valid] < 0).mean()) if valid.any() else np.nan
    rho = float(spearmanr(q, a).statistic) if np.ptp(q) and np.ptp(a) else np.nan
    overlap = len(set(qi) & set(ai)) / k
    return dict(
        bins=len(t), top_k=k,
        query_video_mass=float(qraw.sum()), answer_video_mass=float(araw.sum()),
        video_mass_ratio=float(araw.sum() / qraw.sum()),
        tv=float(np.abs(q-a).sum()/2),
        js_divergence_bits=float(jensenshannon(q, a, base=2)**2),
        cosine=float(q @ a / (np.linalg.norm(q)*np.linalg.norm(a))),
        spearman=rho,
        top20_overlap=overlap, top20_random_expectation=k/len(t),
        top20_excess=overlap-k/len(t),
        peak_same_bin=int(q.argmax()==a.argmax()),
        peak_shift_fraction=float(abs(t[q.argmax()]-t[a.argmax()])/duration),
        wasserstein_duration=float(wasserstein_distance(t, t, q, a)/duration),
        centroid_shift_fraction=float((a-q)@t/duration),
        query_early_peak=int(t[q.argmax()] < duration*.1),
        answer_early_peak=int(t[a.argmax()] < duration*.1),
        adjacent_slope_agreement=float((np.sign(dq[valid])==np.sign(da[valid])).mean()) if valid.any() else np.nan,
        adjacent_slope_random_expectation=float(chance_slope),
        adjacent_slope_excess=float((np.sign(dq[valid])==np.sign(da[valid])).mean()-chance_slope) if valid.any() else np.nan,
        query_normalized_entropy=float(-np.sum(q[q>0]*np.log(q[q>0]))/np.log(len(q))),
        answer_normalized_entropy=float(-np.sum(a[a>0]*np.log(a[a>0]))/np.log(len(a))),
    )


def summary(rows, keys):
    rng = np.random.default_rng(SEED)
    result = {'n': len(rows)}
    for key in keys:
        values = np.array([r[key] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            result[key] = {'n': 0}
            continue
        bootstrap = values[rng.integers(0, len(values), (4000, len(values)))]
        result[key] = dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
            q25=float(np.quantile(values,.25)), q75=float(np.quantile(values,.75)),
            mean_ci95=np.quantile(bootstrap.mean(1),[.025,.975]).tolist(),
            median_ci95=np.quantile(np.median(bootstrap,axis=1),[.025,.975]).tolist())
    result['spearman_positive_count'] = sum(r['spearman'] > 0 for r in rows)
    result['spearman_ge_05_count'] = sum(r['spearman'] >= .5 for r in rows)
    result['spearman_ge_07_count'] = sum(r['spearman'] >= .7 for r in rows)
    return result


def rebin(raw, edges, new_edges):
    overlap = np.maximum(0, np.minimum(new_edges[1:,None],edges[None,1:])
                         - np.maximum(new_edges[:-1,None],edges[None,:-1]))
    result = (overlap/np.diff(edges)) @ (raw/raw.sum())
    assert np.isclose(result.sum(),1)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    samples = json.loads((relocate_path(RESULTS/'samples.json')).read_text())
    previous = {r['sample_key']:r for r in json.loads((relocate_path(RESULTS/'sample_metrics.json')).read_text())}
    assert len(samples) == 300
    assert len({(x['dataset'], x['vid']) for x in samples}) == 300
    rows, curves, hashes = [], {}, {}
    for sample in samples:
        key = sample['sample_key']; folder = relocate_path(RESULTS/ sample["dataset"] / key)
        meta = json.loads((relocate_path(folder/'metadata.json')).read_text())
        path = relocate_path(folder/'temporal_comparison.csv')
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = np.genfromtxt(path, delimiter=',', names=True)
        assert np.allclose(data['query_video_conditional'], relocate_path(data['query']/data['query'].sum()), atol=1e-10, rtol=0)
        t, duration = data['time_seconds'], meta['duration_video']
        assert np.allclose(t, meta['sampled_timestamps'])
        edges = np.r_[0, (t[:-1]+t[1:])/2, duration]
        assert np.all(np.diff(edges)>0)
        early_fraction = np.maximum(0,np.minimum(edges[1:],duration*.1)-edges[:-1])/np.diff(edges)
        curves[key] = (t/duration, data['query_video_conditional'], data['answer_video_conditional'])
        for scope in ['answer','predictive','coordinates']:
            if scope not in data.dtype.names:
                continue
            qraw, araw = data['query'], data[scope]
            if not np.isfinite(araw).all():
                continue
            assert np.allclose(data[scope+'_video_conditional'],araw/araw.sum(),atol=1e-10)
            old = previous[key]['scopes'][scope]
            base = dict(sample_key=key,dataset=sample['dataset'],scope=scope,
                        truncated=previous[key]['truncated'],gt_mass_gain=old['gt_mass_gain_over_query'])
            for region in ['full','after_first_10pct']:
                mask = np.ones(len(t),dtype=bool) if region=='full' else t>=duration*.1
                if mask.sum()<3:
                    continue
                metrics = compare(qraw[mask],araw[mask],t[mask],duration)
                if region=='full':
                    assert np.isclose(metrics['tv'],old['total_variation_from_query'],atol=1e-10,rtol=0)
                    assert np.isclose(metrics['answer_video_mass'],old['total_video_attention'],atol=1e-10,rtol=0)
                    metrics['query_early_mass'] = float((qraw/qraw.sum())@early_fraction)
                    metrics['answer_early_mass'] = float((araw/araw.sum())@early_fraction)
                    metrics['early_mass_gain'] = metrics['answer_early_mass']-metrics['query_early_mass']
                rows.append(dict(base,region=region,**metrics))
    keys = list(compare(np.array([1.,2.,3.]),np.array([2.,1.,3.]),np.arange(3),3))
    fullkeys = keys+['query_early_mass','answer_early_mass','early_mass_gain','gt_mass_gain']
    summaries = {}
    for dataset in DATASETS+['All']:
        summaries[dataset] = {}
        for scope in ['answer','predictive','coordinates']:
            for region in ['full','after_first_10pct']:
                group = [r for r in rows if (dataset=='All' or r['dataset']==dataset) and r['scope']==scope and r['region']==region]
                summaries[dataset][scope+'__'+region] = summary(group,fullkeys if region=='full' else keys)
        group = [r for r in rows if (dataset=='All' or r['dataset']==dataset) and r['scope']=='answer' and r['region']=='full' and not r['truncated']]
        summaries[dataset]['answer__excluding_truncated'] = summary(group,fullkeys)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (relocate_path(OUT/'sample_metrics.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    write_result_text((relocate_path(OUT/'summary.json')), json.dumps(summaries,indent=2,allow_nan=False)+'\n')
    hashes['analyze_distribution.py'] = hashlib.sha256(relocate_path(Path(__file__)).read_bytes()).hexdigest()
    write_result_text((relocate_path(OUT/'source_sha256.json')), json.dumps(hashes,indent=2)+'\n')
    plot(rows, samples, curves)
    report(summaries, rows)
    print(json.dumps({d:{k:summaries[d]['answer__full'][k] for k in ['tv','spearman','top20_overlap','peak_same_bin','early_mass_gain','centroid_shift_fraction']} for d in DATASETS},indent=2))


def plot(rows, samples, curves):
    fig, axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    panels = [('tv','Total variation (0 = identical)'),('spearman','Spearman rank correlation'),
              ('top20_overlap','Overlap of top 20% bins'),('wasserstein_duration','Wasserstein / video duration'),
              ('early_mass_gain','First 10% mass: answer - query'),('adjacent_slope_agreement','Adjacent rise/fall agreement')]
    rng=np.random.default_rng(SEED)
    for ax,(metric,title) in zip(axes.flat,panels):
        for i,dataset in enumerate(DATASETS):
            group=[r for r in rows if r['dataset']==dataset and r['scope']=='answer' and r['region']=='full']
            y=[r[metric] for r in group]
            ax.scatter(i+rng.uniform(-.17,.17,len(y)),y,s=11,alpha=.45)
            ax.plot([i-.22,i+.22],[np.median(y)]*2,color='black',lw=2)
            if metric in ['spearman','top20_overlap']:
                cropped=[r[metric] for r in rows if r['dataset']==dataset and r['scope']=='answer' and r['region']=='after_first_10pct']
                ax.plot(i,np.median(cropped),marker='D',color='#bc4a20',label='Median excluding first 10%' if i==0 else None)
        ax.set(title=title,xticks=range(3),xticklabels=['Charades','QVHighlights','ActivityNet'])
        ax.grid(axis='y',alpha=.2)
        if metric in ['early_mass_gain','spearman']: ax.axhline(0,color='gray',ls=':',lw=1)
        if metric in ['spearman','top20_overlap']: ax.legend(fontsize=7)
    fig.suptitle('Query vs generated answer: paired temporal distributions\n100 samples per dataset; layers 8-21; black bars = medians')
    fig.savefig(relocate_path(OUT/'distribution_summary.png'),dpi=180);fig.savefig(relocate_path(OUT/'distribution_summary.pdf'));plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    target=np.linspace(0,1,51)
    for ax,dataset in zip(axes,DATASETS):
        rebinned=[]
        for sample in samples:
            if sample['dataset']!=dataset:continue
            t,q,a=curves[sample['sample_key']]
            edges=np.r_[0,(t[:-1]+t[1:])/2,1]
            rebinned.append([rebin(q,edges,target),rebin(a,edges,target)])
        means=np.mean(rebinned,axis=0)
        for j,label in enumerate(['Query','Generated answer']):
            ax.plot((target[:-1]+target[1:])/2,means[j]/np.diff(target),label=label)
        ax.axvspan(0,.1,color='gray',alpha=.12)
        ax.set(title=dataset,xlabel='Relative video time',ylabel='Mean probability density')
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Equal-weight sample averages; 50 common bins; shaded = first 10%')
    fig.savefig(relocate_path(OUT/'relative_time_profiles.png'),dpi=180);fig.savefig(relocate_path(OUT/'relative_time_profiles.pdf'));plt.close(fig)


def report(s,rows):
    lines=['# Query–generated answer attention 분포 비교','',
        '**시간별 attention의 상대적 순위는 상당 부분 유지되지만, 확률 질량 배분과 최고점은 달라진다. 답변에서는 영상 초반 집중이 더 강해진다.** 세 데이터셋의 TV 중앙값은 0.160–0.225, Spearman 중앙값은 0.653–0.749다. 초반 10%를 제외해도 Spearman 0.600–0.691이 남는다. 이는 평균 곡선의 관찰이며, 개별 토큰의 동작이나 인과관계를 뜻하지 않는다.','',
        '기존 실제 greedy 생성 300개(데이터셋당 100개)의 저장된 곡선을 재계산했다. 레이어 8–21, 모든 head 및 해당 토큰을 평균한 attention이다. 주 분석은 답변 토큰을 입력으로 처리하는 행, EOS 제외 답변 전체다. 영상 내 합을 1로 맞춘 뒤 같은 샘플끼리 비교한다. 각 샘플을 동일 가중치로 집계한다. 모델 재실행은 하지 않았다.','',
        '## 분포 차이와 경향성','',
        '아래 연속형 수치는 샘플별 지표의 중앙값이다. Peak 일치는 샘플 비율이다.','',
        '| Dataset | TV | JS divergence (bits) | Cosine | Spearman | Top-20% overlap | Same peak | Wasserstein / duration |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        a=s[d]['answer__full'];m=lambda k:a[k]['median']
        lines.append(f"| {d} | {m('tv'):.4f} | {m('js_divergence_bits'):.4f} | {m('cosine'):.4f} | {m('spearman'):.4f} | {m('top20_overlap'):.1%} | {a['peak_same_bin']['mean']:.1%} | {m('wasserstein_duration'):.1%} |")
    lines += ['',
        '- TV = ½Σ|p−q|. 0은 동일, 1은 질량이 완전히 분리됨. 1−TV는 두 분포가 공유하는 확률 질량이다.',
        '- JS는 base-2 Jensen–Shannon **divergence**, 범위 0–1. JS distance의 제곱이다.',
        '- Spearman은 시간 bin별 attention 크기의 순위 상관(-1–1)이다. Pearson 계수는 계산하지 않았다.',
        '- Top-20%는 k=ceil(0.2×bin 수)인 상위 bin 두 집합의 교집합/k. 독립 무작위 선택 기대치는 k/bin 수(약 20%)다. 시계열 자기상관을 고려한 유의성 검정은 아니다.',
        '- Wasserstein은 timestamp 지점에 질량을 놓았을 때 필요한 평균 이동 거리를 영상 길이로 나눈 값이다.',
        '- 동일 peak는 정확히 같은 bin에서 최댓값을 갖는 비율이다. 단순 영상 초반 쏠림도 유사도를 높일 수 있다.','',
        '## 영상 초반 10%를 제외한 민감도 분석','',
        'timestamp가 영상 길이의 10% 미만인 bin을 제거하고 남은 질량을 각각 다시 정규화했다. 경계에 걸친 bin을 분할하지 않는다.','',
        '| Dataset | Full → cropped Spearman | Full → cropped Top-20% overlap | Cropped TV | Positive rank correlation | Full rho ≥ 0.5 | Full rho ≥ 0.7 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        a,b=s[d]['answer__full'],s[d]['answer__after_first_10pct']
        lines.append(f"| {d} | {a['spearman']['median']:.3f} → {b['spearman']['median']:.3f} | {a['top20_overlap']['median']:.1%} → {b['top20_overlap']['median']:.1%} | {b['tv']['median']:.3f} | {b['spearman_positive_count']}/{b['n']} | {a['spearman_ge_05_count']}/{a['n']} | {a['spearman_ge_07_count']}/{a['n']} |")
    lines += ['', '## 변화 방향', '',
        '영상 총량과 early peak 비율은 평균, 나머지는 paired 변화량의 중앙값이다. 첫 10% 질량은 timestamp 중점 bin 안에서 균일한 질량을 가정해 경계 중첩을 반영한다.', '',
        '| Dataset | Query → answer video mass | Query → answer early peak | First 10% mass gain | Centroid shift / duration | Adjacent rise/fall agreement | GT mass gain |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        a=s[d]['answer__full']
        lines.append(f"| {d} | {a['query_video_mass']['mean']:.1%} → {a['answer_video_mass']['mean']:.1%} | {a['query_early_peak']['mean']:.1%} → {a['answer_early_peak']['mean']:.1%} | {a['early_mass_gain']['median']*100:+.2f}%p | {a['centroid_shift_fraction']['median']*100:+.2f}% | {a['adjacent_slope_agreement']['median']:.1%} | {a['gt_mass_gain']['median']*100:+.2f}%p |")
    lines += ['', '음수 중심 이동은 영상 앞쪽으로 이동함을 뜻한다. 인접 bin 상승/하강 일치는 0 변화가 있는 쌍을 제외한 부호 일치율이다. 무작위 기대치는 두 곡선의 상승/하강 비율의 곱으로 계산해 JSON에 저장했다.','',
        '## 불확실성과 토큰 선택 민감도','',
        '샘플 단위 bootstrap 4,000회, seed=20260929. 각 데이터셋의 100개 샘플은 영상 ID가 모두 다르다. 아래는 중앙값의 95% percentile 구간이다. 샘플 변동성을 나타내며, 모델·프롬프트·다른 seed로의 일반화를 보장하지 않는다.','',
        '| Dataset | TV median 95% CI | Spearman median 95% CI | Predictive TV / rho | Timestamp digits n; TV / rho | Non-truncated n; TV / rho |',
        '|---|---|---|---|---|---|']
    for d in DATASETS:
        a=s[d]['answer__full'];p=s[d]['predictive__full'];c=s[d]['coordinates__full'];n=s[d]['answer__excluding_truncated']
        ci=lambda key:'–'.join(f'{v:.3f}' for v in a[key]['median_ci95'])
        lines.append(f"| {d} | {ci('tv')} | {ci('spearman')} | {p['tv']['median']:.3f} / {p['spearman']['median']:.3f} | {c['n']}; {c['tv']['median']:.3f} / {c['spearman']['median']:.3f} | {n['n']}; {n['tv']['median']:.3f} / {n['spearman']['median']:.3f} |")
    lines += ['',
        'Predictive는 각 답변 토큰을 생성하기 직전 행이다. Timestamp digits는 구간 파싱 성공 샘플만 포함한다. 두 경우 모두 동일 샘플의 query와 비교했다. 레이어/head/답변 토큰을 평균한 곡선에 대한 결과이므로 각 개별 토큰·head까지 동일 경향이라고 해석할 수 없다. 영상 attention 총량 감소에는 답변이 볼 수 있는 key 집합 증가가 포함되므로 시각 정보 사용량의 인과적 감소라고 단정하지 않는다.','',
        '## 검증과 산출물','',
        '모든 샘플·토큰 그룹에서 재계산한 TV와 영상 총량이 기존 sample_metrics.json과 atol=1e-10으로 일치했다. timestamp와 정규화된 곡선도 검증했다. 원본 CSV 및 분석 스크립트 SHA256을 저장했다.','',
        '- [샘플 분포 그림](distribution_summary.png) / [PDF](distribution_summary.pdf)',
        '- [상대 시간 평균 분포](relative_time_profiles.png) / [PDF](relative_time_profiles.pdf)',
        '- [샘플별 수치](sample_metrics.csv) / [집계 및 CI](summary.json)',
        '- 재실행: 실험 폴더에서 `.venv/bin/python analyze_distribution.py`','']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines))


if __name__=='__main__':
    main()
