from result_layout import relocate_path, write_result_text, result_glob
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Reproducible distribution metrics, plots, CSVs, and a self-contained HTML report."""
from pathlib import Path
import argparse
import csv
import html
import json
import hashlib
import numpy as np
from scipy.special import softmax
from scipy.optimize import brentq
from scipy.stats import spearmanr, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS = HIDDEN
DATASETS = ['Charades-STA','QVHighlights','ActivityNet-Captions']
TEMPERATURES = [.01,.02,.05,.1,.2,1.]
COLORS = ['#2563eb','#e87924','#159570']


def write_json(path, obj):
    write_result_text(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def entropy(p):
    p = p[p > 0]
    return float(-p @ np.log(p))


def compare(scores, raw_attention, timestamps, duration, tau):
    scores, raw_attention, timestamps = [np.asarray(x,dtype=np.float64) for x in (scores,raw_attention,timestamps)]
    assert len(scores)>1 and np.isfinite(scores).all() and np.isfinite(raw_attention).all()
    assert (raw_attention>=0).all() and raw_attention.sum()>0
    p, q = softmax(scores/tau), raw_attention/raw_attention.sum()
    k = max(1,int(np.ceil(.2*len(p))))
    top_p, top_q = np.argsort(-scores,kind='stable')[:k], np.argsort(-q,kind='stable')[:k]
    rho = float(spearmanr(scores,q).statistic) if np.ptp(scores)>0 and np.ptp(q)>0 else None
    return dict(spearman=rho, tv=float(np.abs(p-q).sum()/2),
        js_bits=float(jensenshannon(p,q,base=2)**2), top20_overlap=len(set(top_p)&set(top_q))/k,
        top20_random=k/len(p), peak_same=int(scores.argmax()==q.argmax()),
        peak_shift_fraction=float(abs(timestamps[scores.argmax()]-timestamps[q.argmax()])/duration),
        wasserstein_fraction=float(wasserstein_distance(timestamps,timestamps,p,q)/duration),
        similarity_entropy=entropy(p)/np.log(len(p)),attention_entropy=entropy(q)/np.log(len(q)),
        similarity_first_bin=float(p[0]),attention_first_bin=float(q[0]),
        cosine_min=float(scores.min()),cosine_max=float(scores.max()),cosine_std=float(scores.std()))


def fractions(edges,windows):
    # Merge overlapping intervals before integrating bin mass.
    merged=[]
    for start,end in sorted(windows):
        start,end=max(0,start),min(edges[-1],end)
        if end<=start: continue
        if merged and start<=merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
        else: merged.append([start,end])
    return sum((np.maximum(0,np.minimum(edges[1:],e)-np.maximum(edges[:-1],s))/np.diff(edges)
                for s,e in merged),np.zeros(len(edges)-1))


def summarize(rows):
    rng=np.random.default_rng(20260930)
    result={'n':len(rows)}
    for metric in ['spearman','tv','js_bits','top20_overlap','top20_random','peak_same',
                   'peak_shift_fraction','wasserstein_fraction','similarity_entropy','attention_entropy',
                   'similarity_first_bin','attention_first_bin','similarity_early_mass','attention_early_mass',
                   'similarity_gt_mass','attention_gt_mass','cosine_min','cosine_max','cosine_std']:
        x=np.array([r[metric] for r in rows if r.get(metric) is not None],dtype=float)
        if not len(x): continue
        means=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
        result[metric]={'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
                        'mean_ci95':np.quantile(means,[.025,.975]).tolist()}
    return result


def write_csv(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as file:
        writer=csv.DictWriter(file,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def sample_plot(folder,meta,data,metrics):
    t=np.asarray(meta['sampled_timestamps']); duration=meta['duration_video']
    sim=data['cosine_query_frame'][8:22].astype(float).mean((0,1))
    att=data['attention_query_frame'][8:22].astype(float).mean((0,1)); att/=att.sum()
    p=softmax(sim/.1)
    heat_height=max(1.8,len(meta['query_tokens'])*.14)
    fig,axes=plt.subplots(4,1,figsize=(11,5.5+2*heat_height),sharex=True,
                          gridspec_kw={'height_ratios':[2,1,heat_height,heat_height]})
    axes[0].plot(t,att,label='Query attention (video-normalized)',color=COLORS[0])
    axes[0].plot(t,p,label='Hidden cosine → softmax (tau=0.1)',color=COLORS[1])
    for ax in axes:
        for j,(s,e) in enumerate(meta['gt_windows']):
            ax.axvspan(s,e,color=COLORS[2],alpha=.13,
                       label='Ground truth' if ax is axes[0] and j==0 else None)
        ax.set_xlim(0,max(duration,max(e for s,e in meta['gt_windows'])))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel('Probability / temporal bin')
    axes[0].set_title(f"{meta['sample_key']} · Spearman={metrics['spearman']:.3f}, TV={metrics['tv']:.3f}, JS={metrics['js_bits']:.3f} bits\n{meta['query']}",fontsize=10,wrap=True)
    axes[1].plot(t,sim,color=COLORS[1]);axes[1].set_ylabel('Mean pair cosine')
    c=data['cosine_query_frame'][8:22].mean(0)
    edges=np.r_[0,(t[:-1]+t[1:])/2,duration]
    qa=data['attention_query_frame'][8:22].mean(0)
    qa=qa/qa.sum(-1,keepdims=True)
    qs=softmax(c/.1,axis=-1)
    vmax=max(qa.max(),qs.max())
    for ax,values,label in [(axes[2],qs,'Hidden similarity'),(axes[3],qa,'Attention')]:
        im=ax.pcolormesh(edges,np.arange(len(c)+1),values,cmap='viridis',shading='flat',vmin=0,vmax=vmax)
        ax.set_yticks(np.arange(len(c))+.5,meta['query_tokens'],fontsize=6)
        ax.set_ylabel(label+'\nQuery token')
        fig.colorbar(im,ax=ax,label='Probability / bin',pad=.01)
    axes[3].set_xlabel('Video time (seconds)')
    fig.tight_layout()
    # Reserve the colorbar gutter in the line plots too: seconds align vertically.
    pos=axes[2].get_position()
    for ax in axes[:2]:
        previous=ax.get_position()
        ax.set_position([pos.x0,previous.y0,pos.width,previous.height])
    fig.savefig(relocate_path(folder/'comparison.png'),dpi=140)
    plt.close(fig)
    rows=[dict(time_seconds=float(x),cosine=float(c),similarity_probability=float(s),attention_probability=float(a))
          for x,c,s,a in zip(t,sim,p,att)]
    write_csv(relocate_path(folder/'curves.csv'),rows)
    write_result_text((relocate_path(folder/'plot_version.txt')), '5\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    samples=json.loads((relocate_path(RESULTS/'samples.json')).read_text())
    available=[s for s in samples if (relocate_path(RESULTS/s['dataset']/s['sample_key']/'validation.json')).exists()]
    assert args.partial or len(available)==len(samples),f'Only {len(available)}/{len(samples)} complete'
    assert available
    rows=[];layer_rows=[];token_rows=[];query_rows=[];matched_rows=[];validations=[];curves=[]
    for sample in available:
        folder=relocate_path(RESULTS/sample['dataset']/sample['sample_key'])
        meta=json.loads((relocate_path(folder/'metadata.json')).read_text())
        validations.append(json.loads((relocate_path(folder/'validation.json')).read_text()))
        with np.load(relocate_path(folder/'distributions.npz')) as z:
            data={k:z[k] for k in z.files}
        t=data['sampled_timestamps'];duration=meta['duration_video']
        edges=np.r_[0,(t[:-1]+t[1:])/2,duration]
        assert np.all(np.diff(edges)>0)
        early=fractions(edges,[[0,.1*duration]]);gt=fractions(edges,meta['gt_windows'])
        base=dict(dataset=sample['dataset'],sample_key=sample['sample_key'])
        for name,band in [('L8_21',slice(8,22)),('all_layers',slice(None)),('last_layer',slice(35,36))]:
            c=data['cosine_query_frame'][band].astype(float).mean((0,1))
            a=data['attention_query_frame'][band].astype(float).mean((0,1))
            for tau in TEMPERATURES:
                for region in ['full','after_first_10pct']:
                    mask=np.ones(len(t),dtype=bool) if region=='full' else t>=duration*.1
                    if mask.sum()<3:continue
                    values=compare(c[mask],a[mask],t[mask],duration,tau)
                    if region=='full':
                        p,q=softmax(c/tau),a/a.sum()
                        values.update(similarity_early_mass=float(p@early),attention_early_mass=float(q@early),
                                      similarity_gt_mass=float(p@gt),attention_gt_mass=float(q@gt))
                    row=dict(base,scope=name,temperature=tau,region=region,**values);rows.append(row)
                    if name=='L8_21' and tau==.1 and region=='full':
                        primary=row
            if name=='L8_21':
                curves.append((meta,c,a/a.sum()))
                # Diagnostic: match entropy, without optimizing a distance or ranks.
                target=entropy(a/a.sum())
                log_tau=brentq(lambda x:entropy(softmax(c/np.exp(x)))-target,-20,8)
                matched_tau=float(np.exp(log_tau))
                matched=compare(c,a,t,duration,matched_tau)
                assert abs(matched['similarity_entropy']-matched['attention_entropy'])<1e-9
                matched_rows.append(dict(base,temperature=matched_tau,**matched))
                tc=data['cosine_video_token'][band].astype(float).mean(0)
                ta=data['attention_video_token'][band].astype(float).mean(0)
                tt=np.repeat(t,meta['spatial_tokens_per_bin'])
                token_rows.append(dict(base,**compare(tc,ta,tt,duration,.1)))
        for layer in range(36):
            c=data['cosine_query_frame'][layer].astype(float).mean(0)
            a=data['attention_query_frame'][layer].astype(float).mean(0)
            layer_rows.append(dict(base,layer=layer,**compare(c,a,t,duration,.1)))
        cq=data['cosine_query_frame'][8:22].astype(float).mean(0)
        aq=data['attention_query_frame'][8:22].astype(float).mean(0)
        for i,token in enumerate(meta['query_tokens']):
            query_rows.append(dict(base,query_token_index=i,query_token=token,
                                   sequence_position=meta['query_positions'][i],
                                   **compare(cq[i],aq[i],t,duration,.1)))
        version=relocate_path(folder/'plot_version.txt')
        if not version.exists() or version.read_text().strip()!='5':
            sample_plot(folder,meta,data,primary)
    write_csv(relocate_path(RESULTS/'metrics.csv'),rows);write_csv(relocate_path(RESULTS/'layer_metrics.csv'),layer_rows)
    write_csv(relocate_path(RESULTS/'video_token_metrics.csv'),token_rows)
    write_csv(relocate_path(RESULTS/'entropy_matched_metrics.csv'),matched_rows)
    write_csv(relocate_path(RESULTS/'query_token_metrics.csv'),query_rows)
    summary={}
    for dataset in DATASETS+['All']:
        group=[r for r in rows if dataset=='All' or r['dataset']==dataset]
        if not group:continue
        summary[dataset]={}
        for scope in ['L8_21','all_layers','last_layer']:
            for tau in TEMPERATURES:
                for region in ['full','after_first_10pct']:
                    g=[r for r in group if r['scope']==scope and r['temperature']==tau and r['region']==region]
                    summary[dataset][f'{scope}__tau{tau}__{region}']=summarize(g)
        summary[dataset]['video_tokens_L8_21_tau0.1']=summarize([r for r in token_rows if dataset=='All' or r['dataset']==dataset])
        summary[dataset]['entropy_matched']=summarize([r for r in matched_rows if dataset=='All' or r['dataset']==dataset])
    write_json(relocate_path(RESULTS/'summary.json'),summary)
    write_json(relocate_path(RESULTS/'validation_summary.json'),dict(n_complete=len(available),n_expected=len(samples),
        all_attention_bit_identical=all(v['prior_attention_bit_identical'] for v in validations),
        all_inputs_identical=all(v['input_positions_and_text_identical'] for v in validations),
        max_attention_difference=max(v['max_abs_attention_difference'] for v in validations),
        extraction_seconds=sum(v['elapsed_seconds'] for v in validations),
        max_gpu_allocated_gib=max(v['peak_gpu_allocated_gib'] for v in validations)))
    primary=[r for r in rows if r['scope']=='L8_21' and r['temperature']==.1 and r['region']=='full']
    present=[d for d in DATASETS if d in summary]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,metric,label in zip(axes.flat,['spearman','tv','js_bits','top20_overlap','similarity_entropy','attention_entropy'],
                             ['Rank agreement (Spearman)','Total variation distance','JS divergence (bits)','Top 20% overlap','Hidden similarity entropy','Attention entropy']):
        values=[[r[metric] for r in primary if r['dataset']==d] for d in present]
        ax.boxplot(values,tick_labels=[d.split('-')[0] for d in present],showfliers=False)
        for i,v in enumerate(values):ax.scatter(i+1,np.mean(v),marker='D',color=COLORS[i],zorder=3)
        ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2)
        if metric=='spearman':ax.set_ylim(-1.05,1.05);ax.axhline(0,color='gray',linewidth=.7)
        else:ax.set_ylim(-.02,1.02)
    fig.suptitle(f'Output hidden similarity vs query attention · {len(available)} samples · L8–21 · tau=0.1\nDiamonds: means; boxes: medians and quartiles',fontsize=12)
    fig.tight_layout();fig.savefig(relocate_path(RESULTS/'summary.png'),dpi=170);fig.savefig(relocate_path(RESULTS/'summary.pdf'));plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,metric in zip(axes.flat,['spearman','tv','js_bits','top20_overlap']):
        for d,color in zip(present,COLORS):
            y=[np.mean([r[metric] for r in layer_rows if r['dataset']==d and r['layer']==l]) for l in range(36)]
            ax.plot(range(36),y,label=d,color=color)
        ax.axvspan(8,21,color='gray',alpha=.1);ax.set_title(metric);ax.set_xlabel('Decoder layer (0-based)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(relocate_path(RESULTS/'layers.png'),dpi=160);fig.savefig(relocate_path(RESULTS/'layers.pdf'));plt.close(fig)
    # Paged full-cohort curves, with no cherry-picked examples.
    for d in present:
        subset=[c for c in curves if c[0]['dataset']==d]
        with PdfPages(relocate_path(RESULTS/f'{d}_curves.pdf')) as pdf:
            for start in range(0,len(subset),12):
                fig,axes=plt.subplots(4,3,figsize=(15,13))
                for ax in axes.flat:ax.set_visible(False)
                for ax,(m,c,a) in zip(axes.flat,subset[start:start+12]):
                    ax.set_visible(True);t=m['sampled_timestamps']
                    ax.plot(t,a,color=COLORS[0],linewidth=1,label='Attention')
                    ax.plot(t,softmax(c/.1),color=COLORS[1],linewidth=1,label='Hidden similarity')
                    for s,e in m['gt_windows']:ax.axvspan(s,e,color=COLORS[2],alpha=.12)
                    ax.set_title(m['sample_key']+'\n'+m['query'][:72],fontsize=7,wrap=True)
                    ax.set_xlabel('Seconds',fontsize=7);ax.tick_params(labelsize=6)
                axes.flat[0].legend(fontsize=6)
                fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    render_report(summary,primary,len(available))
    write_json(relocate_path(RESULTS/'analysis_provenance.json'),dict(
        temperatures=TEMPERATURES, primary_temperature=.1, entropy_matching='per-sample numerical root for equal entropy',
        bootstrap_seed=20260930, bootstrap_replicates=2000,
        script_sha256=hashlib.sha256(relocate_path(Path(__file__)).read_bytes()).hexdigest(),
        artifact_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(result_glob(RESULTS, '*/*/distributions.npz'))}))
    print(json.dumps({d:summary[d]['L8_21__tau0.1__full'] for d in summary},indent=2))


def render_report(summary,primary,n):
    total=summary['All']['L8_21__tau0.1__full']
    matched=summary['All']['entropy_matched']
    positive=sum(r['spearman']>0 for r in primary)
    lead=(f"전체 평균 Spearman ρ는 {total['spearman']['mean']:.3f}, "
          f"TV는 {total['tv']['mean']:.3f}, JS divergence는 {total['js_bits']['mean']:.3f} bits다. "
          f"양의 순위 상관은 {positive}/{n}개, 최고점 일치는 {total['peak_same']['mean']:.1%}다. "
          f"Entropy를 같게 맞춘 보조 조건에서는 TV {matched['tv']['mean']:.3f}, "
          f"JS {matched['js_bits']['mean']:.3f} bits다.")
    def table(scope):
        lines=['<tr><th>Dataset</th><th>N</th><th>Spearman ρ</th><th>TV</th><th>JS (bits)</th><th>Top-20% overlap</th><th>Peak match</th></tr>']
        for d,s in summary.items():
            s=s[scope]
            lines.append('<tr><td>'+d+'</td><td>'+str(s['n'])+'</td>'+''.join(
                f'<td>{s[k]["mean"]:.3f}<small>95% CI {s[k]["mean_ci95"][0]:.3f}–{s[k]["mean_ci95"][1]:.3f}</small></td>'
                for k in ['spearman','tv','js_bits','top20_overlap','peak_same'])+'</tr>')
        return '<table>'+''.join(lines)+'</table>'
    temp=['<tr><th>Dataset</th><th>τ</th><th>TV</th><th>JS (bits)</th><th>Similarity entropy</th><th>Attention entropy</th></tr>']
    concentration=['<tr><th>Dataset</th><th>Similarity first 10%</th><th>Attention first 10%</th><th>Similarity GT mass</th><th>Attention GT mass</th></tr>']
    for d,s in summary.items():
        for tau in TEMPERATURES:
            v=s[f'L8_21__tau{tau}__full']
            temp.append(f'<tr><td>{d}</td><td>{tau}</td>'+''.join(f'<td>{v[k]["mean"]:.3f}</td>' for k in ['tv','js_bits','similarity_entropy','attention_entropy'])+'</tr>')
        v=s['L8_21__tau0.1__full']
        concentration.append(f'<tr><td>{d}</td>'+''.join(f'<td>{v[k]["mean"]:.3f}</td>' for k in ['similarity_early_mass','attention_early_mass','similarity_gt_mass','attention_gt_mass'])+'</tr>')
    links=[]
    for r in primary:
        key=r['sample_key'];rel=f'{r["dataset"]}/{key}'
        meta=json.loads((relocate_path(RESULTS/rel/'metadata.json')).read_text())
        links.append(f'<tr><td><a href="{rel}/comparison.png">{key}</a></td><td>{html.escape(meta["query"])}</td><td>{r["spearman"]:.3f}</td><td>{r["tv"]:.3f}</td><td><a href="{rel}/curves.csv">CSV</a> · <a href="{rel}/distributions.npz">NPZ</a></td></tr>')
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Hidden similarity vs attention</title>
<style>body{{font:16px/1.65 system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#172338}}table{{border-collapse:collapse;width:100%;margin:16px 0 28px}}th,td{{padding:9px;border-bottom:1px solid #d8e0e8;text-align:left}}th{{background:#eef3f8}}small{{display:block;font-size:10px;color:#617086}}img{{max-width:100%}}code{{background:#eef3f8;padding:2px 5px}}a{{color:#1857a8}}.note{{background:#f0f5fb;padding:18px}}input{{padding:12px;width:90%}}</style>
<h1>Transformer output hidden state 유사도와 attention 비교</h1>
<p>Qwen3-VL-8B-Instruct · 기존 고정 샘플 {n}/300개 · 레이어 8–21 (0-based) · 정답을 입력하지 않은 prefill</p>
<p><b>{lead}</b></p>
<p class="note">각 영상 토큰과 각 이벤트 쿼리 토큰의 <b>동일 레이어 최종 hidden state</b> 사이 cosine을 계산했다.
Attention은 같은 forward의 원래 causal softmax다. 처리한 모든 샘플에서 이전 실험과 attention 배열의 비트 일치를 검증했다.
TV와 JS는 softmax 온도에 의존하므로, 원래 cosine 순위 기반 Spearman 및 상위 프레임 겹침과 함께 해석한다.</p>
<h2>주 결과: 시간 bin별 분포</h2>{table('L8_21__tau0.1__full')}<img src="summary.png">
<p>TV=0, JS=0이면 분포가 같다. 둘 다 최대 1이다. Spearman은 −1에서 1이며 순위가 같으면 1이다.
Top-20% overlap은 각 분포의 상위 ceil(0.2T)개 bin 중 교집합 비율이다. 무작위 기대치는 약 0.2(정확한 값은 CSV)다.
각 데이터셋의 샘플 평균과 샘플 단위 bootstrap 95% CI(2,000회)를 제시한다. CI는 이 고정 표본의 탐색적 불확실성이며 레이어를 독립 표본으로 취급하지 않는다.</p>
<h2>정의와 집계 순서</h2>
<p><code>s[l,q,t] = mean_spatial cosine(h[l,q], h[l,t,spatial])</code>.<br>
<code>s[t] = mean_layers,query s[l,q,t]</code>, <code>p[t] = softmax(s[t]/τ)</code>.<br>
<code>a[t] = mean_layers,heads,query sum_spatial attention[l,head,q,t,spatial]</code>, <code>q[t] = a[t]/sum(a)</code>.</p>
<p>벡터를 평균한 뒤 cosine을 계산하지 않고 토큰 쌍의 cosine을 먼저 계산한다. 모든 영상 토큰과 쿼리 토큰을 사용한다.
주 온도 τ=0.1은 GT나 attention 일치도를 보고 최적화하지 않았다. 아래 τ=0.01, 0.02, 0.05, 0.2, 1.0 결과도 함께 제공한다.
공간 토큰 수는 한 샘플의 모든 시간 bin에서 같다. 프레임이라는 표현은 Qwen processor의 temporal patch bin을 뜻하며 원본 영상의 개별 프레임과 동일하지 않다.</p>
<p>Output은 attention residual + MLP residual을 마친 decoder block 출력이다. 모델 최종 RMSNorm 전이며,
레이어 0–2에서는 block 밖에서 더해지는 DeepStack 주입 전이다. 주 범위 8–21에는 이 구분이 영향을 주지 않는다.
BF16 모델 출력으로 FP32 cosine을 계산했고, native eager의 독립적인 head를 나누어 계산해 메모리 사용을 줄였다.
처음에는 모든 입력을 4개 head씩 계산했고, 이후 4,800 tokens 이하 입력은 전체 head를 한 번에 계산했다.
두 실행 방식 모두 기존 저장 attention과 비트 단위로 일치함을 샘플마다 검증했다.
쿼리는 Event: 뒤 이벤트 문장만 포함한다. 영상 토큰은 causal 순서상 뒤의 쿼리를 볼 수 없고 쿼리는 앞의 영상을 볼 수 있으므로 이 관측은 인과적 중요도 평가가 아니다.</p>
<h2>온도 민감도</h2><table>{''.join(temp)}</table>
<h2>분포의 집중도를 맞춘 보조 비교</h2>{table('entropy_matched')}
<p>샘플마다 hidden similarity의 entropy가 attention entropy와 같아지는 온도를 수치적으로 찾았다.
TV/JS가 단순히 집중도 차이에서 비롯되는지 확인하는 보조 분석이며, attention을 참조해 맞춘 조건이다.
GT나 TV/JS 자체를 최적화하지 않았으며 순위는 바뀌지 않는다. <a href="entropy_matched_metrics.csv">샘플별 온도와 수치</a>.</p>
<h2>초반 10%를 제외하고 다시 정규화한 비교</h2>{table('L8_21__tau0.1__after_first_10pct')}
<p>대표 timestamp가 영상 길이의 10% 미만인 bin을 제거했다. 시간에 따른 attention 편향만으로 주 결과가 설명되는지 확인하는 보조 분석이다.</p>
<h2>공간 토큰을 합치기 전 비교</h2>{table('video_tokens_L8_21_tau0.1')}
<p>레이어·쿼리 토큰 평균 후 개별 영상 토큰 위치를 비교했다. Attention은 head도 평균한다. τ=0.1이며 영상 토큰 전체에서 정규화한다.</p>
<h2>전체 레이어 평균 / 최종 레이어</h2>{table('all_layers__tau0.1__full')}{table('last_layer__tau0.1__full')}
<img src="layers.png"><h2>초반 집중과 GT 질량</h2><table>{''.join(concentration)}</table>
<p>시간 bin 경계는 실제 sampled timestamp의 중점, 양 끝은 0과 실제 영상 길이다. GT 합집합과의 시간 중첩 비율로 질량을 계산한다.
범위 밖 GT는 원본을 보존하고 질량 계산에서만 영상과 교차한다. GT 질량은 localization 성능이나 인과성 지표가 아니다.</p>
<h2>재현과 원자료</h2>
<p><a href="metrics.csv">전체 조건 CSV</a> · <a href="layer_metrics.csv">레이어별 CSV</a> · <a href="video_token_metrics.csv">영상 토큰별 비교 CSV</a> · <a href="summary.json">요약 JSON</a> · <a href="validation_summary.json">검증</a> · <a href="run_config.json">실행 설정</a></p>
<p><a href="query_token_metrics.csv">각 쿼리 토큰별 분포 비교 CSV</a>: 토큰별 시간 분포를 L8–21 평균, τ=0.1로 비교한 값이다.
주 표는 쿼리 토큰을 평균한 뒤 샘플 단위로 집계하며, 긴 쿼리가 더 큰 가중치를 받지 않는다.</p>
<p>{' · '.join(f'<a href="{d}_curves.pdf">{d} 전체 곡선 PDF</a>' for d in DATASETS if d in summary)}</p>
<p><code>.venv/bin/python run.py run --stage hidden</code><br>
<code>.venv/bin/python run.py analyze --stage hidden</code></p>
<p>NPZ에는 <code>cosine_query_frame[L,Q,T]</code>, <code>cosine_video_token[L,V]</code>(쿼리 평균),
<code>attention_query_frame[L,Q,T]</code>(head 평균), <code>attention_video_token[L,V]</code>(head·쿼리 평균)을 FP32로 보존한다.
원래 hidden vector 전체는 디스크에 보관하지 않으며, 추출 중 계산한 유사도를 보존한다.</p>
<h2>전체 샘플</h2><input id="search" placeholder="데이터셋, 샘플, 쿼리 검색"><table id="samples"><tr><th>Sample</th><th>Query</th><th>Spearman</th><th>TV</th><th>Data</th></tr>{''.join(links)}</table>
<script>document.getElementById('search').oninput=function(){{for(const r of document.querySelectorAll('#samples tr:not(:first-child)'))r.style.display=r.textContent.toLowerCase().includes(this.value.toLowerCase())?'':'none';}};</script></html>'''
    body=body.replace('<style>',"<style>@font-face{font-family:ReportKorean;src:url('fonts/NotoSansCJKkr-Regular.otf');font-style:normal;font-weight:400;}")
    body=body.replace('font:16px/1.65 system-ui;','font:16px/1.65 ReportKorean,system-ui;')
    write_result_text((relocate_path(RESULTS/'report.html')), body)


if __name__=='__main__':main()
