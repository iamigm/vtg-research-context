from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Per-layer, per-head temporal attention profiles and early-video concentration.
Run with .venv/bin/python analyze_heads.py; reuses actual-generation arrays.
"""
from pathlib import Path
import csv, hashlib, html, json, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

SRC=GENERATION
OUT=relocate_path(SRC/'head_analysis')
DATASETS=['Charades-STA','QVHighlights','ActivityNet-Captions']
SCOPES=['query','answer']
METRICS=['video_mass','early05','early10','early20','early_peak','centroid','first_bin_mass','raw_early10','uniform_key_early10']
EDGES=np.linspace(0,1,51)
SEED=20260929

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def extract(sample):
    folder=relocate_path(SRC/sample['dataset']/sample['sample_key'])
    meta=json.loads((relocate_path(folder/'metadata.json')).read_text())
    t=np.array(meta['sampled_timestamps'])/meta['duration_video']
    edges=np.r_[0,(t[:-1]+t[1:])/2,1]
    assert np.all(np.diff(edges)>0)
    overlap=np.maximum(0,np.minimum(EDGES[1:,None],edges[None,1:])-np.maximum(EDGES[:-1,None],edges[None,:-1]))/np.diff(edges)
    fractions=[np.maximum(0,np.minimum(edges[1:],f)-edges[:-1])/np.diff(edges) for f in [.05,.1,.2]]
    metrics=[];profiles=[];raws=[]
    with np.load(relocate_path(folder/'attention.npz')) as z:
        for scope in SCOPES:
            tensor=z[scope+'_temporal']
            assert tensor.shape[:2]==(36,32) and tensor.shape[-1]==len(t)
            raw=tensor.mean(axis=2,dtype=np.float64)
            del tensor
            mass=raw.sum(-1)
            assert np.all(mass>0) and np.all(np.isfinite(raw)) and np.all(raw>=0)
            p=raw/mass[...,None]
            early=[p@f for f in fractions]
            metrics.append(np.stack([mass,*early,(t[p.argmax(-1)]<.1),p@t,p[...,0],raw@fractions[1],np.full_like(mass,fractions[1].mean())]))
            prof=p@overlap.T
            assert np.allclose(prof.sum(-1),1,rtol=0,atol=1e-12)
            assert np.allclose(prof[...,:5].sum(-1),early[1],rtol=0,atol=1e-12)
            profiles.append(prof);raws.append(raw)
    # Check the layer/head aggregate against the previously plotted curves.
    saved=np.genfromtxt(relocate_path(folder/'temporal_comparison.csv'),delimiter=',',names=True)
    for i,scope in enumerate(SCOPES):
        assert np.allclose(raws[i][8:22].mean((0,1)),saved[scope],atol=2e-7,rtol=2e-5)
    dest=relocate_path(OUT/sample['dataset']/sample['sample_key']);dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(relocate_path(dest/'head_attention.npz'),query_temporal=raws[0],answer_temporal=raws[1],relative_timestamps=t,
                        metrics=np.array(metrics),metric_names=np.array(METRICS),scopes=np.array(SCOPES))
    return np.array(metrics),np.array(profiles)

def bootstrap_ci(values):
    rng=np.random.default_rng(SEED)
    means=values[rng.integers(0,len(values),(4000,len(values)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()

def rows_for(dataset,metrics,profiles):
    mean=metrics.mean(0);rows=[]
    for si,scope in enumerate(SCOPES):
        ranking=np.argsort(-mean[si,2].ravel(),kind='stable')
        ranks=np.empty(1152,int);ranks[ranking]=np.arange(1,1153)
        total_raw=mean[si,7].sum()
        band_raw=mean[si,7,8:22].sum()
        for layer in range(36):
            for head in range(32):
                row=dict(dataset=dataset,scope=scope,layer=layer,head=head,n=len(metrics),rank=int(ranks[layer*32+head]))
                row.update({k:float(mean[si,mi,layer,head]) for mi,k in enumerate(METRICS)})
                row.update(early10_excess_over_uniform_time=row['early10']-.1,
                    early10_excess_over_uniform_keys=row['early10']-row['uniform_key_early10'],
                    early10_median=float(np.median(metrics[:,si,2,layer,head])),
                    early10_gt_half_rate=float(np.mean(metrics[:,si,2,layer,head]>.5)),
                    raw_early_share_all=float(mean[si,7,layer,head]/total_raw),
                    raw_early_share_band=float(mean[si,7,layer,head]/band_raw) if 8<=layer<=21 else 0.,
                    answer_minus_query_early10=float(mean[1,2,layer,head]-mean[0,2,layer,head]))
                rows.append(row)
    return rows

def heatmap(dataset,means,dest):
    fig,axes=plt.subplots(2,3,figsize=(17,11),constrained_layout=True)
    settings=[(means[0,2]*100,'Query: first 10% mass (%)',0,100,'viridis'),
              (means[1,2]*100,'Answer: first 10% mass (%)',0,100,'viridis'),
              ((means[1,2]-means[0,2])*100,'Answer - query: first 10% mass (pp)',-100,100,'coolwarm'),
              (means[0,0]*100,'Query: total video attention (%)',0,100,'viridis'),
              (means[1,0]*100,'Answer: total video attention (%)',0,100,'viridis'),
              (means[1,4]*100,'Answer: peak in first 10% (% samples)',0,100,'viridis')]
    for ax,(a,title,vmin,vmax,cmap) in zip(axes.flat,settings):
        im=ax.imshow(a,aspect='auto',origin='upper',vmin=vmin,vmax=vmax,cmap=cmap)
        ax.set(title=title,xlabel='Head (0-based)',ylabel='Layer (0-based)',xticks=np.arange(0,32,4),yticks=np.arange(0,36,4))
        ax.axhline(7.5,color='white',ls='--',lw=.8);ax.axhline(21.5,color='white',ls='--',lw=.8)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(f'{dataset}: equal-weight sample means; dashed lines = layers 8-21')
    fig.savefig(relocate_path(dest/'bias_heatmap.png'),dpi=160);fig.savefig(relocate_path(dest/'bias_heatmap.pdf'));plt.close(fig)

def grid(profiles,means,layer):
    fig,axes=plt.subplots(4,8,figsize=(20,11),constrained_layout=True)
    x=(EDGES[:-1]+EDGES[1:])/2
    for h,ax in enumerate(axes.flat):
        for si,label in enumerate(['Query','Generated answer']):
            ax.plot(x,profiles[si,layer,h]*50,label=label,lw=1.1,color=['#2474b5','#d46a28'][si])
        ax.axvspan(0,.1,color='gray',alpha=.12);ax.axhline(1,color='gray',lw=.5,ls=':')
        ax.set(title=f'H{h:02d}  early Q {means[0,2,layer,h]:.0%} / A {means[1,2,layer,h]:.0%}',xlim=(0,1))
        ax.tick_params(labelsize=7);ax.set_title(ax.get_title(),fontsize=9)
    axes[0,0].legend(fontsize=7)
    fig.supxlabel('Relative video time');fig.supylabel('Mean video-conditional probability density')
    fig.suptitle(f'Layer {layer:02d}: all 32 heads · 300 samples · each sample normalized before averaging',fontsize=15)
    return fig

def top_plot(rows,profiles,means,dest,dataset):
    tops=sorted([r for r in rows if r['scope']=='answer'],key=lambda r:r['rank'])[:8]
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    for ax,r in zip(axes.flat,tops):
        l,h=r['layer'],r['head']
        for si,label in enumerate(['Query','Answer']):
            ax.plot((EDGES[:-1]+EDGES[1:])/2,profiles[si,l,h]*50,label=label,color=['#2474b5','#d46a28'][si])
        ax.axvspan(0,.1,color='gray',alpha=.12);ax.axhline(1,color='gray',ls=':',lw=.7)
        ax.set(title=f'L{l:02d} H{h:02d}: early={r["early10"]:.1%}\nvideo mass={r["video_mass"]:.2%}',xlabel='Relative video time',ylabel='Probability density')
        ax.legend(fontsize=8)
    fig.suptitle(dataset+': strongest answer early-video concentration (all layers)')
    fig.savefig(relocate_path(dest/'top_heads.png'),dpi=160);fig.savefig(relocate_path(dest/'top_heads.pdf'));plt.close(fig)

def viewer(aggregates):
    payload={d:{'profiles':np.round(p,6).tolist(),'metrics':np.round(m,6).tolist()} for d,(m,p) in aggregates.items()}
    template='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Head별 attention</title>
+<style>body{font:15px system-ui;margin:24px;background:#fafafa}select{padding:7px;margin:8px}#grid{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:12px}.card{background:white;padding:10px;border:1px solid #ddd}svg{width:100%;height:155px}.q{color:#2474b5}.a{color:#d46a28}@media(max-width:1000px){#grid{grid-template-columns:repeat(2,1fr)}}</style>
+<h1>레이어·head별 영상 attention</h1><p><a href="REPORT.md">분석 보고서</a> · <a href="all_layers_heads.pdf">36개 레이어 PDF</a> · <a href="rankings.csv">전체 순위 CSV</a></p>
+<p>각 샘플에서 영상 내부 합을 1로 정규화한 뒤 평균. x축: 상대 영상 시간 0–1, y축: 확률밀도. 회색: 앞 10%. 번호는 0-based.</p>
+<label>데이터셋 <select id="dataset"></select></label><label>레이어 <select id="layer"></select></label><span class="q">━ Query</span> / <span class="a">━ Generated answer</span><div id="grid"></div>
+<script>const data=__DATA__;const ds=document.getElementById('dataset'),ls=document.getElementById('layer');Object.keys(data).forEach(k=>ds.add(new Option(k,k)));for(let l=0;l<36;l++)ls.add(new Option('Layer '+l,l));
+function draw(){const d=data[ds.value],l=+ls.value;document.getElementById('grid').innerHTML=Array.from({length:32},(_,h)=>{const q=d.profiles[0][l][h].map(v=>v*50),a=d.profiles[1][l][h].map(v=>v*50),ymax=Math.max(...q,...a,1)*1.05;const line=v=>v.map((y,i)=>`${35+(i+.5)*5.2},${130-y/ymax*110}`).join(' ');return `<div class="card"><b>L${l} H${h}</b><svg viewBox="0 0 310 155"><rect x="35" y="20" width="26" height="110" fill="#eee"/><path d="M35 20 V130 H295" fill="none" stroke="#aaa"/><polyline points="${line(q)}" fill="none" stroke="#2474b5" stroke-width="1.8"/><polyline points="${line(a)}" fill="none" stroke="#d46a28" stroke-width="1.8"/><text x="0" y="25" font-size="11">${ymax.toFixed(1)}</text><text x="15" y="132" font-size="11">0</text><text x="32" y="146" font-size="11">0</text><text x="158" y="146" font-size="11">0.5</text><text x="290" y="146" font-size="11">1</text></svg><div>앞 10% 질량 Q ${(d.metrics[0][2][l][h]*100).toFixed(1)}% / A ${(d.metrics[1][2][l][h]*100).toFixed(1)}%</div><div>영상 총량 Q ${(d.metrics[0][0][l][h]*100).toFixed(2)}% / A ${(d.metrics[1][0][l][h]*100).toFixed(2)}%</div></div>`}).join('')}ds.onchange=ls.onchange=draw;draw();</script></html>'''.replace('\n+','\n')
    write_result_text((relocate_path(OUT/'report.html')), template.replace('__DATA__',json.dumps(payload,separators=(',',':'))))

def report(rows,metrics,samples):
    allrows=[r for r in rows if r['dataset']=='All'];lines=['# 레이어·head별 영상 초반 편향','',
      '실제 생성 300개(각 벤치마크 100개), 36개 레이어 × 32개 head를 분석했다. **L/H는 모두 0-based**이며, 서로 다른 레이어의 같은 head 번호는 별개의 head다. Query 이벤트 토큰과 EOS 제외 답변 전체 토큰을 각각 평균했다. 답변은 생성된 토큰을 입력으로 처리하는 행이다.','',
      '주 편향 점수는 **각 샘플의 영상 내 attention 중 앞 10%에 배정된 질량을 계산한 뒤 샘플 평균**한 값이다. 시간 bin 경계는 실제 timestamp의 중점이고, 경계와 겹치는 bin 질량을 시간 비율로 배분했다. 균일한 시간 분포의 기준은 10%. 값이 클수록 앞부분 집중이 강하다. 영상에 보내는 attention 총량도 함께 봐야 한다.','',
      '[인터랙티브 32-head 분포](report.html) · [36개 레이어 전체 PDF](all_layers_heads.pdf) · [전체 순위 CSV](rankings.csv) · [전체 heatmap](bias_heatmap.png)','']
    selected={}
    for scope in SCOPES:
        si=SCOPES.index(scope)
        for label,eligible in [('all_layers',allrows),('layers_8_21',[r for r in allrows if 8<=r['layer']<=21])]:
            top=sorted([r for r in eligible if r['scope']==scope],key=lambda r:-r['early10'])[:10]
            selected[scope+'_'+label]=[]
            lines += [f'## {scope}: {label} 상위 10개','', '| 순위 | Layer / Head | 앞 10% 질량 평균 [95% CI] | 영상 총량 | 초반 peak 비율 | 앞 10% 질량>50% 샘플 |','|---|---|---|---:|---:|---:|']
            for i,r in enumerate(top):
                ci=bootstrap_ci(metrics[:,si,2,r['layer'],r['head']]);selected[scope+'_'+label].append(dict(r,early10_mean_ci95=ci))
                lines.append(f"| {i+1} | L{r['layer']} H{r['head']} | {r['early10']:.2%} [{ci[0]:.2%}, {ci[1]:.2%}] | {r['video_mass']:.2%} | {r['early_peak']:.1%} | {r['early10_gt_half_rate']:.1%} |")
    lines+=['','## 데이터셋별 1위','', '| 데이터셋 | Query 1위 | 질량 | Answer 1위 | 질량 |','|---|---|---:|---|---:|']
    for d in DATASETS:
        best=[min([r for r in rows if r['dataset']==d and r['scope']==scope],key=lambda r:r['rank']) for scope in SCOPES]
        q,a=best;lines.append(f"| {d} | L{q['layer']} H{q['head']} | {q['early10']:.2%} | L{a['layer']} H{a['head']} | {a['early10']:.2%} |")
    lines+=['','## 원래 평균 곡선의 초반 질량에 가장 많이 기여하는 head','',
      '조건부 집중률 순위와 별개로, 레이어 8–21에서 실제 softmax의 초반 영상 질량을 많이 보내는 head를 정렬했다. 기여율은 모든 샘플·선택 head의 raw 초반 질량 합에서 해당 head가 차지하는 비율이다. 개별 샘플의 기여율을 평균한 값과는 다르다.','',
      '| Scope | Layer / Head | Raw 초반 질량 | 선택 head 내 기여율 | 조건부 초반 질량 |','|---|---|---:|---:|---:|']
    for scope in SCOPES:
        top=sorted([r for r in allrows if r['scope']==scope and 8<=r['layer']<=21],key=lambda r:-r['raw_early10'])[:5]
        selected[scope+'_band_raw_contributors']=top
        for r in top:lines.append(f"| {scope} | L{r['layer']} H{r['head']} | {r['raw_early10']:.3%} | {r['raw_early_share_band']:.2%} | {r['early10']:.2%} |")
    lines+=['','## 기준 변경과 해석 범위','']
    sensitivity={}
    for si,scope in enumerate(SCOPES):
        mean=metrics[:,si].mean(0);scores=mean[2].ravel();sensitivity[scope]={}
        for mi,label in [(1,'first_5pct'),(3,'first_20pct')]:
            best=int(mean[mi].argmax());rho=float(spearmanr(scores,mean[mi].ravel()).statistic)
            sensitivity[scope][label]=dict(top_layer=best//32,top_head=best%32,rank_correlation_with_first10=rho)
            lines.append(f"- {scope}, {label}: 1위 L{best//32} H{best%32}; 앞 10% 순위와 Spearman {rho:.3f}.")
    lines+=['', 'Bootstrap은 샘플 단위 4,000회(seed=20260929)로 각 고정 head의 평균 CI를 구했다. 데이터로 상위 head를 선택했으므로 CI는 1위의 통계적 우월성이나 다중비교 보정을 뜻하지 않는다. 고정 모델·프롬프트·샘플에 대한 기술적 분석이다. 초반 편향이 위치 효과인지 영상 내용 때문인지는 이 관측만으로 분리할 수 없다.', '',
      '정규화 후 높은 초반 비율이라도 전체 영상 attention이 작을 수 있다. `rankings.csv`에 전체 영상 질량, raw 초반 질량, uniform-video-key 기준, 질량 중심, 첫 bin 질량을 함께 저장했다. 매우 짧은 영상은 하나의 bin이 앞 10%보다 넓으므로 시간 중첩 비율을 적용한 값의 상한이 낮아질 수 있다.', '',
      '각 head 분포는 영상 내 합을 1로 정규화한 뒤 동일 가중치로 평균했다. 영상 attention 총량으로 가중된 기존 전체 평균 곡선과는 집계 순서가 다르다. 전 레이어에서 같은 head 번호를 합산해 특정 head라고 부르지 않았다.', '',
      '## 파일','',
      '- `layers/`: 36개 레이어별 32-head 그림.',
      '- `<dataset>/bias_heatmap.png`, `<dataset>/top_heads.png`: 벤치마크별 비교.',
      '- `<dataset>/<sample_key>/head_attention.npz`: 토큰 평균 후 head별 원래 시간 곡선 및 지표.',
      '- `aggregate_profiles.npz`: 50개 상대 시간 bin에 질량을 보존해 재배분한 head별 평균 곡선.',
      '- 재실행: `.venv/bin/python analyze_heads.py`. 기존 원본 데이터와 생성 결과는 수정하지 않는다.','']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines))
    write_result_text((relocate_path(OUT/'top_heads.json')), json.dumps(dict(selected=selected,sensitivity=sensitivity),indent=2)+'\n')

def main():
    OUT.mkdir(exist_ok=True);samples=json.loads((relocate_path(SRC/'samples.json')).read_text());assert len(samples)==300
    metrics=[];profiles=[]
    for i,s in enumerate(samples):
        m,p=extract(s);metrics.append(m);profiles.append(p)
        if (i+1)%25==0:print('Extracted',i+1,'/ 300',flush=True)
    metrics=np.array(metrics);profiles=np.array(profiles);rows=[];aggregates={}
    for dataset in ['All']+DATASETS:
        ids=[i for i,s in enumerate(samples) if dataset=='All' or s['dataset']==dataset]
        m,p=metrics[ids],profiles[ids]
        group=rows_for(dataset,m,p);rows.extend(group);aggregates[dataset]=(m.mean(0),p.mean(0))
        dest=OUT if dataset=='All' else relocate_path(OUT/dataset);dest.mkdir(exist_ok=True)
        write_csv(relocate_path(dest/'rankings.csv'),group)
        heatmap(dataset,*[m.mean(0),dest]);top_plot(group,p.mean(0),m.mean(0),dest,dataset)
    write_csv(relocate_path(OUT/'rankings.csv'),rows)
    np.savez_compressed(relocate_path(OUT/'aggregate_profiles.npz'),**{d.replace('-','_'):p for d,(m,p) in aggregates.items()},relative_bin_edges=EDGES)
    np.savez_compressed(relocate_path(OUT/'sample_head_metrics.npz'),metrics=metrics,metric_names=np.array(METRICS),scopes=np.array(SCOPES),sample_keys=np.array([s['sample_key'] for s in samples]),datasets=np.array([s['dataset'] for s in samples]))
    report(rows,metrics,samples);viewer(aggregates)
    (relocate_path(OUT/'layers')).mkdir(exist_ok=True)
    with PdfPages(relocate_path(OUT/'all_layers_heads.pdf')) as pdf:
        means,profs=aggregates['All']
        for layer in range(36):
            fig=grid(profs,means,layer);pdf.savefig(fig);fig.savefig(relocate_path(OUT/'layers'/f'L{layer:02d}.png'),dpi=130);plt.close(fig)
            print('Rendered layer',layer,'/ 35',flush=True)
    validation=dict(samples=300,layers=36,heads_per_layer=32,scopes=SCOPES,normalized_profile_sums=True,first10_rebin_matches_direct=True,
        aggregate_matches_previous_curves=True,script_sha256=hashlib.sha256(relocate_path(Path(__file__)).read_bytes()).hexdigest())
    write_result_text((relocate_path(OUT/'validation.json')), json.dumps(validation,indent=2)+'\n')
    print('Complete:',OUT,flush=True)

if __name__=='__main__':main()
