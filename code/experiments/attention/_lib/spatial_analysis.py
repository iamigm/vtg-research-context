from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Spatial inspection of late-GT videos with early query-attention peaks."""
from pathlib import Path
import json,csv,html
import numpy as np
from scipy.stats import spearmanr,rankdata
from scipy.special import softmax
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hidden_analysis import RESULTS, fractions, write_json, write_csv

OUT=SPATIAL
OUT.mkdir(exist_ok=True)


def load(sample):
    folder=relocate_path(RESULTS/sample['dataset']/sample['sample_key'])
    m=json.loads((relocate_path(folder/'metadata.json')).read_text())
    with np.load(relocate_path(folder/'distributions.npz')) as z:
        a=z['attention_video_token'][8:22].astype(float).mean(0)
        c=z['cosine_video_token'][8:22].astype(float).mean(0)
        t=z['sampled_timestamps']
    T,H,W=m['video_grid_thw'];H//=2;W//=2
    a=a.reshape(T,H,W);c=c.reshape(T,H,W)
    temporal=a.sum((1,2));temporal/=temporal.sum()
    edges=np.r_[0,(t[:-1]+t[1:])/2,m['duration_video']]
    return m,a,c,t,temporal,edges


def frame_metrics(a,c):
    p=a.ravel()/a.sum();v=c.ravel();k=int(np.ceil(.2*len(p)))
    top=np.argsort(-p,kind='stable')[:k];ctop=np.argsort(-v,kind='stable')[:k]
    ranked=(rankdata(v)-1)/(len(v)-1)
    h,w=a.shape
    border=np.zeros((h,w),bool);border[[0,-1],:]=True;border[:,[0,-1]]=True
    return dict(spatial_spearman=float(spearmanr(p,v).statistic),
        attention_entropy=float(-p@np.log(p)/np.log(len(p))),
        top20_attention_mass=float(np.sort(p)[-k:].sum()),
        top20_similarity_overlap=len(set(top)&set(ctop))/k,
        attention_weighted_similarity_percentile=float(p@ranked),
        first_patch_mass=float(p[0]),last_patch_mass=float(p[-1]),
        border_mass=float(p@border.ravel()),border_area=float(border.mean()),
        peak_row=int(p.argmax()//w),peak_col=int(p.argmax()%w),
        cosine_min=float(v.min()),cosine_max=float(v.max()),
        patches=len(p),height=h,width=w)


def decode(m,bins):
    import decord
    vr=decord.VideoReader(m['video_path'],ctx=decord.cpu(0),num_threads=2)
    indices=m['decoder_metadata']['frames_indices']
    # Temporal patch size=2. Both constituent frames are retained for inspection.
    ids=[indices[min(2*b+j,len(indices)-1)] for b in bins for j in [0,1]]
    frames=vr.get_batch(ids).asnumpy()
    return frames.reshape(len(bins),2,*frames.shape[1:]),np.array(ids).reshape(-1,2)


def render(m,a,c,t,temporal,edges,selected):
    folder=relocate_path(OUT/m['sample_key']);folder.mkdir(exist_ok=True)
    roles=list(selected);bins=list(selected.values())
    frames,indices=decode(m,bins)
    np.savez_compressed(relocate_path(folder/'frames_and_maps.npz'),frames=frames,frame_indices=indices,
        bins=bins,attention=a[bins],cosine=c[bins],timestamps=t[bins])
    rows=[]
    for role,b in selected.items():
        rows.append(dict(sample_key=m['sample_key'],dataset=m['dataset'],role=role,bin=b,
                         timestamp=float(t[b]),temporal_attention=float(temporal[b]),**frame_metrics(a[b],c[b])))
    fig=plt.figure(figsize=(15,2+3*len(bins)))
    gs=fig.add_gridspec(len(bins)+1,4,height_ratios=[.8]+[1]*len(bins))
    ax=fig.add_subplot(gs[0,:]);ax.plot(t,temporal,color='#2563eb',label='Query attention')
    for j,(s,e) in enumerate(m['gt_windows']):ax.axvspan(s,e,color='#16a34a',alpha=.2,label='GT interval' if j==0 else None)
    ax.axvspan(0,.1*m['duration_video'],color='#e87924',alpha=.1,label='First 10%')
    for role,b in selected.items():ax.axvline(t[b],color='gray',ls=':',lw=.8);ax.annotate(role,(t[b],temporal[b]),fontsize=7)
    ax.set_xlim(0,max(m['duration_video'],max(e for s,e in m['gt_windows'])))
    ax.set_xlabel('Seconds');ax.set_ylabel('Video-normalized mass');ax.legend(fontsize=7,loc='upper right')
    ax.set_title(m['sample_key']+' | '+m['query']+'\nGT '+str(m['gt_windows']),fontsize=10,wrap=True)
    for i,(role,b) in enumerate(selected.items()):
        p=a[b]/a[b].sum();h,w=p.shape;v=c[b]
        percentile=(rankdata(v.ravel()).reshape(h,w)-1)/(h*w-1)
        frame=frames[i,0];ih,iw=frame.shape[:2]
        for col in range(4):
            ax=fig.add_subplot(gs[i+1,col]);ax.imshow(frames[i,1] if col==1 else frame);ax.axis('off')
            if col==0:ax.set_title(f'{role}: {t[b]:.2f}s\nframe {indices[i,0]}',fontsize=9)
            if col==1:ax.set_title(f'Paired frame {indices[i,1]}\nSame temporal token',fontsize=9)
            if col==2:
                # Intensity is scaled per frame for localization; the numeric peak is explicit.
                im=ax.imshow(p,extent=(0,iw,ih,0),cmap='inferno',alpha=.62,interpolation='nearest',vmin=0,vmax=p.max())
                ax.set_title(f'Attention within frame\nmax={p.max():.3f}; uniform={1/(h*w):.3f}',fontsize=9)
            if col==3:
                ax.imshow(percentile,extent=(0,iw,ih,0),cmap='viridis',alpha=.6,interpolation='nearest',vmin=0,vmax=1)
                ax.set_title(f'Hidden cosine spatial percentile\nraw range {v.min():.3f}–{v.max():.3f}',fontsize=9)
    fig.tight_layout();fig.savefig(relocate_path(folder/'spatial.png'),dpi=140);plt.close(fig)
    return rows


def main():
    manifest=json.loads((relocate_path(RESULTS/'samples.json')).read_text())
    cohort=[];allrows=[];strong=[]
    for s in manifest:
        m,a,c,t,temporal,edges=load(s);d=m['duration_video']
        start=min(x[0] for x in m['gt_windows'])
        if not (.1*d<=start<d and t[temporal.argmax()]<.1*d):continue
        early_mass=float(temporal@fractions(edges,[[0,.1*d]]))
        gt=fractions(edges,m['gt_windows'])
        if not (gt>0).any():continue
        early=np.flatnonzero(t<.1*d)
        peak=int(temporal.argmax())
        gtpeak=int(np.where(gt>0,temporal,-1).argmax())
        # GT bin is chosen by representative timestamp where possible, not just a boundary overlap.
        gtpoint=np.array([any(lo<=x<=hi for lo,hi in m['gt_windows']) for x in t])
        if gtpoint.any():gtpeak=int(np.where(gtpoint,temporal,-1).argmax())
        second=[int(x) for x in early if x!=peak]
        selected={'early_peak':peak}
        if second:selected['early_other']=max(second,key=lambda b:temporal[b])
        selected['GT_peak']=gtpeak
        row=dict(sample_key=m['sample_key'],dataset=m['dataset'],query=m['query'],gt_windows=m['gt_windows'],
            duration=d,early_mass=early_mass,gt_start_fraction=start/d,strong=early_mass>=.2,selected_bins=selected)
        cohort.append(row)
        for role,b in selected.items():
            allrows.append(dict(sample_key=m['sample_key'],dataset=m['dataset'],strong=row['strong'],role=role,
                bin=b,timestamp=float(t[b]),**frame_metrics(a[b],c[b])))
        if row['strong']:
            strong.append(row)
            render(m,a,c,t,temporal,edges,selected)
            print('Rendered',m['sample_key'],flush=True)
    write_json(relocate_path(OUT/'cohort.json'),cohort);write_json(relocate_path(OUT/'strong_samples.json'),strong)
    write_csv(relocate_path(OUT/'spatial_metrics.csv'),allrows)
    # Raw-frame contact sheets for visual review, five videos per sheet.
    for start in range(0,len(strong),5):
        part=strong[start:start+5]
        fig,axes=plt.subplots(len(part),3,figsize=(15,3.4*len(part)),squeeze=False)
        for i,s in enumerate(part):
            z=np.load(relocate_path(OUT/s['sample_key']/'frames_and_maps.npz'));roles=list(s['selected_bins'])
            ei=roles.index('early_peak');gi=roles.index('GT_peak')
            for col,(j,second) in enumerate([(ei,0),(ei,1),(gi,0)]):
                axes[i,col].imshow(z['frames'][j,second]);axes[i,col].axis('off')
                axes[i,col].set_title(('EARLY' if col<2 else 'GT')+f' {z["timestamps"][j]:.1f}s',fontsize=9)
            axes[i,0].set_title(s['sample_key']+'\n'+s['query']+'\nEARLY',fontsize=9,wrap=True)
        fig.tight_layout();fig.savefig(relocate_path(OUT/f'contact_{start//5+1}.jpg'),dpi=130);plt.close(fig)
    print('Cohort',len(cohort),'strong',len(strong),flush=True)


if __name__=='__main__':main()
