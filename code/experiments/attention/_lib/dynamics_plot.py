from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Static scientific figures and an offline interactive token-step viewer."""
import html,json,textwrap,time,sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dynamics import ROOT,OUT,DATASETS
sys.path.insert(0,str(relocate_path(ROOT/'python_deps')))
from pypdf import PdfWriter

def render(d):
    n=len(d['tokens']);steps=np.arange(1,n+1);m=d['scopes']['predictive']['metrics']
    raw=np.array(d['scopes']['predictive']['raw']);p=raw/raw.sum(-1,keepdims=True)
    height=max(3,n*.09)
    fig,axes=plt.subplots(6,1,figsize=(13,11+height),gridspec_kw={'height_ratios':[height,.5,2,1.8,1.8,1.8]},layout='constrained')
    title=f"{d['sample_key']} · attention used to select each generated token\nQuery: {d['query']}\nAnswer: {d['answer']}"
    if d['truncated']:title+=' [TRUNCATED at 128 tokens]'
    fig.suptitle('\n'.join(textwrap.wrap(title,115,replace_whitespace=False)),fontsize=11)
    ax=axes[0];im=ax.pcolormesh(d['edges'],np.arange(n+1)+.5,p,shading='flat',cmap='magma')
    ax.set_ylim(n+.5,.5);ax.set_yticks(steps,[f"{i+1}: {t!r}" for i,t in enumerate(d['tokens'])],fontsize=7)
    ax.set(xlabel='Video time (seconds)',ylabel='Token selected at this step',title='Video-conditional attention · every row sums to 1')
    ax.plot(m['peak_seconds'],steps,color='cyan',marker='.',ms=3,lw=.8,label='Peak');ax.legend(fontsize=8,loc='upper right')
    fig.colorbar(im,ax=ax,label='Attention within video')
    xmax=max([d['duration']]+[e for s,e in d['gt']]+[e for s,e in d['predictions']])
    ax.set_xlim(0,xmax)
    for y,windows,color in [(1,d['gt'],'#319966'),(0,d['predictions'],'#ce676a')]:
        for s,e in windows:axes[1].broken_barh([(s,e-s)],(y-.3,.6),facecolors=color)
    axes[1].set(yticks=[0,1],yticklabels=['Answer spans','GT'],xlim=(0,xmax),xlabel='Video time (seconds)')
    keys=np.array(m['keys']);axes[2].stackplot(steps,*keys.T,labels=['Video','Input query','Previous answer','Other prefix'],colors=['#2474b5','#64aa70','#dc8538','#a7a7b7'])
    axes[2].set(ylabel='Raw attention mass',ylim=(0,1.01),title='Where does attention go at each generation step?');axes[2].legend(ncol=4,fontsize=8,loc='upper right')
    axes[3].plot(steps,m['gt_mass'],label='Current step',color='#2474b5')
    axes[3].axhline(d['query_gt_mass'],color='gray',ls='--',label='Input-query mean')
    ix=d['timestamp_token_indices']
    axes[3].scatter(steps[ix],np.array(m['gt_mass'])[ix],color='#9b40aa',s=20,label='Timestamp digit selected',zorder=3)
    axes[3].set(ylabel='GT mass within video',ylim=(0,1),title='GT concentration over generated tokens');axes[3].legend(fontsize=8,ncol=3)
    axes[4].plot(steps,m['peak_seconds'],marker='.',label='Peak',color='#d16c22')
    axes[4].plot(steps,m['centroid_seconds'],label='Attention centroid',color='#2474b5')
    for s,e in d['gt']:axes[4].axhspan(s,e,color='#319966',alpha=.15)
    axes[4].set(ylabel='Video time (seconds)',ylim=(0,max(d['duration'],max(e for s,e in d['gt']))),title='Attention location · green bands = GT');axes[4].legend(fontsize=8)
    axes[5].plot(steps,m['tv_previous'],label='From previous generation step',color='#2474b5')
    axes[5].plot(steps,m['tv_from_query'],label='From input-query mean',color='#a276ad',ls='--')
    axes[5].set(ylabel='Total variation (0–1)',xlabel='Generation step / emitted token index (1-based)',ylim=(0,1),title='How much does the temporal distribution change?');axes[5].legend(fontsize=8)
    for ax in axes[2:]:ax.set_xlim(.8,n+.2);ax.grid(alpha=.2)
    folder=relocate_path(OUT/ d['dataset'] / d['sample_key']);fig.savefig(relocate_path(folder/'token_dynamics.png'),dpi=145);fig.savefig(relocate_path(folder/'token_dynamics.pdf'));plt.close(fig)

def summary_plot(summary):
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    cols=['#2474b5','#cf7734','#7c59ad'];x=np.arange(10)*10+5
    for ds,col in zip(DATASETS,cols):
        s=summary[ds]['scopes']['predictive']
        for ax,key in zip(axes.flat,['video_attention','gt_mass','tv_previous']):
            y=[b[key]['median'] for b in s['progress_bins']];lo=[b[key]['q25'] for b in s['progress_bins']];hi=[b[key]['q75'] for b in s['progress_bins']]
            ax.plot(x,y,color=col,label=ds);ax.fill_between(x,lo,hi,color=col,alpha=.1);ax.set_xlabel('Relative answer progress (%)')
        axes[1,1].hist([r['mean_tv'] for r in s['sample_summaries']],bins=np.linspace(0,1,26),histtype='step',color=col,label=ds,lw=2)
    for ax,title,ylabel in zip(axes.flat,['Attention assigned to video','GT concentration within video','Change from preceding token','Within-answer variation per sample'],['Raw attention mass','GT attention mass','Total variation','Samples']):
        ax.set(title=title,ylabel=ylabel);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[1,1].set_xlabel('Mean adjacent-token total variation')
    fig.suptitle('Token-by-token attention during actual generation · same 100 samples per dataset\nPredictive rows · layers 8–21 · curves: median / IQR across samples, after averaging within each progress bin',fontsize=12)
    fig.savefig(relocate_path(OUT/'summary.png'),dpi=180);fig.savefig(relocate_path(OUT/'summary.pdf'));plt.close(fig)

def make_html(data):
    template=(relocate_path(ROOT/'_lib/viewer_template.html')).read_text()
    template=template.replace('__SAMPLE_ASSETS__',json.dumps(os.path.relpath(relocate_path(ROOT/'artifacts/runs/dynamics/samples'),relocate_path(OUT))+'/'))
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    write_result_text((relocate_path(OUT/'report.html')), template.replace('__PAYLOAD__',payload))

def main():
    samples=json.loads((relocate_path(OUT/'samples.json')).read_text());data=[]
    for i,s in enumerate(samples):
        folder=relocate_path(OUT/ s['dataset'] / s['sample_key'])
        while not (relocate_path(folder/'validation.json')).exists():time.sleep(3)
        d=json.loads((relocate_path(folder/'dynamics.json')).read_text());data.append(d)
        if not (relocate_path(folder/'PLOTTED')).exists():render(d);(relocate_path(folder/'PLOTTED')).touch()
        if (i+1)%25==0:print('Rendered',i+1,flush=True)
    while not (relocate_path(OUT/'summary.json')).exists():time.sleep(3)
    summary_plot(json.loads((relocate_path(OUT/'summary.json')).read_text()));make_html(data)
    all_pdf=PdfWriter()
    for ds in DATASETS:
        pdf=PdfWriter()
        for d in data:
            if d['dataset']==ds:
                path=relocate_path(OUT/ d['dataset'] / d['sample_key']/'token_dynamics.pdf');pdf.append(path);all_pdf.append(path)
        pdf.write(relocate_path(OUT/f'{ds}_token_dynamics.pdf'));pdf.close()
    all_pdf.write(relocate_path(OUT/'all_token_dynamics.pdf'));all_pdf.close()
    print('All 300 plots, interactive viewer and PDFs complete',flush=True)

if __name__=='__main__':main()
