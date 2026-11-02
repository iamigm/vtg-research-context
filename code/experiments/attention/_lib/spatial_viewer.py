from result_layout import relocate_path, write_result_text, result_glob
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""All sampled-frame attention maps, visual notes, and counterfactual summaries."""
from pathlib import Path
import json,csv,html
import numpy as np
from scipy.stats import spearmanr,rankdata
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spatial_analysis import OUT,RESULTS,load
from hidden_analysis import write_json,write_csv

# Exploratory visual inspection of both constituent frames, not box/segmentation GT.
NOTES={
'Charades-STA_16':('background','사람과 종이봉투는 초반에도 보이지만, 최고점은 왼쪽 위 계단/배경에 있다. 약의 내용물은 판별하지 않았다.'),
'Charades-STA_36':('background','사람은 보이지만 최고점은 위쪽 전자레인지/선반이다.'),
'Charades-STA_46':('context_or_ambiguous','최고점은 사람 왼쪽의 붉은 의자/소파 부근이다. 쿼리의 목적지와 관련될 수 있으나 신발이나 던지는 동작에 대한 증거는 약하다.'),
'Charades-STA_48':('context_or_ambiguous','최고점은 펼친 책 가장자리와 이불의 경계 부근이다. 책은 보이지만 해당 patch가 책만을 선택한다고 단정하기 어렵다.'),
'Charades-STA_51':('background','노트북은 초반에도 보이지만 최고점은 화면 오른쪽 벽이다. GT 내부 최고점은 노트북 부근으로 이동한다.'),
'Charades-STA_74':('background','앉은 사람은 보이지만 초반 최고점은 오른쪽 문/배경이다. GT 내부 최고점은 얼굴 부근이다.'),
'Charades-STA_78':('background','사람은 보이지만 초반 최고점은 오른쪽 위 벽이다. GT 내부에서는 입과 손 부근으로 이동한다.'),
'Charades-STA_85':('background','소파/담요/사람은 보이지만 최고점은 왼쪽 위 문틀/벽이다.'),
'Charades-STA_92':('background','달리는 사람이 초반에도 보이지만 최고점은 오른쪽 아래 바닥이다.'),
'Charades-STA_96':('background','소파의 사람이 보이지만 초반 최고점은 오른쪽 위 조명/벽이다. GT 내부는 얼굴 부근이다.'),
'Charades-STA_98':('background','열린 냉장고와 사람은 보이지만 초반 최고점은 오른쪽 조리대/벽 부근이다. GT 내부 최고점은 냉장고 문 쪽이다.'),
'QVHighlights_15':('related','초반 최고점은 커피컵을 든 손 부근이다. 관련 물체가 초반에도 존재하지만, 차에 들어가는 동작의 시점을 확인한 것은 아니다.'),
'QVHighlights_40':('background','오토바이와 도로가 보이지만 초반 최고점은 위쪽 하늘이다.'),
'QVHighlights_50':('context_or_ambiguous','회색 옷의 사람은 보이며 최고점은 어깨/배경 경계 부근이다. 초반에는 실내에 앉아 있어 밖에서 안으로 들어오는 동작과 구별된다.'),
'QVHighlights_52':('background','도로 장면은 있지만 최고점은 왼쪽 위 하늘이다. 거울을 통한 시점인지 이 프레임만으로 확인하지 않았다.'),
'QVHighlights_58':('context_or_ambiguous','여성이 초반에도 보인다. 최고점 patch는 머리 왼쪽 위 배경과 머리 경계 부근으로, 인물만을 보는지 불명확하다.'),
'QVHighlights_70':('context_or_ambiguous','temporal bin 내부에서 샷이 바뀐다. 초반에는 뱀과 뱀을 든 남성이 보이고, 쿼리의 큰 도마뱀은 보이지 않는다.'),
'QVHighlights_97':('context_or_ambiguous','첫 프레임에서는 뒤로 쓴 모자 부근이 최고점이지만 두 번째 프레임은 다른 샷이다. 석조 기념물은 초반에 보이지 않는다.'),
'ActivityNet-Captions_21':('background','두 입력 프레임은 검은 화면과 매니큐어 제목 화면이다. 최고점은 오른쪽 아래 배경이며 손톱/바르는 동작은 없다.'),
'ActivityNet-Captions_27':('background','여성과 술잔은 초반에도 보이지만 최고점은 왼쪽 가장자리 벽/포스터다.'),
'ActivityNet-Captions_38':('background','경기 중인 사람들이 초반에도 보이지만 최고점은 위쪽 나무/배경이다.'),
'ActivityNet-Captions_43':('related','최고점은 바닥에 놓인 공 부근이다. 공은 초반에도 있으나 사람이 구부려 집는 시점은 이후 GT와 구별된다.'),
'ActivityNet-Captions_47':('related','최고점은 회사명 텍스트 부근이다. 회사명/웹사이트가 적힌 거의 같은 타이틀 카드가 영상 앞과 끝에 반복되어, 시각적 관련성과 ends라는 시간 조건이 분리된다.'),
'ActivityNet-Captions_62':('background','줄다리기 선수들이 보이지만 최고점은 오른쪽 아래 잔디이다.'),
'ActivityNet-Captions_79':('background','체조 선수가 보이지만 최고점은 오른쪽 위 관중석/배경이다.'),
}


def examples(cases):
    chosen=['Charades-STA_51','ActivityNet-Captions_62','QVHighlights_15']
    by={s['sample_key']:s for s in cases}
    fig=plt.figure(figsize=(16,9))
    gs=fig.add_gridspec(6,4,height_ratios=[.2,1,.2,1,.2,1])
    for i,key in enumerate(chosen):
        s=by[key];z=np.load(relocate_path(OUT/key/'frames_and_maps.npz'));roles=list(s['selected_bins'])
        ei,gi=roles.index('early_peak'),roles.index('GT_peak')
        maps=[z['attention'][j]/z['attention'][j].sum() for j in [ei,gi]]
        vmax=max(p.max() for p in maps)
        header=fig.add_subplot(gs[2*i,:]);header.axis('off')
        header.text(0,.3,key+' — '+s['query']+' | GT: '+str(s['gt_windows']),fontsize=11)
        for col in range(4):
            j=[ei,gi][col//2];f=z['frames'][j,0];H,W=f.shape[:2];ax=fig.add_subplot(gs[2*i+1,col])
            ax.imshow(f);ax.axis('off')
            title=('Early' if col<2 else 'Inside GT')+f' bin {z["timestamps"][j]:.1f}s'
            if col%2:
                p=maps[col//2];h,w=p.shape
                ax.imshow(p,extent=(0,W,H,0),interpolation='nearest',alpha=.62,cmap='inferno',vmin=0,vmax=vmax)
                y,x=np.unravel_index(p.argmax(),p.shape);ax.scatter((x+.5)*W/w,(y+.5)*H/h,c='cyan',marker='x',s=50)
                title+=f' | peak={p.max():.3f}'
            else:title+=' | original'
            ax.set_title(title,fontsize=10)
    fig.suptitle('Frame attention maps · L8–21 · query/head mean · cyan × = highest-attention patch\nWithin-frame probabilities; each row uses the same color scale for early and GT maps.',fontsize=12,y=.995)
    fig.subplots_adjust(top=.88,bottom=.015,left=.015,right=.985,wspace=.04,hspace=.35)
    fig.savefig(relocate_path(OUT/'frame_attention_examples.png'),dpi=160);plt.close(fig)


def export_frames(s):
    import decord
    folder=relocate_path(OUT/s['sample_key']);dst=relocate_path(folder/'frames');dst.mkdir(exist_ok=True)
    m,a,c,t,temporal,edges=load(s)
    inds=m['decoder_metadata']['frames_indices'];fps=m['decoder_metadata']['fps']
    marker=relocate_path(folder/'viewer_data.json')
    if marker.exists() and len(list(result_glob(dst, '*.jpg')))==len(inds):return
    original=decord.VideoReader(m['video_path'],num_threads=2)
    H,W=original[0].shape[:2];del original
    scale=min(1,640/max(H,W));height,width=round(H*scale),round(W*scale)
    vr=decord.VideoReader(m['video_path'],width=width,height=height,num_threads=2)
    for first in range(0,len(inds),24):
        batch=vr.get_batch(inds[first:first+24]).asnumpy()
        for i,frame in enumerate(batch,first):Image.fromarray(frame).save(dst/f'{i:04d}.jpg',quality=84)
    probs=a/a.sum((1,2),keepdims=True)
    ranks=np.stack([(rankdata(v.ravel()).reshape(v.shape)-1)/(v.size-1) for v in c])
    write_json(marker,dict(sample_key=s['sample_key'],query=s['query'],gt_windows=m['gt_windows'],
        duration=m['duration_video'],frame_indices=inds,frame_times=(np.array(inds)/fps).tolist(),
        timestamps=t.tolist(),temporal_attention=temporal.tolist(),height=a.shape[1],width=a.shape[2],
        attention=np.round(probs,8).tolist(),cosine_rank=np.round(ranks,6).tolist(),
        selected_bins=s['selected_bins'],early_mass=s['early_mass'],review=NOTES[s['sample_key']]))
    print('Frames',s['sample_key'],len(inds),flush=True)


def counterfactual_metrics(cases):
    rows=[]
    for s in cases:
        z=np.load(relocate_path(OUT/s['sample_key']/'counterfactual.npz'))['attention']
        for role,b in s['selected_bins'].items():
            p=z[0,b].ravel();p/=p.sum();k=int(np.ceil(.2*len(p)))
            for i in [1,2]:
                q=z[i,b].ravel();q/=q.sum()
                rows.append(dict(sample_key=s['sample_key'],dataset=s['dataset'],role=role,control=i,
                    tv=float(abs(p-q).sum()/2),spearman=float(spearmanr(p,q).statistic),
                    peak_same=int(p.argmax()==q.argmax()),top20_overlap=len(set(np.argsort(-p)[:k])&set(np.argsort(-q)[:k]))/k))
    write_csv(relocate_path(OUT/'counterfactual_metrics.csv'),rows)
    summary={}
    for role in ['early_peak','early_other','GT_peak']:
        r=[x for x in rows if x['role']==role]
        summary[role]={k:float(np.mean([x[k] for x in r])) for k in ['tv','spearman','peak_same','top20_overlap']}
        summary[role]['n_video']=len(set(x['sample_key'] for x in r));summary[role]['n_pairs']=len(r)
    write_json(relocate_path(OUT/'counterfactual_summary.json'),summary)
    return summary


def write_viewer(cases,summary):
    data={s['sample_key']:json.loads((relocate_path(OUT/s['sample_key']/'viewer_data.json')).read_text()) for s in cases}
    write_result_text((relocate_path(OUT/'viewer_data.js')), 'const DATA = '+json.dumps(data,ensure_ascii=False)+';\n')
    notes=[dict(sample_key=k,classification=v[0],note=v[1]) for k,v in NOTES.items()]
    write_json(relocate_path(OUT/'visual_review.json'),notes)
    rows=''.join(f'<tr><td><a href="{s["sample_key"]}/spatial.png">{s["sample_key"]}</a></td><td>{s["early_mass"]:.1%}</td><td>{html.escape(NOTES[s["sample_key"]][1])}</td></tr>' for s in cases)
    cf=summary['early_peak'];gt=summary['GT_peak']
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>프레임별 spatial attention</title>
<style>@font-face{font-family:KR;src:url('../hidden/fonts/NotoSansCJKkr-Regular.otf')}body{font:16px/1.65 KR,system-ui;max-width:1500px;margin:24px auto;padding:0 20px;color:#1b2b40}h1{font-size:28px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,button,input{font:inherit;padding:7px}#slider{width:100%}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.images canvas{width:100%;height:auto;background:#eee}#timeline{width:100%;height:150px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.note{background:#eef4fa;padding:16px}a{color:#175ab8}#query{font-size:19px;font-weight:bold}#where{color:#166534}img{max-width:100%}small{font-size:12px}</style>
<h1>초반 attention이 큰 영상의 프레임별 공간 지도</h1>
<p>GT가 첫 10% 이후에 있고 attention 최고점이 첫 10%에 있는 92개 중, 첫 10%가 attention 질량 20% 이상인 25개를 표시한다. 레이어 8–21, 모든 head·이벤트 쿼리 토큰 평균이다.</p>
<div class="controls"><label>영상 <select id="sample"></select></label><button id="prev">이전 프레임</button><button id="next">다음 프레임</button><button id="play">재생</button><button id="early">초반 최고점</button><button id="gt">GT 내부 최고점</button></div>
<p id="query"></p><p id="where"></p><canvas id="timeline" width="1450" height="150"></canvas><input id="slider" type="range" min="0" value="0"><p id="status"></p>
<div class="images"><div>원본 입력 프레임<canvas id="raw"></canvas></div><div>Query → video attention<canvas id="attn"></canvas><small>프레임 내부 합=1. 색: 낮음(보라) → 높음(노랑). 흰색 ×: 최고점.</small></div><div>Hidden cosine의 공간 순위<canvas id="cosine"></canvas><small>프레임 내부 percentile 0–1. 색 차이는 원래 cosine 차이의 크기를 나타내지 않는다.</small></div></div>
<p id="review" class="note"></p>
<p>Qwen3-VL은 샘플링된 프레임 두 장을 같은 temporal token 묶음으로 처리한다. 따라서 프레임 0·1, 2·3 등의 공간 지도는 공유된다. 슬라이더는 실제 입력 프레임 각각을 보여 준다. 샷 전환이 한 묶음 안에 들어간 경우 지도는 두 장 모두를 반영한다. GT는 초록 막대/음영, 초반 10%는 주황 음영, 현재 프레임은 빨강 선이다.</p>
<p>Attention overlay의 색 범위는 해당 temporal bin의 0–최대값이다. 서로 다른 프레임의 밝기를 직접 비교하지 말고 표시한 최고점 확률과 시간 곡선을 함께 확인한다. 검은 배경/벽 등에도 높은 attention이 생길 수 있으며, attention이나 hidden cosine을 독립적인 의미 정답으로 취급하지 않는다.</p>
<h2>실제 프레임 비교 예시</h2><img src="frame_attention_examples.png">
<h2>쿼리를 바꾼 대조 실험</h2><p>원래 영상을 그대로 두고 이벤트 문장만 “A dog catches a frisbee in a park.”와 “A violinist plays a violin on a stage.”로 교체했다. 원래 쿼리 재실행의 attention은 25개 모두 기존 값과 비트 단위로 같았다. 대조 쿼리의 토큰 수는 원래와 다르므로 쿼리 의미만 완전히 고립한 실험은 아니다.</p>
CF_TEXT
<p>25개 초반 최고점 patch에 대한 육안 기록이다. 공식 spatial GT나 객관적 정확도 평가가 아니며, 경계/샷 전환 사례는 불명확으로 남겼다. 대상이 화면에 존재하는 것과 attention 최고점이 그 대상을 보는 것, 그리고 해당 동작의 시점이 맞는 것은 각각 구분한다.</p>
<table><tr><th>샘플 / 상세 공간 지도</th><th>초반 10% 질량</th><th>초반 최고점 관찰</th></tr>REVIEW_ROWS</table>
<p><a href="spatial_metrics.csv">92개 공간 비교 수치</a> · <a href="counterfactual_metrics.csv">쿼리 교체 비교 수치</a> · <a href="visual_review.json">육안 관찰 기록</a> · <a href="../hidden/report.html">기존 전체 300개 보고서</a></p>
<script src="viewer_data.js"></script><script>
const sel=document.getElementById('sample'),slider=document.getElementById('slider');let current,frame=0,serial=0,timer=null;
for(const key of Object.keys(DATA)){let o=document.createElement('option');o.value=key;o.textContent=key;sel.appendChild(o)}
function color(v){v=Math.max(0,Math.min(1,v));return [Math.round(40+210*v),Math.round(5+235*v*v),Math.round(85*(1-v))]}
function timeline(){const c=document.getElementById('timeline'),ctx=c.getContext('2d'),w=c.width,h=c.height,d=current.duration;ctx.clearRect(0,0,w,h);const x=t=>40+t/d*(w-55);ctx.fillStyle='#fbe8d4';ctx.fillRect(x(0),0,x(.1*d)-x(0),h-25);ctx.fillStyle='#bce4c6';for(const [s,e] of current.gt_windows)ctx.fillRect(x(s),0,x(Math.min(e,d))-x(s),h-25);let max=Math.max(...current.temporal_attention);ctx.strokeStyle='#2563eb';ctx.lineWidth=2;ctx.beginPath();current.timestamps.forEach((t,i)=>{const y=115-current.temporal_attention[i]/max*85;i?ctx.lineTo(x(t),y):ctx.moveTo(x(t),y)});ctx.stroke();ctx.fillStyle='#166534';ctx.fillText('GT',4,15);for(const [s,e]of current.gt_windows){ctx.fillRect(x(s),6,x(Math.min(e,d))-x(s),7);ctx.fillText(s.toFixed(1)+'–'+e.toFixed(1)+'s',x(s),28)}ctx.strokeStyle='#d52232';ctx.beginPath();ctx.moveTo(x(current.frame_times[frame]),0);ctx.lineTo(x(current.frame_times[frame]),125);ctx.stroke();ctx.fillStyle='#333';for(let i=0;i<=10;i++)ctx.fillText((d*i/10).toFixed(0)+'s',x(d*i/10)-8,145);ctx.fillText('temporal attention',50,45)}
function canvasImage(id,img,values,max,peak){let c=document.getElementById(id);c.width=img.naturalWidth;c.height=img.naturalHeight;let ctx=c.getContext('2d');ctx.drawImage(img,0,0);if(!values)return;let H=current.height,W=current.width;for(let y=0;y<H;y++)for(let x=0;x<W;x++){let col=color(values[y][x]/max);ctx.fillStyle='rgba('+col.join(',')+',0.6)';ctx.fillRect(x*c.width/W,y*c.height/H,c.width/W+.3,c.height/H+.3)}if(peak){let flat=values.flat(),i=flat.indexOf(Math.max(...flat)),x=(i%W+.5)*c.width/W,y=(Math.floor(i/W)+.5)*c.height/H;ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x-5,y-5);ctx.lineTo(x+5,y+5);ctx.moveTo(x+5,y-5);ctx.lineTo(x-5,y+5);ctx.stroke()}}
function draw(){let call=++serial;let b=Math.min(Math.floor(frame/2),current.attention.length-1),a=current.attention[b],max=Math.max(...a.flat());slider.value=frame;let time=current.frame_times[frame],inside=current.gt_windows.some(([s,e])=>s<=time&&time<=e);document.getElementById('query').textContent=current.query;document.getElementById('where').textContent='GT '+JSON.stringify(current.gt_windows)+' s — '+(inside?'현재 프레임은 GT 내부':'현재 프레임은 GT 외부');document.getElementById('status').textContent=`입력 프레임 ${frame+1}/${current.frame_indices.length} · 원본 frame ${current.frame_indices[frame]} · ${time.toFixed(2)}초 · temporal bin ${b} (${current.timestamps[b].toFixed(2)}초) · 공간 최고점 ${(max*100).toFixed(1)}% · 균등 기준 ${(100/(current.height*current.width)).toFixed(2)}%`;document.getElementById('review').textContent=current.review[1];timeline();let img=new Image();img.onload=()=>{if(call!==serial)return;canvasImage('raw',img);canvasImage('attn',img,a,max,true);canvasImage('cosine',img,current.cosine_rank[b],1,false);document.body.dataset.rendered=current.sample_key+':'+frame};img.src=current.sample_key+'/frames/'+String(frame).padStart(4,'0')+'.jpg'}
function select(){current=DATA[sel.value];slider.max=current.frame_indices.length-1;frame=current.selected_bins.early_peak*2;draw()};sel.onchange=select;slider.oninput=()=>{frame=Number(slider.value);draw()};document.getElementById('prev').onclick=()=>{frame=Math.max(0,frame-1);draw()};document.getElementById('next').onclick=()=>{frame=Math.min(current.frame_indices.length-1,frame+1);draw()};document.getElementById('early').onclick=()=>{frame=current.selected_bins.early_peak*2;draw()};document.getElementById('gt').onclick=()=>{frame=current.selected_bins.GT_peak*2;draw()};document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='재생'}else{document.getElementById('play').textContent='정지';timer=setInterval(()=>{frame=(frame+1)%current.frame_indices.length;draw()},350)}};select();
</script></html>'''
    doc=doc.replace('REVIEW_ROWS',rows).replace('CF_TEXT',f'<p>초반 최고점 프레임에서 원래 쿼리와 대조 쿼리 사이 평균 spatial TV는 <b>{cf["tv"]:.3f}</b>, Spearman은 <b>{cf["spearman"]:.3f}</b>, 최고점 유지율은 <b>{cf["peak_same"]:.1%}</b>였다. GT 내부에서는 각각 {gt["tv"]:.3f}, {gt["spearman"]:.3f}, {gt["peak_same"]:.1%}였다. 쿼리에 따른 변화도 있으므로 모든 초반 attention이 쿼리와 무관하다고 단정할 수는 없지만, 육안으로 확인한 배경 최고점을 의미 있는 객체 주시로 해석하는 것은 부적절하다.</p>')
    write_result_text((relocate_path(OUT/'report.html')), doc)


def main():
    cases=json.loads((relocate_path(OUT/'strong_samples.json')).read_text())
    examples(cases)
    for s in cases:export_frames(s)
    summary=counterfactual_metrics(cases)
    write_viewer(cases,summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
