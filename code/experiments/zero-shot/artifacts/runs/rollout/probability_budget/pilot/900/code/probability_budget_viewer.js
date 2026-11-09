'use strict';
const cfg=window.SEARCH,base=new URL(cfg.store+'/',location.href),root=new URL(cfg.root+'/',location.href),$=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(x,k=3)=>Number.isFinite(x)?x.toFixed(k):'—',pct=(x,k=2)=>Number.isFinite(x)?(x>0&&x<1e-6?(100*x).toExponential(1):(100*x).toFixed(k))+'%':'—';
const seconds=x=>Number(x).toFixed(2).replace(/0+$/,'').replace(/\.$/,''),span=x=>x?x.map(seconds).join(' – ')+' s':'유효 구간 없음';
const iou=(a,b)=>a&&b?Math.max(0,Math.min(a[1],b[1])-Math.max(a[0],b[0]))/Math.max(1e-12,Math.max(a[1],b[1])-Math.min(a[0],b[0])):0;
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b),methodNames={levin:'LevinTS',native_mass:'Native mass',numeric_novelty:'Numeric novelty'};
let catalog=[],filtered=[],pageIndex=0,selectedKey=null,serial=0,tree=null,snap=null,query=null,intervals=[],picked=null,pathLeaf=0,cursor=0,sourceTrace=new Map(),supplement=new Map(),prefixes=[],playEnd=null,refreshing=false;
const pageSize=16,cache=new Map();
async function fetchJSON(url,optional=false){const r=await fetch(url,{cache:'no-store'});if(optional&&r.status===404)return null;if(!r.ok)throw Error(`HTTP ${r.status}: ${url}`);return r.json()}
const get=(path,optional=false)=>fetchJSON(new URL(path,base),optional);
function row(){return catalog.find(r=>r.key===selectedKey)}
function visible(n){return n&&n.created_step<=snap.logical_step}
function visited(n){return visible(n)&&n.visited_step!=null&&n.visited_step<=snap.logical_step}
function children(n){return (n.children||[]).filter(id=>visible(tree.nodes[id]))}
function pathTo(id){const out=[];while(id!=null){const n=tree.nodes[id];out.push(n);id=n.parent}return out.reverse()}
function textTo(id){return pathTo(id).filter(n=>n.token!=null&&!n.piece.startsWith('<|')).map(n=>n.piece).join('')}
function role(n){if(n.role)return n.role;const text=textTo(n.id);for(const [end,anchor] of [['start','The event happens in '],['end',' - ']]){const tail=text.split(anchor);if(tail.length<2)continue;const part=tail[tail.length-1];if(/^[0-9]*$/.test(part))return end+'.integer_or_decimal';if(/^[0-9]+\.[0-9]*$/.test(part))return end+'.fraction'}return 'format'}
function positionName(n){const r=role(n);return r==='format'?'응답 문구·형식':(r.startsWith('start')?'시작 시각':'종료 시각')+(r.endsWith('fraction')?' · 소수 자릿수':' · 정수부·소수점')}
function updateURL(){if(!selectedKey)return;const url=new URL(location.href);for(const [k,v] of Object.entries({sample:selectedKey,method:$('method').value,budget:$('budget').value,node:cursor,path:pathLeaf}))url.searchParams.set(k,v);if(!tree){url.searchParams.delete('node');url.searchParams.delete('path')}history.replaceState(null,'',url);window.dispatchEvent(new CustomEvent('rollout:context'))}
function listing(keepPage=false){
 filtered=catalog.filter(r=>(r.key+' '+r.query).toLowerCase().includes($('filter').value.toLowerCase())&&(!$('dataset').value||r.dataset===$('dataset').value)&&(!$('completion').value||(r.complete?'complete':'partial')===$('completion').value));
 if(!keepPage)pageIndex=0;pageIndex=Math.max(0,Math.min(pageIndex,Math.max(0,Math.ceil(filtered.length/pageSize)-1)));
 $('sample-count').textContent=`${catalog.filter(r=>r.complete).length}/${catalog.length} 완료`;
 $('list-count').textContent=`조건에 맞는 query ${filtered.length}개`;
 $('samples').innerHTML=filtered.slice(pageIndex*pageSize,(pageIndex+1)*pageSize).map(r=>`<button data-key="${esc(r.key)}" class="${r.key===selectedKey?'selected':''}" aria-current="${r.key===selectedKey?'true':'false'}"><small>${esc(r.dataset.replace('-timelens',''))} · ${r.complete?'완료':r.methods.length?'진행 '+r.methods.length+'/3':'대기'}</small><strong>${esc(r.query)}</strong><small class="sample-id">${esc(r.key)}</small></button>`).join('')||'<p class="small">조건에 맞는 query가 없습니다.</p>';
 $('samples').querySelectorAll('[data-key]').forEach(b=>b.onclick=()=>load(b.dataset.key));$('page-count').textContent=`${filtered.length?pageIndex+1:0} / ${Math.ceil(filtered.length/pageSize)}`;$('list-prev').disabled=pageIndex===0;$('list-next').disabled=(pageIndex+1)*pageSize>=filtered.length;
 const index=filtered.findIndex(r=>r.key===selectedKey);$('query-prev').disabled=index<=0;$('query-next').disabled=index<0||index>=filtered.length-1;
}
function play(pair,autoplay=true){if(!pair)return;const v=$('video');playEnd=autoplay?pair[1]:null;v.currentTime=pair[0];$('video-status').textContent=`${span(pair)}${autoplay?' 재생 중':''}`;if(autoplay)v.play().catch(()=>{$('video-status').textContent='재생 버튼을 눌러 영상을 시작하세요.'})}
function answers(){
 const r=row(),pairs=[['GT',r.gt,'var(--truth)',null],['Zero-shot greedy',query?.greedy?.span,'var(--greedy)',iou(query?.greedy?.span,r.gt)],['현재 후보',picked?.span,'var(--picked)',picked?iou(picked.span,r.gt):null]];
 $('answer-cards').innerHTML=pairs.map(([name,pair,color,score],i)=>`<div class="answer-card" style="--card:${color}"><small>${name}</small><strong>${span(pair)}</strong><span>${i===0?'사후 평가 기준':`IoU ${num(score)}`}</span><button data-play="${i}" ${pair?'':'disabled'}>▶ 구간 재생</button></div>`).join('');
 $('timeline').innerHTML=pairs.map(([name,pair,color],i)=>`<div class="timeline-row"><div class="timeline-label"><span style="color:${color}">${name} · ${span(pair)}</span></div><div class="track"><div class="bar" style="--bar:${color};left:${pair?100*pair[0]/r.duration:0}%;width:${pair?100*(pair[1]-pair[0])/r.duration:0}%"></div></div></div>`).join('')+`<div class="timeline-ticks"><span>0 s</span><span>${seconds(r.duration)} s</span></div>`;
 $('answer-cards').querySelectorAll('[data-play]').forEach(b=>b.onclick=()=>play(pairs[Number(b.dataset.play)][1]));
}
function bestCompletion(id){
 let best=null;for(const l of tree.leaves.slice(0,snap.leaf_count)){if(!l.valid)continue;const path=pathTo(l.node);if(path.some(n=>n.id===id)&&(!best||l.logp>best.logp))best=l}if(best)return best.node;
 let n=tree.nodes[id];while(visited(n)&&children(n).length)n=tree.nodes[children(n).reduce((a,b)=>tree.nodes[a].p>=tree.nodes[b].p?a:b)];return n.id;
}
function setLeaf(id,focus=null,seek=false){
 pathLeaf=id;const leaf=tree.nodes[id];picked=visited(leaf)&&leaf.span?intervals.find(i=>same(i.span,leaf.span))||null:null;
 if(focus!=null)cursor=focus;else{const forks=pathTo(id).filter(n=>visited(n)&&children(n).length>1);cursor=forks[0]?.id??tree.nodes[id].parent??0}
 if(!pathTo(pathLeaf).some(n=>n.id===cursor))cursor=0;renderRoute();answers();renderCandidates();renderPlot();updateURL();if(seek&&picked)play(picked.span,false);
}
function intervalPick(index,seek=true){const item=intervals[index];if(item)setLeaf(item.leaf,null,seek)}
function distFor(n){if(n.top30)return {top:n.top30,kind:'native'};const key=JSON.stringify(prefixes[n.id]);const added=supplement.get(key);if(added)return {top:added.top30,kind:'replay',meta:added};const old=sourceTrace.get(key);if(old)return {top:old.top.map(t=>({token:t.id,text:t.token,p:t.probability})),kind:'reference',meta:old};return null}
function branchReason(n,d){
 const r=role(n),cs=children(n);if(!visited(n))return n.status==='pending'||!n.visited_step?'현재 예산에서 아직 확장하지 않은 prefix입니다. 분기를 하지 않기로 결정한 위치와 구분합니다.':'현재 checkpoint 뒤에 계산한 위치입니다.';
 if(cs.length>1)return `정수부·소수점 위치에서 native top-p 0.99와 확률 1% 기준을 통과한 ${cs.length}개 token을 추적했습니다.`;
 if(n.status==='valid'||n.status==='invalid')return 'EOS로 응답이 끝난 위치입니다. EOS 자체의 선택 근거는 문장 속 EOS를 눌러 확인합니다.';
 if(n.status==='context_limit')return '모델 context 한도 때문에 이 prefix의 다음 분포를 계산하지 않았습니다.';
 if(n.status==='dead')return '확률·정책 기준을 통과한 다음 token이 없어 이 경로를 끝냈습니다.';
 if(r.endsWith('fraction'))return '소수 자릿수는 greedy 정책입니다. 대안 확률이 높아도 분기하지 않습니다. 분포를 표시하는 것과 branch entropy 측정은 별개입니다.';
 if(r==='format')return '응답 문구·형식은 greedy 정책입니다. 정수부 숫자·소수점 위치에서만 분기하므로 가장 높은 확률 token 하나를 추적했습니다.';
 if((n.decision_reason||d?.meta?.decision_reason)==='single_probability_candidate')return '이 숫자 위치에서 native top-p 0.99와 절대 확률 1%를 통과한 token이 하나뿐입니다.';
 if(d){const eligible=nativeEligible(d.top);if(eligible.size<=1)return '이 숫자 위치에서 native top-p 0.99와 절대 확률 1%를 통과한 token이 하나뿐입니다.'}
 return '숫자 위치이지만 확률 필터와 숫자 형식 정책을 거친 대안이 하나뿐입니다. 숫자가 아닌 대안과 소수 자릿수만 다른 atomic token은 추가 분기하지 않습니다.';
}
function nativeEligible(top){let c=0;const ids=new Set();for(const t of top){if(c>=.99)break;if(t.p<.01)break;ids.add(t.token);c+=t.p}return ids}
function distribution(){
 const n=tree.nodes[cursor],d=distFor(n),cs=children(n),allowed=new Set(cs.map(id=>tree.nodes[id].token)),current=pathTo(pathLeaf).find(x=>x.parent===cursor),reason=branchReason(n,d);
 let html=`<h3>이 위치의 token distribution</h3><p class="why">${esc(reason)}</p>`;
 if(!visited(n)){html+='<p>탐색하지 않은 위치의 분포를 이미 관측한 것처럼 표시하지 않습니다.</p>';$('token-distribution').innerHTML=html;return}
 if(!d){html+='<p class="notice">이전 기록에 분포가 없어 보충 계산을 기다리는 위치입니다. 자식 token의 확률만으로 전체 분포를 추정하지 않습니다.</p>';$('token-distribution').innerHTML=html;return}
 const top=d.top,eligible=nativeEligible(top),policy=role(n),max=top[0]?.p||1;
 function tokenStatus(t){if(allowed.has(t.token))return cs.length>1?'추적한 분기':'추적 · greedy 또는 유일 대안';if(t.p<.01)return '제외 · 확률 1% 미만';if(!eligible.has(t.token))return '제외 · top-p 0.99 밖';if(policy.endsWith('fraction'))return '제외 · 소수 자릿수 greedy 정책';if(policy==='format')return '제외 · 문구·형식 greedy 정책';return '제외 · 숫자 형식·atomic token 정책'}
 const bars=top.map(t=>`<div class="prob-row ${allowed.has(t.token)?'chosen':''}"><div class="prob-name"><code title="Token ID ${t.token}">${esc(t.text||'∅')}</code></div><div><small>${esc(tokenStatus(t))}</small><div class="prob-track"><span style="width:${100*t.p/max}%"></span></div></div><div class="prob-value">${pct(t.p,t.p<.0001?5:2)}${current?.token===t.token?'<span class="badge">현재 경로</span>':''}</div></div>`);
 html+=`<p>전체 vocabulary의 native softmax 확률입니다. 아래 token 사이에서 다시 정규화하지 않습니다. 막대 길이는 이 위치의 최대 확률에 비례합니다.</p><div class="prob-list">${bars.slice(0,8).join('')}</div><details><summary>나머지 상위 ${Math.max(0,top.length-8)}개 token 보기</summary><div class="prob-list">${bars.slice(8).join('')}</div></details>`;
 const provenance=d.kind==='native'?'탐색 시 직접 저장한 분포':d.kind==='replay'?(d.meta.source==='same_input_prefix_other_method'?'동일 입력·prefix에서 다른 방법이 직접 저장한 분포':'동일 입력·prefix를 single-token native 실행으로 재계산한 분포 · 비용은 search와 별도'):'동일 입력 hash·동일 token prefix의 기존 deterministic native trace · 참고 재사용';
 html+=`<p class="provenance">${esc(provenance)}${d.meta?.max_edge_probability_error!=null?' · 저장된 경로 확률과 최대 차이 '+num(d.meta.max_edge_probability_error,6):''}</p>`;$('token-distribution').innerHTML=html;
}
function renderRoute(){
 if(!tree)return;const path=pathTo(pathLeaf),n=tree.nodes[cursor],next=path.find(x=>x.parent===cursor),forks=path.filter(n=>visited(n)&&children(n).length>1);
 $('path-summary').textContent=`${picked?span(picked.span)+' · '+picked.paths.length+'개 sequence':'미완성 경로'} · 이 경로의 분기 ${forks.length}곳`;
 $('path-select-label').hidden=!picked||picked.paths.length<2;if(picked){$('path-select').innerHTML=[...picked.paths].sort((a,b)=>b.logp-a.logp).map((p,i)=>`<option value="${p.node}" ${p.node===pathLeaf?'selected':''}>${i===0?'대표 경로':'경로 '+(i+1)} · native ${pct(Math.exp(p.logp),4)} · ${esc(textTo(p.node).slice(20))}</option>`).join('')}
 $('clipping-note')?.remove();const match=textTo(pathLeaf).match(/in ([0-9.]+) - ([0-9.]+)/);if(picked&&match&&!same([Number(match[1]),Number(match[2])],picked.span)){const note=document.createElement('p');note.id='clipping-note';note.className='small';note.textContent='생성 원문은 아래 문장에 그대로 표시합니다. 후보 구간은 [0, 영상 길이]로 clipping한 값입니다.';$('sentence').after(note)}
 $('sentence').innerHTML=path.filter(n=>n.token!=null).map(n=>{const p=tree.nodes[n.parent],fork=visited(p)&&children(p).length>1;return `<button class="token ${fork?'branch-token':''} ${p.id===cursor?'active':''} ${n.piece.startsWith('<|')?'eos':''}" data-position="${p.id}" aria-pressed="${p.id===cursor}" title="${esc(positionName(p))} · 이 token 직전 분포"><code>${esc(n.piece.startsWith('<|')?'EOS':n.piece)}</code><small>${pct(n.p,1)}</small></button>`}).join('');
 if(!next&&visible(n)&&!visited(n))$('sentence').innerHTML+='<span class="small"> · 이 prefix부터 미탐색</span>';
 $('sentence').querySelectorAll('[data-position]').forEach(b=>b.onclick=()=>{cursor=Number(b.dataset.position);renderRoute();updateURL()});
 $('position-label').textContent=`${positionName(n)}${next?' · 다음 token: '+(next.piece.startsWith('<|')?'EOS':JSON.stringify(next.piece)):''} · prefix depth ${n.depth}`;
 const positions=path.slice(0,-1).map(n=>n.id),idx=positions.indexOf(cursor);$('token-prev').disabled=idx<=0;$('token-next').disabled=idx<0||idx>=positions.length-1;
 $('route-nav').innerHTML=forks.map(f=>{const selected=path.find(x=>x.parent===f.id);return `<button data-node="${f.id}" class="${f.id===cursor?'active':''}"><span class="small">${role(f).startsWith('start')?'시작':'종료'}</span><span class="fork-context">${esc(textTo(f.id).slice(-30))}</span><b>${esc(selected?.piece)}</b><span class="small">${children(f).length}갈래 ↗</span></button>`}).join('')||'<p class="small">이 경로에는 관측된 분기 지점이 없습니다.</p>';
 $('route-nav').querySelectorAll('[data-node]').forEach(b=>b.onclick=()=>{cursor=Number(b.dataset.node);renderRoute();updateURL()});
 $('prefix-window').textContent=textTo(cursor)||'(응답 시작 전)';
 const cs=children(n);
 $('branch-options').innerHTML=`<p class="branch-reason">${esc(visited(n)?cs.length>1?'이 위치에서 이어갈 token을 선택하세요. 선택하면 그 자식 경로 중 native 확률이 가장 높은 유효 완성 응답을 보여줍니다. 유효 완성이 없으면 관측된 prefix까지만 보여줍니다.':'이 위치에서 추적한 다음 token':'이 prefix는 현재 checkpoint에서 미탐색입니다.')}</p><div class="branches">${visited(n)?cs.map(id=>{const c=tree.nodes[id];return `<button data-child="${id}" aria-pressed="${next?.id===id}"><code>${esc(c.piece.startsWith('<|')?'EOS':c.piece)}</code><small>${pct(c.p)}</small></button>`}).join(''):''}</div>`;
 $('branch-options').querySelectorAll('[data-child]').forEach(b=>b.onclick=()=>{const child=Number(b.dataset.child);setLeaf(bestCompletion(child),cursor,true)});
 const alternatives=visited(n)&&cs.length>1?cs.filter(id=>id!==next?.id).sort((a,b)=>tree.nodes[b].p-tree.nodes[a].p):[];
 if(alternatives.length){const other=bestCompletion(alternatives[0]),otherPath=pathTo(other),chosenIndex=path.findIndex(x=>x.parent===cursor),otherIndex=otherPath.findIndex(x=>x.parent===cursor);const answer=(p,j)=>esc(p.slice(0,j).filter(x=>x.token!=null&&!x.piece.startsWith('<|')).map(x=>x.piece).join(''))+`<mark>${esc(p[j]?.piece)}</mark>`+esc(p.slice(j+1).filter(x=>!x.piece.startsWith('<|')).map(x=>x.piece).join(''));
 $('branch-preview').innerHTML=`<div class="branch-comparison"><p>현재 경로의 완성 응답 · ${span(picked?.span)}</p><code>${answer(path,chosenIndex)}</code></div><div class="branch-comparison"><p>확률이 가장 높은 다른 분기 · ${span(tree.nodes[other].span)}${tree.nodes[other].status==='pending'?' · 미완성':''}</p><code>${answer(otherPath,otherIndex)}</code></div>`;
 }else $('branch-preview').innerHTML='';distribution();
}
function renderCandidates(){
 const term=$('candidate-filter').value.toLowerCase(),sort=$('candidate-sort').value,items=intervals.map((x,index)=>({...x,index})).filter(x=>span(x.span).includes(term));items.sort(sort==='start'?(a,b)=>a.span[0]-b.span[0]||a.span[1]-b.span[1]:sort==='iou'?(a,b)=>iou(b.span,row().gt)-iou(a.span,row().gt):(a,b)=>b.mass-a.mass);
 $('candidate-count').textContent=`${items.length} / ${intervals.length}개`;
 $('candidates').innerHTML=items.map(x=>`<tr class="${picked&&same(x.span,picked.span)?'selected':''}"><td><button data-interval="${x.index}" aria-pressed="${!!picked&&same(x.span,picked.span)}">${span(x.span)}</button>${same(x.span,snap.selected_span)?'<span class="native-mass">최대 native 질량</span>':''}</td><td>${pct(x.weight)}<span class="native-mass">native ${pct(x.mass,3)}</span></td><td>${num(iou(x.span,row().gt))}</td></tr>`).join('');$('candidates').querySelectorAll('[data-interval]').forEach(b=>b.onclick=()=>intervalPick(Number(b.dataset.interval)));
}
function niceStep(range){const raw=range/5,power=10**Math.floor(Math.log10(raw)),x=raw/power;return (x<=1?1:x<=2?2:x<=5?5:10)*power}
function ticks(lo,hi){const step=niceStep(hi-lo),out=[];for(let t=Math.ceil((lo-1e-9)/step)*step;t<=hi+1e-9;t+=step)out.push(Math.max(0,t));if(!out.length||hi-out[out.length-1]>step*.4)out.push(hi);return out}
function renderPlot(){
 if(!tree)return;const r=row(),T=r.duration,L=61,top=24,size=346,bottom=top+size;
 let xlo=0,ylo=0,range=T;if($('plot-range').value==='fit'){
 const points=intervals.map(x=>x.span).concat([r.gt,query.greedy.span].filter(Boolean));if(points.length){const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);range=Math.min(T,Math.max(1,Math.max(maxx-minx,maxy-miny)*1.2));xlo=Math.max(0,Math.min(T-range,(minx+maxx-range)/2));ylo=Math.max(0,Math.min(T-range,(miny+maxy-range)/2))}}
 const x=t=>L+(t-xlo)/range*size,y=t=>bottom-(t-ylo)/range*size;
 const grid=ticks(xlo,xlo+range).map(t=>`<line class="grid-line" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${bottom}"/><text class="tick-label" x="${x(t)}" y="${bottom+19}" text-anchor="middle">${seconds(t)}</text>`).join('')+ticks(ylo,ylo+range).map(t=>`<line class="grid-line" x1="${L}" x2="${L+size}" y1="${y(t)}" y2="${y(t)}"/><text class="tick-label" x="${L-9}" y="${y(t)+3}" text-anchor="end">${seconds(t)}</text>`).join('');
 const dots=[...intervals.map((item,index)=>({item,index}))].sort((a,b)=>b.item.mass-a.item.mass).map(({item,index})=>`<circle class="candidate-point ${picked&&same(item.span,picked.span)?'selected':''}" data-point="${index}" cx="${x(item.span[0])}" cy="${y(item.span[1])}" r="${2.5+Math.sqrt(item.weight)*16}" tabindex="0" role="button" aria-label="후보 ${span(item.span)} · 비중 ${pct(item.weight)}"><title>${span(item.span)} · 비중 ${pct(item.weight)} · IoU ${num(iou(item.span,r.gt))}</title></circle>`).join('');
 const gt=r.gt,g=query.greedy.span;
 const marker=(pair,type)=>{if(!pair)return '';const px=x(pair[0]),py=y(pair[1]);return type==='gt'?`<path d="M${px-6} ${py-6}l12 12m0 -12l-12 12" stroke="#277c68" stroke-width="2.5" fill="none"/>`:`<path d="M${px} ${py-7}l7 7l-7 7l-7 -7Z" stroke="#2678b5" stroke-width="2" fill="white" fill-opacity=".6"/>`};
 $('scatter').innerHTML=`<svg viewBox="0 0 438 427" role="img" aria-label="후보 interval 분포: 시작·종료 시각, 초 단위 눈금"><defs><clipPath id="plot-clip"><rect x="${L}" y="${top}" width="${size}" height="${size}"/></clipPath></defs><rect x="${L}" y="${top}" width="${size}" height="${size}" fill="#fbfcfa"/>${grid}<g clip-path="url(#plot-clip)"><polygon points="${x(0)},${y(0)} ${x(T)},${y(T)} ${x(T)},${y(0)}" fill="#ebeeea"/><path d="M${x(0)} ${y(0)}L${x(T)} ${y(T)}" stroke="#c7d1c5" stroke-dasharray="4 4"/>${dots}<g pointer-events="none">${marker(gt,'gt')}${marker(g,'greedy')}${picked?`<circle cx="${x(picked.span[0])}" cy="${y(picked.span[1])}" r="${6+Math.sqrt(picked.weight)*16}" stroke="#b96328" stroke-width="1.5" fill="none"/>`:''}</g></g><rect x="${L}" y="${top}" width="${size}" height="${size}" stroke="#c3d0c3" fill="none" pointer-events="none"/><text class="axis-label" x="${L+size/2}" y="${bottom+44}" text-anchor="middle">시작 시각 (s)</text><text class="axis-label" transform="translate(16 ${top+size/2}) rotate(-90)" text-anchor="middle">종료 시각 (s)</text></svg>`;
 $('scatter').querySelectorAll('[data-point]').forEach(p=>{const choose=()=>intervalPick(Number(p.dataset.point));p.onclick=choose;p.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose()}}});
 const s=snap.distribution,cov=s.covariance;$('distribution-stats').textContent=`관측 interval entropy ${num(s.entropy)} bits · start/end variance ${num(cov?.[0]?.[0])} / ${num(cov?.[1]?.[1])} s². 회색 영역은 종료≤시작인 무효 구간입니다. 미탐색 부분을 제외한 통계입니다.`;
}
function details(){
 const s=snap,r=row(),oracle=Math.max(0,...intervals.map(x=>iou(x.span,r.gt))),method=$('method').value;
 $('query-detail').innerHTML=`<div class="search-summary"><span><b>${s.actual_inputs.toLocaleString()} / ${s.budget.toLocaleString()}</b><small>전체 경로의 decoder inputs</small></span><span><b>${num(s.cost.wall_seconds,2)} s</b><small>이 방법의 누적 search 비용</small></span><span><b>${intervals.length}개</b><small>서로 다른 exact interval</small></span><span><b>${pct(s.unseen_valid_mass_upper_bound)}</b><small>미탐색 유효 질량 상한</small></span><span><b>${num(iou(s.selected_span,r.gt))} / ${num(oracle)}</b><small>최대 native 질량 후보 / oracle IoU</small></span></div><p class="method-note">${method==='levin'?'LevinTS: 깊이 대비 native path probability가 높은 prefix부터 확장합니다.':method==='native_mass'?'Native mass: 초기 subtree의 남은 native 질량에 비례해 연산을 배분합니다.':'Numeric novelty: 남은 native 질량에 시간적 새로움을 최대 1.1배로 보정해 배분합니다.'} · ${s.reason==='mass_target_reached'?'미탐색 상한 1%에 도달해 종료':s.reason==='decoder_budget_exhausted'?'누적 decoder 예산에 도달':'탐색 종료: '+esc(s.reason)}. 최대 native 질량 후보는 proposal 진단값이며 independent verifier의 final answer가 아닙니다.</p>`;
 $('cost-details').innerHTML=`<p>그룹은 첫 numeric fork의 subtree이며 semantic mode가 아닙니다. Prefill은 query당 공유하며 search 상한 밖의 비용입니다. 이 query의 prefill ${query.input?.input_tokens?.toLocaleString()??'—'} tokens · ${num(query.input?.prefill_wall_seconds)} s, greedy decode ${num(query.greedy?.cost?.wall_seconds)} s.</p><div class="rows"><table><thead><tr><th>Subtree</th><th>남은 native 질량</th><th>Decoder inputs</th><th>유효 경로</th><th>시간적 새로움</th></tr></thead><tbody>${s.groups.map(g=>`<tr><td>${g.id}</td><td>${pct(g.pending_mass,3)}</td><td>${g.spent}</td><td>${g.valid_leaves}</td><td>${method==='numeric_novelty'?(g.valid_leaves?num(g.novelty):'관측 전'):'사용 안 함'}</td></tr>`).join('')}</tbody></table></div><p>완성 유효 ${pct(s.mass.valid,3)} · 무효 ${pct(s.mass.invalid,3)} · 확률 필터 제외 ${pct(s.mass.filtered,3)} · greedy·숫자 형식 정책 제외 ${pct(s.mass.policy,3)} · frontier ${pct(s.frontier_mass,3)} · context 한도 ${pct(s.mass.context,3)}.</p><p>실제 decoder inputs ${s.cost.decoder_inputs} (replay ${s.cost.replay_inputs}) · CUDA decode ${num(s.cost.decode_cuda_seconds)} s · KV 저장·복원 ${num(s.cost.kv_store_cuda_seconds+s.cost.kv_restore_cuda_seconds)} s · reward ${num(s.cost.reward_wall_seconds)} s. 보충 token distribution 계산은 이 search 비용에 합산하지 않습니다.</p>`;
}
async function loadTraces(key,seq){
 const [old,extra]=await Promise.all([cfg.traces?fetchJSON(new URL(cfg.traces+'/'+key+'/index.json',location.href),true):null,get('token_distributions/'+key+'.json',true)]);if(seq!==serial)return;
 sourceTrace=new Map();supplement=new Map();const hashes=query.input?.hashes;
 const hashesMatch=h=>h&&hashes&&Object.keys(hashes).every(k=>h[k]===hashes[k]);
 if(old?.validation?.passed&&hashesMatch(old.input_hashes))for(const n of Object.values(old.nodes))sourceTrace.set(JSON.stringify(n.prefix_ids),n);
 if(extra?.complete&&hashesMatch(extra.input_hashes))for(const n of extra.nodes)supplement.set(JSON.stringify(n.prefix_ids),n);
 renderRoute();
}
async function load(key,fromURL=false){
 const different=selectedKey!==key,oldPair=picked?.span,oldPrefix=prefixes[cursor],seq=++serial,r=catalog.find(r=>r.key===key);if(!r)return;selectedKey=key;listing(true);
 if(different){sourceTrace=new Map();supplement=new Map();query=null;tree=null;snap=null;cursor=0;pathLeaf=0;picked=null;prefixes=[];$('video').src=new URL(r.video,root);playEnd=null;$('video-status').textContent='GT·greedy·현재 후보의 ‘구간 재생’으로 비교합니다.';$('query-title').textContent=r.query;$('query-meta').textContent=`${r.dataset} · ${seconds(r.duration)} s · ${r.key}`;$('answer-cards').innerHTML='';$('timeline').innerHTML='';$('explorer').hidden=true;$('interval-panel').hidden=true}
 $('query-detail').textContent='저장된 탐색 결과를 읽는 중…';$('explorer').classList.add('loading');
 try{
 const method=$('method').value,budget=$('budget').value,cacheKey=key+'/'+method;
 const [q,m]=await Promise.all([get('queries/'+key+'.json',true),cache.has(cacheKey)?cache.get(cacheKey):get('search/'+key+'/'+method+'.json',true)]);if(seq!==serial)return;
 query=q;if(!q||!m){tree=null;picked=null;answers();$('explorer').hidden=true;$('interval-panel').hidden=true;$('query-detail').innerHTML='<p class="notice">이 query·방법은 실행 중이거나 대기 중입니다. 진행 상태는 자동 갱신됩니다.</p>';updateURL();return}
 cache.set(cacheKey,m);tree=m;snap=m.snapshots[budget];prefixes=[];for(const n of tree.nodes)prefixes[n.id]=n.parent==null?[]:prefixes[n.parent].concat([n.token]);
 const map=new Map();for(const l of m.leaves.slice(0,snap.leaf_count)){if(!l.valid)continue;const k=JSON.stringify(l.span),item=map.get(k)||{span:l.span,mass:0,leaf:l.node,logp:l.logp,paths:[]};item.paths.push(l);item.mass+=Math.exp(l.logp);if(l.logp>item.logp){item.leaf=l.node;item.logp=l.logp}map.set(k,item)}
 intervals=[...map.values()].sort((a,b)=>b.mass-a.mass);const total=intervals.reduce((s,x)=>s+x.mass,0);for(const i of intervals)i.weight=i.mass/total;
 $('explorer').hidden=false;$('interval-panel').hidden=false;const preserve=!different&&intervals.find(x=>same(x.span,oldPair));const initial=preserve||intervals.find(x=>same(x.span,snap.selected_span))||intervals[0];
 let target=initial?.leaf??bestCompletion(0),focus=null;if(!different&&oldPrefix){const p=pathTo(target).find(n=>same(prefixes[n.id],oldPrefix));if(p)focus=p.id}
 if(fromURL){const params=new URL(location.href).searchParams,leaf=Number(params.get('path')),node=Number(params.get('node'));if(params.has('path')&&visible(tree.nodes[leaf]))target=leaf;if(params.has('node')&&pathTo(target).some(n=>n.id===node))focus=node}
 setLeaf(target,focus);details();loadTraces(key,seq).catch(e=>{if(seq===serial)console.warn('Supplemental trace unavailable',e)});
 }catch(e){if(seq===serial){$('query-detail').innerHTML=`<p class="notice error">결과를 불러오지 못했습니다. ${esc(e.message)}</p>`;console.error(e)}}finally{if(seq===serial)$('explorer').classList.remove('loading')}
}
async function refresh(initial=false){
 if(refreshing)return;refreshing=true;try{catalog=await fetchJSON(new URL(cfg.catalog,location.href));listing(true);const state=await get('metadata/status.json');$('status').textContent=`${state.state==='complete'?'완료':'진행 중'} · 현재 범위 ${catalog.filter(r=>r.complete).length}/${catalog.length} query\n누적 ${state.completed??0}/${state.total??900} · ${state.timestamp?.replace('T',' ').slice(0,19)||''}`;
 if(initial){const p=new URL(location.href).searchParams;if(Object.keys(methodNames).includes(p.get('method')))$('method').value=p.get('method');if(['512','1024','2048'].includes(p.get('budget')))$('budget').value=p.get('budget');const key=catalog.some(r=>r.key===p.get('sample'))?p.get('sample'):catalog[0]?.key;if(key){const idx=catalog.findIndex(r=>r.key===key);pageIndex=Math.floor(idx/pageSize);await load(key,true)}}else if(selectedKey&&(!tree||supplement.size===0)){if(!tree)await load(selectedKey);else await loadTraces(selectedKey,serial)}
 }catch(e){$('status').textContent='상태 갱신 실패 · '+e.message}finally{refreshing=false}}
for(const id of ['filter','dataset','completion'])$(id).addEventListener(id==='filter'?'input':'change',()=>listing());
$('list-prev').onclick=()=>{pageIndex--;listing(true)};$('list-next').onclick=()=>{pageIndex++;listing(true)};
for(const [id,delta] of [['query-prev',-1],['query-next',1]])$(id).onclick=()=>{const idx=filtered.findIndex(r=>r.key===selectedKey),target=filtered[idx+delta];if(target){pageIndex=Math.floor((idx+delta)/pageSize);load(target.key)}};
for(const id of ['method','budget'])$(id).onchange=()=>{if(selectedKey)load(selectedKey)};
for(const [id,delta] of [['token-prev',-1],['token-next',1]])$(id).onclick=()=>{const positions=pathTo(pathLeaf).slice(0,-1).map(n=>n.id),idx=positions.indexOf(cursor);if(positions[idx+delta]!=null){cursor=positions[idx+delta];renderRoute();updateURL()}};
$('path-select').onchange=()=>setLeaf(Number($('path-select').value),null);$('plot-range').onchange=renderPlot;$('candidate-filter').oninput=renderCandidates;$('candidate-sort').onchange=renderCandidates;
$('video').addEventListener('timeupdate',()=>{if(playEnd!=null&&$('video').currentTime>=playEnd){$('video').pause();playEnd=null}});
$('video').addEventListener('error',()=>{$('video-status').textContent='영상을 불러오지 못했습니다. 파일 경로와 서버 응답을 확인해야 합니다.'});
refresh(true);setInterval(()=>refresh(),30000);
