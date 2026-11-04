'use strict';
(()=>{
const $=id=>document.getElementById(id),BASE=window.ROLLOUT_DISTRIBUTIONS_BASE||window.ROLLOUT_ASSETS+'/../token_traces';
const COLORS=['#2678b5','#c4672f','#218b69','#9264b2','#a98a22','#ba4e85','#597d49','#597b96'];
const ds={data:null,candidate:null,focus:null,trace:null,step:0,serial:0,drawCounter:0,cache:new Map(),arrays:new Map(),ranked:new Map(),limits:new Map(),vocab:null,timer:null,disabled:new Set(),visibilityInitialized:false,autoJump:false,autoK:true};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const visible=t=>t===' '?'공백':String(t).replace(/\n/g,'↵').replace(/ /g,'·');
const interval=c=>c.pair?c.pair.map(x=>Number(x.toFixed(2))).join(' – ')+'초':'유효한 구간 없음';
const probability=p=>p===0?'0%':p>=.000001?`${(100*p).toFixed(5)}%`:`${(100*p).toExponential(3)}%`;
const pathProbability=log=>{if(log===0)return '1';if(!Number.isFinite(log))return '0';const x=log/Math.LN10,e=Math.floor(x);return `${Math.pow(10,x-e).toFixed(4)} × 10^${e}`;};
const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
const tokenProbability=(node,id)=>node?.observed.find(t=>t.id===id)?.probability??node?.top.find(t=>t.id===id)?.probability??0;
const nodeFor=(c,step)=>c&&ds.trace?.nodes[ds.trace.candidates[String(c.index)]?.nodes[step]];
function setFocus(c,j){ds.focus=c;ds.step=j;ds.disabled.clear();ds.visibilityInitialized=false;renderStrip();renderNavigator();}
async function render(data,candidate){
 const serial=++ds.serial;clearTimeout(ds.timer);ds.data=data;ds.candidate=candidate;ds.trace=null;ds.limits.clear();
 setFocus(candidate,candidate.branch?.position??0);$('distributionTree').innerHTML='';$('distributionStatus').textContent='저장된 분포를 불러오는 중…';
 try{
  let trace=ds.cache.get(data.sample.key);
  if(!trace){const r=await fetch(`${BASE}/${data.sample.key}/index.json`,{cache:'no-store'});if(r.status===404){await pendingStatus(serial);return;}if(!r.ok)throw Error(`분포 자료 로드 실패 (${r.status})`);trace=await r.json();ds.cache.set(data.sample.key,trace);if(ds.cache.size>5)ds.cache.delete(ds.cache.keys().next().value);}
  if(serial!==ds.serial)return;ds.trace=trace;renderStrip();
  $('distributionStatus').textContent='원래 응답의 정확한 토큰 이력을 다시 입력해 측정했습니다. 출력 형식은 바꾸지 않았습니다. ·는 공백, ↵는 줄바꿈입니다.';
  await draw(serial);
 }catch(e){if(serial===ds.serial)$('distributionStatus').textContent=e.message;}
}
async function pendingStatus(serial){
 let note='이 샘플의 probability distribution는 아직 측정 중입니다. 위의 예측 구간과 IoU는 원래 실험의 결과입니다.';
 try{const r=await fetch(BASE+'/../metadata/distribution_status.json',{cache:'no-store'});if(r.ok){const s=await r.json();note+=` 전체 ${s.completed}/${s.total}개 완료.`;}}catch(e){}
 if(serial!==ds.serial)return;$('distributionStatus').textContent=note;
 ds.timer=setTimeout(()=>{if(serial===ds.serial)render(ds.data,ds.candidate);},30000);
}
const shortProbability=p=>p===null?'—':p===0?'0%':`${Number((100*p).toPrecision(4))}%`;
const shortProduct=log=>{if(log===null)return '—';if(log===0)return '1';if(!Number.isFinite(log))return '0';let e=Math.floor(log/Math.LN10),m=Number(Math.pow(10,log/Math.LN10-e).toFixed(2));if(m>=10){m=1;e++;}return `${m.toFixed(2)}e${e}`;};
function stepValues(c,j){const n=nodeFor(c,j);if(!n)return {p:null,log:null};const p=tokenProbability(n,c.token_ids[j]);return {p,log:p>0?n.prefix_log_probability+Math.log(p):-Infinity};}
function traceMatrix(c,start,end,common=false){
 const indices=Array.from({length:end-start},(_,i)=>i+start),branch=ds.candidate.branch;
 const buttons=indices.map(j=>{const t=c.tokens[j],point=branch&&c.index===ds.candidate.index&&j===branch.position,active=j===ds.step&&(common||c.index===ds.focus.index);
  return `<button data-dist-step="${j}" data-trace-candidate="${c.index}" class="${active?'active ':''}${point?'branch-point':''}" aria-pressed="${active}" title="${j+1}번째 토큰 · ${esc(t)}"><span class="token-step">${j+1}${point?' · 분기':''}</span><code>${esc(t==='<|im_end|>'?'EOS':visible(t))}</code></button>`;}).join('');
 const probabilities=indices.map(j=>{const v=stepValues(c,j);return `<span class="trace-value" data-step="${j}" data-probability="${v.p??''}" title="P(${esc(c.tokens[j])} | prefix) = ${v.p===null?'측정 중':v.p.toPrecision(9)}">${shortProbability(v.p)}</span>`;}).join('');
 const products=indices.map(j=>{const v=stepValues(c,j);return `<span class="trace-value" data-step="${j}" data-log-probability="${v.log??''}" title="${v.log===null?'측정 중':pathProbability(v.log)}">${shortProduct(v.log)}</span>`;}).join('');
 return `<div class="trace-scroll"><div class="trace-matrix" style="--steps:${indices.length};min-width:${66+54*indices.length}px"><div class="trace-token-row"><span class="trace-row-label">Step / 토큰</span>${buttons}</div><div class="trace-p-row"><span class="trace-row-label">확률 pᵢ</span>${probabilities}</div><div class="trace-product-row"><span class="trace-row-label">누적곱 ∏pᵢ</span>${products}</div></div></div>`;
}
function renderStrip(){
 const scrolls=[...$('distTokenStrip').querySelectorAll('.trace-scroll')].map(el=>el.scrollLeft);
 const main=ds.candidate,parent=main.branch?ds.data.candidates[main.branch.parent-1]:main;
 const paths=[{c:parent,title:parent.index===1?'greedy 응답':`부모 후보 #${parent.index}`,color:COLORS[0]}];
 if(main.index!==parent.index)paths.push({c:main,title:`분기 후보 #${main.index}`,color:COLORS[1]});
 if(!paths.some(p=>p.c.index===ds.focus.index))paths.push({c:ds.focus,title:'탐색 중인 경로',color:COLORS[2]});
 let fork=0;
 if(paths.length>1){const max=Math.min(...paths.map(p=>p.c.token_ids.length));while(fork<max&&paths.every(p=>p.c.token_ids[fork]===paths[0].c.token_ids[fork]))fork++;}
 else fork=main.token_ids.length;
 $('distFocusLabel').textContent=`${ds.step+1}번째 토큰 선택 · ${interval(ds.focus)}`;
 let html=fork?`<div class="trace-common"><div class="trace-section-heading"><b>${paths.length>1?'공통 prefix':'응답 경로'}</b><span>${paths.length>1?'분기 전까지 동일한 토큰':'생성한 전체 토큰'}</span></div>${traceMatrix(main,0,fork,true)}</div>`:'';
 if(paths.length>1){html+=`<div class="trace-fork-label">↙ ${fork+1}번째 토큰에서 분기 ↘</div><div class="trace-paths">`;
 for(const path of paths){const v=stepValues(path.c,path.c.token_ids.length-1);html+=`<div class="trace-path" style="--trace-color:${path.color}" data-probability-path="${path.c.index}" data-full-log-probability="${v.log??''}"><div class="trace-path-heading"><b>${path.title} · ${esc(interval(path.c))}</b><span title="EOS까지 확률의 곱">P = ${shortProduct(v.log)}</span></div>${traceMatrix(path.c,fork,path.c.token_ids.length)}</div>`;}
 html+='</div>';}
 $('distTokenStrip').innerHTML=html;
 $('distTokenStrip').querySelectorAll('.trace-scroll').forEach((el,i)=>{el.scrollLeft=scrolls[i]||0;});
 $('distTokenStrip').querySelectorAll('[data-dist-step]').forEach(b=>b.onclick=()=>{setFocus(ds.data.candidates[Number(b.dataset.traceCandidate)-1],Number(b.dataset.distStep));if(ds.trace)draw(ds.serial);});
}
function renderNavigator(){
 const c=ds.focus;if(!c)return;const j=ds.step,branch=ds.candidate.branch,offset=branch?j-branch.position:null;
 const relative=offset===null?'greedy 경로':offset===0?'분기 위치':offset>0?`분기 위치에서 +${offset} token`:`분기 위치보다 ${-offset} token 앞`;
 $('distributionNavigator').innerHTML=`<div class="position-heading"><b>후보 #${c.index} · ${j+1} / ${c.tokens.length}번째 토큰 분석</b><span>${relative} · prefix ${j}개 토큰 확인</span></div><div class="position-actions"><button data-position="previous" ${j===0?'disabled':''}>← 이전 위치</button><button data-position="branch" ${branch?'':'disabled'}>분기 위치로 돌아가기</button></div><div class="position-trail"><span>시작 → </span>${c.tokens.slice(0,j+1).map((t,i)=>`<button data-position-step="${i}" aria-current="${i===j?'step':'false'}" title="${i+1}번째 토큰 분석"><small>${i+1}</small><code>${esc(t==='<|im_end|>'?'EOS':visible(t))}</code></button>`).join('<span class="trail-arrow">→</span>')}<span class="position-now">← 현재 분석</span></div>`;
 $('distributionNavigator').querySelector('[data-position="previous"]').onclick=()=>chooseStep(j-1);
 $('distributionNavigator').querySelector('[data-position="branch"]').onclick=()=>{setFocus(ds.candidate,branch.position);if(ds.trace)draw(ds.serial);};
 $('distributionNavigator').querySelectorAll('[data-position-step]').forEach(button=>button.onclick=()=>chooseStep(Number(button.dataset.positionStep)));
 const trail=$('distributionNavigator').querySelector('.position-trail');trail.scrollLeft=trail.scrollWidth;
}
function chooseStep(j){if(!ds.focus)return;setFocus(ds.focus,Math.max(0,Math.min(ds.focus.tokens.length-1,j)));if(ds.trace)draw(ds.serial);}
async function raw(node){
 const key=ds.data.sample.key+'/'+node.file;if(ds.arrays.has(key))return ds.arrays.get(key);
 const r=await fetch(`${BASE}/${ds.data.sample.key}/${node.file}`);if(!r.ok)throw Error('전체 어휘 확률 로드 실패');
 const values=new Float32Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
 if(values.length!==node.vocabulary_size)throw Error('어휘 크기가 저장된 분포와 다릅니다.');ds.arrays.set(key,values);if(ds.arrays.size>20)ds.arrays.delete(ds.arrays.keys().next().value);return values;
}
async function ranked(node){
 const key=ds.data.sample.key+'/'+node.file;
 if(ds.ranked.has(key))return ds.ranked.get(key);
 const values=await raw(node),ids=Array.from({length:values.length},(_,i)=>i);ids.sort((a,b)=>values[b]-values[a]||a-b);
 const result={ids,values};ds.ranked.set(key,result);if(ds.ranked.size>10)ds.ranked.delete(ds.ranked.keys().next().value);return result;
}
async function vocabulary(){if(!ds.vocab){const r=await fetch(BASE+'/vocab.json');if(!r.ok)throw Error('어휘 목록을 불러오지 못했습니다.');ds.vocab=await r.json();}return ds.vocab;}
async function rowsFor(node){
 const query=$('distSearch').value,mode=$('distMode').value,k=Math.max(1,Math.min(node.vocabulary_size,Math.floor(Number($('distK').value)||5))),p=Math.max(.000001,Math.min(1,Number($('distP').value)||.95));
 const limit=ds.limits.get(node.file)||80;let selected=[],mass=0,source=null;
 if(query){source=await ranked(node);const vocab=await vocabulary();selected=source.ids.filter(id=>(vocab[id]??'').includes(query)||String(id)===query);for(const id of selected)mass+=source.values[id];}
 else if(mode==='top_k'&&k<=node.top.length){selected=node.top.slice(0,k).map(t=>t.id);mass=node.top.slice(0,k).reduce((a,t)=>a+t.probability,0);}
 else if(mode==='top_p'&&node.top.reduce((a,t)=>a+t.probability,0)>=p){for(const t of node.top){selected.push(t.id);mass+=t.probability;if(mass>=p)break;}}
 else {source=await ranked(node);for(const id of source.ids){selected.push(id);mass+=source.values[id];if(mode==='top_k'?selected.length>=k:mass>=p)break;}}
 const selectedSet=new Set(selected),known=new Map([...node.top,...node.observed].map(t=>[t.id,t]));
 const shown=selected.slice(0,limit);for(const t of node.observed)if(!shown.includes(t.id))shown.push(t.id);
 if(shown.some(id=>!known.has(id)))await vocabulary();
 const rows=shown.map(id=>({id,token:known.get(id)?.token??ds.vocab?.[id]??`<${id}>`,probability:known.get(id)?.probability??source.values[id],contextOnly:!selectedSet.has(id)}));
 rows.sort((a,b)=>b.probability-a.probability||a.id-b.id);
 Object.assign(rows,{selectionCount:selected.length,selectionMass:mass,extraCount:rows.filter(r=>r.contextOnly).length,hasMore:selected.length>limit,selectionText:query?`검색 일치 ${selected.length}개`:mode==='top_k'?`top-k ${k} → ${selected.length}개`:`top-p ${p} → ${selected.length}개`});return rows;
}
function distributionRows(rows,node,colors,clickable=false){
 return '<div class="probability-rows">'+rows.map(row=>{
  const color=colors.get(row.id),style=color?`style="border-left:3px solid ${color};background:${color}12"`:'';
  return `<div class="prob-row ${color?'measured-choice':''}" ${style} ${clickable?`data-root-token="${row.id}"`:''} data-context-only="${row.contextOnly}"><code title="Token ID ${row.id}${row.contextOnly?' · 표시 기준 밖의 경로 토큰':''}">${esc(visible(row.token))}${row.contextOnly?'<sup>경로</sup>':''}</code><div class="prob-bar"><i style="width:${100*row.probability}%;background:${color||'#a4b3ac'}"></i></div><span>${probability(row.probability)}</span></div>`;
 }).join('')+'</div>'+(rows.hasMore?`<button class="show-more-prob" data-more-prob="${node.file}">다음 80개 표시 (${rows.selectionCount}개 중)</button>`:'');
}
function massNote(rows){return `${rows.selectionText} · 누적확률 ${(100*rows.selectionMass).toFixed(4)}%${rows.extraCount?` · 기준 밖 경로 토큰 ${rows.extraCount}개 별도 표시`:''}`;}
function tableHeader(){return '<div class="prob-row prob-heading"><span>토큰</span><span>확률 막대</span><span>조건부 확률</span></div>';}
async function draw(serial){
 try{
 const drawId=++ds.drawCounter,alive=()=>serial===ds.serial&&drawId===ds.drawCounter,rootExpanded=$('distributionTree').querySelector('.distribution-root details')?.open;
 const c=ds.focus,main=ds.candidate,j=ds.step,node=nodeFor(c,j);if(!node)return;
 const parent=main.branch?ds.data.candidates[main.branch.parent-1]:main;
 const atBranch=main.branch&&j===main.branch.position&&same(parent.token_ids.slice(0,j),c.token_ids.slice(0,j));
 const mainTokens=atBranch?[parent.token_ids[j],main.token_ids[j]]:[c.token_ids[j]];
 const tokens=[...new Set([...mainTokens,...node.observed.map(t=>t.id)])].filter(id=>node.observed.some(t=>t.id===id));
 if(!ds.visibilityInitialized){for(const id of tokens)if(!mainTokens.includes(id))ds.disabled.add(id);ds.visibilityInitialized=true;}
 const primaryColor=ds.focus.index===main.index&&parent.index!==main.index?COLORS[1]:ds.focus.index===parent.index?COLORS[0]:COLORS[2];
 const colors=new Map(tokens.map((id,i)=>[id,atBranch?COLORS[i%COLORS.length]:i===0?primaryColor:COLORS[(i+2)%COLORS.length]]));
 if(ds.autoK)$('distK').value='5';
 $('distK').max=String(node.vocabulary_size);
 $('distKDefault').textContent=`기본 k = 5${ds.autoK?'':' · 직접 입력한 k 적용 중 (위치를 바꿔도 유지)'}`;
 const rootRows=await rowsFor(node);if(!alive())return;
 $('distributionContext').innerHTML=atBranch?'공통으로 생성한 부분 다음에서 파란색과 주황색 토큰으로 갈렸습니다. 각 토큰을 고르면 다음 토큰의 확률도 다르게 조건화됩니다.':'선택한 자리까지 생성한 응답을 조건으로 token distribution와 후속 경로를 보여줍니다. 위의 정답·분기 전후 예측은 유지됩니다. ‘분기’ 표시가 있는 토큰을 누르면 기본 비교로 돌아갑니다.';
 let html=`<div class="distribution-root"><h4>${atBranch?'분기 위치의 분포':'선택한 위치의 분포'}</h4><p class="prefix-caption">${atBranch?'분기 직전 prefix':`${j}개 토큰을 생성한 prefix`}</p><pre class="dist-prefix">${esc(node.prefix_text||'(응답을 시작하기 전)')}<b>▏</b></pre>${tableHeader()}${distributionRows(rootRows,node,colors,true)}<p class="small">${massNote(rootRows)}</p><div class="root-legend">${atBranch?`<span style="color:${COLORS[0]}">● 파랑: 분기 전 선택 <code>${esc(parent.tokens[j])}</code></span><span style="color:${COLORS[1]}">● 주황: 바꿔 고른 토큰 <code>${esc(main.tokens[j])}</code></span>`:'색이 있는 토큰은 저장된 응답에서 선택한 적 있는 토큰입니다.'}</div><details ${rootExpanded?'open':''}><summary>비교할 경로 추가·숨기기</summary><p class="small">저장된 다른 응답에서 선택한 토큰도 펼칠 수 있습니다. 분포만 있고 후속 응답이 없는 토큰은 펼치지 않습니다.</p><div class="distribution-chips">`;
 for(const id of tokens){const t=node.observed.find(x=>x.id===id),color=colors.get(id);html+=`<label style="border-color:${color};color:${color}"><input type="checkbox" data-child-token="${id}" ${ds.disabled.has(id)?'':'checked'}> ${esc(visible(t.token))}</label>`;}
 html+='</div></details></div><div class="distribution-children">';
 for(const id of tokens){
  if(ds.disabled.has(id))continue;
  const token=node.observed.find(x=>x.id===id),color=colors.get(id);
  const matches=x=>x.token_ids.length>j&&same(x.token_ids.slice(0,j),c.token_ids.slice(0,j))&&x.token_ids[j]===id;
  const representative=atBranch&&matches(parent)?parent:atBranch&&matches(main)?main:matches(c)?c:ds.data.candidates.find(matches);
  const next=nodeFor(representative,j+1);
  const before=atBranch&&id===parent.token_ids[j],after=atBranch&&id===main.token_ids[j];
  html+=`<article class="distribution-child" style="--child-color:${color}" data-path-candidate="${representative.index}"><h4><span class="token-color" style="background:${color}"></span>${before?'분기 전':after?'분기 후':'현재 prefix의 경로'}: <code>${esc(visible(token.token))}</code> 선택 · #${representative.index}</h4><div class="path-legend"><span class="path-prefix">확인한 prefix</span><span class="path-picked">고른 토큰</span><span class="path-suffix">이어서 생성</span></div><p class="path-sentence">응답 <code class="path-answer"><span class="path-prefix">${esc(representative.tokens.slice(0,j).join(''))}</span><mark class="path-picked">${esc(token.token)}</mark><span class="path-suffix">${esc(representative.tokens.slice(j+1).filter(t=>!t.startsWith('<|')).join(''))}</span></code> → 최종 구간 <strong>${esc(interval(representative))}</strong>.</p>`;
  if(!next){html+='<p class="small">응답 종료 토큰(EOS)을 골라 응답이 끝났습니다.</p></article>';continue;}
  const nextRows=await rowsFor(next);if(!alive())return;
  const nextId=representative.token_ids[j+1];
  html+=`<button class="follow-path" data-next-node="${representative.index}">다음 위치: ${j+2}번째 토큰 <code>${esc(visible(representative.tokens[j+1]))}</code> 분석 →</button><p class="prefix-caption">고른 토큰까지의 prefix 다음 token distribution</p>${tableHeader()}${distributionRows(nextRows,next,new Map([[nextId,color]]))}<p class="small">${massNote(nextRows)}</p><a class="raw-prob" href="${BASE}/${ds.data.sample.key}/${next.file}" download>전체 어휘 FP32 확률 다운로드</a></article>`;
 }
 html+='</div>';if(!alive())return;$('distributionTree').innerHTML=html;
 $('distributionTree').querySelectorAll('[data-child-token]').forEach(input=>input.onchange=()=>{const id=Number(input.dataset.childToken);if(input.checked)ds.disabled.delete(id);else ds.disabled.add(id);draw(ds.serial);});
 $('distributionTree').querySelectorAll('[data-root-token]').forEach(row=>row.onclick=()=>{const id=Number(row.dataset.rootToken);if(!tokens.includes(id))return;if(ds.disabled.has(id))ds.disabled.delete(id);else ds.disabled.add(id);draw(ds.serial);});
 $('distributionTree').querySelectorAll('[data-next-node]').forEach(button=>button.onclick=async()=>{const y=$('distributionNavigator').getBoundingClientRect().top;setFocus(ds.data.candidates[Number(button.dataset.nextNode)-1],j+1);await draw(ds.serial);window.scrollBy(0,$('distributionNavigator').getBoundingClientRect().top-y);});
 $('distributionTree').querySelectorAll('[data-more-prob]').forEach(button=>button.onclick=()=>{const file=button.dataset.moreProb;ds.limits.set(file,(ds.limits.get(file)||80)+80);draw(ds.serial);});
 if(!ds.autoJump&&new URLSearchParams(location.hash.slice(1)).get('panel')==='distributions'){ds.autoJump=true;$('distributionPanel').scrollIntoView({block:'start'});}
 }catch(e){if(serial===ds.serial)$('distributionStatus').textContent=e.message;}
}
function bind(){
 const signature=()=>JSON.stringify([$('distMode').value,$('distK').value,$('distP').value,$('distSearch').value]);
 let applied=signature(),timer;
 function selectionChanged(){const current=signature();if(current===applied)return;applied=current;ds.limits.clear();if(ds.trace)draw(ds.serial);}
 function delayed(){clearTimeout(timer);timer=setTimeout(selectionChanged,200);}
 $('distMode').onchange=()=>{$('distKLabel').hidden=$('distMode').value!=='top_k';$('distPLabel').hidden=$('distMode').value!=='top_p';selectionChanged();};
 $('distK').onchange=()=>{ds.autoK=false;selectionChanged();};$('distP').onchange=selectionChanged;
 $('distK').oninput=()=>{ds.autoK=false;delayed();};$('distP').oninput=delayed;$('distSearch').oninput=delayed;
}
window.ROLLOUT_DISTRIBUTIONS={render,state:ds,chooseStep,pathProbability,rowsFor};bind();
})();
