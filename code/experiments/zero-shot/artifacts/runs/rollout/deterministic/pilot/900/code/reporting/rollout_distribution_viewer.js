'use strict';
(()=>{
const $=id=>document.getElementById(id),BASE=window.ROLLOUT_DISTRIBUTIONS_BASE||window.ROLLOUT_ASSETS+'/../token_traces';
const COLORS=['#2678b5','#c4672f','#218b69','#9264b2','#a98a22','#ba4e85','#597d49','#597b96'];
const ds={data:null,candidate:null,focus:null,trace:null,step:0,serial:0,drawCounter:0,cache:new Map(),arrays:new Map(),ranked:new Map(),limits:new Map(),vocab:null,timer:null,disabled:new Set(),visibilityInitialized:false,autoJump:false,autoK:true,showProb:false};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const visible=t=>t===' '?'공백':String(t).replace(/\n/g,'↵').replace(/ /g,'·');
const interval=c=>c.pair?c.pair.map(x=>Number(x.toFixed(2))).join(' – ')+'초':'유효한 구간 없음';
const probability=p=>p===0?'0%':p>=.000001?`${(100*p).toFixed(5)}%`:`${(100*p).toExponential(3)}%`;
const pathProbability=log=>{if(log===0)return '1';if(!Number.isFinite(log))return '0';const x=log/Math.LN10,e=Math.floor(x);return `${Math.pow(10,x-e).toFixed(4)} × 10^${e}`;};
const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
const tokenProbability=(node,id)=>node?.observed.find(t=>t.id===id)?.probability??node?.top.find(t=>t.id===id)?.probability??0;
const nodeFor=(c,step)=>c&&ds.trace?.nodes[ds.trace.candidates[String(c.index)]?.nodes[step]];
function setFocus(c,j){ds.focus=c;ds.step=j;ds.disabled.clear();ds.visibilityInitialized=false;renderNavigator();}
async function render(data,candidate){
 const serial=++ds.serial;clearTimeout(ds.timer);ds.data=data;ds.candidate=candidate;ds.trace=null;ds.limits.clear();
 setFocus(candidate,candidate.branch?.position??0);$('distributionTree').innerHTML='';$('distributionStatus').textContent='저장된 분포를 불러오는 중…';
 try{
  let trace=ds.cache.get(data.sample.key);
  if(!trace){const r=await fetch(`${BASE}/${data.sample.key}/index.json`,{cache:'no-store'});if(r.status===404){await pendingStatus(serial);return;}if(!r.ok)throw Error(`분포 자료 로드 실패 (${r.status})`);trace=await r.json();ds.cache.set(data.sample.key,trace);if(ds.cache.size>5)ds.cache.delete(ds.cache.keys().next().value);}
  if(serial!==ds.serial)return;ds.trace=trace;
  $('distributionStatus').textContent='저장된 응답의 정확한 prefix에서 측정한 확률입니다. ␣는 공백, ↵는 줄바꿈, EOS는 응답 종료입니다.';
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
function routeSentence(c,color){
 const main=ds.candidate,parent=main.branch?ds.data.candidates[main.branch.parent-1]:main;
 const name=c.index===parent.index?(parent.index===1?'greedy 응답':'부모 응답'):c.index===main.index?'분기 응답':'추가 경로';
 const total=stepValues(c,c.token_ids.length-1),focused=c.index===ds.focus.index;
 const buttons=c.tokens.map((t,j)=>{
  const v=stepValues(c,j),active=focused&&j===ds.step,point=main.branch&&j===main.branch.position;
  const token=t==='<|im_end|>'?'EOS':t===' '?'␣':t.replace(/\n/g,'↵');
  const title=`${j+1} / ${c.tokens.length} · ${t} · p = ${v.p===null?'측정 중':v.p.toPrecision(9)} · 누적곱 = ${v.log===null?'측정 중':pathProbability(v.log)}`;
  return `<button data-dist-step="${j}" data-trace-candidate="${c.index}" data-probability="${v.p??''}" data-log-probability="${v.log??''}" class="route-token ${active?'active ':''}${point?'branch-point ':''}${t.startsWith('<|')?'eos-token':''}" aria-pressed="${active}" title="${esc(title)}"><code class="route-token-text">${esc(token)}</code><span class="route-token-stats"><span>p ${shortProbability(v.p)}</span><span>∏ ${shortProduct(v.log)}</span></span></button>`;
 }).join('');
 const next=Math.min(ds.step+1,c.tokens.length-1),ended=ds.step>=c.tokens.length-1;
 return `<article class="distribution-child route-card ${focused?'focused':''}" style="--child-color:${color}" data-path-candidate="${c.index}" data-full-log-probability="${total.log??''}"><div class="route-heading"><h4>${name} #${c.index}</h4><span>${esc(interval(c))}</span></div><div class="route-sentence ${ds.showProb?'show-token-prob':''}" aria-label="${name}의 문장에서 분석할 토큰 선택">${buttons}</div><div class="route-footer"><span title="EOS까지의 조건부 확률 곱">path probability <b>${shortProduct(total.log)}</b></span>${ended?'<span class="route-ended">응답이 끝났습니다 · EOS</span>':`<button class="follow-path" data-next-node="${c.index}" data-next-step="${next}">다음 토큰 <code>${esc(visible(c.tokens[next]==='<|im_end|>'?'EOS':c.tokens[next]))}</code> →</button>`}</div></article>`;
}
function renderNavigator(){
 const c=ds.focus;if(!c)return;const j=ds.step,branch=ds.candidate.branch,offset=branch?j-branch.position:null;
 const relative=offset===null?'greedy 경로':offset===0?'분기 위치':offset>0?`분기 위치에서 +${offset} token`:`분기 위치보다 ${-offset} token 앞`;
 $('distributionNavigator').innerHTML=`<div class="position-heading"><b>후보 #${c.index} · <code>${esc(c.tokens[j]==='<|im_end|>'?'EOS':visible(c.tokens[j]))}</code> 분석</b><span>${relative}</span></div><div class="position-actions"><button data-position="previous" ${j===0?'disabled':''}>← 이전 토큰</button><button data-position="branch" ${branch?'':'disabled'}>분기 위치로 돌아가기</button></div>`;
 $('distributionNavigator').querySelector('[data-position="previous"]').onclick=()=>chooseStep(j-1);
 $('distributionNavigator').querySelector('[data-position="branch"]').onclick=()=>{setFocus(ds.candidate,branch.position);if(ds.trace)draw(ds.serial);};
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
 const matchesPrefix=x=>x.token_ids.length>j&&same(x.token_ids.slice(0,j),c.token_ids.slice(0,j));
 const primary=[parent,main].filter(matchesPrefix),mainTokens=[...new Set([...primary.map(x=>x.token_ids[j]),c.token_ids[j]])];
 const tokens=[...new Set([...mainTokens,...node.observed.map(t=>t.id)])].filter(id=>node.observed.some(t=>t.id===id));
 if(!ds.visibilityInitialized){for(const id of tokens)if(!mainTokens.includes(id))ds.disabled.add(id);ds.visibilityInitialized=true;}
 const paths=[parent];if(main.index!==parent.index)paths.push(main);if(!paths.some(x=>x.index===c.index))paths.push(c);
 for(const id of tokens){if(ds.disabled.has(id)||paths.some(x=>matchesPrefix(x)&&x.token_ids[j]===id))continue;const row=ds.data.candidates.find(x=>matchesPrefix(x)&&x.token_ids[j]===id);if(row&&!paths.some(x=>x.index===row.index))paths.push(row);}
 const colorFor=row=>row.index===parent.index?COLORS[0]:row.index===main.index?COLORS[1]:COLORS[(paths.findIndex(x=>x.index===row.index)+2)%COLORS.length];
 const colors=new Map(tokens.map((id,i)=>{const row=paths.find(x=>matchesPrefix(x)&&x.token_ids[j]===id);return [id,row?colorFor(row):COLORS[(i+2)%COLORS.length]];}));
 if(ds.autoK)$('distK').value='5';
 $('distK').max=String(node.vocabulary_size);
 $('distKDefault').textContent=`기본 k = 5${ds.autoK?'':' · 직접 입력한 k 적용 중 (위치를 바꿔도 유지)'}`;
 const rootRows=await rowsFor(node);if(!alive())return;
 let html=`<div class="distribution-children" aria-label="문장에서 토큰 선택">${paths.map(row=>routeSentence(row,colorFor(row))).join('')}</div><div class="distribution-root"><h4>${atBranch?'분기 위치의 분포':'선택한 위치의 분포'}</h4><p class="prefix-caption">아래 prefix 다음에 올 토큰의 확률</p><pre class="dist-prefix">${esc(node.prefix_text||'(응답을 시작하기 전)')}<b>▏</b></pre>${tableHeader()}${distributionRows(rootRows,node,colors,true)}<p class="small">${massNote(rootRows)}</p><p class="root-legend">색이 있는 토큰을 누르면 그 토큰을 선택한 저장 경로로 이동합니다.</p><details ${rootExpanded?'open':''}><summary>같은 prefix의 다른 저장 경로</summary><p class="small">부모·분기 응답을 유지하면서 비교할 경로를 추가합니다.</p><div class="distribution-chips">`;
 for(const id of tokens){const t=node.observed.find(x=>x.id===id),color=colors.get(id),fixed=mainTokens.includes(id);html+=`<label style="border-color:${color};color:${color}"><input type="checkbox" data-child-token="${id}" ${fixed||!ds.disabled.has(id)?'checked':''} ${fixed?'disabled':''}> ${esc(visible(t.token))}</label>`;}
 html+=`</div></details><a class="raw-prob" href="${BASE}/${ds.data.sample.key}/${node.file}" download>현재 분포의 전체 어휘 FP32 확률 다운로드</a></div>`;
 if(!alive())return;$('distributionTree').innerHTML=html;
 $('distributionTree').querySelectorAll('[data-dist-step]').forEach(button=>button.onclick=()=>{setFocus(ds.data.candidates[Number(button.dataset.traceCandidate)-1],Number(button.dataset.distStep));draw(ds.serial);});
 $('distributionTree').querySelectorAll('[data-child-token]').forEach(input=>input.onchange=()=>{const id=Number(input.dataset.childToken);if(input.checked)ds.disabled.delete(id);else ds.disabled.add(id);draw(ds.serial);});
 $('distributionTree').querySelectorAll('[data-root-token]').forEach(row=>{const id=Number(row.dataset.rootToken);if(!tokens.includes(id))return;row.setAttribute('role','button');row.tabIndex=0;row.onclick=()=>{const target=paths.find(x=>matchesPrefix(x)&&x.token_ids[j]===id)||ds.data.candidates.find(x=>matchesPrefix(x)&&x.token_ids[j]===id);if(target){setFocus(target,j);draw(ds.serial);}};row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();row.click();}};});
 $('distributionTree').querySelectorAll('[data-next-node]').forEach(button=>button.onclick=async()=>{const y=$('distributionNavigator').getBoundingClientRect().top;setFocus(ds.data.candidates[Number(button.dataset.nextNode)-1],Number(button.dataset.nextStep));await draw(ds.serial);window.scrollBy(0,$('distributionNavigator').getBoundingClientRect().top-y);});
 $('distributionTree').querySelectorAll('[data-more-prob]').forEach(button=>button.onclick=()=>{const file=button.dataset.moreProb;ds.limits.set(file,(ds.limits.get(file)||80)+80);draw(ds.serial);});
 if(!ds.autoJump&&new URLSearchParams(location.hash.slice(1)).get('panel')==='distributions'){ds.autoJump=true;$('distributionPanel').scrollIntoView({block:'start'});}
 }catch(e){if(serial===ds.serial)$('distributionStatus').textContent=e.message;}
}
function bind(){
 $('distShowProb').onchange=()=>{ds.showProb=$('distShowProb').checked;document.querySelectorAll('.route-sentence').forEach(el=>el.classList.toggle('show-token-prob',ds.showProb));};
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
