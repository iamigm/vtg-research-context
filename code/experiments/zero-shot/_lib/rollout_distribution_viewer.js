'use strict';
(()=>{
const $=id=>document.getElementById(id),BASE=window.ROLLOUT_ASSETS+'/../token_traces';
const COLORS=['#2678b5','#d16b32','#218b69','#9264b2','#a98a22','#ba4e85','#597d49','#597b96','#7d6651','#78486a'];
const ds={data:null,candidate:null,trace:null,step:0,serial:0,drawCounter:0,cache:new Map(),arrays:new Map(),vocab:null,pending:null,timer:null,disabled:new Set(),visibilityInitialized:false,autoJump:false};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const visible=t=>t===' '?'공백':String(t).replace(/\n/g,'↵').replace(/ /g,'·');
const probability=p=>p===0?'0%':p>=.000001?`${(100*p).toFixed(5)}%`:`${(100*p).toExponential(3)}%`;
const pathProbability=log=>{
 if(log===0)return '1';if(!Number.isFinite(log))return '0';
 const x=log/Math.LN10,exponent=Math.floor(x);return `${Math.pow(10,x-exponent).toFixed(4)} × 10^${exponent}`;
};
const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
const tokenProbability=(node,id)=>node.observed.find(t=>t.id===id)?.probability??node.top.find(t=>t.id===id)?.probability??0;
const nodeFor=(candidate,step)=>ds.trace?.nodes[ds.trace.candidates[String(candidate.index)]?.nodes[step]];

async function render(data,candidate){
 const serial=++ds.serial;clearTimeout(ds.timer);ds.data=data;ds.candidate=candidate;ds.trace=null;ds.disabled.clear();ds.visibilityInitialized=false;
 ds.step=ds.pending?.key===data.sample.key&&ds.pending.candidate===candidate.index?ds.pending.step:(candidate.branch?.position??0);ds.pending=null;
 $('distStep').innerHTML=candidate.tokens.map((token,j)=>`<option value="${j}">Step ${j+1}: ${esc(visible(token))}</option>`).join('');
 $('distStep').value=String(ds.step);$('distributionTree').innerHTML='';$('distributionProduct').innerHTML='';
 $('distributionStatus').textContent='저장된 토큰 분포를 확인하는 중…';
 renderStrip();
 try{
  let trace=ds.cache.get(data.sample.key);
  if(!trace){const response=await fetch(`${BASE}/${data.sample.key}/index.json`,{cache:'no-store'});if(response.status===404){await pendingStatus(serial);return;}if(!response.ok)throw Error(`분포 자료 로드 실패 (${response.status})`);trace=await response.json();ds.cache.set(data.sample.key,trace);if(ds.cache.size>5)ds.cache.delete(ds.cache.keys().next().value);}
  if(serial!==ds.serial)return;ds.trace=trace;
  $('distributionStatus').textContent='같은 입력·정확한 prefix로 재측정한 전체 어휘 분포입니다. Generation step은 영상 시각이 아니라 응답 토큰 순번입니다. 토큰 표시의 ·는 공백, ↵는 줄바꿈입니다.';
  await draw(serial);
 }catch(e){if(serial===ds.serial)$('distributionStatus').textContent=e.message;}
}
async function pendingStatus(serial){
 let note='이 샘플의 분포는 아직 측정되지 않았습니다.';
 try{const response=await fetch(window.ROLLOUT_ASSETS+'/../metadata/distribution_status.json',{cache:'no-store'});if(response.ok){const s=await response.json();note+=` 전체 ${s.completed}/${s.total}개 완료 · ${s.state==='failed'?'측정 오류 발생: 로그 확인 필요':s.state==='complete'?'측정 종료':'백그라운드 측정 중'}.`;}}catch(e){}
 if(serial!==ds.serial)return;$('distributionStatus').innerHTML=esc(note)+' <button id="distReadyExample">측정 완료된 1.3 → 107.3 사례 보기</button>';
 $('distReadyExample').onclick=()=>{window.ROLLOUT_VIEWER.applyPreset('greedy_short');window.ROLLOUT_VIEWER.selectCase('activitynet-timelens_01723',3,true);};
 ds.timer=setTimeout(()=>{if(serial===ds.serial)render(ds.data,ds.candidate);},30000);
}
function renderStrip(){
 const c=ds.candidate;$('distTokenStrip').innerHTML=c.tokens.map((t,j)=>`<button data-dist-step="${j}" class="${j===ds.step?'active':''}" title="Generation step ${j+1}">${esc(visible(t))}</button>`).join('');
 $('distTokenStrip').querySelectorAll('[data-dist-step]').forEach(b=>b.onclick=()=>chooseStep(Number(b.dataset.distStep)));
 $('distPrevious').disabled=ds.step===0;$('distNext').disabled=ds.step>=c.tokens.length-1;
}
function chooseStep(j){
 if(!ds.candidate)return;ds.step=Math.max(0,Math.min(ds.candidate.tokens.length-1,j));ds.disabled.clear();ds.visibilityInitialized=false;
 $('distStep').value=String(ds.step);renderStrip();if(ds.trace)draw(ds.serial);
}
async function raw(node){
 const cacheKey=ds.data.sample.key+'/'+node.file;
 if(ds.arrays.has(cacheKey))return ds.arrays.get(cacheKey);
 const response=await fetch(`${BASE}/${ds.data.sample.key}/${node.file}`);if(!response.ok)throw Error('전체 어휘 확률 로드 실패');
 const stream=response.body.pipeThrough(new DecompressionStream('gzip'));
 const values=new Float32Array(await new Response(stream).arrayBuffer());
 if(values.length!==node.vocabulary_size)throw Error('어휘 크기가 저장된 분포와 다릅니다.');
 ds.arrays.set(cacheKey,values);if(ds.arrays.size>20)ds.arrays.delete(ds.arrays.keys().next().value);return values;
}
async function rowsFor(node){
 const query=$('distSearch').value,limit=Number($('distTop').value);
 let rows=node.top.slice(0,limit);
 if(query){
  if(!ds.vocab){const response=await fetch(BASE+'/vocab.json');ds.vocab=await response.json();}
  const values=await raw(node);rows=[];
  for(let id=0;id<values.length;id++)if((ds.vocab[id]??'').includes(query)||String(id)===query)rows.push({id,token:ds.vocab[id]??`<${id}>`,probability:values[id]});
  rows.sort((a,b)=>b.probability-a.probability||a.id-b.id);rows=rows.slice(0,limit);
 }
 for(const row of node.observed)if(!rows.some(r=>r.id===row.id))rows.push(row);
 rows.sort((a,b)=>b.probability-a.probability||a.id-b.id);return rows;
}
function distributionRows(rows,node,colors,clickable=false){
 return rows.map(row=>{
  const color=colors.get(row.id),style=color?`style="border-left:4px solid ${color};background:${color}12"`:'';
  const log=row.probability>0?node.prefix_log_probability+Math.log(row.probability):-Infinity;
  return `<div class="prob-row" ${style} ${clickable?`data-root-token="${row.id}"`:''}><code title="Token ID ${row.id}">${esc(visible(row.token))}</code><div class="prob-bar"><i style="width:${100*row.probability}%;background:${color||'#9baaaa'}"></i></div><span>${probability(row.probability)}</span><span class="path-prob">${pathProbability(log)}</span></div>`;
 }).join('');
}
function massNote(rows){const mass=rows.reduce((a,b)=>a+b.probability,0);return `표시 토큰 확률 합 ${(100*mass).toFixed(4)}% · 나머지 전체 어휘 ${(100*Math.max(0,1-mass)).toFixed(4)}%`;}
function tableHeader(){return '<div class="prob-row prob-heading"><span>토큰</span><span>확률 막대</span><span>조건부 확률</span><span>누적 경로 확률</span></div>';}

async function draw(serial){
 try{
 const drawId=++ds.drawCounter,alive=()=>serial===ds.serial&&drawId===ds.drawCounter;
 const c=ds.candidate,j=ds.step,node=nodeFor(c,j);if(!node)return;
 const parent=c.branch?ds.data.candidates[c.branch.parent-1]:c;
 const preferred=[];
 if(same(parent.token_ids.slice(0,j),c.token_ids.slice(0,j)))preferred.push(parent.token_ids[j]);
 preferred.push(c.token_ids[j]);preferred.push(...node.observed.map(t=>t.id));
 const tokens=[...new Set(preferred)].filter(id=>node.observed.some(t=>t.id===id));
 if(!ds.visibilityInitialized){
  const mainTokens=new Set([c.token_ids[j]]);
  if(same(parent.token_ids.slice(0,j),c.token_ids.slice(0,j)))mainTokens.add(parent.token_ids[j]);
  for(const id of tokens)if(!mainTokens.has(id))ds.disabled.add(id);
  ds.visibilityInitialized=true;
 }
 const colors=new Map(tokens.map((id,i)=>[id,COLORS[i%COLORS.length]]));
 const rootRows=await rowsFor(node);if(!alive())return;
 let html=`<div class="distribution-root"><h4>부모 단계 · Step ${j+1}의 토큰을 선택하기 전</h4><pre class="dist-prefix">${esc(node.prefix_text||'(아직 응답 토큰 없음)')}</pre><p class="formula">P(prefix) = 앞선 ${j}개 step 확률의 곱 = <b>${pathProbability(node.prefix_log_probability)}</b></p>${tableHeader()}${distributionRows(rootRows,node,colors,true)}<p class="small">${massNote(rootRows)} · 색이 있는 행은 저장된 응답에서 실제 선택한 토큰입니다.</p><div class="distribution-chips">`;
 for(const id of tokens){const t=node.observed.find(x=>x.id===id),color=colors.get(id);html+=`<label style="border-color:${color};color:${color}"><input type="checkbox" data-child-token="${id}" ${ds.disabled.has(id)?'':'checked'}> ${esc(visible(t.token))} 선택 후</label>`;}
 html+='</div></div><div class="distribution-children">';
 for(const id of tokens){
  if(ds.disabled.has(id))continue;
  const token=node.observed.find(x=>x.id===id),color=colors.get(id);
  const representative=ds.data.candidates.find(x=>x.token_ids.length>j&&same(x.token_ids.slice(0,j),c.token_ids.slice(0,j))&&x.token_ids[j]===id);
  const next=nodeFor(representative,j+1),cumulative=node.prefix_log_probability+Math.log(token.probability);
  html+=`<article class="distribution-child" style="--child-color:${color}"><h4><span class="token-color" style="background:${color}"></span> ${esc(visible(token.token))} 선택 후 · Step ${j+2}</h4><p class="formula">P(${esc(visible(token.token))} | prefix) = <b>${probability(token.probability)}</b><br>P(prefix + ${esc(visible(token.token))})<br>= ${pathProbability(node.prefix_log_probability)} × ${token.probability.toPrecision(5)}<br>= <b>${pathProbability(cumulative)}</b></p>`;
  if(!next){html+='<p>EOS로 응답이 끝났으므로 다음 generation step은 없습니다.</p></article>';continue;}
  const nextRows=await rowsFor(next);if(!alive())return;
  const childColors=new Map(next.observed.map(t=>[t.id,color]));
  html+=`<pre class="dist-prefix">${esc(next.prefix_text)}</pre>${tableHeader()}${distributionRows(nextRows,next,childColors)}<p class="small">${massNote(nextRows)}</p><p class="formula">각 행의 누적값 = P(prefix) × P(${esc(visible(token.token))} | prefix) × 그 행의 다음 토큰 조건부 확률</p><button data-next-node="${representative.index}">이 경로의 다음 시점을 부모로 펼치기</button><a class="raw-prob" href="${BASE}/${ds.data.sample.key}/${next.file}" download>전체 어휘 FP32 확률</a></article>`;
 }
 html+='</div>';if(!alive())return;$('distributionTree').innerHTML=html;
 $('distributionTree').querySelectorAll('[data-child-token]').forEach(input=>input.onchange=()=>{const id=Number(input.dataset.childToken);if(input.checked)ds.disabled.delete(id);else ds.disabled.add(id);draw(ds.serial);});
 $('distributionTree').querySelectorAll('[data-root-token]').forEach(row=>row.onclick=()=>{
  const id=Number(row.dataset.rootToken);if(!tokens.includes(id))return;
  if(ds.disabled.has(id))ds.disabled.delete(id);else ds.disabled.add(id);draw(ds.serial);
 });
 $('distributionTree').querySelectorAll('[data-next-node]').forEach(button=>button.onclick=()=>{
  const id=Number(button.dataset.nextNode);ds.pending={key:ds.data.sample.key,candidate:id,step:j+1};window.ROLLOUT_VIEWER.showCandidate(id);
 });
 renderProduct(c);
 if(!ds.autoJump&&new URLSearchParams(location.hash.slice(1)).get('panel')==='distributions'){
  ds.autoJump=true;$('distributionPanel').scrollIntoView({block:'start'});
 }
 }catch(e){if(serial===ds.serial)$('distributionStatus').textContent=e.message;}
}
function renderProduct(c){
 let sum=0;const factors=[],rows=[];
 for(let j=0;j<c.token_ids.length;j++){
  const node=nodeFor(c,j),p=tokenProbability(node,c.token_ids[j]);sum+=p>0?Math.log(p):-Infinity;
  factors.push(p.toPrecision(4));rows.push(`<tr><td>${j+1}</td><td><code>${esc(visible(c.tokens[j]))}</code></td><td>${probability(p)}</td><td>${pathProbability(sum)}</td></tr>`);
 }
 const saved=ds.trace.candidates[String(c.index)];
 $('distributionProduct').innerHTML=`<div class="probability-product"><h4>확인 중인 후보 #${c.index}: generation step 확률의 곱</h4><p class="formula">P(전체 응답 | 영상·질문) = p₁ × p₂ × … × p${c.token_ids.length} = <b>${pathProbability(sum)}</b></p><p class="small">EOS까지 포함한 곱입니다. 원래 저장한 전체 응답 log P = ${saved.saved_log_probability.toFixed(6)}, 재측정 log P = ${saved.log_probability.toFixed(6)}.</p><details><summary>각 step의 확률과 누적곱 펼치기</summary><p class="formula factors">${factors.join(' × ')}</p><div class="table-scroll product-table"><table><thead><tr><th>Generation step</th><th>선택 토큰</th><th>해당 step 조건부 확률</th><th>이 step까지 누적곱</th></tr></thead><tbody>${rows.join('')}</tbody></table></div></details></div>`;
}
function bind(){
 $('distPrevious').onclick=()=>chooseStep(ds.step-1);$('distNext').onclick=()=>chooseStep(ds.step+1);
 $('distStep').onchange=()=>chooseStep(Number($('distStep').value));
 $('distBranchPoint').onclick=()=>chooseStep(ds.candidate?.branch?.position??0);
 $('distTop').onchange=()=>{if(ds.trace)draw(ds.serial);};
 let timer;$('distSearch').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{if(ds.trace)draw(ds.serial);},250);};
}
window.ROLLOUT_DISTRIBUTIONS={render,state:ds,chooseStep,pathProbability};bind();
})();
