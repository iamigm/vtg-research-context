'use strict';
(()=>{
const $=id=>document.getElementById(id),BASE=window.ROLLOUT_DISTRIBUTIONS_BASE||window.ROLLOUT_ASSETS+'/../token_traces';
const COLORS=['#2678b5','#c4672f','#218b69','#9264b2','#a98a22','#ba4e85','#597d49','#597b96'];
const ds={data:null,candidate:null,focus:null,trace:null,step:0,serial:0,drawCounter:0,cache:new Map(),arrays:new Map(),vocab:null,timer:null,disabled:new Set(),visibilityInitialized:false,autoJump:false};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const visible=t=>t===' '?'공백':String(t).replace(/\n/g,'↵').replace(/ /g,'·');
const interval=c=>c.pair?c.pair.map(x=>Number(x.toFixed(2))).join(' – ')+'초':'유효한 구간 없음';
const probability=p=>p===0?'0%':p>=.000001?`${(100*p).toFixed(5)}%`:`${(100*p).toExponential(3)}%`;
const pathProbability=log=>{if(log===0)return '1';if(!Number.isFinite(log))return '0';const x=log/Math.LN10,e=Math.floor(x);return `${Math.pow(10,x-e).toFixed(4)} × 10^${e}`;};
const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
const tokenProbability=(node,id)=>node?.observed.find(t=>t.id===id)?.probability??node?.top.find(t=>t.id===id)?.probability??0;
const nodeFor=(c,step)=>c&&ds.trace?.nodes[ds.trace.candidates[String(c.index)]?.nodes[step]];
function setFocus(c,j){ds.focus=c;ds.step=j;ds.disabled.clear();ds.visibilityInitialized=false;renderStrip();}
async function render(data,candidate){
 const serial=++ds.serial;clearTimeout(ds.timer);ds.data=data;ds.candidate=candidate;ds.trace=null;
 setFocus(candidate,candidate.branch?.position??0);$('distributionTree').innerHTML='';$('distributionProduct').innerHTML='';$('distributionStatus').textContent='저장된 분포를 불러오는 중…';
 try{
  let trace=ds.cache.get(data.sample.key);
  if(!trace){const r=await fetch(`${BASE}/${data.sample.key}/index.json`,{cache:'no-store'});if(r.status===404){await pendingStatus(serial);return;}if(!r.ok)throw Error(`분포 자료 로드 실패 (${r.status})`);trace=await r.json();ds.cache.set(data.sample.key,trace);if(ds.cache.size>5)ds.cache.delete(ds.cache.keys().next().value);}
  if(serial!==ds.serial)return;ds.trace=trace;
  $('distributionStatus').textContent='원래 응답의 정확한 토큰 이력을 다시 입력해 측정했습니다. 출력 형식은 바꾸지 않았습니다. ·는 공백, ↵는 줄바꿈입니다.';
  await draw(serial);
 }catch(e){if(serial===ds.serial)$('distributionStatus').textContent=e.message;}
}
async function pendingStatus(serial){
 let note='이 샘플의 확률 분포는 아직 측정 중입니다. 위의 예측 구간과 겹침 점수는 원래 실험의 결과입니다.';
 try{const r=await fetch(BASE+'/../metadata/distribution_status.json',{cache:'no-store'});if(r.ok){const s=await r.json();note+=` 전체 ${s.completed}/${s.total}개 완료.`;}}catch(e){}
 if(serial!==ds.serial)return;$('distributionStatus').textContent=note;
 ds.timer=setTimeout(()=>{if(serial===ds.serial)render(ds.data,ds.candidate);},30000);
}
function renderStrip(){
 const c=ds.focus,branch=ds.candidate.branch;
 $('distFocusLabel').textContent=`${ds.step+1}번째 토큰 · ${interval(c)} 경로`;
 $('distTokenStrip').innerHTML=c.tokens.map((t,j)=>{
  const branchPoint=branch&&j===branch.position&&same(c.token_ids.slice(0,j),ds.candidate.token_ids.slice(0,j));
  return `<button data-dist-step="${j}" class="${j===ds.step?'active ':''}${branchPoint?'branch-point':''}" aria-pressed="${j===ds.step}" title="응답의 ${j+1}번째 토큰${branchPoint?' · 기본 분기 위치':''}"><span class="token-step">${j+1}</span><code>${esc(visible(t))}</code>${branchPoint?'<span class="branch-point-label">분기</span>':''}</button>`;
 }).join('');
 $('distTokenStrip').querySelectorAll('[data-dist-step]').forEach(b=>b.onclick=()=>chooseStep(Number(b.dataset.distStep)));
}
function chooseStep(j){if(!ds.focus)return;setFocus(ds.focus,Math.max(0,Math.min(ds.focus.tokens.length-1,j)));if(ds.trace)draw(ds.serial);}
async function raw(node){
 const key=ds.data.sample.key+'/'+node.file;if(ds.arrays.has(key))return ds.arrays.get(key);
 const r=await fetch(`${BASE}/${ds.data.sample.key}/${node.file}`);if(!r.ok)throw Error('전체 어휘 확률 로드 실패');
 const values=new Float32Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
 if(values.length!==node.vocabulary_size)throw Error('어휘 크기가 저장된 분포와 다릅니다.');ds.arrays.set(key,values);if(ds.arrays.size>20)ds.arrays.delete(ds.arrays.keys().next().value);return values;
}
async function rowsFor(node){
 const query=$('distSearch').value,limit=Number($('distTop').value);let rows=node.top.slice(0,limit);
 if(query){if(!ds.vocab){const r=await fetch(BASE+'/vocab.json');ds.vocab=await r.json();}const values=await raw(node);rows=[];
  for(let id=0;id<values.length;id++)if((ds.vocab[id]??'').includes(query)||String(id)===query)rows.push({id,token:ds.vocab[id]??`<${id}>`,probability:values[id]});
  rows.sort((a,b)=>b.probability-a.probability||a.id-b.id);rows=rows.slice(0,limit);}
 for(const row of node.observed)if(!rows.some(r=>r.id===row.id))rows.push(row);rows.sort((a,b)=>b.probability-a.probability||a.id-b.id);return rows;
}
function distributionRows(rows,node,colors,clickable=false){return rows.map(row=>{
 const color=colors.get(row.id),style=color?`style="border-left:3px solid ${color};background:${color}12"`:'';
 const log=row.probability>0?node.prefix_log_probability+Math.log(row.probability):-Infinity;
 return `<div class="prob-row ${color?'measured-choice':''}" ${style} ${clickable?`data-root-token="${row.id}"`:''}><code title="Token ID ${row.id}">${esc(visible(row.token))}</code><div class="prob-bar"><i style="width:${100*row.probability}%;background:${color||'#a4b3ac'}"></i></div><span>${probability(row.probability)}</span><span class="path-prob" title="여기까지 선택한 각 토큰의 조건부 확률을 곱한 값">누적 ${pathProbability(log)}</span></div>`;
 }).join('');}
function massNote(rows){const mass=rows.reduce((a,b)=>a+b.probability,0);return `표시한 토큰 ${(100*mass).toFixed(2)}% · 나머지 어휘 ${(100*Math.max(0,1-mass)).toFixed(2)}%`;}
function tableHeader(){return '<div class="prob-row prob-heading"><span>토큰</span><span>확률 막대</span><span>다음 토큰 확률</span><span>누적 = 여기까지 고른 토큰 확률의 곱</span></div>';}
async function draw(serial){
 try{
 const drawId=++ds.drawCounter,alive=()=>serial===ds.serial&&drawId===ds.drawCounter,rootExpanded=$('distributionTree').querySelector('.distribution-root details')?.open;
 const c=ds.focus,main=ds.candidate,j=ds.step,node=nodeFor(c,j);if(!node)return;
 const parent=main.branch?ds.data.candidates[main.branch.parent-1]:main;
 const atBranch=main.branch&&j===main.branch.position&&same(parent.token_ids.slice(0,j),c.token_ids.slice(0,j));
 const mainTokens=atBranch?[parent.token_ids[j],main.token_ids[j]]:[c.token_ids[j]];
 const tokens=[...new Set([...mainTokens,...node.observed.map(t=>t.id)])].filter(id=>node.observed.some(t=>t.id===id));
 if(!ds.visibilityInitialized){for(const id of tokens)if(!mainTokens.includes(id))ds.disabled.add(id);ds.visibilityInitialized=true;}
 const colors=new Map(tokens.map((id,i)=>[id,COLORS[i%COLORS.length]]));
 const rootRows=await rowsFor(node);if(!alive())return;
 $('distributionContext').innerHTML=atBranch?'공통으로 생성한 부분 다음에서 파란색과 주황색 토큰으로 갈렸습니다. 각 토큰을 고르면 다음 토큰의 확률도 다르게 조건화됩니다.':'선택한 자리까지 생성한 응답을 조건으로 토큰 분포와 후속 경로를 보여줍니다. 위의 정답·분기 전후 예측은 유지됩니다. ‘분기’ 표시가 있는 토큰을 누르면 기본 비교로 돌아갑니다.';
 let html=`<div class="distribution-root"><h4>${atBranch?'두 답이 갈린 순간':'지금 살펴보는 토큰 자리'}</h4><p class="prefix-caption">이때까지 생성한 공통 부분${atBranch?'':' / 탐색 중인 경로'}</p><pre class="dist-prefix">${esc(node.prefix_text||'(응답을 시작하기 전)')}<b>▏</b></pre><p class="formula">여기까지의 확률 곱 <b>${pathProbability(node.prefix_log_probability)}</b></p>${tableHeader()}${distributionRows(rootRows,node,colors,true)}<p class="small">${massNote(rootRows)}</p><div class="root-legend">${atBranch?`<span style="color:${COLORS[0]}">● 파랑: 분기 전 선택 <code>${esc(parent.tokens[j])}</code></span><span style="color:${COLORS[1]}">● 주황: 바꿔 고른 토큰 <code>${esc(main.tokens[j])}</code></span>`:'색이 있는 토큰은 저장된 응답에서 선택한 적 있는 토큰입니다.'}</div><details ${rootExpanded?'open':''}><summary>비교할 경로 추가·숨기기</summary><p class="small">저장된 다른 응답에서 선택한 토큰도 펼칠 수 있습니다. 분포만 있고 후속 응답이 없는 토큰은 펼치지 않습니다.</p><div class="distribution-chips">`;
 for(const id of tokens){const t=node.observed.find(x=>x.id===id),color=colors.get(id);html+=`<label style="border-color:${color};color:${color}"><input type="checkbox" data-child-token="${id}" ${ds.disabled.has(id)?'':'checked'}> ${esc(visible(t.token))}</label>`;}
 html+='</div></details></div><div class="distribution-children">';
 for(const id of tokens){
  if(ds.disabled.has(id))continue;
  const token=node.observed.find(x=>x.id===id),color=colors.get(id);
  const matches=x=>x.token_ids.length>j&&same(x.token_ids.slice(0,j),c.token_ids.slice(0,j))&&x.token_ids[j]===id;
  const representative=atBranch&&matches(parent)?parent:atBranch&&matches(main)?main:matches(c)?c:ds.data.candidates.find(matches);
  const next=nodeFor(representative,j+1),cumulative=node.prefix_log_probability+Math.log(token.probability);
  const before=atBranch&&id===parent.token_ids[j],after=atBranch&&id===main.token_ids[j];
  html+=`<article class="distribution-child" style="--child-color:${color}" data-path-candidate="${representative.index}"><h4><span class="token-color" style="background:${color}"></span>${before?'분기 전':after?'분기 후':'다른 경로'}: <code>${esc(visible(token.token))}</code> 선택</h4><span class="child-result">→ ${esc(interval(representative))}</span><p class="formula">이 토큰 선택 확률 <b>${probability(token.probability)}</b><br>선택 후 누적 = ${pathProbability(node.prefix_log_probability)} × ${token.probability.toPrecision(5)}<br>= <b>${pathProbability(cumulative)}</b></p>`;
  if(!next){html+='<p class="small">응답 종료 토큰(EOS)을 골라 응답이 끝났습니다.</p></article>';continue;}
  const nextRows=await rowsFor(next);if(!alive())return;
  const nextId=representative.token_ids[j+1],nextP=tokenProbability(next,nextId);
  html+=`<div class="next-choice">이 조건에서 다음으로 고른 토큰은 <code>${esc(visible(representative.tokens[j+1]))}</code> <b>${probability(nextP)}</b><br>이어서 생성한 최종 구간이 위의 ${esc(interval(representative))}입니다.</div><p class="prefix-caption">선택 후 다음 토큰 분포</p>${tableHeader()}${distributionRows(nextRows,next,new Map([[nextId,color]]))}<p class="small">${massNote(nextRows)}</p><details class="child-more"><summary>선택 후 원문·다음 위치 분석</summary><pre class="dist-prefix">${esc(next.prefix_text)}</pre><p class="formula">누적 = 이전까지 확률 곱 × 선택 토큰 확률 × 그 행의 다음 토큰 확률</p><button data-next-node="${representative.index}">이 경로에서 한 토큰 더 따라가기</button><a class="raw-prob" href="${BASE}/${ds.data.sample.key}/${next.file}" download>전체 어휘 FP32 확률 다운로드</a></details></article>`;
 }
 html+='</div>';if(!alive())return;$('distributionTree').innerHTML=html;
 $('distributionTree').querySelectorAll('[data-child-token]').forEach(input=>input.onchange=()=>{const id=Number(input.dataset.childToken);if(input.checked)ds.disabled.delete(id);else ds.disabled.add(id);draw(ds.serial);});
 $('distributionTree').querySelectorAll('[data-root-token]').forEach(row=>row.onclick=()=>{const id=Number(row.dataset.rootToken);if(!tokens.includes(id))return;if(ds.disabled.has(id))ds.disabled.delete(id);else ds.disabled.add(id);draw(ds.serial);});
 $('distributionTree').querySelectorAll('[data-next-node]').forEach(button=>button.onclick=()=>{setFocus(ds.data.candidates[Number(button.dataset.nextNode)-1],j+1);draw(ds.serial);});
 renderProduct(main,parent);
 if(!ds.autoJump&&new URLSearchParams(location.hash.slice(1)).get('panel')==='distributions'){ds.autoJump=true;$('distributionPanel').scrollIntoView({block:'start'});}
 }catch(e){if(serial===ds.serial)$('distributionStatus').textContent=e.message;}
}
function renderProduct(c,parent){
 const expanded=$('distributionProduct').querySelector('details')?.open;
 function product(candidate,title){let sum=0;const rows=[];for(let j=0;j<candidate.token_ids.length;j++){const p=tokenProbability(nodeFor(candidate,j),candidate.token_ids[j]);sum+=p>0?Math.log(p):-Infinity;rows.push(`<tr><td>${j+1}</td><td><code>${esc(visible(candidate.tokens[j]))}</code></td><td>${probability(p)}</td><td>${pathProbability(sum)}</td></tr>`);}return {sum,html:`<div><h4>${title}: ${esc(interval(candidate))}</h4><div class="table-scroll product-table"><table><thead><tr><th>응답 토큰 순번</th><th>선택한 토큰</th><th>이 토큰의 조건부 확률</th><th>여기까지 확률 곱</th></tr></thead><tbody>${rows.join('')}</tbody></table></div></div>`};}
 const before=product(parent,'분기 전 응답'),after=product(c,'분기 후 응답');
 $('distributionProduct').innerHTML=`<div class="probability-product"><h4>토큰 하나의 확률과 전체 응답의 확률은 다릅니다</h4><p class="formula">P(전체 응답 | 영상·질문) = p₁ × p₂ × … × pₙ</p><div class="product-comparison"><div><span class="before-text">분기 전 전체 응답</span><strong>${pathProbability(before.sum)}</strong></div><div><span class="after-text">분기 후 전체 응답</span><strong>${pathProbability(after.sum)}</strong></div></div><p class="small">첫 토큰부터 응답 종료 토큰(EOS)까지 각 조건부 확률을 모두 곱한 값입니다. 숫자가 클수록 모델이 더 높은 확률을 준 응답이며, 정답과 더 가깝다는 뜻은 아닙니다.</p><details ${expanded?'open':''}><summary>두 응답의 토큰별 확률과 누적곱 확인</summary><div class="product-tables">${before.html}${after.html}</div></details></div>`;
}
function bind(){
 $('distTop').onchange=()=>{if(ds.trace)draw(ds.serial);};let timer;$('distSearch').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{if(ds.trace)draw(ds.serial);},250);};
}
window.ROLLOUT_DISTRIBUTIONS={render,state:ds,chooseStep,pathProbability};bind();
})();
