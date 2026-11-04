'use strict';
const $=id=>document.getElementById(id),ASSETS=window.ROLLOUT_ASSETS;
const DOT='end.seconds.decimal_separator',PAGE_SIZE=10;
const state={manifest:null,map:new Map(),filtered:[],grouped:[],page:0,preset:'all',flag:'',current:null,candidate:1,serial:0,segment:null,cache:new Map()};
const escapeHTML=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=3)=>Number.isFinite(x)?x.toFixed(n):'—';
const seconds=x=>Number.isFinite(x)?Number(x.toFixed(2)).toString():'—';
const pairText=p=>p?`${seconds(p[0])} – ${seconds(p[1])}초`:'유효한 구간 없음';
const score=x=>Number.isFinite(x)?x.toFixed(3):'—';
const delta=x=>`<span class="${x>1e-9?'positive':x<-1e-9?'negative':'zero'}">${x>=0?'+':''}${x.toFixed(3)}</span>`;
const roleText=roles=>roles.map(r=>(state.manifest.roles[r]||r).replace(' · 초 필드 · ','의 ').replace('소수점 자체','소수점')).join(' / ');
const dsName=s=>({'charades-timelens':'Charades','activitynet-timelens':'ActivityNet','qvhighlights-timelens':'QVHighlights'}[s.dataset]||s.dataset);
function basicMatch(s){const q=$('search').value.trim().toLowerCase(),ds=$('dataset').value;return (!ds||s.dataset===ds)&&(!q||`${s.key} ${s.query}`.toLowerCase().includes(q));}
const numericFields={initial:b=>state.map.get(b.sample).greedy_iou,parent:b=>b.parent_iou,candidate:b=>b.iou,greedyDelta:b=>b.greedy_delta,parentDelta:b=>b.parent_delta,step:b=>b.position+1,tokenP:b=>b.probability};
const numericIds=Object.keys(numericFields).flatMap(key=>[key+'Min',key+'Max']);
function scenarioMatch(b,name=state.preset){
 const s=state.map.get(b.sample);
 if(name==='recovered')return s.greedy_iou<.5&&b.iou>=.5;
 if(name==='improved')return b.greedy_delta>=.01;
 if(name==='worsened')return b.greedy_delta<=-.01;
 if(name==='parent_improved')return b.parent_delta>=.01;
 if(name==='dot')return b.dot;
 if(name==='greedy_short')return s.greedy_short&&b.dot;
 if(name==='parent_short')return b.short_parent;
 return true;
}
function applyPreset(name,update=true){
 state.preset=name;state.flag='';
 document.querySelectorAll('[data-preset]').forEach(b=>{b.classList.toggle('active',b.dataset.preset===name);b.setAttribute('aria-pressed',String(b.dataset.preset===name));if(b.querySelector('span'))b.title=b.querySelector('span').textContent;});
 $('scenarioDescription').textContent=document.querySelector(`[data-preset="${name}"] span`)?.textContent||'';
 if(update){filterBranches();const first=state.grouped[0];if(first)selectCase(first.sample,first.candidate,true);updateHash();}
}
function filterBranches(){
 const samples=state.manifest.samples.filter(basicMatch),allowed=new Set(samples.map(s=>s.key));
 let invalid=false;const ranges=Object.entries(numericFields).map(([key,get])=>{
  const a=$(key+'Min'),b=$(key+'Max'),low=a.value===''?-Infinity:Number(a.value),high=b.value===''?Infinity:Number(b.value);
  const valid=a.validity.valid&&b.validity.valid&&low<=high;if(!valid)invalid=true;
  a.setAttribute('aria-invalid',String(!valid));b.setAttribute('aria-invalid',String(!valid));return {get,low,high};
 });
 $('numericStatus').textContent=invalid?'입력 범위와 최솟값·최댓값 순서를 확인하세요.':'';
 state.filtered=invalid?[]:state.manifest.branches.filter(b=>allowed.has(b.sample)&&scenarioMatch(b)&&ranges.every(r=>r.get(b)>=r.low&&r.get(b)<=r.high));
 const sort=$('sort').value,comparison=['parent_improved','dot','parent_short'].includes(state.preset)?'parent_delta':'greedy_delta';
 state.filtered.sort((a,b)=>(sort==='asc'?a[comparison]-b[comparison]:b[comparison]-a[comparison])||a.sample.localeCompare(b.sample)||a.candidate-b.candidate);
 const groups=new Map();for(const b of state.filtered){if(!groups.has(b.sample))groups.set(b.sample,{...b,matchingBranches:0});groups.get(b.sample).matchingBranches++;}
 if(state.preset==='all'&&!invalid&&!numericIds.some(id=>$(id).value!==''))for(const s of samples)if(!groups.has(s.key))groups.set(s.key,{sample:s.key,candidate:1,parent:1,parent_pair:s.greedy_pair,pair:s.greedy_pair,parent_iou:s.greedy_iou,iou:s.greedy_iou,roles:[],before:'',after:'',parent_delta:0,greedy_delta:0,matchingBranches:0});
 state.grouped=[...groups.values()];
 if(sort==='sample')state.grouped.sort((a,b)=>a.sample.localeCompare(b.sample));
 if(sort==='greedy_low'||sort==='greedy_high')state.grouped.sort((a,b)=>(sort==='greedy_low'?1:-1)*(state.map.get(a.sample).greedy_iou-state.map.get(b.sample).greedy_iou)||a.sample.localeCompare(b.sample));
 $('scope').textContent=`${state.grouped.length}개 샘플 · 샘플마다 조건에 맞는 분기 중 ${comparison==='parent_delta'?'부모 후보':'greedy'} 응답 대비 ΔIoU가 가장 ${sort==='asc'?'작은':'큰'} 분기를 표시합니다.`;
 const baseAllowed=new Set(samples.map(s=>s.key));
 for(const name of ['recovered','improved','dot','worsened','parent_improved','greedy_short','parent_short']){
  const count=new Set(state.manifest.branches.filter(b=>baseAllowed.has(b.sample)&&!invalid&&ranges.every(r=>r.get(b)>=r.low&&r.get(b)<=r.high)&&scenarioMatch(b,name)).map(b=>b.sample)).size;
  const el=document.querySelector(`[data-count="${name}"]`);if(el)el.textContent=`${count}개`;
  if(['greedy_short','parent_short'].includes(name))document.querySelector(`[data-preset="${name}"]`).disabled=count===0;
 }
 const allCount=numericIds.some(id=>$(id).value!=='')?new Set(state.manifest.branches.filter(b=>baseAllowed.has(b.sample)&&!invalid&&ranges.every(r=>r.get(b)>=r.low&&r.get(b)<=r.high)).map(b=>b.sample)).size:baseAllowed.size;
 document.querySelector('[data-count="all"]').textContent=`${allCount}개`;
 state.page=0;renderCatalog();
}
function activeBranch(fallback){
 if(state.current?.sample.key!==fallback.sample)return fallback;
 const c=state.current.candidates[state.candidate-1],b=c.branch,p=b?state.current.candidates[b.parent-1]:c;
 return {...fallback,candidate:c.index,parent:p.index,parent_pair:p.pair,pair:c.pair,parent_iou:p.iou,iou:c.iou,roles:b?.roles||[],before:b?.original_token||'',after:b?.forced_token||'',parent_delta:c.iou-p.iou,greedy_delta:c.iou-state.current.candidates[0].iou};
}
function renderCatalog(){
 const pages=Math.max(1,Math.ceil(state.grouped.length/PAGE_SIZE));state.page=Math.max(0,Math.min(state.page,pages-1));
 const entries=state.grouped.slice(state.page*PAGE_SIZE,(state.page+1)*PAGE_SIZE);
 $('branches').innerHTML=entries.length?entries.map(entry=>{
  const s=state.map.get(entry.sample),active=state.current?.sample.key===entry.sample,b=activeBranch(entry);
  return `<button class="case-card ${active?'selected':''}" data-sample="${escapeHTML(s.key)}" data-candidate="${b.candidate}" aria-pressed="${active}"><span class="case-header"><span class="case-id">${escapeHTML(s.key)}</span>${active?'<span class="case-selected-label">선택됨</span>':''}</span><strong>${escapeHTML(s.query)}</strong><span class="case-truth">정답 <b>${escapeHTML(pairText(s.gt))}</b></span><span class="case-predictions"><span class="before-text">전 ${escapeHTML(pairText(b.parent_pair))}</span><span>→</span><span class="after-text">후 ${escapeHTML(pairText(b.pair))}</span></span><span class="case-change">${b.roles.length?`${escapeHTML(roleText(b.roles))} <code>${escapeHTML(b.before)}</code> → <code>${escapeHTML(b.after)}</code>`:'추가 분기 없음'}</span><span class="case-score">IoU ${score(b.parent_iou)} → <b>${score(b.iou)}</b> ${delta(b.parent_delta)}</span>${b.parent!==1?`<span class="case-origin">greedy 예측 ${escapeHTML(pairText(s.greedy_pair))} · 위 ‘전’은 탐색 중 후보</span>`:''}</button>`;
 }).join(''):'<div class="empty-state">조건에 맞는 샘플이 없습니다.<br>검색어나 필터를 조정하세요.</div>';
 $('branches').querySelectorAll('[data-sample]').forEach(card=>card.onclick=()=>selectCase(card.dataset.sample,Number(card.dataset.candidate),true));
 $('pageInfo').textContent=`${state.page+1} / ${pages}`;$('previousPage').disabled=state.page===0;$('nextPage').disabled=state.page===pages-1;
 const index=state.grouped.findIndex(x=>x.sample===state.current?.sample.key);
 $('previousSample').disabled=index<=0;$('nextSample').disabled=index>=state.grouped.length-1||!state.grouped.length;
}
function defaultCandidate(key){
 const entry=state.grouped.find(x=>x.sample===key);if(entry)return entry.candidate;
 const branches=state.manifest.branches.filter(b=>b.sample===key).sort((a,b)=>b.greedy_delta-a.greedy_delta||a.candidate-b.candidate);return branches[0]?.candidate||1;
}
async function selectCase(key,candidate=null,scroll=false){
 const serial=++state.serial;$('loadStatus').textContent='사례를 불러오는 중…';
 try{
  let data=state.cache.get(key);
  if(!data){const response=await fetch(`${window.ROLLOUT_SAMPLE_BASE||ASSETS+'/samples'}/${encodeURIComponent(key)}.json`);if(!response.ok)throw Error(`샘플 불러오기 실패: ${response.status}`);data=await response.json();state.cache.set(key,data);if(state.cache.size>6)state.cache.delete(state.cache.keys().next().value);}
  if(serial!==state.serial)return;const changed=state.current?.sample.key!==key;state.current=data;
  $('sampleId').textContent=data.sample.key;$('query').textContent=data.sample.query;
  $('sampleMeta').textContent=`${dsName(data.sample)} · 영상 ${seconds(data.sample.duration)}초 · 저장된 예측 ${data.candidates.length}개`;
  if(changed){state.segment=null;$('video').pause();$('video').src=window.ROLLOUT_VIDEO_BASE?new URL(data.video,new URL(window.ROLLOUT_VIDEO_BASE,location.href)).href:data.video;$('videoStatus').textContent='';$('videoDetails').open=false;}
  const listIndex=state.grouped.findIndex(entry=>entry.sample===key);if(listIndex>=0)state.page=Math.floor(listIndex/PAGE_SIZE);
  showCandidate(candidate||defaultCandidate(key));$('loadStatus').textContent='';
  if(scroll&&window.innerWidth<1000)$('samplePanel').scrollIntoView({behavior:'smooth',block:'start'});
 }catch(error){$('loadStatus').textContent=error.message;}
}
function answerBlock(title,candidate,position){
 const tokens=candidate.tokens.map((token,index)=>token.startsWith('<|')?`<span class="faded">${escapeHTML(token)}</span>`:index===position?`<mark class="changed">${escapeHTML(token)}</mark>`:escapeHTML(token)).join('');
 return `<div class="answer"><b>${escapeHTML(title)} · ${pairText(candidate.pair)} · IoU ${score(candidate.iou)}</b><pre>${tokens}</pre></div>`;
}
function responseCard(candidate,kind,branch){
 const label=candidate.index===1?'greedy 응답':kind==='before'?'부모 후보':'분기 후보';
 return `<article class="fork-response ${kind}" data-response-candidate="${candidate.index}"><div class="fork-response-heading"><b>${label} #${candidate.index}</b><span class="decoding-method">${candidate.index===1?'greedy decoding':'forced token → greedy decoding'}</span></div><code class="fork-sentence">${branchSentence(candidate,branch?.position,kind)}</code><div class="comparison-metrics"><div><span>예측 구간</span><span class="comparison-span"><strong class="fork-outcome">${escapeHTML(pairText(candidate.pair))}</strong><button class="comparison-play" data-overview-play="${kind}" aria-label="${label} #${candidate.index} 구간 재생">▶ 재생</button></span></div><div><span>IoU</span><strong class="comparison-iou">${score(candidate.iou)}</strong></div></div></article>`;
}
function branchSentence(candidate,position,kind='after'){
 const prefix=candidate.tokens.slice(0,position).filter(t=>!t.startsWith('<|')).join('');
 const picked=candidate.tokens[position],suffix=candidate.tokens.slice(position+1).filter(t=>!t.startsWith('<|')).join('');
 if(position==null)return escapeHTML(candidate.answer);
 const visible=picked?.startsWith('<|')?'EOS':picked?.trim()?picked:picked?.replace(/ /g,'␣').replace(/\n/g,'↵').replace(/\t/g,'⇥');
 return `<span class="sentence-prefix">${escapeHTML(prefix)}</span><mark class="sentence-pivot ${kind}" title="${escapeHTML(picked)}">${escapeHTML(visible)}</mark><span class="sentence-suffix">${escapeHTML(suffix)}</span>`;
}
function renderBranchMap(current){
 const data=state.current,chain=[];
 for(let row=current;row;row=row.branch?data.candidates[row.branch.parent-1]:null)chain.unshift(row);
 const ancestors=new Set(chain.map(row=>row.index));
 $('branchCount').textContent=`${data.candidates.length}개 경로 · 선택 #${current.index}`;
 $('branchAncestry').innerHTML='<span class="ancestry-label">선택한 경로의 이력</span>'+chain.map(row=>`<button data-branch-candidate="${row.index}" title="${escapeHTML(row.answer)}" aria-current="${row.index===current.index?'step':'false'}">${row.branch?`#${row.index} · <code>${escapeHTML(row.branch.original_token)}</code> → <code>${escapeHTML(row.branch.forced_token)}</code>`:'greedy #1'}</button>`).join('<span aria-hidden="true">→</span>');
 const children=new Map();for(const row of data.candidates){const parent=row.branch?.parent||0;if(!children.has(parent))children.set(parent,[]);children.get(parent).push(row);}
 function tree(parent,depth){return (children.get(parent)||[]).map(row=>{
  const b=row.branch,selected=row.index===current.index;
  return `<div class="branch-map-node"><button data-branch-candidate="${row.index}" class="branch-map-choice ${selected?'selected':ancestors.has(row.index)?'ancestor':''}" style="--depth:${depth}" aria-pressed="${selected}"><span class="branch-connector" aria-hidden="true">${depth?'└':'●'}</span><span class="branch-choice-body"><span class="branch-choice-meta"><b>#${row.index}</b><span>${b?`부모 #${b.parent} · ${escapeHTML(roleText(b.roles))}`:'greedy · 시작 경로'}</span>${b?`<span class="branch-choice-swap"><code class="before-token">${escapeHTML(b.original_token)}</code> → <code class="after-token">${escapeHTML(b.forced_token)}</code></span>`:''}<span class="branch-map-result">IoU ${score(row.iou)}${selected?' · 선택됨':''}</span></span><code class="branch-choice-sentence">${branchSentence(row,b?.position)}</code></span></button>${tree(row.index,depth+1)}</div>`;
 }).join('');}
 const map=$('branchMap'),top=map.scrollTop;map.innerHTML=tree(0,0);map.scrollTop=top;
 const selected=map.querySelector('.selected');if(selected&&(selected.offsetTop<map.scrollTop||selected.offsetTop+selected.offsetHeight>map.scrollTop+map.clientHeight))map.scrollTop=selected.offsetTop-map.clientHeight/2;
 document.querySelectorAll('[data-branch-candidate]').forEach(button=>button.onclick=()=>showCandidate(Number(button.dataset.branchCandidate)));
}
function renderResponseComparison(current){
 const data=state.current,b=current.branch,parent=b?data.candidates[b.parent-1]:current;
 const truth=`<div class="comparison-truth"><span>정답 구간 <small>GT</small></span><strong>${escapeHTML(pairText(data.sample.gt))}</strong><button data-overview-play="truth">이 구간 재생 ↗</button></div>`;
 const prefix=b?`<div class="fork-prefix"><span>공통 prefix</span><code>${escapeHTML(current.tokens.slice(0,b.position).join(''))}</code><span class="fork-point">여기서 분기 ↙ ↘</span></div>`:'';
 $('branchPreview').innerHTML=`<div class="fork-heading"><b>${b?'같은 문장에서 갈라진 두 응답':'greedy 응답과 정답 비교'}</b>${b?`<span class="comparison-delta">부모 대비 ΔIoU ${delta(current.iou-parent.iou)}</span>`:''}</div>${truth}${prefix}<div class="fork-responses ${b?'':'single'}">${responseCard(parent,'before',b)}${b?responseCard(current,'after',b):''}</div>${b?'<p class="fork-legend">옅은 부분은 공통 prefix · 테두리 토큰은 분기 선택 · 이어지는 부분은 각 선택 이후의 완성 문장</p>':'<p class="fork-legend">위에서 분기 경로를 선택하면 부모 응답과 분기 응답을 비교합니다.</p>'}`;
 const pairs={truth:data.sample.gt,before:parent.pair,after:current.pair};
 $('branchPreview').querySelectorAll('[data-overview-play]').forEach(button=>button.onclick=()=>playSpan(pairs[button.dataset.overviewPlay]));
}
function showCandidate(index){
 const data=state.current;if(!data)return;const c=data.candidates.find(x=>x.index===index)||data.candidates[0];state.candidate=c.index;
 if(window.ROLLOUT_EXPLORER_OPTIONS?.caseOnly){renderCatalog();updateHash();window.dispatchEvent(new CustomEvent('rollout:candidate',{detail:{data,candidate:c}}));return;}
 const b=c.branch,parent=b?data.candidates[b.parent-1]:c,greedy=data.candidates[0];
 $('originContext').innerHTML=parent.index!==1?`<div class="origin-context"><b>부모 후보 #${parent.index}는 greedy 응답과 다릅니다.</b> greedy는 ${escapeHTML(pairText(greedy.pair))}를 예측했고, 탐색 중 얻은 ${escapeHTML(pairText(parent.pair))}에서 다시 분기했습니다. greedy 응답과 비교한 ΔIoU는 ${delta(c.iou-greedy.iou)}입니다. <button id="showOriginal">greedy 응답 보기</button> <button id="parentJump">분기 전 후보 보기</button></div>`:'';
 if($('showOriginal'))$('showOriginal').onclick=()=>showCandidate(1);if($('parentJump'))$('parentJump').onclick=()=>showCandidate(parent.index);
 $('answerComparison').innerHTML=b?answerBlock('분기 부모 후보',parent,b.position)+answerBlock('분기 후 응답',c,b.position)+(parent.index!==1?answerBlock('greedy 응답',greedy,null):''):answerBlock('greedy 응답',c,null);
 renderBranchMap(c);renderResponseComparison(c);renderSelectionAnalysis();renderTimeline(c,parent);renderCatalog();updateHash();
 if(window.ROLLOUT_DISTRIBUTIONS)window.ROLLOUT_DISTRIBUTIONS.render(data,c);
 window.dispatchEvent(new CustomEvent('rollout:candidate',{detail:{data,candidate:c}}));
}
function renderSelectionAnalysis(){
 const data=state.current;if(!data)return;const k=$('budget').value,choice=data.candidates[data.choices[k][$('method').value]-1],oracle=data.candidates[data.choices[k].oracle-1],current=data.candidates[state.candidate-1];
 $('choiceSummary').innerHTML=`<div class="selection-results"><div><b>${escapeHTML($('method').selectedOptions[0].textContent)} 선택</b><strong>${escapeHTML(pairText(choice.pair))}</strong><span>IoU ${score(choice.iou)} · 후보 ${choice.index}</span></div><div><b>oracle 선택</b><strong>${escapeHTML(pairText(oracle.pair))}</strong><span>IoU ${score(oracle.iou)} · 후보 ${oracle.index}</span></div></div><p class="explain">지금 확인 중인 분기 후 예측은 ${pairText(current.pair)}입니다. ${current.index===choice.index?'선택기가 이 후보를 골랐습니다.':'선택기는 이 후보를 고르지 않았습니다.'}${current.index>Number(k)?' 현재 후보는 선택기에 준 후보 수 범위 밖입니다.':''}</p>`;
 $('candidates').innerHTML=data.candidates.filter(row=>row.index<=Number(k)).map(row=>{const br=row.branch,tags=[];if(row.index===choice.index)tags.push('선택된 답');if(row.index===oracle.index)tags.push('GT 최선');if(row.index===1)tags.push('greedy 응답');return `<tr data-candidate="${row.index}" class="${row.index===state.candidate?'selected':''}"><td>${row.index}${br?` / ${br.parent}`:''}</td><td>${escapeHTML(br?roleText(br.roles):'greedy 응답')}</td><td>${br?`${escapeHTML(br.original_token)} → ${escapeHTML(br.forced_token)}`:'—'}</td><td>${escapeHTML(pairText(row.pair))}</td><td>${score(row.iou)}</td><td>${delta(row.iou-data.candidates[0].iou)}</td><td>${br?delta(br.parent_delta):'—'}</td><td>${tags.map(t=>`<span class="tag">${t}</span>`).join('')}</td></tr>`;}).join('');
 $('candidates').querySelectorAll('[data-candidate]').forEach(row=>row.onclick=()=>showCandidate(Number(row.dataset.candidate)));
}
function renderTimeline(c,parent){
 const s=state.current.sample,tracks=[['정답',s.gt,'#277c68'],['분기 전 예측',parent.pair,'#2678b5'],['분기 후 예측',c.pair,'#d16b32']];
 if(parent.index!==1)tracks.push(['greedy 응답',state.current.candidates[0].pair,'#8c9a9c']);
 $('timeline').innerHTML=tracks.map(([name,pair,color],index)=>{const a=pair?Math.max(0,Math.min(1,pair[0]/s.duration)):0,b=pair?Math.max(0,Math.min(1,pair[1]/s.duration)):0;return `<div class="timeline-row"><div class="timeline-label"><span>${name} ${escapeHTML(pairText(pair))}</span><button data-play="${index}" ${pair?'':'disabled'}>재생</button></div><div class="track"><div class="bar" style="left:${a*100}%;width:${Math.max(0,b-a)*100}%;--bar-color:${color}"></div></div></div>`;}).join('');
 $('timeline').querySelectorAll('[data-play]').forEach(b=>b.onclick=()=>playSpan(tracks[Number(b.dataset.play)][1]));
}
function playSpan(pair){
 if(!pair)return;$('videoDetails').open=true;const video=$('video'),duration=Number.isFinite(video.duration)?video.duration:state.current.sample.duration;
 const a=Math.max(0,Math.min(duration,pair[0])),b=Math.max(0,Math.min(duration,pair[1]));if(b<=a){$('videoStatus').textContent='영상 범위 밖인 구간입니다.';return;}
 state.segment=[a,b];video.currentTime=a;$('videoStatus').textContent=`${seconds(a)} – ${seconds(b)}초 재생 중`;
 $('videoDetails').scrollIntoView({behavior:'smooth',block:'nearest'});video.play().catch(e=>$('videoStatus').textContent=e.message);
}
function updateHash(){
 if(!state.current)return;const params=new URLSearchParams({preset:state.preset,sample:state.current.sample.key,candidate:state.candidate,flag:state.flag});
 for(const id of ['budget','method'])if($(id))params.set(id,$(id).value);
 const old=new URLSearchParams(location.hash.slice(1));for(const key of ['panel','review','axis','approach'])if(old.get(key))params.set(key,old.get(key));
 for(const id of ['search','dataset','sort',...numericIds])if($(id).value)params.set(id,$(id).value);
 history.replaceState(null,'',`${location.pathname}${location.search}#${params}`);window.dispatchEvent(new CustomEvent('rollout:context'));
}
async function restoreHash(){
 const hash=new URLSearchParams(location.hash.slice(1));$('search').value='';$('dataset').value='';if($('budget'))$('budget').value='32';if($('method'))$('method').value='probability_weighted_medoid';$('sort').value='sample';for(const id of numericIds)$(id).value='';applyPreset(hash.get('preset')||'all',false);
 for(const id of ['search','dataset','sort','budget','method',...numericIds])if(hash.has(id)&&$(id))$(id).value=hash.get(id);
 filterBranches();
 const requested=hash.get('sample'),key=state.map.has(requested)?requested:state.grouped[0]?.sample||state.manifest.samples[0].key;
 await selectCase(key,requested&&requested!==key?null:Number(hash.get('candidate'))||null);
}
async function init(){
 const response=await fetch(`${ASSETS}/manifest.json`);if(!response.ok)throw Error('사례 목록을 불러오지 못했습니다.');state.manifest=await response.json();state.map=new Map(state.manifest.samples.map(s=>[s.key,s]));
 $('summary').textContent=`${state.manifest.summary.samples}개 샘플`;
 document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>applyPreset(b.dataset.preset));
 for(const id of ['search','dataset','sort',...numericIds])$(id).addEventListener(id==='search'||numericIds.includes(id)?'input':'change',()=>{filterBranches();updateHash();});
 $('previousPage').onclick=()=>{state.page--;renderCatalog();};$('nextPage').onclick=()=>{state.page++;renderCatalog();};
 $('resetNumeric').onclick=()=>{for(const id of numericIds)$(id).value='';filterBranches();updateHash();};
 for(const id of ['budget','method'])if($(id))$(id).onchange=()=>{renderSelectionAnalysis();updateHash();};
 function adjacent(direction){const i=state.grouped.findIndex(x=>x.sample===state.current?.sample.key),entry=state.grouped[i+direction];if(entry)selectCase(entry.sample,entry.candidate);}
 $('previousSample').onclick=()=>adjacent(-1);$('nextSample').onclick=()=>adjacent(1);
 $('video').ontimeupdate=()=>{if(state.segment&&$('video').currentTime>=state.segment[1]){$('video').pause();state.segment=null;}};
 $('video').onerror=()=>{$('videoStatus').textContent='영상을 불러오지 못했습니다.';};
 window.addEventListener('hashchange',restoreHash);
 await restoreHash();
}
window.ROLLOUT_VIEWER={state,selectCase,showCandidate,applyPreset,playSpan};
init().catch(error=>{$('summary').textContent=error.message;$('loadStatus').textContent=error.message;});
