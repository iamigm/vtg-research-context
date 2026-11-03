'use strict';
const $=id=>document.getElementById(id),ASSETS=window.ROLLOUT_ASSETS;
const DOT='end.seconds.decimal_separator',PAGE_SIZE=10;
const state={manifest:null,map:new Map(),filtered:[],grouped:[],page:0,preset:'all',flag:'',current:null,candidate:1,serial:0,segment:null,cache:new Map()};
const escapeHTML=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=3)=>Number.isFinite(x)?x.toFixed(n):'—';
const seconds=x=>Number.isFinite(x)?Number(x.toFixed(2)).toString():'—';
const pairText=p=>p?`${seconds(p[0])} – ${seconds(p[1])}초`:'유효한 구간 없음';
const score=x=>`${(100*x).toFixed(2)}%`;
const delta=x=>`<span class="${x>1e-9?'positive':x<-1e-9?'negative':'zero'}">${x>=0?'+':''}${(100*x).toFixed(2)}%p</span>`;
const roleText=roles=>roles.map(r=>(state.manifest.roles[r]||r).replace(' · 초 필드 · ','의 ').replace('소수점 자체','소수점')).join(' / ');
const dsName=s=>({'charades-timelens':'Charades','activitynet-timelens':'ActivityNet','qvhighlights-timelens':'QVHighlights'}[s.dataset]||s.dataset);
function basicMatch(s){const q=$('search').value.trim().toLowerCase(),ds=$('dataset').value;return (!ds||s.dataset===ds)&&(!q||`${s.key} ${s.query}`.toLowerCase().includes(q));}
function sampleMatch(s){if(!basicMatch(s))return false;return !(state.flag==='dot'&&!s.dot_count||state.flag==='greedy_short'&&!s.greedy_short||state.flag==='parent_short'&&!s.any_short);}
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
 state.preset=name;state.flag='';$('role').value='';$('comparison').value='greedy_delta';$('outcome').value='all';$('minimum').value='0';
 if(name==='improved'){$('outcome').value='improved';$('minimum').value='.01';}
 if(name==='worsened'){$('outcome').value='worsened';$('minimum').value='.01';}
 if(name==='parent_improved'){$('comparison').value='parent_delta';$('outcome').value='improved';$('minimum').value='.01';}
 if(['dot','greedy_short','parent_short'].includes(name)){state.flag=name;$('role').value=DOT;if(name==='dot')$('comparison').value='parent_delta';}
 document.querySelectorAll('[data-preset]').forEach(b=>{b.classList.toggle('active',b.dataset.preset===name);b.setAttribute('aria-pressed',String(b.dataset.preset===name));if(b.querySelector('span'))b.title=b.querySelector('span').textContent;});
 $('scenarioDescription').textContent=document.querySelector(`[data-preset="${name}"] span`)?.textContent||'세부 조건에 맞는 사례를 표시합니다.';
 if(update){filterBranches();const first=state.grouped[0];if(first)selectCase(first.sample,first.candidate,true);updateHash();}
}
function filterBranches(){
 const samples=state.manifest.samples.filter(sampleMatch),allowed=new Set(samples.map(s=>s.key));
 const role=$('role').value,comparison=$('comparison').value,outcome=$('outcome').value,min=Math.max(0,Number($('minimum').value)||0);
 state.filtered=state.manifest.branches.filter(b=>{
  if(!allowed.has(b.sample)||!scenarioMatch(b)||role&&!b.roles.includes(role))return false;
  const d=b[comparison];
  if(outcome==='improved'&&!(d>1e-9&&d>=min))return false;
  if(outcome==='worsened'&&!(d<-1e-9&&-d>=min))return false;
  if(outcome==='same'&&Math.abs(d)>1e-9)return false;
  return !(outcome==='all'&&Math.abs(d)<min);
 });
 const sort=$('sort').value;
 state.filtered.sort((a,b)=>(sort==='asc'?a[comparison]-b[comparison]:b[comparison]-a[comparison])||a.sample.localeCompare(b.sample)||a.candidate-b.candidate);
 const groups=new Map();for(const b of state.filtered){if(!groups.has(b.sample))groups.set(b.sample,{...b,matchingBranches:0});groups.get(b.sample).matchingBranches++;}
 if(state.preset==='all'&&!role&&outcome==='all'&&min===0)for(const s of samples)if(!groups.has(s.key))groups.set(s.key,{sample:s.key,candidate:1,parent:1,parent_pair:s.greedy_pair,pair:s.greedy_pair,parent_iou:s.greedy_iou,iou:s.greedy_iou,roles:[],before:'',after:'',parent_delta:0,greedy_delta:0,matchingBranches:0});
 state.grouped=[...groups.values()];
 if(sort==='sample')state.grouped.sort((a,b)=>a.sample.localeCompare(b.sample));
 if(sort==='greedy_low'||sort==='greedy_high')state.grouped.sort((a,b)=>(sort==='greedy_low'?1:-1)*(state.map.get(a.sample).greedy_iou-state.map.get(b.sample).greedy_iou)||a.sample.localeCompare(b.sample));
 $('scope').textContent=`${state.grouped.length}개 샘플 · 같은 샘플은 한 번씩 표시합니다. 샘플별로 조건에 맞는 변경 중 겹침 ${sort==='asc'?'감소':'증가'}가 가장 큰 변경을 대표로 표시합니다.`;
 const current=$('sampleSelect').value;
 $('sampleSelect').innerHTML=samples.map(s=>`<option value="${escapeHTML(s.key)}">${escapeHTML(s.query.slice(0,48))} · ${escapeHTML(s.key)}</option>`).join('');
 if(samples.some(s=>s.key===current))$('sampleSelect').value=current;
 const baseAllowed=new Set(state.manifest.samples.filter(basicMatch).map(s=>s.key));
 for(const name of ['recovered','improved','dot','worsened']){
  const count=new Set(state.manifest.branches.filter(b=>baseAllowed.has(b.sample)&&scenarioMatch(b,name)).map(b=>b.sample)).size;
  const el=document.querySelector(`[data-count="${name}"]`);if(el)el.textContent=`${count}개`;
 }
 document.querySelector('[data-count="all"]').textContent=`${baseAllowed.size}개`;
 for(const name of ['greedy_short','parent_short']){const button=document.querySelector(`[data-preset="${name}"]`);button.disabled=!state.manifest.samples.some(s=>name==='greedy_short'?s.greedy_short:s.any_short);}
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
  return `<button class="case-card ${active?'selected':''}" data-sample="${escapeHTML(s.key)}" data-candidate="${b.candidate}" aria-pressed="${active}"><span class="case-header"><span class="case-id">${escapeHTML(s.key)}</span>${active?'<span class="case-selected-label">선택됨</span>':''}</span><strong>${escapeHTML(s.query)}</strong><span class="case-truth">정답 <b>${escapeHTML(pairText(s.gt))}</b></span><span class="case-predictions"><span class="before-text">전 ${escapeHTML(pairText(b.parent_pair))}</span><span>→</span><span class="after-text">후 ${escapeHTML(pairText(b.pair))}</span></span><span class="case-change">${b.roles.length?`${escapeHTML(roleText(b.roles))} <code>${escapeHTML(b.before)}</code> → <code>${escapeHTML(b.after)}</code>`:'추가 분기 없음'}</span><span class="case-score">정답과 겹침 ${score(b.parent_iou)} → <b>${score(b.iou)}</b> ${delta(b.parent_delta)}</span>${b.parent!==1?`<span class="case-origin">최초 예측 ${escapeHTML(pairText(s.greedy_pair))} · 위 ‘전’은 탐색 중 후보</span>`:''}</button>`;
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
  $('sampleSelect').value=key;$('sampleId').textContent=data.sample.key;$('query').textContent=data.sample.query;
  $('sampleMeta').textContent=`${dsName(data.sample)} · 영상 ${seconds(data.sample.duration)}초 · 저장된 예측 ${data.candidates.length}개`;
  if(changed){state.segment=null;$('video').pause();$('video').src=data.video;$('videoStatus').textContent='';$('videoDetails').open=false;$('rawAnswers').open=false;}
  const listIndex=state.grouped.findIndex(entry=>entry.sample===key);if(listIndex>=0)state.page=Math.floor(listIndex/PAGE_SIZE);
  showCandidate(candidate||defaultCandidate(key));$('loadStatus').textContent='';
  if(scroll&&window.innerWidth<1000)$('samplePanel').scrollIntoView({behavior:'smooth',block:'start'});
 }catch(error){$('loadStatus').textContent=error.message;}
}
function answerBlock(title,candidate,position){
 const tokens=candidate.tokens.map((token,index)=>token.startsWith('<|')?`<span class="faded">${escapeHTML(token)}</span>`:index===position?`<mark class="changed">${escapeHTML(token)}</mark>`:escapeHTML(token)).join('');
 return `<div class="answer"><b>${escapeHTML(title)} · ${pairText(candidate.pair)} · 겹침 ${score(candidate.iou)}</b><pre>${tokens}</pre></div>`;
}
function resultCard(label,pair,kind,candidate=null){
 return `<article class="result-card ${kind}"><span class="result-label">${label}</span><strong>${escapeHTML(pairText(pair))}</strong><span class="result-score">${kind==='truth'?'데이터셋의 정답 구간':`정답과 겹침 <b>${score(candidate.iou)}</b>`}</span><button data-overview-play="${kind}">이 구간 재생 ↗</button></article>`;
}
function showCandidate(index){
 const data=state.current;if(!data)return;const c=data.candidates.find(x=>x.index===index)||data.candidates[0];state.candidate=c.index;
 const b=c.branch,parent=b?data.candidates[b.parent-1]:c,greedy=data.candidates[0];
 $('resultOverview').innerHTML=resultCard('정답',data.sample.gt,'truth')+resultCard('분기 전 예측',parent.pair,'before',parent)+resultCard('분기 후 예측',c.pair,'after',c);
 const pairs={truth:data.sample.gt,before:parent.pair,after:c.pair};$('resultOverview').querySelectorAll('[data-overview-play]').forEach(button=>button.onclick=()=>playSpan(pairs[button.dataset.overviewPlay]));
 if(b){
  const verb=b.original_token==='.'&&/^\d+$/.test(b.forced_token)?'소수점을 찍는 대신 숫자를 이어 썼습니다.':`${escapeHTML(roleText(b.roles))}에서 다른 토큰을 골랐습니다.`;
  $('branchDetail').innerHTML=`<div class="branch-story-top"><span class="story-label">무엇을 바꿨나요?</span><div class="token-swap"><code class="before-token">${escapeHTML(b.original_token)}</code><span>→</span><code class="after-token">${escapeHTML(b.forced_token)}</code><span>${escapeHTML(roleText(b.roles))}</span></div></div><p>${verb} 그 뒤는 모델이 끝까지 이어서 생성했습니다.</p><p class="story-effect">정답과 겹침 <b>${score(parent.iou)} → ${score(c.iou)}</b> ${delta(c.iou-parent.iou)}</p>`;
 }else $('branchDetail').innerHTML='<b>최초 응답입니다.</b><p>다른 토큰을 강제하지 않았습니다. 다른 변경을 선택하면 분기 전후를 비교할 수 있습니다.</p>';
 $('originContext').innerHTML=parent.index!==1?`<div class="origin-context"><b>이 분기의 ‘전’은 최초 응답이 아닙니다.</b> 처음에는 ${escapeHTML(pairText(greedy.pair))}를 예측했고, 탐색 중 얻은 ${escapeHTML(pairText(parent.pair))}에서 다시 분기했습니다. 최초 응답과 비교한 겹침 변화는 ${delta(c.iou-greedy.iou)}입니다. <button id="showOriginal">최초 응답 보기</button> <button id="parentJump">분기 전 후보 보기</button></div>`:'';
 if($('showOriginal'))$('showOriginal').onclick=()=>showCandidate(1);if($('parentJump'))$('parentJump').onclick=()=>showCandidate(parent.index);
 $('branchSelect').innerHTML=data.candidates.map(row=>{const rb=row.branch,p=rb?data.candidates[rb.parent-1]:row;return `<option value="${row.index}">${row.index===1?'최초 응답':`변경 ${row.index-1} · ${roleText(rb.roles)} · ${rb.original_token}→${rb.forced_token}`} · ${pairText(p.pair)} → ${pairText(row.pair)} · 겹침 ${score(row.iou)}</option>`;}).join('');$('branchSelect').value=String(c.index);
 $('answerComparison').innerHTML=answerBlock('분기 직전 응답',parent,b?.position)+answerBlock('분기 후 응답',c,b?.position)+(parent.index!==1?answerBlock('최초 응답',greedy,null):'');
 renderSelectionAnalysis();renderTimeline(c,parent);renderCatalog();updateHash();
 if(window.ROLLOUT_DISTRIBUTIONS)window.ROLLOUT_DISTRIBUTIONS.render(data,c);
}
function renderSelectionAnalysis(){
 const data=state.current;if(!data)return;const k=$('budget').value,choice=data.candidates[data.choices[k][$('method').value]-1],oracle=data.candidates[data.choices[k].oracle-1],current=data.candidates[state.candidate-1];
 $('choiceSummary').innerHTML=`<div class="selection-results"><div><b>선택기가 고른 답</b><strong>${escapeHTML(pairText(choice.pair))}</strong><span>겹침 ${score(choice.iou)} · 후보 ${choice.index}</span></div><div><b>정답을 보고 고른 최선의 답</b><strong>${escapeHTML(pairText(oracle.pair))}</strong><span>겹침 ${score(oracle.iou)} · 후보 ${oracle.index}</span></div></div><p class="explain">지금 확인 중인 분기 후 예측은 ${pairText(current.pair)}입니다. ${current.index===choice.index?'선택기가 이 후보를 골랐습니다.':'선택기는 이 후보를 고르지 않았습니다.'}${current.index>Number(k)?' 현재 후보는 선택기에 준 후보 수 범위 밖입니다.':''}</p>`;
 $('candidates').innerHTML=data.candidates.filter(row=>row.index<=Number(k)).map(row=>{const br=row.branch,tags=[];if(row.index===choice.index)tags.push('선택된 답');if(row.index===oracle.index)tags.push('GT 최선');if(row.index===1)tags.push('최초 응답');return `<tr data-candidate="${row.index}" class="${row.index===state.candidate?'selected':''}"><td>${row.index}${br?` / ${br.parent}`:''}</td><td>${escapeHTML(br?roleText(br.roles):'최초 응답')}</td><td>${br?`${escapeHTML(br.original_token)} → ${escapeHTML(br.forced_token)}`:'—'}</td><td>${escapeHTML(pairText(row.pair))}</td><td>${score(row.iou)}</td><td>${delta(row.iou-data.candidates[0].iou)}</td><td>${br?delta(br.parent_delta):'—'}</td><td>${tags.map(t=>`<span class="tag">${t}</span>`).join('')}</td></tr>`;}).join('');
 $('candidates').querySelectorAll('[data-candidate]').forEach(row=>row.onclick=()=>showCandidate(Number(row.dataset.candidate)));
}
function renderTimeline(c,parent){
 const s=state.current.sample,tracks=[['정답',s.gt,'#277c68'],['분기 전 예측',parent.pair,'#2678b5'],['분기 후 예측',c.pair,'#d16b32']];
 if(parent.index!==1)tracks.push(['최초 응답',state.current.candidates[0].pair,'#8c9a9c']);
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
 if(!state.current)return;const params=new URLSearchParams({preset:state.preset,sample:state.current.sample.key,candidate:state.candidate,budget:$('budget').value,method:$('method').value,flag:state.flag});
 const old=new URLSearchParams(location.hash.slice(1));if(old.get('panel'))params.set('panel',old.get('panel'));
 for(const id of ['search','dataset','role','comparison','outcome','minimum','sort'])if($(id).value)params.set(id,$(id).value);
 history.replaceState(null,'',`${location.pathname}${location.search}#${params}`);
}
async function restoreHash(){
 const hash=new URLSearchParams(location.hash.slice(1));$('search').value='';$('dataset').value='';$('budget').value='32';$('method').value='probability_weighted_medoid';$('sort').value='sample';applyPreset(hash.get('preset')||'all',false);
 for(const id of ['search','dataset','role','comparison','outcome','minimum','sort','budget','method'])if(hash.has(id))$(id).value=hash.get(id);
 if(hash.has('flag'))state.flag=hash.get('flag');filterBranches();
 const first=hash.get('sample')||state.grouped[0]?.sample||state.manifest.samples[0].key;
 await selectCase(state.map.has(first)?first:state.manifest.samples[0].key,Number(hash.get('candidate'))||null);
}
async function init(){
 const response=await fetch(`${ASSETS}/manifest.json`);if(!response.ok)throw Error('사례 목록을 불러오지 못했습니다.');state.manifest=await response.json();state.map=new Map(state.manifest.samples.map(s=>[s.key,s]));
 $('summary').textContent=`${state.manifest.summary.samples}개 샘플`;
 $('role').innerHTML='<option value="">모든 위치</option>'+Object.entries(state.manifest.roles).map(([code,label])=>`<option value="${escapeHTML(code)}">${escapeHTML(label)}</option>`).join('');
 document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>applyPreset(b.dataset.preset));
 for(const id of ['search','dataset','role','comparison','outcome','minimum','sort'])$(id).addEventListener(['search','minimum'].includes(id)?'input':'change',()=>{filterBranches();updateHash();});
 $('previousPage').onclick=()=>{state.page--;renderCatalog();};$('nextPage').onclick=()=>{state.page++;renderCatalog();};
 $('sampleSelect').onchange=()=>selectCase($('sampleSelect').value,defaultCandidate($('sampleSelect').value));$('branchSelect').onchange=()=>showCandidate(Number($('branchSelect').value));
 for(const id of ['budget','method'])$(id).onchange=()=>{renderSelectionAnalysis();updateHash();};
 function adjacent(direction){const i=state.grouped.findIndex(x=>x.sample===state.current?.sample.key),entry=state.grouped[i+direction];if(entry)selectCase(entry.sample,entry.candidate);}
 $('previousSample').onclick=()=>adjacent(-1);$('nextSample').onclick=()=>adjacent(1);
 $('video').ontimeupdate=()=>{if(state.segment&&$('video').currentTime>=state.segment[1]){$('video').pause();state.segment=null;}};
 $('video').onerror=()=>{$('videoStatus').textContent='영상을 불러오지 못했습니다.';};
 window.addEventListener('hashchange',restoreHash);
 await restoreHash();
}
window.ROLLOUT_VIEWER={state,selectCase,showCandidate,applyPreset};
init().catch(error=>{$('summary').textContent=error.message;$('loadStatus').textContent=error.message;});
