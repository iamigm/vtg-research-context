'use strict';
const $=id=>document.getElementById(id),ASSETS=window.ROLLOUT_ASSETS;
const DOT='end.seconds.decimal_separator',PAGE_SIZE=40;
const state={manifest:null,map:new Map(),filtered:[],page:0,preset:'improved',flag:'',current:null,candidate:1,serial:0,segment:null,cache:new Map()};
const escapeHTML=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=3)=>Number.isFinite(x)?x.toFixed(n):'—';
const pairText=p=>p?`[${fmt(p[0],1)}, ${fmt(p[1],1)}]`:'유효 구간 없음';
const delta=x=>`<span class="${x>1e-9?'positive':x<-1e-9?'negative':'zero'}">${x>=0?'+':''}${fmt(x,4)}</span>`;
const roleText=roles=>roles.map(r=>state.manifest.roles[r]||r).join(' / ');

function sampleMatch(s){
 const q=$('search').value.trim().toLowerCase(),ds=$('dataset').value;
 if(ds&&s.dataset!==ds)return false;
 if(q&&!`${s.key} ${s.query}`.toLowerCase().includes(q))return false;
 if(state.flag==='dot'&&!s.dot_count)return false;
 if(state.flag==='greedy_short'&&!s.greedy_short)return false;
 if(state.flag==='parent_short'&&!s.any_short)return false;
 return true;
}
function applyPreset(name,update=true){
 state.preset=name;state.flag='';$('search').value='';$('dataset').value='';$('role').value='';
 $('comparison').value='greedy_delta';$('outcome').value='improved';$('minimum').value='0.01';$('sort').value='desc';
 if(name==='parent_improved')$('comparison').value='parent_delta';
 if(['dot','greedy_short','parent_short'].includes(name)){
  state.flag=name;$('role').value=DOT;$('outcome').value='all';$('minimum').value='0';
  if(name==='dot')$('comparison').value='parent_delta';
 }
 if(name==='all'){$('outcome').value='all';$('minimum').value='0';}
 document.querySelectorAll('[data-preset]').forEach(b=>b.classList.toggle('active',b.dataset.preset===name));
 if(update){filterBranches();const keys=[...$('sampleSelect').options].map(o=>o.value);if(keys.length)selectCase(keys[0]);updateHash();}
}
function filterBranches(){
 const samples=state.manifest.samples.filter(sampleMatch),allowed=new Set(samples.map(s=>s.key));
 const role=$('role').value,comparison=$('comparison').value,outcome=$('outcome').value,min=Math.max(0,Number($('minimum').value)||0);
 state.filtered=state.manifest.branches.filter(b=>{
  if(!allowed.has(b.sample)||role&&!b.roles.includes(role))return false;
  if(state.flag==='parent_short'&&!b.short_parent)return false;
  const d=b[comparison];
  if(outcome==='improved'&&!(d>1e-9&&d>=min))return false;
  if(outcome==='worsened'&&!(d<-1e-9&&-d>=min))return false;
  if(outcome==='same'&&Math.abs(d)>1e-9)return false;
  if(outcome==='all'&&Math.abs(d)<min)return false;
  return true;
 });
 const sort=$('sort').value;
 state.filtered.sort((a,b)=>sort==='sample'?a.sample.localeCompare(b.sample)||a.candidate-b.candidate:
  (sort==='asc'?a[comparison]-b[comparison]:b[comparison]-a[comparison])||a.sample.localeCompare(b.sample)||a.candidate-b.candidate);
 const count=new Set(state.filtered.map(b=>b.sample)).size;
 $('scope').textContent=`${state.filtered.length.toLocaleString()}개 분기 · ${count}개 질문 · 비교: ${comparison==='greedy_delta'?'최초 greedy 응답':'분기 직전 부모 후보'} · 40개씩 표시`;
 const current=$('sampleSelect').value;
 $('sampleSelect').innerHTML=samples.map(s=>`<option value="${escapeHTML(s.key)}">${escapeHTML(s.key)} · GT ${escapeHTML(pairText(s.gt))}</option>`).join('');
 if(samples.some(s=>s.key===current))$('sampleSelect').value=current;
 state.page=0;renderCatalog();
}
function renderCatalog(){
 const pages=Math.max(1,Math.ceil(state.filtered.length/PAGE_SIZE));state.page=Math.max(0,Math.min(state.page,pages-1));
 const rows=state.filtered.slice(state.page*PAGE_SIZE,(state.page+1)*PAGE_SIZE);
 $('branches').innerHTML=rows.length?rows.map(b=>{
  const s=state.map.get(b.sample),active=state.current?.sample.key===b.sample&&state.candidate===b.candidate;
  return `<tr data-sample="${escapeHTML(b.sample)}" data-candidate="${b.candidate}" class="${active?'selected':''}"><td><span class="mono">${escapeHTML(b.sample)}</span><br><span class="small">${escapeHTML(s.query)}</span></td><td>#${b.candidate}<br><span class="small">부모 #${b.parent}</span></td><td>${escapeHTML(roleText(b.roles))}<br><code>${escapeHTML(b.before)}</code> → <code>${escapeHTML(b.after)}</code></td><td>${fmt(b.iou,4)}</td><td>${delta(b.greedy_delta)}</td><td>${delta(b.parent_delta)}</td><td><button>분기 보기</button></td></tr>`;
 }).join(''):'<tr><td colspan="7">조건에 맞는 분기가 없습니다. 최소 변화량이나 검색 조건을 조정하세요.</td></tr>';
 $('branches').querySelectorAll('[data-sample]').forEach(row=>row.onclick=()=>selectCase(row.dataset.sample,Number(row.dataset.candidate),true));
 $('pageInfo').textContent=`${state.page+1} / ${pages}`;$('previousPage').disabled=state.page===0;$('nextPage').disabled=state.page===pages-1;
}
async function selectCase(key,candidate=null,scroll=false){
 const serial=++state.serial;$('loadStatus').textContent='샘플을 불러오는 중…';
 try{
  let data=state.cache.get(key);
  if(!data){const response=await fetch(`${ASSETS}/samples/${encodeURIComponent(key)}.json`);if(!response.ok)throw Error(`샘플 불러오기 실패: ${response.status}`);data=await response.json();state.cache.set(key,data);if(state.cache.size>6)state.cache.delete(state.cache.keys().next().value);}
  if(serial!==state.serial)return;
  const changed=state.current?.sample.key!==key;state.current=data;
  if(candidate&&candidate>Number($('budget').value))$('budget').value='32';
  $('sampleSelect').value=key;$('sampleId').textContent=key;$('query').textContent=data.sample.query;
  $('sampleMeta').textContent=`${data.sample.dataset} · 영상 ${fmt(data.sample.duration,1)}초 · GT ${pairText(data.sample.gt)}초 · 완성 후보 ${data.candidates.length}개`;
  if(changed){state.segment=null;$('video').pause();$('video').src=data.video;$('videoStatus').textContent='';}
  const choice=data.choices[$('budget').value][$('method').value];
  showCandidate(candidate||choice);$('loadStatus').textContent='';
  if(scroll)$('samplePanel').scrollIntoView({behavior:'smooth',block:'start'});
 }catch(error){$('loadStatus').textContent=error.message;}
}
function answerBlock(title,candidate,position){
 const tokens=candidate.tokens.map((token,index)=>{
  if(token.startsWith('<|'))return `<span class="faded">${escapeHTML(token)}</span>`;
  return index===position?`<mark class="changed">${escapeHTML(token)}</mark>`:escapeHTML(token);
 }).join('');
 return `<div class="answer"><b>${escapeHTML(title)} · ${pairText(candidate.pair)} · IoU ${fmt(candidate.iou,4)}</b><pre>${tokens}</pre></div>`;
}
function showCandidate(index){
 const data=state.current;if(!data)return;
 const pool=data.candidates.filter(c=>c.index<=Number($('budget').value));
 const c=pool.find(c=>c.index===index)||pool[0];state.candidate=c.index;
 const b=c.branch,parent=b?data.candidates[b.parent-1]:c,greedy=data.candidates[0];
 const chosen=data.choices[$('budget').value][$('method').value],oracle=data.choices[$('budget').value].oracle;
 $('choiceSummary').innerHTML=`지정한 선택 기준의 답: <b>#${chosen}</b> · GT로 고른 상한: <b>#${oracle}</b> · 현재 확인 중: <b>#${c.index}</b>`;
 if(b){
  $('branchDetail').innerHTML=`<b>${escapeHTML(roleText(b.roles))}</b><br>토큰 <code>${escapeHTML(b.original_token)}</code> → <code>${escapeHTML(b.forced_token)}</code> · 부모 #${b.parent} → 후보 #${c.index}<br>부모 대비 ${delta(b.parent_delta)} · 최초 응답 대비 ${delta(b.greedy_delta)}<br><span class="small">강제한 토큰의 조건부 확률 ${(100*b.probability).toFixed(3)}%. 정답 확률이 아닙니다. 전체 응답 로그 확률 ${fmt(c.log_probability,3)}.</span><br><button id="parentJump">부모 후보 #${b.parent} 확인</button>`;
  $('parentJump').onclick=()=>showCandidate(b.parent);
 }else $('branchDetail').textContent='최초 greedy 응답입니다. 다른 토큰을 강제하지 않았습니다.';
 $('answerComparison').innerHTML=answerBlock('분기 직전 부모',parent,b?.position)+answerBlock('분기 후 후보',c,b?.position)+(parent.index!==1?answerBlock('최초 greedy',greedy,null):'');
 $('candidates').innerHTML=pool.map(row=>{
  const br=row.branch,tags=[];if(row.index===chosen)tags.push('선택된 답');if(row.index===oracle)tags.push('oracle');if(row.index===1)tags.push('최초 응답');
  return `<tr data-candidate="${row.index}" class="${row.index===c.index?'selected':''}"><td>#${row.index}${br?` / #${br.parent}`:''}</td><td>${escapeHTML(br?roleText(br.roles):'최초 응답')}</td><td>${br?`<code>${escapeHTML(br.original_token)}</code> → <code>${escapeHTML(br.forced_token)}</code>`:'—'}</td><td>${escapeHTML(pairText(row.pair))}</td><td>${fmt(row.iou,4)}</td><td>${delta(row.iou-greedy.iou)}</td><td>${br?delta(br.parent_delta):'—'}</td><td>${tags.map(tag=>`<span class="tag">${tag}</span>`).join('')}</td></tr>`;
 }).join('');
 $('candidates').querySelectorAll('[data-candidate]').forEach(row=>row.onclick=()=>showCandidate(Number(row.dataset.candidate)));
 renderTimeline(c,parent,data.candidates[chosen-1],data.candidates[oracle-1]);renderCatalog();updateHash();
 if(window.ROLLOUT_DISTRIBUTIONS)window.ROLLOUT_DISTRIBUTIONS.render(data,c);
}
function renderTimeline(c,parent,chosen,oracle){
 const s=state.current.sample,greedy=state.current.candidates[0];
 const tracks=[['GT',s.gt,'#3e7762'],['최초 greedy',greedy.pair,'#81969f'],[`부모 #${parent.index}`,parent.pair,'#c1924f'],[`분기 후보 #${c.index}`,c.pair,'#307da4'],[`선택된 답 #${chosen.index}`,chosen.pair,'#7764a0'],[`Oracle #${oracle.index}`,oracle.pair,'#7b984f']];
 $('timeline').innerHTML=tracks.map(([label,pair,color],index)=>{
  let start=pair?Math.max(0,Math.min(1,pair[0]/s.duration)):0,end=pair?Math.max(0,Math.min(1,pair[1]/s.duration)):0;
  return `<div class="timeline-row"><div class="timeline-label"><span>${escapeHTML(label)} ${escapeHTML(pairText(pair))}</span><button data-play="${index}" ${pair?'':'disabled'}>구간 재생</button></div><div class="track"><div class="bar" style="left:${start*100}%;width:${Math.max(0,end-start)*100}%;--bar-color:${color}"></div></div></div>`;
 }).join('');
 $('timeline').querySelectorAll('[data-play]').forEach(button=>button.onclick=()=>playSpan(tracks[Number(button.dataset.play)][1]));
}
function playSpan(pair){
 if(!pair)return;const video=$('video'),duration=Number.isFinite(video.duration)?video.duration:state.current.sample.duration;
 const a=Math.max(0,Math.min(duration,pair[0])),b=Math.max(0,Math.min(duration,pair[1]));
 if(b<=a){$('videoStatus').textContent='이 구간은 재생 가능한 영상 범위 밖입니다.';return;}
 state.segment=[a,b];video.currentTime=a;$('videoStatus').textContent=`${fmt(a,1)}∼${fmt(b,1)}초 재생`;
 video.play().catch(e=>$('videoStatus').textContent=e.message);
}
function updateHash(){
 if(!state.current)return;const params=new URLSearchParams({preset:state.preset,sample:state.current.sample.key,candidate:state.candidate,budget:$('budget').value,method:$('method').value,flag:state.flag});
 const previousHash=new URLSearchParams(location.hash.slice(1));if(previousHash.get('panel'))params.set('panel',previousHash.get('panel'));
 for(const id of ['search','dataset','role','comparison','outcome','minimum','sort'])if($(id).value)params.set(id,$(id).value);
 history.replaceState(null,'',`${location.pathname}${location.search}#${params}`);
}
async function init(){
 const response=await fetch(`${ASSETS}/manifest.json`);if(!response.ok)throw Error('분기 목록을 불러오지 못했습니다.');
 state.manifest=await response.json();state.map=new Map(state.manifest.samples.map(s=>[s.key,s]));
 const summary=state.manifest.summary;
 $('summary').textContent=`${summary.samples}개 질문 · ${summary.branches.toLocaleString()}개 분기 · 종료 소수점 ${summary.dot_samples}질문 / ${summary.dot_branches}분기`;
 $('role').innerHTML='<option value="">모든 위치</option>'+Object.entries(state.manifest.roles).map(([code,label])=>`<option value="${escapeHTML(code)}">${escapeHTML(label)}</option>`).join('');
 const hash=new URLSearchParams(location.hash.slice(1));applyPreset(hash.get('preset')||'improved',false);
 for(const id of ['search','dataset','role','comparison','outcome','minimum','sort','budget','method'])if(hash.has(id))$(id).value=hash.get(id);
 if(hash.has('flag'))state.flag=hash.get('flag');filterBranches();
 document.querySelectorAll('[data-preset]').forEach(button=>button.onclick=()=>applyPreset(button.dataset.preset));
 for(const id of ['search','dataset','role','comparison','outcome','minimum','sort'])$(id).addEventListener(['search','minimum'].includes(id)?'input':'change',()=>{filterBranches();updateHash();});
 $('previousPage').onclick=()=>{state.page--;renderCatalog();};$('nextPage').onclick=()=>{state.page++;renderCatalog();};
 $('sampleSelect').onchange=()=>selectCase($('sampleSelect').value);
 for(const id of ['budget','method'])$(id).onchange=()=>{if(state.current)showCandidate(state.current.choices[$('budget').value][$('method').value]);};
 $('video').ontimeupdate=()=>{if(state.segment&&$('video').currentTime>=state.segment[1]){$('video').pause();state.segment=null;}};
 $('video').onerror=()=>{$('videoStatus').textContent='영상 로드에 실패했습니다. 서버 연결을 확인하세요.';};
 const first=hash.get('sample')||state.filtered[0]?.sample||state.manifest.samples[0].key;
 await selectCase(state.map.has(first)?first:state.manifest.samples[0].key,Number(hash.get('candidate'))||null);
}
window.ROLLOUT_VIEWER={state,selectCase,showCandidate,applyPreset};
init().catch(error=>{$('summary').textContent=error.message;$('loadStatus').textContent=error.message;});
