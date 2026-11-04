'use strict';
(()=>{
const config=window.RECONSIDERATION_CONFIG,$=id=>document.getElementById(id);
const groups=[['agent','agent','Action → observation → final regeneration'],['verifier','verifier','후보별 평가 → 점수 비교 → 선택 → final regeneration'],['tts','test-time scaling','후보 대진 → 양방향 비교 → 승자 → final regeneration'],['direct','direct regeneration','추가 평가 없이 같은 ledger로 재응답하는 대조 조건']];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const span=p=>p?p.map(x=>Number(x.toFixed(2))).join(' – ')+'초':'유효한 구간 없음';
const pair=a=>a?.parsed?.positive&&a.stopped_on_eos?[a.parsed.start,a.parsed.end]:null;
const iou=(a,b)=>a&&b?Math.max(0,Math.min(a[1],b[1])-Math.max(a[0],b[0]))/Math.max(1e-12,Math.max(a[1],b[1])-Math.min(a[0],b[0])):0;
const pct=x=>(100*x).toFixed(1)+'%',signed=x=>`<span class="${x>1e-9?'positive':x<-1e-9?'negative':'zero'}">${x>=0?'+':''}${x.toFixed(3)}</span>`;
const params=new URLSearchParams(location.hash.slice(1));
const state={data:null,record:null,key:null,serial:0,cache:new Map(),variant:params.get('review')||(/^(agent|verifier|tts|direct)_/.test(params.get('method')||'')?params.get('method'):null),approach:params.get('approach')||'agent',choices:new Map(),axis:params.get('axis')||'all'};
function pool(v,cs=state.data.candidates){return cs.slice(0,v.pool);}
function unique(cs){const seen=new Set();return cs.filter(c=>{if(!c.pair||c.stopped_on_eos===false||c.pair[1]<=c.pair[0])return false;const key=JSON.stringify(c.pair);if(seen.has(key))return false;seen.add(key);return true;});}
function diverse(cs,n){
 let available=unique(cs);if(!available.length)return [];
 const first=available.find(c=>c.index===1)||available.reduce((a,b)=>b.log_probability>a.log_probability?b:a),out=[first];available=available.filter(c=>c.index!==first.index);
 while(available.length&&out.length<n){
  let best=null,bestDistance=-Infinity;
  for(const c of available){const d=1-Math.max(...out.map(x=>iou(c.pair,x.pair)));if(!best||d>bestDistance||d===bestDistance&&(c.log_probability>best.log_probability||c.log_probability===best.log_probability&&c.index<best.index)){best=c;bestDistance=d;}}
  out.push(best);available=available.filter(c=>c.index!==best.index);
 }
 return out;
}
function bracket(v,cs=state.data.candidates){
 const seeds=diverse(pool(v,cs),v.amount).map(c=>c.index);let active=[...seeds],used=0;const rounds=[];
 while(active.length>1){const matches=[],next=[];
  for(let i=0;i<active.length;i+=2){const a=active[i],b=active[i+1];if(b===undefined){matches.push({bye:true,winner:a});next.push(a);continue;}
   const r=v.reviews[used++];if(!r||r.a!==a||r.b!==b)throw Error('저장 대진과 후보 순서가 일치하지 않습니다.');matches.push(r);next.push(r.winner);
  }rounds.push(matches);active=next;
 }
 if(used!==v.reviews.length||(active[0]||1)!==v.selected_candidate)throw Error('저장된 최종 승자와 대진이 일치하지 않습니다.');
 return {seeds,rounds,winner:active[0]||1};
}
const ledgerIDs=v=>[...v.ledger.matchAll(/^#(\d+)\s+\[/gm)].map(m=>Number(m[1]));
function sourceURL(id){const p=new URLSearchParams(location.hash.slice(1));if(id)p.set('candidate',id);return config.sourceViewer+'#'+p;}
function play(p){return `<button class="process-play" data-play-pair="${esc(JSON.stringify(p))}" ${p?'':'disabled'} aria-label="${esc(span(p))} 재생">▶</button>`;}
function candidate(id){const c=state.data.candidates[id-1];return `<span class="trajectory-chip" data-trajectory="${id}"><a href="${esc(sourceURL(id))}" target="_parent">#${id}</a><span>${esc(span(c?.pair))}</span>${play(c?.pair)}</span>`;}
function inputNote(v,selection){const p=pool(v),n=unique(p).length;return `<p class="process-input">기존 trajectory ${p.length}개 중 유효한 고유 구간 ${n}개 · GT 없이 farthest-first로 ${esc(selection)} · <a href="${esc(sourceURL())}" target="_parent">trajectory 원본 ↗</a></p>`;}
function calls(v){return v.model_calls.reduce((m,c)=>(m[c.kind]=(m[c.kind]||0)+1,m),{});}
function costs(v){return `<div class="process-cost"><span>의사결정 <b>${v.cost_decision.model_evaluations}</b> model evaluations</span><span>model seconds <b>${v.cost_decision.model_seconds.toFixed(3)}s</b></span><span>기존 pool 생성 <b>${v.proposal_cost.original_generation_seconds.toFixed(3)}s</b></span></div>`;}
function outcome(v,heading='최종 결론'){
 const p=pair(v.answer),s=iou(p,state.data.sample.gt),greedy=state.data.candidates[0],selected=Number.isInteger(v.selected_candidate)?state.data.candidates[v.selected_candidate-1]:null;
 return `<section class="process-outcome" data-outcome="${esc(v.name)}"><div class="outcome-heading"><h4>${heading}</h4><span>greedy 대비 ΔIoU ${signed(s-greedy.iou)}</span></div>${selected?`<div class="selection-to-final"><span>선택 ${candidate(selected.index)}</span><span aria-hidden="true">→</span><span>final regeneration</span><span class="small">선택 후보 IoU ${selected.iou.toFixed(3)} · 재생성 변화 ${signed(s-selected.iou)}</span></div>`:''}<p class="final-answer">${esc(v.answer.text)}</p><div class="final-metrics"><strong>${span(p)}</strong>${play(p)}<span>IoU <b data-final-iou>${s.toFixed(3)}</b></span></div>${costs(v)}</section>`;
}
function variantsFor(method){return Object.values(state.record.variants).filter(v=>v.method===method).sort((a,b)=>a.pool-b.pool||a.amount-b.amount);}
function variantControls(){
 const items=variantsFor(state.approach);if(state.approach==='direct'){$('methodVariants').replaceChildren();return;}
 const html=items.map(v=>{const n=v.method==='tts'?diverse(pool(v),v.amount).length:v.reviews.length;const budget=v.method==='tts'?`select${v.amount} · 실제 ${n}개`:v.method==='agent'?`최대 ${v.amount} rounds · 실제 ${n}`:`최대 ${v.amount}개 평가 · 실제 ${n}`;
  return `<button data-variant="${v.name}" class="variant-option ${v.name===state.variant?'active':''}" aria-pressed="${v.name===state.variant}"><b>pool${v.pool} · ${budget}</b><span>final ${span(pair(v.answer))} · IoU ${iou(pair(v.answer),state.data.sample.gt).toFixed(3)} · ${v.cost_decision.model_evaluations} evaluations</span></button>`;
 }).join('');
 $('methodVariants').innerHTML=`<div class="variant-options ${state.approach==='tts'?'factorial-options':''}" aria-label="같은 접근 안의 예산 비교">${html}</div>`;
}
function agent(v){
 const c=calls(v),extra=(c.agent_plan||0)-v.reviews.length;
 return `<section class="agent-process"><h3>agent의 action–observation 이력</h3>${inputNote(v,`controller에게 최대8개 후보를 전달`)}<div class="agent-menu"><b>Controller 후보 메뉴 · ${ledgerIDs(v).length}개</b><div class="trajectory-list">${ledgerIDs(v).map(candidate).join('')}</div></div><ol class="agent-timeline">${v.reviews.map(r=>`<li class="agent-round" data-round="${r.round}"><div class="agent-action"><span class="round-number">${r.round}</span><b>${r.action==='event'?'event comparison':'boundary inspection'}</b>${r.fallback?'<span class="fallback-label">fallback 적용</span>':''}</div><div class="agent-targets">${candidate(r.a)}${r.action==='event'?`<span>와 비교</span>${candidate(r.b)}`:''}</div><blockquote><small>Visual inspector observation · 원문</small><p>${esc(r.observation)}</p></blockquote><details class="controller-plan"><summary>Controller action 원문${r.fallback?' · 적용 action과 비교':''}</summary><pre>${esc(r.plan)}</pre></details></li>`).join('')}</ol><div class="agent-stop"><b>${v.reviews.length}개 observation을 모아 final regeneration</b><span>controller ${c.agent_plan||0}회 · inspector ${c.visual_inspector||0}회 · final ${c.final_regeneration||0}회${extra>0?` · inspection으로 이어지지 않은 controller 호출 ${extra}회`:''}</span><p>별도의 selected candidate 기록은 없습니다. 마지막 inspection 대상을 최종 선택으로 간주하지 않습니다.</p></div>${outcome(v)}</section>`;
}
function meter(p){return `<span class="judge-meter"><i style="width:${p*100}%"></i><b>${pct(p)}</b></span>`;}
function verifier(v){
 const ranked=[...v.reviews].sort((a,b)=>b.score-a.score||a.candidate-b.candidate);
 return `<section class="verifier-process"><h3>독립 평가로 후보를 고르는 과정</h3>${inputNote(v,`최대 ${v.amount}개 평가 후보를 수집`)}<div class="verifier-rule"><b>선택 점수 = P(event Yes) × P(boundary Yes)</b><span>두 값은 각각 Yes/No 안에서 정규화한 모델 확률입니다. GT IoU는 선택에 쓰지 않은 사후 평가값입니다.</span></div><div class="table-scroll verifier-table"><table><thead><tr><th>평가 후보</th><th>event Yes</th><th>boundary Yes</th><th>선택 점수</th><th>GT IoU</th><th>판정</th></tr></thead><tbody>${ranked.map(r=>`<tr data-verifier-candidate="${r.candidate}" data-score="${r.score}" class="${r.candidate===v.selected_candidate?'verifier-selected':''}"><td>${candidate(r.candidate)}</td><td>${meter(r.event.conditional_probabilities[0])}</td><td>${meter(r.boundary.conditional_probabilities[0])}</td><td class="judge-score">${r.score.toFixed(4)}</td><td>${state.data.candidates[r.candidate-1].iou.toFixed(3)}</td><td>${r.candidate===v.selected_candidate?'<b>선택</b>':'탈락'}</td></tr>`).join('')}</tbody></table></div><p class="process-note">최대 점수 후보 #${v.selected_candidate}를 선택한 뒤, 평가 결과와 ledger를 보고 final answer를 다시 생성했습니다.</p>${outcome(v)}</section>`;
}
function tournament(v){
 const plan=bracket(v);
 const columns=plan.rounds.map((round,i)=>`<section class="match-column"><h4>Round ${i+1}</h4><div class="match-column-body">${round.map(r=>r.bye?`<article class="match-bye" data-bye="${r.winner}">${candidate(r.winner)}<span>bye · 비교 없이 진출</span></article>`:`<article class="match-card" data-winner="${r.winner}" data-pa="${r.p_a}"><div class="match-side ${r.winner===r.a?'match-winner':''}">${candidate(r.a)}<b>${pct(r.p_a)}</b></div><div class="match-side ${r.winner===r.b?'match-winner':''}">${candidate(r.b)}<b>${pct(1-r.p_a)}</b></div><p>P(#${r.a}): 정방향 ${pct(r.forward.conditional_probabilities[0])} · 역방향 ${pct(r.reverse.conditional_probabilities[1])}</p><strong>#${r.winner} 진출 →</strong></article>`).join('')}</div></section>`).join('');
 return `<section class="tts-process"><h3>Pairwise knockout 대진</h3>${inputNote(v,`select${v.amount} 목표에서 실제 ${plan.seeds.length}개 대진 후보를 수집`)}<div class="tournament-seeds"><b>출전 순서</b><div class="trajectory-list">${plan.seeds.map(candidate).join('')}</div></div><p class="process-note">P(a) = [P(A | A=a, B=b) + P(B | A=b, B=a)] / 2. 평균이 0.5 이상이면 a가 진출합니다. 각 대진에서 presentation order를 뒤집어 두 번 평가합니다.</p><div class="bracket-scroll"><div class="tournament-bracket" style="--rounds:${Math.max(1,plan.rounds.length)}">${columns||'<p class="single-entrant">저장된 대진이 없습니다. 기록된 선택 후보를 final regeneration에 전달했습니다.</p>'}</div></div><div class="tournament-winner">최종 승자 ${candidate(plan.winner)} <span>${v.reviews.length}개 대진 · pairwise 평가 ${v.reviews.length*2}회</span></div>${outcome(v)}</section>`;
}
function direct(){
 return `<section class="direct-process"><h3>추가 평가 없이 재응답</h3><p class="process-note">pool 크기만 바꾼 두 대조 조건입니다. 모델의 후보 평가나 winner selection 없이, 기존 ledger를 한 번 읽고 final answer를 생성했습니다.</p><div class="direct-comparison">${variantsFor('direct').map(v=>`<article class="direct-condition ${v.name===state.variant?'active':''}" data-direct="${v.name}"><h4>pool${v.pool}</h4><p class="small">기존 경로 ${pool(v).length}개 → ledger 후보 ${ledgerIDs(v).length}개 → final regeneration 1회</p><div class="trajectory-list">${ledgerIDs(v).map(candidate).join('')}</div>${outcome(v,`pool${v.pool}의 결론`)}<button class="direct-inspect" data-variant="${v.name}">pool${v.pool}의 전달 원문·사후 진단 선택</button></article>`).join('')}</div></section>`;
}
function diagnostics(v){
 const root=$('reconsiderationDiagnostics'),wasOpen=root.querySelector('details')?.open;
 root.innerHTML=`<details id="reviewDiagnostics" class="advanced"><summary><b>사후 token distribution 진단</b><span>검토 과정과 별도로 측정한 고정 probe의 review 전후 변화입니다.</span></summary><p class="process-note">원래 trajectory의 prefix를 고정한 counterfactual replay입니다. 실제 final regeneration의 경로로 해석하지 않습니다. log P가 클수록 해당 경로의 모델 확률이 높고, JS divergence는 0이면 같은 분포입니다.</p>${v.probes.map(p=>`<article class="diagnostic-probe"><h4>${p.axis} proxy · #${p.parent} → #${p.candidate}</h4><div class="table-scroll"><table><thead><tr><th>고정 경로</th><th>log P 전</th><th>log P 후</th></tr></thead><tbody>${Object.entries(p.paths).map(([name,x])=>`<tr><td>${name==='parent'?'부모':'분기'} #${x.candidate}</td><td>${x.before_log_probability.toFixed(4)}</td><td>${x.after_log_probability.toFixed(4)}</td></tr>`).join('')}</tbody></table></div><div class="diagnostic-nodes">${p.nodes.map(n=>`<div><b>${({root:'분기 직전',original_child:'부모 선택 직후',alternative_child:'분기 선택 직후'})[n.kind]}</b><p>JS divergence ${n.js_divergence.toFixed(6)}</p><table><thead><tr><th>token</th><th>전</th><th>후</th></tr></thead><tbody>${n.top.map(t=>`<tr><td><code>${esc(t.token)}</code></td><td>${pct(t.before)}</td><td>${pct(t.after)}</td></tr>`).join('')}</tbody></table><a href="${config.store}/${n.file}" download>전체 어휘 확률</a></div>`).join('')}</div></article>`).join('')||'<p>이 조건에는 저장된 probe가 없습니다.</p>'}</details>`;
 root.querySelector('details').open=!!wasOpen;
}
function save(){const p=new URLSearchParams(location.hash.slice(1));p.set('review',state.variant);p.set('approach',state.approach);p.delete('method');history.replaceState(null,'',location.pathname+location.search+'#'+p);window.dispatchEvent(new CustomEvent('rollout:context'));}
function paint(){
 if(!state.data||!state.record)return;
 try{
  if(!state.record.variants[state.variant])state.variant=variantsFor(state.approach)[0]?.name||Object.keys(state.record.variants)[0];
  const v=state.record.variants[state.variant];state.approach=v.method;state.choices.set(v.method,v.name);
  $('approachTabs').innerHTML=groups.filter(([id])=>variantsFor(id).length).map(([id,title,description])=>`<button data-approach="${id}" class="${id===state.approach?'active':''}" aria-pressed="${id===state.approach}" title="${description}"><b>${title}</b><span>${variantsFor(id).length}개 예산 조건</span></button>`).join('');
  const gt=state.data.sample.gt,g=state.data.candidates[0];
  $('caseContext').innerHTML=`<span>GT <b>${span(gt)}</b> ${play(gt)}</span><span>greedy <b>${span(g.pair)}</b> ${play(g.pair)}</span><span>greedy IoU <b>${g.iou.toFixed(3)}</b></span>`;
  $('sourceTrajectory').href=sourceURL();variantControls();
  $('processCanvas').innerHTML=({agent:()=>agent(v),verifier:()=>verifier(v),tts:()=>tournament(v),direct})[v.method]();
  $('workflowStatus').textContent=groups.find(([id])=>id===v.method)[2];
  $('ledgerSource').textContent=v.ledger;$('callSource').textContent=JSON.stringify(v.model_calls,null,2);diagnostics(v);bind();save();
 }catch(error){$('workflowStatus').textContent=error.message;}
}
function bind(){
 $('samplePanel').querySelectorAll('[data-variant]').forEach(b=>b.onclick=()=>{state.variant=b.dataset.variant;paint();});
 $('samplePanel').querySelectorAll('[data-approach]').forEach(b=>b.onclick=()=>{state.approach=b.dataset.approach;state.variant=state.choices.get(state.approach)||variantsFor(state.approach)[0].name;paint();});
 $('samplePanel').querySelectorAll('[data-play-pair]').forEach(b=>b.onclick=()=>window.ROLLOUT_VIEWER.playSpan(JSON.parse(b.dataset.playPair)));
}
async function onCandidate(event){
 state.data=event.detail.data;const key=state.data.sample.key;
 const s=state.data.sample;const dataset=({'activitynet-timelens':'ActivityNet','charades-timelens':'Charades','qvhighlights-timelens':'QVHighlights'})[s.dataset]||s.dataset;$('sampleMeta').textContent=`${dataset} · 영상 ${Number(s.duration.toFixed(2))}초 · 기존 trajectory ${state.data.candidates.length}개`;
 if(key===state.key&&state.record){paint();return;}
 const serial=++state.serial;state.key=key;state.record=state.cache.get(key)||null;$('processCanvas').replaceChildren();$('workflowStatus').textContent='검토 기록을 불러오는 중…';
 try{if(!state.record){const r=await fetch(`${config.store}/responses/${encodeURIComponent(key)}.json`);if(!r.ok)throw Error(`HTTP ${r.status}`);const record=await r.json();if(serial!==state.serial)return;state.record=record;state.cache.set(key,record);if(state.cache.size>6)state.cache.delete(state.cache.keys().next().value);}if(serial===state.serial)paint();}
 catch(error){if(serial===state.serial)$('workflowStatus').textContent='검토 기록을 불러오지 못했습니다: '+error.message;}
}
window.addEventListener('rollout:candidate',onCandidate);
window.addEventListener('hashchange',()=>{const p=new URLSearchParams(location.hash.slice(1));state.variant=p.get('review')||null;state.approach=p.get('approach')||'agent';});
window.RECONSIDERATION={state,diverse,bracket,ledgerIDs};
})();
