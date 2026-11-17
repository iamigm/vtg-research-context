/* One sample selection drives path, distribution, feasibility and supporting views. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id),cfg=window.TOKEN_VIEWER_CONFIG;
  const fmt=(v,n=3)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(n);
  const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  const absolute=p=>new URL(p,location.href).href;
  const cache=new Map(),blueCache=new Map(),savedCache=new Map();let savedLookup={samples:{}},savedLookupURL=null,source=cfg.default_source||'original',pathId='1',reference='parent',pathBundle=null,activeCandidate=null,baselineTrace=null,baselineRaw=null,loadedModel=null,referenceRaw=null;let catalog={},filtered=[],current=null,trace=null,raw=null,integrated=null,blue=[],step=null,selectedPresets=new Set(),selectedRegions=new Set(),page=0,serial=0,segment=null;
  const pageSize=50,originalAvailable=cfg.original_available!==false;
  const probabilityText=p=>p==null?'—':p>0&&p<.00001?Number(p).toExponential(3)+' (raw probability)':fmt(p*100,5)+'%';
  const metrics=[['delta_previous_token','오류 − 직전 token (nat)'],['delta_previous','오류 − 직전 정답 가능 숫자 (기존, nat)'],['delta_prior_mean','오류 − 이전 숫자 평균 (nat)'],['error_entropy','최초 오류 H₃₀ (nat)'],['risk_mean_entropy','경로 risk 숫자 평균 H₃₀ (nat)'],['iou','전체 구간 IoU'],['min_safe','가능성 유지 숫자의 최소 safe q']];
  const presets=[['all','전체','추가 조건 없음'],['paired','오류 전후 비교 가능','최초 numeric 오류와 이전 정답 가능 숫자가 모두 있는 경로'],['increase','직전 숫자 대비 상승 (기존)','최초 numeric 오류 H − 직전 정답 가능 numeric step H > 0. 중간 소수점·서식 token은 건너뜁니다.'],['decrease','직전 숫자 대비 감소 (기존)','최초 numeric 오류 H − 직전 정답 가능 numeric step H < 0. 중간 소수점·서식 token은 건너뜁니다.'],['token_increase','직전 token 대비 상승','최초 오류(numeric·format) H(k) − 바로 직전 decoding token H(k−1) > 0. 소수점·공백·endpoint 밖 token도 포함합니다.'],['token_decrease','직전 token 대비 감소','최초 오류(numeric·format) H(k) − 바로 직전 decoding token H(k−1) < 0. 소수점·공백·endpoint 밖 token도 포함합니다.'],['low_error','낮은 entropy에서 오류','최초 오류 H₃₀ ≤ 0.5 nat'],['high_safe','높은 entropy인데 가능성 유지','정답 가능성을 유지한 numeric step 중 H₃₀ ≥ 1.5 nat이 존재'],['next_failure','유지 직후 다음 숫자 오류','가능성 유지 숫자 다음 numeric step에서 최초 오류 발생'],['disagree','두 모델 판정 불일치','선택한 endpoint의 반올림 정답 여부가 두 모델에서 다름(둘 다 지원 표기)']];
  const regions=[['all','전체'],['integer','정수부 최초 오류'],['fractional','소수부 최초 오류'],['format','소수점·서식 최초 오류'],['none','오류 없음']];
  const conditions={first_digit_correct:'첫 숫자 가능성 유지',first_digit_wrong:'첫 숫자 오류',previous_correct_now_correct:'이전·현재 가능성 유지',previous_correct_now_wrong:'이전 유지 → 최초 오류',after_error:'오류 이후',format_correct:'서식 유지',format_first_error:'서식 최초 오류',format_after_error:'오류 이후 서식',all_correct_endpoints:'끝까지 정답',not_all_correct_endpoints:'최종 오답'};
  const stateName=s=>!s?'기타 token':s.first_error?(s.selection_kind==='forced'?'최초 오류 · 강제 선택':s.selection_kind==='conditioned'?'최초 오류 · 고정 prefix':'최초 오류'):s.prefix_correct?'가능성 유지':'오류 이후';
  const stateColor=s=>!s?'#c1cbd0':s.first_error?'#c44536':s.prefix_correct?'#247ba0':'#87929b';
  const selection=()=>({model:$('model').value,endpoint:$('endpoint').value});
  const pathFor=s=>catalog[$('model').value+'/'+s.sample_key]?.endpoints?.[$('endpoint').value]||{};
  const savedAvailable=s=>$('model').value==='qwen3'&&!!savedLookup.samples?.[s.sample_key];
  const activeMetrics=()=>source==='deterministic'?(trace?.endpoints?.[$('endpoint').value]?.metrics||{}):(current?pathFor(current):{});
  function value(s,key){return key==='iou'?s.models[$('model').value]?.iou:pathFor(s)[key==='error_entropy'?'token_error_entropy':key];}
  function matchesPreset(s,preset){const p=pathFor(s),m=s.models,ep=$('endpoint').value+'_correct';return preset==='all'||preset==='paired'&&p.paired||preset==='increase'&&p.delta_previous>0||preset==='decrease'&&p.delta_previous<0||preset==='token_increase'&&p.delta_previous_token>0||preset==='token_decrease'&&p.delta_previous_token<0||preset==='low_error'&&p.token_error_entropy!=null&&p.token_error_entropy<=.5||preset==='high_safe'&&p.safe_high||preset==='next_failure'&&p.next_failure||preset==='disagree'&&m.qwen25?.supported&&m.qwen3?.supported&&m.qwen25[ep]!==m.qwen3[ep];}
  function rangesMatch(s){return metrics.every(([key])=>{const a=$(key+'-min').value,b=$(key+'-max').value;if(a===''&&b==='')return true;const x=value(s,key);return Number.isFinite(x)&&(a===''||x>=Number(a))&&(b===''||x<=Number(b));});}
  function remember(){const h=new URLSearchParams({model:$('model').value,dataset:$('dataset').value,endpoint:$('endpoint').value,preset:[...selectedPresets].join(',')||'all',error_region:[...selectedRegions].join(',')||'all',search:$('search').value,sort:$('sort').value,range:$('chart-range').value,distribution:$('distribution-mode').value,source,path:pathId,reference,paths:$('path-availability').value});if(current)h.set('sample',current.sample_key);if(step!=null)h.set('step',step);for(const[k]of metrics)for(const bound of ['min','max'])if($(k+'-'+bound).value!=='')h.set(k+'-'+bound,$(k+'-'+bound).value);history.replaceState(null,'','#'+h);window.dispatchEvent(new CustomEvent('rollout:context'));}
  function clearCase(message){serial++;current=trace=raw=integrated=baselineTrace=baselineRaw=pathBundle=activeCandidate=referenceRaw=null;source=cfg.default_source||'original';pathId='1';blue=[];step=null;segment=null;$('video').pause();$('video').removeAttribute('src');$('video').load();$('case-content').hidden=true;$('empty').hidden=false;$('status').hidden=false;$('status').textContent=message;remember();}
  function applyFilters(wanted,requestedStep,requestedPath){
    const old=wanted||current?.sample_key,ds=$('dataset').value,q=$('search').value.trim().toLowerCase();
    filtered=MANIFEST.filter(s=>($('path-availability').value!=='deterministic'||savedAvailable(s))&&(!ds||s.dataset===ds)&&(!q||(s.sample_key+' '+s.query).toLowerCase().includes(q))&&(!selectedRegions.size||selectedRegions.has(pathFor(s).error_region))&&(!selectedPresets.size||[...selectedPresets].some(p=>matchesPreset(s,p)))&&rangesMatch(s));
    const order=$('sort').value,key=order.startsWith('token_delta_')?'delta_previous_token':order.startsWith('delta_')?'delta_previous':order.startsWith('error_')?'error_entropy':'iou',direction=order.endsWith('_desc')?-1:1;
    filtered.sort((a,b)=>{if(order!=='sample'){const x=value(a,key),y=value(b,key);if(x==null&&y!=null)return 1;if(y==null&&x!=null)return -1;if(x!=null&&y!=null&&x!==y)return direction*(x-y);}return a.sample_key.localeCompare(b.sample_key,'en',{numeric:true});});
    $('count').textContent=filtered.length.toLocaleString()+' / '+MANIFEST.length.toLocaleString()+' queries';
    for(const b of $('error-regions').children)b.setAttribute('aria-pressed',String(b.dataset.region==='all'?!selectedRegions.size:selectedRegions.has(b.dataset.region)));
    for(const b of $('presets').children)b.setAttribute('aria-pressed',String(b.dataset.preset==='all'?!selectedPresets.size:selectedPresets.has(b.dataset.preset)));$('preset-help').replaceChildren(...presets.filter(p=>selectedPresets.size?selectedPresets.has(p[0]):p[0]==='all').map(p=>el('span',p[1]+': '+p[2]+' ')));
    const chosen=filtered.find(s=>s.sample_key===old)||filtered[0];page=chosen?Math.floor(filtered.indexOf(chosen)/pageSize):0;
    if(chosen)selectCase(chosen,requestedStep,requestedPath);else clearCase('필터에 해당하는 sample이 없습니다.');renderList();
  }
  function renderList(){const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pages-1);$('samples').replaceChildren();for(const s of filtered.slice(page*pageSize,(page+1)*pageSize)){const b=el('button');b.dataset.sample=s.sample_key;b.setAttribute('aria-current',String(current?.sample_key===s.sample_key));b.append(el('strong',s.sample_key),el('span',s.query.length>85?s.query.slice(0,85)+'…':s.query),el('small','숫자 대비 '+fmt(pathFor(s).delta_previous)+' · token 대비 '+fmt(pathFor(s).delta_previous_token)+' nat · IoU '+fmt(s.models[$('model').value]?.iou)));if(savedAvailable(s))b.append(el('small','저장된 deterministic 경로 '+savedLookup.samples[s.sample_key].paths+'개'));b.onclick=()=>selectCase(s);$('samples').append(b);}$('page-count').textContent=(page+1)+' / '+pages;$('prev-page').disabled=page===0;$('next-page').disabled=page>=pages-1;}
  async function get(url,type='json'){if(!cache.has(url))cache.set(url,fetch(url).then(r=>{if(!r.ok)throw Error('HTTP '+r.status+' · '+url);return r.text();}).then(s=>type==='trace'?JSON.parse(s.slice(s.indexOf(',')+1).replace(/\);\s*$/,'')):JSON.parse(s)).catch(e=>{cache.delete(url);throw e;}));return cache.get(url);}
  async function blueRows(key,endpoint){const refs=(catalog[key]?.blue_ranges||[]).filter(r=>!endpoint||r[0]===endpoint);return Promise.all(refs.map(async([ep,st,a,b])=>{const id=key+'/'+ep+'/'+st;if(!blueCache.has(id))blueCache.set(id,fetch(absolute(cfg.blue),{headers:{Range:`bytes=${a}-${b}`}}).then(async r=>{if(r.status!==206)throw Error('후보 자료의 부분 읽기에 실패했습니다 (HTTP '+r.status+')');return r.json();}).catch(e=>{blueCache.delete(id);throw e;}));return blueCache.get(id);}));}
  function notice(text){$('path-notice').textContent=text||'';$('path-notice').hidden=!text;}
  function configurePathControls(){
    const available=current&&savedAvailable(current);$('decoding-source').value=source;
    $('decoding-source').querySelector('[value="original"]').disabled=!originalAvailable;
    const deterministicOption=$('decoding-source').querySelector('[value="deterministic"]');deterministicOption.disabled=!available;deterministicOption.textContent=cfg.saved_paths?'deterministic rollout · '+cfg.scope.replace('/',' '):'deterministic rollout · pilot 단계에서 확인';
    $('decoding-path').replaceChildren();$('decoding-path').disabled=source!=='deterministic';
    $('path-reference').value=reference;$('path-reference').disabled=source!=='deterministic';
    $('saved-path-legend').hidden=source!=='deterministic';$('all-paths').hidden=source!=='deterministic';
    $('distribution-mode').querySelector('[value="processed"]').disabled=source==='deterministic';
    if(source==='deterministic'){
      $('distribution-mode').value='raw';
      for(const c of pathBundle.record.result.candidates){const m=pathBundle.metadata.candidates[String(c.index)];$('decoding-path').add(new Option('#'+c.index+(c.branch?' · 부모 #'+c.branch.parent+' · 분기 step '+(c.branch.position+1):' · greedy')+' · IoU '+fmt(m.iou),String(c.index)));}
      $('decoding-path').value=pathId;
      $('path-source-note').textContent='deterministic rollout · '+cfg.scope.replace('/',' ')+' · Qwen3-VL-8B-Instruct · '+(savedLookup.source?.input_description||'2 FPS, spatial target 100,352 pixels, video_query. 원본 greedy와 입력 조건이 다릅니다').replace(/[.。]+$/,'')+(originalAvailable?'.':'. 이 단계에는 deterministic 경로만 있습니다.')+' 겹쳐 보는 곡선은 같은 deterministic 입력 안의 경로끼리 비교합니다.';
    }else{
      $('decoding-path').add(new Option('원본 greedy','1'));
      $('path-source-note').textContent='원본 greedy · '+cfg.scope+'. '+(available?'이 sample의 deterministic 경로 '+savedLookup.samples[current.sample_key].paths+'개도 위 출처 선택에서 볼 수 있습니다.':!cfg.saved_paths?'pilot 경로는 pilot 300·600·900의 별도 화면에서 확인하세요.':$('model').value!=='qwen3'?'다른 저장 경로는 Qwen3에만 있습니다.':'이 sample에는 저장된 deterministic 경로가 없습니다.');
    }
    $('jump-branch').disabled=source!=='deterministic'||!activeCandidate?.branch_step;
  }
  async function savedBundle(sample){
    if(!savedCache.has(sample)){
      const entry=savedLookup.samples[sample];
      const fetchJSON=async ref=>{const url=new URL(ref,savedLookupURL).href,r=await fetch(url);if(!r.ok)throw Error('HTTP '+r.status+' · '+url);return r.json();};
      savedCache.set(sample,Promise.all([fetchJSON(entry.record),fetchJSON(entry.index),fetchJSON(entry.metadata)]).then(([record,index,metadata])=>({record,index,metadata})).catch(e=>{savedCache.delete(sample);throw e;}));
      while(savedCache.size>4)savedCache.delete(savedCache.keys().next().value);
    }
    return savedCache.get(sample);
  }
  function rawCandidate(id){
    const c=pathBundle.record.result.candidates.find(c=>String(c.index)===String(id)),m=pathBundle.metadata.candidates[String(id)];
    if(!c||!m)throw Error('저장 경로 #'+id+'를 찾을 수 없습니다.');
    const nodes=pathBundle.index.candidates[String(id)].nodes,forced=new Set(m.forced_steps||[]);
    return {steps:c.ids.map((token_id,i)=>{const node=pathBundle.index.nodes[nodes[i]],chosen=node.observed.find(t=>t.id===token_id)||node.top.find(t=>t.id===token_id);if(!chosen)throw Error('실제 선택 token의 저장 자료가 없습니다.');return {step:i+1,token_id,token:chosen.token,chosen_probability:chosen.probability,raw:{top30:node.top},prefix:node.prefix_text,node:nodes[i],selection_kind:forced.has(i+1)?'forced':i<m.forced_prefix_length?'conditioned':'free'};})};
  }
  function activatePath(id,requestedStep){
    const m=pathBundle.metadata.candidates[String(id)];
    if(!m){notice('요청한 경로 #'+id+'가 없어 저장된 greedy 경로를 표시합니다.');id='1';}
    pathId=String(id);activeCandidate=pathBundle.metadata.candidates[pathId];
    const c=pathBundle.record.result.candidates.find(c=>String(c.index)===pathId);
    raw=rawCandidate(pathId);trace={answer:c.answer,timestamps:c.parsed?[[c.parsed.start,c.parsed.end]]:[],iou:activeCandidate.iou,endpoints:activeCandidate.endpoints||{}};
    blue=[];integrated=null;
    const preferred=activeCandidate.branch_step||ep()?.steps.find(s=>s.first_error)?.ar_step||raw.steps[0].step;
    step=raw.steps.some(r=>r.step===requestedStep)?requestedStep:preferred;
    if(!displaySteps().some(r=>r.step===step))$('chart-range').value='response';
  }
  function updateIdentity(){
    $('identity').textContent=cfg.scope+' · '+$('model').value+' · '+current.sample_key+' · '+$('endpoint').value+(source==='deterministic'?' · deterministic #'+pathId:' · 원본 greedy');
    $('query').textContent=current.query;$('answer').textContent=trace.answer;
    $('gt').textContent=JSON.stringify(current.span[0])+' 초';$('prediction').textContent=trace.timestamps?.[0]?JSON.stringify(trace.timestamps[0])+' 초':'지원되는 구간 없음';$('sample-iou').textContent=fmt(trace.iou);
    $('status').textContent='';$('status').hidden=true;
  }
  async function selectCase(s,requestedStep,requestedPath){
    const {model,endpoint}=selection(),same=current?.sample_key===s.sample_key&&loadedModel===model;
    const desiredSource=requestedPath?.source||(same?source:cfg.default_source||'original'),desiredPath=requestedPath?.pathId||(same?pathId:'1');
    const ticket=++serial,key=model+'/'+s.sample_key;current=s;loadedModel=model;source=desiredSource;pathId=String(desiredPath);
    trace=raw=integrated=pathBundle=activeCandidate=referenceRaw=null;blue=[];segment=null;notice(requestedPath?.notice||'');
    $('video').pause();$('case-content').hidden=true;$('empty').hidden=true;$('status').hidden=false;$('status').textContent='sample과 decoding 경로를 불러오는 중…';renderList();
    if(source==='original'&&!originalAvailable){source='deterministic';pathId='1';notice('이 단계는 deterministic 경로만 제공하므로 저장된 greedy를 표시합니다.');}
    if(source==='deterministic'&&!savedAvailable(s)){if(!originalAvailable){clearCase('이 sample에는 현재 범위의 저장된 decoding 경로가 없습니다.');return;}source='original';pathId='1';notice(cfg.saved_paths?'선택한 모델·sample에는 deterministic 경로가 없습니다. 원본 greedy를 표시합니다.':'full에서는 원본 greedy를 표시합니다. pilot 경로는 pilot 300·600·900의 별도 화면에서 확인하세요.');}
    try{
      if(source==='deterministic'){
        const bundle=await savedBundle(s.sample_key);if(ticket!==serial)return;pathBundle=bundle;activatePath(pathId,requestedStep);
      }else{
        const [tr,ra]=await Promise.all([get(absolute(cfg.trace+key+'.js'),'trace'),get(absolute(cfg.raw+key+'.json'))]);if(ticket!==serial)return;
        baselineTrace=tr;baselineRaw=ra;trace=tr;raw=ra;pathId='1';const e=ep(),preferred=e?.steps.find(x=>x.first_error)?.ar_step||e?.steps[0]?.ar_step||raw.steps[0]?.step;
        step=raw.steps.some(r=>r.step===requestedStep)?requestedStep:preferred;
      }
      configurePathControls();updateIdentity();$('video').src=new URL(s.media,new URL(cfg.media_base,location.href)).href;$('case-content').hidden=false;
      render();remember();
      if(source==='deterministic')return;
      const extra=await Promise.allSettled([blueRows(key,null),cfg.integrated?get(absolute(cfg.integrated+key+'.js'),'trace'):Promise.resolve(null)]);if(ticket!==serial)return;
      if(extra[0].status==='fulfilled')blue=extra[0].value;else{$('status').hidden=false;$('status').textContent='경로는 표시했습니다. 정답 가능성 자료 로드 실패: '+extra[0].reason.message;}
      if(extra[1].status==='fulfilled')integrated=extra[1].value;else{$('status').hidden=false;$('status').textContent+=($('status').textContent?' · ':'')+'GT 고정 상세 로드 실패';}renderDistribution();renderAdditional();
    }catch(e){if(ticket===serial){clearCase('sample 로드 실패: '+e.message);$('empty').textContent='자료를 불러오지 못했습니다. 다른 sample을 선택하거나 새로고침하세요.';}}
  }
  function selectSavedPath(id){
    if(!pathBundle||source!=='deterministic')return;serial++;segment=null;$('video').pause();notice('');activatePath(id);configurePathControls();updateIdentity();render();remember();
  }
  function referenceId(){return source!=='deterministic'||reference==='none'?null:reference==='greedy'?(pathId==='1'?null:'1'):(activeCandidate?.parent==null?null:String(activeCandidate.parent));}
  function renderPathRows(){
    $('path-rows').replaceChildren();if(source!=='deterministic')return;
    for(const c of pathBundle.record.result.candidates){const m=pathBundle.metadata.candidates[String(c.index)],tr=el('tr'),b=el('button','#'+c.index+(c.branch?'':' · greedy'));
      tr.dataset.pathId=c.index;tr.setAttribute('aria-current',String(String(c.index)===pathId));b.onclick=()=>selectSavedPath(String(c.index));const td=el('td');td.append(b);tr.append(td);
      const nums=Object.values(m.endpoints||{}).flatMap(e=>e.steps.filter(s=>s.numeric)),mean=nums.length?nums.reduce((v,r)=>v+r.entropy,0)/nums.length:null;
      for(const value of [c.branch?'#'+c.branch.parent+' / '+(c.branch.position+1):'—',c.parsed?c.parsed.start+' – '+c.parsed.end:'지원 표기 밖',fmt(m.iou),fmt(mean)])tr.append(el('td',value));$('path-rows').append(tr);
    }
  }
  const ep=()=>trace?.endpoints?.[$('endpoint').value];
  function labelFor(st){for(const[name,e]of Object.entries(trace?.endpoints||{})){const s=e.steps.find(x=>x.ar_step===st);if(s)return {...s,endpoint:name};}return null;}
  function displaySteps(){if($('chart-range').value==='endpoint'&&ep())return raw.steps.filter(r=>ep().steps.some(s=>s.ar_step===r.step));return raw.steps;}
  function entropy(r,mode='raw'){const top=r[mode].top30,mass=top.reduce((a,t)=>a+t.probability,0);return -top.reduce((a,t)=>{const q=t.probability/mass;return a+(q>0?q*Math.log(q):0);},0);}
  function setStep(n){if(!raw?.steps.some(r=>r.step===n))return;if(!displaySteps().some(r=>r.step===n))$('chart-range').value='response';step=n;renderPath();renderDistribution();renderAdditional();remember();}
  function cards(root,values){root.replaceChildren();for(const[label,value]of values){const c=el('div',undefined,'metric-card');c.append(el('span',label),el('strong',value));root.append(c);}}
  function render(){const p=activeMetrics();cards($('path-summary'),[[source==='deterministic'&&p.error_selection_kind&&p.error_selection_kind!=='free'?'최초 오류 · '+(p.error_selection_kind==='forced'?'강제 선택':'고정 prefix'):'최초 오류',fmt(p.token_error_entropy??p.error_entropy)+' nat'],['바로 직전 token '+JSON.stringify(raw?.steps.find(r=>r.step===p.previous_token_step)?.token??'—'),fmt(p.previous_token_entropy)+' nat'],['직전 token 대비 차이',(p.delta_previous_token>0?'+':'')+fmt(p.delta_previous_token)+' nat'],['직전 정답 가능 숫자 '+JSON.stringify(raw?.steps.find(r=>r.step===p.previous_step)?.token??'—'),fmt(p.previous_entropy)+' nat'],['직전 숫자 대비 차이 (기존)',(p.delta_previous>0?'+':'')+fmt(p.delta_previous)+' nat']]);renderPathRows();renderPath();renderDistribution();renderAdditional();}

  function renderPath(){if(!raw)return;const rows=displaySteps();if(!rows.length)return;const rid=referenceId();referenceRaw=rid?rawCandidate(rid):null;const referenceByStep=new Map((referenceRaw?.steps||[]).map(r=>[r.step,r]));if(!rows.some(r=>r.step===step))step=rows[0].step;const svg=$('path-plot'),width=Math.max(360,$('path-plot').parentElement.clientWidth,rows.length*64+100),height=250;svg.replaceChildren();svg.style.width=width+'px';svg.setAttribute('viewBox',`0 0 ${width} ${height}`);const unit=(width-100)/rows.length,x=i=>50+unit*(i+.5),y=v=>235-v/Math.log(30)*190;
    const draw=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;};
    $('path-y-axis').replaceChildren();const unitLabel=el('span','H₃₀ nat');unitLabel.style.top='0px';$('path-y-axis').append(unitLabel);
    for(const v of [0,1,2,3,Math.log(30)]){const tick=el('span',fmt(v,1));tick.style.top=(y(v)-8)+'px';$('path-y-axis').append(tick);draw('line',{x1:52,x2:width-20,y1:y(v),y2:y(v),stroke:'#dce5df'});draw('text',{x:45,y:y(v)+4,'text-anchor':'end','font-size':11,fill:'#52685b'},fmt(v,1));}
    draw('text',{x:12,y:18,'font-size':12,fill:'#344b3e'},'H₃₀ (nat) · raw');
    for(const name of ['start','end']){const indices=rows.map((r,i)=>trace.endpoints?.[name]?.steps.some(s=>s.ar_step===r.step)?i:-1).filter(i=>i>=0);if(indices.length){const a=x(indices[0])-13,b=x(indices.at(-1))+13;draw('rect',{x:a,y:27,width:b-a,height:217,fill:name==='start'?'#247ba00c':'#9370b50c'});draw('text',{x:(a+b)/2,y:39,'text-anchor':'middle','font-size':11,fill:'#52685b'},name);}}
    if(referenceRaw){const points=rows.map((r,i)=>referenceByStep.has(r.step)?x(i)+','+y(entropy(referenceByStep.get(r.step))):null).filter(Boolean);if(points.length)draw('polyline',{points:points.join(' '),fill:'none',stroke:'#9b78ad','stroke-width':2,'stroke-dasharray':'6 4',class:'reference-curve','data-path-id':rid,'data-reference':reference});}
    draw('polyline',{points:rows.map((r,i)=>x(i)+','+y(entropy(r))).join(' '),fill:'none',stroke:'#667b71','stroke-width':2});$('token-strip').replaceChildren();$('token-strip').style.width=width+'px';$('token-strip').classList.toggle('has-reference',!!referenceRaw);
    rows.forEach((r,i)=>{const originalLabel=labelFor(r.step),s=originalLabel?{...originalLabel,selection_kind:r.selection_kind}:null,h=entropy(r),selected=r.step===step;const point=draw('circle',{cx:x(i),cy:y(h),r:selected?7:5,fill:stateColor(s),stroke:selected?'#7354aa':'white','stroke-width':selected?3:1,role:'button',tabindex:0,'aria-label':`step ${r.step} ${JSON.stringify(r.token)} entropy ${fmt(h)} ${stateName(s)}`,'data-step':r.step,'data-selection-kind':r.selection_kind||'free'});point.style.cursor='pointer';point.onclick=()=>setStep(r.step);point.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setStep(r.step);}};
      if(r.selection_kind==='forced')draw('rect',{x:x(i)-9,y:y(h)-9,width:18,height:18,fill:'none',stroke:'#d18b22','stroke-width':2,class:'forced-step-marker','data-step':r.step});
      if(s?.first_error)draw('text',{x:x(i),y:Math.max(55,y(h)-14),'text-anchor':'middle','font-size':11,fill:'#b42318'},s.selection_kind==='forced'?'강제 오류':s.selection_kind==='conditioned'?'고정 prefix 오류':'최초 오류');if(selected)draw('text',{x:x(i),y:Math.min(225,y(h)+21),'text-anchor':'middle','font-size':12,fill:'#7354aa'},fmt(h));
      const b=el('button',undefined,selected?'selected':'');b.dataset.step=r.step;b.dataset.selectionKind=r.selection_kind||'free';b.style.position='absolute';b.style.left=(x(i)-unit/2+3)+'px';b.style.width=(unit-6)+'px';b.style.borderTop='3px solid '+stateColor(s);b.append(el('small',String(r.step)),el('span',JSON.stringify(r.token)));if(r.selection_kind&&r.selection_kind!=='free')b.append(el('small',r.selection_kind==='forced'?'강제 선택':'고정 prefix','selection-tag'));const rr=referenceByStep.get(r.step);if(rr)b.append(el('small','#'+rid+' '+JSON.stringify(rr.token),'reference-token'));b.title=stateName(s)+' · '+fmt(h)+' nat';b.onclick=()=>setStep(r.step);$('token-strip').append(b);
    });
    const p=activeMetrics(),target=rows.find(r=>r.step===step),label=labelFor(step),i=rows.indexOf(target);const selectedToken=$('token-strip').children[i];$('path-plot').parentElement.scrollLeft=Math.max(0,x(i)-$('path-plot').parentElement.clientWidth/2);$('step-slider').min=0;$('step-slider').max=rows.length-1;$('step-slider').value=i;$('prev-step').disabled=i===0;$('next-step').disabled=i===rows.length-1;$('jump-error').disabled=p.error_step==null;$('jump-branch').disabled=!activeCandidate?.branch_step;
    const before=label&&trace.endpoints[label.endpoint].steps.filter(s=>s.ar_step<step).at(-1)?.prefix||'';
    const choice=target.selection_kind==='forced'?'강제 선택':target.selection_kind==='conditioned'?'고정 prefix':'모델 자유 생성';
    $('step-info').textContent=(source==='deterministic'?'경로 #'+pathId+' · '+choice+' · 부모 '+(activeCandidate.parent==null?'없음':'#'+activeCandidate.parent)+' · 분기 step '+(activeCandidate.branch_step||'없음')+' · IoU '+fmt(activeCandidate.iou)+'\n':'')+`선택: step ${step} · token ${JSON.stringify(target.token)} · H₃₀ ${fmt(entropy(target))} nat\n`+(source==='deterministic'?'선택 전 전체 prefix: '+JSON.stringify(target.prefix)+'\n':'')+(label?`${label.endpoint} prefix: ${JSON.stringify(before)} → ${JSON.stringify(label.prefix)} · ${stateName(label?{...label,selection_kind:target.selection_kind}:null)}`:'시각 숫자 밖의 token입니다. 분포는 확인할 수 있지만 endpoint 정답 가능성 판정은 없습니다.');
    const rr=referenceByStep.get(step);$('reference-info').hidden=!referenceRaw;$('reference-info').textContent=referenceRaw?'점선: '+(reference==='parent'?'부모':'greedy')+' 경로 #'+rid+' · 선택한 step '+step+(rr?'의 token '+JSON.stringify(rr.token)+' · H₃₀ '+fmt(entropy(rr))+' nat':'은 비교 경로에 없습니다.')+'. 가로 위치는 같은 decoding 순번이며 분기 이후 token·prefix·의미는 다를 수 있습니다. 아래 가로축에 비교 token도 표시합니다.':'';
    $('path-explanation').textContent=(p.paired?`이 sample의 ${$('endpoint').value}에서는 최초 오류 ${fmt(p.error_entropy)} − 직전 정답 가능 숫자(서식 건너뜀) ${fmt(p.previous_entropy)} = ${p.delta_previous>0?'+':''}${fmt(p.delta_previous)} nat입니다. 오류 순간에 entropy가 ${p.delta_previous>0?'높아졌습니다':p.delta_previous<0?'낮아졌습니다':'같았습니다'}. `:p.error_kind==='none'?'선택한 endpoint에는 최초 오류가 없어 paired 차이를 계산하지 않습니다. ':p.error_kind==='format'?'최초 오류가 서식 step이므로 numeric paired 비교에서는 제외합니다. ':p.error_kind==='numeric'?'첫 숫자부터 오류여서 이전 숫자와의 paired 차이를 계산할 수 없습니다. ':'지원 표기 밖이므로 endpoint 판정 없이 전체 decoding entropy를 표시합니다. ')+ (p.delta_previous_token!=null?`별도의 직전 token 기준은 step ${p.error_step}의 ${fmt(p.token_error_entropy)} − step ${p.previous_token_step} ${JSON.stringify(raw.steps.find(r=>r.step===p.previous_token_step)?.token)}의 ${fmt(p.previous_token_entropy)} = ${p.delta_previous_token>0?'+':''}${fmt(p.delta_previous_token)} nat입니다. `:'') + (source==='deterministic'?'이 값은 현재 경로의 판정입니다. 주황은 실험에서 강제 선택한 token이며, 고정 prefix 안의 오류를 모델의 새 자유 생성 오류로 세지 않습니다. ':'') + '곡선은 생성 순서의 실제 entropy이고, 색은 token 선택 후의 판정입니다. 빨간 점이 반드시 가장 높지는 않습니다. 회색 오류 이후 step은 paired 비교에서 제외합니다. 점이나 token을 누르면 아래 후보 분포가 바뀝니다.';
  }
  function renderDistribution(){if(!raw)return;const row=raw.steps.find(r=>r.step===step),mode=$('distribution-mode').value,top=row[mode].top30,mass=top.reduce((a,t)=>a+t.probability,0),ranked=top.map(t=>({...t,q:t.probability/mass})),br=source==='original'?blue.find(r=>r.ar_step===step):null,states=new Map((mode==='raw'?br?.candidates||[]:[]).map(r=>[r.id,r.state]));const labels={safe:'정답 가능',unsafe:'정답 불가',unresolved:'미확인'};
    $('distribution-info').textContent=`step ${step} · 실제 token ${JSON.stringify(row.token)} · ${mode} H₃₀ ${fmt(entropy(row,mode))} nat · 재정규화 전 top-30 probability mass ${fmt(mass*100,4)}%`+(source==='deterministic'?' · 실제 선택의 전체 vocabulary raw probability '+probabilityText(row.chosen_probability)+(top.some(t=>t.id===row.token_id)?'':' · 실제 선택은 top-30 밖 (H₃₀·후보 표에는 포함하지 않음)'):'');
    if(br&&mode==='raw'){cards($('mass-summary'),[['정답 가능 q 합',fmt(br.safe_lower)],['정답 불가 q 합',fmt(br.unsafe_mass)],['미확인 q 합',fmt(br.unresolved_mass)]]);$('feasibility-note').textContent='이 step은 정답 가능성을 유지한 숫자입니다. 후보별 판정과 q 합은 저장된 raw top-30 분석입니다. 합계는 1이며 최종 정답 확률과 다릅니다.';}
    else{$('mass-summary').replaceChildren();$('feasibility-note').textContent=mode==='processed'?'processed 분포입니다. raw 후보의 정답 가능성 판정을 다른 분포에 그대로 붙이지 않습니다.':'이 step에는 별도로 저장된 가능성 유지 숫자의 후보 판정이 없습니다. 후보 확률만 표시하며 미계산을 정답 불가로 간주하지 않습니다.';}
    $('candidate-bars').replaceChildren();$('candidate-rows').replaceChildren();for(const[t,r]of ranked.entries()){const selected=r.id===row.token_id,state=labels[states.get(r.id)]||'판정 자료 없음';const tr=el('tr');for(const v of [t+1,JSON.stringify(r.token),fmt(r.q*100,5)+'%',selected?'실제 선택':'',state])tr.append(el('td',v));$('candidate-rows').append(tr);if(t<10){const line=el('div',undefined,'candidate-row'+(selected?' selected':'')),track=el('div',undefined,'bar-track'),fill=el('div',undefined,'bar-fill');fill.style.width=100*r.q+'%';track.append(fill);line.append(el('code',JSON.stringify(r.token)+(selected?' ✓':'')),track,el('span',fmt(r.q*100,2)+'%'),el('span',state));$('candidate-bars').append(line);}}
  }
  function renderAdditional(){if(!trace)return;$('additional').hidden=source==='deterministic';if(source==='deterministic')return;const endpoint=$('endpoint').value,e=ep();cards($('endpoint-scores'),[['전체 구간 IoU',fmt(trace.iou)],['start만 평가 · end GT 고정',fmt(trace.endpoints?.start?.endpoint_iou)],['end만 평가 · start GT 고정',fmt(trace.endpoints?.end?.endpoint_iou)]]);$('iou-details').replaceChildren();const ie=integrated?.endpoints?.[endpoint];if(ie){const table=el('table'),head=el('tr');for(const t of ['step','token','전체 구간 가능성','GT 고정 가능성','GT 고정 최대 IoU 전 → 후'])head.append(el('th',t));table.append(head);for(const s of ie.steps){const tr=el('tr');for(const t of [s.ar_step,JSON.stringify(s.token),integrated.joint?.steps.find(t=>t.step===s.ar_step)?.label_50||'자료 없음',s.label_50,fmt(s.oracle_before)+' → '+fmt(s.oracle_after)])tr.append(el('td',t));table.append(tr);}$('iou-details').append(table,el('p','위 표는 전체 구간과 반대쪽 endpoint를 GT로 고정한 IoU≥0.5 completion 판정을 나란히 비교합니다. retains_feasibility는 가능성 유지, first_loss는 최초 상실, after_loss는 상실 이후입니다. 메인 화면의 반올림 endpoint 정답 판정과 정의가 다릅니다.','caption'));}else $('iou-details').append(el('p','step별 GT 고정 IoU 상세는 기존 pilot 분석에만 있습니다. 이 범위에서는 저장된 endpoint 요약만 표시합니다.','caption'));
    $('cohort-rows').replaceChildren();for(const s of window.CONDITION_STATS||[]){if(s.model!==$('model').value||s.dataset!==current.dataset||s.endpoint!==endpoint)continue;const tr=el('tr');for(const v of [conditions[s.condition]||s.condition,s.n_queries+' / '+s.n_steps,fmt(s.mean_entropy)+' ['+s.ci95.map(x=>fmt(x)).join(', ')+']'])tr.append(el('td',v));$('cohort-rows').append(tr);}
    $('role-comparison').replaceChildren();const selected=e?.steps.find(s=>s.ar_step===step);if(selected){$('role-comparison').append(el('h3','같은 GT/생성 자릿수 · '+selected.role));for(const s of window.ROLE_STRATA||[]){if(s.model===$('model').value&&s.dataset===current.dataset&&s.endpoint===endpoint&&s.gt_digits===e.gt_digits&&s.generated_digits===e.generated_digits&&s.role===selected.role){const r=el('div',undefined,'role-row');r.append(el('span',conditions[s.condition]||s.condition),el('span',fmt(s.mean_entropy)+' nat · N='+s.n_queries));$('role-comparison').append(r);}}$('role-comparison').append(el('p','위 수치는 같은 모델·dataset·endpoint·GT/출력 자릿수·role의 query별 평균을 집계한 값입니다. 현재 sample의 경로 내부 차이와 구분합니다.','caption'));}
  }
  function play(span){if(!span)return;const v=$('video'),duration=Number.isFinite(v.duration)?v.duration:current.duration,a=Math.max(0,Math.min(duration,span[0])),b=Math.max(0,Math.min(duration,span[1]));if(a>=b)return;segment=b;v.currentTime=a;v.play().catch(e=>{if(e.name!=='AbortError'){$('status').hidden=false;$('status').textContent=e.message;}});}
  function readChoices(text,choices){return new Set((text||'').split(',').filter(id=>id!=='all'&&choices.some(c=>c[0]===id)));}
  function toggleChoice(set,id){if(id==='all')set.clear();else if(set.has(id))set.delete(id);else set.add(id);applyFilters();}
  function restore(){
    const h=new URLSearchParams(location.hash.slice(1));let restoreNotice='';
    for(const id of ['model','dataset','endpoint','sort','chart-range','distribution-mode','path-availability']){
      const key=id==='chart-range'?'range':id==='distribution-mode'?'distribution':id==='path-availability'?'paths':id;
      if([...$(id).options].some(o=>!o.disabled&&o.value===h.get(key)))$(id).value=h.get(key);
    }
    if(!cfg.saved_paths&&h.get('paths')==='deterministic'){$('path-availability').value='all';restoreNotice='full에는 pilot 경로 필터를 적용하지 않습니다. pilot 경로는 pilot 300·600·900의 별도 화면에서 확인하세요.';}
    if(!originalAvailable&&h.get('model')==='qwen25')restoreNotice='이 pilot 단계에는 Qwen3 deterministic 경로만 있으므로 해당 모델을 표시합니다.';
    $('search').value=h.get('search')||'';selectedRegions=readChoices(h.get('error_region'),regions);selectedPresets=readChoices(h.get('preset'),presets);
    if(!originalAvailable)selectedPresets.delete('disagree');
    for(const[k]of metrics)for(const b of ['min','max'])$(k+'-'+b).value=$(k+'-'+b).disabled?'':h.get(k+'-'+b)||'';
    reference=['parent','greedy','none'].includes(h.get('reference'))?h.get('reference'):'parent';
    applyFilters(h.get('sample'),h.has('step')?Number(h.get('step')):undefined,{source:h.has('source')?(h.get('source')==='deterministic'?'deterministic':'original'):(cfg.default_source||'original'),pathId:h.get('path')||'1',notice:restoreNotice});
  }
  for(const [key,title]of metrics){const root=el('div',undefined,'range-row');root.append(el('span',title));const fields=el('div');for(const b of ['min','max']){const i=el('input');i.type='number';i.step='any';i.id=key+'-'+b;i.placeholder=b==='min'?'최소':'최대';i.setAttribute('aria-label',title+' '+i.placeholder);if(!originalAvailable&&key==='min_safe'){i.disabled=true;i.placeholder='판정 자료 없음';root.title='이 단계에는 후보별 정답 가능성 판정이 없어 safe q 필터를 제공하지 않습니다.';}i.onchange=()=>applyFilters();fields.append(i);}root.append(fields);$('ranges').append(root);}
  for(const[id,label]of regions){const b=el('button',label);b.dataset.region=id;b.onclick=()=>{toggleChoice(selectedRegions,id);};$('error-regions').append(b);}
  for(const [id,label,help]of presets){const b=el('button',label);b.dataset.preset=id;b.title=help;if(id==='disagree'&&!originalAvailable){b.disabled=true;b.title='이 단계에는 Qwen3 경로만 있어 두 모델 비교를 제공하지 않습니다.';}b.onclick=()=>{toggleChoice(selectedPresets,id);};$('presets').append(b);}
  $('path-availability').querySelector('[value="deterministic"]').disabled=!cfg.saved_paths;
  $('model').querySelector('[value="qwen25"]').disabled=!originalAvailable;
  $('decoding-source').querySelector('[value="original"]').disabled=!originalAvailable;
  $('filter-basis').textContent='아래 오류·정량 필터와 sample 목록의 수치는 '+(cfg.filter_source_label||'원본 greedy')+' 경로 기준입니다. 선택한 다른 경로의 수치는 본문에 표시합니다.';
  for(const ds of [...new Set(MANIFEST.map(s=>s.dataset))])$('dataset').add(new Option(ds,ds));$('scope').textContent=cfg.scope.replace('/',' · ')+' queries · within-path 우선';
  for(const id of ['model','dataset','endpoint','sort','path-availability'])$(id).onchange=()=>applyFilters();let timer;$('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>applyFilters(),150);};$('reset').onclick=()=>{selectedPresets.clear();selectedRegions.clear();$('search').value='';$('dataset').value='';$('sort').value='sample';$('path-availability').value='all';for(const[k]of metrics)for(const b of ['min','max'])$(k+'-'+b).value='';applyFilters();};
  $('prev-page').onclick=()=>{page--;renderList();};$('next-page').onclick=()=>{page++;renderList();};$('chart-range').onchange=()=>{renderPath();renderDistribution();remember();};$('distribution-mode').onchange=()=>{renderDistribution();remember();};$('step-slider').oninput=()=>setStep(displaySteps()[Number($('step-slider').value)].step);$('prev-step').onclick=()=>{const rows=displaySteps();setStep(rows[rows.findIndex(r=>r.step===step)-1].step);};$('next-step').onclick=()=>{const rows=displaySteps();setStep(rows[rows.findIndex(r=>r.step===step)+1].step);};$('jump-error').onclick=()=>setStep(activeMetrics().error_step);$('jump-branch').onclick=()=>setStep(activeCandidate?.branch_step);$('decoding-source').onchange=()=>{if(current)selectCase(current,undefined,{source:$('decoding-source').value,pathId:'1'});};$('decoding-path').onchange=()=>selectSavedPath($('decoding-path').value);$('path-reference').onchange=()=>{reference=$('path-reference').value;renderPath();remember();};$('play-gt').onclick=()=>play(current?.span[0]);$('play-pred').onclick=()=>play(trace?.timestamps?.[0]);$('video').ontimeupdate=()=>{if(segment!=null&&$('video').currentTime>=segment){$('video').pause();segment=null;}};
  window.addEventListener('resize',()=>{if(raw)renderPath();});window.addEventListener('hashchange',restore);window.TokenViewer={state:()=>({sample:current?.sample_key,model:$('model').value,endpoint:$('endpoint').value,step,preset:[...selectedPresets].join(',')||'all',errorRegion:[...selectedRegions].join(',')||'all',presets:[...selectedPresets],errorRegions:[...selectedRegions],count:filtered.length,visible:filtered.map(s=>s.sample_key),path:current?activeMetrics():null,source,pathId,reference,referencePathId:referenceId(),availablePaths:current&&savedAvailable(current)?savedLookup.samples[current.sample_key].paths:0,forcedSteps:activeCandidate?.forced_steps||[],pathAvailability:$('path-availability').value,ready:!!raw}),filter:applyFilters};
  savedLookupURL=cfg.saved_paths?absolute(cfg.saved_paths):null;Promise.all([get(absolute(cfg.catalog)),savedLookupURL?get(savedLookupURL):Promise.resolve({samples:{}})]).then(([data,saved])=>{catalog=data;savedLookup=saved;restore();}).catch(e=>$('status').textContent='목록 로드 실패: '+e.message);
})();
