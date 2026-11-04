(async()=>{
 const $=id=>document.getElementById(id),root=$('family-app');let config,active,serial=0;const contexts=new Map();
 window.addEventListener('experiment-refresh',event=>{
  if(!active)return;
  event.preventDefault();show(active.id,true,location.hash,true);
 });
 $('toggle-sidebar').addEventListener('click',()=>{
  const sidebar=$('family-sidebar'),expanded=sidebar.hidden;
  sidebar.hidden=!expanded;
  root.classList.toggle('sidebar-collapsed',!expanded);
  $('toggle-sidebar').setAttribute('aria-expanded',String(expanded));
  $('toggle-sidebar').setAttribute('aria-label',expanded?'실험 목록 접기':'실험 목록 펼치기');
  $('toggle-sidebar').title=expanded?'실험 목록 접기':'실험 목록 펼치기';
 });
 $('analysis-tabs').addEventListener('click',event=>{
  const button=event.target.closest('button[data-view]');
  if(button&&!button.disabled&&button.dataset.view!==active?.id)show(button.dataset.view);
 });
 $('analysis-tabs').addEventListener('keydown',event=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  const tabs=[...$('analysis-tabs').querySelectorAll('button:not(:disabled)')];if(!tabs.length)return;
  event.preventDefault();const index=tabs.indexOf(document.activeElement);
  const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
  const view=tabs[next].dataset.view;show(view).then(()=>[...$('analysis-tabs').querySelectorAll('button')].find(b=>b.dataset.view===view)?.focus());
 });
 document.addEventListener('click',event=>{for(const id of ['reference-menu','view-browser'])if(!$(id).contains(event.target))$(id).open=false;});
 $('view-browser').addEventListener('toggle',()=>{if($('view-browser').open)positionBrowser();});
 $('close-view-browser').onclick=()=>{$('view-browser').open=false;$('view-browser').querySelector('summary').focus();};
 $('view-browser').addEventListener('keydown',event=>{if(event.key==='Escape'){$('view-browser').open=false;$('view-browser').querySelector('summary').focus();}});
 $('view-tree-items').addEventListener('click',event=>{const button=event.target.closest('button[data-view]');if(button&&!button.disabled){$('view-browser').open=false;show(button.dataset.view);}});
 $('reference-menu').addEventListener('keydown',event=>{if(event.key==='Escape'){$('reference-menu').open=false;$('reference-menu').querySelector('summary').focus();}});
 const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const urlFor=e=>new URL(e.href,location.href).href;
 function markdown(text,base){
  const tokens=[];const save=s=>{tokens.push(s);return `\u0000${tokens.length-1}\u0000`};
  function inline(raw){
   let s=escape(raw);
   s=s.replace(/`([^`]+)`/g,(_,t)=>save('<code>'+t+'</code>'));
   s=s.replace(/(!?)\[([^\]]*)\]\(([^\s)]+)(?:\s+&quot;[^)]*&quot;)?\)/g,(_,img,label,ref)=>{
    const decoded=ref.replace(/&amp;/g,'&');let url;
    try{url=new URL(decoded,base);if(!['http:','https:'].includes(url.protocol))return label}catch{return label}
    return save(img?`<img loading="lazy" src="${escape(url.href)}" alt="${label}">`:`<a href="${escape(url.href)}">${label}</a>`)
   });
   s=s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\*([^*]+)\*/g,'<em>$1</em>');
   return s.replace(/\u0000(\d+)\u0000/g,(_,i)=>tokens[+i]);
  }
  const lines=text.split('\n'),out=[];let i=0;
  const cells=line=>line.trim().replace(/^\||\|$/g,'').split('|');
  while(i<lines.length){const line=lines[i];
   if(!line.trim()){i++;continue}
   if(line.startsWith('```')){const code=[];i++;while(i<lines.length&&!lines[i].startsWith('```'))code.push(lines[i++]);i++;out.push('<pre><code>'+escape(code.join('\n'))+'</code></pre>');continue}
   const heading=line.match(/^(#{1,6})\s+(.*)$/);if(heading){out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);i++;continue}
   if(line.includes('|')&&i+1<lines.length&&/^\s*\|?\s*:?-{3}/.test(lines[i+1])){
    const header=cells(line);i+=2;const rows=[];while(i<lines.length&&lines[i].trim().startsWith('|'))rows.push(cells(lines[i++]));
    out.push('<table><thead><tr>'+header.map(c=>'<th>'+inline(c.trim())+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+inline(c.trim())+'</td>').join('')+'</tr>').join('')+'</tbody></table>');continue
   }
   if(/^\s*[-*]\s+/.test(line)){const rows=[];while(i<lines.length&&/^\s*[-*]\s+/.test(lines[i]))rows.push(lines[i++].replace(/^\s*[-*]\s+/,''));out.push('<ul>'+rows.map(r=>'<li>'+inline(r)+'</li>').join('')+'</ul>');continue}
   if(/^\s*\d+[.)]\s+/.test(line)){const rows=[];while(i<lines.length&&/^\s*\d+[.)]\s+/.test(lines[i]))rows.push(lines[i++].replace(/^\s*\d+[.)]\s+/,''));out.push('<ol>'+rows.map(r=>'<li>'+inline(r)+'</li>').join('')+'</ol>');continue}
   if(/^<!--/.test(line)){while(i<lines.length&&!lines[i].includes('-->'))i++;i++;continue}
   out.push('<p>'+inline(line)+'</p>');i++;
  }
  return out.join('\n');
 }
 const documentLabels={SPEC:'실험 설계',STRATA:'조건별 결과',END_DECIMAL_CASES:'소수점 분기 사례',SCHEMA:'데이터 구조',SPATIAL_AUDIT:'공간 입력 검증',CACHE_REPORT:'캐시 분석',PIPELINE_REVIEW:'파이프라인 검토',NEXT_EXPERIMENT:'후속 실험',ANALYSIS:'상세 분석',REPORT:'추가 보고서'};
 function indexNavigation(){
  if(!config.navigation)throw Error('탐색 구조가 없습니다. 결과 목록을 다시 생성하세요.');
  function visit(node,scope,labels=[]){
   if(node.kind==='scope')scope=node;
   if(node.kind==='analysis'){
    labels=[...labels,node.label];for(const id of node.entries)contexts.set(id,{node,scope,labels});
   }
   for(const child of node.children||[])visit(child,scope,labels);
  }
  visit(config.navigation,null);
 }
 function scopeShort(scope){return scope.queries?scope.id.split('/')[0]+scope.queries:scope.label;}
 function availability(){
  const all=new Map();
  function visit(node,scope,labels=[]){
   if(node.kind==='scope')scope=node;
   if(node.kind==='analysis'){
    labels=[...labels,node.label];
    if(node.result||node.report){const key=JSON.stringify(node.path);if(!all.has(key))all.set(key,{label:labels.length>1?labels.slice(1).join(' › '):node.label,results:[],reports:[]});const row=all.get(key);if(node.result)row.results.push(scopeShort(scope));else if(node.report)row.reports.push(scopeShort(scope));}
   }
   for(const child of node.children||[])visit(child,scope,labels);
  }
  visit(config.navigation,null);return [...all.values()];
 }
 function navigationButton(entryId,label,e){
  const button=document.createElement('button');button.type='button';button.textContent=label;button.dataset.view=entryId||'';button.disabled=!entryId;
  if(entryId===e.id)button.setAttribute('aria-current','page');
  if(entryId)button.title=config.entries.find(x=>x.id===entryId)?.title||label;
  return button;
 }
 function renderTree(e){
  const tree=$('view-tree-items');tree.replaceChildren();
  function analysisNode(node,scope){
   const li=document.createElement('li'),row=document.createElement('div');row.className='view-tree-row';row.dataset.analysisPath=node.path.join('/');
   if(node.entries.includes(e.id))row.classList.add('current');
   const name=document.createElement('span');name.textContent=node.label;row.append(name);
   if(node.result||node.report){const actions=document.createElement('div');actions.className='view-tree-actions';actions.append(navigationButton(node.result,'결과',e),navigationButton(node.report,'보고서',e));row.append(actions);}
   li.append(row);
   if(node.children.length){const ul=document.createElement('ul');for(const child of node.children)ul.append(analysisNode(child,scope));li.append(ul);}
   return li;
  }
  for(const mode of config.navigation.children){
   const group=document.createElement('section');group.className='view-tree-mode';const heading=document.createElement('h3');heading.textContent=mode.label;group.append(heading);
   for(const scope of mode.children){
    const section=document.createElement('section');section.className='view-tree-scope';section.dataset.scope=scope.id;
    const title=document.createElement('h4');title.textContent=scope.label;section.append(title);
    const list=document.createElement('ul');for(const node of scope.children)list.append(analysisNode(node,scope));section.append(list);group.append(section);
   }
   tree.append(group);
  }
  const rows=availability();$('view-availability').replaceChildren();
  for(const row of rows){const line=document.createElement('div'),label=document.createElement('b'),coverage=document.createElement('span');label.textContent=row.label;coverage.textContent=(row.results.length?'결과: '+row.results.join(' · '):'')+(row.reports.length?(row.results.length?' / ':'')+'보고서만: '+row.reports.join(' · '):'');line.append(label,coverage);$('view-availability').append(line);}
  $('view-availability-summary').textContent=rows.length<=3?rows.map(row=>row.label+': '+(row.results.length?row.results.join('·'):'보고서만')).join(' / '):rows.length+'개 분석 · 실제 결과가 있는 범위 확인';
 }
 function renderContext(e){
  const context=contexts.get(e.id);if(!context)throw Error('선택한 자료의 탐색 경로가 없습니다.');
  const {node,scope,labels}=context,tabs=$('analysis-tabs'),references=$('reference-list');tabs.replaceChildren();references.replaceChildren();
  const container=document.createElement('div');container.className='analysis-tab-group';container.dataset.analysis=e.analysis;
  const pair=document.createElement('div');pair.className='view-tabs';
  for(const [mode,id,label] of [['result',node.result,'결과'],['report',e.kind==='md'?e.id:node.report,'보고서']]){
   const button=navigationButton(id,label,e);button.id='tab-'+mode;button.dataset.view=id||'';
   const selected=id===e.id;button.setAttribute('role','tab');button.setAttribute('aria-selected',String(selected));button.setAttribute('aria-controls','view-panel');button.setAttribute('aria-label',node.label+' · '+label);button.tabIndex=selected?0:-1;pair.append(button);
  }
  container.append(pair);tabs.append(container);
  $('view-location').textContent=scope.label+' › '+labels.join(' › ');
  for(const id of node.documents){
   const extra=config.entries.find(x=>x.id===id),stem=id.split('/').pop().replace(/\.md$/i,'').toUpperCase(),link=document.createElement('a');
   link.textContent=documentLabels[stem]||extra.title;link.dataset.view=id;
   const url=new URL(location.href);url.searchParams.set('view',id);url.hash='';link.href=url;
   if(id===e.id)link.setAttribute('aria-current','page');
   link.onclick=event=>{if(event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;event.preventDefault();$('reference-menu').open=false;show(id);};references.append(link);
  }
  $('reference-menu').hidden=e.kind!=='md'||!node.documents.length;if(e.kind!=='md')$('reference-menu').open=false;
  $('reference-menu').querySelector('summary').textContent=node.documents.includes(e.id)?(documentLabels[e.id.split('/').pop().replace(/\.md$/i,'').toUpperCase()]||'참고'):'참고';
  $('view-panel').setAttribute('aria-labelledby',e.kind==='html'?'tab-result':'tab-report');renderTree(e);
 }
 function positionBrowser(){
  const box=$('view-browser').querySelector('summary').getBoundingClientRect(),panel=$('view-tree'),width=Math.min(720,innerWidth-24);
  panel.style.width=width+'px';panel.style.left=Math.max(12,Math.min(box.left,innerWidth-width-12))+'px';panel.style.top=(box.bottom+8)+'px';panel.style.maxHeight=Math.max(120,Math.min(innerHeight*.7,innerHeight-box.bottom-20))+'px';
 }
 window.addEventListener('resize',()=>{if($('view-browser').open)positionBrowser();});
 async function show(id,replace=false,detailHash,refresh=false){
  const e=config.entries.find(v=>v.id===id)||config.entries.find(v=>v.id===config.default)||config.entries[0];if(!e)return;
  if(detailHash===undefined&&active&&e.kind==='html'){try{detailHash=replace?location.hash:$('detail-frame').contentWindow.location.hash||location.hash}catch{detailHash=location.hash}}
  active=e;const call=++serial;renderContext(e);$('view-title').textContent=e.scope+' · '+e.title;$('view-title').title=$('view-title').textContent;
  $('source-md').hidden=e.kind!=='md';
  if(e.kind==='md')$('source-md').href=urlFor(e);else $('source-md').removeAttribute('href');
  const url=new URL(location.href);url.searchParams.set('view',e.id);url.hash=detailHash!==undefined?detailHash:(replace?location.hash:'');history[replace?'replaceState':'pushState'](null,'',url);
  $('original').href=e.kind==='md'?url.href:urlFor(e)+location.hash;
  $('original').title=$('original').ariaLabel=e.kind==='md'?'현재 보고서 새 탭에서 열기':'현재 결과 새 탭에서 열기';
  const frame=$('detail-frame'),report=$('report');frame.hidden=e.kind!=='html';report.hidden=e.kind==='html';$('message').hidden=true;
  if(e.kind==='html'){
   report.replaceChildren();frame.onload=()=>{try{const doc=frame.contentDocument;doc.querySelectorAll('[data-experiment-home]').forEach(n=>n.hidden=true);const top=doc.querySelector('body');if(top)top.dataset.familyEmbedded='true';const syncDetailHash=()=>{if(active?.id!==e.id||active.kind!=='html')return;const outer=new URL(location.href);outer.hash=frame.contentWindow.location.hash;history.replaceState(null,'',outer)};frame.contentWindow.addEventListener('hashchange',syncDetailHash);frame.contentWindow.addEventListener('rollout:context',syncDetailHash);syncDetailHash()}catch{}};
   // Only the outer selection owns a history entry. Iframe loads must not add
   // extra browser-history steps between the result and its report.
   const target=urlFor(e)+location.hash;
   if(frame.contentWindow.location.href!==target)frame.contentWindow.location.replace(target);
   else if(refresh)frame.contentWindow.location.reload();
  }else{
   try{frame.contentDocument?.querySelectorAll('video,audio').forEach(media=>media.pause())}catch{}
   report.textContent='보고서를 불러오는 중…';
   try{const response=await fetch(urlFor(e),{cache:refresh?'no-store':'default'});if(!response.ok)throw Error(response.status);const text=await response.text();if(call!==serial)return;report.innerHTML=markdown(text,urlFor(e));report.scrollTop=0;report.querySelectorAll('a').forEach(a=>{const target=config.entries.find(x=>new URL(urlFor(x)).pathname===new URL(a.href).pathname);if(target)a.onclick=ev=>{ev.preventDefault();show(target.id,false,new URL(a.href).hash)};else{a.target='_blank';a.rel='noopener'}})}catch(error){if(call===serial)report.textContent='보고서를 불러오지 못했습니다: '+error.message}
  }
 }
 try{
  const response=await fetch(root.dataset.manifest);if(!response.ok)throw Error(response.status);config=await response.json();
  indexNavigation();
  window.addEventListener('popstate',()=>show(new URL(location.href).searchParams.get('view'),true));
  await show(new URL(location.href).searchParams.get('view'),true);root.dataset.ready='true';
 }catch(error){$('message').hidden=false;$('message').textContent='결과 목록을 불러오지 못했습니다: '+error.message;root.dataset.error='true'}
})();
