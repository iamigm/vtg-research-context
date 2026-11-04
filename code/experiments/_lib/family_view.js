(async()=>{
 const $=id=>document.getElementById(id),root=$('family-app');let config,active,serial=0;
 $('toggle-sidebar').addEventListener('click',()=>{
  const sidebar=$('family-sidebar'),expanded=sidebar.hidden;
  sidebar.hidden=!expanded;
  root.classList.toggle('sidebar-collapsed',!expanded);
  $('toggle-sidebar').setAttribute('aria-expanded',String(expanded));
  $('toggle-sidebar').setAttribute('aria-label',expanded?'목록 접기':'목록 펼치기');
  $('toggle-sidebar').title=expanded?'목록 접기':'목록 펼치기';
 });
 for(const id of ['tab-result','tab-report'])$(id).addEventListener('click',()=>{
  const target=$(id).dataset.view;if(target&&target!==active?.id)show(target);
 });
 document.querySelector('.view-tabs').addEventListener('keydown',event=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  const tabs=[...document.querySelectorAll('.view-tabs button:not(:disabled)')];
  if(!tabs.length)return;
  event.preventDefault();const index=tabs.indexOf(document.activeElement);
  const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
  tabs[next].focus();tabs[next].click();
 });
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
 function renderNav(){
  const scope=$('scope').value,groups=new Map();
  for(const e of config.entries){if(scope&&e.scope_key!==scope)continue;if(!groups.has(e.scope))groups.set(e.scope,[]);groups.get(e.scope).push(e)}
  $('view-nav').replaceChildren();
  for(const [label,entries] of groups){const d=document.createElement('details');d.open=true;const s=document.createElement('summary');s.textContent=label;d.append(s);
   for(const e of entries){const b=document.createElement('button');b.type='button';b.dataset.view=e.id;b.setAttribute('aria-current',String(active?.id===e.id));b.append(document.createTextNode(e.title));const sub=document.createElement('small');sub.textContent=e.analysis+' · '+(e.kind==='html'?'interactive':'Markdown report');b.append(sub);b.onclick=()=>show(e.id);d.append(b)}$('view-nav').append(d)
  }
 }
 function renderTabs(e){
  const report=e.kind==='md'?e:config.entries.find(v=>v.id===e.report);
  const result=e.kind==='html'?e:config.entries.find(v=>v.kind==='html'&&v.scope_key===e.scope_key&&v.report===e.id&&v.analysis===e.analysis)
   ||config.entries.find(v=>v.kind==='html'&&v.scope_key===e.scope_key&&v.report===e.id);
  for(const [id,entry] of [['tab-result',result],['tab-report',report]]){
   const button=$(id),selected=entry?.id===e.id;
   button.disabled=!entry;button.dataset.view=entry?.id||'';
   button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;
  }
  $('view-panel').setAttribute('aria-labelledby',e.kind==='html'?'tab-result':'tab-report');
 }
 async function show(id,replace=false,detailHash){
  const e=config.entries.find(v=>v.id===id)||config.entries.find(v=>v.id===config.default)||config.entries[0];if(!e)return;
  active=e;const call=++serial;$('scope').value=e.scope_key;renderNav();renderTabs(e);$('view-title').textContent=e.scope+' · '+e.title;$('view-title').title=$('view-title').textContent;$('original').href=urlFor(e);const reportPath=e.kind==='md'?e.href:e.report;$('source-md').hidden=!reportPath;if(reportPath)$('source-md').href=new URL(reportPath,location.href).href;
  const url=new URL(location.href);url.searchParams.set('view',e.id);url.hash=detailHash!==undefined?detailHash:(replace?location.hash:'');history[replace?'replaceState':'pushState'](null,'',url);
  const frame=$('detail-frame'),report=$('report');frame.hidden=e.kind!=='html';report.hidden=e.kind==='html';$('message').hidden=true;
  if(e.kind==='html'){
   report.replaceChildren();frame.onload=()=>{try{const doc=frame.contentDocument;doc.querySelectorAll('[data-experiment-home]').forEach(n=>n.hidden=true);const top=doc.querySelector('body');if(top)top.dataset.familyEmbedded='true';frame.contentWindow.addEventListener('hashchange',()=>{if(active?.id!==e.id||active.kind!=='html')return;const outer=new URL(location.href);outer.hash=frame.contentWindow.location.hash;history.replaceState(null,'',outer)})}catch{}};
   // Only the outer selection owns a history entry. Iframe loads must not add
   // extra browser-history steps between the result and its report.
   const target=urlFor(e)+location.hash;
   if(frame.contentWindow.location.href!==target)frame.contentWindow.location.replace(target);
  }else{
   try{frame.contentDocument?.querySelectorAll('video,audio').forEach(media=>media.pause())}catch{}
   report.textContent='보고서를 불러오는 중…';
   try{const response=await fetch(urlFor(e));if(!response.ok)throw Error(response.status);const text=await response.text();if(call!==serial)return;report.innerHTML=markdown(text,urlFor(e));report.scrollTop=0;report.querySelectorAll('a').forEach(a=>{const target=config.entries.find(x=>new URL(urlFor(x)).pathname===new URL(a.href).pathname);if(target)a.onclick=ev=>{ev.preventDefault();show(target.id,false,new URL(a.href).hash)};else{a.target='_blank';a.rel='noopener'}})}catch(error){if(call===serial)report.textContent='보고서를 불러오지 못했습니다: '+error.message}
  }
 }
 try{
  const response=await fetch(root.dataset.manifest);if(!response.ok)throw Error(response.status);config=await response.json();
  for(const [key,label] of new Map(config.entries.map(e=>[e.scope_key,e.scope])))$('scope').add(new Option(label,key));
  $('scope').onchange=()=>{const e=config.entries.find(v=>v.scope_key===$('scope').value&&v.kind==='html')||config.entries.find(v=>v.scope_key===$('scope').value);if(e)show(e.id)};
  window.addEventListener('popstate',()=>show(new URL(location.href).searchParams.get('view'),true));
  await show(new URL(location.href).searchParams.get('view'),true);root.dataset.ready='true';
 }catch(error){$('message').hidden=false;$('message').textContent='결과 목록을 불러오지 못했습니다: '+error.message;root.dataset.error='true'}
})();
