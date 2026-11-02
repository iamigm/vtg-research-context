'use strict';
const $ = id => document.getElementById(id);
const cases = window.VIDEO_CASES;
const video = $('video');
let current, tracks = [], clip = null, domain = [0, 1];
const colors = {gt:'#18805a',qwen25:'#b96518',qwen3:'#276abe'};
const fmt = x => Number.isInteger(x) ? String(x) : x.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
const el = (tag, text, cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const interval = t => t.span ? `${fmt(t.span[0])}–${fmt(t.span[1])}s` : '파싱된 구간 없음';
const valid = t => t.span && t.span.every(Number.isFinite) && t.span[1] > t.span[0];
const playableEnd = () => Math.min(current.decoded_duration, Number.isFinite(video.duration) ? video.duration : Infinity);
const position = t => 100*(t-domain[0])/(domain[1]-domain[0]);
function iou(span, gt) {if(!span || span[1]<=span[0])return 0;return Math.max(0,Math.min(span[1],gt[1])-Math.max(span[0],gt[0]))/(Math.max(span[1],gt[1])-Math.min(span[0],gt[0]));}
function filtered() {const q=$('search').value.toLowerCase(),ds=$('dataset').value;return cases.filter(c=>(ds==='all'||c.sample_id.startsWith(ds+'/'))&&(c.sample_id+' '+c.query).toLowerCase().includes(q));}
function renderList() {
 const list=filtered();$('count').textContent=`${list.length} / ${cases.length}`;$('caseList').replaceChildren();
 for(const c of list){const b=el('button',undefined,'case'+(c===current?' selected':''));b.setAttribute('aria-pressed',String(c===current));b.append(el('strong',c.sample_id.replace('-timelens','')),el('span',c.query),el('small',c.review?'직접 검토한 사례':'회복·악화 비교 사례'));b.onclick=()=>selectCase(c);$('caseList').append(b);}
 if(!list.length)$('caseList').append(el('p','검색 결과가 없습니다.','hint'));
 const index=list.indexOf(current);$('previous').disabled=index<=0;$('next').disabled=index<0||index>=list.length-1;
}
function selectCase(c) {
 video.pause();clip=null;current=c;video.src=c.media;video.load();$('error').hidden=true;
 $('sampleId').textContent=c.sample_id;$('query').textContent=c.query;$('duration').textContent=`영상 ${c.decoded_duration.toFixed(2)}초 · GT ${interval({span:c.gt})}`;
 $('metadata').href=c.key+'.json';$('sheet').href=c.key+'.jpg';$('pdf').href=c.key+'.pdf';$('detailLink').replaceChildren();
 if(c.review?.detail_timestamps){const a=el('a','GT·예측 주변 확대 ↗');a.href=c.key+'_detail.jpg';$('detailLink').append(' · ',a);}
 $('reviewPanel').hidden=!c.review;
 if(c.review){$('observation').textContent=c.review.observation;$('hypothesis').textContent='해석 후보: '+c.review.hypothesis;$('limitation').textContent='검토 한계: '+c.review.limitation;}
 $('answers').replaceChildren();for(const m of c.models){const row=el('div',undefined,'answer');row.append(el('strong',`${m.model_key==='qwen25'?'Qwen2.5':'Qwen3'} · ${m.variant==='baseline'?'기본 입력':'TimeLens-7B식 입력'}`),el('p',m.answer));$('answers').append(row);}
 try{history.replaceState(null,'','#'+encodeURIComponent(c.sample_id));}catch(_){}
 renderList();renderTracks();$('playState').textContent='전체 영상 재생';update();
}
function renderTracks() {
 clip=null;const raw=$('timeMode').value==='raw';
 tracks=[{id:'gt',label:'GT',span:current.gt,color:colors.gt}];
 for(const [variant,model] of [['baseline','qwen25'],['timestamp','qwen25'],['baseline','qwen3']]){
  const m=current.models.find(x=>x.variant===variant&&x.model_key===model);if(!m)continue;
  const span=(raw?m.raw_timestamps:m.timestamps)?.[0]||null;
  tracks.push({id:model+'-'+variant,label:(model==='qwen25'?'Qwen2.5':'Qwen3')+' · '+(variant==='timestamp'?'TimeLens-7B식 입력':'기본'),span,color:colors[model],baseline:variant==='baseline'&&model==='qwen25',iou:iou(span,current.gt)});
 }
 const ends=tracks.flatMap(t=>t.span||[]).filter(Number.isFinite);domain=[Math.min(0,...ends),Math.max(current.decoded_duration,...ends)];
 $('tracks').replaceChildren();$('ruler').replaceChildren();$('ruler').setAttribute('aria-valuemax',String(current.decoded_duration));
 for(let i=0;i<=5;i++){const tick=el('span',fmt(domain[0]+(domain[1]-domain[0])*i/5));tick.style.left=i*20+'%';$('ruler').append(tick);}
 for(const t of tracks){
  const row=el('div',undefined,'track');row.dataset.track=t.id;
  const label=el('div',undefined,'track-name');label.append(el('strong',t.label),el('small',interval(t)));
  const outside=t.span&&(t.span[0]<0||t.span[1]>current.decoded_duration);
  if(!valid(t))label.append(el('small',t.span?'길이 0 또는 역전 구간':'구간 없음','status-note'));
  else if(outside)label.append(el('small','영상 범위 밖 예측 포함','status-note'));
  const plot=el('div',undefined,'track-plot');
  for(const [a,b] of [[domain[0],0],[current.decoded_duration,domain[1]]])if(b>a){const shade=el('div',undefined,'outside');shade.style.left=position(a)+'%';shade.style.width=(position(b)-position(a))+'%';plot.append(shade);}
  const end=el('div',undefined,'video-end');end.style.left=position(current.decoded_duration)+'%';plot.append(end);
  if(t.span){const b=el('button',undefined,'segment'+(t.baseline?' baseline':'')+(!valid(t)?' zero':''));b.style.setProperty('--color',t.color);b.style.left=position(Math.min(...t.span))+'%';b.style.width=Math.max(0,Math.abs(t.span[1]-t.span[0])/(domain[1]-domain[0])*100)+'%';b.title=`${t.label} ${interval(t)} · 구간 재생`;b.setAttribute('aria-label',b.title);b.disabled=!valid(t)||t.span[0]>=current.decoded_duration||t.span[1]<=0;b.onclick=()=>playSegment(t);plot.append(b);}
  const head=el('div',undefined,'playhead');plot.append(head);
  const score=el('div',t.id==='gt'?'기준':t.iou.toFixed(3),'score '+(t.id==='gt'||t.iou>=.5?'good':'bad'));row.append(label,plot,score);$('tracks').append(row);
 }
 $('playState').textContent='전체 영상 재생';update();
}
function seek(t){clip=null;video.currentTime=Math.max(0,Math.min(t,playableEnd()));$('playState').textContent='전체 영상 재생';update();}
function playSegment(t){const start=Math.max(0,t.span[0]),end=Math.min(playableEnd(),t.span[1]);if(end<=start)return;clip={id:t.id,start,end,label:t.label};video.currentTime=start;$('playState').textContent=`${t.label} · ${interval(t)} 재생`+(start!==t.span[0]||end!==t.span[1]?' (영상 내 부분만)':'');video.play().catch(e=>{$('error').hidden=false;$('error').textContent='재생하지 못했습니다: '+e.message;});update();}
function update(){if(!current)return;const now=video.currentTime||0;
 if(clip&&!video.paused&&now>=clip.end){if($('loop').checked){video.currentTime=clip.start;}else{video.pause();video.currentTime=clip.end;}}
 $('clock').textContent=`${now.toFixed(2)}s / ${current.decoded_duration.toFixed(2)}s`;$('play').textContent=video.paused?'▶ 재생':'Ⅱ 일시정지';$('ruler').setAttribute('aria-valuenow',now.toFixed(2));
 document.querySelectorAll('.playhead').forEach(n=>n.style.left=position(now)+'%');document.querySelectorAll('.track').forEach(n=>n.querySelector('.segment')?.classList.toggle('chosen',clip?.id===n.dataset.track));
 const active=tracks.filter(t=>valid(t)&&now>=t.span[0]&&now<t.span[1]);$('activeLabels').replaceChildren();for(const t of active){const b=el('span',t.label,'badge');b.style.setProperty('--color',t.color);$('activeLabels').append(b);}
}
function animate(){if(!video.paused)update();requestAnimationFrame(animate);}
for(const event of ['timeupdate','seeked','play','pause','loadedmetadata'])video.addEventListener(event,update);
video.addEventListener('ended',()=>{if(clip&&$('loop').checked){video.currentTime=clip.start;video.play();}});
video.addEventListener('error',()=>{$('error').hidden=false;$('error').textContent='영상을 읽지 못했습니다. cases/media 폴더가 함께 있는지 확인해주세요.';});
video.addEventListener('seeking',()=>{if(clip&&(video.currentTime<clip.start-.1||video.currentTime>clip.end+.1)){clip=null;$('playState').textContent='전체 영상 재생';}});
function togglePlay(){if(video.paused){if(clip&&video.currentTime>=clip.end-.02)video.currentTime=clip.start;video.play().catch(()=>{});}else video.pause();}
function move(delta){const list=filtered();const i=list.indexOf(current)+delta;if(i>=0&&i<list.length)selectCase(list[i]);}
function rulerSeek(e){const box=$('ruler').getBoundingClientRect();seek(domain[0]+Math.max(0,Math.min(1,(e.clientX-box.left)/box.width))*(domain[1]-domain[0]));}
let dragging=false;$('ruler').onpointerdown=e=>{dragging=true;$('ruler').setPointerCapture(e.pointerId);rulerSeek(e);};$('ruler').onpointermove=e=>{if(dragging)rulerSeek(e);};$('ruler').onpointerup=()=>dragging=false;$('ruler').onpointercancel=()=>dragging=false;
for(const id of ['search','dataset'])$(id).addEventListener(id==='search'?'input':'change',()=>{const list=filtered();if(list.length&&!list.includes(current))selectCase(list[0]);else renderList();});
for(const id of ['timeMode'])$(id).onchange=renderTracks;
function step(delta){seek((video.currentTime||0)+delta);}
$('back').onclick=()=>step(-1);$('forward').onclick=()=>step(1);$('play').onclick=togglePlay;$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('ruler').onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();if(e.key==='Home')seek(0);else if(e.key==='End')seek(playableEnd());else step(e.key==='ArrowLeft'?-1:1);}};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','VIDEO'].includes(e.target.tagName)||e.target.id==='ruler')return;if(e.code==='Space'){e.preventDefault();togglePlay();}else if(e.key==='ArrowRight')step(1);else if(e.key==='ArrowLeft')step(-1);});
let hash='';try{hash=decodeURIComponent(location.hash.slice(1));}catch(_){}
selectCase(cases.find(c=>c.sample_id===hash)||cases.find(c=>c.sample_id==='charades-timelens/81VSN/0')||cases[0]);requestAnimationFrame(animate);
