/* Shared step visualization. Reads the existing top-30 derived trace only. */
(() => {
  const fmt = x => Number(x).toFixed(3);
  const state = s => s.first_error ? '최초 오류' : s.prefix_correct ? '정답 가능성 유지' : '오류 이후';
  const color = s => s.first_error ? '#c24132' : s.prefix_correct ? '#247ba0' : '#89939e';
  const node = (tag, text, cls) => { const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(cls)e.className=cls; return e; };
  function styles() {
    if(document.getElementById('step-path-style'))return;
    const css=node('style');css.id='step-path-style';css.textContent=`
    .step-path{min-width:0;margin:20px 0;padding:18px;border:1px solid #cbd5e1;border-radius:12px;background:var(--surface,#fff);color:var(--text,#172b3a)}
    .step-path h2{margin:0 0 8px}.step-path h3{margin:14px 0 8px}.step-path p{line-height:1.7;margin:10px 0}
    .step-path .sp-scroll{overflow-x:auto;max-width:100%}.step-path .sp-tokens{display:flex;gap:8px;padding:8px 2px}
    .step-path button,.step-path select{font:inherit;padding:8px;border:1px solid #a9b7c3;border-radius:7px;background:#fff;color:#172b3a;cursor:pointer}
    .step-path button:focus-visible{outline:3px solid #5946b2;outline-offset:2px}.step-path button:disabled{opacity:.45;cursor:default}
    .step-path .sp-token{flex:0 0 104px;text-align:left;border-top:5px solid var(--step-color)}.step-path .sp-token strong{display:block;font:700 22px monospace;white-space:pre-wrap;overflow-wrap:anywhere}
    .step-path .sp-token small{display:block;font-size:12px}.step-path [aria-pressed=true]{outline:3px solid #5946b2;outline-offset:-2px}
    .step-path .sp-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.step-path input[type=range]{flex:1;min-width:100px;max-width:420px}
    .step-path .sp-chart{display:block;height:300px!important;max-width:none!important}.step-path .sp-note{padding:12px 16px;background:#f1f5f9;border-left:4px solid #64748b;color:#172b3a}
    .step-path .sp-detail{padding:12px 16px;border:1px solid #cbd5e1;border-radius:8px}.step-path .sp-prefix{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.step-path code{font-size:18px;overflow-wrap:anywhere}
    .step-path .sp-compare{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.step-path .sp-value{padding:10px;border:1px solid #cbd5e1;border-radius:8px;min-width:140px}.step-path .sp-value strong{display:block;font-size:24px}
    @media(max-width:600px){.step-path{padding:12px}.step-path .sp-token{flex-basis:96px}}`;
    document.head.append(css);
  }
  window.renderStepPath = (root, ep, selected, onselect) => {
    styles();root.classList.add('step-path');root.replaceChildren();
    if(!ep?.steps?.length){root.append(node('p','이 응답은 지원 표기 밖이므로 step별 정답 가능성 비교를 표시하지 않습니다.'));return;}
    const steps=ep.steps,idx=Math.max(0,steps.findIndex(s=>s.ar_step===selected)),s=steps[idx];
    const risk=steps.filter(x=>x.numeric&&x.previous_correct),mean=risk.reduce((a,x)=>a+x.entropy,0)/Math.max(1,risk.length);
    const error=steps.find(x=>x.first_error),prior=steps.filter(x=>x.numeric&&x.prefix_correct&&error&&x.ar_step<error.ar_step),previous=prior.at(-1);
    root.append(node('h2','생성 순서대로 entropy 살펴보기'),node('p','왼쪽부터 실제로 생성한 token입니다. token이나 그래프의 막대를 누르면 그 순간까지 쓴 답과 정답 가능성을 확인할 수 있습니다.'));
    const tokens=node('div',undefined,'sp-tokens'),strip=node('div',undefined,'sp-scroll');strip.append(tokens);
    for(const step of steps){
      const b=node('button',undefined,'sp-token');b.type='button';b.dataset.step=step.ar_step;b.style.setProperty('--step-color',color(step));b.style.setProperty('border-top','5px solid '+color(step),'important');b.setAttribute('aria-pressed',String(step.ar_step===s.ar_step));
      b.append(node('small','step '+step.ar_step+(step.numeric?' · 숫자':' · 서식')),node('strong',JSON.stringify(step.token)),node('small',state(step)),node('small',fmt(step.entropy)+' nat'));b.onclick=()=>onselect(step.ar_step);tokens.append(b);
    }
    root.append(strip);
    const controls=node('div',undefined,'sp-controls'),back=node('button','← 이전 step'),next=node('button','다음 step →'),slider=node('input');
    back.disabled=idx===0;next.disabled=idx===steps.length-1;back.onclick=()=>onselect(steps[idx-1].ar_step);next.onclick=()=>onselect(steps[idx+1].ar_step);
    slider.type='range';slider.min=0;slider.max=steps.length-1;slider.value=idx;slider.setAttribute('aria-label','생성 step 선택');slider.oninput=()=>onselect(steps[Number(slider.value)].ar_step);
    const label=node('label','그래프 기준 '),scale=node('select');scale.setAttribute('aria-label','entropy 그래프 기준');scale.add(new Option('entropy 원래 값','raw'));scale.add(new Option('같은 경로 평균과의 차이','centered'));scale.value=root.dataset.scale||'raw';label.append(scale);scale.onchange=()=>{root.dataset.scale=scale.value;window.renderStepPath(root,ep,s.ar_step,onselect);};controls.append(back,slider,next,label);root.append(controls);
    const centered=scale.value==='centered',values=steps.map(x=>x.entropy-(centered?mean:0)),min=centered?Math.min(0,...values)-.1:0,max=centered?Math.max(.1,...values)+.1:Math.log(30),width=Math.max(660,steps.length*94+84),height=300;
    const scroll=node('div',undefined,'sp-scroll'),svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('sp-chart');svg.style.width=width+'px';svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('role','img');svg.setAttribute('aria-label','생성 step별 top-30 entropy; 막대 선택으로 상세 확인');scroll.append(svg);root.append(scroll);
    const x=i=>80+(width-110)*(i+.5)/steps.length,y=v=>238-(v-min)*190/(max-min);
    const draw=(tag,attrs,text)=>{const e=document.createElementNS(svg.namespaceURI,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;};
    for(const v of centered?[min,0,max]:[0,1,2,3,Math.log(30)]){draw('line',{x1:65,x2:width-15,y1:y(v),y2:y(v),stroke:'#d8e0e6'});draw('text',{x:57,y:y(v)+4,'text-anchor':'end','font-size':12,fill:'#334155'},fmt(v));}
    draw('text',{x:14,y:17,'font-size':12,fill:'#334155'},centered?'평균과의 차이 (nat)':'entropy (nat)');
    draw('text',{x:width-18,y:291,'text-anchor':'end','font-size':12,fill:'#334155'},'생성 step →');
    steps.forEach((step,i)=>{
      const v=values[i],bar=draw('rect',{x:x(i)-23,y:Math.min(y(v),y(0)),width:46,height:Math.max(2,Math.abs(y(v)-y(0))),fill:color(step),stroke:step.ar_step===s.ar_step?'#5946b2':color(step),'stroke-width':step.ar_step===s.ar_step?4:1,tabindex:0,role:'button','aria-label':`step ${step.ar_step}, token ${JSON.stringify(step.token)}, ${state(step)}, ${fmt(step.entropy)} nat`,'data-step':step.ar_step});
      bar.style.cursor='pointer';bar.onclick=()=>onselect(step.ar_step);bar.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onselect(step.ar_step);}};
      draw('text',{x:x(i),y:Math.max(37,y(v)-8),'text-anchor':'middle','font-size':12,fill:'#172b3a'},fmt(v));
      draw('text',{x:x(i),y:260,'text-anchor':'middle','font-size':12,fill:'#172b3a'},step.ar_step+' · '+JSON.stringify(step.token));
      if(step.first_error)draw('text',{x:x(i),y:277,'text-anchor':'middle','font-size':12,fill:'#b42318'},'최초 오류');
    });
    const caption=node('p',undefined,'sp-note');caption.textContent=(centered?'위 그래프는 각 step의 entropy에서 이 endpoint의 오류 전·최초 오류 numeric step 평균 '+fmt(mean)+' nat을 뺀 값입니다. 0 위면 그 평균보다 높다는 뜻이며, 0은 정답·오답 임계값이 아닙니다.':'위 그래프는 token을 선택하기 직전의 top-30 entropy입니다. 막대가 높을수록 다음 token 후보 사이에 확률이 더 퍼져 있습니다. 세로축은 항상 0–ln(30) ≈ 3.401 nat이며, 빨간 막대가 가장 높아야 하는 것은 아닙니다.')+' 파랑은 선택 후 정답 가능성 유지, 빨강은 최초 상실, 회색은 오류 이후입니다. 소수점·종료 같은 서식도 표시합니다.';root.append(caption);
    const detail=node('div',undefined,'sp-detail');detail.dataset.selectedStep=s.ar_step;detail.append(node('h3','선택한 step '+s.ar_step+' · '+JSON.stringify(s.token)+' · '+state(s)));
    const before=idx?steps[idx-1].prefix:'',prefix=node('div',undefined,'sp-prefix');prefix.append(node('span','선택 전'),node('code',before||'(빈 prefix)'),node('span','→ 선택 후'),node('code',s.prefix||'(빈 prefix)'));detail.append(prefix);
    detail.append(node('p','이 step에서 '+JSON.stringify(s.token)+'을 선택하기 직전의 entropy는 '+fmt(s.entropy)+' nat입니다. GT endpoint는 '+ep.gt+'초이며 최종 출력은 '+ep.prediction+'초입니다.'));
    detail.append(node('p',s.first_error?'이 token을 선택하면서 반올림 정답으로 완성할 가능성이 처음 사라졌습니다. 색은 이 판정을 표시하며 entropy의 높낮이로 정한 것이 아닙니다.':s.prefix_correct?'이 prefix 뒤에 허용된 숫자·서식을 붙여 반올림 정답을 만들 가능성이 유지됩니다. 이미 정답을 완성했다거나 GT 문자열과 일치한다는 뜻은 아닙니다.':'앞선 step에서 이미 정답 가능성이 사라졌습니다. 이 step은 화면에 표시하지만 최초 오류의 경로 내부 비교에서는 제외합니다.'));root.append(detail);
    root.append(node('h3','이 sample 안에서 최초 오류 전후 비교'));
    if(error?.numeric&&previous){
      const delta=error.entropy-previous.entropy,compare=node('div',undefined,'sp-compare');
      for(const [title,value] of [[`직전 숫자 · step ${previous.ar_step} ${JSON.stringify(previous.token)}`,previous.entropy],[`최초 오류 · step ${error.ar_step} ${JSON.stringify(error.token)}`,error.entropy],['오류 − 직전 숫자',delta]]){const box=node('div',undefined,'sp-value');box.append(node('span',title),node('strong',(title.startsWith('오류')&&value>0?'+':'')+fmt(value)+' nat'));compare.append(box);}
      root.append(compare,node('p',`이 sample에서는 최초 오류의 entropy가 직전 정답 가능 숫자보다 ${fmt(Math.abs(delta))} nat ${delta>0?'높았습니다':delta<0?'낮았습니다':'차이가 없었습니다'}. 가운데 소수점 등 서식은 이 두 숫자 비교에서 건너뜁니다. 이 차이가 전체 보고서의 Δprevious에 들어갑니다.`, 'sp-note'));
    } else root.append(node('p',!error?'이 endpoint에는 최초 오류가 없습니다. 따라서 오류 전후 차이를 계산하지 않습니다.':!error.numeric?'최초 오류가 숫자가 아닌 서식 step입니다. 이 sample은 numeric 오류의 paired 평균에서 제외됩니다.':'첫 숫자부터 오류여서 이전 정답 가능 숫자가 없습니다. 오류 전후 차이를 계산하지 않습니다.','sp-note'));
    // Keep the selected step visible inside each horizontal scroller on mobile.
    strip.scrollLeft=Math.max(0,tokens.children[idx].offsetLeft-tokens.children[0].offsetLeft-strip.clientWidth/2+52);
    scroll.scrollLeft=Math.max(0,x(idx)-scroll.clientWidth/2);
  };
})();
