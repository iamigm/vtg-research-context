/* Post-hoc ground-truth display; reads the cohort catalog, never inference inputs. */
(() => {
  const valid = x => Array.isArray(x) && x.length === 2 && x.every(Number.isFinite) && x[0] < x[1];
  const interval = x => valid(x) ? `${x[0].toFixed(3)} – ${x[1].toFixed(3)} s` : '대기 중';
  const overlap = (a, b) => {
    if (!valid(b)) return null;
    if (!valid(a)) return 0;
    const intersection = Math.max(0, Math.min(a[1], b[1]) - Math.max(a[0], b[0]));
    return intersection / (a[1] - a[0] + b[1] - b[0] - intersection);
  };
  const style = document.createElement('style');
  style.textContent = `
    .gt-comparison{padding:15px;border:2px solid #24784f;border-radius:10px;background:#f5fbf7}
    .gt-comparison h3{margin:0 0 12px;font-size:17px}.gt-comparison p{margin:8px 0}
    .gt-compare-row{display:grid;grid-template-columns:86px minmax(0,1fr) 210px;gap:10px;align-items:center;margin:12px 0}
    .gt-label{font-weight:700}.gt-value{min-width:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums;font-size:12px}
    .gt-value small{display:block}.gt-track{height:18px;position:relative;overflow:hidden;background:#e4e9ef;border-radius:4px}
    .gt-band{position:absolute;top:0;height:100%;min-width:2px;border-radius:3px}.gt-reference{position:absolute;top:5px;height:12px;background:#25835a;min-width:2px;border-radius:2px}
    .mode .timeline.gt-overlay{height:42px}.mode .timeline.gt-overlay>.range{top:25px;height:12px}
    .candidate-iou{display:block;margin:-3px 0 9px;font-size:12px;color:#344b61;font-variant-numeric:tabular-nums}
    .gt-seek{padding:5px 9px;font-size:12px}.gt-legend{font-size:12px;color:#496456}
    @media(max-width:600px){.gt-compare-row{grid-template-columns:74px minmax(0,1fr);gap:5px 8px}.gt-label{grid-row:span 2}.gt-value{grid-column:2}.gt-track{min-width:0}}
  `;
  document.head.appendChild(style);

  function paintGroundTruth(overlays = true) {
    if (!selected || !$('heading')) return;
    const current = data && (!data.sample || data.sample.sample_key === selected.key) ? data : null;
    const route = current?.results?.[$('method').value + ':' + $('seed').value];
    const result = route?.checkpoints?.[$('budget').value];
    const gt = selected.gt;
    const duration = Number(current?.clipping_duration ?? selected.duration);
    const percent = value => Math.min(100, Math.max(0, 100 * value / duration));
    const position = x => `left:${percent(x[0])}%;width:${Math.max(0, percent(x[1]) - percent(x[0]))}%;`;
    let panel = $('gt-comparison');
    if (!panel) {
      panel = document.createElement('section');
      panel.id = 'gt-comparison';
      panel.className = 'gt-comparison';
      $('heading').appendChild(panel);
    }
    const rows = [
      ['GT', gt, '#25835a', null, true],
      ['Zero-shot', current?.baseline?.span, '#718096', current?.baseline ? overlap(current.baseline.span, gt) : null, !!current?.baseline],
      ['최종 선택', result?.selection?.span, '#3974b9', result ? overlap(result.selection.span, gt) : null, !!result]
    ];
    panel.innerHTML = `<h3>GT와 예측 비교</h3>` + rows.map(([label, span, color, score, present]) =>
      `<div class="gt-compare-row"><span class="gt-label">${label}</span>` +
      `<div class="gt-track" role="img" aria-label="${label} ${interval(span)}">${valid(span) ? `<i class="gt-band" style="${position(span)}background:${color}"></i>` : ''}</div>` +
      `<span class="gt-value" title="${valid(span) ? span.join(' – ') : ''}">${valid(span) ? interval(span) : present ? '유효 interval 없음' : '대기 중'}${score !== null ? `<small>IoU vs GT ${score.toFixed(4)}</small>` : ''}</span></div>`
    ).join('') + `<p class="gt-legend">공통 시간축 0 – ${Number.isFinite(duration) ? duration.toFixed(3) : '—'} s · 초록: GT · 회색: zero-shot · 파랑: 최종 선택</p>` +
      `<button type="button" class="gt-seek" ${valid(gt) && current && $('video') ? '' : 'disabled'}>GT 시작으로 이동</button>` +
      `<p class="muted">GT는 사후 비교용입니다. 아래 후보 timeline의 위쪽 초록 막대가 GT이며, 아래쪽 막대가 해당 후보입니다.</p>`;
    panel.querySelector('.gt-seek').onclick = () => {
      const video = $('video');
      if (video && valid(gt)) video.currentTime = Math.max(0, gt[0]);
    };

    if (!overlays || !result || !valid(gt)) return;
    for (const kind of ['main', 'tail']) {
      const cards = document.querySelectorAll('.mode.' + kind);
      (result[kind + '_modes'] ?? []).forEach((group, index) => {
        const card = cards[index];
        if (!card) return;
        card.querySelectorAll('.candidate-iou').forEach(x => x.remove());
        card.querySelectorAll('.timeline').forEach((timeline, i) => {
          const prediction = group.representatives[i];
          timeline.classList.add('gt-overlay');
          timeline.querySelectorAll('.gt-reference').forEach(x => x.remove());
          const reference = document.createElement('i');
          reference.className = 'gt-reference';
          reference.style.cssText = position(gt);
          timeline.appendChild(reference);
          timeline.setAttribute('role', 'img');
          timeline.setAttribute('aria-label', `GT ${interval(gt)} · 후보 ${interval(prediction)}`);
          const score = overlap(prediction, gt);
          const annotation = document.createElement('small');
          annotation.className = 'candidate-iou';
          annotation.textContent = `IoU vs GT ${score === null ? '—' : score.toFixed(4)}`;
          timeline.after(annotation);
        });
      });
    }
    const table = document.querySelector('#content .rows table');
    if (table && !table.querySelector('.gt-iou-heading')) {
      const th = document.createElement('th');
      th.className = 'gt-iou-heading';
      th.textContent = 'IoU vs GT';
      table.querySelector('thead tr').appendChild(th);
    }
    table?.querySelectorAll('[data-draw]').forEach(row => {
      let cell = row.querySelector('.gt-iou-cell');
      if (!cell) {
        cell = document.createElement('td');
        cell.className = 'gt-iou-cell';
        row.appendChild(cell);
      }
      const draw = route.draws?.[Number(row.dataset.draw)];
      const score = draw ? overlap(current.sequences[draw.sequence]?.span, gt) : null;
      cell.textContent = score === null ? '—' : score.toFixed(4);
    });
  }

  const previousRender = render;
  render = function () { previousRender(); paintGroundTruth(); };
  const previousLoad = load;
  load = async function (...args) {
    const pending = previousLoad(...args);
    paintGroundTruth(false);
    await pending;
    paintGroundTruth();
  };
  paintGroundTruth();
})();
