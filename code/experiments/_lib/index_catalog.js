(() => {
  const $ = id => document.getElementById(id);
  const projects = [...document.querySelectorAll('details[data-project]')];
  const byId = new Map(projects.map(node => [node.dataset.project, node]));
  const defaults = projects.map(node => node.dataset.project);
  const key = 'experiment-project-layout';
  const pinned = $('pinned-projects'), other = $('other-projects');
  const status = $('order-status');
  let drag;

  function readLayout(raw) {
    try {
      const data = JSON.parse(raw);
      if (data?.version === 1 && Array.isArray(data.order) && Array.isArray(data.pinned)) {
        return {
          order: [...new Set([...data.order.filter(id => byId.has(id)), ...defaults])],
          pinned: new Set(data.pinned.filter(id => byId.has(id))),
        };
      }
    } catch {}
    return {order: defaults, pinned: new Set()};
  }
  function applyLayout(layout) {
    for (const id of layout.order) (layout.pinned.has(id) ? pinned : other).append(byId.get(id));
    updateActions();
  }
  function updateActions() {
    for (const list of [pinned, other]) {
      const rows = [...list.querySelectorAll('.project')];
      rows.forEach((row, index) => {
        const isPinned = list === pinned, button = row.querySelector('.pin-project');
        button.setAttribute('aria-pressed', String(isPinned));
        button.textContent = isPinned ? '★' : '☆';
        button.title = isPinned ? '상단 고정 해제' : '상단 고정';
        button.setAttribute('aria-label', row.dataset.label + ' ' + button.title);
        row.querySelector('.move-up').disabled = index === 0;
        row.querySelector('.move-down').disabled = index === rows.length - 1;
      });
    }
    pinned.querySelector('.pin-empty').hidden = !!pinned.querySelector('.project');
    // Keep the project filter in the same personal order as the catalog.
    for (const row of [...pinned.querySelectorAll('.project'), ...other.querySelectorAll('.project')]) {
      const option = [...$('project').options].find(option => option.value === row.dataset.project);
      if (option) $('project').append(option);
    }
  }
  function save(message) {
    updateActions();
    const order = [...pinned.querySelectorAll('.project'), ...other.querySelectorAll('.project')].map(row => row.dataset.project);
    const data = {version: 1, order, pinned: [...pinned.querySelectorAll('.project')].map(row => row.dataset.project)};
    try { localStorage.setItem(key, JSON.stringify(data)); status.classList.add('sr-only'); }
    catch { message += ' 브라우저 저장을 사용할 수 없어 현재 화면에만 적용됩니다.'; status.classList.remove('sr-only'); }
    status.textContent = message;
  }
  try { applyLayout(readLayout(localStorage.getItem(key))); } catch { applyLayout(readLayout(null)); }
  window.addEventListener('storage', event => {
    if (event.key === key && !drag) applyLayout(readLayout(event.newValue));
  });
  projects.forEach(row => {
    row.querySelector('summary').addEventListener('click', event => {
      if (event.target.closest('button')) event.preventDefault();
    });
    row.querySelector('.pin-project').addEventListener('click', () => {
      const wasPinned = row.parentElement === pinned;
      (wasPinned ? other : pinned).append(row);
      row.querySelector('.pin-project').focus();
      save(row.dataset.label + (wasPinned ? ' 고정을 해제했습니다.' : ' 프로젝트를 상단에 고정했습니다.'));
    });
    for (const [selector, direction] of [['.move-up', -1], ['.move-down', 1]]) {
      row.querySelector(selector).addEventListener('click', () => {
        const neighbor = direction < 0 ? row.previousElementSibling : row.nextElementSibling;
        if (!neighbor?.matches('.project')) return;
        if (direction < 0) neighbor.before(row); else neighbor.after(row);
        save(row.dataset.label + ' 프로젝트 순서를 저장했습니다.');
        // The moved action can become disabled at a boundary; retain keyboard focus.
        const action = row.querySelector(selector);
        (action.disabled ? row.querySelector('.pin-project') : action).focus();
      });
    }
    row.querySelector('.drag-handle').addEventListener('pointerdown', event => {
      if (event.button !== 0 || drag) return;
      const handle = event.currentTarget;
      drag = {row, handle, pointer: event.pointerId, startX: event.clientX, startY: event.clientY, x: event.clientX, y: event.clientY};
      handle.setPointerCapture(event.pointerId);
    });
  });
  function clearTargets() {
    document.querySelectorAll('.drop-before,.drop-after,.drop-zone').forEach(node => node.classList.remove('drop-before', 'drop-after', 'drop-zone'));
  }
  function findTarget() {
    clearTargets();
    const hit = document.elementFromPoint(drag.x, drag.y);
    let list = hit?.closest('.project-list');
    if (!list) list = hit?.closest('.project-zone')?.querySelector('.project-list');
    drag.target = null;
    if (!list) return;
    const row = hit.closest('.project');
    if (row && row !== drag.row) {
      const summary = row.querySelector('summary').getBoundingClientRect();
      const before = drag.y < summary.top + summary.height / 2;
      row.classList.add(before ? 'drop-before' : 'drop-after');
      drag.target = {list, row, before};
    } else if (!row) {
      list.classList.add('drop-zone');
      drag.target = {list};
    }
  }
  function scrollDrag() {
    if (!drag?.active) return;
    if (drag.y < 65) window.scrollBy(0, -12);
    else if (drag.y > innerHeight - 65) window.scrollBy(0, 12);
    findTarget();
    drag.raf = requestAnimationFrame(scrollDrag);
  }
  document.addEventListener('pointermove', event => {
    if (!drag || drag.pointer !== event.pointerId) return;
    drag.x = event.clientX; drag.y = event.clientY;
    if (!drag.active && Math.hypot(drag.x - drag.startX, drag.y - drag.startY) > 6) {
      drag.active = true;
      drag.ghost = document.createElement('div');
      drag.ghost.className = 'drag-ghost'; drag.ghost.textContent = drag.row.dataset.label;
      document.body.append(drag.ghost); document.body.classList.add('project-dragging');
      drag.row.classList.add('dragging'); scrollDrag();
    }
    if (drag.active) {
      event.preventDefault();
      drag.ghost.style.left = Math.min(drag.x + 14, innerWidth - drag.ghost.offsetWidth - 8) + 'px';
      drag.ghost.style.top = drag.y + 14 + 'px';
      findTarget();
    }
  }, {passive: false});
  function finish(cancel = false) {
    if (!drag) return;
    const {row, handle, target, pointer} = drag;
    cancelAnimationFrame(drag.raf);
    if (!cancel && drag.active && target) {
      if (target.row) {
        if (target.before) target.row.before(row); else target.row.after(row);
      } else target.list.append(row);
      save(row.dataset.label + ' 프로젝트 순서를 저장했습니다.');
      handle.focus();
    }
    drag.ghost?.remove(); row.classList.remove('dragging');
    document.body.classList.remove('project-dragging'); clearTargets();
    drag = null;
    if (handle.hasPointerCapture(pointer)) handle.releasePointerCapture(pointer);
  }
  document.addEventListener('pointerup', event => { if (drag?.pointer === event.pointerId) finish(); });
  document.addEventListener('pointercancel', () => finish(true));
  document.addEventListener('lostpointercapture', () => finish(true));
  document.addEventListener('keydown', event => { if (event.key === 'Escape') finish(true); });
  $('reset-project-order').addEventListener('click', () => { applyLayout(readLayout(null)); save('기본 프로젝트 순서로 되돌렸습니다.'); });

  const search = $('search'), project = $('project');
  function filter() {
    let n = 0, p = 0;
    const query = search.value.trim().toLowerCase();
    projects.forEach(group => {
      let visible = 0;
      group.querySelectorAll('article').forEach(row => {
        row.hidden = !!((project.value && project.value !== group.dataset.project) || !row.dataset.search.includes(query));
        if (!row.hidden) visible++;
      });
      group.querySelectorAll('[data-topic]').forEach(topic => {
        const count = [...topic.querySelectorAll('article')].filter(row => !row.hidden).length;
        topic.hidden = !count;
        topic.querySelector('.topic-count').textContent = count + '개 실험 계열';
      });
      group.hidden = !visible;
      group.open = !!visible && !!(query || project.value);
      if (visible) p++;
      n += visible;
    });
    $('count').textContent = `${p}개 프로젝트 · ${n}개 실험 계열`;
    $('empty').hidden = !!n;
    try { sessionStorage.setItem('experiment-filter', JSON.stringify([search.value, project.value])); } catch {}
  }
  try {
    const saved = JSON.parse(sessionStorage.getItem('experiment-filter'));
    if (Array.isArray(saved) && saved.length === 2) [search.value, project.value] = saved;
  } catch {}
  search.addEventListener('input', filter); project.addEventListener('change', filter); filter();
})();
