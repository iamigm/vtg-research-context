/* Global family bookmarks, ordered using the index's saved project preferences. */
(() => {
  const sidebar = document.getElementById('family-sidebar');
  const nav = document.getElementById('experiment-nav');
  const search = document.getElementById('bookmark-search');
  const count = document.getElementById('bookmark-count');
  let catalog;
  document.getElementById('close-bookmarks').addEventListener('click', () => {
    document.getElementById('toggle-sidebar').click();
    document.getElementById('toggle-sidebar').focus();
  });
  function render() {
    if (!catalog) return;
    let preferences = {};
    try { preferences = JSON.parse(localStorage.getItem('experiment-project-layout')) || {}; } catch {}
    if (preferences.version !== 1) preferences = {};
    const defaults = catalog.project_order || [...new Set(catalog.families.map(family => family.project))];
    const order = [...new Set([...(Array.isArray(preferences.order) ? preferences.order : []), ...defaults])];
    const pinned = new Set(Array.isArray(preferences.pinned) ? preferences.pinned : []);
    order.sort((a, b) => Number(pinned.has(b)) - Number(pinned.has(a)));
    const query = search.value.trim().toLowerCase();
    const open = new Set([...nav.querySelectorAll('details[open]')].map(node => node.dataset.project));
    const fragment = document.createDocumentFragment();
    let visible = 0;
    for (const project of order) {
      const families = catalog.families.filter(family => family.project === project &&
        (family.project_label + ' ' + family.topic_title + ' ' + family.title + ' ' + family.search).toLowerCase().includes(query));
      if (!families.length) continue;
      const group = document.createElement('details');
      group.dataset.project = project;
      group.open = !!query || pinned.has(project) || open.has(project) || families.some(family => family.href === sidebar.dataset.current);
      group.classList.toggle('bookmark-pinned', pinned.has(project));
      const summary = document.createElement('summary');
      const label = document.createElement('span');
      label.className = 'bookmark-project-label';
      label.textContent = families[0].project_label;
      const meta = document.createElement('span'); meta.className = 'bookmark-project-meta';
      const favorite = document.createElement('span'); favorite.className = 'bookmark-favorite';
      favorite.textContent = '★';
      if (pinned.has(project)) {
        favorite.setAttribute('aria-label', '즐겨찾는 프로젝트'); favorite.title = '즐겨찾는 프로젝트';
      } else { favorite.classList.add('empty'); favorite.setAttribute('aria-hidden', 'true'); }
      const total = document.createElement('small'); total.textContent = families.length;
      meta.append(favorite, total); summary.append(label, meta); group.append(summary);
      const topics = new Map();
      for (const family of families) {
        if (!topics.has(family.topic_title)) topics.set(family.topic_title, []);
        topics.get(family.topic_title).push(family);
      }
      for (const [topic, members] of topics) {
        const section = document.createElement('section'); section.className = 'bookmark-topic';
        const heading = document.createElement('h3');
        const topicTitle = document.createElement('span'); topicTitle.textContent = topic;
        const topicCount = document.createElement('small'); topicCount.textContent = members.length;
        heading.append(topicTitle, topicCount); section.append(heading);
        const links = document.createElement('div'); links.className = 'bookmark-families';
        for (const family of members) {
          const link = document.createElement('a');
          link.href = new URL(family.href, new URL(sidebar.dataset.home, location.href));
          link.target = '_blank'; link.rel = 'noopener';
          link.dataset.family = family.href;
          link.title = family.title + ' · 새 탭에서 열기';
          if (family.href === sidebar.dataset.current) link.setAttribute('aria-current', 'page');
          const title = document.createElement('span'); title.textContent = family.title;
          const scope = document.createElement('small'); scope.textContent = family.scopes.join(' · ');
          link.append(title, scope); links.append(link); visible++;
        }
        section.append(links); group.append(section);
      }
      fragment.append(group);
    }
    if (!visible) {
      const empty = document.createElement('p'); empty.className = 'bookmark-status';
      empty.textContent = '일치하는 실험이 없습니다.'; fragment.append(empty);
    }
    nav.replaceChildren(fragment); count.textContent = visible + '개';
    sidebar.dataset.bookmarksReady = 'true';
  }
  search.addEventListener('input', render);
  window.addEventListener('storage', event => {
    if (event.key === 'experiment-project-layout' || event.key === null) render();
  });
  fetch(sidebar.dataset.catalog, {cache: 'no-store'}).then(response => {
    if (!response.ok) throw Error(response.status);
    return response.json();
  }).then(data => { catalog = data; render(); }).catch(() => {
    nav.textContent = '실험 목록을 불러오지 못했습니다.';
  });
})();
