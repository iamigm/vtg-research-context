(() => {
  const $ = id => document.getElementById(id);
  const choices = [...document.querySelectorAll('input[name="compare"]')];
  const pages = new Map(choices.map(choice => {
    const article = choice.closest('article');
    return [choice.value, `${article.querySelector('h3').textContent} · ${article.querySelector('.run').textContent}`];
  }));
  let selected = [];
  for (const side of ['left', 'right']) {
    for (const [href, title] of pages) $(side + '-experiment').add(new Option(title, href));
    $(side + '-experiment').addEventListener('change', () => {
      open($('left-experiment').value, $('right-experiment').value);
    });
  }
  function updateSelection() {
    choices.forEach(choice => {
      choice.checked = selected.includes(choice.value);
      choice.disabled = selected.length === 2 && !choice.checked;
    });
    $('selection-status').textContent = selected.length === 2
      ? '2개 선택됨 · 두 계열을 함께 열 수 있습니다'
      : `비교할 실험 계열 2개를 선택하세요 (${selected.length}/2)`;
    $('open-comparison').disabled = selected.length !== 2;
    $('clear-selection').disabled = !selected.length;
  }
  choices.forEach(choice => choice.addEventListener('change', () => {
    selected = choice.checked ? [...selected, choice.value].slice(0, 2) : selected.filter(href => href !== choice.value);
    updateSelection();
  }));
  function render(left, right) {
    const comparing = pages.has(left) && pages.has(right);
    $('catalog').hidden = comparing;
    $('comparison').hidden = !comparing;
    document.body.classList.toggle('comparing', comparing);
    document.title = comparing ? '두 계열 비교 · 실험 목록' : '실험 목록';
    if (comparing) {
      selected = [...new Set([left, right])];
      for (const [side, href] of [['left', left], ['right', right]]) {
        $(side + '-experiment').value = href;
        $(side + '-open').href = href;
        const frame = $(side + '-frame');
        if (frame.getAttribute('src') !== href) {
          const next = document.createElement('iframe');
          next.id = frame.id;
          next.title = `${side === 'left' ? '실험 A' : '실험 B'}: ${pages.get(href)}`;
          next.src = href;
          frame.replaceWith(next);
        }
      }
    } else {
      for (const side of ['left', 'right']) {
        const frame = $(side + '-frame');
        if (frame.hasAttribute('src')) {
          const empty = frame.cloneNode(false);
          empty.removeAttribute('src');
          frame.replaceWith(empty);
        }
      }
    }
    updateSelection();
  }
  function open(left, right) {
    const url = new URL(location.href);
    url.searchParams.set('left', left);
    url.searchParams.set('right', right);
    history.pushState(null, '', url);
    render(left, right);
  }
  $('open-comparison').addEventListener('click', () => open(...selected));
  $('clear-selection').addEventListener('click', () => { selected = []; updateSelection(); });
  $('close-comparison').addEventListener('click', () => {
    const url = new URL(location.href);
    url.searchParams.delete('left');
    url.searchParams.delete('right');
    history.pushState(null, '', url);
    render();
  });
  $('swap-comparison').addEventListener('click', () => open($('right-experiment').value, $('left-experiment').value));
  function restore() {
    const params = new URLSearchParams(location.search);
    render(params.get('left'), params.get('right'));
  }
  window.addEventListener('popstate', restore);
  restore();
})();
