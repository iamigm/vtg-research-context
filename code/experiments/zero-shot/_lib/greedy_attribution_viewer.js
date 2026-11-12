'use strict';

(() => {
  const config = window.ATTRIBUTION;
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[character]));
  const finite = value => value !== null && value !== undefined && Number.isFinite(Number(value));
  const number = (value, digits = 3) => finite(value) ? Number(value).toFixed(digits) : '—';
  const scientific = value => finite(value)
    ? (Math.abs(value) > 0 && Math.abs(value) < 0.0001 ? Number(value).toExponential(2) : number(value, 4))
    : '—';
  const percent = value => finite(value) ? `${(100 * value).toFixed(value < 0.0001 ? 5 : 2)}%` : '—';
  const signed = value => finite(value) ? `${value > 0 ? '+' : ''}${scientific(value)}` : '—';
  const empty = message => `<p class="empty">${escape(message)}</p>`;
  const sourceLabel = group => ({
    video_pixels: 'Video pixels', video: 'Visual embedding', visual: 'Visual embedding',
    query: 'Query', timestamp: 'Timestamp', instruction: 'Instruction', prompt: 'Instruction',
    generated: 'Generated prefix', special: 'Special token', other: 'Other'
  }[group] || group);
  const datasetLabel = value => ({
    'charades-timelens': 'Charades', 'activitynet-timelens': 'ActivityNet',
    'qvhighlights-timelens': 'QVHighlights'
  }[value] || value);
  const resolve = (path, base = location.href) => new URL(path, base).href;
  const rootURL = resolve(`${config.root}/`);
  const storeURL = resolve(`${config.store}/`);
  let initialParams = new URLSearchParams(location.hash.slice(1) || location.search);
  const PAGE_SIZE = 10;
  const POSITION_PAGE_SIZE = 160;
  let catalog = {}, samples = [], filtered = [], listPage = 0;
  let record = null, interpretation = null, selected = null;
  let step = 0, targetIndex = -1, positionPage = 0, frameIndex = 0, loadSerial = 0;
  const payloadPromises = new WeakMap();

  if (window.self !== window.top) document.body.classList.add('embedded');

  async function fetchJSON(url, optional = false) {
    const response = await fetch(url, {cache: 'no-store'});
    if (!response.ok) {
      if (optional && response.status === 404) return null;
      throw new Error(`${response.status}: ${url}`);
    }
    return response.json();
  }

  const steps = () => record?.steps || [];
  const targets = () => record?.targets || [];
  const isBranch = target => target?.kind === 'logit_margin' && target.alternative !== null;
  const targetStep = target => Number(target?.step ?? target?.position ?? -1);
  const chosenID = item => typeof item.chosen === 'object' ? item.chosen.id : item.chosen ?? item.token_id ?? item.id;
  const tokenText = item => item.chosen?.token ?? item.token ?? item.token_text ?? item.text ?? String(chosenID(item) ?? '');
  const spanText = span => Array.isArray(span) ? `${number(span[0], 2)} – ${number(span[1], 2)} s` : '미관측';
  const epsilon = () => Number(catalog.config?.screen_epsilon ?? 0.03125);

  function updateURL() {
    const url = new URL(location.href), state = new URLSearchParams();
    state.set('sample', selected?.key || '');
    state.set('step', String(step));
    state.set('target', String(targetIndex));
    url.hash = state.toString();
    history.replaceState(null, '', url);
    window.dispatchEvent(new Event('rollout:context'));
  }

  function renderSummary() {
    const summary = catalog.summary || {};
    const completed = summary.completed ?? samples.filter(sample => sample.complete).length;
    const items = [
      [`${completed} / ${config.scope}`, '계측 완료 query'],
      [summary.branch_targets ?? '—', '상세 계측 분기'],
      [summary.targets ?? '—', 'Backward target'],
      [`${summary.completed_unique_videos ?? '—'} / ${summary.unique_videos ?? '—'}`, '완료 / 계획 고유 영상']
    ];
    $('summary').innerHTML = items.map(([value, label]) => `<div><b>${escape(value)}</b><span>${escape(label)}</span></div>`).join('');
    const counts = new Map();
    for (const sample of samples) {
      const count = counts.get(sample.dataset) || {total: 0, complete: 0};
      count.total++;
      count.complete += Number(sample.complete);
      counts.set(sample.dataset, count);
    }
    $('dataset-progress').textContent = [...counts].map(([name, count]) => `${datasetLabel(name)} ${count.complete}/${count.total} query`).join(' · ')
      + (completed < config.scope ? ' · 진행 중인 부분집합입니다. Dataset 전체 비교로 해석하지 않습니다.' : ' · 같은 고정 cohort의 누적 결과입니다.');
    $('status').textContent = completed === config.scope ? '계측 완료 · 검증 상태는 보고서 참조' : `진행 중 · ${completed}/${config.scope} query`;
    $('sample-count').textContent = `pilot ${config.scope}`;
  }

  function renderList() {
    const search = $('filter').value.toLowerCase();
    const dataset = $('dataset').value, completion = $('completion').value;
    filtered = samples.filter(sample =>
      (!search || `${sample.key} ${sample.query}`.toLowerCase().includes(search)) &&
      (!dataset || sample.dataset === dataset) &&
      (!completion || (completion === 'complete') === sample.complete)
    );
    listPage = Math.max(0, Math.min(listPage, Math.ceil(filtered.length / PAGE_SIZE) - 1));
    $('list-count').textContent = `${filtered.length} query`;
    $('page-count').textContent = `${listPage + 1} / ${Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))}`;
    $('list-prev').disabled = listPage === 0;
    $('list-next').disabled = (listPage + 1) * PAGE_SIZE >= filtered.length;
    $('samples').innerHTML = filtered.slice(listPage * PAGE_SIZE, (listPage + 1) * PAGE_SIZE).map(sample => `
      <button class="sample ${selected?.key === sample.key ? 'selected' : ''}" data-key="${escape(sample.key)}">
        <span class="query">${escape(sample.query)}</span>
        <span>${escape(datasetLabel(sample.dataset))} · ${sample.complete ? '계측 완료' : '대기·진행'}</span>
        <span>${escape(sample.key)}</span>
      </button>`).join('') || empty('검색 결과가 없습니다.');
  }

  // A matching descendant alone never establishes a direct conditional completion.
  function branchFacts(target) {
    const verified = interpretation?.branches?.find(branch => branch.target_id === target.id);
    if (verified) return verified;
    const sourceStep = steps()[targetStep(target)] || {}, branch = target.branch || {};
    return {
      target_id: target.id, step: targetStep(target), prefix: sourceStep.prefix,
      greedy: {...sourceStep.chosen, span: record.greedy?.span},
      alternative: {id: target.alternative, token: branch.token, probability: branch.probability, source_probability: branch.source_probability},
      margin: {value: target.score, source: branch.source_margin, sensitive_to_replay: branch.margin_sensitive_to_replay},
      competition: {
        current: !!branch.current_competitor, potential: !!branch.potential_competitor,
        potential_only: !!branch.potential_competitor && !branch.current_competitor,
        probe_only: !branch.current_competitor && !branch.potential_competitor,
        screen_gain: branch.screen_gain, screen_gap: branch.screen_gap,
        screen_direction: branch.screen_direction, screen_sign: branch.screen_sign,
        observed_best_gap: branch.observed_best_gap,
        observed_pairwise_reversal: finite(branch.observed_best_gap) ? branch.observed_best_gap < 0 : null
      },
      continuation: {status: branch.mode?.candidates?.length ? 'provenance_unverified' : 'unobserved', descendant_count: 0},
      comparison: {greedy_span: record.greedy?.span, alternative_span: null, direct_distinct_mode_candidate: null},
      groups: target.groups || [], pixels: {signed_sum: target.pixel_signed_sum, grad_l2: target.pixel_gradient_l2}
    };
  }

  function continuationLabel(facts) {
    const status = facts.continuation?.status;
    if (status === 'direct') return facts.comparison?.direct_distinct_mode_candidate ? '직접 continuation · 구간 분리' : '직접 continuation';
    return {
      descendants_only: '자손 경로만 관측', unobserved: 'Continuation 미관측',
      provenance_unverified: '직접 결과 여부 미확인', conflicting_direct: '직접 경로 기록 충돌'
    }[status] || 'Continuation 미관측';
  }

  function competitionBadges(facts) {
    const competition = facts.competition || {}, labels = [];
    if (competition.current) labels.push('<span class="badge">현재 경쟁</span>');
    if (competition.potential) labels.push('<span class="badge potential">잠재 경쟁</span>');
    if (competition.probe_only) labels.push('<span class="badge probe">낮은 probability probe</span>');
    return labels.join(' ');
  }

  function defaultTarget() {
    const branches = targets().map((target, index) => ({target, index})).filter(item => isBranch(item.target));
    if (!branches.length) return targets().length ? 0 : -1;
    function tier(facts) {
      if (facts.continuation?.status === 'direct' && facts.comparison?.direct_distinct_mode_candidate) return 0;
      if (facts.continuation?.status === 'direct') return 1;
      return 2;
    }
    branches.sort((left, right) => {
      const a = branchFacts(left.target), b = branchFacts(right.target);
      return tier(a) - tier(b) || (b.alternative?.source_probability ?? b.alternative?.probability ?? 0)
        - (a.alternative?.source_probability ?? a.alternative?.probability ?? 0) || left.index - right.index;
    });
    return branches[0].index;
  }

  function renderBranchCards() {
    const branches = targets().map((target, index) => ({target, index})).filter(item => isBranch(item.target));
    $('branch-panel').hidden = !branches.length;
    $('branch-count').textContent = `${branches.length}개 · GT를 선택에 사용하지 않음`;
    $('branch-cards').innerHTML = branches.map(({target, index}) => {
      const facts = branchFacts(target), greedy = facts.greedy || {}, alternative = facts.alternative || {};
      const directSpan = facts.continuation?.status === 'direct' ? facts.comparison?.alternative_span : null;
      const result = directSpan ? `<strong>${spanText(facts.comparison.greedy_span)}</strong><span>→</span><strong>${spanText(directSpan)}</strong>`
        : `<strong>${spanText(facts.comparison?.greedy_span)}</strong><span>→</span><strong class="unknown">직접 결과 미관측</strong>`;
      return `<button class="branch-card ${targetIndex === index ? 'selected active' : ''}" data-branch-target="${index}" aria-pressed="${targetIndex === index}">
        <span class="branch-card-top"><span>Step ${facts.step}</span><span class="card-margin">Margin ${number(facts.margin?.value)}</span></span>
        <span class="token-pair"><span><small>greedy</small><code>${escape(greedy.token)}</code><b>${percent(greedy.source_probability ?? greedy.probability)}</b></span><i>↔</i><span><small>alternative</small><code>${escape(alternative.token)}</code><b>${percent(alternative.source_probability ?? alternative.probability)}</b></span></span>
        <span class="branch-result">${result}</span><span class="continuation-status">${escape(continuationLabel(facts))}</span>
        <span class="card-badges">${competitionBadges(facts)}</span>
      </button>`;
    }).join('');
    $('branch-default-note').textContent = '기본 선택: 직접 continuation의 구간 분리가 확인된 분기 중 alternative probability가 가장 큰 항목. 해당 항목이 없으면 직접 관측된 분기, 그다음 전체 계측 분기에서 같은 기준을 적용합니다.';
  }

  function renderTimeline(alternativeSpan = null) {
    const duration = Number(selected?.duration || 1);
    const rows = [['GT', selected?.gt, 'gt'], ['greedy', record?.greedy?.span || selected?.greedy_span, 'greedy']];
    if (alternativeSpan) rows.push(['alternative', alternativeSpan, 'alternative']);
    $('timeline').innerHTML = rows.map(([label, span, kind]) => `
      <div class="timeline-row"><span>${label}</span><div class="timeline-track">${Array.isArray(span) ?
        `<button class="timeline-span ${kind}" data-seek="${Math.max(0, span[0])}" aria-label="${label} 시작 시각으로 이동" style="left:${Math.max(0, span[0]) / duration * 100}%;width:${Math.max(0, Math.min(duration, span[1]) - Math.max(0, span[0])) / duration * 100}%"></button>` : ''}</div></div>
    `).join('') + `<div class="timeline-scale"><span>0 s</span><span>${number(duration, 1)} s</span></div>`;
  }

  async function selectSample(key, initial = false) {
    const sample = samples.find(item => item.key === key);
    if (!sample) return;
    selected = sample;
    record = null;
    interpretation = null;
    const serial = ++loadSerial;
    positionPage = 0;
    frameIndex = 0;
    step = 0;
    targetIndex = -1;
    $('workspace').classList.add('loading');
    renderList();
    $('query-title').textContent = sample.query;
    $('query-meta').textContent = `${datasetLabel(sample.dataset)} · ${sample.key} · ${number(sample.duration, 2)} s`;
    $('query-state').textContent = sample.complete ? '' : '이 query의 계측이 아직 완료되지 않았습니다.';
    for (const id of ['branch-panel', 'token-panel', 'attribution-panel', 'frame-panel', 'layer-panel', 'message-panel', 'record-panel']) $(id).hidden = true;
    for (const id of ['group-details', 'position-details', 'token-details', 'layer-details', 'message-details']) $(id).open = false;
    $('receiver').replaceChildren();
    if (sample.video) {
      $('video').src = resolve(sample.video, rootURL);
      $('video-status').textContent = '원본 시간축 · 구간 막대와 frame을 눌러 영상 위치를 이동할 수 있습니다.';
    } else {
      $('video').removeAttribute('src');
      $('video-status').textContent = '영상 연결 없음';
    }
    $('answer-cards').innerHTML = `<div><span>GT · 사후 참조</span><b>${spanText(sample.gt)}</b></div><div><span>Source greedy</span><b>${spanText(sample.greedy_span)}</b></div>`;
    renderTimeline();
    $('validation').textContent = '';
    if (!sample.complete || !sample.record) {
      $('workspace').classList.remove('loading');
      updateURL();
      return;
    }
    try {
      const query = await fetchJSON(resolve(sample.record, rootURL));
      if (serial !== loadSerial) return;
      record = query;
      try {
        const interpretationURL = sample.interpretation ? resolve(sample.interpretation, rootURL)
          : resolve(`web/interpretations/${encodeURIComponent(key)}.json`, storeURL);
        interpretation = await fetchJSON(interpretationURL, true);
      } catch {
        interpretation = null;
      }
      if (serial !== loadSerial) return;
      if (interpretation && interpretation.sample_key !== key) throw new Error('Interpretation sample key mismatch');
      $('answer-cards').innerHTML = `<div><span>GT · 사후 참조</span><b>${spanText(sample.gt)}</b></div><div><span>Source greedy · IoU ${number(sample.greedy_iou)}</span><b>${spanText(query.greedy?.span)}</b></div>`;
      const explicitTarget = initial && initialParams.has('target') ? Number(initialParams.get('target')) : null;
      const explicitStep = initial && initialParams.has('step') ? Number(initialParams.get('step')) : null;
      if (explicitTarget !== null && targets()[explicitTarget]) {
        targetIndex = explicitTarget;
        step = targetStep(targets()[targetIndex]);
      } else if (explicitStep !== null && steps()[explicitStep]) {
        step = explicitStep;
        targetIndex = targets().findIndex(target => targetStep(target) === step);
      } else {
        targetIndex = defaultTarget();
        step = targetStep(targets()[targetIndex]);
      }
      renderRecord();
      renderBranchCards();
      $('token-panel').hidden = false;
      $('record-panel').hidden = false;
      await renderStep(true);
      $('query-state').textContent = '';
    } catch (error) {
      $('query-state').textContent = `원자료를 불러오지 못했습니다: ${error.message}`;
    } finally {
      if (serial === loadSerial) $('workspace').classList.remove('loading');
    }
  }

  async function selectBranch(index, scroll = false) {
    if (!isBranch(targets()[index])) return;
    targetIndex = index;
    step = targetStep(targets()[index]);
    positionPage = 0;
    await renderStep(true);
    if (scroll) $('attribution-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
  }

  async function renderStep(keepTarget = false) {
    const current = steps()[step];
    if (!current) return;
    $('token-count').textContent = `${steps().length} step`;
    const measured = new Set(targets().map(targetStep));
    $('sentence').innerHTML = steps().map((item, index) => `
      <button class="token ${index === step ? 'selected' : ''} ${measured.has(index) ? 'measured' : ''}" data-step="${index}" title="Step ${index} · p=${scientific(item.chosen?.probability)}">${escape(tokenText(item)).replace(/ /g, '&nbsp;')}</button>
    `).join('');
    $('step-label').textContent = `Step ${step} · source p=${scientific(current.chosen?.source_probability)} → replay p=${scientific(current.chosen?.probability)}`;
    $('prefix').textContent = current.prefix ?? steps().slice(0, step).map(tokenText).join('');
    $('token-prev').disabled = step === 0;
    $('token-next').disabled = step >= steps().length - 1;
    $('distribution-meta').textContent = `Top30 mass ${percent(current.top30_mass)} · omitted ${percent(current.omitted_mass)} · entropy contribution ${number(current.top30_entropy_contribution)} nats`;
    const byID = new Map((current.top || []).map(candidate => [candidate.id, candidate]));
    for (const candidate of current.candidates || []) byID.set(candidate.id, {...byID.get(candidate.id), ...candidate});
    $('branches').innerHTML = [...byID.values()].sort((a, b) => b.probability - a.probability).map(candidate => {
      const isChosen = candidate.id === chosenID(current);
      const found = targets().findIndex(target => targetStep(target) === step && target.alternative === candidate.id);
      const reason = candidate.selection_reason || '분포 기록';
      return `<tr class="${isChosen ? 'branch-greedy' : ''}"><td><code>${escape(candidate.token)}</code>${isChosen ? ' <span class="badge">greedy</span>' : ''}</td>
        <td>${percent(candidate.source_probability)}</td><td>${percent(candidate.probability)}<div class="metric-tip">Δ ${number(100 * (candidate.probability - candidate.source_probability))} pp</div></td>
        <td>${number(candidate.source_margin ?? (isChosen ? 0 : null))} → ${number(candidate.margin ?? (isChosen ? 0 : null))}${candidate.margin_sensitive_to_replay ? '<span class="badge potential">재현 오차 민감</span>' : ''}${finite(candidate.screen_gap) ? `<div class="metric-tip">Screen gap ${number(candidate.screen_gap)}</div>` : ''}</td>
        <td><span class="badge ${candidate.potential_competitor ? 'potential' : ''}">${escape(reason)}</span>${candidate.screen_direction ? `<div class="metric-tip">${escape(candidate.screen_direction)} ${candidate.screen_sign > 0 ? '+' : '−'}</div>` : ''}</td>
        <td>${found >= 0 ? `<button data-target="${found}">계측 보기</button>` : '—'}</td></tr>`;
    }).join('');
    $('screening-note').textContent = 'Source KV의 top30 token을 고정하고 원래 prefill과 one-token decoding을 재현합니다. Probability를 top30 안에서 재정규화하지 않습니다. Top30 밖은 미계측이며 entropy contribution은 부분합입니다.';
    const available = targets().map((target, index) => ({target, index})).filter(item => targetStep(item.target) === step || targetStep(item.target) < 0);
    $('target').innerHTML = available.map(({target, index}) => `<option value="${index}">${escape(target.kind)}${target.branch?.token ? ` · ${escape(target.branch.token)}` : ''}</option>`).join('');
    if (!keepTarget || !available.some(item => item.index === targetIndex)) targetIndex = available[0]?.index ?? -1;
    $('target').value = String(targetIndex);
    renderBranchCards();
    await renderTarget();
    updateURL();
  }

  async function loadTarget(target) {
    if (!target.data_file || target._loaded) return target;
    if (!payloadPromises.has(target)) payloadPromises.set(target, (async () => {
      const response = await fetch(resolve(target.data_file, storeURL));
      if (!response.ok) throw new Error(String(response.status));
      const payload = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
      Object.assign(target, payload, {_loaded: true});
      return target;
    })());
    return payloadPromises.get(target);
  }

  function renderNarrative(target) {
    if (!isBranch(target)) {
      $('branch-title').textContent = '선택된 token의 민감도';
      $('branch-narrative').innerHTML = `<p>현재 목표는 <b>${escape(target.kind)}</b>입니다. 선택된 token 또는 endpoint의 log probability에 대한 민감도를 표시합니다. 대안과의 차이는 위 분기 카드를 선택해 확인하세요.</p>`;
      $('target-metrics').innerHTML = `<div><span>목표값</span><b>${scientific(target.score)}</b></div>`;
      $('screening-summary').innerHTML = '';
      renderTimeline();
      return;
    }
    const facts = branchFacts(target), alternative = facts.alternative || {}, greedy = facts.greedy || {};
    const continuation = facts.continuation || {}, comparison = facts.comparison || {}, margin = facts.margin?.value;
    $('branch-title').innerHTML = `<code>${escape(greedy.token)}</code>와 <code>${escape(alternative.token)}</code> 사이의 선택`;
    let consequence;
    if (continuation.status === 'direct') {
      consequence = comparison.alternative_span
        ? `이 위치에서 <code>${escape(alternative.token)}</code> 하나를 고정하고 나머지를 greedy로 완성한 기존 결과는 <b>${spanText(comparison.alternative_span)}</b>입니다.`
        : '이 token을 고정한 직접 continuation은 관측했지만 유효한 시간 구간으로 파싱되지 않았습니다.';
      if (comparison.direct_distinct_mode_candidate === true) consequence += ` Greedy와 경계가 최대 <b>${number(comparison.max_endpoint_difference, 2)} s</b> 달라 구간 분리 기준(${number(comparison.endpoint_tolerance, 2)} s)을 넘습니다.`;
      if (comparison.direct_distinct_mode_candidate === false) consequence += ' 관측한 구간 차이는 설정된 분리 기준 이내입니다.';
    } else if (continuation.status === 'descendants_only') {
      consequence = `같은 분기를 거친 자손 경로 ${continuation.descendant_count}개가 있지만 이후 token도 추가로 바뀌었습니다. <b>이 선택 하나의 직접 결과는 아직 확인하지 못했습니다.</b>`;
    } else if (continuation.status === 'provenance_unverified') {
      consequence = '같은 prefix와 token을 공유하는 경로는 있으나, 이 선택 이후 추가 강제 분기가 없었는지 확인되지 않았습니다. 직접 결과로 표시하지 않습니다.';
    } else if (continuation.status === 'conflicting_direct') {
      consequence = '직접 continuation으로 분류된 기록들이 충돌하여 대표 구간을 정하지 않았습니다. 원자료의 실행 경로를 확인해야 합니다.';
    } else consequence = '이 alternative를 선택한 기존 continuation이 없습니다. 현재는 token 선택과 입력 민감도만 비교할 수 있습니다.';
    const odds = finite(greedy.probability) && alternative.probability > 0 ? greedy.probability / alternative.probability : null;
    $('branch-narrative').innerHTML = `<div class="selected-prefix"><span>같은 prefix</span><code>${escape(facts.prefix || '(응답 시작 전)')}</code></div><p>${consequence}</p>`;
    $('target-metrics').innerHTML = [
      [percent(greedy.probability), `greedy “${greedy.token}” probability`],
      [percent(alternative.probability), `alternative “${alternative.token}” probability`],
      [number(margin), `Margin · greedy 우세 ${finite(odds) ? number(odds, 2) + '배' : '—'}`]
    ].map(([value, label]) => `<div><span>${escape(label)}</span><b>${value}</b></div>`).join('');
    renderTimeline(continuation.status === 'direct' ? comparison.alternative_span : null);
    const competition = facts.competition || {};
    const sign = competition.screen_sign > 0 ? '+' : '−';
    const directional = finite(competition.screen_gain) && competition.screen_gain > 0
      ? `${escape(sourceLabel(competition.screen_direction))} ${sign}${number(100 * epsilon(), 3)}%`
      : '측정 방향의 1차 gap 감소 없음';
    const observed = competition.observed_pairwise_reversal === true ? '측정한 변화 중 두 token의 순서가 바뀜'
      : competition.observed_pairwise_tie ? '측정한 변화 중 두 token의 점수가 같아짐'
      : finite(competition.observed_best_gap) ? '측정한 변화에서 greedy 우세 유지' : '측정 기록 없음';
    $('screening-summary').innerHTML = `<div><span class="section-kicker">분기 선별을 위한 screening</span><p><b>${directional}</b></p><p class="small">Gap을 가장 줄이는 1차 추정 방향</p></div><div><span>기준 margin → 1차 추정 gap</span><b>${number(margin)} → ${number(competition.screen_gap)}</b><small>선택한 방향의 국소 추정</small></div><div><span>실제로 측정한 최소 gap</span><b>${number(competition.observed_best_gap)}</b><small>${observed}</small></div><p class="screening-footnote">실제 최소 gap은 모든 시험 방향·부호 중 최솟값입니다. 위 1차 추정 방향과 같은 방향에서 나왔다고 가정하지 않습니다. 두 token의 역전도 전체 vocabulary top1 또는 최종 mode 전환을 확정하지 않습니다.</p>`;
  }

  function directionWords(target) {
    if (isBranch(target)) {
      const facts = branchFacts(target);
      return {positive: `greedy “${facts.greedy.token}” 쪽`, negative: `alternative “${facts.alternative.token}” 쪽`};
    }
    return {positive: '목표 log probability 증가', negative: '목표 log probability 감소'};
  }

  function renderInputDirections(target) {
    const words = directionWords(target), scale = epsilon();
    const groups = new Map((target.groups || []).map(group => [group.group, group]));
    const rows = [{group: 'video_pixels', signed: target.pixel_signed_sum, absolute: null, count: null}];
    for (const group of ['timestamp', 'query', 'instruction', 'special', 'generated']) {
      const value = groups.get(group);
      if (value) rows.push({group, signed: value.signed_sum, absolute: value.absolute_sum, count: value.count});
    }
    const maximum = Math.max(...rows.map(row => Math.abs((row.signed || 0) * scale)), 1e-20);
    const legend = `<span class="alternative-color">← 음수: ${escape(words.negative)}</span><span class="greedy-color">양수: ${escape(words.positive)} →</span>`;
    $('direction-legend').innerHTML = legend;
    $('layer-legend').innerHTML = legend;
    $('direction-scale').textContent = `+${number(scale * 100, 3)}% 배율에 대한 gradient 1차 예측`;
    $('input-directions').innerHTML = rows.map(row => {
      const delta = finite(row.signed) ? row.signed * scale : null;
      const width = finite(delta) ? Math.abs(delta) / maximum * 50 : 0;
      const toward = !finite(delta) ? '미계측' : delta === 0 ? '1차 변화 0' : delta > 0 ? words.positive : words.negative;
      const absolute = finite(row.absolute) ? `Net ${signed(row.signed)} · absolute sum ${scientific(row.absolute)}` : `전체 pixel signed sum ${signed(row.signed)}`;
      return `<div class="direction-row"><div class="direction-source"><b>${escape(sourceLabel(row.group))}</b><small>${row.group === 'video_pixels' ? 'Normalized pixel 표현' : `${row.count}개 입력 위치`}</small></div><div><div class="diverging-track"><span class="zero-line"></span><span class="diverging-bar ${delta < 0 ? 'negative' : 'positive'}" style="${delta < 0 ? 'right:50%' : 'left:50%'};width:${width}%"></span></div><small class="net-note">${absolute}</small></div><div class="direction-value ${delta < 0 ? 'alternative-color' : 'greedy-color'}"><b>${signed(delta)}</b><small>${escape(toward)}</small></div></div>`;
    }).join('');
    $('direction-note').textContent = `표시값은 ε×〈입력 표현, 목표 gradient〉입니다(ε=${scale}). 막대는 같은 차트의 최대 절댓값에 맞춘 크기이며 evidence 비중이 아닙니다. 배율 확대가 정보나 의미를 강화한다는 뜻은 아닙니다. Net이 작아도 위치별 양수·음수가 상쇄될 수 있어 absolute sum을 함께 표시합니다. 실제 ±배율 재계산의 screening과 이 원래 계산점의 gradient 예측은 다를 수 있습니다.`;
  }

  function metricValue(value) {
    return Number($('metric').value === 'grad_l2'
      ? value.grad_l2 ?? value.gradient_norm ?? value.grad_norm ?? 0
      : value.activation_gradient ?? value.activation_dot_gradient ?? value.signed_sum ?? value.actgrad ?? 0);
  }

  function positionRows(target) {
    const context = record.context || {};
    const mapping = new Map((context.visual_map || []).map(position => [position.position, position]));
    const base = context.ids?.map((id, index) => ({index, id, text: context.tokens[index], type: context.groups[index], ...mapping.get(index)})) || [];
    const limit = targetStep(target) >= 0 ? record.input.prompt_length + targetStep(target) : base.length;
    return base.slice(0, limit).map((position, index) => ({
      ...position, grad_l2: target.context?.grad_norm?.[index],
      activation_gradient: target.context?.activation_dot_gradient?.[index]
    }));
  }

  async function renderTarget() {
    const index = targetIndex, target = targets()[index], query = record;
    $('attribution-panel').hidden = !target;
    $('attribution-panel').dataset.targetId = target?.id || '';
    $('attribution-panel').dataset.targetReady = 'false';
    for (const id of ['frame-panel', 'layer-panel', 'message-panel']) $(id).hidden = true;
    if (!target) return;
    renderNarrative(target);
    renderInputDirections(target);
    $('target-description').textContent = `${target.kind} · Source greedy prefix를 유지한 계측 replay의 국소 민감도입니다. 큰 값만으로 인과적 근거나 사건 이해를 확정하지 않습니다.`;
    $('positions').innerHTML = empty('선택한 target의 상세 gradient를 불러오는 중…');
    try {
      await loadTarget(target);
      if (query !== record || index !== targetIndex) return;
      const positions = positionRows(target);
      const types = [...new Set(positions.map(position => position.type || 'other'))];
      const previous = $('source-type').value;
      $('source-type').innerHTML = '<option value="">전체</option>' + types.map(type => `<option value="${escape(type)}">${escape(sourceLabel(type))}</option>`).join('');
      $('source-type').value = types.includes(previous) ? previous : '';
      renderGroups(target);
      renderPositions();
      renderFrames(target, positions);
      renderLayers(target);
      renderMessages(target);
      $('attribution-panel').dataset.targetReady = 'true';
      updateURL();
    } catch (error) {
      $('positions').innerHTML = empty(`Gradient 원자료 로딩 실패: ${error.message}`);
    }
  }

  function renderGroups(target) {
    const groups = target.groups || [];
    const maximum = Math.max(...groups.map(group => Math.abs(metricValue(group))), 1e-20);
    const norm = $('metric').value === 'grad_l2';
    $('groups').innerHTML = groups.map(group => {
      const value = metricValue(group);
      return `<div class="bar-row"><span>${escape(sourceLabel(group.group))}</span><div class="bar-track"><div class="bar-value ${norm ? '' : value < 0 ? 'negative' : 'positive'}" style="width:${Math.abs(value) / maximum * 100}%"></div></div><span class="bar-number">${scientific(value)}</span></div>`;
    }).join('');
    $('group-stat-table').innerHTML = `<table><thead><tr><th>입력 group</th><th>위치 수</th><th>Gradient L2</th><th>Net signed sum</th><th>Absolute sum</th></tr></thead><tbody>${groups.map(group => `<tr><td>${escape(sourceLabel(group.group))}</td><td>${group.count}</td><td>${scientific(group.grad_l2)}</td><td>${signed(group.signed_sum)}</td><td>${scientific(group.absolute_sum)}</td></tr>`).join('')}</tbody></table>`;
  }

  function heatColor(value, maximum, signedMetric) {
    const alpha = Math.max(0.04, Math.min(0.85, Math.abs(value) / (maximum || 1) * 0.85));
    if (!signedMetric) return `rgba(43,111,145,${alpha})`;
    return value < 0 ? `rgba(178,120,55,${alpha})` : `rgba(43,111,145,${alpha})`;
  }

  function renderPositions() {
    const target = targets()[targetIndex];
    if (!target?._loaded && target?.data_file) return;
    const all = positionRows(target), type = $('source-type').value;
    let rows = all.filter(position => !type || position.type === type);
    if ($('source-sort').value === 'magnitude') rows.sort((a, b) => Math.abs(metricValue(b)) - Math.abs(metricValue(a)));
    positionPage = Math.max(0, Math.min(positionPage, Math.ceil(rows.length / POSITION_PAGE_SIZE) - 1));
    const maximum = Math.max(...all.map(position => Math.abs(metricValue(position))), 1e-20);
    const signedMetric = $('metric').value !== 'grad_l2';
    $('position-summary').textContent = `${rows.length} / ${all.length} 선행 위치 · 색상은 전체 위치의 최대 절댓값 기준`;
    $('positions').innerHTML = rows.slice(positionPage * POSITION_PAGE_SIZE, (positionPage + 1) * POSITION_PAGE_SIZE).map(position => `
      <button class="position" data-position="${position.index}" data-time="${position.timestamp ?? ''}" style="background:${heatColor(metricValue(position), maximum, signedMetric)}" title="${escape(sourceLabel(position.type))} · position ${position.index}">${escape(position.text)}<small>${position.index} · ${scientific(metricValue(position))}</small></button>
    `).join('') || empty('이 조건의 선행 위치가 없습니다.');
    $('position-page').textContent = `${positionPage + 1} / ${Math.max(1, Math.ceil(rows.length / POSITION_PAGE_SIZE))}`;
    $('position-prev').disabled = positionPage === 0;
    $('position-next').disabled = (positionPage + 1) * POSITION_PAGE_SIZE >= rows.length;
  }

  function framesFor(target, positions) {
    if ($('frame-space').value === 'pixels' && target.pixel_frames?.gradient_norm) {
      return target.pixel_frames.gradient_norm.map((norm, index) => {
        const dots = target.pixel_frames.activation_dot_gradient[index];
        return {index, timestamp: record.input.timestamps?.[index], grid_h: norm.length, grid_w: norm[0]?.length,
          values: ($('metric').value === 'grad_l2' ? norm : dots).flat(),
          grad_l2: Math.sqrt(norm.flat().reduce((sum, value) => sum + value * value, 0)),
          activation_gradient: dots.flat().reduce((sum, value) => sum + value, 0)};
      });
    }
    const grouped = new Map();
    for (const position of positions) {
      if (position.frame_pair === undefined) continue;
      const frame = grouped.get(position.frame_pair) || {index: position.frame_pair, timestamp: position.timestamp, positions: []};
      frame.positions.push(position);
      grouped.set(position.frame_pair, frame);
    }
    return [...grouped.values()].sort((a, b) => a.index - b.index).map(frame => {
      const norm = target.frames?.gradient_norm?.[frame.index], dots = target.frames?.activation_dot_gradient?.[frame.index];
      return {...frame, grid_h: norm?.length, grid_w: norm?.[0]?.length,
        values: ($('metric').value === 'grad_l2' ? norm : dots)?.flat() || [],
        grad_l2: norm ? Math.sqrt(norm.flat().reduce((sum, value) => sum + value * value, 0)) : 0,
        activation_gradient: dots?.flat().reduce((sum, value) => sum + value, 0) || 0};
    });
  }

  function renderFrames(target, positions) {
    $('frame-title').textContent = $('metric').value === 'grad_l2' ? '언제, 어디에 민감한가' : '어느 위치의 확대가 선택을 기울이는가';
    const frames = framesFor(target, positions);
    $('frame-panel').hidden = !frames.length;
    if (!frames.length) return;
    frameIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
    $('frame').innerHTML = frames.map((frame, index) => `<option value="${index}">${index} · ${number(frame.timestamp, 2)} s</option>`).join('');
    $('frame').value = String(frameIndex);
    const maximum = Math.max(...frames.map(frame => Math.abs(metricValue(frame))), 1e-20);
    const signedMetric = $('metric').value !== 'grad_l2';
    $('frame-timeline').innerHTML = frames.map((frame, index) => `<button class="frame-cell ${index === frameIndex ? 'selected' : ''}" data-frame="${index}" style="background:${heatColor(metricValue(frame), maximum, signedMetric)}" title="${number(frame.timestamp, 2)} s · ${scientific(metricValue(frame))}">${index}</button>`).join('');
    $('frame-description').textContent = ($('frame-space').value === 'pixels'
      ? 'Source frame의 pixel patch 지도입니다. Vision encoder와 DeepStack 경로를 포함합니다.'
      : '두 frame을 묶은 visual embedding 지도입니다. DeepStack 추가 주입 이전의 language-model 입력을 측정합니다.')
      + (signedMetric ? ' 파랑은 목표를 높이는 방향, 황토색은 낮추는 방향입니다. 정답·오답 표시가 아닙니다.' : ' 색이 진할수록 gradient norm이 큽니다. 지지·억제의 부호는 포함하지 않습니다.');
    const frame = frames[frameIndex], values = frame.values || [];
    const localMaximum = Math.max(...values.map(Math.abs), 1e-20);
    $('spatial-map').style.gridTemplateColumns = `repeat(${frame.grid_w || 1},1fr)`;
    $('spatial-map').style.aspectRatio = `${frame.grid_w || 16}/${frame.grid_h || 9}`;
    $('spatial-map').innerHTML = values.map((value, index) => `<div class="spatial-cell" style="background:${heatColor(value, localMaximum, signedMetric)}" title="Patch ${index} · ${scientific(value)}"></div>`).join('');
    $('spatial-note').textContent = `${values.length} spatial 위치 · 현재 frame 내 최대 절댓값으로 색상 표시`;
    $('frame-details').innerHTML = `<table class="record-table"><tbody><tr><th>원본 timestamp</th><td>${number(frame.timestamp, 3)} s</td></tr><tr><th>Spatial grid</th><td>${frame.grid_h} × ${frame.grid_w}</td></tr><tr><th>표시값</th><td>${scientific(metricValue(frame))}</td></tr><tr><th>지도 읽기</th><td>${signedMetric ? 'Signed activation × gradient' : 'Gradient L2 norm'}</td></tr></tbody></table><p class="small">Frame 버튼을 누르면 영상도 같은 시각으로 이동합니다. 같은 계산에서의 민감도이며 프레임을 독립적으로 제거한 효과가 아닙니다.</p>`;
  }

  function renderLayers(target) {
    const layers = target.layers || {}, kinds = ['attention', 'mlp', 'residual_input'];
    if (!$('receiver').options.length || Number($('receiver').dataset.target) !== targetIndex) {
      const maximum = Math.max(0, targetStep(target));
      $('receiver').innerHTML = Array.from({length: maximum + 1}, (_, index) => `<option value="${index}">Step ${index} predictor · position ${record.input.prompt_length + index - 1}</option>`).join('');
      $('receiver').value = String(maximum);
      $('receiver').dataset.target = String(targetIndex);
    }
    const receiver = Number($('receiver').value);
    const indices = [...new Set(kinds.flatMap(kind => (layers[kind] || []).map(row => row.layer)))].sort((a, b) => a - b);
    $('layer-panel').hidden = !indices.length;
    if (!indices.length) return;
    const lookup = Object.fromEntries(kinds.map(kind => [kind, new Map((layers[kind] || []).map(row => [row.layer, row.activation_dot_gradient?.[receiver]]))]));
    const maximum = Math.max(...indices.flatMap(index => kinds.map(kind => Math.abs(lookup[kind].get(index) ?? 0))), 1e-20);
    $('layers').innerHTML = '<div class="layer-row"><b>Layer</b><b>Attention</b><b>MLP</b><b>Residual input</b></div>' + indices.map(index => `<div class="layer-row"><b>${index}</b>${kinds.map(kind => {
      const value = lookup[kind].get(index);
      return `<div class="layer-value"><div class="bar-track"><div class="bar-value ${value < 0 ? 'negative' : 'positive'}" style="width:${finite(value) ? Math.abs(value) / maximum * 100 : 0}%"></div></div><span>${signed(value)}</span></div>`;
    }).join('')}</div>`).join('');
  }

  function renderMessages(target) {
    const data = target.messages || {}, layers = data.layers || [];
    $('message-panel').hidden = !layers.length;
    if (!layers.length) return;
    const sign = $('message-sign').value;
    const receiver = record.input.prompt_length + Number($('receiver').value || targetStep(target)) - 1;
    const rows = layers.flatMap(layer => (layer.edges || []).map(edge => ({...edge, layer: layer.layer})))
      .filter(edge => edge.receiver_position === receiver && (!sign || edge.sign === sign))
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
    $('message-description').textContent = `Layer ${(data.selected_layers || []).join(', ')}의 prediction receiver에서 보존한 head별 상위 양수·음수 edge입니다. Visual-to-visual 및 다른 간접 receiver는 미계측입니다. 현재 receiver position ${receiver}.`;
    $('messages').innerHTML = rows.slice(0, 100).map(edge => `<tr><td>${edge.layer} / ${edge.head}</td><td><code>${escape(record.context.tokens[edge.source_position])}</code><div class="small">${edge.source_position} · ${escape(sourceLabel(record.context.groups[edge.source_position]))}</div></td><td>${edge.receiver_position}</td><td>${scientific(edge.attention_weight)}</td><td class="${edge.score < 0 ? 'alternative-color' : 'greedy-color'}">${signed(edge.score)}</td></tr>`).join('') || '<tr><td colspan="5">해당 receiver·부호의 보존된 edge가 없습니다.</td></tr>';
    $('message-count').textContent = `${rows.length}개 중 최대 100개를 절댓값순으로 표시 · score=Aᵢⱼ〈∂L/∂oᵢ,vⱼ〉 · cross-layer 합산 없음`;
  }

  function renderRecord() {
    const validation = record.validation || {}, cost = record.cost || {}, input = record.input || {};
    const passed = validation.input_hashes_match && validation.causality_passed && validation.greedy_argmax_match
      && validation.gradient_forward_max_logit_error === 0
      && validation.replay_max_probability_error <= (catalog.config?.replay_probability_tolerance ?? 0.001);
    $('validation').innerHTML = `<span class="${passed ? 'ok' : 'fail'}">${passed ? 'Source greedy 재현 검증 통과' : '계측 검증 상태 확인 필요'}</span><br>Max top30 |Δp| ${scientific(validation.replay_max_probability_error)} · argmax ${validation.greedy_argmax_match ? '전체 일치' : '차이 있음'}`;
    const pairs = [
      ['Source record', record.source?.record], ['모델', catalog.config?.model],
      ['Input tokens', input.prompt_length], ['Visual tokens', record.context?.visual_positions?.length],
      ['Input H×W', input.hw?.join(' × ')], ['Wall seconds', cost.total_seconds],
      ['Peak allocated GiB', finite(cost.peak_allocated_bytes) ? cost.peak_allocated_bytes / 2 ** 30 : null],
      ...Object.entries(validation).map(([key, value]) => [key, typeof value === 'object' ? JSON.stringify(value) : String(value)])
    ];
    $('record-details').innerHTML = `<p><a class="raw-link" href="${escape(resolve(selected.record, rootURL))}" target="_blank" rel="noopener">Query 원자료 ↗</a>${selected.interpretation ? `<a class="raw-link" href="${escape(resolve(selected.interpretation, rootURL))}" target="_blank" rel="noopener">분기 해석·출처 검증 ↗</a>` : ''}${catalog.config_path ? `<a class="raw-link" href="${escape(resolve(catalog.config_path, rootURL))}" target="_blank" rel="noopener">동결 실행 설정 ↗</a>` : ''}</p><table class="record-table"><tbody>${pairs.filter(([, value]) => value !== null && value !== undefined).map(([key, value]) => `<tr><th>${escape(key)}</th><td>${escape(value)}</td></tr>`).join('')}</tbody></table>`;
  }

  function seekFrame(index) {
    const target = targets()[targetIndex];
    if (!target) return;
    frameIndex = index;
    const frames = framesFor(target, positionRows(target));
    if (finite(frames[index]?.timestamp)) $('video').currentTime = Number(frames[index].timestamp);
    renderFrames(target, positionRows(target));
  }

  $('samples').addEventListener('click', event => {
    const button = event.target.closest('[data-key]');
    if (button) selectSample(button.dataset.key);
  });
  for (const id of ['filter', 'dataset', 'completion']) $(id).addEventListener(id === 'filter' ? 'input' : 'change', () => {listPage = 0; renderList();});
  $('list-prev').onclick = () => {listPage--; renderList();};
  $('list-next').onclick = () => {listPage++; renderList();};
  for (const [id, delta] of [['query-prev', -1], ['query-next', 1]]) $(id).onclick = () => {
    const index = filtered.findIndex(sample => sample.key === selected?.key);
    if (filtered[index + delta]) selectSample(filtered[index + delta].key);
  };
  $('branch-cards').onclick = event => {
    const button = event.target.closest('[data-branch-target]');
    if (button) selectBranch(Number(button.dataset.branchTarget));
  };
  $('sentence').onclick = event => {
    const button = event.target.closest('[data-step]');
    if (button) {step = Number(button.dataset.step); renderStep();}
  };
  for (const [id, delta] of [['token-prev', -1], ['token-next', 1]]) $(id).onclick = () => {step += delta; renderStep();};
  $('branches').onclick = event => {
    const button = event.target.closest('[data-target]');
    if (button) selectBranch(Number(button.dataset.target), true);
  };
  $('target').onchange = () => {
    targetIndex = Number($('target').value);
    positionPage = 0;
    renderBranchCards();
    renderTarget();
  };
  $('metric').onchange = () => renderTarget();
  for (const id of ['source-type', 'source-sort']) $(id).onchange = () => {positionPage = 0; renderPositions();};
  $('position-prev').onclick = () => {positionPage--; renderPositions();};
  $('position-next').onclick = () => {positionPage++; renderPositions();};
  $('positions').onclick = event => {
    const button = event.target.closest('[data-time]');
    if (button && button.dataset.time !== '' && finite(button.dataset.time)) $('video').currentTime = Number(button.dataset.time);
  };
  $('receiver').onchange = () => {const target = targets()[targetIndex]; renderLayers(target); renderMessages(target);};
  $('message-sign').onchange = () => renderMessages(targets()[targetIndex]);
  $('frame-space').onchange = () => {frameIndex = 0; const target = targets()[targetIndex]; renderFrames(target, positionRows(target));};
  $('frame').onchange = () => seekFrame(Number($('frame').value));
  $('frame-timeline').onclick = event => {
    const button = event.target.closest('[data-frame]');
    if (button) seekFrame(Number(button.dataset.frame));
  };
  $('timeline').onclick = event => {
    const button = event.target.closest('[data-seek]');
    if (button) $('video').currentTime = Number(button.dataset.seek);
  };
  $('video').addEventListener('error', () => {$('video-status').textContent = '영상 재생 오류 · 원자료 경로를 확인하세요.';});

  // External links and browser history may change only the hash, without
  // reloading this document. Restore the full query/step/target state then.
  window.addEventListener('hashchange', () => {
    initialParams = new URLSearchParams(location.hash.slice(1) || location.search);
    const key = initialParams.get('sample');
    if (key && samples.some(sample => sample.key === key)) selectSample(key, true);
  });

  fetchJSON(resolve(config.catalog)).then(data => {
    catalog = Array.isArray(data) ? {samples: data} : data;
    samples = catalog.samples || [];
    $('dataset').innerHTML = '<option value="">전체</option>' + [...new Set(samples.map(sample => sample.dataset))]
      .map(dataset => `<option value="${escape(dataset)}">${escape(datasetLabel(dataset))}</option>`).join('');
    renderSummary();
    renderList();
    const key = initialParams.get('sample') || samples.find(sample => sample.complete)?.key || samples[0]?.key;
    if (key) selectSample(key, true);
  }).catch(error => {$('status').textContent = `목록 로딩 실패: ${error.message}`;});
})();
