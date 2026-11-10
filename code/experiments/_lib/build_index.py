"""Build the single experiment entry point from existing result pages."""
from pathlib import Path
from html import escape
import os
import re

ROOT = Path(str(Path(__file__).resolve().parents[1]).replace('/home1/irteam/', '/home/irteam/'))
PROJECTS = {
    'dataset_metadata': 'Dataset metadata',
    'attention': 'Attention',
    'zero-shot': 'Zero-shot',
    'similarity': 'Similarity',
    'analysis/failure_mode_timestamp/failure_cases': 'Failure analysis',
    'audio_caption_resources_20260930': 'Audio caption resources',
}

# Research questions group related runs without changing their stable paths.
ZERO_SHOT_GROUPS = {
    'baseline': ('기준 성능·입력 조건', '전체 집계 → 입력 순서·일관성 → Thinking/Instruct 비교'),
    'uncertainty': ('토큰 불확실성·오류 진단', 'Full-head 분포 → 엔트로피·최초 오류 → 자릿수별 정답 가능성'),
    'sampling': ('영상 샘플링·공간 통제', '공간 예산 비교 → 전체/경계 배정 → 고정 해상도 FPS 비교. GT 활용 조건은 진단 실험입니다.'),
    'rollout': ('후보 탐색·Rollout', 'Stochastic · 조건부 예산 · 결정적 탐색과300/600/900 누적 비교'),
    'operations': ('운영·입력 검증', '캐시 이관, 영상 디코딩, 전체 추론 병목 조사'),
}


def zero_shot_group(relative):
    parts = relative.split('/')
    run = parts[0]
    if run in ('full', 'pilot', 'overview') and len(parts)>1:
        return parts[1]
    if run in ZERO_SHOT_GROUPS:
        return run
    if any(part in relative.split('/') for part in ('performance', 'mmco_diagnosis')) or run == 'selected_frame_cache_migration_v1':
        return 'operations'
    if run in ('all_zeroshot_v1', 'input_order', 'thinking_vs_instruct_v1'):
        return 'baseline'
    if run.startswith(('spatial_', 'fps_')):
        return 'sampling'
    if 'rollout' in run:
        return 'rollout'
    return 'uncertainty'


def scope_label(path):
    value = str(path)
    if '/results/' not in value:
        return ''
    relative = value.split('/results/', 1)[1]
    if 'additional600' in relative:
        return 'pilot900 중 추가600질문 (기존300 제외)'
    if '/pilot/' in '/' + relative:
        if '/overview/' in '/' + relative:
            return 'pilot overview'
        match = re.search(r'/pilot/(\d+)(?:/|$)', '/' + relative)
        return f'pilot · {int(match[1]):,} queries' if match else 'pilot'
    if '/full/' in '/' + relative:
        match = re.search(r'/full/(\d+)(?:/|$)', '/' + relative)
        return f'full · {int(match[1]):,} queries' if match else 'full · source-specific populations'
    if '/shared/' in '/' + relative:
        return 'shared resources'
    if 'zero-shot/results/' in value:
        if 'full9404_pilot300' in value or '/baseline/all_zeroshot/overview/' in value or '/baseline/input_order/overview/' in value:
            return 'full · 9,404 queries / pilot · 300 queries'
        for n in (900, 600, 300, 30):
            if f'pilot{n}' in value:
                return f'pilot{n} ({n}질문)'
        if 'additional600' in value:
            return 'pilot900 중 추가600질문'
        if 'timelens_rollout' in value:
            return 'pilot300 (모델별 300질문)'
    return ''


def history_controls(home="index.html", refresh_label="목록 새로고침"):
    return f'<nav class="history-controls" aria-label="페이지 이동"><button type="button" data-history-back aria-label="뒤로 가기" title="뒤로 가기" disabled><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 12H5m7-7-7 7 7 7"/></svg></button><button type="button" data-history-forward aria-label="앞으로 가기" title="앞으로 가기" disabled><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-7-7 7 7-7 7"/></svg></button><button type="button" data-refresh-current aria-label="{escape(refresh_label, quote=True)}" title="{escape(refresh_label, quote=True)}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5M6.1 6.1A8 8 0 0 1 19.6 9M4.4 15A8 8 0 0 0 17.9 17.9"/></svg></button><a class="home-control" href="{escape(home, quote=True)}" aria-label="홈" title="홈 · 전체 실험"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m3 10 9-7 9 7M5 9v12h5v-7h4v7h5V9"/></svg></a></nav>'


def add_navigation(path, text):
    # Inject at generation time, including old documents with an implicit head/body.
    # Keep this idempotent so rebuilding never accumulates theme tags.
    text = re.sub(r'<link\b[^>]*data-experiment-theme[^>]*>', '', text, flags=re.I)
    if not re.search(r'<html\b', text, re.I):
        text = re.sub(r'(<!doctype[^>]*>)', r'\1<html lang="ko" data-experiment-detail>', text, count=1, flags=re.I) if re.search(r'<!doctype', text, re.I) else '<html lang="ko" data-experiment-detail>' + text
    elif not re.search(r'<html\b[^>]*data-experiment-detail', text, re.I):
        text = re.sub(r'<html\b', '<html data-experiment-detail', text, count=1, flags=re.I)
    if not re.search(r'<meta\b[^>]*name=[\'"]viewport[\'"]', text, re.I):
        text = re.sub(r'(<html\b[^>]*>)', r'\1<meta name="viewport" content="width=device-width,initial-scale=1">', text, count=1, flags=re.I)
    theme = os.path.relpath(ROOT / '_lib/experiment_theme.css', path.parent)
    link = f'<link data-experiment-theme rel="stylesheet" href="{theme}">'
    # Last stylesheet wins over each detail viewer's original visual skin.
    styles = list(re.finditer(r'</style\s*>|<link\b[^>]*rel=[\'"]stylesheet[\'"][^>]*>', text, re.I))
    pos = styles[-1].end() if styles else re.search(r'<html\b[^>]*>', text, re.I).end()
    text = text[:pos] + link + text[pos:]
    history_js = os.path.relpath(ROOT / '_lib/history_controls.js', path.parent)
    text = re.sub(r'<script\b[^>]*data-experiment-history[^>]*>.*?</script>', '', text, flags=re.I | re.S)
    text = text.replace(link, link + f'<script data-experiment-history defer src="{history_js}"></script>', 1)
    nav = '<div data-experiment-home>' + history_controls(os.path.relpath(ROOT / 'index.html', path.parent), '페이지 새로고침') + '</div>'
    if 'data-experiment-home' in text:
        return re.sub(r'<(nav|div)\b[^>]*data-experiment-home[^>]*>.*?</\1>', lambda m: nav, text, count=1, flags=re.S)
    body = re.search(r'<body\b[^>]*>', text, re.I)
    if body:
        return text[:body.end()] + nav + text[body.end():]
    first = re.search(r'<(?:h[1-6]|main|article|div|p|header)\b', text, re.I)
    pos = first.start() if first else len(text)
    return text[:pos] + nav + text[pos:]


def build():
    from pilot_catalog import build_pilot_catalog
    from project_pilot_catalog import build_project_pilot_catalogs
    from family_catalog import build_family_catalog
    from report_appendices import build_report_appendices
    build_pilot_catalog()
    build_project_pilot_catalogs()
    build_report_appendices(PROJECTS)
    families = build_family_catalog(PROJECTS)
    sections = []
    for project, label in PROJECTS.items():
        values = [f for f in families if f['project'] == project]
        topics = {}
        for f in values:
            topics.setdefault(f['topic_title'], []).append(f)
        groups = []
        for topic, members in topics.items():
            rows = []
            for f in members:
                search = (label + ' ' + topic + ' ' + f['title'] + ' ' + f['search']).lower()
                scopes = ' · '.join(f['scopes'])
                rows.append(f'<article data-search="{escape(search, quote=True)}"><div class="family-name"><h3><a href="{f["href"]}" target="_blank" rel="noopener">{escape(f["title"])}</a></h3><p>{escape(scopes)}</p><p class="run">{escape(f["topic"] + "/" + f["family"])}</p></div><a class="open" href="{f["href"]}" target="_blank" rel="noopener">새 탭에서 보기 ↗</a><label class="compare-choice"><input type="checkbox" name="compare" value="{f["href"]}"> 비교</label></article>')
            groups.append(f'<section data-topic><header class="topic-heading"><h2>{escape(topic)}</h2><span class="topic-count">{len(members)}개 실험 계열</span></header>{"".join(rows)}</section>')
        sections.append(f'''<details class="project" data-project="{project}" data-label="{escape(label, quote=True)}"><summary><button type="button" class="drag-handle" aria-label="{escape(label)} 프로젝트 끌어서 이동" title="끌어서 순서 변경">⠿</button><span class="project-title">{escape(label)}<small>{len(values)}개 실험 계열</small></span><span class="project-actions"><button type="button" class="move-up" aria-label="{escape(label)} 위로 이동" title="위로 이동">↑</button><button type="button" class="move-down" aria-label="{escape(label)} 아래로 이동" title="아래로 이동">↓</button><button type="button" class="pin-project" aria-label="{escape(label)} 상단 고정" aria-pressed="false" title="상단 고정">☆</button></span></summary><div class="project-body">{"".join(groups)}</div></details>''')
    options = ''.join(f'<option value="{p}">{label}</option>' for p, label in PROJECTS.items())
    document = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실험 목록</title><link rel="stylesheet" href="_lib/experiment_theme.css"><link rel="stylesheet" href="_lib/index_catalog.css"><link rel="stylesheet" href="_lib/index_compare.css"><script defer src="_lib/history_controls.js"></script></head><body><main id="catalog"><header>__HISTORY_CONTROLS__<div class="eyebrow">EXPERIMENTS / RESEARCH FAMILIES</div><h1>실험 목록</h1><p>프로젝트에서 주제와 실험 계열을 탐색하세요.<br>실험과 비교 화면은 새 탭에서 열립니다.</p></header>
<div class="controls"><input id="search" type="search" aria-label="실험 검색" placeholder="주제, 실험 계열, full/pilot 또는 분석명 검색"><select id="project" aria-label="프로젝트 선택"><option value="">전체 프로젝트</option>__OPTIONS__</select></div><p id="count" aria-live="polite"></p>
<div class="order-help"><p>⠿ 핸들을 끌어 순서를 바꾸거나 ☆로 상단에 고정하세요. 순서는 이 브라우저에 저장됩니다.</p><button id="reset-project-order" type="button">기본 순서로</button></div>
<section class="project-zone" aria-labelledby="pinned-heading"><h2 id="pinned-heading" class="zone-heading">고정한 프로젝트 <span>PINNED</span></h2><div id="pinned-projects" class="project-list" data-pinned="true"><p class="pin-empty">자주 보는 프로젝트를 여기로 끌어다 놓으세요.</p></div></section>
<section class="project-zone" aria-labelledby="projects-heading"><h2 id="projects-heading" class="zone-heading">프로젝트</h2><div id="other-projects" class="project-list" data-pinned="false">__SECTIONS__</div></section><p id="empty" hidden>검색 조건에 맞는 실험 계열이 없습니다.</p><p id="order-status" class="sr-only" role="status"></p>
<div id="compare-tray"><span id="selection-status" role="status">비교할 실험 계열 2개를 선택하세요 (0/2)</span><button id="clear-selection" type="button" disabled>선택 해제</button><button id="open-comparison" type="button" disabled>두 계열 비교 ↗</button></div></main>
<div id="comparison" hidden><div class="comparison-toolbar">__HISTORY_CONTROLS__<h1>실험 계열 비교</h1><button id="swap-comparison" type="button">좌우 바꾸기</button></div><div class="comparison-panes"><div class="comparison-pane"><div class="pane-toolbar"><label for="left-experiment">실험 A</label><select id="left-experiment"></select><a id="left-open" target="_blank" rel="noopener">새 탭 ↗</a></div><iframe id="left-frame" title="실험 A 결과"></iframe></div><div class="comparison-pane"><div class="pane-toolbar"><label for="right-experiment">실험 B</label><select id="right-experiment"></select><a id="right-open" target="_blank" rel="noopener">새 탭 ↗</a></div><iframe id="right-frame" title="실험 B 결과"></iframe></div></div></div>
<script src="_lib/index_catalog.js"></script><script src="_lib/index_compare.js"></script></body></html>'''
    document = document.replace('__OPTIONS__', options).replace('__SECTIONS__', '\n'.join(sections)).replace('__HISTORY_CONTROLS__', history_controls())
    target = ROOT / 'index.html'
    target.write_text(document)
    print(f'Built {target}: {len(families)} experiment families across {len(PROJECTS)} projects')


if __name__ == '__main__':
    build()
