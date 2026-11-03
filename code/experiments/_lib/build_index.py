"""Build the single experiment entry point from existing result pages."""
from result_layout import relocate_path
from pathlib import Path
from html import escape, unescape
from urllib.parse import quote
import os
import re

ROOT = Path(__file__).resolve().parents[1]
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


def add_navigation(path, text):
    scope = scope_label(path)
    href = os.path.relpath(ROOT / 'index.html', path.parent)
    badge = f'<span data-experiment-scope style="margin-left:24px">{escape(scope)}</span>' if scope else ''
    family_link = ''
    normalized = str(path).replace('/home1/irteam/', '/home/irteam/')
    for project in sorted(PROJECTS, key=len, reverse=True):
        marker = '/experiments/' + project + '/results/'
        if marker not in normalized:
            continue
        parts = normalized.split(marker, 1)[1].split('/')
        if len(parts) >= 3:
            family = Path(normalized.split(marker)[0] + marker) / parts[0] / parts[1]
            view = os.path.relpath(family / 'viewer.html', path.parent) + '?view=' + quote('/'.join(parts[2:]), safe='')
            family_link = f'<a style="color:inherit;margin-left:18px" href="{view}">계열 통합 보기</a>'
        break
    nav = f'<nav data-experiment-home style="position:relative;z-index:1000;padding:12px 20px;background:#edf4f2;color:#163b32;font:14px system-ui"><a style="color:inherit" href="{href}">← 전체 실험 목록</a>{family_link}{badge}</nav>'
    if 'data-experiment-home' in text:
        return re.sub(r'<nav\b[^>]*data-experiment-home[^>]*>.*?</nav>', lambda m: nav, text, count=1, flags=re.S)
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
    build_pilot_catalog()
    build_project_pilot_catalogs()
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
                rows.append(f'<article data-search="{escape(search, quote=True)}"><div class="family-name"><h3><a href="{f["href"]}">{escape(f["title"])}</a></h3><p>{escape(scopes)}</p><p class="run">{escape(f["topic"] + "/" + f["family"])}</p></div><a class="open" href="{f["href"]}">통합 보기 →</a><label class="compare-choice"><input type="checkbox" name="compare" value="{f["href"]}"> 비교</label></article>')
            groups.append(f'<section data-topic><h2>{escape(topic)}</h2>{"".join(rows)}</section>')
        sections.append(f'<details class="project" data-project="{project}"><summary><span>{escape(label)}</span><small>{len(values)}개 실험 계열</small></summary><div class="project-body">{"".join(groups)}</div></details>')
    options = ''.join(f'<option value="{p}">{label}</option>' for p, label in PROJECTS.items())
    document = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실험 목록</title><link rel="stylesheet" href="_lib/index_compare.css"><style>
:root{font:15px/1.6 system-ui,sans-serif;color:#203730;background:#f5f7f4}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:46px 26px 100px}header{padding-bottom:22px;border-bottom:1px solid #d2ded5}h1{font-size:38px;margin:4px 0 12px}header p{margin:0;color:#62736b}.eyebrow{font-size:12px;letter-spacing:.15em;color:#527366}.controls{display:flex;gap:12px;margin:24px 0 12px}input,select,button{font:inherit}#search,#project{padding:12px;border:1px solid #c7d4cc;border-radius:9px;background:#fff}#search{flex:1;min-width:0}#count{color:#62736b;font-size:13px}.project{border:1px solid #d8e1da;border-radius:12px;background:#fff;margin:12px 0;overflow:hidden}.project>summary{cursor:pointer;display:flex;justify-content:space-between;align-items:center;padding:19px 22px;gap:14px;font-size:19px;font-weight:600}.project>summary:before{content:'▸';color:#6f8b7a}.project[open]>summary:before{content:'▾'}.project>summary span{flex:1}.project small{font-size:12px;font-weight:400;color:#62736b}.project-body{padding:0 22px 16px}section h2{font-size:13px;color:#62736b;margin:18px 0 7px}article{display:flex;align-items:center;gap:18px;padding:13px 0;border-top:1px solid #edf0ec;min-width:0}.family-name{flex:1;min-width:0}h3{margin:0;font-size:16px;overflow-wrap:anywhere}.family-name p{margin:3px 0 0;font-size:12px;color:#62736b;overflow-wrap:anywhere}.run{display:none}a{color:#256650;text-decoration:none}a:hover{text-decoration:underline}.open{font-size:13px;white-space:nowrap}.compare-choice{font-size:12px;white-space:nowrap;color:#62736b}button{cursor:pointer}a:focus-visible,summary:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #85b9a6;outline-offset:3px}[hidden]{display:none!important}#empty{padding:25px;color:#62736b}@media(max-width:650px){main{padding:25px 16px 110px}h1{font-size:30px}.controls{flex-direction:column}.project>summary{padding:15px;font-size:16px}.project-body{padding:0 15px 12px}article{flex-wrap:wrap;gap:8px}.family-name{flex-basis:100%}.open{margin-right:auto}}
</style></head><body><main id="catalog"><header><div class="eyebrow">EXPERIMENTS / RESEARCH FAMILIES</div><h1>실험 목록</h1><p>프로젝트 → 주제 → 실험 계열 순서로 탐색합니다.<br>각 통합 뷰에서 전체·pilot, 질문 수, 하위 분석을 전환할 수 있습니다.</p></header>
<div class="controls"><input id="search" type="search" aria-label="실험 검색" placeholder="주제, 실험 계열, full/pilot 또는 분석명 검색"><select id="project" aria-label="프로젝트 선택"><option value="">전체 프로젝트</option>__OPTIONS__</select></div><p id="count" aria-live="polite"></p>__SECTIONS__<p id="empty" hidden>검색 조건에 맞는 실험 계열이 없습니다.</p>
<div id="compare-tray"><span id="selection-status" role="status">비교할 실험 계열 2개를 선택하세요 (0/2)</span><button id="clear-selection" type="button" disabled>선택 해제</button><button id="open-comparison" type="button" disabled>두 계열 함께 보기</button></div></main>
<div id="comparison" hidden><div class="comparison-toolbar"><button id="close-comparison" type="button">← 실험 목록</button><h1>실험 계열 비교</h1><button id="swap-comparison" type="button">좌우 바꾸기</button></div><div class="comparison-panes"><div class="comparison-pane"><div class="pane-toolbar"><label for="left-experiment">실험 A</label><select id="left-experiment"></select><a id="left-open" target="_blank" rel="noopener">새 탭 ↗</a></div><iframe id="left-frame" title="실험 A 결과"></iframe></div><div class="comparison-pane"><div class="pane-toolbar"><label for="right-experiment">실험 B</label><select id="right-experiment"></select><a id="right-open" target="_blank" rel="noopener">새 탭 ↗</a></div><iframe id="right-frame" title="실험 B 결과"></iframe></div></div></div>
<script>
const search=document.querySelector('#search'),project=document.querySelector('#project');
function filter(){let n=0,p=0;const q=search.value.trim().toLowerCase();document.querySelectorAll('details[data-project]').forEach(group=>{let visible=0;group.querySelectorAll('article').forEach(row=>{row.hidden=!!((project.value&&project.value!==group.dataset.project)||!row.dataset.search.includes(q));if(!row.hidden)visible++});group.querySelectorAll('[data-topic]').forEach(topic=>{topic.hidden=!Array.from(topic.querySelectorAll('article')).some(row=>!row.hidden)});group.hidden=!visible;group.open=!!visible&&!!(q||project.value);if(visible)p++;n+=visible});document.querySelector('#count').textContent=p+'개 프로젝트 · '+n+'개 실험 계열';document.querySelector('#empty').hidden=!!n;sessionStorage.setItem('experiment-filter',JSON.stringify([search.value,project.value]))}
try{const saved=JSON.parse(sessionStorage.getItem('experiment-filter'));if(saved){[search.value,project.value]=saved}}catch(e){}search.addEventListener('input',filter);project.addEventListener('change',filter);filter();
</script><script src="_lib/index_compare.js"></script></body></html>'''
    document = document.replace('__OPTIONS__', options).replace('__SECTIONS__', '\n'.join(sections))
    target = ROOT / 'index.html'
    target.write_text(document)
    print(f'Built {target}: {len(families)} experiment families across {len(PROJECTS)} projects')


if __name__ == '__main__':
    build()
