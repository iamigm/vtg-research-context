"""Build the single experiment entry point from existing result pages."""
from result_layout import relocate_path
from pathlib import Path
from html import escape, unescape
import os
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    'dataset_metadata': '데이터셋 메타데이터 통계',
    'attention': 'Attention 분석',
    'zero-shot': 'Zero-shot · TimeLens',
    'similarity': 'Embedding 유사도',
    'analysis/failure_mode_timestamp/failure_cases': '실패 사례 분석',
    'audio_caption_resources_20260930': 'Audio caption 자료 조사',
}

# Research questions group related runs without changing their stable paths.
ZERO_SHOT_GROUPS = {
    'baseline': ('기준 성능·입력 조건', '전체 집계 → 입력 순서·일관성 → Thinking/Instruct 비교'),
    'uncertainty': ('토큰 불확실성·오류 진단', 'Full-head 분포 → 엔트로피·최초 오류 → 자릿수별 정답 가능성'),
    'sampling': ('영상 샘플링·공간 통제', '공간 예산 비교 → 전체/경계 배정 → 고정 해상도 FPS 비교. GT 활용 조건은 진단 실험입니다.'),
    'rollout': ('Rollout · 이번 정리 제외', '진행 중인 실험으로 기존 HTML 링크만 유지합니다.'),
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
    if '/pilot/' in '/' + relative:
        if '/overview/' in '/' + relative:
            return 'pilot 누적 통합 · 등록된 표본 범위'
        match = re.search(r'/pilot/(\d+)(?:/|$)', '/' + relative)
        return f'pilot{match[1]} ({match[1]}질문)' if match else 'pilot'
    if '/full/' in '/' + relative:
        match = re.search(r'/full/(\d+)(?:/|$)', '/' + relative)
        return f'전체 full{match[1]} ({int(match[1]):,}질문)' if match else '전체 · 출처별 모수 구분'
    if '/shared/' in '/' + relative:
        return '공유 자료 · 표본 실험과 구분'
    if 'zero-shot/results/' in value:
        if 'full9404_pilot300' in value or '/baseline/all_zeroshot/overview/' in value or '/baseline/input_order/overview/' in value:
            return '전체 full9404 · pilot300 (질문 수)'
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
    # Rollout pages retain their existing presentation during active capture.
    if 'rollout' in str(path) and 'data-experiment-home' in text:
        return text
    href = os.path.relpath(ROOT / 'index.html', path.parent)
    badge = f'<span data-experiment-scope style="margin-left:24px">{escape(scope)}</span>' if scope else ''
    nav = f'<nav data-experiment-home style="position:relative;z-index:1000;padding:12px 20px;background:#edf4f2;color:#163b32;font:14px system-ui"><a style="color:inherit" href="{href}">← 전체 실험 목록</a>{badge}</nav>'
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
    build_pilot_catalog()
    from project_pilot_catalog import build_project_pilot_catalogs
    build_project_pilot_catalogs()
    sections = []
    total = 0
    for project, label in PROJECTS.items():
        cards = []
        grouped = {key: [] for key in ZERO_SHOT_GROUPS}
        result_root = relocate_path(ROOT / project / 'results')
        pages = list(result_root.rglob('*.html'))
        if project in PROJECTS:
            # Include reports/specifications for families without interactive pages.
            report_dirs = {p.parent for name in ('FINAL_REPORT.md', 'REPORT.md', 'final_report.md', 'FINDINGS.md', 'SPEC.md') for p in result_root.rglob(name)}
            for run in sorted(report_dirs):
                if any('rollout' in part for part in run.relative_to(result_root).parts) or list(run.glob('*.html')):
                    continue
                report = next((run / name for name in ('FINAL_REPORT.md', 'REPORT.md', 'final_report.md', 'FINDINGS.md', 'SPEC.md') if (run / name).exists()), None)
                if report:
                    pages.append(report)
        for path in sorted(pages):
            text = path.read_text()
            match = re.search(r'<title[^>]*>(.*?)</title>', text, re.S | re.I)
            title = unescape(re.sub('<[^>]+>', '', match.group(1))).strip() if match else path.parent.name
            if path.suffix == '.md':
                title = next((line.lstrip('# ').strip() for line in text.splitlines() if line.startswith('# ')), title)
            relative = path.relative_to(relocate_path(ROOT / project / 'results')).as_posix()
            href = path.relative_to(ROOT).as_posix()
            report = path.with_name('REPORT.md')
            md = f'<a class="secondary" href="{report.relative_to(ROOT).as_posix()}">보고서 MD ↗</a>' if report.exists() and path.suffix == '.html' else ''
            group = zero_shot_group(relative) if project == 'zero-shot' else None
            group_title = ZERO_SHOT_GROUPS[group][0] if group else ''
            compare = f'<label class="compare-choice"><input type="checkbox" name="compare" value="{href}"> 비교에 추가</label>' if path.suffix == '.html' else ''
            action = '실험 페이지 열기 →' if path.suffix == '.html' else ('실험 사양 MD ↗' if path.name == 'SPEC.md' else '보고서 MD ↗')
            scope = scope_label(path)
            if scope and scope not in title:
                title += " · " + scope
            card = f'<article data-search="{escape((label + " " + group_title + " " + title + " " + relative).lower(), quote=True)}"><p class="run">{escape(relative.rsplit("/", 1)[0])}</p><h3><a href="{href}">{escape(title)}</a></h3><div class="links"><a href="{href}">{action}</a>{md}</div>{compare}</article>'
            cards.append(card)
            if group:
                grouped[group].append(card)
            updated = add_navigation(path, text) if path.suffix == '.html' else text
            if updated != text:
                path.write_text(updated)
        total += len(cards)
        content = f'<div class="grid">{"".join(cards)}</div>'
        if project == 'zero-shot':
            content = ''.join(f'<div data-experiment-group><h3>{title} <small>{len(grouped[key])}</small></h3><p>{description}</p><div class="grid">{"".join(grouped[key])}</div></div>' for key, (title, description) in ZERO_SHOT_GROUPS.items() if grouped[key])
        sections.append(f'<section data-project="{project}"><h2>{label} <small>{len(cards)}</small></h2>{content}</section>')
    options = ''.join(f'<option value="{p}">{label}</option>' for p, label in PROJECTS.items())
    document = '''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실험 목록</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#203730;background:#f5f7f4}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:64px 32px}header{border-bottom:1px solid #cedbd3;padding-bottom:32px}.eyebrow{font-size:12px;letter-spacing:.2em;color:#527366}h1{font-size:clamp(32px,5vw,54px);letter-spacing:-.04em;margin:12px 0}header p{color:#62736b;line-height:1.7}.controls{display:flex;gap:12px;margin:32px 0 16px}input,select{font:inherit;border:1px solid #c7d4cc;border-radius:10px;padding:14px;background:white;color:inherit}input{flex:1;min-width:0}#count{color:#62736b;font-size:14px}section{margin-top:40px}h2{font-size:22px}small{font-size:13px;font-weight:400;color:#62736b;margin-left:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}article{background:white;border:1px solid #dbe3dd;border-radius:14px;padding:24px;display:flex;flex-direction:column;min-height:188px}article:hover{border-color:#779a86}.run{font-size:12px;overflow-wrap:anywhere;color:#718078;margin:0}h3{overflow-wrap:anywhere;font-size:18px;line-height:1.5;margin:14px 0 24px}a{color:#256650;text-decoration:none}h3 a{color:inherit}a:hover{text-decoration:underline}a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #85b9a6;outline-offset:4px}.links{margin-top:auto;display:flex;gap:20px;flex-wrap:wrap;font-size:14px}.secondary{color:#758078}[hidden]{display:none!important}#empty{padding:48px 0;color:#62736b}@media(max-width:600px){main{padding:32px 18px}.controls{flex-direction:column}article{padding:20px}}
</style><link rel="stylesheet" href="_lib/index_compare.css"></head><body><main id="catalog"><header><div class="eyebrow">EXPERIMENTS / INDEX</div><h1>실험 목록</h1><p>프로젝트별 실험과 분석 결과를 한곳에서 찾아보세요.<br>각 페이지에서 영상, 토큰, 샘플별 결과를 탐색할 수 있습니다. 두 페이지를 선택하면 한 화면에서 비교할 수 있습니다.</p></header>
<div class="controls"><input id="search" type="search" aria-label="실험 검색" placeholder="실험명, 실행 ID, 분석 주제로 검색"><select id="project" aria-label="프로젝트 선택"><option value="">전체 프로젝트</option>__OPTIONS__</select></div><p id="count" aria-live="polite">전체 __TOTAL__개 페이지</p>
__SECTIONS__<p id="empty" hidden>검색 조건에 맞는 실험이 없습니다.</p>
<div id="compare-tray"><span id="selection-status" role="status">비교할 실험 2개를 선택하세요 (0/2)</span><button id="clear-selection" type="button" disabled>선택 해제</button><button id="open-comparison" type="button" disabled>두 실험 함께 보기</button></div></main>
<div id="comparison" hidden>
<div class="comparison-toolbar"><button id="close-comparison" type="button">← 실험 목록</button><h1>두 실험 비교</h1><button id="swap-comparison" type="button">좌우 바꾸기</button></div>
<div class="comparison-panes">
<div class="comparison-pane"><div class="pane-toolbar"><label for="left-experiment">실험 A</label><select id="left-experiment"></select><a id="left-open" target="_blank" rel="noopener">새 탭 ↗</a></div><iframe id="left-frame" title="실험 A 결과"></iframe></div>
<div class="comparison-pane"><div class="pane-toolbar"><label for="right-experiment">실험 B</label><select id="right-experiment"></select><a id="right-open" target="_blank" rel="noopener">새 탭 ↗</a></div><iframe id="right-frame" title="실험 B 결과"></iframe></div>
</div></div>
<script>
const search=document.querySelector('#search'),project=document.querySelector('#project');
function filter(){let n=0;const q=search.value.trim().toLowerCase();document.querySelectorAll('section').forEach(section=>{let visible=0;section.querySelectorAll('article').forEach(card=>{card.hidden=!!((project.value&&project.value!==section.dataset.project)||!card.dataset.search.includes(q));if(!card.hidden)visible++});section.querySelectorAll('[data-experiment-group]').forEach(group=>{group.hidden=!Array.from(group.querySelectorAll('article')).some(card=>!card.hidden)});section.hidden=!visible;n+=visible});document.querySelector('#count').textContent=n+'개 페이지';document.querySelector('#empty').hidden=!!n;sessionStorage.setItem('experiment-filter',JSON.stringify([search.value,project.value]))}
try{const saved=JSON.parse(sessionStorage.getItem('experiment-filter'));if(saved){[search.value,project.value]=saved}}catch(e){}search.addEventListener('input',filter);project.addEventListener('change',filter);filter();
</script><script src="_lib/index_compare.js"></script></body></html>'''
    (ROOT / 'index.html').write_text(document.replace('__OPTIONS__', options).replace('__TOTAL__', str(total)).replace('__SECTIONS__', '\n'.join(sections)))
    print(f'Built {ROOT / "index.html"}: {total} pages')


if __name__ == '__main__':
    build()
