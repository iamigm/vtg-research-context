"""Keep reviewed report sections in the main MD and move detail to appendix MD.

Parts retain their original bytes and ordering. Rebuilding first reconstructs the
source from both documents; a freshly generated unsplit report replaces that
source. No predictions or experiment artifacts are copied or recalculated.
"""
import hashlib
import re
from pathlib import Path

ROOT = Path('/home/irteam/experiments')
PART = re.compile(r'<!-- report-part:(\d+):begin -->\n(.*?)<!-- report-part:\1:end -->\n', re.S)
HEADER = re.compile(r'<!-- report-appendix:(\S+) parts=(\d+) -->\n')
# Section choices are semantic, shared by all query counts of the same analysis.
POLICIES = [
 ('dataset_metadata/', r'먼저 알아둘|1\. 지금|5\. 쉽게|9\. 계산'),
 ('attention/', r'데이터셋별 1위|원래 평균 곡선|기준 변경|파일'),
 ('similarity/results/frame_query/', r'^설정|^지표 정의|^순위 성능$|^GT mass|^검증 및 한계'),
 ('similarity/results/compute_cost/', r'비교 조건|샘플당 시간|Instruct 대비|측정 정의|FLOPs 정의|정량 성능'),
 ('failure_cases/', r'정확한 입력|성능|Qwen2.5 입력|검증과 산출물'),
 ('baseline/all_zeroshot/', r'전체 실행 목록|전체 벤치마크|동일 pilot|해석 및 중복'),
 ('rollout/deterministic/', r'^영상 검토|^샘플별 분기|^Oracle interval overlap'),
 ('rollout/event_tree_search/', r'^지표$|^전체 성능$|^Analytic coverage audit|^범위별 결과'),
 ('rollout/probability_budget/', r'^후보 품질|^실제 연산 비용|^Greedy 대비 오버헤드|^해석 범위'),
 ('rollout/temporal_mode_search/', r'^비용과 관측 mode|^재현과 검증'),
 ('sampling/spatial_allocation/budget_comparison/', r'^실제 입력 크기'),
 ('sampling/spatial_allocation/allocation_sweep/', r'^비용$'),
 ('sampling/fps_spatial_control/', r'^실제 입력과 실패'),
 ('uncertainty/span_head/', r'^비교 대상|^정량 결과|^Paired 변화|^해석$'),
 ('digit_correctness/blue_analysis/', r'^A와 B의 정의|^관측 결과와 해석|^원자료'),
]


def policy(path):
    rel = path.relative_to(ROOT).as_posix()
    if '/reconsideration/' in rel and '/shared/' not in rel:
        return r'^지표와 비교|^전체 성능|^Paired comparison|^Reconsideration 프로세스'
    if '/oracle_diversity/' in rel:
        return r'^Measurement|^Oracle performance|^Greedy → oracle · K=32|^Reproducibility'
    if '/end_to_end/' in rel:
        return r'^최종 답변|^답변 변화|^해석 범위'
    if '/capped_rollouts/' in rel:
        return r'^예산 비교|^연산 비용과 성능 대비 비용|^Independent trace work|^실제 시도|^사례 탐색'
    if '/shared/design_principles/' in rel:
        return r'^1\.|^2\.|^10\.|^12\.|^실행 명세'
    if '/rollout/stochastic/' in rel or '/rollout/conditional_budget/' in rel:
        return r'.*'  # Only the long method block moves out of these reports.
    return next((rule for prefix, rule in POLICIES if prefix in rel), None)


def parts(text):
    """Split on headings outside fenced code and preserve method/details blocks."""
    result = []; start = 0; offset = 0; fenced = False; methods = False
    for line in text.splitlines(keepends=True):
        if line.startswith('<!-- rollout-methods:begin -->'):
            if offset > start: result.append(text[start:offset])
            start = offset; methods = True
        elif methods and line.startswith('<!-- rollout-methods:end -->'):
            result.append(text[start:offset + len(line)]); start = offset + len(line); methods = False
        elif not methods:
            if line.startswith('```'): fenced = not fenced
            if not fenced and re.match(r'#{2,4} ', line):
                if offset > start: result.append(text[start:offset])
                start = offset
        offset += len(line)
    if start < len(text): result.append(text[start:])
    assert ''.join(result) == text
    return result


def read_source(path):
    text = path.read_text()
    header = HEADER.search(text)
    if not header: return text
    appendix = path.with_name(header[1])
    found = PART.findall(text) + PART.findall(appendix.read_text())
    ordered = sorted((int(key), value) for key, value in found)
    assert [key for key, _ in ordered] == list(range(int(header[2]))), f'Incomplete appendix: {path}'
    return ''.join(value for _, value in ordered)


def tables(text):
    return re.findall(r'^\|[^\n]+\n\|[-: |]+\n(?:\|[^\n]*\n)+', text, re.M)


def compact_tables(text, kind):
    """Copy only explicitly selected aggregate rows; full tables stay in appendix."""
    if kind == 'fps':
        table = tables(text)[0].splitlines()
        rows = [line for line in table[2:] if line.split('|')[3].strip() == 'all']
        assert len(rows) == 20
        return '## 전체 query의 FPS 비교\n\n' + '\n'.join(table[:2] + rows) + '\n\n'
    if kind == 'budget':
        output = ['## 데이터셋·공간 조건별 핵심 결과', '',
                  '| Dataset | Spatial target | 조건 | mIoU % (Δpp) | R@.3 % (Δpp) | R@.5 % (Δpp) | R@.7 % (Δpp) |',
                  '|---|---|---|---:|---:|---:|---:|']
        ci = []
        for section in re.split(r'(?=^## )', text, flags=re.M):
            if not re.match(r'## (Charades|ActivityNet|QVHighlights)', section): continue
            dataset = section.splitlines()[0][3:].split(' —')[0]
            for subsection in re.split(r'(?=^### )', section, flags=re.M):
                match = re.match(r'### (low|high)\n', subsection)
                if not match: continue
                group = tables(subsection)
                for row in group[0].splitlines()[2:]:
                    if re.match(r'\| (Uniform|GT |Random 평균)', row):
                        output.append(f'| {dataset} | {match[1]} ' + row)
                if len(group) > 1:
                    for row in group[1].splitlines()[2:]:
                        if '| mIoU |' in row: ci.append(f'| {dataset} | {match[1]} ' + row)
        assert len(output) == 4 + 18
        output += ['', '## mIoU 변화의 95% 구간', '', '| Dataset | Spatial target | 비교 | 지표 | 변화 pp [95% 구간] |', '|---|---|---|---|---:|'] + ci
        return '\n'.join(output) + '\n\n'
    if kind == 'allocation':
        output = ['## Dataset·조건별 mIoU 비교', '',
                  '| 배정 / 공간 | Dataset | Uniform | GT 전체 8배 (Δpp) | GT 경계 8배 (Δpp) |', '|---|---|---:|---:|---:|']
        for section in re.split(r'(?=^### )', text, flags=re.M):
            match = re.match(r'### ([AB] / (?:low|high))', section)
            if not match: continue
            for group in re.split(r'(?=^#### )', section, flags=re.M):
                dataset = re.match(r'#### (Charades|ActivityNet|QVHighlights)', group)
                if not dataset: continue
                rows = tables(group)[0].splitlines()[2:]
                values = {row.split('|')[1].strip(): row.split('|')[2].strip() for row in rows}
                output.append(f'| {match[1]} | {dataset[1]} | {values["Uniform"]} | {values["GT 전체 8배"]} | {values["GT 경계 8배"]} |')
        assert len(output) == 16
        return '\n'.join(output) + '\n\nR@.3/.5/.7·전체 배율·Random 대비·신뢰구간은 appendix의 전체 조건표에서 확인한다.\n\n'
    return ''


def split_report(path):
    path = Path(str(Path(path).absolute()).replace('/home1/irteam/', '/home/irteam/'))
    rule = policy(path)
    if rule is None: return None
    source = read_source(path)
    if len(source.splitlines()) < 100 and len(source) < 8500: return None
    units = parts(source); keep = []
    rel = path.relative_to(ROOT).as_posix()
    kind = 'budget' if '/budget_comparison/' in rel else 'allocation' if '/allocation_sweep/' in rel else 'fps' if '/fps_spatial_control/' in rel else ''
    if kind == 'fps':
        table = tables(units[0])[0]
        prefix, suffix = units[0].split(table, 1)
        units = [prefix, table, suffix] + units[1:]
    intro = True
    for unit in units:
        heading = re.match(r'#{2,4} (.*)', unit)
        if unit.startswith('<!-- rollout-methods:begin -->'): keep.append(False); continue
        if heading: intro = False
        selected = intro or bool(heading and re.search(rule, heading[1]))
        if kind == 'fps' and unit.startswith('|'): selected = False
        keep.append(selected)
    return save_split(path, source, units, keep, compact_tables(source, kind))


def save_split(path, source, units, keep, summary):
    if all(keep) or not any(keep): return None
    appendix = path.with_name('APPENDIX.md')
    title = source.splitlines()[0].removeprefix('# ')
    main = [f'<!-- report-appendix:{appendix.name} parts={len(units)} -->\n']
    detail = [f'# Appendix · {title}\n\n[핵심 보고서]({path.name})\n\n상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.\n\n']
    contexts = {}; shown = [{}, {}]
    for index, (unit, selected) in enumerate(zip(units, keep)):
        destination = main if selected else detail
        heading = re.match(r'(#{2,4}) (.*)', unit)
        if heading:
            depth = len(heading[1])
            contexts = {level: label for level, label in contexts.items() if level < depth}
            for level, label in contexts.items():
                if shown[selected].get(level) != label:
                    destination.append('#' * level + ' ' + label + '\n\n')
                    shown[selected][level] = label
            contexts[depth] = heading[2]
            shown[selected] = {level: label for level, label in shown[selected].items() if level < depth}
            shown[selected][depth] = heading[2]
        rendered = f'<!-- report-part:{index}:begin -->\n{unit}<!-- report-part:{index}:end -->\n'
        destination.append(rendered)
    core = ''.join(unit for unit, selected in zip(units, keep) if selected)
    if ('rollout-methods:begin' in source and 'mIoU=' not in core and 'mIoU =' not in core) or ('/budget_comparison/' in str(path) and 'mIoU는' not in core):
        main.append('## 핵심 지표 정의\n\nIoU=예측·GT 시간 구간의 교집합 길이/합집합 길이, mIoU=평가 query의 IoU 평균, R@τ=IoU≥τ인 query 수/평가 query 수(τ=.3/.5/.7)다. 모두 높을수록 좋으며 % 표시는 100을 곱한다. Δpp=100×(방법−기준)이다. Oracle은 GT로 고른 후보 집합의 사후 상한이며 실제 선택 성능과 구분한다. 반복 응답은 query 안에서 평균한 뒤 query 간 집계한다.\n\n')
    if summary: main.append(summary)
    main.append(f'## Appendix\n\n[상세 방법·조건별 결과·검증]({appendix.name})는 상단 `appendix` 메뉴에서도 열 수 있다.\n')
    main_text = ''.join(main); detail_text = ''.join(detail)
    reconstructed = ''.join(v for _, v in sorted((int(k), v) for k, v in PART.findall(main_text) + PART.findall(detail_text)))
    assert reconstructed == source, path
    for target, value in [(appendix, detail_text), (path, main_text)]:
        if not target.exists() or target.read_text() != value: target.write_text(value)
    return dict(path=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(source.encode()).hexdigest(),source_lines=len(source.splitlines()),main_lines=len(PART.sub(lambda m:m[2],main_text).splitlines()),appendix=str(appendix.relative_to(ROOT)),parts=len(units))


def build_report_appendices(projects):
    records = []
    for project in projects:
        result = ROOT/project/'results'
        for path in sorted(result.rglob('*.md')):
            if path.name not in {'REPORT.md','FINAL_REPORT.md','final_report.md','FINDINGS.md'}: continue
            # Existing supplementary FINDINGS already lives in the appendix menu.
            if path.name == 'FINDINGS.md' and (path.with_name('REPORT.md').exists() or path.with_name('FINAL_REPORT.md').exists()): continue
            record = split_report(path)
            if record: records.append(record)
    return records
