"""Validate integrated decoding paths against canonical saved token distributions."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlencode, urljoin, urlsplit

ROOT = Path('/home/irteam/experiments/zero-shot')
RESULT = ROOT / 'results/uncertainty/token_analysis'
ART = ROOT / 'artifacts/runs/uncertainty/token_analysis'
LOOKUP = ART / 'overview/web/paths/lookup.json'
BASE = 'http://127.0.0.1:8780/'


def file_url(path):
    return BASE + path.relative_to(ROOT.parent).as_posix()


def local_url(url):
    parsed = urlsplit(url)
    assert parsed.netloc == '127.0.0.1:8780', url
    return ROOT.parent / parsed.path.lstrip('/')


def entropy(top):
    mass = sum(t['probability'] for t in top)
    return -sum((t['probability'] / mass) * math.log(t['probability'] / mass)
                for t in top if t['probability'] > 0)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(scopes=('pilot/300', 'pilot/600', 'pilot/900', 'full/9404')):
    resources = ROOT.parent / '_shared/timelens'
    sys.path.insert(0, str(resources / 'browser_deps'))
    os.environ.update(
        PLAYWRIGHT_BROWSERS_PATH=str(resources / 'browser_cache'),
        LD_LIBRARY_PATH=str(resources / 'browser_libs/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
        FONTCONFIG_FILE=str(resources / 'browser_libs/fonts.conf'))
    from playwright.sync_api import sync_playwright

    lookup = json.loads(LOOKUP.read_text())
    lookup_url = file_url(LOOKUP)
    assert lookup['source']['model'] == 'qwen3'
    assert lookup['source']['scope'] == 'pilot/900'
    assert len(lookup['samples']) == 900
    assert sum(x['paths'] for x in lookup['samples'].values()) == 27460
    errors, checks, snapshots, bundles, cohort_sets, representative_cases = [], [], {}, {}, {}, {}

    def bundle(key):
        if key not in bundles:
            entry = lookup['samples'][key]
            data = {}
            for field in ('record', 'index', 'metadata'):
                path = local_url(urljoin(lookup_url, entry[field]))
                snapshots[path] = digest(path)
                data[field] = json.loads(path.read_text())
            assert data['index']['sample_key'] == key
            assert data['record']['sample']['sample_key'] == key
            data['candidates'] = {str(c['index']): c for c in data['record']['result']['candidates']}
            assert len(data['candidates']) == entry['paths']
            bundles[key] = data
        return bundles[key]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
                                                       '--autoplay-policy=no-user-gesture-required'])
        page = browser.new_page(viewport={'width': 1500, 'height': 1100})
        page.on('pageerror', lambda e: errors.append(str(e)))

        def wait_path(key, candidate):
            page.wait_for_function('([sample,id]) => {const s=window.TokenViewer?.state();return s?.ready && s.sample===sample && s.source==="deterministic" && String(s.pathId)===String(id);}', arg=[key, candidate])

        def inspect_layout(key, candidate, mobile=False):
            data = bundle(key)
            row = data['candidates'][str(candidate)]
            metadata = data['metadata']['candidates'][str(candidate)]
            assert page.locator('#status').is_hidden(), 'Ready samples should not show a redundant status line'
            assert page.locator('#query').inner_text() == data['record']['sample']['query']
            assert page.locator('.sample-result-summary h2').all_text_contents() == ['Answer', 'GT', 'Prediction', 'IoU']
            assert page.locator('#answer').inner_text() == row['answer']
            assert json.loads(page.locator('#gt').inner_text().removesuffix(' 초')) == data['record']['sample']['span'][0]
            expected_prediction = [row['parsed']['start'], row['parsed']['end']] if row.get('parsed') else None
            assert json.loads(page.locator('#prediction').inner_text().removesuffix(' 초')) == expected_prediction
            iou = page.locator('#sample-iou').inner_text()
            assert (iou == '—') if metadata['iou'] is None else abs(float(iou) - metadata['iou']) <= 0.000501
            boxes = page.evaluate('''() => {
                const rect=id=>{const r=document.querySelector(id).getBoundingClientRect();return {top:r.top,bottom:r.bottom,left:r.left,right:r.right,width:r.width,height:r.height};};
                return {query:rect('#query'),video:rect('#video'),summary:rect('.sample-result-summary'),header:rect('.case-header'),hero:rect('.sample-hero'),
                        queryFont:parseFloat(getComputedStyle(document.getElementById('query')).fontSize)};
            }''')
            assert boxes['query']['bottom'] <= boxes['video']['top']
            assert page.locator('#video-box').evaluate('(e)=>e.tagName') == 'DIV'
            assert page.locator('.sample-hero #video').count() == 1
            assert page.locator('.sample-hero .sample-result-summary').count() == 1
            if mobile:
                assert boxes['summary']['top'] >= boxes['video']['bottom']
            else:
                assert boxes['summary']['left'] >= boxes['video']['right']
                assert boxes['summary']['top'] < boxes['video']['bottom']
                assert boxes['video']['width'] >= 1.5 * boxes['summary']['width']
                assert abs(boxes['header']['top'] - boxes['summary']['top']) < 1
                assert abs(boxes['header']['bottom'] - boxes['summary']['bottom']) < 1
            assert boxes['queryFont'] >= (22 if mobile else 26)
            assert boxes['video']['height'] >= (240 if mobile else 350)
            assert boxes['video']['width'] >= (280 if mobile else 650)
            tag_fonts = page.locator('#token-strip .selection-tag').evaluate_all('(es)=>es.map(e=>getComputedStyle(e).fontFamily)')
            assert all('TokenKorean' in family for family in tag_fonts), tag_fonts

        def inspect_candidate(key, candidate):
            data = bundle(key)
            row = data['candidates'][str(candidate)]
            index = data['index']
            node_ids = index['candidates'][str(candidate)]['nodes']
            assert page.locator('#path-plot circle[data-step]').count() == len(row['ids'])
            assert page.locator('#token-strip button[data-step]').count() == len(row['ids'])
            assert page.locator('#answer').inner_text() == row['answer']
            assert page.locator('#distribution-mode').input_value() == 'raw'
            assert page.locator('#distribution-mode option[value="processed"]').evaluate('(e)=>e.disabled')
            assert page.locator('#mass-summary').evaluate('(e)=>e.children.length') == 0
            assert page.locator('#additional').is_hidden()
            inspect_layout(key, candidate)
            forced = set()
            ancestor = row
            while ancestor['branch']:
                forced.add(ancestor['branch']['position'] + 1)
                ancestor = data['candidates'][str(ancestor['branch']['parent'])]
            state = page.evaluate('TokenViewer.state()')
            assert sorted(state['forcedSteps']) == sorted(forced)
            metadata = data['metadata']['candidates'][str(candidate)]
            expected_metrics = metadata.get('endpoints', {}).get(state['endpoint'], {}).get('metrics', {})
            actual_metrics = state.get('path') or {}
            for field in ('error_kind', 'error_step', 'delta_previous_token', 'delta_previous', 'error_selection_kind'):
                expected, actual = expected_metrics.get(field), actual_metrics.get(field)
                if isinstance(expected, float):
                    assert actual is not None and abs(actual - expected) < 1e-10, (field, expected, actual)
                else:
                    assert actual == expected, (field, expected, actual)
            conditioned_length = len(row['branch']['forced_prefix_ids']) if row['branch'] else 0
            max_entropy_error = 0.0
            for offset, (token, node_id) in enumerate(zip(row['ids'], node_ids)):
                step = offset + 1
                node = index['nodes'][node_id]
                chosen = next(t for t in node['observed'] if t['id'] == token)
                expected_h = entropy(node['top'])
                point = page.locator(f'#path-plot circle[data-step="{step}"]')
                match = re.search(r'entropy\s+([\d.eE+-]+)', point.get_attribute('aria-label') or '')
                assert match, point.get_attribute('aria-label')
                error = abs(float(match[1]) - expected_h)
                max_entropy_error = max(max_entropy_error, error)
                assert error <= 0.000501, (key, candidate, step, expected_h, match[1])
                token_text = page.locator(f'#token-strip button[data-step="{step}"] span').text_content()
                assert json.loads(token_text) == chosen['token']
                kind = 'forced' if step in forced else 'conditioned' if offset < conditioned_length else 'free'
                assert point.get_attribute('data-selection-kind') == kind, (key, candidate, step, kind)
            # Inspect a distribution at an actual fork and the free step after it.
            inspect_steps = [1, min(len(row['ids']), conditioned_length + 1)]
            if row['branch']:
                inspect_steps.append(row['branch']['position'] + 1)
            for step in sorted(set(inspect_steps)):
                page.locator(f'#token-strip button[data-step="{step}"]').click()
                assert page.evaluate('TokenViewer.state().step') == step
                node = index['nodes'][node_ids[step - 1]]
                mass = sum(t['probability'] for t in node['top'])
                cells = page.locator('#candidate-rows tr td:nth-child(3)').all_text_contents()
                observed = [float(v.replace('%', '')) / 100 for v in cells]
                assert len(observed) == len(node['top'])
                assert max(abs(a - b['probability'] / mass) for a, b in zip(observed, node['top'])) < 1e-7
                displayed_tokens = page.locator('#candidate-rows tr td:nth-child(2)').all_text_contents()
                assert [json.loads(t) for t in displayed_tokens] == [t['token'] for t in node['top']]
                assert set(page.locator('#candidate-rows tr td:nth-child(5)').all_text_contents()) == {'판정 자료 없음'}
                bar_widths = page.locator('#candidate-bars .bar-fill').evaluate_all('(es)=>es.map(e=>parseFloat(e.style.width)/100)')
                assert len(bar_widths) == min(10, len(node['top']))
                # Browser style serialization rounds percentages to roughly six significant digits.
                assert max(abs(a - b['probability'] / mass) for a, b in zip(bar_widths, node['top'])) < 1e-6
            return {'sample': key, 'candidate': int(candidate), 'steps': len(row['ids']),
                    'forced_steps': sorted(forced), 'max_display_entropy_error': max_entropy_error}

        for scope in scopes:
            dest = RESULT / scope
            page.goto(file_url(dest / 'report.html'))
            page.wait_for_function('window.TokenViewer?.state().ready')
            manifest = page.evaluate('MANIFEST.map(s=>({sample_key:s.sample_key,dataset:s.dataset}))')
            config = page.evaluate('TOKEN_VIEWER_CONFIG')
            scope_keys = {s['sample_key'] for s in manifest}
            present = scope_keys & lookup['samples'].keys()
            if scope == 'full/9404':
                assert len(scope_keys) == 9404
                assert not config.get('saved_paths'), 'Full must not include pilot decoding paths'
                assert page.evaluate('TokenViewer.state().availablePaths') == 0
                assert page.locator('#decoding-source option[value="deterministic"]').evaluate('(e)=>e.disabled')
                assert page.locator('#path-availability option[value="deterministic"]').evaluate('(e)=>e.disabled')
                for key in (next(iter(present)), next(k for k in sorted(scope_keys) if k not in present)):
                    page.goto(file_url(dest / 'report.html') + '#' + urlencode({'model': 'qwen3', 'sample': key,
                                                                              'source': 'deterministic', 'path': '2', 'preset': 'all'}))
                    page.wait_for_function('s=>TokenViewer.state().ready && TokenViewer.state().sample===s', arg=key)
                    state = page.evaluate('TokenViewer.state()')
                    assert state['source'] == 'original' and state['availablePaths'] == 0
                    assert page.locator('#path-plot .reference-curve').count() == 0
                checks.append({'scope': scope, 'available_queries': 0, 'pilot_paths_excluded': True,
                               'invalid_path_url_falls_back_to_original': True})
                print(scope + ': pilot paths excluded and original source preserved', flush=True)
                continue
            expected_count = int(scope.rsplit('/', 1)[1])
            assert len(scope_keys) == len(present) == expected_count
            cohort = json.loads((ROOT / f'artifacts/common/cohorts/nested_pilot_v1/pilot_{expected_count}/samples.json').read_text())
            assert [s['sample_key'] for s in manifest] == [s['sample_key'] for s in cohort], 'Pilot order must match canonical cohort'
            cohort_sets[expected_count] = scope_keys
            original_available = config.get('original_available', True)
            if not original_available:
                assert page.evaluate('TokenViewer.state().source') == 'deterministic'
                assert page.locator('#decoding-source option[value="original"]').evaluate('(e)=>e.disabled')
                assert not page.locator('#model option[value="qwen25"]').count() or page.locator('#model option[value="qwen25"]').evaluate('(e)=>e.disabled')
            cases = []
            for dataset in sorted({s['dataset'] for s in manifest}):
                keys = sorted(s['sample_key'] for s in manifest if s['dataset'] == dataset and s['sample_key'] in present)
                for key in keys:
                    data = bundle(key)
                    candidates = data['candidates']
                    candidate = next((c for c in candidates.values() if c['branch'] and c['branch']['parent'] > 1
                                      and (c.get('parsed') or {}).get('positive')
                                      and data['metadata']['candidates'][str(c['index'])]['supported']), None)
                    if candidate:
                        cases.append((key, str(candidate['index'])))
                        break
            assert len(cases) == 3
            representative_cases[scope] = cases[0]
            scope_checks = []
            for key, candidate in cases:
                params = urlencode({'model': 'qwen3', 'sample': key, 'preset': 'all', 'source': 'original' if original_available else 'deterministic',
                                    'range': 'response', 'paths': 'all'})
                page.goto(file_url(dest / 'report.html') + '#' + params)
                page.wait_for_function('s=>TokenViewer.state().ready && TokenViewer.state().sample===s && TokenViewer.state().availablePaths>0', arg=key)
                expected_options = set(bundle(key)['candidates'])
                if original_available:
                    page.select_option('#distribution-mode', 'processed')
                    page.select_option('#decoding-source', 'deterministic')
                page.wait_for_function('TokenViewer.state().ready && TokenViewer.state().source==="deterministic"')
                options = page.locator('#decoding-path option').evaluate_all('(es)=>es.map(e=>e.value).filter(Boolean)')
                assert set(options) == expected_options
                page.select_option('#decoding-path', candidate)
                wait_path(key, candidate)
                scope_checks.append(inspect_candidate(key, candidate))
                parent = str(bundle(key)['candidates'][candidate]['branch']['parent'])
                page.select_option('#path-reference', 'parent')
                assert page.locator('#path-plot .reference-curve').get_attribute('data-path-id') == parent
                assert str(page.evaluate('TokenViewer.state().referencePathId')) == parent
                reference_points = page.locator('#path-plot .reference-curve').get_attribute('points').split()
                parent_nodes = bundle(key)['index']['candidates'][parent]['nodes']
                assert len(reference_points) == min(len(parent_nodes), len(bundle(key)['candidates'][candidate]['ids']))
                for xy, node_id in zip(reference_points, parent_nodes):
                    expected_y = 235 - entropy(bundle(key)['index']['nodes'][node_id]['top']) / math.log(30) * 190
                    assert abs(float(xy.split(',')[1]) - expected_y) <= 0.001
                page.select_option('#path-reference', 'greedy')
                assert page.locator('#path-plot .reference-curve').get_attribute('data-path-id') == '1'
                page.select_option('#path-reference', 'none')
                assert page.locator('#path-plot .reference-curve').count() == 0
                page.select_option('#path-reference', 'parent')
                page.click('#jump-branch')
                assert page.evaluate('TokenViewer.state().step') == bundle(key)['candidates'][candidate]['branch']['position'] + 1
                state = page.evaluate('TokenViewer.state()')
                saved_url = page.url
                page.reload()
                wait_path(key, candidate)
                restored = page.evaluate('TokenViewer.state()')
                for field in ('sample', 'model', 'source', 'pathId', 'reference', 'step', 'forcedSteps'):
                    assert restored[field] == state[field], (field, restored[field], state[field])
                assert page.url == saved_url
                # Last path selection must win, with no stale response or token table.
                others = sorted(expected_options, key=int)
                page.locator('#decoding-path').evaluate('(e,ids)=>{for(const id of ids){e.value=id;e.dispatchEvent(new Event("change",{bubbles:true}));}}', [others[-1], candidate])
                wait_path(key, candidate)
                assert page.locator('#answer').inner_text() == bundle(key)['candidates'][candidate]['answer']

            # Filter availability refers to this scope/model, and does not fabricate Qwen2.5 paths.
            page.select_option('#path-availability', 'deterministic')
            page.wait_for_function('n=>TokenViewer.state().count===n', arg=len(present))
            assert set(page.evaluate('TokenViewer.state().visible')) == present
            page.select_option('#path-availability', 'all')
            if original_available:
                page.select_option('#model', 'qwen25')
                page.wait_for_function('TokenViewer.state().ready && TokenViewer.state().model==="qwen25"')
                state = page.evaluate('TokenViewer.state()')
                assert state['source'] == 'original' and state['availablePaths'] == 0
                assert page.locator('#path-plot .reference-curve').count() == 0
                assert not page.locator('#distribution-mode option[value="processed"]').evaluate('(e)=>e.disabled')

            # Unsupported time notation still has a complete token distribution trajectory.
            clock_case = None
            for key in sorted(present):
                # Search lightweight metadata first rather than loading every index in RAM.
                entry = lookup['samples'][key]
                meta_path = local_url(urljoin(lookup_url, entry['metadata']))
                metadata = json.loads(meta_path.read_text())
                if all(c['supported'] for c in metadata['candidates'].values()):
                    continue
                record_path = local_url(urljoin(lookup_url, entry['record']))
                record = json.loads(record_path.read_text())
                candidate = next((c for c in record['result']['candidates'] if (c.get('parsed') or {}).get('format') == 'clock'), None)
                if candidate:
                    clock_case = key, str(candidate['index'])
                    break
            if expected_count == 900:
                assert clock_case, 'The complete pilot must exercise an unsupported clock-format path'
            if clock_case:
                key, candidate = clock_case
                page.goto(file_url(dest / 'report.html') + '#' + urlencode({'model': 'qwen3', 'sample': key,
                                                                          'source': 'deterministic', 'path': candidate, 'preset': 'all'}))
                wait_path(key, candidate)
                scope_checks.append(inspect_candidate(key, candidate))
                page.select_option('#chart-range', 'endpoint')
                assert page.locator('#path-plot circle[data-step]').count() > 0
                page.select_option('#chart-range', 'response')

            # Record a representative nested branch and verify video + mobile controls.
            key, candidate = cases[0]
            page.goto(file_url(dest / 'report.html') + '#' + urlencode({'model': 'qwen3', 'sample': key,
                                                                      'source': 'deterministic', 'path': candidate, 'reference': 'parent', 'preset': 'all'}))
            wait_path(key, candidate)
            page.wait_for_function('document.getElementById("video").readyState>=2')
            page.click('#play-pred')
            page.wait_for_function('!document.getElementById("video").paused')
            page.locator('#video').evaluate('(e)=>e.pause()')
            page.evaluate('document.fonts.ready')
            page.screenshot(path=str(dest / 'preview_paths.png'), full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            inspect_layout(key, candidate, mobile=True)
            page.screenshot(path=str(dest / 'preview_paths_mobile.png'), full_page=True)
            page.set_viewport_size({'width': 1500, 'height': 1100})
            checks.append({'scope': scope, 'available_queries': len(present), 'cases': scope_checks,
                           'native_top30_values': True, 'forced_ancestry': True, 'reference_paths': True,
                           'url_restore': True, 'rapid_switch': True, 'availability_filter': True,
                           'model_reset': True if original_available else 'qwen3_only', 'unsupported_notation': bool(clock_case),
                           'video_playback': True, 'mobile': True, 'query_video_and_result_cards': True,
                           'korean_selection_tag_font': True})
            print(scope + ': saved path selection, distributions, references and video verified', flush=True)
        # Every real pilot stage is independently reachable through the family viewer.
        family_url = file_url(RESULT / 'viewer.html')
        page.goto(family_url + '?' + urlencode({'view': 'pilot/300/report.html'}))
        page.wait_for_selector('#family-app[data-ready="true"]')
        for scope in scopes:
            for filename in ('report.html', 'REPORT.md'):
                assert page.locator(f'#view-tree-items [data-view="{scope}/{filename}"]').count() == 1
            page.locator('#view-browser summary').click()
            page.locator(f'#view-tree-items [data-view="{scope}/report.html"]').click()
            page.wait_for_function('n=>{const s=document.getElementById("detail-frame").contentWindow.TokenViewer?.state();return s?.ready&&s.count===n;}', arg=int(scope.rsplit('/', 1)[1]))
            frame = page.locator('#detail-frame').element_handle().content_frame()
            assert frame is not None
            config = frame.evaluate('TOKEN_VIEWER_CONFIG')
            assert bool(config.get('saved_paths')) == scope.startswith('pilot/')
            page.click('#tab-report')
            page.wait_for_function('!document.getElementById("report").hidden && document.getElementById("report").textContent.length>100')
            assert f'view={scope.replace("/", "%2F")}%2FREPORT.md' in page.url
            page.click('#tab-result')
            page.wait_for_function('n=>{const s=document.getElementById("detail-frame").contentWindow.TokenViewer?.state();return s?.ready&&s.count===n;}', arg=int(scope.rsplit('/', 1)[1]))
        key, candidate = representative_cases['pilot/300']
        page.goto(family_url + '?' + urlencode({'view': 'pilot/300/report.html'}) + '#' + urlencode(
            {'model': 'qwen3', 'sample': key, 'source': 'original', 'preset': 'all'}))
        page.wait_for_selector('#family-app[data-ready="true"]')
        page.wait_for_function('s=>{const a=document.getElementById("detail-frame").contentWindow.TokenViewer?.state();return a?.ready&&a.sample===s;}', arg=key)
        frame = page.locator('#detail-frame').element_handle().content_frame()
        frame.select_option('#decoding-source', 'deterministic')
        frame.wait_for_function('TokenViewer.state().ready && TokenViewer.state().source==="deterministic"')
        frame.select_option('#decoding-path', candidate)
        frame.select_option('#path-reference', 'parent')
        frame.click('#jump-branch')
        saved = frame.evaluate('TokenViewer.state()')
        page.wait_for_function('([id,step])=>{const p=new URLSearchParams(location.hash.slice(1));return p.get("source")==="deterministic"&&p.get("path")===id&&p.get("step")===String(step);}', arg=[candidate, saved['step']])
        page.reload()
        page.wait_for_selector('#family-app[data-ready="true"]')
        page.wait_for_function('s=>{const a=document.getElementById("detail-frame").contentWindow.TokenViewer?.state();return a?.ready&&a.sample===s;}', arg=key)
        restored = page.locator('#detail-frame').element_handle().content_frame().evaluate('TokenViewer.state()')
        for field in ('sample', 'source', 'pathId', 'reference', 'step'):
            assert restored[field] == saved[field], (field, restored[field], saved[field])
        checks.append({'family_navigation': True, 'scope_result_report_pairs': len(scopes),
                       'path_source_selection_url_restore': True})
        browser.close()
    for smaller, larger in zip(sorted(cohort_sets), sorted(cohort_sets)[1:]):
        assert cohort_sets[smaller] <= cohort_sets[larger], (smaller, larger)
    assert not errors, errors
    assert all(digest(path) == before for path, before in snapshots.items()), 'Canonical source files changed'
    result = {'passed': True, 'checks': checks, 'page_errors': errors, 'source_files_unchanged': len(snapshots),
              'canonical_queries': 900, 'canonical_paths': 27460}
    path = ART / 'overview/validation/token_paths.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    check()
