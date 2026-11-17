"""Check the new appendix through the existing family report navigation."""
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urljoin
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8780/'
FAMILY = 'zero-shot/results/uncertainty/token_analysis/'


def check():
    resources = ROOT.parent / '_shared/timelens'
    sys.path.insert(0, str(resources / 'browser_deps'))
    os.environ.update(
        PLAYWRIGHT_BROWSERS_PATH=str(resources / 'browser_cache'),
        LD_LIBRARY_PATH=str(resources / 'browser_libs/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
        FONTCONFIG_FILE=str(resources / 'browser_libs/fonts.conf'))
    from playwright.sync_api import sync_playwright
    errors, records = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1080})
        page.on('pageerror', lambda error: errors.append(str(error)))
        for scope in ('pilot/300', 'full/9404'):
            entry = scope + '/entropy_error/REPORT.md'
            dest = ROOT / 'results/uncertainty/token_analysis' / scope / 'entropy_error'
            text = (dest / 'REPORT.md').read_text()
            assert 'top-30 재정규화' in text
            assets = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
            assert assets, 'No scientific figure in appendix'
            for asset in assets:
                with urlopen(urljoin(BASE + FAMILY + entry, asset)) as response:
                    assert response.status == 200
            page.goto(BASE + FAMILY + 'viewer.html?view=' + scope + '/REPORT.md')
            page.wait_for_selector('#family-app[data-ready="true"]')
            page.locator('#reference-menu summary').click()
            page.locator('#reference-list a[data-view="' + entry + '"]').click()
            page.wait_for_function('document.querySelector("#report").textContent.includes("top-30 재정규화")')
            page.wait_for_function('() => [...document.querySelectorAll("#report img")].length > 0 && [...document.querySelectorAll("#report img")].every(i=>i.complete && i.naturalWidth>0)')
            assert page.locator('#report table').count() >= 4
            assert page.locator('#detail-frame').is_hidden()
            saved_url = page.url
            page.reload()
            page.wait_for_function('() => [...document.querySelectorAll("#report img")].length > 0 && [...document.querySelectorAll("#report img")].every(i=>i.complete && i.naturalWidth>0)')
            assert page.url == saved_url
            page.evaluate('document.fonts.ready')
            page.screenshot(path=str(dest / 'preview.png'))
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            page.screenshot(path=str(dest / 'preview_mobile.png'))
            page.set_viewport_size({'width': 1440, 'height': 1080})
            page.locator('#tab-report').click()
            page.wait_for_function('document.querySelector("#report").textContent.includes("무엇을 먼저 보는가")')
            page.locator('#tab-result').click()
            page.frame_locator('#detail-frame').locator('#path-plot circle').first.wait_for()
            records.append({'scope': scope, 'appendix_menu': True, 'figures': len(assets), 'url_restore': True,
                            'main_report_return': True, 'sample_view_return': True, 'mobile': True})
            print(scope + ': appendix navigation and figures verified', flush=True)
        browser.close()
    assert not errors, errors
    with urlopen(BASE) as response:
        assert response.status == 200
    result = {'passed': True, 'checks': records, 'page_errors': errors}
    path = ROOT / 'artifacts/runs/uncertainty/token_analysis/overview/validation/entropy_error_appendix.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def check_saved():
    """Open canonical saved-step appendices from the Token uncertainty report."""
    from playwright.sync_api import sync_playwright
    cases = [('deterministic', '900'), ('oracle_switch_controls', '60')]
    errors, checks = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1080})
        page = context.new_page()
        page.goto(BASE + FAMILY + 'viewer.html?view=full/9404/REPORT.md')
        page.wait_for_selector('#family-app[data-ready="true"]')
        for family, count in cases:
            selector = '#report a[href*="rollout/' + family + '/viewer.html"]'
            page.locator(selector).wait_for()
            with context.expect_page() as popup:
                page.locator(selector).click()
            detail = popup.value
            detail.on('pageerror', lambda e: errors.append(str(e)))
            detail.wait_for_selector('#family-app[data-ready="true"]')
            detail.locator('#report table').first.wait_for()
            detail.evaluate('document.querySelectorAll("#report img").forEach(i=>i.loading="eager")')
            detail.wait_for_function('() => [...document.querySelectorAll("#report img")].length > 0 && [...document.querySelectorAll("#report img")].every(i=>i.complete && i.naturalWidth>0)')
            entry = 'pilot/' + count + '/APPENDIX_ENTROPY.md'
            assert 'APPENDIX_ENTROPY.md' in detail.url
            assert detail.locator('#report table').count() >= 2
            dest = ROOT / 'results/rollout' / family / 'pilot' / count
            detail.evaluate('document.fonts.ready')
            detail.screenshot(path=str(dest / 'preview_entropy.png'))
            detail.set_viewport_size({'width': 390, 'height': 844})
            assert detail.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            detail.screenshot(path=str(dest / 'preview_entropy_mobile.png'))
            detail.set_viewport_size({'width': 1440, 'height': 1080})
            detail.locator('#tab-report').click()
            detail.wait_for_function('() => !location.search.includes("APPENDIX_ENTROPY")')
            detail.locator('#reference-menu summary').click()
            detail.locator('#reference-list a[data-view="' + entry + '"]').click()
            detail.wait_for_function('location.search.includes("APPENDIX_ENTROPY")')
            checks.append({'family': family, 'queries': int(count), 'canonical_scope': True,
                           'cross_family_new_tab': True, 'appendix_menu': True, 'figures': True, 'mobile': True})
            detail.close()
            print(f'{family} pilot/{count}: cross-family appendix verified', flush=True)
        browser.close()
    assert not errors, errors
    path = ROOT / 'artifacts/runs/uncertainty/token_analysis/overview/validation/entropy_error_appendix.json'
    record = json.loads(path.read_text())
    record['saved_decoding_appendices'] = checks
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    check()
    check_saved()
