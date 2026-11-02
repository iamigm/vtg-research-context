from result_layout import relocate_path, write_result_text
from timelens_paths import RESOURCES
"""Exercise the unified report's search and links in an offline browser."""
import json
import os
import sys
from paths import ROOT, RESULTS

sys.path.insert(0, str(relocate_path(RESOURCES / 'browser_deps')))
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(relocate_path(RESOURCES / 'browser_cache'))
os.environ['LD_LIBRARY_PATH'] = str(relocate_path(RESOURCES / 'browser_libs/usr/lib/x86_64-linux-gnu')) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['FONTCONFIG_FILE'] = str(relocate_path(RESOURCES / 'browser_libs/fonts.conf'))
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1600, 'height': 1100})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto((relocate_path(RESULTS / 'report.html')).as_uri(), wait_until='load')
    assert page.locator('#samples tr:visible').count() == 300
    for dataset in ['Charades-STA', 'QVHighlights', 'ActivityNet-Captions']:
        page.locator('#search').fill(dataset)
        assert page.locator('#samples tr:visible').count() == 100
    page.locator('#search').fill('person opening a box')
    assert page.locator('#samples tr:visible').count() >= 1
    page.locator('#search').fill('no-matching-sample-92874')
    assert page.locator('#samples tr:visible').count() == 0
    page.locator('#search').fill('')
    assert page.locator('#samples tr:visible').count() == 300
    assert page.locator('nav a').count() == 7
    page.screenshot(path=str(relocate_path(RESULTS / 'preview.png')), full_page=False)
    assert not errors, errors
    browser.close()

result = dict(passed=True, samples=300, dataset_search=True, query_search=True,
              empty_search=True, reset_search=True, analysis_links=7, page_errors=errors)
write_result_text((relocate_path(RESULTS / 'browser_validation.json')), json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
