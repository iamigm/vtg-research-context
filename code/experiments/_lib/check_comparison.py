"""Check comparison selection, independent frames and proxy-relative navigation."""
import json
import os
import sys
from pathlib import Path
ROOT = Path('/home/irteam/experiments')
R = ROOT / '_shared/timelens'
sys.path.insert(0, str(R / 'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(R / 'browser_cache')
os.environ['LD_LIBRARY_PATH'] = str(R / 'browser_libs/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['FONTCONFIG_FILE'] = str(R / 'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright

BASE = 'http://127.0.0.1:8888/proxy/8780/'
left = 'zero-shot/results/uncertainty/token_analysis/timelens_top30/report.html'
right = 'zero-shot/results/timelens_rollout/report.html'
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage', '--autoplay-policy=no-user-gesture-required'])
    page = browser.new_page(viewport={'width': 1600, 'height': 1000})
    errors = []
    failed = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('response', lambda r: failed.append([r.status, r.url]) if r.status >= 400 else None)
    page.goto(BASE, wait_until='networkidle')
    assert page.locator('#open-comparison').is_disabled()
    page.locator(f'input[name=compare][value="{left}"]').check()
    assert page.locator('#open-comparison').is_disabled()
    page.locator('#search').fill('rollout')
    page.locator(f'input[name=compare][value="{right}"]').check()
    assert page.locator('input[name=compare]:disabled').count() == 19
    page.locator('#open-comparison').click()
    page.wait_for_url('**/?left=**')
    for side in ['left', 'right']:
        page.wait_for_function("side => {const f=document.getElementById(side+'-frame');return f.contentDocument?.querySelector('video')?.readyState>=2}", arg=side, timeout=30000)
        frame = page.frame_locator('#' + side + '-frame')
        frame.locator('video').first.evaluate('(v)=>{v.muted=true;v.play()}')
        page.wait_for_function("side=>document.getElementById(side+'-frame').contentDocument.querySelector('video').currentTime>.15", arg=side)
        frame.locator('video').first.evaluate('(v)=>v.pause()')
    boxes = [page.locator('#' + s + '-frame').bounding_box() for s in ['left', 'right']]
    assert boxes[0]['x'] < boxes[1]['x'] and boxes[0]['y'] == boxes[1]['y']
    page.locator('#right-frame').evaluate('(f)=>f.contentWindow.comparisonMarker=42')
    replacement = 'attention/results/dynamics/report.html'
    page.locator('#left-experiment').select_option(replacement)
    page.wait_for_function("document.getElementById('left-frame').contentDocument?.title.includes('Token') || document.getElementById('left-frame').contentWindow.location.pathname.includes('/dynamics/')")
    assert page.locator('#right-frame').evaluate('(f)=>f.contentWindow.comparisonMarker') == 42
    page.locator('#swap-comparison').click()
    assert page.locator('#left-frame').get_attribute('src') == right
    assert page.locator('#right-frame').get_attribute('src') == replacement
    page.reload(wait_until='networkidle')
    assert page.locator('#comparison').is_visible()
    assert page.locator('#left-frame').get_attribute('src') == right
    page.set_viewport_size({'width': 390, 'height': 844})
    boxes = [page.locator('#' + s + '-frame').bounding_box() for s in ['left', 'right']]
    assert boxes[0]['y'] < boxes[1]['y']
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.locator('#close-comparison').click()
    assert page.locator('#catalog').is_visible()
    assert page.locator('#left-frame').get_attribute('src') is None
    page.evaluate('history.back()')
    page.locator('#comparison').wait_for(state='visible')
    page.goto(BASE + '?left=https://example.com&right=' + right, wait_until='networkidle')
    assert page.locator('#catalog').is_visible()
    assert page.locator('#left-frame').get_attribute('src') is None
    assert not errors, errors
    assert not failed, failed
    browser.close()
result = dict(passed=True, proxy_path=True, selection_limit=True, both_videos_play=True,
              independent_frames=True, replacement=True, swap=True, url_restore=True,
              browser_back=True, mobile_stacking=True, invalid_path_rejected=True,
              page_errors=errors, failed_requests=failed)
(ROOT / 'artifacts/maintenance/comparison_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
