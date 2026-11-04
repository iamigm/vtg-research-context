"""Exercise saved project order, pointer drag, new tabs and the shared visual theme."""
import json, os, sys
from pathlib import Path

ROOT = Path('/home/irteam/experiments')
shared = ROOT / '_shared/timelens'
sys.path.insert(0, str(shared / 'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(shared / 'browser_cache')
os.environ['LD_LIBRARY_PATH'] = str(shared / 'browser_libs/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['FONTCONFIG_FILE'] = str(shared / 'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright

BASE = 'http://127.0.0.1:8780/'
KEY = 'experiment-project-layout'

def ids(page, zone='other'):
    return page.locator(f'#{zone}-projects > .project').evaluate_all('(rows)=>rows.map(row=>row.dataset.project)')

def project(page, name):
    return page.locator(f'details[data-project="{name}"]')

def drag(page, source, destination, after=False):
    box = project(page, source).locator('.drag-handle').bounding_box()
    target = destination.bounding_box()
    page.mouse.move(box['x'] + box['width']/2, box['y'] + box['height']/2)
    page.mouse.down()
    page.mouse.move(target['x'] + target['width']/2, target['y'] + target['height'] * (.85 if after else .15), steps=18)
    page.mouse.up()

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
    context = browser.new_context(viewport={'width':1440, 'height':1100})
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(BASE)
    defaults = ids(page)
    assert len(defaults) == 6 and not ids(page, 'pinned')
    row = project(page, 'zero-shot')
    row.locator('.pin-project').click()
    assert ids(page, 'pinned') == ['zero-shot']
    assert not row.evaluate('(row)=>row.open'), 'Pin must not expand the project'
    page.reload()
    assert ids(page, 'pinned') == ['zero-shot']
    # Pointer drag changes both order and pinned membership.
    drag(page, 'similarity', project(page, 'dataset_metadata').locator('summary'))
    assert ids(page)[0] == 'similarity'
    drag(page, 'attention', project(page, 'zero-shot').locator('summary'))
    assert ids(page, 'pinned') == ['attention', 'zero-shot']
    page.reload()
    assert ids(page, 'pinned') == ['attention', 'zero-shot'] and ids(page)[0] == 'similarity'
    project(page, 'zero-shot').locator('.move-up').click()
    assert ids(page, 'pinned') == ['zero-shot', 'attention']
    assert not project(page, 'zero-shot').evaluate('(row)=>row.open')
    # Saved choices propagate to another open catalog tab.
    second = context.new_page(); second.goto(BASE)
    project(page, 'attention').locator('.pin-project').click()
    second.wait_for_function("document.querySelector('#pinned-projects').querySelectorAll('.project').length === 1")
    assert ids(second, 'pinned') == ['zero-shot']
    second.close()
    page.locator('#search').fill('deterministic rollout')
    assert page.locator('article:visible').count() == 1
    assert ids(page, 'pinned') == ['zero-shot']
    page.locator('#search').fill('nothing-found-xyz')
    assert page.locator('#empty').is_visible()
    page.locator('#search').fill('')
    page.locator('#project').select_option('similarity')
    original_url = page.url
    for selector in ['article:visible h3 a', 'article:visible a.open']:
        with context.expect_page() as opened:
            page.locator(selector).first.click()
        experiment = opened.value
        experiment.wait_for_selector('#family-app[data-ready="true"]')
        assert page.url == original_url and page.locator('#catalog').is_visible()
        assert experiment.evaluate('window.opener === null')
        experiment.close()
    page.locator('article:visible input[name=compare]').nth(0).check()
    page.locator('article:visible input[name=compare]').nth(1).check()
    with context.expect_page() as opened: page.locator('#open-comparison').click()
    comparison = opened.value
    comparison.wait_for_selector('body.comparing')
    assert page.url == original_url and page.locator('#catalog').is_visible()
    assert comparison.evaluate('window.opener === null')
    for side in ['left', 'right']:
        comparison.frame_locator('#'+side+'-frame').locator('#family-app[data-ready="true"]').wait_for()
    comparison.locator('#swap-comparison').click()
    url = comparison.url; comparison.reload(); comparison.wait_for_selector('body.comparing')
    assert comparison.url == url
    comparison.close()
    page.locator('#project').select_option('')
    page.locator('#reset-project-order').click()
    assert ids(page) == defaults and not ids(page, 'pinned')
    # Escape cancels a drag without changing the saved layout.
    start = project(page, 'attention').locator('.drag-handle').bounding_box()
    target = page.locator('#pinned-projects').bounding_box()
    page.mouse.move(start['x']+10, start['y']+10); page.mouse.down()
    page.mouse.move(target['x']+30, target['y']+20, steps=12)
    page.keyboard.press('Escape'); page.mouse.up()
    assert ids(page) == defaults and not ids(page, 'pinned')
    # An empty pin zone accepts a drop, including when nothing was previously pinned.
    drag(page, 'zero-shot', page.locator('#pinned-projects'))
    assert ids(page, 'pinned') == ['zero-shot']
    page.locator('#reset-project-order').click()
    page.set_viewport_size({'width':390, 'height':844})
    project(page, 'zero-shot').locator('.pin-project').click()
    project(page, 'similarity').locator('.move-up').focus()
    project(page, 'similarity').locator('.move-up').press('Enter')
    assert ids(page, 'pinned') == ['zero-shot']
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='/tmp/experiment-index-mobile.png', full_page=True)
    page.set_viewport_size({'width':1440, 'height':1100})
    page.screenshot(path='/tmp/experiment-index-desktop.png', full_page=True)
    # Invalid storage and unavailable storage must leave navigation usable.
    page.evaluate('(key)=>localStorage.setItem(key,"{invalid")', KEY)
    page.reload(); assert ids(page) == defaults
    blocked = browser.new_context()
    blocked.add_init_script("Storage.prototype.setItem = function(){throw new Error('storage unavailable')}")
    blocked_page = blocked.new_page(); blocked_page.goto(BASE)
    project(blocked_page, 'zero-shot').locator('.pin-project').click()
    assert ids(blocked_page, 'pinned') == ['zero-shot']
    assert '현재 화면에만' in blocked_page.locator('#order-status').inner_text()
    blocked.close()
    # A real touch sequence exercises pointer capture and touch-action on phones.
    mobile = browser.new_context(viewport={'width':390, 'height':1000}, is_mobile=True, has_touch=True)
    phone = mobile.new_page(); phone.goto(BASE)
    handle = project(phone, 'attention').locator('.drag-handle').bounding_box()
    drop = phone.locator('#pinned-projects').bounding_box()
    cdp = mobile.new_cdp_session(phone)
    x, y = handle['x'] + handle['width']/2, handle['y'] + handle['height']/2
    end_x, end_y = drop['x'] + drop['width']/2, drop['y'] + drop['height']/2
    cdp.send('Input.dispatchTouchEvent', {'type':'touchStart', 'touchPoints':[{'x':x, 'y':y}]})
    for step in range(1, 13):
        cdp.send('Input.dispatchTouchEvent', {'type':'touchMove', 'touchPoints':[{'x':x+(end_x-x)*step/12, 'y':y+(end_y-y)*step/12}]})
    cdp.send('Input.dispatchTouchEvent', {'type':'touchEnd', 'touchPoints':[]})
    assert ids(phone, 'pinned') == ['attention']
    phone.reload(); assert ids(phone, 'pinned') == ['attention']
    mobile.close()
    # Verify every registered detail uses the same body/header/control theme.
    families = json.loads((ROOT/'artifacts/maintenance/family_catalog.json').read_text())['families']
    checked = 0
    for family in families:
        manifest = json.loads((ROOT/family['manifest']).read_text())
        for entry in manifest['entries']:
            if entry['kind'] != 'html': continue
            detail = context.new_page()
            # Some existing details construct thousands of options on load.
            detail.goto(BASE + str((Path(family['href']).parent / entry['href'])), wait_until='domcontentloaded', timeout=90000)
            assert detail.locator('link[data-experiment-theme]').count() == 1
            assert detail.evaluate("getComputedStyle(document.body).backgroundColor") == 'rgb(245, 247, 244)', entry['href']
            assert detail.locator('meta[name=viewport]').count() == 1
            assert detail.locator('[data-experiment-home]').is_visible()
            if entry['href'] == 'pilot/300/report.html' and family['family'] == 'stochastic':
                detail.screenshot(path='/tmp/experiment-detail-desktop.png')
            checked += 1
            detail.close()
    assert not errors, errors
    browser.close()

result = dict(passed=True, saved_order=True, pin_and_unpin=True, pointer_drag=True, touch_drag=True, drag_cancel=True,
              keyboard_controls=True, cross_tab_sync=True, new_experiment_tabs=True,
              new_comparison_tab=True, search=True, mobile=True, storage_fallback=True,
              themed_details=checked)
(ROOT/'artifacts/maintenance/index_ui_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
