import os,sys,json
from pathlib import Path
root=Path('/home/irteam/experiments');shared=root/'_shared/timelens'
sys.path.insert(0,str(shared/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:8780/'
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);p=b.new_page(viewport={'width':1440,'height':1000})
 errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
 p.goto(BASE);p.goto(BASE+'zero-shot/results/rollout/stochastic/viewer.html');p.locator('#family-app[data-ready=true]').wait_for()
 assert p.locator('#family-sidebar').is_hidden()
 assert p.locator('#toggle-sidebar').inner_text()=='목록'
 assert p.locator('.home-link').count()==0
 assert p.locator('#tab-result').get_attribute('aria-selected')=='true'
 assert not p.locator('#tab-report').is_disabled()
 p.locator('#tab-report').click();p.wait_for_function("document.querySelector('#report h1') !== null")
 report_url=p.url
 assert p.locator('#tab-report').get_attribute('aria-selected')=='true'
 assert p.locator('#report').is_visible()
 assert not p.locator('[data-history-back]').is_disabled()
 p.locator('[data-history-back]').click();p.wait_for_function("document.querySelector('#tab-result')?.getAttribute('aria-selected')==='true'")
 assert not p.locator('[data-history-forward]').is_disabled()
 p.locator('[data-history-forward]').click();p.wait_for_function("document.querySelector('#tab-report')?.getAttribute('aria-selected')==='true'")
 assert p.url==report_url
 p.locator('#tab-report').focus();p.locator('#tab-report').press('ArrowLeft')
 p.wait_for_function("document.querySelector('#tab-result')?.getAttribute('aria-selected')==='true'")
 p.locator('#toggle-sidebar').click();assert p.locator('#family-sidebar').is_visible()
 p.locator('#toggle-sidebar').click();assert p.locator('#family-sidebar').is_hidden()
 p.evaluate('document.fonts.ready');p.screenshot(path='/tmp/view-controls-desktop.png')
 p.set_viewport_size({'width':390,'height':844})
 assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 p.screenshot(path='/tmp/view-controls-mobile.png')
 # Actual cross-document back and forward, including the index's forward arrow.
 p.goto(BASE);p.goto(BASE+'zero-shot/results/baseline/all_zeroshot/viewer.html');p.locator('#family-app[data-ready=true]').wait_for()
 assert p.locator('#tab-result').is_disabled() and p.locator('#tab-report').get_attribute('aria-selected')=='true'
 p.locator('[data-history-back]').click();p.wait_for_url(BASE)
 p.locator('#catalog [data-history-forward]').click();p.wait_for_selector('#family-app[data-ready=true]')
 assert 'all_zeroshot' in p.url
 # Opening a new experiment tab keeps real browser history boundaries.
 single=b.new_page();single.goto(BASE+'zero-shot/results/baseline/all_zeroshot/viewer.html');single.locator('#family-app[data-ready=true]').wait_for()
 assert single.locator('[data-history-back]').is_disabled() and single.locator('[data-history-forward]').is_disabled()
 assert not errors,errors
 b.close()
result=dict(passed=True,result_report_tabs=True,keyboard=True,history_navigation=True,new_tab_history_boundaries=True,sidebar_collapsed=True,mobile=True)
(root/'artifacts/maintenance/view_controls_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
