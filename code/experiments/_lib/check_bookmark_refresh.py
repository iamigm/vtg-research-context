"""Verify global experiment bookmarks, local analysis selection and scoped refresh."""
import json,os,sys
from pathlib import Path
ROOT=Path('/home/irteam/experiments');shared=ROOT/'_shared/timelens'
sys.path.insert(0,str(shared/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
from family_view_test_utils import select_view
BASE='http://127.0.0.1:8780/'
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
 ctx=browser.new_context(viewport={'width':1440,'height':1000});home=ctx.new_page();home.goto(BASE)
 home.evaluate("localStorage.setItem('experiment-project-layout',JSON.stringify({version:1,order:['similarity','zero-shot'],pinned:['similarity']}))")
 exp=ctx.new_page();exp.goto(BASE+'zero-shot/results/rollout/stochastic/viewer.html')
 exp.locator('#family-app[data-ready=true]').wait_for()
 exp.locator('#family-sidebar[data-bookmarks-ready=true]').wait_for(state='attached')
 assert exp.locator('#family-sidebar').is_hidden()
 exp.locator('#toggle-sidebar').click()
 total=len(json.loads((ROOT/'artifacts/maintenance/family_catalog.json').read_text())['families'])
 assert exp.locator('#experiment-nav a[data-family]').count()==total
 assert exp.locator('#experiment-nav a[aria-current=page]').count()==1
 assert exp.locator('#experiment-nav>details').first.get_attribute('data-project')=='similarity'
 assert exp.locator('#experiment-nav>details').first.get_attribute('class')=='bookmark-pinned'
 exp.locator('#bookmark-search').fill('Embedding vs generation cost (16 FPS)')
 assert exp.locator('#experiment-nav a[data-family]').count()==1
 with ctx.expect_page() as opened:exp.locator('#experiment-nav a[data-family]').click()
 opened.value.locator('#family-app[data-ready=true]').wait_for();opened.value.close()
 exp.locator('#bookmark-search').fill('not-a-real-experiment-xyz')
 assert exp.locator('#experiment-nav a[data-family]').count()==0
 exp.locator('#bookmark-search').fill('')
 assert exp.locator('#experiment-nav a[data-family]').count()==total
 # A single result/report pair is one local analysis, separate from global bookmarks.
 assert exp.locator('#analysis').count()==0
 assert exp.locator('#analysis-tabs .analysis-tab-group').count()==1
 exp.goto(BASE+'zero-shot/results/uncertainty/token_analysis/viewer.html?view=pilot%2F300%2Fintegrated%2FREPORT.md')
 exp.locator('#family-app[data-ready=true]').wait_for()
 assert exp.locator('#view-tree-items .view-tree-row').count()>1
 assert exp.locator('#analysis-tabs [aria-selected=true]').evaluate('(el)=>el.closest(".analysis-tab-group").dataset.analysis')=='integrated'
 select_view(exp,'pilot/300/digit_correctness/REPORT.md')
 exp.wait_for_url('**view=pilot%2F300%2Fdigit_correctness%2FREPORT.md')
 assert exp.locator('#tab-report').get_attribute('aria-selected')=='true'
 select_view(exp,'full/9404/digit_correctness/REPORT.md')
 exp.wait_for_url('**view=full%2F9404%2Fdigit_correctness%2FREPORT.md')
 assert exp.locator('#analysis-tabs [aria-selected=true]').evaluate('(el)=>el.closest(".analysis-tab-group").dataset.analysis')=='digit_correctness'
 exp.goto(BASE+'zero-shot/results/rollout/deterministic/viewer.html?view=pilot%2F900%2FSTRATA.md')
 exp.locator('#family-app[data-ready=true]').wait_for()
 assert exp.locator('#reference-list [aria-current=page]').get_attribute('data-view')=='pilot/900/STRATA.md'
 exp.locator('#reference-menu>summary').click()
 assert exp.locator('#reference-list [aria-current=page]').is_visible()
 exp.locator('#reference-menu>summary').press('Escape')
 assert exp.locator('#reference-menu').get_attribute('open') is None
 assert exp.locator('#tab-result').is_enabled()
 # Refresh the current report without reloading index, the outer page or bookmarks.
 exp.locator('#family-sidebar[data-bookmarks-ready=true]').wait_for(state='attached')
 exp.locator('#toggle-sidebar').click()
 exp.evaluate("window.outerMarker=123;document.querySelector('#experiment-nav').marker=456")
 home.evaluate('window.indexMarker=789')
 before_url=exp.url;before_history=exp.evaluate('history.length')
 with exp.expect_response(lambda response: response.url.endswith('/pilot/900/STRATA.md')):
  exp.locator('[data-refresh-current]').click()
 exp.wait_for_function("document.querySelector('#report h1')!==null")
 assert exp.url==before_url and exp.evaluate('history.length')==before_history
 assert exp.evaluate('window.outerMarker')==123 and home.evaluate('window.indexMarker')==789
 assert exp.evaluate("document.querySelector('#experiment-nav').marker")==456
 assert exp.locator('#family-sidebar').is_visible()
 # Index refresh affects its own document, retaining saved ordering and filters.
 home.locator('#search').fill('Rollout')
 home.locator('#catalog [data-refresh-current]').click();home.wait_for_load_state()
 assert home.evaluate('window.indexMarker') is None
 assert home.locator('#search').input_value()=='Rollout'
 assert exp.evaluate('window.outerMarker')==123
 assert home.locator('#pinned-projects>.project').first.get_attribute('data-project')=='similarity'
 # Interactive refresh reloads the detail, retaining the family context.
 exp.goto(BASE+'zero-shot/results/rollout/stochastic/viewer.html');exp.locator('#family-app[data-ready=true]').wait_for()
 exp.frame_locator('#detail-frame').locator('body').wait_for()
 exp.evaluate("window.outerMarker=321;document.querySelector('#detail-frame').contentWindow.innerMarker=654")
 before_url=exp.url;before_history=exp.evaluate('history.length')
 target=exp.evaluate("document.querySelector('#detail-frame').contentWindow.location.href")
 with exp.expect_response(lambda response: response.url==target and response.request.resource_type=='document'):
  exp.locator('[data-refresh-current]').click()
 exp.wait_for_function("document.querySelector('#detail-frame').contentWindow.innerMarker===undefined")
 assert exp.evaluate('window.outerMarker')==321 and exp.url==before_url
 assert exp.evaluate('history.length')==before_history
 exp.locator('#toggle-sidebar').click();exp.evaluate('document.fonts.ready')
 exp.set_viewport_size({'width':390,'height':844})
 assert exp.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 browser.close()
result=dict(passed=True,global_experiment_bookmarks=total,saved_project_order=True,bookmark_search=True,
            bookmark_new_tab=True,scope_and_analysis=True,supplementary_documents=True,
            independent_index_refresh=True,independent_experiment_refresh=True,refresh_preserves_context=True,mobile=True)
(ROOT/'artifacts/maintenance/bookmark_refresh_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
