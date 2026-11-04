import os,sys,json
from pathlib import Path
root=Path('/home/irteam/experiments');shared=root/'_shared/timelens'
sys.path.insert(0,str(shared/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
from family_catalog import build_navigation,analysis_parts,scope_info
from family_view_test_utils import select_view
BASE='http://127.0.0.1:8780/'
# A third-level subexperiment is discovered without any experiment-specific UI branch.
fixture=[]
for scope,path,kind,name in [('pilot/300',[], 'html','viewer.html'),('pilot/300',['search','calibration','temperature'], 'html','viewer.html'),('pilot/300',['search','calibration','temperature'], 'md','REPORT.md'),('pilot/600',['report_only'], 'md','REPORT.md')]:
 id=scope+('/'+'/'.join(path) if path else '')+'/'+name
 fixture.append(dict(id=id,kind=kind,scope_key=scope,scope=scope,queries=int(scope.split('/')[1]),analysis_path=path,report=id if kind=='md' else id.replace('viewer.html','REPORT.md') if path else None))
assert analysis_parts(Path('pilot/300/region/full/REPORT.md'))==['region','full']
assert scope_info(Path('pilot/300/region/full/REPORT.md'))==('pilot/300','pilot · 300 queries',300)
assert scope_info(Path('overview/REPORT.md'))==('overview','family overview',0)
nav=build_navigation('Example',fixture)
assert len(nav['children'][0]['children'])==2
node=nav['children'][0]['children'][0]['children'][0]
for part in ('search','calibration','temperature'):
 node=next(child for child in node['children'] if child['path'][-1]==part)
assert node['result'].endswith('temperature/viewer.html') and node['report'].endswith('temperature/REPORT.md')
report_only=nav['children'][0]['children'][1]['children'][0]['children'][0]
assert report_only['result'] is None and report_only['report'].endswith('report_only/REPORT.md')
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);p=b.new_page(viewport={'width':1440,'height':1000})
 errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
 p.goto(BASE);p.goto(BASE+'zero-shot/results/rollout/stochastic/viewer.html');p.locator('#family-app[data-ready=true]').wait_for()
 assert p.locator('#family-sidebar').is_hidden()
 assert p.locator('#toggle-sidebar').inner_text()=='실험'
 assert p.locator('.home-control').count()==1
 assert p.locator('#analysis,#analysis-control,.toolbar').count()==0
 assert p.locator('#analysis-tabs .analysis-tab-group').count()==1
 assert p.locator('.family-heading').bounding_box()['y']<p.locator('.viewer-controls').bounding_box()['y']
 assert p.locator('#scope').count()==0
 assert p.locator('#view-browser').evaluate('(el)=>!!el.closest(".viewer-controls")')
 assert p.locator('#analysis-tabs').evaluate('(el)=>!!el.closest(".viewer-controls")')
 assert p.locator('#view-browser').is_visible()
 p.locator('#view-browser>summary').click()
 assert p.locator('#view-tree').is_visible()
 p.locator('#view-browser>summary').press('Escape')
 assert p.locator('#view-browser').get_attribute('open') is None
 assert p.locator('#tab-result').get_attribute('aria-selected')=='true'
 assert p.locator('#source-md').is_hidden()
 assert p.locator('#source-md').get_attribute('href') is None
 assert not p.locator('#tab-report').is_disabled()
 p.locator('#tab-report').click();p.wait_for_function("document.querySelector('#report h1') !== null")
 report_url=p.url
 assert p.locator('#tab-report').get_attribute('aria-selected')=='true'
 assert p.locator('#report').is_visible()
 assert p.locator('#source-md').is_visible()
 raw_url=p.locator('#source-md').get_attribute('href')
 with p.expect_popup() as popup:
  p.locator('#original').click()
 rendered=popup.value;rendered.locator('#report h1').wait_for()
 assert rendered.url==report_url
 assert rendered.locator('#tab-report').get_attribute('aria-selected')=='true'
 rendered.close()
 with p.expect_popup() as popup:
  p.locator('#source-md').click()
 raw=popup.value;raw.wait_for_load_state()
 assert raw.url==raw_url
 assert raw.evaluate('document.characterSet')=='UTF-8'
 assert raw.locator('body').inner_text()==p.request.get(raw_url).body().decode('utf-8')
 raw.close()
 assert not p.locator('[data-history-back]').is_disabled()
 p.locator('[data-history-back]').click();p.wait_for_function("document.querySelector('#tab-result')?.getAttribute('aria-selected')==='true'")
 assert not p.locator('[data-history-forward]').is_disabled()
 p.locator('[data-history-forward]').click();p.wait_for_function("document.querySelector('#tab-report')?.getAttribute('aria-selected')==='true'")
 assert p.url==report_url
 p.locator('#tab-report').focus();p.locator('#tab-report').press('ArrowLeft')
 p.wait_for_function("document.querySelector('#tab-result')?.getAttribute('aria-selected')==='true'")
 assert p.locator('#source-md').is_hidden()
 p.locator('#toggle-sidebar').click();assert p.locator('#family-sidebar').is_visible()
 p.locator('#toggle-sidebar').click();assert p.locator('#family-sidebar').is_hidden()
 p.set_viewport_size({'width':390,'height':844})
 assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 # Actual cross-document back and forward, including the index's forward arrow.
 p.goto(BASE);p.goto(BASE+'zero-shot/results/baseline/all_zeroshot/viewer.html');p.locator('#family-app[data-ready=true]').wait_for()
 assert p.locator('#tab-result').is_disabled() and p.locator('#tab-report').get_attribute('aria-selected')=='true'
 p.locator('[data-history-back]').click();p.wait_for_url(BASE)
 p.locator('#catalog [data-history-forward]').click();p.wait_for_selector('#family-app[data-ready=true]')
 assert 'all_zeroshot' in p.url
 # Opening a new experiment tab keeps real browser history boundaries.
 single=b.new_page();single.goto(BASE+'zero-shot/results/baseline/all_zeroshot/viewer.html');single.locator('#family-app[data-ready=true]').wait_for()
 assert single.locator('[data-history-back]').is_disabled() and single.locator('[data-history-forward]').is_disabled()
 single.locator('.home-control').click();single.wait_for_url('**/index.html')
 assert single.locator('#catalog').is_visible()
 # Check every registered Markdown endpoint, including nested subexperiments.
 catalog=json.loads((root/'artifacts/maintenance/family_catalog.json').read_text())
 document_count=0
 for family in catalog['families']:
  manifest=json.loads((root/family['manifest']).read_text())
  for entry in manifest['entries']:
   if entry['kind']!='md':continue
   response=p.request.get(BASE+str(Path(family['href']).parent/entry['href']))
   assert response.status==200,(family['href'],entry['id'],response.status)
   assert response.headers['content-type']=='text/plain; charset=utf-8'
   response.body().decode('utf-8');document_count+=1
 for entry in ('pilot/600/viewer.html','pilot/600/reconsideration/viewer.html'):
  p.goto(BASE+'zero-shot/results/rollout/deterministic/viewer.html?view='+entry)
  p.locator('#family-app[data-ready=true]').wait_for()
  assert p.locator('#source-md').is_hidden()
  p.locator('#tab-report').click();p.locator('#report h1').wait_for()
  raw=b.new_page();raw.goto(p.locator('#source-md').get_attribute('href'))
  assert raw.evaluate('document.characterSet')=='UTF-8'
  assert '완료' in raw.locator('body').inner_text() or '질문' in raw.locator('body').inner_text()
  raw.close()
 assert not errors,errors
 b.close()
result=dict(passed=True,title_above_controls=True,hierarchical_navigation=True,recursive_subexperiments=True,no_body_toolbar=True,result_report_tabs=True,markdown_only_in_report=True,markdown_utf8_documents=document_count,rendered_report_new_tab=True,keyboard=True,history_navigation=True,new_tab_history_boundaries=True,sidebar_collapsed=True,home_button=True,mobile=True)
(root/'artifacts/maintenance/view_controls_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
