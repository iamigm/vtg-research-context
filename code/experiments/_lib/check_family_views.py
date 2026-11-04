"""Directory-aligned catalog, family navigation, cohort transitions and comparison checks."""
import hashlib,json,os,re,sys
from pathlib import Path
from urllib.parse import urljoin
ROOT=Path('/home/irteam/experiments');S=ROOT/'artifacts/maintenance/rollout_views_20261104'
sys.path.insert(0,str(ROOT/'_lib'))
from result_layout import relocate_path
from family_view_test_utils import select_view

def preservation():
 data=json.loads((S/'unified_sources.json').read_text());store=Path(data['single_source_store']);assert len(list(store.glob('*.json')))==900
 for key,sha in data['response_sha256'].items():assert hashlib.sha256((store/(key+'.json')).read_bytes()).hexdigest()==sha,key
 base=store.parent.parent
 for n in [300,600,900]:
  out=base/str(n);cohort=json.loads((out/'metadata/cohort.json').read_text());refs=json.loads((out/'metadata/source_records.json').read_text());assert len(cohort)==len(refs)==n
  assert [Path(r['path']).stem for r in refs]==[s['sample_key'] for s in cohort]
  assert all(Path(r['path']).parent==store for r in refs)
  if n<900:assert not (out/'rollouts').exists() and not (out/'inputs').exists()
 assert not (store.parent/'additional600').exists()
 assert not list(store.parent.rglob('source_batches'))
 public=[]
 for project in __import__('build_index').PROJECTS:
  for p in (ROOT/project/'results').rglob('*'):
   if p.is_dir():assert not re.search(r'(?:^|[/_])v\d+(?:$|[/_])',str(p)),p
   if p.suffix=='.html':public.append(p)
 catalog=json.loads((ROOT/'artifacts/maintenance/family_catalog.json').read_text())['families']
 assert len(catalog)<len(public)
 for family in catalog:
  m=json.loads((ROOT/family['manifest']).read_text());assert len(m['entries'])==family['entry_count']
  for e in m['entries']:
   p=ROOT/family['href'];target=p.parent/e['href'];assert target.exists(),target
   assert 'additional600' not in e['id'] and not re.search(r'(?:^|[/_])v\d+(?:$|[/_])',e['id']),e
 print('Unified900 response hashes, cohort references and catalog paths: PASS',flush=True)
 (S/'preservation.json').write_text(json.dumps(dict(passed=True,responses=900,cohorts=[300,600,900],representative_families=len(catalog),html_files=len(public)),indent=2)+'\n')

def browser_checks():
 shared=ROOT/'_shared/timelens';sys.path.insert(0,str(shared/'browser_deps'));os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache');os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','');os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
 from playwright.sync_api import sync_playwright
 catalog=json.loads((ROOT/'artifacts/maintenance/family_catalog.json').read_text())['families'];rows=[];summary={}
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
  home=browser.new_page(viewport={'width':1440,'height':1000});home.goto('http://127.0.0.1:8780/',wait_until='networkidle')
  assert home.locator('article').count()==len(catalog)
  assert home.locator('details[data-project][open]').count()==0
  assert all(href.endswith('/viewer.html') for href in home.locator('article h3 a').evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("href"))'))
  home.locator('#search').fill('no-such-experiment-xyz');assert home.locator('#empty').is_visible()
  home.locator('#search').fill('deterministic rollout');assert home.locator('article:visible').count()==1
  home.locator('#search').fill('');home.locator('#project').select_option('similarity');assert home.locator('article:visible').count()==2
  summary['index_collapsed_search_filter']=True
  for family in catalog:
   errors=[];failed=[];page=browser.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:failed.append([r.status,r.url]) if r.status>=400 and r.url.startswith('http://127.0.0.1:8780/') else None)
   row=dict(family=family['href'])
   try:
    page.goto('http://127.0.0.1:8780/'+family['href']);page.wait_for_selector('#family-app[data-ready="true"]',timeout=45000)
    assert page.locator('#family-sidebar').is_hidden()
    assert page.locator('#toggle-sidebar').get_attribute('aria-expanded')=='false'
    page.locator('#toggle-sidebar').click();assert page.locator('#family-sidebar').is_visible()
    page.wait_for_timeout(700)
    page.locator('#experiment-nav a[aria-current=page]').wait_for()
    assert page.locator('#experiment-nav a[aria-current=page]').count()==1
    assert page.locator('#analysis-tabs [aria-selected=true]').count()==1
    assert page.locator('.toolbar,#analysis').count()==0
    m=json.loads((ROOT/family['manifest']).read_text());active=next(e for e in m['entries'] if e['id']==m['default'])
    if active['kind']=='html':
     frame=page.frame_locator('#detail-frame');frame.locator('body').wait_for(timeout=30000)
     # Lazy videos inside a closed details panel have no source until opened.
     if frame.locator('video:visible').count():
      page.wait_for_function("document.querySelector('#detail-frame').contentDocument.querySelector('video')?.readyState>=2",timeout=45000)
      frame.locator('video').first.evaluate('(v)=>{v.muted=true;v.play().catch(()=>{})}')
      frame.locator('video').first.evaluate('(v)=>v.readyState')
      page.wait_for_function("document.querySelector('#detail-frame').contentDocument.querySelector('video')?.currentTime>.1",timeout=20000);row['video_playback']=True
    else:page.wait_for_function("document.querySelector('#report').textContent.length>100 && !document.querySelector('#report').textContent.includes('불러오는 중')");row['markdown']=True
    if family['family']=='deterministic':
     frame=page.frame_locator('#detail-frame');frame.locator('#branches .case-card').first.wait_for(timeout=60000)
     for n in [300,600,900]:
      select_view(page,'pilot/'+str(n)+'/viewer.html');page.wait_for_function('(n)=>document.querySelector("#detail-frame").contentWindow.ROLLOUT_VIEWER?.state.grouped.length===n',arg=n,timeout=60000)
     row['cohort_switch_300_600_900']=True
     # Full token distributions load from the same canonical900 store.
     url=page.url.split('#')[0]+'#preset=all&sample=activitynet-timelens_01723&candidate=3&panel=distributions'
     page.goto(url);page.wait_for_function("document.querySelector('#detail-frame').contentWindow.ROLLOUT_DISTRIBUTIONS?.state.trace",timeout=60000);row['token_distribution']=True
    # A Markdown report is an in-family selection, not a new catalog item.
    md=next((e for e in m['entries'] if e['kind']=='md' and e['id'].endswith('REPORT.md')),None)
    if md:
     page.goto('http://127.0.0.1:8780/'+family['href']+'?view='+__import__('urllib.parse',fromlist=['quote']).quote(md['id'],safe=''))
     page.wait_for_selector('#family-app[data-ready="true"]');page.wait_for_function("document.querySelector('#report').textContent.length>80 && !document.querySelector('#report').textContent.includes('불러오는 중')");assert page.locator('#report h1').count()==1
     row['report_selection']=True
    page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    row['passed']=not errors and not failed
   except Exception as e:row.update(passed=False,error=str(e))
   row.update(errors=errors,failed_requests=failed);rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True);page.close()
  # Independent scope selection is retained in each comparison pane.
  home.locator('article:visible input[name="compare"]').nth(0).check();home.locator('article:visible input[name="compare"]').nth(1).check()
  with home.context.expect_page() as opened:home.locator('#open-comparison').click()
  comparison=opened.value;comparison.wait_for_load_state()
  for side in ['left','right']:
   outer=comparison.frame_locator('#'+side+'-frame');outer.locator('#family-app[data-ready="true"]').wait_for(timeout=45000)
  comparison.reload();comparison.frame_locator('#left-frame').locator('#family-app[data-ready="true"]').wait_for(timeout=45000)
  summary['comparison_and_url_restore']=True
  comparison.close();home.set_viewport_size({'width':390,'height':844});assert home.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  browser.close()
 result=dict(passed=all(r['passed'] for r in rows),summary=summary,families=rows)
 (S/'family_browser.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');assert result['passed']
if __name__=='__main__':
 if '--browser' in sys.argv:browser_checks()
 else:preservation()
