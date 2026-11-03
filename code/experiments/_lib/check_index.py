"""Validate the shared entry point, search, navigation and video playback."""
import json,os,sys
from pathlib import Path
ROOT=Path('/home/irteam/experiments'); R=ROOT/'_shared/timelens'
sys.path.insert(0,str(R/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(R/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(R/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(R/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
rows=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
 page=browser.new_page(viewport={'width':1440,'height':1000})
 page.goto('http://127.0.0.1:8780/',wait_until='networkidle')
 links=page.locator('article h3 a').evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("href")).filter(href=>href.endsWith(".html"))')
 assert len(links)==sum(len(list((ROOT/project/'results').rglob('*.html'))) for project in __import__('build_index').PROJECTS)
 page.locator('#search').fill('no-such-experiment-xyz');assert page.locator('#empty').is_visible()
 page.locator('#search').fill('');page.locator('#project').select_option('similarity')
 assert page.locator('article:visible').count()==3
 page.locator('#project').select_option('')
 for link in links:
  errors=[];failed=[]
  view=browser.new_page()
  view.on('pageerror',lambda e:errors.append(str(e)))
  view.on('response',lambda r:failed.append([r.status,r.url]) if r.status>=400 else None)
  response=view.goto('http://127.0.0.1:8780/'+link,wait_until='networkidle',timeout=30000)
  assert response.status==200
  home=view.locator('[data-experiment-home] a');assert home.count()==1
  assert home.evaluate('(a)=>a.href')=='http://127.0.0.1:8780/index.html'
  if '/pilot/uncertainty/token_analysis/300/report.html' in link:
   view.wait_for_function("[...document.querySelectorAll('video')].some(v=>v.readyState>=2&&v.videoWidth>0)",timeout=30000)
   video=view.locator('video').first;video.evaluate('(v)=>{v.muted=true;v.play()}')
   view.wait_for_function("document.querySelector('video').currentTime>.15")
   rows.append({'video_playback':True})
  assert not errors,(link,errors)
  assert not failed,(link,failed)
  rows.append({'page':link,'passed':True})
  view.close()
 page.set_viewport_size({'width':390,'height':844})
 assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 browser.close()
result={'pages':len(links),'search':True,'project_filter':True,'mobile_layout':True,'results':rows}
(ROOT/'artifacts/maintenance/unified_index_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
