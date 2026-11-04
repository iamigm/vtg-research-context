"""Check independent family scopes and nested video playback in the comparison view."""
import json,os,sys
from pathlib import Path
ROOT=Path('/home/irteam/experiments');shared=ROOT/'_shared/timelens'
sys.path.insert(0,str(shared/'browser_deps'));os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache');os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','');os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required']);p=b.new_page(viewport={'width':1600,'height':1000})
 p.goto('http://127.0.0.1:8780/');p.select_option('#project','similarity')
 for i in [0,1]:p.locator('article:visible input[name="compare"]').nth(i).check()
 with p.context.expect_page() as opened:p.click('#open-comparison')
 catalog=p;p=opened.value;p.wait_for_load_state()
 assert catalog.locator('#catalog').is_visible() and not catalog.url.endswith('?left=')
 for side in ['left','right']:
  outer=p.frame_locator('#'+side+'-frame');outer.locator('#family-app[data-ready="true"]').wait_for(timeout=60000)
  p.wait_for_function('(side)=>document.querySelector("#"+side+"-frame").contentDocument.querySelector("#detail-frame").contentDocument?.querySelector("video")?.readyState>=2',arg=side,timeout=60000)
  video=outer.frame_locator('#detail-frame').locator('video').first;video.evaluate('(v)=>{v.muted=true;v.play()}')
  p.wait_for_function('(side)=>document.querySelector("#"+side+"-frame").contentDocument.querySelector("#detail-frame").contentDocument.querySelector("video").currentTime>.1',arg=side,timeout=20000)
 assert p.frame_locator('#left-frame').locator('#scope').input_value()=='pilot/300'
 assert p.frame_locator('#right-frame').locator('#scope').input_value()=='full/9404'
 p.frame_locator('#right-frame').locator('#scope').select_option('pilot/300')
 assert p.frame_locator('#left-frame').locator('#scope').input_value()=='pilot/300'
 p.frame_locator('#right-frame').locator('#scope').select_option('full/9404')
 assert p.frame_locator('#left-frame').locator('#scope').input_value()=='pilot/300'
 url=p.url;p.reload();p.frame_locator('#left-frame').locator('#family-app[data-ready="true"]').wait_for(timeout=45000);assert p.url==url
 p.click('#close-comparison');p.set_viewport_size({'width':390,'height':844});assert p.evaluate('document.documentElement.scrollWidth<=innerWidth')
 b.close()
result=dict(passed=True,two_family_videos_play=True,scope_selection_independent=True,url_restore=True,mobile=True)
(ROOT/'artifacts/maintenance/rollout_views_20261104/comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
