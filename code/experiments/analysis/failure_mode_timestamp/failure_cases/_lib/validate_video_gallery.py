"""End-to-end checks using the installed local Playwright browser bundle."""
from result_layout import relocate_path, write_result_text
import json,os,sys,time
from pathlib import Path
support=relocate_path(Path('/home/irteam/experiments/mtla_attention'))
sys.path.insert(0,str(relocate_path(support/'browser_deps')))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(relocate_path(support/'browser_cache'))
os.environ['LD_LIBRARY_PATH']=str(relocate_path(support/'browser_libs/usr/lib/x86_64-linux-gnu'))+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(relocate_path(support/'browser_libs/fonts.conf'))
from playwright.sync_api import sync_playwright
root=relocate_path(Path(__file__)).resolve().parents[1];out=relocate_path(root/'reports/pilot_100/cases')
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1200});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((relocate_path(out/'report.html')).as_uri());page.wait_for_function('document.getElementById("video").readyState >= 2',timeout=20000)
 assert page.locator('.case').count()==14
 assert page.locator('.track').count()==4
 page.locator('[data-track="gt"] .segment').click();page.wait_for_timeout(550)
 assert page.evaluate('video.currentTime')>19
 assert not page.evaluate('video.paused')
 page.locator('#play').click()
 page.evaluate('document.fonts.ready');page.screenshot(path=str(relocate_path(out/'video_viewer_preview.png')),full_page=True)
 # Seek on axis; active labels follow current timestamp.
 page.locator('#ruler').click(position={'x':10,'y':8});page.wait_for_timeout(200)
 assert page.evaluate('video.currentTime')<5
 page.locator('#ruler').focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(200)
 # Segment endpoint pause and replay.
 page.locator('[data-track="gt"] .segment').click()
 page.evaluate('video.currentTime=23.85');page.wait_for_function('!video.seeking && video.readyState>=2');page.wait_for_timeout(700)
 assert page.evaluate('video.paused')
 assert abs(page.evaluate('video.currentTime')-24)<.08
 page.locator('#loop').check();page.locator('[data-track="gt"] .segment').click();page.evaluate('video.currentTime=23.9');page.wait_for_function('!video.seeking && video.readyState>=2');page.wait_for_timeout(600)
 assert not page.evaluate('video.paused');assert page.evaluate('video.currentTime')<20
 page.locator('#play').click();page.locator('#loop').uncheck()
 # Zero-length rounded interval vs valid raw prediction on same case.
 page.locator('#search').fill('71Fa81Wr');page.wait_for_function('video.readyState >= 2')
 zero=page.locator('[data-track="qwen25-timestamp"] .segment');assert zero.is_disabled()
 assert '146–146s' in page.locator('#tracks').inner_text()
 page.locator('#timeMode').select_option('raw');assert not zero.is_disabled()
 assert '145.5–146.5s' in page.locator('#tracks').inner_text()
 zero.click();page.wait_for_timeout(200);assert 145.5<=page.evaluate('video.currentTime')<146.6
 page.locator('#play').click();page.locator('#timeMode').select_option('rounded')
 # Wholly out-of-bounds prediction disabled, original interval still shown.
 page.locator('#search').fill('trcIoufe');page.wait_for_function('video.readyState >= 2')
 assert page.locator('[data-track="qwen25-timestamp"] .segment').is_disabled()
 assert '150–152s' in page.locator('#tracks').inner_text()
 # Partly out-of-bounds prediction playable only up to actual end.
 page.locator('#search').fill('3YQE2');page.wait_for_function('video.readyState >= 2')
 page.locator('[data-track="qwen25-timestamp"] .segment').click();assert '영상 내 부분만' in page.locator('#playState').inner_text()
 page.locator('#play').click()
 # Load/decode and seek every media file, no server needed.
 media_checks=[]
 for c in json.loads((relocate_path(out/'selection.json')).read_text())['cases']:
  page.locator('#search').fill(c['sample_id']);page.wait_for_function('video.readyState >= 2',timeout=15000)
  page.evaluate('video.currentTime=video.duration*.65');page.wait_for_function('!video.seeking && video.readyState>=2',timeout=15000)
  dimensions=page.evaluate('({width:video.videoWidth,height:video.videoHeight,duration:video.duration})')
  assert dimensions['width']>0 and dimensions['height']>0
  assert abs(dimensions['duration']-c['decoded_duration'])<.15,(c['sample_id'],dimensions)
  media_checks.append({'sample_id':c['sample_id'],**dimensions})
 # Narrow screen: no page overflow; controls and bars remain reachable.
 page.set_viewport_size({'width':390,'height':844});page.locator('#search').fill('81VSN');page.wait_for_function('video.readyState >= 2')
 assert page.evaluate('document.documentElement.scrollWidth')<=392
 page.screenshot(path=str(relocate_path(out/'video_viewer_mobile.png')),full_page=True)
 assert not errors,errors
 write_result_text((relocate_path(out/'browser_validation.json')), json.dumps({'passed':True,'engine':'Chromium','local_file_mode':True,'all_media_decode_and_seek':media_checks,'checks':['segment playback','endpoint pause','loop','timeline pointer/keyboard seek','raw vs rounded intervals','zero-length disabled','out-of-bounds visible and not playable','partial out-of-bounds playback clipped only','mobile no overflow'],'page_errors':errors},indent=2)+'\n')
 browser.close()
print('Video gallery browser validation passed')
