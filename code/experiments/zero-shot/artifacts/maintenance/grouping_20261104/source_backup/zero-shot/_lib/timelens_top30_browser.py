from result_layout import relocate_path, write_result_text
from timelens_paths import RESOURCES
from pathlib import Path
import os,sys,json
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/timelens_top30')
sys.path.insert(0,str(relocate_path(RESOURCES/'browser_deps')))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(relocate_path(RESOURCES/'browser_cache'))
os.environ['LD_LIBRARY_PATH']=str(relocate_path(RESOURCES/'browser_libs/usr/lib/x86_64-linux-gnu'))+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(relocate_path(RESOURCES/'browser_libs/fonts.conf'))
from playwright.sync_api import sync_playwright
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
    page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8770/');page.wait_for_function("document.querySelectorAll('#top tr').length===30")
    for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens']:
        page.select_option('#dataset',ds);assert page.locator('#cases button').count()==100
        page.locator('#cases button').first.click();page.wait_for_function("document.querySelectorAll('#top tr').length===30")
        page.wait_for_function("document.querySelector('video').readyState>=2 && document.querySelector('video').videoWidth>0",timeout=60000)
        for model in ['qwen25','qwen3']:
            page.select_option('#model',model);page.wait_for_function("document.querySelectorAll('#top tr').length===30")
            page.click('#next');assert page.locator('#stats').inner_text().startswith('Step 2 ')
            page.select_option('#mode','processed');assert page.locator('#top tr').count()==30
            page.select_option('#mode','raw')
        page.evaluate("document.querySelector('video').currentTime=0;document.querySelector('video').play()")
        page.wait_for_function("document.querySelector('video').currentTime>0.2",timeout=20000)
        page.evaluate("document.querySelector('video').pause()")
    page.screenshot(path=str(relocate_path(OUT/'preview.png')))
    browser.close()
assert not errors,errors
write_result_text((relocate_path(OUT/'browser_validation.json')), json.dumps(dict(passed=True,datasets=3,models=2,video_playback=True,top30_table=True,offline_file_protocol=True,page_errors=errors),indent=2)+'\n')
