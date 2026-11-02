from result_layout import relocate_path, write_result_text
from timelens_paths import RESOURCES
from pathlib import Path
import os,sys,json
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/timelens_rollout')
sys.path.insert(0,str(relocate_path(RESOURCES/'browser_deps')));os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(relocate_path(RESOURCES/'browser_cache'));os.environ['LD_LIBRARY_PATH']=str(relocate_path(RESOURCES/'browser_libs/usr/lib/x86_64-linux-gnu'))+':'+os.environ.get('LD_LIBRARY_PATH','');os.environ['FONTCONFIG_FILE']=str(relocate_path(RESOURCES/'browser_libs/fonts.conf'))
from playwright.sync_api import sync_playwright
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8771/')
    for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens']:
        page.select_option('#dataset',ds);assert page.locator('#cases button').count()==100;page.locator('#cases button').first.click()
        page.wait_for_function("document.querySelector('video').readyState>=2&&document.querySelector('video').videoWidth>0",timeout=60000)
        for model in ['qwen25','qwen3']:
            page.select_option('#model',model)
            for var in ['top_k','top_p']:
                page.select_option('#variant',var)
                for roll in ['0','4']:
                    page.select_option('#rollout',roll)
                    page.wait_for_function("([m,v,r])=>trace&&trace.model_key===m&&trace.variant===v&&trace.rollout===Number(r)&&document.querySelectorAll('#top tr').length>0",arg=[model,var,roll],timeout=60000)
                    count=page.evaluate('trace.steps[step].candidate_count');assert page.locator('#top tr').count()==min(100,count)
                    if count>100:
                        page.click('#candidateNext');assert page.locator('#candidatePage').inner_text().startswith('2 /')
                    page.click('#next');page.wait_for_function("document.querySelector('#stats').textContent.startsWith('Step 2 ')")
        page.evaluate("document.querySelector('video').currentTime=0;document.querySelector('video').play()")
        page.wait_for_function("document.querySelector('video').currentTime>0.2",timeout=20000);page.evaluate("document.querySelector('video').pause()")
    page.screenshot(path=str(relocate_path(OUT/'preview.png')));browser.close()
assert not errors,errors
write_result_text((relocate_path(OUT/'browser_validation.json')), json.dumps(dict(passed=True,datasets=3,models=2,variants=2,rollouts_checked=[0,4],full_candidate_pagination=True,video_playback=True,page_errors=errors),indent=2)+'\n')
