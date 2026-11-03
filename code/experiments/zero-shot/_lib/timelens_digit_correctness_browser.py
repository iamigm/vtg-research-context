from result_layout import relocate_path, write_result_text
from pathlib import Path
import sys,os,json
from timelens_paths import RESOURCES
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300/digit_correctness')
sys.path.insert(0,str(relocate_path(RESOURCES/'browser_deps')));os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(relocate_path(RESOURCES/'browser_cache'));os.environ['LD_LIBRARY_PATH']=str(relocate_path(RESOURCES/'browser_libs/usr/lib/x86_64-linux-gnu'))+':'+os.environ.get('LD_LIBRARY_PATH','');os.environ['FONTCONFIG_FILE']=str(relocate_path(RESOURCES/'browser_libs/fonts.conf'))
from playwright.sync_api import sync_playwright
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required']);page=browser.new_page(viewport={'width':1500,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)));page.goto('http://127.0.0.1:8773/')
    page.wait_for_function("current && document.querySelectorAll('#tokens tr').length>0")
    for model in ['qwen25','qwen3']:
        page.select_option('#model',model)
        for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens']:
            page.select_option('#dataset',ds);assert page.locator('#samples option').count()==100
            page.wait_for_function("([m,d])=>current&&current.model===m&&current.sample.dataset===d&&document.querySelectorAll('#tokens tr').length>0",arg=[model,ds])
            for ep in ['start','end']:
                page.click('#'+ep);assert page.locator('#tokens tr').count()>0;page.locator('#tokens tr').first.click();assert 'H=' in page.locator('#selection').inner_text();assert page.locator('#conditions tr').count()>0
            page.wait_for_function("document.querySelector('video').readyState>=2 && document.querySelector('video').videoWidth>0",timeout=60000)
            page.click('#playGT');page.wait_for_function("!document.querySelector('video').paused",timeout=10000);page.evaluate("document.querySelector('video').pause()")
    page.select_option('#dataset','activitynet-timelens');page.select_option('#samples','activitynet-timelens_00646');page.wait_for_function("current && current.sample.sample_key==='activitynet-timelens_00646'");assert page.locator('#tokens tr').count()==0;assert '범위 밖' in page.locator('#endpointInfo').inner_text()
    page.fill('#search','NO_MATCH_993811');assert page.locator('#samples option').count()==0;page.fill('#search','');page.select_option('#dataset','charades-timelens');page.wait_for_function("current && current.sample.dataset==='charades-timelens' && document.querySelectorAll('#tokens tr').length>0")
    page.select_option('#outcome','fail');assert page.locator('#samples option').count()>0
    page.evaluate('document.fonts.ready');page.screenshot(path=str(relocate_path(OUT/'preview.png')),full_page=True);browser.close()
assert not errors,errors
write_result_text((relocate_path(OUT/'browser_validation.json')), json.dumps(dict(passed=True,models=2,datasets=3,endpoint_toggle=True,role_selection=True,video_playback=True,excluded_format=True,search_empty=True,page_errors=errors),indent=2)+'\n')
write_result_text((relocate_path(OUT/'status.json')), json.dumps(dict(state='complete',completed=600,total=600,percent=100,browser_validation=True),indent=2)+'\n')
