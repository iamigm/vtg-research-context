"""Browser checks used as the first300 gate."""
import argparse,sys
from pathlib import Path
from interval_distribution_core import *
def run(count):
 sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
 from temporal_mode_view_check import sync_playwright
 errors=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);page=browser.new_page(viewport=dict(width=1600,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
  key=read(COHORT)[0]['sample_key'];page.goto(f'http://127.0.0.1:8780/zero-shot/results/rollout/interval_distribution/pilot/{count}/detail.html?sample={key}',wait_until='networkidle');page.wait_for_selector('#query-detail svg')
  assert page.locator('#samples button').count()==count;assert 'GT' in page.locator('#content').inner_text();assert '독립 모델이 다시 본 최종 답' in page.locator('#content').inner_text()
  response=page.request.get(page.locator('video').get_attribute('src'),headers={'Range':'bytes=0-127'});assert response.status==206;assert len(response.body())==128
  result=ROOT/f'results/rollout/interval_distribution/pilot/{count}';page.screenshot(path=str(result/'preview_desktop.png'),full_page=True);page.set_viewport_size(dict(width=390,height=844));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path=str(result/'preview_mobile.png'),full_page=True)
  assert not errors,errors;browser.close()
 write(OUT/'validation'/f'viewer_pilot{count}.json',dict(passed=True,queries=count,page_errors=errors,timestamp=stamp()));print('VIEWER_VALIDATED',count)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',type=int,default=300);run(p.parse_args().scope)
