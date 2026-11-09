"""Postprocessing viewer checks using actual completed production queries."""
import argparse,sys
from probability_completion_core import *
def check(count):
 sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
 from temporal_mode_view_check import sync_playwright
 result=ROOT/f'results/rollout/probability_budget/pilot/{count}/end_to_end';url=f'http://127.0.0.1:8780/zero-shot/results/rollout/probability_budget/pilot/{count}/end_to_end/detail.html';errors=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);page=browser.new_page(viewport=dict(width=1600,height=1100));page.on('pageerror',lambda e:errors.append(str(e)));page.goto(url,wait_until='networkidle');page.wait_for_selector('#answers .answer');assert page.locator('#answers .answer').count()==4;assert page.locator('#answers .answer small').first.inner_text()=='greedy';assert page.locator('#reviews tr').count()>0;assert 'Cosine' in page.locator('body').inner_text();response=page.request.get(page.locator('video').get_attribute('src'),headers={'Range':'bytes=0-127'});assert response.status==206
  for method in METHODS:
   page.select_option('#method',method)
   for budget in BUDGETS:
    page.select_option('#budget',str(budget));assert page.locator('#answers .answer').count()==4
  for pool in ('instruct_visual_mean','instruct_last','embedding_last'):page.select_option('#pool',pool);assert 'cosine' in page.locator('#semantic-summary').inner_text()
  page.select_option('#method','native_mass');page.select_option('#budget','1024');page.screenshot(path=str(result/'preview_desktop.png'),full_page=True);page.set_viewport_size(dict(width=390,height=844));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path=str(result/'preview_mobile.png'),full_page=True);assert not errors,errors;browser.close()
 write(OUT/'validation'/f'viewer_pilot{count}.json',dict(passed=True,scope_queries=count,actual_completed_query=True,four_answers=True,all_conditions=True,both_semantic_models=True,video_range=True,mobile_no_overflow=True,page_errors=errors,timestamp=stamp()));print('COMPLETION_VIEWER_VALIDATED',count)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',type=int,default=300);a=p.parse_args();check(a.scope)
