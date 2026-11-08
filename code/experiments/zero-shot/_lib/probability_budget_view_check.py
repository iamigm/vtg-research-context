"""Common navigation, actual media and branch/prefix interaction checks."""
import argparse,json,sys
from probability_budget_core import *
def check(count=300,initial=False):
 sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
 from temporal_mode_view_check import sync_playwright
 errors=[];cohort=read(COHORT)[:count];key=cohort[0]['sample_key'];result=ROOT/f'results/rollout/probability_budget/pilot/{count}';base=f'http://127.0.0.1:8780/zero-shot/results/rollout/probability_budget/pilot/{count}/detail.html'
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);page=browser.new_page(viewport=dict(width=1600,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
  if initial:
   # Intercept real GPU smoke data in memory; never publish it as pilot output.
   m=read(OUT/'validation/smoke_tree.json');s=m['snapshots']['64'];m=dict(m,snapshots={str(b):dict(s,budget=b) for b in BUDGETS});old=read(SOURCE/'rollouts'/f'{key}.json')['result']['candidates'][0];q=dict(greedy=dict(span=[old['parsed']['start'],old['parsed']['end']]))
   page.route('**/queries/'+key+'.json',lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(q)))
   page.route('**/search/'+key+'/*.json',lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(m)))
  page.goto(base+'?sample='+key+'&method=native_mass',wait_until='networkidle');page.wait_for_selector('#query-detail svg');assert page.locator('#samples button').count()==count;assert 'GT' in page.locator('#content').inner_text();src=page.locator('video').get_attribute('src');response=page.request.get(src,headers={'Range':'bytes=0-127'});assert response.status==206 and len(response.body())==128
  page.wait_for_selector('[data-child]');page.locator('[data-child]').first.click();assert page.locator('#prefix-window').inner_text().startswith('The event happens in');assert all('The' not in text for text in page.locator('#prefix-window .branch-token').all_text_contents());assert page.locator('#token-distribution').inner_text().find('token distribution')>=0
  page.select_option('#budget','512');page.wait_for_selector('#query-detail svg');page.select_option('#method','levin');page.wait_for_selector('#query-detail svg');page.screenshot(path=str(result/'preview_desktop.png'),full_page=True);page.set_viewport_size(dict(width=390,height=844));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path=str(result/'preview_mobile.png'),full_page=True)
  if initial:
   page.unroute_all();page.goto(base,wait_until='networkidle');assert page.locator('#samples button').count()==count;assert 'GT' in page.locator('#content').inner_text();page.set_viewport_size(dict(width=1600,height=1100));page.screenshot(path=str(result/'preview_desktop.png'),full_page=True);page.set_viewport_size(dict(width=390,height=844));page.screenshot(path=str(result/'preview_mobile.png'),full_page=True)
  page.goto('http://127.0.0.1:8780/zero-shot/results/rollout/probability_budget/viewer.html',wait_until='networkidle');page.wait_for_selector('#analysis-tabs button');assert '결과' in page.locator('#analysis-tabs').inner_text() and '보고서' in page.locator('#analysis-tabs').inner_text();assert not errors,errors;browser.close()
 write(OUT/'validation'/('viewer_initial.json' if initial else f'viewer_pilot{count}.json'),dict(passed=True,queries=count,initial_gpu_fixture_only=initial,prefix_interaction=True,no_word_highlighting=True,token_distribution=True,video_range=True,mobile_no_overflow=True,page_errors=errors,timestamp=stamp()));print('VIEWER_VALIDATED',count,initial)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',type=int,default=300);p.add_argument('--initial',action='store_true');a=p.parse_args();check(a.scope,a.initial)
