"""Browser, video, filter, deep-link and source-metric checks for the rollout UI."""
import os,sys,json,csv
from pathlib import Path
from urllib.parse import urljoin
from deterministic_rollout_viewer import OUT,REPORT,WEB,ROOT
from deterministic_rollout300 import read,write

SHARED=ROOT.parent/'_shared/timelens'
sys.path.insert(0,str(SHARED/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(SHARED/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(SHARED/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(SHARED/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright

BASE='http://127.0.0.1:8780/zero-shot/results/deterministic_rollout_pilot900_v2/viewer.html'

def run():
 manifest=read(WEB/'manifest.json');assert len(manifest['samples'])==900
 assert len(manifest['branches'])==26560
 expected={x['role']:x for x in csv.DictReader((OUT/'metrics/branch_role_summary.csv').open())}
 for role,summary in expected.items():
  bs=[b for b in manifest['branches'] if role in b['roles']]
  assert len(bs)==int(summary['branches'])
  assert abs(sum(b['parent_delta'] for b in bs)/len(bs)-float(summary['mean_delta_iou']))<1e-12
  assert abs(sum(b['greedy_delta'] for b in bs)/len(bs)-float(summary['mean_delta_vs_greedy']))<1e-12
 errors=[];checks=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
  page=browser.new_page(viewport={'width':1500,'height':1100})
  page.on('pageerror',lambda error:errors.append(str(error)))
  page.goto(BASE)
  page.wait_for_function('window.ROLLOUT_VIEWER?.state.current && !document.querySelector("#loadStatus").textContent',timeout=60000)
  assert page.locator('#branches tr').count()<=40
  assert page.evaluate('ROLLOUT_VIEWER.state.filtered.every(b=>b.greedy_delta>=.01)')
  checks.append('default_filter_is_improvement_over_initial_greedy')
  page.click('[data-preset="parent_improved"]')
  assert page.evaluate('ROLLOUT_VIEWER.state.filtered.every(b=>b.parent_delta>=.01)')
  page.click('[data-preset="dot"]')
  page.wait_for_function('ROLLOUT_VIEWER.state.filtered.length===243')
  assert page.locator('#sampleSelect option').count()==84
  page.click('#nextPage');assert page.locator('#pageInfo').inner_text().startswith('2 /')
  checks.append('all_243_dot_branches_in_84_samples_and_pagination')
  page.click('[data-preset="parent_short"]')
  assert page.locator('#sampleSelect option').count()==4
  assert page.evaluate('ROLLOUT_VIEWER.state.filtered.every(b=>b.short_parent)')
  page.click('[data-preset="greedy_short"]')
  assert page.locator('#sampleSelect option').count()==2
  page.select_option('#sampleSelect','activitynet-timelens_01723')
  page.wait_for_function('ROLLOUT_VIEWER.state.current?.sample.key==="activitynet-timelens_01723"')
  page.locator('#candidates tr[data-candidate="3"]').click()
  assert page.evaluate('ROLLOUT_VIEWER.state.candidate')==3
  assert '107.3' in page.locator('#answerComparison').inner_text()
  assert page.locator('#answerComparison mark').count()==2
  assert '. → 0' in page.locator('#branchDetail').inner_text()
  checks.append('actual_1_point_3_to_107_point_3_token_highlight')
  deep_link=page.url;page.reload()
  page.wait_for_function('window.ROLLOUT_VIEWER?.state.current?.sample.key==="activitynet-timelens_01723" && ROLLOUT_VIEWER.state.candidate===3')
  page.select_option('#budget','4')
  assert page.locator('#candidates tr').count()==4
  page.select_option('#method','oracle')
  assert page.evaluate('ROLLOUT_VIEWER.state.candidate===ROLLOUT_VIEWER.state.current.choices["4"].oracle')
  page.select_option('#budget','32')
  checks.append('deep_link_and_budget_and_selector')
  page.click('[data-preset="all"]')
  assert page.locator('#sampleSelect option').count()==900
  for ds in ['charades-timelens','activitynet-timelens','qvhighlights-timelens']:
   page.select_option('#dataset',ds);assert page.locator('#sampleSelect option').count()==300
   key=page.locator('#sampleSelect option').first.get_attribute('value')
   page.select_option('#sampleSelect',key)
   page.wait_for_function('(key)=>ROLLOUT_VIEWER.state.current?.sample.key===key',arg=key)
   page.wait_for_function('document.querySelector("video").readyState>=2 && document.querySelector("video").videoWidth>0',timeout=60000)
   page.click('#timeline [data-play="0"]')
   page.wait_for_function('!document.querySelector("video").paused',timeout=15000)
   page.wait_for_function('document.querySelector("video").currentTime>ROLLOUT_VIEWER.state.current.sample.gt[0]+.1',timeout=15000)
   page.evaluate('document.querySelector("video").pause()')
   response=page.request.get(page.locator('#video').get_attribute('src') if page.locator('#video').get_attribute('src').startswith('http') else urljoin(BASE,page.locator('#video').get_attribute('src')),headers={'Range':'bytes=0-1023'})
   assert response.status==206 and len(response.body())==1024
  checks.append('three_dataset_filters_video_playback_and_byte_ranges')
  page.fill('#search','NO_MATCH_999999')
  assert page.locator('#sampleSelect option').count()==0
  assert '조건에 맞는 분기가 없습니다' in page.locator('#branches').inner_text()
  page.click('[data-preset="greedy_short"]');page.select_option('#sampleSelect','activitynet-timelens_01723')
  page.wait_for_function('ROLLOUT_VIEWER.state.current?.sample.key==="activitynet-timelens_01723"')
  page.select_option('#method','probability_weighted_medoid')
  page.locator('#candidates tr[data-candidate="3"]').click()
  page.wait_for_function('document.querySelector("video").readyState>=2 && document.querySelector("video").videoWidth>0',timeout=60000)
  page.click('#timeline [data-play="0"]')
  page.wait_for_function('document.querySelector("video").currentTime>.2',timeout=15000)
  page.evaluate('document.querySelector("video").pause()')
  page.evaluate('window.scrollTo(0,0)');page.evaluate('document.fonts.ready')
  assert page.evaluate('Array.from(document.fonts).some(f=>f.family==="RolloutKorean" && f.status==="loaded")')
  page.screenshot(path=str(REPORT/'preview_desktop.png'),full_page=True)
  page.set_viewport_size({'width':390,'height':844});page.evaluate('window.scrollTo(0,0)')
  assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
  page.screenshot(path=str(REPORT/'preview_mobile.png'),full_page=True)
  checks.append('empty_search_and_mobile_layout')
  browser.close()
 assert not errors,errors
 write(OUT/'validation/viewer_browser.json',dict(passed=True,checks=checks,page_errors=errors,deep_link=deep_link,
     all_branch_metrics_match_source=True,reused_existing_video_cache=True))
 print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False))

if __name__=='__main__':run()
