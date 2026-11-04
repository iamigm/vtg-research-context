"""Check case grouping, meaningful before/after navigation and unchanged source metrics."""
import os,sys,json,csv
from urllib.parse import urljoin
from deterministic_rollout_viewer import OUT,REPORT,WEB,ROOT
from deterministic_rollout300 import read,write
SHARED=ROOT.parent/'_shared/timelens'
sys.path.insert(0,str(SHARED/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(SHARED/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(SHARED/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(SHARED/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:8780/zero-shot/results/rollout/deterministic/pilot/900/viewer.html'
KEY='activitynet-timelens_01723'
def ready(page):page.wait_for_function('window.ROLLOUT_VIEWER?.state.current && !document.querySelector("#loadStatus").textContent',timeout=60000)
def run():
 manifest=read(WEB/'manifest.json');assert len(manifest['samples'])==900 and len(manifest['branches'])==26560
 expected={x['role']:x for x in csv.DictReader((OUT/'metrics/branch_role_summary.csv').open())}
 for role,summary in expected.items():
  bs=[b for b in manifest['branches'] if role in b['roles']]
  assert len(bs)==int(summary['branches'])
  assert abs(sum(b['parent_delta'] for b in bs)/len(bs)-float(summary['mean_delta_iou']))<1e-12
  assert abs(sum(b['greedy_delta'] for b in bs)/len(bs)-float(summary['mean_delta_vs_greedy']))<1e-12
 errors=[];checks=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
  page=browser.new_page(viewport={'width':1600,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(BASE);ready(page)
  assert page.evaluate('ROLLOUT_VIEWER.state.preset==="all" && ROLLOUT_VIEWER.state.grouped.length===900')
  assert page.locator('.presets button').first.get_attribute('data-preset')=='all'
  assert page.locator('[data-preset="all"]').get_attribute('aria-pressed')=='true'
  assert page.locator('#sort').input_value()=='sample'
  assert page.locator('#sort').is_visible()
  assert page.locator('#search').bounding_box()['y']==page.locator('#dataset').bounding_box()['y']
  assert page.locator('#search').bounding_box()['height']==page.locator('#dataset').bounding_box()['height']
  for order in ['greedy_low','greedy_high']:
   page.select_option('#sort',order)
   assert page.evaluate('(order)=>{const s=ROLLOUT_VIEWER.state;return s.grouped.every((b,i,a)=>!i||(order==="greedy_low"?s.map.get(a[i-1].sample).greedy_iou<=s.map.get(b.sample).greedy_iou:s.map.get(a[i-1].sample).greedy_iou>=s.map.get(b.sample).greedy_iou));}',order)
  page.select_option('#dataset','activitynet-timelens');page.fill('#search','man');page.select_option('#sort','greedy_low');page.click('[data-preset="improved"]');ready(page)
  assert page.locator('#dataset').input_value()=='activitynet-timelens' and page.locator('#search').input_value()=='man' and page.locator('#sort').input_value()=='greedy_low'
  page.select_option('#dataset','');page.fill('#search','');page.select_option('#sort','sample')
  checks.append('all_samples_default_aligned_search_dataset_and_independent_basic_sorting')
  assert page.locator('#branches .case-card').count()<=10
  first=page.locator('#branches .case-card').first
  assert first.locator(':scope>span').first.get_attribute('class')=='case-header'
  assert first.locator('.case-id').inner_text()==first.get_attribute('data-sample')
  assert first.locator('.case-dataset').count()==0 and '개 변경 중' not in first.inner_text()
  for scenario in ['recovered','improved','worsened','dot','all']:
   page.click(f'[data-preset="{scenario}"]');ready(page)
   assert page.evaluate('new Set(ROLLOUT_VIEWER.state.grouped.map(b=>b.sample)).size===ROLLOUT_VIEWER.state.grouped.length')
   if scenario=='recovered':assert page.evaluate('ROLLOUT_VIEWER.state.filtered.every(b=>b.iou>=.5 && ROLLOUT_VIEWER.state.map.get(b.sample).greedy_iou<.5)')
   if scenario=='improved':assert page.evaluate('ROLLOUT_VIEWER.state.filtered.every(b=>b.greedy_delta>=.01)')
   if scenario=='worsened':assert page.evaluate('ROLLOUT_VIEWER.state.filtered.every(b=>b.greedy_delta<=-.01)')
   if scenario=='dot':assert page.evaluate('ROLLOUT_VIEWER.state.filtered.length===243 && ROLLOUT_VIEWER.state.grouped.length===84')
   if scenario=='all':assert page.evaluate('ROLLOUT_VIEWER.state.grouped.length===900')
   card=page.locator('.case-card').first;key=card.get_attribute('data-sample');candidate=int(card.get_attribute('data-candidate'));card.click()
   page.wait_for_function('([key,c])=>ROLLOUT_VIEWER.state.current.sample.key===key&&ROLLOUT_VIEWER.state.candidate===c',arg=[key,candidate])
   assert page.locator('#resultOverview .result-card').count()==3
   assert page.locator('#branchSelect').input_value()==str(candidate)
  checks.append('scenario_semantics_unique_sample_cards_and_card_matches_inspected_branch')
  assert page.locator('#sampleSelect,.secondary-presets,.filter-details [data-preset],.score-explain').count()==0
  assert page.locator('.filter-details select').count()==0
  assert page.locator('.filter-details input[type="number"]').count()==14
  assert page.locator('[data-preset="parent_improved"]').is_visible()
  page.locator('.filter-details>summary').click()
  page.fill('#initialMax','0.5');page.fill('#candidateMin','0.5');page.fill('#greedyDeltaMin','0.1');page.fill('#stepMin','8');page.fill('#stepMax','18');page.fill('#tokenPMax','0.4')
  assert page.evaluate('ROLLOUT_VIEWER.state.filtered.length>0 && ROLLOUT_VIEWER.state.filtered.every(b=>ROLLOUT_VIEWER.state.map.get(b.sample).greedy_iou<=.5&&b.iou>=.5&&b.greedy_delta>=.1&&b.position+1>=8&&b.position+1<=18&&b.probability<=.4)')
  filtered_url=page.url;page.reload();ready(page)
  assert page.locator('#candidateMin').input_value()=='0.5' and page.locator('#stepMax').input_value()=='18'
  page.locator('.filter-details>summary').click();page.fill('#candidateMax','0.2')
  assert page.evaluate('ROLLOUT_VIEWER.state.grouped.length===0') and page.locator('#numericStatus').inner_text()
  page.click('#resetNumeric');assert page.evaluate('ROLLOUT_VIEWER.state.grouped.length===900')
  page.locator('.filter-details>summary').click()
  assert '겹침 점수' not in page.locator('body').inner_text()
  checks.append('numeric_range_filters_validation_reset_and_shared_URL_without_categorical_controls')

  page.click('#nextPage');assert page.locator('#pageInfo').inner_text().startswith('2 /')
  page.goto(BASE+f'#preset=dot&sample={KEY}&candidate=3');page.wait_for_function('ROLLOUT_VIEWER.state.current.sample.key==="'+KEY+'" && ROLLOUT_VIEWER.state.candidate===3');ready(page)
  overview=page.locator('#resultOverview').inner_text();assert all(t in overview for t in ['0 – 106초','0 – 1.3초','0 – 107.3초','IoU 0.988'])
  assert page.locator('#answerComparison mark').count()==2
  assert page.locator('#answerComparison').evaluate('(el)=>el.closest("details").id')=='videoDetails'
  assert page.locator('details#rawAnswers').count()==0
  assert page.locator('#distAdvanced,#distPrevious,#distNext,#distStep,#distBranchPoint').count()==0
  assert page.locator('#distTokenStrip').is_visible() and page.locator('#distSearch').is_visible()
  assert page.locator('#selectionAnalysis').get_attribute('open') is None
  assert page.locator('#originContext').inner_text()==''
  page.locator('#selectionAnalysis>summary').click();page.select_option('#budget','4');page.select_option('#method','oracle')
  assert page.evaluate('ROLLOUT_VIEWER.state.candidate===3')
  assert page.locator('#candidates tr').count()==4
  page.select_option('#budget','32');page.select_option('#method','probability_weighted_medoid')
  page.locator('#selectionAnalysis>summary').click()
  deep_link=page.url;page.reload();ready(page)
  assert page.evaluate('ROLLOUT_VIEWER.state.current.sample.key==="'+KEY+'" && ROLLOUT_VIEWER.state.candidate===3')
  checks.append('GT_before_after_token_change_deep_link_and_selector_independence')
  parent_branch=next(b for b in manifest['branches'] if b['parent']!=1)
  page.evaluate('([key,c])=>ROLLOUT_VIEWER.selectCase(key,c)',[parent_branch['sample'],parent_branch['candidate']])
  page.wait_for_function('(key)=>ROLLOUT_VIEWER.state.current.sample.key===key',arg=parent_branch['sample'])
  assert 'greedy 응답과 다릅니다' in page.locator('#originContext').inner_text()
  checks.append('intermediate_parent_explicitly_distinguished_from_initial_response')
  page.click('[data-preset="all"]');ready(page)
  for dataset in ['charades-timelens','activitynet-timelens','qvhighlights-timelens']:
   page.select_option('#dataset',dataset)
   assert page.evaluate('ROLLOUT_VIEWER.state.grouped.length===300')
   page.locator('.case-card').first.click();ready(page)
   page.wait_for_function('document.querySelector("video").readyState>=2&&document.querySelector("video").videoWidth>0',timeout=60000)
   page.locator('[data-overview-play="truth"]').click()
   page.wait_for_function('!document.querySelector("video").paused',timeout=15000)
   page.wait_for_function('document.querySelector("video").currentTime>ROLLOUT_VIEWER.state.current.sample.gt[0]+.1',timeout=15000)
   page.evaluate('document.querySelector("video").pause()')
   assert page.locator('#videoDetails').get_attribute('open') is not None
   response=page.request.get(urljoin(BASE,page.locator('#video').get_attribute('src')),headers={'Range':'bytes=0-1023'})
   assert response.status==206 and len(response.body())==1024
  page.fill('#search','NO_MATCH_999999');assert page.locator('.case-card').count()==0 and '조건에 맞는 샘플이 없습니다' in page.locator('#branches').inner_text()
  checks.append('dataset_filters_replay_opens_video_three_datasets_byte_ranges_and_empty_search')
  page.goto(BASE+f'#preset=dot&sample={KEY}&candidate=3');page.wait_for_function('ROLLOUT_VIEWER.state.current.sample.key==="'+KEY+'" && ROLLOUT_VIEWER.state.candidate===3');ready(page)
  page.wait_for_function('document.querySelectorAll(".distribution-child").length===2',timeout=60000);page.evaluate('document.fonts.ready');page.evaluate('scrollTo(0,0)')
  assert page.evaluate('Array.from(document.fonts).some(f=>f.family==="RolloutKorean"&&f.status==="loaded")')
  page.screenshot(path=str(REPORT/'preview_desktop.png'))
  page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  page.locator('.case-card').first.click();page.wait_for_timeout(450)
  assert page.evaluate('document.querySelector("#samplePanel").getBoundingClientRect().top<100')
  page.screenshot(path=str(REPORT/'preview_mobile.png'))
  checks.append('korean_font_mobile_no_overflow_and_case_selection_scrolls_to_comparison')
  browser.close()
 assert not errors,errors
 write(OUT/'validation/viewer_browser.json',dict(passed=True,checks=checks,page_errors=errors,deep_link=deep_link,all_branch_metrics_match_source=True,reused_existing_video_cache=True))
 print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False))
if __name__=='__main__':run()
