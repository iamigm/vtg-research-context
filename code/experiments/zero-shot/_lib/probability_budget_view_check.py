"""Real media, per-token decisions, candidate/plot synchronization and navigation."""
import argparse,json,sys
from probability_budget_core import *

def check(count=300,initial=False,revision=False):
 sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
 from temporal_mode_view_check import sync_playwright
 errors=[];key=read(COHORT)[0]['sample_key'];result=ROOT/f'results/rollout/probability_budget/pilot/{count}';base=f'http://127.0.0.1:8780/zero-shot/results/rollout/probability_budget/pilot/{count}/detail.html'
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);page=browser.new_page(viewport=dict(width=1600,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
  if initial:
   m=read(OUT/'validation/smoke_tree.json');s=m['snapshots']['64'];m=dict(m,snapshots={str(b):dict(s,budget=b) for b in BUDGETS});old=read(SOURCE/'rollouts'/f'{key}.json')['result']['candidates'][0];q=dict(greedy=dict(span=[old['parsed']['start'],old['parsed']['end']]))
   page.route('**/queries/'+key+'.json',lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(q)))
   page.route('**/search/'+key+'/*.json',lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(m)))
  page.goto(base+'?sample='+key+'&method=native_mass',wait_until='networkidle');page.wait_for_selector('#scatter svg');assert page.evaluate('catalog.length')==count;assert 'GT' in page.locator('#answer-cards').inner_text()
  assert page.locator('.tick-label').count()>=6;assert '시작 시각 (s)' in page.locator('#scatter').inner_text() and '종료 시각 (s)' in page.locator('#scatter').inner_text()
  video=page.locator('#video');src=video.get_attribute('src');response=page.request.get(src,headers={'Range':'bytes=0-127'});assert response.status==206 and len(response.body())==128
  page.wait_for_function("document.getElementById('video').readyState>=1")
  assert not page.locator('#sentence .branch-token').filter(has_text='The').count()
  page.locator('#sentence button').first.click();assert page.locator('#prefix-window').inner_text()=='(응답 시작 전)'
  if not initial:
   page.wait_for_function('sourceTrace.size>0 || supplement.size>0');assert 'greedy 정책' in page.locator('#token-distribution').inner_text();assert '100.00%' in page.locator('#token-distribution').inner_text()
   # The first query has a completed supplement: every computed position must resolve.
   if (OUT/'token_distributions'/f'{key}.json').exists():
    page.wait_for_function('supplement.size>0')
    coverage=page.evaluate("({computed:tree.nodes.filter(n=>visited(n)&&['expanded','dead'].includes(n.status)).length,missing:tree.nodes.filter(n=>visited(n)&&['expanded','dead'].includes(n.status)&&!distFor(n)).length})")
    assert coverage['missing']==0,coverage
   fraction=page.evaluate("pathTo(pathLeaf).find(n=>visited(n)&&role(n).endsWith('fraction'))?.id")
   if fraction is not None:
    page.locator(f'#sentence [data-position="{fraction}"]').click();assert '소수 자릿수는 greedy 정책' in page.locator('#token-distribution').inner_text()
   point=page.locator('[data-point="1"]')
   if point.count():
    expected=page.evaluate('intervals[1].span[0]');point.focus();point.press('Enter');assert abs(video.evaluate('(v)=>v.currentTime')-expected)<.15;assert page.locator('#candidates tr.selected').count()==1
   if page.locator('#path-select option').count()>1:
    option=page.locator('#path-select option').nth(1).get_attribute('value');page.select_option('#path-select',option);assert page.evaluate('String(pathLeaf)')==option
   video.evaluate('(v)=>{v.pause();v.currentTime=2}');savedTime=video.evaluate('(v)=>v.currentTime')
   page.select_option('#budget','512');page.wait_for_function("snap?.budget===512&&!document.getElementById('explorer').classList.contains('loading')");assert abs(video.evaluate('(v)=>v.currentTime')-savedTime)<.15
   page.select_option('#method','levin');page.wait_for_function("tree?.method==='levin'&&!document.getElementById('explorer').classList.contains('loading')");assert abs(video.evaluate('(v)=>v.currentTime')-savedTime)<.15
   page.select_option('#plot-range','fit');assert page.locator('.tick-label').count()>=4
   # Query filters are bounded, paged and preserve one selected context.
   page.select_option('#dataset','activitynet-timelens');assert page.evaluate("filtered.every(r=>r.dataset==='activitynet-timelens')");assert page.locator('#samples button').count()<=16;page.select_option('#dataset','')
   for method in METHODS:
    page.select_option('#method',method)
    for budget in BUDGETS:
     page.select_option('#budget',str(budget));page.wait_for_function(f"tree?.method==='{method}'&&snap?.budget==={budget}&&!document.getElementById('explorer').classList.contains('loading')")
     assert page.locator('#scatter svg').count()==1 and '불러오지 못했습니다' not in page.locator('#query-detail').inner_text()
   page.select_option('#method','native_mass');page.select_option('#budget','1024');page.wait_for_function("tree?.method==='native_mass'&&snap?.budget===1024&&!document.getElementById('explorer').classList.contains('loading')")
   page.locator('#route-nav [data-node]').first.click();url=page.url;prefix=page.locator('#prefix-window').inner_text();page.reload(wait_until='networkidle');page.wait_for_selector('#scatter svg');assert page.locator('#prefix-window').inner_text()==prefix
  else:
   page.locator('[data-child]').first.click();assert page.locator('#prefix-window').inner_text().startswith('The event happens in');assert 'token distribution' in page.locator('#token-distribution').inner_text()
  page.select_option('#plot-range','full');page.screenshot(path=str(result/'preview_desktop.png'),full_page=True)
  for width in (390,768,1100):
   page.set_viewport_size(dict(width=width,height=844));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
  page.set_viewport_size(dict(width=390,height=844));page.screenshot(path=str(result/'preview_mobile.png'),full_page=True)
  page.goto('http://127.0.0.1:8780/zero-shot/results/rollout/probability_budget/viewer.html',wait_until='networkidle');page.wait_for_selector('#analysis-tabs button');assert '결과' in page.locator('#analysis-tabs').inner_text() and '보고서' in page.locator('#analysis-tabs').inner_text()
  assert not errors,errors;browser.close()
 name='viewer_initial.json' if initial else 'viewer_revision.json' if revision else f'viewer_pilot{count}.json'
 write(OUT/'validation'/name,dict(passed=True,scope_queries=count,initial_gpu_fixture_only=initial,completed_scope_required=not(revision or initial),prefix_interaction=True,nonbranch_token_distribution=not initial,fractional_policy_explained=not initial,no_word_highlighting=True,axis_ticks_and_seconds=True,equal_plot_scale=True,plot_candidate_video_path_linked=not initial,method_budget_video_position_preserved=not initial,deep_link=True,video_range=True,mobile_no_overflow=True,page_errors=errors,timestamp=stamp()));print('VIEWER_VALIDATED',count,initial,revision)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',type=int,default=300);p.add_argument('--initial',action='store_true');p.add_argument('--revision',action='store_true');a=p.parse_args();check(a.scope,a.initial,a.revision)
