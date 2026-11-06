"""Budget switching, common family navigation and responsive mode interaction."""
import copy
import json
from urllib.request import Request,urlopen
from temporal_mode_capped_core import *
from temporal_mode_view_check import sync_playwright


def run():
    origin='http://127.0.0.1:8780/'
    page_url=origin+'zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/detail.html'
    sample=read(OUT/'web/samples.json')[0]
    main=dict(id='main',ids=[1,9],text='The event happens in 1 - 3 seconds.',raw=[1.,3.],span=[1.,3.],
        raw_logp=-1.,valid=True,clipped=False)
    tail=dict(main,id='tail',ids=[2,9],raw=[5.,8.],span=[5.,8.])
    group=dict(id=0,paths=['main'],intervals=[[1.,3.]],representatives=[[1.,3.]],unique_intervals=1,observed_log_mass=-1.)
    tgroup=dict(group,id=1,paths=['tail'],intervals=[[5.,8.]],representatives=[[5.,8.]])
    route=dict(checkpoints={},draws=[])
    previous=0
    for k in CHECKPOINTS:
        for kind in ('main','tail'):
            route['draws'] += [dict(sequence=kind,kind=kind,mode=0 if kind=='main' else 1) for _ in range(k-previous)]
        state=dict(spent=k*3,cap=65536,rollout_cap=k,rollouts=k,scheduler=dict(adaptive=k-8),
            unvisited_edges=0,unfinished_minimum=0,mi=[])
        route['checkpoints'][str(k)]=dict(main=state,tail=dict(state),main_modes=[group],tail_modes=[tgroup],
            main_candidate_hash='fixture',tail_isolation_verified=True,draw_cutoff=len(route['draws']),
            selection=dict(span=[1.,3.],status='single_main_candidate',pool=[group],cost=0,comparisons=[],assessments=[]))
        previous=k
    fixture=dict(baseline=main,sequences=dict(main=main,tail=tail),distributions=[dict(prefix=[],prefix_text='',
        entropy_bits=.5,threshold=.009,excluded_mass=.05,accepted=[dict(token=1,text='1',p=.9,core=True),dict(token=2,text='5',p=.05,core=False)])],
        results={f'{method}:{seed}':copy.deepcopy(route) for method in METHODS for seed in CONFIG['seeds']})
    summary=dict(groups=[dict(method=method,budget=k,queries=1,iou=.4+.02*__import__('math').log2(k/16)) for method in METHODS for k in CHECKPOINTS])
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1600,height=1100))
        page.on('pageerror',lambda e:errors.append(str(e)))
        pattern='**/capped_rollouts/responses/'+sample['key']+'.json'
        metric_pattern='**/capped_rollouts/metrics/summary.json'
        page.route(pattern,lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(fixture)))
        page.route(metric_pattern,lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(summary)))
        page.goto(page_url+'?sample='+sample['key'],wait_until='networkidle')
        page.wait_for_selector('.mode.main')
        assert page.locator('#budget option').count()==4
        assert page.locator('#seed option').all_text_contents()==['42','43']
        for k in CHECKPOINTS:
            page.select_option('#budget',str(k))
            page.wait_for_function('(k)=>document.getElementById("content").innerText.includes("Rollout main "+k+"/"+k)',arg=k)
            assert page.locator('.mode.main').count()==1 and page.locator('.mode.tail').count()==1
        page.locator('[data-draw="0"]').click()
        assert 'token 위치' in page.locator('#branch-detail').inner_text()
        assert page.locator('#budget-curve polyline').count()==3
        page.set_viewport_size(dict(width=390,height=844))
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
        page.unroute(pattern)
        page.unroute(metric_pattern)
        page.goto(page_url+'?sample='+sample['key'],wait_until='networkidle')
        page.screenshot(path=str(RESULT/'preview_mobile.png'),full_page=True)
        page.set_viewport_size(dict(width=1600,height=1100))
        page.screenshot(path=str(RESULT/'preview_desktop.png'),full_page=True)
        page.goto(origin+'zero-shot/results/rollout/temporal_mode_search/viewer.html',wait_until='networkidle')
        page.wait_for_selector('#analysis-tabs button')
        assert '결과' in page.locator('#analysis-tabs').inner_text() and '보고서' in page.locator('#analysis-tabs').inner_text()
        manifest=read(LEGACY.parent.parent/'overview/web/view_manifest.json')
        assert manifest['default']=='pilot/600/capped_rollouts/detail.html'
        assert not errors,errors
        browser.close()
    if sample['video']:
        with urlopen(Request(origin.rstrip('/')+sample['video'],headers={'Range':'bytes=0-127'}),timeout=10) as response:
            assert response.status==206
            response.read(128)
    with urlopen(origin,timeout=10) as response:
        assert response.status==200
    write(OUT/'validation/viewer.json',dict(passed=True,powers_of_two_selector=True,main_tail_cards=True,
        common_result_report_tabs=True,family_default_is_current_condition=True,mobile_no_overflow=True,
        budget_curves=True,page_errors=errors,timestamp=stamp()))
    print('CAPPED_VIEWER_CHECK_PASSED')


if __name__=='__main__':
    run()
