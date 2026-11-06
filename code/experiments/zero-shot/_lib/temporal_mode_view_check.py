"""Check common navigation plus mode/branch interaction without fake result files."""
import argparse
import copy
import json
import os
import sys
from urllib.request import Request, urlopen
from temporal_mode_core import *

SHARED = ROOT.parent/'_shared/timelens'
sys.path.insert(0, str(SHARED/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(SHARED/'browser_cache')
os.environ['LD_LIBRARY_PATH'] = str(SHARED/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE'] = str(SHARED/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright


def run():
    base = 'http://127.0.0.1:8780/'
    scope = base+'zero-shot/results/rollout/temporal_mode_search/pilot/600/'
    sample = read(OUT/'web/samples.json')[0]
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=['--no-sandbox','--disable-dev-shm-usage'])
        page = browser.new_page(viewport=dict(width=1600,height=1100))
        page.on('pageerror', lambda error: errors.append(str(error)))
        # Fixture is intercepted in memory only; it never enters the source store.
        seq = dict(id='main', ids=[1,9], text='The event happens in 1 - 3 seconds.', raw_logp=-1.,
                   raw=[1.,3.],span=[1.,3.],valid=True,clipped=False)
        tail_seq = dict(seq, id='tail', ids=[2,9], raw=[5.,8.],span=[5.,8.])
        group = dict(id=0,paths=['main'],intervals=[[1.,3.]],representatives=[[1.,3.]],unique_intervals=1,observed_log_mass=-1.)
        tail_group = dict(group,id=1,paths=['tail'],intervals=[[5.,8.]],representatives=[[5.,8.]])
        result = dict(main_modes=[group],tail_modes=[tail_group],main_candidate_hash='fixture',tail_isolation_verified=True,
            main=dict(spent=50,cap=CONFIG['main_budget'],unfinished_minimum=0,mi=[]),
            tail=dict(spent=50,cap=CONFIG['tail_budget'],unfinished_minimum=0),
            selection=dict(span=[1.,3.],status='single_main_candidate',cost=0,comparisons=[],assessments=[]),
            draws=[dict(sequence='main',kind='main',mode=0),dict(sequence='tail',kind='tail',mode=1)])
        fixture = dict(baseline=seq,sequences=dict(main=seq,tail=tail_seq),
            distributions=[dict(prefix=[],prefix_text='',accepted=[dict(token=1,p=.9,core=True,text='1'),dict(token=2,p=.05,core=False,text='5')],
                entropy_bits=.4,threshold=.009,excluded_mass=.05)],
            results={f'{method}:{seed}':result for method in METHODS for seed in CONFIG['seeds']})
        pattern = '**/responses/'+sample['key']+'.json'
        page.route(pattern, lambda route: route.fulfill(status=200,content_type='application/json',body=json.dumps(fixture)))
        page.goto(scope+'detail.html?sample='+sample['key'],wait_until='networkidle')
        page.wait_for_selector('.mode.main')
        assert page.locator('.mode.main').count() == 1
        assert page.locator('.mode.tail').count() == 1
        page.locator('[data-draw="0"]').click()
        assert 'token 위치' in page.locator('#branch-detail').inner_text()
        page.select_option('#method','probability')
        page.wait_for_selector('.mode.main')
        page.select_option('#seed','43')
        page.wait_for_selector('.mode.main')
        page.unroute(pattern)
        page.goto(scope+'detail.html?sample='+sample['key'],wait_until='networkidle')
        page.screenshot(path=str(RESULT/'preview_desktop.png'),full_page=True)
        page.set_viewport_size(dict(width=390,height=844))
        page.screenshot(path=str(RESULT/'preview_mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
        page.set_viewport_size(dict(width=1600,height=1100))
        page.goto(scope.replace('pilot/600/','')+'viewer.html',wait_until='networkidle')
        page.wait_for_selector('#analysis-tabs button',timeout=30000)
        labels = page.locator('#analysis-tabs').inner_text()
        assert '결과' in labels and '보고서' in labels, labels
        assert not errors, errors
        browser.close()
    if sample['video']:
        with urlopen(Request(base.rstrip('/')+sample['video'],headers={'Range':'bytes=0-127'}),timeout=10) as response:
            assert response.status == 206
            response.read(128)
    with urlopen(base,timeout=10) as response:
        assert response.status==200
    write(OUT/'validation/viewer.json',dict(passed=True,common_result_report_tabs=True,
        mode_cards=True,main_tail_separation=True,prefix_drilldown=True,mobile_no_overflow=True,
        video_range=True,page_errors=errors,timestamp=stamp()))
    print('VIEWER_CHECK_PASSED')


if __name__ == '__main__':
    run()
