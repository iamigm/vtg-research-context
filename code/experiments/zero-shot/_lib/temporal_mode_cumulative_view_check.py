"""Scope selection and GT/greedy presentation for shared cumulative responses."""
import json
from pathlib import Path
import temporal_mode_cumulative_scope as scope


def run():
    core=scope.configure(900)
    from temporal_mode_view_check import sync_playwright
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1600,height=1100))
        page.on('pageerror',lambda error:errors.append(str(error)))
        for count in (300,600,900):
            rows=core.read(scope.COHORT_ROOT/f'pilot_{count}/samples.json')
            key=rows[0]['sample_key']
            url=f'http://127.0.0.1:8780/zero-shot/results/rollout/temporal_mode_search/pilot/{count}/capped_rollouts/detail.html?sample={key}'
            page.goto(url,wait_until='networkidle')
            page.wait_for_selector('#gt-comparison')
            assert page.locator('#samples button').count()==10
            assert page.locator('#count').inner_text().startswith(f'{count} / {count} query')
            assert page.locator('.mode.main').count()>0
            assert '/pilot/900/capped_rollouts/responses/' in page.locator('#raw').get_attribute('href')
            assert page.locator('#budget option').count()==4 and page.locator('#seed option').all_text_contents()==['42','43']
            result=core.ROOT/f'results/rollout/temporal_mode_search/pilot/{count}/capped_rollouts'
            page.screenshot(path=str(result/'preview_desktop.png'),full_page=True)
            page.set_viewport_size(dict(width=390,height=844))
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.screenshot(path=str(result/'preview_mobile.png'),full_page=True)
            page.set_viewport_size(dict(width=1600,height=1100))
        assert not errors,errors
        browser.close()
    core.write(core.OUT/'validation/cumulative_viewer.json',dict(passed=True,scopes=[300,600,900],
        shared_canonical_response_urls=True,gt_visible=True,greedy_report=True,page_errors=errors,timestamp=core.stamp()))
    print('CUMULATIVE_VIEWER_PASSED')


if __name__=='__main__':
    run()
