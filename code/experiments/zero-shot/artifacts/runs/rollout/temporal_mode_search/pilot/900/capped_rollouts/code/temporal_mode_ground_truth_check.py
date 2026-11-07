"""Ground-truth rendering, async scope changes, pending data and missing media."""
import json
from temporal_mode_capped_core import *
from temporal_mode_view_check import sync_playwright


def run():
    catalog=read(OUT/'web/samples.json')
    sample=catalog[0]
    key=sample['key']
    raw=read(OUT/'responses'/f'{key}.json')
    pending=next((s for s in reversed(catalog) if not (OUT/'responses'/f'{s["key"]}.json').exists()
        and not (OUT/'responses'/f'{s["key"]}.partial.json').exists()),None)
    origin='http://127.0.0.1:8780/zero-shot/results/rollout/temporal_mode_search/pilot/600/capped_rollouts/detail.html'
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1600,height=1100))
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(origin+f'?sample={key}&method=probability&seed=42&budget=16',wait_until='networkidle')
        page.wait_for_selector('#gt-comparison .gt-compare-row')
        assert page.locator('#gt-comparison .gt-label').all_text_contents()==['GT','Zero-shot','최종 선택']
        assert f'{sample["gt"][0]:.3f}' in page.locator('#gt-comparison').inner_text()
        for budget in CHECKPOINTS:
            page.select_option('#budget',str(budget))
            page.wait_for_function('(k)=>document.getElementById("content").innerText.includes("Rollout main "+k+"/"+k)',arg=budget)
            score=iou(raw['results']['probability:42']['checkpoints'][str(budget)]['selection']['span'],sample['gt'])
            assert f'{score:.4f}' in page.locator('#gt-comparison .gt-value').nth(2).inner_text()
            assert page.locator('#gt-comparison').count()==1
            assert page.locator('.gt-reference').count()==page.locator('.mode .timeline').count()
            assert page.locator('.candidate-iou').count()==page.locator('.mode .timeline').count()
            assert page.locator('.gt-iou-heading').count()==1
        page.locator('[data-draw="0"]').click()
        assert page.locator('#branch-detail').inner_text()
        page.select_option('#seed','43')
        page.select_option('#method','token_entropy')
        page.wait_for_load_state('networkidle')
        assert page.locator('.gt-reference').count()>0
        page.set_viewport_size(dict(width=390,height=844))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.screenshot(path=str(RESULT/'preview_mobile.png'),full_page=True)
        page.set_viewport_size(dict(width=1600,height=1100))
        page.screenshot(path=str(RESULT/'preview_desktop.png'),full_page=True)
        if pending:
            page.goto(origin+f'?sample={pending["key"]}',wait_until='networkidle')
            page.wait_for_selector('#gt-comparison')
            assert f'{pending["gt"][0]:.3f}' in page.locator('#gt-comparison').inner_text()
            assert '대기 중' in page.locator('#gt-comparison').inner_text()
        modified=[dict(s,video=None) if s['key']==key else s for s in catalog]
        page.route('**/capped_rollouts/web/samples.json*',lambda route:route.fulfill(status=200,
            content_type='application/json',body=json.dumps(modified)))
        page.goto(origin+f'?sample={key}&budget=16',wait_until='networkidle')
        page.wait_for_selector('.mode.main')
        assert page.locator('video').count()==0 and page.locator('#gt-comparison').count()==1
        assert page.locator('.gt-reference').count()>0
        assert not errors,errors
        browser.close()
    write(OUT/'validation/ground_truth_viewer.json',dict(passed=True,ground_truth_from_cohort=True,
        gt_zero_shot_final=True,mode_gt_overlays=True,path_iou_column=True,
        budget_seed_method_switching=True,pending_query_gt_visible=bool(pending),
        gt_visible_without_video=True,mobile_no_overflow=True,page_errors=errors,timestamp=stamp()))
    print('GROUND_TRUTH_VIEWER_CHECK_PASSED')


if __name__=='__main__':
    run()
