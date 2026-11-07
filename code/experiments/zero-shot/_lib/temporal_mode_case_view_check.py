"""Verify actual query navigation, prefix branches, tail hiding, and report figures."""
import json
import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())
sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
from temporal_mode_view_check import sync_playwright


def run():
    errors=[];checked=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1600,height=1100))
        page.on('pageerror',lambda error:errors.append(str(error)))
        for count in (300,600,900):
            base=ROOT/f'artifacts/runs/rollout/temporal_mode_search/pilot/{count}/capped_rollouts'
            rows=json.loads((base/'web/samples.json').read_text())
            chosen=[rows[0]]
            if count==900:
                chosen += [next(r for r in rows if r['dataset']==ds) for ds in sorted({r['dataset'] for r in rows}) if ds!=rows[0]['dataset']]
            for row in chosen:
                url=f'http://127.0.0.1:8780/zero-shot/results/rollout/temporal_mode_search/pilot/{count}/capped_rollouts/detail.html?sample={row["key"]}'
                page.goto(url,wait_until='networkidle');page.wait_for_selector('#path-browser')
                assert page.locator('#budget-curve,#cost-frontier').count()==0
                assert page.locator('#query-budget tbody tr').count()==4
                assert page.locator('#query-budget th').count()==5
                # A branch distribution can be opened without choosing any leaf.
                page.locator('.distribution-open:visible').first.click()
                distribution=page.evaluate("() => {const d=prefixMap.get(JSON.stringify(distributionPrefix));return d.accepted.filter(a=>a.core).map(a=>({id:a.token,p:(100*a.p).toFixed(3)+'%'}))}")
                assert page.locator('.token-probabilities tbody tr').count()==len(distribution)
                sentence=page.evaluate("() => ({expected:prefixMap.get(JSON.stringify(distributionPrefix)).prefix_text, actual:document.querySelector('#sentence-text').textContent})")
                assert sentence['actual']==sentence['expected']+'▌'
                roots=page.evaluate("[...new Set(current().draws.filter(d=>d.kind==='main').map(d=>data.sequences[d.sequence].ids[0]))].length")
                if roots==1:
                    assert page.locator('.branch-tree>li>details>summary>.branch-token').count()==0

                for edge in distribution:
                    assert edge['p'] in page.locator(f'[data-probability-token="{edge["id"]}"]').inner_text()

                assert not page.locator('#show-tail').is_checked()
                assert page.locator('.path-leaf.tail,#tail-candidates').count()==0
                assert page.locator('#gt-comparison').is_visible()
                assert page.evaluate('document.querySelector("video").src.startsWith(new URL(window.TEMPORAL_EXPERIMENT_ROOT+"/",location.href).href)')
                # Traverse real collapsed branches rather than clicking hidden leaves.
                for _ in range(40):
                    if page.locator('.path-leaf.main:visible').count():break
                    page.locator('#branch-map details:not([open])>summary:visible').first.click()
                leaf=page.locator('.path-leaf.main:visible').first;leaf.click()
                assert page.locator('#branch-detail').inner_text().find('Zero-shot과')>=0
                if page.locator('#branch-position option').count():
                    expected=page.locator('#branch-position').input_value()
                    actual=page.evaluate('document.getElementById("token-distribution").textContent')
                    assert str(int(expected)+1)+'번째 token 선택 직전' in actual
                page.check('#show-tail');assert page.locator('#tail-candidates').count()==1
                page.uncheck('#show-tail');assert page.locator('.path-leaf.tail,#tail-candidates').count()==0
                if count==900:
                    for method in ('probability','token_entropy','temporal_mode'):
                        page.select_option('#method',method)
                        for seed in ('42','43'):
                            page.select_option('#seed',seed)
                            for budget in ('16','32','64','128'):
                                page.select_option('#budget',budget)
                                # Compare displayed values with stored checkpoint values.
                                check=page.evaluate('''() => {const p=current();const actual=iou(p.selection.span,selected.gt);return {shown:document.querySelector('#query-budget .active-budget td:nth-child(2)').textContent,actual:actual.toFixed(3),tail:document.querySelectorAll('.path-leaf.tail').length}}''')
                                assert check['shown']==check['actual'] and check['tail']==0,check
                page.select_option('#method','temporal_mode');page.select_option('#seed','42');page.select_option('#budget','128')
                if row==chosen[0]:
                    result=ROOT/f'results/rollout/temporal_mode_search/pilot/{count}/capped_rollouts'
                    page.screenshot(path=str(result/'preview_desktop.png'),full_page=True)
                    page.set_viewport_size(dict(width=390,height=844))
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                    page.screenshot(path=str(result/'preview_mobile.png'),full_page=True)
                    page.set_viewport_size(dict(width=1600,height=1100))
                checked.append(dict(scope=count,sample=row['key']))
        # Report tab contains aggregate figures and selected/oracle comparison.
        page.goto('http://127.0.0.1:8780/zero-shot/results/rollout/temporal_mode_search/viewer.html?view=pilot%2F900%2Fcapped_rollouts%2FREPORT.md',wait_until='networkidle')
        page.locator('#report h1').wait_for()
        assert 'Rollout 예산별 mIoU · 최종 선택과 oracle' in page.locator('#report').inner_text()
        images=page.locator('#report img');assert images.count()>=2
        for image in images.all():
            image.scroll_into_view_if_needed()
            page.wait_for_function('(img)=>img.complete&&img.naturalWidth>0',arg=image.element_handle())
        assert images.evaluate_all('(els)=>els.every(e=>e.complete&&e.naturalWidth>0)')
        assert not errors,errors
        browser.close()
    run=ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts'
    (run/'validation/case_presentation.json').write_text(json.dumps(dict(passed=True,checked=checked,
        conditional_controls_verified=True,default_tail_hidden=True,query_metrics_match=True,
        report_images_loaded=True,page_errors=errors),ensure_ascii=False,indent=2)+'\n')
    print('CASE_PRESENTATION_PASSED',len(checked))

if __name__=='__main__':run()
