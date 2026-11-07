"""Verify actual query navigation, prefix branches, tail hiding, and report figures."""
import json
import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())
sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
from temporal_mode_view_check import sync_playwright


def sample_layout(page):
    layout=page.evaluate('''() => {
        const list=document.querySelector('#samples'),box=list.getBoundingClientRect();
        const rows=[...list.querySelectorAll('.sample')].map(row=>{
            const r=row.getBoundingClientRect(),q=row.querySelector('.sample-query').getBoundingClientRect(),
                  score=row.querySelector('.sample-score').getBoundingClientRect();
            return {top:r.top,bottom:r.bottom,queryBottom:q.bottom,scoreTop:score.top,scoreBottom:score.bottom};
        });
        return {height:innerHeight,top:box.top,bottom:box.bottom,scroll:list.scrollHeight,
                client:list.clientHeight,rows};
    }''')
    assert len(layout['rows'])==10,layout
    assert layout['top']>=0 and layout['bottom']<=layout['height'],layout
    assert layout['scroll']<=layout['client']+1,layout
    for row in layout['rows']:
        assert row['top']>=layout['top'] and row['bottom']<=layout['bottom'],row
        assert row['queryBottom']<=row['scoreTop']+1 and row['scoreBottom']<=row['bottom'],row
    return layout


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
                assert page.locator('#branch-detail').inner_text().find('greedy와')>=0
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
                # The video precedes evaluation; mode membership comes from the checkpoint.
                assert page.evaluate('document.querySelector("#heading").nextElementSibling.id === "content"')
                assert page.evaluate('document.querySelector("#content").firstElementChild.id === "source-video"')
                assert page.evaluate('document.querySelector("#source-video").nextElementSibling.id === "gt-comparison"')
                assert 'zero-shot' not in page.locator('body').inner_text().lower()
                assert page.evaluate('''() => current().main_modes.every(g=>{
                    const node=document.querySelector(`[data-mode-group="main:${g.id}"]`);
                    return node && node.querySelectorAll('[data-sequence]').length===g.paths.length &&
                      g.paths.every(id=>node.querySelector(`[data-sequence="${id}"]`));
                })''')
                assert page.locator('#samples button').count()<=10
                if row==chosen[0]:
                    sample_layout(page)
                    page.locator('.finder-options>summary').click()
                    metrics=[r for r in json.loads((base/'metrics/per_query.json').read_text())
                             if r['method']=='temporal_mode' and r['seed']==42 and r['budget']==128]
                    expected_counts=dict(
                        recovered=sum(r['baseline']<.5 and r['iou']>=.5 for r in metrics),
                        improved=sum(r['iou']-r['baseline']>=.01-1e-9 for r in metrics),
                        worsened=sum(r['iou']-r['baseline']<=-.01+1e-9 for r in metrics),
                        missed=sum(r['baseline']<.5 and r['main_oracle']>=.5 and r['iou']<.5 for r in metrics),
                        unrecovered=sum(r['baseline']<.5 and r['main_oracle']<.5 for r in metrics),
                        multiple=sum(r['main_modes']>=2 for r in metrics),
                        invalid=sum(r['failed'] for r in metrics))
                    expected_counts['tail-opportunity']=sum(r['baseline']<.5 and r['main_oracle']<.5 and r['diagnostic_oracle']>=.5 for r in metrics)
                    for preset in ('recovered','improved','worsened','missed','unrecovered','multiple','tail-opportunity','invalid'):
                        page.select_option('#case-filter',preset)
                        expected=expected_counts[preset]
                        assert page.locator('#count').inner_text().startswith(str(expected)+' /')
                    page.select_option('#case-filter','all')
                    page.locator('.numeric-filter>summary').click()
                    page.fill('#greedy-min','0.2');page.fill('#greedy-max','0.1')
                    assert page.locator('#samples button').count()==0
                    assert page.locator('[aria-invalid="true"]').count()==2
                    page.click('#reset-cases')
                    page.locator('.numeric-filter>summary').click()
                    page.locator('.finder-options>summary').click()
                    page.click('#next-page')
                    assert page.locator('#page-info').inner_text().startswith('2 /')
                    restored_url=page.url
                    page.reload(wait_until='networkidle');page.wait_for_selector('#path-browser')
                    assert page.url==restored_url
                    assert page.locator('#page-info').inner_text().startswith('2 /')
                    page.click('#previous-page')
                    page.locator('[data-mode-group]').first.locator('summary').click()
                    candidate=page.locator('[data-candidate-path]:visible').first
                    candidate.click()
                    assert 'greedy와' in page.locator('#branch-detail').inner_text()
                    page.locator('[data-mode-group]').first.locator('summary').click()
                if row==chosen[0]:
                    result=ROOT/f'results/rollout/temporal_mode_search/pilot/{count}/capped_rollouts'
                    page.evaluate('scrollTo(0,0)')
                    page.screenshot(path=str(result/'preview_desktop.png'),full_page=True)
                    page.set_viewport_size(dict(width=390,height=844))
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                    assert not page.locator('#samples').is_visible()
                    page.click('#toggle-finder')
                    assert page.locator('#samples').is_visible()
                    sample_layout(page)
                    page.click('#toggle-finder')
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
        # A shared family URL restores the query and the exact filter context.
        page.goto('http://127.0.0.1:8780/zero-shot/results/rollout/temporal_mode_search/viewer.html',wait_until='networkidle')
        frame=page.frame_locator('#detail-frame')
        frame.locator('#path-browser').wait_for()
        frame.locator('.finder-options>summary').click()
        frame.locator('#dataset').select_option('qvhighlights-timelens')
        frame.locator('#case-filter').select_option('improved')
        frame.locator('.finder-options>summary').click()
        frame.locator('#samples button').first.click()
        frame.locator('#path-browser').wait_for()
        key=frame.locator('#selected-context b').inner_text()
        assert 'sample='+key in page.url
        page.reload(wait_until='networkidle')
        frame.locator('#path-browser').wait_for()
        assert frame.locator('#selected-context b').inner_text()==key
        assert frame.locator('#case-filter').input_value()=='improved'
        assert frame.locator('#dataset').input_value()=='qvhighlights-timelens'
        assert not errors,errors
        browser.close()
    run=ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts'
    (run/'validation/case_presentation.json').write_text(json.dumps(dict(passed=True,checked=checked,
        conditional_controls_verified=True,default_tail_hidden=True,query_metrics_match=True,
        report_images_loaded=True,case_filters_match_metrics=True,aligned_mode_membership=True,
        video_before_gt=True,url_context_restored=True,pagination_explicit=True,page_errors=errors),ensure_ascii=False,indent=2)+'\n')
    print('CASE_PRESENTATION_PASSED',len(checked))

if __name__=='__main__':run()
