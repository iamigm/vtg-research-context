"""Real browser validation for pending and completed oracle-switch pilot3 views."""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from oracle_switch_core import OUT, RESULTS, ROOT, read, write, stamp

def require(condition,message):
    if not condition: raise AssertionError(message)

def check():
    selection=read(OUT/'metadata/selection.json'); samples=selection['samples']
    require(selection['scope_queries']==len(samples)==3,'Expected the selected pilot3 cohort')
    require((RESULTS/'detail.html').is_file() and (RESULTS/'REPORT.md').is_file(),'Report must be generated before browser validation')
    records={sample['sample_key']:read(path) if (path:=OUT/'queries'/f"{sample['sample_key']}.json").exists() else {}
             for sample in samples}
    completed=sum(bool(record.get('complete')) for record in records.values())
    shared=ROOT.parent/'_shared/timelens'
    sys.path.insert(0,str(shared/'browser_deps'))
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
    os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
    from playwright.sync_api import sync_playwright
    base='http://127.0.0.1:8780'
    family=base+'/zero-shot/results/rollout/oracle_switch/viewer.html'
    detail=base+'/zero-shot/results/rollout/oracle_switch/pilot/3/detail.html'
    errors=[]; verified_links=set(); media_checks=[]; tested_directions=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
            page=browser.new_page(viewport=dict(width=1600,height=1100))
            page.on('pageerror',lambda error:errors.append(str(error)))
            def links(locator):
                for url in locator.evaluate_all("nodes=>nodes.map(node=>node.href||node.src).filter(Boolean)"):
                    parsed=urlparse(url)
                    if parsed.netloc!='127.0.0.1:8780' or url in verified_links: continue
                    response=page.request.head(url)
                    require(response.status==200,f'Local result/source link failed: {url} ({response.status})')
                    verified_links.add(url)
            def visible_case(key):
                case=page.locator(f'article.case[data-case="{key}"]')
                case.wait_for(state='visible')
                require(page.locator('article.case:visible').count()==1,'Sample switch left multiple cases visible')
                require(page.locator(f'.query-tabs button[data-sample="{key}"]').get_attribute('aria-pressed')=='true',
                        'Selected query lacks an accessible active state')
                require(page.evaluate("new URLSearchParams(location.hash.slice(1)).get('sample')")==key,'Sample deep link differs')
                return case

            response=page.goto(detail,wait_until='networkidle')
            require(response and response.status==200,'Detail page did not load')
            require('pilot3' in page.title(),'Detail scope differs from pilot3')
            require(page.locator('.query-tabs button[data-sample]').evaluate_all('nodes=>nodes.map(node=>node.dataset.sample)')
                    ==[sample['sample_key'] for sample in samples],'Viewer sample order differs from selection')
            require(page.locator('.summary > div > b').first.inner_text()==f'{completed} / 3','Displayed completed query count differs')
            links(page.locator('a[href],link[href]'))
            for sample in samples:
                key=sample['sample_key']; record=records[key]
                button=page.locator(f'.query-tabs button[data-sample="{key}"]'); button.focus(); button.press('Enter')
                case=visible_case(key)
                score_text=case.locator('.scores b').all_inner_texts()
                expected=[sample['greedy']['iou'],sample['oracle']['iou'],sample['delta_iou']]
                require(len(score_text)==3 and all(math.isclose(float(a),b,abs_tol=5.1e-5) for a,b in zip(score_text,expected)),
                        f'{key}: displayed source scores differ')
                video=case.locator('video'); require(video.count()==1,'Original video missing or duplicated')
                source=video.evaluate('video=>video.src')
                expected_path=Path(sample['video_path']).resolve().relative_to(ROOT.parent.resolve()).as_posix()
                require(urlparse(source).path=='/'+expected_path,'Viewer video differs from selected reused media')
                response=page.request.get(source,headers={'Range':'bytes=0-127'})
                require(response.status==206 and len(response.body())==128,'Original video Range request failed')
                page.wait_for_function("key=>document.querySelector(`article[data-case='${key}'] video`).readyState>=1",arg=key,timeout=30000)
                seek=case.locator('.timeline-span.oracle'); require(seek.count()==1,'Oracle timeline is missing')
                expected_time=float(seek.get_attribute('data-seek')); require(expected_time==max(0,sample['oracle']['span'][0]),'Oracle seek time differs')
                seek.click()
                page.wait_for_function("""({key,time})=>{const v=document.querySelector(`article[data-case='${key}'] video`);
                    return v.paused && !v.seeking && Math.abs(v.currentTime-time)<.15;}""",arg=dict(key=key,time=expected_time),timeout=30000)
                media_checks.append(dict(sample_key=key,url=source,range=True,seek_time=expected_time))
                directions=record.get('directions',[])
                if isinstance(directions,dict): directions=[dict(value,id=name) for name,value in directions.items()]
                if not directions:
                    require(case.locator('.direction-picker').count()==0 and case.locator('.empty').count()>0,
                            'Pending case must explain missing new measurements')
                else:
                    picker=case.locator('.direction-picker'); require(picker.count()==1,'Measured support selector missing')
                    require(picker.locator('option').evaluate_all('nodes=>nodes.map(node=>node.value)')==[item['id'] for item in directions],
                            'Measured support coverage differs from query data')
                    for direction in directions:
                        name=direction['id']; picker.select_option(name)
                        panel=case.locator(f'.direction-view[data-direction="{name}"]'); panel.wait_for(state='visible')
                        require(case.locator('.direction-view:visible').count()==1,'Support selection left multiple curves visible')
                        require(panel.locator('.margin-curve').count()==1 and panel.locator('.margin-curve circle').count()>0,
                                'Measured margin curve missing')
                        for picture in panel.locator('.preview-grid img').all():
                            picture.scroll_into_view_if_needed()
                            picture.evaluate('image=>image.decode()')
                            require(picture.evaluate('image=>image.complete && image.naturalWidth>0'),'Input perturbation preview failed to decode')
                        tested_directions.append(dict(sample_key=key,direction=name))
                links(case.locator('a[href]'))
                page.reload(wait_until='networkidle'); visible_case(key)
                page.set_viewport_size(dict(width=390,height=844))
                require(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),f'{key}: detail overflows at390px')
                page.set_viewport_size(dict(width=1600,height=1100))

            preview=next((sample for sample in samples if records[sample['sample_key']].get('complete')),samples[0])
            page.goto(detail+'#sample='+preview['sample_key'],wait_until='networkidle'); visible_case(preview['sample_key'])
            page.screenshot(path=str(RESULTS/'preview_desktop.png'),full_page=True)
            page.set_viewport_size(dict(width=390,height=844))
            require(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Mobile preview overflow')
            page.screenshot(path=str(RESULTS/'preview_mobile.png'),full_page=True)

            page.set_viewport_size(dict(width=1600,height=1100))
            response=page.goto(family+'?view=pilot%2F3%2Fdetail.html',wait_until='networkidle')
            require(response and response.status==200,'Family viewer missing')
            page.wait_for_selector('#analysis-tabs button')
            require('결과' in page.locator('#analysis-tabs').inner_text() and '보고서' in page.locator('#analysis-tabs').inner_text(),
                    'Common result/report tabs missing')
            require(page.locator('#view-tree-items button[data-view="pilot/3/detail.html"]').count()==1,'Family pilot3 scope missing')
            frame=page.frame_locator('#detail-frame'); frame.locator('.query-tabs button[data-sample]').first.wait_for()
            family_key=samples[-1]['sample_key']
            frame.locator(f'.query-tabs button[data-sample="{family_key}"]').click()
            page.wait_for_function("key=>new URLSearchParams(location.hash.slice(1)).get('sample')===key",arg=family_key)
            page.reload(wait_until='networkidle')
            frame=page.frame_locator('#detail-frame'); frame.locator(f'article.case[data-case="{family_key}"]').wait_for(state='visible')
            page.locator('#tab-report').click(); page.wait_for_selector('#report h1')
            require('3' in page.locator('#report h1').inner_text() and page.locator('#source-md').is_visible(),'Report scope/source link missing')
            links(page.locator('#report a[href],#source-md'))
            page.locator('#tab-result').click(); frame.locator(f'article.case[data-case="{family_key}"]').wait_for(state='visible')
            require(not page.locator('#source-md').is_visible(),'MD source link leaked into result tab')
            page.set_viewport_size(dict(width=390,height=844))
            require(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Family viewer overflows at390px')
            require(page.request.get(base+'/').status==200,'Unified result index failed')
            require(not errors,f'Browser errors: {errors}')
            browser.close()
        report=dict(passed=True,scope='pilot',scope_queries=3,completed_queries=completed,pending_queries=3-completed,
            complete=completed==3,sample_selection=True,sample_deep_links=True,source_scores=True,
            media=media_checks,support_switches=tested_directions,desktop_and_mobile=True,
            family_scope3=True,result_report_navigation=True,family_deep_link=True,
            verified_local_links=sorted(verified_links),page_errors=errors,unified_index=True,
            previews=[str(RESULTS/'preview_desktop.png'),str(RESULTS/'preview_mobile.png')],timestamp=stamp())
        write(OUT/'validation/viewer.json',report)
        return report
    except BaseException as error:
        write(OUT/'validation/viewer.json',dict(passed=False,scope_queries=3,completed_queries=completed,
            error=str(error),page_errors=errors,timestamp=stamp()))
        raise

if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False))
