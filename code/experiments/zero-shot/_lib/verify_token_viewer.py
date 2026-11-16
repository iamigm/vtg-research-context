"""Verify the unified token viewer using stored per-path values and raw traces."""
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import urlopen,Request

ROOT=Path('/home/irteam/experiments/zero-shot')
sys.path.insert(0,str(ROOT/'_lib'))
from result_layout import relocate_path
from publish_token_viewer import RESULT,ART,save


def check(scopes=('pilot/300','full/9404')):
    resources=ROOT.parent/'_shared/timelens';sys.path.insert(0,str(resources/'browser_deps'))
    os.environ.update(PLAYWRIGHT_BROWSERS_PATH=str(resources/'browser_cache'),
                     LD_LIBRARY_PATH=str(resources/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH',''),
                     FONTCONFIG_FILE=str(resources/'browser_libs/fonts.conf'))
    from playwright.sync_api import sync_playwright
    base='http://127.0.0.1:8780/';errors=[];checks=[];blue_responses=[]
    url=lambda p:base+str(p.relative_to(ROOT.parent))
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page(viewport={'width':1500,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('response',lambda r:blue_responses.append(r.status) if '/blue_analysis/web/data.js' in r.url else None)
        for scope in scopes:
            dest=RESULT/scope;data=json.loads((ART/scope/'metrics/viewer_catalog.json').read_text())
            s=relocate_path(dest/'digit_correctness/manifest.js').read_text();manifest,_=json.JSONDecoder().raw_decode(s,s.index('=')+1)
            page.goto(url(dest/'report.html'));page.wait_for_function('window.TokenViewer?.state().ready')
            for model in ('qwen25','qwen3'):
                page.select_option('#model',model)
                for endpoint in ('start','end'):
                    page.select_option('#endpoint',endpoint)
                    expected={s['sample_key']:data.get(model+'/'+s['sample_key'],{}).get('endpoints',{}).get(endpoint,{}) for s in manifest}
                    tests={
                        'all':lambda r:True,'paired':lambda r:r.get('paired'),
                        'increase':lambda r:r.get('delta_previous') is not None and r['delta_previous']>0,
                        'decrease':lambda r:r.get('delta_previous') is not None and r['delta_previous']<0,
                        'low_error':lambda r:r.get('error_entropy') is not None and r['error_entropy']<=.5,
                        'high_safe':lambda r:r.get('safe_high'),
                        'next_failure':lambda r:r.get('next_failure')}
                    for preset,predicate in tests.items():
                        page.click('[data-preset="'+preset+'"]')
                        ids=sorted(k for k,r in expected.items() if predicate(r))
                        page.wait_for_function('n=>TokenViewer.state().count===n',arg=len(ids))
                        assert sorted(page.evaluate('TokenViewer.state().visible'))==ids,(scope,model,endpoint,preset)
                    page.click('[data-preset="disagree"]')
                    key=endpoint+'_correct'
                    disagreement=sorted(s['sample_key'] for s in manifest if s['models']['qwen25']['supported'] and s['models']['qwen3']['supported'] and s['models']['qwen25'][key]!=s['models']['qwen3'][key])
                    assert sorted(page.evaluate('TokenViewer.state().visible'))==disagreement
                    page.click('[data-preset="all"]');page.locator('#metric-filter').evaluate('(e)=>e.open=true')
                    page.fill('#delta_previous-min','0');page.fill('#error_entropy-max','1');page.locator('#error_entropy-max').dispatch_event('change')
                    ids=sorted(k for k,r in expected.items() if r.get('delta_previous') is not None and r['delta_previous']>=0 and r.get('error_entropy') is not None and r['error_entropy']<=1)
                    assert sorted(page.evaluate('TokenViewer.state().visible'))==ids
                    page.click('#reset')
            # A valid path and raw probability distribution must follow the selected step.
            page.click('[data-preset="paired"]');page.wait_for_function('TokenViewer.state().ready')
            state=page.evaluate('TokenViewer.state()');raw_path=relocate_path(dest/'digit_correctness/raw'/state['model']/(state['sample']+'.json'))
            raw=json.loads(raw_path.read_text());assert page.locator('#path-plot circle').count()==len(raw['steps'])
            target=raw['steps'][1];page.locator('#path-plot circle[data-step="'+str(target['step'])+'"]').click()
            assert page.evaluate('TokenViewer.state().step')==target['step']
            assert 'step '+str(target['step']) in page.locator('#distribution-info').inner_text()
            assert page.locator('#candidate-rows tr').count()==len(target['raw']['top30'])
            probabilities=[float(x.replace('%',''))/100 for x in page.locator('#candidate-rows tr td:nth-child(3)').all_text_contents()]
            mass=sum(t['probability'] for t in target['raw']['top30'])
            assert max(abs(a-b['probability']/mass) for a,b in zip(probabilities,target['raw']['top30']))<1e-7
            page.select_option('#distribution-mode','processed')
            probabilities=[float(x.replace('%',''))/100 for x in page.locator('#candidate-rows tr td:nth-child(3)').all_text_contents()]
            processed_mass=sum(t['probability'] for t in target['processed']['top30'])
            assert max(abs(a-b['probability']/processed_mass) for a,b in zip(probabilities,target['processed']['top30']))<1e-7
            page.select_option('#distribution-mode','raw')
            # Range offsets reconstruct an original record without downloading the full candidate file.
            key=state['model']+'/'+state['sample'];ref=data[key]['blue_ranges'][0];ep,st,a,b=ref
            source=relocate_path(dest/'digit_correctness/blue_analysis/data.js')
            with source.open('rb') as stream:stream.seek(a);original=json.loads(stream.read(b-a+1))
            with urlopen(Request(url(source),headers={'Range':f'bytes={a}-{b}'})) as response:
                assert response.status==206;assert json.loads(response.read())==original
            unsupported=next((s for s in manifest if not s['models']['qwen3']['supported']),None)
            if unsupported:
                page.goto(url(dest/'report.html')+'#model=qwen3&preset=all&sample='+unsupported['sample_key'])
                page.wait_for_function('TokenViewer.state().ready')
                assert page.locator('#path-plot circle').count()>0
                assert '지원 표기 밖' in page.locator('#path-explanation').inner_text()
            # No-result filtering must clear previous sample, graph and video state.
            page.fill('#search','__absent_query__');page.wait_for_function('TokenViewer.state().count===0')
            assert not page.evaluate('TokenViewer.state().ready') and page.locator('#case-content').is_hidden()
            assert page.locator('#video').get_attribute('src') is None
            page.click('#reset');page.click('[data-preset="decrease"]');page.wait_for_function('TokenViewer.state().ready')
            page.click('#jump-error');state=page.evaluate('TokenViewer.state()');saved_url=page.url
            page.reload();page.wait_for_function('TokenViewer.state().ready')
            restored=page.evaluate('TokenViewer.state()');assert all(restored[k]==state[k] for k in ('sample','model','endpoint','step','preset','count'))
            assert page.url==saved_url
            page.locator('#video-box').evaluate('(e)=>e.open=true');page.wait_for_function('document.getElementById("video").readyState>=2')
            page.click('#play-gt');page.wait_for_function('!document.getElementById("video").paused');page.locator('#video').evaluate('(e)=>e.pause()')
            page.locator('#video-box').evaluate('(e)=>e.open=false');page.locator('#metric-filter').evaluate('(e)=>e.open=false')
            page.evaluate('document.fonts.ready');page.screenshot(path=str(dest/'preview.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.screenshot(path=str(dest/'preview_mobile.png'),full_page=True);page.set_viewport_size({'width':1500,'height':1100})
            # Retired nested result selections resolve to the shared scope viewer.
            page.goto(url(RESULT/'viewer.html')+'?view='+scope+'/digit_correctness/report.html')
            page.wait_for_selector('#family-app[data-ready="true"]');page.frame_locator('#detail-frame').locator('#path-plot circle').first.wait_for()
            from urllib.parse import parse_qs,urlsplit
            assert parse_qs(urlsplit(page.url).query)['view']==[scope+'/report.html'],page.url
            page.locator('#tab-report').click();page.wait_for_function('document.querySelector("#report").textContent.includes("같은 경로의 오류 전후 차이")')
            page.locator('#reference-menu summary').click();page.locator('#reference-list a[data-view="'+scope+'/digit_correctness/within_path/REPORT.md"]').click()
            page.wait_for_function('document.querySelector("#report").textContent.includes("표 해설")')
            page.locator('#tab-result').click();page.frame_locator('#detail-frame').locator('#path-plot circle').first.wait_for()
            checks.append(dict(scope=scope,preset_counts=True,range_intersection=True,raw_probabilities=True,byte_range_identity=True,
                               empty_state=True,url_restore=True,video_playback=True,report_appendix=True,mobile=True))
            print(scope+': unified viewer checks passed',flush=True)
        browser.close()
    assert not errors,errors
    assert blue_responses and set(blue_responses)=={206},blue_responses
    save(ART/'overview/validation/unified_viewer.json',dict(passed=True,checks=checks,page_errors=errors,candidate_reads='HTTP 206 only'))
    return checks


if __name__=='__main__':check()
