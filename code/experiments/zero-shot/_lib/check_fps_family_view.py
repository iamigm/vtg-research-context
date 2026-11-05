"""Verify completed FPS reports, canonical family navigation and shared hosting."""
import os,sys,json,re,csv
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urljoin,urlsplit,unquote
ROOT=Path('/home/irteam/experiments');PROJECT=ROOT/'zero-shot'
FAMILY=PROJECT/'results/sampling/fps_spatial_control'
OUT=PROJECT/'artifacts/runs/sampling/fps_spatial_control/pilot/300'
shared=ROOT/'_shared/timelens'
sys.path[:0]=[str(ROOT/'_lib'),str(shared/'browser_deps')]
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
from family_view_test_utils import select_view

def main():
    manifest=json.loads((PROJECT/'artifacts/runs/sampling/fps_spatial_control/overview/web/view_manifest.json').read_text())
    assert manifest['default']=='pilot/300/REPORT.md'
    assert {e['scope_key'] for e in manifest['entries']}=={'pilot/300','pilot/overview'}
    assert len(list((OUT/'responses').rglob('*.json')))==6000
    assert json.loads((OUT/'validation/final.json').read_text())['passed']
    rows=list(csv.DictReader((OUT/'metrics/summary.csv').open()))
    text=(FAMILY/'pilot/300/REPORT.md').read_text()
    table=[line.split('|')[1:-1] for line in text.splitlines() if line.startswith('| ') and line.count('|')==14][1:]
    expected={(r['model'],r['order'],r['dataset'],r['fps']):r for r in rows}
    assert len(table)==len(rows)==80
    for cells in table:
        cells=[v.strip() for v in cells];r=expected[tuple(cells[:4])];assert int(cells[4])==int(r['n'])
        for i,key in enumerate(['miou','r03','r05','r07']):
            assert cells[5+2*i]==f"{100*float(r[key]):.2f}"
            assert cells[6+2*i]==f"{float(r['delta_'+key]):+.2f}"
    # Every local Markdown link resolves in the new layout, including raw metrics.
    for md in (FAMILY/'pilot').rglob('*.md'):
        for href in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if href.startswith(('http:','https:','#')):continue
            assert (md.parent/unquote(urlsplit(href).path)).exists(),(md,href)
    base='http://127.0.0.1:8780/'
    relative='zero-shot/results/sampling/fps_spatial_control/viewer.html'
    failures=[];errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('response',lambda response:failures.append([response.status,response.url]) if response.status>=400 and response.url.startswith(base) else None)
        page.goto(base,wait_until='networkidle')
        family=page.locator('article').filter(has=page.locator('h3 a[href="'+relative+'"]'))
        assert family.count()==1
        assert family.locator('h3 a').get_attribute('target')=='_blank'
        page.locator('details[data-project="zero-shot"] > summary .project-title').click()
        with page.expect_popup() as pop:
            family.locator('h3 a').click()
        view=pop.value
        view.on('pageerror',lambda error:errors.append(str(error)))
        view.on('response',lambda response:failures.append([response.status,response.url]) if response.status>=400 and response.url.startswith(base) else None)
        view.wait_for_selector('#family-app[data-ready="true"]')
        view.wait_for_function("document.querySelector('#report').textContent.includes('6,000')")
        assert view.locator('#report table').count()==2
        assert view.locator('#tab-report').get_attribute('aria-selected')=='true'
        assert view.locator('#tab-result').is_disabled()
        assert view.locator('#source-md').is_visible()
        assert 'viewer.html' in view.locator('#original').get_attribute('href')
        view.locator('#reference-menu summary').click()
        view.locator('#reference-list a[data-view="pilot/300/AXES.md"]').click()
        view.wait_for_function("document.querySelector('#report table') && document.querySelectorAll('#report table').length===48")
        assert 'AXES.md' in view.url
        view.reload();view.wait_for_selector('#family-app[data-ready="true"]')
        view.wait_for_function("document.querySelectorAll('#report table').length===48")
        select_view(view,'pilot/overview/REPORT.md')
        view.wait_for_function("new URL(location.href).searchParams.get('view')==='pilot/overview/REPORT.md'")
        select_view(view,'pilot/300/REPORT.md')
        view.wait_for_function("document.querySelector('#report').textContent.includes('6,000')")
        view.set_viewport_size({'width':390,'height':844})
        view.wait_for_timeout(250)
        assert view.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        # One latest desktop/mobile preview per screen type.
        view.screenshot(path=str(FAMILY/'pilot/300/preview_mobile.png'))
        view.set_viewport_size({'width':1440,'height':1000});view.screenshot(path=str(FAMILY/'pilot/300/preview_desktop.png'))
        browser.close()
    assert not errors,errors
    assert not failures,failures
    with urlopen(base+relative) as response:assert response.status==200
    with urlopen(base+'zero-shot/results/sampling/fps_spatial_control/pilot/300/REPORT.md') as response:
        assert 'charset=utf-8' in response.headers['Content-Type'].lower()
    validation=dict(passed=True,responses=6000,queries=300,metric_rows_compared=80,index_family_entries=1,
                    primary_report_default=True,new_tab_entry=True,reference_and_scope_navigation=True,url_restore=True,
                    desktop_mobile_layout=True,raw_markdown_utf8=True,failed_requests=failures,browser_errors=errors)
    (OUT/'validation/view.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False))
if __name__=='__main__':main()
