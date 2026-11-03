from result_layout import relocate_path, write_result_text
from timelens_paths import RESOURCES
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
import json,os,sys
from pathlib import Path
sys.path.insert(0,str(relocate_path(RESOURCES/'browser_deps')))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(relocate_path(RESOURCES/'browser_cache'))
os.environ['LD_LIBRARY_PATH']=str(relocate_path(RESOURCES/'browser_libs/usr/lib/x86_64-linux-gnu'))+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(relocate_path(RESOURCES/'browser_libs/fonts.conf'))
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1350,'height':950})
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((relocate_path(DYNAMICS / 'report.html')).as_uri(),wait_until='load')
    assert page.locator('#sample option').count()==300
    first=page.locator('#raw').inner_html()
    page.locator('#next').click();assert page.locator('#step').inner_text().startswith('2 /')
    assert page.locator('#raw').inner_html()!=first
    page.locator('#mode').select_option('consumed');assert page.locator('#raw').inner_html()!=first
    page.locator('#dataset').select_option('QVHighlights');assert page.locator('#sample option').count()==100
    page.locator('#search').fill('QVHighlights_18');assert page.locator('#sample option').count()==1
    page.locator('#slider').evaluate('(e)=>{e.value=128;e.dispatchEvent(new Event("input"))}')
    assert page.locator('#step').inner_text()=='128 / 128'
    page.locator('#mode').select_option('predictive')
    page.locator('#play').click();page.wait_for_timeout(650);page.locator('#play').click()
    assert page.locator('#step').inner_text()!='128 / 128'
    page.locator('#search').fill('');page.locator('#dataset').select_option('ActivityNet-Captions')
    assert page.locator('#sample option').count()==100
    page.locator('#tokens button').nth(6).click();assert page.locator('#step').inner_text().startswith('7 /')
    # Check the moved sample assets in each benchmark folder.
    for dataset in ['Charades-STA', 'QVHighlights', 'ActivityNet-Captions']:
        page.locator('#dataset').select_option(dataset)
        page.wait_for_function('document.getElementById("staticImg").complete && document.getElementById("staticImg").naturalWidth > 0')
        assert page.locator('#staticImg').get_attribute('src').__contains__('/'+dataset+'/'+dataset+'_')
        for href in page.locator('#links a').evaluate_all('(links) => links.map(a => a.getAttribute("href"))'):
            assert (relocate_path(DYNAMICS/href)).is_file(), href
    page.screenshot(path=str(relocate_path(DYNAMICS / 'viewer_preview.png')),full_page=False)
    assert not errors,errors
    write_result_text((relocate_path(DYNAMICS / 'browser_validation.json')), json.dumps(dict(passed=True,engine='Chromium',
        local_file_mode=True,samples=300,dataset_filter=True,search=True,slider=True,token_selection=True,
        raw_curve_changes=True,predictive_consumed_toggle=True,autoplay=True,truncated_128_token_sample=True,
        page_errors=errors),indent=2)+'\n')
    browser.close()
print('Browser interactions passed')
