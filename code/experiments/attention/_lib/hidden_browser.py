from result_layout import relocate_path, write_result_text
from timelens_paths import RESOURCES
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Check the completed local report, links, search, and image rendering."""
from pathlib import Path
import json
import os
import sys

DEPS=RESOURCES
sys.path.insert(0,str(relocate_path(DEPS/'browser_deps')))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(relocate_path(DEPS/'browser_cache'))
os.environ['LD_LIBRARY_PATH']=str(relocate_path(DEPS/'browser_libs/usr/lib/x86_64-linux-gnu'))+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(relocate_path(DEPS/'browser_libs/fonts.conf'))
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1400,'height':1100})
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((relocate_path(HIDDEN / 'report.html')).as_uri(),wait_until='load')
    assert page.locator('#samples tr:not(:first-child)').count()==300
    for dataset in ['Charades-STA','QVHighlights','ActivityNet-Captions']:
        page.locator('#search').fill(dataset)
        assert page.locator('#samples tr:not(:first-child):visible').count()==100
    page.locator('#search').fill('QVHighlights_01')
    assert page.locator('#samples tr:not(:first-child):visible').count()==1
    page.locator('#search').fill('')
    links=page.locator('a').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("href"))')
    for link in links:
        assert (relocate_path(HIDDEN/link)).is_file(),link
    assert page.locator('img').evaluate_all('(xs)=>xs.every(x=>x.complete && x.naturalWidth>0)')
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.evaluate('document.fonts.ready')
    assert page.evaluate('document.fonts.check("16px ReportKorean")')
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(relocate_path(HIDDEN / 'report_preview.png')),full_page=False)
    assert not errors,errors
    write_result_text((relocate_path(HIDDEN / 'browser_validation.json')), json.dumps(dict(
        passed=True,samples=300,dataset_search_counts=[100,100,100],single_sample_search=True,
        checked_links=len(links),images_rendered=True,no_horizontal_overflow=True,page_errors=errors),indent=2)+'\n')
    browser.close()
print('Report browser verification passed')
