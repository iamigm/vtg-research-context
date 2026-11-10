"""Verify lossless report partition, regeneration, appendix navigation and media."""
import hashlib,json,os,sys
from pathlib import Path
from urllib.parse import quote
from report_appendices import ROOT,HEADER,PART,read_source,split_report
from build_index import PROJECTS

records=[]
for project in PROJECTS:
 for path in sorted((ROOT/project/'results').rglob('*.md')):
  if not HEADER.search(path.read_text()):continue
  source=read_source(path);main=path.read_text();appendix=path.with_name('APPENDIX.md');detail=appendix.read_text()
  # Repeat the final generation stage, including reconstructing existing sections.
  split_report(path)
  assert path.read_text()==main and appendix.read_text()==detail,path
  assert read_source(path)==source,path
  records.append(dict(path=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(source.encode()).hexdigest(),source_lines=len(source.splitlines()),main_lines=len(PART.sub(lambda m:m[2],main).splitlines())))
assert records
catalog=json.loads((ROOT/'artifacts/maintenance/family_catalog.json').read_text())['families']
lookup={}
for family in catalog:
 manifest=json.loads((ROOT/family['manifest']).read_text())
 def visit(node):
  if node.get('kind')=='analysis':
   for entry in node['entries']:lookup[str(Path(family['href']).parent/entry)]=(family,node,manifest)
  for child in node.get('children',[]):visit(child)
 visit(manifest['navigation'])
for record in records:
 family,node,manifest=lookup[record['path']]
 relative=Path(record['path']).relative_to(Path(family['href']).parent).as_posix()
 appendix=str(Path(relative).with_name('APPENDIX.md'))
 assert node['report']==relative and appendix in node['documents'],record['path']
 assert not any(child['path'][-1:] == ['appendix'] for child in node['children'])

shared=ROOT/'_shared/timelens';sys.path.insert(0,str(shared/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:8780/'
errors=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
 page=browser.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)))
 assert page.request.get(BASE).status==200
 for record in records:
  family,node,manifest=lookup[record['path']];report=node['report'];appendix=str(Path(report).with_name('APPENDIX.md'))
  page.goto(BASE+family['href']+'?view='+quote(report,safe=''))
  page.locator('#family-app[data-ready=true]').wait_for();page.locator('#report h1').wait_for()
  assert page.locator('#reference-menu>summary').inner_text()=='appendix'
  page.locator('#reference-menu>summary').click()
  page.locator('#reference-list a[data-view='+json.dumps(appendix)+']').click()
  page.wait_for_function("document.querySelector('#report h1')?.textContent.startsWith('Appendix')")
  assert page.locator('#reference-menu>summary').inner_text()=='appendix'
  assert page.locator('#source-md').get_attribute('href').endswith('/APPENDIX.md')
  assert page.locator('#tab-report').get_attribute('data-view')==report
  assert page.locator('#tab-report').get_attribute('aria-selected')=='true'
  if 'blue_analysis' in appendix:assert page.locator('#report table').count()>=3
  page.reload();page.locator('#family-app[data-ready=true]').wait_for()
  page.wait_for_function("document.querySelector('#report h1')?.textContent.startsWith('Appendix')")
  page.locator('#tab-report').click()
  page.wait_for_function("document.querySelector('#report h1')&&!document.querySelector('#report h1').textContent.startsWith('Appendix')")
  assert page.locator('#source-md').get_attribute('href').endswith('/'+report)
 # Existing supplementary files with nonstandard names are reachable, too.
 page.goto(BASE+'zero-shot/results/sampling/spatial_allocation/viewer.html?view=allocation_sweep%2Fpilot%2F900%2FREPORT.md')
 page.locator('#family-app[data-ready=true]').wait_for();page.locator('#report h1').wait_for()
 page.locator('#reference-menu>summary').click()
 assert page.locator('#reference-list a[data-view="allocation_sweep/pilot/900/A_low.md"]').is_visible()
 # Mobile menu selection and returning to the core report.
 page.set_viewport_size({'width':390,'height':844})
 page.goto(BASE+'zero-shot/results/rollout/deterministic/viewer.html?view=pilot%2F900%2FREPORT.md')
 page.locator('#family-app[data-ready=true]').wait_for();page.locator('#report h1').wait_for()
 page.locator('#reference-menu>summary').click();page.locator('#reference-list a[data-view="pilot/900/APPENDIX.md"]').click()
 page.wait_for_function("document.querySelector('#report h1')?.textContent.startsWith('Appendix')")
 assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
 page.locator('#tab-report').click()
 page.wait_for_function("document.querySelector('#report h1')&&!document.querySelector('#report h1').textContent.startsWith('Appendix')")
 # Result tab and original video remain usable after appendix navigation.
 page.locator('#tab-result').click();frame=page.frame_locator('#detail-frame');video=frame.locator('video').first
 frame.locator('#videoDetails>summary').click()
 video.wait_for();video.evaluate('(v)=>{v.muted=true;v.play().catch(()=>{});}');page.wait_for_timeout(1200)
 assert video.evaluate('(v)=>v.readyState>=2 && v.currentTime>0')
 assert page.locator('#reference-menu').is_hidden()
 assert not errors,errors
 browser.close()
result=dict(passed=True,reports=len(records),original_lines=sum(r['source_lines'] for r in records),core_lines=sum(r['main_lines'] for r in records),lossless_partition=True,idempotent_generation=True,appendix_navigation=True,core_report_return=True,scope_and_analysis_context=True,deep_links_and_reload=True,existing_detail_documents=True,raw_html_tables=True,mobile=True,video_playback=True,records=records)
(ROOT/'artifacts/maintenance/report_appendices_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
