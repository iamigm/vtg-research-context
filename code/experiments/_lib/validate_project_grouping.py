"""Verify preservation, links and browser behavior for the project grouping."""
import ast,hashlib,json,os,re,sys
from pathlib import Path
from urllib.parse import unquote
from result_layout import ROOT,relocate_path
S=ROOT/'artifacts/maintenance/project_grouping_20261104'
PROJECTS=['attention','similarity','dataset_metadata','analysis/failure_mode_timestamp/failure_cases','audio_caption_resources_20260930']

def check_files():
 j=json.loads((S/'migration.json').read_text());changed=set(j['changed']);errors=[];count=0
 for path,(inode,size) in ([] if '--documents' in sys.argv else j['preserved'].items()):
  f=Path(path)
  if not f.exists():errors.append(['missing',path]);continue
  st=f.stat()
  if st.st_ino!=inode:errors.append(['inode',path])
  if path not in changed and f.suffix not in {'.md','.html','.js','.css'} and st.st_size!=size:errors.append(['size',path])
  count+=1
 print('File identity checked:',count,flush=True)
 hashes=0
 for path,sha in ([] if '--documents' in sys.argv else j['numeric_hashes'].items()):
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:errors.append(['numeric_hash',path])
  hashes+=1
 if '--documents' in sys.argv:
  previous=json.loads((S/'preservation_validation.json').read_text());count=previous['files_identity_checked'];hashes=previous['array_metric_sha256_checked']
 print('Metric/array SHA256 checked:',hashes,flush=True)
 broken=[];links=0;numeric_tables=0;code=0
 for project in PROJECTS:
  p=ROOT/project
  for f in [*p.glob('*.py'),*(p/'_lib').glob('*.py')]:
   if not f.is_symlink():ast.parse(f.read_text());code+=1
  for f in [*p.glob('*.md'),p/'artifacts/README.md',*(p/'results').rglob('*.md'),*(p/'results').rglob('*.html')]:
   if not f.exists():continue
   text=f.read_text();refs=re.findall(r'\]\(([^)]+)\)',text) if f.suffix=='.md' else re.findall(r'(?:href|src)=[\'\"]([^\'\"]+)',text)
   for ref in refs:
    if ref.startswith(('http:','https:','data:','#','javascript:','/data/')) or any(c in ref for c in ['{','}','<','>','$']):continue
    ref=ref.split('#')[0].split('?')[0]
    target=f.parent/unquote(ref)
    if not target.exists():broken.append([str(f),ref])
    links+=1
 for f in (S/'source_backup').rglob('*.md'):
  if '/results/' not in str(f):continue
  target=relocate_path(ROOT/f.relative_to(S/'source_backup'))
  def numbers(s):return [re.findall(r'[-+]?\d+(?:\.\d+)?',re.sub(r'\]\([^)]+\)','](path)',line)) for line in s.splitlines() if line.startswith('|')]
  if numbers(f.read_text())!=numbers(target.read_text()):errors.append(['numeric_report_table',str(target)])
  numeric_tables+=1
 result=dict(files_identity_checked=count,array_metric_sha256_checked=hashes,syntax_files=code,numeric_reports_checked=numeric_tables,document_links=links,broken_links=broken,errors=errors)
 (S/'preservation_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='broken_links'},ensure_ascii=False),flush=True);print('Broken links:',len(broken),flush=True)
 assert not errors,errors[:8]
 assert not broken,broken[:8]

def check_browser():
 shared=ROOT/'_shared/timelens';sys.path.insert(0,str(shared/'browser_deps'))
 os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache');os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','');os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
 from playwright.sync_api import sync_playwright
 rows=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
  index=browser.new_page(viewport={'width':1440,'height':1000});index.goto('http://127.0.0.1:8780/',wait_until='networkidle')
  for project in PROJECTS:
   index.locator('#project').select_option(project);assert index.locator('article:visible').count()>0,project
   for f in sorted((ROOT/project/'results').rglob('*.html')):
    errors=[];failed=[];page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:failed.append([r.status,r.url]) if r.status>=400 and r.url.startswith('http://127.0.0.1:8780/') else None)
    row=dict(page=str(f.relative_to(ROOT)))
    try:
     response=page.goto('http://127.0.0.1:8780/'+str(f.relative_to(ROOT)),wait_until='networkidle',timeout=45000);assert response.status==200
     page.wait_for_timeout(1000)
     assert page.locator('[data-experiment-home] a').first.evaluate('(a)=>a.href')=='http://127.0.0.1:8780/index.html'
     assert page.locator('[data-experiment-scope]').count()==1
     row['video_present']=page.locator('video').count()>0
     if row['video_present']:
      page.locator('video').first.evaluate('(v)=>{v.muted=true;v.play().catch(()=>{})}')
      page.wait_for_function("[...document.querySelectorAll('video')].some(v=>v.currentTime>.1&&v.videoWidth>0)",timeout=15000);row['video_playback']=True
     row['passed']=not errors and not failed
    except Exception as e:row.update(passed=False,error=str(e))
    row.update(errors=errors,failed_requests=failed);rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True);page.close()
  index.locator('#project').select_option('');index.set_viewport_size({'width':390,'height':844});assert index.evaluate('document.documentElement.scrollWidth<=innerWidth')
  browser.close()
 (S/'browser_validation.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in rows),pages=rows),ensure_ascii=False,indent=2)+'\n')
 assert all(r['passed'] for r in rows)
if __name__=='__main__':
 if '--browser' in sys.argv:check_browser()
 else:check_files()
