"""Verify grouped file identity, links, numeric tables and browser interactions."""
import ast,json,os,re,sys
from pathlib import Path
from urllib.parse import unquote
ROOT=Path('/home/irteam/experiments');P=ROOT/'zero-shot';S=P/'artifacts/maintenance/grouping_20261104'
sys.path.insert(0,str(ROOT/'_lib'))
from result_layout import relocate_path
journal=json.loads((S/'migration.json').read_text());failures=[];verified=0
for path,ino,size in journal['preserved']:
 f=Path(path)
 if not f.exists():failures.append(('missing',path));continue
 if f.stat().st_ino!=ino:failures.append(('inode',path))
 if f.suffix not in {'.md','.html','.js','.css','.py'} and f.name!='media_manifest.json' and str(f) not in journal['changed'] and f.stat().st_size!=size:failures.append(('size',path))
 verified+=1
for f in (P/'_lib').glob('*.py'):
 if not f.is_symlink():ast.parse(f.read_text())
# Resolve local document links; only newly introduced failures block this migration.
links=[];broken=[]
for f in [P/'README.md',P/'artifacts/README.md',*(P/'results').rglob('*.md'),*(P/'results').rglob('*.html')]:
 if 'rollout' in str(f):continue
 text=f.read_text();refs=re.findall(r'\]\(([^)]+)\)',text) if f.suffix=='.md' else re.findall(r'(?:href|src)=[\'\"]([^\'\"]+)',text)
 for ref in refs:
  if ref.startswith(('http:','https:','data:','#','javascript:','/data/')) or '{' in ref or '$' in ref:continue
  target=(f.parent/unquote(ref.split('#')[0].split('?')[0]))
  if not target.exists():broken.append([str(f),ref])
  links.append([str(f),ref])
# Markdown numeric table cells must equal their pre-migration versions.
tables=0
for backup in (S/'source_backup/zero-shot/results').rglob('*.md'):
 current=relocate_path(ROOT/backup.relative_to(S/'source_backup'))
 if not current.exists():continue
 def values(t):return [re.findall(r'[-+]?\d+(?:\.\d+)?',line) for line in t.splitlines() if line.startswith('|')]
 if values(backup.read_text())!=values(current.read_text()):failures.append(('numeric_table',str(current)))
 tables+=1
result={'files_identity_checked':verified,'numeric_reports_checked':tables,'local_links_checked':len(links),'broken_links':broken,'failures':failures}
(S/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='broken_links'},ensure_ascii=False));print('Broken links:',len(broken))
assert not failures,failures[:5]
# Browser: all moved HTML pages must load data and play a local video.
shared=ROOT/'_shared/timelens';sys.path.insert(0,str(shared/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
rows=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
 page=browser.new_page(viewport={'width':1440,'height':1000})
 page.goto('http://127.0.0.1:8780/',wait_until='networkidle')
 page.locator('#project').select_option('zero-shot');assert page.locator('article:visible').count()>10
 page.locator('#search').fill('full9404');assert page.locator('article:visible').count()>0
 page.locator('#search').fill('no-such-experiment');assert page.locator('#empty').is_visible()
 for f in (P/'results').rglob('*.html'):
  if 'rollout' in str(f):continue
  errors=[];bad=[];v=browser.new_page()
  v.on('pageerror',lambda e:errors.append(str(e)))
  v.on('response',lambda r:bad.append([r.status,r.url]) if r.status>=400 else None)
  response=v.goto('http://127.0.0.1:8780/'+str(f.relative_to(ROOT)),wait_until='networkidle',timeout=45000)
  v.wait_for_timeout(1200)
  assert response.status==200
  assert v.locator('[data-experiment-home] a').evaluate('(a)=>a.href')=='http://127.0.0.1:8780/index.html'
  assert v.locator('[data-experiment-scope]').count()==1
  playback=False
  if v.locator('video').count():
   v.locator('video').first.evaluate('(v)=>{v.muted=true;v.play().catch(()=>{})}')
   try:v.wait_for_function("[...document.querySelectorAll('video')].some(v=>v.currentTime>.1 && v.videoWidth>0)",timeout=10000);playback=True
   except Exception:pass
  row={'page':str(f.relative_to(P)),'errors':errors,'failed_requests':bad,'video_playback':playback};rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
  v.close()
 page.set_viewport_size({'width':390,'height':844});page.locator('#search').fill('');assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 browser.close()
result['browser']=rows;result['browser_passed']=all(not r['errors'] and not r['failed_requests'] for r in rows)
(S/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
assert result['browser_passed']
