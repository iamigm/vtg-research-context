"""Exercise existing local viewers after result reorganization; overwrite previews."""
import os,sys,json
from pathlib import Path
ROOT=Path('/home/irteam/experiments');R=ROOT/'_shared/timelens'
sys.path.insert(0,str(R/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(R/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(R/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']=str(R/'browser_libs/fonts.conf')
from playwright.sync_api import sync_playwright
D=json.loads((ROOT/'artifacts/maintenance/layout.json').read_text());rows=[]
cases=[(8780,'/zero-shot/results/timelens_top30/report.html',ROOT/'zero-shot/results/timelens_top30'),(8780,'/zero-shot/results/timelens_rollout/report.html',ROOT/'zero-shot/results/timelens_rollout'),(8780,'/zero-shot/results/timelens_top30/integrated/report.html',ROOT/'zero-shot/results/timelens_top30/integrated'),(8780,'/zero-shot/results/timelens_top30/digit_correctness/report.html',ROOT/'zero-shot/results/timelens_top30/digit_correctness'),(8780,'/zero-shot/results/timelens_top30_full/digit_correctness/report.html',ROOT/'zero-shot/results/timelens_top30_full/digit_correctness'),(8780,'/zero-shot/results/input_order/consistency_v1/report.html',ROOT/'zero-shot/results/input_order/consistency_v1'),(8780,'/analysis/failure_mode_timestamp/failure_cases/artifacts/runs/pilot_100/data/cases/report.html',ROOT/'analysis/failure_mode_timestamp/failure_cases/results/pilot_100')]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
 for port,path,root in cases:
  errors=[];failed=[];page=browser.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:failed.append([r.status,r.url]) if r.status>=400 else None)
  row=dict(port=port,url='http://127.0.0.1:'+str(port)+path)
  try:
   page.goto(row['url'],wait_until='networkidle',timeout=30000)
   page.wait_for_function("[...document.querySelectorAll('video')].some(v=>v.readyState>=2&&v.videoWidth>0)",timeout=30000)
   video=page.locator('video').first
   video.evaluate('(v)=>{v.muted=true;v.currentTime=0;v.play()}')
   page.wait_for_function("document.querySelector('video').currentTime>.15",timeout=10000)
   video.evaluate('(v)=>v.pause()')
   row['video_playback']=True
   preview=D['previews'].get(str(root)+'/desktop')
   if preview:page.screenshot(path=preview)
   mobile=D['previews'].get(str(root)+'/mobile')
   if mobile:page.set_viewport_size({'width':390,'height':844});page.screenshot(path=mobile)
   row['passed']=not errors and not failed
  except Exception as e:row.update(passed=False,error=str(e))
  row.update(page_errors=errors,failed_requests=failed);rows.append(row);print(json.dumps(row),flush=True);page.close()
 browser.close()
(ROOT/'artifacts/maintenance/viewer_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
