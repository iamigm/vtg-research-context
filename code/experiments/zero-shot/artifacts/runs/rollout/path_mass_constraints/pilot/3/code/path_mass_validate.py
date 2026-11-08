"""Verify actual integer-only branches, scope, metrics and browser media."""
import sys,math,re
from urllib.request import urlopen,Request
from path_mass_core import *
from path_mass_runtime import numeric_role

def run():
 cohort=read(OUT/'metadata/cohort.json');assert len(cohort)==3 and len({r['dataset'] for r in cohort})==3
 for row in cohort:
  q=read(OUT/'queries'/f'{row["sample_key"]}.json');assert q['complete']
  for v in q['variants'].values():
   for point in v['branch_audit']:assert numeric_role(point['prefix']).endswith('integer_or_decimal') and point['entropy'] is not None
   for s in [v['final'],*v['snapshots'].values()]:assert abs(s['mass_total']-1)<3e-4 and s['fractional_branch_points']==0
   for eps,s in v['snapshots'].items():assert s['unseen_valid_mass_upper_bound']<=float(eps)
 for entry in read(OUT/'metadata/manifest.json')['files']:assert sha(entry['path'])==entry['sha256']
 # Every Markdown table row must have the same number of cells as its header.
 width=None
 for line in (RESULT/'REPORT.md').read_text().splitlines():
  if line.startswith('|'):
   n=line.count('|')
   if width is None:width=n
   assert n==width,(n,width,line)
  else:width=None
 sys.path.insert(0,str(ROOT/'artifacts/runs/rollout/temporal_mode_search/pilot/600/code'))
 from temporal_mode_view_check import sync_playwright
 errors=[];base='http://127.0.0.1:8780/zero-shot/results/rollout/path_mass_constraints/'
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);page=browser.new_page(viewport=dict(width=1600,height=1100));page.on('pageerror',lambda error:errors.append(str(error)));page.goto(base+'pilot/3/detail.html',wait_until='networkidle');page.wait_for_selector('svg');assert page.locator('#sample option').count()==3
  for row in cohort:
   page.select_option('#sample',row['sample_key']);page.wait_for_function('(text)=>document.querySelector("#content h2")?.textContent===text',arg=row['query']);page.wait_for_selector('svg');assert 'GT' in page.locator('#content').inner_text();src=page.locator('video').get_attribute('src');response=page.request.get(src,headers={'Range':'bytes=0-127'});assert response.status==206 and len(response.body())==128
  page.select_option('#checkpoint','0.01');page.wait_for_function('document.querySelector("#content").textContent.includes("도달하지 못")');page.select_option('#checkpoint','0.05');page.wait_for_selector('svg');page.select_option('#variant','format_and_mass');page.wait_for_selector('svg')
  page.screenshot(path=str(RESULT/'preview_desktop.png'),full_page=True);page.set_viewport_size(dict(width=390,height=844));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path=str(RESULT/'preview_mobile.png'),full_page=True)
  page.goto(base+'viewer.html',wait_until='networkidle');page.wait_for_selector('#analysis-tabs button');tabs=page.locator('#analysis-tabs').inner_text();assert '결과' in tabs and '보고서' in tabs;assert not errors,errors;browser.close()
 with urlopen('http://127.0.0.1:8780/') as r:assert r.status==200
 assert not (ROOT/'results/rollout/interval_distribution').exists() and not (ROOT/'artifacts/runs/rollout/interval_distribution').exists()
 write(OUT/'validation/viewer_and_results.json',dict(passed=True,queries=3,actual_fractional_branch_positions=0,actual_format_branch_positions=0,mass_conservation=True,targets_not_mislabelled=True,all_media_range_requests=True,checkpoint_switch=True,common_result_report_tabs=True,mobile_no_overflow=True,markdown_table_widths=True,page_errors=errors,timestamp=stamp()));print('RESULT_AND_VIEWER_VALIDATED')

if __name__=='__main__':run()
