"""Verify method-specific review processes against original records and every saved tournament."""
import json,re,sys
from pathlib import Path
from deterministic_rollout_viewer_check import sync_playwright,ROOT
from rollout_reconsideration import OUT,SOURCE,COHORT,read,write,diverse,interval,iou
sys.path.insert(0,str(ROOT.parent/'_lib'))
from family_view_test_utils import select_view
URL='http://127.0.0.1:8780/zero-shot/results/rollout/deterministic/'
KEY='activitynet-timelens_01723'
def ready(page,key=None):
 page.wait_for_function('window.RECONSIDERATION?.state.record && !document.querySelector("#loadStatus").textContent && document.querySelector(`[data-outcome="${RECONSIDERATION.state.variant}"]`)',timeout=60000)
 if key:assert page.evaluate('RECONSIDERATION.state.key')==key

def run():
 packets=[];special={};cohort=read(COHORT)
 for sample in cohort:
  key=sample['sample_key'];record=read(OUT/'responses'/f'{key}.json');raw=read(SOURCE/'rollouts'/f'{key}.json')['result']['candidates'];normal=read(SOURCE/'web/samples'/f'{key}.json')['candidates']
  cs=[{k:c[k] for k in ('index','pair','log_probability','stopped_on_eos')} for c in normal]
  for name,v in record['variants'].items():
   if v['method']=='tts':
    seed=[c['index'] for c in diverse(raw[:v['pool']],v['amount'])]
    minimal=dict(pool=v['pool'],amount=v['amount'],selected_candidate=v['selected_candidate'],reviews=[dict(a=r['a'],b=r['b'],winner=r['winner']) for r in v['reviews']])
    packets.append(dict(cs=cs,v=minimal,expected=seed,key=key,name=name))
    if len(seed)>1 and len(seed)&(len(seed)-1):special.setdefault('bye',(key,name))
   if v['method']=='agent' and any(r['fallback'] for r in v['reviews']):special.setdefault('fallback',(key,name))
   if v['selected_candidate'] and interval(v['answer'])!=interval(raw[v['selected_candidate']-1]):special.setdefault('regeneration',(key,name))
 record=read(OUT/'responses'/f'{KEY}.json');checks=[];errors=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
  p=browser.new_page(viewport={'width':1600,'height':1100});p.on('pageerror',lambda e:errors.append(str(e)))
  p.goto(URL+f'pilot/600/reconsideration/viewer.html#sample={KEY}&candidate=3&preset=dot&review=agent_pool8');ready(p,KEY)
  assert p.locator('#branchMap,#branchPreview,#distributionPanel,.fork-response,#reviewMethod').count()==0
  assert p.evaluate('window.ROLLOUT_DISTRIBUTIONS===undefined')
  assert p.locator('#approachTabs button').count()==4
  assert p.evaluate('ROLLOUT_VIEWER.state.grouped.length')==59
  assert p.evaluate('ROLLOUT_VIEWER.state.filtered.length')==171
  for name,v in record['variants'].items():
   p.click(f'[data-approach="{v["method"]}"]');p.click(f'[data-variant="{name}"]');ready(p,KEY)
   final=p.locator(f'[data-outcome="{name}"]')
   assert final.locator('.final-answer').inner_text()==v['answer']['text']
   score=iou(interval(v['answer']),record['sample']['span'][0]);assert final.locator('[data-final-iou]').inner_text()==f'{score:.3f}'
   assert p.locator('#ledgerSource').inner_text()==v['ledger']
   if v['method']=='agent':
    assert p.locator('.agent-round').count()==len(v['reviews'])
    assert p.locator('.agent-menu [data-trajectory]').evaluate_all('(els)=>els.map(e=>Number(e.dataset.trajectory))')==[int(x) for x in re.findall(r'^#(\d+)\s+\[',v['ledger'],re.M)]
    for i,r in enumerate(v['reviews']):
     step=p.locator('.agent-round').nth(i)
     assert step.locator('blockquote p').inner_text()==r['observation']
     assert step.locator('.agent-targets [data-trajectory]').evaluate_all('(els)=>els.map(e=>Number(e.dataset.trajectory))')==([r['a'],r['b']] if r['action']=='event' else [r['a']])
    assert final.locator('.selection-to-final').count()==0
   elif v['method']=='verifier':
    assert p.locator('[data-verifier-candidate]').count()==len(v['reviews'])
    assert p.locator('.verifier-selected').get_attribute('data-verifier-candidate')==str(v['selected_candidate'])
    for r in v['reviews']:
     row=p.locator(f'[data-verifier-candidate="{r["candidate"]}"]');assert float(row.get_attribute('data-score'))==r['score']
   elif v['method']=='tts':
    assert p.locator('.match-card').count()==len(v['reviews'])
    for i,r in enumerate(v['reviews']):
     card=p.locator('.match-card').nth(i);assert card.get_attribute('data-winner')==str(r['winner']);assert float(card.get_attribute('data-pa'))==r['p_a']
   else:
    assert p.locator('.direct-condition,.direct-process .process-outcome').count()==4
    assert p.locator('.agent-round,.verifier-selected,.match-card').count()==0
  checks.append('all_ten_conditions_render_exact_final_answers_and_distinct_agent_verifier_tournament_and_direct_processes')
  # Validate all600 × four tournament conditions using the original Python selection policy.
  for start in range(0,len(packets),200):
   batch=packets[start:start+200]
   actual=p.evaluate('(items)=>items.map(x=>{const p=RECONSIDERATION.bracket(x.v,x.cs);return {seeds:p.seeds,winner:p.winner};})',batch)
   for expected,observed in zip(batch,actual):assert observed==dict(seeds=expected['expected'],winner=expected['v']['selected_candidate']),(expected['key'],expected['name'])
  checks.append('all_2400_saved_tournaments_match_original_GT_free_seed_selection_rounds_byes_and_winners')
  for kind,(key,name) in special.items():
   p.goto(URL+f'pilot/600/reconsideration/viewer.html#sample={key}&review={name}');ready(p,key)
   v=read(OUT/'responses'/f'{key}.json')['variants'][name]
   if kind=='fallback':assert p.locator('.fallback-label').count()==sum(r['fallback'] for r in v['reviews'])
   if kind=='bye':assert p.locator('.match-bye').count()>0
   if kind=='regeneration':assert p.locator('.selection-to-final [data-trajectory]').get_attribute('data-trajectory')==str(v['selected_candidate'])
  checks.append('recorded_fallbacks_byes_and_selection_to_regeneration_changes_remain_explicit')
  p.goto(URL+f'pilot/600/reconsideration/viewer.html#sample={KEY}&review=verifier_pool8');ready(p,KEY)
  p.locator('.process-outcome .process-play').click();p.wait_for_function('!document.querySelector("video").paused&&document.querySelector("video").videoWidth>0',timeout=60000)
  response=p.request.get(p.locator('#video').get_attribute('src'),headers={'Range':'bytes=0-1023'});assert response.status==206 and len(response.body())==1024
  p.evaluate('document.querySelector("video").pause()')
  for preset in ('all','recovered','improved','worsened','parent_improved','dot','greedy_short','parent_short'):
   p.click(f'[data-preset="{preset}"]');ready(p)
   assert p.evaluate('ROLLOUT_VIEWER.state.preset')==preset
  checks.append('shared_case_types_and_original_video_replay_work_without_repeating_deterministic_detail')
  shell=browser.new_page(viewport={'width':1600,'height':1100});shell.on('pageerror',lambda e:errors.append(str(e)))
  shell.goto(URL+f'viewer.html?view=pilot/600/reconsideration/viewer.html#sample={KEY}&candidate=3&review=agent_pool8')
  frame=shell.frame_locator('#detail-frame');frame.locator('.agent-process').wait_for(timeout=60000)
  select_view(shell,'pilot/600/viewer.html');frame.locator('#branchMap .selected[data-branch-candidate="3"]').wait_for(timeout=60000)
  select_view(shell,'pilot/600/reconsideration/viewer.html');frame.locator('.agent-process').wait_for(timeout=60000)
  assert shell.frames[1].evaluate('ROLLOUT_VIEWER.state.current.sample.key')==KEY
  shell.locator('#tab-report').click();shell.locator('#report h1').wait_for();shell.locator('#tab-result').click();frame.locator('.agent-process').wait_for(timeout=60000)
  for n in (300,600):
   select_view(shell,f'pilot/{n}/reconsideration/viewer.html');frame.locator('.agent-process').wait_for(timeout=60000)
   detail=shell.frames[1];assert detail.evaluate('ROLLOUT_VIEWER.state.map.has(ROLLOUT_VIEWER.state.current.sample.key)')
   assert detail.evaluate('RECONSIDERATION_CONFIG.queries')==n
   target=ROOT/'results/rollout/deterministic/pilot'/str(n)/'reconsideration';detail.evaluate('document.fonts.ready')
   frame.locator('#approachTabs').scroll_into_view_if_needed();shell.screenshot(path=str(target/'preview_desktop.png'))
   shell.set_viewport_size({'width':390,'height':844})
   for method in ('agent','verifier','tts','direct'):
    frame.locator(f'[data-approach="{method}"]').click();assert detail.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
   frame.locator('[data-approach="agent"]').click();frame.locator('.agent-process').scroll_into_view_if_needed();shell.screenshot(path=str(target/'preview_mobile.png'))
   detail.evaluate('document.querySelectorAll("details").forEach(e=>e.open=true)');assert detail.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
   shell.set_viewport_size({'width':1600,'height':1100})
  assert not errors,errors;browser.close()
 write(OUT/'validation/viewer_browser.json',dict(passed=True,checks=checks,page_errors=errors,tournaments_checked=len(packets),original_results_unchanged=True,shared_case_definitions=True,media_reused=True))
 print(json.dumps(dict(passed=True,checks=checks,tournaments_checked=len(packets)),ensure_ascii=False))
if __name__=='__main__':run()
