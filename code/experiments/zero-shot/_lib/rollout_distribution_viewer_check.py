"""Validate measured probabilities, two-level fan-out UI and cumulative products."""
import gzip,math,json
import numpy as np
from deterministic_rollout_viewer_check import BASE,OUT,REPORT,sync_playwright
from deterministic_rollout300 import read,write

KEY='activitynet-timelens_01723'
def run():
 root=OUT/'token_traces'/KEY;trace=read(root/'index.json');payload=read(OUT/'web/samples'/(KEY+'.json'))
 c=payload['candidates'][2];position=c['branch']['position'];path=trace['candidates']['3']['nodes']
 node=trace['nodes'][path[position]]
 assert node['prefix_text']=='The event happens in 0.0 - 1'
 values=np.frombuffer(gzip.decompress((root/node['file']).read_bytes()),dtype='<f4')
 assert abs(float(values.sum(dtype=np.float64))-1)<3e-5
 original=next(t for t in node['observed'] if t['token']=='.')
 forced=next(t for t in node['observed'] if t['token']=='0')
 assert abs(original['probability']-forced['probability'])<1e-8
 logp=0.
 for j,key in enumerate(path):
  nd=trace['nodes'][key];prob=next(t['probability'] for t in nd['observed'] if t['id']==c['token_ids'][j])
  assert abs(nd['prefix_log_probability']-logp)<1e-5
  logp+=math.log(prob)
 assert abs(logp-trace['candidates']['3']['saved_log_probability'])<1e-5
 errors=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
  page=browser.new_page(viewport={'width':1500,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(BASE+'#preset=greedy_short&sample='+KEY+'&candidate=3&panel=distributions')
  page.wait_for_function('window.ROLLOUT_DISTRIBUTIONS?.state.trace && document.querySelectorAll(".distribution-child").length===2',timeout=60000)
  assert page.evaluate('ROLLOUT_DISTRIBUTIONS.state.step')==position
  assert page.evaluate('ROLLOUT_DISTRIBUTIONS.state.autoJump')
  dot=page.locator(f'.distribution-root [data-root-token="{original["id"]}"]')
  zero=page.locator(f'.distribution-root [data-root-token="{forced["id"]}"]')
  assert dot.evaluate('(e)=>getComputedStyle(e).borderLeftColor')!=zero.evaluate('(e)=>getComputedStyle(e).borderLeftColor')
  assert '35.05341%' in dot.inner_text() and '35.05341%' in zero.inner_text()
  page.locator('.distribution-root details>summary').click()
  checkbox=page.locator(f'[data-child-token="{original["id"]}"]');checkbox.uncheck()
  page.wait_for_function('document.querySelectorAll(".distribution-child").length===1')
  page.locator(f'[data-child-token="{original["id"]}"]').check()
  page.wait_for_function('document.querySelectorAll(".distribution-child").length===2')
  page.locator('[data-child-token="16"]').check();page.locator('[data-child-token="19"]').check()
  page.wait_for_function('document.querySelectorAll(".distribution-child").length===4')
  page.locator('[data-child-token="16"]').uncheck();page.locator('[data-child-token="19"]').uncheck()
  assert page.locator('#distAdvanced,#distPrevious,#distNext,#distStep,#distBranchPoint').count()==0
  assert page.locator('#distTokenStrip').bounding_box()['y']<page.locator('#distributionTree').bounding_box()['y']
  assert page.locator(f'[data-dist-step="{position}"]').get_attribute('aria-pressed')=='true'
  page.locator(f'[data-dist-step="{position+1}"]').click()
  page.wait_for_function('(step)=>ROLLOUT_DISTRIBUTIONS.state.step===step',arg=position+1)
  page.wait_for_function('document.querySelector(".distribution-root .dist-prefix").textContent.endsWith("10▏")')
  page.locator('#distTokenStrip .branch-point').click()
  page.wait_for_function('document.querySelectorAll(".distribution-child").length===2')
  page.fill('#distSearch','6486')
  page.wait_for_function('ROLLOUT_DISTRIBUTIONS.state.arrays.size>0',timeout=30000)
  page.wait_for_function('document.querySelectorAll(".distribution-root [data-root-token]").length===5',timeout=30000)
  page.wait_for_selector('.distribution-root [data-root-token="6486"]',timeout=30000)
  assert page.evaluate('ROLLOUT_DISTRIBUTIONS.state.arrays.size')>0
  page.fill('#distSearch','')
  page.wait_for_function('document.querySelectorAll(".distribution-root .prob-row[data-root-token]").length>=8')
  page.locator('#distributionProduct summary').click()
  assert page.locator('#distributionProduct tbody tr').count()==len(c['token_ids'])+len(payload['candidates'][0]['token_ids'])
  assert '(EOS)까지' in page.locator('#distributionProduct').inner_text()
  assert 'p₁ × p₂' in page.locator('#distributionProduct').inner_text()
  # Traversing a conditioned child must not change the case under inspection.
  page.locator('.distribution-child .child-more>summary').first.click()
  page.locator('[data-next-node]').first.click()
  page.wait_for_function('(step)=>ROLLOUT_DISTRIBUTIONS.state.step===step',arg=position+1)
  assert page.evaluate('ROLLOUT_VIEWER.state.candidate')==3
  page.locator('#distTokenStrip .branch-point').click()
  page.wait_for_function('document.querySelectorAll(".distribution-child").length===2')
  # Keep only the latest previews.
  page.locator('#distributionProduct summary').click()
  page.evaluate('window.scrollTo(0,0)');page.evaluate('document.fonts.ready')
  page.screenshot(path=str(REPORT/'preview_desktop.png'),full_page=False)
  page.set_viewport_size({'width':390,'height':844})
  assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
  page.locator('#samplePanel').scroll_into_view_if_needed()
  page.screenshot(path=str(REPORT/'preview_mobile.png'),full_page=False)
  browser.close()
 assert not errors,errors
 write(OUT/'validation/distribution_viewer.json',dict(passed=True,sample_key=KEY,conditional_parent_probability=forced['probability'],
  parent_token_colors_distinct=True,four_conditioned_child_distributions=True,child_toggle=True,
  token_strip_above_distributions=True,default_branch_point_selected=True,arbitrary_generation_step=True,full_vocabulary_search=True,full_vector_probability_sum_checked=True,
  cumulative_probability_products_match_original=True,step_product_table=True,mobile_no_overflow=True,page_errors=errors))
 print(json.dumps(dict(passed=True,conditional_parent_probability=forced['probability'],two_level_children=4),ensure_ascii=False))

if __name__=='__main__':run()
