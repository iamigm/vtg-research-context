"""Compare the compact path display and top-k/top-p against saved full-vocabulary probabilities."""
import gzip,math,json
import numpy as np
from deterministic_rollout_viewer_check import BASE,OUT,REPORT,sync_playwright
from deterministic_rollout300 import read,write
KEY='activitynet-timelens_01723'
def run():
 root=OUT/'token_traces'/KEY;trace=read(root/'index.json');payload=read(OUT/'web/samples'/(KEY+'.json'))
 c=payload['candidates'][2];j=c['branch']['position'];path=trace['candidates']['3']['nodes'];node=trace['nodes'][path[j]]
 values=np.frombuffer(gzip.decompress((root/node['file']).read_bytes()),dtype='<f4');rank=np.lexsort((np.arange(len(values)),-values));cdf=np.cumsum(values[rank],dtype=np.float64)
 assert abs(float(values.sum(dtype=np.float64))-1)<3e-5
 errors=[];checks=[]
 with sync_playwright() as pw:
  b=pw.chromium.launch(headless=True,args=['--no-sandbox']);p=b.new_page(viewport={'width':1600,'height':1100});p.on('pageerror',lambda e:errors.append(str(e)))
  p.goto(BASE+'#preset=all&sample='+KEY+'&candidate=3&panel=distributions')
  p.wait_for_function('window.ROLLOUT_DISTRIBUTIONS?.state.trace && document.querySelectorAll(".distribution-child").length===2',timeout=60000)
  assert p.evaluate('ROLLOUT_DISTRIBUTIONS.state.step')==j
  default_k=1+len({t['id'] for t in node['observed']})
  assert p.locator('#distK').input_value()==str(default_k)
  assert f'색상 토큰 {default_k-1}개' in p.locator('#distKDefault').inner_text()
  assert p.locator('.distribution-root .measured-choice').count()+1==default_k
  assert p.locator('#distributionProduct').count()==0
  assert p.locator('.trace-common .trace-p-row').count()==1 and p.locator('.trace-common .trace-product-row').count()==1
  # Validate every displayed conditional probability and cumulative log probability, including EOS.
  for index in (1,3):
   record=payload['candidates'][index-1];nodes=trace['candidates'][str(index)]['nodes'];running=0.
   for pos,key in enumerate(nodes):
    n=trace['nodes'][key];prob=next(t['probability'] for t in n['observed'] if t['id']==record['token_ids'][pos]);running+=math.log(prob)
    selector='.trace-common' if pos<j else f'[data-probability-path="{index}"]'
    observed=p.locator(selector+f' .trace-p-row [data-step="{pos}"]').get_attribute('data-probability')
    cumulative=p.locator(selector+f' .trace-product-row [data-step="{pos}"]').get_attribute('data-log-probability')
    assert abs(float(observed)-prob)<1e-12 and abs(float(cumulative)-running)<1e-5
   assert abs(running-trace['candidates'][str(index)]['saved_log_probability'])<1e-5
   assert abs(float(p.locator(f'[data-probability-path="{index}"]').get_attribute('data-full-log-probability'))-running)<1e-5
  checks.append('two_horizontal_probability_rows_shared_prefix_and_two_suffixes_match_source_through_EOS')
  # Changing a token position/path keeps the compared pair fixed; changing the pair updates the response.
  p.locator(f'[data-trace-candidate="1"][data-dist-step="{j+1}"]').click()
  p.wait_for_function('(j)=>ROLLOUT_DISTRIBUTIONS.state.focus.index===1 && ROLLOUT_DISTRIBUTIONS.state.step===j',arg=j+1)
  assert p.evaluate('ROLLOUT_VIEWER.state.candidate')==3 and p.locator('#branchSelect').input_value()=='3'
  p.locator('.branch-point').click();p.wait_for_function('document.querySelectorAll(".distribution-child").length===2')
  p.select_option('#branchSelect','4');p.wait_for_function('ROLLOUT_VIEWER.state.candidate===4')
  p.select_option('#branchSelect','3');p.wait_for_function('ROLLOUT_DISTRIBUTIONS.state.candidate.index===3 && document.querySelectorAll(".distribution-child").length===2')
  assert p.locator('#distK').input_value()==str(default_k)
  checks.append('response_pair_and_token_position_selection_distinct_dynamic_top_k_default')
  # Exact top-k above the saved top30 requires the full FP32 vector.
  p.fill('#distK','50');p.locator('#distK').dispatch_event('change')
  p.wait_for_function('document.querySelector(".distribution-root").textContent.includes("top-k 50 → 50개")',timeout=30000)
  actual=p.locator('.distribution-root [data-root-token][data-context-only="false"]').evaluate_all('(rows)=>rows.map(e=>Number(e.dataset.rootToken))')
  assert actual==rank[:50].tolist(),(actual,rank[:50].tolist())
  p.select_option('#distMode','top_p');p.fill('#distP','0.95');p.locator('#distP').dispatch_event('change')
  p.wait_for_function('document.querySelector(".distribution-root").textContent.includes("top-p 0.95")',timeout=30000)
  actual=p.locator('.distribution-root [data-root-token][data-context-only="false"]').evaluate_all('(rows)=>rows.map(e=>Number(e.dataset.rootToken))')
  count=min(len(values),int(np.searchsorted(cdf,.95))+1);assert actual==rank[:count].tolist()
  # p=1 can need many tokens: keep exact membership/mass and page the rendered rows.
  p.fill('#distP','1');p.locator('#distP').dispatch_event('change')
  p.wait_for_function('document.querySelector(".distribution-root").textContent.includes("top-p 1 →")',timeout=30000)
  meta=p.evaluate('async()=>{const s=ROLLOUT_DISTRIBUTIONS.state,n=s.trace.nodes[s.trace.candidates["3"].nodes[s.step]],r=await ROLLOUT_DISTRIBUTIONS.rowsFor(n);return {count:r.selectionCount,mass:r.selectionMass,visible:r.length,more:r.hasMore};}')
  count=min(len(values),int(np.searchsorted(cdf,1.))+1);assert meta['count']==count and abs(meta['mass']-float(cdf[count-1]))<1e-10
  assert meta['visible']<=80+len(node['observed'])
  if meta['more']:
   p.locator('.distribution-root [data-more-prob]').click()
   p.wait_for_function('document.querySelectorAll(".distribution-root [data-root-token]").length>80')
  checks.append('top_k_50_and_top_p_minimal_prefix_match_full_numpy_sort_and_large_sets_are_paged')
  p.fill('#distSearch','6486');p.wait_for_function('document.querySelector(".distribution-root").textContent.includes("검색 일치")',timeout=30000);p.wait_for_selector('.distribution-root [data-root-token="6486"]',timeout=30000)
  assert '검색 일치' in p.locator('.distribution-root').inner_text()
  p.fill('#distSearch','');p.select_option('#distMode','top_k');p.fill('#distK','1');p.locator('#distK').dispatch_event('change')
  p.wait_for_function('document.querySelector(".distribution-root").textContent.includes("top-k 1 → 1개")')
  assert p.locator('.distribution-root [data-context-only="true"]').count()>0
  assert '기준 밖 경로 토큰' in p.locator('.distribution-root').inner_text()
  checks.append('full_vocabulary_search_and_pinned_path_tokens_explicitly_distinguished_from_cutoff')
  p.locator('.branch-point').click();p.wait_for_function('(k)=>document.querySelector(".distribution-root").textContent.includes("top-k "+k+" → "+k+"개")',arg=default_k)
  p.evaluate('document.fonts.ready');p.evaluate('scrollTo(0,0)');p.screenshot(path=str(REPORT/'preview_desktop.png'),full_page=True)
  p.set_viewport_size({'width':390,'height':844});assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  p.locator('#distributionPanel').scroll_into_view_if_needed();p.screenshot(path=str(REPORT/'preview_mobile.png'))
  p.evaluate('document.querySelectorAll("details").forEach(e=>e.open=true)');assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  assert not errors,errors;b.close()
 write(OUT/'validation/distribution_viewer.json',dict(passed=True,sample_key=KEY,checks=checks,page_errors=errors,cumulative_probability_products_match_original=True,top_k_and_top_p_match_full_vocabulary=True,mobile_no_overflow=True))
 print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False))
if __name__=='__main__':run()
