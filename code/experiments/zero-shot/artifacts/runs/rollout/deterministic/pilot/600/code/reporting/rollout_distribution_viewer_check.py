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
  default_k=5
  assert p.locator('#distK').input_value()==str(default_k)
  assert '기본 k = 5' in p.locator('#distKDefault').inner_text()
  assert p.locator('.distribution-root .measured-choice').count()==len(node['observed'])
  assert p.locator('#branchMap [data-branch-candidate]').count()==len(payload['candidates'])
  assert p.locator('#branchMap .selected').get_attribute('data-branch-candidate')=='3'
  assert f'{j+1}번째' in p.locator('#branchPreview').inner_text()
  assert p.locator('#branchPreview pre').inner_text().startswith(c['tokens'][:j] and ''.join(c['tokens'][:j]) or '')
  assert p.locator('.child-more,.next-choice').count()==0
  for child in p.locator('.distribution-child').all():
   index=int(child.get_attribute('data-path-candidate'));answer=payload['candidates'][index-1]
   assert child.locator('.path-prefix').last.inner_text()==''.join(answer['tokens'][:j])
   assert child.locator('mark.path-picked').inner_text()==answer['tokens'][j]
   assert child.locator('.path-suffix').last.inner_text()==''.join(t for t in answer['tokens'][j+1:] if not t.startswith('<|'))
   assert child.locator('[data-next-node]').is_visible()
  checks.append('visual_branch_map_and_inline_prefix_selected_token_completion_match_saved_response')
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
  # Advance twice from a visible action; keep the response pair and make location reversible.
  p.locator('.distribution-child[data-path-candidate="3"] [data-next-node]').click()
  p.wait_for_function('(j)=>ROLLOUT_DISTRIBUTIONS.state.step===j && document.querySelector(".distribution-root h4").textContent==="선택한 위치의 분포"',arg=j+1)
  assert p.evaluate('ROLLOUT_VIEWER.state.candidate')==3
  assert f'{j+2} /' in p.locator('#distributionNavigator').inner_text()
  assert '분기 위치에서 +1 token' in p.locator('#distributionNavigator').inner_text()
  assert p.locator('#distributionNavigator [aria-current="step"]').get_attribute('data-position-step')==str(j+1)
  p.locator('.distribution-child[data-path-candidate="3"] [data-next-node]').click()
  p.wait_for_function('(j)=>ROLLOUT_DISTRIBUTIONS.state.step===j && document.querySelector(".distribution-child [data-next-node]")?.textContent.includes((j+2)+"번째")',arg=j+2)
  p.locator('[data-position="previous"]').click()
  p.wait_for_function('(j)=>ROLLOUT_DISTRIBUTIONS.state.step===j',arg=j+1)
  p.locator('[data-position="branch"]').click()
  p.wait_for_function('document.querySelectorAll(".distribution-child").length===2')
  assert p.evaluate('ROLLOUT_DISTRIBUTIONS.state.step')==j
  # Map navigation and its ancestry match a candidate whose direct parent is another branch.
  nested=next(row for row in payload['candidates'] if row.get('branch') and row['branch']['parent']!=1)
  p.locator(f'#branchMap [data-branch-candidate="{nested["index"]}"]').click()
  p.wait_for_function('(i)=>ROLLOUT_DISTRIBUTIONS.state.candidate.index===i && document.querySelectorAll(".distribution-child").length===2',arg=nested['index'])
  assert p.locator('#branchAncestry [data-branch-candidate]').count()>=3
  assert f'부모 #{nested["branch"]["parent"]}' in p.locator('#branchMap .selected').inner_text()
  p.select_option('#branchSelect','3')
  p.wait_for_function('ROLLOUT_DISTRIBUTIONS.state.candidate.index===3 && document.querySelectorAll(".distribution-child").length===2')
  checks.append('fixed_top_k_5_and_visible_step_navigation_preserve_pair_and_show_nested_ancestry')
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
  p.locator('.branch-point').click();p.wait_for_function('document.querySelector(".distribution-root").textContent.includes("top-k 1 → 1개")')
  assert p.locator('#distK').input_value()=='1'
  p.select_option('#branchSelect','4');p.wait_for_function('ROLLOUT_DISTRIBUTIONS.state.candidate.index===4 && document.querySelector(".distribution-root").textContent.includes("top-k 1 → 1개")')
  p.select_option('#branchSelect','3');p.wait_for_function('ROLLOUT_DISTRIBUTIONS.state.candidate.index===3 && document.querySelectorAll(".distribution-child").length===2')
  assert p.locator('#distK').input_value()=='1'
  checks.append('manual_top_k_survives_position_and_candidate_changes')
  p.fill('#distK','5');p.locator('#distK').dispatch_event('change')
  p.wait_for_function('document.querySelector(".distribution-root").textContent.includes("top-k 5 → 5개")')
  end=len(c['token_ids'])-1
  p.locator(f'[data-trace-candidate="3"][data-dist-step="{end}"]').click()
  p.wait_for_function('(j)=>ROLLOUT_DISTRIBUTIONS.state.step===j && document.querySelector(".distribution-child")?.textContent.includes("응답이 끝났습니다")',arg=end)
  assert p.locator('.distribution-child [data-next-node]').count()==0
  p.locator('[data-position="branch"]').click();p.wait_for_function('document.querySelectorAll(".distribution-child").length===2')
  checks.append('EOS_path_ends_without_forward_action_and_can_return_to_branch')
  p.evaluate('document.fonts.ready');assert p.evaluate('[...document.fonts].every(f=>f.status==="loaded")')
  p.locator('#distributionNavigator').scroll_into_view_if_needed();p.screenshot(path=str(REPORT/'preview_desktop.png'))
  p.set_viewport_size({'width':390,'height':844});assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  p.locator('.distribution-child').first.scroll_into_view_if_needed();p.screenshot(path=str(REPORT/'preview_mobile.png'))
  assert 0<=p.locator('#distributionNavigator').bounding_box()['y']<=10
  p.locator('.distribution-child[data-path-candidate="3"] [data-next-node]').click()
  p.wait_for_function('(j)=>ROLLOUT_DISTRIBUTIONS.state.step===j && document.querySelector(".distribution-child [data-next-node]")?.textContent.includes((j+2)+"번째")',arg=j+1)
  assert '분기 위치에서 +1 token' in p.locator('#distributionNavigator').inner_text()
  assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  p.screenshot(path=str(REPORT/'preview_mobile.png'))
  p.evaluate('document.querySelectorAll("details").forEach(e=>e.open=true)');assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  # All fixed cohorts use the updated viewer through the family shell.
  shell=b.new_page(viewport={'width':1600,'height':1100});shell.on('pageerror',lambda e:errors.append(str(e)))
  shell.goto(BASE.replace('/pilot/900/viewer.html','/viewer.html'))
  shell.wait_for_function('document.querySelector("#family-app").dataset.ready==="true"')
  for size in (300,600,900):
   shell.select_option('#scope',f'pilot/{size}')
   frame=shell.frame_locator('#detail-frame')
   frame.locator('#branchMap .selected').wait_for(timeout=60000)
   shell.wait_for_function('(size)=>{const w=document.querySelector("#detail-frame").contentWindow;return w.ROLLOUT_VIEWER?.state.manifest?.summary.samples===size && w.ROLLOUT_DISTRIBUTIONS?.state.trace}',arg=size,timeout=60000)
   assert frame.locator('#distK').input_value()=='5'
   assert frame.locator('.branch-map-choice').count()>1
   if size!=900:
    target=REPORT.parent/str(size);detail=shell.frames[1]
    detail.evaluate('document.fonts.ready');assert detail.evaluate('[...document.fonts].every(f=>f.status==="loaded")')
    frame.locator('#branchMap').scroll_into_view_if_needed();shell.screenshot(path=str(target/'preview_desktop.png'))
    shell.set_viewport_size({'width':390,'height':844});frame.locator('.distribution-child').first.scroll_into_view_if_needed();shell.screenshot(path=str(target/'preview_mobile.png'))
    assert detail.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    shell.set_viewport_size({'width':1600,'height':1100})
  checks.append('family_scope_switching_pilot300_600_900_uses_visual_map_top_k_5_and_updated_previews')
  assert not errors,errors;b.close()
 write(OUT/'validation/distribution_viewer.json',dict(passed=True,sample_key=KEY,checks=checks,page_errors=errors,cumulative_probability_products_match_original=True,top_k_and_top_p_match_full_vocabulary=True,mobile_no_overflow=True))
 print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False))
if __name__=='__main__':run()
