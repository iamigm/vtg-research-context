"""Validate original sources, gradient causality, reports and real browser views."""
from __future__ import annotations
import argparse, collections, gzip, json, math, os, sys
from pathlib import Path
import numpy as np
from greedy_attribution_core import BASE, COHORT, CONFIG, GROUPS, RESULTS, ROOT, SOURCE, digest, read, sha, stamp, storage, write

COHORT_HASHES = {
    300:'b4232e420a9f9f1a5a4f07fc8f84dde2f372be29935f6dc1dc152a3a9e3d0fdf',
    600:'d7fb4b7cc1c28842be0fa81e9b277c688ccfddf862371a3253a1000db25e4752',
    900:'e92620e9b56da0ca62af58b2da07ab2fb3fa5bac58ab20aa880b709426124db7',
}

class ValidationError(ValueError): pass

def require(condition,message):
    if not condition: raise ValidationError(message)

def close(actual,expected,label,atol=1e-6,rtol=2e-5):
    require(math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),abs_tol=atol,rel_tol=rtol),
            f'{label}: {actual} != {expected}')

def finite(value,label='record'):
    if isinstance(value,float): require(math.isfinite(value),f'{label}: nonfinite number')
    elif isinstance(value,dict):
        for key,item in value.items(): finite(item,f'{label}.{key}')
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value): finite(item,f'{label}[{i}]')

def array(value,shape,label,nonnegative=False):
    data=np.asarray(value)
    require(data.shape==tuple(shape),f'{label}: shape {data.shape}, expected {tuple(shape)}')
    require(data.dtype.kind in 'iuf' and bool(np.isfinite(data).all()),f'{label}: nonnumeric/nonfinite data')
    if nonnegative: require(bool((data>=0).all()),f'{label}: negative norm')
    return data

def within(root,relative):
    path=(root/relative).resolve()
    require(path.is_relative_to(root.resolve()),f'Artifact path escapes store: {relative}')
    return path

def validate_context(context,original,ids):
    prompt=len(original['input_ids']); expected=original['input_ids']+ids[:-1]; length=len(expected)
    require(context['ids']==expected,'Context differs from original prompt plus greedy prefix')
    groups=context['groups']
    require(len(context['tokens'])==length and len(groups)==length,'Context array length mismatch')
    require(''.join(context['tokens'][:prompt])==original['expanded_prompt'],'Context labels do not reconstruct the saved prompt')
    require(all(g in GROUPS for g in groups),'Unknown context group')
    require(all(g!='generated' for g in groups[:prompt]) and all(g=='generated' for g in groups[prompt:]),'Generated group overlap/gap')
    visual=[i for i,g in enumerate(groups) if g=='visual']
    require(visual==context['visual_positions']==[i for i,x in enumerate(expected) if x==151656],'Visual group coverage mismatch')
    t,h,w=original['grid']; require(h%2==w%2==0 and len(visual)==t*h*w//4,'Visual grid mismatch')
    mapping=context['visual_map']; require([m['position'] for m in mapping]==visual,'Visual map order mismatch')
    for j,entry in enumerate(mapping):
        frame,spatial=divmod(j,(h//2)*(w//2))
        require((entry['frame_pair'],entry['row'],entry['col'])==(frame,spatial//(w//2),spatial%(w//2)),'Visual coordinates mismatch')
        times=original['plan']['source_pts'][2*frame:2*frame+2]
        close(entry['timestamp'],sum(times)/len(times),'Visual timestamp')
    return prompt,length,groups

def validate_target(target,payload,prompt,length,groups,grid,count):
    ident=target['id']; step=target['step']; require(isinstance(step,int) and 0<=step<count,f'{ident}: invalid step')
    for key in ('id','kind','step','alternative','score','groups','data_file'):
        require(payload[key]==target[key],f'{ident}: payload metadata differs: {key}')
    require(math.isfinite(target['score']),f'{ident}: nonfinite objective')
    norms=array(payload['context']['grad_norm'],(length,),f'{ident}.context.norm',True)
    dots=array(payload['context']['activation_dot_gradient'],(length,),f'{ident}.context.signed')
    end=prompt+step; future=float(np.max(norms[end:])) if end<length else 0.
    require(future<1e-6,f'{ident}: future gradient leakage {future}')
    close(target['causality_max_future_gradient'],future,f'{ident}.future',atol=1e-8)
    require(not len(dots[end:]) or float(np.max(np.abs(dots[end:])))<1e-5,f'{ident}: future signed attribution leakage')
    embedding=payload['embedding']; require(embedding is not None,f'{ident}: text embedding gradients missing')
    for name in ('gradient_norm_l2','activation_dot_gradient'):
        values=array(embedding[name],(length,),f'{ident}.embedding.{name}',name=='gradient_norm_l2')
        require(not len(values[end:]) or float(np.max(np.abs(values[end:])))<1e-5,f'{ident}: future embedding gradient')
    rows=target['groups']; require(len(rows)==len(GROUPS) and {r['group'] for r in rows}==set(GROUPS),f'{ident}: incomplete group partition')
    for row in rows:
        positions=np.asarray([i for i,g in enumerate(groups[:end]) if g==row['group']],dtype=np.int64)
        require(row['count']==len(positions),f'{ident}: prior group count mismatch')
        close(row['signed_sum'],float(np.sum(dots[positions],dtype=np.float64)),f'{ident}.signed_sum')
        close(row['absolute_sum'],float(np.sum(np.abs(dots[positions]),dtype=np.float64)),f'{ident}.absolute_sum')
        close(row['grad_l2'],float(np.linalg.norm(norms[positions].astype(np.float64))),f'{ident}.grad_l2')
    t,h,w=grid
    for field,shape in [('frames',(t,h//2,w//2)),('pixel_frames',(2*t,h,w))]:
        for metric in ('gradient_norm','activation_dot_gradient'):
            array(payload[field][metric],shape,f'{ident}.{field}.{metric}',metric=='gradient_norm')
    visual=np.asarray([i for i,g in enumerate(groups) if g=='visual'],dtype=np.int64)
    require(np.array_equal(np.asarray(payload['frames']['gradient_norm']).ravel(),norms[visual]),f'{ident}: visual map norm differs')
    require(np.array_equal(np.asarray(payload['frames']['activation_dot_gradient']).ravel(),dots[visual]),f'{ident}: visual map sign differs')
    pixel_norm=np.asarray(payload['pixel_frames']['gradient_norm'],dtype=np.float64)
    pixel_dot=np.asarray(payload['pixel_frames']['activation_dot_gradient'],dtype=np.float64)
    close(target['pixel_gradient_l2'],float(np.linalg.norm(pixel_norm)),f'{ident}.pixel_l2',rtol=2e-4)
    close(target['pixel_signed_sum'],float(pixel_dot.sum()),f'{ident}.pixel_signed_sum',atol=2e-5,rtol=2e-4)
    require(payload['predictor_positions']==list(range(prompt-1,length)),f'{ident}: predictor position shift')
    layers=payload['layers']; require(set(layers)=={'attention','mlp','residual_input'},f'{ident}: decoder components missing')
    for kind,entries in layers.items():
        require([row['layer'] for row in entries]==list(range(36)),f'{ident}: incomplete {kind} layers')
        for row in entries:
            layer_signed=array(row['activation_dot_gradient'],(count,),f'{ident}.{kind}.signed')
            layer_norm=array(row['gradient_norm_l2'],(count,),f'{ident}.{kind}.norm',True)
            require(not len(layer_norm[step+1:]) or float(np.max(layer_norm[step+1:]))<1e-6,f'{ident}: future decoder gradient')
            require(not len(layer_signed[step+1:]) or float(np.max(np.abs(layer_signed[step+1:])))<1e-5,f'{ident}: future decoder signed signal')
    message_data=payload['messages']; finite(message_data,f'{ident}.messages')
    require(target['message_validation']['passed'] is True and payload['message_validation']==target['message_validation'],f'{ident}: message gate missing or failed')
    selected=message_data['selected_layers']; require(selected==[0,17,35],f'{ident}: message layer coverage differs')
    recon={r['layer']:r for r in message_data['reconstruction']}; measured={r['layer']:r for r in message_data['layers']}
    require(set(recon)==set(measured)==set(selected),f'{ident}: reconstructed/message layer coverage differs')
    for layer in selected:
        require(recon[layer]['rms_relative_difference']<=.03,f'{ident}: reconstructed attention RMS error')
        heads=array(measured[layer]['head_score_sum'],(32,count),f'{ident}.message_head_scores')
        summed=heads.sum(axis=0); block=np.asarray(layers['attention'][layer]['activation_dot_gradient'])
        delta=np.abs(summed-block); tolerance=1e-4+.05*np.maximum(np.abs(summed),np.abs(block))
        require(bool((delta<=tolerance).all()),f'{ident}: message sum does not match attention scaling derivative')
        for edge in measured[layer]['edges']:
            require(0<=edge['source_position']<=edge['receiver_position']<length,f'{ident}: noncausal message edge')
            require(edge['receiver_position'] in payload['predictor_positions'],f'{ident}: unexpected message receiver')
            require(0<=edge['head']<32 and 0<=edge['attention_weight']<=1.0001,f'{ident}: invalid message head/weight')
    return future

def validate_record(record,original,saved_input,trace,canonical,reference):
    key=original['sample']['sample_key']; finite(record,key)
    require(record['complete'] is True and record['sample_key']==key and record['sample']==original['sample'],f'{key}: sample/incomplete mismatch')
    require(record['config_hash']==digest(CONFIG),f'{key}: config hash differs')
    source=record['source']; require(source['record_sha256']==reference['sha256'],f'{key}: source checksum differs')
    require(Path(source['record']).resolve()==(SOURCE/'rollouts'/f'{key}.json').resolve(),f'{key}: source record path differs')
    trace_path=SOURCE/'token_traces'/key/'index.json'
    require(Path(source['trace']).resolve()==trace_path.resolve() and source['trace_sha256']==sha(trace_path),f'{key}: source trace checksum differs')
    require(trace['raw_record_sha256']==reference['sha256'],f'{key}: trace source mismatch')
    greedy=original['result']['candidates'][0]; ids=greedy['ids']; count=len(ids)
    require(greedy['index']==1 and greedy['branch'] is None and record['greedy']['ids']==ids,f'{key}: greedy tokens changed')
    require(len(record['greedy']['tokens'])==count and record['greedy']['text']==greedy['answer'],f'{key}: greedy text changed')
    prompt,length,groups=validate_context(record['context'],saved_input,ids)
    inp=record['input']; require(inp['prompt_length']==prompt and inp['total_length']==length,f'{key}: length mismatch')
    require(inp['hashes']==saved_input['hashes']==original['input_hashes']==trace['input_hashes'],f'{key}: input hash mismatch')
    require(inp['grid']==saved_input['grid'] and inp['hw']==saved_input['hw'],f'{key}: input geometry changed')
    require(inp['timestamps']==saved_input['plan']['source_pts'],f'{key}: timestamp changed')
    require([s['step'] for s in record['steps']]==list(range(count)),f'{key}: greedy step coverage mismatch')
    require([s['chosen']['id'] for s in record['steps']]==ids,f'{key}: chosen IDs mismatch')
    validation=record['validation']; require(validation['input_hashes_match'] is True and validation['causality_passed'] is True,f'{key}: replay/causality failed')
    require(validation['gradient_forward_max_logit_error']==0,f'{key}: gradient forward changed logits')
    require(validation['replay_max_probability_error']<=CONFIG['replay_probability_tolerance'],f'{key}: probability tolerance exceeded')
    arrays=record['arrays']; array_path=within(canonical,arrays['file']); require(sha(array_path)==arrays['sha256'],f'{key}: array checksum differs')
    with np.load(array_path,allow_pickle=False) as saved:
        tracked=CONFIG['candidate_top_k']; require(arrays['tracked_tokens']==tracked==30,f'{key}: top30 scope differs')
        token_ids=array(saved['token_ids'],(count,tracked),f'{key}.token_ids')
        expected_ids=np.asarray([[r['id'] for r in trace['nodes'][node]['top'][:tracked]] for node in trace['candidates']['1']['nodes']])
        require(np.array_equal(token_ids,expected_ids),f'{key}: frozen original top30 IDs changed')
        logits=array(saved['logits'],(count,tracked),f'{key}.logits')
        prob=array(saved['probabilities'],(count,tracked),f'{key}.probabilities',True)
        source_prob=array(saved['source_probabilities'],(count,tracked),f'{key}.source_probabilities',True)
        expected_prob=np.asarray([[r['probability'] for r in trace['nodes'][node]['top'][:tracked]] for node in trace['candidates']['1']['nodes']],dtype=np.float32)
        require(np.array_equal(source_prob,expected_prob),f'{key}: source top30 probabilities changed')
        close(validation['replay_max_probability_error'],float(np.max(np.abs(prob-source_prob))),f'{key}.replay_max')
        require(bool((prob<=1).all()) and bool((prob.sum(-1,dtype=np.float64)<=1.00003).all()),f'{key}: invalid unnormalized top30 probability')
        argmax_index=logits.argmax(-1); argmax=token_ids[np.arange(count),argmax_index]
        chosen_indices=np.asarray([list(token_ids[j]).index(token) for j,token in enumerate(ids)])
        mismatch=np.flatnonzero(argmax!=np.asarray(ids)).tolist()
        require(validation['greedy_argmax_match']==(not mismatch) and validation['greedy_near_tie_steps']==mismatch,f'{key}: replay top1 audit inconsistent')
        require(all(logits[j,argmax_index[j]]-logits[j,chosen_indices[j]]<=.125 for j in mismatch),f'{key}: substantive greedy top1 mismatch')
        directions=saved['directions'].tolist(); require(directions==arrays['directions'],f'{key}: screening direction mismatch')
        for field,shape in [('slopes',(len(directions),count,tracked)),('changes',(len(directions),2,count,tracked)),('screen_gaps',(count,tracked))]:
            data=array(saved[field],shape,f'{key}.{field}'); del data
        for j,step in enumerate(record['steps']):
            close(step['chosen']['probability'],prob[j,chosen_indices[j]],f'{key}.chosen_probability')
            close(step['top30_mass'],float(prob[j].sum(dtype=np.float64)),f'{key}.top30_mass')
            close(step['omitted_mass'],max(0.,1-float(prob[j].sum(dtype=np.float64))),f'{key}.omitted_mass')
            require(step['tracked_tokens']==30 and {r['id'] for r in step['top']}==set(token_ids[j]),f'{key}: top30 step coverage differs')
            node=trace['nodes'][trace['candidates']['1']['nodes'][j]]
            require(node['prefix_ids']==ids[:j],f'{key}: source prefix mismatch')
            expected=next(r['probability'] for r in node['observed'] if r['id']==ids[j])
            require(abs(step['chosen']['probability']-expected)<=CONFIG['replay_probability_tolerance'],f'{key}: chosen replay error')
            alternatives=step['candidates']; require(len({a['id'] for a in alternatives})==len(alternatives),f'{key}: repeated alternative')
            for alternative in alternatives:
                alt=alternative['id']; require(alt!=ids[j] and alt in token_ids[j],f'{key}: alternative outside original top30')
                column=list(token_ids[j]).index(alt)
                close(alternative['probability'],prob[j,column],f'{key}.alternative_probability')
                close(alternative['margin'],logits[j,chosen_indices[j]]-logits[j,column],f'{key}.alternative_margin')
        del logits,prob
    targets=record['targets']; require(len({t['id'] for t in targets})==len(targets),f'{key}: duplicate target')
    chosen={t['step']:t for t in targets if t['kind']=='chosen_log_probability'}
    require(len(chosen)==count and sum(t['kind']=='chosen_log_probability' for t in targets)==count and sorted(chosen)==list(range(count)),f'{key}: chosen target coverage including EOS differs')
    require(sum(t['kind']=='logit_margin' for t in targets)<=CONFIG['detailed_branch_targets'],f'{key}: detailed margin budget exceeded')
    future=0.
    for target in targets:
        kind=target['kind']; require(kind in {'chosen_log_probability','start_log_probability','end_log_probability','logit_margin'},f'{key}: unknown objective')
        if kind=='logit_margin':
            branch=next((a for a in record['steps'][target['step']]['candidates'] if a['id']==target['alternative']),None)
            require(branch is not None,f'{key}: margin branch missing'); close(target['score'],branch['margin'],f'{key}.margin')
        elif kind!='chosen_log_probability':
            selected=target['steps']; require(selected and selected==sorted(set(selected)) and all(i in chosen for i in selected),f'{key}: endpoint steps invalid')
            require(target['step']==max(selected),f'{key}: endpoint causal boundary mismatch')
            close(target['score'],math.fsum(chosen[i]['score'] for i in selected),f'{key}.endpoint',atol=2e-4)
        data_path=within(canonical,target['data_file']); require(sha(data_path)==target['sha256'] and data_path.stat().st_size==target['bytes'],f'{key}: target checksum differs')
        with gzip.open(data_path,'rt') as stream: payload=json.load(stream)
        future=max(future,validate_target(target,payload,prompt,length,groups,inp['grid'],count)); del payload
    if greedy.get('parsed'):
        require(all(sum(t['kind']==kind for t in targets)==1 for kind in ('start_log_probability','end_log_probability')),f'{key}: endpoint aggregate missing')
    return dict(sample_key=key,steps=count,targets=len(targets),future_max=future)

def validate_scope(scope,allow_partial=False):
    require(scope in CONFIG['stages'],'Unsupported scope'); canonical=storage(); cohort=read(COHORT)
    require(read(canonical/'metadata/config.json')==CONFIG,'Frozen config differs')
    require(read(canonical/'metadata/cohort.json')==cohort[:int(canonical.name)],'Active cohort changed')
    for n,expected in COHORT_HASHES.items():
        path=ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json'
        require(sha(path)==expected and read(path)==cohort[:n],f'Frozen cohort{n} hash/prefix changed')
    require(read(SOURCE/'metadata/cohort.json')==cohort,'Original source cohort differs')
    refs=read(canonical/'metadata/source_manifest.json'); require(isinstance(refs,list) and len(refs)==900,'Source reference coverage differs')
    original_refs=read(SOURCE/'metadata/source_records.json')
    require([(r['path'],r['sha256']) for r in refs]==[(r['path'],r['sha256']) for r in original_refs],'Frozen source references differ')
    provenance=read(canonical/'metadata/provenance.json')
    require(provenance['config_hash']==digest(CONFIG) and provenance['cohort_sha256']==COHORT_HASHES[900],'Provenance mismatch')
    completed=[]; missing=[]
    for row,ref in zip(cohort[:scope],refs[:scope]):
        key=row['sample_key']; path=SOURCE/'rollouts'/f'{key}.json'
        require(ref['sample_key']==key and sha(path)==ref['sha256'],f'{key}: source checksum changed')
        query=canonical/'queries'/f'{key}.json'
        if not query.exists(): missing.append(key); continue
        record=read(query)
        if not record.get('complete'): missing.append(key); continue
        original=read(path); require(original['sample']==row,f'{key}: sample/cohort differs')
        completed.append(validate_record(record,original,read(SOURCE/'inputs'/f'{key}.json'),read(SOURCE/'token_traces'/key/'index.json'),canonical,ref))
    require(allow_partial or not missing,f'pilot{scope}: {len(missing)} incomplete queries')
    counts=dict(completed=len(completed),steps=sum(r['steps'] for r in completed),targets=sum(r['targets'] for r in completed),
                unique_videos=len({r['video_path'] for r in cohort[:scope] if r['sample_key'] not in missing}))
    report=dict(passed=True,scope_queries=scope,complete=not missing,allow_partial=allow_partial,canonical=str(canonical),**counts,
                missing_queries=missing,source_hashes_verified=True,input_hashes_verified=True,all_greedy_tokens_including_eos=True,
                exclusive_context_groups=True,future_gradient_max=max((r['future_max'] for r in completed),default=None),timestamp=stamp())
    write(BASE/str(scope)/'validation/integrity.json',report); return report

def validate_catalog(scope):
    canonical=storage(); cohort=read(COHORT)[:scope]; catalog=read(BASE/str(scope)/'web/catalog.json')
    require(catalog['scope']==scope and [s['key'] for s in catalog['samples']]==[r['sample_key'] for r in cohort],'Catalog cohort coverage/order differs')
    completed=[]; step_count=target_count=0; videos=set()
    for sample,row in zip(catalog['samples'],cohort):
        path=canonical/'queries'/f'{sample["key"]}.json'; record=read(path) if path.exists() else None
        actual=bool(record and record.get('complete')); require(sample['complete']==actual,'Catalog completion state differs')
        if actual:
            completed.append(sample); step_count+=len(record['steps']); target_count+=len(record['targets']); videos.add(row['video_path'])
    expected=dict(completed=len(completed),steps=step_count,targets=target_count,
                  unique_videos=len({r['video_path'] for r in cohort}),completed_unique_videos=len(videos))
    for key,value in expected.items(): require(catalog['summary'][key]==value,f'Catalog {key} != actual unique-query aggregate')
    require((RESULTS/'pilot'/str(scope)/'REPORT.md').exists(),'MD report missing')
    return catalog,completed

def check_viewer(scope=300,allow_partial=False):
    catalog,completed=validate_catalog(scope); require(completed,'Browser validation requires a real completed query')
    require(allow_partial or len(completed)==scope,'Complete scope required for browser check')
    shared=ROOT.parent/'_shared/timelens'; sys.path.insert(0,str(shared/'browser_deps'))
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
    os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
    from playwright.sync_api import sync_playwright
    sample=completed[0]; record=read(storage()/'queries'/f'{sample["key"]}.json'); errors=[]
    base='http://127.0.0.1:8780'; detail=f'{base}/zero-shot/results/rollout/greedy_attribution/pilot/{scope}/detail.html'
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1600,height=1100)); page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(detail+'?sample='+sample['key'],wait_until='networkidle'); page.wait_for_selector('#sentence button[data-step]',timeout=60000)
        require(page.locator('#sentence button[data-step]').count()==len(record['steps']),'Viewer greedy token coverage differs')
        require(page.locator('#samples button[data-key]').count()<=10,'Query list not paginated')
        page.locator('#sentence button[data-step]').last.click(); page.wait_for_selector('#positions .position',timeout=60000)
        prefix=page.locator('#prefix').inner_text(); url=page.url
        page.reload(wait_until='networkidle'); page.wait_for_selector('#positions .position',timeout=60000)
        require(page.locator('#prefix').inner_text()==prefix,'Deep link lost selected token')
        margin=next((t for t in record['targets'] if t['kind']=='logit_margin'),None)
        if margin:
            page.locator(f'#sentence [data-step="{margin["step"]}"]').click()
            page.select_option('#target',str(record['targets'].index(margin)))
            page.wait_for_selector('#positions .position',timeout=60000)
            page.wait_for_function("document.getElementById('target-description').textContent.includes('logit_margin')")
        for metric in ('grad_l2','activation_gradient'):
            page.select_option('#metric',metric); require(page.locator('#positions .position').count()>0,'Metric switch lost positions')
        page.select_option('#source-type','query')
        require(page.locator('#positions .position').count()>0,'Query-only context filter empty')
        require(page.locator('#positions .position').evaluate_all("els=>els.every(e=>e.title.startsWith('Query'))"),'Context group filter mixed sources')
        page.select_option('#source-type','')
        require(page.locator('#spatial-map .spatial-cell').count()>0,'Spatial map empty')
        require(page.locator('#layers .layer-row').count()==37,'Decoder layer table incomplete')
        for space in ('pixels','visual'):
            page.select_option('#frame-space',space)
            require(page.locator('#spatial-map .spatial-cell').count()>0,f'{space} map empty')
        receivers=page.locator('#receiver option')
        if receivers.count()>1:
            page.select_option('#receiver',receivers.first.get_attribute('value'))
            require(page.locator('#layers .layer-row').count()==37,'Receiver switch lost layer table')
            page.select_option('#receiver',receivers.last.get_attribute('value'))
        require(page.locator('#message-panel').is_visible(),'Signed message panel missing')
        video=page.locator('#video'); src=video.get_attribute('src'); require(src,'Video source missing')
        response=page.request.get(src,headers={'Range':'bytes=0-127'}); require(response.status==206 and len(response.body())==128,'Video Range failed')
        page.wait_for_function("document.getElementById('video').readyState>=1",timeout=30000)
        options=page.locator('#frame option')
        if options.count()>1:
            page.select_option('#frame',options.nth(1).get_attribute('value'))
            expected_time=sum(record['input']['timestamps'][2:4])/2
            close(video.evaluate('v=>v.currentTime'),expected_time,'Temporal patch seek',atol=.15)
        result=RESULTS/'pilot'/str(scope); page.screenshot(path=str(result/'preview_desktop.png'),full_page=True)
        for width in (390,768,1100):
            page.set_viewport_size(dict(width=width,height=844)); require(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),f'Overflow at {width}px')
        page.set_viewport_size(dict(width=390,height=844)); page.screenshot(path=str(result/'preview_mobile.png'),full_page=True)
        page.goto(f'{base}/zero-shot/results/rollout/greedy_attribution/viewer.html?view=pilot%2F{scope}%2Fdetail.html',wait_until='networkidle'); page.wait_for_selector('#analysis-tabs button')
        labels=page.locator('#analysis-tabs').inner_text(); require('결과' in labels and '보고서' in labels,'Common tabs missing')
        family_scopes=[]
        for n in CONFIG['stages']:
            if not (RESULTS/'pilot'/str(n)/'detail.html').exists(): continue
            family_scopes.append(n)
            page.locator('#view-browser > summary').click()
            page.locator(f'#view-tree-items button[data-view="pilot/{n}/detail.html"]').click()
            page.frame_locator('#detail-frame').locator('#sentence button[data-step]').first.wait_for(timeout=60000)
            require(f'pilot/{n}/detail.html' in page.url.replace('%2F','/'),'Scope did not switch')
            page.locator('#tab-report').click(); page.wait_for_selector('#report h1')
            require(str(n) in page.locator('#report h1').inner_text(),'MD report scope differs')
            require(page.locator('#source-md').is_visible(),'MD source link missing on report tab')
            page.locator('#tab-result').click(); page.frame_locator('#detail-frame').locator('#sentence button[data-step]').first.wait_for(timeout=60000)
            require(not page.locator('#source-md').is_visible(),'MD source link leaked into result tab')
        require(page.request.get(base+'/').status==200,'Unified index server did not respond')
        require(not errors,f'Browser errors: {errors}'); browser.close()
    report=dict(passed=True,scope_queries=scope,completed_queries=len(completed),allow_partial=allow_partial,real_sample=sample['key'],
                token_navigation=True,context_positions=True,spatial_map=True,layer_table=True,deep_link=url,video_range=True,mobile_no_overflow=True,
                family_scopes=family_scopes,result_report_switch=True,unified_index=True,page_errors=errors,timestamp=stamp())
    write(BASE/str(scope)/'validation/viewer.json',report); return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--scope',type=int,default=300); p.add_argument('--allow-partial',action='store_true'); p.add_argument('--viewer',action='store_true'); a=p.parse_args()
    result=validate_scope(a.scope,a.allow_partial)
    if a.viewer: result['viewer']=check_viewer(a.scope,a.allow_partial)
    print(json.dumps(result,ensure_ascii=False))
