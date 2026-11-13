"""CPU integrity and real browser gates for the fixed visual-controls pilot60."""
from __future__ import annotations
import collections
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from oracle_switch_controls_core import CONFIG,OUT,RESULTS,ROOT,read,write,sha,stamp,digest,iou
from oracle_switch_controls_select import DATASETS,POPULATION_COUNTS,sampling_hash,COHORT,COHORT_SHA256

SELECTION_SHA256='a41cd09ad9333415063d5e86206d846390dda7e806dc92cd3d58eda344ed8001'

def require(condition,message):
    if not condition: raise ValueError(message)

def close(actual,expected,label,tolerance=1e-6):
    require(math.isfinite(float(actual)) and abs(float(actual)-float(expected))<=tolerance,
            f'{label}: {actual} != {expected}')

def finite(value,label='record'):
    if isinstance(value,float): require(math.isfinite(value),f'{label}: nonfinite scalar')
    elif isinstance(value,dict):
        for name,item in value.items(): finite(item,f'{label}.{name}')
    elif isinstance(value,list):
        for index,item in enumerate(value): finite(item,f'{label}[{index}]')

def validate_selection():
    path=OUT/'metadata/selection.json'; require(sha(path)==SELECTION_SHA256,'Frozen pilot60 selection changed')
    selection=read(path); samples=selection['samples']; cohort=read(COHORT)
    require(sha(COHORT)==COHORT_SHA256,'Original cohort900 hash changed')
    require(len(samples)==len({sample['sample_key'] for sample in samples})==60,'Selected query coverage differs')
    require(collections.Counter(sample['dataset'] for sample in samples)=={dataset:20 for dataset in DATASETS},'Dataset denominator differs')
    for ref in selection['source_refs'].values(): require(sha(ref['path'])==ref['sha256'],'Selection source reference changed')
    excluded={row['sample_key'] for row in selection['excluded_extreme_refs']}
    for row in selection['excluded_extreme_refs']: require(sha(row['query']['path'])==row['query']['sha256'],'Excluded extreme response changed')
    rankings=read(selection['source_refs']['source_rankings']['path'])['rankings']
    population=[row for row in rankings if row['eligible'] and row['sample_key'] not in excluded]
    require(len(population)==805 and dict(collections.Counter(row['dataset'] for row in population))==POPULATION_COUNTS,'Conditional population differs')
    expected=[]
    for dataset in DATASETS:
        expected.extend(row['sample_key'] for row in sorted((row for row in population if row['dataset']==dataset),key=lambda row:(sampling_hash(row['sample_key']),row['sample_key']))[:20])
    require([sample['sample_key'] for sample in samples]==expected,'Hash-selected cohort order differs')
    require(not excluded.intersection(expected),'Extreme pilot3 overlaps the new study')
    for sample in samples:
        require(cohort[sample['source_cohort_index']]==sample['row'],'Selected source cohort row differs')
        require(sample['selection_hash']==sampling_hash(sample['sample_key']),'Selection hash differs')
        for name,source in sample['source'].items(): require(sha(source)==sample['source_hashes'][name],f"{sample['sample_key']}: source {name} changed")
        require(sha(sample['source_trace']['path'])==sample['source_trace']['sha256'],'Original top30 trace changed')
        require(read(sample['source']['input'])['hashes']==sample['input_hashes'],'Original input hash differs')
    precise=[sample['sample_key'] for sample in samples if sample['selection_rank_within_dataset']<=2]
    require(selection['prespecified_precision_subset']==precise and len(precise)==6,'Prespecified precision subset changed')
    if (OUT/'metadata/config.json').exists(): require(read(OUT/'metadata/config.json')==CONFIG,'Frozen control config differs')
    return selection

def validate_probe(probe,sample):
    finite(probe); branch=sample['branch']; top=probe['top30']
    require([row['id'] for row in top]==branch['top_ids'] and len(top)==30,'Fixed original top30 differs')
    require(all(0<=row['probability']<=1 for row in top) and sum(row['probability'] for row in top)<=1.00003,'Invalid raw top30 probabilities')
    by_id={row['id']:row for row in top}; greedy=by_id[branch['greedy_token']]; oracle=by_id[branch['oracle_token']]
    close(probe['margin'],greedy['logit']-oracle['logit'],'Paired margin')
    close(probe['greedy_p'],greedy['probability'],'Greedy probability')
    close(probe['oracle_p'],oracle['probability'],'Oracle probability')
    prefix=probe['prefix_argmax']==branch['prefix_ids']; target=probe['top1_id']==branch['oracle_token']
    require(probe['prefix_argmax_match']==prefix and probe['target_top1']==target,'Prefix/target flags differ')
    require(probe['branch_reached']==(prefix and target),'Conditional branch flag differs')
    require(probe['strict_target_top1']==(probe['oracle_margin']>0),'Strict top1 flag differs')
    require(probe['paired_reversal']==(probe['margin']<0),'Paired reversal flag differs')
    require(probe['top1_in_original_top30']==(probe['top1_id'] in by_id),'Native argmax membership differs')
    tracked_other=max(row['logit'] for row in top if row['id']!=branch['oracle_token'])
    require(probe['oracle_margin']<=oracle['logit']-tracked_other+1e-6,'Full-vocabulary oracle margin exceeds tracked-other bound')

def validate_free(outcome,sample):
    finite(outcome); ids=outcome['ids']; branch=sample['branch']; step=branch['step']; original=sample['greedy']['ids']
    require(isinstance(ids,list) and all(isinstance(token,int) for token in ids),'Invalid free-decoding token IDs')
    reached=len(ids)>step and ids[:step]==branch['prefix_ids'] and ids[step]==branch['oracle_token']
    require(outcome['oracle_branch_reached']==reached and outcome['oracle_path_equal']==(ids==sample['oracle']['ids']),'Free-decoding path flags differ')
    first=next((index for index,(a,b) in enumerate(zip(ids,original)) if a!=b),None)
    if first is None and len(ids)!=len(original): first=min(len(ids),len(original))
    require(outcome['first_divergence_step']==first,'Free first divergence differs')
    span=outcome['span']
    if span is not None: require(outcome['stopped_on_eos'] and len(span)==2 and span[1]>span[0],'Invalid parsed free span')
    close(outcome['iou'],iou(span,sample['row']['span'][0]) or 0.,'Free GT IoU')
    close(outcome['oracle_span_iou'],iou(span,sample['oracle']['span']) or 0.,'Free oracle span IoU')

def validate_rgb(rgb,epsilon_255,rounded=False):
    finite(rgb); linf=rgb['actual_rgb_linf']; rms=rgb['actual_rgb_rms_full']
    # Quantization can add half a uint8 level; the rounded trial records its
    # actual physical change and is not falsely checked against continuous eps.
    allowance=(abs(epsilon_255)+(.5 if rounded else 0.))/255+1e-6
    require(0<=rms<=linf+1e-7 and 0<=linf<=allowance,'Actual RGB delta exceeds its continuous/rounded budget')
    for name in ('changed_fraction','changed_fraction_full','changed_fraction_support','clipped_fraction'):
        if name in rgb: require(0<=rgb[name]<=1,f'{name}: invalid physical fraction')

def validate_cast(cast,grid):
    finite(cast); components=cast['components']; require(components>0,'Empty input-cast measurement')
    require(math.prod(cast['shape'])==components,'Input-cast shape/count differs')
    require(cast['shape']==[math.prod(grid),1536],'Input-cast geometry differs from original video patch packing')
    require(cast['original_dtype']==cast['perturbed_dtype']=='torch.float32','Pre-cast normalized input precision differs')
    require(cast['cast_dtype']=='torch.bfloat16','Native input cast differs')
    for name,prefix in [('before_cast','precast'),('after_cast','postcast')]:
        values=cast[name]; count=values['changed_components']; require(0<=count<=components,'Invalid cast changed count')
        close(values['changed_fraction'],count/components,'Cast fraction',1e-12)
        close(cast[prefix+'_changed_fraction'],values['changed_fraction'],'Cast fraction alias',1e-12)
        close(cast[prefix+'_linf'],values['max_abs'],'Cast L-infinity alias',1e-12)
        close(cast[prefix+'_rms'],values['rms'],'Cast RMS alias',1e-12)
        require(0<=values['rms']<=values['max_abs']+1e-12,'Invalid cast norm')
    equal=cast['after_cast']['changed_components']==0
    require(cast['postcast_equal']==cast['input_pair_identical_after_cast']==equal,'Cast equality flag differs')

def expected_trials():
    methods=[('targeted',None),('opposite',None),*[('random',seed) for seed in CONFIG['random_seeds']],('repeat_gradient',None),('rounded8_targeted',None)]
    return [(method,seed,budget) for budget in CONFIG['common_epsilon_255'] for method,seed in methods]

def validate_controls(record,sample,query,precision_selected):
    finite(record); key=sample['sample_key']
    require(record['sample_key']==key and record['config_hash']==digest(CONFIG),'Control source/config differs')
    require(record['input_hashes']==sample['input_hashes'],'Control input hashes differ')
    require(record['baseline']==query['baseline'] and record['baseline_free']==query['baseline_free'],'Native control baseline differs from measured query')
    validate_probe(record['baseline'],sample); validate_free(record['baseline_free'],sample)
    trials=record['trials']; actual=[(trial['method'],trial['seed'],trial['epsilon_255']) for trial in trials]
    expected=expected_trials()
    require(actual==expected[:len(actual)],'Control method/seed/budget coverage/order differs')
    if record['complete']: require(actual==expected and len(trials)==14,'Completed query lacks exactly 14 matched controls')
    for trial in trials:
        validate_probe(trial['probe'],sample); validate_free(trial['free'],sample)
        method=trial['method']; validate_rgb(trial['rgb'],trial['epsilon_255'],method=='rounded8_targeted'); validate_cast(trial['precision_input'],query['input']['grid'])
        if method=='random':
            seed=int.from_bytes(hashlib.sha256(f"oracle_switch_controls|{trial['seed']}|{key}".encode()).digest()[:8],'big')
            require(trial['random_generator_seed']==seed,'Random generator seed differs')
        else: require(trial['random_generator_seed'] is None,'Unexpected random seed for fixed direction')
        if method=='rounded8_targeted':
            rounded=trial['roundtrip']; require(0<=rounded['rounded_rgb_changed_fraction']<=1,'Invalid rounded change fraction')
            require(rounded['rounded_rgb_identical_to_baseline']==(rounded['rounded_rgb_changed_fraction']==0),'Rounded byte identity differs')
            if rounded['rounded_rgb_identical_to_baseline']: require(rounded['normalized_tensor_identical_to_baseline'],'Identical rounded frames did not preserve original input')
            if rounded['normalized_tensor_identical_to_baseline']:
                require(trial['probe']==record['baseline'] and trial['free']['ids']==record['baseline_free']['ids'],'Identical rounded input changed baseline outputs')
                require(trial['precision_input']['postcast_equal'],'Identical rounded input changed BF16 values')
                close(trial['rgb']['actual_rgb_linf'],0.,'Identical rounded RGB delta',1e-7)
        else: require('roundtrip' not in trial,'Round-trip record attached to another method')
    audit=record['precision_audit']; require(audit['selected']==precision_selected,'Precision subset membership differs')
    modes=audit['modes']; expected_modes=CONFIG['precision_modes'] if precision_selected else []
    require([mode['mode'] for mode in modes]==expected_modes[:len(modes)],'Precision mode coverage/order differs')
    if record['complete']: require([mode['mode'] for mode in modes]==expected_modes,'Completed precision audit lacks modes')
    for mode in modes:
        validate_probe(mode['baseline'],sample); validate_free(mode['baseline_free'],sample)
        expected_match=mode['baseline']['prefix_argmax_match'] and mode['baseline']['top1_id']==record['baseline']['top1_id']
        require(mode['baseline_matches_native_prefix_and_token']==expected_match,'Precision baseline comparison flag differs')
        require(mode['metadata']['full_fp32_model'] is False and mode['metadata']['scope']=='initial_visual_patch_projection_only','Precision scope is misrepresented')
        conditions=mode['conditions']; budgets=[point['epsilon_255'] for point in conditions]
        require(budgets==CONFIG['common_epsilon_255'][:len(budgets)],'Precision condition budgets differ')
        if record['complete']: require(budgets==CONFIG['common_epsilon_255'],'Completed precision audit lacks conditions')
        for point in conditions:
            validate_probe(point['probe'],sample); validate_free(point['free'],sample); validate_rgb(point['rgb'],point['epsilon_255'])
            close(point['delta_oracle_margin'],point['probe']['oracle_margin']-mode['baseline']['oracle_margin'],'Within-mode oracle-margin change')

def validate_query(query,sample):
    finite(query); require(query['complete'] and query['sample_key']==sample['sample_key'],'Native scan incomplete or mismatched')
    require(query['config_hash']==digest(CONFIG) and query['input']['hashes']==sample['input_hashes'],'Native scan input/config differs')
    require(query['baseline_free']['ids']==sample['greedy']['ids'],'Successful native baseline did not reproduce original greedy')
    validate_probe(query['baseline'],sample); validate_free(query['baseline_free'],sample)
    require([direction['id'] for direction in query['directions']]==CONFIG['supports'],'Native support coverage differs')
    for direction in query['directions']:
        for trial in direction['trials']:
            validate_probe(trial,sample)
            epsilon=abs(trial['epsilon']); rgb=trial['perturbation']
            if 'actual_rgb_linf' in rgb: validate_rgb(rgb,epsilon*255)
            else: require(epsilon==0 and rgb['rgb_linf']==0,'Missing physical perturbation stats')
        for name in ('free_low','free_high','free_limit','conditional_high'):
            if direction.get(name): validate_free(direction[name],sample)
        if direction.get('boundary'): require('verified' in direction['boundary'],'Boundary verification outcome missing')
    gradient=query['gradient']
    for path_key,hash_key in [('direction_file','direction_sha256'),('repeat_sign_file','repeat_sign_sha256')]:
        path=Path(gradient[path_key]); require(path.resolve().is_relative_to(OUT.resolve()) and sha(path)==gradient[hash_key],'Saved gradient direction checksum differs')

def validate(allow_partial=True):
    selection=validate_selection(); samples=selection['samples']; allowed={sample['sample_key'] for sample in samples}
    for folder in (OUT/'queries',OUT/'analyses/controls'):
        require(all(path.stem in allowed for path in folder.glob('*.json')),'Unexpected query outside frozen pilot60')
    counters={dataset:dict(planned=20,completed=0,failed=0,pending=0) for dataset in DATASETS}
    pending=[]; failed=[]; successful=[]; partial=[]; trial_count=0
    for sample in samples:
        key=sample['sample_key']; path=OUT/'analyses/controls'/f'{key}.json'; counts=counters[sample['dataset']]
        if not path.exists(): pending.append(key); counts['pending']+=1; continue
        record=read(path); finite(record)
        require(record['sample_key']==key and record['config_hash']==digest(CONFIG),'Control record source/config differs')
        if record.get('failed'):
            require(record['complete'] and record['failure_kind']=='baseline_replay' and bool(record['error']) and isinstance(record['details'],dict),'Invalid explicit failure record')
            require(record['details']['input']['hashes']==sample['input_hashes'],'Failed baseline input hash differs')
            validate_probe(record['details']['baseline'],sample)
            if record['details'].get('baseline_free'): validate_free(record['details']['baseline_free'],sample)
            failed.append(key); counts['failed']+=1; continue
        query_path=OUT/'queries'/f'{key}.json'; require(query_path.exists(),'Controls lack their native scan')
        query=read(query_path); validate_query(query,sample)
        validate_controls(record,sample,query,key in selection['prespecified_precision_subset']); trial_count+=len(record['trials'])
        if record['complete']: successful.append(key); counts['completed']+=1
        else: partial.append(key); pending.append(key); counts['pending']+=1
    require(allow_partial or not pending,'Study has unattempted/incomplete selected queries')
    report=dict(passed=True,scope_queries=60,population_queries=805,complete=not pending,allow_partial=allow_partial,
        completed_queries=len(successful),failed_queries=len(failed),terminal_queries=len(successful)+len(failed),pending_queries=len(pending),
        trial_records=trial_count,dataset_denominators=counters,population_denominators=POPULATION_COUNTS,
        completed_keys=successful,failed_keys=failed,pending_keys=pending,partial_keys=partial,
        selection_sha256=SELECTION_SHA256,source_hashes_verified=True,failed_queries_retained=True,
        independent_query_denominator=60,random_seeds_are_within_query_repeats=True,timestamp=stamp())
    write(OUT/'validation/integrity.json',report); return report


def check_viewer(allow_partial=True):
    """Check the published snapshot while the immutable cohort runs continue."""
    selection=read(OUT/'metadata/selection.json'); samples=selection['samples']
    require(sha(OUT/'metadata/selection.json')==SELECTION_SHA256,'Viewer cohort changed')
    require((RESULTS/'detail.html').is_file() and (RESULTS/'REPORT.md').is_file(),'Generate the report before checking its viewer')
    snapshot_hash=sha(RESULTS/'detail.html'); report_hash=sha(RESULTS/'REPORT.md')
    controls={sample['sample_key']:read(path) if (path:=OUT/'analyses/controls'/f"{sample['sample_key']}.json").exists() else {} for sample in samples}
    shared=ROOT.parent/'_shared/timelens'
    sys.path.insert(0,str(shared/'browser_deps'))
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(shared/'browser_cache')
    os.environ['LD_LIBRARY_PATH']=str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['FONTCONFIG_FILE']=str(shared/'browser_libs/fonts.conf')
    from playwright.sync_api import sync_playwright
    base='http://127.0.0.1:8780'; family=base+'/zero-shot/results/rollout/oracle_switch_controls/viewer.html'
    detail=base+'/zero-shot/results/rollout/oracle_switch_controls/pilot/60/detail.html'
    errors=[]; verified_links=set(); media=[]; filters=[]; measured_rows=0; published={}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
            page=browser.new_page(viewport=dict(width=1600,height=1100))
            page.on('pageerror',lambda error:errors.append(str(error)))
            def links(locator):
                for url in locator.evaluate_all('nodes=>nodes.map(node=>node.href).filter(Boolean)'):
                    if urlparse(url).netloc!='127.0.0.1:8780' or url in verified_links: continue
                    response=page.request.head(url)
                    require(response.status==200,f'Local result/source link failed: {url} ({response.status})')
                    verified_links.add(url)
            def visible_case(key):
                case=page.locator(f'.query-case[data-key="{key}"]'); case.wait_for(state='visible')
                require(page.locator('.query-case:visible').count()==1,'Multiple query cases visible')
                require(page.locator('#query-select').input_value()==key,'Query selection differs from displayed case')
                require(page.evaluate("new URLSearchParams(location.hash.slice(1)).get('sample')")==key,'Query deep link differs')
                return case
            def no_overflow(label):
                require(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),label+' overflows at390px')

            response=page.goto(detail,wait_until='networkidle')
            require(response and response.status==200 and 'pilot60' in page.title(),'Detail page or scope failed')
            keys=[sample['sample_key'] for sample in samples]
            require(page.locator('.query-case').evaluate_all('nodes=>nodes.map(node=>node.dataset.key)')==keys,'Displayed frozen cohort/order differs')
            require(page.locator('#query-select option').evaluate_all('nodes=>nodes.map(node=>node.value)')==keys,'Query options differ from frozen cohort')
            counts=page.locator('#study-summary > div > b').all_inner_texts()
            require(len(counts)==4 and counts[3]=='805 → 60','Conditional population/selected denominator differs')
            terminal,native=[int(value.split('/')[0].strip()) for value in counts[:2]]; failed=int(counts[2])
            require(all(value.strip().endswith('/ 60') for value in counts[:2]),'Published denominator differs from60')
            require(0<=failed<=terminal<=60 and 0<=native<=60,'Invalid published completion counts')
            require(terminal<=sum(bool(record.get('complete')) for record in controls.values()),'Published completed count exceeds actual terminal records')
            require(failed<=sum(bool(record.get('failed')) for record in controls.values()),'Published failed count exceeds explicit failures')
            require(native<=sum((OUT/'queries'/f'{key}.json').exists() for key in keys),'Published native count exceeds actual query records')
            require(allow_partial or terminal==60,'Published snapshot is still partial')
            published=dict(terminal_queries=terminal,native_queries=native,failed_queries=failed,pending_queries=60-terminal,complete=terminal==60)
            require(page.locator('#dataset-filter option').evaluate_all('nodes=>nodes.map(node=>node.value)')==['all',*DATASETS],'Dataset filter options differ')
            require(page.locator('#budget-filter option').evaluate_all('nodes=>nodes.map(node=>node.value)')==['0.25','1'],'Budget filter options differ')
            require(page.locator('#query-video').count()==1,'Expected one reused-media player')
            links(page.locator('a[href],link[href]'))

            # The report may precede the latest raw query. Test each displayed
            # measured row against its immutable saved result; pending rows stay
            # pending rather than being inferred from the advancing worker.
            for sample in samples:
                key=sample['sample_key']; case=page.locator(f'.query-case[data-key="{key}"]')
                require(case.get_attribute('data-dataset')==sample['dataset'],'Query dataset differs')
                for budget in ('0.25','1'):
                    rows=case.locator(f'.query-budget[data-budget="{budget}"] .query-controls tbody tr')
                    require(rows.count()==7,'Expected seven control rows per fixed budget')
                    expected=[entry for entry in expected_trials() if entry[2]==float(budget)]
                    for index,(method,seed,_) in enumerate(expected):
                        cells=rows.nth(index).locator('td').all_inner_texts()
                        require(len(cells)==12,'Control table columns differ')
                        if cells[1] in ('미측정','재현 실패'): continue
                        trial=next((trial for trial in controls[key].get('trials',[]) if trial['method']==method and trial['seed']==seed and trial['epsilon_255']==float(budget)),None)
                        require(trial is not None,'Published control has no corresponding saved trial')
                        require(cells[1]==('예' if trial['free']['oracle_branch_reached'] else '아니오'),'Primary displayed result is not unforced branch reach')
                        close(float(cells[2]),trial['free']['iou'],'Displayed measured GT IoU',5.1e-5)
                        require(cells[5]==('예' if trial['probe']['branch_reached'] else '아니오'),'Probe reach is not separately displayed')
                        measured_rows+=1

            for dataset in ('all',*DATASETS):
                page.locator('#dataset-filter').select_option(dataset)
                expected=[sample['sample_key'] for sample in samples if dataset=='all' or sample['dataset']==dataset]
                require(page.locator('#query-select option').evaluate_all('nodes=>nodes.map(node=>node.value)')==expected,'Dataset filter query coverage/order differs')
                for budget in ('0.25','1'):
                    page.locator('#budget-filter').select_option(budget)
                    selected=page.locator('.summary-slice:visible')
                    require(selected.count()==1 and selected.get_attribute('data-dataset')==dataset and selected.get_attribute('data-budget')==budget,'Summary filter selection differs')
                    key=page.locator('#query-select').input_value(); case=visible_case(key)
                    panel=case.locator('.query-budget:visible')
                    require(panel.count()==1 and panel.get_attribute('data-budget')==budget,'Query budget selection differs')
                    require(page.locator('#study-summary > div > b').all_inner_texts()==counts,'Dataset filter changed the study denominator')
                    filters.append(dict(dataset=dataset,budget=budget,query_options=len(expected)))
                page.set_viewport_size(dict(width=390,height=844)); no_overflow(dataset)
                page.set_viewport_size(dict(width=1600,height=1100))

            page.locator('#dataset-filter').select_option('all')
            representatives=[next(sample for sample in samples if sample['dataset']==dataset) for dataset in DATASETS]
            for sample in representatives:
                key=sample['sample_key']; page.locator('#query-select').select_option(key); case=visible_case(key)
                video=page.locator('#query-video'); source=video.evaluate('video=>video.src')
                expected_path=Path(sample['video_path']).resolve().relative_to(ROOT.parent.resolve()).as_posix()
                require(urlparse(source).path=='/'+expected_path,'Query video differs from selected reused source media')
                response=page.request.get(source,headers={'Range':'bytes=0-127'})
                require(response.status==206 and len(response.body())==128,'Reused video Range request failed')
                page.wait_for_function("document.querySelector('#query-video').readyState>=1",timeout=30000)
                duration=video.evaluate('video=>video.duration'); require(math.isfinite(duration) and duration>0,'Video duration invalid')
                seek=min(max(0,float(sample['oracle']['span'][0])),max(0,duration-.2))
                video.evaluate('(video,time)=>{video.pause();video.currentTime=time}',seek)
                page.wait_for_function("time=>{const v=document.querySelector('#query-video');return !v.seeking&&v.paused&&Math.abs(v.currentTime-time)<.15}",arg=seek,timeout=30000)
                media.append(dict(sample_key=key,url=source,range=True,seek_time=seek))
                links(case.locator('a[href]'))
                page.reload(wait_until='networkidle'); visible_case(key)
                page.set_viewport_size(dict(width=390,height=844)); no_overflow(key)
                page.set_viewport_size(dict(width=1600,height=1100))

            preview=next((sample for sample in samples if controls[sample['sample_key']].get('complete') and not controls[sample['sample_key']].get('failed')),samples[0])
            page.goto(detail+'#sample='+preview['sample_key'],wait_until='networkidle'); case=visible_case(preview['sample_key'])
            if case.locator('.precision-detail').count():
                case.locator('.precision-detail > summary').click()
                require(case.locator('.precision-budget:visible').count()==1 and case.locator('.precision-table:visible tbody tr').count()==3,'Per-query precision audit does not show three own-baseline modes')
                page.locator('#budget-filter').select_option('1')
                require(case.locator('.precision-budget:visible').get_attribute('data-budget')=='1','Precision budget filter differs')
                case.locator('.precision-detail > summary').click(); page.locator('#budget-filter').select_option('0.25')
            page.screenshot(path=str(RESULTS/'preview_desktop.png'),full_page=True)
            page.set_viewport_size(dict(width=390,height=844)); no_overflow('Mobile preview')
            page.screenshot(path=str(RESULTS/'preview_mobile.png'),full_page=True)
            page.set_viewport_size(dict(width=1600,height=1100))
            response=page.goto(family+'?view=pilot%2F60%2Fdetail.html',wait_until='networkidle')
            require(response and response.status==200,'Family viewer missing')
            page.wait_for_selector('#analysis-tabs button')
            require('결과' in page.locator('#analysis-tabs').inner_text() and '보고서' in page.locator('#analysis-tabs').inner_text(),'Common result/report tabs missing')
            require(page.locator('#view-tree-items button[data-view="pilot/60/detail.html"]').count()==1,'Family pilot60 scope missing')
            frame=page.frame_locator('#detail-frame'); frame.locator('#query-select').wait_for()
            family_key=samples[-1]['sample_key']; frame.locator('#query-select').select_option(family_key)
            page.wait_for_function("key=>new URLSearchParams(location.hash.slice(1)).get('sample')===key",arg=family_key)
            page.reload(wait_until='networkidle'); frame=page.frame_locator('#detail-frame')
            frame.locator(f'.query-case[data-key="{family_key}"]').wait_for(state='visible')
            page.locator('#tab-report').click(); page.wait_for_selector('#report h1')
            require('60' in page.locator('#report h1').inner_text() and page.locator('#source-md').is_visible(),'Report scope/source link missing')
            links(page.locator('#report a[href],#source-md'))
            page.locator('#tab-result').click(); frame.locator(f'.query-case[data-key="{family_key}"]').wait_for(state='visible')
            require(not page.locator('#source-md').is_visible(),'MD source link leaked into result view')
            page.set_viewport_size(dict(width=390,height=844)); no_overflow('Family viewer')
            require(page.request.get(base+'/').status==200,'Unified result index failed')
            require(not errors,f'Browser errors: {errors}')
            browser.close()
        require(sha(RESULTS/'detail.html')==snapshot_hash and sha(RESULTS/'REPORT.md')==report_hash,'Published snapshot changed during validation; rerun against the new snapshot')
        report=dict(passed=True,scope='pilot',scope_queries=60,population_queries=805,published_snapshot=published,
            published_html_sha256=snapshot_hash,published_report_sha256=report_hash,selection_sha256=SELECTION_SHA256,
            frozen_query_options=True,filters=filters,checked_measured_rows=measured_rows,media=media,
            source_links=sorted(verified_links),sample_deep_links=True,family_deep_link=True,result_report_navigation=True,
            desktop_and_mobile=True,page_errors=errors,unified_index=True,
            previews=[str(RESULTS/'preview_desktop.png'),str(RESULTS/'preview_mobile.png')],timestamp=stamp())
        write(OUT/'validation/viewer.json',report); return report
    except BaseException as error:
        write(OUT/'validation/viewer.json',dict(passed=False,scope_queries=60,published_snapshot=published,error=str(error),page_errors=errors,timestamp=stamp()))
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--viewer-only',action='store_true'); parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    result=check_viewer(allow_partial=not args.require_complete) if args.viewer_only else validate(allow_partial=not args.require_complete)
    print(json.dumps(result,ensure_ascii=False))

