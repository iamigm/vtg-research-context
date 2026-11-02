from result_layout import relocate_path, write_result_text, result_glob
from timelens_paths import RESOURCES
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Verify spatial indexing, map aggregation, exported frames, controls and viewer."""
from pathlib import Path
import os,sys,json
import numpy as np
from spatial_analysis import OUT,RESULTS,load
from paths import ROOT


def arrays():
    # Known 3x4 merged coordinates, including temporal-pair packing, match row-major maps.
    values=np.repeat(np.repeat(np.arange(12).reshape(3,4),2,0),2,1)
    video=np.broadcast_to(values,(1,2,1,6,8)).copy()
    patches=video.reshape(1,1,2,1,3,2,1,4,2,1).transpose(0,1,4,7,5,8,3,2,6,9).reshape(1,48,2)
    merged=patches.reshape(12,4,2).mean((1,2))
    np.testing.assert_array_equal(merged,np.arange(12))
    cases=json.loads((relocate_path(OUT/'strong_samples.json')).read_text());assert len(cases)==25
    frame_count=0
    for s in cases:
        m,a,c,t,temporal,edges=load(s)
        assert min(x[0] for x in m['gt_windows'])>=.1*m['duration_video']
        assert t[temporal.argmax()]<.1*m['duration_video'] and s['early_mass']>=.2
        original=np.load(relocate_path(RESULTS/s['dataset']/s['sample_key']/'distributions.npz'))
        np.testing.assert_allclose(a.sum((1,2)),original['attention_query_frame'][8:22].astype(float).mean((0,1)),atol=2e-7)
        folder=relocate_path(OUT/s['sample_key']);d=json.loads((relocate_path(folder/'viewer_data.json')).read_text())
        images=list(result_glob((relocate_path(folder/'frames')), '*.jpg'));assert len(images)==len(m['decoder_metadata']['frames_indices'])
        frame_count+=len(images)
        assert len(images) in (2*len(t),2*len(t)-1)
        np.testing.assert_allclose(np.asarray(d['attention']).sum((1,2)),1,atol=2e-6)
        np.testing.assert_allclose(d['attention'],a/a.sum((1,2),keepdims=True),atol=5.1e-9)
        with np.load(relocate_path(folder/'frames_and_maps.npz')) as z:
            for i,b in enumerate(z['bins']):
                assert z['frame_indices'][i,0]==m['decoder_metadata']['frames_indices'][2*b]
                np.testing.assert_array_equal(z['attention'][i],a[b])
        v=json.loads((relocate_path(folder/'counterfactual_validation.json')).read_text())
        assert v['original_attention_bit_identical'] and v['video_tensor_reused'] and v['prefix_before_event_identical']
        z=np.load(relocate_path(folder/'counterfactual.npz'));assert z['attention'].shape==(3,*a.shape)
        np.testing.assert_allclose(z['attention'][0],a,atol=1e-8,rtol=2e-6)
    return dict(merged_patch_coordinates=True,spatial_temporal_sums=True,selected_videos=25,
                all_sampled_frames=frame_count,control_forwards=50,original_reproduction_forwards=25,
                all_original_attention_bit_identical=True,gt_selection=True)


def browser():
    deps=RESOURCES
    sys.path.insert(0,str(relocate_path(deps/'browser_deps')))
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(relocate_path(deps/'browser_cache'))
    os.environ['LD_LIBRARY_PATH']=str(relocate_path(deps/'browser_libs/usr/lib/x86_64-linux-gnu'))+':'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['FONTCONFIG_FILE']=str(relocate_path(deps/'browser_libs/fonts.conf'))
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=b.new_page(viewport={'width':1600,'height':1120});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((relocate_path(OUT/'report.html')).as_uri(),wait_until='load')
        page.wait_for_function('document.body.dataset.rendered')
        assert page.locator('#sample option').count()==25
        for key in ['Charades-STA_51','QVHighlights_15','ActivityNet-Captions_62']:
            page.locator('#sample').select_option(key)
            page.wait_for_function('(k)=>document.body.dataset.rendered.startsWith(k+":")',arg=key)
            first=page.locator('#attn').screenshot()
            page.locator('#gt').click()
            page.wait_for_function('document.body.dataset.rendered===current.sample_key+":"+frame')
            assert first!=page.locator('#attn').screenshot()
            page.locator('#slider').evaluate('(e)=>{e.value=e.max;e.dispatchEvent(new Event("input"))}')
            page.wait_for_function('document.body.dataset.rendered===current.sample_key+":"+frame')
        page.locator('#sample').select_option('Charades-STA_51')
        page.wait_for_function('document.body.dataset.rendered.startsWith("Charades-STA_51:")')
        page.locator('#play').click();page.wait_for_timeout(900);page.locator('#play').click()
        page.locator('#early').click();page.wait_for_function('document.body.dataset.rendered===current.sample_key+":"+frame')
        await_fonts=page.evaluate('document.fonts.ready')
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        for link in page.locator('a').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("href"))'):
            assert (relocate_path(OUT/link)).is_file(),link
        page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(relocate_path(OUT/'viewer_preview.png')))
        assert not errors,errors;b.close()
    return dict(local_file_mode=True,sample_selection=True,gt_jump=True,last_frame=True,
                map_changes=True,autoplay=True,page_errors=errors)


if __name__=='__main__':
    result={'arrays':arrays(),'browser':browser()}
    write_result_text((relocate_path(OUT/'validation.json')), json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
