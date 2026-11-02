from result_layout import relocate_path, write_result_text
"""Exercise the offline viewer with real cached embeddings or final results."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from prepare import ROOT,OUT,VIEW,video_key,write
from metrics import aggregate,metrics,labels,TEMPERATURES

RESOURCES=Path('/home/irteam/experiments/_shared/timelens')
sys.path.insert(0,str(RESOURCES/'browser_deps'))
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(RESOURCES/'browser_cache')
os.environ['LD_LIBRARY_PATH']=str(RESOURCES/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['FONTCONFIG_FILE']='/home/irteam/qwen-chat/runtime/browser-libs/fonts.conf'

def preview():
    rows=json.loads((OUT/'samples.json').read_text());samples=[];records=[]
    q={m:np.load(OUT/m/'queries.npz')['embeddings'] for m in ['clip','qwen']}
    for i,s in enumerate(rows):
        if not all((OUT/m/'video_embeddings'/f'{video_key(s["video_path"])}.npz').exists() for m in q):continue
        meta=json.loads((OUT/'frames'/video_key(s['video_path'])/'metadata.json').read_text())
        item={k:s[k] for k in ['sample_key','dataset','query','duration','span']}
        item.update(video_key=video_key(s['video_path']),timestamps=meta['timestamps'],models={})
        for m in q:
            v=np.load(OUT/m/'video_embeddings'/f'{video_key(s["video_path"])}.npz')
            scores=v['embeddings']@q[m][i]
            r=dict(model=m,dataset=s['dataset'],sample_key=s['sample_key'],**metrics(scores,labels(meta['timestamps'],s['span'])))
            records.append(r);item['models'][m]=dict(scores=scores.tolist(),metrics=r)
        samples.append(item)
        if len(samples)>=6:break
    assert samples
    groups={ds:{m:aggregate([r for r in records if r['model']==m and (ds=='all' or r['dataset']==ds)]) for m in q} for ds in ['all']+sorted({s['dataset'] for s in samples})}
    data=dict(samples=samples,summary=dict(groups=groups,temperatures=TEMPERATURES))
    target=VIEW/'.viewer_preview.html'
    template=(ROOT/'_lib/viewer.html').read_text().replace('artifacts/',os.path.relpath(OUT,VIEW)+'/')
    write_result_text(target, template.replace('__DATA__',json.dumps(data).replace('<','\\u003c')))
    return target

def main():
    from playwright.sync_api import sync_playwright
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    target=preview() if args.preview else VIEW/'report.html'
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1440,height=1100))
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(target.as_uri(),timeout=120000)
        page.wait_for_selector('#raw svg')
        assert page.locator('#raw path').count()==2
        assert page.locator('#mass path').count()==2
        expected_frames=page.evaluate('current.timestamps.length')
        for plot in ['raw','mass']:
            for model in ['clip','qwen']:
                assert page.locator(f'#{plot} circle.measurement[data-model="{model}"]').count()==expected_frames
            assert page.evaluate("""(id)=>[...document.querySelectorAll('#'+id+' circle.measurement')].every(dot=>{
                const i=+dot.dataset.frameIndex,t=current.timestamps[i];
                const max=Math.max(current.duration,current.timestamps.at(-1));
                return +dot.dataset.timestamp===t && Math.abs(+dot.getAttribute('cx')-(63+t/max*1019))<1e-7;
            })""",plot)
        assert page.locator('#metrics tr').count()==3
        has_adjacent=page.evaluate("Array.isArray(current.models.clip.adjacent_scores)")
        if has_adjacent:
            assert page.locator('#adjacentCard').is_visible()
            for model in ['clip','qwen']:
                assert page.locator(f'#adjacent circle.measurement[data-model="{model}"]').count()==expected_frames-1
            assert page.evaluate("""()=>[...document.querySelectorAll('#adjacent circle.measurement')].every(dot=>{
                const i=+dot.dataset.frameIndex;return i>=1 && i<current.timestamps.length && +dot.dataset.timestamp===current.timestamps[i];
            })""")
            page.locator('#adjacent').click(position=dict(x=350,y=100))
            page.wait_for_function("document.querySelectorAll('#adjacentFrames img').length===2 && [...document.querySelectorAll('#adjacentFrames img')].every(x=>x.naturalWidth>0)")
            assert page.evaluate("""()=>{const imgs=[...document.querySelectorAll('#adjacentFrames img')];
                const indices=imgs.map(im=>+im.getAttribute('src').match(/\/(\d+)\.jpg/)[1]);
                return indices[1]===indices[0]+1;}""")
        before=page.locator('#query').inner_text();page.click('#next');assert before!=page.locator('#query').inner_text()
        for t in TEMPERATURES:
            page.select_option('#tau',format(t,'g'));assert page.locator('#mass path').count()==2
        page.select_option('#tau','0.07')
        for ds in page.locator('#dataset option').evaluate_all('(els)=>els.map(e=>e.value)'):
            page.select_option('#dataset',ds)
            if not args.preview and ds!='all':
                expected=page.evaluate('(ds)=>D.samples.filter(s=>s.dataset===ds).length',ds)
                assert page.locator('#sample option').count()==expected
            page.wait_for_function("document.querySelector('#videoPlayer').readyState>=2 && document.querySelector('#videoPlayer').videoWidth>0")
            page.evaluate("document.querySelector('#videoPlayer').play()")
            page.wait_for_function("document.querySelector('#videoPlayer').currentTime>0.2")
            page.evaluate("document.querySelector('#videoPlayer').pause()")
            page.locator('#gtSeek button').first.click()
            page.wait_for_function("Math.abs(document.querySelector('#videoPlayer').currentTime-current.span[0][0])<0.1")
            assert page.locator('#raw .playhead').count()==1
        page.select_option('#dataset','all')
        page.select_option('#sort','delta');page.select_option('#sort','clip');page.select_option('#sort','original')
        key=page.locator('#sample').input_value()
        page.fill('#search',key);assert page.locator('#sample option').count()==1
        page.fill('#search','')
        page.locator('#raw').click(position=dict(x=350,y=100))
        page.wait_for_function("document.querySelector('#framePreview img')?.naturalWidth>0")
        page.wait_for_function("Math.abs(document.querySelector('#videoPlayer').currentTime-parseFloat(document.querySelector('#framePreview p').textContent))<0.1")
        video_time=page.evaluate("document.querySelector('#videoPlayer').currentTime")
        page.select_option('#tau','0.1')
        assert abs(page.evaluate("document.querySelector('#videoPlayer').currentTime")-video_time)<.1
        page.locator('#topQwen .frameSeek').first.click()
        page.wait_for_function("Math.abs(document.querySelector('#videoPlayer').currentTime-parseFloat(document.querySelector('#framePreview p').textContent))<0.1")
        page.wait_for_function("[...document.querySelectorAll('#topClip img')].every(x=>x.complete && x.naturalWidth>0)")
        if not args.preview:
            page.screenshot(path=str(VIEW/'viewer_desktop.png'),full_page=True)
        page.set_viewport_size(dict(width=390,height=844))
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        if not args.preview:
            page.screenshot(path=str(VIEW/'viewer_mobile.png'),full_page=True)
        browser.close()
    assert not errors,errors
    if args.preview:
        target.unlink()
        print('PREVIEW BROWSER VALIDATION PASSED (no persistent preview artifacts)')
        return
    write(OUT/'browser_validation.json',dict(passed=True,
        offline_file_protocol=True,raw_curves=2,mass_curves=2,temperature_switches=6,
        sample_navigation=True,dataset_filter=True,search=True,sort=True,frame_preview=True,
        images_loaded=True,mobile_no_overflow=True,video_playback_all_datasets=True,
        measured_frame_markers=True,marker_timestamps_and_x_coordinates=True,
        adjacent_frame_cosine=has_adjacent,
        gt_video_seek=True,graph_video_seek=True,thumbnail_video_seek=True,page_errors=errors))
    for name in ['preview_desktop.png','preview_mobile.png','preview_browser_validation.json']:
        (VIEW/name).unlink(missing_ok=True)
    print('BROWSER VALIDATION PASSED')

if __name__=='__main__':main()
