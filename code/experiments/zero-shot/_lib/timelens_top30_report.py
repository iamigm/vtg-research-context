"""Validate all full-head traces and build the separate video viewer / quantitative report."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import json, re, statistics, hashlib, csv
ROOT=relocate_path(Path(__file__)).resolve().parents[1]
OUT=relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300')
from timelens_paths import PILOT_SAMPLES
def read(p):return json.loads(p.read_text())
def avg(v):return statistics.mean(v) if v else 0

def build():
    samples=read(PILOT_SAMPLES); media=read(relocate_path(OUT/'media_manifest.json')); records=[]; manifest=[]
    assert len(samples)==300 and len({s['sample_key'] for s in samples})==300
    for s in samples:
        item=dict(s,media=media[s['video_path']]['media'],models={})
        for model in ['qwen25','qwen3']:
            r=read(relocate_path(OUT/model/(s['sample_key']+'.json')));steps=r['steps']
            assert r['sample']==s and r['baseline_tokens_identical'] and len(steps)==len(r['generated_ids'])
            baseline=relocate_path(Path(r['baseline_path']))
            assert hashlib.sha256(baseline.read_bytes()).hexdigest()==r['baseline_sha256']
            assert read(baseline)['generated_ids']==r['generated_ids']==[t['token_id'] for t in steps]
            for t in steps:
                # topk may order equal-probability tokens differently from argmax.
                assert t['processed']['top30'][0]['probability']==t['processed_selected_probability']
                assert any(x['id']==t['token_id'] and x['probability']==t['processed_selected_probability'] for x in t['processed']['top30'])
                for mode in ['raw','processed']:
                    p=t[mode];top=p['top30'];probs=[x['probability'] for x in top]
                    assert len(top)==len({x['id'] for x in top})==30 and probs==sorted(probs,reverse=True)
                    assert all(0<=x<=1 for x in probs) and abs(p['probability_sum']-1)<2e-5 and sum(probs)<=1.00002
            records.append(r)
            item['models'][model]={'answer':r['answer'],'iou':r['iou'],'timestamps':r['timestamps']}
        manifest.append(item)
    groups=[]
    for model in ['qwen25','qwen3']:
        for ds in dict.fromkeys(s['dataset'] for s in samples):
            rs=[r for r in records if r['model_key']==model and r['sample']['dataset']==ds];ts=[t for r in rs for t in r['steps']]
            nums=[t for t in ts if re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',t['token'])]
            g=dict(model=model,dataset=ds,n=len(rs),steps=len(ts),mean_output_tokens=avg([len(r['steps']) for r in rs]),miou=100*avg([r['iou'] for r in rs]),
                **{f'r{int(th*100)}':100*avg([r['iou']>=th for r in rs]) for th in [.3,.5,.7]},
                selected_probability=avg([t['raw_selected_probability'] for t in ts]),top30_mass=avg([t['raw']['top30_mass'] for t in ts]),
                entropy_nats=avg([t['raw']['entropy_nats'] for t in ts]),numeric_steps=len(nums),numeric_selected_probability=avg([t['raw_selected_probability'] for t in nums]),
                raw_argmax_disagreements=sum(t['raw_selected_rank']>1 for t in ts),selected_outside_raw_top30=sum(t['raw_selected_rank']>30 for t in ts))
            assert g['n']==100;groups.append(g)
    summary=dict(samples=300,model_query_runs=600,total_steps=sum(g['steps'] for g in groups),baseline_exact_matches=600,groups=groups)
    write_result_text((relocate_path(OUT/'summary.json')), json.dumps(summary,indent=2)+'\n')
    with (relocate_path(OUT/'step_metrics.csv')).open('w') as f:
        writer=csv.writer(f);writer.writerow(['model','dataset','sample','step','token_id','token','raw_probability','raw_rank','processed_probability','top30_mass','entropy_nats'])
        for r in records:
            for t in r['steps']:writer.writerow([r['model_key'],r['sample']['dataset'],r['sample']['sample_key'],t['step'],t['token_id'],t['token'],t['raw_selected_probability'],t['raw_selected_rank'],t['processed_selected_probability'],t['raw']['top30_mass'],t['raw']['entropy_nats']])
    lines=['# TimeLens pilot: full-head autoregressive top-30','',f"기존 pilot 100개 × 3개 벤치마크 × 2개 모델 = 600개 응답, 총 {summary['total_steps']:,}개 생성 step을 기록했다. EOS step도 포함한다.",'',
        '기존 `experiments/zero-shot/predictions/cohorts/pilot_300/samples.json`의 seed 42 표본을 그대로 사용했다. 별도의 원본 TimeLens pilot manifest는 발견되지 않아 앞선 실험에서 정한 표본이며, 이번 실험에서 재추출하지 않았다.', '',
        '모든 600개 응답에서 생성 token ID 전체가 저장된 기존 zero-shot full-head 결과와 정확히 일치했다. 따라서 아래 grounding 지표도 기존 full-head와 동일하다. 숫자 제한 head를 적용하지 않았다.', '',
        '| 모델 | 벤치마크 | N | R@0.3 (%) | R@0.5 (%) | R@0.7 (%) | mIoU (%) | 평균 생성 토큰 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['model']} | {g['dataset']} | {g['n']} | {g['r30']:.2f} | {g['r50']:.2f} | {g['r70']:.2f} | {g['miou']:.2f} | {g['mean_output_tokens']:.2f} |")
    lines+=['','확률은 BF16 모델의 logits를 FP32로 변환하여 **전체 LM-head 어휘**에 softmax를 적용한 값이다. Top-30 안에서 재정규화하지 않았다. HTML 기본 표시는 generation processor 이전 raw 분포이며, 보정 후 분포도 선택할 수 있다. Qwen2.5의 repetition penalty는 1.05, Qwen3는 1.0이다. Greedy decoding이므로 이 분포는 실제 무작위 샘플링을 의미하지 않는다.','',
        '| 모델 | 벤치마크 | 모든 step 평균 선택 확률 (%) | 숫자 step 평균 선택 확률 (%) | 평균 top-30 질량 (%) | 평균 엔트로피 (nat) | raw 최대 확률보다 낮은 토큰 선택 |','|---|---|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['model']} | {g['dataset']} | {100*g['selected_probability']:.2f} | {100*g['numeric_selected_probability']:.2f} | {100*g['top30_mass']:.2f} | {g['entropy_nats']:.3f} | {g['raw_argmax_disagreements']} |")
    lines+=['','동률 토큰의 top-k 표시 순서는 greedy argmax의 선택 순서와 다를 수 있다. 선택 확률이 최대 확률과 같은지 검증하며 동률은 불일치로 세지 않는다. 숫자 step은 토큰 문자열이 공백을 제외하고 숫자 또는 소수인 경우로 정의했다. 각 step에 동일한 가중치를 주므로 긴 응답의 영향이 더 크다. 자연어 정형 문구의 높은 확률이 전체 평균을 높일 수 있어 숫자 step을 별도로 제시했다. 토큰 확률은 영상 구간 정답 확률이나 calibration 지표가 아니다.','']
    for ds in dict.fromkeys(s['dataset'] for s in samples):
        a=next(g for g in groups if g['model']=='qwen25' and g['dataset']==ds);b=next(g for g in groups if g['model']=='qwen3' and g['dataset']==ds)
        lines.append(f"- {ds}: Qwen3의 mIoU는 Qwen2.5보다 {b['miou']-a['miou']:+.2f} percentage points, R@0.5는 {b['r50']-a['r50']:+.2f} percentage points 차이였다.")
    lines+=['','동일한 입력·프롬프트·프레임 전처리와 greedy 설정(max_new_tokens=512)을 재사용했다. 공식 시간 파서 및 반올림 후 첫 구간 IoU를 사용했다. GPU는 H200에서 CUDA MPS로 두 모델의 4개 shard를 병렬 실행했다. 확률 기록은 logits를 변경하지 않는 forward hook과 processor로 수행했다. 기록 비용이 포함되므로 실행 시간은 모델 순수 추론 속도 비교용이 아니다.','',
        '[영상 및 step별 top-30 탐색](report.html) · [정량 JSON](summary.json) · [step별 CSV](step_metrics.csv)','',
        'HTML은 로컬 HTTP 서버로 열고 모델별 원본 JSON을 직접 읽는다. 별도의 응답 복제본을 만들지 않는다. 영상은 브라우저용 무음 H.264 사본이며 모델 입력 원본은 변경하지 않았다. GT/예측 구간 버튼으로 영상 시각을 이동할 수 있다. 생성 step 번호는 영상 시각이 아니다.']
    if (relocate_path(OUT/'mmco_diagnosis/results.json')).exists():
        audit=read(relocate_path(OUT/'mmco_diagnosis/results.json'))
        affected=[r for r in audit if r['sparse']['mmco_messages']]
        different=[r for r in audit if r.get('sampled_frame_hash_mismatches')]
        lines+=['', f'디코더 진단: {len(audit)}개 고유 영상 중 {len(affected)}개에서 sparse Decord MMCO 경고가 재현됐고, 그중 {len(different)}개는 순차 Decord와 표본 RGB hash가 달랐다. 기존 baseline과 동일 생성 토큰이라는 사실은 입력 영상의 디코딩 정확성까지 보증하지 않는다. 원본 입력/decoder를 그대로 유지한 비교 결과이며, 자세한 범위와 증거는 [디코더 진단 보고서](mmco_diagnosis/REPORT.md)를 참고한다.']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
    write_result_text((relocate_path(OUT/'manifest.js')), 'window.MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';\n')
    write_result_text((relocate_path(OUT/'report.html')), HTML)
    write_result_text((relocate_path(OUT/'viewer.js')), JS)
    write_result_text((relocate_path(OUT/'validation.json')), json.dumps(dict(passed=True,runs=600,steps=summary['total_steps'],all_baseline_tokens_identical=True,top30_sorted_unique_normalized=True),indent=2))
    print(json.dumps(summary),flush=True)

HTML='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TimeLens · Full-head top-30</title><style>
*{box-sizing:border-box}body{margin:0;background:#101722;color:#e3eaf5;font:15px system-ui}header{padding:20px;border-bottom:1px solid #344257}a{color:#8ac8ff}main{display:grid;grid-template-columns:290px 1fr;gap:22px;padding:20px}aside{height:86vh;overflow:auto}select,input,button{background:#1e2a3d;color:inherit;border:1px solid #53647d;border-radius:5px;padding:8px;margin:3px;cursor:pointer}input{width:95%}#cases button{display:block;text-align:left;width:96%}#cases .active,.selected{background:#315179}video{width:100%;max-height:420px;background:#000}section{min-width:0}#tokens{padding:10px 0;line-height:2.6}#tokens button{font-family:monospace;padding:3px 7px}#step{width:65%}table{width:100%;border-collapse:collapse}td,th{padding:7px;text-align:left;border-bottom:1px solid #344257}td:nth-child(3){white-space:pre-wrap;font-family:monospace;word-break:break-all}.bar{height:5px;background:#64b5f6}small,.muted{color:#b1bdd0}#details{white-space:pre-wrap}#status{color:#93caff}@media(max-width:800px){main{grid-template-columns:1fr}aside{height:220px}}
</style><header><strong>TimeLens · Full-head autoregressive top-30</strong> <a href="REPORT.md">정량 보고서 (MD)</a><br><small>각 벤치마크 100개 · Qwen2.5 / Qwen3 · 전체 어휘 softmax · top-30 재정규화 없음</small></header><main><aside><select id="dataset"><option value="">모든 벤치마크</option></select><input id="search" placeholder="질의 / sample ID 검색"><p id="count"></p><div id="cases"></div></aside><section><h2 id="query"></h2><div id="status">데이터 불러오는 중</div><video id="video" controls muted preload="metadata"></video><p><button id="gt">GT 구간 재생</button><button id="pred">예측 구간 재생</button><label><input type="checkbox" id="loop" style="width:auto">구간 반복</label></p><p id="details"></p><select id="model"><option value="qwen3">Qwen3-VL-8B</option><option value="qwen25">Qwen2.5-VL-7B</option></select><select id="mode"><option value="raw">Raw LM-head 확률</option><option value="processed">Generation processor 보정 후 확률</option></select><p id="answer"></p><div id="tokens"></div><button id="prev">이전 step</button><input id="step" type="range" min="0" value="0"><button id="next">다음 step</button><p id="stats"></p><small>공백: ␠ · 줄바꿈: ↵ · EOS 포함. 생성 step은 영상 시각이 아닙니다. 토큰 확률은 구간 정답 확률이 아닙니다.</small><table><thead><tr><th>순위</th><th>Token ID</th><th>토큰</th><th>확률</th></tr></thead><tbody id="top"></tbody></table></section></main><script src="manifest.js"></script><script src="viewer.js"></script></html>'''
JS=r'''
const $=id=>document.getElementById(id),cache={},waiting={};let current=0,trace=null,step=0,segment=null;
const visible=t=>t.replace(/ /g,'␠').replace(/\n/g,'↵');const pct=p=>(p*100>=.0001?(p*100).toFixed(5):(p*100).toExponential(3))+'%';
window.receiveTrace=(key,data)=>{cache[key]=data;if(waiting[key]){waiting[key](data);delete waiting[key]}};
function load(key){if(cache[key])return Promise.resolve(cache[key]);return fetch('artifacts/samples/'+key+'.json').then(response=>{if(!response.ok)throw Error('Record load failed: '+key);return response.json()}).then(data=>{cache[key]=data;return data})}
for(const ds of [...new Set(MANIFEST.map(s=>s.dataset))]){let o=document.createElement('option');o.value=o.textContent=ds;$('dataset').appendChild(o)}
function list(){let count=0;$('cases').replaceChildren();MANIFEST.forEach((s,i)=>{if($('dataset').value&&s.dataset!==$('dataset').value)return;if(!(s.query+' '+s.sample_key).toLowerCase().includes($('search').value.toLowerCase()))return;count++;let b=document.createElement('button');b.textContent=s.sample_key+' · '+s.query;b.className=i===current?'active':'';b.onclick=()=>select(i);$('cases').appendChild(b)});$('count').textContent=count+' / 300 samples'}
async function select(i){let changed=i!==current||!$('video').getAttribute('src');current=i;let s=MANIFEST[i],key=$('model').value+'/'+s.sample_key;trace=null;$('top').replaceChildren();$('tokens').replaceChildren();$('status').textContent='불러오는 중…';$('query').textContent=s.query;if(changed){$('video').src=s.media;segment=null}list();try{let r=await load(key);if(key!==$('model').value+'/'+MANIFEST[current].sample_key)return;trace=r;step=0;$('status').textContent=s.sample_key+' · 기존 baseline token 전체 일치';$('answer').textContent=r.answer;$('details').textContent='GT '+JSON.stringify(s.span)+' s | 예측 '+JSON.stringify(r.timestamps)+' s | IoU '+r.iou.toFixed(4)+' | annotation duration '+s.duration.toFixed(2)+' s';$('step').max=r.steps.length-1;$('tokens').replaceChildren();r.steps.forEach((t,j)=>{let b=document.createElement('button');b.textContent=visible(t.token);b.title='Step '+(j+1)+' / ID '+t.token_id;b.onclick=()=>{step=j;render()};$('tokens').appendChild(b)});render()}catch(e){$('status').textContent=e.message}}
function render(){if(!trace)return;step=Math.max(0,Math.min(trace.steps.length-1,step));let t=trace.steps[step],p=t[$('mode').value];$('step').value=step;[...$('tokens').children].forEach((b,i)=>b.className=i===step?'selected':'');$('stats').textContent='Step '+(step+1)+' / '+trace.steps.length+' · 선택 '+visible(t.token)+' (ID '+t.token_id+') · raw P '+pct(t.raw_selected_probability)+' / rank '+t.raw_selected_rank+' · 보정 후 P '+pct(t.processed_selected_probability)+' · top-30 질량 '+pct(p.top30_mass)+' · 나머지 '+pct(Math.max(0,1-p.top30_mass))+' · entropy '+p.entropy_nats.toFixed(3)+' nat';$('top').replaceChildren();p.top30.forEach((x,i)=>{let tr=document.createElement('tr');if(x.id===t.token_id)tr.className='selected';[i+1,x.id,visible(x.token),pct(x.probability)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td)});let bar=document.createElement('div');bar.className='bar';bar.style.width=(100*x.probability)+'%';tr.lastChild.appendChild(bar);$('top').appendChild(tr)})}
function playSpan(span){if(!span)return;let v=$('video'),duration=Number.isFinite(v.duration)?v.duration:MANIFEST[current].duration;segment=[Math.max(0,Math.min(duration-.01,span[0])),Math.max(0,Math.min(duration,span[1]))];if(segment[1]<=segment[0]){$('status').textContent='이 구간은 재생 가능한 영상 범위 밖입니다.';segment=null;return}v.currentTime=segment[0];v.play().catch(e=>$('status').textContent=e.message)}
$('video').ontimeupdate=()=>{if(segment&&$('video').currentTime>=segment[1]){if($('loop').checked){$('video').currentTime=segment[0];$('video').play()}else{$('video').pause();segment=null}}};
$('gt').onclick=()=>playSpan(MANIFEST[current].span[0]);$('pred').onclick=()=>trace&&playSpan(trace.timestamps[0]);$('dataset').onchange=list;$('search').oninput=list;$('model').onchange=()=>select(current);$('mode').onchange=render;$('step').oninput=()=>{step=+$('step').value;render()};$('prev').onclick=()=>{step--;render()};$('next').onclick=()=>{step++;render()};select(0);
'''
if __name__=='__main__':build()
