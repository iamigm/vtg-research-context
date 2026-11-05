"""Validate rollout candidates, compare paired baselines, and build video viewer."""
from result_layout import relocate_path, write_result_text
from rollout_method_docs import formalize
from pathlib import Path
import json,statistics,hashlib,os,gzip,base64
import numpy as np
from timelens_paths import PILOT_SAMPLES
from timelens_top30_report import HTML,JS
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/timelens_rollout')
def read(p):return json.loads(p.read_text())
def avg(x):return statistics.mean(x)
def build():
    config=read(relocate_path(OUT/'config.json'));samples=read(PILOT_SAMPLES);media=read(relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300/media_manifest.json'));manifest=[];records=[];steps_total=0
    assert len(samples)==300 and hashlib.sha256(PILOT_SAMPLES.read_bytes()).hexdigest()==config['manifest_sha256']
    for s in samples:
        manifest.append(dict(s,media=media[s['video_path']]['media']))
        dest=relocate_path(OUT/media[s['video_path']]['media']);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300'/media[s['video_path']]['media']),dest)
        for model in ['qwen25','qwen3']:
            for variant in ['top_k','top_p']:
                for roll in range(config['rollouts_per_variant']):
                    r=read(relocate_path(OUT/'records'/model/s['sample_key']/f'{variant}_{roll}.json'))
                    assert r['sample']==s and r['variant']==variant and r['rollout']==roll
                    assert hashlib.sha256(relocate_path(Path(r['baseline_path'])).read_bytes()).hexdigest()==r['baseline_sha256']
                    assert r['generated_ids']==[t['token_id'] for t in r['steps']]
                    for t in r['steps']:
                        if 'dense' in t:
                            d=t['dense'];arrays={k:np.frombuffer(gzip.decompress(base64.b64decode(v)),dtype='<f4') for k,v in d['arrays'].items()}
                            for k,v in d['aliases'].items():arrays[k]=arrays[v]
                            probs=arrays['probability'];assert len(probs)==d['vocab_size']
                            ids=np.arange(len(probs)) if d['ids']=='all' else np.asarray(d['ids'])
                            assert len(ids)==t['candidate_count'] and np.isfinite(probs).all() and (probs>=0).all()
                            assert abs(float(probs.sum())-1)<2e-5 and abs(float(probs[ids].sum())-1)<2e-5
                            assert float(probs[t['token_id']])==t['selected_probability']>0
                            assert int((probs>0).sum())==t['positive_probability_count']
                        else:
                            cs=t['candidates'];assert len(cs)==len({c['id'] for c in cs})==t['candidate_count']
                            probs=[c['probability'] for c in cs];assert probs==sorted(probs,reverse=True)
                            assert all(0<=p<=1 for p in probs) and abs(sum(probs)-1)<2e-5
                            assert any(c['id']==t['token_id'] and c['probability']==t['selected_probability'] and c['probability']>0 for c in cs)
                            assert len([c for c in cs if c['probability']>0])==t['positive_probability_count']
                    steps_total+=len(r['steps'])
                    r['mean_candidate_count']=avg([t['candidate_count'] for t in r['steps']]);r['singleton_fraction']=avg([t['positive_probability_count']==1 for t in r['steps']]);r.pop('steps');records.append(r)
    groups=[]
    for model in ['qwen25','qwen3']:
        for variant in ['top_k','top_p']:
            for ds in dict.fromkeys(s['dataset'] for s in samples):
                rs=[r for r in records if r['model_key']==model and r['variant']==variant and r['sample']['dataset']==ds]
                by={s['sample_key']:[r for r in rs if r['sample']['sample_key']==s['sample_key']] for s in samples if s['dataset']==ds}
                assert len(by)==100 and all(len(v)==config['rollouts_per_variant'] for v in by.values())
                groups.append(dict(model=model,variant=variant,dataset=ds,queries=100,rollouts=len(rs),miou=100*avg([r['iou'] for r in rs]),r50=100*avg([r['iou']>=.5 for r in rs]),baseline_miou=100*avg([v[0]['baseline']['iou'] for v in by.values()]),baseline_r50=100*avg([v[0]['baseline']['iou']>=.5 for v in by.values()]),oracle_any5_r50=100*avg([any(r['iou']>=.5 for r in v) for v in by.values()]),mean_unique_answers=avg([len({r['answer'] for r in v}) for v in by.values()]),mean_candidate_count=avg([r['mean_candidate_count'] for r in rs]),singleton_fraction=avg([r['singleton_fraction'] for r in rs])))
    summary=dict(queries=300,runs=len(records),steps=steps_total,groups=groups)
    write_result_text((relocate_path(OUT/'summary.json')), json.dumps(summary,indent=2)+'\n')
    lines=['# TimeLens stochastic rollout: top-k vs top-p','',f'동일 pilot 300개 × 두 모델 × 두 sampling 방식 × {config["rollouts_per_variant"]}회 = {len(records):,}개 응답, {steps_total:,}개 생성 단계. 기존 zero-shot과 같은 입력 hash를 확인했다.','',
        '**두 실험은 top-k만 적용(p=1) / top-p만 적용(k=0)하는 분리 실험이다.** Qwen3 k=20/p=0.8, Qwen2.5 라이브러리 기본 k=50/p=1.0을 사용한다. 모델 카드에서 k와 p를 함께 쓰는 설정 자체의 재현은 아니다. 매 단계에는 truncation 후 유한 score를 가진 후보 전체를 저장했다. 고정 top-30 기록은 하지 않는다.','',
        'Qwen3 [모델 카드](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)는 일반 태스크에 temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0, presence_penalty=1.5를 제시한다. Presence penalty는 이미 생성된 토큰에 한 번씩 차감한다. 최대 길이는 zero-shot과 비교하도록 512토큰을 유지했다.','',config['qwen25_setting_note'],'',
        '설정 원문과 retrieval 시각/해시는 `sources/`에 보존했다. 실제 설정은 [config.json](config.json)에 기록했다. 각 sample/rollout의 seed는 sample ID·rollout 번호·42에서 SHA-256으로 결정하며 두 방식과 모델 사이에 동일하게 대응시켰다. 동일 seed는 서로 다른 분포에서 동일 응답을 보장하지 않는다.','',
        '| 모델 | 방식 | 벤치마크 | baseline mIoU (%) | rollout 평균 mIoU (%) | baseline R@.5 (%) | rollout 평균 R@.5 (%) | oracle any-of-5 R@.5 (%) | 평균 고유 응답 수 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['model']} | {g['variant']} | {g['dataset']} | {g['baseline_miou']:.2f} | {g['miou']:.2f} | {g['baseline_r50']:.2f} | {g['r50']:.2f} | {g['oracle_any5_r50']:.2f} | {g['mean_unique_answers']:.2f} |")
    lines+=['','평균 지표는 query마다 rollout 5개를 동일 가중 평균한 후 100개 query를 집계한 값이다. 500개 rollout을 독립 query로 해석하지 않는다. oracle any-of-5는 GT를 보고 가장 좋은 답을 고르는 상한이며 실제 배포 성능이 아니다.','',
        '| 모델 | 방식 | 벤치마크 | 평균 후보 수 | 양의 확률 후보가 1개인 step 비율 (%) |','|---|---|---|---:|---:|']
    for g in groups:lines.append(f"| {g['model']} | {g['variant']} | {g['dataset']} | {g['mean_candidate_count']:.2f} | {100*g['singleton_fraction']:.2f} |")
    lines+=['','후보 수 통계는 응답 안에서 step 평균을 내고 응답 간 평균했다. top-k 경계의 logit 동률은 Hugging Face 구현에 따라 k개보다 많이 남을 수 있다. HTML은 해당 후보도 모두 표시한다. top-p는 temperature·penalty 적용 후 분포에 적용하며 다른 top-k 제한을 사용하지 않는다.','',
        'sampling 확률은 필터 후 분포를 재정규화한 실제 추출 확률이다. 필터 전 확률과 raw LM-head 확률을 함께 표시해 구분한다. 유한 score라도 FP32 softmax underflow로 확률 0이 될 수 있어 유한 후보 수와 양의 확률 후보 수를 따로 기록했다. 토큰 확률을 구간 정답 확률로 해석하면 안 된다.','']
    for model in ['qwen25','qwen3']:
        for var in ['top_k','top_p']:
            gs=[g for g in groups if g['model']==model and g['variant']==var]
            lines.append(f"- {model} {var}: 세 벤치마크 평균 mIoU 차이 {avg([g['miou']-g['baseline_miou'] for g in gs]):+.2f} pp, 평균 고유 응답 {avg([g['mean_unique_answers'] for g in gs]):.2f}/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.")
    lines+=['','기존 decoder와 프레임 전처리를 유지했다. 앞선 MMCO 진단에서 sparse Decord와 순차 Decord 사이에 9개 영상의 표본 픽셀 차이가 있었으므로 입력 디코딩 품질에 관한 기존 한계도 유지된다.','',
        '[영상·생성 단계별 전체 후보 집합 HTML](report.html) · [정량 JSON](summary.json) · [진행 로그](logs/progress.log)']
    write_result_text((relocate_path(OUT/'REPORT.md')), formalize('\n'.join(lines)+'\n','stochastic',300))
    write_result_text((relocate_path(OUT/'manifest.js')), 'window.MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';\n')
    html=HTML.replace('Full-head autoregressive top-30','Stochastic rollout · top-k / top-p').replace('Full-head top-30','Stochastic rollout').replace('전체 어휘 softmax · top-30 재정규화 없음','단계별 실제 후보 전체 · sampling 확률 / 필터 전 확률 / raw 확률')
    a=html.index('<select id="mode">');b=html.index('</select>',a)+len('</select>')
    html=html[:a]+'<select id="variant"><option value="top_k">Top-k only</option><option value="top_p">Top-p only</option></select><select id="rollout">'+''.join(f'<option value="{r}">Rollout {r+1}</option>' for r in range(config['rollouts_per_variant']))+'</select>'+html[b:]
    html=html.replace('<table>', '<p><input id="candidateSearch" placeholder="후보 토큰 / ID 검색"><button id="candidatePrev">이전 후보 페이지</button><button id="candidateNext">다음 후보 페이지</button><span id="candidatePage"></span></p><table>').replace('<script src="viewer.js">','<script src="vocab/qwen25.js"></script><script src="vocab/qwen3.js"></script><script src="viewer.js">')
    html=html.replace('<th>확률</th>','<th>Sampling 확률</th><th>필터 전 확률</th><th>Raw 확률</th>')
    js=JS.replace("$('model').value+'/'+s.sample_key","$('model').value+'/'+$('variant').value+'/'+s.sample_key+'/'+$('rollout').value").replace("$('model').value+'/'+MANIFEST[current].sample_key","$('model').value+'/'+$('variant').value+'/'+MANIFEST[current].sample_key+'/'+$('rollout').value")
    js=js.replace("' · 기존 baseline token 전체 일치'","' · '+r.variant+' · rollout '+(r.rollout+1)+' · seed '+r.seed")
    js=js.replace("+' | annotation duration '+s.duration.toFixed(2)+' s'","+' | baseline IoU '+r.baseline.iou.toFixed(4)+' / '+r.baseline.answer+' | T='+r.settings.temperature+', k='+r.settings.top_k+', p='+r.settings.top_p")
    a=js.index('function render()');b=js.index('function playSpan',a)
    js=js[:a]+RENDER+'\n'+js[b:]
    js=js.replace("$('mode').onchange=render;","$('variant').onchange=()=>select(current);$('rollout').onchange=()=>select(current);")
    js=js.replace("window.receiveTrace=(key,data)=>{", "window.receiveTrace=(key,data)=>{if(Object.keys(cache).length>3)delete cache[Object.keys(cache)[0]];")
    a=js.index('function load(');b=js.index('\n',a);js=js[:a]+"async function load(key){if(cache[key])return cache[key];const [model,variant,sample,roll]=key.split('/');const url='records/'+model+'/'+sample+'/'+variant+'_'+roll+'.json';const response=await fetch(url);if(!response.ok)throw Error('Record load failed: '+url);const data=await response.json();if(Object.keys(cache).length>3)delete cache[Object.keys(cache)[0]];cache[key]=data;return data}"+js[b:]
    write_result_text((relocate_path(OUT/'report.html')), html);write_result_text((relocate_path(OUT/'viewer.js')), js)
    write_result_text((relocate_path(OUT/'validation.json')), json.dumps(dict(passed=True,runs=len(records),steps=steps_total,candidate_probabilities_valid=True,selected_token_in_candidates=True),indent=2)+'\n')
RENDER=r'''
let candidateRows=[],candidatePage=0,renderSerial=0;
async function unpack(d){if(d.decoded)return d.decoded;let arrays={};for(const [k,v] of Object.entries(d.arrays)){let bytes=Uint8Array.from(atob(v),c=>c.charCodeAt(0));let stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));arrays[k]=new Float32Array(await new Response(stream).arrayBuffer())}for(const [k,v] of Object.entries(d.aliases))arrays[k]=arrays[v];d.decoded=arrays;return arrays}
function renderCandidates(){let term=$('candidateSearch').value.toLowerCase();let rows=term?candidateRows.filter(x=>x.token.toLowerCase().includes(term)||String(x.id)===term):candidateRows;let pages=Math.max(1,Math.ceil(rows.length/100));candidatePage=Math.min(Math.max(0,candidatePage),pages-1);$('candidatePage').textContent=(candidatePage+1)+' / '+pages+' · '+rows.length+' candidates (100/page)';$('top').replaceChildren();rows.slice(candidatePage*100,(candidatePage+1)*100).forEach((x,i)=>{let tr=document.createElement('tr');if(trace&&x.id===trace.steps[step].token_id)tr.className='selected';[candidatePage*100+i+1,x.id,visible(x.token),pct(x.probability),pct(x.prefilter_probability),pct(x.raw_probability)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td)});$('top').appendChild(tr)})}
async function render(){if(!trace)return;let serial=++renderSerial;step=Math.max(0,Math.min(trace.steps.length-1,step));let t=trace.steps[step],model=trace.model_key;$('step').value=step;[...$('tokens').children].forEach((b,i)=>b.className=i===step?'selected':'');$('stats').textContent='Step '+(step+1)+' / '+trace.steps.length+' · 선택 '+visible(t.token)+' (ID '+t.token_id+') · P '+pct(t.selected_probability)+' · 후보 전체 '+t.candidate_count+'개 / 양의 확률 '+t.positive_probability_count+'개 · 필터 전 보존 질량 '+pct(t.retained_prefilter_mass);$('top').replaceChildren();candidatePage=0;if(t.dense){let a=await unpack(t.dense);if(serial!==renderSerial)return;let ids=t.dense.ids==='all'?Array.from({length:t.dense.vocab_size},(_,i)=>i):t.dense.ids;ids.sort((x,y)=>a.probability[y]-a.probability[x]||x-y);candidateRows=ids.map(id=>({id,token:window.VOCABS[model][id]||'<unused:'+id+'>',probability:a.probability[id],prefilter_probability:a.prefilter_probability[id],raw_probability:a.raw_probability[id]}))}else candidateRows=t.candidates;renderCandidates()}
$('candidateSearch').oninput=()=>{candidatePage=0;renderCandidates()};$('candidatePrev').onclick=()=>{candidatePage--;renderCandidates()};$('candidateNext').onclick=()=>{candidatePage++;renderCandidates()};
'''
if __name__=='__main__':build()
