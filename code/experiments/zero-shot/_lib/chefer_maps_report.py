"""CPU-only report and synchronized video viewer for Chefer decoder maps."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

PROJECT = Path(__file__).resolve().parents[1]
RELATIVE = Path('rollout/greedy_attribution/pilot/3/chefer_maps')
ARTIFACTS = PROJECT / 'artifacts/runs' / RELATIVE
OUTPUT = PROJECT / 'results' / RELATIVE
PAPER = 'https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf'
COMMIT = 'c3e578f76b954e8528afeaaee26de3f07e3fe559'
OFFICIAL = f'https://github.com/hila-chefer/Transformer-Explainability/tree/{COMMIT}'


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def relative(path, output, base=ARTIFACTS):
    if not path:
        return ''
    path = str(path)
    if path.startswith(('https://', 'http://')):
        return path
    # Browser video_url values are relative to the unified experiments server.
    if path.startswith('/zero-shot/'):
        path = str(PROJECT.parent / path.lstrip('/'))
    path = Path(path)
    if not path.is_absolute():
        path = Path(base) / path
    return quote(os.path.relpath(path.resolve(), Path(output).resolve()), safe='/')


def escape(value):
    return html.escape(str(value if value is not None else ''), quote=True)


def rows(selection, artifacts, output):
    result = []
    for sample in selection.get('samples', []):
        key = sample['sample_key']
        query_path = artifacts / 'queries' / f'{key}.json'
        query = read(query_path, {})
        display = sample.get('display') or {}
        video = display.get('video_url') or sample.get('video_path') or sample.get('video_url')
        timestamps = display.get('timestamps') or []
        row = sample.get('row') or {}
        target = query.get('target') or dict(
            step=sample.get('branch', {}).get('step'),
            token_id=sample.get('branch', {}).get('greedy_token'),
            text=display.get('target_token_text') or sample.get('branch', {}).get('greedy_token_text'),
            objective='logit')
        result.append(dict(
            key=key, dataset=sample.get('dataset'), query=row.get('query', key),
            duration=row.get('duration'), video=relative(video, output, artifacts),
            timestamps=timestamps, grid=display.get('grid'), input_hw=display.get('input_hw'),
            target=target, greedy=sample.get('greedy'), oracle=sample.get('oracle'),
            gt=row.get('span'), complete=bool(query.get('complete')),
            maps=relative(query.get('maps_file'), output, artifacts),
            raw=relative(query_path, output) if query_path.exists() else None,
            validation=query.get('validation'), cost=query.get('cost')))
    return result


def report(selection, status, entries, artifacts, output):
    complete = sum(entry['complete'] for entry in entries)
    state = status.get('state', 'queued')
    result = [
        '# Chefer decoder maps · pilot3', '',
        f'**지도 완료 {complete}/{len(entries)} query · 상태 `{state}`**', '',
        '기존 oracle-switch 극단 사례 3개에서 원래 greedy 분기 token의 logit을 설명한다. '
        '동일 입력·동일 target·동일 video time에서 attention rollout과 final C를 비교한다. '
        '3개는 성능 차이로 선택한 진단 사례이며 전체 benchmark의 평균을 대표하지 않는다.', '',
    ]
    if complete < len(entries):
        result += ['아직 생성되지 않은 지도는 미측정이다. GPU 대기 또는 실행 중인 사례에 빈 지도만 표시하며, 색상이나 수치를 임의로 채우지 않는다. 원영상과 저장된 frame timestamp는 대기 중에도 탐색할 수 있다.', '']
    result += [
        '## 비교하는 두 화면', '',
        '| 화면 | 정의 | 해석 |', '| --- | --- | --- |',
        '| Attention rollout | 원래 attention A의 head 평균에 residual I를 더하고 행 정규화 후 layer를 합성 | Target logit gradient·LRP를 사용하지 않는 baseline |',
        '| Final C | mean_h max((∂z_t/∂A)⊙R_A,0)에 I를 더해 전체 decoder layer 합성 | 선택 target logit의 gradient로 가중한 LRP attention 경로 |', '',
        '두 지도는 값의 단위와 변환이 달라 같은 색을 같은 절댓값으로 비교하지 않는다.', '',
        '## 수식과 target', '',
        'A_l^h는 layer l, head h의 causal attention matrix, R_l^h는 해당 matrix에 대한 LRP relevance, '
        'G_l^h=∂z_t/∂A_l^h는 선택한 greedy token logit z_t의 gradient다. [x]₊=max(x,0), '
        'E_h는 head의 산술평균이며 positive clipping은 head 평균 **전**에 적용한다. 모든 입력 token을 포함해 계산한 뒤 예측 위치 p의 visual-token 열만 지도에 추출한다.', '',
        'LRP의 output seed는 선택 token 위치의 one-hot1이다. Linear에는 공식 α=1 규칙, residual Add에는 relevance 재정규화, bilinear 연산에는 두 operand의 절반 분배를 사용한다. Gradient의 objective는 별도로 실제 logit z_t다. 0-based step s에서 predictor 위치는 p=N_prompt+s−1이다.', '',
        '- Attention baseline: B_l=RowNorm(I+E_h[A_l^h]), map=e_pᵀB₃₅⋯B₀.',
        '- Final C: C_l=I+E_h[[G_l^h⊙R_l^h]₊], map=e_pᵀC₃₅⋯C₀.',
        '',
        'Final C는 공식 LRP 구현에 맞춰 layer matrix의 행 정규화를 하지 않는다. '
        'Attention baseline에만 RowNorm을 적용하며 분모는 해당 행의 모든 token 열 합이다. '
        '역방향 vector–matrix product로 필요한 target 행을 계산하는 것은 이 행렬곱의 대수적 계산 방식이며 일부 token을 생략한 근사가 아니다. '
        '지도는 확률·GT IoU·자연어 근거의 비율이 아니며 높은 값만으로 causal effect를 증명하지 않는다.', '',
        '## Qwen decoder에 적용한 범위', '',
        '이는 Chefer et al.(CVPR2021)의 ViT/BERT 분류 구현을 Qwen의 decoder target logit에 이식한 분석이다. '
        'Vision encoder 전체에 LRP를 적용한 pixel attribution으로 부르지 않는다. Decoder visual token 위치에 배치된 contextual representation을 표시한다. '
        'DeepStack 경로의 relevance는 별도 분기로 분리한다. RMSNorm·SiLU의 relevance 전달은 identity, '
        'RoPE는 고정 linear z-rule, SwiGLU bilinear 연산은 두 operand에 절반씩 전달하는 이식 가정을 사용한다. '
        '원 논문과 architecture가 다르므로 원 논문의 relevance conservation·설명 품질이 이 모델에서 자동 보장된다고 주장하지 않는다. '
        'Forward 재현과 수치 검증은 query별 validation 원자료를 확인한다.', '',
        '## 영상·색상 읽기', '',
        '각 visual token의 시간 위치는 실제 입력 frame 두 장의 pair에 대응한다. 두 source frame에 같은 pair 지도를 표시하며 '
        '두 시점 사이의 새로운 지도를 보간하지 않는다. 재생 중에는 현재 video time에 가장 가까운 저장 input timestamp의 pair를 사용한다. '
        '수동으로 선택한 반복 timestamp의 input slot은 정지 상태에서 보존한다. Frame index는 위, 초 단위 소수 timestamp는 아래에 표시한다.', '',
        '원본 영상의 실제 가시 영역에 지도 전체 좌표를 맞춘다. 모델 입력은 direct resize를 사용하므로 원영상과 H×W 또는 종횡비가 다를 수 있다. '
        'Spatial cell은 contextual visual-token 위치이며 해당 원본 pixel 영역만의 영향이라고 해석하지 않는다.', '',
        '각 map의 모든 frame·cell 최대값 M을 고정해 color ratio=v/M으로 표시한다. '
        'Frame마다 최대값을 다시 맞추지 않는다. M=0이면 모든 값이0임을 명시한다. 다색 palette는 낮은 값에서 높은 값 순서다. '
        '투명도0%는 모든 cell을 완전 불투명하게 표시하며, 투명도100%는 원영상만 표시한다. '
        'Overlay의 alpha는 1−투명도/100이며 값의 크기는 alpha가 아닌 색에 반영한다.', '',
        '## 사례와 원자료', '',
        '| Query | Target step (0-based) | Token | 지도 |', '| --- | --- | --- | --- |',
    ]
    for entry in entries:
        text = str(entry['target'].get('text') or '').replace('|', '\\|').replace('\n', ' ')
        result.append(f'| {entry["key"]} | {entry["target"].get("step")} | `{text}` | {"완료" if entry["complete"] else "미측정"} |')
    result += ['', f'- [선정·원본 입력 참조]({relative(artifacts/"metadata/selection.json",output)})']
    for filename, label in [('config.json', '실행 설정'), ('status.json', '현재 상태')]:
        if (artifacts/'metadata'/filename).exists():
            result.append(f'- [{label}]({relative(artifacts/"metadata"/filename,output)})')
    result += ['', '## 논문과 공식 구현', '',
               f'- [Chefer, Gur, Wolf · Transformer Interpretability Beyond Attention Visualization · CVPR2021]({PAPER})',
               f'- [공식 Transformer-Explainability 구현 · commit {COMMIT[:12]}]({OFFICIAL})',
               f'- [공식 LRP layer 합성과 final C]({OFFICIAL.replace("/tree/", "/blob/")}/baselines/ViT/ViT_LRP.py)',
               f'- [공식 attention rollout baseline]({OFFICIAL.replace("/tree/", "/blob/")}/baselines/ViT/ViT_explanation_generator.py)', '']
    return '\n'.join(result)


STYLE = '''
main{max-width:1800px;margin:auto;padding:24px}h1,h2,h3,p{margin-top:0}.page-heading{margin:-24px -24px 22px}.eyebrow{font-size:11px;letter-spacing:.12em;color:var(--ui-accent)}.panel{padding:20px;margin:16px 0;background:var(--ui-surface);border:1px solid var(--ui-line);border-radius:12px;min-width:0}.small{font-size:12px;line-height:1.8;color:var(--ui-muted)}.notice{padding:12px 16px;border-left:3px solid #b88c42;background:#faf7ef;font-size:13px;line-height:1.8}.toolbar{display:flex;flex-wrap:wrap;gap:12px 22px;align-items:center}.toolbar label{display:flex;align-items:center;gap:8px;font-size:12px}select,button{padding:8px}#sample-select{max-width:100%;width:550px}#target-description{font-size:13px;margin:14px 0;line-height:1.8}.comparison{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px}.map-card{border:1px solid var(--ui-line);border-radius:10px;overflow:hidden;min-width:0;background:var(--ui-surface)}.map-heading{padding:15px 15px 12px;min-height:150px}.map-heading h3{font-size:17px;margin:0 0 7px}.map-heading p{font-size:11px;color:var(--ui-muted);line-height:1.7;margin:0}.map-heading .toolbar{gap:8px;margin-top:8px}.map-heading select{padding:4px;max-width:100%;font-size:11px}.video-stage{position:relative;width:100%;height:280px;background:#101817;overflow:hidden}.video-stage video{display:block;width:100%;height:100%;object-fit:contain}.map-overlay{position:absolute;pointer-events:none;image-rendering:pixelated;z-index:2}.map-overlay[hidden]{display:none!important}.map-empty{position:absolute;left:12px;top:12px;padding:6px 9px;border-radius:5px;background:rgba(255,255,255,.92);color:#34453c;font-size:11px;pointer-events:none;z-index:3}.map-empty[hidden]{display:none!important}.map-meta{font-size:11px;line-height:1.8;padding:12px 14px;min-height:68px;color:var(--ui-muted)}.legend-gradient{height:7px;background:linear-gradient(90deg,#30123b,#3250b4,#14b496,#c8e628,#f57814,#820014);border-radius:4px;margin-bottom:5px}.legend-labels{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}.player-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:16px}#time-seek{flex:1;min-width:160px}#time-readout{font-size:12px;font-variant-numeric:tabular-nums;min-width:110px}.frame-timeline{display:flex;gap:4px;overflow-x:auto;padding:6px 2px 12px;margin-top:12px}.frame-button{padding:6px 8px!important;min-width:58px;font-size:11px;white-space:nowrap}.frame-button b,.frame-button small{display:block}.frame-button small{font-size:10px;color:var(--ui-muted);margin-top:3px}.frame-button.selected{background:var(--ui-soft)!important;border-color:var(--ui-accent)!important;outline:1px solid var(--ui-accent)}.frame-button.paired{box-shadow:inset 0 -3px #b0cbbd}.frame-button .repeat{font-size:9px}#frame-summary{font-size:12px;line-height:1.8;min-height:22px}#frame-select{max-width:280px}.source-links a{font-size:12px;margin-right:16px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid var(--ui-line);text-align:left}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:280px;overflow:auto}summary{cursor:pointer;font-size:13px;font-weight:600}.embedded .page-heading{display:none}body.embedded main{padding-top:8px}#sample-state:empty{display:none}input[type=range]{accent-color:var(--ui-accent)}
@media(max-width:1000px){.comparison{grid-template-columns:1fr}.map-heading{min-height:0}.video-stage{height:310px}}@media(max-width:600px){main{padding:14px}.page-heading{margin:-14px -14px 15px}.panel{padding:14px}.toolbar{gap:10px}.toolbar label{flex-wrap:wrap}.comparison{gap:12px}.video-stage{height:230px}#sample-select{width:100%}.player-controls{gap:8px}.map-heading{padding:12px}#frame-select{max-width:225px}}
'''


SCRIPT = r'''
(() => {
  const cfg = JSON.parse(document.getElementById('viewer-data').textContent);
  const $ = id => document.getElementById(id);
  const videos = [...document.querySelectorAll('.map-video')];
  const names = ['attention', 'final'];
  const cache = new Map();
  let current = null, payload = null, generation = 0, manualFrame = null, manualTime = null;
  let syncing = false, selectedFrame = -1, activeMaps = {}, maxima = {}, lastMapKey = '';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = value => !finite(value) ? '—' : value !== 0 && Math.abs(value) < .0001 ? value.toExponential(3) : value.toPrecision(5);
  const timestamps = () => current?.timestamps || [];
  if (window.self !== window.top) document.body.classList.add('embedded');

  function remember() {
    if (!current) return;
    const url = new URL(location.href);
    url.hash = new URLSearchParams({sample: current.key, frame: String(Math.max(0, selectedFrame))});
    history.replaceState(null, '', url);
    window.dispatchEvent(new Event('rollout:context'));
  }

  async function loadMaps(url) {
    if (!cache.has(url)) cache.set(url, (async () => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`지도 HTTP ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes[0] === 31 && bytes[1] === 139) {
        if (!window.DecompressionStream) throw new Error('이 브라우저는 gzip 지도 해제를 지원하지 않습니다.');
        return JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
      }
      return JSON.parse(new TextDecoder().decode(bytes));
    })());
    try { return await cache.get(url); } catch (error) { cache.delete(url); throw error; }
  }

  function maximum(map) {
    let result = 0;
    for (const frame of map || []) for (const row of frame || []) for (const value of row || []) if (finite(value)) result = Math.max(result, value);
    return result;
  }

  function updateMaps() {
    activeMaps = {attention: payload?.attention_rollout, final: payload?.final_c};
    maxima = Object.fromEntries(names.map(name => [name, maximum(activeMaps[name])]));
    lastMapKey = '';
    paintFrame();
    remember();
  }

  function nearestFrame(time) {
    const points = timestamps();
    let best = 0, distance = Infinity;
    for (let index = 0; index < points.length; index++) {
      const next = Math.abs(points[index] - time);
      if (next < distance) { best = index; distance = next; }
    }
    return best;
  }

  function color(ratio) {
    const palette = [[48,18,59],[50,80,180],[20,180,150],[200,230,40],[245,120,20],[130,0,20]];
    const x = Math.max(0, Math.min(1, ratio)) * (palette.length - 1);
    const index = Math.min(palette.length - 2, Math.floor(x)), fraction = x - index;
    return palette[index].map((value, i) => Math.round(value + fraction * (palette[index + 1][i] - value)));
  }

  function alignOverlay(name) {
    const video = $(`${name}-video`), stage = video.parentElement, canvas = $(`${name}-map`);
    const width = video.videoWidth, height = video.videoHeight;
    if (!width || !height || !stage.clientWidth || !stage.clientHeight) return;
    const scale = Math.min(stage.clientWidth / width, stage.clientHeight / height);
    Object.assign(canvas.style, {left: `${(stage.clientWidth-width*scale)/2}px`, top: `${(stage.clientHeight-height*scale)/2}px`, width: `${width*scale}px`, height: `${height*scale}px`});
  }

  function appearance() {
    const transparency = Number($('transparency').value);
    $('transparency-value').value = `${transparency}%`;
    for (const name of names) {
      const canvas = $(`${name}-map`);
      canvas.style.opacity = String(1-transparency/100);
      canvas.hidden = !$('overlay-visible').checked || !activeMaps[name];
    }
  }

  function paintFrame() {
    if (!current) return;
    const time = videos[0].currentTime || 0;
    const preserveManual = manualFrame !== null && videos[0].paused && Math.abs(time-manualTime) < .08;
    const frame = preserveManual ? manualFrame : nearestFrame(time);
    selectedFrame = frame;
    const pair = Math.floor(frame/2), points = timestamps();
    const mapKey = String(pair);
    $('frame-select').value = String(frame);
    $('time-readout').textContent = `${time.toFixed(3)} / ${Number(current.duration || videos[0].duration || 0).toFixed(3)} s`;
    $('time-seek').value = String(time);
    $('frame-summary').textContent = points.length ? `Input slot ${frame} · ${points[frame].toFixed(3)} s · 공동 visual pair ${pair}: ${points[pair*2]?.toFixed(3)} / ${points[Math.min(pair*2+1,points.length-1)]?.toFixed(3)} s${frame > 0 && points[frame] === points[frame-1] ? ' · 반복 입력 slot' : ''}` : '입력 frame timestamp 준비 중';
    document.querySelectorAll('.frame-button').forEach(button => {
      button.classList.toggle('selected', Number(button.dataset.frame) === frame);
      button.classList.toggle('paired', Math.floor(Number(button.dataset.frame)/2) === pair);
      button.setAttribute('aria-pressed', String(Number(button.dataset.frame) === frame));
    });
    if (mapKey !== lastMapKey) {
      for (const name of names) {
        const map = activeMaps[name], values = map?.[pair], canvas = $(`${name}-map`), empty = $(`${name}-empty`);
        empty.hidden = !!values;
        empty.textContent = payload ? '이 지도는 미측정' : current.complete ? '지도 불러오는 중' : cfg.status.state === 'failed' ? '실행 실패 · 지도 미측정' : cfg.status.state === 'queued' ? 'GPU 대기 · 지도 미측정' : '계산 대기/진행 · 지도 미측정';
        if (!values?.length || !values[0]?.length) {
          canvas.hidden = true;
          $(`${name}-maximum`).textContent = '미측정';
          continue;
        }
        canvas.width = values[0].length; canvas.height = values.length;
        const context = canvas.getContext('2d');
        const image = context.createImageData(canvas.width, canvas.height);
        for (let row=0; row<canvas.height; row++) for (let col=0; col<canvas.width; col++) {
          const value = values[row][col], rgb = color(maxima[name] > 0 ? value/maxima[name] : 0);
          const offset = (row*canvas.width+col)*4;
          image.data.set([...rgb, 255], offset);
        }
        context.putImageData(image,0,0);
        canvas.dataset.pair = String(pair); canvas.dataset.maximum = String(maxima[name]);
        $(`${name}-maximum`).textContent = `${number(maxima[name])}${maxima[name] === 0 ? ' · 전체 값0' : ''}`;
        alignOverlay(name);
      }
      lastMapKey = mapKey;
      appearance();
    }
  }

  function seek(time, frame = null) {
    videos.forEach(video => video.pause());
    manualFrame = frame; manualTime = frame === null ? null : time;
    syncing = true;
    for (const video of videos) if (Number.isFinite(time) && Math.abs(video.currentTime-time) > .001) video.currentTime = time;
    syncing = false;
    paintFrame(); remember();
  }

  async function play() {
    if (syncing) return;
    manualFrame = null; manualTime = null; syncing = true;
    const time = videos[0].currentTime;
    for (const video of videos.slice(1)) if (Math.abs(video.currentTime-time) > .03) video.currentTime = time;
    await Promise.allSettled(videos.map(video => video.play()));
    syncing = false;
    $('play-toggle').textContent = videos[0].paused ? '동기 재생' : '일시정지';
  }

  function pause() {
    syncing = true; videos.forEach(video => video.pause()); syncing = false;
    $('play-toggle').textContent = '동기 재생';
    remember();
  }

  async function chooseSample(key, desiredFrame = 0) {
    const token = ++generation;
    pause(); current = cfg.samples.find(sample => sample.key === key) || cfg.samples[0];
    if (!current) return;
    payload = null; activeMaps = {}; maxima = {}; lastMapKey = ''; selectedFrame = -1;
    $('comparison').dataset.sample=current.key;
    $('comparison').dataset.ready=current.complete?'loading':'pending';
    $('sample-select').value = current.key;
    $('query-title').textContent = current.query;
    const target = current.target;
    $('target-description').textContent = `원래 greedy step ${target.step} · token ${JSON.stringify(target.text || '')} (ID ${target.token_id}) · objective: ${target.objective || 'logit'} · 두 화면은 동일 target`;
    $('sample-state').textContent = current.complete ? '완료된 지도를 불러오는 중입니다.' : cfg.status.state === 'failed' ? '실행 실패 · 지도 미측정. 실패 원인은 현재 상태 원자료에 기록됩니다.' : cfg.status.state === 'queued' ? 'GPU 사용 가능 상태를 기다립니다. 원영상은 탐색할 수 있으며 지도는 아직 측정되지 않았습니다.' : '계산 대기 또는 진행 중인 사례입니다. 지도는 아직 측정되지 않았습니다.';
    $('input-description').textContent = `모델 입력 ${current.input_hw?.join(' × ') || '미확인'} · visual token은 실제 입력 frame 두 장의 contextual representation · 가장 가까운 저장 timestamp, 보간 없음`;
    $('time-seek').max = String(current.duration || timestamps().at(-1) || 1);
    $('frame-select').replaceChildren(...timestamps().map((time,index) => {
      const option = document.createElement('option'); option.value = String(index);
      option.textContent = `Input ${index} · ${time.toFixed(3)} s${index>0 && time===timestamps()[index-1] ? ' · 반복 slot' : ''}`; return option;
    }));
    $('frame-timeline').replaceChildren(...timestamps().map((time,index) => {
      const button = document.createElement('button'); button.type='button'; button.className='frame-button'; button.dataset.frame=String(index);
      const label=document.createElement('b');label.textContent=String(index);
      const seconds=document.createElement('small');seconds.textContent=`${time.toFixed(3)} s`;
      button.append(label,seconds);button.onclick=()=>seek(time,index);return button;
    }));
    $('raw-link').hidden=!current.raw; if(current.raw)$('raw-link').href=current.raw;
    $('maps-link').hidden=!current.maps; if(current.maps)$('maps-link').href=current.maps;
    $('validation-data').textContent=JSON.stringify({validation:current.validation,cost:current.cost},null,2);
    for (const video of videos) { if (video.getAttribute('src') !== current.video) { video.src=current.video; video.load(); } }
    desiredFrame=Math.max(0,Math.min(timestamps().length-1,desiredFrame));
    seek(timestamps()[desiredFrame] || 0, desiredFrame);
    updateMaps();
    if (current.complete && current.maps) {
      try {
        const loaded=await loadMaps(current.maps); if(token!==generation)return;
        payload=loaded; $('sample-state').textContent=''; $('comparison').dataset.ready='true';
        updateMaps();
      } catch(error) { if(token===generation){$('sample-state').textContent=`지도 로딩 실패: ${error.message}`;$('comparison').dataset.ready='error';} }
    } else $('comparison').dataset.ready='pending';
    $('comparison').dataset.sample=current.key;
  }

  videos.forEach((video,index) => {
    video.addEventListener('loadedmetadata',()=>{
      alignOverlay(names[index]);
      if(manualFrame!==null && finite(manualTime) && Math.abs(video.currentTime-manualTime)>.001)video.currentTime=manualTime;
      paintFrame();
    });
    video.addEventListener('play',()=>{
      if(syncing)return;
      if(index!==0 && Math.abs(videos[0].currentTime-video.currentTime)>.03)videos[0].currentTime=video.currentTime;
      play();
    });
    video.addEventListener('pause',()=>{if(!syncing && videos.some(other=>!other.paused))pause();});
    video.addEventListener('seeking',()=>{
      if(syncing)return;
      const time=video.currentTime;
      if(manualTime!==null && Math.abs(time-manualTime)>.08){manualFrame=null;manualTime=null;}
      syncing=true;videos.forEach(other=>{if(other!==video && Math.abs(other.currentTime-time)>.03)other.currentTime=time;});syncing=false;
      paintFrame();
    });
    video.addEventListener('timeupdate',()=>{
      if(index!==0)return;
      if(!video.paused){for(const other of videos.slice(1))if(Math.abs(other.currentTime-video.currentTime)>.12)other.currentTime=video.currentTime;}
      paintFrame();
    });
  });
  const observer=new ResizeObserver(()=>names.forEach(alignOverlay));
  document.querySelectorAll('.video-stage').forEach(stage=>observer.observe(stage));
  $('sample-select').onchange=()=>chooseSample($('sample-select').value);
  $('transparency').oninput=appearance; $('overlay-visible').onchange=appearance;
  $('frame-select').onchange=()=>{const index=Number($('frame-select').value);seek(timestamps()[index],index);};
  $('time-seek').oninput=()=>seek(Number($('time-seek').value));
  $('play-toggle').onclick=()=>videos[0].paused?play():pause();
  $('previous-frame').onclick=()=>{const index=Math.max(0,selectedFrame-1);seek(timestamps()[index],index);};
  $('next-frame').onclick=()=>{const index=Math.min(timestamps().length-1,selectedFrame+1);seek(timestamps()[index],index);};
  function restore(){const params=new URLSearchParams(location.hash.slice(1));chooseSample(params.get('sample'),Number(params.get('frame')||0));}
  window.addEventListener('hashchange',restore);
  restore();
})();
'''


def document(entries, status, output):
    complete = sum(entry['complete'] for entry in entries)
    options = ''.join(f'<option value="{escape(entry["key"])}">{escape(entry["key"])} · {escape(entry["query"])}</option>' for entry in entries)
    cards = []
    for name, title, description in [
        ('attention', 'Attention rollout', '원래 attention A · residual + 행 정규화 · target gradient 없음'),
        ('final', 'Final C', 'Gradient × LRP relevance · residual 경로와 모든 decoder layer 합성')]:
        cards.append(f'<article class="map-card" id="{name}-panel"><div class="map-heading"><h3>{title}</h3><p>{description}</p></div><div class="video-stage"><video id="{name}-video" class="map-video" controls muted playsinline preload="metadata" aria-label="{title} 원영상"></video><canvas id="{name}-map" class="map-overlay" aria-label="{title} heatmap" hidden></canvas><div id="{name}-empty" class="map-empty">지도 미측정</div></div><div class="map-meta"><div class="legend-gradient"></div><div class="legend-labels"><span>0 · 낮음</span><span id="{name}-maximum">미측정</span></div><div>해당 지도 전체 frame의 고정 색 범위</div></div></article>')
    data = json.dumps(dict(samples=entries,status=status), ensure_ascii=False).replace('</', '<\\/')
    return f'''<!doctype html><html lang="ko" data-experiment-detail><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chefer decoder maps · pilot3</title><style>{STYLE}</style><link rel="stylesheet" href="{escape(relative(PROJECT.parent/'_lib/experiment_theme.css',output))}"></head><body><main>
<header class="page-heading"><p class="eyebrow">GREEDY ATTRIBUTION / PILOT 3 / CHEFER MAPS</p><h1>같은 순간, attention rollout과 final C</h1><p>원래 attention 경로와 gradient로 가중한 LRP 경로를 비교합니다</p></header>
<p class="notice" id="study-status">지도 완료 {complete}/{len(entries)} query · {escape(status.get('state','queued'))}. 기존 극단3사례이며 아직 계산하지 않은 지도는 미측정으로 표시합니다.</p>
<section class="panel"><div class="toolbar"><label>Query<select id="sample-select">{options}</select></label></div><h2 id="query-title" style="margin-top:18px"></h2><p id="target-description"></p><p id="sample-state" class="notice"></p>
<div class="toolbar"><label><input type="checkbox" id="overlay-visible" checked> Heatmap 표시</label><label>투명도<input id="transparency" type="range" min="0" max="100" value="40"><output id="transparency-value">40%</output></label><span class="small">0%: Heatmap만 · 100%: 원영상만</span></div>
<div class="comparison" id="comparison">{''.join(cards)}</div>
<div class="player-controls"><button id="play-toggle" type="button">동기 재생</button><input id="time-seek" type="range" min="0" max="1" step="0.001" value="0" aria-label="두 영상 동기 탐색"><output id="time-readout"></output></div>
<div class="toolbar" style="margin-top:14px"><button id="previous-frame" type="button" aria-label="이전 입력 frame">←</button><label>Input frame<select id="frame-select"></select></label><button id="next-frame" type="button" aria-label="다음 입력 frame">→</button></div>
<p id="frame-summary"></p><div id="frame-timeline" class="frame-timeline" aria-label="입력 frame index와 실제 timestamp"></div>
<p class="small" id="input-description"></p><p class="small">각 지도의 모든 frame에서 최대값을 고정합니다. 색상은 해당 지도 내부의 상대 크기이며 panel 사이 절댓값 비교가 아닙니다. 지도는 contextual visual token 위치이며 pixel별 causal effect나 확률이 아닙니다.</p>
<details><summary>검증·원자료</summary><p class="source-links"><a id="raw-link" target="_blank" rel="noopener" hidden>Query 원자료 ↗</a><a id="maps-link" target="_blank" rel="noopener" hidden>지도 gzip JSON ↗</a></p><pre id="validation-data"></pre></details></section>
<p class="small">Chefer et al. CVPR2021의 decoder 적용 분석 · <a href="{PAPER}" target="_blank" rel="noopener">논문 ↗</a> · <a href="{OFFICIAL}" target="_blank" rel="noopener">공식 구현 ↗</a>. 수식·Qwen 이식 가정·정량 해석은 대표 viewer의 보고서 탭에서 확인합니다.</p>
</main><script type="application/json" id="viewer-data">{data}</script><script>{SCRIPT}</script></body></html>'''


def generate(artifacts=None, output=None):
    artifacts, output = Path(artifacts or ARTIFACTS), Path(output or OUTPUT)
    selection = read(artifacts/'metadata/selection.json')
    if not selection or len(selection.get('samples', [])) != 3:
        raise ValueError('The fixed three-query selection is required before publishing Chefer maps.')
    status = read(artifacts/'metadata/status.json', {'state':'queued'})
    entries = rows(selection, artifacts, output)
    write(output/'REPORT.md', report(selection, status, entries, artifacts, output))
    write(output/'detail.html', document(entries, status, output))
    return dict(queries=len(entries), complete=sum(entry['complete'] for entry in entries), report=str(output/'REPORT.md'), detail=str(output/'detail.html'))


def check_viewer():
    """Check the actual published snapshot; pending UI is not GPU validation."""
    import datetime
    import gzip
    import hashlib
    import math
    import re
    import sys

    shared = PROJECT.parent/'_shared/timelens'
    sys.path.insert(0, str(shared/'browser_deps'))
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(shared/'browser_cache')
    os.environ['LD_LIBRARY_PATH'] = str(shared/'browser_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['FONTCONFIG_FILE'] = str(shared/'browser_libs/fonts.conf')
    from playwright.sync_api import sync_playwright

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    html_text = (OUTPUT/'detail.html').read_text()
    snapshot = json.loads(re.search(r'<script type="application/json" id="viewer-data">(.*?)</script>', html_text, re.S).group(1))
    hashes = {name: hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() for name in ('detail.html','REPORT.md')}
    base = 'http://127.0.0.1:8780'
    detail = base+'/zero-shot/results/'+RELATIVE.as_posix()+'/detail.html'
    view = 'pilot/3/chefer_maps/detail.html'
    family = base+'/zero-shot/results/rollout/greedy_attribution/viewer.html?view='+quote(view,safe='')
    errors, media, measured = [], [], []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--no-sandbox','--disable-dev-shm-usage'])
            page = browser.new_page(viewport=dict(width=1600,height=1100))
            page.on('pageerror', lambda error: errors.append(str(error)))

            def no_overflow(label):
                dimensions = page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
                require(dimensions['scroll'] <= dimensions['width']+1, f'{label}: page overflow {dimensions}')

            for sample in snapshot['samples']:
                key = sample['key']
                response = page.goto(detail+'#sample='+key, wait_until='networkidle')
                require(response is None or response.status == 200, 'Detail page HTTP failed')
                page.wait_for_function('key=>document.getElementById("comparison").dataset.sample===key && ["true","pending"].includes(document.getElementById("comparison").dataset.ready)', arg=key, timeout=60000)
                require(page.locator('.map-video').count() == 2, 'Expected only rollout and final C videos')
                require(page.locator('#layer-select,#relevance-variant').count() == 0, 'Removed relevance controls reappeared')
                require(page.locator('#sample-select option').count() == 3, 'Wrong query scope')
                page.wait_for_function('Array.from(document.querySelectorAll(".map-video")).every(v=>v.readyState>=1)', timeout=30000)
                sources = page.locator('.map-video').evaluate_all('nodes=>nodes.map(v=>v.currentSrc)')
                require(sources[0] == sources[1], 'Videos do not share the same source')
                response = page.request.get(sources[0],headers={'Range':'bytes=0-127'})
                require(response.status == 206 and len(response.body()) == 128, 'Video Range response failed')
                points = sample['timestamps']
                index = min(3,len(points)-1)
                page.locator('#frame-select').select_option(str(index))
                page.wait_for_function('time=>Array.from(document.querySelectorAll(".map-video")).every(v=>!v.seeking && Math.abs(v.currentTime-time)<.08)',arg=points[index],timeout=30000)
                require(page.locator('.frame-button.selected').get_attribute('data-frame') == str(index), 'Selected input slot differs')
                label = page.locator(f'.frame-button[data-frame="{index}"]')
                require(label.locator('b').inner_text() == str(index) and label.locator('small').inner_text().endswith(' s'), 'Frame index/timestamp format differs')
                # Repeated source frames must remain individually selectable while paused.
                duplicates = [i for i in range(1,len(points)) if points[i] == points[i-1]]
                if duplicates:
                    last = duplicates[-1]
                    page.locator('#frame-select').select_option(str(last))
                    page.wait_for_function('time=>Array.from(document.querySelectorAll(".map-video")).every(v=>!v.seeking && Math.abs(v.currentTime-time)<.08)',arg=points[last],timeout=30000)
                    require(page.locator('#frame-select').input_value() == str(last), 'Repeated timestamp slot was lost')
                page.locator('#frame-select').select_option('0')
                page.wait_for_function('Array.from(document.querySelectorAll(".map-video")).every(v=>!v.seeking && v.currentTime<.08)')
                page.locator('#play-toggle').click()
                page.wait_for_function('Array.from(document.querySelectorAll(".map-video")).every(v=>!v.paused && v.currentTime>.15)',timeout=15000)
                page.locator('#play-toggle').click()
                page.wait_for_function('Array.from(document.querySelectorAll(".map-video")).every(v=>v.paused)')
                times = page.locator('.map-video').evaluate_all('nodes=>nodes.map(v=>v.currentTime)')
                require(abs(times[0]-times[1]) < .2, 'Video playback synchronization drift')

                if sample['complete']:
                    map_url = page.locator('#maps-link').evaluate('link=>link.href')
                    response = page.request.get(map_url)
                    require(response.status == 200, 'Actual map fetch failed')
                    raw = response.body()
                    maps = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
                    for name, map_name in [('attention','attention_rollout'),('final','final_c')]:
                        maximum = max(value for frame in maps[map_name] for row in frame for value in row)
                        canvas = page.locator(f'#{name}-map')
                        require(canvas.is_visible(), f'{map_name}: actual map is hidden')
                        shown = float(canvas.get_attribute('data-maximum'))
                        require(math.isclose(maximum,shown,rel_tol=1e-12,abs_tol=1e-14), 'Color maximum differs from all-frame raw maximum')
                        shape = canvas.evaluate('c=>[c.height,c.width]')
                        require(shape == maps['shape'][1:], 'Canvas differs from spatial map grid')
                        bounds = page.locator(f'#{name}-video').evaluate('v=>{const c=v.parentElement.querySelector("canvas"),r=c.getBoundingClientRect(),s=v.parentElement.getBoundingClientRect(),k=Math.min(s.width/v.videoWidth,s.height/v.videoHeight);return {actual:[r.x-s.x,r.y-s.y,r.width,r.height],expected:[(s.width-v.videoWidth*k)/2,(s.height-v.videoHeight*k)/2,v.videoWidth*k,v.videoHeight*k]}}')
                        require(all(abs(a-b)<1 for a,b in zip(bounds['actual'],bounds['expected'])), 'Overlay does not align to visible video content')
                        require(canvas.evaluate('c=>Array.from(c.getContext("2d").getImageData(0,0,c.width,c.height).data).filter((_,i)=>i%4===3).every(a=>a===255)'), 'Heatmap cell alpha is not fully opaque')
                    page.locator('#transparency').evaluate('input=>{input.value="0";input.dispatchEvent(new Event("input"));}')
                    require(page.locator('#attention-map').evaluate('c=>getComputedStyle(c).opacity') == '1', 'Transparency0 is not opaque')
                    page.locator('#transparency').evaluate('input=>{input.value="100";input.dispatchEvent(new Event("input"));}')
                    require(page.locator('#final-map').evaluate('c=>getComputedStyle(c).opacity') == '0', 'Transparency100 does not reveal video')
                    page.locator('#overlay-visible').uncheck()
                    require(not page.locator('#attention-map').is_visible(), 'Overlay visibility toggle failed')
                    page.locator('#overlay-visible').check()
                    page.locator('#transparency').evaluate('input=>{input.value="40";input.dispatchEvent(new Event("input"));}')
                    measured.append(key)
                else:
                    require(page.locator('.map-overlay:visible').count() == 0, 'Pending maps must not be fabricated')
                    require(page.locator('.map-empty:visible').count() == 2, 'Pending map labels missing')
                page.locator('#frame-select').select_option(str(index))
                page.reload(wait_until='networkidle')
                page.wait_for_function('value=>document.getElementById("frame-select").value===value',arg=str(index))
                require(page.locator('#sample-select').input_value()==key, 'Sample deep link failed')
                page.set_viewport_size(dict(width=390,height=844));no_overflow(key)
                page.set_viewport_size(dict(width=1600,height=1100))
                media.append(dict(sample_key=key,range=True,synchronized_seek=True,synchronized_play=True,duplicate_slot_preserved=bool(duplicates)))

            preview = next((sample for sample in snapshot['samples'] if sample['complete']),snapshot['samples'][0])
            page.goto(detail+'#sample='+preview['key'],wait_until='networkidle')
            page.wait_for_function('document.getElementById("comparison").dataset.ready!=="loading"',timeout=60000)
            page.screenshot(path=str(OUTPUT/'preview_desktop.png'),full_page=True)
            page.set_viewport_size(dict(width=390,height=844));no_overflow('Mobile preview')
            page.screenshot(path=str(OUTPUT/'preview_mobile.png'),full_page=True)
            page.set_viewport_size(dict(width=1600,height=1100))
            response = page.goto(family,wait_until='networkidle')
            require(response.status==200,'Family viewer unavailable')
            page.wait_for_selector('#view-tree-items button[data-view="'+view+'"]',state='attached')
            frame = page.frame_locator('#detail-frame')
            frame.locator('#sample-select').wait_for()
            selected = snapshot['samples'][-1]['key']
            frame.locator('#sample-select').select_option(selected)
            page.wait_for_function('key=>new URLSearchParams(location.hash.slice(1)).get("sample")===key',arg=selected)
            page.locator('#tab-report').click();page.wait_for_selector('#report h1')
            require('Chefer' in page.locator('#report h1').inner_text(),'Wrong analysis report')
            page.locator('#tab-result').click();frame.locator('#sample-select').wait_for()
            require(frame.locator('#sample-select').input_value()==selected,'Result/report navigation lost sample')
            require(page.request.get(base+'/').status==200,'Unified index HTTP failed')
            require(not errors,f'Browser errors: {errors}')
            browser.close()
        require(all(hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items()),'Published snapshot changed during browser validation')
        result = dict(passed=True,scope='pilot',queries=3,ui_passed=True,
                      measured_map_queries=len(measured),gpu_validation='pending' if not measured else 'see per-query validation; browser is not a scientific GPU gate',
                      actual_map_checks=measured,media=media,desktop_and_mobile=True,deep_links=True,
                      family_navigation=True,page_errors=errors,published_hashes=hashes,
                      timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(ARTIFACTS/'validation/viewer.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        return result
    except BaseException as error:
        write(ARTIFACTS/'validation/viewer.json',json.dumps(dict(passed=False,ui_passed=False,error=str(error),page_errors=errors,published_hashes=hashes),ensure_ascii=False,indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-viewer', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_viewer() if args.check_viewer else generate(args.artifacts,args.output),ensure_ascii=False))
