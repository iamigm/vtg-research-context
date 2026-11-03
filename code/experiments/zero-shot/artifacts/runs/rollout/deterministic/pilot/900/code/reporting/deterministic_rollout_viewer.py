"""Build the pilot900 branch browser and a curated end-decimal case list."""
from result_layout import relocate_path
import collections,csv,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
from deterministic_rollout300 import ROOT,read,write
from deterministic_rollout300_report import interval,overlap,select
from deterministic_rollout_diagnostics import role_name

RUN='deterministic_rollout_pilot900_v2'
OUT=relocate_path(ROOT/'artifacts/runs'/RUN)
REPORT=relocate_path(ROOT/'results'/RUN)
WEB=OUT/'web'
DOT='end.seconds.decimal_separator'

def build(refresh_index=True,pilot_size=900):
 global OUT,REPORT,WEB
 assert pilot_size in [300,600,900]
 previous=(OUT,REPORT,WEB)
 OUT=relocate_path(ROOT/'artifacts/runs'/f'deterministic_rollout_pilot{pilot_size}_v2')
 REPORT=relocate_path(ROOT/'results'/OUT.name);WEB=OUT/'web'
 try:_build_current(refresh_index)
 finally:OUT,REPORT,WEB=previous

def _build_current(refresh_index=True):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct',local_files_only=True)
 cohort=read(OUT/'metadata/cohort.json');pilot_size=len(cohort)
 all_refs=read(relocate_path(ROOT/'artifacts/runs/deterministic_rollout_pilot900_v2/metadata/source_records.json'))
 refs=all_refs[:pilot_size]
 assert [Path(r['path']).stem for r in refs]==[s['sample_key'] for s in cohort]
 media_source=relocate_path(ROOT/'artifacts/runs/uncertainty/token_analysis/full/9404/metadata/media_manifest.json')
 media=read(media_source);media_report=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404')
 WEB.mkdir(parents=True,exist_ok=True)
 shared_samples=relocate_path(ROOT/'artifacts/runs/deterministic_rollout_pilot900_v2/web/samples')
 if pilot_size==900:(WEB/'samples').mkdir(exist_ok=True)
 manifest=[];catalog=[];case_rows=[];roles=set();checks=[]
 for item in refs:
  original=Path(item['path']);assert hashlib.sha256(original.read_bytes()).hexdigest()==item['sha256']
  rec=read(original);s=rec['sample'];cs=rec['result']['candidates'];gt=s['span'][0]
  scores={c['index']:overlap(interval(c),gt) if interval(c) else 0. for c in cs}
  base=interval(cs[0]);greedy_short=bool(base and 1<=base[1]<2 and 100<=gt[1]<200)
  video=(media_report/media[s['video_path']]['media']).resolve();assert video.is_file(),video
  assert video.is_relative_to(ROOT.resolve())
  candidates=[];dot_children=[];any_short=False
  for c in cs:
   b=c['branch'];branch=None
   if b:
    parent=cs[b['parent']-1];pp=interval(parent)
    isdot=DOT in b['roles'];roles.update(b['roles'])
    short=bool(isdot and pp and 1<=pp[1]<2 and 100<=gt[1]<200)
    any_short|=short
    branch=dict(parent=b['parent'],position=b['position'],roles=b['roles'],
      role_labels=[role_name(r) for r in b['roles']],target_roles=b.get('target_roles',[]),
      original_token=tok.decode([b['original_token']],clean_up_tokenization_spaces=False),
      forced_token=tok.decode([b['forced_token']],clean_up_tokenization_spaces=False),
      probability=b['raw_probability'],parent_delta=scores[c['index']]-scores[parent['index']],
      greedy_delta=scores[c['index']]-scores[1],dot=isdot,short_parent=short)
    catalog.append(dict(sample=s['sample_key'],candidate=c['index'],parent=b['parent'],
      roles=b['roles'],position=b['position'],probability=branch['probability'],parent_delta=branch['parent_delta'],greedy_delta=branch['greedy_delta'],
      iou=scores[c['index']],parent_iou=scores[parent['index']],pair=interval(c),parent_pair=pp,
      before=branch['original_token'],after=branch['forced_token'],
      dot=isdot,short_parent=short))
    if isdot:dot_children.append((c,parent,branch))
   candidates.append(dict(index=c['index'],answer=c['answer'],token_ids=c['ids'],tokens=[tok.decode([i],clean_up_tokenization_spaces=False) for i in c['ids']],
      pair=interval(c),iou=scores[c['index']],log_probability=c['log_probability'],
      branch=branch,decoder_tokens=c['decoder_tokens'],stopped_on_eos=c['stopped_on_eos']))
  choices={}
  for k in [1,4,8,16,32]:
   choices[str(k)]={method:select(cs[:k],method,gt)['index'] for method in
     ['max_probability','medoid','probability_weighted_medoid','oracle']}
  entry=dict(key=s['sample_key'],dataset=s['dataset'],query=s['query'],duration=s['duration'],gt=gt,
    greedy_pair=base,greedy_iou=scores[1],dot_count=len(dot_children),greedy_short=greedy_short,
    any_short=any_short,clock=any(c['parsed'] and c['parsed']['format']=='clock' for c in cs),
    max_parent_delta=max((c['branch']['parent_delta'] for c in candidates if c['branch']),default=0),
    max_greedy_delta=max((c['iou']-scores[1] for c in candidates),default=0))
  manifest.append(entry)
  payload=dict(sample=entry,video=os.path.relpath(video,REPORT),candidates=candidates,choices=choices,
    raw_record=os.path.relpath(original,REPORT))
  if pilot_size==900:write(WEB/'samples'/(s['sample_key']+'.json'),payload)
  else:assert (shared_samples/(s['sample_key']+'.json')).exists()
  if dot_children:
   best=max(dot_children,key=lambda row:(row[2]['parent_delta'],-row[0]['index']))
   c,parent,b=best
   short_rows=[row for row in dot_children if row[2]['short_parent']]
   focus=max(short_rows,key=lambda row:row[2]['parent_delta']) if short_rows else best
   fc,fp,fb=focus
   case_rows.append(dict(sample_key=s['sample_key'],dataset=s['dataset'],query=s['query'],
    gt_start=gt[0],gt_end=gt[1],dot_branches=len(dot_children),greedy_answer=cs[0]['answer'],
    greedy_iou=scores[1],greedy_1x_gt100_199=greedy_short,any_parent_1x_gt100_199=any_short,
    focus_parent=fp['index'],focus_candidate=fc['index'],parent_answer=fp['answer'],
    candidate_answer=fc['answer'],forced_token=fb['forced_token'],parent_iou=scores[fp['index']],
    candidate_iou=scores[fc['index']],parent_delta=fb['parent_delta'],greedy_delta=fb['greedy_delta']))
  checks.append(dict(sample_key=s['sample_key'],candidates=len(cs),video_exists=True))
 assert len(manifest)==pilot_size
 if pilot_size==900:
  assert sum(x['dot_count'] for x in manifest)==243 and len(case_rows)==84
 summary=dict(samples=pilot_size,branches=len(catalog),dot_samples=len(case_rows),dot_branches=sum(x['dot_count'] for x in manifest),
   greedy_short=sum(x['greedy_short'] for x in manifest),any_parent_short=sum(x['any_short'] for x in manifest))
 write(WEB/'manifest.json',dict(samples=manifest,branches=catalog,roles={r:role_name(r) for r in sorted(roles)},summary=summary))
 write(OUT/'metrics/end_decimal_cases.json',dict(summary=summary,cases=case_rows))
 with (OUT/'metrics/end_decimal_cases.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
 prefix=Path(__file__).with_suffix('')
 font=ROOT/'_lib/entropy_viewer/NotoSansKR.ttf';assert font.is_file()
 css=Path(str(prefix)+'.css').read_text().replace('__FONT_URL__',os.path.relpath(font,WEB))
 (WEB/'viewer.css').write_text(css)
 shutil.copyfile(str(prefix)+'.js',WEB/'viewer.js')
 distribution_js=Path(__file__).with_name('rollout_distribution_viewer.js')
 shutil.copyfile(distribution_js,WEB/'distributions.js')
 vocab_size=read(Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct/config.json')).get('text_config',{}).get('vocab_size',len(tok))
 traces_root=relocate_path(ROOT/'artifacts/runs/deterministic_rollout_pilot900_v2/token_traces')
 vocab_path=traces_root/'vocab.json'
 if not vocab_path.exists():
  tokenizer_size=len(tok)
  write(vocab_path,[tok.decode([i],clean_up_tokenization_spaces=False) if i<tokenizer_size else f'<unused:{i}>' for i in range(vocab_size)])
 html=Path(str(prefix)+'.html').read_text().replace('__ASSET_BASE__',os.path.relpath(WEB,REPORT))
 html=html.replace('__PILOT_SIZE__',str(pilot_size)).replace('__DISTRIBUTION_BASE__',os.path.relpath(traces_root,REPORT))
 html=html.replace('__SAMPLE_BASE__',os.path.relpath(shared_samples,REPORT))
 sys.path.insert(0,str(ROOT.parent/'_lib'))
 from build_index import add_navigation
 (REPORT/'viewer.html').write_text(add_navigation(REPORT/'viewer.html',html))
 case_report(case_rows,summary)
 report=REPORT/'REPORT.md';text=report.read_text()
 block=f'''
<!-- rollout-viewer:start -->
## 샘플별 분기 확인

[Rollout 분석 HTML](viewer.html) · [종료 소수점 {summary["dot_samples"]}개 샘플](viewer.html#preset=dot) · [전체 사례 목록](END_DECIMAL_CASES.md)

샘플 탐색은 모든 샘플·ID순이 기본이며, 사례 유형 선택과 IoU·ΔIoU·분기 토큰 순번·대체 토큰 확률의 수치 범위를 함께 적용한다. 응답 쌍 선택은 비교할 완성 후보와 직접 부모를 바꾸며, 디코딩 토큰 선택은 그 응답 쌍을 유지하고 확률 분포를 볼 위치만 바꾼다. greedy 응답과 직접 부모가 다르면 따로 표시한다.

디코딩 경로는 공통 prefix와 분기 이후 응답을 분리하고, 토큰별 조건부 확률과 누적곱을 각각 한 줄씩 가로로 표시한다. 두 경로의 누적곱은 EOS까지 이어진다. 분포는 top-k 또는 top-p로 선택하며, 기본 k는 현재 prefix에서 색으로 표시하고 후속 경로를 추적한 토큰 수+1다. 직접 입력한 k는 해당 위치에서 유지하며 다른 토큰 위치를 고르면 다시 기본값을 계산한다. 상위30개로 기준을 만족하지 못하면 전체 어휘 FP32 원자료에서 정확히 계산한다. 기준 밖 경로 토큰은 별도로 표시하고 재정규화하지 않는다. 검색은 전체 어휘에서 수행한다. 긴 집합은 분포별로80개씩 추가 표시한다.

영상 재생·응답 원문·토큰 비교는 하나의 검토 영역이다. 원래 응답·성능 수치는 유지한다. 원자료는900개 실행의 token_traces와 web/samples를 공유하며 pilot마다 복제하지 않는다.
<!-- rollout-viewer:end -->
'''
 text=re.sub(r'\n<!-- rollout-viewer:start -->.*?<!-- rollout-viewer:end -->\n?','',text,flags=re.S)
 report.write_text(text.rstrip()+'\n'+block)
 from rollout_distribution_integrate import append_distribution_summary
 append_distribution_summary(REPORT,OUT)
 sources=[]
 for suffix in ['.py','.html','.css','.js']:
  p=Path(str(prefix)+suffix);dest=OUT/'code/reporting'/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(p,dest);sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 shutil.copyfile(distribution_js,OUT/'code/reporting'/distribution_js.name)
 sources.append(dict(path=str(distribution_js),sha256=hashlib.sha256(distribution_js.read_bytes()).hexdigest()))
 write(OUT/'validation/viewer_data.json',dict(passed=True,**summary,all_900_media_exist=True,
   raw_record_hashes_verified=True,sources=sources,video_cache_reused=True,distribution_source=str(traces_root),
   shared_korean_font=str(font),font_sha256=hashlib.sha256(font.read_bytes()).hexdigest()))
 if refresh_index:subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
 print(json.dumps(summary,ensure_ascii=False))

def case_report(rows,summary):
 lines=['# 종료 시각의 소수점 분기 사례','',
  '종료 시각의 소수점 기호 `.` 자체를 다른 토큰으로 바꾼 분기다. 내부 코드 `end.seconds.decimal_separator`는 end=종료 시각, seconds=초 필드, decimal_separator=소수점 자체라는 뜻이다. 소수점 아래 숫자를 조정한 분기와 다르다.', '',
  f'{summary["samples"]}질문 중 {summary["dot_samples"]}질문에서 {summary["dot_branches"]}개 분기가 있었다. 최초 greedy의 종료가1.x초인데 GT 종료가100∼199초인 질문은 {summary["greedy_short"]}개다. 탐색 중 생긴 부모 후보까지 포함하면 {summary["any_parent_short"]}개다. 이 두 범위를 혼동하지 않는다.', '',
  '[모든 샘플 HTML](viewer.html) · [종료 소수점 전체](viewer.html#preset=dot) · [최초 greedy부터 1초대였던 경우](viewer.html#preset=greedy_short) · [탐색 부모까지 포함한 경우](viewer.html#preset=parent_short)', '',
  '## GT 종료는100초대인데 부모가1초대로 답한 사례','',
  '아래 각 질문에서는 이 조건에 해당하는 분기 중 부모 대비 IoU 개선이 가장 큰 분기를 표시했다. GT와의 IoU가 낮다고 다른 사건이라고 단정하지 않는다. 모델의 내부 의도는 알 수 없지만, `1.`에서 정수 부분을 끝내어 실제 출력이1초대가 된 것은 확인할 수 있다.', '',
  '| 샘플 | greedy 응답도 해당? | GT(초) | 최초 greedy | 분기 직전 부모 | 분기 후 후보 | 부모 대비 ΔIoU |',
  '|---|---|---|---|---|---|---:|']
 for x in rows:
  if not x['any_parent_1x_gt100_199']:continue
  url=f'viewer.html#sample={x["sample_key"]}&candidate={x["focus_candidate"]}&preset=parent_short'
  lines.append(f'| [{x["sample_key"]}]({url}) | '+('예' if x['greedy_1x_gt100_199'] else '아니오, 탐색 부모만')+f' | [{x["gt_start"]}, {x["gt_end"]}] | {x["greedy_answer"]} | {x["parent_answer"]} | {x["candidate_answer"]} | {x["parent_delta"]:+.4f} |')
 lines += ['', f'## 종료 소수점 분기가 있었던 전체{summary["dot_samples"]}개 질문','',
  '각 질문을 클릭하면 해당 후보로 이동한다. 위 조건에 해당하면 해당 조건 분기를, 그 외에는 부모 대비 개선이 가장 큰 소수점 분기를 표시한다. 평균이 아닌 개별 사례 목록이다. Δ부모=후보 IoU−분기 직전 부모 IoU, Δgreedy=후보 IoU−greedy 응답 IoU. 두 값은 다를 수 있다.', '',
  '| 샘플 | GT(초) | 분기 수 | 비교할 부모 → 후보 | Δ부모 | Δgreedy |',
  '|---|---|---:|---|---:|---:|']
 for x in rows:
  url=f'viewer.html#sample={x["sample_key"]}&candidate={x["focus_candidate"]}&preset=dot'
  lines.append(f'| [{x["sample_key"]}]({url}) | [{x["gt_start"]}, {x["gt_end"]}] | {x["dot_branches"]} | {x["parent_answer"]} → {x["candidate_answer"]} | {x["parent_delta"]:+.4f} | {x["greedy_delta"]:+.4f} |')
 (REPORT/'END_DECIMAL_CASES.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':build()
