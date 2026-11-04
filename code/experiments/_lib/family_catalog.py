"""Directory-derived family navigation shared by every experiment.

The manifest contains existing document entries and a recursive navigation tree:
family → mode → query scope → analysis → child analysis. Analysis nodes reference
result/report entry IDs and supplementary documents; synthetic ancestors have no
actions. New nested analyses are discovered from paths, without renderer changes.
"""
import hashlib,json,os,re
from html import escape,unescape
from pathlib import Path
from urllib.parse import quote
ROOT=Path('/home/irteam/experiments')
TOPICS={'baseline':'Baselines','uncertainty':'Token uncertainty','sampling':'Video sampling','rollout':'Rollout','operations':'Runtime & validation','token_attention':'Token attention','input_order':'Input order','frame_query':'Frame–query similarity','compute_cost':'Compute cost','datasets':'Dataset statistics','failure_modes':'Failure analysis','resources':'Research resources'}
FAMILIES={'reconsideration':'Rollout reconsideration','all_zeroshot':'Zero-shot evaluation','input_order':'Input-order consistency','thinking_vs_instruct':'Thinking vs Instruct','token_analysis':'Token uncertainty & digit errors','span_head':'Span head','spatial_allocation':'Spatial–temporal allocation','fps_spatial_control':'FPS spatial control','stochastic':'Stochastic rollout','conditional_budget':'Conditional rollout budget','deterministic':'Deterministic rollout','selected_frame_cache_migration':'Frame-cache migration','inference_performance':'Inference performance','mmco_diagnosis':'Decoder diagnostics','mtla':'MTLA attention','timelens_instruct':'TimeLens Instruct attention','timelens_thinking':'TimeLens Thinking attention','cosine_2fps':'CLIP / Qwen cosine similarity (2 FPS)','embedding_vs_generation_16fps':'Embedding vs generation cost (16 FPS)','source_statistics':'Dataset metadata','input_settings':'Input settings','audio_caption':'Audio caption resources'}
LEAF_NAMES={'REPORT.md','FINAL_REPORT.md','FINDINGS.md','SPEC.md','ANALYSIS.md','NEXT_EXPERIMENT.md','STRATA.md','SCHEMA.md','END_DECIMAL_CASES.md','SPATIAL_AUDIT.md','AXES.md','PIPELINE_REVIEW.md','CACHE_REPORT.md','final_report.md'}

ANALYSIS_LABELS={'reconsideration':'Reconsideration','dynamics':'토큰별 attention','generation':'Query·답변 attention','hidden':'Hidden state similarity','query':'Query attention','spatial':'Spatial attention','generation / head_analysis':'Layer·head attention','generation / distribution_analysis':'Attention 분포','consistency':'Input order consistency','budget_comparison':'공간 예산 비교','allocation_sweep':'공간·시간 배정','digit_correctness':'자릿수별 정답','digit_correctness / blue_analysis':'정답 가능성','digit_correctness / first_error_distance':'최초 오류 거리','integrated':'Entropy 통합 분석','entropy_iou':'Entropy와 IoU','numeric_uncertainty':'숫자 불확실성','prefix_entropy':'Prefix entropy','all_sources':'데이터 출처 통계'}

def analysis_parts(relative):
 parts=relative.parent.parts
 for i,mode in enumerate(parts[:-1]):
  if mode in ('full','pilot') and (parts[i+1].isdigit() or parts[i+1] in ('overview','all_sources')):
   return list(parts[:i]+parts[i+2:])
 return list(parts[1:] if parts and parts[0] in ('overview','shared','_shared') else parts)

def build_navigation(title,entries):
 """Scope → query count → recursive analysis tree; leaves reference existing entries only."""
 root=dict(kind='family',id='family',label=title,children=[]);modes={};scopes={};nodes={}
 for entry in entries:
  key=entry['scope_key'];mode=key.split('/')[0]
  if mode not in modes:
   modes[mode]=dict(kind='mode',id=mode,label=mode,children=[]);root['children'].append(modes[mode])
  if key not in scopes:
   scope=dict(kind='scope',id=key,label=entry['scope'],queries=entry['queries'],children=[])
   scopes[key]=scope;modes[mode]['children'].append(scope)
   analysis=dict(kind='analysis',id=key+'::',path=[],label=title,entries=[],result=None,report=None,documents=[],children=[])
   scope['children'].append(analysis);nodes[key,()]=analysis
  path=entry.get('analysis_path',[]);parent=nodes[key,()]
  for depth,part in enumerate(path,1):
   prefix=tuple(path[:depth]);node_key=(key,prefix)
   if node_key not in nodes:
    label=ANALYSIS_LABELS.get(' / '.join(prefix),part.replace('_',' ').capitalize())
    child=dict(kind='analysis',id=key+'::'+'/'.join(prefix),path=list(prefix),label=label,entries=[],result=None,report=None,documents=[],children=[])
    nodes[node_key]=child;parent['children'].append(child)
   parent=nodes[node_key]
  parent['entries'].append(entry['id'])
 by_id={entry['id']:entry for entry in entries}
 for node in nodes.values():
  members=[by_id[key] for key in node['entries']]
  result=next((entry for entry in members if entry['kind']=='html'),None)
  report=by_id.get(result.get('report')) if result else next((entry for entry in members if entry.get('report')==entry['id']),None)
  if not report:report=next((entry for entry in members if entry['kind']=='md' and re.search(r'(?:FINAL_REPORT|REPORT|FINDINGS|SPEC)\.md$',entry['id'],re.I)),None)
  if not report:report=next((entry for entry in members if entry['kind']=='md'),None)
  node['result']=result['id'] if result else None;node['report']=report['id'] if report else None
  node['documents']=[entry['id'] for entry in members if entry['id'] not in (node['result'],node['report'])]
 return root

def family_root(project,path):
 result=ROOT/project/'results'
 try:parts=path.relative_to(result).parts
 except ValueError:return None
 if len(parts)<3:return None
 return result/parts[0]/parts[1]

def scope_info(relative):
 parts=relative.parts
 for i,mode in enumerate(parts[:-1]):
  if mode not in ('full','pilot'):continue
  size=parts[i+1]
  if size.isdigit():return f'{mode}/{size}',f'{mode} · {int(size):,} queries',int(size)
  if size=='overview':return f'{mode}/overview',f'{mode} overview',0
  if size=='all_sources':return f'{mode}/all_sources',f'{mode} · source-specific populations',0
 if parts and parts[0] in ('shared','_shared'):return 'shared','shared resources',0
 return 'overview','family overview',0

def title_for(path):
 from result_terms import normalize_terms
 text=path.read_text();title=path.stem
 if path.suffix=='.html':
  m=re.search(r'<title[^>]*>(.*?)</title>',text,re.S|re.I)
  if m:title=unescape(re.sub('<[^>]+>','',m.group(1))).strip()
 else:title=next((line[2:].strip() for line in text.splitlines() if line.startswith('# ')),title)
 title=re.sub(r' · (?:pilot\d+ \(\d+질문\)|전체 full\d+ \([^)]*\)|pilot 누적 통합 · 등록된 표본 범위)$','',title)
 return re.sub(r'(?:\s|_)v\d+\b','',normalize_terms(title)).strip()

def write(path,text):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists() or path.read_text()!=text:path.write_text(text)

def build_family_catalog(projects):
 from build_index import add_navigation,history_controls
 families={}
 for project,project_label in projects.items():
  result=ROOT/project/'results'
  for path in sorted(result.rglob('*')):
   if not path.is_file() or path.suffix not in {'.html','.md'}:continue
   if path.suffix=='.md' and path.name not in LEAF_NAMES:continue
   if path.suffix=='.html' and 'data-family-shell' in path.read_text()[:2000]:continue
   family=family_root(project,path)
   if family is None:continue
   rel=path.relative_to(family);topic,family_id=family.relative_to(result).parts[:2]
   key=family.relative_to(ROOT).as_posix()
   entry=families.setdefault(key,dict(project=project,project_label=project_label,topic=topic,topic_title=TOPICS.get(topic,topic),family=family_id,title=FAMILIES.get(family_id,family_id.replace('_',' ')),root=family,entries=[]))
   scope_key,scope,queries=scope_info(rel)
   report=next((path.with_name(name) for name in ['FINAL_REPORT.md','REPORT.md','FINDINGS.md','final_report.md'] if path.with_name(name).exists()),None)
   parts=analysis_parts(rel)
   analysis=' / '.join(parts) or 'main results'
   entry['entries'].append(dict(id=rel.as_posix(),href=rel.as_posix(),title=title_for(path),kind='html' if path.suffix=='.html' else 'md',scope_key=scope_key,scope=scope,queries=queries,analysis=analysis,analysis_path=parts,report=report.relative_to(family).as_posix() if report else None))
   if path.suffix=='.html':
    original=path.read_text();updated=add_navigation(path,original)
    if updated!=original:path.write_text(updated)
 output=[]
 for key,f in sorted(families.items()):
  family=f['root'];entries=f['entries'];entries.sort(key=lambda e:(e['scope_key'],len(Path(e['id']).parts),e['kind']!='html',e['id']))
  html=[e for e in entries if e['kind']=='html']
  candidates=html or entries
  default=min(candidates,key=lambda e:(-e['queries'],len(Path(e['id']).parts),e['id']))
  # Structural manifest belongs to the family's corresponding artifact tree.
  rel=family.relative_to(ROOT/f['project']/'results');web=ROOT/f['project']/'artifacts/runs'/rel/'overview/web'
  manifest=web/'view_manifest.json'
  config=dict(schema_version=1,title=f['title'],project=f['project'],family=rel.as_posix(),default=default['id'],entries=entries,navigation=build_navigation(f['title'],entries))
  write(manifest,json.dumps(config,ensure_ascii=False,indent=2)+'\n')
  css=os.path.relpath(ROOT/'_lib/family_view.css',family);theme=os.path.relpath(ROOT/'_lib/experiment_theme.css',family);js=os.path.relpath(ROOT/'_lib/family_view.js',family);data=os.path.relpath(manifest,family);history_js=os.path.relpath(ROOT/'_lib/history_controls.js',family);bookmarks_js=os.path.relpath(ROOT/'_lib/family_bookmarks.js',family);home=os.path.relpath(ROOT/'index.html',family);catalog=os.path.relpath(ROOT/'artifacts/maintenance/family_catalog.json',family)
  shell=f'''<!doctype html><html lang="ko" data-family-shell><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(f['title'])} · 실험 계열</title><link rel="stylesheet" href="{theme}"><link rel="stylesheet" href="{css}"><script defer src="{history_js}"></script><script defer src="{bookmarks_js}"></script></head><body>
<header class="family-header"><div class="family-heading"><h1>{escape(f['title'])}</h1><p>{escape(f['project_label'])} / {escape(f['topic_title'])}</p></div><div class="viewer-controls"><div class="viewer-navigation">{history_controls(home, "실험 새로고침")}<span class="control-divider" aria-hidden="true"></span><button id="toggle-sidebar" class="view-control" type="button" aria-controls="family-sidebar" aria-expanded="false" aria-label="실험 목록 펼치기" title="실험 목록 펼치기"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></svg><span>실험</span></button></div><div class="viewer-context"><details id="view-browser" class="view-browser"><summary><span class="view-location-line"><b>결과 탐색</b><span id="view-location"></span></span><span id="view-availability-summary" class="view-availability-summary"></span></summary><nav id="view-tree" aria-label="범위와 하위 실험"><div class="view-tree-heading"><b>범위 → 하위 실험 → 결과·보고서</b><button id="close-view-browser" type="button" aria-label="결과 탐색 닫기">×</button></div><div id="view-availability" class="view-availability"></div><div id="view-tree-items"></div></nav></details><div id="analysis-tabs" class="analysis-tabs" role="tablist" aria-label="현재 분석의 화면 보기"></div><details id="reference-menu" class="reference-menu" hidden><summary>참고</summary><nav id="reference-list" aria-label="참고 문서"></nav></details></div><div class="view-actions"><a id="source-md" hidden target="_blank" rel="noopener" title="원본 Markdown" aria-label="원본 Markdown 열기">MD</a><a id="original" target="_blank" rel="noopener" title="현재 결과 새 탭에서 열기" aria-label="현재 결과 새 탭에서 열기"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3h7v7m0-7L10 14M10 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5"/></svg></a></div></div></header>
<main id="family-app" class="layout sidebar-collapsed" data-manifest="{data}"><aside id="family-sidebar" data-catalog="{catalog}" data-home="{home}" data-current="{(family/"viewer.html").relative_to(ROOT).as_posix()}" hidden><div class="bookmarks-heading"><h2>실험</h2><span id="bookmark-count"></span><button id="close-bookmarks" type="button" aria-label="실험 목록 접기" title="실험 목록 접기">×</button></div><input id="bookmark-search" type="search" aria-label="전체 실험 검색" placeholder="프로젝트·실험 검색"><nav id="experiment-nav" aria-label="전체 실험"><p class="bookmark-status">불러오는 중…</p></nav></aside><section id="view-panel" class="view" role="tabpanel"><span id="view-title" class="sr-only" role="status"></span><div id="message" role="status">결과 목록을 불러오는 중…</div><iframe id="detail-frame" class="detail-frame" title="선택한 인터랙티브 결과" hidden></iframe><article id="report" hidden></article></section></main><script src="{js}"></script></body></html>'''
  write(family/'viewer.html',shell)
  output.append({k:v for k,v in f.items() if k not in ['root','entries']}|dict(href=(family/'viewer.html').relative_to(ROOT).as_posix(),manifest=manifest.relative_to(ROOT).as_posix(),default=default['id'],entry_count=len(entries),html_count=len(html),scopes=list(dict.fromkeys(e['scope'] for e in entries)),search=' '.join(e['title']+' '+e['id']+' '+e['scope'] for e in entries)))
 write(ROOT/'artifacts/maintenance/family_catalog.json',json.dumps(dict(schema_version=1,project_order=list(projects),families=output),ensure_ascii=False,indent=2)+'\n')
 return output
