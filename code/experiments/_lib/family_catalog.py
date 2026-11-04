"""One representative viewer per directory family, with original detail pages."""
import hashlib,json,os,re
from html import escape,unescape
from pathlib import Path
from urllib.parse import quote
ROOT=Path('/home/irteam/experiments')
TOPICS={'baseline':'Baselines','uncertainty':'Token uncertainty','sampling':'Video sampling','rollout':'Rollout','operations':'Runtime & validation','token_attention':'Token attention','input_order':'Input order','frame_query':'Frame–query similarity','compute_cost':'Compute cost','datasets':'Dataset statistics','failure_modes':'Failure analysis','resources':'Research resources'}
FAMILIES={'reconsideration':'Rollout reconsideration','all_zeroshot':'Zero-shot evaluation','input_order':'Input-order consistency','thinking_vs_instruct':'Thinking vs Instruct','token_analysis':'Token uncertainty & digit errors','span_head':'Span head','spatial_allocation':'Spatial–temporal allocation','fps_spatial_control':'FPS spatial control','stochastic':'Stochastic rollout','conditional_budget':'Conditional rollout budget','deterministic':'Deterministic rollout','selected_frame_cache_migration':'Frame-cache migration','inference_performance':'Inference performance','mmco_diagnosis':'Decoder diagnostics','mtla':'MTLA attention','timelens_instruct':'TimeLens Instruct attention','timelens_thinking':'TimeLens Thinking attention','cosine_2fps':'CLIP / Qwen cosine similarity (2 FPS)','embedding_vs_generation_16fps':'Embedding vs generation cost (16 FPS)','source_statistics':'Dataset metadata','input_settings':'Input settings','audio_caption':'Audio caption resources'}
LEAF_NAMES={'REPORT.md','FINAL_REPORT.md','FINDINGS.md','SPEC.md','ANALYSIS.md','NEXT_EXPERIMENT.md','STRATA.md','SCHEMA.md','END_DECIMAL_CASES.md','SPATIAL_AUDIT.md','AXES.md','PIPELINE_REVIEW.md','CACHE_REPORT.md','final_report.md'}

def family_root(project,path):
 result=ROOT/project/'results'
 try:parts=path.relative_to(result).parts
 except ValueError:return None
 if len(parts)<3:return None
 return result/parts[0]/parts[1]

def scope_info(relative):
 parts=relative.parts
 for mode in ['full','pilot']:
  if mode in parts:
   i=parts.index(mode);size=parts[i+1] if len(parts)>i+1 else ''
   if size.isdigit():return f'{mode}/{size}',f'{mode} · {int(size):,} queries',int(size)
   if size=='overview':return 'pilot/overview','pilot overview',0
   if size=='all_sources':return 'full/all_sources','full · source-specific populations',0
 if any(x in parts for x in ['shared','_shared']):return 'shared','shared resources',0
 return 'overview','full · 9,404 queries / pilot · 300 queries',0

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
 from build_index import add_navigation
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
   parts=[x for x in rel.parent.parts if x not in ['full','pilot','overview','shared','_shared'] and not x.isdigit()]
   analysis=' / '.join(parts) or 'main results'
   entry['entries'].append(dict(id=rel.as_posix(),href=rel.as_posix(),title=title_for(path),kind='html' if path.suffix=='.html' else 'md',scope_key=scope_key,scope=scope,queries=queries,analysis=analysis,report=report.relative_to(family).as_posix() if report else None))
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
  config=dict(schema_version=1,title=f['title'],project=f['project'],family=rel.as_posix(),default=default['id'],entries=entries)
  write(manifest,json.dumps(config,ensure_ascii=False,indent=2)+'\n')
  home=os.path.relpath(ROOT/'index.html',family);css=os.path.relpath(ROOT/'_lib/family_view.css',family);theme=os.path.relpath(ROOT/'_lib/experiment_theme.css',family);js=os.path.relpath(ROOT/'_lib/family_view.js',family);data=os.path.relpath(manifest,family)
  shell=f'''<!doctype html><html lang="ko" data-family-shell><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(f['title'])} · 실험 계열</title><link rel="stylesheet" href="{theme}"><link rel="stylesheet" href="{css}"></head><body>
<header class="family-header"><a class="home-link" href="{home}">← 전체 실험 목록</a><div><p class="eyebrow">EXPERIMENTS / RESEARCH FAMILY</p><h1>{escape(f['title'])}</h1><p>{escape(f['project_label'])} / {escape(f['topic_title'])} / {escape(f['family'])}</p></div></header>
<main id="family-app" class="layout" data-manifest="{data}"><aside><label for="scope">평가 범위 · 질문 수</label><select id="scope" aria-label="평가 범위"></select><nav id="view-nav" aria-label="계열 내 분석"></nav></aside><section class="view"><div class="toolbar"><span id="view-title"></span><a id="source-md" hidden target="_blank" rel="noopener">원본 MD</a><a id="original" target="_blank" rel="noopener">현재 결과 직접 열기 ↗</a></div><div id="message" role="status">결과 목록을 불러오는 중…</div><iframe id="detail-frame" class="detail-frame" title="선택한 인터랙티브 결과" hidden></iframe><article id="report" hidden></article></section></main><script src="{js}"></script></body></html>'''
  write(family/'viewer.html',shell)
  output.append({k:v for k,v in f.items() if k not in ['root','entries']}|dict(href=(family/'viewer.html').relative_to(ROOT).as_posix(),manifest=manifest.relative_to(ROOT).as_posix(),default=default['id'],entry_count=len(entries),html_count=len(html),scopes=list(dict.fromkeys(e['scope'] for e in entries)),search=' '.join(e['title']+' '+e['id']+' '+e['scope'] for e in entries)))
 write(ROOT/'artifacts/maintenance/family_catalog.json',json.dumps(dict(schema_version=1,families=output),ensure_ascii=False,indent=2)+'\n')
 return output
