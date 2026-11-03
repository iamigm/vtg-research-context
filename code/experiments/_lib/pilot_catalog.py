"""Build cumulative pilot family catalogs from registered cohort directories."""
import hashlib,json,os
from pathlib import Path
ROOT=Path('/home/irteam/experiments');P=ROOT/'zero-shot'

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
 if not path.exists() or path.read_text()!=text:path.write_text(text)
def write(path,text):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists() or path.read_text()!=text:path.write_text(text)
def build_pilot_catalog():
 base=P/'results';common=P/'artifacts/common/cohorts/nested_pilot_v1'
 cohorts={}
 for n in (300,600,900):
  path=common/f'pilot_{n}/samples.json';rows=json.loads(path.read_text())
  assert len(rows)==n
  cohorts[n]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'keys':[r['sample_key'] for r in rows]}
 assert cohorts[300]['keys']==cohorts[600]['keys'][:300]==cohorts[900]['keys'][:300]
 assert cohorts[600]['keys']==cohorts[900]['keys'][:600]
 families={folder for folder in base.rglob('pilot') if folder.is_dir() and any(child.is_dir() and child.name.isdigit() and ((child/'REPORT.md').exists() or (child/'SPEC.md').exists()) for child in folder.iterdir())}
 for family in sorted(families):
  scopes=sorted((p for p in family.iterdir() if p.is_dir() and p.name.isdigit()),key=lambda p:int(p.name))
  rel=family.relative_to(P/'results');view=family/'overview';out=P/'artifacts/runs'/rel/'overview'
  entries=[];lines=[f'# Pilot overview · {family.relative_to(base)}','', '동일 실험 계열에서 표본이 추가되면300·600·900개 단계를 함께 탐색한다. 아래에 등록된 범위만 현재 자료가 있다.','', '| 표본 범위 | 보고서·사양 | 고정 코호트 |','|---|---|---|']
  for scope in scopes:
   n=int(scope.name)
   if n not in cohorts:
    custom=P/'artifacts/runs'/scope.relative_to(P/'results')/'metadata/cohort.json'
    if not custom.exists():continue
    rows=json.loads(custom.read_text());assert len(rows)==n
    cohorts[n]={'path':str(custom),'sha256':hashlib.sha256(custom.read_bytes()).hexdigest(),'keys':[r['sample_key'] for r in rows]}
   report=scope/'REPORT.md' if (scope/'REPORT.md').exists() else scope/'SPEC.md'
   if not report.exists():continue
   data=P/'artifacts/runs'/scope.relative_to(P/'results');cohort=cohorts[n]
   reference=dict(schema_version=1,queries=n,cohort_id=f'nested_pilot_v1.{n}.cohort' if n in (300,600,900) else f'fixed_diagnostic.{n}.cohort',cohort_path=cohort['path'],cohort_sha256=cohort['sha256'],selection='frozen nested prefix; do not resample',source_records='See original run metadata; responses are not copied by this catalog')
   # Spatial budget300/600 share original900 responses, not independent inference.
   if family.parent.name=='budget_comparison' and n in (300,600):reference['source_run']=str(P/'artifacts/runs'/rel/'900')
   dump(data/'metadata/cohort_reference.json',reference)
   entries.append(dict(queries=n,result=str(scope),artifacts=str(data),report=str(report),cohort_id=reference['cohort_id'],cohort_sha256=cohort['sha256']))
   lines.append(f'| pilot{n} · {n}질문 | [{report.name}]({os.path.relpath(report,view)}) | [고정{n}개]({os.path.relpath(Path(cohort["path"]),view)}) |')
  lines += ['', '300·600·900개를 포함하는 계열의 공통 코호트는300 ⊂ 600 ⊂ 900이며 작은 코호트는 큰 코호트의 정확한 prefix다. 겹치는 질문을 독립 표본처럼 더하지 않는다. 확장 실행의 추가분은 별도 결과로 노출하지 않는다. 이 코호트는 층화 진단 표본으로 전체 벤치마크 평균이 아니다.','', '새600/900개 결과는 같은 계열의 해당 숫자 폴더와 artifacts/runs의 대응 경로에 등록한다. 전체 실험 목록을 재생성하면 이 통합 목록도 갱신된다. 조건·입력·생성 정책이 바뀌면 별도 버전 계열로 둔다.','',f'[전체 실험 목록]({os.path.relpath(ROOT/"index.html",view)})']
  dump(out/'metadata/catalog.json',dict(schema_version=1,family=str(rel),entries=entries,nested=True,aggregation='references only; no independent-sample pooling'))
  write(view/'REPORT.md','\n'.join(lines)+'\n')
if __name__=='__main__':build_pilot_catalog()
