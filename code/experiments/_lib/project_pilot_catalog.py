"""Maintain pilot family overviews without duplicating or pooling source records."""
import hashlib,json,os
from pathlib import Path
ROOT=Path('/home/irteam/experiments')
PROJECTS=['attention','similarity','analysis/failure_mode_timestamp/failure_cases']
REPORTS=['FINAL_REPORT.md','REPORT.md','final_report.md','FINDINGS.md','SPEC.md']
def write(path,text):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists() or path.read_text()!=text:path.write_text(text)
def manifest(data,n):
 for name in ['inputs/samples.json','metadata/samples.json','samples.json','inputs/manifest.json','metadata/manifest.json']:
  f=data/name
  if not f.exists():continue
  obj=json.loads(f.read_text());rows=obj if isinstance(obj,list) else obj.get('samples') if isinstance(obj,dict) else None
  if not isinstance(rows,list) or len(rows)!=n:continue
  keys=[str(row.get('sample_key',row.get('id',json.dumps(row,sort_keys=True,ensure_ascii=False)))) if isinstance(row,dict) else str(row) for row in rows]
  return f,keys
 return None,None

def build_project_pilot_catalogs():
 for project in PROJECTS:
  p=ROOT/project;results=p/'results'
  for pilot in sorted(results.rglob('pilot')):
   if not pilot.is_dir():continue
   entries=[];prior=None;prefix_checked=True
   for cohort in sorted((d for d in pilot.iterdir() if d.is_dir() and d.name.isdigit()),key=lambda d:int(d.name)):
    report=next((cohort/name for name in REPORTS if (cohort/name).exists()),None)
    if report is None:continue
    n=int(cohort.name);data=p/'artifacts/runs'/cohort.relative_to(results);fixed,keys=manifest(data,n)
    if prior and keys:
     assert prior==keys[:len(prior)],f'Changed pilot prefix: {cohort}'
    if keys:prior=keys
    else:prefix_checked=False
    entries.append(dict(queries=n,result=str(cohort),artifacts=str(data),report=str(report),manifest=str(fixed) if fixed else None,manifest_sha256=hashlib.sha256(fixed.read_bytes()).hexdigest() if fixed else None))
   if not entries:continue
   view=pilot/'overview';out=p/'artifacts/runs'/pilot.relative_to(results)/'overview'
   lines=[f'# Pilot 누적 통합 · {pilot.parent.relative_to(results)}','', '같은 실험 계열의 실제 등록된 pilot 단계만 표시한다. 질문 수와 모델·조건별 응답 수를 구분한다.','', '| 표본 범위 | 보고서 | 고정 manifest |','|---|---|---|']
   for e in entries:
    m=f'[원본 manifest]({os.path.relpath(e["manifest"],view)})' if e['manifest'] else '각 원보고서의 표본 선정 기록 참조'
    lines.append(f'| pilot{e["queries"]} · {e["queries"]}질문 | [보고서]({os.path.relpath(e["report"],view)}) | {m} |')
   lines+=['','새600·900개 결과는 같은 계열의 pilot 아래 숫자 폴더에 추가하고 대응 artifacts/runs 경로를 함께 등록한다. 없는 단계의 결과를 만들지 않는다. 원자료를 복제하거나 겹치는 질문을 독립 표본처럼 합산하지 않는다. 다른 프로젝트의 pilot과 동일한 코호트라고 자동 가정하지 않는다.','', '중첩 단계가 여러 개면 각 고정 manifest의 질문 순서와 prefix 포함 관계를 검사한다. 설정·입력·생성 정책이 바뀌면 새 버전 계열로 구분한다.','',f'[전체 실험 목록]({os.path.relpath(ROOT/"index.html",view)})']
   write(view/'REPORT.md','\n'.join(lines)+'\n')
   write(out/'metadata/catalog.json',json.dumps(dict(schema_version=1,family=str(pilot.parent.relative_to(results)),entries=entries,prefix_checked=prefix_checked,nested_multiple_cohorts_checked=prefix_checked and len(entries)>1,aggregation='references only; no pooling'),ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':build_project_pilot_catalogs()
