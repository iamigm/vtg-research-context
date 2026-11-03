"""Fold finished probability capture into the original pilot900 report and UI."""
import argparse,fcntl,hashlib,json,re,time,traceback
from pathlib import Path
from deterministic_rollout300 import ROOT,read,write,stamp

OUT=ROOT/'artifacts/runs/deterministic_rollout_pilot900_v2'
REPORT=ROOT/'results/deterministic_rollout_pilot900_v2'

def append_distribution_summary(report,out):
 validation=out/'validation/distribution_capture.json'
 if not validation.exists():validation=out/'validation/pilot_distribution_ready.json'
 if not validation.exists() or not read(validation).get('passed'):return False
 result=read(validation);n=result['queries'];assert n in [300,600,900]
 if (out/'metadata/cohort.json').exists():assert len(read(out/'metadata/cohort.json'))==n
 size=result['compressed_bytes']/1024**3
 block=f'''\n<!-- rollout-distributions:start -->
## 토큰 분포와 generation step 확률의 곱

기존{n}개 rollout 실험에 토큰 분포 측정을 통합했다. 별도 분포 실험이 아니라 동일 코호트의 기존 HTML에서 확인한다. 총{n}개 질문, {result['prefixes']:,}개 고유 prefix의 전체 어휘 FP32 확률을 기록했다. 참조하는 압축 원자료 크기는 {size:.2f}GiB다. 원래 응답·평가 수치는 유지했다.

조건부 토큰 확률은 P(다음 토큰 | 영상·질문·현재 prefix)이다. 누적 경로 확률은 각 generation step의 조건부 확률을 곱한 값이고, 전체 응답 확률은 EOS까지 포함한다. 부모에서 선택한 토큰을 서로 다른 색으로 표시하고, 각 토큰을 붙인 뒤의 자식 분포는 같은 색으로 비교한다. 표시한 상위 토큰 안에서 재정규화하지 않으며 나머지 확률도 표시한다.

원래 입력 tensor hash, greedy suffix, 강제 토큰 확률과 전체 응답 확률 곱을 검증했다. [기존 샘플별 HTML에서 보기](viewer.html#panel=distributions). 원자료는 전체900개 실행의 token_traces를 공유하며 pilot별로 복제하지 않는다.
<!-- rollout-distributions:end -->
'''
 path=report/'REPORT.md';text=path.read_text()
 text=re.sub(r'\n<!-- rollout-distributions:start -->.*?<!-- rollout-distributions:end -->\n?','',text,flags=re.S)
 path.write_text(text.rstrip()+'\n'+block)
 note=report/'DISTRIBUTIONS.md'
 if note.exists():note.unlink()
 return True

def status(state,stage,**extra):
 value=dict(state=state,stage=stage,timestamp=stamp(),**extra)
 write(OUT/'metadata/distribution_integration.json',value)
 with (OUT/'logs/distribution_integration.log').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
 original=read(OUT/'metadata/status.json');original['token_distributions']=value
 write(OUT/'metadata/status.json',original)
 print(json.dumps(value,ensure_ascii=False),flush=True)

def integrate():
 with (OUT/'metadata/distribution_integration.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  capture=read(OUT/'validation/distribution_capture.json');assert capture['passed'] and capture['queries']==900
  assert len(list((OUT/'token_traces').glob('*/index.json')))==900
  prior=OUT/'validation/distribution_integration.json'
  csv=OUT/'metrics/per_query.csv';before=hashlib.sha256(csv.read_bytes()).hexdigest()
  if prior.exists() and read(prior).get('passed') and '<!-- rollout-distributions:start -->' in (REPORT/'REPORT.md').read_text():
   assert read(prior)['original_per_query_sha256']==before
   status('complete','complete',completed=900,total=900);return
  status('running','integrating',completed=900,total=900)
  records=[]
  for reference in read(OUT/'metadata/source_records.json'):
   path=Path(reference['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256']
   records.append(read(path))
  import deterministic_rollout300_report as report_module
  report_module.OUT=OUT;report_module.REPORT=REPORT
  report_module.report(records=records,cohort_label='고정 중첩 pilot900')
  assert hashlib.sha256(csv.read_bytes()).hexdigest()==before,'Original performance changed'
  assert append_distribution_summary(REPORT,OUT)
  for target in re.findall(r'\]\(([^)]+)\)',(REPORT/'REPORT.md').read_text()):
   if not target.startswith(('http','#')):assert (REPORT/target.split('#')[0]).exists(),target
  assert not (REPORT/'DISTRIBUTIONS.md').exists() and (REPORT/'viewer.html').exists()
  write(prior,dict(passed=True,queries=900,original_per_query_sha256=before,
   original_record_hashes_unchanged=True,existing_html_updated=True,existing_report_updated=True,
   separate_distribution_report_removed=True,separate_run_created=False))
  source=Path(__file__);(OUT/'code/reporting'/source.name).write_bytes(source.read_bytes())
  status('complete','complete',completed=900,total=900)

def wait_and_integrate():
 while True:
  capture=read(OUT/'metadata/distribution_status.json')
  if capture['state']=='failed':
   status('failed','capture_failed',completed=capture['completed'],total=900,error=capture.get('error'));return
  if capture['state']=='complete':break
  status('waiting','waiting_for_capture',completed=capture['completed'],total=900)
  time.sleep(20)
 integrate()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
 try:
  if args.wait:wait_and_integrate()
  else:integrate()
 except BaseException:status('failed','integration_failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
