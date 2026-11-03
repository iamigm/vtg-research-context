"""Two disjoint prefix replay workers, global progress and final validation."""
import os,sys,subprocess,time,hashlib,collections,fcntl,traceback
from pathlib import Path
from rollout_token_distributions import OUT,TRACES,Progress,read,write,stamp

def main():
 with (OUT/'metadata/distribution_parallel.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  refs=read(OUT/'metadata/source_records.json')
  datasets={Path(r['path']).stem:read(r['path'])['sample']['dataset'] for r in refs}
  def completed():return {p.parent.name for p in TRACES.glob('*/index.json')}
  initial=len(completed());p=Progress();processes=[];started=time.monotonic()
  try:
   write(OUT/'metadata/distribution_operation.json',dict(timestamp=stamp(),workers=2,torch_threads_per_worker=2,
    compression_threads_per_worker=2,partition='SHA256(sample key) first8 hex mod2',
    top30='topk threshold + stable sort of threshold ties; exact same values and tie order',
    raw_probabilities_unchanged=True,reused_completed_queries=initial))
   for rank in range(2):
    env=dict(os.environ,ROLLOUT_DIST_RANK=str(rank),ROLLOUT_DIST_WORLD='2')
    with (OUT/'logs'/f'distribution_worker{rank}.log').open('a') as stream:
     process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('rollout_token_distributions.py'))],
       env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    processes.append(process)
   write(OUT/'metadata/distribution_workers.json',dict(pids=[x.pid for x in processes]))
   while True:
    done=completed();states=[]
    for rank in range(2):
     path=OUT/'metadata'/f'distribution_worker{rank}.json';states.append(read(path) if path.exists() else {})
    fresh=len(done)-initial
    p.update(state='running',stage='parallel_replay',completed=len(done),counts=dict(collections.Counter(datasets[x] for x in done)),
     workers=[dict(pid=proc.pid,state=s.get('state'),stage=s.get('stage'),sample_key=s.get('sample_key')) for proc,s in zip(processes,states)],
     eta_seconds=(time.monotonic()-started)/fresh*(900-len(done)) if fresh>0 else None,
     total_prefixes=sum(s.get('total_prefixes',0) for s in states),
     total_compressed_bytes=sum(s.get('total_compressed_bytes',0) for s in states))
    codes=[proc.poll() for proc in processes]
    if any(code is not None and code!=0 for code in codes):raise RuntimeError(f'Prefix replay worker failed: {codes}')
    if all(code==0 for code in codes):break
    time.sleep(20)
   p.update(stage='validating')
   assert len(completed())==900
   total_nodes=total_bytes=0
   for reference in refs:
    key=Path(reference['path']).stem;index=read(TRACES/key/'index.json')
    assert index['validation']['passed'] and index['raw_record_sha256']==reference['sha256']
    assert hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest()==reference['sha256']
    for node in index['nodes'].values():
     assert (TRACES/key/node['file']).stat().st_size==node['bytes']
     total_nodes+=1;total_bytes+=node['bytes']
   write(OUT/'validation/distribution_capture.json',dict(passed=True,queries=900,prefixes=total_nodes,compressed_bytes=total_bytes,
    all_original_answers_preserved=True,input_hashes_identical=True,greedy_suffixes_identical=True,
    forced_probabilities_checked=True,full_answer_probability_products_checked=True,workers=2))
   note=OUT.parents[2]/'results'/OUT.name/'DISTRIBUTIONS.md'
   note.write_text('# 토큰 분포 측정 완료\n\n900개 저장 입력·응답 prefix를 동일하게 재생했다. 전체 어휘 FP32 확률을 보존하며 조건부 확률과 generation step 확률의 누적곱을 구분한다. 새 응답 생성이나 평가 수치 변경은 없다. 부모에서 선택된 토큰은 서로 다른 색, 각 토큰을 붙인 뒤의 자식 분포는 같은 색으로 표시한다.\n')
   p.update(state='complete',stage='complete',completed=900,eta_seconds=0,total_prefixes=total_nodes,total_compressed_bytes=total_bytes)
  except BaseException:
   p.update(state='failed',stage='failed',error=traceback.format_exc())
   for process in processes:
    if process.poll() is None:process.terminate()
   raise
  finally:p.stop.set()

if __name__=='__main__':main()
