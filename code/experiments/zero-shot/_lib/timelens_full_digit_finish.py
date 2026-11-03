"""Full benchmark supervisor: GPU queue, resume inference, complete digit analyses."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
from collections import Counter
import json,time,os,subprocess,traceback,sys
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404');LIB=relocate_path(ROOT/'_lib');PY=str(relocate_path(ROOT/'.venv/bin/python'));ROWS=json.loads((relocate_path(BASE/'samples.json')).read_text());N=len(ROWS)*2;START=time.time();INFER_START=None;INITIAL_DONE=0;CHILDREN=[]
(relocate_path(BASE/'logs')).mkdir(exist_ok=True);(relocate_path(BASE/'digit_correctness')).mkdir(exist_ok=True)
def counts():
    return {m:{d:len(list(result_glob((relocate_path(BASE/m)), d+'_*.json'))) for d in sorted({s['dataset'] for s in ROWS})} for m in ['qwen25','qwen3']}
def status(stage,state='running',**extra):
    c=counts();done=sum(sum(v.values()) for v in c.values());elapsed=time.time()-START
    new=done-INITIAL_DONE;eta=(time.time()-INFER_START)*(N-done)/new if INFER_START and new>0 and stage=='inference' else None
    d=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),state=state,stage=stage,completed=done,total=N,percent=relocate_path(100*done/N),elapsed_seconds=elapsed,eta_seconds=eta,eta_scope='remaining inference only; excludes GPU waiting, media and analysis',counts=c,**extra)
    p=relocate_path(BASE/'status.json');tmp=p.with_suffix('.tmp');write_result_text(tmp, json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p);print(json.dumps(d,ensure_ascii=False),flush=True)
def launch(script,log,args=(),env=None):
    f=(relocate_path(BASE/'logs'/log)).open('a');p=subprocess.Popen([PY,'-u',str(relocate_path(LIB/script)),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);f.close();CHILDREN.append(p);return p
def wait(p,stage):
    while p.poll() is None:status(stage);time.sleep(15)
    if p.returncode:raise RuntimeError(f'{stage}: process {p.pid} failed, exit {p.returncode}; see logs')
def gpu():
    line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0];a,b,c=map(int,line.split(','));return dict(used_mib=a,total_mib=b,utilization=c)
def main():
    global INFER_START,INITIAL_DONE
    INITIAL_DONE=sum(sum(v.values()) for v in counts().values())
    # Prepare HTML media on CPU while waiting for the occupied GPU.
    mp=launch('timelens_full_digit_media.py','media.log') if not (relocate_path(BASE/'media_complete.json')).exists() else None
    while True:
        g=gpu()
        if g['used_mib']<8192 and g['utilization']<15:break
        status('waiting_for_gpu',gpu=g,reason='existing external GPU workload; not interrupted');time.sleep(15)
        if mp and mp.poll() not in [None,0]:raise RuntimeError('Media preparation failed; see logs/media.log')
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    INFER_START=time.time();workers=[]
    for model in ['qwen25','qwen3']:
        workers.append((model,launch('timelens_top30_full_worker.py',model+'.log',['--model',model,'--shards','1'],env)))
    write_result_text((relocate_path(BASE/'execution.json')), json.dumps(dict(supervisor_pid=os.getpid(),workers=[dict(model=m,pid=p.pid) for m,p in workers],reused_pilot=600,full_samples=len(ROWS),settings='unchanged pilot greedy input and generation settings'),indent=2))
    while any(p.poll() is None for _,p in workers):status('inference',gpu=gpu());time.sleep(15)
    # Retry a failed worker once without concurrent inference, resuming saved files.
    for model,p in workers:
        if p.returncode:
            retry=launch('timelens_top30_full_worker.py',model+'.log',['--model',model,'--shards','1'],env);wait(retry,'inference retry '+model)
    assert sum(sum(v.values()) for v in counts().values())==N,counts()
    if mp:wait(mp,'waiting for browser media')
    for script,log,stage in [('timelens_full_digit_prepare.py','endpoint_prepare.log','extract endpoint traces'),('timelens_full_digit_impl.py','digit_analysis.log','digit correctness analysis')]:wait(launch(script,log),stage)
    out=relocate_path(BASE/'digit_correctness');summary=json.loads((relocate_path(out/'summary.json')).read_text());supported=summary['responses'];excluded=len(summary['excluded'])
    p=relocate_path(out/'viewer.js');s=p.read_text().replace('동일 pilot 300 × 두 모델 · 숫자 분석 594/600개',f'전체 {len(ROWS):,}개 × 두 모델 · 숫자 분석 {supported:,}개 응답').replace("list.length+' / 300개'", "list.length+'개 (전체 '+MANIFEST.length+'개)'");write_result_text(p, s)
    p=relocate_path(out/'report.html');s=p.read_text().replace('모델별 동일 pilot 300개. MM:SS 형식 6개는 자릿수 판정에서 제외되어 사유를 표시합니다.',f'모델별 전체 {len(ROWS):,}개. 지원 표기 밖 응답 {excluded:,}개는 자릿수 분석에서 제외하며 영상과 원응답은 볼 수 있습니다.');write_result_text(p, s)
    with (relocate_path(out/'REPORT.md')).open('a') as f:f.write(f'\n## 전체 벤치마크 분석 범위\n\n전체 응답 {N:,}개 중 지원 표기 {supported:,}개, 제외 {excluded:,}개. 제외 목록은 summary.json에 기록했다.\n\n[정답 가능 상태의 세 축 분석](blue_analysis/REPORT.md)\n')
    wait(launch('timelens_full_digit_browser.py','digit_browser.log'),'digit browser validation')
    wait(launch('timelens_full_blue_impl.py','blue_analysis.log'),'blue-state three-axis analysis and browser validation')
    write_result_text((relocate_path(out/'status.json')), json.dumps(dict(state='complete',responses=N,supported=supported,excluded=excluded,browser_validation=True),indent=2));status('complete',state='complete',digit_report='digit_correctness/REPORT.md',blue_report='digit_correctness/blue_analysis/REPORT.md')
if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',state='failed',error=traceback.format_exc())
        for p in CHILDREN:
            if p.poll() is None:p.terminate()
        raise
