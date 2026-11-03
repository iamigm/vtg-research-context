"""Non-invasive progress logger for the detached full-head experiment."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
from datetime import datetime, timezone
import fcntl,json,os,time
from timelens_paths import PILOT_SAMPLES
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/timelens_top30')
def read(path):return json.loads(path.read_text())
def alive(pid):
    try:return relocate_path(Path(f'/proc/{pid}/stat')).read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def main():
    logs=relocate_path(OUT/'logs');logs.mkdir(parents=True,exist_ok=True)
    with (relocate_path(logs/'progress.lock')).open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        execution=read(relocate_path(OUT/'execution.json'));samples=read(PILOT_SAMPLES)
        datasets=list(dict.fromkeys(s['dataset'] for s in samples));started=execution['started_epoch']
        with (relocate_path(logs/'progress.log')).open('a',buffering=1) as human,(relocate_path(logs/'progress.jsonl')).open('a',buffering=1) as machine:
            while True:
                now=time.time();elapsed=now-started
                counts={m:{ds:sum((relocate_path(OUT/m/(s['sample_key']+'.json'))).exists() for s in samples if s['dataset']==ds) for ds in datasets} for m in ['qwen25','qwen3']}
                totals={m:sum(v.values()) for m,v in counts.items()};done=sum(totals.values());total=2*len(samples)
                state=read(relocate_path(OUT/'status.json')) if (relocate_path(OUT/'status.json')).exists() else {'state':'running','stage':'starting'}
                status=state['state'];stage=state.get('stage',status)
                supervisor=read(relocate_path(OUT/'supervisor.json')) if (relocate_path(OUT/'supervisor.json')).exists() else None
                if status=='running' and supervisor and not alive(supervisor['pid']):status='failed';stage='supervisor exited without final status'
                eta=round(elapsed*(total-done)/done) if done else None
                record=dict(timestamp=datetime.now(timezone.utc).isoformat(),state=status,stage=stage,completed=done,total=total,percent=round(done/total*100,2),elapsed_seconds=round(elapsed),inference_eta_seconds=eta,eta_method='mean throughput since experiment start; excludes final report/validation',models=totals,datasets=counts)
                if 'error' in state:record['error']=state['error']
                tmp=relocate_path(OUT/'progress.tmp');write_result_text(tmp, json.dumps(record,indent=2)+'\n');tmp.replace(relocate_path(OUT/'progress.json'))
                machine.write(json.dumps(record)+'\n')
                eta_text='unknown' if eta is None else f'{eta//60}m{eta%60:02d}s'
                human.write(f"{record['timestamp']} state={status} stage={stage} completed={done}/{total} progress={record['percent']:.2f}% elapsed={int(elapsed)//60}m{int(elapsed)%60:02d}s inference_ETA~{eta_text} qwen25={totals['qwen25']}/{len(samples)} qwen3={totals['qwen3']}/{len(samples)}\n")
                human.write('  '+' | '.join(m+': '+', '.join(f'{ds}={n}/100' for ds,n in groups.items()) for m,groups in counts.items())+'\n')
                if status in ['complete','failed']:
                    if 'error' in state:human.write(state['error']+'\n')
                    break
                time.sleep(15)
if __name__=='__main__':main()
