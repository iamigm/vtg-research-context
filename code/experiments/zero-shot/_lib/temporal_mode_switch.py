"""Hold only this experiment's old processes after an atomic condition checkpoint."""
import json
import os
import signal
import time
from pathlib import Path

RUN = Path('/home1/irteam/experiments/zero-shot/artifacts/runs/rollout/temporal_mode_search/pilot/600')


def read(path):
    return json.loads(path.read_text())


def record(value):
    target = RUN/'metadata/parallel_transition.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(target)


def run():
    launcher = read(RUN/'metadata/launcher.json')
    status = read(RUN/'metadata/status.json')
    condition = f'{status["method"]}:{status["seed"]}'
    key = status['sample_key']
    information = dict(state='waiting_for_condition_checkpoint', old_supervisor_pid=launcher['supervisor_pid'],
        old_worker_pid=launcher['worker_pid'], sample_key=key, condition=condition,
        before_status=status, active_seeds=[42,43], requested_workers=2, started_unix=time.time())
    record(information)
    while True:
        for name in (key+'.json', key+'.partial.json'):
            target = RUN/'responses'/name
            if target.exists():
                try:
                    data = read(target)
                except (ValueError,OSError):
                    continue
                if condition in data.get('results',{}) or status['seed']==44:
                    for pid in (launcher['supervisor_pid'],launcher['worker_pid']):
                        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
                        assert b'temporal_mode_runtime.py' in command
                        os.kill(pid,signal.SIGSTOP)
                    information.update(state='checkpoint_ready', checkpoint=str(target),
                        checkpoint_conditions=list(data['results']), held_unix=time.time())
                    record(information)
                    print(json.dumps(information,ensure_ascii=False),flush=True)
                    return
        time.sleep(2)


if __name__=='__main__':
    run()
