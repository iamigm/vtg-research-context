"""Finish a running CPU integrity gate, publish, and resume cumulative pilots."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from greedy_attribution_core import BASE, CONFIG, ROOT, digest, read, stamp, storage, write


def continue_after_integrity(validator_pid, scope):
    canonical = storage()
    scope = int(scope)
    assert read(canonical/'metadata/config.json') == CONFIG
    state = read(canonical/'metadata/status.json')
    state.pop('error', None)
    start = time.monotonic()

    def progress(stage, **updates):
        state.update(state='running', stage=stage, completed=scope, active_queries=scope,
                     total=900, timestamp=stamp(), elapsed_seconds=time.monotonic()-start,
                     validation_log=str(canonical/'logs/integrity_check.log'), **updates)
        write(canonical/'metadata/status.json', state)
        write(ROOT/'artifacts/common/registry/greedy_attribution.live.json',
              dict(canonical=str(canonical), status=str(canonical/'metadata/status.json'),
                   log=str(canonical/'logs/run.log'), **state))
        with (canonical/'logs/progress.log').open('a') as stream:
            import json
            stream.write(json.dumps(state, ensure_ascii=False)+'\n')

    try:
        gate = canonical/'validation'/f'integrity_pilot{scope}.json'
        print(f'Waiting for pilot{scope} integrity PID {validator_pid}; see logs/integrity_check.log', flush=True)
        while True:
            checked = read(gate) if gate.exists() else {}
            if (checked.get('passed') and checked.get('complete') and checked.get('completed') == scope
                    and checked.get('scope_queries') == scope
                    and checked.get('pixel_signed_sum_validation', {}).get('method') == 'fp32_reduction_forward_error_bound'):
                break
            command = Path(f'/proc/{validator_pid}/cmdline')
            if not command.exists() or b'greedy_attribution_check.py' not in command.read_bytes():
                raise RuntimeError('CPU integrity gate exited without a complete passing record; see integrity_check.log')
            progress('validating_saved_records')
            time.sleep(20)
        from greedy_attribution_report import report
        from greedy_attribution_check import check_viewer
        progress('reporting_and_validating_viewer')
        report(scope)
        subprocess.run([sys.executable, str(ROOT.parent/'_lib/build_index.py')], check=True)
        subprocess.run([sys.executable, str(ROOT.parent/'_lib/refresh_result_servers.py')], check=True)
        check_viewer(scope)
        audit_path = canonical/'validation/pixel_sum_audit.json'
        audit = read(audit_path) if audit_path.exists() else None
        write(canonical/'validation'/f'pilot{scope}.json', dict(
            passed=True, queries=scope, report_and_viewer_verified=True, timestamp=stamp(),
            config_hash=digest(CONFIG), gradient_repeatability_verified=False,
            known_backward_audit=dict(path=str(audit_path), sample_key=audit.get('sample_key'),
                                     target_id=audit.get('target_id'), passed=audit.get('passed')) if audit else None))
        report(allow_partial=True)
        subprocess.run([sys.executable, str(ROOT.parent/'_lib/build_index.py')], check=True)
        print(f'pilot{scope} saved-record and viewer gates passed. Backward repeatability audit is separately documented.', flush=True)
        from greedy_attribution_manage import launch
        launch()
    except Exception as error:
        progress('validation_failed')
        state.update(state='failed', error=str(error))
        write(canonical/'metadata/status.json', state)
        write(ROOT/'artifacts/common/registry/greedy_attribution.live.json',
              dict(canonical=str(canonical), status=str(canonical/'metadata/status.json'),
                   log=str(canonical/'logs/run.log'), **state))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validator-pid', type=int, required=True)
    parser.add_argument('--scope', type=int, default=300, choices=CONFIG['stages'])
    args = parser.parse_args()
    with (BASE/'resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        continue_after_integrity(args.validator_pid, args.scope)
