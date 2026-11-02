"""Stop individual experiment viewers and ensure one detached shared server."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path('/home/irteam/experiments')
STATE = ROOT / 'artifacts/maintenance'
LEGACY = {
    'timelens_top30_serve.py', 'timelens_rollout_serve.py',
    'timelens_entropy_integrated_serve.py', 'timelens_digit_correctness_serve.py',
    'timelens_full_digit_serve.py', 'serve_gallery.py', 'serve_report.py',
}
records = []
shared = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        args = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
        scripts = [a for a in args[1:] if a.endswith('.py') and '/experiments/' in a]
        if not scripts:
            continue
        script = scripts[0]
        name = Path(script).name
        if name == 'serve_index.py':
            shared.append(int(proc.name))
        elif name in LEGACY or (name == 'serve.py' and 'consistency_v1' in script):
            os.kill(int(proc.name), signal.SIGTERM)
            records.append({'pid': int(proc.name), 'script': script, 'action': 'stopped'})
    except (OSError, UnicodeError):
        continue
# --restart applies changes to the common server without reviving old viewers.
if '--restart' in sys.argv:
    for pid in shared:
        os.kill(pid, signal.SIGTERM)
    shared = []
    time.sleep(.3)
for pid in shared[1:]:
    os.kill(pid, signal.SIGTERM)
if not shared:
    with (STATE / 'serve_index.log').open('a') as stream:
        process = subprocess.Popen(
            [sys.executable, '-u', str(ROOT / '_lib/serve_index.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    shared = [process.pid]
for attempt in range(50):
    try:
        with urlopen('http://127.0.0.1:8780/', timeout=2) as response:
            assert response.status == 200
            assert '실험 목록' in response.read().decode()
        break
    except Exception:
        if attempt == 49:
            raise
        time.sleep(.2)
result = {'stopped_servers': records, 'active_server': {'pid': shared[0], 'port': 8780,
          'log': str(STATE / 'serve_index.log')},
          'proxy_url': os.environ.get('VSCODE_PROXY_URI', '').replace('{{port}}', '8780')}
(STATE / 'servers.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
