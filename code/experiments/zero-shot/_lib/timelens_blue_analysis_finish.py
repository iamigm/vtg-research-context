from result_layout import relocate_path, write_result_text
from pathlib import Path
import subprocess,json,time
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300/digit_correctness/blue_analysis')
p=subprocess.Popen([str(relocate_path(ROOT/'.venv/bin/python')),'-u',str(relocate_path(ROOT/'_lib/timelens_blue_analysis.py'))])
while True:
    try:
        code=p.wait(timeout=15)
        if code:raise SystemExit(code)
        break
    except subprocess.TimeoutExpired:
        status=json.loads((relocate_path(OUT/'status.json')).read_text()) if (relocate_path(OUT/'status.json')).exists() else {}
        status['heartbeat']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        print(json.dumps(status,ensure_ascii=False),flush=True)
