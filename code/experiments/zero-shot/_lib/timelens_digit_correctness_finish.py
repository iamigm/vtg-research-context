from result_layout import relocate_path, write_result_text
from pathlib import Path
import subprocess,json,os,time,traceback,socket
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/timelens_top30/digit_correctness');PY=str(relocate_path(ROOT/'.venv/bin/python'))
def main():
    try:
        for script in ['timelens_digit_correctness.py','timelens_digit_correctness_browser.py']:
            proc=subprocess.Popen([PY,str(relocate_path(ROOT/'_lib'/script))])
            while proc.poll() is None:time.sleep(5)
            if proc.returncode:raise RuntimeError(f'{script}: exit {proc.returncode}')
    except Exception:
        write_result_text((relocate_path(OUT/'status.json')), json.dumps(dict(state='failed',error=traceback.format_exc()),indent=2)+'\n');raise
if __name__=='__main__':main()
