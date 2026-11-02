"""Serve existing experiments and shared media locally with byte-range seeking."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import sys
ROOT=Path('/home/irteam/experiments/zero-shot/results/input_order/consistency_v1/serve.py').resolve().parents[4]
sys.path.insert(0,str(relocate_path(ROOT/'zero-shot/_lib')))
import timelens_full_digit_serve as server
server.ROOT=ROOT
server.LANDING='/zero-shot/results/input_order/consistency_v1/report.html'
if __name__=='__main__':
 if '--port' not in sys.argv: sys.argv.extend(['--port','8775'])
 server.main()
