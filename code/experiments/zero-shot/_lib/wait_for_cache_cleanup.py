"""Wait for current900 experiment and integrated report, then compact shared RGB."""
from result_layout import relocate_path
import time,subprocess,sys,traceback,fcntl
from adaptive_temporal_sampling import ROOT,read,write,stamp
RUN='selected_frame_cache_migration_v1';OUT=relocate_path(ROOT/'artifacts/runs'/RUN)
TARGETS=['gt_extent_boundary_density_pilot900_v1','gt_actual_density_pilot900_v1','gt_density_allocation_comparison_pilot900_v1']
def main():
    (OUT/'metadata').mkdir(parents=True,exist_ok=True)
    with (OUT/'metadata/watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                states={r:read(relocate_path(ROOT/'artifacts/runs'/r/'metadata/status.json')).get('state') for r in TARGETS}
                if all(v=='complete' for v in states.values()):break
                x=dict(state='waiting',stage='waiting_for_both_experiments_and_report',dependencies=states,timestamp=stamp());write(OUT/'metadata/status.json',x);print(x,flush=True);time.sleep(30)
            subprocess.run([sys.executable,'-u',str(ROOT/'_lib/compact_selected_frames.py')],check=True)
            subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
        except BaseException:
            write(OUT/'metadata/status.json',dict(state='aborted',timestamp=stamp(),error=traceback.format_exc()));raise
if __name__=='__main__':main()
