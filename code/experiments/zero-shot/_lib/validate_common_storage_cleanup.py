"""Validate a completed frame-cache cleanup and record measured space savings."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from adaptive_temporal_sampling import ROOT, read, write, sha, stamp, need
from shared_selected_frames import load_selected


def validate(output):
    plan = read(output / 'frame_subset_plan.json')
    status = read(output / 'frame_subset_status.json')
    need(status['state'] == 'complete' and status['completed'] == len(plan['pairs']), 'Cleanup is incomplete')
    records = []
    for pair in plan['pairs']:
        record = read(output / 'pairs' / (pair['retired_cache_key'] + '.json'))
        need(record['state'] == 'complete', 'Pair retirement incomplete')
        need(not Path(pair['retired_array']).exists(), 'Retired array still exists')
        alias = read(pair['retired_manifest'])
        canonical = read(pair['canonical_manifest'])
        need(alias['canonical_cache_key'] == pair['canonical_cache_key'], 'Alias points elsewhere')
        need('canonical_cache_key' not in canonical and Path(canonical['path']).exists(), 'Canonical storage absent')
        need(sha(pair['canonical_manifest']) == pair['canonical_manifest_sha256'], 'Canonical manifest changed')
        need(sha(pair['retired_metadata']) == record['original_metadata_sha256'], 'Original input metadata changed')
        need(sha(pair['canonical_metadata']) == record['canonical_metadata_sha256'], 'Canonical input metadata changed')
        source = read(pair['retired_metadata'])
        ids = source['ids']
        probe = [ids[-1], ids[0], ids[len(ids) // 2], ids[0]]
        expected = dict(zip(ids, source['resized_rgb_hashes']))
        selected = load_selected(source, probe)
        need(all(hashlib.sha256(frame.tobytes()).hexdigest() == expected[ident]
                 for frame, ident in zip(selected, probe)), 'Post-retirement reader mismatch')
        records.append(record)
    reports = read(output / 'report_hashes.json')
    for relative, digest in reports.items():
        need(sha(ROOT / relative) == digest, 'Original report or view changed: ' + relative)
    test = subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / '_lib/check_shared_frame_alias.py')],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / 'alias_validation.log').write_text(test.stdout)
    need(test.returncode == 0, 'Alias regression checks failed')
    disposed = read(output / 'disposable_cache.json')
    need(disposed['state'] == 'complete', 'Disposable cache cleanup incomplete')
    for entry in disposed['files']:
        need(not Path(entry['path']).exists(), 'Disposable cache was regenerated during cleanup')
    after = int(subprocess.check_output(['du', '-s', '-B1', str(ROOT / 'artifacts/common')], text=True).split()[0])
    before = 915293913088
    summary = dict(state='complete', timestamp=stamp(), common_allocated_bytes_before=before,
                   common_allocated_bytes_after=after, net_common_bytes_freed=before-after,
                   net_common_reduction_percent=100*(before-after)/before,
                   retired_frame_arrays=len(records), canonical_frame_arrays=len({p['canonical_cache_key'] for p in plan['pairs']}),
                   retired_frame_array_bytes=sum(r['retired_bytes'] for r in records),
                   preserved_frames_verified=sum(r['retained_pixel_hashes_verified'] for r in records),
                   temporary_video_cache_files=len(disposed['files']), temporary_video_cache_bytes=disposed['bytes'],
                   original_metadata_hashes_verified=2*len(records), unchanged_reports_and_views=len(reports),
                   public_reader_pairs_verified=len(records), alias_regression_checks=8,
                   canonical_whole_file_hash_recomputed=False,
                   verification='Original array SHA-256 and every retired frame pixel hash verified before retirement; retained canonical arrays unchanged; post-retirement public loader probes passed.',
                   preserved=['model responses', 'token distributions', 'cohorts', 'input metadata', 'plans', 'reports', 'canonical RGB'],
                   code_sha256={name:sha(ROOT/'_lib'/name) for name in ('deduplicate_selected_frames.py','shared_selected_frames.py','check_shared_frame_alias.py','validate_common_storage_cleanup.py')})
    write(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/maintenance/common_storage_cleanup')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    try:
        if args.wait_pid:
            while Path(f'/proc/{args.wait_pid}').exists():
                process_state = Path(f'/proc/{args.wait_pid}/stat').read_text().split()[2]
                if process_state == 'Z':
                    break
                time.sleep(5)
        validate(args.output)
    except BaseException as exc:
        write(args.output/'validation_failure.json', dict(state='failed', timestamp=stamp(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
