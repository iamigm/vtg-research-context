"""Retire only frame arrays whose exact pixels survive in a verified superset.

Consumes an audited pair plan; never discovers deletion targets implicitly.
Original experiment metadata and canonical arrays are not modified.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from adaptive_temporal_sampling import ROOT, read, write, sha, stamp, need
from shared_selected_frames import COMMON, load_selected


@contextlib.contextmanager
def locked(path):
    with path.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def fingerprint(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def process_pair(pair, output):
    key = pair['retired_cache_key']
    canonical_key = pair['canonical_cache_key']
    frames = (COMMON / 'frames').resolve()
    old = Path(pair['retired_array']).resolve()
    kept = Path(pair['canonical_array']).resolve()
    old_mp = Path(pair['retired_manifest']).resolve()
    kept_mp = Path(pair['canonical_manifest']).resolve()
    need(old == frames / (key + '.npy'), 'Retired path outside audited frame store')
    need(kept == frames / (canonical_key + '.npy'), 'Canonical path outside audited frame store')
    need(old_mp == old.with_suffix('.json') and kept_mp == kept.with_suffix('.json'), 'Manifest path mismatch')
    need(key != canonical_key, 'Self reference')
    journal = output / 'pairs' / (key + '.json')
    if journal.exists() and read(journal)['state'] in ('retiring', 'complete') and not old.exists():
        need(not old.exists(), 'Retired array reappeared')
        need(read(old_mp).get('canonical_cache_key') == canonical_key and kept.exists(), 'Completed alias changed')
        record = read(journal)
        source = read(pair['retired_metadata'])
        ids = source['ids']
        probe = [ids[-1], ids[0], ids[len(ids) // 2], ids[0]]
        selected = load_selected(source, probe)
        expected = dict(zip(ids, source['resized_rgb_hashes']))
        need(all(hashlib.sha256(frame.tobytes()).hexdigest() == expected[ident]
                 for frame, ident in zip(selected, probe)), 'Resumed alias reader verification failed')
        if record['state'] != 'complete':
            record.update(state='complete', timestamp=stamp(), new_manifest_sha256=sha(old_mp))
            write(journal, record)
        return record

    with locked(old.with_suffix('.lock')), locked(kept.with_suffix('.lock')):
        source = read(pair['retired_metadata'])
        destination = read(pair['canonical_metadata'])
        original, canonical = read(old_mp), read(kept_mp)
        need('canonical_cache_key' not in canonical, 'Alias target must be canonical')
        need(sha(kept_mp) == pair['canonical_manifest_sha256'], 'Canonical manifest changed since audit')
        need(source['source_video_hash'] == destination['source_video_hash'] == pair['source_video_hash'], 'Source video differs')
        need(source['grid_hw'] == destination['grid_hw'] == pair['grid_hw'], 'Spatial resolution differs')
        need(source['decoder'] == destination['decoder'], 'Decoder differs')
        need(original['cache_key'] == source['cache_key'] == key, 'Retired cache identity differs')
        need(canonical['cache_key'] == destination['cache_key'] == canonical_key, 'Canonical identity differs')
        # A stopped run may have published its alias but not yet retired the array.
        if 'canonical_cache_key' in original:
            need(original['canonical_cache_key'] == canonical_key, 'Unexpected existing alias')
        else:
            need(sha(old_mp) == pair['retired_manifest_sha256'], 'Retired manifest changed since audit')
            need(original['ids'] == source['ids'] and original['rgb_hashes'] == source['resized_rgb_hashes'], 'Original selection metadata differs')
        need(old.exists() and not old.is_symlink() and old.stat().st_nlink == 1, 'Retired file is absent, linked or symbolic')
        need(not kept.is_symlink(), 'Canonical file is symbolic')
        old_stat, kept_stat = fingerprint(old), fingerprint(kept)
        allocated_bytes = old.stat().st_blocks * 512
        need(old.stat().st_size == pair['retired_array_bytes'], 'Retired size changed')
        need(sha(old) == pair['retired_array_expected_sha256'], 'Retired array checksum mismatch')
        need(canonical['sha256'] == pair['canonical_array_expected_sha256'], 'Canonical declared checksum changed')
        loc = {ident: row for row, ident in enumerate(canonical['ids'])}
        need(len(loc) == len(canonical['ids']), 'Duplicate canonical frame IDs')
        expected = dict(zip(source['ids'], source['resized_rgb_hashes']))
        need(len(expected) == len(source['ids']) == pair['retired_frame_count'], 'Source selection changed')
        need(set(expected) <= set(loc), 'Canonical array omits required frame IDs')
        array = np.load(kept, mmap_mode='r', allow_pickle=False)
        original_array = np.load(old, mmap_mode='r', allow_pickle=False)
        need(array.dtype == np.uint8 and array.shape == (len(loc), 3, *source['grid_hw']), 'Canonical array layout differs')
        need(original_array.dtype == array.dtype and original_array.shape == (len(expected), *array.shape[1:]), 'Original array layout differs')
        for original_row, (ident, digest) in enumerate(expected.items()):
            need(canonical['rgb_hashes'][loc[ident]] == digest, 'Recorded pixel hashes differ')
            need(hashlib.sha256(array[loc[ident]].tobytes()).hexdigest() == digest, 'Retained pixel bytes differ')
            need(np.array_equal(original_array[original_row], array[loc[ident]]), 'Original and retained pixels differ')
        del array, original_array
        need(fingerprint(old) == old_stat and fingerprint(kept) == kept_stat, 'Arrays changed during verification')
        alias = dict(original)
        # Full canonical fields also keep historical read-only loaders functional.
        for field in ('path', 'ids', 'rgb_hashes', 'sha256'):
            alias[field] = canonical[field]
        alias.update(canonical_cache_key=canonical_key, deduplicated_at=stamp())
        record = dict(state='verified', retired_cache_key=key, canonical_cache_key=canonical_key,
                      retired_path=str(old), canonical_path=str(kept), retired_bytes=old_stat[2],
                      retired_allocated_bytes=allocated_bytes,
                      original_array_sha256=pair['retired_array_expected_sha256'],
                      original_manifest_sha256=pair['retired_manifest_sha256'],
                      original_metadata=pair['retired_metadata'], original_metadata_sha256=sha(pair['retired_metadata']),
                      canonical_metadata=pair['canonical_metadata'], canonical_metadata_sha256=sha(pair['canonical_metadata']),
                      original_array_sha256_verified=True, retained_pixel_hashes_verified=len(expected),
                      original_and_retained_pixels_equal=True,
                      canonical_whole_file_hash_recomputed=False,
                      original_metadata_preserved=True, canonical_array_unchanged=True, timestamp=stamp())
        write(journal, record)
        write(old_mp, alias)

    # Exercise the real reader with reordered and repeated IDs before unlinking.
    ids = source['ids']
    probe = [ids[-1], ids[0], ids[len(ids) // 2], ids[0]]
    try:
        selected = load_selected(source, probe)
        need(all(hashlib.sha256(frame.tobytes()).hexdigest() == expected[ident]
                 for frame, ident in zip(selected, probe)), 'Public reader verification failed')
    except Exception:
        with locked(old.with_suffix('.lock')):
            write(old_mp, original)
        raise
    with locked(old.with_suffix('.lock')), locked(kept.with_suffix('.lock')):
        need(read(old_mp) == alias and sha(kept_mp) == pair['canonical_manifest_sha256'], 'Manifest changed before retirement')
        need(fingerprint(old) == old_stat and fingerprint(kept) == kept_stat, 'Array changed before retirement')
        record.update(state='retiring', public_reader_verified=True)
        write(journal, record)
        old.unlink()
        record.update(state='complete', timestamp=stamp(), new_manifest_sha256=sha(old_mp))
        write(journal, record)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    plan = read(args.plan)
    output = args.plan.resolve().parent
    pairs = plan['pairs'][:args.limit] if args.limit else plan['pairs']
    need(1 <= args.workers <= 8, 'Use between one and eight CPU workers')
    need(len({p['retired_cache_key'] for p in pairs}) == len(pairs), 'Duplicate retirement target')
    need(not ({p['retired_cache_key'] for p in pairs} & {p['canonical_cache_key'] for p in pairs}), 'Alias chains are not allowed')
    begin = time.monotonic()
    records = []
    def status(state, **extra):
        value = dict(state=state, timestamp=stamp(), completed=len(records), total=len(pairs),
                     retired_bytes=sum(r['retired_bytes'] for r in records),
                     retired_allocated_bytes=sum(r['retired_allocated_bytes'] for r in records),
                     verified_frames=sum(r['retained_pixel_hashes_verified'] for r in records),
                     elapsed_seconds=time.monotonic() - begin, **extra)
        write(output / 'frame_subset_status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    with (output / 'frame_subset.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('running')
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process_pair, p, output) for p in pairs]
                last = begin
                for future in as_completed(futures):
                    records.append(future.result())
                    if time.monotonic() - last >= 10:
                        status('running')
                        last = time.monotonic()
            status('complete')
        except BaseException as exc:
            status('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
