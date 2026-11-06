"""Detached fixed-pilot600 supervisor; per-query resume, heartbeat and finalization."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import collections
import fcntl
import gc
import json
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from temporal_mode_core import *


class Progress:
    def __init__(self):
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.state = dict(state='running', stage='initializing', completed=0, total=600,
                          completed_conditions=0, total_conditions=5400, counts={}, eta_seconds=None,
                          eta_scope='remaining inference, excludes GPU waiting and final reports', model=MODEL.name)
        threading.Thread(target=self.heartbeat, daemon=True).start()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state.update(timestamp=stamp(), elapsed_seconds=time.monotonic()-self.started,
                              percentage=100*self.state['completed']/600)
            write(OUT/'metadata/status.json', self.state)
            text = json.dumps(self.state, ensure_ascii=False)
            with (OUT/'logs/progress.log').open('a') as stream:
                stream.write(text+'\n')
            print(text, flush=True)

    def heartbeat(self):
        while not self.stop.wait(20):
            self.update()


def setup():
    cohort = read(COHORT)
    if Path(__file__).resolve().parent == (OUT/'code').resolve():
        assert read(OUT/'metadata/config.json') == CONFIG
        manifest = read(OUT/'metadata/manifest.json')
        assert sha(COHORT) == manifest['cohort_sha256']
        for source in manifest['files']:
            assert sha(OUT/'code'/Path(source['path']).name) == source['sha256']
        assert read(OUT/'validation/cpu_preflight.json')['passed']
        assert read(OUT/'validation/gpu_smoke.json')['full_config_hash'] == digest(CONFIG)
        return cohort
    assert len(cohort) == 600
    assert collections.Counter(r['dataset'] for r in cohort) == dict.fromkeys(
        ['charades-timelens', 'activitynet-timelens', 'qvhighlights-timelens'], 200)
    assert read(ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_300/samples.json') == cohort[:300]
    assert read(ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json')[:600] == cohort
    for category in ('metadata', 'logs', 'responses', 'inputs', 'validation', 'code', 'metrics', 'web', 'environment'):
        (OUT/category).mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    frozen = OUT/'metadata/config.json'
    if frozen.exists():
        assert read(frozen) == CONFIG, 'Frozen configuration changed'
    else:
        write(frozen, CONFIG)
    files = list(Path(__file__).parent.glob('temporal_mode_*'))
    provenance = []
    for path in files:
        if path.is_file():
            destination = OUT/'code'/path.name
            if destination.exists():
                assert sha(destination) == sha(path), 'Code changed after freezing: '+path.name
            else:
                shutil.copyfile(path, destination)
            provenance.append(dict(path=str(path), sha256=sha(path)))
    manifest = dict(cohort=str(COHORT), cohort_sha256=sha(COHORT), queries=600,
        unique_videos=len({r['video_path'] for r in cohort}), dataset_counts=dict(collections.Counter(r['dataset'] for r in cohort)),
        model_manifest_sha256=sha('/home/irteam/backbones/manifests/registry.json'), files=provenance,
        cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        cpuset=Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text().strip(),
        memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip())
    if not (OUT/'metadata/manifest.json').exists():
        write(OUT/'metadata/manifest.json', dict(**manifest, started_at=stamp()))
    else:
        assert read(OUT/'metadata/manifest.json')['files'] == provenance
    from result_layout import relocate_path
    source_checks = []
    for row in cohort:
        source_input = SOURCE/'inputs'/f'{row["sample_key"]}.json'
        assert source_input.exists()
        assert (SOURCE/'rollouts'/f'{row["sample_key"]}.json').exists()
        prior = read(source_input)
        cache_path = relocate_path(Path(prior['cache_metadata']))
        cache = read(cache_path)
        assert relocate_path(Path(cache['array'])).exists(), cache['array']
        assert prior['plan']['frame_ids'] == cache['plans']['2']['frame_ids']
        assert prior['hw'] == cache['hw']
        source_checks.append(dict(key=row['sample_key'], input=str(source_input), input_sha256=sha(source_input), cache=str(cache_path)))
    write(OUT/'validation/cpu_preflight.json', dict(passed=True, queries=600, input_sources=source_checks,
        nested_cohorts_verified=True, frame_arrays_exist=True, timestamp=stamp()))
    smoke = read(OUT/'validation/gpu_smoke.json')
    assert smoke['passed'] and smoke['full_config_hash'] == digest(CONFIG), 'GPU smoke does not match frozen config'
    from artifact_store import register
    register('temporal_mode_search.pilot600.responses', 'rollouts', OUT/'responses',
        'Fixed pilot600 stochastic temporal-mode experiment; inference and validation pending.',
        producer='temporal_mode_runtime.py', cohort=str(COHORT), state='building')
    return cohort


def wait_gpu(progress, required_gib):
    while True:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True)
        free = float(result.stdout.splitlines()[0])/1024
        if free >= required_gib:
            return
        progress.update(state='waiting', stage='waiting_for_gpu_memory', gpu_free_gib=free, gpu_required_gib=required_gib)
        # Runs in a detached worker; the user can inspect the 20-second heartbeat.
        time.sleep(20)


def prepare(row, processor):
    import fps_spatial_control as module
    from result_layout import relocate_path
    old_path = SOURCE/'inputs'/f'{row["sample_key"]}.json'
    old = read(old_path)
    cache_path = relocate_path(Path(old['cache_metadata']))
    cache = read(cache_path)
    public = {k: row[k] for k in ('sample_key', 'dataset', 'video_path', 'query', 'duration')}
    cpu, detail = module.prepare_input(public, cache, 2, 'video_query', processor)
    assert detail['hashes'] == old['hashes'], 'Frozen tensor hashes changed: '+row['sample_key']
    assert detail['hw'] == old['hw'] and detail['plan']['frame_ids'] == old['plan']['frame_ids']
    write(OUT/'inputs'/f'{row["sample_key"]}.json', dict(source=str(old_path), source_sha256=sha(old_path),
        cache_metadata=str(cache_path), hashes=detail['hashes'], frame_ids=detail['plan']['frame_ids'],
        timestamps=detail['plan']['source_pts'], hw=detail['hw'], visual_tokens=detail['visual_tokens'],
        input_tokens=detail['input_tokens'], clipping_duration=cache['duration'], sample_duration=row['duration'], uses_gt=False))
    return cpu, detail, public, cache['duration']


def run_worker():
    cohort = read(COHORT)
    progress = Progress()
    try:
        # fps_spatial_control installs the shared, frozen Transformers/vendor paths.
        import fps_spatial_control
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForImageTextToText
        torch.set_num_threads(CONFIG['cpu_threads'])
        progress.update(stage='checking_resources')
        wait_gpu(progress, CONFIG['gpu_initial_free_gib'])
        progress.update(state='running', stage='loading_model')
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, do_resize=False)
        model = AutoModelForImageTextToText.from_pretrained(MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map={'': 'cuda:0'}).eval()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        write(OUT/'environment/runtime.json', dict(python=sys.version, executable=sys.executable,
            torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), attention='flash_attention_2', model_path=str(MODEL)))
        # Three datasets are checked early; reporting order always remains the frozen cohort order.
        early = [next(r for r in cohort if r['dataset'] == d) for d in sorted({r['dataset'] for r in cohort})]
        keys = {r['sample_key'] for r in early}
        order = early + [r for r in cohort if r['sample_key'] not in keys]
        done, conditions = 0, 0
        counts = collections.Counter()
        validated = set()
        work_started = time.monotonic()
        for row in order:
            key = row['sample_key']
            path = OUT/'responses'/f'{key}.json'
            partial = path.with_suffix('.partial.json')
            if path.exists():
                saved = read(path)
                assert saved['config_hash'] == digest(CONFIG)
                done += 1
                conditions += len(saved['results'])
                counts[row['dataset']] += 1
                continue
            progress.update(stage='preparing_input', sample_key=key, completed=done, completed_conditions=conditions)
            cpu, detail, public, duration = prepare(row, processor)
            # KV and visual prefill peaks need headroom; never change frames or resolution.
            kv_gib = detail['input_tokens'] * 36*8*128*2*2 / 2**30
            wait_gpu(progress, max(18., 2.5*kv_gib+12.))
            progress.update(state='running', stage='prefill', input_tokens=detail['input_tokens'])
            gpu = cpu.to('cuda')
            backend = Backend(model, gpu, torch, processor.tokenizer, eos)
            backend.duration = duration
            progress.update(stage='greedy_validation')
            baseline = backend.greedy()
            source = read(SOURCE/'rollouts'/f'{key}.json')
            expected = source['result']['candidates'][0]['ids']
            assert baseline['ids'] == expected, (key, 'greedy replay changed')
            if row['dataset'] not in validated:
                write(OUT/'validation'/f'greedy_{row["dataset"]}.json', dict(passed=True, sample_key=key,
                    tensor_hashes_equal=True, greedy_ids_equal=True, baseline_id=baseline['id']))
                validated.add(row['dataset'])
            data = read(partial) if partial.exists() else dict(sample=public, config_hash=digest(CONFIG),
                clipping_duration=duration,
                baseline=baseline, baseline_reference=str(SOURCE/'rollouts'/f'{key}.json'),
                input_hashes=detail['hashes'], prefill_seconds=backend.prefill_seconds, results={},
                sequences={}, distributions=[], choices={})
            # Completed conditions resume without relying on a stale in-memory RNG/cache.
            for sequence_id, sequence in data['sequences'].items():
                backend.sequences[sequence_id] = sequence
            backend.choices.update(data['choices'])
            backend.distributions.update({tuple(d['prefix']): d for d in data['distributions']})
            for seed in CONFIG['seeds']:
                for method in METHODS:
                    condition = f'{method}:{seed}'
                    if condition in data['results']:
                        conditions += 1
                        continue
                    progress.update(stage='main_search', method=method, seed=seed, completed_conditions=conditions)
                    search = Search(backend, public, method, seed, progress.update)
                    main = search.run_main()
                    frozen_main = search.main.summary()
                    before = digest(frozen_main)
                    progress.update(stage='model_judgment', main_modes=len(frozen_main))
                    selection = judge(backend, public, frozen_main)
                    progress.update(stage='tail_search')
                    tail = search.run_tail()
                    assert digest(search.main.summary()) == before, 'Tail changed main candidates'
                    data['results'][condition] = dict(method=method, seed=seed, main=main, tail=tail,
                        main_modes=frozen_main, tail_modes=search.tail.summary(), selection=selection,
                        draws=search.draws, main_candidate_hash=before, tail_isolation_verified=True)
                    data['sequences'] = backend.sequences
                    data['choices'] = backend.choices
                    data['distributions'] = list(backend.distributions.values())
                    data['actual_cost'] = dict(decoder_inputs=backend.actual_tokens,
                        forwards=backend.actual_forwards, wall_seconds=time.monotonic()-backend.started)
                    write(partial, data)
                    conditions += 1
                    progress.update(completed_conditions=conditions, stage='condition_checkpoint')
            data['completed_at'] = stamp()
            write(path, data)
            partial.unlink(missing_ok=True)
            done += 1
            counts[row['dataset']] += 1
            finished = {p.stem for p in (OUT/'responses').glob('*.json') if not p.name.endswith('.partial.json')}
            write(OUT/'web/completed.json', [r['sample_key'] for r in cohort if r['sample_key'] in finished])
            progress.update(stage='query_complete', completed=done, completed_conditions=conditions, counts=dict(counts),
                eta_seconds=(time.monotonic()-work_started)/max(1, done)*(600-done))
            del backend, gpu, cpu, data, source
            gc.collect()
            torch.cuda.empty_cache()
        progress.update(state='complete', stage='inference_complete', completed=600, completed_conditions=5400, eta_seconds=0)
    except BaseException:
        progress.update(state='failed', stage='inference_failed', error=traceback.format_exc())
        raise
    finally:
        progress.stop.set()


def supervise():
    cohort = setup()
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for attempt in range(CONFIG['max_retries']+1):
            stream = (OUT/'logs/worker.log').open('a')
            worker = subprocess.Popen([sys.executable, '-u', __file__, '--worker'], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            stream.close()
            write(OUT/'metadata/launcher.json', dict(supervisor_pid=os.getpid(), worker_pid=worker.pid,
                attempt=attempt, timestamp=stamp(), command=[sys.executable, '-u', __file__, '--worker']))
            code = worker.wait()
            if code == 0:
                break
            if attempt == CONFIG['max_retries']:
                raise RuntimeError('Worker exhausted retries; see logs/worker.log')
        status = read(OUT/'metadata/status.json')
        status.update(state='running', stage='postprocessing', timestamp=stamp())
        write(OUT/'metadata/status.json', status)
        for name, arguments in [('temporal_mode_report.py', []), ('temporal_mode_check.py', ['--validate'])]:
            subprocess.run([sys.executable, str(Path(__file__).with_name(name)), *arguments], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT.parent/'_lib/build_index.py')], check=True)
        subprocess.run([sys.executable, str(ROOT.parent/'_lib/refresh_result_servers.py')], check=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('temporal_mode_view_check.py'))], check=True)
        from artifact_store import register
        register('temporal_mode_search.pilot600.responses', 'rollouts', OUT/'responses',
            'Fixed pilot600; three probability-supported stochastic searches and three seeds; clipping; isolated diagnostic tails; final validation passed.',
            producer='temporal_mode_runtime.py', cohort=str(COHORT), state='complete')
        status.update(state='complete', stage='complete', timestamp=stamp(), eta_seconds=0)
        write(OUT/'metadata/status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--setup', action='store_true')
    args = parser.parse_args()
    if args.setup:
        setup()
    elif args.worker:
        run_worker()
    else:
        try:
            supervise()
        except BaseException:
            write(OUT/'metadata/status.json', dict(state='failed', stage='supervisor_failed', total=600,
                  timestamp=stamp(), error=traceback.format_exc()))
            raise
