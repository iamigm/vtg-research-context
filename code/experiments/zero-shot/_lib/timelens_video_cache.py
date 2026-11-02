"""Bounded, disposable cache of unchanged fetch_video results for full inference."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import contextlib, fcntl, hashlib, inspect, json, os, time
import torch

@contextlib.contextmanager
def locked(path):
    with path.open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield

class VideoCache:
    def __init__(self, root, model, fetch, eligible, limit=8*1024**3):
        self.root = relocate_path(Path(root)); self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.fetch, self.eligible, self.limit = model, fetch, set(eligible), limit
        self.version = hashlib.sha256(relocate_path(Path(inspect.getfile(fetch))).read_bytes()).hexdigest()
        self.events = []

    def __call__(self, ele, *args, **kwargs):
        started = time.perf_counter()
        source = ele.get('video')
        if source not in self.eligible:
            return self.fetch(ele, *args, **kwargs)
        path = relocate_path(Path(source.removeprefix('file://'))).resolve(); stat = path.stat()
        key = hashlib.sha256(json.dumps(dict(model=self.model, source=str(path), size=stat.st_size,
            mtime=stat.st_mtime_ns, element=ele, args=args, kwargs=kwargs,
            implementation=self.version, torch=torch.__version__, reader=os.environ.get('FORCE_QWENVL_VIDEO_READER')),
            sort_keys=True).encode()).hexdigest()
        target = self.root/(key+'.pt')
        with locked(self.root/(key+'.lock')):
            # Open under eviction lock, then load from the open fd even if evicted later.
            with locked(relocate_path(self.root/'eviction.lock')):
                f = target.open('rb') if target.exists() else None
                if f: os.utime(target, None)
            if f:
                with f: result = torch.load(f, map_location='cpu', weights_only=True)
                hit = True
            else:
                result = self.fetch(ele, *args, **kwargs)
                tmp = self.root/(key+f'.{os.getpid()}.tmp')
                try:
                    torch.save(result, tmp)
                    size = tmp.stat().st_size
                    with locked(relocate_path(self.root/'eviction.lock')):
                        files = sorted(result_glob(self.root, '*.pt'), key=lambda p:p.stat().st_mtime_ns)
                        total = sum(p.stat().st_size for p in files)
                        if size <= self.limit:
                            for old in files:
                                if total+size <= self.limit: break
                                total -= old.stat().st_size; old.unlink()
                            tmp.replace(target)
                finally:
                    tmp.unlink(missing_ok=True)
                hit = False
        self.events.append(dict(hit=hit, seconds=time.perf_counter()-started, key=key))
        return result

from timelens_span_head import Prepared
class CachedPrepared(Prepared):
    def __init__(self, rows, model_key, cache_root, eligible):
        super().__init__(rows, model_key)
        self.cache_root, self.eligible, self.cache = cache_root, eligible, None

    def __getitem__(self, i):
        import qwen_vl_utils.vision_process as vision
        if self.cache is None:
            self.cache = VideoCache(self.cache_root, self.model_key, vision.fetch_video, self.eligible)
        original = vision.fetch_video
        self.cache.events = []
        start = time.perf_counter()
        vision.fetch_video = self.cache
        try: item = super().__getitem__(i)
        finally: vision.fetch_video = original
        item['preparation'] = dict(seconds=time.perf_counter()-start, ready_at=time.time(),
            worker_pid=os.getpid(), cache=self.cache.events)
        return item
