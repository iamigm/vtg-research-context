import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import shared_selected_frames as frames


def digest(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


class AliasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='frame_alias_test_')
        self.root = Path(self.tmp.name)
        self.old_common = frames.COMMON
        frames.COMMON = self.root
        (self.root / 'frames').mkdir()
        self.alias_key = 'a' * 64
        self.canonical_key = 'b' * 64
        self.source = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2)
        self.video = self.root / 'source.video'
        self.video.write_bytes(b'synthetic source')
        self.cache = dict(cache_key=self.alias_key, path=str(self.root / 'removed.npy'),
                          ids=[0, 1, 2, 3], resized_rgb_hashes=[digest(a) for a in self.source],
                          video_path=str(self.video), source_video_hash=frames.sha(self.video),
                          N=4, grid_hw=[2, 2])
        self.array_path = self.root / 'frames' / (self.canonical_key + '.npy')
        self.set_canonical([0, 2])
        self.alias = dict(self.canonical, cache_key=self.alias_key,
                          canonical_cache_key=self.canonical_key,
                          retired_array=dict(path=self.cache['path'], sha256='original'))
        self.alias_path = frames.manifest_path(self.cache)
        frames.write(self.alias_path, self.alias)

    def tearDown(self):
        frames.COMMON = self.old_common
        self.tmp.cleanup()

    def set_canonical(self, ids):
        np.save(self.array_path, self.source[ids])
        self.canonical = dict(cache_key=self.canonical_key, path=str(self.array_path),
                              ids=ids, rgb_hashes=[digest(self.source[i]) for i in ids],
                              sha256=frames.sha(self.array_path))
        frames.write(frames.manifest_path(dict(cache_key=self.canonical_key)), self.canonical)

    def test_out_of_order_and_repeated_ids(self):
        got = frames.load_selected(self.cache, [2, 0, 2])
        np.testing.assert_array_equal(got, self.source[[2, 0, 2]])
        self.assertFalse(Path(self.cache['path']).exists())

    def test_resolves_latest_canonical_rows(self):
        self.set_canonical([0, 1, 2, 3])
        np.testing.assert_array_equal(frames.load_selected(self.cache, [2, 0]), self.source[[2, 0]])
        self.assertEqual(frames.read(self.alias_path), self.alias)

    def test_alias_manifest_corruption(self):
        self.alias['rgb_hashes'][1] = 'bad'
        frames.write(self.alias_path, self.alias)
        with self.assertRaisesRegex(RuntimeError, 'Aliased shared RGB checksum mismatch'):
            frames.load_selected(self.cache, [2])

    def test_canonical_pixel_corruption(self):
        bad = self.source[[0, 2]].copy()
        bad[1, 0, 0, 0] ^= 1
        np.save(self.array_path, bad)
        with self.assertRaisesRegex(RuntimeError, 'Selected shared RGB checksum mismatch'):
            frames.load_selected(self.cache, [2])

    def test_invalid_key_self_and_cycle(self):
        for key, expected in [('../outside', 'Invalid canonical'), (self.alias_key, 'alias itself')]:
            self.alias['canonical_cache_key'] = key
            frames.write(self.alias_path, self.alias)
            with self.assertRaisesRegex(RuntimeError, expected):
                frames.load_selected(self.cache, [0])
        self.alias['canonical_cache_key'] = self.canonical_key
        frames.write(self.alias_path, self.alias)
        self.canonical['canonical_cache_key'] = self.alias_key
        frames.write(frames.manifest_path(dict(cache_key=self.canonical_key)), self.canonical)
        with self.assertRaisesRegex(RuntimeError, 'alias chains'):
            frames.load_selected(self.cache, [0])

    def test_missing_canonical_manifest(self):
        frames.manifest_path(dict(cache_key=self.canonical_key)).unlink()
        with self.assertRaisesRegex(RuntimeError, 'Canonical shared RGB manifest missing'):
            frames.load_selected(self.cache, [0])

    def test_real_extension_with_mock_decoder(self):
        # Exercise extend() including sorted row insertion, atomic array replacement,
        # source/pixel checks, and canonical-only manifest updates; no GPU is used.
        raw = self.source.transpose(0, 2, 3, 1)
        class Tensor:
            def __init__(self, array): self.array = array
            def permute(self, *order): return Tensor(self.array.transpose(*order))
            def numpy(self): return self.array
        class Reader:
            def __init__(self, path, num_threads):
                assert path == str(self_outer.video) and num_threads == 1
            def __len__(self): return len(raw)
            def get_batch(self, ids):
                return types.SimpleNamespace(asnumpy=lambda: raw[ids])
        def resize(tensor, hw, **kwargs):
            self.assertEqual(list(tensor.array.shape[-2:]), hw)
            return tensor
        self_outer = self
        mocked = {
            'torch': types.SimpleNamespace(from_numpy=Tensor),
            'decord': types.SimpleNamespace(VideoReader=Reader),
            'torchvision': types.ModuleType('torchvision'),
            'torchvision.transforms': types.SimpleNamespace(
                functional=types.SimpleNamespace(resize=resize),
                InterpolationMode=types.SimpleNamespace(BICUBIC='bicubic')),
        }
        before_alias = self.alias_path.read_bytes()
        with patch.dict(sys.modules, mocked):
            np.testing.assert_array_equal(frames.load_selected(self.cache, [2, 1, 0, 1]),
                                          self.source[[2, 1, 0, 1]])
        current = frames.read(frames.manifest_path(dict(cache_key=self.canonical_key)))
        self.assertEqual(current['ids'], [0, 1, 2])
        self.assertEqual(current['sha256'], frames.sha(self.array_path))
        self.assertEqual(before_alias, self.alias_path.read_bytes())
        self.assertFalse(Path(self.cache['path']).exists())
        np.testing.assert_array_equal(frames.load_selected(self.cache, [2, 0]), self.source[[2, 0]])

    def test_plain_cache_and_legacy_fallback(self):
        direct = dict(self.cache, cache_key=self.canonical_key)
        np.testing.assert_array_equal(frames.load_selected(direct, [2, 0]), self.source[[2, 0]])
        missing = dict(cache_key='c' * 64, path=str(self.array_path), ids=[0, 2])
        np.testing.assert_array_equal(frames.load_selected(missing, [2, 0]), self.source[[2, 0]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
