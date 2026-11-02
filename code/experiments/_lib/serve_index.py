"""Serve the experiment index and all viewers with video byte-range support."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'zero-shot/_lib'))
from build_index import build
import timelens_top30_serve as server
from result_layout import relocate_path


class IndexHandler(server.GalleryHandler):
    def send_head(self):
        if self.path.split('?', 1)[0] == '/':
            self.path = '/index.html'
        return super().send_head()

    def translate_path(self, path):
        return str(relocate_path(Path(super().translate_path(path))))


if __name__ == '__main__':
    build()
    server.ROOT = ROOT
    server.LANDING = '/index.html'
    server.GalleryHandler = IndexHandler
    if '--port' not in sys.argv:
        sys.argv.extend(['--port', '8780'])
    server.main()
