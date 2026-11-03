"""Serve the experiment index and all viewers with video byte-range support."""
import sys
from pathlib import Path
from urllib.parse import urlsplit,unquote,quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'zero-shot/_lib'))
from build_index import build
import timelens_top30_serve as server
from result_layout import relocate_path


class IndexHandler(server.GalleryHandler):
    def send_head(self):
        if self.path.split('?', 1)[0] == '/':
            self.path = '/index.html'
        parts=urlsplit(self.path)
        old=Path(str(ROOT / unquote(parts.path).lstrip('/')).replace('/home1/irteam/','/home/irteam/'))
        new=relocate_path(old)
        if str(new)!=str(old):
            target='/' + new.relative_to(Path(str(ROOT).replace('/home1/irteam/','/home/irteam/'))).as_posix()
            self.send_response(302)
            self.send_header('Location',quote(target,safe='/')+('?' + parts.query if parts.query else ''))
            self.send_header('Content-Length','0')
            self.end_headers()
            return None
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
