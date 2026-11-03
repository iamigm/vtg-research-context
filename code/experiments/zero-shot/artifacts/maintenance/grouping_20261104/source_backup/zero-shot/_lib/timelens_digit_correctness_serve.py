"""Private local gallery server with HTTP byte ranges for video seeking.

Use an SSH tunnel or a remote editor's port forwarding to access port 8773.
"""
from result_layout import relocate_path, write_result_text
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

RESULT_VIEW = relocate_path(relocate_path(Path(__file__)).resolve().parents[1] / 'results/timelens_top30/digit_correctness')
ROOT = Path('/home/irteam/experiments')
LANDING = '/' + str(RESULT_VIEW).replace('/home1/irteam/','/home/irteam/').split('/experiments/',1)[1] + '/report.html'

class GalleryHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_count = None
        if self.path.split('?',1)[0] == '/':
            self.send_response(302)
            self.send_header('Location', LANDING)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        path = relocate_path(Path(self.translate_path(self.path)))
        if not path.resolve().is_relative_to(ROOT.resolve()):
            self.send_error(403)
            return None
        value = self.headers.get('Range')
        if not value or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip())
        start, end = 0, size - 1
        invalid = not match or size == 0
        if match:
            left, right = match.groups()
            if not left and not right:
                invalid = True
            elif left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            else:
                length = int(right)
                start = max(0, size - length)
                invalid |= length == 0
            invalid |= start > end or start >= size
        if invalid:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        stream = path.open('rb')
        stream.seek(start)
        self.byte_count = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(self.byte_count))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Last-Modified', self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return stream

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self.byte_count is None:
            return super().copyfile(source, outputfile)
        remaining = self.byte_count
        while remaining:
            data = source.read(min(256 * 1024, remaining))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser canceled a video fetch after seeking/switching cases.

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8773)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(GalleryHandler, directory=str(ROOT)))
    print(f'Gallery: http://127.0.0.1:{args.port}/ (localhost only)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
