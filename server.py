"""Plain HTML gallery and narrowly scoped JSON settings endpoints."""
from dataclasses import asdict
from collections import deque
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import shutil
import time
from urllib.parse import quote, unquote, urlsplit

from config_store import EDITABLE
from camera import save_photo

TEMPLATE = Path(__file__).parent / 'static/index.html'
PHOTO_NAME = re.compile(r'^\d{8}_\d{6}(?:_[a-zA-Z0-9]+)?\.jpg$')


def photo_caption(name):
    return datetime.strptime(name[:15], '%Y%m%d_%H%M%S').strftime('%H:%M:%S %Y-%m-%d')


class RecentLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = deque(maxlen=100)
        self.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S'))

    def emit(self, record):
        self.lines.append(self.format(record))

    def snapshot(self):
        self.acquire()
        try:
            return list(self.lines)
        finally:
            self.release()


def make_server(store, camera=None, logs=None):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path
            config = store.snapshot()
            if route == '/logs':
                self.send(200, json.dumps(logs.snapshot() if logs is not None else []).encode())
            elif route == '/live.jpg':
                if config.CAMERA_SNAPSHOT_ENABLED:
                    self.send(503, b'{"error": "Live preview is disabled in snapshot mode"}')
                    return
                preview = camera.preview_frame if camera is not None else None
                if preview is None or time.monotonic() - preview[0] > 3:
                    self.send(503, b'{"error": "Camera unavailable"}')
                    return
                import cv2
                try:
                    frame = preview[1]
                    height, width = frame.shape[:2]
                    if width > 960:
                        frame = cv2.resize(frame, (960, max(1, round(height * 960 / width))))
                    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if not ok:
                        raise ValueError('Unable to encode live frame')
                    self.send(200, encoded.tobytes(), 'image/jpeg')
                except (cv2.error, ValueError):
                    self.send(503, b'{"error": "Live frame unavailable"}')
            elif route == '/settings':
                values = asdict(config)
                self.send(200, json.dumps({key: values[key] for key in sorted(EDITABLE)}).encode())
            elif route == '/':
                photos = []
                for path in sorted(config.PHOTO_DIR.glob('*.jpg'), reverse=True):
                    if not PHOTO_NAME.fullmatch(path.name) or path.is_symlink() or not path.is_file():
                        continue
                    try:
                        caption = photo_caption(path.name)
                    except ValueError:
                        continue
                    photos.append(f'<figure><a href="/photos/{quote(path.name)}"><img src="/photos/{quote(path.name)}" loading="lazy" alt=""></a><figcaption>{escape(caption)}</figcaption></figure>')
                try:
                    body = (TEMPLATE.read_text().replace('{{photos}}', '\n'.join(photos))
                            .replace('{{snapshot_mode}}', 'true' if config.CAMERA_SNAPSHOT_ENABLED else 'false').encode())
                    self.send(200, body, 'text/html; charset=utf-8')
                except OSError:
                    logging.exception('Cannot read gallery template')
                    self.send_error(500)
            elif route.startswith('/photos/'):
                name = unquote(route[len('/photos/'):])
                if not PHOTO_NAME.fullmatch(name):
                    self.send_error(404)
                    return
                path = config.PHOTO_DIR / name
                if path.is_symlink():
                    self.send_error(404)
                    return
                try:
                    with path.open('rb') as file:
                        self.send_response(200)
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(path.stat().st_size))
                        self.send_header('X-Content-Type-Options', 'nosniff')
                        self.end_headers()
                        shutil.copyfileobj(file, self.wfile)
                except FileNotFoundError:
                    self.send_error(404)
                except OSError:
                    logging.exception('Cannot serve photograph')
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path not in ('/settings', '/capture'):
                self.send_error(404)
                return
            # JSON-only writes prevent cross-origin HTML forms from changing settings.
            origin = self.headers.get('Origin')
            if (self.headers.get('Content-Type', '').split(';')[0] != 'application/json'
                    or self.headers.get('Sec-Fetch-Site') == 'cross-site'
                    or (origin is not None and origin != 'http://' + self.headers.get('Host', ''))):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('Invalid request size')
                self.connection.settimeout(5)
                updates = json.loads(self.rfile.read(length))
                if not isinstance(updates, dict):
                    raise ValueError('Expected a settings object')
                if self.path == '/capture':
                    if store.snapshot().CAMERA_SNAPSHOT_ENABLED:
                        self.send(409, b'{"error": "Manual capture is disabled in snapshot mode"}')
                        return
                    preview = camera.preview_frame if camera is not None else None
                    if preview is None or time.monotonic() - preview[0] > 3:
                        self.send(503, b'{"error": "Camera unavailable"}')
                        return
                    import cv2
                    try:
                        name = save_photo(preview[1], store.snapshot().PHOTO_DIR)
                    except (OSError, cv2.error):
                        logging.exception('Cannot save manual photograph')
                        self.send(500, b'{"error": "Unable to save photo"}')
                        return
                    self.send(200, json.dumps({'url': '/photos/' + name, 'caption': photo_caption(name)}).encode())
                    return
                store.update(updates)
                self.send(200, b'{"saved": true}')
            except (ValueError, UnicodeError, TimeoutError) as error:
                self.send(400, json.dumps({'error': str(error)}).encode())
            except OSError:
                logging.exception('Cannot update settings')
                self.send(500, b'{"error": "Unable to write configuration"}')

    server = ThreadingHTTPServer(('0.0.0.0', store.snapshot().PORT), Handler)
    server.daemon_threads = True
    return server
