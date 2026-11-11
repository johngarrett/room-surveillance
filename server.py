"""Plain HTML gallery and narrowly scoped JSON settings endpoints."""
from dataclasses import asdict
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote, urlsplit

from config import EDITABLE

TEMPLATE = Path(__file__).parent / 'static/index.html'
PHOTO_NAME = re.compile(r'^\d{8}_\d{6}(?:_[a-zA-Z0-9]+)?\.jpg$')


def make_server(store):
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
            if route == '/settings':
                values = asdict(config)
                self.send(200, json.dumps({key: values[key] for key in sorted(EDITABLE)}).encode())
            elif route == '/':
                photos = []
                for path in sorted(config.PHOTO_DIR.glob('*.jpg'), reverse=True):
                    if not PHOTO_NAME.fullmatch(path.name) or path.is_symlink() or not path.is_file():
                        continue
                    try:
                        caption = datetime.strptime(path.name[:15], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        continue
                    photos.append(f'<figure><a href="/photos/{quote(path.name)}"><img src="/photos/{quote(path.name)}" loading="lazy" alt=""></a><figcaption>{escape(caption)}</figcaption></figure>')
                try:
                    body = TEMPLATE.read_text().replace('{{photos}}', '\n'.join(photos)).encode()
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
            if self.path != '/settings':
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
