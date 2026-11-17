"""Validated immutable snapshots; watchdog publishes changes to the camera worker."""
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from io import StringIO
import math
import os
import re
import tempfile
import threading

from dotenv import dotenv_values
from dotenv.parser import parse_stream
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

EDITABLE = {'CAPTURE_ENABLED', 'MIN_CAPTURE_INTERVAL', 'MAX_CAPTURE_INTERVAL',
            'DETECTION_INTERVAL', 'PERSON_CONFIDENCE', 'EMPTY_TIMEOUT', 'SHOW_HIDDEN', 'SHOW_HIDDEN_BUTTON'}


@dataclass(frozen=True)
class Config:
    CAMERA_DEVICE: str = '/dev/video0'
    CAMERA_WIDTH: int = 1920
    CAMERA_HEIGHT: int = 1080
    CAMERA_FORMAT: str = 'MJPG'
    CAMERA_FPS: int = 25
    CAMERA_CYCLE_ENABLED: bool = False
    CAMERA_SNAPSHOT_ENABLED: bool = False
    CAMERA_SNAPSHOT_INTERVAL: float = 300
    DETECTION_INTERVAL: float = 3
    PERSON_CONFIDENCE: float = .5
    EMPTY_TIMEOUT: float = 30
    CAPTURE_ENABLED: bool = True
    MIN_CAPTURE_INTERVAL: float = 120
    MAX_CAPTURE_INTERVAL: float = 300
    SHOW_HIDDEN: bool = True
    SHOW_HIDDEN_BUTTON: bool = True
    PHOTO_DIR: Path = Path('./photos')
    PORT: int = 5418


def load_config(path: Path) -> Config:
    if not path.is_file():
        raise ValueError(f'Missing configuration: {path}; copy .env.example there')
    content = path.read_text()
    if any(binding.error for binding in parse_stream(StringIO(content))):
        raise ValueError('Invalid .env syntax')
    values = dotenv_values(stream=StringIO(content), interpolate=False)
    defaults = asdict(Config())
    for key in defaults:
        if key not in values:
            continue
        value = values[key]
        if value is None or not value.strip():
            raise ValueError(f'{key} must not be empty')
        if key in ('CAPTURE_ENABLED', 'CAMERA_CYCLE_ENABLED', 'CAMERA_SNAPSHOT_ENABLED', 'SHOW_HIDDEN', 'SHOW_HIDDEN_BUTTON'):
            if value.lower() not in ('true', 'false'):
                raise ValueError(f'{key} must be true or false')
            defaults[key] = value.lower() == 'true'
        elif key in ('PORT', 'CAMERA_WIDTH', 'CAMERA_HEIGHT', 'CAMERA_FPS'):
            defaults[key] = int(value)
        elif key == 'PHOTO_DIR':
            defaults[key] = Path(value).expanduser().resolve()
        elif key in ('CAMERA_DEVICE', 'CAMERA_FORMAT'):
            defaults[key] = value
        else:
            defaults[key] = float(value)
    config = Config(**defaults)
    for key in ('CAMERA_WIDTH', 'CAMERA_HEIGHT'):
        if not 1 <= getattr(config, key) <= 16384:
            raise ValueError(f'{key} must be between 1 and 16384')
    if not re.fullmatch(r'[A-Z0-9]{4}', config.CAMERA_FORMAT):
        raise ValueError('CAMERA_FORMAT must be a four-character format code, e.g. MJPG')
    if not 1 <= config.CAMERA_FPS <= 240:
        raise ValueError('CAMERA_FPS must be between 1 and 240')
    for key in ('DETECTION_INTERVAL', 'MIN_CAPTURE_INTERVAL', 'MAX_CAPTURE_INTERVAL', 'CAMERA_SNAPSHOT_INTERVAL'):
        value = getattr(config, key)
        if not math.isfinite(value) or not 0 < value <= 86400:
            raise ValueError(f'{key} must be greater than 0 and at most 86400')
    if not 0 <= config.PERSON_CONFIDENCE <= 1:
        raise ValueError('PERSON_CONFIDENCE must be between 0 and 1')
    if not 0 <= config.EMPTY_TIMEOUT <= 86400:
        raise ValueError('EMPTY_TIMEOUT must be between 0 and 86400')
    if config.MIN_CAPTURE_INTERVAL > config.MAX_CAPTURE_INTERVAL:
        raise ValueError('Minimum capture interval exceeds maximum')
    if not 1 <= config.PORT <= 65535:
        raise ValueError('PORT must be between 1 and 65535')
    return config


class ConfigStore(FileSystemEventHandler):
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.lock = threading.RLock()
        self.changed = threading.Event()
        self._snapshot = load_config(self.path)
        self.observer = Observer()

    def snapshot(self) -> Config:
        with self.lock:
            return self._snapshot

    def start(self):
        self.observer.schedule(self, str(self.path.parent), recursive=False)
        self.observer.start()
        self.reload()  # Cover changes between the initial read and subscription.

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def on_any_event(self, event):
        if event.is_directory or event.event_type not in ('modified', 'created', 'moved', 'deleted'):
            return
        paths = (event.src_path, getattr(event, 'dest_path', ''))
        if str(self.path) in paths:
            self.reload()

    def reload(self):
        with self.lock:
            try:
                candidate = load_config(self.path)
            except (OSError, ValueError) as error:
                logging.warning('Keeping last valid configuration: %s', error)
                return
            if candidate != self._snapshot:
                if candidate.PORT != self._snapshot.PORT:
                    logging.warning('PORT changed; restart application to apply it')
                self._snapshot = candidate
                self.changed.set()

    def update(self, updates: dict):
        if not updates or not updates.keys() <= EDITABLE:
            raise ValueError('Only supported capture and detection settings may be updated')
        if any(not isinstance(value, (str, int, float, bool)) for value in updates.values()):
            raise ValueError('Settings must be scalar values')
        with self.lock:
            content = self.path.read_text()
            for key, value in updates.items():
                value = str(value).lower() if isinstance(value, bool) else str(value)
                if '\n' in value or '\r' in value or len(value) > 100:
                    raise ValueError('Invalid setting value')
                pattern = re.compile(r'^(\s*(?:export\s+)?' + key + r'\s*=)[^\n]*', re.M)
                # Preserve comments on supported numeric/boolean lines as well as unrelated lines.
                def replace(match):
                    old = match.group(0)
                    comment = ' #' + old.split('#', 1)[1] if '#' in old else ''
                    return match.group(1) + value + comment
                if pattern.search(content):
                    content = pattern.sub(replace, content)
                else:
                    content = content.rstrip('\n') + f'\n{key}={value}\n'
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as file:
                    temporary = Path(file.name)
                    file.write(content)
                    file.flush()
                    os.fsync(file.fileno())
                load_config(temporary)
                os.replace(temporary, self.path)
                # Watchdog performs publication; HTTP never touches camera state.
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
