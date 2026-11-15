"""Continuously drain the camera; detection and saves use the latest original frame."""
from datetime import datetime
import logging
import platform
import threading
import time
import uuid

from scheduler import Scheduler


def save_photo(frame, directory):
    import cv2
    directory.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.jpg'
    target = directory / name
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError('JPEG encoding failed')
    temporary = target.with_suffix('.tmp')
    try:
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    logging.info('Saved %s', target)
    return name


class Occupancy:
    def __init__(self):
        self.occupied = False
        self.positives = 0
        self.last_positive = None

    def update(self, positive, now, timeout):
        self.positives = self.positives + 1 if positive else 0
        if positive:
            self.last_positive = now
        if self.positives >= 2:
            self.occupied = True
        if not positive and self.last_positive is not None and now - self.last_positive >= timeout:
            self.occupied = False
        return self.occupied


class CameraWorker(threading.Thread):
    def __init__(self, store, detector, stop):
        super().__init__(name='camera')
        self.store, self.detector, self.stop_event = store, detector, stop
        self.preview_frame = None

    def run(self):
        import cv2
        capture = None
        device = None
        occupancy = Occupancy()
        scheduler = Scheduler()
        next_detection = 0
        previous_config = None
        try:
            while not self.stop_event.is_set():
                self.store.changed.clear()
                config = self.store.snapshot()
                changed = config != previous_config
                previous_config = config
                if capture is None or device != config.CAMERA_DEVICE:
                    self.preview_frame = None
                    if capture is not None:
                        capture.release()
                    occupancy = Occupancy()
                    scheduler = Scheduler()
                    device = config.CAMERA_DEVICE
                    source = int(device) if device.isdecimal() else device
                    backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_V4L2
                    capture = cv2.VideoCapture(source, backend)
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    next_detection = 0
                ok, frame = capture.read()
                if not ok:
                    self.preview_frame = None
                    logging.warning('Camera unavailable: %s; retrying in 3 seconds', device)
                    capture.release()
                    capture = None
                    self.stop_event.wait(3)
                    continue
                now = time.monotonic()
                self.preview_frame = (now, frame)
                if changed:
                    next_detection = now
                if now < next_detection:
                    continue
                next_detection = now + config.DETECTION_INTERVAL
                try:
                    positive = self.detector.has_person(frame, config.PERSON_CONFIDENCE)
                except Exception:
                    logging.exception('Detection failed; discarding frame')
                    continue
                logging.info('PERSON %s (confidence threshold %.2f)',
                             'detected' if positive else 'not detected', config.PERSON_CONFIDENCE)
                was_occupied = occupancy.occupied
                occupied = occupancy.update(positive, now, config.EMPTY_TIMEOUT)
                if was_occupied != occupied:
                    logging.info('Room %s', 'occupied' if occupied else 'empty')
                scheduler.update(now, occupied, config)
                if positive and scheduler.due(now):
                    # A settings change during inference must be applied before saving.
                    if self.store.snapshot() != config:
                        continue
                    try:
                        save_photo(frame, config.PHOTO_DIR)
                    except (OSError, cv2.error):
                        logging.exception('Unable to save photograph')
                    scheduler.schedule(time.monotonic(), config)
        finally:
            self.preview_frame = None
            if capture is not None:
                capture.release()
