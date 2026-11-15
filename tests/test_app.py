from dataclasses import replace
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

from camera import Occupancy
from config_store import Config, ConfigStore, load_config
from detector import Detector
from scheduler import Scheduler
from server import make_server


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / '.env'
        self.path.write_text('# keep me\nMIN_CAPTURE_INTERVAL=60 # interval\nUNRELATED=hello\nPORT=5418\n')
        self.store = ConfigStore(self.path)

    def test_validation(self):
        for text in ('DETECTION_INTERVAL=0', 'MIN_CAPTURE_INTERVAL=500',
                     'PERSON_CONFIDENCE=nan', 'EMPTY_TIMEOUT=inf',
                     'CAPTURE_ENABLED=yes', 'PORT=0', 'MAX_CAPTURE_INTERVAL=nan',
                     'PERSON_CONFIDENCE="unterminated', 'CAMERA_WIDTH=0',
                     'CAMERA_HEIGHT=16385', 'CAMERA_WIDTH=1.5', 'CAMERA_FORMAT=MJPEG',
                     'CAMERA_CYCLE_ENABLED=yes'):
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(self.path)

    def test_camera_mode_configuration(self):
        config = load_config(self.path)
        self.assertEqual((1920, 1080, 'MJPG'), (config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FORMAT))
        self.path.write_text('CAMERA_WIDTH=1280\nCAMERA_HEIGHT=720\nCAMERA_FORMAT=YUYV\n')
        config = load_config(self.path)
        self.assertEqual((1280, 720, 'YUYV'), (config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FORMAT))

    def test_camera_cycling_is_opt_in(self):
        self.assertFalse(load_config(self.path).CAMERA_CYCLE_ENABLED)
        self.path.write_text('CAMERA_CYCLE_ENABLED=true\n')
        self.assertTrue(load_config(self.path).CAMERA_CYCLE_ENABLED)

    def test_invalid_reload_keeps_snapshot(self):
        before = self.store.snapshot()
        self.path.write_text('DETECTION_INTERVAL=-1')
        self.store.reload()
        self.assertEqual(before, self.store.snapshot())

    def test_camera_selection(self):
        before = self.path.read_text()
        for device in ('0', '/dev/video2'):
            with self.assertRaises(ValueError):
                self.store.update({'CAMERA_DEVICE': device})
            self.assertEqual(before, self.path.read_text())

    def test_web_update_preserves_comments_and_other_keys(self):
        self.store.update({'MIN_CAPTURE_INTERVAL': 90, 'CAPTURE_ENABLED': False})
        text = self.path.read_text()
        self.assertIn('# keep me', text)
        self.assertIn('MIN_CAPTURE_INTERVAL=90 # interval', text)
        self.assertIn('UNRELATED=hello', text)
        self.assertFalse(load_config(self.path).CAPTURE_ENABLED)
        before = text
        for updates in ({'PHOTO_DIR': '/tmp'}, {'MIN_CAPTURE_INTERVAL': 999},
                        {'CAPTURE_ENABLED': 'true\nPORT=1'}, {'EMPTY_TIMEOUT': None}):
            with self.assertRaises(ValueError):
                self.store.update(updates)
            self.assertEqual(before, self.path.read_text())

    def wait_value(self, value):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if self.store.snapshot().DETECTION_INTERVAL == value:
                return
            self.store.changed.wait(.05)
            self.store.changed.clear()
        self.fail('Filesystem event did not publish configuration')

    def test_watcher_edits_replacements_and_web_writes(self):
        self.store.start()
        self.addCleanup(self.store.stop)
        self.path.write_text('DETECTION_INTERVAL=4')
        self.wait_value(4)
        replacement = self.path.with_name('replacement')
        replacement.write_text('DETECTION_INTERVAL=5')
        replacement.replace(self.path)
        self.wait_value(5)
        self.store.update({'DETECTION_INTERVAL': 6})
        self.wait_value(6)


class SchedulerTests(unittest.TestCase):
    def test_lifecycle(self):
        config = Config()
        scheduler = Scheduler()
        scheduler.update(0, False, config)
        self.assertFalse(scheduler.due(1000))
        scheduler.update(10, True, config)
        self.assertTrue(70 <= scheduler.deadline <= 310)
        deadline = scheduler.deadline
        scheduler.update(11, True, config)
        self.assertEqual(deadline, scheduler.deadline)
        self.assertTrue(scheduler.due(deadline))
        config = replace(config, MIN_CAPTURE_INTERVAL=2, MAX_CAPTURE_INTERVAL=2)
        scheduler.update(20, True, config)
        self.assertEqual(22, scheduler.deadline)
        scheduler.update(21, True, replace(config, CAPTURE_ENABLED=False))
        self.assertIsNone(scheduler.deadline)
        scheduler.update(22, True, config)
        self.assertEqual(24, scheduler.deadline)
        scheduler.update(23, False, config)
        self.assertIsNone(scheduler.deadline)
        scheduler.update(1000, True, config)
        self.assertEqual(1002, scheduler.deadline)

    def test_occupancy(self):
        room = Occupancy()
        self.assertFalse(room.update(True, 0, 30))
        self.assertFalse(room.update(False, 3, 30))
        self.assertFalse(room.update(True, 6, 30))
        self.assertTrue(room.update(True, 9, 30))
        self.assertTrue(room.update(False, 38, 30))
        self.assertFalse(room.update(False, 39, 30))
        self.assertFalse(room.update(True, 40, 30))


class DetectorTests(unittest.TestCase):
    def test_class_and_confidence(self):
        model = Mock()
        detector = Detector(model)
        frame = object()
        for class_id, confidence, expected in ((0, .8, True), (1, .8, False), (0, .2, False)):
            box = Mock()
            box.cls.item.return_value = class_id
            box.conf.item.return_value = confidence
            model.predict.return_value = [Mock(boxes=[box])]
            self.assertEqual(expected, detector.has_person(frame, .5))
        model.predict.return_value = [Mock(boxes=[])]
        self.assertFalse(detector.has_person(frame, .5))
        self.assertIs(model.predict.call_args.args[0], frame)
        self.assertEqual(320, model.predict.call_args.kwargs['imgsz'])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.photos = root / 'photos'
        self.photos.mkdir()
        self.path = root / '.env'
        self.path.write_text(f'PHOTO_DIR={self.photos}\n')
        self.store = ConfigStore(self.path)
        # Port zero is assigned by the OS, only for this test server.
        self.store._snapshot = replace(self.store.snapshot(), PORT=0)
        self.camera = Mock(preview_frame=None)
        self.server = make_server(self.store, self.camera)
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
        def cleanup():
            self.server.shutdown()
            thread.join()
            self.server.server_close()
        self.addCleanup(cleanup)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        self.addCleanup(connection.close)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        return response.status, response.read(), response.headers

    def test_gallery_and_photo_boundaries(self):
        for name in ('20261111_102305.jpg', '20261112_102305_abcd.jpg'):
            (self.photos / name).write_bytes(b'jpeg')
        (self.photos / '20261113_102305.jpg').symlink_to(self.path)
        status, body, _ = self.request('GET', '/')
        self.assertEqual(200, status)
        text = body.decode()
        self.assertLess(text.index('2026-11-12'), text.index('2026-11-11'))
        self.assertNotIn('2026-11-13', text)
        self.assertIn('10:23:05 2026-11-11', text)
        status, body, headers = self.request('GET', '/photos/20261111_102305.jpg')
        self.assertEqual((200, b'jpeg', 'image/jpeg'), (status, body, headers['Content-Type']))
        for path in ('/.env', '/photos/../.env', '/photos/%2e%2e%2f.env',
                     '/photos/20261113_102305.jpg', '/photos/20261110_102305.jpg'):
            self.assertEqual(404, self.request('GET', path)[0])

    def test_settings(self):
        status, body, _ = self.request('GET', '/settings')
        self.assertEqual(200, status)
        self.assertNotIn('PHOTO_DIR', json.loads(body))
        self.assertNotIn('CAMERA_DEVICE', json.loads(body))
        headers = {'Content-Type': 'application/json'}
        self.assertEqual(400, self.request('POST', '/settings', '{"CAMERA_DEVICE": "1"}', headers)[0])
        self.assertEqual(200, self.request('POST', '/settings', '{"EMPTY_TIMEOUT": 10}', headers)[0])
        self.assertEqual(10, load_config(self.path).EMPTY_TIMEOUT)
        for body in ('{"PORT": 1}', '{"EMPTY_TIMEOUT": -1}', '[]', 'invalid'):
            self.assertEqual(400, self.request('POST', '/settings', body, headers)[0])
        self.assertEqual(403, self.request('POST', '/settings', '{}')[0])
        self.assertEqual(403, self.request('POST', '/settings', '{}',
                         {**headers, 'Origin': 'http://evil.example'})[0])

    def test_live_feed_and_unavailable_camera(self):
        import cv2
        import numpy as np
        self.assertEqual(503, self.request('GET', '/live.jpg')[0])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.camera.preview_frame = (time.monotonic(), frame)
        status, body, headers = self.request('GET', '/live.jpg?t=1')
        self.assertEqual(200, status)
        self.assertEqual('image/jpeg', headers['Content-Type'])
        self.assertEqual('no-store', headers['Cache-Control'])
        decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((540, 960, 3), decoded.shape)
        self.camera.preview_frame = (time.monotonic() - 4, frame)
        self.assertEqual(503, self.request('GET', '/live.jpg')[0])

    def test_manual_capture(self):
        import cv2
        import numpy as np
        headers = {'Content-Type': 'application/json'}
        self.assertEqual(403, self.request('POST', '/capture', '{}')[0])
        self.assertEqual(503, self.request('POST', '/capture', '{}', headers)[0])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera.preview_frame = (time.monotonic(), frame)
        status, body, _ = self.request('POST', '/capture', '{}', headers)
        self.assertEqual(200, status)
        name = json.loads(body)['url'].split('/')[-1]
        self.assertEqual(f'{name[9:11]}:{name[11:13]}:{name[13:15]} {name[:4]}-{name[4:6]}-{name[6:8]}', json.loads(body)['caption'])
        self.assertEqual(frame.shape, cv2.imread(str(self.photos / name)).shape)
        self.assertEqual([], list(self.photos.glob('*.tmp')))
        self.camera.preview_frame = (time.monotonic() - 4, frame)
        self.assertEqual(503, self.request('POST', '/capture', '{}', headers)[0])


class CameraTests(unittest.TestCase):
    def test_planned_cycle_preserves_photo_deadline(self):
        import cv2
        import numpy as np
        from camera import CameraWorker
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            store = Mock()
            store.snapshot.return_value = replace(Config(), CAMERA_CYCLE_ENABLED=True,
                PHOTO_DIR=Path(directory), DETECTION_INTERVAL=1,
                MIN_CAPTURE_INTERVAL=30, MAX_CAPTURE_INTERVAL=30)
            store.changed = threading.Event()
            stop = threading.Event()
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            capture = Mock()
            def read():
                if capture.read.call_count == 3:
                    stop.set()
                return True, frame
            capture.read.side_effect = read
            detector = Mock()
            detector.has_person.return_value = True
            worker = CameraWorker(store, detector, stop)
            def pause(seconds):
                self.assertEqual(20, seconds)
                self.assertIsNone(worker.preview_frame)
                self.assertEqual(1, capture.release.call_count)
                self.assertEqual([], list(Path(directory).glob('*.jpg')))
            with patch('cv2.VideoCapture', return_value=capture) as open_camera, \
                    patch('camera.time.monotonic', side_effect=[0, 1, 2, 2, 21, 41, 42, 42]), \
                    patch.object(stop, 'wait', side_effect=pause) as wait:
                worker.run()
            wait.assert_called_once_with(20)
            self.assertEqual(2, open_camera.call_count)
            self.assertEqual(2, capture.release.call_count)
            photos = list(Path(directory).glob('*.jpg'))
            self.assertEqual(1, len(photos))
            self.assertEqual(frame.shape, cv2.imread(str(photos[0])).shape)
            self.assertIsNone(worker.preview_frame)

    def test_stop_during_cycle_pause_does_not_reopen(self):
        from camera import CameraWorker
        from unittest.mock import patch
        store = Mock()
        store.snapshot.return_value = replace(Config(), CAMERA_CYCLE_ENABLED=True)
        store.changed = threading.Event()
        stop = threading.Event()
        capture = Mock()
        capture.read.return_value = (True, object())
        detector = Mock()
        detector.has_person.return_value = False
        with patch('cv2.VideoCapture', return_value=capture) as open_camera, \
                patch('camera.time.monotonic', side_effect=[0, 1, 21]), \
                patch.object(stop, 'wait', side_effect=lambda _: stop.set()) as wait:
            CameraWorker(store, detector, stop).run()
        wait.assert_called_once_with(20)
        open_camera.assert_called_once()
        capture.release.assert_called_once()

    def test_failed_capture_uses_cycle_pause_only_when_enabled(self):
        from camera import CameraWorker
        from unittest.mock import patch
        for enabled, delay in ((False, 3), (True, 20)):
            with self.subTest(enabled=enabled):
                store = Mock()
                store.snapshot.return_value = replace(Config(), CAMERA_CYCLE_ENABLED=enabled)
                store.changed = threading.Event()
                stop = threading.Event()
                capture = Mock()
                capture.read.return_value = (False, None)
                with patch('cv2.VideoCapture', return_value=capture), \
                        patch.object(stop, 'wait', side_effect=lambda _: stop.set()) as wait:
                    CameraWorker(store, Mock(), stop).run()
                wait.assert_called_once_with(delay)
                capture.release.assert_called_once()

    def test_camera_mode_changes_reopen_device(self):
        from camera import CameraWorker
        from unittest.mock import patch
        store = Mock()
        config = Config()
        changed = replace(config, CAMERA_WIDTH=1280, CAMERA_HEIGHT=720, CAMERA_FORMAT='YUYV')
        store.snapshot.side_effect = [config, changed]
        store.changed = threading.Event()
        stop = threading.Event()
        capture = Mock()
        def read():
            if capture.read.call_count == 2:
                stop.set()
                return False, None
            return True, object()
        capture.read.side_effect = read
        detector = Mock()
        detector.has_person.return_value = False
        with patch('cv2.VideoCapture', return_value=capture) as open_camera:
            CameraWorker(store, detector, stop).run()
        self.assertEqual(2, open_camera.call_count)
        self.assertEqual(2, capture.release.call_count)

    def test_failed_encoding_does_not_kill_worker(self):
        import cv2
        from camera import CameraWorker
        from unittest.mock import patch
        store = Mock()
        store.snapshot.return_value = replace(Config(), MIN_CAPTURE_INTERVAL=1,
                                               MAX_CAPTURE_INTERVAL=1, DETECTION_INTERVAL=1)
        store.changed = threading.Event()
        stop = threading.Event()
        now = [-1]
        def read():
            now[0] += 1
            if now[0] == 3:
                stop.set()
                return False, None
            return True, object()
        capture = Mock()
        capture.read.side_effect = read
        detector = Mock()
        detector.has_person.return_value = True
        with patch('cv2.VideoCapture', return_value=capture), \
                patch('camera.time.monotonic', side_effect=lambda: now[0]), \
                patch('cv2.imencode', side_effect=cv2.error('encoding failed')) as encode, \
                self.assertLogs(level='ERROR'):
            CameraWorker(store, detector, stop).run()
        encode.assert_called_once()
        capture.release.assert_called()

    def test_only_current_positive_frames_saved_at_original_resolution(self):
        import cv2
        import numpy as np
        from camera import CameraWorker
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            config = replace(Config(), PHOTO_DIR=Path(directory), DETECTION_INTERVAL=1,
                             EMPTY_TIMEOUT=0, MIN_CAPTURE_INTERVAL=2, MAX_CAPTURE_INTERVAL=2)
            store = Mock()
            store.snapshot.return_value = config
            store.changed = threading.Event()
            stop = threading.Event()
            now = [-1]
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            positives = [True, True, False, True, True, False, True, True, True, True]
            def read():
                now[0] += 1
                if now[0] >= len(positives):
                    stop.set()
                    return False, None
                return True, frame.copy()
            capture = Mock()
            capture.read.side_effect = read
            detector = Mock()
            detector.has_person.side_effect = positives
            with patch('cv2.VideoCapture', return_value=capture), patch('camera.platform.system', return_value='Linux'), patch('camera.time.monotonic', side_effect=lambda: now[0]), self.assertLogs(level='INFO') as logs:
                CameraWorker(store, detector, stop).run()
            from unittest.mock import call
            self.assertEqual([
                call(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                call(cv2.CAP_PROP_FRAME_WIDTH, 1920),
                call(cv2.CAP_PROP_FRAME_HEIGHT, 1080),
            ], capture.set.call_args_list[:3])
            self.assertEqual(sum(positives), sum('PERSON detected' in line for line in logs.output))
            self.assertEqual(len(positives) - sum(positives), sum('PERSON not detected' in line for line in logs.output))
            photos = list(Path(directory).glob('*.jpg'))
            self.assertEqual(1, len(photos))
            self.assertEqual(frame.shape, cv2.imread(str(photos[0])).shape)
            self.assertEqual([], list(Path(directory).glob('*.tmp')))
            capture.release.assert_called()


if __name__ == '__main__':
    unittest.main()
