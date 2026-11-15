"""One process: camera worker, watchdog observer, and threaded HTTP serving."""
import logging
from pathlib import Path
import signal
import threading

from camera import CameraWorker
from config import ConfigStore
from detector import Detector
from server import make_server, RecentLogs


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logs = RecentLogs()
    logging.getLogger().addHandler(logs)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    store = None
    server = None
    camera = None
    http_thread = None
    watching = False
    try:
        store = ConfigStore(Path('config/.env'))
        detector = Detector()
        camera = CameraWorker(store, detector, stop)
        server = make_server(store, camera, logs)
        store.start()
        watching = True
        http_thread = threading.Thread(target=server.serve_forever, name='http')
        http_thread.start()
        logging.info('Gallery listening on port %s', server.server_port)
        # macOS camera permission requests require the main thread.
        camera.run()
    except Exception:
        logging.exception('Application failed')
        return 1
    finally:
        stop.set()
        if server is not None:
            if http_thread is not None:
                server.shutdown()
                http_thread.join()
            server.server_close()
        if camera is not None and camera.ident is not None:
            camera.join()
        if watching:
            store.stop()
        logging.getLogger().removeHandler(logs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
