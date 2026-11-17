# Room Camera

A local webcam takes occasional full-resolution JPEGs while a person is present.
One Python process serves a plain HTML gallery and settings on your trusted LAN.
No accounts, database, recorded video, audio, face recognition, or cloud service.
The live preview refreshes up to five times per second while the page is visible;
it uses the same camera as detection and does not save preview images.
Use **take photo** below the live view to save the current full-resolution frame
immediately, even when automatic capture is off or no person is detected.

## Native setup (Python 3.12+)

Run `./script/startup`. It creates the local Python environment and configuration
on first use, defaults to camera 0 on macOS, and preserves existing settings.
Allow camera access when prompted; Ctrl-C stops the app.

Manual setup:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdir -p config photos
cp .env.example config/.env
# On macOS, edit config/.env and set CAMERA_DEVICE=0.
python main.py
```

Open http://localhost:5418 or http://YOUR-LAN-IP:5418 from another device.
Allow camera access for your terminal on macOS. Use native execution on macOS;
Docker Desktop does not provide the Linux webcam passthrough used here.
The first native launch downloads `yolo11n.pt` into the working directory;
subsequent launches reuse it. Keep it there for offline operation.

The camera defaults to 1920×1080 at 25 fps and requests MJPEG (`MJPG`) on Linux before
setting resolution. macOS chooses its native capture format. Configure these in
`config/.env`; mode changes reopen the camera automatically. The driver can
negotiate another resolution; startup logs show the requested and negotiated size.
Detection uses a 320px inference image, while saved photographs use the original
frame at the negotiated camera resolution. Verify resolution on your hardware.

## Linux Docker

On the Linux host, run `./script/cameras` to list attached camera names, device
paths, and matching config/Compose settings. Run it outside Docker so it can see
all host cameras. Some webcams expose metadata nodes as well as capture nodes;
use `v4l2-ctl --device=/dev/videoN --all` to check for video capture capability.
The website can switch only among devices available to the process. Docker must
map the camera you want to use and set `CAMERA_DEVICE` in `config/.env`.

```sh
mkdir -p config photos
cp .env.example config/.env
# Confirm CAMERA_DEVICE=/dev/video0, PHOTO_DIR=./photos, PORT=5418.
docker compose up --build -d
docker compose logs -f
```

The build installs CPU PyTorch and packages YOLO11 nano weights in the image.
The base image is pinned by digest, Python dependency versions are constrained by
`requirements-docker.lock`, and the model checksum is verified during the build.
Update these deliberately when upgrading. Debian package repositories are not
snapshotted; deploy the same built image digest across machines for identical artifacts.
A network connection is required for building, but not for normal startup.
Photos and configuration are bind mounted from `./photos` and `./config` and
survive container rebuilds. Back them up yourself. Nothing deletes old photos.
The Linux host's `/etc/localtime` is mounted read-only so filenames use local time.
For a different Linux webcam, change both the Compose device mapping and
`CAMERA_DEVICE`. The container needs permission to access that device.

Stop with `docker compose down`, or Ctrl-C for native execution. SIGINT/SIGTERM
stop the HTTP server, camera worker, and filesystem observer and release the camera.
A camera driver blocked inside a native read may delay shutdown; Compose forces
termination after its 30-second grace period.

## Configuration

`config/.env` is the source of truth; shell environment variables and Compose
`env_file` are not used. Paths are relative to the process working directory.
`config_store.py` contains configuration loading and validation code; edit
`config/.env` to change your settings.
Run native commands from this project directory. The whole config directory is
mounted so editor atomic replacements remain visible inside Docker.

| Key | Default | Meaning |
| --- | --- | --- |
| CAMERA_DEVICE | /dev/video0 | Linux path or numeric camera index (0 on macOS) |
| CAMERA_WIDTH | 1920 | Requested photo width in pixels, 1–16384 |
| CAMERA_HEIGHT | 1080 | Requested photo height in pixels, 1–16384 |
| CAMERA_FORMAT | MJPG | Four-character Linux capture format code; MJPG means MJPEG |
| CAMERA_FPS | 25 | Requested capture frame rate, 1–240; must be supported by the camera |
| CAMERA_CYCLE_ENABLED | false | Workaround: alternate 5 seconds streaming with 20 seconds released |
| CAMERA_SNAPSHOT_ENABLED | false | Open briefly for one frame, release, then detect and save only if a person is found |
| CAMERA_SNAPSHOT_INTERVAL | 300 | Legacy setting; cooldown bounds now control snapshot attempts |
| DETECTION_INTERVAL | 3 | Seconds between inference checks |
| PERSON_CONFIDENCE | 0.5 | Person confidence threshold, 0–1 |
| EMPTY_TIMEOUT | 30 | Seconds since last positive detection before vacancy |
| CAPTURE_ENABLED | true | `true` or `false` |
| MIN_CAPTURE_INTERVAL | 120 | Minimum seconds between photos |
| MAX_CAPTURE_INTERVAL | 300 | Maximum seconds between photos |
| PHOTO_DIR | ./photos | Output directory |
| PORT | 5418 | HTTP port; restart required to change |

Positive intervals are limited to 86400 seconds; empty timeout allows 0–86400.
For periodic photos without a continuous stream, set `CAMERA_SNAPSHOT_ENABLED=true`
and cooldown bounds `MIN_CAPTURE_INTERVAL=120`, `MAX_CAPTURE_INTERVAL=300` in `config/.env`. The first attempt is immediate;
later attempts wait a newly chosen random cooldown after the previous attempt finishes. Each attempt
opens the configured camera, reads one full-resolution frame, releases the camera,
then runs detection. One positive detection saves that frame; otherwise it is
discarded without writing an image to disk. Each completed attempt, including failures, draws a new random cooldown between the configured bounds.
`CAPTURE_ENABLED=false` prevents snapshot attempts entirely. Snapshot mode ignores
the continuous detection, vacancy, and camera cycling
settings, but uses `PERSON_CONFIDENCE` and the configured camera mode. Live preview
and manual capture are disabled; reload the page after changing modes. Existing
photos remain available. Configuration edits interrupt the wait within a second;
shutdown interrupts it immediately. Reopening can still fail, and blocked camera
calls cannot be forcibly interrupted by this mode. This is not USB power cycling.

Set `CAMERA_CYCLE_ENABLED=true` in `config/.env` to try periodic camera restarts.
Live preview, detection, and manual photos are unavailable during each pause.
Photo deadlines survive planned pauses; a due photo still needs a fresh person
detection after reopening. Shutdown interrupts the pause immediately. This releases
the camera driver without cutting USB power. It cannot interrupt a blocked native
camera call, and is an experimental workaround rather than a confirmed USB fix.
Minimum capture interval cannot exceed maximum. Invalid edits keep the previous
valid snapshot and log a warning. Invalid startup configuration fails clearly.
Missing keys use defaults. Edit settings through the small `settings` disclosure
or a text editor. Web settings permit capture/detection values. Set camera devices,
storage paths, and ports in `config/.env`. Writes are validated and atomic and
preserve comments and unrelated settings. Avoid simultaneous web/editor saves:
as with ordinary files, the last writer wins.

Watchdog subscribes to directory events; configuration is never polled. Changes
apply on the next camera frame. Camera changes reopen the device and reset
occupancy; photo directory changes affect subsequent saves and gallery requests.
A bad/unwritable photo directory logs save failures without deleting old photos.
Changing directories leaves existing images in their original location. Port
changes require restarting; Docker also requires a matching Compose port mapping.
Keep port 5418 in the initial Docker deployment.

Two consecutive positive detections establish occupancy. Vacancy cancels the
schedule. Re-enabling photography or changing interval bounds starts a new
random interval. A due capture requires a positive detection in that very frame;
otherwise it waits. Inference-cycle precision is sufficient: photographs can be
delayed by one detection interval plus inference time. Pauses never produce a
catch-up burst. Camera read failures reset occupancy and retry after 3 seconds
(20 seconds with camera cycling enabled).

Photos use local timestamps, with a unique suffix, e.g.
`20261111_102305_a1b2c3d4.jpg`. The gallery sorts these names newest first, loads
JPEGs separately and needs a page reload to show new photos. Edit
`static/index.html` directly; it is reread on every gallery request.

## Tests

Tests use temporary files, fake model output, and a local HTTP server; no camera
or downloaded model is needed:

```sh
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Before relying on a deployment, test a real Linux camera: seated and standing
people, false positives in the empty room, full saved resolution, disconnect and
reconnect, CPU/RAM over several hours, and SIGTERM. Check web and external editor
updates on the target filesystem, including atomic replacement. Verify LAN
access and persistence across `docker compose down` / `up --build`.

This is an unauthenticated service for a trusted local network. Do not publish
its port to the internet. Anyone on that network can see photos and change the
exposed settings. Only photograph people with their knowledge and consent.
YOLO can miss people or produce false positives; test thresholds in your room.

Run the full suite in the deployment image (without camera passthrough):

```sh
docker compose build
docker run --rm -v "$PWD/tests:/app/tests:ro" \
  webcam-room-photos-room-camera python -m unittest discover -s tests -v
```

The default Compose image name above follows this repository directory name;
use the image name printed by `docker compose images` if you rename the directory.

Implementation verification: the image builds on Linux/ARM64 through Docker
Desktop; all 11 hardware-free tests pass inside it. YOLO initialization and
synthetic empty-frame inference also pass with `--network none`. A camera-less
application process retries the unavailable device and stops promptly on SIGTERM.
Real Linux webcam capture, Linux/AMD64, LAN access, and long-running resource
usage still require deployment hardware. An extra running-process HTTP smoke
check was blocked by local Docker-exec permissions; HTTP and live configuration
behavior are covered by the container test suite.

Gallery settings include **show hidden** and **show hidden button**, both on by default. Hide/unhide writes image names to the `hidden` array in `images/metadata`; photo files remain accessible and are never deleted. Compose persists this directory. Cooldown bounds apply to continuous capture and snapshot attempts; the old `CAMERA_SNAPSHOT_INTERVAL` setting is retained for config compatibility but no longer controls the wait.
