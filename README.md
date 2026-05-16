# Race Vision (CV1) — Mini Racing Telemetry with Webcam

Python project using OpenCV for miniature race telemetry. Tracks multiple
vehicles via colored markers, detects hand-drawn gates (two posts, a
connecting line, and a digit as the gate ID), and runs complete races with
a countdown, lap counting, best-lap times, and text-to-speech announcements.

## Features

- **Live tracking** (HSV + morphology) for multiple vehicles in parallel.
- **Hand-drawn gates**: circle and line detection + MNIST CNN for the gate
  digit; sequence order determined automatically; digit displayed inside the
  empty post circle.
- **Race mode**: Mario Kart countdown (traffic-light overlay + beeps + GO),
  adjustable lap limit via Left/Right arrows, finishing position announced
  at the finish line.
- **Text-to-speech** (espeak-ng): lap announcements `"<color> <lap>"`,
  finishing position + best lap at the finish line, non-blocking in its own
  thread.
- **Trails**: history trail transparent, current lap trail opaque, best-lap
  trail thicker; start/end snapped exactly to the gate crossing point.
- **Overlay/HUD**: info panel top-left (color, HSV, position, speed, times),
  FPS + lap status top-right, help text bottom, HSV color picker under the
  mouse — all togglable with `i`.
- **CSV logging** (`logs/log_<ts>.csv`, columns `t, car, x, y, speed_px_s`)
  plus gate events and lap summary per race.
- **Simulator** as a drop-in camera replacement (paper scene with two colored
  dots on a circular track through three gates) — no hardware needed for
  testing.

## Installation

```bash
pip install -r requirements.txt

# Optional — MNIST CNN for gate digit recognition:
pip install -r requirements-ml.txt
python train_digits.py   # run once; writes models/digits.pt

# Optional — text-to-speech:
sudo apt install espeak-ng
```

## Running

```bash
python main.py
```

On startup a camera list is displayed; press `s` to select the simulator.

## Lighting

Ambient light directly affects the effective camera FPS. In dim conditions
the auto-exposure increases integration time and the real frame rate can drop
to 10–15 fps. 1280×720 @ 30 fps over USB (MJPG) is only achievable with
good lighting. For consistent telemetry use a desk lamp or diffuse overhead
light and avoid hard shadows over the track.

## Controls

### Race workflow

| Key | Action |
|-----|--------|
| `g` | Calibrate gates — detect circles, lines, and digits; saves `configs/gates.json`. |
| `s` | Start countdown (3 beeps + GO, traffic-light overlay). |
| `n` | New race — save log, reset timing and trails; gates are kept. |
| `←` / `→` | Adjust lap limit ±5 (min 5, max 100) — only before race start. |

### Image processing

| Key | Action |
|-----|--------|
| `k` | Toggle CLAHE contrast enhancement. |
| `a` | Open the Adjust window with Brightness / Contrast / Gamma / Saturation sliders and a live histogram. Values are persisted and applied even after the window is closed. |
| `c` | Define a 4-point polygon ROI by clicking corners. Press again to cancel or clear. |
| `h` | Toggle camera mode (Race ↔ Calibration). On startup, available modes are queried via `v4l2-ctl`. Race = lowest resolution with max FPS (≤1280×720); Calibration = highest resolution. Gates and ROI are scaled proportionally; trails are cleared. |

### Display

| Key | Action |
|-----|--------|
| `p` | Pause. |
| `t` | Clear all trails. |
| `f` | Toggle fullscreen. |
| `i` | Toggle HUD (all panels). |
| `q` / `ESC` | Quit and save session. |

### Simulator only

| Key | Action |
|-----|--------|
| `r` | Reverse travel direction. |
| `↑` / `↓` | Increase / decrease simulator speed by ×1.25. |

## File overview

| File | Description |
|------|-------------|
| `main.py` | Main loop, overlay/HUD, key handling, race logic. |
| `vision.py` | Marker tracking (HSV + morphology). |
| `gates.py` | Gate detection — circles, lines, pairing, OCR crop, orientation, classification, crossing check. |
| `digits.py` | MNIST-style CNN for gate digit inference. |
| `train_digits.py` | Training script; writes `models/digits.pt`. |
| `timing.py` | Lap counting, best times, sector splits, gate event log. |
| `geometry.py` | Segment intersection, point–segment distance, Catmull-Rom spline. |
| `sound.py` | Low-latency PCM sounds + espeak-ng TTS, both in worker threads with queues. |
| `sim.py` | Simulator camera (duck-typed `cv2.VideoCapture`). |

## Config files

| File | Description |
|------|-------------|
| `configs/cars.json` | HSV ranges per vehicle. |
| `configs/hud.json` | HUD font, line height, padding, panel alpha. |
| `configs/session.json` | *(auto, gitignored)* Lap limit, ROI polygon, filter values. Written on exit, loaded on startup. |
| `configs/gates.json` | *(auto, gitignored)* Persisted gate calibration. Written after `g`, loaded on startup. |
