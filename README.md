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
- **Text-to-speech** (espeak-ng; Windows PowerShell): enthusiastic English
  lap, finish, and Lewis/Max overtake calls, non-blocking in its own thread.
  Windows keeps the built-in speech engine running between calls and prefers
  an installed English male voice (David/Mark where available), with a lower,
  cinematic pitch; espeak-ng uses its English male voice.
- **Pit call**: pressing `p` pauses and plays a radio-style double tone plus
  “Box, box!”; pressing it again resumes with an English call.
- **Trails**: history trail transparent, current lap trail opaque, best-lap
  trail thicker; start/end snapped exactly to the gate crossing point.
- **Overlay/HUD**: info panel top-left (color, HSV, position, speed, times),
  FPS + lap status top-right, help text bottom, HSV color picker under the
  mouse — togglable with `i`; a Formula 1-style live standings panel on the
  right can be toggled separately with `l`.
- **CSV logging**: positions and speeds (`px/s`, calibrated `m/s` where available),
  plus separate gate-event, sector, completed-lap, and lap-summary files.
- **Speed scale**: live-camera `m/s` is estimated from gate 0's 7.5 cm
  post-center distance. It remains unavailable until gate 0 is classified;
  camera perspective makes the conversion approximate.
- **Simulator** as a drop-in camera replacement (paper scene with two colored
  dots on a circular track through three gates) — no hardware needed for
  testing.

## Installation

```bash
pip install -r requirements.txt

# Optional — MNIST CNN for gate digit recognition:
pip install -r requirements-ml.txt
python train_digits.py   # run once; writes models/digits.pt

# Optional — text-to-speech on Linux (Windows uses built-in PowerShell speech):
sudo apt install espeak-ng
```

Without the optional ML dependencies, gates can still be detected and saved,
but their digits cannot be classified and lap timing by gate order is unavailable.

## Running

```bash
python main.py
```

On startup a camera list is displayed; press `s` to select the simulator.
After the video window opens, click it before using keyboard controls. The
controls are read by the OpenCV window, not by the terminal.

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
| `i` | Toggle the regular HUD panels. |
| `l` | Toggle live standings (ranked by race progress, with the gap to the car ahead). |
| `q` / `ESC` | Quit and save session. |

### Simulator only

| Key | Action |
|-----|--------|
| `r` | Reverse travel direction. |
| `↑` / `↓` | Increase / decrease simulator speed by ×1.25. |

## File overview

| File | Description |
|------|-------------|
| `main.py` | Entry point — initialises subsystems and runs the frame loop. |
| `app.py` | `AppState` (UI flags, gates, capture, mouse) and keyboard dispatcher. |
| `race.py` | `RaceState` (trails, lap times, countdown) and gate-crossing logic. |
| `camera.py` | Camera discovery, mode selection, threaded capture, mode switching. |
| `session.py` | Load/save session config and race-log CSVs. |
| `hud.py` | All drawing helpers — panels, trails, overlays, HSV picker. |
| `vision.py` | HSV marker tracking, frame filters (brightness, gamma, ROI mask). |
| `gates.py` | Gate detection, digit classification, ordering, crossing check, I/O. |
| `perf.py` | Block-level frame-time profiler. |
| `timing.py` | Lap counting, best times, sector splits, gate event log. |
| `geometry.py` | Segment intersection, point–segment distance, Catmull-Rom spline. |
| `sound.py` | Low-latency PCM sounds and espeak-ng TTS in background threads. |
| `sim.py` | Simulator camera — drop-in replacement for a real webcam. |
| `digits.py` | MNIST-style CNN for gate digit inference. |
| `train_digits.py` | Training script; writes `models/digits.pt`. |

## Config files

| File | Description |
|------|-------------|
| `configs/cars.json` | HSV ranges per vehicle. |
| `configs/hud.json` | HUD font, line height, padding, panel alpha. |
| `configs/session.json` | *(auto, gitignored)* Lap limit, ROI polygon, filter values. Written on exit, loaded on startup. |
| `configs/gates.json` | *(auto, gitignored)* Persisted gate calibration. Written after `g`, loaded on startup. |
