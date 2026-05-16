import os
import re
import subprocess
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gates import GateCandidate, scale_gates_and_roi

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30

CamMode = Tuple[str, int, int, float]  # (fourcc, width, height, fps)


class ThreadedCapture:
    """VideoCapture wrapper that reads frames in a background thread.

    Frames are captured as fast as the camera delivers them. read() returns
    immediately with the most recent frame (possibly the same as the previous
    call). frame_id() increments whenever a new frame arrives — the main loop
    can skip vision/gate processing when the image has not changed.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ok = False
        self._frame_id = 0
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                if ok and frame is not None:
                    self._frame = frame
                    self._frame_id += 1
            if not ok:
                time.sleep(0.02)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame

    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, val):
        return self._cap.set(prop, val)

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


def scan_cameras() -> List[Tuple[int, str]]:
    """Scans /sys/class/video4linux and returns capture-capable physical cameras.

    Deduplicates by device name, keeping the lowest index per device.

    Returns:
        List of (index, name) tuples for cameras that can deliver a frame.
    """
    sys_root = "/sys/class/video4linux"
    if not os.path.isdir(sys_root):
        return []
    indices = sorted(
        int(n[len("video"):]) for n in os.listdir(sys_root) if n.startswith("video")
    )
    seen_names = set()
    found: List[Tuple[int, str]] = []
    for i in indices:
        try:
            with open(os.path.join(sys_root, f"video{i}", "name")) as f:
                name = f.read().strip()
        except OSError:
            continue
        if name in seen_names:
            continue
        cap = cv2.VideoCapture(i)   # open camera to test if it delivers frames
        ok_open = cap.isOpened()
        ok_frame = False
        if ok_open:
            ok_frame, frame = cap.read()
            if not ok_frame or frame is None or frame.size == 0:
                ok_frame = False
        cap.release()
        if not ok_frame:
            continue
        seen_names.add(name)
        found.append((i, name))
    return found


def prompt_camera_choice():
    """Prompts the user to select a camera or the simulator.

    Returns:
        Camera index (int) or the string "sim".
    """
    print("[scan] searching for cameras...")
    cams = scan_cameras()
    print("[scan] available sources:")
    for idx, name in cams:
        print(f"  [{idx}] {name}")
    print("  [s] Simulation")
    valid = [str(idx) for idx, _ in cams] + ["s"]
    while True:
        raw = input(f"Select source {valid}: ").strip().lower()
        if raw == "s":
            return "sim"
        try:
            choice = int(raw)
            if choice in [c[0] for c in cams]:
                return choice
        except ValueError:
            pass
        print("invalid choice, try again.")


def list_v4l2_modes(device_index: int) -> List[CamMode]:
    """Parses v4l2-ctl --list-formats-ext output into a list of camera modes.

    Args:
        device_index: /dev/videoN index.

    Returns:
        List of (fourcc, width, height, fps) tuples, or empty if v4l2-ctl is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-formats-ext",
             "-d", f"/dev/video{device_index}"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return []
    modes: List[CamMode] = []
    cur_fmt = ""
    cur_w = cur_h = 0
    for raw in out.splitlines():
        line = raw.strip()
        m = re.match(r"\[\d+\]:\s*'(\w+)'", line)
        if m:
            cur_fmt = m.group(1)
            continue
        m = re.match(r"Size:\s*Discrete\s+(\d+)x(\d+)", line)
        if m:
            cur_w, cur_h = int(m.group(1)), int(m.group(2))
            continue
        m = re.match(r"Interval:\s*Discrete\s+[\d.]+s\s+\(([\d.]+)\s*fps\)",
                     line)
        if m and cur_fmt and cur_w:
            modes.append((cur_fmt, cur_w, cur_h, float(m.group(1))))
    return modes


def pick_default_modes(
    modes: List[CamMode],
) -> Tuple[Optional[CamMode], Optional[CamMode]]:
    """Selects race and calibration modes from the available camera modes.

    Race: highest FPS at ≤1280×720, then largest resolution as tiebreaker.
    Calibration: largest resolution, then highest FPS as tiebreaker.
    MJPG is preferred over other formats.

    Args:
        modes: List of modes from list_v4l2_modes.

    Returns:
        Tuple (race_mode, calibration_mode), either may be None.
    """
    mjpg = [m for m in modes if m[0] == "MJPG"] or modes
    if not mjpg:
        return None, None
    race_pool = [m for m in mjpg if m[1] <= 1280 and m[2] <= 720] or mjpg
    race = max(race_pool, key=lambda m: (m[3], m[1] * m[2]))
    cal = max(mjpg, key=lambda m: (m[1] * m[2], m[3]))
    return race, cal


def open_capture(source, mode: Optional[CamMode] = None):
    """Opens a camera or simulation capture and returns a ready ThreadedCapture.

    For a real camera the codec is set to MJPG before resolution to avoid the
    slow YUYV fallback on USB cameras.  Blocks up to 2 s waiting for the first
    valid frame so the main loop does not see a stale "not ok" status on startup.

    Args:
        source: Integer camera index, or the string ``"sim"`` for simulation mode.
        mode: Optional (fourcc, width, height, fps) tuple; defaults to the
            module-level CAPTURE_* constants.

    Returns:
        A started ThreadedCapture instance (or SimCapture for simulation).

    Raises:
        RuntimeError: If the camera device cannot be opened.
    """
    if source == "sim":
        from sim import SimCapture
        print("[capture] Simulation mode")
        return SimCapture()
    cap = cv2.VideoCapture(source)  # open camera device
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {source}")
    fourcc, w, h, fps = mode or ("MJPG", CAPTURE_WIDTH, CAPTURE_HEIGHT,
                                 float(CAPTURE_FPS))
    # Set MJPG before resolution; otherwise USB defaults to YUYV which at
    # 1280×720 gives ~126 ms/frame instead of ~33 ms.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))  # pixel format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize capture latency
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_str = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4))
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[capture] fourcc={fcc_str} {aw}x{ah} "
          f"fps={cap.get(cv2.CAP_PROP_FPS):.0f}")
    wrapper = ThreadedCapture(cap)
    # Wait for the first valid frame so the main loop does not exit immediately
    # on a stale "not ok" status from the freshly opened capture.
    t_wait0 = time.time()
    while time.time() - t_wait0 < 2.0:
        ok, _ = wrapper.read()
        if ok:
            break
        time.sleep(0.02)
    return wrapper


def setup_camera(source):
    """Queries camera modes, opens the capture, and returns all setup values.

    Starts in calibration mode (highest resolution). The caller can toggle to
    race mode later with switch_camera_mode.

    Args:
        source: Camera index (int) or "sim".

    Returns:
        Tuple (cap, race_mode, cal_mode, current_mode, actual_w, actual_h).
    """
    race_mode: Optional[CamMode] = None
    cal_mode: Optional[CamMode] = None
    if isinstance(source, int):
        available = list_v4l2_modes(source)
        if available:
            race_mode, cal_mode = pick_default_modes(available)
            if race_mode:
                print(f"[modes] race={race_mode[1]}x{race_mode[2]}@"
                      f"{race_mode[3]:.0f}  cal={cal_mode[1]}x{cal_mode[2]}@"
                      f"{cal_mode[3]:.0f} (toggle: h)")
        else:
            print("[modes] v4l2-ctl unavailable or no modes found — "
                  "using default 1280x720@30")
    current_mode: Optional[CamMode] = cal_mode or race_mode
    cap = open_capture(source, current_mode)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[capture] source={source} resolution={actual_w}x{actual_h}")
    return cap, race_mode, cal_mode, current_mode, actual_w, actual_h


def switch_camera_mode(source, cap, current_mode: CamMode,
                       race_mode: CamMode, cal_mode: CamMode,
                       gates: List[GateCandidate], mouse_state: dict,
                       trails: dict, car_names: List[str],
                       lap_trail: dict, best_trail: dict,
                       prev: dict):
    """Switches between race and calibration modes and rescales dependent state.

    Releases the current capture, opens a new one at the target mode, and
    rescales gates, ROI, and trails when the resolution changes.

    Args:
        source: Camera index passed to open_capture.
        cap: Active capture to release.
        current_mode: Currently active CamMode.
        race_mode: Race CamMode (lower resolution, higher FPS).
        cal_mode: Calibration CamMode (higher resolution).
        gates: Gate list rescaled in place on resolution change.
        mouse_state: Dict containing roi_pts, updated in place.
        trails: Per-car history deques, cleared on resolution change.
        car_names: List of car names.
        lap_trail: Per-car current-lap lists, cleared on resolution change.
        best_trail: Per-car best-lap lists, cleared on resolution change.
        prev: Per-car previous position, reset on resolution change.

    Returns:
        Tuple (cap, new_mode, last_frame_id) where last_frame_id is reset to -1.
    """
    new_mode = cal_mode if current_mode == race_mode else race_mode
    old_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    old_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    cap = open_capture(source, new_mode)
    new_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if old_w > 0 and old_h > 0 and (old_w, old_h) != (new_w, new_h):
        sx, sy = new_w / old_w, new_h / old_h
        mouse_state["roi_pts"] = scale_gates_and_roi(
            gates, mouse_state["roi_pts"], sx, sy)
        for tr in trails.values():
            tr.clear()
        for name in car_names:
            lap_trail[name].clear()
            best_trail[name].clear()
            prev[name] = None
    label = "cal" if new_mode == cal_mode else "race"
    print(f"[mode] {label} — {new_w}x{new_h}@{new_mode[3]:.0f}")
    return cap, new_mode, -1
