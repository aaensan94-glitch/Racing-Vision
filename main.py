import os
import json
import time
import csv
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from vision import MultiTracker
from timing import LapTimer
from geometry import distance_to_polyline

Point = Tuple[float, float]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(BASE_DIR, "configs")
CARS_CFG_PATH = os.path.join(CFG_DIR, "cars.json")
IDEAL_LINE_PATH = os.path.join(CFG_DIR, "ideal_line.json")
START_LINE_PATH = os.path.join(CFG_DIR, "start_line.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

WINDOW = "Race Vision CV1"

# ----- UI state -----
drawing_line = False
ideal_points: List[Point] = []
start_line_clicks: List[Point] = []
roi = None  # (x,y,w,h)
paused = False


def load_ideal_line() -> List[Point]:
    try:
        with open(IDEAL_LINE_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return [(float(p[0]), float(p[1])) for p in d.get("points", [])]
    except Exception:
        return []


def save_ideal_line(points: List[Point]):
    with open(IDEAL_LINE_PATH, "w", encoding="utf-8") as f:
        json.dump({"points": [[float(x), float(y)] for (x, y) in points]}, f, indent=2)


def load_start_line() -> Tuple[Optional[Point], Optional[Point]]:
    try:
        with open(START_LINE_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        p1 = d.get("p1")
        p2 = d.get("p2")
        if p1 is None or p2 is None:
            return None, None
        return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))
    except Exception:
        return None, None


def save_start_line(a: Point, b: Point):
    with open(START_LINE_PATH, "w", encoding="utf-8") as f:
        json.dump({"p1": [float(a[0]), float(a[1])], "p2": [float(b[0]), float(b[1])]}, f, indent=2)


def apply_roi(frame):
    global roi
    if roi is None:
        return frame, (0, 0)
    x, y, w, h = roi
    x = max(0, x); y = max(0, y)
    return frame[y:y + h, x:x + w].copy(), (x, y)


def on_mouse(event, x, y, flags, param):
    global drawing_line, ideal_points, start_line_clicks
    ox, oy = param.get("offset", (0, 0))
    gx, gy = x + ox, y + oy
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_line:
            ideal_points.append((gx, gy))
        elif param.get("setting_start_line", False):
            start_line_clicks.append((gx, gy))


def draw_polyline(img, pts: List[Point], color=(255, 255, 0), thickness=2, closed=False):
    if pts is None or len(pts) < 2:
        return
    p = np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [p], isClosed=closed, color=color, thickness=thickness)


CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
DISPLAY_WIDTH = 1280


def scan_cameras() -> List[Tuple[int, str]]:
    """Scan /sys/class/video4linux, return (index, name) for capture-capable,
    deduplicated physical devices (lowest index per device name)."""
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
        cap = cv2.VideoCapture(i)
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


def prompt_camera_choice() -> int:
    print("[scan] searching for cameras...")
    cams = scan_cameras()
    if not cams:
        raise RuntimeError("No cameras found.")
    if len(cams) == 1:
        idx, name = cams[0]
        print(f"[scan] one camera found: [{idx}] {name} — using it.")
        return idx
    print("[scan] available cameras:")
    for idx, name in cams:
        print(f"  [{idx}] {name}")
    valid = [idx for idx, _ in cams]
    while True:
        raw = input(f"Select camera index {valid}: ").strip()
        try:
            choice = int(raw)
            if choice in valid:
                return choice
        except ValueError:
            pass
        print("invalid choice, try again.")


def main():
    global drawing_line, ideal_points, start_line_clicks, roi, paused

    cam_index = prompt_camera_choice()

    tracker = MultiTracker.load(CARS_CFG_PATH)
    car_names = tracker.names()

    # Per-car state
    timers: Dict[str, LapTimer] = {name: LapTimer(min_lap_time_s=2.0) for name in car_names}
    prev: Dict[str, Optional[Tuple[float, float, float]]] = {name: None for name in car_names}
    speed: Dict[str, float] = {name: 0.0 for name in car_names}

    a, b = load_start_line()
    for t_ in timers.values():
        if a is not None and b is not None:
            t_.set_start_line(a, b)
        else:
            t_.reset()

    ideal_points = load_ideal_line()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] index={cam_index} resolution={actual_w}x{actual_h}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"log_{ts}.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["t", "car", "x", "y", "speed_px_s", "dist_to_ideal_px", "lap_event_s"])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if actual_w > 0:
        disp_h = int(actual_h * DISPLAY_WIDTH / actual_w)
        cv2.resizeWindow(WINDOW, DISPLAY_WIDTH, disp_h)
    fullscreen = False
    mouse_state = {"offset": (0, 0), "setting_start_line": False}
    cv2.setMouseCallback(WINDOW, on_mouse, mouse_state)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

        t = time.time()

        view, offset = apply_roi(frame)
        mouse_state["offset"] = offset

        positions = tracker.update(view, t)
        global_positions: Dict[str, Optional[Point]] = {}
        for name, pos in positions.items():
            global_positions[name] = (pos[0] + offset[0], pos[1] + offset[1]) if pos is not None else None

        # Per-car update: lap timing, speed, distance-to-ideal
        dists: Dict[str, Optional[float]] = {}
        lap_events: Dict[str, Optional[float]] = {}
        for name in car_names:
            gp = global_positions[name]
            lap_events[name] = timers[name].update(gp, t) if gp is not None else None

            if gp is not None and prev[name] is not None:
                dt = t - prev[name][0]
                if dt > 1e-6:
                    dx = gp[0] - prev[name][1]
                    dy = gp[1] - prev[name][2]
                    speed[name] = float((dx * dx + dy * dy) ** 0.5 / dt)
            if gp is not None:
                prev[name] = (t, gp[0], gp[1])

            dists[name] = distance_to_polyline(gp, ideal_points) if (gp is not None and len(ideal_points) >= 2) else None

        # ----- Overlay -----
        overlay = frame.copy()

        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (200, 200, 200), 2)

        # start line (shared) — take from first timer
        any_timer = next(iter(timers.values()))
        if any_timer.start_line_a is not None and any_timer.start_line_b is not None:
            ax, ay = int(any_timer.start_line_a[0]), int(any_timer.start_line_a[1])
            bx, by = int(any_timer.start_line_b[0]), int(any_timer.start_line_b[1])
            cv2.line(overlay, (ax, ay), (bx, by), (0, 0, 255), 3)
            cv2.putText(overlay, "START", (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        draw_polyline(overlay, ideal_points, color=(255, 255, 0), thickness=2, closed=False)

        # Per-car HUD block
        y_cursor = 30
        for name in car_names:
            color = tracker.car(name).display_color_bgr()
            gp = global_positions[name]

            if gp is not None:
                gx, gy = int(gp[0]), int(gp[1])
                cnt = tracker.contour(name)
                if cnt is not None:
                    cnt_shifted = cnt + np.array([[offset[0], offset[1]]], dtype=cnt.dtype)
                    cv2.drawContours(overlay, [cnt_shifted], -1, color, 1)
                cv2.circle(overlay, (gx, gy), 6, color, -1)
                cv2.putText(overlay, name, (gx + 10, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            lap_time = timers[name].current_lap_time(t)
            dist = dists[name]
            dist_str = f"{dist:.1f}px" if dist is not None else "NA"
            pos_str = f"({int(gp[0])},{int(gp[1])})" if gp is not None else "none"
            hsv = tracker.hsv(name)
            hsv_str = f"H={hsv[0]:.0f} S={hsv[1]:.0f} V={hsv[2]:.0f}" if hsv is not None else "HSV=NA"
            line = f"{name}[{hsv_str}]: pos={pos_str} speed={speed[name]:.0f}px/s lap={lap_time:.2f}s dist={dist_str} laps={len(timers[name].laps)}"
            cv2.putText(overlay, line, (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_cursor += 28

        mode = "LINE-DRAW" if drawing_line else "NORMAL"
        cv2.putText(overlay, f"mode={mode}  (l=line, b=startline, r=roi, p=pause, n=reset-laps, h=hsv, f=fullscreen)",
                    (20, overlay.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(WINDOW, overlay)

        # CSV: one row per car per frame when seen
        for name in car_names:
            gp = global_positions[name]
            if gp is not None:
                writer.writerow([
                    t, name, gp[0], gp[1], speed[name],
                    dists[name] if dists[name] is not None else "",
                    lap_events[name] if lap_events[name] is not None else "",
                ])

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("l"):
            drawing_line = not drawing_line
            mouse_state["setting_start_line"] = False
        elif key == ord("s"):
            save_ideal_line(ideal_points)
            print(f"[saved] ideal line with {len(ideal_points)} points -> {IDEAL_LINE_PATH}")
        elif key == ord("c"):
            ideal_points = load_ideal_line()
            print(f"[loaded] ideal line with {len(ideal_points)} points")
        elif key == ord("x"):
            ideal_points = []
            print("[cleared] ideal line (RAM only)")
        elif key == ord("b"):
            start_line_clicks = []
            mouse_state["setting_start_line"] = True
            drawing_line = False
            print("[startline] click 2 points in the window...")
            while True:
                cv2.imshow(WINDOW, overlay)
                k2 = cv2.waitKey(10) & 0xFF
                if k2 in (27, ord("q")):
                    break
                if len(start_line_clicks) >= 2:
                    a, b = start_line_clicks[0], start_line_clicks[1]
                    for t_ in timers.values():
                        t_.set_start_line(a, b)
                    save_start_line(a, b)
                    print(f"[saved] start line: {a} -> {b}")
                    break
            mouse_state["setting_start_line"] = False
        elif key == ord("n"):
            for t_ in timers.values():
                t_.reset()
            print("[reset] lap timers")
        elif key == ord("r"):
            paused = True
            r = cv2.selectROI(WINDOW, overlay, fromCenter=False, showCrosshair=True)
            paused = False
            x, y, w, h = r
            if w > 0 and h > 0:
                roi = (int(x), int(y), int(w), int(h))
                print(f"[roi] set to {roi}")
            else:
                roi = None
                print("[roi] cleared")
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        elif key == ord("h"):
            for name in car_names:
                gp = global_positions[name]
                if gp is None:
                    continue
                gx, gy = int(gp[0]), int(gp[1])
                x0, y0 = max(0, gx - 10), max(0, gy - 10)
                x1, y1 = min(frame.shape[1], gx + 10), min(frame.shape[0], gy + 10)
                patch = frame[y0:y1, x0:x1]
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mean = hsv.reshape(-1, 3).mean(axis=0)
                print(f"[HSV around {name}] H={mean[0]:.1f}, S={mean[1]:.1f}, V={mean[2]:.1f}")

    log_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] log saved: {log_path}")


if __name__ == "__main__":
    main()
