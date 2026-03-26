import os
import json
import time
import csv
from typing import List, Tuple, Optional

import cv2
import numpy as np

from vision import MarkerConfig, MarkerTracker
from timing import LapTimer
from geometry import distance_to_polyline

Point = Tuple[float, float]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(BASE_DIR, "configs")
MARKER_CFG_PATH = os.path.join(CFG_DIR, "marker_hsv.json")
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
        pts = d.get("points", [])
        return [(float(p[0]), float(p[1])) for p in pts]
    except Exception:
        return []

def save_ideal_line(points: List[Point]):
    with open(IDEAL_LINE_PATH, "w", encoding="utf-8") as f:
        json.dump({"points": [[float(x), float(y)] for (x,y) in points]}, f, indent=2)

def load_start_line() -> Tuple[Optional[Point], Optional[Point]]:
    try:
        with open(START_LINE_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        p1 = d.get("p1", None)
        p2 = d.get("p2", None)
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
    x,y,w,h = roi
    x = max(0, x); y = max(0, y)
    return frame[y:y+h, x:x+w].copy(), (x, y)

def on_mouse(event, x, y, flags, param):
    global drawing_line, ideal_points, start_line_clicks, roi
    # x,y are in the shown (possibly ROI-cropped) coordinates, we store in ROI coordinates
    offset = param.get("offset", (0,0))
    ox, oy = offset
    gx, gy = x + ox, y + oy

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_line:
            ideal_points.append((gx, gy))
        elif param.get("setting_start_line", False):
            start_line_clicks.append((gx, gy))

def draw_polyline(img, pts: List[Point], color=(255,255,0), thickness=2, closed=False):
    if pts is None or len(pts) < 2:
        return
    p = np.array([[int(x), int(y)] for x,y in pts], dtype=np.int32).reshape((-1,1,2))
    cv2.polylines(img, [p], isClosed=closed, color=color, thickness=thickness)

def main():
    global drawing_line, ideal_points, start_line_clicks, roi, paused

    cfg = MarkerConfig.load(MARKER_CFG_PATH)
    tracker = MarkerTracker(cfg)

    timer = LapTimer(min_lap_time_s=2.0)
    a, b = load_start_line()
    if a is not None and b is not None:
        timer.set_start_line(a, b)
    else:
        timer.reset()

    ideal_points = load_ideal_line()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # logfile
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"log_{ts}.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["t", "x", "y", "speed_px_s", "dist_to_ideal_px", "lap_event_s"])

    cv2.namedWindow(WINDOW)
    mouse_state = {"offset": (0,0), "setting_start_line": False}
    cv2.setMouseCallback(WINDOW, on_mouse, mouse_state)

    prev = None  # (t, x, y)
    speed = 0.0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # comment out if mirrored view is confusing

        t = time.time()

        view, offset = apply_roi(frame)
        mouse_state["offset"] = offset

        pos, mask = tracker.update(view, t)
        # pos is in ROI coordinates; convert to global coordinates
        global_pos = None
        if pos is not None:
            global_pos = (pos[0] + offset[0], pos[1] + offset[1])

        lap_event = None
        if global_pos is not None:
            lap_event = timer.update(global_pos, t)

        # speed from global positions
        if global_pos is not None and prev is not None:
            dt = t - prev[0]
            if dt > 1e-6:
                dx = global_pos[0] - prev[1]
                dy = global_pos[1] - prev[2]
                speed = float((dx*dx + dy*dy) ** 0.5 / dt)
        if global_pos is not None:
            prev = (t, global_pos[0], global_pos[1])

        # distance to ideal line (px)
        dist = None
        if global_pos is not None and len(ideal_points) >= 2:
            dist = distance_to_polyline(global_pos, ideal_points)

        # --- Overlay ---
        overlay = frame.copy()

        # ROI rectangle
        if roi is not None:
            x,y,w,h = roi
            cv2.rectangle(overlay, (x,y), (x+w, y+h), (200,200,200), 2)

        # start line
        if timer.start_line_a is not None and timer.start_line_b is not None:
            ax, ay = int(timer.start_line_a[0]), int(timer.start_line_a[1])
            bx, by = int(timer.start_line_b[0]), int(timer.start_line_b[1])
            cv2.line(overlay, (ax,ay), (bx,by), (0,0,255), 3)
            cv2.putText(overlay, "START", (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ideal line
        draw_polyline(overlay, ideal_points, color=(255,255,0), thickness=2, closed=False)

        # position marker
        if global_pos is not None:
            gx, gy = int(global_pos[0]), int(global_pos[1])
            cv2.circle(overlay, (gx,gy), 6, (0,255,0), -1)
            cv2.putText(overlay, f"pos=({gx},{gy})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.putText(overlay, "pos=(none)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(overlay, f"speed={speed:.1f} px/s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        lap_time = timer.current_lap_time(t)
        cv2.putText(overlay, f"lap_time={lap_time:.2f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if dist is not None:
            cv2.putText(overlay, f"dist_to_ideal={dist:.1f}px", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.putText(overlay, "dist_to_ideal=NA", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Lap list (last 5)
        if len(timer.laps) > 0:
            y0 = 160
            cv2.putText(overlay, "laps:", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            for i, lt in enumerate(timer.laps[-5:]):
                cv2.putText(overlay, f"{len(timer.laps)-len(timer.laps[-5:])+i+1}: {lt:.2f}s",
                            (20, y0 + 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Mode hints
        mode = "LINE-DRAW" if drawing_line else "NORMAL"
        cv2.putText(overlay, f"mode={mode}  (l=toggle line, b=set startline, r=roi, p=pause)", (20, overlay.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Show
        cv2.imshow(WINDOW, overlay)

        # write log row
        if global_pos is not None:
            writer.writerow([t, global_pos[0], global_pos[1], speed, dist if dist is not None else "", lap_event if lap_event is not None else ""])

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
            # startline mode: collect 2 clicks
            start_line_clicks = []
            mouse_state["setting_start_line"] = True
            drawing_line = False
            print("[startline] click 2 points in the window...")
            # wait until 2 clicks are collected (but keep UI responsive)
            while True:
                cv2.imshow(WINDOW, overlay)
                k2 = cv2.waitKey(10) & 0xFF
                if k2 in (27, ord("q")):
                    break
                if len(start_line_clicks) >= 2:
                    a, b = start_line_clicks[0], start_line_clicks[1]
                    timer.set_start_line(a, b)
                    save_start_line(a, b)
                    print(f"[saved] start line: {a} -> {b}")
                    break
            mouse_state["setting_start_line"] = False
        elif key == ord("n"):
            timer.reset()
            print("[reset] lap timer")
        elif key == ord("r"):
            paused = True
            # select ROI on current frame
            r = cv2.selectROI(WINDOW, overlay, fromCenter=False, showCrosshair=True)
            paused = False
            x,y,w,h = r
            if w > 0 and h > 0:
                roi = (int(x), int(y), int(w), int(h))
                print(f"[roi] set to {roi}")
            else:
                roi = None
                print("[roi] cleared")
        elif key == ord("h"):
            # print rough HSV stats from center ROI for debugging
            if global_pos is not None:
                gx, gy = int(global_pos[0]), int(global_pos[1])
                x0, y0 = max(0, gx-10), max(0, gy-10)
                x1, y1 = min(frame.shape[1], gx+10), min(frame.shape[0], gy+10)
                patch = frame[y0:y1, x0:x1]
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mean = hsv.reshape(-1,3).mean(axis=0)
                print(f"[HSV mean around marker] H={mean[0]:.1f}, S={mean[1]:.1f}, V={mean[2]:.1f}")

    log_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] log saved: {log_path}")

if __name__ == "__main__":
    main()
