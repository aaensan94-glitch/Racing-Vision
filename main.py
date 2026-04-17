import os
import json
import time
import csv
from collections import deque
from typing import Dict, Deque, List, Tuple, Optional

import cv2
import numpy as np

from vision import MultiTracker
from gates import (detect_gates, draw_gates, build_crops_panel,
                   classify_gate_digit, order_gates, gate_crossed,
                   GateCandidate)
from geometry import segment_intersection, catmull_rom
from timing import LapTracker
import sound

Point = Tuple[float, float]


def _trail_gate_xpt(prev_pt: Point, gp: Point, gate,
                    trail_tail: List[Point]) -> Optional[Point]:
    """Schnittpunkt der Fahrbahn (Spline oder gerade) mit der Gate-Linie."""
    a, b = gate.post_a, gate.post_b
    # Spline-Pfad wie in gate_crossed
    if trail_tail is not None and len(trail_tail) >= 3:
        p0 = trail_tail[-3]
        p1 = trail_tail[-2]
        p2 = trail_tail[-1]
        p3 = (2 * gp[0] - prev_pt[0], 2 * gp[1] - prev_pt[1])
        pts = catmull_rom(p0, p1, p2, p3, n=10)
        for i in range(len(pts) - 1):
            xpt = segment_intersection(pts[i], pts[i + 1], a, b)
            if xpt is not None:
                return xpt
    # Fallback: gerade Linie
    return segment_intersection(prev_pt, gp, a, b)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(BASE_DIR, "configs")
CARS_CFG_PATH = os.path.join(CFG_DIR, "cars.json")
HUD_CFG_PATH = os.path.join(CFG_DIR, "hud.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


class HudConfig:
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, path: str):
        with open(path) as f:
            cfg = json.load(f)
        self.scale: float = float(cfg["font_scale"])
        self.thickness: int = int(cfg["font_thickness"])
        self.line_h: int = int(cfg["line_height"])
        self.pad: int = int(cfg["padding"])
        c = cfg["text_color"]
        self.color: Tuple[int, int, int] = (int(c[0]), int(c[1]), int(c[2]))
        self.panel_alpha: float = float(cfg["panel_alpha"])

WINDOW = "Race Vision CV1"

paused = False


def _panel(img, x: int, y: int, w: int, h: int, alpha: float = 0.6) -> None:
    """Dark translucent background so Text über hellem Papier lesbar bleibt."""
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0:y1, x0:x1]
    dark = np.zeros_like(sub)
    cv2.addWeighted(sub, 1 - alpha, dark, alpha, 0, sub)


def draw_polyline(img, pts: List[Point], color=(255, 255, 0), thickness=2,
                  closed=False):
    if pts is None or len(pts) < 2:
        return
    if len(pts) >= 4:
        # Phantom-Punkte spiegeln damit Start/Ende auch Kurven werden
        p_start = (2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1])
        p_end = (2 * pts[-1][0] - pts[-2][0], 2 * pts[-1][1] - pts[-2][1])
        ext = [p_start] + list(pts) + [p_end]
        smooth: List[Point] = []
        for i in range(1, len(ext) - 2):
            smooth.extend(catmull_rom(ext[i - 1], ext[i], ext[i + 1],
                                      ext[i + 2], n=6))
        pts = smooth
    p = np.array([[int(x), int(y)] for x, y in pts],
                 dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [p], isClosed=closed, color=color, thickness=thickness)


CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
DISPLAY_WIDTH = 1280
TRAIL_LEN = 0  # 0 = unbegrenzt, sonst max. Anzahl Punkte pro Spur


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


def prompt_camera_choice():
    """Returns int (camera index) or 'sim' for the simulator."""
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


def open_capture(source):
    if source == "sim":
        from sim import SimCapture
        print("[capture] Simulation mode")
        return SimCapture()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    return cap


def main():
    global paused

    source = prompt_camera_choice()

    hud = HudConfig(HUD_CFG_PATH)
    tracker = MultiTracker.load(CARS_CFG_PATH)
    car_names = tracker.names()

    prev: Dict[str, Optional[Tuple[float, float, float]]] = {
        name: None for name in car_names}
    speed: Dict[str, float] = {name: 0.0 for name in car_names}
    trail_maxlen = TRAIL_LEN if TRAIL_LEN > 0 else None
    trails: Dict[str, Deque[Point]] = {
        name: deque(maxlen=trail_maxlen) for name in car_names}
    lap_trail: Dict[str, List[Point]] = {name: [] for name in car_names}
    best_trail: Dict[str, List[Point]] = {name: [] for name in car_names}


    cap = open_capture(source)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[capture] source={source} resolution={actual_w}x{actual_h}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"log_{ts}.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["t", "car", "x", "y", "speed_px_s"])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if actual_w > 0:
        disp_h = int(actual_h * DISPLAY_WIDTH / actual_w)
        cv2.resizeWindow(WINDOW, DISPLAY_WIDTH, disp_h)
    fullscreen = False

    mouse_px: Dict[str, int] = {"x": -1, "y": -1}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            param["x"] = int(x)
            param["y"] = int(y)

    cv2.setMouseCallback(WINDOW, _on_mouse, mouse_px)

    fps = 0.0
    fps_last_t = time.time()
    fps_frames = 0

    gates: List[GateCandidate] = []
    gate_circles = None
    gate_lines = None
    use_clahe = False
    show_hud = True
    digit_classifier = None
    last_gate_hit: Dict[str, Dict[int, float]] = {
        n: {} for n in car_names}
    last_gate_flash: Dict[str, Tuple[float, int]] = {
        n: (0.0, -1) for n in car_names}
    GATE_DEBOUNCE_S = 0.3
    lap_tracker = LapTracker(car_names)
    RACE_LAPS = 5
    race_active = False
    race_finished: Dict[str, bool] = {n: False for n in car_names}
    countdown_t0: Optional[float] = None
    countdown_beeps = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

        t = time.time()

        # ----- Countdown-Ampel -----
        if countdown_t0 is not None:
            elapsed = t - countdown_t0
            # lit: 0s→1, 1s→2, 2s→3, 3s→GO
            lit = min(int(elapsed) + 1, 4)
            if lit <= 3 and lit > countdown_beeps:
                sound.play_countdown()
                countdown_beeps = lit
                print(f"[countdown] {4 - lit}...")
            if lit >= 4 and countdown_beeps < 4:
                # GO — Rennen starten
                sound.play_go()
                countdown_beeps = 4
                race_active = True
                # Reset wie 'n' aber ohne Log (noch nichts da)
                lap_tracker = LapTracker(car_names,
                                         num_gates=lap_tracker.num_gates)
                for tr in trails.values():
                    tr.clear()
                for name in car_names:
                    lap_trail[name].clear()
                    best_trail[name].clear()

                    prev[name] = None
                    speed[name] = 0.0
                    last_gate_hit[name] = {}
                    last_gate_flash[name] = (0.0, -1)
                    race_finished[name] = False
                countdown_t0 = None
                print(f"[race] GO! {RACE_LAPS} laps")

        positions = tracker.update(frame, t)
        global_positions: Dict[str, Optional[Point]] = dict(positions)

        for name in car_names:
            gp = global_positions[name]

            if gp is not None and prev[name] is not None and gates:
                prev_pt = (prev[name][1], prev[name][2])
                for gi, g in enumerate(gates):
                    trail_tail = list(trails[name])[-4:]
                    direction = gate_crossed(
                        prev_pt, gp, g,
                        trail=trail_tail)
                    if direction != 0:
                        if (t - last_gate_hit[name].get(gi, 0.0)) > GATE_DEBOUNCE_S:
                            last_gate_hit[name][gi] = t
                            last_gate_flash[name] = (t, gi)
                            if direction > 0:
                                ev = lap_tracker.on_forward_crossing(
                                    name, g.digit, t)
                                if ev is not None and ev["lap_time_s"] is not None:
                                    sound.play_triple()
                                else:
                                    sound.play()
                                if ev is not None and ev["lap_time_s"] is not None:
                                    # Schnittpunkt VOR clear berechnen
                                    xpt = _trail_gate_xpt(
                                        prev_pt, gp, g, trail_tail)
                                    # Alten Trail bis zum Gate verlängern
                                    if xpt is not None:
                                        lap_trail[name].append(xpt)
                                    # Ist diese Runde die neue Bestzeit?
                                    if ev["lap_time_s"] <= (
                                            lap_tracker.best_lap_time(name)
                                            or float("inf")):
                                        best_trail[name] = lap_trail[name][:]
                                    lap_trail[name].clear()
                                    if xpt is not None:
                                        lap_trail[name].append(xpt)
                                    lap_nr = lap_tracker.lap(name)
                                    if not sound.tts_busy():
                                        sound.say(f"{name} {lap_nr}")
                                    print(f"\n[lap] {name} lap {ev['lap']} "
                                          f"time={ev['lap_time_s']:.3f}s "
                                          f"(best={lap_tracker.best_lap_time(name):.3f}s)")
                                    tbl = lap_tracker.race_table(name)
                                    if tbl:
                                        print(tbl)
                                        print()
                                    # Ziel erreicht?
                                    if (race_active
                                            and lap_tracker.lap(name) >= RACE_LAPS
                                            and not race_finished[name]):
                                        race_finished[name] = True
                                        place = sum(race_finished.values())
                                        sound.play_finish()
                                        bt = lap_tracker.best_lap_time(name)
                                        bt_s = f"{bt:.1f} seconds" if bt else ""
                                        if place == 1:
                                            sound.say(
                                                f"{name} wins! "
                                                f"Best lap {bt_s}",
                                                priority=True, speed=180)
                                        else:
                                            sound.say(
                                                f"{name} finishes {place}nd. "
                                                f"Best lap {bt_s}",
                                                priority=True, speed=180)
                                        print(f"\n*** {name} FINISHED "
                                              f"place {place} — "
                                              f"{RACE_LAPS} laps! ***")
                                        if all(race_finished.values()):
                                            race_active = False
                                            print("\n=== RACE COMPLETE ===")
                                elif ev is not None and ev["gate"] == 0:
                                    # Gate 0 aber ungültige Runde → trotzdem reset
                                    xpt = _trail_gate_xpt(
                                        prev_pt, gp, g, trail_tail)
                                    if xpt is not None:
                                        lap_trail[name].append(xpt)
                                    lap_trail[name].clear()
                                    if xpt is not None:
                                        lap_trail[name].append(xpt)
                                elif ev is not None and ev["sector_s"] is not None:
                                    delta = lap_tracker.sector_delta(name)
                                    delta_s = f"  [{delta}]" if delta else ""
                                    print(f"[sector] {name} gate {ev['gate']} "
                                          f"sector={ev['sector_s']:.3f}s"
                                          f"{delta_s}")
                            else:
                                sound.play_alarm()
                            did = g.digit if g.digit >= 0 else gi
                            arrow = "fwd" if direction > 0 else "WRONG"
                            print(f"[gate] {name} crossed gate #{did} "
                                  f"{arrow} t={t:.2f}s")
                        break

            if gp is not None and prev[name] is not None:
                dt = t - prev[name][0]
                if dt > 1e-6:
                    dx = gp[0] - prev[name][1]
                    dy = gp[1] - prev[name][2]
                    speed[name] = float((dx * dx + dy * dy) ** 0.5 / dt)
            if gp is not None:
                prev[name] = (t, gp[0], gp[1])

            if gp is not None:
                trails[name].append(gp)
                lap_trail[name].append(gp)

        # ----- Overlay -----
        if use_clahe:
            from gates import _clahe
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
            overlay = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            overlay = frame.copy()

        if gates:
            draw_gates(overlay, gates)
            for name in car_names:
                t_hit, gi = last_gate_flash[name]
                if gi >= 0 and (t - t_hit) < 0.4 and gi < len(gates):
                    g = gates[gi]
                    ax, ay = float(g.post_a[0]), float(g.post_a[1])
                    bx, by = float(g.post_b[0]), float(g.post_b[1])
                    dx, dy = bx - ax, by - ay
                    L = float(np.hypot(dx, dy))
                    if L > g.radius_a + g.radius_b:
                        ux, uy = dx / L, dy / L
                        sx, sy = ax + ux * g.radius_a, ay + uy * g.radius_a
                        ex, ey = bx - ux * g.radius_b, by - uy * g.radius_b
                        cv2.line(overlay, (int(sx), int(sy)),
                                 (int(ex), int(ey)), (0, 255, 0), 5)

        # Bahnen zeichnen: History + Best = 50% transparent, aktuelle Runde opak
        trail_layer = overlay.copy()
        for name in car_names:
            color = tracker.car(name).display_color_bgr()
            if len(trails[name]) >= 2:
                draw_polyline(trail_layer, list(trails[name]),
                              color=color, thickness=1, closed=False)
            if len(best_trail[name]) >= 2:
                draw_polyline(trail_layer, best_trail[name],
                              color=color, thickness=3, closed=False)
        cv2.addWeighted(trail_layer, 0.5, overlay, 0.5, 0, overlay)
        for name in car_names:
            color = tracker.car(name).display_color_bgr()
            if len(lap_trail[name]) >= 2:
                draw_polyline(overlay, lap_trail[name],
                              color=color, thickness=3, closed=False)

        # Per-Car Info als Block: erst sammeln, dann Panel + Text
        info_lines: List[Tuple[str, Tuple[int, int, int]]] = []
        for name in car_names:
            color = tracker.car(name).display_color_bgr()
            gp = global_positions[name]

            if gp is not None:
                gx, gy = int(gp[0]), int(gp[1])
                cnt = tracker.contour(name)
                if cnt is not None:
                    cv2.drawContours(overlay, [cnt], -1, color, 1)
                cv2.circle(overlay, (gx, gy), 6, color, -1)
                cv2.putText(overlay, name, (gx + 10, gy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            cv2.LINE_AA)

            pos_str = f"({int(gp[0])},{int(gp[1])})" if gp is not None else "none"
            hsv = tracker.hsv(name)
            hsv_str = (f"HSV {hsv[0]:.0f} {hsv[1]:.0f} {hsv[2]:.0f}"
                       if hsv is not None else "HSV --")
            info_lines.append(
                (f"{name}  {hsv_str}  {pos_str}  {speed[name]:.0f}px/s",
                 color))

            lap = lap_tracker.lap(name)
            last = lap_tracker.last_lap_time(name)
            best = lap_tracker.best_lap_time(name)
            cur = lap_tracker.current_lap_elapsed(name, t)
            last_s = f"{last:.3f}" if last is not None else "--"
            best_s = f"{best:.3f}" if best is not None else "--"
            cur_s = f"{cur:.2f}" if cur is not None else "--"
            info_lines.append(
                (f"  lap {lap}  cur {cur_s}s  last {last_s}s  best {best_s}s",
                 color))

        if show_hud:
            widths = [cv2.getTextSize(t, hud.FONT, hud.scale, hud.thickness)[0][0]
                      for t, _ in info_lines]
            panel_w = (max(widths) if widths else 0) + 2 * hud.pad
            panel_h = len(info_lines) * hud.line_h + 2 * hud.pad
            _panel(overlay, 10, 10, panel_w, panel_h, hud.panel_alpha)
            y_cursor = 10 + hud.pad + hud.line_h - 10
            for text, col in info_lines:
                cv2.putText(overlay, text, (10 + hud.pad, y_cursor),
                            hud.FONT, hud.scale, col, hud.thickness,
                            cv2.LINE_AA)
                y_cursor += hud.line_h

        # Countdown / Race Overlay
        if countdown_t0 is not None:
            lit = min(int(t - countdown_t0) + 1, 4)  # 1→2→3→4(GO)
            # Ampel quer — 3 Lichter horizontal
            lamp_r = 45
            gap = 16
            w_total = 3 * (2 * lamp_r) + 4 * gap
            h_total = 2 * lamp_r + 2 * gap
            cx = overlay.shape[1] // 2
            cy = overlay.shape[0] // 2
            x0 = cx - w_total // 2
            y0 = cy - h_total // 2
            cv2.rectangle(overlay, (x0, y0), (x0 + w_total, y0 + h_total),
                          (20, 20, 20), -1)
            cv2.rectangle(overlay, (x0, y0), (x0 + w_total, y0 + h_total),
                          (60, 60, 60), 3)
            for i in range(3):
                lx = x0 + gap + lamp_r + i * (2 * lamp_r + gap)
                if lit >= 4:
                    color = (0, 200, 0)  # GO — alle grün
                elif i < lit:
                    color = (0, 0, 255)  # Rot an
                else:
                    color = (30, 30, 30)  # Aus
                cv2.circle(overlay, (lx, cy), lamp_r, color, -1)
                cv2.circle(overlay, (lx, cy), lamp_r, (60, 60, 60), 2)

        fps_frames += 1
        now = time.time()
        if now - fps_last_t >= 0.5:
            fps = fps_frames / (now - fps_last_t)
            fps_frames = 0
            fps_last_t = now

        # Top-Right: FPS + Laps/Race-Status
        if show_hud:
            tr_lines: List[str] = [f"{fps:.1f} fps"]
            if race_active:
                laps_done = max(lap_tracker.lap(n) for n in car_names)
                tr_lines.append(f"Race {laps_done}/{RACE_LAPS}")
            else:
                tr_lines.append(f"Laps {RACE_LAPS}")
            tr_widths = [
                cv2.getTextSize(t, hud.FONT, hud.scale, hud.thickness)[0][0]
                for t in tr_lines]
            tr_w = max(tr_widths) + 2 * hud.pad
            tr_h = len(tr_lines) * hud.line_h + 2 * hud.pad
            tr_x = overlay.shape[1] - tr_w - 10
            tr_y = 10
            _panel(overlay, tr_x, tr_y, tr_w, tr_h, hud.panel_alpha)
            tr_cursor = tr_y + hud.pad + hud.line_h - 10
            for text in tr_lines:
                cv2.putText(overlay, text, (tr_x + hud.pad, tr_cursor),
                            hud.FONT, hud.scale, hud.color, hud.thickness,
                            cv2.LINE_AA)
                tr_cursor += hud.line_h

        # Bottom: Hilfe-Zeilen — gesplittet damit sie ins Fenster passen
        if show_hud:
            help_lines: List[str] = []
            if source == "sim":
                help_lines.append("r=reverse  Up/Down=speed")
            help_lines.append(
                "s=start  n=new-race  Left/Right=laps  p=pause  t=clear-trails")
            help_lines.append(
                "f=fullscreen  g=gates  e=clahe  i=hud  q=quit")
            h_widths = [
                cv2.getTextSize(t, hud.FONT, hud.scale, hud.thickness)[0][0]
                for t in help_lines]
            h_w = max(h_widths) + 2 * hud.pad
            h_h = len(help_lines) * hud.line_h + 2 * hud.pad
            h_x = 10
            h_y = overlay.shape[0] - h_h - 10
            _panel(overlay, h_x, h_y, h_w, h_h, hud.panel_alpha)
            hy = h_y + hud.pad + hud.line_h - 10
            for text in help_lines:
                cv2.putText(overlay, text, (h_x + hud.pad, hy),
                            hud.FONT, hud.scale, hud.color, hud.thickness,
                            cv2.LINE_AA)
                hy += hud.line_h

        # Bottom-Right: HSV-Picker an Mausposition
        mx, my = mouse_px["x"], mouse_px["y"]
        F_H, F_W = frame.shape[:2]
        if show_hud and 0 <= mx < F_W and 0 <= my < F_H:
            bgr = frame[my, mx]
            hsv_px = cv2.cvtColor(
                np.array([[bgr]], dtype=np.uint8),
                cv2.COLOR_BGR2HSV)[0, 0]
            pick_lines = [
                f"({mx},{my})",
                f"BGR {bgr[0]} {bgr[1]} {bgr[2]}",
                f"HSV {hsv_px[0]} {hsv_px[1]} {hsv_px[2]}",
            ]
            p_widths = [
                cv2.getTextSize(t, hud.FONT, hud.scale, hud.thickness)[0][0]
                for t in pick_lines]
            sw = hud.line_h  # Farb-Swatch quadratisch, Schrifthöhe
            p_w = max(p_widths) + sw + 3 * hud.pad
            p_h = len(pick_lines) * hud.line_h + 2 * hud.pad
            p_x = overlay.shape[1] - p_w - 10
            p_y = overlay.shape[0] - p_h - 10
            _panel(overlay, p_x, p_y, p_w, p_h, hud.panel_alpha)
            sx0 = p_x + hud.pad
            sy0 = p_y + hud.pad
            cv2.rectangle(overlay, (sx0, sy0), (sx0 + sw, sy0 + sw),
                          (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)
            cv2.rectangle(overlay, (sx0, sy0), (sx0 + sw, sy0 + sw),
                          (80, 80, 80), 1)
            py_cur = p_y + hud.pad + hud.line_h - 10
            tx = sx0 + sw + hud.pad
            for text in pick_lines:
                cv2.putText(overlay, text, (tx, py_cur),
                            hud.FONT, hud.scale, hud.color, hud.thickness,
                            cv2.LINE_AA)
                py_cur += hud.line_h

        cv2.imshow(WINDOW, overlay)

        for name in car_names:
            gp = global_positions[name]
            if gp is not None:
                writer.writerow([t, name, gp[0], gp[1], speed[name]])

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("n"):
            # Log speichern bevor Reset
            events_df = lap_tracker.to_dataframe()
            if not events_df.empty:
                reset_ts = time.strftime("%Y%m%d_%H%M%S")
                ev_path = os.path.join(LOGS_DIR, f"gates_{reset_ts}.csv")
                events_df.to_csv(ev_path, index=False)
                summary = lap_tracker.summary()
                sum_path = os.path.join(LOGS_DIR, f"laps_{reset_ts}.csv")
                summary.to_csv(sum_path, index=False)
                print(f"[reset] saved {ev_path}")
                print(summary.to_string(index=False))
            # Timing + Trails reset, Gates bleiben
            lap_tracker = LapTracker(car_names, num_gates=lap_tracker.num_gates)
            for tr in trails.values():
                tr.clear()
            for name in car_names:
                lap_trail[name].clear()
                best_trail[name].clear()
                prev[name] = None
                speed[name] = 0.0
                last_gate_hit[name] = {}
                last_gate_flash[name] = (0.0, -1)
            print("[reset] new race — gates kept")
        elif key == ord("t"):
            for tr in trails.values():
                tr.clear()
            print("[cleared] trails")
        elif key == ord("s"):
            if gates:
                countdown_t0 = time.time()
                countdown_beeps = 0
                sound.play_countdown()
                countdown_beeps = 1
                print("[countdown] 3...")
            else:
                print("[start] detect gates first (g)")
        elif key == ord("e"):
            use_clahe = not use_clahe
            print(f"[clahe] {'ON' if use_clahe else 'OFF'}")
        elif key == ord("i"):
            show_hud = not show_hud
            print(f"[hud] {'ON' if show_hud else 'OFF'}")
        elif key == ord("g"):
            gates, gate_circles, gate_lines = detect_gates(
                frame, use_clahe=use_clahe)
            print(f"[gates] circles={len(gate_circles)} "
                  f"lines={len(gate_lines)} gates={len(gates)}")
            if gates:
                if digit_classifier is None:
                    from digits import DigitClassifier
                    try:
                        digit_classifier = DigitClassifier.load(
                            "models/digits.pt")
                        print("[gates] loaded models/digits.pt")
                    except FileNotFoundError:
                        print("[gates] models/digits.pt fehlt — "
                              "train_digits.py ausführen")
                if digit_classifier is not None:
                    for g in gates:
                        classify_gate_digit(digit_classifier, frame, g)
                    ordered, warns = order_gates(gates)
                    print("[gates] order: " + " -> ".join(
                        f"{g.digit}({g.digit_confidence:.2f},{g.digit_side})"
                        for g in ordered))
                    for w in warns:
                        print(f"[gates] WARN {w}")
                    if ordered:
                        gates = ordered
                        n = max(g.digit for g in ordered) + 1
                        lap_tracker.num_gates = n
                        for name in car_names:
                            lap_tracker._state[name].expected = 0
                        print(f"[gates] using {n} ordered gates, "
                              f"timing reset")
                panel = build_crops_panel(frame, gates)
                cv2.imshow("Gate Crops", panel)
        elif key == ord("r"):
            if hasattr(cap, "reverse"):
                cap.reverse()
                print("[sim] reversed direction")
        elif key == 82 and hasattr(cap, "speed_up"):  # arrow up
            cap.speed_up()
            print(f"[sim] faster — period={cap._period:.2f}s")
        elif key == 84 and hasattr(cap, "speed_down"):  # arrow down
            cap.speed_down()
            print(f"[sim] slower — period={cap._period:.2f}s")
        elif key == 83 and not race_active and countdown_t0 is None:  # arrow right
            RACE_LAPS = min(100, RACE_LAPS + 5)
            print(f"[race] laps={RACE_LAPS}")
        elif key == 81 and not race_active and countdown_t0 is None:  # arrow left
            RACE_LAPS = max(5, RACE_LAPS - 5)
            print(f"[race] laps={RACE_LAPS}")
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

    log_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] log saved: {log_path}")

    events_df = lap_tracker.to_dataframe()
    if not events_df.empty:
        events_path = os.path.join(LOGS_DIR, f"gates_{ts}.csv")
        summary_path = os.path.join(LOGS_DIR, f"laps_{ts}.csv")
        events_df.to_csv(events_path, index=False)
        summary = lap_tracker.summary()
        summary.to_csv(summary_path, index=False)
        print(f"[done] gate events: {events_path}")
        print(f"[done] lap summary: {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
