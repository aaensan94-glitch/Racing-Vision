import os
import csv
import time
from collections import deque
from typing import Dict, Deque, List, Tuple, Optional

import cv2
import numpy as np

from camera import (CamMode, ThreadedCapture, open_capture, list_v4l2_modes,
                    pick_default_modes, prompt_camera_choice)
from session import (LOGS_DIR, HUD_CFG_PATH, CARS_CFG_PATH,
                     _load_session, _save_session, _load_gates, _save_gates,
                     _gates_path, _session_path)
from hud import HudConfig, _make_histogram, _panel, draw_polyline
from perf import Perf
from vision import MultiTracker
from gates import (detect_gates, draw_gates, build_crops_panel,
                   classify_gate_digit, order_gates, gate_crossed,
                   GateCandidate)
from geometry import segment_intersection, catmull_rom
from timing import LapTracker
import sound

Point = Tuple[float, float]

DISPLAY_WIDTH = 1280
TRAIL_LEN = 0  # 0 = unlimited, otherwise max points per trail
WINDOW = "Race Vision CV1"

paused = False


def _trail_gate_xpt(prev_pt: Point, gp: Point, gate,
                    trail_tail: List[Point]) -> Optional[Point]:
    """Returns the intersection of the car's path (spline or straight) with the gate line.

    Args:
        prev_pt: Previous car position.
        gp: Current car position.
        gate: Gate candidate to intersect with.
        trail_tail: Recent trail points used to build the spline.

    Returns:
        Intersection point, or None if there is no intersection.
    """
    a, b = gate.post_a, gate.post_b
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
    # fallback: straight line
    return segment_intersection(prev_pt, gp, a, b)


def _scale_gates_and_roi(gates: List[GateCandidate],
                         roi_pts: List[Tuple[int, int]],
                         sx: float, sy: float) -> List[Tuple[int, int]]:
    """Scales gates in place and returns rescaled ROI points for a mode switch.

    Args:
        gates: Gate candidates to rescale in place.
        roi_pts: ROI polygon points to rescale (not mutated; new list returned).
        sx: Horizontal scale factor (new_width / old_width).
        sy: Vertical scale factor (new_height / old_height).

    Returns:
        Rescaled ROI point list.
    """
    for g in gates:
        g.post_a = (g.post_a[0] * sx, g.post_a[1] * sy)
        g.post_b = (g.post_b[0] * sx, g.post_b[1] * sy)
        g.line_p1 = (g.line_p1[0] * sx, g.line_p1[1] * sy)
        g.line_p2 = (g.line_p2[0] * sx, g.line_p2[1] * sy)
        g.radius_a = g.radius_a * (sx + sy) * 0.5
        g.radius_b = g.radius_b * (sx + sy) * 0.5
    return [(int(p[0] * sx), int(p[1] * sy)) for p in roi_pts]


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

    # Query available camera modes and auto-select race and calibration modes.
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
    # Start in calibration mode (highest resolution) — calibrate gates, then
    # switch to race mode with 'h'.
    current_mode: Optional[CamMode] = cal_mode or race_mode
    cap = open_capture(source, current_mode)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[capture] source={source} resolution={actual_w}x{actual_h}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"log_{ts}.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["t", "car", "x", "y", "speed_px_s"])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizable display window
    if actual_w > 0:
        disp_h = int(actual_h * DISPLAY_WIDTH / actual_w)
        cv2.resizeWindow(WINDOW, DISPLAY_WIDTH, disp_h)  # set initial window size
    fullscreen = True

    session = _load_session(source)
    saved_roi = session.get("roi_pts", [])
    saved_sess_res = session.get("resolution")
    roi_pts_init: List[Tuple[int, int]] = [
        (int(p[0]), int(p[1])) for p in saved_roi
        if isinstance(p, (list, tuple)) and len(p) == 2
    ] if len(saved_roi) == 4 else []
    # Rescale the saved ROI to the current resolution if the mode was different
    # when the session was saved (e.g. race vs calibration mode across sessions).
    if (roi_pts_init and saved_sess_res and len(saved_sess_res) == 2
            and actual_w > 0 and actual_h > 0):
        osw, osh = int(saved_sess_res[0]), int(saved_sess_res[1])
        if (osw, osh) != (actual_w, actual_h) and osw > 0 and osh > 0:
            sx, sy = actual_w / osw, actual_h / osh
            roi_pts_init = [(int(p[0] * sx), int(p[1] * sy))
                            for p in roi_pts_init]
            print(f"[session] roi rescaled {osw}x{osh} -> "
                  f"{actual_w}x{actual_h}")

    mouse_state: Dict = {
        "x": -1, "y": -1,
        "last_move_t": 0.0,
        "roi_pts": roi_pts_init,
        "roi_editing": False,
    }

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            param["x"] = int(x)
            param["y"] = int(y)
            param["last_move_t"] = time.time()
        elif event == cv2.EVENT_LBUTTONDOWN and param["roi_editing"]:
            pts = param["roi_pts"]
            if len(pts) < 4:
                pts.append((int(x), int(y)))
                print(f"[roi] point {len(pts)}/4: ({x},{y})")
                if len(pts) == 4:
                    param["roi_editing"] = False
                    print("[roi] 4 points set — mask active")

    cv2.setMouseCallback(WINDOW, _on_mouse, mouse_state)  # register mouse handler for ROI editing

    fps = 0.0
    fps_last_t = time.time()
    fps_frames = 0

    gates, saved_gates_res = _load_gates(source)
    if gates:
        print(f"[session] loaded {len(gates)} gates from {_gates_path(source)}")
        if (saved_gates_res and actual_w > 0 and actual_h > 0
                and saved_gates_res != (actual_w, actual_h)):
            sx = actual_w / saved_gates_res[0]
            sy = actual_h / saved_gates_res[1]
            _scale_gates_and_roi(gates, [], sx, sy)
            print(f"[session] gates rescaled "
                  f"{saved_gates_res[0]}x{saved_gates_res[1]} -> "
                  f"{actual_w}x{actual_h}")
    gate_circles = None
    gate_lines = None
    use_clahe = False
    show_hud = True
    adjust_open = False
    ADJUST_WIN = "Image Adjust"
    ADJUST_SLIDERS = [
        ("Brightness", 100, 200),
        ("Contrast", 100, 300),
        ("Gamma", 100, 300),
        ("Saturation", 100, 300),
    ]
    saved_adjust = session.get("adjust", {})
    adjust_vals: Dict[str, int] = {
        name: int(saved_adjust.get(name, default))
        for name, default, _ in ADJUST_SLIDERS
    }
    digit_classifier = None
    last_gate_hit: Dict[str, Dict[int, float]] = {
        n: {} for n in car_names}
    last_gate_flash: Dict[str, Tuple[float, int]] = {
        n: (0.0, -1) for n in car_names}
    GATE_DEBOUNCE_S = 0.3
    lap_tracker = LapTracker(car_names)
    # Per car: True after the first gate-0 crossing — distinguishes the silent
    # initial start crossing from a later messed-up reset announcement.
    armed_before: Dict[str, bool] = {n: False for n in car_names}
    if gates:
        valid_digits = [g.digit for g in gates if g.digit >= 0]
        if valid_digits:
            lap_tracker.num_gates = max(valid_digits) + 1
    RACE_LAPS = int(session.get("lap_limit", 5))
    RACE_LAPS = max(5, min(100, RACE_LAPS))
    race_active = False
    race_finished: Dict[str, bool] = {n: False for n in car_names}
    countdown_t0: Optional[float] = None
    countdown_beeps = 0
    perf = Perf()
    last_frame_id = -1

    while True:
        pending_key = 255  # 255 = no key pressed
        with perf.timed("capture"):
            if not paused:
                # Wait for a new frame from ThreadedCapture. Without this we
                # would loop faster than the camera, render duplicate frames,
                # and flicker in the Adjust window between filtered and
                # unfiltered images. Buffer any key pressed while waiting so
                # the main handler below can process it.
                if hasattr(cap, "frame_id"):
                    while True:
                        fid = cap.frame_id()
                        if fid != last_frame_id:
                            last_frame_id = fid
                            break
                        k = cv2.waitKey(1) & 0xFF
                        if k != 255:
                            pending_key = k
                ok, frame = cap.read()
                if not ok:
                    break

        t = time.time()

        # ----- Countdown lights -----
        if countdown_t0 is not None:
            elapsed = t - countdown_t0
            # lit: 0s→1, 1s→2, 2s→3, 3s→GO
            lit = min(int(elapsed) + 1, 4)
            if lit <= 3 and lit > countdown_beeps:
                sound.play_countdown()
                countdown_beeps = lit
                print(f"[countdown] {4 - lit}...")
            if lit >= 4 and countdown_beeps < 4:
                # GO — reset like 'n' but skip the log (no data yet)
                sound.play_go()
                countdown_beeps = 4
                race_active = True
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
                    armed_before[name] = False
                countdown_t0 = None
                print(f"[race] GO! {RACE_LAPS} laps")

        with perf.timed("filter"):
            if adjust_open:
                for name, _d, _m in ADJUST_SLIDERS:
                    adjust_vals[name] = cv2.getTrackbarPos(name, ADJUST_WIN)  # read slider value
            b = adjust_vals["Brightness"]
            c_ = adjust_vals["Contrast"]
            gm = adjust_vals["Gamma"]
            sa = adjust_vals["Saturation"]
            if (b, c_, gm, sa) != (100, 100, 100, 100):
                alpha = c_ / 100.0
                beta = float(b - 100)
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)  # brightness/contrast
                gamma = max(0.1, gm / 100.0)
                lut = np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255,
                              0, 255).astype(np.uint8)
                frame = cv2.LUT(frame, lut)  # apply gamma correction via lookup table
                if sa != 100:
                    hsv_img = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2HSV).astype(np.int32)  # convert for saturation edit
                    hsv_img[..., 1] = np.clip(
                        hsv_img[..., 1] * sa / 100, 0, 255)
                    frame = cv2.cvtColor(
                        hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)  # convert back to BGR
        if adjust_open:
            hist_canvas = _make_histogram(frame, w=520)
            header = np.full((150, 520, 3), 30, dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(header, "Slider  (100 = neutral)", (14, 26),
                        font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            labels = [("Brightness", b), ("Contrast", c_),
                      ("Gamma", gm), ("Saturation", sa)]
            y = 56
            for lbl, val in labels:
                col = (230, 230, 230) if val == 100 else (80, 200, 255)
                cv2.putText(header, f"{lbl:<11s} {val:>3d}", (18, y),
                            font, 0.7, col, 2, cv2.LINE_AA)
                y += 24
            canvas = np.vstack([header, hist_canvas])
            cv2.imshow(ADJUST_WIN, canvas)  # display adjust window with histogram

        with perf.timed("roi"):
            roi_pts = mouse_state["roi_pts"]
            if len(roi_pts) == 4:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(roi_pts, dtype=np.int32)], 255)  # polygon ROI mask
                frame_proc = cv2.bitwise_and(frame, frame, mask=mask)          # apply mask to frame
            else:
                frame_proc = frame
        with perf.timed("vision"):
            positions = tracker.update(frame_proc, t)
        global_positions: Dict[str, Optional[Point]] = dict(positions)

        _t_gates = perf.now()
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
                                was_armed = armed_before[name]
                                if g.digit == 0:
                                    armed_before[name] = True
                                ev = lap_tracker.on_forward_crossing(
                                    name, g.digit, t)
                                if ev is not None and ev["lap_time_s"] is not None:
                                    sound.play_triple()
                                else:
                                    sound.play()
                                if ev is not None and ev["lap_time_s"] is not None:
                                    # compute intersection before clearing the trail
                                    xpt = _trail_gate_xpt(
                                        prev_pt, gp, g, trail_tail)
                                    if xpt is not None:
                                        lap_trail[name].append(xpt)
                                    is_new_best = ev["lap_time_s"] <= (
                                        lap_tracker.best_lap_time(name)
                                        or float("inf"))
                                    if is_new_best:
                                        best_trail[name] = lap_trail[name][:]
                                    lap_trail[name].clear()
                                    if xpt is not None:
                                        lap_trail[name].append(xpt)
                                    lap_nr = lap_tracker.lap(name)
                                    msg = f"{name} {lap_nr}"
                                    if is_new_best and lap_nr > 1:
                                        msg += ", new best lap"
                                    sound.say(msg, speed=180)
                                    print(f"\n[lap] {name} lap {ev['lap']} "
                                          f"time={ev['lap_time_s']:.3f}s "
                                          f"(best={lap_tracker.best_lap_time(name):.3f}s)")
                                    tbl = lap_tracker.race_table(name)
                                    if tbl:
                                        print(tbl)
                                        print()
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
                                    # Gate 0 but invalid lap → still reset the trail.
                                    # was_armed=False is the very first start crossing
                                    # — skip the audio announcement in that case.
                                    xpt = _trail_gate_xpt(
                                        prev_pt, gp, g, trail_tail)
                                    if xpt is not None:
                                        lap_trail[name].append(xpt)
                                    lap_trail[name].clear()
                                    if xpt is not None:
                                        lap_trail[name].append(xpt)
                                    if was_armed:
                                        sound.say(f"{name}, you messed up",
                                                  speed=180)
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
        perf.mark("gates+trail", perf.now() - _t_gates)

        # ----- Overlay -----
        _t_overlay = perf.now()
        if use_clahe:
            from gates import _clahe
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # convert to LAB for CLAHE on L channel
            lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
            overlay = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert back to BGR
        else:
            overlay = frame.copy()

        if roi_pts:
            roi_color = (0, 200, 255)
            for p in roi_pts:
                cv2.circle(overlay, p, 6, roi_color, -1)  # ROI corner dot
            if len(roi_pts) == 4:
                arr = np.array(roi_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [arr], isClosed=True, color=roi_color,
                              thickness=2)                  # closed ROI polygon
            elif len(roi_pts) >= 2:
                for i in range(len(roi_pts) - 1):
                    cv2.line(overlay, roi_pts[i], roi_pts[i + 1],
                             roi_color, 2)                  # in-progress ROI edges
            if mouse_state["roi_editing"] and roi_pts:
                mx_live = mouse_state["x"]
                my_live = mouse_state["y"]
                if 0 <= mx_live < overlay.shape[1] and 0 <= my_live < overlay.shape[0]:
                    cv2.line(overlay, roi_pts[-1], (mx_live, my_live),
                             roi_color, 1)                  # rubber-band line to cursor

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
                                 (int(ex), int(ey)), (0, 255, 0), 5)  # flash green on crossing

        perf.mark("overlay_base", perf.now() - _t_overlay)

        # Trails: history + best lap at 50% opacity; current lap fully opaque
        _t_trails = perf.now()
        trail_layer = overlay.copy()
        for name in car_names:
            color = tracker.car(name).display_color_bgr()
            if len(trails[name]) >= 2:
                draw_polyline(trail_layer, list(trails[name]),
                              color=color, thickness=1, closed=False)
            if len(best_trail[name]) >= 2:
                draw_polyline(trail_layer, best_trail[name],
                              color=color, thickness=3, closed=False)
        cv2.addWeighted(trail_layer, 0.5, overlay, 0.5, 0, overlay)  # blend trail layer at 50%
        for name in car_names:
            color = tracker.car(name).display_color_bgr()
            if len(lap_trail[name]) >= 2:
                draw_polyline(overlay, lap_trail[name],
                              color=color, thickness=3, closed=False)
        perf.mark("trails", perf.now() - _t_trails)

        _t_hud = perf.now()
        # Collect per-car info lines first, then size the panel to fit
        info_lines: List[Tuple[str, Tuple[int, int, int]]] = []
        for name in car_names:
            color = tracker.car(name).display_color_bgr()
            gp = global_positions[name]

            if gp is not None:
                gx, gy = int(gp[0]), int(gp[1])
                cnt = tracker.contour(name)
                if cnt is not None:
                    cv2.drawContours(overlay, [cnt], -1, color, 1)  # car blob outline
                cv2.circle(overlay, (gx, gy), 6, color, -1)          # car position dot
                cv2.putText(overlay, name, (gx + 10, gy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            cv2.LINE_AA)                              # car name label

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
                      for t, _ in info_lines]  # measure text for panel sizing
            panel_w = (max(widths) if widths else 0) + 2 * hud.pad
            panel_h = len(info_lines) * hud.line_h + 2 * hud.pad
            _panel(overlay, 10, 10, panel_w, panel_h, hud.panel_alpha)
            y_cursor = 10 + hud.pad + hud.line_h - 10
            for text, col in info_lines:
                cv2.putText(overlay, text, (10 + hud.pad, y_cursor),
                            hud.FONT, hud.scale, col, hud.thickness,
                            cv2.LINE_AA)
                y_cursor += hud.line_h

        # Countdown / race traffic-light overlay
        if countdown_t0 is not None:
            lit = min(int(t - countdown_t0) + 1, 4)  # 1→2→3→4(GO)
            # horizontal traffic light — 3 lamps side by side
            lamp_r = 45
            gap = 16
            w_total = 3 * (2 * lamp_r) + 4 * gap
            h_total = 2 * lamp_r + 2 * gap
            cx = overlay.shape[1] // 2
            cy = overlay.shape[0] // 2
            x0 = cx - w_total // 2
            y0 = cy - h_total // 2
            cv2.rectangle(overlay, (x0, y0), (x0 + w_total, y0 + h_total),
                          (20, 20, 20), -1)   # housing fill
            cv2.rectangle(overlay, (x0, y0), (x0 + w_total, y0 + h_total),
                          (60, 60, 60), 3)    # housing border
            for i in range(3):
                lx = x0 + gap + lamp_r + i * (2 * lamp_r + gap)
                if lit >= 4:
                    color = (0, 200, 0)   # GO — all green
                elif i < lit:
                    color = (0, 0, 255)   # red lamp on
                else:
                    color = (30, 30, 30)  # lamp off
                cv2.circle(overlay, (lx, cy), lamp_r, color, -1)   # lamp fill
                cv2.circle(overlay, (lx, cy), lamp_r, (60, 60, 60), 2)  # lamp ring

        fps_frames += 1
        now = time.time()
        if now - fps_last_t >= 0.5:
            fps = fps_frames / (now - fps_last_t)
            fps_frames = 0
            fps_last_t = now

        # Top-right: FPS + lap/race status
        if show_hud:
            if current_mode is not None:
                tr_lines: List[str] = [
                    f"{fps:.1f} fps  "
                    f"{current_mode[1]}x{current_mode[2]}@"
                    f"{current_mode[3]:.0f}"
                ]
            else:
                tr_lines = [f"{fps:.1f} fps"]
            if race_active:
                laps_done = max(lap_tracker.lap(n) for n in car_names)
                tr_lines.append(f"Race {laps_done}/{RACE_LAPS}")
            else:
                tr_lines.append(f"Laps {RACE_LAPS}")
            tr_widths = [
                cv2.getTextSize(t, hud.FONT, hud.scale, hud.thickness)[0][0]
                for t in tr_lines]  # measure text for panel sizing
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

        # Bottom: help text — split across lines to fit the window width
        if show_hud:
            help_lines: List[str] = []
            if source == "sim":
                help_lines.append("r=reverse  Up/Down=speed")
            if (race_mode is not None and cal_mode is not None
                    and race_mode != cal_mode):
                other_mode = (cal_mode if current_mode == race_mode
                              else race_mode)
                h_hint = (f"h={other_mode[1]}x{other_mode[2]}@"
                          f"{other_mode[3]:.0f}")
            else:
                h_hint = "h=hires"
            help_lines.append(
                f"g=gates  k=contrast  a=adjust  c=roi  {h_hint}")
            help_lines.append(
                "s=start  n=new-race  Left/Right=laps  p=pause  t=clear-trails")
            help_lines.append("f=fullscreen  i=hud  q=quit")
            h_widths = [
                cv2.getTextSize(t, hud.FONT, hud.scale, hud.thickness)[0][0]
                for t in help_lines]  # measure text for panel sizing
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

        # HSV color picker follows the mouse; hidden after 10 s of inactivity
        # or when the cursor leaves the frame.
        mx, my = mouse_state["x"], mouse_state["y"]
        F_H, F_W = frame.shape[:2]
        mouse_active = (t - mouse_state["last_move_t"]) < 10.0
        if (show_hud and mouse_active
                and 0 <= mx < F_W and 0 <= my < F_H):
            bgr = frame[my, mx]
            hsv_px = cv2.cvtColor(        # single-pixel BGR→HSV for the picker readout
                np.array([[bgr]], dtype=np.uint8),
                cv2.COLOR_BGR2HSV)[0, 0]
            pick_lines = [
                f"({mx},{my})",
                f"HSV {hsv_px[0]} {hsv_px[1]} {hsv_px[2]}",
            ]
            p_widths = [
                cv2.getTextSize(t, hud.FONT, hud.scale, hud.thickness)[0][0]
                for t in pick_lines]
            sw = hud.line_h  # color swatch: square the height of one text line
            p_w = max(p_widths) + sw + 3 * hud.pad
            p_h = len(pick_lines) * hud.line_h + 2 * hud.pad
            cv2.drawMarker(overlay, (mx, my), (255, 255, 255),
                           cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)  # crosshair at cursor
            cv2.circle(overlay, (mx, my), 6, (0, 0, 0), 1, cv2.LINE_AA)   # inner dot ring
            off = 16
            p_x = mx + off
            p_y = my + off
            if p_x + p_w > overlay.shape[1] - 6:
                p_x = mx - p_w - off
            if p_y + p_h > overlay.shape[0] - 6:
                p_y = my - p_h - off
            p_x = max(6, p_x)
            p_y = max(6, p_y)
            _panel(overlay, p_x, p_y, p_w, p_h, hud.panel_alpha)
            sx0 = p_x + hud.pad
            sy0 = p_y + hud.pad
            cv2.rectangle(overlay, (sx0, sy0), (sx0 + sw, sy0 + sw),
                          (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)  # color swatch fill
            cv2.rectangle(overlay, (sx0, sy0), (sx0 + sw, sy0 + sw),
                          (80, 80, 80), 1)                               # swatch border
            py_cur = p_y + hud.pad + hud.line_h - 10
            tx = sx0 + sw + hud.pad
            for text in pick_lines:
                cv2.putText(overlay, text, (tx, py_cur),
                            hud.FONT, hud.scale, hud.color, hud.thickness,
                            cv2.LINE_AA)
                py_cur += hud.line_h
        perf.mark("hud", perf.now() - _t_hud)

        with perf.timed("imshow"):
            cv2.imshow(WINDOW, overlay)  # display the composite overlay frame

        for name in car_names:
            gp = global_positions[name]
            if gp is not None:
                writer.writerow([t, name, gp[0], gp[1], speed[name]])

        perf.tick()

        key = cv2.waitKey(1) & 0xFF  # poll keyboard; 1 ms timeout caps CPU usage
        if key == 255 and pending_key != 255:
            key = pending_key
        if key in (27, ord("q")):
            break
        elif key == ord("d"):
            perf.on = not perf.on
            perf.acc.clear()
            perf.n = 0
            print(f"[perf] {'ON' if perf.on else 'OFF'}")
        elif key == ord("p"):
            paused = not paused
        elif key == ord("n"):
            # Save the log before resetting
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
            # Reset timing + trails; keep gates. Also reset race state so the
            # Left/Right lap-limit keys work again before the next race.
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
                race_finished[name] = False
                armed_before[name] = False
            race_active = False
            countdown_t0 = None
            countdown_beeps = 0
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
        elif key == ord("k"):
            use_clahe = not use_clahe
            print(f"[clahe] {'ON' if use_clahe else 'OFF'}")
        elif key == ord("i"):
            show_hud = not show_hud
            print(f"[hud] {'ON' if show_hud else 'OFF'}")
        elif key == ord("c"):
            if mouse_state["roi_editing"]:
                mouse_state["roi_editing"] = False
                mouse_state["roi_pts"] = []
                print("[roi] editing cancelled")
            elif mouse_state["roi_pts"]:
                mouse_state["roi_pts"] = []
                print("[roi] cleared")
            else:
                mouse_state["roi_pts"] = []
                mouse_state["roi_editing"] = True
                print("[roi] click 4 corners (c to cancel)")
        elif key == ord("a"):
            if adjust_open:
                cv2.destroyWindow(ADJUST_WIN)  # close adjust window
                adjust_open = False
                print("[adjust] OFF")
            else:
                cv2.namedWindow(ADJUST_WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(ADJUST_WIN, 560, 520)
                for name, _default, maxv in ADJUST_SLIDERS:
                    cv2.createTrackbar(name, ADJUST_WIN, adjust_vals[name],
                                       maxv, lambda _v: None)  # slider; callback unused
                adjust_open = True
                print("[adjust] ON — 100=neutral")
        elif key == ord("g"):
            gates, gate_circles, gate_lines = detect_gates(
                frame_proc, use_clahe=use_clahe)
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
                        print("[gates] models/digits.pt not found — "
                              "run train_digits.py first")
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
                cv2.imshow("Gate Crops", panel)  # debug panel showing all gate crops
            cur_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            _save_gates(gates, source, resolution=cur_res)
            print(f"[session] saved {len(gates)} gates → {_gates_path(source)}")
        elif key == ord("h"):
            if (isinstance(source, int) and race_mode is not None
                    and cal_mode is not None and race_mode != cal_mode):
                new_mode = cal_mode if current_mode == race_mode else race_mode
                old_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                old_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                current_mode = new_mode
                cap = open_capture(source, current_mode)
                new_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                new_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if old_w > 0 and old_h > 0 and (old_w, old_h) != (new_w, new_h):
                    sx = new_w / old_w
                    sy = new_h / old_h
                    mouse_state["roi_pts"] = _scale_gates_and_roi(
                        gates, mouse_state["roi_pts"], sx, sy)
                    for tr in trails.values():
                        tr.clear()
                    for name in car_names:
                        lap_trail[name].clear()
                        best_trail[name].clear()
                        prev[name] = None
                last_frame_id = -1
                label = "cal" if current_mode == cal_mode else "race"
                print(f"[mode] {label} — {new_w}x{new_h}@"
                      f"{current_mode[3]:.0f}")
            else:
                print("[mode] toggle not available (sim or no modes detected)")
        elif key == ord("r"):
            if hasattr(cap, "reverse"):
                cap.reverse()
                print("[sim] reversed direction")
        elif key == 82 and hasattr(cap, "speed_up"):   # arrow up
            cap.speed_up()
            print(f"[sim] faster — period={cap._period:.2f}s")
        elif key == 84 and hasattr(cap, "speed_down"): # arrow down
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
            cv2.setWindowProperty(                    # toggle fullscreen mode
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

    final_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    _save_session({
        "lap_limit": RACE_LAPS,
        "roi_pts": [list(p) for p in mouse_state["roi_pts"]],
        "adjust": adjust_vals,
        "resolution": list(final_res),
    }, source)
    print(f"[session] saved {_session_path(source)}")

    log_f.close()
    cap.release()
    cv2.destroyAllWindows()  # close all OpenCV windows
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
