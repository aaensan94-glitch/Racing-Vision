import os
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
from timing import LapTracker
import sound

Point = Tuple[float, float]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(BASE_DIR, "configs")
CARS_CFG_PATH = os.path.join(CFG_DIR, "cars.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

WINDOW = "Race Vision CV1"

paused = False


def draw_polyline(img, pts: List[Point], color=(255, 255, 0), thickness=2,
                  closed=False):
    if pts is None or len(pts) < 2:
        return
    from geometry import catmull_rom
    if len(pts) >= 4:
        smooth: List[Point] = []
        for i in range(1, len(pts) - 2):
            smooth.extend(catmull_rom(pts[i - 1], pts[i], pts[i + 1],
                                      pts[i + 2], n=6))
        if smooth:
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

    fps = 0.0
    fps_last_t = time.time()
    fps_frames = 0

    gates: List[GateCandidate] = []
    gate_circles = None
    gate_lines = None
    show_gate_candidates = False
    use_clahe = False
    digit_classifier = None
    last_gate_hit: Dict[str, Tuple[float, int]] = {
        n: (0.0, -1) for n in car_names}
    GATE_DEBOUNCE_S = 0.5
    lap_tracker = LapTracker(car_names)
    RACE_LAPS = 15
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
            beep_idx = int(elapsed)  # 0, 1, 2 = Vorbereitungstöne, 3 = GO
            if beep_idx > countdown_beeps and countdown_beeps < 3:
                sound.play()
                countdown_beeps = beep_idx
                print(f"[countdown] {3 - countdown_beeps}...")
            if beep_idx >= 3 and countdown_beeps == 3:
                # GO — Rennen starten
                sound.play_finish()
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
                    last_gate_hit[name] = (0.0, -1)
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
                    direction = gate_crossed(
                        prev_pt, gp, g,
                        trail=list(trails[name]))
                    if direction != 0:
                        if (t - last_gate_hit[name][0]) > GATE_DEBOUNCE_S:
                            last_gate_hit[name] = (t, gi)
                            if direction > 0:
                                sound.play()
                                ev = lap_tracker.on_forward_crossing(
                                    name, g.digit, t)
                                if ev is not None and ev["lap_time_s"] is not None:
                                    # Ist diese Runde die neue Bestzeit?
                                    if ev["lap_time_s"] <= (
                                            lap_tracker.best_lap_time(name)
                                            or float("inf")):
                                        best_trail[name] = lap_trail[name][:]
                                    lap_trail[name].clear()
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
                                        sound.play_finish()
                                        print(f"\n*** {name} FINISHED "
                                              f"{RACE_LAPS} laps! ***")
                                        if all(race_finished.values()):
                                            race_active = False
                                            print("\n=== RACE COMPLETE ===")
                                elif ev is not None and ev["gate"] == 0:
                                    # Gate 0 aber ungültige Runde → trotzdem reset
                                    lap_trail[name].clear()
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

        if gates or show_gate_candidates:
            draw_gates(overlay, gates, gate_circles, gate_lines,
                       show_candidates=show_gate_candidates)
            for name in car_names:
                t_hit, gi = last_gate_hit[name]
                if gi >= 0 and (t - t_hit) < 0.4 and gi < len(gates):
                    g = gates[gi]
                    ax, ay = int(g.post_a[0]), int(g.post_a[1])
                    bx, by = int(g.post_b[0]), int(g.post_b[1])
                    cv2.line(overlay, (ax, ay), (bx, by), (0, 255, 255), 5)

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

        y_cursor = 30
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            pos_str = f"({int(gp[0])},{int(gp[1])})" if gp is not None else "none"
            hsv = tracker.hsv(name)
            hsv_str = (f"H={hsv[0]:.0f} S={hsv[1]:.0f} V={hsv[2]:.0f}"
                       if hsv is not None else "HSV=NA")
            line = (f"{name}[{hsv_str}]: pos={pos_str} "
                    f"speed={speed[name]:.0f}px/s")
            cv2.putText(overlay, line, (20, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_cursor += 24

            lap = lap_tracker.lap(name)
            last = lap_tracker.last_lap_time(name)
            best = lap_tracker.best_lap_time(name)
            cur = lap_tracker.current_lap_elapsed(name, t)
            last_s = f"{last:.3f}" if last is not None else "--"
            best_s = f"{best:.3f}" if best is not None else "--"
            cur_s = f"{cur:.2f}" if cur is not None else "--"
            lap_line = (f"  lap {lap}  cur={cur_s}s  last={last_s}s  "
                        f"best={best_s}s")
            cv2.putText(overlay, lap_line, (20, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y_cursor += 28

        # Countdown / Race Overlay
        if countdown_t0 is not None:
            remaining = max(0, 3 - int(t - countdown_t0))
            txt = str(remaining) if remaining > 0 else "GO!"
            clr = (0, 0, 255) if remaining > 0 else (0, 255, 0)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)
            cx = (overlay.shape[1] - tw) // 2
            cy = (overlay.shape[0] + th) // 2
            cv2.putText(overlay, txt, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, clr, 8, cv2.LINE_AA)
        elif race_active:
            laps_done = max(lap_tracker.lap(n) for n in car_names)
            cv2.putText(overlay, f"Race: {laps_done}/{RACE_LAPS}",
                        (overlay.shape[1] - 280, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        fps_frames += 1
        now = time.time()
        if now - fps_last_t >= 0.5:
            fps = fps_frames / (now - fps_last_t)
            fps_frames = 0
            fps_last_t = now
        cv2.putText(overlay, f"{fps:.1f} fps",
                    (overlay.shape[1] - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        keys_help = ("s=start  n=new-race  p=pause  t=clear-trails  "
                     "h=hsv  f=fullscreen  g=gates  G=debug  e=clahe  q=quit")
        if source == "sim":
            keys_help += "  r=reverse  Up/Dn=speed"
        cv2.putText(overlay, keys_help,
                    (20, overlay.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
                last_gate_hit[name] = (0.0, -1)
            print("[reset] new race — gates kept")
        elif key == ord("t"):
            for tr in trails.values():
                tr.clear()
            print("[cleared] trails")
        elif key == ord("s"):
            if gates:
                countdown_t0 = time.time()
                countdown_beeps = 0
                print("[countdown] 3...")
                sound.play()
            else:
                print("[start] detect gates first (g)")
        elif key == ord("e"):
            use_clahe = not use_clahe
            print(f"[clahe] {'ON' if use_clahe else 'OFF'}")
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
        elif key == ord("G"):
            show_gate_candidates = not show_gate_candidates
            print(f"[gates] show_candidates={show_gate_candidates}")
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
                print(f"[HSV around {name}] "
                      f"H={mean[0]:.1f}, S={mean[1]:.1f}, V={mean[2]:.1f}")

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
