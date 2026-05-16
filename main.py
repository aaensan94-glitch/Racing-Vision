import time
from typing import Dict, Optional, Tuple

import cv2

from app import AppState, WINDOW, ADJUST_WIN, ADJUST_SLIDERS, handle_key
from camera import setup_camera, prompt_camera_choice
from session import (LOGS_DIR, HUD_CFG_PATH, CARS_CFG_PATH,
                     _load_session, _save_session, _session_path,
                     save_race_logs, open_log)
from hud import (HudConfig, draw_trails, draw_traffic_light, draw_info_panel,
                 draw_status_panel, draw_help_panel, draw_hsv_picker,
                 draw_roi_overlay, draw_gate_flash, draw_adjust_window,
                 draw_car_markers, build_help_lines)
from perf import Perf
from vision import MultiTracker, apply_image_adjustments, apply_roi_mask
from gates import draw_gates
from race import RaceState, tick_countdown, process_gate_crossings

Point = Tuple[float, float]

DISPLAY_WIDTH = 1280
TRAIL_LEN = 0  # 0 = unlimited, otherwise max points per trail


def main():
    source = prompt_camera_choice()
    hud = HudConfig(HUD_CFG_PATH)
    tracker = MultiTracker.load(CARS_CFG_PATH)
    car_names = tracker.names()

    cap, race_mode, cal_mode, current_mode, actual_w, actual_h = setup_camera(source)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_f, writer, log_path = open_log(LOGS_DIR, ts)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizable display window
    if actual_w > 0:
        cv2.resizeWindow(WINDOW, DISPLAY_WIDTH,
                         int(actual_h * DISPLAY_WIDTH / actual_w))

    session = _load_session(source)
    app = AppState.create(cap, source, session, actual_w, actual_h, current_mode)
    cv2.setMouseCallback(WINDOW, app.on_mouse)  # register mouse handler for ROI editing

    num_gates = max((g.digit for g in app.gates if g.digit >= 0), default=-1) + 1
    trail_maxlen = TRAIL_LEN if TRAIL_LEN > 0 else None
    race_laps = max(5, min(100, int(session.get("lap_limit", 5))))
    state = RaceState.create(car_names, num_gates=max(0, num_gates),
                             race_laps=race_laps, trail_maxlen=trail_maxlen)
    perf = Perf()
    fps, fps_last_t, fps_frames = 0.0, time.time(), 0

    while True:
        pending_key = 255  # 255 = no key pressed
        with perf.timed("capture"):
            if not app.paused:
                # Sync to camera FPS: wait for a new frame_id before reading.
                # Buffer any key pressed while waiting so the handler below
                # can process it without dropping input.
                if hasattr(app.cap, "frame_id"):
                    while True:
                        fid = app.cap.frame_id()
                        if fid != app.last_frame_id:
                            app.last_frame_id = fid
                            break
                        k = cv2.waitKey(1) & 0xFF
                        if k != 255:
                            pending_key = k
                ok, frame = app.cap.read()
                if not ok:
                    break

        t = time.time()
        tick_countdown(state, t)

        with perf.timed("filter"):
            if app.adjust_open:
                for name, _d, _m in ADJUST_SLIDERS:
                    app.adjust_vals[name] = cv2.getTrackbarPos(name, ADJUST_WIN)  # read slider
            frame = apply_image_adjustments(frame, app.adjust_vals)

        if app.adjust_open:
            draw_adjust_window(frame, app.adjust_vals, ADJUST_SLIDERS, ADJUST_WIN)

        with perf.timed("roi"):
            frame_proc = apply_roi_mask(frame, app.mouse_state["roi_pts"])

        with perf.timed("vision"):
            positions = tracker.update(frame_proc, t)
        global_positions: Dict[str, Optional[Point]] = dict(positions)

        _t_gates = perf.now()
        process_gate_crossings(state, app.gates, global_positions, t)
        perf.mark("gates+trail", perf.now() - _t_gates)

        # ----- Overlay -----
        _t_overlay = perf.now()
        if app.use_clahe:
            from gates import _clahe
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # convert to LAB for CLAHE on L channel
            lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
            overlay = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert back to BGR
        else:
            overlay = frame.copy()
        draw_roi_overlay(overlay, app.mouse_state["roi_pts"],
                         app.mouse_state["roi_editing"],
                         app.mouse_state["x"], app.mouse_state["y"])
        if app.gates:
            draw_gates(overlay, app.gates)
            draw_gate_flash(overlay, app.gates, state.last_gate_flash, car_names, t)
        perf.mark("overlay_base", perf.now() - _t_overlay)

        _t_trails = perf.now()
        colors = {name: tracker.car(name).display_color_bgr() for name in car_names}
        draw_trails(overlay, car_names, state.trails, state.best_trail,
                    state.lap_trail, colors)
        perf.mark("trails", perf.now() - _t_trails)

        _t_hud = perf.now()
        info_lines = draw_car_markers(overlay, car_names, global_positions,
                                      tracker, state.speed, state.lap_tracker,
                                      t, colors)
        if app.show_hud:
            draw_info_panel(overlay, hud, info_lines)
        if state.countdown_t0 is not None:
            draw_traffic_light(overlay, state.countdown_t0, t)
        fps_frames += 1
        now_t = time.time()
        if now_t - fps_last_t >= 0.5:
            fps = fps_frames / (now_t - fps_last_t)
            fps_frames = 0
            fps_last_t = now_t
        if app.show_hud:
            laps_done = max(state.lap_tracker.lap(n) for n in car_names)
            draw_status_panel(overlay, hud, fps, app.current_mode,
                              state.race_active, laps_done, state.race_laps)
            draw_help_panel(overlay, hud,
                            build_help_lines(source, race_mode, cal_mode,
                                             app.current_mode))
            draw_hsv_picker(overlay, frame, app.mouse_state["x"],
                            app.mouse_state["y"],
                            app.mouse_state["last_move_t"], t, hud)
        perf.mark("hud", perf.now() - _t_hud)

        with perf.timed("imshow"):
            cv2.imshow(WINDOW, overlay)  # display the composite overlay frame

        for name in car_names:
            gp = global_positions[name]
            if gp is not None:
                writer.writerow([t, name, gp[0], gp[1], state.speed[name]])

        perf.tick()

        key = cv2.waitKey(1) & 0xFF  # poll keyboard; 1 ms timeout caps CPU usage
        if key == 255 and pending_key != 255:
            key = pending_key
        if handle_key(key, app, state, frame, frame_proc,
                      source, car_names, race_mode, cal_mode, perf):
            break

    final_res = (int(app.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(app.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    _save_session({
        "lap_limit": state.race_laps,
        "roi_pts": [list(p) for p in app.mouse_state["roi_pts"]],
        "adjust": app.adjust_vals,
        "resolution": list(final_res),
    }, source)
    print(f"[session] saved {_session_path(source)}")
    log_f.close()
    app.cap.release()
    cv2.destroyAllWindows()  # close all OpenCV windows
    print(f"[done] log saved: {log_path}")
    save_race_logs(state.lap_tracker, LOGS_DIR, ts, tag="done")


if __name__ == "__main__":
    main()
