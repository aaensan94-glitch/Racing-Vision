"""Application state and keyboard dispatcher.

Defines AppState (UI flags, gate list, capture handle, mouse state) and the
handle_key() dispatcher that maps key presses to system actions such as gate
calibration, race control, ROI editing, and camera mode switching.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2

import sound
from camera import CamMode, switch_camera_mode
from gates import (GateCandidate, detect_and_save_gates, gates_path,
                   load_gates, scale_gates_and_roi)
from race import RaceState
from session import LOGS_DIR, save_race_logs

WINDOW = "Race Vision CV1"
ADJUST_WIN = "Image Adjust"
ADJUST_SLIDERS: List[Tuple[str, int, int]] = [
    ("Brightness", 100, 200),
    ("Contrast",   100, 300),
    ("Gamma",      100, 300),
    ("Saturation", 100, 300),
]


@dataclass
class AppState:
    """Application-level state: capture, UI flags, gates, and mouse interaction.

    Attributes:
        cap: Active ThreadedCapture or SimCapture.
        mouse_state: Dict with keys x, y, last_move_t, roi_pts, roi_editing.
        gates: Detected and ordered gate list.
        gate_circles: Raw Hough circles from the last gate detection, or None.
        gate_lines: Raw Hough lines from the last gate detection, or None.
        digit_classifier: Lazy-loaded DigitClassifier, or None.
        use_clahe: Apply CLAHE contrast enhancement to the overlay.
        show_hud: Render the HUD overlay.
        adjust_open: The image-adjust trackbar window is open.
        adjust_vals: Current slider values keyed by slider name.
        fullscreen: Window is in fullscreen mode.
        paused: Frame capture and processing is paused.
        current_mode: Active CamMode, or None in simulation/default mode.
        last_frame_id: frame_id of the last processed frame.
    """

    cap: Any
    mouse_state: Dict
    gates: List[GateCandidate]
    gate_circles: Any
    gate_lines: Any
    digit_classifier: Any
    use_clahe: bool
    show_hud: bool
    adjust_open: bool
    adjust_vals: Dict[str, int]
    fullscreen: bool
    paused: bool
    current_mode: Optional[CamMode]
    last_frame_id: int

    @classmethod
    def create(cls, cap: Any, source, session: dict,
               actual_w: int, actual_h: int,
               current_mode: Optional[CamMode]) -> "AppState":
        """Creates AppState, loading and rescaling saved gates and ROI.

        Args:
            cap: Active capture object.
            source: Camera source identifier (int or "sim").
            session: Loaded session dict.
            actual_w: Current capture width in pixels.
            actual_h: Current capture height in pixels.
            current_mode: Currently active CamMode, or None.

        Returns:
            New AppState ready for the main loop.
        """
        # Rescale saved ROI to the current resolution
        saved_roi = session.get("roi_pts", [])
        saved_sess_res = session.get("resolution")
        roi_pts: List[Tuple[int, int]] = [
            (int(p[0]), int(p[1])) for p in saved_roi
            if isinstance(p, (list, tuple)) and len(p) == 2
        ] if len(saved_roi) == 4 else []
        if (roi_pts and saved_sess_res and len(saved_sess_res) == 2
                and actual_w > 0 and actual_h > 0):
            osw, osh = int(saved_sess_res[0]), int(saved_sess_res[1])
            if (osw, osh) != (actual_w, actual_h) and osw > 0 and osh > 0:
                sx, sy = actual_w / osw, actual_h / osh
                roi_pts = [(int(p[0] * sx), int(p[1] * sy)) for p in roi_pts]
                print(f"[session] roi rescaled {osw}x{osh} -> "
                      f"{actual_w}x{actual_h}")

        # Load and rescale saved gates
        gates, saved_gates_res = load_gates(source)
        if gates:
            print(f"[session] loaded {len(gates)} gates from {gates_path(source)}")
            if (saved_gates_res and actual_w > 0 and actual_h > 0
                    and saved_gates_res != (actual_w, actual_h)):
                sx = actual_w / saved_gates_res[0]
                sy = actual_h / saved_gates_res[1]
                scale_gates_and_roi(gates, [], sx, sy)
                print(f"[session] gates rescaled "
                      f"{saved_gates_res[0]}x{saved_gates_res[1]} -> "
                      f"{actual_w}x{actual_h}")

        saved_adjust = session.get("adjust", {})
        adjust_vals = {
            name: int(saved_adjust.get(name, default))
            for name, default, _ in ADJUST_SLIDERS
        }

        return cls(
            cap=cap,
            mouse_state={
                "x": -1, "y": -1,
                "last_move_t": 0.0,
                "roi_pts": roi_pts,
                "roi_editing": False,
            },
            gates=gates,
            gate_circles=None,
            gate_lines=None,
            digit_classifier=None,
            use_clahe=False,
            show_hud=True,
            adjust_open=False,
            adjust_vals=adjust_vals,
            fullscreen=True,
            paused=False,
            current_mode=current_mode,
            last_frame_id=-1,
        )

    def on_mouse(self, event: int, x: int, y: int,
                 flags: int, param: Any) -> None:
        """OpenCV mouse callback; updates position and ROI points in mouse_state."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_state["x"] = int(x)
            self.mouse_state["y"] = int(y)
            self.mouse_state["last_move_t"] = time.time()
        elif event == cv2.EVENT_LBUTTONDOWN and self.mouse_state["roi_editing"]:
            pts = self.mouse_state["roi_pts"]
            if len(pts) < 4:
                pts.append((int(x), int(y)))
                print(f"[roi] point {len(pts)}/4: ({x},{y})")
                if len(pts) == 4:
                    self.mouse_state["roi_editing"] = False
                    print("[roi] 4 points set — mask active")


def handle_key(key: int, app: AppState, state: RaceState,
               frame, frame_proc, source, car_names: List[str],
               race_mode: Optional[CamMode], cal_mode: Optional[CamMode],
               perf) -> bool:
    """Dispatches a keypress; mutates app and state in place.

    Args:
        key: Key code from cv2.waitKey (255 = no key pressed).
        app: Application state, mutated in place.
        state: Race state, mutated in place.
        frame: Current BGR frame (used by 'g' for gate detection).
        frame_proc: ROI-masked frame (used by 'g' for gate detection).
        source: Camera source identifier.
        car_names: Ordered car names.
        race_mode: Race CamMode, or None.
        cal_mode: Calibration CamMode, or None.
        perf: Perf instance.

    Returns:
        True if the caller should exit the main loop.
    """
    if key in (27, ord("q")):
        return True
    elif key == ord("d"):
        perf.on = not perf.on
        perf.acc.clear()
        perf.n = 0
        print(f"[perf] {'ON' if perf.on else 'OFF'}")
    elif key == ord("p"):
        app.paused = not app.paused
    elif key == ord("n"):
        save_race_logs(state.lap_tracker, LOGS_DIR,
                       time.strftime("%Y%m%d_%H%M%S"), tag="reset")
        state.new_race()
        print("[reset] new race — gates kept")
    elif key == ord("t"):
        for tr in state.trails.values():
            tr.clear()
        print("[cleared] trails")
    elif key == ord("s"):
        if app.gates:
            state.countdown_t0 = time.time()
            state.countdown_beeps = 0
            sound.play_countdown()
            state.countdown_beeps = 1
            print("[countdown] 3...")
        else:
            print("[start] detect gates first (g)")
    elif key == ord("k"):
        app.use_clahe = not app.use_clahe
        print(f"[clahe] {'ON' if app.use_clahe else 'OFF'}")
    elif key == ord("i"):
        app.show_hud = not app.show_hud
        print(f"[hud] {'ON' if app.show_hud else 'OFF'}")
    elif key == ord("c"):
        if app.mouse_state["roi_editing"]:
            app.mouse_state["roi_editing"] = False
            app.mouse_state["roi_pts"] = []
            print("[roi] editing cancelled")
        elif app.mouse_state["roi_pts"]:
            app.mouse_state["roi_pts"] = []
            print("[roi] cleared")
        else:
            app.mouse_state["roi_pts"] = []
            app.mouse_state["roi_editing"] = True
            print("[roi] click 4 corners (c to cancel)")
    elif key == ord("a"):
        if app.adjust_open:
            cv2.destroyWindow(ADJUST_WIN)
            app.adjust_open = False
            print("[adjust] OFF")
        else:
            cv2.namedWindow(ADJUST_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(ADJUST_WIN, 560, 520)
            for name, _default, maxv in ADJUST_SLIDERS:
                cv2.createTrackbar(name, ADJUST_WIN, app.adjust_vals[name],
                                   maxv, lambda _v: None)  # callback unused
            app.adjust_open = True
            print("[adjust] ON — 100=neutral")
    elif key == ord("g"):
        app.gates, app.gate_circles, app.gate_lines, app.digit_classifier = (
            detect_and_save_gates(frame, frame_proc, app.use_clahe,
                                  app.digit_classifier, state.lap_tracker,
                                  car_names, app.cap, source))
    elif key == ord("h"):
        if (isinstance(source, int) and race_mode is not None
                and cal_mode is not None and race_mode != cal_mode):
            app.cap, app.current_mode, app.last_frame_id = switch_camera_mode(
                source, app.cap, app.current_mode, race_mode, cal_mode,
                app.gates, app.mouse_state, state.trails, state.car_names,
                state.lap_trail, state.best_trail, state.prev)
        else:
            print("[mode] toggle not available (sim or no modes detected)")
    elif key == ord("r"):
        if hasattr(app.cap, "reverse"):
            app.cap.reverse()
            print("[sim] reversed direction")
    elif key == 82 and hasattr(app.cap, "speed_up"):    # arrow up
        app.cap.speed_up()
        print(f"[sim] faster — period={app.cap._period:.2f}s")
    elif key == 84 and hasattr(app.cap, "speed_down"):  # arrow down
        app.cap.speed_down()
        print(f"[sim] slower — period={app.cap._period:.2f}s")
    elif key == 83 and not state.race_active and state.countdown_t0 is None:  # right
        state.race_laps = min(100, state.race_laps + 5)
        print(f"[race] laps={state.race_laps}")
    elif key == 81 and not state.race_active and state.countdown_t0 is None:  # left
        state.race_laps = max(5, state.race_laps - 5)
        print(f"[race] laps={state.race_laps}")
    elif key == ord("f"):
        app.fullscreen = not app.fullscreen
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if app.fullscreen else cv2.WINDOW_NORMAL,
        )
    return False
