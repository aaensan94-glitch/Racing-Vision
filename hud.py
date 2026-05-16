import json
from collections import deque
from typing import Dict, Deque, List, Optional, Tuple

import cv2
import numpy as np

from geometry import catmull_rom

Point = Tuple[float, float]


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


def _make_histogram(bgr: np.ndarray, w: int = 420, h: int = 200) -> np.ndarray:
    """Returns a BGR histogram with axis labels, sized to fit the Adjust window.

    Args:
        bgr: BGR source image.
        w: Canvas width in pixels.
        h: Canvas height in pixels.

    Returns:
        BGR histogram image.
    """
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    axis_y = h - 22
    plot_h = axis_y - 10
    channel_colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]
    for i, col in enumerate(channel_colors):
        hist = cv2.calcHist([bgr], [i], None, [256], [0, 256]).flatten()  # per-channel histogram
        m = float(hist.max()) or 1.0
        pts = np.zeros((256, 2), dtype=np.int32)
        for x in range(256):
            pts[x, 0] = int(x * (w - 1) / 255)
            pts[x, 1] = axis_y - int(hist[x] / m * plot_h)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, col, 1,
                      cv2.LINE_AA)  # draw histogram curve
    cv2.line(canvas, (0, axis_y), (w, axis_y), (90, 90, 90), 1)  # x-axis baseline
    font = cv2.FONT_HERSHEY_SIMPLEX
    for val, lx in ((0, 2), (64, w // 4 - 8), (128, w // 2 - 12),
                    (192, 3 * w // 4 - 12), (255, w - 32)):
        cv2.line(canvas, (lx + 10, axis_y), (lx + 10, axis_y + 3),
                 (120, 120, 120), 1)  # tick mark
        cv2.putText(canvas, str(val), (lx, axis_y + 16), font, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)  # tick label
    return canvas


def _panel(img, x: int, y: int, w: int, h: int, alpha: float = 0.6) -> None:
    """Draws a dark translucent background so text remains readable over bright images.

    Args:
        img: BGR image to draw on (modified in place).
        x: Panel left edge.
        y: Panel top edge.
        w: Panel width.
        h: Panel height.
        alpha: Darkness factor (0 = transparent, 1 = fully black).
    """
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0:y1, x0:x1]
    dark = np.zeros_like(sub)
    cv2.addWeighted(sub, 1 - alpha, dark, alpha, 0, sub)  # blend toward black in place


def draw_polyline(img, pts: List[Point], color=(255, 255, 0), thickness=2,
                  closed=False):
    """Draws a Catmull-Rom smoothed polyline onto img.

    For four or more points, phantom endpoints are mirrored at both ends so the
    curve rounds cleanly at the start and finish.  Each pair of original points
    is subdivided into six interpolated segments before drawing.  Fewer than
    four points fall back to a plain polyline.

    Args:
        img: BGR image to draw on (modified in place).
        pts: Ordered list of (x, y) points along the path.
        color: BGR draw color.
        thickness: Line thickness in pixels.
        closed: Whether to close the polyline back to the first point.
    """
    if pts is None or len(pts) < 2:
        return
    if len(pts) >= 4:
        # mirror phantom points so the path curves at both endpoints
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
    cv2.polylines(img, [p], isClosed=closed, color=color, thickness=thickness)  # smooth trail curve


def draw_trails(overlay: np.ndarray, car_names: List[str],
                trails: Dict[str, Deque[Point]],
                best_trail: Dict[str, List[Point]],
                lap_trail: Dict[str, List[Point]],
                colors: Dict[str, Tuple[int, int, int]]) -> None:
    """Draws history, best-lap, and current-lap trails onto the overlay.

    History and best-lap trails are blended at 50% opacity; the current-lap
    trail is drawn fully opaque on top.

    Args:
        overlay: BGR image to draw on (modified in place).
        car_names: Ordered list of car names.
        trails: Full position history per car.
        best_trail: Best-lap path per car.
        lap_trail: Current-lap path per car.
        colors: BGR draw color per car.
    """
    trail_layer = overlay.copy()
    for name in car_names:
        color = colors[name]
        if len(trails[name]) >= 2:
            draw_polyline(trail_layer, list(trails[name]),
                          color=color, thickness=1, closed=False)
        if len(best_trail[name]) >= 2:
            draw_polyline(trail_layer, best_trail[name],
                          color=color, thickness=3, closed=False)
    cv2.addWeighted(trail_layer, 0.5, overlay, 0.5, 0, overlay)  # blend trail layer at 50%
    for name in car_names:
        color = colors[name]
        if len(lap_trail[name]) >= 2:
            draw_polyline(overlay, lap_trail[name],
                          color=color, thickness=3, closed=False)


def draw_traffic_light(overlay: np.ndarray, countdown_t0: float,
                       t: float) -> None:
    """Draws a horizontal 3-lamp traffic light centered on the overlay.

    Args:
        overlay: BGR image to draw on (modified in place).
        countdown_t0: Timestamp when the countdown started.
        t: Current timestamp.
    """
    lit = min(int(t - countdown_t0) + 1, 4)  # 1→2→3→4(GO)
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
            color = (0, 200, 0)    # GO — all green
        elif i < lit:
            color = (0, 0, 255)    # red lamp on
        else:
            color = (30, 30, 30)   # lamp off
        cv2.circle(overlay, (lx, cy), lamp_r, color, -1)         # lamp fill
        cv2.circle(overlay, (lx, cy), lamp_r, (60, 60, 60), 2)   # lamp ring


def draw_info_panel(overlay: np.ndarray, hud: HudConfig,
                    info_lines: List[Tuple[str, Tuple[int, int, int]]]) -> None:
    """Draws the top-left per-car info panel (HSV, position, speed, lap times).

    Args:
        overlay: BGR image to draw on (modified in place).
        hud: HUD configuration (font, padding, etc.).
        info_lines: List of (text, color) tuples, one or two per car.
    """
    widths = [cv2.getTextSize(line, hud.FONT, hud.scale, hud.thickness)[0][0]
              for line, _ in info_lines]
    panel_w = (max(widths) if widths else 0) + 2 * hud.pad
    panel_h = len(info_lines) * hud.line_h + 2 * hud.pad
    _panel(overlay, 10, 10, panel_w, panel_h, hud.panel_alpha)
    y = 10 + hud.pad + hud.line_h - 10
    for text, col in info_lines:
        cv2.putText(overlay, text, (10 + hud.pad, y),
                    hud.FONT, hud.scale, col, hud.thickness, cv2.LINE_AA)
        y += hud.line_h


def draw_status_panel(overlay: np.ndarray, hud: HudConfig, fps: float,
                      current_mode, race_active: bool,
                      laps_done: int, race_laps: int) -> None:
    """Draws the top-right FPS and race-status panel.

    Args:
        overlay: BGR image to draw on (modified in place).
        hud: HUD configuration.
        fps: Current frames per second.
        current_mode: Active CamMode tuple, or None.
        race_active: Whether a race is currently running.
        laps_done: Highest lap count across all cars.
        race_laps: Total laps required to finish the race.
    """
    if current_mode is not None:
        lines = [f"{fps:.1f} fps  "
                 f"{current_mode[1]}x{current_mode[2]}@{current_mode[3]:.0f}"]
    else:
        lines = [f"{fps:.1f} fps"]
    lines.append(f"Race {laps_done}/{race_laps}" if race_active
                 else f"Laps {race_laps}")
    widths = [cv2.getTextSize(ln, hud.FONT, hud.scale, hud.thickness)[0][0]
              for ln in lines]
    w = max(widths) + 2 * hud.pad
    h = len(lines) * hud.line_h + 2 * hud.pad
    x = overlay.shape[1] - w - 10
    y = 10
    _panel(overlay, x, y, w, h, hud.panel_alpha)
    cursor = y + hud.pad + hud.line_h - 10
    for text in lines:
        cv2.putText(overlay, text, (x + hud.pad, cursor),
                    hud.FONT, hud.scale, hud.color, hud.thickness, cv2.LINE_AA)
        cursor += hud.line_h


def draw_help_panel(overlay: np.ndarray, hud: HudConfig,
                    help_lines: List[str]) -> None:
    """Draws the bottom help-text panel.

    Args:
        overlay: BGR image to draw on (modified in place).
        hud: HUD configuration.
        help_lines: Lines of help text to display.
    """
    widths = [cv2.getTextSize(ln, hud.FONT, hud.scale, hud.thickness)[0][0]
              for ln in help_lines]
    w = max(widths) + 2 * hud.pad
    h = len(help_lines) * hud.line_h + 2 * hud.pad
    x = 10
    y = overlay.shape[0] - h - 10
    _panel(overlay, x, y, w, h, hud.panel_alpha)
    cursor = y + hud.pad + hud.line_h - 10
    for text in help_lines:
        cv2.putText(overlay, text, (x + hud.pad, cursor),
                    hud.FONT, hud.scale, hud.color, hud.thickness, cv2.LINE_AA)
        cursor += hud.line_h


def draw_hsv_picker(overlay: np.ndarray, frame: np.ndarray,
                    mx: int, my: int, last_move_t: float,
                    now: float, hud: HudConfig) -> None:
    """Draws a floating HSV color-picker tooltip under the mouse cursor.

    Hidden when the cursor is outside the frame or has been idle for 10 s.

    Args:
        overlay: BGR image to draw on (modified in place).
        frame: Source frame used to sample the pixel color.
        mx: Mouse x coordinate in frame pixels.
        my: Mouse y coordinate in frame pixels.
        last_move_t: Timestamp of the last mouse-move event.
        now: Current timestamp.
        hud: HUD configuration.
    """
    F_H, F_W = frame.shape[:2]
    if (now - last_move_t) >= 10.0 or not (0 <= mx < F_W and 0 <= my < F_H):
        return
    bgr = frame[my, mx]
    hsv_px = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8),
                           cv2.COLOR_BGR2HSV)[0, 0]
    pick_lines = [f"({mx},{my})", f"HSV {hsv_px[0]} {hsv_px[1]} {hsv_px[2]}"]
    p_widths = [cv2.getTextSize(ln, hud.FONT, hud.scale, hud.thickness)[0][0]
                for ln in pick_lines]
    sw = hud.line_h  # color swatch: square the height of one text line
    p_w = max(p_widths) + sw + 3 * hud.pad
    p_h = len(pick_lines) * hud.line_h + 2 * hud.pad
    cv2.drawMarker(overlay, (mx, my), (255, 255, 255),
                   cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)  # crosshair at cursor
    cv2.circle(overlay, (mx, my), 6, (0, 0, 0), 1, cv2.LINE_AA)   # inner dot ring
    off = 16
    p_x = min(mx + off, overlay.shape[1] - p_w - 6)
    p_y = min(my + off, overlay.shape[0] - p_h - 6)
    if p_x < mx and p_x < 6:  # flip left if still off-screen
        p_x = mx - p_w - off
    p_x = max(6, p_x)
    p_y = max(6, p_y)
    _panel(overlay, p_x, p_y, p_w, p_h, hud.panel_alpha)
    sx0 = p_x + hud.pad
    sy0 = p_y + hud.pad
    cv2.rectangle(overlay, (sx0, sy0), (sx0 + sw, sy0 + sw),
                  (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)  # color swatch fill
    cv2.rectangle(overlay, (sx0, sy0), (sx0 + sw, sy0 + sw),
                  (80, 80, 80), 1)                               # swatch border
    cursor = p_y + hud.pad + hud.line_h - 10
    tx = sx0 + sw + hud.pad
    for text in pick_lines:
        cv2.putText(overlay, text, (tx, cursor),
                    hud.FONT, hud.scale, hud.color, hud.thickness, cv2.LINE_AA)
        cursor += hud.line_h
