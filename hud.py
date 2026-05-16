import json
from typing import List, Tuple

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
