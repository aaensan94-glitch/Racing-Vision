"""Simulated webcam for testing without physical hardware.

Renders a white-paper scene with three hand-drawn gates and two colored dots
traveling on a circular track through all three gates.

Duck-type compatible with cv2.VideoCapture: .read(), .get(), .set(),
.isOpened(), .release().
"""
import json
import os
import time
from typing import Tuple

import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
FPS = 30

# Center and radius of the circular track (pixels)
TRACK_CX = WIDTH // 2
TRACK_CY = HEIGHT // 2
TRACK_R = 240

# Three gates evenly spaced on the track (angles in radians)
GATE_ANGLES = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
GATE_DIGITS = (0, 2, 1)
POST_R = 28
POST_GAP = 160  # center-to-center distance between posts (driveable gap ≈ GAP - 2R)

DOT_R = 12
DOT_PERIOD_S = 8.0  # seconds per lap


def _dot_bgr_from_cars(cfg_path: str, car: str = "cyan") -> Tuple[int, int, int]:
    """Returns a BGR color at the midpoint of the HSV range for the given car.

    Falls back to pure cyan or magenta if the config is unavailable.

    Args:
        cfg_path: Path to cars.json.
        car: Car name key in the config.

    Returns:
        BGR tuple.
    """
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        lo = cfg[car]["hsv_lower"]
        hi = cfg[car]["hsv_upper"]
        h = int((lo[0] + hi[0]) / 2)
        s = int((lo[1] + hi[1]) / 2)
        v = int((lo[2] + hi[2]) / 2)
        px = np.array([[[h, s, v]]], dtype=np.uint8)
        bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]  # single-pixel HSV→BGR
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    except (OSError, KeyError, ValueError):
        return (255, 255, 0) if car == "cyan" else (255, 0, 255)


_CARS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "configs", "cars.json")
DOT_BGR = _dot_bgr_from_cars(_CARS_JSON, "cyan")
DOT2_BGR = _dot_bgr_from_cars(_CARS_JSON, "magenta")
DOT2_PHASE = np.pi   # 180° offset on the track
DOT_R_OFFSET = -20   # cyan runs slightly inside the track center
DOT2_R_OFFSET = 20   # magenta runs slightly outside


def _draw_gate(img: np.ndarray, cx: float, cy: float, tangent_angle: float,
               digit: int) -> None:
    """Draws a gate perpendicular to the track tangent at the given position.

    Args:
        img: Canvas to draw on (modified in place).
        cx: Gate center x in pixels.
        cy: Gate center y in pixels.
        tangent_angle: Track tangent angle in radians at this gate.
        digit: Gate number to inscribe in the inner post.
    """
    # posts are perpendicular to the tangent, i.e. along the radial direction
    nx, ny = -np.sin(tangent_angle), np.cos(tangent_angle)
    half = POST_GAP / 2.0
    ax = int(cx - nx * half); ay = int(cy - ny * half)
    bx = int(cx + nx * half); by = int(cy + ny * half)

    # inner post (toward track center) carries the digit
    inner_dist = np.hypot(ax - TRACK_CX, ay - TRACK_CY)
    outer_dist = np.hypot(bx - TRACK_CX, by - TRACK_CY)
    if inner_dist > outer_dist:
        ax, bx = bx, ax
        ay, by = by, ay

    cv2.circle(img, (ax, ay), POST_R, (40, 40, 40), 2)  # left (digit) post outline
    cv2.circle(img, (bx, by), POST_R, (40, 40, 40), 2)  # right (empty) post outline
    # connecting line only between the inner post tangent points
    dx, dy = bx - ax, by - ay
    L = float(np.hypot(dx, dy))
    if L > 2 * POST_R:
        ux, uy = dx / L, dy / L
        sx, sy = int(ax + ux * POST_R), int(ay + uy * POST_R)
        ex, ey = int(bx - ux * POST_R), int(by - uy * POST_R)
        cv2.line(img, (sx, sy), (ex, ey), (40, 40, 40), 2)  # traversable segment

    # digit inscribed in the inner post, pre-rotated so it reads upright after
    # the pipeline's gate-alignment rotation
    text = str(digit)
    side = int(POST_R * 2)
    canvas = np.full((side, side, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_COMPLEX  # serif → "1" has a foot, closer to MNIST style
    (tw, th), _ = cv2.getTextSize(text, font, 1.1, 2)
    cv2.putText(canvas, text,
                (side // 2 - tw // 2, side // 2 + th // 2),
                font, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    angle_deg = float(np.degrees(np.arctan2(by - ay, bx - ax)))
    # pipeline rotates by +angle_deg, so pre-compensate by -angle_deg here
    M = cv2.getRotationMatrix2D((side / 2, side / 2), -angle_deg, 1.0)  # rotation matrix
    rot = cv2.warpAffine(canvas, M, (side, side),                        # apply rotation
                         flags=cv2.INTER_LINEAR,
                         borderValue=(245, 245, 245))
    x0, y0 = ax - side // 2, ay - side // 2
    x1, y1 = x0 + side, y0 + side
    if 0 <= x0 and x1 <= img.shape[1] and 0 <= y0 and y1 <= img.shape[0]:
        # min blend: dark digit ink wins over paper without erasing the post circle
        roi = img[y0:y1, x0:x1]
        np.minimum(roi, rot, out=roi)


def _build_background() -> np.ndarray:
    bg = np.full((HEIGHT, WIDTH, 3), 245, dtype=np.uint8)  # light-gray paper
    # slight noise makes Hough detection behave more realistically
    noise = np.random.randint(-6, 7, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for ang, d in zip(GATE_ANGLES, GATE_DIGITS):
        cx = TRACK_CX + TRACK_R * np.cos(ang)
        cy = TRACK_CY + TRACK_R * np.sin(ang)
        _draw_gate(bg, cx, cy, ang + np.pi / 2, d)
    return bg


class SimCapture:
    """Simulated camera that produces frames at a fixed FPS.

    Attributes:
        SPEED_FACTOR: Multiplier applied to the lap period on each
            speed_up / speed_down call.
    """

    SPEED_FACTOR = 1.25

    def __init__(self):
        self._bg = _build_background()
        self._t0 = time.time()
        self._opened = True
        self._dir = -1  # -1 = forward convention, +1 = reverse (for testing)
        self._period = DOT_PERIOD_S

    def _current_ang(self, now: float) -> float:
        return self._dir * 2 * np.pi * ((now - self._t0) / self._period)

    def _retune(self, new_period: float) -> None:
        """Changes lap period without a phase jump at the current position."""
        now = time.time()
        ang = self._current_ang(now)
        self._period = new_period
        # adjust t0 so the current angle stays consistent under the new period
        self._t0 = now - ang * self._period / (self._dir * 2 * np.pi)

    def reverse(self) -> None:
        """Reverses travel direction without a position jump."""
        now = time.time()
        self._t0 = 2 * now - self._t0
        self._dir = -self._dir

    def speed_up(self) -> None:
        """Increases lap speed by SPEED_FACTOR."""
        self._retune(self._period / self.SPEED_FACTOR)

    def speed_down(self) -> None:
        """Decreases lap speed by SPEED_FACTOR."""
        self._retune(self._period * self.SPEED_FACTOR)

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, np.ndarray]:
        """Returns (True, frame) at the simulated FPS rate."""
        if not self._opened:
            return False, None
        frame = self._bg.copy()
        t = time.time() - self._t0
        ang = self._dir * 2 * np.pi * (t / self._period)
        r1 = TRACK_R + DOT_R_OFFSET
        x = int(TRACK_CX + r1 * np.cos(ang))
        y = int(TRACK_CY + r1 * np.sin(ang))
        cv2.circle(frame, (x, y), DOT_R, DOT_BGR, -1)       # filled cyan dot
        r2 = TRACK_R + DOT2_R_OFFSET
        x2 = int(TRACK_CX + r2 * np.cos(ang + DOT2_PHASE))
        y2 = int(TRACK_CY + r2 * np.sin(ang + DOT2_PHASE))
        cv2.circle(frame, (x2, y2), DOT_R, DOT2_BGR, -1)    # filled magenta dot (180° offset)
        time.sleep(max(0.0, 1.0 / FPS - 0.001))              # cap loop to simulated FPS
        return True, frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(WIDTH)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(HEIGHT)
        if prop == cv2.CAP_PROP_FPS:
            return float(FPS)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True  # ignored: simulation has a fixed resolution

    def release(self) -> None:
        self._opened = False
