"""Simulierte Webcam: weisses Papier mit drei handgezeichneten Gates und
einem cyan Punkt, der auf einer Kreisbahn durch die Gates faehrt.

Duck-type-kompatibel zu cv2.VideoCapture: .read(), .get(), .set(),
.isOpened(), .release()."""
import json
import os
import time
from typing import Tuple

import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
FPS = 30

# Kreisbahn-Mittelpunkt und Radius (px)
TRACK_CX = WIDTH // 2
TRACK_CY = HEIGHT // 2
TRACK_R = 240

# Drei Gates gleichmaessig auf der Bahn (Winkel in rad)
GATE_ANGLES = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
GATE_DIGITS = (0, 2, 1)
POST_R = 28
POST_GAP = 160  # Mittelpunktsabstand der Pfosten (befahrbar ~ GAP-2R)

# Cyan-Punkt — Farbe dynamisch aus cars.json, damit Tuning nicht bricht
DOT_R = 12
DOT_PERIOD_S = 8.0  # eine Runde


def _dot_bgr_from_cars(cfg_path: str, car: str = "cyan") -> Tuple[int, int, int]:
    """Wähle eine BGR-Farbe, die sicher in der HSV-Range von `car` liegt
    (Mittelpunkt der Range). Fallback: reines Cyan."""
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        lo = cfg[car]["hsv_lower"]
        hi = cfg[car]["hsv_upper"]
        h = int((lo[0] + hi[0]) / 2)
        s = int((lo[1] + hi[1]) / 2)
        v = int((lo[2] + hi[2]) / 2)
        px = np.array([[[h, s, v]]], dtype=np.uint8)
        bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    except (OSError, KeyError, ValueError):
        return (255, 255, 0)


DOT_BGR = _dot_bgr_from_cars(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "configs", "cars.json"))


def _draw_gate(img: np.ndarray, cx: float, cy: float, tangent_angle: float,
               digit: int) -> None:
    """Zeichne ein Gate quer zur Bahn (Pfosten links/rechts der Tangente)."""
    # Pfosten-Verbindung steht senkrecht auf der Tangente -> radial
    nx, ny = -np.sin(tangent_angle), np.cos(tangent_angle)  # Normale = radial
    half = POST_GAP / 2.0
    ax = int(cx - nx * half); ay = int(cy - ny * half)
    bx = int(cx + nx * half); by = int(cy + ny * half)

    # Linker Pfosten = innen (Richtung Bahnmitte) -> mit Ziffer
    inner_dist = np.hypot(ax - TRACK_CX, ay - TRACK_CY)
    outer_dist = np.hypot(bx - TRACK_CX, by - TRACK_CY)
    if inner_dist > outer_dist:
        ax, bx = bx, ax
        ay, by = by, ay

    # Pfosten-Kreise (schwarz, Linie)
    cv2.circle(img, (ax, ay), POST_R, (40, 40, 40), 2)
    cv2.circle(img, (bx, by), POST_R, (40, 40, 40), 2)
    # Verbindungslinie nur zwischen den inneren Tangentenpunkten
    dx, dy = bx - ax, by - ay
    L = float(np.hypot(dx, dy))
    if L > 2 * POST_R:
        ux, uy = dx / L, dy / L
        sx, sy = int(ax + ux * POST_R), int(ay + uy * POST_R)
        ex, ey = int(bx - ux * POST_R), int(by - uy * POST_R)
        cv2.line(img, (sx, sy), (ex, ey), (40, 40, 40), 2)

    # Ziffer im linken Pfosten — orthogonal zur Pfostenlinie ausgerichtet
    # damit sie im rotierten Gate-Frame aufrecht ist.
    text = str(digit)
    side = int(POST_R * 2)
    canvas = np.full((side, side, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_COMPLEX  # Serif → "1" mit Fuß, näher an MNIST
    (tw, th), _ = cv2.getTextSize(text, font, 1.1, 2)
    cv2.putText(canvas, text,
                (side // 2 - tw // 2, side // 2 + th // 2),
                font, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    angle_deg = float(np.degrees(np.arctan2(by - ay, bx - ax)))
    # Pipeline rotiert spaeter um +angle_deg, also hier um -angle_deg
    # vorkompensieren, damit die Ziffer im rotierten Frame aufrecht steht.
    M = cv2.getRotationMatrix2D((side / 2, side / 2), -angle_deg, 1.0)
    rot = cv2.warpAffine(canvas, M, (side, side),
                         flags=cv2.INTER_LINEAR,
                         borderValue=(245, 245, 245))
    x0, y0 = ax - side // 2, ay - side // 2
    x1, y1 = x0 + side, y0 + side
    if 0 <= x0 and x1 <= img.shape[1] and 0 <= y0 and y1 <= img.shape[0]:
        # min: dunkler Text gewinnt gegen Papier, ohne Pfostenkreis zu loeschen
        roi = img[y0:y1, x0:x1]
        np.minimum(roi, rot, out=roi)


def _build_background() -> np.ndarray:
    bg = np.full((HEIGHT, WIDTH, 3), 245, dtype=np.uint8)  # leicht graues Papier
    # leichtes Rauschen -> Hough verhaelt sich realistischer
    noise = np.random.randint(-6, 7, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for ang, d in zip(GATE_ANGLES, GATE_DIGITS):
        cx = TRACK_CX + TRACK_R * np.cos(ang)
        cy = TRACK_CY + TRACK_R * np.sin(ang)
        _draw_gate(bg, cx, cy, ang + np.pi / 2, d)
    return bg


class SimCapture:
    def __init__(self):
        self._bg = _build_background()
        self._t0 = time.time()
        self._opened = True
        self._dir = -1  # -1 = Konventions-Vorwärts, +1 = Rückwärts (Test)

    def reverse(self) -> None:
        """Fahrrichtung umkehren ohne Phasensprung am aktuellen Punkt."""
        now = time.time()
        # ang = dir * 2π * (now - t0)/T  → dir flippen und t0 spiegeln, sodass ang gleich bleibt
        self._t0 = 2 * now - self._t0
        self._dir = -self._dir

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._opened:
            return False, None
        frame = self._bg.copy()
        t = time.time() - self._t0
        ang = self._dir * 2 * np.pi * (t / DOT_PERIOD_S)
        x = int(TRACK_CX + TRACK_R * np.cos(ang))
        y = int(TRACK_CY + TRACK_R * np.sin(ang))
        cv2.circle(frame, (x, y), DOT_R, DOT_BGR, -1)
        # an FPS koppeln, damit der Loop nicht durchrennt
        time.sleep(max(0.0, 1.0 / FPS - 0.001))
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
        return True  # ignoriert: Simulation hat feste Aufloesung

    def release(self) -> None:
        self._opened = False
