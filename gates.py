from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from geometry import segments_intersect, point_segment_distance

Point = Tuple[float, float]

# Hough-Parameter — experimentell anpassen
CIRCLE_DP = 1.2
CIRCLE_MIN_DIST = 30
CIRCLE_PARAM1 = 100
CIRCLE_PARAM2 = 30
CIRCLE_MIN_R = 8
CIRCLE_MAX_R = 60

CANNY_LOW = 50
CANNY_HIGH = 150
LINE_THRESHOLD = 40
LINE_MIN_LEN = 30
LINE_MAX_GAP = 10

# Endpunkt-Toleranz als Vielfaches des Kreisradius
ENDPOINT_TOL = 1.5


@dataclass
class GateCandidate:
    post_a: Point
    post_b: Point
    radius_a: float
    radius_b: float
    line_p1: Point
    line_p2: Point


def detect_circles(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=CIRCLE_DP, minDist=CIRCLE_MIN_DIST,
        param1=CIRCLE_PARAM1, param2=CIRCLE_PARAM2,
        minRadius=CIRCLE_MIN_R, maxRadius=CIRCLE_MAX_R,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    return circles[0]


def detect_lines(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=LINE_THRESHOLD,
        minLineLength=LINE_MIN_LEN,
        maxLineGap=LINE_MAX_GAP,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines[:, 0, :]


def pair_gates(circles: np.ndarray, lines: np.ndarray) -> List[GateCandidate]:
    """Verbinde Linien, deren Endpunkte nahe zweier unterschiedlicher Kreiszentren liegen."""
    gates: List[GateCandidate] = []
    used_pairs = set()

    for (x1, y1, x2, y2) in lines:
        best1 = None  # (idx, dist)
        best2 = None
        for ci, (cx, cy, r) in enumerate(circles):
            tol = r * ENDPOINT_TOL
            d1 = float(np.hypot(x1 - cx, y1 - cy))
            d2 = float(np.hypot(x2 - cx, y2 - cy))
            if d1 < tol and (best1 is None or d1 < best1[1]):
                best1 = (ci, d1)
            if d2 < tol and (best2 is None or d2 < best2[1]):
                best2 = (ci, d2)
        if best1 is None or best2 is None or best1[0] == best2[0]:
            continue
        pair = tuple(sorted([best1[0], best2[0]]))
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        cxa, cya, ra = circles[best1[0]]
        cxb, cyb, rb = circles[best2[0]]
        gates.append(GateCandidate(
            post_a=(float(cxa), float(cya)),
            post_b=(float(cxb), float(cyb)),
            radius_a=float(ra), radius_b=float(rb),
            line_p1=(float(x1), float(y1)),
            line_p2=(float(x2), float(y2)),
        ))
    return gates


def detect_gates(frame_bgr: np.ndarray):
    """Return (gates, circles, lines). Circles/lines sind alle Kandidaten (für Debug)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    circles = detect_circles(gray)
    lines = detect_lines(gray)
    gates = pair_gates(circles, lines)
    return gates, circles, lines


def extract_gate_crops(frame_bgr: np.ndarray, gate: GateCandidate,
                       margin_factor: float = 0.15,
                       height_factor: float = 1.0):
    """Rotiert den Frame so, dass die Gate-Linie horizontal liegt (post_a links),
    und schneidet oberhalb und unterhalb der Linie je einen Ausschnitt zwischen
    den Pfosten aus.

    Returns (above, below). `below` ist 180° gedreht, sodass eine Ziffer, die
    auf dieser Seite mit Unterkante zur Linie gezeichnet wurde, aufrecht steht.
    Die Seite mit der Ziffer wird erst im nächsten Schritt (Ziffern-Blob) gewählt.
    """
    ax, ay = gate.post_a
    bx, by = gate.post_b
    dx, dy = bx - ax, by - ay
    L = float(np.hypot(dx, dy))
    if L < 4.0:
        return None, None

    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    h, w = frame_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((mx, my), angle_deg, 1.0)
    rotated = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    margin = L * margin_factor
    crop_w = max(1, int(L - 2 * margin))
    crop_h = max(1, int(L * height_factor))
    cx = int(round(mx))
    cy = int(round(my))

    x0 = max(0, cx - crop_w // 2)
    x1 = min(w, cx + crop_w // 2)
    y_up0 = max(0, cy - crop_h)
    y_up1 = cy
    y_dn0 = cy
    y_dn1 = min(h, cy + crop_h)

    above = rotated[y_up0:y_up1, x0:x1].copy() if y_up1 > y_up0 and x1 > x0 else None
    below = rotated[y_dn0:y_dn1, x0:x1].copy() if y_dn1 > y_dn0 and x1 > x0 else None
    if below is not None and below.size > 0:
        below = cv2.rotate(below, cv2.ROTATE_180)
    return above, below


def build_crops_panel(frame_bgr: np.ndarray, gates: List[GateCandidate],
                      tile: int = 96) -> np.ndarray:
    """Panel: eine Zeile pro Gate, links=above, rechts=below."""
    if not gates:
        return np.zeros((tile, tile * 2, 3), dtype=np.uint8)
    rows = []
    for i, g in enumerate(gates):
        above, below = extract_gate_crops(frame_bgr, g)
        row_tiles = []
        for label, img in (("above", above), ("below", below)):
            if img is None or img.size == 0:
                t = np.zeros((tile, tile, 3), dtype=np.uint8)
            else:
                t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)
            row_tiles.append(t)
        row = np.hstack(row_tiles)
        cv2.putText(row, f"G{i}", (4, tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        rows.append(row)
    return np.vstack(rows)


def gate_crossed(prev: Point, curr: Point, gate: GateCandidate) -> bool:
    """True wenn Trajektorie prev->curr die Linie zwischen den Pfosten kreuzt,
    ohne einen Pfosten zu überfahren.

    Regeln:
    - Kreuzung nur gültig zwischen post_a und post_b (Segment, nicht unendliche Linie)
    - Weder prev noch curr darf innerhalb eines Pfostens liegen
    - Trajektorie darf keinem Pfosten näher kommen als dessen Radius
    """
    if not segments_intersect(prev, curr, gate.post_a, gate.post_b):
        return False
    # Pfosten-Überfahrt ausschließen: Trajektorie-Segment darf Kreise nicht berühren
    if point_segment_distance(gate.post_a, prev, curr) < gate.radius_a:
        return False
    if point_segment_distance(gate.post_b, prev, curr) < gate.radius_b:
        return False
    return True


def draw_gates(img: np.ndarray, gates: List[GateCandidate],
               circles: np.ndarray = None, lines: np.ndarray = None,
               show_candidates: bool = False):
    if show_candidates:
        if circles is not None:
            for (x, y, r) in circles:
                cv2.circle(img, (int(x), int(y)), int(r), (80, 80, 80), 1)
        if lines is not None:
            for (x1, y1, x2, y2) in lines:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                         (60, 60, 160), 1)
    for i, g in enumerate(gates):
        ax, ay = int(g.post_a[0]), int(g.post_a[1])
        bx, by = int(g.post_b[0]), int(g.post_b[1])
        cv2.circle(img, (ax, ay), int(g.radius_a), (0, 255, 255), 2)
        cv2.circle(img, (bx, by), int(g.radius_b), (0, 255, 255), 2)
        cv2.line(img, (ax, ay), (bx, by), (0, 255, 0), 2)
        mx, my = (ax + bx) // 2, (ay + by) // 2
        cv2.putText(img, f"G{i}", (mx + 5, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
