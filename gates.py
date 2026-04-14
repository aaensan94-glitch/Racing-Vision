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


def _circle_fill_mean(gray: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Mittlere Grauwert-Intensität im Kreisinneren (kleiner Schrumpfradius,
    um Rand zu vermeiden). Hell = outline/leer, dunkel = gefüllt."""
    h, w = gray.shape[:2]
    rr = max(2, int(r * 0.6))
    x0 = max(0, int(cx) - rr); x1 = min(w, int(cx) + rr + 1)
    y0 = max(0, int(cy) - rr); y1 = min(h, int(cy) + rr + 1)
    if x1 <= x0 or y1 <= y0:
        return 255.0
    patch = gray[y0:y1, x0:x1]
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= rr * rr
    if not mask.any():
        return float(patch.mean())
    return float(patch[mask].mean())


def _orient_posts(gray: np.ndarray, gate: GateCandidate):
    """Gibt (left_post, right_post, left_r, right_r) zurück. Der hellere
    (leere) Pfosten soll in Fahrtrichtung rechts liegen — d.h. bezüglich
    der Rotation, die post_a links, post_b rechts legt, ist der hellere
    Pfosten post_b."""
    ma = _circle_fill_mean(gray, gate.post_a[0], gate.post_a[1], gate.radius_a)
    mb = _circle_fill_mean(gray, gate.post_b[0], gate.post_b[1], gate.radius_b)
    if ma > mb:
        # post_a ist heller (leer) → muss rechts sein → swap
        return (gate.post_b, gate.post_a, gate.radius_b, gate.radius_a)
    return (gate.post_a, gate.post_b, gate.radius_a, gate.radius_b)


def _rotate_for_gate(frame_bgr: np.ndarray, gate: GateCandidate):
    """Rotiert den Frame so, dass die Pfostenverbindung horizontal liegt
    und der leere (hellere) Pfosten rechts landet. Liefert
    (rotated, mx, my, L, ra, rb)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    (ax, ay), (bx, by), ra, rb = _orient_posts(gray, gate)
    dx, dy = bx - ax, by - ay
    L = float(np.hypot(dx, dy))
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    h, w = frame_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((mx, my), angle_deg, 1.0)
    rotated = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, mx, my, L, float(ra), float(rb)


def _roi_bounds(mx: float, my: float, L: float, ra: float, rb: float,
                height_factor: float, w: int, h: int):
    """Horizontal: inkl. äußerer Pfostenränder. Vertikal: crop_h oberhalb /
    unterhalb der Linie."""
    x0 = max(0, int(round(mx - L / 2 - ra)))
    x1 = min(w, int(round(mx + L / 2 + rb)))
    crop_h = max(1, int(L * height_factor))
    cy = int(round(my))
    y_up0 = max(0, cy - crop_h); y_up1 = cy
    y_dn0 = cy; y_dn1 = min(h, cy + crop_h)
    return x0, x1, y_up0, y_up1, y_dn0, y_dn1


def extract_gate_crops(frame_bgr: np.ndarray, gate: GateCandidate,
                       height_factor: float = 1.0):
    """Oberhalb/unterhalb der Linie, horizontal über beide Pfosten hinweg.
    below ist 180° gedreht."""
    if np.hypot(gate.post_b[0] - gate.post_a[0],
                gate.post_b[1] - gate.post_a[1]) < 4.0:
        return None, None
    rotated, mx, my, L, ra, rb = _rotate_for_gate(frame_bgr, gate)
    h, w = rotated.shape[:2]
    x0, x1, y_up0, y_up1, y_dn0, y_dn1 = _roi_bounds(
        mx, my, L, ra, rb, height_factor, w, h)
    if x1 - x0 < 4:
        return None, None
    above = rotated[y_up0:y_up1, x0:x1].copy() if y_up1 > y_up0 else None
    below = rotated[y_dn0:y_dn1, x0:x1].copy() if y_dn1 > y_dn0 else None
    if below is not None and below.size > 0:
        below = cv2.rotate(below, cv2.ROTATE_180)
    return above, below


def extract_gate_overview(frame_bgr: np.ndarray, gate: GateCandidate,
                          pad_factor: float = 0.2) -> np.ndarray:
    """Rotierte Übersicht: beide Pfosten + Linie + Ziffer-Regionen sichtbar.
    Linie und Pfostenkreise werden als Overlay eingezeichnet."""
    rotated, mx, my, L, ra, rb = _rotate_for_gate(frame_bgr, gate)
    h, w = rotated.shape[:2]
    pad_x = int(L * pad_factor + max(ra, rb))
    pad_y = int(L)
    x0 = max(0, int(mx - L / 2 - pad_x))
    x1 = min(w, int(mx + L / 2 + pad_x))
    y0 = max(0, int(my - pad_y))
    y1 = min(h, int(my + pad_y))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    crop = rotated[y0:y1, x0:x1].copy()

    # Overlay in Crop-Koordinaten
    ax_c = int(mx - L / 2) - x0
    bx_c = int(mx + L / 2) - x0
    y_c = int(my) - y0
    cv2.line(crop, (ax_c, y_c), (bx_c, y_c), (0, 255, 0), 1)
    cv2.circle(crop, (ax_c, y_c), int(ra), (0, 255, 255), 1)
    cv2.circle(crop, (bx_c, y_c), int(rb), (0, 255, 255), 1)

    # ROI-Rechtecke (above/below) einzeichnen
    _, _, y_up0, y_up1, y_dn0, y_dn1 = _roi_bounds(
        mx, my, L, ra, rb, 1.0, rotated.shape[1], rotated.shape[0])
    roi_x0 = int(mx - L / 2 - ra) - x0
    roi_x1 = int(mx + L / 2 + rb) - x0
    up_y0, up_y1 = y_up0 - y0, y_up1 - y0
    dn_y0, dn_y1 = y_dn0 - y0, y_dn1 - y0
    cv2.rectangle(crop, (roi_x0, up_y0), (roi_x1, up_y1), (255, 0, 255), 1)
    cv2.rectangle(crop, (roi_x0, dn_y0), (roi_x1, dn_y1), (255, 0, 255), 1)
    return crop


def build_crops_panel(frame_bgr: np.ndarray, gates: List[GateCandidate],
                      tile: int = 96) -> np.ndarray:
    """Panel: eine Zeile pro Gate. Spalten: Übersicht (rotiert, overlay),
    above-Crop, below-Crop."""
    if not gates:
        return np.zeros((tile, tile * 3, 3), dtype=np.uint8)
    overview_w = tile * 2
    rows = []
    for i, g in enumerate(gates):
        above, below = extract_gate_crops(frame_bgr, g)
        overview = extract_gate_overview(frame_bgr, g)

        if overview.size > 0:
            oh, ow = overview.shape[:2]
            scale = min(overview_w / ow, tile / oh)
            new_w = max(1, int(ow * scale))
            new_h = max(1, int(oh * scale))
            ov = cv2.resize(overview, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ov_tile = np.zeros((tile, overview_w, 3), dtype=np.uint8)
            y_off = (tile - new_h) // 2
            x_off = (overview_w - new_w) // 2
            ov_tile[y_off:y_off + new_h, x_off:x_off + new_w] = ov
        else:
            ov_tile = np.zeros((tile, overview_w, 3), dtype=np.uint8)
        cv2.putText(ov_tile, f"G{i} rotated", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        row_tiles = [ov_tile]
        for label, img in (("above", above), ("below", below)):
            if img is None or img.size == 0:
                t = np.zeros((tile, tile, 3), dtype=np.uint8)
            else:
                t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)
            row_tiles.append(t)
        rows.append(np.hstack(row_tiles))
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
