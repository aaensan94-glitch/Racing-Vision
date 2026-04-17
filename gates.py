from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from geometry import segments_intersect, point_segment_distance, catmull_rom

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
    digit: int = -1
    digit_confidence: float = 0.0
    digit_side: str = ""  # "above" | "below" | ""
    forward: Point = (0.0, 0.0)  # Konventions-Vorwärtsrichtung (aus Gate-Rotation)


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


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def detect_gates(frame_bgr: np.ndarray, use_clahe: bool = False):
    """Return (gates, circles, lines). Circles/lines sind alle Kandidaten (für Debug).
    Gates werden so kanonisiert, dass post_a der dunklere Pfosten (mit Ziffer) ist."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        gray = _clahe.apply(gray)
    circles = detect_circles(gray)
    lines = detect_lines(gray)
    gates = pair_gates(circles, lines)
    for g in gates:
        ma = _circle_fill_mean(gray, g.post_a[0], g.post_a[1], g.radius_a)
        mb = _circle_fill_mean(gray, g.post_b[0], g.post_b[1], g.radius_b)
        if ma > mb:  # post_a heller (leer) → swap, damit Ziffer-Pfosten links/a ist
            g.post_a, g.post_b = g.post_b, g.post_a
            g.radius_a, g.radius_b = g.radius_b, g.radius_a
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
    (rotated, mx, my, L, ra, rb, forward)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    (ax, ay), (bx, by), ra, rb = _orient_posts(gray, gate)
    dx, dy = bx - ax, by - ay
    L = float(np.hypot(dx, dy))
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    # Forward-Richtung: senkrecht auf digit→leer, "von unten" im rotierten Frame
    forward = (dy / L, -dx / L) if L > 1e-6 else (0.0, 0.0)
    h, w = frame_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((mx, my), angle_deg, 1.0)
    rotated = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, mx, my, L, float(ra), float(rb), forward


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
    rotated, mx, my, L, ra, rb, _fwd = _rotate_for_gate(frame_bgr, gate)
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


def extract_digit_crop(frame_bgr: np.ndarray, gate: GateCandidate):
    """Crop über den LINKEN (dunkleren) Pfostenkreis — die Ziffer ist dort
    eingezeichnet. Benutzt das dem Kreis EINBESCHRIEBENE Quadrat
    (Halbseite = r/sqrt(2)), damit der Kreis selbst nicht im Crop landet."""
    if np.hypot(gate.post_b[0] - gate.post_a[0],
                gate.post_b[1] - gate.post_a[1]) < 4.0:
        return None
    rotated, mx, my, L, ra, rb, _fwd = _rotate_for_gate(frame_bgr, gate)
    h, w = rotated.shape[:2]
    cx_l = mx - L / 2.0
    cy = my
    half = ra / float(np.sqrt(2.0)) * 0.9
    x0 = max(0, int(round(cx_l - half)))
    x1 = min(w, int(round(cx_l + half)))
    y0 = max(0, int(round(cy - half)))
    y1 = min(h, int(round(cy + half)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return rotated[y0:y1, x0:x1].copy()


def extract_gate_overview(frame_bgr: np.ndarray, gate: GateCandidate,
                          pad_factor: float = 0.2) -> np.ndarray:
    """Rotierte Übersicht: beide Pfosten + Linie + Ziffer-Regionen sichtbar.
    Linie und Pfostenkreise werden als Overlay eingezeichnet."""
    rotated, mx, my, L, ra, rb, _fwd = _rotate_for_gate(frame_bgr, gate)
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
    # Befahrbares Segment: nur zwischen den inneren Pfostenrändern
    cv2.line(crop, (ax_c + int(ra), y_c), (bx_c - int(rb), y_c),
             (0, 255, 0), 1)
    cv2.circle(crop, (ax_c, y_c), int(ra), (0, 255, 255), 1)
    cv2.circle(crop, (bx_c, y_c), int(rb), (0, 255, 255), 1)

    # OCR-Rechteck: einbeschriebenes Quadrat im linken Pfostenkreis
    half = ra / float(np.sqrt(2.0)) * 0.9
    cx_l = mx - L / 2.0
    ocr_x0 = int(round(cx_l - half)) - x0
    ocr_x1 = int(round(cx_l + half)) - x0
    ocr_y0 = int(round(my - half)) - y0
    ocr_y1 = int(round(my + half)) - y0
    cv2.rectangle(crop, (ocr_x0, ocr_y0), (ocr_x1, ocr_y1), (255, 0, 255), 1)
    return crop


def build_crops_panel(frame_bgr: np.ndarray, gates: List[GateCandidate],
                      tile: int = 96) -> np.ndarray:
    """Panel: eine Zeile pro Gate. Spalten: Übersicht (rotiert, overlay),
    above-Crop, below-Crop."""
    if not gates:
        return np.zeros((tile, tile * 3, 3), dtype=np.uint8)
    overview_w = tile * 2
    ocr_w = tile * 2
    sorted_gates = sorted(
        gates, key=lambda g: (g.digit < 0, g.digit if g.digit >= 0 else 0))
    rows = []
    for i, g in enumerate(sorted_gates):
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
        title = (f"#{g.digit} ({g.digit_confidence:.2f},{g.digit_side})"
                 if g.digit >= 0 else f"G{i} ?")
        cv2.putText(ov_tile, title, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # OCR-Input: dynamisches ROI nach Otsu + größter Komponente (wie an den
        # Classifier geht)
        raw = extract_digit_crop(frame_bgr, g)
        if raw is None:
            ocr_img = None
        else:
            from digits import preprocess_canvas
            canvas = preprocess_canvas(raw)
            ocr_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        if ocr_img is None or ocr_img.size == 0:
            ocr_tile = np.zeros((tile, ocr_w, 3), dtype=np.uint8)
        else:
            oh, ow = ocr_img.shape[:2]
            scale = min(ocr_w / ow, tile / oh)
            new_w = max(1, int(ow * scale))
            new_h = max(1, int(oh * scale))
            resized = cv2.resize(ocr_img, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            ocr_tile = np.zeros((tile, ocr_w, 3), dtype=np.uint8)
            y_off = (tile - new_h) // 2
            x_off = (ocr_w - new_w) // 2
            ocr_tile[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        cv2.putText(ocr_tile, "ocr", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        rows.append(np.hstack([ov_tile, ocr_tile]))
    return np.vstack(rows)


def classify_gate_digit(classifier, frame_bgr: np.ndarray,
                        gate: GateCandidate):
    """Klassifiziert den Digit-Crop, setzt digit + forward-Vektor aus der
    stabilen Gate-Rotation. Mutiert das Gate in-place."""
    # Forward-Vektor aus der Rotation herleiten (stabil, unabhängig von
    # der möglicherweise instabilen Helligkeits-Kanonisierung in detect_gates)
    rotated, mx, my, L, ra, rb, fwd = _rotate_for_gate(frame_bgr, gate)
    gate.forward = fwd
    crop = extract_digit_crop(frame_bgr, gate)
    if crop is None or crop.size == 0:
        gate.digit, gate.digit_confidence, gate.digit_side = -1, 0.0, ""
        return (-1, 0.0, "")
    d, c = classifier.predict(crop)
    gate.digit, gate.digit_confidence, gate.digit_side = d, c, "left_post"
    return (d, c, "left_post")


def order_gates(gates: List[GateCandidate]) -> Tuple[List[GateCandidate], List[str]]:
    """Sortiert Gates nach erkannter Ziffer. Liefert (sorted_gates, warnings).
    Warnt bei Duplikaten oder Lücken (0..N erwartet)."""
    warnings: List[str] = []
    valid = [g for g in gates if g.digit >= 0]
    valid.sort(key=lambda g: g.digit)
    ids = [g.digit for g in valid]
    seen = set()
    for i in ids:
        if i in seen:
            warnings.append(f"Duplicate gate id {i}")
        seen.add(i)
    if ids:
        expected = set(range(max(ids) + 1))
        missing = expected - set(ids)
        if missing:
            warnings.append(f"Missing gate ids: {sorted(missing)}")
    return valid, warnings


def _check_segment(a: Point, b: Point, gate: GateCandidate) -> int:
    """Prüft ein einzelnes Segment a->b gegen ein Gate.
    0 = kein Crossing, +1 = forward, -1 = wrong direction."""
    if not segments_intersect(a, b, gate.post_a, gate.post_b):
        return 0
    if point_segment_distance(gate.post_a, a, b) < gate.radius_a:
        return 0
    if point_segment_distance(gate.post_b, a, b) < gate.radius_b:
        return 0
    # Forward aus stabiler Gate-Rotation, Fallback auf post_a→post_b Normale
    fx, fy = gate.forward
    if abs(fx) < 1e-6 and abs(fy) < 1e-6:
        dx = gate.post_b[0] - gate.post_a[0]
        dy = gate.post_b[1] - gate.post_a[1]
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-6:
            return 0
        fx, fy = dy / L, -dx / L
    vx, vy = b[0] - a[0], b[1] - a[1]
    return 1 if (vx * fx + vy * fy) >= 0 else -1


def gate_crossed(prev: Point, curr: Point, gate: GateCandidate,
                 trail: List[Point] = None, spline_n: int = 10) -> int:
    """0 = keine Überquerung, +1 = forward, -1 = wrong direction.

    Wenn trail mind. 3 Punkte hat (vor prev), wird Catmull-Rom zwischen
    prev und curr interpoliert und jedes Sub-Segment geprüft. Sonst
    Fallback auf gerade Linie prev->curr."""
    if trail is not None and len(trail) >= 3:
        # Catmull-Rom braucht 4 Punkte: p0, p1(=prev), p2(=curr), p3
        p0 = trail[-3]
        p1 = trail[-2]  # ~ prev
        p2 = trail[-1]  # ~ curr (gerade angehängt)
        # p3 extrapolieren: curr + (curr - prev) als Tangentenstütze
        p3 = (2 * curr[0] - prev[0], 2 * curr[1] - prev[1])
        pts = catmull_rom(p0, p1, p2, p3, n=spline_n)
        for i in range(len(pts) - 1):
            result = _check_segment(pts[i], pts[i + 1], gate)
            if result != 0:
                return result
        return 0
    return _check_segment(prev, curr, gate)


def draw_gates(img: np.ndarray, gates: List[GateCandidate]):
    for i, g in enumerate(gates):
        ax, ay = int(g.post_a[0]), int(g.post_a[1])
        bx, by = int(g.post_b[0]), int(g.post_b[1])
        cv2.circle(img, (ax, ay), int(g.radius_a), (0, 255, 255), 2)
        cv2.circle(img, (bx, by), int(g.radius_b), (0, 255, 255), 2)
        # Befahrbares Segment: nur zwischen den inneren Tangenten der Pfosten
        dx, dy = bx - ax, by - ay
        L = float(np.hypot(dx, dy))
        mx, my = (ax + bx) // 2, (ay + by) // 2
        if L > g.radius_a + g.radius_b:
            ux, uy = dx / L, dy / L
            sx, sy = ax + ux * g.radius_a, ay + uy * g.radius_a
            ex, ey = bx - ux * g.radius_b, by - uy * g.radius_b
            cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)),
                     (0, 255, 255), 2)
            # Fahrrichtung: aus stabiler Gate-Rotation (gesetzt in classify_gate_digit),
            # Fallback auf (uy, -ux) wenn noch nicht klassifiziert
            fx, fy = g.forward
            if abs(fx) < 1e-6 and abs(fy) < 1e-6:
                fx, fy = uy, -ux
            arrow_len = 0.6 * L
            tail_x = int(mx - fx * arrow_len / 2)
            tail_y = int(my - fy * arrow_len / 2)
            head_x = int(mx + fx * arrow_len / 2)
            head_y = int(my + fy * arrow_len / 2)
            cv2.arrowedLine(img, (tail_x, tail_y), (head_x, head_y),
                            (0, 255, 255), 2, tipLength=0.3)
        if g.digit >= 0:
            text = str(g.digit)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.6, g.radius_b / 20.0)
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            tx = bx - tw // 2
            ty = by + th // 2
            cv2.putText(img, text, (tx, ty), font, scale,
                        (0, 0, 0), thick, cv2.LINE_AA)
        else:
            cv2.putText(img, f"G{i}", (mx + 5, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)
