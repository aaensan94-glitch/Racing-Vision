"""Gate detection, digit classification, ordering, crossing check, and I/O.

Detects hand-drawn gates (circle posts + connecting line) via Hough transforms,
classifies the gate digit with an MNIST-style CNN, orders gates into a sequence,
checks per-frame gate crossings, and persists calibration to configs/gates.json.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from geometry import segments_intersect, point_segment_distance, catmull_rom

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

Point = Tuple[float, float]

# Hough detection parameters — tune these per camera/lighting
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

# Line endpoint tolerance as a multiple of the circle radius
ENDPOINT_TOL = 1.5


@dataclass
class GateCandidate:
    """Detected gate defined by two post circles and the connecting line.

    Attributes:
        post_a: Center of the digit (left/inner) post.
        post_b: Center of the empty (right/outer) post.
        radius_a: Radius of post_a in pixels.
        radius_b: Radius of post_b in pixels.
        line_p1: First endpoint of the detected line segment.
        line_p2: Second endpoint of the detected line segment.
        digit: Classified gate number (-1 if unclassified).
        digit_confidence: Softmax confidence of the digit prediction.
        digit_side: Which post carries the digit ("above", "below", or "").
        forward: Unit vector pointing in the forward (legal) crossing direction.
    """

    post_a: Point
    post_b: Point
    radius_a: float
    radius_b: float
    line_p1: Point
    line_p2: Point
    digit: int = -1
    digit_confidence: float = 0.0
    digit_side: str = ""  # "above" | "below" | ""
    forward: Point = (0.0, 0.0)  # canonical forward direction from gate rotation


def detect_circles(gray: np.ndarray) -> np.ndarray:
    """Detects circular gate posts in a grayscale image.

    Args:
        gray: Grayscale input image.

    Returns:
        Array of shape (N, 3) with columns (cx, cy, radius),
        or an empty array if no circles are found.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)  # smooth to suppress noise before Hough
    circles = cv2.HoughCircles(                      # gradient-based circle detection
        blurred, cv2.HOUGH_GRADIENT,
        dp=CIRCLE_DP, minDist=CIRCLE_MIN_DIST,
        param1=CIRCLE_PARAM1, param2=CIRCLE_PARAM2,
        minRadius=CIRCLE_MIN_R, maxRadius=CIRCLE_MAX_R,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    return circles[0]


def detect_lines(gray: np.ndarray) -> np.ndarray:
    """Detects line segments connecting gate posts.

    Args:
        gray: Grayscale input image.

    Returns:
        Array of shape (N, 4) with columns (x1, y1, x2, y2),
        or an empty array if no lines are found.
    """
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)  # edge map for Hough line input
    lines = cv2.HoughLinesP(                          # probabilistic Hough line segments
        edges, 1, np.pi / 180,
        threshold=LINE_THRESHOLD,
        minLineLength=LINE_MIN_LEN,
        maxLineGap=LINE_MAX_GAP,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines[:, 0, :]


def pair_gates(circles: np.ndarray, lines: np.ndarray) -> List[GateCandidate]:
    """Pairs detected lines with circle pairs to form gate candidates.

    A line is accepted as a gate if both endpoints lie within the tolerance
    radius of two different circles.

    Args:
        circles: Output of detect_circles — shape (N, 3).
        lines: Output of detect_lines — shape (M, 4).

    Returns:
        List of GateCandidate objects, one per unique circle pair.
    """
    gates: List[GateCandidate] = []
    used_pairs = set()

    for (x1, y1, x2, y2) in lines:
        best1 = None  # (circle_index, distance)
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


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # adaptive histogram equalizer


def detect_gates(frame_bgr: np.ndarray, use_clahe: bool = False):
    """Detects all gates in a frame and canonicalizes post order.

    post_a is normalized to the darker (digit) post so downstream code can
    assume a consistent left/right orientation.

    Args:
        frame_bgr: Full BGR camera frame.
        use_clahe: If True, applies CLAHE contrast enhancement before detection.

    Returns:
        Tuple (gates, circles, lines) where circles and lines are all raw
        candidates, useful for debug visualization.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # convert to grayscale for Hough
    if use_clahe:
        gray = _clahe.apply(gray)  # enhance local contrast before circle/line detection
    circles = detect_circles(gray)
    lines = detect_lines(gray)
    gates = pair_gates(circles, lines)
    for g in gates:
        ma = _circle_fill_mean(gray, g.post_a[0], g.post_a[1], g.radius_a)
        mb = _circle_fill_mean(gray, g.post_b[0], g.post_b[1], g.radius_b)
        if ma > mb:  # post_a is brighter (empty) → swap so digit post is always post_a
            g.post_a, g.post_b = g.post_b, g.post_a
            g.radius_a, g.radius_b = g.radius_b, g.radius_a
    return gates, circles, lines


def _circle_fill_mean(gray: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Returns the mean gray intensity inside a circle, shrunk to avoid the edge.

    Bright = outline/empty post, dark = filled/digit post.

    Args:
        gray: Grayscale image.
        cx: Circle center x.
        cy: Circle center y.
        r: Circle radius.

    Returns:
        Mean pixel intensity in [0, 255].
    """
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
    """Returns posts ordered so the digit post is on the left in track direction.

    The brighter (empty) post should be on the right relative to the forward
    direction, making the digit post always the left (post_a) post.

    Args:
        gray: Grayscale image.
        gate: Gate candidate to orient.

    Returns:
        Tuple (left_post, right_post, left_r, right_r).
    """
    ma = _circle_fill_mean(gray, gate.post_a[0], gate.post_a[1], gate.radius_a)
    mb = _circle_fill_mean(gray, gate.post_b[0], gate.post_b[1], gate.radius_b)
    if ma > mb:
        # post_a is brighter (empty) → it must be on the right → swap
        return (gate.post_b, gate.post_a, gate.radius_b, gate.radius_a)
    return (gate.post_a, gate.post_b, gate.radius_a, gate.radius_b)


def _rotate_for_gate(frame_bgr: np.ndarray, gate: GateCandidate):
    """Rotates the frame so the gate's post line is horizontal.

    The empty (brighter) post lands on the right after rotation, giving a
    canonical orientation for digit extraction.

    Args:
        frame_bgr: Full BGR frame.
        gate: Gate candidate.

    Returns:
        Tuple (rotated, mx, my, L, ra, rb, forward) where mx/my are the gate
        midpoint, L is the post separation, ra/rb are post radii, and forward
        is the unit vector pointing in the legal crossing direction.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # grayscale for brightness comparison
    (ax, ay), (bx, by), ra, rb = _orient_posts(gray, gate)
    dx, dy = bx - ax, by - ay
    L = float(np.hypot(dx, dy))
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    # forward direction: perpendicular to the post line, pointing "upward" in the rotated frame
    forward = (dy / L, -dx / L) if L > 1e-6 else (0.0, 0.0)
    h, w = frame_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((mx, my), angle_deg, 1.0)       # rotation matrix around gate midpoint
    rotated = cv2.warpAffine(frame_bgr, M, (w, h),              # rotate image to align gate horizontally
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, mx, my, L, float(ra), float(rb), forward


def _roi_bounds(mx: float, my: float, L: float, ra: float, rb: float,
                height_factor: float, w: int, h: int):
    """Computes crop bounds for the region above and below the gate line.

    Args:
        mx: Gate midpoint x.
        my: Gate midpoint y.
        L: Post separation distance.
        ra: Left post radius.
        rb: Right post radius.
        height_factor: Crop height relative to L.
        w: Image width.
        h: Image height.

    Returns:
        Tuple (x0, x1, y_up0, y_up1, y_dn0, y_dn1).
    """
    # horizontal: include outer post edges
    x0 = max(0, int(round(mx - L / 2 - ra)))
    x1 = min(w, int(round(mx + L / 2 + rb)))
    crop_h = max(1, int(L * height_factor))
    cy = int(round(my))
    y_up0 = max(0, cy - crop_h); y_up1 = cy
    y_dn0 = cy; y_dn1 = min(h, cy + crop_h)
    return x0, x1, y_up0, y_up1, y_dn0, y_dn1


def extract_gate_crops(frame_bgr: np.ndarray, gate: GateCandidate,
                       height_factor: float = 1.0):
    """Extracts the region above and below the gate line.

    The below crop is returned rotated 180° so it reads in the same
    orientation as the above crop.

    Args:
        frame_bgr: Full BGR frame.
        gate: Gate candidate.
        height_factor: Crop height as a fraction of the post separation.

    Returns:
        Tuple (above, below) as BGR arrays, or (None, None) if gate is too small.
    """
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
        below = cv2.rotate(below, cv2.ROTATE_180)  # flip so digit reads in the same direction
    return above, below


def extract_digit_crop(frame_bgr: np.ndarray, gate: GateCandidate):
    """Extracts a tight square crop around the digit inside the left post.

    Uses the inscribed square of the post circle (half-side = r/√2) so the
    circle outline itself is outside the crop.

    Args:
        frame_bgr: Full BGR frame.
        gate: Gate candidate.

    Returns:
        BGR crop array, or None if the gate is too small.
    """
    if np.hypot(gate.post_b[0] - gate.post_a[0],
                gate.post_b[1] - gate.post_a[1]) < 4.0:
        return None
    rotated, mx, my, L, ra, rb, _fwd = _rotate_for_gate(frame_bgr, gate)
    h, w = rotated.shape[:2]
    cx_l = mx - L / 2.0
    cy = my
    half = ra / float(np.sqrt(2.0)) * 0.9  # 90% of inscribed square half-side
    x0 = max(0, int(round(cx_l - half)))
    x1 = min(w, int(round(cx_l + half)))
    y0 = max(0, int(round(cy - half)))
    y1 = min(h, int(round(cy + half)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return rotated[y0:y1, x0:x1].copy()


def extract_gate_overview(frame_bgr: np.ndarray, gate: GateCandidate,
                          pad_factor: float = 0.2) -> np.ndarray:
    """Returns a rotated overview crop showing both posts, the line, and digit regions.

    Draws gate geometry as an overlay directly on the crop.

    Args:
        frame_bgr: Full BGR frame.
        gate: Gate candidate.
        pad_factor: Extra padding around posts as a fraction of post separation.

    Returns:
        BGR overview image.
    """
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

    # overlay in crop coordinates
    ax_c = int(mx - L / 2) - x0
    bx_c = int(mx + L / 2) - x0
    y_c = int(my) - y0
    # traversable segment: only between the inner post edges
    cv2.line(crop, (ax_c + int(ra), y_c), (bx_c - int(rb), y_c),
             (0, 255, 0), 1)                                         # driveable gap in green
    cv2.circle(crop, (ax_c, y_c), int(ra), (0, 255, 255), 1)        # left post circle
    cv2.circle(crop, (bx_c, y_c), int(rb), (0, 255, 255), 1)        # right post circle

    # OCR region: inscribed square in the left post circle
    half = ra / float(np.sqrt(2.0)) * 0.9
    cx_l = mx - L / 2.0
    ocr_x0 = int(round(cx_l - half)) - x0
    ocr_x1 = int(round(cx_l + half)) - x0
    ocr_y0 = int(round(my - half)) - y0
    ocr_y1 = int(round(my + half)) - y0
    cv2.rectangle(crop, (ocr_x0, ocr_y0), (ocr_x1, ocr_y1), (255, 0, 255), 1)  # OCR box
    return crop


def build_crops_panel(frame_bgr: np.ndarray, gates: List[GateCandidate],
                      tile: int = 96) -> np.ndarray:
    """Builds a debug panel showing one row per gate.

    Each row contains a rotated overview with overlay and the preprocessed
    OCR input image as sent to the classifier.

    Args:
        frame_bgr: Full BGR frame.
        gates: List of gate candidates to display.
        tile: Tile height in pixels.

    Returns:
        Stacked BGR panel image.
    """
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
            ov = cv2.resize(overview, (new_w, new_h), interpolation=cv2.INTER_AREA)  # scale to tile
            ov_tile = np.zeros((tile, overview_w, 3), dtype=np.uint8)
            y_off = (tile - new_h) // 2
            x_off = (overview_w - new_w) // 2
            ov_tile[y_off:y_off + new_h, x_off:x_off + new_w] = ov
        else:
            ov_tile = np.zeros((tile, overview_w, 3), dtype=np.uint8)
        title = (f"#{g.digit} ({g.digit_confidence:.2f},{g.digit_side})"
                 if g.digit >= 0 else f"G{i} ?")
        cv2.putText(ov_tile, title, (4, 14),   # gate label in top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # OCR input: dynamic ROI via Otsu + largest connected component (matches classifier input)
        raw = extract_digit_crop(frame_bgr, g)
        if raw is None:
            ocr_img = None
        else:
            from digits import preprocess_canvas
            canvas = preprocess_canvas(raw)
            ocr_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)  # grayscale to BGR for stacking
        if ocr_img is None or ocr_img.size == 0:
            ocr_tile = np.zeros((tile, ocr_w, 3), dtype=np.uint8)
        else:
            oh, ow = ocr_img.shape[:2]
            scale = min(ocr_w / ow, tile / oh)
            new_w = max(1, int(ow * scale))
            new_h = max(1, int(oh * scale))
            resized = cv2.resize(ocr_img, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)  # scale to tile
            ocr_tile = np.zeros((tile, ocr_w, 3), dtype=np.uint8)
            y_off = (tile - new_h) // 2
            x_off = (ocr_w - new_w) // 2
            ocr_tile[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        cv2.putText(ocr_tile, "ocr", (4, 14),  # label OCR column
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        rows.append(np.hstack([ov_tile, ocr_tile]))
    return np.vstack(rows)


def classify_gate_digit(classifier, frame_bgr: np.ndarray,
                        gate: GateCandidate):
    """Classifies the digit in a gate's left post and sets the forward vector.

    Mutates the gate in place.

    Args:
        classifier: DigitClassifier instance.
        frame_bgr: Full BGR frame.
        gate: Gate candidate to update.

    Returns:
        Tuple (digit, confidence, digit_side).
    """
    # derive the forward vector from the stable gate rotation, independent of
    # the brightness-based canonicalization in detect_gates
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
    """Sorts gates by classified digit and reports ordering issues.

    Args:
        gates: List of gate candidates with digit set.

    Returns:
        Tuple (sorted_gates, warnings) where warnings lists duplicate or
        missing gate IDs.
    """
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
    """Tests whether segment a→b crosses the gate line.

    Args:
        a: Segment start.
        b: Segment end.
        gate: Gate to test against.

    Returns:
        0 = no crossing, +1 = forward direction, -1 = wrong direction.
    """
    if not segments_intersect(a, b, gate.post_a, gate.post_b):
        return 0
    if point_segment_distance(gate.post_a, a, b) < gate.radius_a:
        return 0
    if point_segment_distance(gate.post_b, a, b) < gate.radius_b:
        return 0
    # use stable forward vector from gate rotation; fall back to post_a→post_b normal
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
    """Checks whether the car path crosses a gate between two positions.

    If the trail has at least 3 prior points, a Catmull-Rom spline is used
    to interpolate the path; otherwise falls back to a straight segment.

    Args:
        prev: Previous car position.
        curr: Current car position.
        gate: Gate to test.
        trail: Recent position history (up to last 3 points before prev).
        spline_n: Number of spline sub-segments for the crossing check.

    Returns:
        0 = no crossing, +1 = forward, -1 = wrong direction.
    """
    if trail is not None and len(trail) >= 3:
        # Catmull-Rom needs 4 points: p0, p1 (≈ prev), p2 (≈ curr), p3
        p0 = trail[-3]
        p1 = trail[-2]
        p2 = trail[-1]
        # extrapolate p3 as tangent support beyond curr
        p3 = (2 * curr[0] - prev[0], 2 * curr[1] - prev[1])
        pts = catmull_rom(p0, p1, p2, p3, n=spline_n)
        for i in range(len(pts) - 1):
            result = _check_segment(pts[i], pts[i + 1], gate)
            if result != 0:
                return result
        return 0
    return _check_segment(prev, curr, gate)


def draw_gates(img: np.ndarray, gates: List[GateCandidate]):
    """Draws all gates onto the overlay image with direction arrows and digit labels.

    Args:
        img: BGR image to draw on (modified in place).
        gates: List of gate candidates to render.
    """
    for i, g in enumerate(gates):
        ax, ay = int(g.post_a[0]), int(g.post_a[1])
        bx, by = int(g.post_b[0]), int(g.post_b[1])
        cv2.circle(img, (ax, ay), int(g.radius_a), (0, 255, 255), 2)  # digit post circle
        cv2.circle(img, (bx, by), int(g.radius_b), (0, 255, 255), 2)  # empty post circle
        dx, dy = bx - ax, by - ay
        L = float(np.hypot(dx, dy))
        mx, my = (ax + bx) // 2, (ay + by) // 2
        if L > g.radius_a + g.radius_b:
            ux, uy = dx / L, dy / L
            sx, sy = ax + ux * g.radius_a, ay + uy * g.radius_a
            ex, ey = bx - ux * g.radius_b, by - uy * g.radius_b
            cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)),
                     (0, 255, 255), 2)          # traversable segment between post edges
            # forward direction from stable gate rotation; fall back to post normal
            fx, fy = g.forward
            if abs(fx) < 1e-6 and abs(fy) < 1e-6:
                fx, fy = uy, -ux
            arrow_len = 0.6 * L
            tail_x = int(mx - fx * arrow_len / 2)
            tail_y = int(my - fy * arrow_len / 2)
            head_x = int(mx + fx * arrow_len / 2)
            head_y = int(my + fy * arrow_len / 2)
            cv2.arrowedLine(img, (tail_x, tail_y), (head_x, head_y),
                            (0, 255, 255), 2, tipLength=0.3)  # direction arrow
        if g.digit >= 0:
            text = str(g.digit)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.6, g.radius_b / 20.0)
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            tx = bx - tw // 2
            ty = by + th // 2
            cv2.putText(img, text, (tx, ty), font, scale,
                        (0, 0, 0), thick, cv2.LINE_AA)  # digit label on the empty post
        else:
            cv2.putText(img, f"G{i}", (mx + 5, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)  # fallback index label when digit is unknown


def scale_gates_and_roi(gates: List["GateCandidate"],
                        roi_pts: List[Tuple[int, int]],
                        sx: float, sy: float) -> List[Tuple[int, int]]:
    """Scales gates in place and returns rescaled ROI points for a resolution switch.

    Args:
        gates: Gate candidates to rescale in place.
        roi_pts: ROI polygon points to rescale (not mutated; new list returned).
        sx: Horizontal scale factor (new_width / old_width).
        sy: Vertical scale factor (new_height / old_height).

    Returns:
        Rescaled ROI point list.
    """
    for g in gates:
        g.post_a = (g.post_a[0] * sx, g.post_a[1] * sy)
        g.post_b = (g.post_b[0] * sx, g.post_b[1] * sy)
        g.line_p1 = (g.line_p1[0] * sx, g.line_p1[1] * sy)
        g.line_p2 = (g.line_p2[0] * sx, g.line_p2[1] * sy)
        g.radius_a = g.radius_a * (sx + sy) * 0.5
        g.radius_b = g.radius_b * (sx + sy) * 0.5
    return [(int(p[0] * sx), int(p[1] * sy)) for p in roi_pts]


def _source_suffix(source) -> str:
    return "_sim" if source == "sim" else ""


def gates_path(source) -> str:
    """Returns the config file path for the gates of the given source."""
    return os.path.join(_CFG_DIR, f"gates{_source_suffix(source)}.json")


def load_gates(
    source,
) -> Tuple[List[GateCandidate], Optional[Tuple[int, int]]]:
    """Returns saved gates and the resolution at save time (used to rescale).

    Args:
        source: Camera index or "sim".

    Returns:
        Tuple (gates, resolution) where resolution may be None for legacy files.
    """
    try:
        with open(gates_path(source)) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], None
    # legacy format: bare list. Current format: dict with resolution + gates.
    if isinstance(raw, list):
        data = raw
        res = None
    else:
        data = raw.get("gates", [])
        r = raw.get("resolution")
        res = (int(r[0]), int(r[1])) if r and len(r) == 2 else None
    gates = [GateCandidate(
        post_a=tuple(g["post_a"]),
        post_b=tuple(g["post_b"]),
        radius_a=float(g["radius_a"]),
        radius_b=float(g["radius_b"]),
        line_p1=tuple(g["line_p1"]),
        line_p2=tuple(g["line_p2"]),
        digit=int(g.get("digit", -1)),
        digit_confidence=float(g.get("digit_confidence", 0.0)),
        digit_side=g.get("digit_side", ""),
        forward=tuple(g.get("forward", [0.0, 0.0])),
    ) for g in data]
    return gates, res


def save_gates(gates: List[GateCandidate], source,
               resolution: Optional[Tuple[int, int]] = None) -> None:
    """Persists gates and the capture resolution to the config file.

    Args:
        gates: Gate candidates to save.
        source: Camera index or "sim"; determines the file name.
        resolution: (width, height) at the time of detection, or None.
    """
    data = [{
        "post_a": list(g.post_a),
        "post_b": list(g.post_b),
        "radius_a": g.radius_a,
        "radius_b": g.radius_b,
        "line_p1": list(g.line_p1),
        "line_p2": list(g.line_p2),
        "digit": g.digit,
        "digit_confidence": g.digit_confidence,
        "digit_side": g.digit_side,
        "forward": list(g.forward),
    } for g in gates]
    payload: dict = {"gates": data}
    if resolution is not None:
        payload["resolution"] = [int(resolution[0]), int(resolution[1])]
    with open(gates_path(source), "w") as f:
        json.dump(payload, f, indent=2)


def detect_and_save_gates(frame: np.ndarray, frame_proc: np.ndarray,
                          use_clahe: bool, digit_classifier,
                          lap_tracker, car_names: List[str],
                          cap, source):
    """Detects gates, classifies digits, orders them, and persists to disk.

    Args:
        frame: Full-resolution BGR frame (used for digit crop extraction).
        frame_proc: ROI-masked frame passed to the gate detector.
        use_clahe: Whether to apply CLAHE contrast enhancement during detection.
        digit_classifier: Loaded DigitClassifier, or None to trigger lazy load.
        lap_tracker: LapTracker whose gate count and state are updated in place.
        car_names: Car names whose expected-gate state is reset on reorder.
        cap: Active capture object, queried for the current resolution.
        source: Camera source identifier used for the save path.

    Returns:
        Tuple (gates, gate_circles, gate_lines, digit_classifier).
    """
    gates, gate_circles, gate_lines = detect_gates(frame_proc,
                                                   use_clahe=use_clahe)
    print(f"[gates] circles={len(gate_circles)} "
          f"lines={len(gate_lines)} gates={len(gates)}")
    if gates:
        if digit_classifier is None:
            from digits import DigitClassifier
            try:
                digit_classifier = DigitClassifier.load("models/digits.pt")
                print("[gates] loaded models/digits.pt")
            except FileNotFoundError:
                print("[gates] models/digits.pt not found — "
                      "run train_digits.py first")
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
                print(f"[gates] using {n} ordered gates, timing reset")
        panel = build_crops_panel(frame, gates)
        cv2.imshow("Gate Crops", panel)  # debug panel showing all gate crops
    cur_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    save_gates(gates, source, resolution=cur_res)
    print(f"[session] saved {len(gates)} gates → {gates_path(source)}")
    return gates, gate_circles, gate_lines, digit_classifier
