"""Geometry primitives for trail and gate-crossing calculations.

Provides segment intersection, point-to-segment distance, Catmull-Rom spline
evaluation, and trail_gate_xpt() which finds the exact crossing point where a
car's smoothed trail passes through a gate segment.
"""

import math
from typing import List, Optional, Tuple

Point = Tuple[float, float]


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Euclidean distance from point p to segment ab."""
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / denom
    t = clamp(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def side_of_line(p: Point, a: Point, b: Point) -> float:
    """Sign indicates which side of directed line a->b the point lies on."""
    px, py = p
    ax, ay = a
    bx, by = b
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def segment_intersection(p1: Point, p2: Point, a: Point, b: Point) -> Optional[Point]:
    """Intersection point of segments p1-p2 and a-b, or None."""
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = b[0] - a[0], b[1] - a[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-12:
        return None
    t = ((a[0] - p1[0]) * d2y - (a[1] - p1[1]) * d2x) / denom
    u = ((a[0] - p1[0]) * d1y - (a[1] - p1[1]) * d1x) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0] + t * d1x, p1[1] + t * d1y)
    return None


def segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    """True if segment p1-p2 properly intersects segment a-b."""
    d1 = side_of_line(p1, a, b)
    d2 = side_of_line(p2, a, b)
    d3 = side_of_line(a, p1, p2)
    d4 = side_of_line(b, p1, p2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def catmull_rom(p0: Point, p1: Point, p2: Point, p3: Point,
                n: int = 10) -> List[Point]:
    """Returns n+1 evenly spaced points on the Catmull-Rom segment between p1 and p2.

    Args:
        p0: Control point before p1 (tangent support).
        p1: Start of the interpolated segment.
        p2: End of the interpolated segment.
        p3: Control point after p2 (tangent support).
        n: Number of intervals; n+1 points are returned.

    Returns:
        List of (x, y) points along the spline from p1 to p2.
    """
    pts: List[Point] = []
    for i in range(n + 1):
        t = i / n
        t2, t3 = t * t, t * t * t
        x = 0.5 * ((2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                    (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
        y = 0.5 * ((2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                    (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
        pts.append((x, y))
    return pts


def trail_gate_xpt(prev_pt: Point, gp: Point, gate,
                   trail_tail: List[Point]) -> Optional[Point]:
    """Returns the intersection of the car's path with the gate line.

    Uses a Catmull-Rom spline through the last trail points for accuracy;
    falls back to a straight segment when fewer than three trail points exist.

    Args:
        prev_pt: Previous car position.
        gp: Current car position.
        gate: Gate candidate (must have post_a and post_b attributes).
        trail_tail: Recent trail points used to build the spline.

    Returns:
        Intersection point, or None.
    """
    a, b = gate.post_a, gate.post_b
    if trail_tail is not None and len(trail_tail) >= 3:
        p0 = trail_tail[-3]
        p1 = trail_tail[-2]
        p2 = trail_tail[-1]
        p3 = (2 * gp[0] - prev_pt[0], 2 * gp[1] - prev_pt[1])
        pts = catmull_rom(p0, p1, p2, p3, n=10)
        for i in range(len(pts) - 1):
            xpt = segment_intersection(pts[i], pts[i + 1], a, b)
            if xpt is not None:
                return xpt
    # fallback: straight line
    return segment_intersection(prev_pt, gp, a, b)
