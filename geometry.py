import math
from typing import Tuple

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
