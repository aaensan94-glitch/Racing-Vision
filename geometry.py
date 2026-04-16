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
    """n gleichmäßige Punkte auf dem Catmull-Rom Segment zwischen p1 und p2,
    mit p0/p3 als Tangenten-Stützpunkte."""
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
