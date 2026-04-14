import math
from typing import List, Tuple, Optional

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
    denom = abx*abx + aby*aby
    if denom < 1e-12:
        # a and b are the same point
        return math.hypot(px - ax, py - ay)
    t = (apx*abx + apy*aby) / denom
    t = clamp(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)

def distance_to_polyline(p: Point, poly: List[Point]) -> Optional[float]:
    """Minimum distance from point p to a polyline (list of points)."""
    if poly is None or len(poly) < 2:
        return None
    best = float("inf")
    for i in range(len(poly) - 1):
        d = point_segment_distance(p, poly[i], poly[i+1])
        if d < best:
            best = d
    return best

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


def crossed_line(prev_p: Point, curr_p: Point, a: Point, b: Point) -> bool:
    """True if segment prev->curr crosses the infinite line through a-b (sign change)."""
    s1 = side_of_line(prev_p, a, b)
    s2 = side_of_line(curr_p, a, b)
    if abs(s1) < 1e-9 or abs(s2) < 1e-9:
        # On the line: treat carefully; we'll rely on debounce in timing
        return True
    return (s1 > 0) != (s2 > 0)
