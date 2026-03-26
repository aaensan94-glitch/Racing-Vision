from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

from geometry import crossed_line

Point = Tuple[float, float]

@dataclass
class LapTimer:
    """Line-crossing lap timer with debounce."""
    start_line_a: Optional[Point] = None
    start_line_b: Optional[Point] = None
    min_lap_time_s: float = 2.0  # debounce
    laps: List[float] = field(default_factory=list)

    _t_last_cross: float = 0.0
    _t_lap_start: float = 0.0
    _prev_pos: Optional[Point] = None

    def reset(self):
        self.laps.clear()
        self._t_last_cross = 0.0
        self._t_lap_start = time.time()
        self._prev_pos = None

    def set_start_line(self, a: Point, b: Point):
        self.start_line_a = a
        self.start_line_b = b
        self.reset()

    def update(self, pos: Optional[Point], t: float) -> Optional[float]:
        """
        Returns new lap time if a lap was completed, else None.
        """
        if self.start_line_a is None or self.start_line_b is None:
            self._prev_pos = pos
            return None
        if pos is None or self._prev_pos is None:
            self._prev_pos = pos
            return None

        did_cross = crossed_line(self._prev_pos, pos, self.start_line_a, self.start_line_b)
        if did_cross:
            if (t - self._t_last_cross) > self.min_lap_time_s:
                lap_time = t - self._t_lap_start
                # ignore the very first "lap" right after setting the line
                if lap_time > 0.2:
                    self.laps.append(lap_time)
                self._t_last_cross = t
                self._t_lap_start = t
                self._prev_pos = pos
                return lap_time

        self._prev_pos = pos
        return None

    def current_lap_time(self, t: float) -> float:
        return max(0.0, t - self._t_lap_start)
