import json
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

@dataclass
class MarkerConfig:
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    min_area_px: int
    morph_kernel: int
    morph_iter: int
    smooth_n: int

    @staticmethod
    def load(path: str) -> "MarkerConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return MarkerConfig(
            hsv_lower=tuple(d["hsv_lower"]),
            hsv_upper=tuple(d["hsv_upper"]),
            min_area_px=int(d["min_area_px"]),
            morph_kernel=int(d["morph_kernel"]),
            morph_iter=int(d["morph_iter"]),
            smooth_n=int(d["smooth_n"]),
        )

class MarkerTracker:
    """Simple HSV blob tracker for CV1."""
    def __init__(self, cfg: MarkerConfig):
        self.cfg = cfg
        self.buf = deque(maxlen=max(1, cfg.smooth_n))
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_seen_t: float = 0.0

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(self.cfg.hsv_lower, dtype=np.uint8)
        upper = np.array(self.cfg.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        k = max(1, self.cfg.morph_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.cfg.morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.cfg.morph_iter)
        return mask

    def update(self, frame_bgr: np.ndarray, t: float) -> Tuple[Optional[Tuple[float, float]], np.ndarray]:
        """
        Returns (smoothed_position or None, mask)
        """
        mask = self._mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best = cnt

        if best is None or best_area < self.cfg.min_area_px:
            # allow short occlusions: keep last known position for 0.2s
            if self.last_pos is not None and (t - self.last_seen_t) < 0.2:
                return self.last_pos, mask
            return None, mask

        M = cv2.moments(best)
        if abs(M["m00"]) < 1e-9:
            return None, mask

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        self.buf.append((cx, cy))
        x = float(np.mean([p[0] for p in self.buf]))
        y = float(np.mean([p[1] for p in self.buf]))

        self.last_pos = (x, y)
        self.last_seen_t = t
        return (x, y), mask
