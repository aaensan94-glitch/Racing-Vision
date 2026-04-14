import json
from dataclasses import dataclass
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Global tracker defaults — tuned per camera/lighting
MIN_AREA_PX = 200
MORPH_KERNEL = 5
MORPH_ITER = 2
SMOOTH_N = 5
OCCLUSION_HOLD_S = 0.2

Position = Tuple[float, float]


@dataclass
class CarConfig:
    name: str
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]

    def display_color_bgr(self) -> Tuple[int, int, int]:
        h = (self.hsv_lower[0] + self.hsv_upper[0]) // 2
        hsv = np.uint8([[[h, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])


class MarkerTracker:
    """HSV blob tracker for a single car."""
    def __init__(self, car: CarConfig):
        self.car = car
        self.buf: deque = deque(maxlen=SMOOTH_N)
        self.last_pos: Optional[Position] = None
        self.last_seen_t: float = 0.0
        self.last_hsv: Optional[Tuple[float, float, float]] = None

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(self.car.hsv_lower, dtype=np.uint8)
        upper = np.array(self.car.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)
        return mask

    def update(self, frame_bgr: np.ndarray, t: float) -> Tuple[Optional[Position], np.ndarray]:
        mask = self._mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best = cnt

        if best is None or best_area < MIN_AREA_PX:
            if self.last_pos is not None and (t - self.last_seen_t) < OCCLUSION_HOLD_S:
                return self.last_pos, mask
            return None, mask

        M = cv2.moments(best)
        if abs(M["m00"]) < 1e-9:
            return None, mask

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # Mean HSV of pixels inside the blob (uses same HSV conversion as the mask)
        hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [best], -1, 255, thickness=cv2.FILLED)
        mean_hsv = cv2.mean(hsv_full, mask=blob_mask)[:3]
        self.last_hsv = (float(mean_hsv[0]), float(mean_hsv[1]), float(mean_hsv[2]))

        self.buf.append((cx, cy))
        x = float(np.mean([p[0] for p in self.buf]))
        y = float(np.mean([p[1] for p in self.buf]))

        self.last_pos = (x, y)
        self.last_seen_t = t
        return (x, y), mask


class MultiTracker:
    """Runs one MarkerTracker per car defined in configs/cars.json."""
    def __init__(self, cars: Dict[str, CarConfig]):
        self.trackers: Dict[str, MarkerTracker] = {
            name: MarkerTracker(cfg) for name, cfg in cars.items()
        }

    @staticmethod
    def load(path: str) -> "MultiTracker":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cars = {
            name: CarConfig(
                name=name,
                hsv_lower=tuple(entry["hsv_lower"]),
                hsv_upper=tuple(entry["hsv_upper"]),
            )
            for name, entry in raw.items()
        }
        if not cars:
            raise ValueError(f"No cars defined in {path}")
        return MultiTracker(cars)

    def update(self, frame_bgr: np.ndarray, t: float) -> Dict[str, Optional[Position]]:
        return {name: tr.update(frame_bgr, t)[0] for name, tr in self.trackers.items()}

    def car(self, name: str) -> CarConfig:
        return self.trackers[name].car

    def hsv(self, name: str) -> Optional[Tuple[float, float, float]]:
        return self.trackers[name].last_hsv

    def names(self):
        return list(self.trackers.keys())
