import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Global tracker defaults — tuned per camera/lighting
MIN_AREA_PX = 200
MORPH_KERNEL = 5
MORPH_ITER = 2
SMOOTH_ALPHA = 0.85  # EMA weight for new position (0=frozen, 1=no smoothing)
OCCLUSION_HOLD_S = 0.2

Position = Tuple[float, float]


@dataclass
class CarConfig:
    """HSV color range and display settings for a single car.

    Attributes:
        name: Car identifier matching the key in cars.json.
        hsv_lower: Lower HSV bound for color thresholding.
        hsv_upper: Upper HSV bound for color thresholding.
    """

    name: str
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]

    def display_color_bgr(self) -> Tuple[int, int, int]:
        """Returns the BGR overlay color at the midpoint of the car's HSV range."""
        h = (self.hsv_lower[0] + self.hsv_upper[0]) // 2
        hsv = np.uint8([[[h, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]  # single-pixel HSV→BGR for display color
        return int(bgr[0]), int(bgr[1]), int(bgr[2])


class MarkerTracker:
    """HSV blob tracker for a single car.

    Tracks the largest blob within the car's HSV range, applies EMA
    smoothing, and holds the last known position during brief occlusions.

    Attributes:
        car: Car configuration with HSV bounds and name.
    """

    def __init__(self, car: CarConfig):
        self.car = car
        self.smoothed: Optional[Position] = None
        self.last_pos: Optional[Position] = None
        self.last_seen_t: float = 0.0
        self.last_hsv: Optional[Tuple[float, float, float]] = None
        self.last_contour: Optional[np.ndarray] = None

    def _mask(self, hsv_full: np.ndarray) -> np.ndarray:
        """Creates a binary mask for pixels within the car's HSV range.

        Args:
            hsv_full: Full-frame HSV image.

        Returns:
            Binary uint8 mask (255 = in range).
        """
        lower = np.array(self.car.hsv_lower, dtype=np.uint8)
        upper = np.array(self.car.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_full, lower, upper)  # 255 where pixel falls within HSV bounds

        kernel = cv2.getStructuringElement(  # elliptical structuring element for morphology
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,   # remove small noise blobs
                                iterations=MORPH_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,  # fill small holes in blob
                                iterations=MORPH_ITER)
        return mask

    def update(self, hsv_full: np.ndarray, t: float) -> Tuple[Optional[Position], np.ndarray]:
        """Updates the tracker with a new HSV frame and returns the estimated position.

        Args:
            hsv_full: Full-frame HSV image (from cv2.cvtColor BGR→HSV).
            t: Current timestamp in seconds.

        Returns:
            Tuple (position, mask) where position is (x, y) in pixel coordinates
            or None if the car is not detected and the occlusion hold has expired.
        """
        mask = self._mask(hsv_full)
        contours, _ = cv2.findContours(  # trace blob outlines in the binary mask
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)  # blob area in pixels
            if area > best_area:
                best_area = area
                best = cnt

        if best is None or best_area < MIN_AREA_PX:
            self.last_contour = None
            if self.last_pos is not None and (t - self.last_seen_t) < OCCLUSION_HOLD_S:
                return self.last_pos, mask
            return None, mask

        M = cv2.moments(best)  # image moments for centroid computation
        if abs(M["m00"]) < 1e-9:
            return None, mask

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # Mean HSV of pixels inside the blob (same HSV frame as the mask)
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [best], -1, 255, thickness=cv2.FILLED)  # fill blob interior
        mean_hsv = cv2.mean(hsv_full, mask=blob_mask)[:3]  # average HSV inside the blob
        self.last_hsv = (float(mean_hsv[0]), float(mean_hsv[1]), float(mean_hsv[2]))
        self.last_contour = best

        if self.smoothed is None:
            self.smoothed = (cx, cy)
        else:
            a = SMOOTH_ALPHA
            self.smoothed = (a * cx + (1 - a) * self.smoothed[0],
                             a * cy + (1 - a) * self.smoothed[1])

        self.last_pos = self.smoothed
        self.last_seen_t = t
        return self.smoothed, mask


class MultiTracker:
    """Runs one MarkerTracker per car defined in configs/cars.json.

    Attributes:
        trackers: Mapping from car name to its MarkerTracker.
    """

    def __init__(self, cars: Dict[str, CarConfig]):
        self.trackers: Dict[str, MarkerTracker] = {
            name: MarkerTracker(cfg) for name, cfg in cars.items()
        }

    @staticmethod
    def load(path: str) -> "MultiTracker":
        """Loads car configurations from a JSON file and creates a MultiTracker.

        Args:
            path: Path to cars.json.

        Returns:
            Initialized MultiTracker.

        Raises:
            ValueError: If no cars are defined in the config file.
        """
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
        """Converts the frame to HSV and updates all car trackers.

        Args:
            frame_bgr: Full BGR camera frame.
            t: Current timestamp in seconds.

        Returns:
            Dict mapping car name to (x, y) position or None if not detected.
        """
        hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)  # convert once, shared across all trackers
        return {name: tr.update(hsv_full, t)[0] for name, tr in self.trackers.items()}

    def car(self, name: str) -> CarConfig:
        """Returns the CarConfig for the given car name."""
        return self.trackers[name].car

    def hsv(self, name: str) -> Optional[Tuple[float, float, float]]:
        """Returns the last measured mean HSV of the car's blob, or None."""
        return self.trackers[name].last_hsv

    def contour(self, name: str) -> Optional[np.ndarray]:
        """Returns the last detected contour array for the car, or None."""
        return self.trackers[name].last_contour

    def names(self):
        """Returns the list of car names in insertion order."""
        return list(self.trackers.keys())


def apply_image_adjustments(frame: np.ndarray,
                             adjust_vals: Dict[str, int]) -> np.ndarray:
    """Applies brightness, contrast, gamma, and saturation adjustments to a frame.

    All sliders are neutral at 100. Returns the original frame unchanged when
    all values are at neutral so the fast path avoids unnecessary copies.

    Args:
        frame: BGR input frame.
        adjust_vals: Dict with keys Brightness, Contrast, Gamma, Saturation
            (integer slider values; 100 = neutral).

    Returns:
        Adjusted BGR frame (may be a new array or the original).
    """
    b = adjust_vals["Brightness"]
    c = adjust_vals["Contrast"]
    gm = adjust_vals["Gamma"]
    sa = adjust_vals["Saturation"]
    if (b, c, gm, sa) == (100, 100, 100, 100):
        return frame
    frame = cv2.convertScaleAbs(frame, alpha=c / 100.0,
                                beta=float(b - 100))  # brightness/contrast
    gamma = max(0.1, gm / 100.0)
    lut = np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255,
                  0, 255).astype(np.uint8)
    frame = cv2.LUT(frame, lut)  # apply gamma correction via lookup table
    if sa != 100:
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv_img[..., 1] = np.clip(hsv_img[..., 1] * sa / 100, 0, 255)
        frame = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return frame


def apply_roi_mask(frame: np.ndarray,
                   roi_pts: list) -> np.ndarray:
    """Masks the frame to the ROI polygon; returns the original if no ROI set.

    Args:
        frame: BGR input frame.
        roi_pts: List of four (x, y) polygon corners, or fewer if not yet set.

    Returns:
        Masked frame (new array) or the original frame when roi_pts has < 4 points.
    """
    if len(roi_pts) != 4:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_pts, dtype=np.int32)], 255)  # polygon ROI mask
    return cv2.bitwise_and(frame, frame, mask=mask)
