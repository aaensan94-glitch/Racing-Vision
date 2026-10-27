import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hud import draw_car_markers
from race import RaceState
from vision import MultiTracker


class _Tracker:
    def __init__(self):
        self.labels = {"cyan": "Max", "magenta": "Lewis"}

    def car(self, name):
        return SimpleNamespace(display_name=self.labels[name])

    def hsv(self, _name):
        return None


class _LapTracker:
    def lap(self, _name):
        return 2

    def last_lap_time(self, name):
        return {"cyan": 8.1, "magenta": 9.25}[name]

    def best_lap_time(self, _name):
        return 8.1

    def current_lap_elapsed(self, _name, _now):
        return 3.456


class CarDisplayTests(unittest.TestCase):
    def test_car_display_names_load_from_config(self):
        tracker = MultiTracker.load("configs/cars.json")

        self.assertEqual(tracker.car("magenta").display_name, "Lewis")
        self.assertEqual(tracker.car("cyan").display_name, "Max")

    def test_race_state_keeps_display_names(self):
        state = RaceState.create(
            ["magenta", "cyan"],
            display_names={"magenta": "Lewis", "cyan": "Max"},
        )

        self.assertEqual(state.display_names, {"magenta": "Lewis", "cyan": "Max"})

    def test_hud_shows_names_and_lap_times_in_seconds(self):
        with patch("hud.cv2.putText"):
            lines = draw_car_markers(
                overlay=np.zeros((40, 40, 3), dtype=np.uint8),
                car_names=["cyan", "magenta"],
                global_positions={"cyan": None, "magenta": None},
                tracker=_Tracker(),
                speed={"cyan": 0.0, "magenta": 0.0},
                lap_tracker=_LapTracker(),
                t=10.0,
                colors={"cyan": (255, 255, 0), "magenta": (255, 0, 255)},
            )

        text = "\n".join(line for line, _ in lines)
        self.assertIn("Max", text)
        self.assertIn("Lewis", text)
        self.assertIn("current 3.46s", text)
        self.assertIn("last 8.10s", text)
        self.assertIn("best 8.10s", text)


if __name__ == "__main__":
    unittest.main()
