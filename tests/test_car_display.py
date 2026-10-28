import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hud import draw_car_markers, draw_standings_panel
from race import RaceState
from timing import LapTracker
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


class _StandingsTracker:
    def race_progress(self, name):
        return {"cyan": (1, 1), "magenta": (1, 2)}[name]

    def best_lap_time(self, name):
        return {"cyan": 8.1, "magenta": 9.25}[name]


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

    def test_standings_panel_ranks_cars_by_sector_progress(self):
        overlay = np.zeros((240, 320, 3), dtype=np.uint8)
        hud = SimpleNamespace(
            FONT=0, scale=0.5, thickness=1, line_h=20, pad=8,
            panel_alpha=0.5, color=(255, 255, 255),
        )
        with patch("hud.cv2.putText") as put_text:
            draw_standings_panel(
                overlay,
                hud,
                ["cyan", "magenta"],
                {"cyan": "Max", "magenta": "Lewis"},
                _StandingsTracker(),
                {"cyan": (255, 255, 0), "magenta": (255, 0, 255)},
            )

        rows = [call.args[1] for call in put_text.call_args_list]
        self.assertIn(" 1  Lewis", rows[1])
        self.assertIn(" 2  Max", rows[2])

    def test_race_progress_advances_through_gates_and_laps(self):
        tracker = LapTracker(["cyan"], num_gates=3)

        self.assertEqual(tracker.race_progress("cyan"), (0, -1))
        tracker.on_forward_crossing("cyan", 0, 0.0)
        self.assertEqual(tracker.race_progress("cyan"), (0, 0))
        tracker.on_forward_crossing("cyan", 1, 1.0)
        self.assertEqual(tracker.race_progress("cyan"), (0, 1))
        tracker.on_forward_crossing("cyan", 2, 2.0)
        self.assertEqual(tracker.race_progress("cyan"), (0, 2))
        tracker.on_forward_crossing("cyan", 0, 3.0)
        self.assertEqual(tracker.race_progress("cyan"), (1, 0))


if __name__ == "__main__":
    unittest.main()
