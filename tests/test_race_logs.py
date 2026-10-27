import csv
import os
import tempfile
import unittest

from gates import GateCandidate, meters_per_pixel_from_gate
from session import open_log, save_race_logs
from timing import LapTracker


class RaceLogTests(unittest.TestCase):
    def test_scale_uses_gate_zero_post_center_distance(self):
        gate = GateCandidate(
            post_a=(0.0, 0.0),
            post_b=(100.0, 0.0),
            radius_a=5.0,
            radius_b=5.0,
            line_p1=(0.0, 0.0),
            line_p2=(100.0, 0.0),
            digit=0,
        )

        self.assertAlmostEqual(meters_per_pixel_from_gate([gate]), 0.00075)
        gate.digit = -1
        self.assertIsNone(meters_per_pixel_from_gate([gate]))

    def test_position_log_has_both_speed_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_file, writer, path = open_log(tmp, "test")
            writer.writerow([1.0, "Max", 10, 20, 100, 0.075])
            log_file.close()

            with open(path, newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["speed_px_s"], "100")
        self.assertEqual(rows[0]["speed_m_s"], "0.075")

    def test_sector_and_completed_lap_logs_are_separate(self):
        tracker = LapTracker(["cyan"], num_gates=2)
        tracker.on_forward_crossing("cyan", 0, 10.0)
        tracker.on_forward_crossing("cyan", 1, 14.0)
        tracker.on_forward_crossing("cyan", 0, 20.0)

        with tempfile.TemporaryDirectory() as tmp:
            save_race_logs(
                tracker, tmp, "test",
                display_names={"cyan": "Max"},
            )
            with open(os.path.join(tmp, "sectors_test.csv"),
                      newline="", encoding="utf-8") as csv_file:
                sectors = list(csv.DictReader(csv_file))
            with open(os.path.join(tmp, "laps_test.csv"),
                      newline="", encoding="utf-8") as csv_file:
                laps = list(csv.DictReader(csv_file))

        self.assertEqual(len(sectors), 2)
        self.assertEqual({row["car"] for row in sectors}, {"Max"})
        self.assertEqual({row["lap"] for row in sectors}, {"1"})
        self.assertEqual(len(laps), 1)
        self.assertEqual(laps[0]["lap"], "1")
        self.assertEqual(laps[0]["lap_time_s"], "10.0")


if __name__ == "__main__":
    unittest.main()
