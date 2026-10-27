import unittest

from gates import GateCandidate, gate_crossed
from geometry import trail_gate_xpt


class GateCrossingTests(unittest.TestCase):
    def setUp(self):
        self.gate = GateCandidate(
            post_a=(0.0, -10.0),
            post_b=(0.0, 10.0),
            radius_a=2.0,
            radius_b=2.0,
            line_p1=(0.0, -10.0),
            line_p2=(0.0, 10.0),
            forward=(1.0, 0.0),
        )

    def test_spline_checks_current_segment(self):
        trail = [(-3.0, 0.0), (-2.0, 0.0), (-1.0, 0.0)]

        self.assertEqual(gate_crossed((-1.0, 0.0), (1.0, 0.0),
                                      self.gate, trail=trail), 1)

    def test_spline_detects_crossing_point_on_current_segment(self):
        trail = [(-3.0, 0.0), (-2.0, 0.0), (-1.0, 0.0)]

        crossing = trail_gate_xpt((-1.0, 0.0), (1.0, 0.0),
                                  self.gate, trail)

        self.assertIsNotNone(crossing)
        self.assertAlmostEqual(crossing[0], 0.0)
        self.assertAlmostEqual(crossing[1], 0.0)

    def test_spline_does_not_reuse_an_old_crossing(self):
        trail = [(-3.0, 0.0), (-2.0, 0.0), (-1.0, 0.0)]

        self.assertEqual(gate_crossed((-1.0, 0.0), (-0.5, 0.0),
                                      self.gate, trail=trail), 0)


if __name__ == "__main__":
    unittest.main()
