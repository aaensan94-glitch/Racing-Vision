import threading
import unittest

import numpy as np

from camera import ThreadedCapture


class ThreadedCaptureTests(unittest.TestCase):
    def test_read_keeps_last_frame_after_capture_failure(self):
        capture = ThreadedCapture.__new__(ThreadedCapture)
        capture._lock = threading.Lock()
        capture._frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture._ok = False

        ok, frame = capture.read()

        self.assertTrue(ok)
        self.assertIs(frame, capture._frame)

    def test_read_fails_when_no_frame_has_been_captured(self):
        capture = ThreadedCapture.__new__(ThreadedCapture)
        capture._lock = threading.Lock()
        capture._frame = None
        capture._ok = False

        ok, frame = capture.read()

        self.assertFalse(ok)
        self.assertIsNone(frame)


if __name__ == "__main__":
    unittest.main()
