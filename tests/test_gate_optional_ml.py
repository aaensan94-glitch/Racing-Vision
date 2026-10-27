import unittest
from unittest.mock import patch

import gates


class _Capture:
    def get(self, _property):
        return 1280


class OptionalGateMlTests(unittest.TestCase):
    def test_missing_torch_does_not_abort_gate_detection(self):
        gate = gates.GateCandidate(
            post_a=(10.0, 10.0),
            post_b=(30.0, 10.0),
            radius_a=4.0,
            radius_b=4.0,
            line_p1=(10.0, 10.0),
            line_p2=(30.0, 10.0),
        )
        original_import = __import__

        def import_without_torch(name, *args, **kwargs):
            if name == "digits":
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(gates, "detect_gates", return_value=([gate], [], [])),
            patch.object(gates, "save_gates") as save_gates,
            patch.object(gates, "build_crops_panel") as crop_panel,
            patch("builtins.__import__", side_effect=import_without_torch),
        ):
            detected, _, _, classifier = gates.detect_and_save_gates(
                frame=None,
                frame_proc=None,
                use_clahe=False,
                digit_classifier=None,
                lap_tracker=None,
                car_names=[],
                cap=_Capture(),
                source="sim",
            )

        self.assertEqual(detected, [gate])
        self.assertIsNone(classifier)
        save_gates.assert_called_once()
        crop_panel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
