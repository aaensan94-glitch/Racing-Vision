import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sound
from app import handle_key
from race import RaceState, overtake_announcement


class AudioControlTests(unittest.TestCase):
    def test_box_sound_is_a_nonempty_pcm_signal(self):
        pcm = np.frombuffer(sound._build_box(), dtype=np.int16)

        self.assertGreater(pcm.size, 0)
        self.assertGreater(int(np.max(np.abs(pcm))), 0)

    def test_pause_plays_box_call_and_resume_announces_return(self):
        app = SimpleNamespace(paused=False)

        with patch("app.sound.play_box") as play_box, \
                patch("app.sound.say") as say:
            handle_key(ord("p"), app, None, None, None, None, [],
                       None, None, None)
            self.assertTrue(app.paused)
            play_box.assert_called_once()
            self.assertEqual(say.call_args.args[0], "Box, box!")
            self.assertEqual(say.call_args.kwargs["speed"], 270)

            handle_key(ord("p"), app, None, None, None, None, [],
                       None, None, None)
            self.assertFalse(app.paused)
            self.assertEqual(say.call_args.args[0], "Back on track! Let's go.")

    def test_windows_extended_arrow_keys_change_laps_before_start(self):
        state = RaceState.create(["car"], race_laps=10)
        app = SimpleNamespace(cap=object(), show_standings=True)

        handle_key(0x270000, app, state, None, None, None, [],
                   None, None, None)
        self.assertEqual(state.race_laps, 15)
        handle_key(0x250000, app, state, None, None, None, [],
                   None, None, None)
        self.assertEqual(state.race_laps, 10)

    def test_standings_toggle_is_independent_of_hud_toggle(self):
        state = RaceState.create(["car"])
        app = SimpleNamespace(cap=object(), show_standings=True, show_hud=True)

        handle_key(ord("l"), app, state, None, None, None, [],
                   None, None, None)

        self.assertFalse(app.show_standings)
        self.assertTrue(app.show_hud)

    def test_lap_numbers_have_english_spoken_forms(self):
        self.assertEqual(sound.english_number(1), "one")
        self.assertEqual(sound.english_number(2), "two")
        self.assertEqual(sound.english_number(21), "twenty one")
        self.assertEqual(sound.english_number(100), "one hundred")

    def test_lap_number_outside_supported_range_is_rejected(self):
        with self.assertRaises(ValueError):
            sound.english_number(101)

    def test_windows_tts_script_keeps_speech_engine_ready(self):
        script = sound._powershell_tts_script()

        self.assertIn("[Console]::In.ReadLine()", script)
        self.assertIn('pitch=\"-15%\"', script)
        self.assertIn("SpeakSsmlAsync", script)
        self.assertIn("SpeakAsyncCancelAll", script)

    def test_overtake_calls_only_play_when_lewis_or_max_passes_the_other(self):
        state = RaceState.create(
            ["lewis", "max"],
            num_gates=3,
            display_names={"lewis": "Lewis", "max": "Max"},
        )
        tracker = state.lap_tracker
        tracker.on_forward_crossing("lewis", 0, 0.0)
        tracker.on_forward_crossing("max", 0, 0.5)
        previous_order = tracker.standings_order(state.car_names)

        self.assertEqual(previous_order, ["lewis", "max"])
        self.assertIsNone(
            overtake_announcement(state, "max", previous_order))

        tracker.on_forward_crossing("max", 1, 2.5)
        self.assertEqual(
            overtake_announcement(state, "max", previous_order),
            "Max P1!",
        )

        previous_order = tracker.standings_order(state.car_names)
        self.assertIsNone(
            overtake_announcement(state, "max", previous_order))

    def test_lewis_overtake_call(self):
        state = RaceState.create(
            ["lewis", "max"],
            num_gates=3,
            display_names={"lewis": "Lewis", "max": "Max"},
        )
        tracker = state.lap_tracker
        tracker.on_forward_crossing("max", 0, 0.0)
        tracker.on_forward_crossing("lewis", 0, 0.5)
        previous_order = tracker.standings_order(state.car_names)
        self.assertEqual(previous_order, ["max", "lewis"])

        tracker.on_forward_crossing("lewis", 1, 2.5)
        self.assertEqual(
            overtake_announcement(state, "lewis", previous_order),
            "Lewis P1!",
        )


if __name__ == "__main__":
    unittest.main()
