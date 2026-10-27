"""Race state management and gate-crossing logic.

Defines RaceState (trails, lap times, countdown) and the two per-frame
functions that advance it: tick_countdown() drives the traffic-light sequence,
and process_gate_crossings() detects when a car passes a gate, updates speed,
appends trail points, and triggers lap/finish announcements.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sound
from gates import GateCandidate, gate_crossed
from geometry import trail_gate_xpt
from timing import LapTracker

Point = Tuple[float, float]

GATE_DEBOUNCE_S = 0.3


@dataclass
class RaceState:
    """All mutable state for an active race session.

    Attributes:
        car_names: Ordered list of car names.
        trails: Full position history deques per car.
        lap_trail: Current-lap path lists per car.
        best_trail: Best-lap path lists per car.
        prev: Previous (t, x, y) per car, or None before first detection.
        speed: Speed in px/s per car.
        last_gate_hit: Last debounce timestamp per (car, gate index).
        last_gate_flash: (timestamp, gate index) of most recent crossing per car.
        race_active: True while a timed race is running.
        race_finished: Per-car finish flags.
        armed_before: True after the first gate-0 crossing per car.
        countdown_t0: Timestamp when the countdown started, or None.
        countdown_beeps: Number of countdown beeps already played.
        lap_tracker: LapTracker managing gate sequence and lap times.
        race_laps: Target lap count for the race.
    """

    car_names: List[str]
    trails: Dict[str, deque]
    lap_trail: Dict[str, List[Point]]
    best_trail: Dict[str, List[Point]]
    prev: Dict[str, Optional[Tuple[float, float, float]]]
    speed: Dict[str, float]
    last_gate_hit: Dict[str, Dict[int, float]]
    last_gate_flash: Dict[str, Tuple[float, int]]
    race_active: bool
    race_finished: Dict[str, bool]
    armed_before: Dict[str, bool]
    countdown_t0: Optional[float]
    countdown_beeps: int
    lap_tracker: LapTracker
    race_laps: int

    @classmethod
    def create(cls, car_names: List[str], num_gates: int = 0,
               race_laps: int = 5,
               trail_maxlen: Optional[int] = None) -> "RaceState":
        """Creates a RaceState with zeroed per-car dicts.

        Args:
            car_names: Ordered list of car names.
            num_gates: Number of ordered gates; passed to LapTracker.
            race_laps: Target laps for the race.
            trail_maxlen: Maximum trail deque length (None = unlimited).

        Returns:
            New RaceState ready for a race session.
        """
        return cls(
            car_names=car_names,
            trails={n: deque(maxlen=trail_maxlen) for n in car_names},
            lap_trail={n: [] for n in car_names},
            best_trail={n: [] for n in car_names},
            prev={n: None for n in car_names},
            speed={n: 0.0 for n in car_names},
            last_gate_hit={n: {} for n in car_names},
            last_gate_flash={n: (0.0, -1) for n in car_names},
            race_active=False,
            race_finished={n: False for n in car_names},
            armed_before={n: False for n in car_names},
            countdown_t0=None,
            countdown_beeps=0,
            lap_tracker=LapTracker(car_names, num_gates=num_gates),
            race_laps=race_laps,
        )

    def reset(self) -> None:
        """Resets per-car mutable state in place; keeps race_laps and gate config."""
        for tr in self.trails.values():
            tr.clear()
        for name in self.car_names:
            self.lap_trail[name].clear()
            self.best_trail[name].clear()
            self.prev[name] = None
            self.speed[name] = 0.0
            self.last_gate_hit[name] = {}
            self.last_gate_flash[name] = (0.0, -1)
            self.race_finished[name] = False
            self.armed_before[name] = False

    def new_race(self) -> None:
        """Resets all state for a new race; keeps gates configuration."""
        self.lap_tracker = LapTracker(self.car_names,
                                      num_gates=self.lap_tracker.num_gates)
        self.reset()
        self.race_active = False
        self.countdown_t0 = None
        self.countdown_beeps = 0


def tick_countdown(state: RaceState, t: float) -> None:
    """Advances the countdown sequence, plays sounds, and fires race start.

    Does nothing when no countdown is active. Mutates state in place.

    Args:
        state: Current race state.
        t: Current timestamp in seconds.
    """
    if state.countdown_t0 is None:
        return
    elapsed = t - state.countdown_t0
    # lit: 0s→1, 1s→2, 2s→3, 3s→GO
    lit = min(int(elapsed) + 1, 4)
    if lit <= 3 and lit > state.countdown_beeps:
        sound.play_countdown()
        state.countdown_beeps = lit
        print(f"[countdown] {4 - lit}...")
    if lit >= 4 and state.countdown_beeps < 4:
        sound.play_go()
        state.countdown_beeps = 4
        state.race_active = True
        state.lap_tracker = LapTracker(state.car_names,
                                       num_gates=state.lap_tracker.num_gates)
        state.reset()
        state.countdown_t0 = None
        print(f"[race] GO! {state.race_laps} laps")


def process_gate_crossings(
    state: RaceState,
    gates: List[GateCandidate],
    global_positions: Dict[str, Optional[Point]],
    t: float,
) -> None:
    """Checks every car against every gate and reacts to crossings.

    Updates state in place: last_gate_hit, last_gate_flash, lap_trail,
    best_trail, trails, prev, speed, race_active, race_finished, armed_before.

    Args:
        state: Current race state, mutated in place.
        gates: Active gate list.
        global_positions: Latest per-car positions from the tracker.
        t: Current timestamp in seconds.
    """
    for name in state.car_names:
        gp = global_positions[name]

        if gp is not None and state.prev[name] is not None and gates:
            prev_pt = (state.prev[name][1], state.prev[name][2])
            for gi, g in enumerate(gates):
                trail_tail = list(state.trails[name])[-4:]
                direction = gate_crossed(prev_pt, gp, g, trail=trail_tail)
                if direction != 0:
                    if (t - state.last_gate_hit[name].get(gi, 0.0)) > GATE_DEBOUNCE_S:
                        state.last_gate_hit[name][gi] = t
                        state.last_gate_flash[name] = (t, gi)
                        if direction > 0:
                            was_armed = state.armed_before[name]
                            if g.digit == 0:
                                state.armed_before[name] = True
                            ev = state.lap_tracker.on_forward_crossing(
                                name, g.digit, t)
                            if ev is not None and ev["lap_time_s"] is not None:
                                sound.play_triple()
                                xpt = trail_gate_xpt(prev_pt, gp, g, trail_tail)
                                if xpt is not None:
                                    state.lap_trail[name].append(xpt)
                                is_new_best = ev["lap_time_s"] <= (
                                    state.lap_tracker.best_lap_time(name)
                                    or float("inf"))
                                if is_new_best:
                                    state.best_trail[name] = state.lap_trail[name][:]
                                state.lap_trail[name].clear()
                                if xpt is not None:
                                    state.lap_trail[name].append(xpt)
                                lap_nr = state.lap_tracker.lap(name)
                                msg = f"{name} {lap_nr}"
                                if is_new_best and lap_nr > 1:
                                    msg += ", new best lap"
                                sound.say(msg, speed=180)
                                print(f"\n[lap] {name} lap {ev['lap']} "
                                      f"time={ev['lap_time_s']:.3f}s "
                                      f"(best="
                                      f"{state.lap_tracker.best_lap_time(name):.3f}s)")
                                tbl = state.lap_tracker.race_table(name)
                                if tbl:
                                    print(tbl)
                                    print()
                                if (state.race_active
                                        and state.lap_tracker.lap(name) >= state.race_laps
                                        and not state.race_finished[name]):
                                    state.race_finished[name] = True
                                    place = sum(state.race_finished.values())
                                    sound.play_finish()
                                    bt = state.lap_tracker.best_lap_time(name)
                                    bt_s = f"{bt:.1f} seconds" if bt else ""
                                    if place == 1:
                                        sound.say(
                                            f"{name} wins! Best lap {bt_s}",
                                            priority=True, speed=180)
                                    else:
                                        sound.say(
                                            f"{name} finishes {place}nd. "
                                            f"Best lap {bt_s}",
                                            priority=True, speed=180)
                                    print(f"\n*** {name} FINISHED "
                                          f"place {place} — "
                                          f"{state.race_laps} laps! ***")
                                    if all(state.race_finished.values()):
                                        state.race_active = False
                                        print("\n=== RACE COMPLETE ===")
                            else:
                                sound.play()
                                if ev is not None and ev["gate"] == 0:
                                    # Gate 0 but invalid lap — reset the trail.
                                    # was_armed=False means the very first start
                                    # crossing — skip the audio announcement then.
                                    xpt = trail_gate_xpt(prev_pt, gp, g, trail_tail)
                                    if xpt is not None:
                                        state.lap_trail[name].append(xpt)
                                    state.lap_trail[name].clear()
                                    if xpt is not None:
                                        state.lap_trail[name].append(xpt)
                                    if was_armed:
                                        sound.say(f"{name}, you messed up", speed=180)
                                elif ev is not None and ev["sector_s"] is not None:
                                    delta = state.lap_tracker.sector_delta(name)
                                    delta_s = f"  [{delta}]" if delta else ""
                                    print(f"[sector] {name} gate {ev['gate']} "
                                          f"sector={ev['sector_s']:.3f}s{delta_s}")
                        else:
                            sound.play_alarm()
                        did = g.digit if g.digit >= 0 else gi
                        arrow = "fwd" if direction > 0 else "WRONG"
                        print(f"[gate] {name} crossed gate #{did} "
                              f"{arrow} t={t:.2f}s")
                    break

        if gp is not None and state.prev[name] is not None:
            dt = t - state.prev[name][0]
            if dt > 1e-6:
                dx = gp[0] - state.prev[name][1]
                dy = gp[1] - state.prev[name][2]
                state.speed[name] = float((dx * dx + dy * dy) ** 0.5 / dt)
        if gp is not None:
            state.prev[name] = (t, gp[0], gp[1])

        if gp is not None:
            state.trails[name].append(gp)
            state.lap_trail[name].append(gp)
