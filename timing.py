"""Runden- und Sektoren-Timing pro Fahrzeug.

Gate 0 = Start/Ziel. Sektoren werden zwischen aufeinanderfolgenden
vorwärtigen Gate-Überquerungen in erwarteter Reihenfolge gemessen.
Rückwärts-Crossings und Sequenz-Brüche werden ignoriert (die Runde bricht
aber ab: bis zum nächsten 0-Gate läuft kein Timing)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from tabulate import tabulate


@dataclass
class _CarState:
    expected: int = 0            # nächstes erwartetes Gate
    lap: int = 0                 # aktuelle Rundennummer (0 = noch vor erstem Ziel)
    lap_start_t: Optional[float] = None
    last_gate_t: Optional[float] = None
    last_lap_time: Optional[float] = None
    best_lap_time: Optional[float] = None
    armed: bool = False          # True nachdem Gate 0 zum ersten Mal erreicht wurde


class LapTracker:
    def __init__(self, car_names: List[str], num_gates: int = 3):
        self.num_gates = num_gates
        self._state: Dict[str, _CarState] = {n: _CarState() for n in car_names}
        self.events: List[dict] = []

    def on_forward_crossing(self, car: str, gate_digit: int, t: float) -> Optional[dict]:
        """Verarbeitet eine vorwärtige Gate-Überquerung.

        Regeln:
        - Gate 0 startet IMMER eine neue Runde. Die eben beendete Runde zählt
          nur wenn davor alle anderen Gates in Reihenfolge überfahren wurden
          (d.h. expected == 0 beim 0-Crossing).
        - Alle anderen Gates zählen nur wenn sie die erwartete Ziffer sind,
          sonst werden sie ignoriert (State unverändert, Gate darf später
          korrekt durchfahren werden).
        """
        if gate_digit < 0:
            return None
        st = self._state[car]

        if gate_digit == 0:
            lap_time: Optional[float] = None
            sector_s: Optional[float] = None
            ev_lap = st.lap  # Closing-Event behaelt den Lap der gerade endet
            if st.armed and st.expected == 0 and st.lap_start_t is not None:
                lap_time = t - st.lap_start_t
                sector_s = (t - st.last_gate_t
                            if st.last_gate_t is not None else None)
                st.last_lap_time = lap_time
                if st.best_lap_time is None or lap_time < st.best_lap_time:
                    st.best_lap_time = lap_time
                st.lap += 1  # naechste Sektoren gehoeren zur neuen Runde
            # neue Runde immer starten
            st.lap_start_t = t
            st.last_gate_t = t
            st.expected = 1 % self.num_gates
            st.armed = True
            ev = {"t": t, "car": car, "lap": ev_lap, "gate": 0,
                  "sector_s": sector_s, "lap_time_s": lap_time}
            self.events.append(ev)
            return ev

        # Non-zero Gate: nur wenn erwartet
        if gate_digit != st.expected:
            return None

        sector_s = (None if st.last_gate_t is None
                    else t - st.last_gate_t)
        st.last_gate_t = t
        st.expected = (gate_digit + 1) % self.num_gates
        ev = {"t": t, "car": car, "lap": st.lap, "gate": gate_digit,
              "sector_s": sector_s, "lap_time_s": None}
        self.events.append(ev)
        return ev

    # --- Read-only Zugriff fuer Overlay ---
    def lap(self, car: str) -> int:
        return self._state[car].lap

    def last_lap_time(self, car: str) -> Optional[float]:
        return self._state[car].last_lap_time

    def best_lap_time(self, car: str) -> Optional[float]:
        return self._state[car].best_lap_time

    def current_lap_elapsed(self, car: str, now: float) -> Optional[float]:
        st = self._state[car]
        if not st.armed or st.lap_start_t is None:
            return None
        return now - st.lap_start_t

    # --- Auswertung ---
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events,
                            columns=["t", "car", "lap", "gate",
                                     "sector_s", "lap_time_s"])

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["car", "laps", "best_lap_s", "mean_lap_s"])
        laps = df.dropna(subset=["lap_time_s"])
        if laps.empty:
            return pd.DataFrame([{"car": c, "laps": 0,
                                  "best_lap_s": None, "mean_lap_s": None}
                                 for c in df["car"].unique()])
        g = laps.groupby("car")["lap_time_s"]
        return g.agg(laps="count", best_lap_s="min", mean_lap_s="mean").reset_index()

    def sector_delta(self, car: str) -> Optional[str]:
        """Vergleicht die aktuellen Sektoren der laufenden Runde mit der
        besten Runde. Gibt einen formatierten Einzeiler zurück."""
        df = self.to_dataframe()
        if df.empty:
            return None
        cf = df[df["car"] == car]
        completed_laps = cf[cf["lap_time_s"].notna()]
        if completed_laps.empty:
            return None
        # beste Runde: Sektoren für gate > 0
        best_lap_nr = completed_laps.loc[
            completed_laps["lap_time_s"].idxmin(), "lap"]
        best = (cf[(cf["lap"] == best_lap_nr) & (cf["gate"] > 0)]
                .set_index("gate")["sector_s"])
        # aktuelle Runde = alles nach dem letzten gate-0-Event, nur gate > 0
        gate0_times = cf[cf["gate"] == 0]["t"]
        if gate0_times.empty:
            return None
        last_g0_t = gate0_times.max()
        cur = (cf[(cf["t"] >= last_g0_t) & (cf["gate"] > 0)]
               .set_index("gate")["sector_s"])
        parts = []
        cum_delta = 0.0
        for gate in sorted(cur.index):
            if gate not in best.index:
                continue
            cs, bs = cur.get(gate), best.get(gate)
            if pd.isna(cs) or pd.isna(bs):
                continue
            d = cs - bs
            cum_delta += d
            sign = "+" if d >= 0 else ""
            parts.append(f"S{int(gate)}:{sign}{d:.3f}")
        if not parts:
            return None
        sign = "+" if cum_delta >= 0 else ""
        parts.append(f"cum:{sign}{cum_delta:.3f}")
        return "  ".join(parts)

    def race_table(self, car: str) -> Optional[str]:
        """Pivot-Tabelle: Runde × Sektor + Rundenzeit. Gibt formatierten
        String zurück, oder None wenn keine Daten."""
        df = self.to_dataframe()
        if df.empty:
            return None
        cf = df[df["car"] == car].copy()
        if cf.empty:
            return None
        # nur Runden die mindestens 1 Event haben
        laps_with_sectors = cf[cf["sector_s"].notna()].copy()
        if laps_with_sectors.empty:
            return None
        # Pivot: rows=lap, cols=gate -> sector_s
        piv = laps_with_sectors.pivot_table(
            index="lap", columns="gate", values="sector_s", aggfunc="first")
        piv.columns = [f"S{int(c)}" for c in piv.columns]
        # lap_time dazuhängen (aus gate 0 Events)
        lap_times = cf[cf["lap_time_s"].notna()].set_index("lap")["lap_time_s"]
        piv["total"] = lap_times
        # delta zur besten Runde
        if not lap_times.empty:
            best = lap_times.min()
            piv["delta"] = lap_times - best
            piv["delta"] = piv["delta"].map(
                lambda x: f"+{x:.3f}" if pd.notna(x) and x > 0 else "")
        piv.index.name = "lap"
        return tabulate(piv, headers="keys", tablefmt="fancy_grid",
                        showindex=True, floatfmt=".3f",
                        missingval="--")
