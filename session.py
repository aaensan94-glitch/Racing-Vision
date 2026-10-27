"""Session and race-log persistence.

Manages per-source session JSON files (ROI, filter values, lap limit) and
writes position, gate-event, sector, lap, and summary CSVs to the logs/ directory.
"""

import csv
import json
import os
from typing import Dict, Optional

from timing import LapTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(BASE_DIR, "configs")
CARS_CFG_PATH = os.path.join(CFG_DIR, "cars.json")
HUD_CFG_PATH = os.path.join(CFG_DIR, "hud.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def _source_suffix(source) -> str:
    """Returns the config file suffix for the given source.

    Simulator and camera configs are separate — gate positions, ROI, and
    image filters are incompatible between the two sources.
    """
    return "_sim" if source == "sim" else ""


def _session_path(source) -> str:
    return os.path.join(CFG_DIR, f"session{_source_suffix(source)}.json")


def _load_session(source) -> dict:
    try:
        with open(_session_path(source)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_session(data: dict, source) -> None:
    with open(_session_path(source), "w") as f:
        json.dump(data, f, indent=2)


def open_log(logs_dir: str, ts: str):
    """Opens a position log CSV and writes the header row.

    Args:
        logs_dir: Directory for the log file.
        ts: Timestamp string used in the file name.

    Returns:
        Tuple (log_f, writer, log_path).

    The CSV includes both pixel speed and calibrated metric speed. The metric
    value is empty when no physical scale has been established.
    """
    log_path = os.path.join(logs_dir, f"log_{ts}.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["t", "car", "x", "y", "speed_px_s", "speed_m_s"])
    return log_f, writer, log_path


def save_race_logs(lap_tracker: LapTracker, logs_dir: str, ts: str,
                   tag: str = "done",
                   display_names: Optional[Dict[str, str]] = None) -> None:
    """Saves separate gate-event, sector, lap, and summary CSVs.

    Does nothing when the lap tracker has no events recorded yet.

    Args:
        lap_tracker: LapTracker instance with the current race data.
        logs_dir: Directory path where CSV files are written.
        ts: Timestamp string used in the file names.
        tag: Log prefix printed with each saved path.
        display_names: Optional mapping from internal car IDs to display names.
    """
    events_df = lap_tracker.to_dataframe()
    if events_df.empty:
        return
    ev_path = os.path.join(logs_dir, f"gate_events_{ts}.csv")
    sector_path = os.path.join(logs_dir, f"sectors_{ts}.csv")
    lap_path = os.path.join(logs_dir, f"laps_{ts}.csv")
    sum_path = os.path.join(logs_dir, f"summary_{ts}.csv")
    if display_names:
        events_df["car"] = events_df["car"].map(
            lambda car: display_names.get(car, car))
    sector_df = events_df.loc[
        events_df["sector_s"].notna(),
        ["t", "car", "lap", "gate", "sector_s"],
    ].copy()
    lap_df = events_df.loc[
        events_df["lap_time_s"].notna(),
        ["t", "car", "lap", "lap_time_s"],
    ].copy()
    sector_df["lap"] += 1
    lap_df["lap"] += 1
    summary = lap_tracker.summary()
    if display_names and not summary.empty:
        summary["car"] = summary["car"].map(
            lambda car: display_names.get(car, car))
    events_df.to_csv(ev_path, index=False)
    sector_df.to_csv(sector_path, index=False)
    lap_df.to_csv(lap_path, index=False)
    summary.to_csv(sum_path, index=False)
    print(f"[{tag}] gate events: {ev_path}")
    print(f"[{tag}] sectors: {sector_path}")
    print(f"[{tag}] laps: {lap_path}")
    print(f"[{tag}] lap summary: {sum_path}")
    print(summary.to_string(index=False))
