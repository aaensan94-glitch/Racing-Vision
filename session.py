import json
import os
from typing import List, Optional, Tuple

from gates import GateCandidate

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


def _gates_path(source) -> str:
    return os.path.join(CFG_DIR, f"gates{_source_suffix(source)}.json")


def _load_session(source) -> dict:
    try:
        with open(_session_path(source)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_session(data: dict, source) -> None:
    with open(_session_path(source), "w") as f:
        json.dump(data, f, indent=2)


def _load_gates(
    source,
) -> Tuple[List[GateCandidate], Optional[Tuple[int, int]]]:
    """Returns saved gates and the resolution at save time (used to rescale).

    Args:
        source: Camera index or "sim".

    Returns:
        Tuple (gates, resolution) where resolution may be None for legacy files.
    """
    try:
        with open(_gates_path(source)) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], None
    # legacy format: bare list. Current format: dict with resolution + gates.
    if isinstance(raw, list):
        data = raw
        res = None
    else:
        data = raw.get("gates", [])
        r = raw.get("resolution")
        res = (int(r[0]), int(r[1])) if r and len(r) == 2 else None
    gates = [GateCandidate(
        post_a=tuple(g["post_a"]),
        post_b=tuple(g["post_b"]),
        radius_a=float(g["radius_a"]),
        radius_b=float(g["radius_b"]),
        line_p1=tuple(g["line_p1"]),
        line_p2=tuple(g["line_p2"]),
        digit=int(g.get("digit", -1)),
        digit_confidence=float(g.get("digit_confidence", 0.0)),
        digit_side=g.get("digit_side", ""),
        forward=tuple(g.get("forward", [0.0, 0.0])),
    ) for g in data]
    return gates, res


def _save_gates(gates: List[GateCandidate], source,
                resolution: Optional[Tuple[int, int]] = None) -> None:
    data = [{
        "post_a": list(g.post_a),
        "post_b": list(g.post_b),
        "radius_a": g.radius_a,
        "radius_b": g.radius_b,
        "line_p1": list(g.line_p1),
        "line_p2": list(g.line_p2),
        "digit": g.digit,
        "digit_confidence": g.digit_confidence,
        "digit_side": g.digit_side,
        "forward": list(g.forward),
    } for g in gates]
    payload = {"gates": data}
    if resolution is not None:
        payload["resolution"] = [int(resolution[0]), int(resolution[1])]
    with open(_gates_path(source), "w") as f:
        json.dump(payload, f, indent=2)
