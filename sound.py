"""Low-Latency Sound-Trigger: hält einen persistenten aplay-Stream offen und
schreibt bei jedem play() ein vorgerechnetes PCM-Click direkt in stdin."""
import shutil
import subprocess
from typing import Optional

import numpy as np

SAMPLE_RATE = 22050
DURATION_S = 0.06
FREQ_HZ = 1200

_proc: Optional[subprocess.Popen] = None
_click_bytes: Optional[bytes] = None


def _build_click() -> bytes:
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n) / SAMPLE_RATE
    env = np.exp(-t * 40.0)  # schneller Decay → knackig
    wave = np.sin(2 * np.pi * FREQ_HZ * t) * env
    pcm = (wave * 32767 * 0.6).astype(np.int16)
    return pcm.tobytes()


def _open() -> None:
    global _proc, _click_bytes
    if _proc is not None:
        return
    _click_bytes = _build_click()
    if shutil.which("pw-cat"):
        cmd = ["pw-cat", "--playback", "-",
               "--rate", str(SAMPLE_RATE),
               "--channels", "1",
               "--format", "s16",
               "--latency", "10ms"]
    elif shutil.which("aplay"):
        cmd = ["aplay", "-q", "-f", "S16_LE",
               "-r", str(SAMPLE_RATE), "-c", "1",
               "--buffer-size=1024"]
    else:
        return
    _proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def play() -> None:
    if _proc is None:
        _open()
    if _proc is None or _proc.stdin is None or _click_bytes is None:
        return
    try:
        _proc.stdin.write(_click_bytes)
        _proc.stdin.flush()
    except (BrokenPipeError, ValueError):
        pass
