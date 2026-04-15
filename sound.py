"""Low-Latency Sound-Trigger: hält einen persistenten aplay-Stream offen und
schreibt bei jedem play() ein vorgerechnetes PCM-Sample direkt in stdin."""
import shutil
import subprocess
from typing import Optional

import numpy as np

SAMPLE_RATE = 22050
DURATION_S = 0.06
FREQ_HZ = 1200
ALARM_DURATION_S = 0.18
ALARM_FREQ_HZ = 220  # tief + Square → kratzig

_proc: Optional[subprocess.Popen] = None
_click_bytes: Optional[bytes] = None
_alarm_bytes: Optional[bytes] = None


def _build_click() -> bytes:
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n) / SAMPLE_RATE
    env = np.exp(-t * 40.0)  # schneller Decay → knackig
    wave = np.sin(2 * np.pi * FREQ_HZ * t) * env
    pcm = (wave * 32767 * 0.6).astype(np.int16)
    return pcm.tobytes()


def _build_alarm() -> bytes:
    n = int(SAMPLE_RATE * ALARM_DURATION_S)
    t = np.arange(n) / SAMPLE_RATE
    # Square + 30 Hz Tremolo → kratziger Buzz, schneller Attack, kurzer Decay
    square = np.sign(np.sin(2 * np.pi * ALARM_FREQ_HZ * t))
    tremolo = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 30.0 * t))
    env = np.minimum(1.0, t * 80.0) * np.exp(-t * 6.0)
    wave = square * tremolo * env
    pcm = (wave * 32767 * 0.5).astype(np.int16)
    return pcm.tobytes()


def _open() -> None:
    global _proc, _click_bytes, _alarm_bytes
    if _proc is not None:
        return
    _click_bytes = _build_click()
    _alarm_bytes = _build_alarm()
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


def _write(buf: Optional[bytes]) -> None:
    if _proc is None:
        _open()
    if _proc is None or _proc.stdin is None or buf is None:
        return
    try:
        _proc.stdin.write(buf)
        _proc.stdin.flush()
    except (BrokenPipeError, ValueError):
        pass


def play() -> None:
    if _proc is None:
        _open()
    _write(_click_bytes)


def play_alarm() -> None:
    if _proc is None:
        _open()
    _write(_alarm_bytes)
