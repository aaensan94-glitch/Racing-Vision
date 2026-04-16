"""Low-Latency Sound-Trigger: hält einen persistenten aplay-Stream offen und
schreibt bei jedem play() ein vorgerechnetes PCM-Sample direkt in stdin."""
import shutil
import subprocess
import threading
from queue import Queue
from typing import Optional

import numpy as np

SAMPLE_RATE = 22050
DURATION_S = 0.06
FREQ_HZ = 1200
ALARM_DURATION_S = 0.18
ALARM_FREQ_HZ = 220  # tief + Square → kratzig

FINISH_DURATION_S = 1.5
FINISH_FREQ_HZ = 1600

_proc: Optional[subprocess.Popen] = None


def _build_click() -> bytes:
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n) / SAMPLE_RATE
    env = np.exp(-t * 40.0)  # schneller Decay → knackig
    wave = np.sin(2 * np.pi * FREQ_HZ * t) * env
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()


def _build_triple() -> bytes:
    """3 schnelle Clicks hintereinander für Start/Ziel-Gate bei gültiger Runde."""
    click = _build_click()
    gap = np.zeros(int(SAMPLE_RATE * 0.04), dtype=np.int16).tobytes()
    return click + gap + click + gap + click


def _build_alarm() -> bytes:
    n = int(SAMPLE_RATE * ALARM_DURATION_S)
    t = np.arange(n) / SAMPLE_RATE
    # Square + 30 Hz Tremolo → kratziger Buzz, schneller Attack, kurzer Decay
    square = np.sign(np.sin(2 * np.pi * ALARM_FREQ_HZ * t))
    tremolo = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 30.0 * t))
    env = np.minimum(1.0, t * 80.0) * np.exp(-t * 6.0)
    wave = square * tremolo * env
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()


def _build_finish() -> bytes:
    """Aufsteigende Fanfare: 3 Töne + langer Schlusston (~1.5s)."""
    parts = []
    # 3 kurze aufsteigende Töne
    for freq in (800, 1200, 1600):
        dur = 0.15
        n = int(SAMPLE_RATE * dur)
        t = np.arange(n) / SAMPLE_RATE
        env = np.minimum(1.0, t * 60.0) * np.exp(-t * 8.0)
        wave = np.sin(2 * np.pi * freq * t) * env
        parts.append((wave * 32767).astype(np.int16))
        parts.append(np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.int16))
    # Langer Schlusston
    dur = 0.8
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    env = np.minimum(1.0, t * 30.0) * np.exp(-t * 2.0)
    wave = np.sin(2 * np.pi * 1600 * t) * env
    # Leichte Obertöne für Fülle
    wave += 0.3 * np.sin(2 * np.pi * 2400 * t) * env
    wave = wave / wave.max()  # normalisieren
    parts.append((wave * 32767).astype(np.int16))
    return np.concatenate(parts).tobytes()


def _build_countdown() -> bytes:
    """Mario-Kart-Style Countdown-Beep: tiefer, kurzer Ton."""
    n = int(SAMPLE_RATE * 0.15)
    t = np.arange(n) / SAMPLE_RATE
    env = np.ones_like(t)
    env[-int(SAMPLE_RATE * 0.02):] = np.linspace(1, 0, int(SAMPLE_RATE * 0.02))
    wave = np.sin(2 * np.pi * 600 * t) * env
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()


def _build_go() -> bytes:
    """Mario-Kart-Style GO: höher und länger als Countdown."""
    n = int(SAMPLE_RATE * 0.35)
    t = np.arange(n) / SAMPLE_RATE
    env = np.ones_like(t)
    env[-int(SAMPLE_RATE * 0.05):] = np.linspace(1, 0, int(SAMPLE_RATE * 0.05))
    wave = np.sin(2 * np.pi * 1000 * t) * env
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()


_click_bytes = _build_click()
_triple_bytes = _build_triple()
_alarm_bytes = _build_alarm()
_finish_bytes = _build_finish()
_countdown_bytes = _build_countdown()
_go_bytes = _build_go()


def _open() -> None:
    global _proc
    if _proc is not None:
        return
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


_pcm_queue: Queue = Queue()
_pcm_thread: Optional[threading.Thread] = None


def _pcm_worker() -> None:
    global _proc
    while True:
        buf = _pcm_queue.get()
        if buf is None:
            break
        if _proc is None:
            _open()
        if _proc is not None and _proc.poll() is not None:
            _proc = None
            _open()
        if _proc is not None and _proc.stdin is not None:
            try:
                _proc.stdin.write(buf)
                _proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                _proc = None
        _pcm_queue.task_done()


def _write(buf: Optional[bytes]) -> None:
    global _pcm_thread
    if buf is None:
        return
    if _pcm_thread is None or not _pcm_thread.is_alive():
        _pcm_thread = threading.Thread(target=_pcm_worker, daemon=True)
        _pcm_thread.start()
    _pcm_queue.put(buf)


def play() -> None:
    _write(_click_bytes)


def play_triple() -> None:
    _write(_triple_bytes)


def play_alarm() -> None:
    _write(_alarm_bytes)


def play_finish() -> None:
    _write(_finish_bytes)


def play_countdown() -> None:
    _write(_countdown_bytes)


def play_go() -> None:
    _write(_go_bytes)


_tts_cmd: Optional[str] = None
_tts_queue: Queue = Queue()
_tts_thread: Optional[threading.Thread] = None


def _tts_worker() -> None:
    while True:
        item = _tts_queue.get()
        if item is None:
            break
        text, speed = item
        if _tts_cmd is not None:
            subprocess.run(
                [_tts_cmd, "-s", str(speed), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        _tts_queue.task_done()


def say(text: str, priority: bool = False, speed: int = 300) -> None:
    """Non-blocking queued TTS. Nachrichten werden nacheinander abgespielt.
    priority=True leert die Queue vorher (für Finish-Ansagen)."""
    global _tts_cmd, _tts_thread
    if _tts_cmd is None:
        _tts_cmd = shutil.which("espeak-ng") or shutil.which("espeak")
    if _tts_cmd is None:
        return
    if _tts_thread is None or not _tts_thread.is_alive():
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        _tts_thread.start()
    if priority:
        # Queue leeren damit Finish-Ansage sofort drankommt
        while not _tts_queue.empty():
            try:
                _tts_queue.get_nowait()
            except Exception:
                break
    _tts_queue.put((text, speed))


def tts_busy() -> bool:
    """True wenn gerade gesprochen wird oder Ansagen in der Queue sind."""
    return not _tts_queue.empty()
