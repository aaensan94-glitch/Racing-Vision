"""Low-latency sound trigger.

Keeps a persistent aplay/pw-cat stream open and writes pre-built PCM samples
directly to stdin on each play() call. Text-to-speech runs in a separate
worker thread via espeak-ng.
"""
import io
import shutil
import subprocess
import sys
import threading
import wave
from queue import Queue
from typing import Optional

import numpy as np

SAMPLE_RATE = 22050
DURATION_S = 0.06
FREQ_HZ = 1200
ALARM_DURATION_S = 0.18
ALARM_FREQ_HZ = 220  # low + square wave → harsh buzz

FINISH_DURATION_S = 1.5
FINISH_FREQ_HZ = 1600

_proc: Optional[subprocess.Popen] = None


def _build_click() -> bytes:
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n) / SAMPLE_RATE
    env = np.exp(-t * 40.0)  # fast decay → sharp click
    wave = np.sin(2 * np.pi * FREQ_HZ * t) * env
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()


def _build_triple() -> bytes:
    """Three rapid clicks for a valid start/finish gate crossing."""
    click = _build_click()
    gap = np.zeros(int(SAMPLE_RATE * 0.04), dtype=np.int16).tobytes()
    return click + gap + click + gap + click


def _build_alarm() -> bytes:
    n = int(SAMPLE_RATE * ALARM_DURATION_S)
    t = np.arange(n) / SAMPLE_RATE
    # square wave + 30 Hz tremolo → harsh buzz; fast attack, short decay
    square = np.sign(np.sin(2 * np.pi * ALARM_FREQ_HZ * t))
    tremolo = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 30.0 * t))
    env = np.minimum(1.0, t * 80.0) * np.exp(-t * 6.0)
    wave = square * tremolo * env
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()


def _build_finish() -> bytes:
    """Rising fanfare: three short tones followed by a long closing note (~1.5 s)."""
    parts = []
    for freq in (800, 1200, 1600):
        dur = 0.15
        n = int(SAMPLE_RATE * dur)
        t = np.arange(n) / SAMPLE_RATE
        env = np.minimum(1.0, t * 60.0) * np.exp(-t * 8.0)
        wave = np.sin(2 * np.pi * freq * t) * env
        parts.append((wave * 32767).astype(np.int16))
        parts.append(np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.int16))
    # long closing tone
    dur = 0.8
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    env = np.minimum(1.0, t * 30.0) * np.exp(-t * 2.0)
    wave = np.sin(2 * np.pi * 1600 * t) * env
    wave += 0.3 * np.sin(2 * np.pi * 2400 * t) * env  # slight overtone for fullness
    wave = wave / wave.max()  # normalize to prevent clipping
    parts.append((wave * 32767).astype(np.int16))
    return np.concatenate(parts).tobytes()


def _build_countdown() -> bytes:
    """Mario Kart-style countdown beep: low, short tone."""
    n = int(SAMPLE_RATE * 0.15)
    t = np.arange(n) / SAMPLE_RATE
    env = np.ones_like(t)
    env[-int(SAMPLE_RATE * 0.02):] = np.linspace(1, 0, int(SAMPLE_RATE * 0.02))
    wave = np.sin(2 * np.pi * 600 * t) * env
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()


def _build_go() -> bytes:
    """Mario Kart-style GO tone: higher and longer than the countdown beep."""
    n = int(SAMPLE_RATE * 0.35)
    t = np.arange(n) / SAMPLE_RATE
    env = np.ones_like(t)
    env[-int(SAMPLE_RATE * 0.05):] = np.linspace(1, 0, int(SAMPLE_RATE * 0.05))
    wave = np.sin(2 * np.pi * 1000 * t) * env
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()


def _build_box() -> bytes:
    """Builds a short two-tone radio-style pit-call chime."""
    parts = []
    for freq in (1050, 780):
        n = int(SAMPLE_RATE * 0.11)
        t = np.arange(n) / SAMPLE_RATE
        env = np.minimum(1.0, t * 100.0) * np.exp(-t * 5.0)
        tone = np.sin(2 * np.pi * freq * t) * env
        parts.append((tone * 22000).astype(np.int16))
        parts.append(np.zeros(int(SAMPLE_RATE * 0.055), dtype=np.int16))
    return np.concatenate(parts).tobytes()


_click_bytes = _build_click()
_triple_bytes = _build_triple()
_alarm_bytes = _build_alarm()
_finish_bytes = _build_finish()
_countdown_bytes = _build_countdown()
_go_bytes = _build_go()
_box_bytes = _build_box()


def _open() -> None:
    global _proc
    if _proc is not None or sys.platform == "win32":
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


def _play_windows_pcm(pcm: bytes) -> None:
    import winsound

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)
    winsound.PlaySound(buffer.getvalue(), winsound.SND_MEMORY)


def _pcm_worker() -> None:
    global _proc
    while True:
        buf = _pcm_queue.get()
        if buf is None:
            break
        if sys.platform == "win32":
            try:
                _play_windows_pcm(buf)
            except (OSError, RuntimeError, wave.Error):
                pass
            _pcm_queue.task_done()
            continue
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
    """Plays a single gate-crossing click."""
    _write(_click_bytes)


def play_triple() -> None:
    """Plays the triple-click sound for a valid lap completion."""
    _write(_triple_bytes)


def play_alarm() -> None:
    """Plays the wrong-direction alarm buzz."""
    _write(_alarm_bytes)


def play_finish() -> None:
    """Plays the finish fanfare."""
    _write(_finish_bytes)


def play_countdown() -> None:
    """Plays one countdown beep."""
    _write(_countdown_bytes)


def play_go() -> None:
    """Plays the GO tone at race start."""
    _write(_go_bytes)


def play_box() -> None:
    """Plays the distinctive two-tone cue for a pit stop."""
    _write(_box_bytes)


# espeak-ng mispronounces colour names that don't match English phonics rules.
_TTS_FIX = {
    "cyan": "sigh-an",
    "magenta": "ma-jenta",
}

_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS_WORDS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}


def english_number(value: int) -> str:
    """Returns an English spoken form for race lap numbers from 0 to 100."""
    if not 0 <= value <= 100:
        raise ValueError("English lap number must be between 0 and 100")
    if value < len(_NUMBER_WORDS):
        return _NUMBER_WORDS[value]
    if value == 100:
        return "one hundred"
    tens, ones = divmod(value, 10)
    return _TENS_WORDS[tens] + (f" {_NUMBER_WORDS[ones]}" if ones else "")


def _fix_pronunciation(text: str) -> str:
    for word, phonetic in _TTS_FIX.items():
        text = text.replace(word, phonetic)
    return text


_tts_cmd: Optional[str] = None
_tts_backend: Optional[str] = None
_tts_queue: Queue = Queue()
_tts_thread: Optional[threading.Thread] = None


def _tts_worker() -> None:
    while True:
        item = _tts_queue.get()
        if item is None:
            break
        text, speed = item
        if _tts_cmd is not None and _tts_backend == "espeak":
            subprocess.run(
                [_tts_cmd, "-v", "en+m3", "-p", "65", "-a", "180",
                 "-s", str(speed), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif _tts_cmd is not None and _tts_backend == "powershell":
            rate = max(-10, min(10, round((speed - 180) / 30)))
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$voices = $synth.GetInstalledVoices() | "
                "Where-Object { $_.VoiceInfo.Culture.Name -like 'en-*' -and "
                "$_.VoiceInfo.Gender -eq 'Male' }; "
                "$voice = $voices | Where-Object { "
                "$_.VoiceInfo.Name -match 'David|Mark' } | Select-Object -First 1; "
                "if (-not $voice) { $voice = $voices | Select-Object -First 1 }; "
                "if ($voice) { $synth.SelectVoice($voice.VoiceInfo.Name) }; "
                f"$synth.Rate = {rate}; "
                "$synth.Volume = 100; "
                "$text = [Console]::In.ReadToEnd(); "
                "$synth.Speak($text); $synth.Dispose()"
            )
            subprocess.run(
                [_tts_cmd, "-NoProfile", "-NonInteractive", "-Command", script],
                input=text,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        _tts_queue.task_done()


def say(text: str, priority: bool = False, speed: int = 300) -> None:
    """Queues a TTS announcement without blocking the main loop.

    Messages are spoken in order. Use priority=True to flush pending
    announcements first (for finish-line calls).

    Args:
        text: Text to speak.
        priority: If True, clears the queue before adding this message.
        speed: Speech rate in words per minute.
    """
    global _tts_cmd, _tts_backend, _tts_thread
    if _tts_cmd is None:
        _tts_cmd = shutil.which("espeak-ng") or shutil.which("espeak")
        if _tts_cmd is not None:
            _tts_backend = "espeak"
        elif sys.platform == "win32":
            _tts_cmd = shutil.which("powershell.exe") or shutil.which("powershell")
            if _tts_cmd is not None:
                _tts_backend = "powershell"
    if _tts_cmd is None:
        return
    if _tts_thread is None or not _tts_thread.is_alive():
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        _tts_thread.start()
    if priority:
        # drain the queue so the finish announcement plays immediately
        while not _tts_queue.empty():
            try:
                _tts_queue.get_nowait()
            except Exception:
                break
    _tts_queue.put((_fix_pronunciation(text), speed))


def tts_busy() -> bool:
    """Returns True if speech is in progress or announcements are queued."""
    return not _tts_queue.empty()
