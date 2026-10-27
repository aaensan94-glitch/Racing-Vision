"""Block-level frame-time profiler.

Perf accumulates named timing samples via the timed() context manager and the
mark() / now() helpers, then prints a per-block summary on tick() once per
second.  Used in the main loop to measure capture, filter, vision, and HUD
costs without introducing external dependencies.
"""

import time
from contextlib import contextmanager
from typing import Dict


class Perf:
    """Block-level performance timer.

    Toggle with 'd'. Prints an accumulated time breakdown per label every
    PRINT_EVERY frames.
    """

    PRINT_EVERY = 30

    def __init__(self):
        self.on = False
        self.acc: Dict[str, float] = {}
        self.n = 0

    @contextmanager
    def timed(self, label: str):
        if not self.on:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.acc[label] = self.acc.get(label, 0.0) + (time.perf_counter() - t0)

    def mark(self, label: str, dt: float) -> None:
        if self.on:
            self.acc[label] = self.acc.get(label, 0.0) + dt

    def now(self) -> float:
        return time.perf_counter() if self.on else 0.0

    def tick(self):
        if not self.on:
            return
        self.n += 1
        if self.n < self.PRINT_EVERY:
            return
        avg_ms = sorted(
            ((k, v / self.n * 1000) for k, v in self.acc.items()),
            key=lambda kv: -kv[1])
        total = sum(v for _, v in avg_ms)
        parts = "  ".join(f"{k}={v:.1f}" for k, v in avg_ms)
        print(f"[perf/{self.n}] total={total:.1f}ms  {parts}")
        self.acc.clear()
        self.n = 0
