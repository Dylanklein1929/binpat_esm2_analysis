# progress.py
from __future__ import annotations
from dataclasses import dataclass
import sys
import time

@dataclass
class Progress:
    total: int
    label: str
    width: int = 28
    every: int = 1          # update every N items
    stream = sys.stderr     # stderr is nice for logs
    start_t: float = time.time()
    last_print_t: float = 0.0
    min_interval_s: float = 0.2  # avoid spamming logs

    def update(self, done: int, extra: str = "") -> None:
        if self.total <= 0:
            return
        if done % self.every != 0 and done != self.total:
            return

        now = time.time()
        if (now - self.last_print_t) < self.min_interval_s and done != self.total:
            return

        frac = max(0.0, min(1.0, done / self.total))
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)

        msg = f"\r[{bar}] {done}/{self.total} {self.label}"
        if extra:
            msg += f" ({extra})"

        self.stream.write(msg)
        self.stream.flush()
        self.last_print_t = now

        if done == self.total:
            self.stream.write("\n")
            self.stream.flush()
