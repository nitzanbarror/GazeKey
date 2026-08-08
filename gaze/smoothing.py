"""GazeKey — One-Euro filter + I-DT fixation detector.
VERIFIED REFERENCE IMPLEMENTATION — use as-is.
"""
from __future__ import annotations
import math
from collections import deque


class _LowPass:
    def __init__(self):
        self.y = None
    def apply(self, x: float, alpha: float) -> float:
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """Casiez et al. 2012. One instance per axis."""
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x, self._dx = _LowPass(), _LowPass()
        self._t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, x: float, t: float) -> float:
        if self._t is None:
            self._t = t
            self._dx.apply(0.0, 1.0)
            return self._x.apply(x, 1.0)
        dt = max(t - self._t, 1e-6)
        self._t = t
        prev = self._x.y
        dx = (x - prev) / dt
        edx = self._dx.apply(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.apply(x, self._alpha(cutoff, dt))


class FixationDetector:
    """Dispersion-threshold (I-DT): fixating if all points within the last
    `window_ms` fit inside `dispersion_px`."""
    def __init__(self, window_ms: float = 150.0, dispersion_px: float = 60.0):
        self.window_s = window_ms / 1000.0
        self.dispersion = dispersion_px
        self.buf: deque = deque()

    def update(self, x: float, y: float, t: float) -> bool:
        self.buf.append((x, y, t))
        while self.buf and t - self.buf[0][2] > self.window_s:
            self.buf.popleft()
        if not self.buf or (t - self.buf[0][2]) < self.window_s * 0.8:
            return False  # not enough history yet
        xs = [p[0] for p in self.buf]; ys = [p[1] for p in self.buf]
        return (max(xs) - min(xs)) + (max(ys) - min(ys)) <= self.dispersion

    def reset(self):
        self.buf.clear()
