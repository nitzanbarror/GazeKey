"""GazeKey — target practice: measured aiming drill after a calibration.

A dot appears somewhere on screen; holding a fixated gaze inside its hit radius
for ``hold_s`` pops it and the next one appears. After ``targets`` rounds the
session reports how many were hit, how far off the gaze sat while holding, and
how long each one took.

The rules deliberately mirror the keyboard's dwell (:mod:`interaction.
controller`) so practice measures the same thing typing depends on:

* the hold only advances while the fixation detector agrees the gaze is held;
* leaving the radius **decays** the hold over the grace period instead of
  throwing it away;
* an invalid stream (blink held too long, face lost) **freezes** the hold.

Unlike the keyboard there is no hysteresis: the ring the user sees is exactly
the region that counts, because this screen is a measurement, not a target to
be helped along.

UI-free, so the whole drill can be driven from scripted gaze in the tests.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from gaze.region import Region, full_screen_region

#: floor for the hit radius, however good the calibration claims to be
MIN_HIT_RADIUS_PX = 60.0

#: keep targets this far in from the screen edges
EDGE_MARGIN_RATIO = 0.12

#: consecutive targets must be at least this fraction of the diagonal apart,
#: so every round needs a real saccade rather than a nudge
MIN_SEPARATION_RATIO = 0.25


class PracticePhase(str, Enum):
    RUNNING = "running"
    DONE = "done"


def hit_radius_for(validation_error_px: float,
                   floor: float = MIN_HIT_RADIUS_PX) -> float:
    """Hit radius for a given calibration accuracy: ``max(60, error)``."""
    if (validation_error_px is None
            or validation_error_px != validation_error_px       # NaN
            or validation_error_px == float("inf")):
        return floor
    return max(floor, float(validation_error_px))


@dataclass
class PracticeAttempt:
    """One target, hit or missed."""

    index: int
    target: Tuple[float, float]
    hit: bool
    time_s: float
    mean_error_px: float
    samples: int
    #: mean distance to the target over **every** sample the target was up
    #: for, hit or miss. Unlike ``mean_error_px`` this is not conditioned on
    #: already being inside the hit radius, so it is the honest in-region
    #: accuracy number — the one to compare two calibrations with.
    aim_error_px: float = float("nan")
    aim_samples: int = 0


@dataclass
class PracticeStats:
    """What the summary screen reports."""

    attempts: List[PracticeAttempt] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def hits(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.hit)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def mean_error_px(self) -> float:
        errors = [a.mean_error_px for a in self.attempts
                  if a.hit and a.mean_error_px == a.mean_error_px]
        return float(sum(errors) / len(errors)) if errors else float("nan")

    @property
    def mean_aim_error_px(self) -> float:
        """Mean distance to the target over every sample of every attempt.

        Weighted by sample count so a long, badly-aimed attempt counts for what
        it was. This is the comparable accuracy figure across two calibrations:
        it includes the approach and the misses, where ``mean_error_px`` only
        looks at gaze that had already landed inside the radius.
        """
        weighted = [(a.aim_error_px, a.aim_samples) for a in self.attempts
                    if a.aim_samples and a.aim_error_px == a.aim_error_px]
        total = sum(n for _, n in weighted)
        if not total:
            return float("nan")
        return float(sum(error * n for error, n in weighted) / total)

    @property
    def mean_time_s(self) -> float:
        times = [a.time_s for a in self.attempts if a.hit]
        return float(sum(times) / len(times)) if times else float("nan")

    def summary_lines(self) -> List[str]:
        """Headline numbers, ready for the screen and the console."""
        if not self.total:
            return ["No targets attempted."]

        def px(value: float) -> str:
            return "-" if value != value else f"{value:.0f} px"

        time_s = ("-" if self.mean_time_s != self.mean_time_s
                  else f"{self.mean_time_s:.1f} s")
        return [
            f"Hits: {self.hits} of {self.total}  ({self.hit_rate * 100:.0f}%)",
            f"Average aim error: {px(self.mean_aim_error_px)}",
            f"Average error while holding: {px(self.mean_error_px)}",
            f"Average time to hit: {time_s}",
        ]


class PracticeSession:
    """Drives the drill from a stream of gaze samples."""

    def __init__(
        self,
        screen_size: Tuple[int, int],
        hit_radius_px: float,
        targets: int = 10,
        hold_s: float = 0.8,
        grace_ms: float = 200.0,
        timeout_s: float = 12.0,
        rng: Optional[random.Random] = None,
        region: Optional[Region] = None,
    ) -> None:
        self.screen_size = screen_size
        self.hit_radius_px = float(hit_radius_px)
        self.targets = int(targets)
        self.hold_s = float(hold_s)
        self.grace_s = max(grace_ms / 1000.0, 1e-6)
        self.timeout_s = float(timeout_s)
        self._rng = rng or random.Random()

        #: targets stay inside the calibrated region — a drill that measured
        #: aim where the model was never fitted would be measuring nothing
        self.region = region or full_screen_region(screen_size)
        self.min_separation = MIN_SEPARATION_RATIO * self.region.diagonal

        self.phase = PracticePhase.RUNNING
        self.stats = PracticeStats()
        self.index = 0
        self._target: Tuple[float, float] = self._pick_target(None)
        self._reset_target(None)

    # ------------------------------------------------------------------ state
    @property
    def current_target(self) -> Optional[Tuple[float, float]]:
        return None if self.phase is PracticePhase.DONE else self._target

    @property
    def progress(self) -> float:
        """0..1 of the way to popping the current target."""
        return min(1.0, self._held / self.hold_s) if self.hold_s else 0.0

    @property
    def inside(self) -> bool:
        """Was the last sample within the hit radius?"""
        return self._inside

    @property
    def remaining(self) -> int:
        return max(0, self.targets - self.index)

    def distance_to_target(self, x: float, y: float) -> float:
        return math.hypot(x - self._target[0], y - self._target[1])

    # ------------------------------------------------------------------- feed
    def update(self, x: float, y: float, is_fixating: bool,
               stream_valid: bool, now: float) -> Optional[str]:
        """Consume one sample. Returns ``"hit"`` or ``"miss"`` on a transition."""
        if self.phase is PracticePhase.DONE:
            return None
        if self._shown_at is None:
            self._shown_at = now
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now

        usable = stream_valid and math.isfinite(x) and math.isfinite(y)
        if not usable:
            self._inside = False
            return None                       # freeze: no advance, no decay

        if now - self._shown_at >= self.timeout_s:
            return self._finish(hit=False, now=now)

        distance = self.distance_to_target(x, y)
        self._inside = distance <= self.hit_radius_px
        self._aim.append(distance)          # every usable sample, hit or miss

        if self._inside and is_fixating:
            self._leaving = False
            self._held += dt
            self._errors.append(distance)
            if self._held >= self.hold_s:
                return self._finish(hit=True, now=now)
        elif not self._inside:
            self._decay(dt)
        return None

    def skip(self) -> None:
        """Abandon the drill (Esc)."""
        self.phase = PracticePhase.DONE

    # -------------------------------------------------------------- internals
    def _decay(self, dt: float) -> None:
        if self._held <= 0.0:
            return
        if not self._leaving:
            self._leaving = True
            self._leave_held = max(self._held, 1e-9)
        self._held = max(0.0, self._held - self._leave_held * dt / self.grace_s)

    def _finish(self, hit: bool, now: float) -> str:
        errors, aim = self._errors, self._aim
        self.stats.attempts.append(PracticeAttempt(
            index=self.index + 1,
            target=self._target,
            hit=hit,
            time_s=now - (self._shown_at or now),
            mean_error_px=(float(sum(errors) / len(errors)) if errors
                           else float("nan")),
            samples=len(errors),
            aim_error_px=(float(sum(aim) / len(aim)) if aim else float("nan")),
            aim_samples=len(aim),
        ))
        self.index += 1
        if self.index >= self.targets:
            self.phase = PracticePhase.DONE
        else:
            previous = self._target
            self._target = self._pick_target(previous)
            self._reset_target(now)
        return "hit" if hit else "miss"

    def _reset_target(self, now: Optional[float]) -> None:
        self._held = 0.0
        self._errors: List[float] = []
        self._aim: List[float] = []
        self._shown_at = now
        self._last_t = now
        self._leaving = False
        self._leave_held = 0.0
        self._inside = False

    def _pick_target(self, previous: Optional[Tuple[float, float]]
                     ) -> Tuple[float, float]:
        """A random position, inset from the region edges and away from the last."""
        region = self.region
        margin_x = region.w * EDGE_MARGIN_RATIO
        margin_y = region.h * EDGE_MARGIN_RATIO
        low_x, high_x = region.x + margin_x, region.x + region.w - margin_x
        low_y, high_y = region.y + margin_y, region.y + region.h - margin_y
        candidate = region.centre
        for _ in range(64):
            candidate = (self._rng.uniform(low_x, high_x),
                         self._rng.uniform(low_y, high_y))
            if previous is None:
                return candidate
            if math.hypot(candidate[0] - previous[0],
                          candidate[1] - previous[1]) >= self.min_separation:
                return candidate
        return candidate
