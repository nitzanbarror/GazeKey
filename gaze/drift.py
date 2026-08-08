"""GazeKey — drift monitoring and the one-point touch-up (spec Section 5.5).

Calibration decays during a session: the head settles a little deeper into its
support, the chair rolls back a centimetre, the light changes. The mapping is
still *shaped* right, it is just **offset** — so the fix is a translation, not
another nine points.

Two pieces live here, both free of Qt and of the camera so the tests can drive
them from scripted gaze:

:class:`DriftMonitor`
    Watches typing for evidence that the aim has slipped, and says so. It never
    acts: no auto-recalibration, no interruption, no dialog. The UI turns the
    evidence into one quiet line in the corner and the user decides.

:class:`TouchUpSession`
    The correction itself. One centre target, ~2 s of samples, a median offset
    applied to the model's two constant terms via
    :meth:`~gaze.calibration.CalibrationModel.apply_offset` (the only mutation
    the verified core exposes for exactly this purpose).

**Evidence, and its limits.** Every key activation is weak ground truth: the
user *was* looking at that key when it fired. Comparing the key's centre with
the gaze centroid that triggered it gives a per-activation offset, and an
exponentially-weighted mean of those offsets is the drift estimate. The limit
is structural and worth stating plainly: a dwell only completes *inside* the
key, so a measured offset can never exceed half a key — drift larger than that
shows up as keys that simply stop activating. That is why the second signal
matters: **type-then-backspace within 3 s** counts a correction, and three of
them inside a minute is enough on its own to raise the flag.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

from gaze.calibration import CalibrationModel
from gaze.region import Region, full_screen_region
from interaction.layouts import Key

# ---------------------------------------------------------------- drift tuning
#: gaze window whose centroid stands in for "where the user was actually looking"
CENTROID_WINDOW_MS = 300.0

#: weight of the newest activation in the exponentially-weighted mean
ALPHA = 0.3

#: offsets from fewer activations than this are noise, not evidence
MIN_ACTIVATIONS = 5

#: a backspace this soon after a character counts as fixing a mis-selection
CORRECTION_GAP_S = 3.0

#: corrections are counted over this rolling window (spec: 60 s)
CORRECTION_WINDOW_S = 60.0

#: this many corrections inside the window is drift on its own (spec: 3)
CORRECTIONS_FOR_DRIFT = 3

#: floor for the offset that counts as a full unit of drift evidence
MIN_OFFSET_PX = 50.0

#: once flagged, the flag only clears below this fraction of the threshold, so
#: the corner indicator cannot flicker on and off between two keystrokes
CLEAR_RATIO = 0.7


def offset_threshold_px(validation_error_px: float,
                        configured: float = 0.0) -> float:
    """How large an offset counts as one full unit of drift evidence.

    ``configured`` (config ``drift_offset_px``) wins when set; otherwise the
    threshold tracks the measured calibration accuracy, because an offset the
    size of the calibration error is what actually starts costing keystrokes.
    """
    if configured and configured > 0:
        return float(configured)
    error = validation_error_px
    if (error is None or error != error                     # NaN
            or error in (float("inf"), 0.0)):
        return MIN_OFFSET_PX
    return max(MIN_OFFSET_PX, float(error))


@dataclass
class DriftState:
    """Everything the monitor knows, in one readable object."""

    offset: Tuple[float, float]
    offset_px: float
    activations: int
    corrections: int
    score: float
    drifting: bool

    def summary(self) -> str:
        return (f"offset {self.offset[0]:+.0f},{self.offset[1]:+.0f} px "
                f"({self.offset_px:.0f} px over {self.activations} keys), "
                f"{self.corrections} correction(s), score {self.score:.2f}")


class DriftMonitor:
    """Accumulates weak evidence that the calibration has slipped.

    Feed it :meth:`record_sample` every frame and :meth:`record_activation`
    every key that fires; read :attr:`drifting` whenever the UI repaints.
    """

    def __init__(
        self,
        validation_error_px: float = float("nan"),
        configured_offset_px: float = 0.0,
        centroid_window_ms: float = CENTROID_WINDOW_MS,
        alpha: float = ALPHA,
        min_activations: int = MIN_ACTIVATIONS,
        correction_gap_s: float = CORRECTION_GAP_S,
        correction_window_s: float = CORRECTION_WINDOW_S,
        corrections_for_drift: int = CORRECTIONS_FOR_DRIFT,
    ) -> None:
        self.threshold_px = offset_threshold_px(validation_error_px,
                                                configured_offset_px)
        self.centroid_window_s = centroid_window_ms / 1000.0
        self.alpha = alpha
        self.min_activations = min_activations
        self.correction_gap_s = correction_gap_s
        self.correction_window_s = correction_window_s
        self.corrections_for_drift = max(1, corrections_for_drift)

        self._samples: Deque[Tuple[float, float, float]] = deque()
        self._corrections: List[float] = []
        self._offset = (0.0, 0.0)
        self._activations = 0
        self._last_char_at: Optional[float] = None
        self._now = 0.0
        self._flagged = False

    # ------------------------------------------------------------------ inputs
    def record_sample(self, x: float, y: float, now: float) -> None:
        """One frame of usable gaze. Held (blink) samples should be skipped."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        self._now = now
        self._samples.append((x, y, now))
        self._prune(now)

    def centroid(self, now: Optional[float] = None
                 ) -> Optional[Tuple[float, float]]:
        """Mean gaze over the last ``centroid_window_ms``, or ``None``."""
        now = self._now if now is None else now
        points = [(x, y) for x, y, t in self._samples
                  if now - t <= self.centroid_window_s]
        if not points:
            return None
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))

    def record_activation(self, key: Key, now: float) -> None:
        """Fold one activated key into the evidence.

        Only ``char`` keys contribute an offset: Space and the control keys are
        wide or oddly shaped, so their centres say very little about where the
        user was looking. A ``backspace`` soon after a character is counted as a
        correction instead.
        """
        self._now = now
        self._prune(now)
        action = getattr(key, "action", "")
        if action == "backspace":
            if (self._last_char_at is not None
                    and now - self._last_char_at <= self.correction_gap_s):
                self._corrections.append(now)
            return
        if action != "char":
            return

        self._last_char_at = now
        centre = self.centroid(now)
        if centre is None:
            return
        dx = key.centre[0] - centre[0]
        dy = key.centre[1] - centre[1]
        if self._activations == 0:
            self._offset = (dx, dy)
        else:
            self._offset = (
                (1.0 - self.alpha) * self._offset[0] + self.alpha * dx,
                (1.0 - self.alpha) * self._offset[1] + self.alpha * dy,
            )
        self._activations += 1

    def _prune(self, now: float) -> None:
        while self._samples and now - self._samples[0][2] > self.centroid_window_s:
            self._samples.popleft()
        self._corrections = [t for t in self._corrections
                             if now - t <= self.correction_window_s]

    # ----------------------------------------------------------------- verdict
    @property
    def offset(self) -> Tuple[float, float]:
        """Exponentially-weighted mean of ``key centre - gaze centroid``."""
        return self._offset

    @property
    def offset_px(self) -> float:
        return math.hypot(*self._offset)

    @property
    def activations(self) -> int:
        return self._activations

    @property
    def corrections(self) -> int:
        return len(self._corrections)

    @property
    def score(self) -> float:
        """Combined evidence; ``>= 1`` is drift.

        Either signal can reach 1 alone — an offset the size of the threshold,
        or the spec's three corrections in a minute — and partial evidence from
        both adds up, because two weak signals pointing the same way are worth
        more than either on its own.
        """
        offset_score = (0.0 if self._activations < self.min_activations
                        else self.offset_px / max(self.threshold_px, 1e-6))
        correction_score = self.corrections / float(self.corrections_for_drift)
        return offset_score + correction_score

    @property
    def drifting(self) -> bool:
        """Has the evidence crossed the threshold? (sticky until it clears)"""
        score = self.score
        if self._flagged:
            self._flagged = score >= CLEAR_RATIO
        else:
            self._flagged = score >= 1.0
        return self._flagged

    def state(self) -> DriftState:
        return DriftState(
            offset=self._offset,
            offset_px=self.offset_px,
            activations=self._activations,
            corrections=self.corrections,
            score=self.score,
            drifting=self.drifting,
        )

    def reset(self) -> None:
        """Forget everything — after a touch-up or a fresh calibration."""
        self._samples.clear()
        self._corrections.clear()
        self._offset = (0.0, 0.0)
        self._activations = 0
        self._last_char_at = None
        self._flagged = False


# ----------------------------------------------------------------- the touch-up
class TouchUpPhase(str, Enum):
    SETTLING = "settling"       # saccade to the dot, deliberately not measured
    COLLECTING = "collecting"
    DONE = "done"


#: an offset further than this fraction of the screen diagonal is not drift —
#: it is a broken calibration, and translating it would only hide the problem
MAX_OFFSET_RATIO = 0.25

#: fewer samples than this and the measurement is not worth trusting
MIN_TOUCHUP_SAMPLES = 10


@dataclass
class TouchUpResult:
    """What a touch-up measured, and whether it is safe to apply."""

    dx: float = 0.0
    dy: float = 0.0
    samples: int = 0
    accepted: bool = False
    reason: str = ""

    @property
    def offset_px(self) -> float:
        return math.hypot(self.dx, self.dy)

    def message(self) -> str:
        if self.accepted:
            return f"aim corrected by {self.offset_px:.0f} px"
        return self.reason or "touch-up not applied"


class TouchUpSession:
    """One centre target, ~2 s of gaze, one translation correction.

    Deliberately the shortest possible interruption: the whole thing is
    :attr:`budget_s` from the moment the key fires, well inside the 10 s that
    makes the difference between a quick fix and abandoning the sentence.

    The dot sits at the centre of the **calibration region**, not of the
    screen: a correction measured outside the fitted area would be reading the
    model's extrapolation and calling it drift.
    """

    def __init__(
        self,
        screen_size: Tuple[int, int],
        settle_ms: float = 400.0,
        collect_s: float = 2.0,
        min_samples: int = MIN_TOUCHUP_SAMPLES,
        max_offset_ratio: float = MAX_OFFSET_RATIO,
        target: Optional[Tuple[float, float]] = None,
        region: Optional[Region] = None,
    ) -> None:
        self.screen_size = screen_size
        self.settle_s = settle_ms / 1000.0
        self.collect_s = float(collect_s)
        self.min_samples = int(min_samples)
        self.region = region or full_screen_region(screen_size)
        self.target = target or self.region.centre
        self.max_offset_px = max_offset_ratio * self.region.diagonal

        self.phase = TouchUpPhase.SETTLING
        self.result: Optional[TouchUpResult] = None
        self._start: Optional[float] = None
        self._elapsed = 0.0
        self._xs: List[float] = []
        self._ys: List[float] = []
        self._fixating = 0

    # ------------------------------------------------------------------ timing
    @property
    def budget_s(self) -> float:
        """Wall-clock the screen costs the user, settle plus measurement."""
        return self.settle_s + self.collect_s

    def elapsed(self, now: float) -> float:
        return 0.0 if self._start is None else max(0.0, now - self._start)

    @property
    def progress(self) -> float:
        """0..1 of the measurement, for the ring around the dot."""
        if self.phase is TouchUpPhase.DONE:
            return 1.0
        if self._start is None or self.phase is TouchUpPhase.SETTLING:
            return 0.0
        return min(1.0, max(0.0, (self._elapsed - self.settle_s) / self.collect_s))

    @property
    def samples(self) -> int:
        return len(self._xs)

    # -------------------------------------------------------------------- feed
    def update(self, x: float, y: float, is_fixating: bool,
               stream_valid: bool, now: float) -> Optional[TouchUpResult]:
        """Consume one sample; returns the result on the frame it completes."""
        if self.phase is TouchUpPhase.DONE:
            return None
        if self._start is None:
            self._start = now
        self._elapsed = elapsed = max(0.0, now - self._start)

        if elapsed < self.settle_s:
            return None                    # saccade to the dot, not measured
        self.phase = TouchUpPhase.COLLECTING

        if stream_valid and math.isfinite(x) and math.isfinite(y):
            self._xs.append(x)
            self._ys.append(y)
            if is_fixating:
                self._fixating += 1

        if elapsed >= self.settle_s + self.collect_s:
            return self._finish()
        return None

    def cancel(self) -> None:
        self.phase = TouchUpPhase.DONE

    # ----------------------------------------------------------------- outcome
    def _finish(self) -> TouchUpResult:
        self.phase = TouchUpPhase.DONE
        if len(self._xs) < self.min_samples:
            self.result = TouchUpResult(
                samples=len(self._xs), accepted=False,
                reason=("could not see your eyes for long enough - "
                        "touch-up not applied"),
            )
            return self.result

        dx = self.target[0] - _median(self._xs)
        dy = self.target[1] - _median(self._ys)
        if math.hypot(dx, dy) > self.max_offset_px:
            self.result = TouchUpResult(
                dx=dx, dy=dy, samples=len(self._xs), accepted=False,
                reason=(f"gaze was {math.hypot(dx, dy):.0f} px off the dot - "
                        f"too far to be drift, recalibrate instead"),
            )
            return self.result

        self.result = TouchUpResult(dx=dx, dy=dy, samples=len(self._xs),
                                    accepted=True)
        return self.result

    def apply_to(self, model: CalibrationModel) -> bool:
        """Apply an accepted result to the model's constant terms only."""
        if self.result is None or not self.result.accepted:
            return False
        model.apply_offset(self.result.dx, self.result.dy)
        return True


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])
