"""GazeKey — the 5-second setup check that runs before the nine dots.

A bad sitting is **visible before calibration and invisible during it**. When
the camera sits below eye height the eyelids crop the iris, the vertical iris
ratio barely moves between the top and the bottom of the screen, and the fit
has almost no vertical signal to work with. The user finds this out forty
seconds later as a MARGINAL verdict they cannot interpret, and the app has
already spent their patience.

The measurement is the one the calibration diagnostics already report — the
**hy span**, how far the vertical iris ratio travels — taken from two targets
instead of nine:

* one at the top of the calibration region, one at the bottom, at exactly the
  10% and 90% heights the calibration grid uses, so the number is directly
  comparable with the ``hy span`` printed after a session;
* the IQR-rejected median hy at each — through the verified
  :func:`~gaze.calibration.aggregate_point`, so a target is aggregated exactly
  the way a calibration point is — and the span between them;
* below :data:`MIN_HY_SPAN` the screen says what to move, and nothing is
  calibrated until the user either passes the check or chooses to continue.

Reference measurements from the development machine: **0.051** and **0.042**
sitting well (calibrating to 44.3 px and 57.3 px, both PASS), **0.024** with
the camera below eye height (117.5 px, and nothing typable). The threshold sits
between them, nearer the bad one, because a check that cries wolf on a usable
sitting is worse than no check at all.

**Pacing exists to make the number mean something.** The first version measured
from the instant the screen appeared, and a median flips wholesale once more
than half its samples are stale: a user who took longer than ~1.05 s to get
their eyes onto the dot was measured *entirely* on where they had been looking
before, so the span collapsed toward zero rather than degrading gracefully. It
read 0.006–0.016 on a sitting that calibrates to 0.042. Hence, and none of it
is decoration:

* a **lead-in** (:data:`LEAD_IN_S`) that shows the instructions with **no dot
  and no measurement**, so the copy is read before anything is recorded and the
  first dot's appearance is itself the cue to look;
* a settle and a collection window sized by the property that actually
  matters — **a target tolerates `settle + collect/2` of reaction** before its
  median tips to the stale side. At 1000 ms and 30 samples that is **1.5 s**,
  against the ~0.3 s a saccade to a dot appearing on an otherwise still screen
  really takes. It is longer than calibration's 700 ms deliberately:
  calibration's user is already in the rhythm of the dots, this one has just
  been handed a screen;
* the screen **discards whatever gaze is already queued** before the first
  target (see :class:`~ui.setup_check_screen.SetupCheckScreen`), so frames
  captured before the dot existed cannot spend the settle.

Qt-free and camera-free, like :mod:`gaze.calibration_session`, so the whole
gate can be driven from synthetic features in the tests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from gaze.calibration import MIN_SAMPLES_PER_POINT, aggregate_point
from gaze.features import FrameFeatures
from gaze.region import Region, full_screen_region

#: Vertical iris travel below which the camera is almost certainly too low.
#: Between the measured sittings (bad 0.024; good 0.042 and 0.051) and
#: deliberately nearer the bad one — see the module docstring.
MIN_HY_SPAN = 0.035

#: fewest samples a target must keep after IQR rejection for its median to mean
#: anything — the verified core's own figure, so a check target is judged by
#: exactly the rule a calibration point is
MIN_SAMPLES = MIN_SAMPLES_PER_POINT

#: instructions-only time before the first dot appears. Nothing is measured
#: here: it buys the user the reading and orienting that used to be charged to
#: the first target's median.
LEAD_IN_S = 1.5

#: where the two targets sit in the region: the same rows as the 3x3 grid
TOP_FRACTION = 0.1
BOTTOM_FRACTION = 0.9


class SetupPhase(str, Enum):
    LEAD_IN = "lead_in"
    MEASURING = "measuring"
    PASSED = "passed"
    FAILED = "failed"


class SetupFailure(str, Enum):
    NO_FACE = "no_face"        # could not see the eyes for long enough
    LOW_SPAN = "low_span"      # eyes tracked fine, but barely moved vertically


FAILURE_TEXT = {
    SetupFailure.NO_FACE: (
        "I could not see your eyes",
        "Sit about 60 cm from the screen, face the camera, and make sure your "
        "face is lit from the front rather than from behind.",
    ),
    SetupFailure.LOW_SPAN: (
        "Camera looks too low",
        "Raise the laptop so the camera is at eye height — on a stand or a "
        "couple of books. From below, your eyelids hide most of the up-down "
        "movement of your eyes and calibration has almost nothing to work "
        "with.",
    ),
}


@dataclass
class SetupCheckResult:
    """What the check measured, and what it means."""

    hy_span: float
    min_hy_span: float
    hy_top: float = float("nan")
    hy_bottom: float = float("nan")
    #: samples kept per target after IQR rejection
    samples: Tuple[int, int] = (0, 0)
    #: ...out of this many valid frames collected
    collected: Tuple[int, int] = (0, 0)
    failure: Optional[SetupFailure] = None
    #: the user chose to calibrate anyway after a failed check
    overridden: bool = False

    @property
    def passed(self) -> bool:
        return self.failure is None

    @property
    def text(self) -> Tuple[str, str]:
        """``(headline, guidance)`` for the screen."""
        if self.passed:
            return ("Camera position looks fine", "Starting calibration…")
        return FAILURE_TEXT[self.failure]

    def measurement_line(self) -> str:
        """The number itself, always shown and always printed."""
        if self.failure is SetupFailure.NO_FACE:
            return (f"only {min(self.samples)} usable samples on one of the two "
                    f"targets - at least {MIN_SAMPLES} are needed")
        return (f"up-down eye movement {self.hy_span:.3f} "
                f"(needs {self.min_hy_span:.3f})")

    def console_line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        if self.overridden:
            verdict += " (continuing anyway)"
        return (f"setup check {verdict}: hy span {self.hy_span:.3f} "
                f"(top {self.hy_top:.3f}, bottom {self.hy_bottom:.3f}, "
                f"kept {self.samples[0]}/{self.collected[0]}"
                f"+{self.samples[1]}/{self.collected[1]}, "
                f"threshold {self.min_hy_span:.3f})")


@dataclass
class SetupTargetState:
    """What the screen should draw right now."""

    position: Tuple[int, int]
    index: int              # 1-based, for "1 / 2"
    total: int
    settling: bool
    progress: float
    samples: int


class SetupCheckSession:
    """Drives the two-target check from a stream of :class:`FrameFeatures`.

    Feed every frame to :meth:`update`; poll :attr:`phase` and
    :meth:`current_target` to render, and read :attr:`result` once it is done.
    """

    def __init__(
        self,
        screen_size: Tuple[int, int],
        region: Optional[Region] = None,
        min_hy_span: float = MIN_HY_SPAN,
        settle_ms: float = 1000.0,
        collect_samples: int = 30,
        collect_max_s: float = 2.0,
        min_samples: int = MIN_SAMPLES,
        lead_in_s: float = LEAD_IN_S,
    ) -> None:
        self.screen_size = screen_size
        self.region = region or full_screen_region(screen_size)
        self.min_hy_span = float(min_hy_span)
        self.lead_in_s = max(0.0, float(lead_in_s))
        self.settle_s = settle_ms / 1000.0
        self.collect_samples = max(1, int(collect_samples))
        self.collect_max_s = float(collect_max_s)
        #: at least one, so a target that saw nothing is always a NO_FACE and
        #: the span comparison below never has to reason about NaN
        self.min_samples = max(1, int(min_samples))

        self.targets: List[Tuple[int, int]] = [
            self._point(TOP_FRACTION), self._point(BOTTOM_FRACTION)]
        self.result: Optional[SetupCheckResult] = None
        self._reset()

    def _point(self, fy: float) -> Tuple[int, int]:
        x, y = self.region.point(0.5, fy)
        return int(round(x)), int(round(y))

    def _reset(self) -> None:
        self.phase = SetupPhase.LEAD_IN if self.lead_in_s else SetupPhase.MEASURING
        self._index = 0
        self._start: Optional[float] = None
        self._lead_in_start: Optional[float] = None
        self._now = 0.0
        self._samples: List[np.ndarray] = []
        self._collected: List[List[np.ndarray]] = []

    def restart(self) -> None:
        """Run the check again — the user moved the camera and wants a retry.

        Back through the lead-in, not straight into a dot: the user has just
        been reading a failure screen and moving a camera, and measuring them
        mid-fidget is how this went wrong the first time.
        """
        self.result = None
        self._reset()

    @property
    def budget_s(self) -> float:
        """Worst case wall clock — every target running to its clock cap."""
        return self.lead_in_s + len(self.targets) * (self.settle_s
                                                     + self.collect_max_s)

    @property
    def typical_s(self) -> float:
        """What it usually takes, for the copy: collection ends on the count."""
        per_target = self.settle_s + self.collect_samples / 30.0
        return self.lead_in_s + len(self.targets) * per_target

    @property
    def reaction_budget_s(self) -> float:
        """How slowly the user may get their eyes onto a dot and still be
        measured correctly.

        A median tips wholesale once more than half its samples come from
        somewhere else, so the budget is the settle plus half the collection —
        not the whole target. This is the number the check was missing: at
        500 ms and 30 samples it was 1.05 s, and a user reading three lines of
        instructions took longer than that every single time.
        """
        return self.settle_s + (self.collect_samples / 30.0) / 2.0

    @property
    def lead_in_progress(self) -> float:
        """0..1 through the instructions-only phase."""
        if self.phase is not SetupPhase.LEAD_IN or self._lead_in_start is None:
            return 1.0
        return min(1.0, (self._now - self._lead_in_start)
                   / max(self.lead_in_s, 1e-6))

    @property
    def is_finished(self) -> bool:
        return self.phase in (SetupPhase.PASSED, SetupPhase.FAILED)

    # -------------------------------------------------------------------- feed
    def current_target(self) -> Optional[SetupTargetState]:
        if self.phase is not SetupPhase.MEASURING:
            return None
        elapsed = 0.0 if self._start is None else self._now - self._start
        settling = elapsed < self.settle_s
        collecting = max(0.0, elapsed - self.settle_s)
        return SetupTargetState(
            position=self.targets[self._index],
            index=self._index + 1,
            total=len(self.targets),
            settling=settling,
            progress=0.0 if settling else min(1.0, max(
                len(self._samples) / self.collect_samples,
                collecting / max(self.collect_max_s, 1e-6),
            )),
            samples=len(self._samples),
        )

    def observe(self, sample) -> None:
        """See the whole gaze sample, before :meth:`update` gets its features.

        A no-op here — the check needs nothing but ``hy``. ``--feature-lab``
        overrides it to take the frame's landmarks, which is the only thing
        that cannot be recovered from :class:`~gaze.features.FrameFeatures`
        after the fact.
        """

    def update(self, features: FrameFeatures) -> Optional[SetupCheckResult]:
        """Consume one frame; returns the result on the frame it completes."""
        if self.is_finished:
            return None
        self._now = features.timestamp or time.time()

        if self.phase is SetupPhase.LEAD_IN:
            # Instructions only: no dot, nothing recorded. The user reads here
            # instead of on the first target's median.
            if self._lead_in_start is None:
                self._lead_in_start = self._now
            if self._now - self._lead_in_start >= self.lead_in_s:
                self.phase = SetupPhase.MEASURING
            return None

        if self._start is None:
            self._start = self._now
        elapsed = self._now - self._start
        if elapsed < self.settle_s:
            return None                 # saccade to the dot, not measured

        if features.valid:
            self._samples.append(features.vector())
        collecting = elapsed - self.settle_s
        if (len(self._samples) >= self.collect_samples
                or collecting >= self.collect_max_s):
            return self._finish_target()
        return None

    def _finish_target(self) -> Optional[SetupCheckResult]:
        self._collected.append(self._samples)
        self._samples = []
        self._start = None
        self._index += 1
        if self._index < len(self.targets):
            return None
        return self._evaluate()

    # ----------------------------------------------------------------- verdict
    def _aggregate(self, block: List[np.ndarray]) -> Tuple[float, int]:
        """``(median hy, kept)`` through the verified aggregation.

        The same 1.5x IQR rejection and median a calibration point gets, so the
        span this reports is the same quantity the session diagnostics print
        rather than a lookalike computed a second way.
        """
        if not block:
            return float("nan"), 0
        samples = np.asarray(block, dtype=float)
        aggregate, kept = aggregate_point(samples)
        if aggregate is None:
            return float("nan"), kept
        return float(aggregate[1]), kept

    def _evaluate(self) -> SetupCheckResult:
        collected = tuple(len(block) for block in self._collected)
        aggregated = [self._aggregate(block) for block in self._collected]
        medians = [value for value, _ in aggregated]
        kept = tuple(count for _, count in aggregated)
        span = abs(medians[1] - medians[0])

        failure: Optional[SetupFailure] = None
        if min(kept) < self.min_samples or not span == span:      # NaN-safe
            failure = SetupFailure.NO_FACE
        elif span < self.min_hy_span:
            failure = SetupFailure.LOW_SPAN

        self.result = SetupCheckResult(
            hy_span=span, min_hy_span=self.min_hy_span,
            hy_top=medians[0], hy_bottom=medians[1],
            samples=(kept[0], kept[1]), collected=(collected[0], collected[1]),
            failure=failure,
        )
        self.phase = SetupPhase.PASSED if failure is None else SetupPhase.FAILED
        return self.result

    def override(self) -> SetupCheckResult:
        """The user chose to calibrate anyway. Returns the marked result."""
        if self.result is None:
            self.result = SetupCheckResult(hy_span=float("nan"),
                                           min_hy_span=self.min_hy_span,
                                           failure=SetupFailure.NO_FACE)
        self.result.overridden = True
        return self.result
