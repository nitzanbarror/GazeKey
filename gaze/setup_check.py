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
* the median hy at each, and the span between them;
* below :data:`MIN_HY_SPAN` the screen says what to move, and nothing is
  calibrated until the user either passes the check or chooses to continue.

Reference measurements from the development machine: a good sitting spans
**0.051**, a sitting with the camera below eye height spans **0.024**. The
default threshold sits between them, nearer the bad one, because a check that
cries wolf on a usable sitting would be worse than no check at all.

Qt-free and camera-free, like :mod:`gaze.calibration_session`, so the whole
gate can be driven from synthetic features in the tests.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from gaze.features import FrameFeatures
from gaze.region import Region, full_screen_region

#: Vertical iris travel below which the camera is almost certainly too low.
#: Between the two measured sittings (bad 0.024, good 0.051) and deliberately
#: nearer the bad one — see the module docstring.
MIN_HY_SPAN = 0.035

#: fewest valid frames a target needs before its median means anything
MIN_SAMPLES = 10

#: where the two targets sit in the region: the same rows as the 3x3 grid
TOP_FRACTION = 0.1
BOTTOM_FRACTION = 0.9


class SetupPhase(str, Enum):
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
    samples: Tuple[int, int] = (0, 0)
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
            return (f"only {min(self.samples)} usable frames on one of the two "
                    f"targets - at least {MIN_SAMPLES} are needed")
        return (f"up-down eye movement {self.hy_span:.3f} "
                f"(needs {self.min_hy_span:.3f})")

    def console_line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        if self.overridden:
            verdict += " (continuing anyway)"
        return (f"setup check {verdict}: hy span {self.hy_span:.3f} "
                f"(top {self.hy_top:.3f}, bottom {self.hy_bottom:.3f}, "
                f"{self.samples[0]}+{self.samples[1]} samples, "
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
        settle_ms: float = 500.0,
        collect_samples: int = 30,
        collect_max_s: float = 2.0,
        min_samples: int = MIN_SAMPLES,
    ) -> None:
        self.screen_size = screen_size
        self.region = region or full_screen_region(screen_size)
        self.min_hy_span = float(min_hy_span)
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
        self.phase = SetupPhase.MEASURING
        self._index = 0
        self._start: Optional[float] = None
        self._now = 0.0
        self._samples: List[float] = []
        self._collected: List[List[float]] = []

    def restart(self) -> None:
        """Run the check again — the user moved the camera and wants a retry."""
        self.result = None
        self._reset()

    @property
    def budget_s(self) -> float:
        """Worst-case wall clock, for the copy that promises "about 5 seconds"."""
        return len(self.targets) * (self.settle_s + self.collect_max_s)

    @property
    def is_finished(self) -> bool:
        return self.phase is not SetupPhase.MEASURING

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

    def update(self, features: FrameFeatures) -> Optional[SetupCheckResult]:
        """Consume one frame; returns the result on the frame it completes."""
        if self.phase is not SetupPhase.MEASURING:
            return None
        self._now = features.timestamp or time.time()
        if self._start is None:
            self._start = self._now
        elapsed = self._now - self._start
        if elapsed < self.settle_s:
            return None                 # saccade to the dot, not measured

        if features.valid and features.hy == features.hy:      # not NaN
            self._samples.append(float(features.hy))
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
    def _evaluate(self) -> SetupCheckResult:
        counts = tuple(len(block) for block in self._collected)
        medians = [statistics.median(block) if block else float("nan")
                   for block in self._collected]
        span = abs(medians[1] - medians[0]) if all(counts) else float("nan")

        failure: Optional[SetupFailure] = None
        if min(counts) < self.min_samples:
            failure = SetupFailure.NO_FACE
        elif span < self.min_hy_span:
            failure = SetupFailure.LOW_SPAN

        self.result = SetupCheckResult(
            hy_span=span, min_hy_span=self.min_hy_span,
            hy_top=medians[0], hy_bottom=medians[1],
            samples=(counts[0], counts[1]), failure=failure,
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
