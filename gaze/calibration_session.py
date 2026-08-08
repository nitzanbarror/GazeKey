"""GazeKey — calibration session sequencing (spec Section 5.2 / 5.3).

This is the **UI-free driver** for a calibration session: it decides which
target is showing, when to start collecting, when a point needs repeating and
when to move on. Every piece of actual mathematics — pose compensation,
outlier rejection, aggregation, the polynomial fit, the validation verdict —
lives in the verified :mod:`gaze.calibration` and is only called from here.

Being free of Qt and of the camera, the whole session can be driven from a
synthetic eye in the tests.

Flow::

    SWEEP ──ok──> POINTS(9) ──fit──> VALIDATION(3) ──> RESULTS
      │                ^                    │
      └─fail─> SWEEP_FAILED               FAIL, once:
               (retry_sweep)              re-collect the 2 worst points,
                                          refit, re-validate
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from gaze.calibration import (
    MIN_SAMPLES_PER_POINT,
    MIN_SWEEP_POSE_RANGE_DEG,
    MIN_SWEEP_SAMPLES,
    CalibrationModel,
    aggregate_point,
    reject_outliers,
)
from gaze.features import FrameFeatures
from gaze.region import Region, attach_region, full_screen_region

#: fewest usable 9-point samples that still allow a meaningful 6-term fit
MIN_USABLE_POINTS = 6

# ------------------------------------------------------- point-level repair
#: A single glance away during one target poisons the whole fit: the ridge
#: polynomial has only nine points to work with, so one that lands hundreds of
#: pixels out drags the surface toward it everywhere. Spec 5.3 already
#: re-collects the two worst points, but only once the *verdict* has failed —
#: which misses exactly the case that motivated this: an otherwise 20-25 px
#: session dragged to a MARGINAL 102.7 px by one bad point. So the same
#: mechanism also runs on point-level evidence, before validation.
#:
#: A point is suspect when its residual is both this multiple of the median...
REPAIR_RESIDUAL_RATIO = 3.0
#: ...and this many pixels in absolute terms, so a tight fit where everything
#: is within a few px is never disturbed by ordinary scatter.
REPAIR_RESIDUAL_MIN_PX = 60.0
#: At most this many points are re-collected. Three or more outliers is not a
#: glance away, it is a systemic problem (lighting, head slipping, a camera
#: knocked) that re-collecting will not fix, and the diagnostics say so instead.
MAX_REPAIRED_POINTS = 2

#: typography the UI uses freely but a Windows console may not be able to encode
_ASCII_MAP = {"°": " deg", "–": "-", "—": "-", "≤": "<=", "≥": ">=", "…": "...",
              "·": "-", "×": "x"}


def to_ascii(text: str) -> str:
    """Transliterate report text so any console code page can print it."""
    for source, replacement in _ASCII_MAP.items():
        text = text.replace(source, replacement)
    return text


class Phase(str, Enum):
    """Where the session currently is."""

    SWEEP = "sweep"
    SWEEP_FAILED = "sweep_failed"
    POINTS = "points"
    VALIDATION = "validation"
    RESULTS = "results"


class SweepFailure(str, Enum):
    """Why :meth:`CalibrationModel.fit_pose_compensation` refused the sweep."""

    NO_FACE = "no_face"        # too few valid frames
    NO_MOTION = "no_motion"    # head barely moved
    UNKNOWN = "unknown"


SWEEP_FAILURE_TEXT = {
    SweepFailure.NO_FACE: (
        "I could not see your eyes for long enough.",
        "Sit about 60 cm away, face the camera, and make sure your face is "
        "well lit from the front.",
    ),
    SweepFailure.NO_MOTION: (
        "Your head barely moved.",
        "Keep your eyes locked on the dot and slowly turn your head left and "
        "right, then up and down — about 10-15 degrees each way.",
    ),
    SweepFailure.UNKNOWN: (
        "The head-sweep step did not produce a usable result.",
        "Keep your eyes on the dot and rotate your head slowly through the "
        "whole range.",
    ),
}


@dataclass
class TargetState:
    """What the UI should draw right now during a target stage."""

    position: Tuple[int, int]
    index: int              # 1-based, for "3 / 9"
    total: int
    settling: bool          # inside the saccade-settling delay
    settle_progress: float  # 0..1 countdown until collection starts
    progress: float         # 0..1 collection progress
    retry: int              # how many times this point has been repeated
    stage: Phase
    samples: int


@dataclass
class PointDiagnostics:
    """Per-target evidence, for the results screen and the console report."""

    index: int                  # 1-based presentation order
    target: Tuple[int, int]
    collected: int              # valid frames gathered
    kept: int                   # survived 1.5x IQR outlier rejection
    retries: int
    hx: float = float("nan")
    hy: float = float("nan")
    prediction: Optional[Tuple[float, float]] = None
    residual_px: float = float("nan")
    #: this point was re-collected because it fitted far worse than the rest
    repaired: bool = False

    @property
    def rejected(self) -> int:
        return max(0, self.collected - self.kept)


@dataclass
class SessionDiagnostics:
    """Everything measured during a session, in one reportable object."""

    verdict: str
    error_px: float
    sweep_samples: int
    sweep_yaw_range: float
    sweep_pitch_range: float
    points: List[PointDiagnostics]
    validation: List[PointDiagnostics]
    hx_range: Tuple[float, float]
    hy_range: Tuple[float, float]
    likely_cause: str
    fixed_head: bool = True
    region: Optional[Region] = None
    screen_size: Tuple[int, int] = (0, 0)
    #: what the point-level repair pass did, in the order it happened
    repairs: List[str] = field(default_factory=list)

    @property
    def hx_span(self) -> float:
        return self.hx_range[1] - self.hx_range[0]

    @property
    def hy_span(self) -> float:
        return self.hy_range[1] - self.hy_range[0]

    def feature_line(self) -> str:
        return (f"hx {self.hx_range[0]:.3f}–{self.hx_range[1]:.3f} "
                f"(span {self.hx_span:.3f})     "
                f"hy {self.hy_range[0]:.3f}–{self.hy_range[1]:.3f} "
                f"(span {self.hy_span:.3f})")

    def sweep_line(self) -> str:
        """Which head mode ran, and what the sweep covered if there was one."""
        if self.fixed_head:
            return ("head mode: fixed (head rest) — stage A skipped, "
                    "pose compensation off")
        return (f"head mode: free — sweep {self.sweep_samples} samples, "
                f"yaw {self.sweep_yaw_range:.1f}°, "
                f"pitch {self.sweep_pitch_range:.1f}°")

    def region_line(self) -> str:
        """Where this error applies — the measurement is only in-region."""
        if self.region is None:
            return "measured over: the whole screen"
        return f"measured over: {self.region.describe(self.screen_size)}"

    def console_report(self) -> str:
        """The full table, printed after every session (ASCII only).

        Windows consoles still default to code pages that cannot encode the
        typographic characters the on-screen version uses, so the report is
        transliterated on the way out.
        """
        out = ["[GazeKey] calibration diagnostics", f"  {self.sweep_line()}",
               f"  {self.region_line()}"]
        out.append("  calibration points   target        residual   kept/valid  retries")
        for point in self.points:
            residual = ("     -" if not np.isfinite(point.residual_px)
                        else f"{point.residual_px:7.1f} px")
            out.append(
                f"    {point.index:>2}  ({point.target[0]:5d},{point.target[1]:5d})"
                f"  {residual}   {point.kept:>3}/{point.collected:<3}"
                f"     {point.retries}"
                f"{'   re-collected' if point.repaired else ''}"
            )
        for line in self.repairs:
            out.append(f"    * {line}")
        out.append("  validation targets   target     ->  prediction      error")
        for point in self.validation:
            if point.prediction is None:
                out.append(f"    {point.index:>2}  ({point.target[0]:5d},"
                           f"{point.target[1]:5d})  ->  (no samples)")
                continue
            out.append(
                f"    {point.index:>2}  ({point.target[0]:5d},{point.target[1]:5d})"
                f"  ->  ({point.prediction[0]:6.0f},{point.prediction[1]:6.0f})"
                f"  {point.residual_px:7.1f} px"
            )
        out.append(f"  feature spread: {self.feature_line()}")
        measured = f" {self.error_px:.1f} px" if np.isfinite(self.error_px) else ""
        out.append(f"  verdict       : {self.verdict}{measured}")
        out.append(f"  most likely   : {self.likely_cause}")
        return to_ascii("\n".join(out))


class CalibrationSession:
    """Drives one calibration session from a stream of :class:`FrameFeatures`.

    Feed it every frame with :meth:`update`; poll :attr:`phase`,
    :meth:`current_target` and :attr:`sweep_progress` to render.
    """

    def __init__(
        self,
        screen_size: Tuple[int, int],
        camera_index: int = 0,
        fixed_head: bool = True,
        settle_ms: float = 700.0,
        collect_samples: int = 45,
        collect_max_s: float = 4.0,
        sweep_samples: int = 90,
        sweep_max_s: float = 8.0,
        max_retries: int = 2,
        rng: Optional[random.Random] = None,
        region: Optional[Region] = None,
    ) -> None:
        self.screen_size = screen_size
        self.camera_index = camera_index
        self.fixed_head = fixed_head
        #: the screen area this session fits and measures over (spec 5.6).
        #: Defaults to the whole screen, which is what calibration did before
        #: regions existed.
        self.region = region or full_screen_region(screen_size)
        self.settle_s = settle_ms / 1000.0
        self.collect_samples = max(MIN_SAMPLES_PER_POINT, int(collect_samples))
        self.collect_max_s = collect_max_s
        self.sweep_samples = sweep_samples
        self.sweep_max_s = sweep_max_s
        self.max_retries = max_retries
        self._rng = rng or random.Random()

        self.model = CalibrationModel(
            screen_size=screen_size, camera_index=camera_index
        )
        self.error_px: float = float("nan")
        self.verdict: str = ""
        self.results_note: str = ""
        self.sweep_failure: Optional[SweepFailure] = None
        self.refit_pass = False
        self._refit_done = False
        self._reset_all()

    # ------------------------------------------------------------------ setup
    def _reset_all(self) -> None:
        self.cal_targets: List[Tuple[int, int]] = list(self.region.grid())
        self._rng.shuffle(self.cal_targets)
        self.val_targets: List[Tuple[int, int]] = list(self.region.validation())
        self.cal_features: List[Optional[np.ndarray]] = [None] * len(self.cal_targets)
        self.val_features: List[Optional[np.ndarray]] = [None] * len(self.val_targets)
        self._retries: dict[Tuple[Phase, int], int] = {}
        #: (phase, index) -> (valid frames collected, frames kept after IQR)
        self._stats: dict[Tuple[Phase, int], Tuple[int, int]] = {}
        self._queue: List[int] = []
        self._stage_size: int = 0
        self._sweep: List[np.ndarray] = []
        self._sweep_start: Optional[float] = None
        self._target_start: Optional[float] = None
        self._samples: List[np.ndarray] = []
        self._now: float = 0.0
        self._fitted = False

        #: running commentary, for the console (see :meth:`take_messages`)
        self.messages: List[str] = []
        self._message_cursor = 0
        #: indices of points the repair pass re-collected
        self.repaired: List[int] = []
        self._repair_done = False
        #: idx -> (feature, residual, stats) kept while a repair is in flight,
        #: so the worse of the two collections can be thrown away afterwards
        self._repair_before: dict = {}

        if self.fixed_head:
            # The head is supported, so pose does not vary: the compensator
            # keeps its zero sensitivities (compensate() is then the identity)
            # and stage A is skipped entirely.
            self.model.comp.s[:] = 0.0
            self.model.comp.ref_yaw = 0.0
            self.model.comp.ref_pitch = 0.0
            self.model.comp.ok = True
            self._begin_stage(Phase.POINTS, range(len(self.cal_targets)))
        else:
            self.phase = Phase.SWEEP

    def restart(self) -> None:
        """Start over from the head-sweep with a fresh model."""
        self.model = CalibrationModel(
            screen_size=self.screen_size, camera_index=self.camera_index
        )
        self.error_px = float("nan")
        self.verdict = ""
        self.results_note = ""
        self.sweep_failure = None
        self.refit_pass = False
        self._refit_done = False
        self._reset_all()

    def retry_sweep(self) -> None:
        """Repeat the head-sweep after it was rejected (free-head mode only)."""
        if self.fixed_head:
            return
        self.sweep_failure = None
        self._sweep = []
        self._sweep_start = None
        self.phase = Phase.SWEEP

    # ------------------------------------------------------------------- feed
    def update(self, features: FrameFeatures) -> None:
        """Consume one frame. Safe to call in any phase."""
        self._now = features.timestamp or time.time()
        if self.phase is Phase.SWEEP:
            self._update_sweep(features)
        elif self.phase in (Phase.POINTS, Phase.VALIDATION):
            self._update_target(features)

    # ------------------------------------------------------------- stage A
    @property
    def sweep_progress(self) -> float:
        """0..1 — how much of the head-sweep has been collected."""
        by_count = len(self._sweep) / max(self.sweep_samples, 1)
        if self._sweep_start is None:
            return 0.0
        by_time = (self._now - self._sweep_start) / max(self.sweep_max_s, 1e-6)
        return float(min(1.0, max(by_count, by_time)))

    @property
    def sweep_pose_range(self) -> Tuple[float, float]:
        """Yaw/pitch range covered so far, in degrees (for live UI feedback)."""
        if len(self._sweep) < 2:
            return 0.0, 0.0
        arr = np.asarray(self._sweep)
        return float(np.ptp(arr[:, 2])), float(np.ptp(arr[:, 3]))

    @property
    def sweep_failure_message(self) -> Tuple[str, str]:
        """``(headline, guidance)`` explaining a rejected sweep."""
        return SWEEP_FAILURE_TEXT[self.sweep_failure or SweepFailure.UNKNOWN]

    def _update_sweep(self, features: FrameFeatures) -> None:
        if self._sweep_start is None:
            self._sweep_start = self._now
        if features.valid:
            self._sweep.append(features.vector())
        elapsed = self._now - self._sweep_start
        if len(self._sweep) >= self.sweep_samples or elapsed >= self.sweep_max_s:
            self._finish_sweep()

    def _finish_sweep(self) -> None:
        samples = (
            np.asarray(self._sweep, dtype=float)
            if self._sweep
            else np.zeros((0, 4), dtype=float)
        )
        accepted = (
            len(samples) >= MIN_SWEEP_SAMPLES
            and self.model.fit_pose_compensation(samples)
        )
        if accepted:
            self._begin_stage(Phase.POINTS, list(range(len(self.cal_targets))))
            return
        self.sweep_failure = self._diagnose_sweep(samples)
        self.phase = Phase.SWEEP_FAILED

    @staticmethod
    def _diagnose_sweep(samples: np.ndarray) -> SweepFailure:
        if len(samples) < MIN_SWEEP_SAMPLES:
            return SweepFailure.NO_FACE
        yaw_range, pitch_range = np.ptp(samples[:, 2]), np.ptp(samples[:, 3])
        if max(yaw_range, pitch_range) < MIN_SWEEP_POSE_RANGE_DEG:
            return SweepFailure.NO_MOTION
        return SweepFailure.UNKNOWN

    # ------------------------------------------------------------- stage B / validation
    def current_target(self) -> Optional[TargetState]:
        """The target being shown, or ``None`` outside a target stage."""
        if self.phase not in (Phase.POINTS, Phase.VALIDATION) or not self._queue:
            return None
        idx = self._queue[0]
        targets = self._targets_for(self.phase)
        elapsed = 0.0 if self._target_start is None else self._now - self._target_start
        settling = elapsed < self.settle_s
        settle_progress = min(1.0, elapsed / max(self.settle_s, 1e-6))
        collecting = max(0.0, elapsed - self.settle_s)
        progress = 0.0 if settling else min(1.0, max(
            len(self._samples) / max(self.collect_samples, 1),
            collecting / max(self.collect_max_s, 1e-6),
        ))
        done = self._stage_size - len(self._queue)
        return TargetState(
            position=targets[idx],
            index=done + 1,
            total=self._stage_size,
            settling=settling,
            settle_progress=settle_progress,
            progress=progress,
            retry=self._retries.get((self.phase, idx), 0),
            stage=self.phase,
            samples=len(self._samples),
        )

    def _targets_for(self, phase: Phase) -> List[Tuple[int, int]]:
        return self.cal_targets if phase is Phase.POINTS else self.val_targets

    def _features_for(self, phase: Phase) -> List[Optional[np.ndarray]]:
        return self.cal_features if phase is Phase.POINTS else self.val_features

    def _begin_stage(self, phase: Phase, queue: Sequence[int]) -> None:
        self.phase = phase
        self._queue = list(queue)
        self._stage_size = len(self._queue)
        self._reset_target()

    def _reset_target(self) -> None:
        self._target_start = None
        self._samples = []

    def _update_target(self, features: FrameFeatures) -> None:
        if not self._queue:
            return
        if self._target_start is None:
            self._target_start = self._now
        elapsed = self._now - self._target_start
        if elapsed < self.settle_s:
            return  # saccade settling — deliberately discard these frames

        if features.valid:
            self._samples.append(features.vector())
        collecting = elapsed - self.settle_s
        if (len(self._samples) >= self.collect_samples
                or collecting >= self.collect_max_s):
            self._finish_target()

    def _finish_target(self) -> None:
        idx = self._queue[0]
        samples = (
            np.asarray(self._samples, dtype=float)
            if self._samples
            else np.zeros((0, 4), dtype=float)
        )
        aggregate, kept = (aggregate_point(samples) if len(samples) else (None, 0))
        key = (self.phase, idx)

        if aggregate is None:
            self._retries[key] = self._retries.get(key, 0) + 1
            if self._retries[key] <= self.max_retries:
                self._reset_target()      # repeat this point automatically
                return
            aggregate = self._salvage(samples)   # "continue with what exists"
            kept = len(reject_outliers(samples)) if len(samples) else 0

        self._stats[key] = (len(samples), kept)
        self._features_for(self.phase)[idx] = aggregate
        self._queue.pop(0)
        self._reset_target()
        if not self._queue:
            self._finish_stage()

    @staticmethod
    def _salvage(samples: np.ndarray) -> Optional[np.ndarray]:
        """Best effort aggregate after the retries are used up."""
        if not len(samples):
            return None
        clean = reject_outliers(samples)
        return np.median(clean if len(clean) else samples, axis=0)

    # ---------------------------------------------------------------- fitting
    def _usable(self, phase: Phase):
        features = self._features_for(phase)
        targets = self._targets_for(phase)
        indices = [i for i, f in enumerate(features) if f is not None]
        return (
            indices,
            np.array([features[i] for i in indices], dtype=float),
            np.array([targets[i] for i in indices], dtype=float),
        )

    def _finish_stage(self) -> None:
        if self.phase is Phase.POINTS:
            self._fit_and_validate()
        else:
            self._evaluate()

    def _fit_and_validate(self) -> None:
        # Returning from a repair pass? Settle it *before* refitting: the
        # current model is the referee that both collections are judged by.
        if self._repair_before:
            self._resolve_repair()

        indices, features, targets = self._usable(Phase.POINTS)
        if len(indices) < MIN_USABLE_POINTS:
            self._fail_out(
                "Too few calibration points produced usable data "
                f"({len(indices)} of {len(self.cal_targets)})."
            )
            return
        self.model.fit(features, targets)
        self._fitted = True

        if not self._repair_done:
            suspect = self._suspect_points()
            if suspect:
                self._start_repair(suspect)
                return

        self.refit_pass = False
        self._begin_stage(Phase.VALIDATION, list(range(len(self.val_targets))))

    # -------------------------------------------------------- point repair
    def take_messages(self) -> List[str]:
        """Console lines produced since the last call (never re-reported)."""
        pending = self.messages[self._message_cursor:]
        self._message_cursor = len(self.messages)
        return pending

    def _residuals(self):
        """``(indices, residuals)`` of the fitted points, in original order."""
        indices, features, targets = self._usable(Phase.POINTS)
        if not len(indices):
            return [], np.zeros(0)
        predicted = self.model.predict_many(features)
        return indices, np.linalg.norm(predicted - targets, axis=1)

    def _residual_of(self, feature, target) -> float:
        """How far this one aggregate lands from its target, in px."""
        if feature is None:
            return float("inf")
        px, py = self.model.predict(np.asarray(feature, dtype=float))
        return float(np.hypot(px - target[0], py - target[1]))

    def _suspect_points(self) -> List[int]:
        """Points that fit far worse than the rest — one glance away, usually.

        Empty unless one or two points stand out: three or more is systemic,
        and re-collecting would only spend the user's patience.
        """
        indices, residuals = self._residuals()
        if len(indices) < MIN_USABLE_POINTS:
            return []
        median = float(np.median(residuals))
        suspect = [
            indices[i] for i, residual in enumerate(residuals)
            if residual > REPAIR_RESIDUAL_RATIO * median
            and residual > REPAIR_RESIDUAL_MIN_PX
        ]
        return suspect if 1 <= len(suspect) <= MAX_REPAIRED_POINTS else []

    def _start_repair(self, suspect: List[int]) -> None:
        """Re-collect the suspect points once, then refit and validate."""
        self._repair_done = True                 # only ever one repair pass
        indices, residuals = self._residuals()
        by_index = dict(zip(indices, residuals))
        median = float(np.median(residuals))

        for idx in suspect:
            self._repair_before[idx] = (
                self.cal_features[idx],
                float(by_index[idx]),
                self._stats.get((Phase.POINTS, idx)),
            )
            self.messages.append(
                f"point {idx + 1} fit poorly "
                f"({by_index[idx]:.0f} px vs {median:.0f} px median) "
                f"- re-collecting"
            )
        self.refit_pass = True                   # the screen says "Improving"
        self._begin_stage(Phase.POINTS, suspect)

    def _resolve_repair(self) -> None:
        """Keep whichever collection fits better, and never repeat the pass.

        Both candidates are measured against the *same* model — the fit that
        flagged the point — so the comparison is fair, and a re-collection that
        went worse (or produced nothing) is simply discarded.
        """
        before, self._repair_before = self._repair_before, {}
        for idx, (old_feature, old_residual, old_stats) in before.items():
            target = self.cal_targets[idx]
            new_residual = self._residual_of(self.cal_features[idx], target)
            if new_residual <= old_residual:
                self.messages.append(
                    f"point {idx + 1} re-collected: {new_residual:.0f} px "
                    f"(was {old_residual:.0f} px)"
                )
            else:
                self.cal_features[idx] = old_feature
                if old_stats is not None:
                    self._stats[(Phase.POINTS, idx)] = old_stats
                self.messages.append(
                    f"point {idx + 1} re-collected no better "
                    f"({new_residual:.0f} px vs {old_residual:.0f} px) "
                    f"- keeping the first"
                )
            self.repaired.append(idx)

    def _evaluate(self) -> None:
        indices, features, targets = self._usable(Phase.VALIDATION)
        if not len(indices):
            self._fail_out("No usable validation samples were collected.")
            return

        self.error_px, self.verdict = self.model.validate(features, targets)
        if self.verdict == "FAIL" and not self._refit_done:
            self._refit_done = True
            self.refit_pass = True
            self._begin_stage(Phase.POINTS, self._worst_calibration_points())
            return
        self.phase = Phase.RESULTS

    def _worst_calibration_points(self, k: int = 2) -> List[int]:
        """Original indices of the k worst-fitting calibration points."""
        indices, features, targets = self._usable(Phase.POINTS)
        worst = self.model.worst_fit_points(features, targets, k=min(k, len(indices)))
        return [indices[i] for i in worst]

    def _fail_out(self, note: str) -> None:
        self.error_px = float("inf")
        self.verdict = "FAIL"
        self.results_note = note
        self.phase = Phase.RESULTS

    # ---------------------------------------------------------------- results
    @property
    def is_finished(self) -> bool:
        return self.phase is Phase.RESULTS

    @property
    def should_save(self) -> bool:
        """PASS and MARGINAL models are worth keeping; FAIL is not."""
        return self.verdict in ("PASS", "MARGINAL")

    @property
    def verdict_text(self) -> Tuple[str, str]:
        """``(headline, guidance)`` for the results screen."""
        if self.verdict == "PASS":
            return ("Good", "Calibration passed. You are ready to go.")
        if self.verdict == "MARGINAL":
            return (
                "Usable",
                "Accuracy is a little low — prefer a larger keyboard, or "
                "recalibrate for a better result.",
            )
        return (
            "Too low",
            self.results_note
            or "Face the camera, improve the lighting, sit about 60 cm away "
               "and keep your head reasonably still, then try again.",
        )

    # ------------------------------------------------------------ diagnostics
    def _point_diagnostics(self, phase: Phase) -> List[PointDiagnostics]:
        features = self._features_for(phase)
        targets = self._targets_for(phase)
        out: List[PointDiagnostics] = []
        for idx, (feature, target) in enumerate(zip(features, targets)):
            collected, kept = self._stats.get((phase, idx), (0, 0))
            point = PointDiagnostics(
                index=idx + 1,
                target=target,
                collected=collected,
                kept=kept,
                retries=self._retries.get((phase, idx), 0),
                repaired=(phase is Phase.POINTS and idx in self.repaired),
            )
            if feature is not None:
                point.hx, point.hy = float(feature[0]), float(feature[1])
                if self._fitted:
                    px, py = self.model.predict(feature)
                    point.prediction = (px, py)
                    point.residual_px = float(
                        np.hypot(px - target[0], py - target[1])
                    )
            out.append(point)
        return out

    def diagnostics(self) -> SessionDiagnostics:
        """Everything measured this session — shown on screen and printed."""
        points = self._point_diagnostics(Phase.POINTS)
        validation = self._point_diagnostics(Phase.VALIDATION)
        aggregated = [f for f in self.cal_features if f is not None]
        if aggregated:
            block = np.asarray(aggregated, dtype=float)
            hx_range = (float(block[:, 0].min()), float(block[:, 0].max()))
            hy_range = (float(block[:, 1].min()), float(block[:, 1].max()))
        else:
            hx_range = hy_range = (float("nan"), float("nan"))
        yaw_range, pitch_range = self.sweep_pose_range
        diagnostics = SessionDiagnostics(
            verdict=self.verdict or "INCOMPLETE",
            error_px=self.error_px,
            sweep_samples=len(self._sweep),
            sweep_yaw_range=yaw_range,
            sweep_pitch_range=pitch_range,
            points=points,
            validation=validation,
            hx_range=hx_range,
            hy_range=hy_range,
            likely_cause="",
            fixed_head=self.fixed_head,
            region=self.region,
            screen_size=self.screen_size,
            repairs=list(self.messages),
        )
        diagnostics.likely_cause = self._likely_cause(diagnostics)
        return diagnostics

    def _likely_cause(self, d: SessionDiagnostics) -> str:
        """One line naming the most probable limit on accuracy."""
        residuals = [p.residual_px for p in d.points if np.isfinite(p.residual_px)]
        val_errors = [p.residual_px for p in d.validation
                      if np.isfinite(p.residual_px)]
        collected = sum(p.collected for p in d.points)
        kept = sum(p.kept for p in d.points)
        rejected_ratio = 1.0 - (kept / collected) if collected else 0.0
        repeated = sum(1 for p in d.points if p.retries)

        if not residuals:
            return ("No calibration point produced usable data — check that "
                    "your face is visible and well lit.")

        if self.verdict == "PASS":
            return (f"Consistent across the screen (best {min(residuals):.0f} px, "
                    f"worst {max(residuals):.0f} px) — no action needed.")

        pose_span = max(d.sweep_yaw_range, d.sweep_pitch_range)
        if not self.fixed_head and pose_span < 12.0:
            return (f"The head sweep only covered {pose_span:.0f}° of rotation — "
                    f"repeat it and turn your head slowly through a much wider arc.")

        if np.isfinite(d.hx_span) and (d.hx_span < 0.08 or d.hy_span < 0.05):
            return (f"Your iris barely moved between targets "
                    f"(hx span {d.hx_span:.3f}, hy span {d.hy_span:.3f}) — "
                    f"sit closer to the screen, or move the camera to eye height.")

        if rejected_ratio > 0.25 or repeated >= 2:
            return (f"{rejected_ratio * 100:.0f}% of samples were discarded and "
                    f"{repeated} point(s) had to repeat — likely blinking or dim, "
                    f"uneven lighting on your face.")

        if val_errors and float(np.mean(val_errors)) > 2.0 * float(
            np.mean(residuals)
        ) + 20.0:
            drifted = (f"The fit is tight on the calibration points "
                       f"(mean {np.mean(residuals):.0f} px) but loose on fresh "
                       f"ones (mean {np.mean(val_errors):.0f} px) — your head "
                       f"most likely shifted between the two stages.")
            if self.fixed_head:
                return (drifted + " Check the head rest is actually holding, or "
                        "recalibrate with --with-head-sweep.")
            return drifted

        worst = max(d.points, key=lambda p: p.residual_px
                    if np.isfinite(p.residual_px) else -1.0)
        median = float(np.median(residuals))
        if np.isfinite(worst.residual_px) and worst.residual_px > 3.0 * median:
            return (f"Point {worst.index} at {worst.target} fits far worse than the "
                    f"rest ({worst.residual_px:.0f} px vs {median:.0f} px median) — "
                    f"you probably looked away while it was showing.")

        return ("No single dominant cause — accuracy is limited by overall noise. "
                "Improve the front lighting, sit about 60 cm away, and try again.")

    def save(self, path: str) -> None:
        """Persist the fitted model and the region it was fitted over (5.4).

        The region rides alongside as an extra key: ``CalibrationModel.save``
        is verified core and must not learn about regions.
        """
        self.model.screen_size = self.screen_size
        self.model.camera_index = self.camera_index
        self.model.created_at = time.time()
        self.model.save(path)
        attach_region(path, self.region)
