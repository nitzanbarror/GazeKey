"""GazeKey — ``--feature-lab``: what is wrong with the vertical channel?

**Read-only.** Nothing here feeds calibration, changes a model or writes a
file. It exists to answer one measured question with evidence instead of
argument.

The question
------------
Across every sitting, ``hx`` spans 0.13–0.22 between the left and right edges
of the calibration region while ``hy`` spans only 0.02–0.06 between the top and
the bottom. The vertical channel carries roughly a fifth of the horizontal
one's signal, and that — not the fitting, not the procedure — is the ceiling on
vertical accuracy.

The hypothesis is that the asymmetry is **baked into the feature definitions**
(``gaze/features.py``, verified core, untouched by this module)::

    hx = (iris_cx - inner_corner_x) / (outer_corner_x - inner_corner_x)
    hy = (iris_cy - upper_lid_y)    / (lower_lid_y   - upper_lid_y)

``hx`` is referenced to the **eye corners**, which are as good as skull-fixed.
``hy`` is referenced to the **eyelids**, which follow vertical gaze: the upper
lid rides up as you look up and down as you look down, at something like
85–90% of the eye's own rotation. Both the numerator's origin *and* the
denominator's scale therefore chase the iris, and a ratio whose reference moves
with its subject reports almost nothing.

The four candidates below are a 2×2 decomposition of exactly that claim, all
computed from the same frames so the comparison is like-for-like:

===============  =====================  ============================
candidate        numerator referenced   denominator
===============  =====================  ============================
**a** ``hy``     upper lid              lid-to-lid (moves)
**b** corner     inter-corner line      inter-corner distance (fixed)
**c** aperture   — (lid-to-lid itself)  inter-corner distance (fixed)
**d** lid/width  upper lid              inter-corner distance (fixed)
===============  =====================  ============================

Read as a table rather than a race:

* **b ≫ a** — the reference is the problem, and a corner-referenced vertical
  feature is the fix;
* **d ≈ b** — only the *denominator* was hurting; the upper lid is a usable
  origin once it is not also the scale;
* **d ≈ a** — the *numerator* is the problem: the lid tracks the iris so
  closely that its offset carries nothing;
* **c large** — direct confirmation that the lid moves with gaze at all, which
  is the mechanism the whole hypothesis rests on.

Everything is measured in **pixels**, never in MediaPipe's normalised
coordinates: normalised x and y are divided by different numbers (640 and 480
on this camera), so any quantity mixing the two axes — a perpendicular
distance, a length, an angle — is distorted in that space. The verified
``hx``/``hy`` are ratios *within* one axis and so are immune, which is worth
knowing before anyone "fixes" them.

Signal alone is not the answer either, so every candidate is reported with the
**within-target IQR** and the ratio ``span / IQR``: a feature that moves twice
as far but is four times noisier is worse, and only the ratio says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gaze.features import (
    L_INNER,
    L_IRIS,
    L_LO,
    L_OUTER,
    L_UP,
    R_INNER,
    R_IRIS,
    R_LO,
    R_OUTER,
    R_UP,
)
from gaze.region import Region
from gaze.setup_check import SetupCheckResult, SetupCheckSession, SetupPhase

# --------------------------------------------------- did the eyes do the work?
#: Anthropometric constants, both stable across adults to a few percent: the
#: eyeball's radius and the palpebral fissure's width (corner to corner). Their
#: **ratio** is what candidate b is measured in, so it needs no camera
#: calibration and no distance estimate — b moves by
#: ``(R/W) * sin(theta)``, i.e. about 0.007 per degree of eye rotation.
EYEBALL_RADIUS_MM = 12.0
FISSURE_WIDTH_MM = 30.0
EYE_GEOMETRY_RATIO = EYEBALL_RADIUS_MM / FISSURE_WIDTH_MM

#: Assumptions behind the *expected* angle only, and printed with the result so
#: they can be argued with: the distance the app tells the user to sit at, and
#: the development machine's panel (14" 16:9).
VIEWING_DISTANCE_MM = 600.0
SCREEN_HEIGHT_MM = 174.0

#: head pitch travels alongside the candidates, as context rather than as a
#: candidate — it never appears in the comparison table
PITCH_KEY = "_pitch"

#: head share below this is an eyes-only sitting, above the second is
#: head-driven; between them the evidence is mixed
EYES_ONLY_BELOW = 0.25
HEAD_DRIVEN_ABOVE = 0.60

#: less than this fraction of the required angle accounted for by eyes plus
#: head means the dots were probably not looked at at all
ACCOUNTED_MIN = 0.5


def eye_rotation_deg(b_span: float) -> float:
    """Degrees of eyeball rotation behind a given span of candidate b."""
    if not np.isfinite(b_span):
        return float("nan")
    return float(np.degrees(np.arcsin(
        np.clip(abs(b_span) / EYE_GEOMETRY_RATIO, -1.0, 1.0))))


def subtended_deg(separation_px: float, screen_px: float,
                  screen_mm: float = SCREEN_HEIGHT_MM,
                  viewing_mm: float = VIEWING_DISTANCE_MM) -> float:
    """How far apart the two dots are, in degrees of gaze at the eye."""
    if screen_px <= 0 or viewing_mm <= 0:
        return float("nan")
    millimetres = separation_px / screen_px * screen_mm
    return float(2.0 * np.degrees(np.arctan(millimetres / 2.0 / viewing_mm)))


@dataclass
class HeadMotion:
    """Who actually pointed at the dots — the eyes, or the head?

    Without this a table is uninterpretable, and worse, quietly misleading. If
    the head does the pointing then the eyeball barely rotates, every
    candidate's span collapses toward zero, and what survives is whatever
    responds to **head pitch** rather than to gaze. Candidate c is the trap:
    pitching the head foreshortens the eye vertically, so the apparent lid
    aperture shrinks by roughly ``cos(pitch)`` while the eye width, being
    perpendicular to the pitch axis, does not — c then scores well as a pure
    head-pose proxy that would be worthless in a fixed-head session.

    Candidates b and d are foreshortened the same way (a vertical displacement
    over a horizontal length). The baseline ``hy`` is the one that is *not*:
    numerator and denominator are both vertical, so the compression cancels.
    """

    eye_deg: float
    head_deg: float
    required_deg: float
    viewing_mm: float = VIEWING_DISTANCE_MM
    screen_mm: float = SCREEN_HEIGHT_MM

    @property
    def total_deg(self) -> float:
        return self.eye_deg + self.head_deg

    @property
    def head_fraction(self) -> float:
        """Share of the measured pointing done by the head. No geometry needed
        for this one: both angles are measured, not assumed."""
        total = self.total_deg
        if not np.isfinite(total) or total <= 1e-6:
            return float("nan")
        return self.head_deg / total

    @property
    def accounted(self) -> float:
        """Measured travel over the travel the dots demanded."""
        if not np.isfinite(self.required_deg) or self.required_deg <= 1e-6:
            return float("nan")
        return self.total_deg / self.required_deg

    @property
    def label(self) -> str:
        share = self.head_fraction
        if not np.isfinite(share):
            return "unclear"
        if np.isfinite(self.accounted) and self.accounted < ACCOUNTED_MIN:
            return "not-looking"
        if share < EYES_ONLY_BELOW:
            return "eyes-only"
        if share > HEAD_DRIVEN_ABOVE:
            return "head-driven"
        return "mixed"

    @property
    def trustworthy(self) -> bool:
        """Is this sitting evidence about *features* at all?"""
        return self.label == "eyes-only"

    @property
    def discredits_the_table(self) -> bool:
        """Is there positive evidence that the rows are not about gaze?

        Deliberately not the negation of :attr:`trustworthy`: "unclear" means
        the head motion could not be measured, and an absent measurement is
        not grounds for overruling the table.
        """
        return self.label in ("head-driven", "not-looking")

    def lines(self) -> List[str]:
        share = self.head_fraction
        if not np.isfinite(share):
            return ["  head motion   : not measured - no head pose in these "
                    "frames, so who did the pointing is unknown"]
        out = [
            f"  head motion   : {self.label.upper()} - eyes "
            f"{self.eye_deg:.1f} deg, head {self.head_deg:.1f} deg "
            f"({share * 100:.0f}% of the pointing done by the head)",
            f"  expected      : the dots subtend {self.required_deg:.1f} deg "
            f"at {self.viewing_mm:.0f} mm on a {self.screen_mm:.0f} mm panel; "
            f"eyes+head covered {self.total_deg:.1f} deg",
        ]
        if self.label == "not-looking":
            out.append("  WARNING       : neither the eyes nor the head covered "
                       "the distance between the dots - they were probably not "
                       "looked at, and no row above means anything.")
        elif self.label == "head-driven":
            out.append("  WARNING       : the head did the pointing, so this "
                       "table is not evidence about features. Every span is "
                       "compressed by the eyeball barely rotating, and c is "
                       "actively misleading - pitching the head foreshortens "
                       "the eye vertically, so lid aperture tracks head pose "
                       "rather than gaze. b and d foreshorten the same way; "
                       "hy alone is immune (both its terms are vertical). "
                       "Rest the head properly and re-run.")
        elif self.label == "mixed":
            out.append("  CAUTION       : the head did a substantial share, so "
                       "the spans are part gaze and part head pose - treat the "
                       "ranking as provisional and re-run with the head rested.")
        return out


#: one eye's landmark indices: (iris, corner, corner, upper lid, lower lid).
#: The two corners are handed over unordered on purpose — every candidate here
#: sorts them along the image x-axis, so neither the "inner"/"outer" labelling
#: nor which side of the face the eye is on can flip a sign. Averaging two eyes
#: whose signs disagree cancels the signal, which is a defect this project has
#: already been bitten by once on the horizontal axis.
EYES = (
    (R_IRIS, R_INNER, R_OUTER, R_UP, R_LO),
    (L_IRIS, L_INNER, L_OUTER, L_UP, L_LO),
)


# --------------------------------------------------------------- the geometry
def _eye_frame(px: np.ndarray, corner_a: int, corner_b: int):
    """``(origin, along, down, width)`` for one eye, in pixels.

    ``along`` runs left→right across the eye in **image** order and ``down`` is
    its perpendicular, so "more positive" always means "lower on screen" for
    both eyes regardless of which corner MediaPipe calls inner.
    """
    a, b = px[corner_a][:2], px[corner_b][:2]
    origin, far = (a, b) if a[0] <= b[0] else (b, a)
    vector = far - origin
    width = float(np.hypot(*vector))
    if width < 1e-9:
        return None
    along = vector / width
    down = np.array([-along[1], along[0]])      # image y grows downward
    return origin, along, down, width


def _iris_centre(px: np.ndarray, iris: Sequence[int]) -> np.ndarray:
    return px[list(iris), :2].mean(axis=0)


def _per_eye(px: np.ndarray, measure: Callable) -> float:
    """Average a per-eye measurement over both eyes, NaN if either is unusable."""
    values = []
    for iris, corner_a, corner_b, up, lo in EYES:
        frame = _eye_frame(px, corner_a, corner_b)
        if frame is None:
            return float("nan")
        values.append(measure(px, frame, iris, up, lo))
    return float(np.mean(values)) if all(np.isfinite(values)) else float("nan")


def iris_vs_corner_line(px: np.ndarray) -> float:
    """**Candidate b** — the vertical analogue of ``hx``.

    Signed distance from the iris centre to the line joining the two eye
    corners, normalised by the distance between them. Both reference and scale
    are corner-based, so nothing in it moves with the eyelid.
    """
    def measure(px, frame, iris, up, lo):
        origin, _along, down, width = frame
        return float(np.dot(_iris_centre(px, iris) - origin, down) / width)

    return _per_eye(px, measure)


def lid_aperture(px: np.ndarray) -> float:
    """**Candidate c** — how open the eye is, normalised by eye width.

    Not a gaze feature so much as the *mechanism*: if this moves between
    looking up and looking down, the eyelid really is chasing the iris and the
    baseline's reference is compromised.
    """
    def measure(px, frame, iris, up, lo):
        _origin, _along, _down, width = frame
        return float(np.hypot(*(px[lo][:2] - px[up][:2])) / width)

    return _per_eye(px, measure)


def lid_to_iris(px: np.ndarray) -> float:
    """**Candidate d** — the baseline's numerator over a fixed denominator.

    Distance from the upper lid down to the iris centre, normalised by eye
    width instead of by the (moving) lid-to-lid distance. Isolates how much of
    the baseline's collapse is the denominator's fault.
    """
    def measure(px, frame, iris, up, lo):
        origin, _along, down, width = frame
        upper = np.dot(px[up][:2] - origin, down)
        return float((np.dot(_iris_centre(px, iris) - origin, down) - upper) / width)

    return _per_eye(px, measure)


@dataclass(frozen=True)
class Candidate:
    """One vertical feature under test."""

    key: str
    label: str
    #: ``None`` for the baseline, which is read from the verified core's output
    #: rather than recomputed here — the point of comparison is what
    #: calibration actually consumes today.
    compute: Optional[Callable[[np.ndarray], float]]

    def value(self, features, px) -> float:
        if self.compute is None:
            return float(features.hy)
        return self.compute(px)


CANDIDATES: Tuple[Candidate, ...] = (
    Candidate("a_hy", "hy  (baseline, verified core)", None),
    Candidate("b_corner", "iris vs corner line / width", iris_vs_corner_line),
    Candidate("c_aperture", "lid aperture / width", lid_aperture),
    Candidate("d_lid_iris", "upper lid to iris / width", lid_to_iris),
)


# ------------------------------------------------------------------ statistics
def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 4:
        return float("nan")
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


@dataclass
class Visit:
    """One showing of one dot: every candidate's samples while it was up."""

    target_index: int                 # 0 = top, 1 = bottom
    cycle: int
    values: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, key: str, value: float) -> None:
        if np.isfinite(value):
            self.values.setdefault(key, []).append(value)

    @property
    def samples(self) -> int:
        return max((len(v) for v in self.values.values()), default=0)


@dataclass
class CandidateReport:
    """What one candidate measured, across the whole run."""

    key: str
    label: str
    top: float
    bottom: float
    span: float
    iqr: float
    per_cycle_spans: List[float]

    @property
    def snr(self) -> float:
        """Span in units of its own within-target noise — the figure of merit."""
        if not np.isfinite(self.span) or not self.iqr or not np.isfinite(self.iqr):
            return float("nan")
        return self.span / self.iqr

    @property
    def stability(self) -> float:
        """Spread of the per-cycle spans, as a fraction of the pooled span.

        A candidate whose span is different every cycle is not measuring what
        it claims to, however good its average looks.
        """
        usable = [s for s in self.per_cycle_spans if np.isfinite(s)]
        if len(usable) < 2 or not self.span:
            return float("nan")
        return float(np.std(usable) / abs(self.span))


class FeatureLabReport:
    """The comparison table, and the one-line reading of it."""

    def __init__(self, visits: List[Visit], baseline_key: str = "a_hy",
                 separation_px: float = 0.0, screen_px: float = 0.0,
                 screen_mm: float = SCREEN_HEIGHT_MM,
                 viewing_mm: float = VIEWING_DISTANCE_MM) -> None:
        self.visits = visits
        self.baseline_key = baseline_key
        self.separation_px = separation_px
        self.screen_px = screen_px
        self.screen_mm = screen_mm
        self.viewing_mm = viewing_mm
        self.candidates = [self._report(c) for c in CANDIDATES]

    # ------------------------------------------------------------ head motion
    def head_motion(self) -> HeadMotion:
        """Eye rotation from candidate b, head rotation from the pose pitch.

        Two independent measurements — one from the iris against the eye
        corners, one from ``solvePnP`` — so they corroborate rather than
        assume each other. Only the *expected* angle needs geometry.
        """
        corner = self.find("b_corner")
        pitch_top = _median(self._pooled(PITCH_KEY, 0))
        pitch_bottom = _median(self._pooled(PITCH_KEY, 1))
        return HeadMotion(
            eye_deg=eye_rotation_deg(corner.span if corner else float("nan")),
            head_deg=abs(pitch_bottom - pitch_top),
            required_deg=subtended_deg(self.separation_px, self.screen_px,
                                       self.screen_mm, self.viewing_mm),
            viewing_mm=self.viewing_mm,
            screen_mm=self.screen_mm,
        )

    # ------------------------------------------------------------- statistics
    def _pooled(self, key: str, target_index: int) -> List[float]:
        return [v for visit in self.visits if visit.target_index == target_index
                for v in visit.values.get(key, [])]

    def _report(self, candidate: Candidate) -> CandidateReport:
        top, bottom = self._pooled(candidate.key, 0), self._pooled(candidate.key, 1)
        iqrs = [_iqr(visit.values.get(candidate.key, [])) for visit in self.visits]
        usable = [q for q in iqrs if np.isfinite(q)]
        return CandidateReport(
            key=candidate.key,
            label=candidate.label,
            top=_median(top),
            bottom=_median(bottom),
            span=abs(_median(bottom) - _median(top)),
            iqr=float(np.median(usable)) if usable else float("nan"),
            per_cycle_spans=self._per_cycle_spans(candidate.key),
        )

    def _per_cycle_spans(self, key: str) -> List[float]:
        spans = []
        for cycle in sorted({visit.cycle for visit in self.visits}):
            by_index = {
                visit.target_index: _median(visit.values.get(key, []))
                for visit in self.visits if visit.cycle == cycle
            }
            if len(by_index) == 2:
                spans.append(abs(by_index[1] - by_index[0]))
        return spans

    def find(self, key: str) -> Optional[CandidateReport]:
        return next((c for c in self.candidates if c.key == key), None)

    @property
    def baseline(self) -> Optional[CandidateReport]:
        return self.find(self.baseline_key)

    @property
    def samples(self) -> int:
        return sum(visit.samples for visit in self.visits)

    @property
    def cycles(self) -> int:
        return len({visit.cycle for visit in self.visits})

    def winner(self) -> Optional[CandidateReport]:
        """The best candidate by SNR that is not the baseline itself."""
        rivals = [c for c in self.candidates
                  if c.key != self.baseline_key and np.isfinite(c.snr)]
        return max(rivals, key=lambda c: c.snr) if rivals else None

    def improvement(self) -> float:
        """The winner's SNR as a multiple of the baseline's."""
        baseline, winner = self.baseline, self.winner()
        if baseline is None or winner is None or not baseline.snr:
            return float("nan")
        return winner.snr / baseline.snr

    # ---------------------------------------------------------------- report
    def lines(self) -> List[str]:
        """The console table (ASCII only, for any Windows code page).

        **Read the ``span/IQR`` column, not the ``span`` column.** Each
        candidate normalises by a different length — the baseline by the lid
        aperture, the rest by the eye width — so their raw spans are in
        different units and cannot be ranked against each other. Only the
        dimensionless ratio can.
        """
        if not self.samples:
            return ["[feature-lab] no usable frames - nothing to compare."]
        out = [
            f"[feature-lab] vertical feature comparison - {self.cycles} cycle(s), "
            f"{len(self.visits)} visit(s), {self.samples} samples",
            "  candidate                        top      bottom      span"
            "       IQR   span/IQR   vs base",
        ]
        baseline = self.baseline
        for report in self.candidates:
            relative = ("    -" if report.key == self.baseline_key
                        or baseline is None or not np.isfinite(baseline.snr)
                        or not baseline.snr
                        else f"{report.snr / baseline.snr:6.2f}x")
            out.append(
                f"  {report.label:<30} {report.top:8.4f} {report.bottom:9.4f} "
                f"{report.span:9.4f} {report.iqr:9.4f} {report.snr:9.1f}   "
                f"{relative}"
            )
        for report in self.candidates:
            spans = "  ".join(f"{s:.4f}" for s in report.per_cycle_spans)
            if spans:
                out.append(f"  per-cycle span  {report.key:<12} {spans}")
        out.extend(self.head_motion().lines())
        out.append(f"  reading       : {self.reading()}")
        out.append("  note          : only the vertical axis is exercised here; "
                   "hx is not measured by this protocol.")
        return out

    def reading(self) -> str:
        """One line of interpretation, in the terms the hypothesis was set in."""
        baseline, winner = self.baseline, self.winner()
        if baseline is None or winner is None or not np.isfinite(winner.snr):
            return "not enough data to compare."

        motion = self.head_motion()
        if motion.discredits_the_table:
            return (f"{motion.label} sitting - the head did "
                    f"{motion.head_fraction * 100:.0f}% of the pointing, so "
                    f"these rows rank responses to head pose, not to gaze. "
                    f"Not evidence about features either way.")
        aperture = self.find("c_aperture")
        lid_iris = self.find("d_lid_iris")
        factor = self.improvement()

        if factor < 1.25:
            return (f"no candidate clearly beats the baseline "
                    f"(best {winner.key} at {factor:.2f}x) - the vertical "
                    f"limit is not the feature definition.")

        parts = [f"{winner.key} carries {factor:.1f}x the baseline's "
                 f"signal-to-noise ({winner.snr:.1f} vs {baseline.snr:.1f})"]
        # Only ever compare candidates by SNR. Their raw spans are in different
        # units — hy is normalised by the lid aperture (~19 px here) and b by
        # the eye width (~64 px) — so "b's span is bigger" would be a statement
        # about the denominators, not about the signal.
        if aperture is not None and np.isfinite(aperture.snr):
            if aperture.snr > 3.0:
                parts.append("the lid aperture itself moves with gaze "
                             f"(SNR {aperture.snr:.1f}), so the baseline's "
                             "reference really is chasing the iris")
        if (lid_iris is not None and np.isfinite(lid_iris.snr)
                and lid_iris.snr > 1.5 * baseline.snr):
            parts.append("and the moving denominator is a large part of it - "
                         "d beats a on the same numerator")
        return "; ".join(parts) + "."


# ------------------------------------------------------------------- session
class FeatureLabSession(SetupCheckSession):
    """The setup check's protocol, timing and geometry — measuring everything.

    Subclassed rather than rewritten so the lab inherits the pacing that was
    fixed in spec 5.1c (lead-in, settle, the reaction budget): a diagnostic
    measured before the user's eyes arrive would repeat exactly the mistake
    that made this investigation necessary.
    """

    def __init__(self, screen_size, region: Optional[Region] = None,
                 seconds: float = 12.0,
                 screen_mm: float = SCREEN_HEIGHT_MM,
                 viewing_mm: float = VIEWING_DISTANCE_MM, **kwargs) -> None:
        super().__init__(screen_size, region=region, **kwargs)
        #: only the *expected* angle depends on these, and the report prints
        #: them so the assumption can be argued with
        self.screen_mm = float(screen_mm)
        self.viewing_mm = float(viewing_mm)
        base = list(self.targets)
        self.cycles = max(1, int(round(
            seconds / max(2.0 * (self.settle_s + self.collect_samples / 30.0),
                          1e-6))))
        #: top, bottom, top, bottom, ... so a slow drift shows up as a spread
        #: between cycles instead of quietly biasing one end
        self.targets = base * self.cycles
        self.visits: List[Visit] = []
        self._pending_px: Optional[np.ndarray] = None
        self.report: Optional[FeatureLabReport] = None

    # -------------------------------------------------------------- recording
    def observe(self, sample) -> None:
        """Take the landmarks off the sample before :meth:`update` runs."""
        self._pending_px = getattr(sample, "landmarks_px", None)

    def _collecting_at(self, timestamp: float) -> Tuple[int, bool]:
        if self.phase is not SetupPhase.MEASURING or self._start is None:
            return -1, False
        return self._index, (timestamp - self._start) >= self.settle_s

    def update(self, features) -> Optional[SetupCheckResult]:
        index, collecting = self._collecting_at(features.timestamp
                                                or self._now)
        result = super().update(features)
        if collecting and features.valid and self._pending_px is not None:
            self._record(index, features, self._pending_px)
        return result

    def _record(self, index: int, features, px: np.ndarray) -> None:
        visit = self._visit(index)
        for candidate in CANDIDATES:
            visit.record(candidate.key, candidate.value(features, px))
        # context, not a candidate: how much the *head* moved between the dots
        visit.record(PITCH_KEY, float(features.pitch))

    def _visit(self, index: int) -> Visit:
        if not self.visits or self.visits[-1].target_index != index % 2 \
                or self.visits[-1].cycle != index // 2:
            self.visits.append(Visit(target_index=index % 2, cycle=index // 2))
        return self.visits[-1]

    # ---------------------------------------------------------------- verdict
    def _evaluate(self) -> SetupCheckResult:
        """Build the comparison; the gate's own pass/fail is not the point.

        The result still carries the baseline span so the screen and the app
        can treat this exactly like a finished check and get out of the way.
        """
        result = super()._evaluate()
        rows = sorted({y for _, y in self.targets})
        self.report = FeatureLabReport(
            self.visits,
            separation_px=abs(rows[-1] - rows[0]),
            screen_px=float(self.screen_size[1]),
            screen_mm=self.screen_mm, viewing_mm=self.viewing_mm)
        # A lab run is not a gate. It must not block, and — the point of
        # ``gated`` — it must not claim a verdict it never applied: printing
        # "PASS" beside a span under the threshold is worse than printing no
        # verdict at all. The failure is left intact rather than cleared, so
        # nothing here quietly rewrites what was measured.
        result.gated = False
        result.label = "feature-lab baseline"
        self.phase = SetupPhase.PASSED
        return result

    def restart(self) -> None:
        super().restart()
        self.visits = []
        self.report = None
        self._pending_px = None
