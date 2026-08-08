"""--feature-lab: the vertical-channel diagnostic (read-only).

The synthetic eye here has one property the one in ``test_features.py`` does
not: **the eyelid follows the iris**. That is the hypothesis under test — that
``hy``'s collapse is caused by its reference moving with its subject — so the
tests drive a lid that tracks vertical gaze at a settable fraction and check
that the lab reports what is actually happening.

Nothing in here may touch the verified core, and a test at the bottom pins
that the lab's baseline is the core's own ``hy`` rather than a re-derivation.
"""
import numpy as np
import pytest

from gaze.feature_lab import (
    CANDIDATES,
    FeatureLabReport,
    FeatureLabSession,
    Visit,
    iris_vs_corner_line,
    lid_aperture,
    lid_to_iris,
)
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
    extract_features,
)
from gaze.setup_check import SetupPhase

SCREEN = (1366, 768)
FRAME = (640, 480)
DT = 1.0 / 30.0

EYE_HALF_W = 0.05
EYE_HALF_H = 0.02
EYE_Y = 0.40
R_EYE_CX, L_EYE_CX = 0.40, 0.60


def _set_iris(lm, idx, cx, cy, r=0.006):
    lm[idx[0], :2] = (cx, cy)
    lm[idx[1], :2] = (cx + r, cy)
    lm[idx[2], :2] = (cx - r, cy)
    lm[idx[3], :2] = (cx, cy + r)
    lm[idx[4], :2] = (cx, cy - r)


def make_face(gaze_v=0.0, lid_follow=0.875, noise=0.0, rng=None, aperture=0.0):
    """A face looking ``gaze_v`` up/down whose lids chase the iris.

    Two separate real effects, because they break different halves of ``hy``:

    Args:
        gaze_v: vertical gaze, -1 (up) .. +1 (down).
        lid_follow: how much of the iris's vertical travel the eyelids **copy**
            — this moves ``hy``'s *origin*. 0 = lids fixed to the skull, 1 =
            lids ride the iris exactly and ``hy`` reports nothing at all.
        aperture: how much the fissure **narrows** looking down and widens
            looking up, as a fraction. Applied to the **lower** lid only, so
            it moves ``hy``'s denominator without touching the upper-lid origin
            that ``hy`` and candidate d share — which is what makes d a clean
            separator rather than just another feature.
        noise: per-landmark jitter, in normalised units.
    """
    lm = np.zeros((478, 3), dtype=np.float64)
    dy = gaze_v * 0.8 * EYE_HALF_H
    lid_dy = lid_follow * dy
    lower = EYE_HALF_H * (1.0 - aperture * gaze_v)

    lm[R_OUTER, :2] = (R_EYE_CX - EYE_HALF_W, EYE_Y)
    lm[R_INNER, :2] = (R_EYE_CX + EYE_HALF_W, EYE_Y)
    lm[L_INNER, :2] = (L_EYE_CX - EYE_HALF_W, EYE_Y)
    lm[L_OUTER, :2] = (L_EYE_CX + EYE_HALF_W, EYE_Y)

    for cx, up, lo, iris in ((R_EYE_CX, R_UP, R_LO, R_IRIS),
                             (L_EYE_CX, L_UP, L_LO, L_IRIS)):
        lm[up, :2] = (cx, EYE_Y - EYE_HALF_H + lid_dy)
        lm[lo, :2] = (cx, EYE_Y + lower + lid_dy)
        _set_iris(lm, iris, cx, EYE_Y + dy)

    lm[1, :2] = (0.50, 0.55)
    lm[152, :2] = (0.50, 0.80)
    lm[61, :2] = (0.43, 0.70)
    lm[291, :2] = (0.57, 0.70)
    if noise and rng is not None:
        lm[:, :2] += rng.normal(0.0, noise, lm[:, :2].shape)
    return lm


def to_px(lm):
    return lm[:, :2] * np.array(FRAME, dtype=np.float64)


def hy_of(lm):
    return extract_features(lm, 0.0, 0.0, 0.0).hy


# ------------------------------------------------------- the hypothesis itself
def test_a_lid_that_chases_the_iris_collapses_hy():
    """Why the vertical channel is weak, reproduced from the definitions."""
    fixed = hy_of(make_face(gaze_v=+1.0, lid_follow=0.0)) - \
        hy_of(make_face(gaze_v=-1.0, lid_follow=0.0))
    chasing = hy_of(make_face(gaze_v=+1.0, lid_follow=0.875)) - \
        hy_of(make_face(gaze_v=-1.0, lid_follow=0.875))

    assert abs(chasing) < 0.2 * abs(fixed), \
        "a lid following the iris at 87.5% should gut hy's span"


def test_the_corner_referenced_candidate_survives_the_same_lid():
    """Candidate b is referenced to the corners, so the lid cannot touch it.

    The eye travels 2 x 0.8 x 0.02 normalised y = 15.4 px against a 64 px eye
    width, so 0.24 is the geometry, and it must not move with the lid at all.
    """
    for follow in (0.0, 0.5, 0.875, 1.0):
        span = abs(iris_vs_corner_line(to_px(make_face(+1.0, follow)))
                   - iris_vs_corner_line(to_px(make_face(-1.0, follow))))
        assert span == pytest.approx(0.24, abs=0.01), \
            f"lid_follow={follow} changed a corner-referenced feature"


def test_raw_spans_are_not_comparable_between_candidates():
    """Why the table is read on span/IQR, and why this test exists at all.

    Each candidate divides by a different length — the baseline by the ~19 px
    lid aperture, the others by the ~64 px eye width — so their spans are in
    different units. A bigger span can mean nothing but a smaller denominator.
    """
    def span(fn):
        return abs(fn(to_px(make_face(+1.0, 0.0)))
                   - fn(to_px(make_face(-1.0, 0.0))))

    fixed_lid_hy = abs(hy_of(make_face(+1.0, 0.0)) - hy_of(make_face(-1.0, 0.0)))
    assert fixed_lid_hy > span(iris_vs_corner_line), \
        "with fixed lids hy has the *larger* raw span and is no better for it"


def test_the_candidates_decompose_the_cause():
    """The 2x2, judged on signal-to-noise under one common landmark noise."""
    def snr(fn, follow, noise=0.0006, samples=400, seed=2):
        rng = np.random.default_rng(seed)
        values = {}
        for gaze_v in (-1.0, +1.0):
            values[gaze_v] = [
                fn(to_px(make_face(gaze_v, follow, noise=noise, rng=rng)))
                for _ in range(samples)]
        span = abs(np.median(values[+1.0]) - np.median(values[-1.0]))
        iqr = np.median([np.subtract(*np.percentile(v, [75, 25]))
                         for v in values.values()])
        return span / iqr

    def hy_snr(follow, noise=0.0006, samples=400, seed=2):
        rng = np.random.default_rng(seed)
        values = {g: [hy_of(make_face(g, follow, noise=noise, rng=rng))
                      for _ in range(samples)] for g in (-1.0, +1.0)}
        span = abs(np.median(values[+1.0]) - np.median(values[-1.0]))
        iqr = np.median([np.subtract(*np.percentile(v, [75, 25]))
                         for v in values.values()])
        return span / iqr

    chasing = 0.875
    assert snr(iris_vs_corner_line, chasing) > 3 * hy_snr(chasing), \
        "b must clearly beat the baseline when the lid chases the iris"
    assert hy_snr(0.0) > 3 * hy_snr(chasing), \
        "and the baseline is only weak *because* the lid moves"


def test_d_isolates_the_denominator_and_a_does_not():
    """What "d isolates the denominator" actually means, tested as such.

    ``d`` shares the baseline's numerator (upper lid to iris) but divides by
    the fixed eye width, so a fissure that opens and closes with vertical gaze
    must move ``hy``'s span and leave ``d``'s alone. Note which way round the
    finding goes: in this model the lid *translating* is what guts the signal,
    not the fissure changing size, so ``d`` is a diagnostic separator rather
    than a candidate replacement.
    """
    def span(fn, **face):
        return abs(fn(to_px(make_face(+1.0, **face)))
                   - fn(to_px(make_face(-1.0, **face))))

    def hy_span(**face):
        return abs(hy_of(make_face(+1.0, **face)) - hy_of(make_face(-1.0, **face)))

    rigid = dict(lid_follow=0.875, aperture=0.0)
    squeezing = dict(lid_follow=0.875, aperture=0.25)

    assert hy_span(**squeezing) != pytest.approx(hy_span(**rigid), rel=0.02), \
        "hy must feel a fissure that changes size - it is its denominator"
    assert span(lid_to_iris, **squeezing) == pytest.approx(
        span(lid_to_iris, **rigid), rel=1e-6), \
        "d must not, or it is not isolating the denominator"
    assert span(iris_vs_corner_line, **squeezing) == pytest.approx(
        span(iris_vs_corner_line, **rigid), rel=1e-6), \
        "and b must be blind to the eyelid entirely"


def test_the_aperture_candidate_sees_the_fissure_and_only_the_fissure():
    """c is the mechanism check: flat under translation, alive under squeeze."""
    def span(**face):
        return abs(lid_aperture(to_px(make_face(+1.0, **face)))
                   - lid_aperture(to_px(make_face(-1.0, **face))))

    assert span(lid_follow=0.875, aperture=0.0) < 1e-9, \
        "lids moving together do not change how open the eye is"
    assert span(lid_follow=0.875, aperture=0.25) > 0.05


def test_aperture_moves_when_the_lids_squeeze_rather_than_ride():
    """c is the mechanism check: it only moves if the fissure itself changes."""
    wide = lid_aperture(to_px(make_face(gaze_v=-1.0, lid_follow=0.0)))
    narrowed = make_face(gaze_v=+1.0, lid_follow=0.0)
    narrowed[R_UP, 1] += 0.008                      # upper lid drops
    narrowed[L_UP, 1] += 0.008
    assert lid_aperture(to_px(narrowed)) < wide


# ------------------------------------------------------------ candidate maths
def test_every_candidate_is_finite_on_an_ordinary_face():
    px = to_px(make_face(gaze_v=0.2))
    features = extract_features(make_face(gaze_v=0.2), 0.0, 0.0, 0.0)
    for candidate in CANDIDATES:
        assert np.isfinite(candidate.value(features, px)), candidate.key


def test_the_candidates_are_monotonic_in_vertical_gaze():
    for fn in (iris_vs_corner_line, lid_to_iris):
        values = [fn(to_px(make_face(gaze_v=g)))
                  for g in np.linspace(-1.0, 1.0, 9)]
        assert all(b > a for a, b in zip(values, values[1:])), (fn.__name__, values)


def test_both_eyes_are_averaged_with_the_same_sign():
    """The defect this project has already been bitten by, on the other axis.

    The two eyes' corner pairs are labelled in opposite orders, so a candidate
    that trusts the labels gets opposite signs and averages the signal away.
    """
    face = make_face(gaze_v=+1.0)
    one_eye = face.copy()
    for idx in (L_IRIS[0], L_IRIS[1], L_IRIS[2], L_IRIS[3], L_IRIS[4]):
        one_eye[idx, :2] = face[idx, :2]

    both = iris_vs_corner_line(to_px(face))
    assert both > 0.0, "looking down must be positive for the average"
    # and each eye alone must agree in sign with the average
    for eye_corners, iris in (((R_INNER, R_OUTER), R_IRIS),
                              ((L_INNER, L_OUTER), L_IRIS)):
        px = to_px(face)
        from gaze.feature_lab import _eye_frame, _iris_centre

        origin, _along, down, width = _eye_frame(px, *eye_corners)
        value = float(np.dot(_iris_centre(px, iris) - origin, down) / width)
        assert value > 0.0, "one eye disagreed in sign with the other"


def test_a_degenerate_eye_gives_nan_not_a_wrong_number():
    face = make_face()
    face[R_INNER, :2] = face[R_OUTER, :2]
    for fn in (iris_vs_corner_line, lid_aperture, lid_to_iris):
        assert not np.isfinite(fn(to_px(face))), fn.__name__


def test_pixels_not_normalised_coordinates():
    """A candidate that mixes axes must be measured in pixels.

    Normalised x and y are divided by different numbers, so the same face
    measured in normalised space gives a different answer — this pins that the
    lab is fed pixels, because otherwise the geometry is quietly wrong.
    """
    face = make_face(gaze_v=0.6)
    in_pixels = iris_vs_corner_line(to_px(face))
    in_normalised = iris_vs_corner_line(face[:, :2])
    assert abs(in_pixels - in_normalised) > 0.01, \
        "if these agree the test no longer proves anything"


# ------------------------------------------------------------------ reporting
def _visits(values_by_target, cycles=2, key="a_hy", noise=0.0, seed=0):
    """Visits carrying a single candidate's samples, for report maths."""
    rng = np.random.default_rng(seed)
    visits = []
    for cycle in range(cycles):
        for index in (0, 1):
            visit = Visit(target_index=index, cycle=cycle)
            for _ in range(30):
                visit.record(key, values_by_target[index]
                             + rng.normal(0.0, noise))
            visits.append(visit)
    return visits


def test_the_report_measures_span_and_noise_separately():
    visits = _visits({0: 0.40, 1: 0.50}, noise=0.01, seed=3)
    report = FeatureLabReport(visits).find("a_hy")

    assert report.span == pytest.approx(0.10, abs=0.01)
    assert report.iqr == pytest.approx(0.0135, abs=0.006), "IQR of a 0.01 sd"
    assert report.snr == pytest.approx(report.span / report.iqr)


def test_a_noisier_candidate_with_a_bigger_span_can_still_lose():
    """The reason span alone is not the answer."""
    quiet = FeatureLabReport(_visits({0: 0.40, 1: 0.44}, noise=0.002,
                                     seed=1)).find("a_hy")
    loud = FeatureLabReport(_visits({0: 0.40, 1: 0.60}, noise=0.05,
                                    seed=1)).find("a_hy")

    assert loud.span > quiet.span
    assert loud.snr < quiet.snr, "span/IQR is what ranks them"


def test_per_cycle_spans_expose_an_unstable_candidate():
    visits = _visits({0: 0.40, 1: 0.50}, cycles=3, noise=0.001)
    visits[3].values["a_hy"] = [0.40] * 30          # cycle 1's bottom drifted
    report = FeatureLabReport(visits).find("a_hy")

    assert len(report.per_cycle_spans) == 3
    assert report.stability > 0.1, "a cycle that disagreed should show up"


def test_the_reading_says_when_nothing_beats_the_baseline():
    same = [0.40, 0.50]
    visits = []
    for cycle in range(2):
        for index in (0, 1):
            visit = Visit(target_index=index, cycle=cycle)
            for candidate in CANDIDATES:
                for i in range(30):
                    visit.record(candidate.key,
                                 same[index] + (0.001 if i % 2 else -0.001))
            visits.append(visit)

    reading = FeatureLabReport(visits).reading()
    assert "no candidate clearly beats the baseline" in reading


def test_an_empty_run_reports_nothing_rather_than_dividing_by_zero():
    report = FeatureLabReport([])
    assert report.samples == 0
    assert "no usable frames" in report.lines()[0]


def test_every_reported_line_is_console_safe():
    visits = _visits({0: 0.40, 1: 0.50}, noise=0.01)
    "\n".join(FeatureLabReport(visits).lines()).encode("cp1255")


# -------------------------------------------------------------- the session
class Sample:
    def __init__(self, features, px):
        self.features = features
        self.landmarks_px = px
        self.stream_valid = True
        self.has_gaze = False
        self.timestamp = features.timestamp


def drive(session, seconds=60.0, lid_follow=0.875, noise=0.0, seed=0):
    """Feed the session a user who looks at whichever dot is showing.

    The clock starts at ``DT`` rather than 0: a ``FrameFeatures.timestamp`` of
    exactly 0.0 reads as "unset" and falls back to the wall clock.
    """
    rng = np.random.default_rng(seed)
    now = DT
    for _ in range(int(seconds / DT)):
        if session.is_finished:
            break
        target = session.current_target()
        # target 1 is the top dot (look up, gaze_v -1), target 2 the bottom
        gaze_v = 0.0 if target is None else (-1.0 if target.index % 2 else +1.0)
        face = make_face(gaze_v=gaze_v, lid_follow=lid_follow,
                         noise=noise, rng=rng)
        features = extract_features(face, 0.0, 0.0, 0.0, now)
        sample = Sample(features, to_px(face))
        session.observe(sample)
        session.update(features)
        now += DT
    return session.report


def test_the_session_reuses_the_setup_check_protocol():
    from gaze.setup_check import SetupCheckSession

    session = FeatureLabSession(SCREEN, seconds=12.0)
    assert isinstance(session, SetupCheckSession)
    assert session.phase is SetupPhase.LEAD_IN, "the lead-in is inherited"
    assert session.reaction_budget_s >= 1.4, "and so is the reaction budget"


def test_seconds_controls_how_many_cycles_run():
    short = FeatureLabSession(SCREEN, seconds=6.0)
    long = FeatureLabSession(SCREEN, seconds=30.0)
    assert short.cycles < long.cycles
    assert len(long.targets) == 2 * long.cycles
    assert long.targets[0] == long.targets[2], "top, bottom, top, bottom..."
    assert long.targets[0] != long.targets[1]


def test_a_run_reports_every_candidate_from_the_same_frames():
    session = FeatureLabSession(SCREEN, seconds=12.0)
    report = drive(session)

    assert report is not None
    assert {c.key for c in report.candidates} == {c.key for c in CANDIDATES}
    per_candidate = {len(v.values[c.key]) for v in report.visits
                     for c in CANDIDATES}
    assert len(per_candidate) == 1, \
        "every candidate must be computed from exactly the same frames"


def test_a_run_finds_the_corner_candidate_wins_on_a_chasing_lid():
    session = FeatureLabSession(SCREEN, seconds=12.0)
    report = drive(session, lid_follow=0.875, noise=0.0004, seed=5)

    assert report.winner().key == "b_corner"
    assert report.improvement() > 2.0
    assert "carries" in report.reading()


def test_a_run_says_so_when_the_lid_is_not_the_problem():
    """Control: with skull-fixed lids the baseline is fine and the lab says so."""
    session = FeatureLabSession(SCREEN, seconds=12.0)
    report = drive(session, lid_follow=0.0, noise=0.0004, seed=5)

    assert report.improvement() < 2.0, \
        "with fixed lids hy is a perfectly good feature"


def test_the_lab_never_blocks_and_never_saves():
    session = FeatureLabSession(SCREEN, seconds=6.0)
    drive(session)
    assert session.phase is SetupPhase.PASSED, "a diagnostic is not a gate"
    assert session.result.passed
    assert not hasattr(session, "save")


def test_frames_before_the_settle_are_not_recorded():
    session = FeatureLabSession(SCREEN, seconds=6.0, lead_in_s=0.0)
    now = DT
    for _ in range(int(0.5 / DT)):                  # inside the 1 s settle
        face = make_face()
        session.observe(Sample(extract_features(face, 0.0, 0.0, 0.0, now),
                               to_px(face)))
        session.update(extract_features(face, 0.0, 0.0, 0.0, now))
        now += DT
    assert session.visits == []


def test_the_lead_in_records_nothing():
    session = FeatureLabSession(SCREEN, seconds=6.0)
    now = DT
    for _ in range(int(1.0 / DT)):
        face = make_face()
        session.observe(Sample(extract_features(face, 0.0, 0.0, 0.0, now),
                               to_px(face)))
        session.update(extract_features(face, 0.0, 0.0, 0.0, now))
        now += DT
    assert session.phase is SetupPhase.LEAD_IN
    assert session.visits == []


def test_a_frame_without_landmarks_is_skipped_not_guessed():
    session = FeatureLabSession(SCREEN, seconds=6.0, lead_in_s=0.0)
    now = DT
    for _ in range(int(3.0 / DT)):
        face = make_face()
        features = extract_features(face, 0.0, 0.0, 0.0, now)
        session.observe(Sample(features, None))     # landmarks not enabled
        session.update(features)
        now += DT
    assert session.visits == [], "no landmarks means no candidates, not zeros"


# ------------------------------------------------------------- the core is safe
def test_the_baseline_is_the_verified_cores_own_hy():
    """Not a re-derivation: the comparison has to be against what ships."""
    baseline = next(c for c in CANDIDATES if c.key == "a_hy")
    assert baseline.compute is None

    face = make_face(gaze_v=0.4)
    features = extract_features(face, 0.0, 0.0, 0.0)
    assert baseline.value(features, to_px(face)) == features.hy


def test_the_lab_imports_nothing_it_could_mutate():
    """gaze/features.py is read for its landmark indices and nothing else."""
    import inspect

    import gaze.feature_lab as lab

    source = inspect.getsource(lab)
    assert "extract_features" not in source, \
        "the lab must not re-run the core's extraction, only read hy"
    for forbidden in ("features.hx =", "features.hy =", "def extract_features"):
        assert forbidden not in source
