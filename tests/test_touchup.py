"""One-point touch-up tests (spec 5.5) — the session logic and the correction.

The touch-up exists to fix a *translation*: the mapping is still the right
shape, it just sits a few dozen pixels off. These check that it measures that
translation, applies it to nothing but the two constant terms, and refuses when
the evidence does not support it.
"""
import numpy as np
import pytest

from gaze.calibration import CalibrationModel
from gaze.drift import TouchUpPhase, TouchUpResult, TouchUpSession

SCREEN = (1366, 768)
CENTRE = (683.0, 384.0)
DT = 1.0 / 30.0


def feed(session, gaze, seconds, start=0.0, stream_valid=True,
         is_fixating=True, jitter=0.0):
    """Drive the session with a gaze position for a stretch of time."""
    now = start
    result = None
    steps = max(1, int(round(seconds / DT)))
    for i in range(steps):
        now += DT
        wobble = jitter * (1 if i % 2 else -1)
        outcome = session.update(gaze[0] + wobble, gaze[1] - wobble,
                                 is_fixating, stream_valid, now)
        result = outcome if outcome is not None else result
    return result, now


# ------------------------------------------------------------------- the target
def test_the_target_is_the_centre_of_the_screen():
    assert TouchUpSession(SCREEN).target == CENTRE


def test_the_whole_thing_fits_well_inside_ten_seconds():
    """'Back to typing in under 10 s' is the point of a touch-up."""
    session = TouchUpSession(SCREEN)
    assert session.budget_s == pytest.approx(2.4)
    assert session.budget_s < 10.0


# ------------------------------------------------------------------- the phases
def test_the_first_moments_are_not_measured():
    """The saccade to the dot must not drag the median with it."""
    session = TouchUpSession(SCREEN, settle_ms=400.0)
    feed(session, (50.0, 50.0), 0.3)
    assert session.phase is TouchUpPhase.SETTLING
    assert session.samples == 0
    assert session.progress == 0.0


def test_measuring_starts_after_the_settle_and_shows_progress():
    session = TouchUpSession(SCREEN, settle_ms=400.0, collect_s=2.0)
    feed(session, CENTRE, 1.0)
    assert session.phase is TouchUpPhase.COLLECTING
    assert 0.0 < session.progress < 1.0
    assert session.samples > 10


# ------------------------------------------------------------- the measurement
def test_a_steady_offset_becomes_the_correction():
    """Gaze reads 40 px right and 25 px low: correct by the opposite."""
    session = TouchUpSession(SCREEN)
    result, _ = feed(session, (CENTRE[0] + 40.0, CENTRE[1] + 25.0), 3.0)
    assert result is not None and result.accepted
    assert result.dx == pytest.approx(-40.0, abs=0.5)
    assert result.dy == pytest.approx(-25.0, abs=0.5)
    assert result.offset_px == pytest.approx(np.hypot(40.0, 25.0), abs=0.5)


def test_the_median_shrugs_off_a_wobble():
    session = TouchUpSession(SCREEN)
    result, _ = feed(session, (CENTRE[0] + 40.0, CENTRE[1]), 3.0, jitter=15.0)
    assert result.accepted
    assert result.dx == pytest.approx(-40.0, abs=8.0)


def test_a_result_is_only_returned_once():
    session = TouchUpSession(SCREEN)
    result, now = feed(session, CENTRE, 3.0)
    assert result is not None
    again, _ = feed(session, CENTRE, 1.0, start=now)
    assert again is None
    assert session.phase is TouchUpPhase.DONE


# ------------------------------------------------------------------ refusals
def test_too_few_samples_is_refused_rather_than_guessed():
    session = TouchUpSession(SCREEN)
    result, _ = feed(session, CENTRE, 3.0, stream_valid=False)
    assert result is not None and not result.accepted
    assert result.samples == 0
    assert "long enough" in result.reason


def test_an_implausible_offset_is_refused():
    """Half a screen away is not drift — applying it would hide a real problem."""
    session = TouchUpSession(SCREEN)
    result, _ = feed(session, (60.0, 60.0), 3.0)
    assert not result.accepted
    assert "recalibrate" in result.reason


def test_cancelling_ends_the_session_without_a_result():
    session = TouchUpSession(SCREEN)
    feed(session, CENTRE, 1.0)
    session.cancel()
    assert session.phase is TouchUpPhase.DONE
    assert session.result is None
    result, _ = feed(session, CENTRE, 3.0, start=5.0)
    assert result is None


# ------------------------------------------------------- applying to the model
def fitted_model() -> CalibrationModel:
    """A model mapping a small grid of ratios onto the screen."""
    model = CalibrationModel(screen_size=SCREEN)
    model.comp.ok = True
    features, targets = [], []
    for hx in (0.35, 0.5, 0.65):
        for hy in (0.35, 0.5, 0.65):
            features.append([hx, hy, 0.0, 0.0])
            targets.append([(hx - 0.35) / 0.3 * 1300 + 33,
                            (hy - 0.35) / 0.3 * 700 + 34])
    model.fit(np.array(features), np.array(targets))
    return model


def test_the_correction_shifts_every_prediction_by_exactly_the_offset():
    model = fitted_model()
    probe = np.array([0.5, 0.5, 0.0, 0.0])
    before = model.predict(probe)

    session = TouchUpSession(SCREEN)
    feed(session, (CENTRE[0] + 40.0, CENTRE[1] + 25.0), 3.0)
    assert session.apply_to(model)

    after = model.predict(probe)
    assert after[0] - before[0] == pytest.approx(-40.0, abs=0.5)
    assert after[1] - before[1] == pytest.approx(-25.0, abs=0.5)


def test_only_the_constant_terms_move():
    """A translation must not touch the shape of the fit (spec 5.5)."""
    model = fitted_model()
    wx, wy = model.wx.copy(), model.wy.copy()

    session = TouchUpSession(SCREEN)
    feed(session, (CENTRE[0] + 40.0, CENTRE[1] + 25.0), 3.0)
    session.apply_to(model)

    assert np.allclose(model.wx[1:], wx[1:]), "curvature must not change"
    assert np.allclose(model.wy[1:], wy[1:])
    assert model.wx[0] != wx[0] and model.wy[0] != wy[0]


def test_a_refused_result_is_never_applied():
    model = fitted_model()
    wx = model.wx.copy()
    session = TouchUpSession(SCREEN)
    feed(session, (60.0, 60.0), 3.0)          # implausible, refused above
    assert not session.apply_to(model)
    assert np.allclose(model.wx, wx)


def test_an_unfinished_session_applies_nothing():
    model = fitted_model()
    wx = model.wx.copy()
    session = TouchUpSession(SCREEN)
    feed(session, CENTRE, 0.5)
    assert not session.apply_to(model)
    assert np.allclose(model.wx, wx)


# ------------------------------------------------------------------- messaging
def test_the_outcome_is_reportable_either_way():
    assert "34 px" in TouchUpResult(dx=30.0, dy=16.0, accepted=True).message()
    assert TouchUpResult(accepted=False, reason="nope").message() == "nope"
    assert TouchUpResult().message()
