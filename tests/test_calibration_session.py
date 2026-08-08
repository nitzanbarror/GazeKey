"""Calibration-session tests — driven by a synthetic eye, no camera, no Qt.

The eye model is the one the verified core is validated against
(``tests/test_calibration_pipeline.py``): non-linear in gaze, coupled to head
pose, with the natural "users turn their head toward the target" behaviour.
"""
import random

import numpy as np
import pytest

from gaze.calibration import MARGINAL_PX, PASS_PX
from gaze.calibration_session import (
    MIN_USABLE_POINTS,
    CalibrationSession,
    Phase,
    SweepFailure,
)
from gaze.features import FrameFeatures
from tests.test_calibration_pipeline import (
    H,
    W,
    natural_pose_for_target,
    true_features_for_screen_point,
)

FRAME_DT = 1.0 / 30.0
SCREEN = (W, H)


def features_from(vector, timestamp: float, valid: bool = True) -> FrameFeatures:
    hx, hy, yaw, pitch = vector
    return FrameFeatures(valid, hx, hy, yaw, pitch, 0.0, timestamp)


class SyntheticUser:
    """Produces frames for whatever the session is currently asking for."""

    def __init__(
        self,
        seed: int = 0,
        noise: float = 0.004,
        still_head: bool = False,
        blind: bool = False,
        settle_offset_px: tuple[float, float] = (0.0, 0.0),
        validation_offset_px: tuple[float, float] = (0.0, 0.0),
        invalid_points: dict[int, int] | None = None,
        glance_points: dict[int, tuple[float, float]] | None = None,
        glance_forever: bool = False,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.still_head = still_head
        self.blind = blind
        self.settle_offset_px = settle_offset_px
        self.validation_offset_px = validation_offset_px
        #: target index -> how many attempts to sabotage with invalid frames
        self.invalid_points = dict(invalid_points or {})
        #: target index -> px the user's gaze wanders off by while it collects,
        #: i.e. "looked away while that dot was up"
        self.glance_points = dict(glance_points or {})
        #: keep glancing away on every re-collection, not just the first
        self.glance_forever = glance_forever
        self.sweep_phase = 0.0
        self._attempt_seen: dict[int, int] = {}
        self._visits: dict = {}
        self._last_key = None
        self._was_settling = False

    def _visit(self, target) -> int:
        """Which showing of this target we are in (1 = the first).

        Keyed on (stage, position) so it works for validation targets too,
        which do not appear in the 9-point grid at all.
        """
        key = (target.stage, target.position)
        if key != self._last_key or (target.settling and not self._was_settling):
            self._visits[key] = self._visits.get(key, 0) + 1
        self._last_key, self._was_settling = key, target.settling
        return self._visits[key]

    def _noisy(self, vector):
        return vector + np.concatenate([
            self.rng.normal(0, self.noise, 2), self.rng.normal(0, 0.3, 2)
        ])

    def sweep_frame(self, t: float) -> FrameFeatures:
        if self.blind:
            return FrameFeatures(valid=False, timestamp=t)
        self.sweep_phase += FRAME_DT
        if self.still_head:
            yaw, pitch = 0.4, -0.2
        else:
            yaw = 12.0 * np.sin(2 * np.pi * 0.5 * self.sweep_phase)
            pitch = 8.0 * np.sin(2 * np.pi * 0.35 * self.sweep_phase + 1.3)
        centre = true_features_for_screen_point(W / 2, H / 2, yaw, pitch)
        return features_from(self._noisy(centre), t)

    def target_frame(self, session: CalibrationSession, t: float) -> FrameFeatures:
        target = session.current_target()
        assert target is not None
        px, py = target.position

        if target.stage is Phase.POINTS and target.retry < self.invalid_points.get(
            _target_key(session, target), -1
        ):
            return FrameFeatures(valid=False, timestamp=t)   # e.g. a long blink

        visit = self._visit(target)
        if target.stage is Phase.POINTS and not target.settling:
            glance = self.glance_points.get(_target_key(session, target))
            if glance and (self.glance_forever or visit == 1):
                px += glance[0]
                py += glance[1]

        if target.settling:
            px += self.settle_offset_px[0]
            py += self.settle_offset_px[1]
        elif target.stage is Phase.VALIDATION:
            px += self.validation_offset_px[0]
            py += self.validation_offset_px[1]

        yaw, pitch = natural_pose_for_target(px, py)
        vector = true_features_for_screen_point(px, py, yaw, pitch)
        return features_from(self._noisy(vector), t)

    def frame(self, session: CalibrationSession, t: float) -> FrameFeatures:
        if session.phase is Phase.SWEEP:
            return self.sweep_frame(t)
        if session.phase in (Phase.POINTS, Phase.VALIDATION):
            return self.target_frame(session, t)
        return FrameFeatures(valid=False, timestamp=t)


def _target_key(session: CalibrationSession, target) -> int:
    """Index of the target inside the shuffled 3x3 grid."""
    return session.cal_targets.index(target.position)


def drive(session: CalibrationSession, user: SyntheticUser,
          max_seconds: float = 180.0, on_frame=None) -> float:
    """Feed frames until the session finishes; returns the simulated duration."""
    t = 1_000.0
    start = t
    while not session.is_finished and (t - start) < max_seconds:
        if session.phase is Phase.SWEEP_FAILED:
            break
        t += FRAME_DT
        session.update(user.frame(session, t))
        if on_frame is not None:
            on_frame(session)
    return t - start


def new_session(**kwargs) -> CalibrationSession:
    kwargs.setdefault("rng", random.Random(1234))
    return CalibrationSession(SCREEN, camera_index=0, **kwargs)


# --------------------------------------------------------------- happy path
def test_full_session_passes_the_validation_gate():
    session = new_session()
    elapsed = drive(session, SyntheticUser(seed=1))

    assert session.phase is Phase.RESULTS
    assert session.verdict == "PASS", f"{session.verdict} at {session.error_px:.1f}px"
    assert session.error_px <= PASS_PX
    assert session.should_save
    assert all(f is not None for f in session.cal_features)
    assert elapsed < 40.0, f"session took {elapsed:.1f}s of frames"


def test_session_visits_nine_then_three_targets():
    session = new_session()
    seen_points, seen_validation = set(), set()

    def record(s: CalibrationSession):
        target = s.current_target()
        if target is None:
            return
        if target.stage is Phase.POINTS:
            seen_points.add(target.position)
        else:
            seen_validation.add(target.position)

    drive(session, SyntheticUser(seed=2), on_frame=record)
    assert len(seen_points) == 9
    assert len(seen_validation) == 3
    assert not seen_points & seen_validation, "validation must use fresh targets"


def test_targets_are_presented_in_randomised_order():
    orders = set()
    for seed in range(6):
        session = CalibrationSession(SCREEN, rng=random.Random(seed))
        orders.add(tuple(session.cal_targets))
    assert len(orders) > 1, "target order is not randomised"


def test_grid_and_validation_targets_match_the_spec():
    session = new_session()
    assert set(session.cal_targets) == {
        (int(x * W), int(y * H))
        for x in (0.1, 0.5, 0.9) for y in (0.1, 0.5, 0.9)
    }
    assert session.val_targets == [
        (int(0.3 * W), int(0.3 * H)),
        (int(0.7 * W), int(0.7 * H)),
        (int(0.7 * W), int(0.3 * H)),
    ]


# ------------------------------------------------------------- stage A gate
def test_sweep_without_head_movement_is_rejected():
    session = new_session(fixed_head=False)
    drive(session, SyntheticUser(seed=3, still_head=True))
    assert session.phase is Phase.SWEEP_FAILED
    assert session.sweep_failure is SweepFailure.NO_MOTION
    headline, guidance = session.sweep_failure_message
    assert "moved" in headline
    assert "turn your head" in guidance


def test_sweep_without_a_visible_face_is_rejected():
    session = new_session(fixed_head=False)
    drive(session, SyntheticUser(seed=4, blind=True))
    assert session.phase is Phase.SWEEP_FAILED
    assert session.sweep_failure is SweepFailure.NO_FACE
    assert "see your eyes" in session.sweep_failure_message[0]


def test_rejected_sweep_can_be_repeated_and_then_succeeds():
    session = new_session(fixed_head=False)
    drive(session, SyntheticUser(seed=5, still_head=True))
    assert session.phase is Phase.SWEEP_FAILED

    session.retry_sweep()
    assert session.phase is Phase.SWEEP
    assert session.sweep_failure is None

    drive(session, SyntheticUser(seed=6))
    assert session.phase is Phase.RESULTS
    assert session.verdict == "PASS"


def test_pose_compensation_is_fitted_before_any_target_is_shown():
    session = new_session(fixed_head=False)
    user = SyntheticUser(seed=7)
    t = 1_000.0
    while session.phase is Phase.SWEEP:
        t += FRAME_DT
        assert session.current_target() is None
        assert not session.model.comp.ok
        session.update(user.frame(session, t))
    assert session.phase is Phase.POINTS
    assert session.model.comp.ok, "stage A must be fitted before stage B"


def test_sweep_reports_live_progress_and_pose_range():
    session = new_session(fixed_head=False)
    user = SyntheticUser(seed=8)
    t = 1_000.0
    assert session.sweep_progress == 0.0
    for _ in range(45):
        t += FRAME_DT
        session.update(user.frame(session, t))
    assert 0.0 < session.sweep_progress < 1.0
    yaw_range, pitch_range = session.sweep_pose_range
    assert yaw_range > 5.0 and pitch_range > 2.0


# ------------------------------------------------------- collection mechanics
def test_settling_delay_discards_frames_before_the_eye_lands():
    """Frames from before the eye lands on the target must not be aggregated."""
    session = new_session()
    user = SyntheticUser(seed=9, settle_offset_px=(600.0, 400.0))
    drive(session, user)

    # a huge offset applied only during settling must not disturb the fit
    assert session.verdict == "PASS", f"{session.error_px:.1f}px"

    first_target = session.cal_targets[0]
    yaw, pitch = natural_pose_for_target(*first_target)
    expected = true_features_for_screen_point(*first_target, yaw, pitch)
    np.testing.assert_allclose(session.cal_features[0][:2], expected[:2], atol=0.01)


def test_point_with_no_usable_samples_is_repeated_then_salvaged():
    session = new_session(max_retries=2)
    retries_seen = []

    def record(s: CalibrationSession):
        target = s.current_target()
        if target is not None and target.stage is Phase.POINTS:
            retries_seen.append(target.retry)

    # sabotage the first grid point for its first two attempts
    user = SyntheticUser(seed=10, invalid_points={0: 2})
    drive(session, user, on_frame=record)

    assert max(retries_seen) == 2, "point should have been repeated twice"
    assert session.phase is Phase.RESULTS
    assert session.cal_features[0] is not None, "salvaged aggregate expected"


def test_retries_stop_after_the_limit():
    session = new_session(max_retries=2)
    user = SyntheticUser(seed=11, invalid_points={0: 99})   # never recovers
    drive(session, user)
    assert session.phase is Phase.RESULTS
    assert session.cal_features[0] is None, "hopeless point is dropped"
    assert sum(f is not None for f in session.cal_features) >= MIN_USABLE_POINTS


def test_progress_and_counters_are_reported_for_the_ui():
    session = new_session()
    user = SyntheticUser(seed=12)
    seen = []

    def record(s: CalibrationSession):
        target = s.current_target()
        if target is not None:
            seen.append((target.stage, target.index, target.total, target.progress))

    drive(session, user, on_frame=record)
    point_states = [s for s in seen if s[0] is Phase.POINTS]
    validation_states = [s for s in seen if s[0] is Phase.VALIDATION]
    assert {s[2] for s in point_states} == {9}
    assert {s[1] for s in point_states} == set(range(1, 10))
    assert {s[2] for s in validation_states} == {3}
    assert max(s[3] for s in point_states) == pytest.approx(1.0, abs=0.2)
    assert min(s[3] for s in point_states) == 0.0   # settling


# ----------------------------------------------------------- validation gate
def test_bad_validation_triggers_exactly_one_refit_pass():
    session = new_session()
    user = SyntheticUser(seed=13, validation_offset_px=(420.0, 300.0))

    phases = []

    def record(s: CalibrationSession):
        entry = (s.phase, s.refit_pass)
        if not phases or phases[-1] != entry:
            phases.append(entry)

    drive(session, user, on_frame=record)

    refit_blocks = [p for p in phases if p == (Phase.POINTS, True)]
    assert len(refit_blocks) == 1, phases
    assert session.phase is Phase.RESULTS
    assert session.verdict in ("PASS", "MARGINAL", "FAIL")


def test_refit_pass_recollects_only_two_points():
    session = new_session()
    user = SyntheticUser(seed=14, validation_offset_px=(420.0, 300.0))
    totals = set()

    def record(s: CalibrationSession):
        target = s.current_target()
        if target is not None and target.stage is Phase.POINTS and s.refit_pass:
            totals.add(target.total)

    drive(session, user, on_frame=record)
    assert totals == {2}, totals


def test_verdict_thresholds_match_the_spec():
    session = new_session()
    drive(session, SyntheticUser(seed=15))
    for error, expected in ((10.0, "PASS"), (PASS_PX, "PASS"),
                            (PASS_PX + 0.1, "MARGINAL"), (MARGINAL_PX, "MARGINAL"),
                            (MARGINAL_PX + 0.1, "FAIL")):
        features = np.array([session.cal_features[0]])
        predicted = session.model.predict_many(features)
        targets = predicted + np.array([[error, 0.0]])
        measured, verdict = session.model.validate(features, targets)
        assert verdict == expected, f"{measured:.1f}px -> {verdict}"


def test_results_text_is_provided_for_every_verdict():
    session = new_session()
    drive(session, SyntheticUser(seed=16))
    for verdict in ("PASS", "MARGINAL", "FAIL"):
        session.verdict = verdict
        headline, guidance = session.verdict_text
        assert headline and guidance
    session.verdict = "FAIL"
    assert not session.should_save


# ------------------------------------------------------------------- pacing
def advance_to_points(session: CalibrationSession, user: SyntheticUser,
                      timestamp: float = 1_000.0) -> float:
    while session.phase is Phase.SWEEP:
        timestamp += FRAME_DT
        session.update(user.frame(session, timestamp))
    return timestamp


@pytest.mark.parametrize("settle_ms", [400, 700, 1000])
def test_settle_time_is_configurable(settle_ms):
    session = new_session(settle_ms=settle_ms)
    user = SyntheticUser(seed=20)
    t = advance_to_points(session, user)

    started = None
    while session.current_target().settling:
        t += FRAME_DT
        if started is None:
            started = t
        session.update(user.frame(session, t))
    measured_ms = (t - started) * 1000.0
    assert measured_ms == pytest.approx(settle_ms, abs=1000 * FRAME_DT * 1.5)


def test_settle_progress_runs_from_zero_to_one():
    session = new_session(settle_ms=700)
    user = SyntheticUser(seed=21)
    t = advance_to_points(session, user)

    seen = []
    while session.current_target().settling:
        t += FRAME_DT
        seen.append(session.current_target().settle_progress)
        session.update(user.frame(session, t))
    assert seen[0] == pytest.approx(0.0, abs=0.05)
    assert max(seen) > 0.9
    assert seen == sorted(seen), "countdown must advance monotonically"
    assert session.current_target().settle_progress == 1.0
    # collection has only just begun on the frame after the countdown ended
    assert session.current_target().progress < 0.1


@pytest.mark.parametrize("wanted", [20, 45, 60])
def test_collect_samples_controls_how_much_is_measured(wanted):
    session = new_session(settle_ms=300, collect_samples=wanted)
    drive(session, SyntheticUser(seed=22))
    assert session.is_finished
    collected = [p.collected for p in session.diagnostics().points]
    assert set(collected) == {wanted}, collected


def test_collection_gives_up_at_the_wall_clock_cap():
    """An unreachable sample target must still end each point on time."""
    session = new_session(settle_ms=200, collect_samples=10_000, collect_max_s=1.0)
    drive(session, SyntheticUser(seed=23))
    assert session.is_finished
    for point in session.diagnostics().points:
        assert 25 <= point.collected <= 35, point.collected


def test_slower_pacing_still_passes_the_gate():
    session = new_session(settle_ms=1000, collect_samples=60)
    drive(session, SyntheticUser(seed=24))
    assert session.verdict == "PASS", f"{session.error_px:.1f} px"
    assert all(p.collected == 60 for p in session.diagnostics().points)


def test_collect_samples_cannot_go_below_the_outlier_floor():
    from gaze.calibration import MIN_SAMPLES_PER_POINT

    session = new_session(collect_samples=3)
    assert session.collect_samples == MIN_SAMPLES_PER_POINT


# -------------------------------------------------------------- diagnostics
def test_diagnostics_report_every_calibration_point():
    session = new_session()
    drive(session, SyntheticUser(seed=25))
    report = session.diagnostics()

    assert len(report.points) == 9
    assert [p.index for p in report.points] == list(range(1, 10))
    for point, target in zip(report.points, session.cal_targets):
        assert point.target == target
        assert point.collected == session.collect_samples
        assert 0 < point.kept <= point.collected
        assert point.rejected == point.collected - point.kept
        assert np.isfinite(point.residual_px)
        assert np.isfinite(point.hx) and np.isfinite(point.hy)
    assert max(p.residual_px for p in report.points) < 100


def test_diagnostics_report_validation_arrows():
    session = new_session()
    drive(session, SyntheticUser(seed=26))
    report = session.diagnostics()

    assert len(report.validation) == 3
    for point, target in zip(report.validation, session.val_targets):
        assert point.target == target
        assert point.prediction is not None
        assert point.residual_px == pytest.approx(
            float(np.hypot(point.prediction[0] - target[0],
                           point.prediction[1] - target[1])), abs=1e-6
        )
    mean_error = float(np.mean([p.residual_px for p in report.validation]))
    assert mean_error == pytest.approx(session.error_px, abs=1e-6)


def test_diagnostics_report_feature_and_sweep_spans():
    session = new_session(fixed_head=False)
    drive(session, SyntheticUser(seed=27))
    report = session.diagnostics()

    assert report.hx_range[0] < report.hx_range[1]
    assert report.hy_range[0] < report.hy_range[1]
    assert report.hx_span > 0.05 and report.hy_span > 0.02
    assert report.sweep_samples == session.sweep_samples
    assert report.sweep_yaw_range > 10.0 and report.sweep_pitch_range > 5.0
    assert "hx" in report.feature_line() and "span" in report.feature_line()
    assert "sweep 90 samples" in report.sweep_line()


def test_diagnostics_count_retries():
    session = new_session()
    drive(session, SyntheticUser(seed=28, invalid_points={0: 2}))
    repeated = [p for p in session.diagnostics().points if p.retries]
    assert repeated and repeated[0].retries == 2


def test_console_report_is_ascii_and_complete():
    session = new_session(fixed_head=False)
    drive(session, SyntheticUser(seed=29))
    text = session.diagnostics().console_report()

    text.encode("ascii")            # must survive any Windows code page
    assert "calibration diagnostics" in text
    assert "head mode: free" in text
    assert text.count("px") >= 12   # 9 residuals + 3 validation errors
    assert "verdict" in text and session.verdict in text
    assert "most likely" in text
    for target in session.val_targets:
        assert f"{target[0]:5d}" in text


def test_diagnostics_before_a_fit_have_no_residuals():
    session = new_session()
    user = SyntheticUser(seed=30)
    advance_to_points(session, user)
    report = session.diagnostics()
    assert all(not np.isfinite(p.residual_px) for p in report.points)
    assert report.verdict == "INCOMPLETE"
    assert "usable data" in report.likely_cause


# ------------------------------------------------------- most-likely-cause
def diagnostics_stub(session: CalibrationSession, **overrides):
    from gaze.calibration_session import PointDiagnostics, SessionDiagnostics

    def point(index, residual, collected=45, kept=44, retries=0):
        return PointDiagnostics(index=index, target=(100 * index, 100),
                                collected=collected, kept=kept, retries=retries,
                                hx=0.5, hy=0.5, prediction=(0.0, 0.0),
                                residual_px=residual)

    base = dict(
        verdict="FAIL", error_px=150.0,
        sweep_samples=90, sweep_yaw_range=25.0, sweep_pitch_range=16.0,
        points=[point(i, 20.0) for i in range(1, 10)],
        validation=[point(i, 25.0) for i in range(1, 4)],
        hx_range=(0.40, 0.60), hy_range=(0.44, 0.58), likely_cause="",
    )
    base.update(overrides)
    return SessionDiagnostics(**base)


def test_cause_for_a_passing_session_is_reassuring():
    session = new_session()
    drive(session, SyntheticUser(seed=31))
    assert session.verdict == "PASS"
    assert "no action needed" in session.diagnostics().likely_cause


@pytest.mark.parametrize("overrides,expected", [
    (dict(hx_range=(0.50, 0.53)), "iris barely moved"),
    (dict(points=[]), "No calibration point produced usable data"),
])
def test_cause_names_the_dominant_problem(overrides, expected):
    session = new_session()
    session.verdict = "FAIL"
    assert expected in session._likely_cause(diagnostics_stub(session, **overrides))


def test_cause_blames_a_short_sweep_only_in_free_head_mode():
    narrow = dict(sweep_yaw_range=4.0, sweep_pitch_range=3.0)

    free = new_session(fixed_head=False)
    free.verdict = "FAIL"
    assert "head sweep" in free._likely_cause(diagnostics_stub(free, **narrow))

    # with a head rest there is no sweep to blame — say something useful instead
    fixed = new_session()
    fixed.verdict = "FAIL"
    cause = fixed._likely_cause(diagnostics_stub(fixed, **narrow))
    assert "head sweep" not in cause


def test_cause_blames_discarded_samples():
    session = new_session()
    session.verdict = "FAIL"
    from gaze.calibration_session import PointDiagnostics

    noisy = [PointDiagnostics(index=i, target=(100 * i, 100), collected=45,
                              kept=20, retries=1, residual_px=30.0)
             for i in range(1, 10)]
    cause = session._likely_cause(diagnostics_stub(session, points=noisy))
    assert "discarded" in cause and "lighting" in cause


def test_cause_blames_head_movement_between_stages():
    session = new_session()
    session.verdict = "FAIL"
    from gaze.calibration_session import PointDiagnostics

    loose = [PointDiagnostics(index=i, target=(500 * i, 300), collected=45,
                              kept=44, retries=0, prediction=(0.0, 0.0),
                              residual_px=400.0)
             for i in range(1, 4)]
    cause = session._likely_cause(diagnostics_stub(session, validation=loose))
    assert "shifted between" in cause


def test_cause_points_at_a_single_bad_target():
    session = new_session()
    session.verdict = "MARGINAL"
    from gaze.calibration_session import PointDiagnostics

    points = [PointDiagnostics(index=i, target=(100 * i, 100), collected=45,
                               kept=44, retries=0, prediction=(0.0, 0.0),
                               residual_px=15.0)
              for i in range(1, 9)]
    points.append(PointDiagnostics(index=9, target=(1700, 900), collected=45,
                                   kept=44, retries=0, prediction=(0.0, 0.0),
                                   residual_px=220.0))
    cause = session._likely_cause(diagnostics_stub(session, points=points))
    assert "Point 9" in cause and "looked away" in cause


def test_cause_falls_back_to_general_noise():
    session = new_session()
    session.verdict = "MARGINAL"
    cause = session._likely_cause(diagnostics_stub(session))
    assert "No single dominant cause" in cause


def test_to_ascii_transliterates_report_typography():
    from gaze.calibration_session import to_ascii

    assert to_ascii("25.0° — hx 0.4–0.6 ≤ 80") == "25.0 deg - hx 0.4-0.6 <= 80"


# ----------------------------------------------------------- persistence 5.4
def test_saved_model_reloads_only_for_the_same_screen_and_camera(tmp_path):
    from gaze.calibration import CalibrationModel

    session = CalibrationSession(SCREEN, camera_index=2, rng=random.Random(99))
    drive(session, SyntheticUser(seed=17))
    assert session.verdict == "PASS"

    path = str(tmp_path / "calibration.json")
    session.save(path)

    reloaded = CalibrationModel.load(path, SCREEN, 2)
    assert reloaded is not None
    assert reloaded.validation_error_px == pytest.approx(session.error_px)
    assert reloaded.screen_size == SCREEN
    assert reloaded.camera_index == 2

    feature = session.cal_features[0]
    np.testing.assert_allclose(reloaded.predict(feature), session.model.predict(feature))

    assert CalibrationModel.load(path, (1280, 720), 2) is None   # other screen
    assert CalibrationModel.load(path, SCREEN, 0) is None        # other camera


def test_restart_clears_previous_results():
    session = new_session(fixed_head=False)
    drive(session, SyntheticUser(seed=18))
    assert session.phase is Phase.RESULTS

    session.restart()
    assert session.phase is Phase.SWEEP
    assert session.verdict == ""
    assert not session.model.comp.ok
    assert all(f is None for f in session.cal_features)

    drive(session, SyntheticUser(seed=19))
    assert session.verdict == "PASS"


# --------------------------------------------------------- fixed-head mode
def test_head_rest_mode_is_the_default():
    from config import DEFAULTS

    assert DEFAULTS["fixed_head"] is True
    assert new_session().fixed_head is True


def test_fixed_head_starts_at_the_first_target_with_no_sweep():
    session = new_session()
    assert session.phase is Phase.POINTS
    assert session.current_target() is not None
    assert session.current_target().index == 1
    assert session.current_target().total == 9


def test_fixed_head_leaves_pose_compensation_at_zero():
    session = new_session()
    compensator = session.model.comp
    assert compensator.ok, "the compensator is deliberately configured, not unfitted"
    assert np.allclose(compensator.s, 0.0), "sensitivities must be zero"
    for vector in ([0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 25.0, -18.0]):
        hx_c, hy_c = compensator.compensate(np.array(vector, dtype=float))
        assert (hx_c, hy_c) == (0.5, 0.5), "compensation must be the identity"


def test_fixed_head_session_passes_the_gate():
    session = new_session()
    drive(session, SyntheticUser(seed=50))
    assert session.phase is Phase.RESULTS
    assert session.verdict == "PASS", f"{session.error_px:.1f} px"
    assert session.diagnostics().sweep_samples == 0


def test_fixed_head_is_faster_than_running_the_sweep():
    fixed = new_session()
    free = new_session(fixed_head=False)
    assert drive(fixed, SyntheticUser(seed=51)) < drive(free, SyntheticUser(seed=51))


def test_retry_sweep_does_nothing_with_a_head_rest():
    session = new_session()
    session.retry_sweep()
    assert session.phase is Phase.POINTS, "there is no sweep to repeat"


def test_restart_keeps_the_head_mode():
    session = new_session()
    drive(session, SyntheticUser(seed=52))
    session.restart()
    assert session.fixed_head
    assert session.phase is Phase.POINTS
    assert np.allclose(session.model.comp.s, 0.0)


@pytest.mark.parametrize("fixed_head,expected", [
    (True, "head mode: fixed (head rest)"),
    (False, "head mode: free"),
])
def test_diagnostics_name_the_head_mode(fixed_head, expected):
    session = new_session(fixed_head=fixed_head)
    drive(session, SyntheticUser(seed=53))
    report = session.diagnostics()
    assert report.fixed_head is fixed_head
    assert expected in report.sweep_line()
    assert expected in report.console_report().replace(" - ", " — ")


def test_fixed_head_drift_advice_points_at_the_head_sweep():
    from gaze.calibration_session import PointDiagnostics

    session = new_session()
    session.verdict = "FAIL"
    loose = [PointDiagnostics(index=i, target=(500 * i, 300), collected=45,
                              kept=44, retries=0, prediction=(0.0, 0.0),
                              residual_px=400.0)
             for i in range(1, 4)]
    cause = session._likely_cause(diagnostics_stub(session, validation=loose))
    assert "head rest" in cause and "--with-head-sweep" in cause


# ------------------------------------------------- point-level repair (5.3)
# Offsets below are measured, not guessed: each reproduces a specific shape of
# damage against this synthetic eye (see the table in the session module).
GLANCE_ONE = (2, (-250.0, 150.0))        # one point clears 3x median and 60 px
GLANCE_TWO = (8, (250.0, -150.0))        # two do
GLANCE_SYSTEMIC = (5, (250.0, -150.0))   # three do — not a glance, a problem
GLANCE_SMALL = (8, (150.0, -90.0))       # none do


def glanced(key, offset, forever=False, seed=11, **kwargs):
    """Drive a session in which one target is looked away from as it collects."""
    session = new_session(**kwargs)
    drive(session, SyntheticUser(seed=seed, glance_points={key: offset},
                                 glance_forever=forever))
    return session


def repair_notices(session):
    return [m for m in session.messages if "fit poorly" in m]


def test_a_glanced_away_point_is_re_collected_before_validation():
    key, offset = GLANCE_ONE
    session = new_session()
    seen = []
    drive(session, SyntheticUser(seed=11, glance_points={key: offset}),
          on_frame=lambda s: seen.append((s.phase, s.refit_pass)))

    assert session.repaired == [key]
    notices = repair_notices(session)
    assert len(notices) == 1
    assert notices[0].startswith(f"point {key + 1} fit poorly (")
    assert "px vs" in notices[0] and notices[0].endswith("median) - re-collecting")

    # it ran off point-level evidence, before any verdict existed
    first_validation = next(i for i, (phase, _) in enumerate(seen)
                            if phase is Phase.VALIDATION)
    repair_frames = [i for i, (phase, refit) in enumerate(seen)
                     if phase is Phase.POINTS and refit]
    assert repair_frames, "no repair pass ran"
    assert max(repair_frames) < first_validation


def test_the_repair_rescues_the_accuracy(monkeypatch):
    """The motivating case: one bad point dragging an otherwise good session."""
    key, offset = GLANCE_ONE
    with_repair = glanced(key, offset)

    monkeypatch.setattr(CalibrationSession, "_suspect_points", lambda self: [])
    without = glanced(key, offset)

    assert without.error_px > 50.0, "the bad point should hurt when left alone"
    assert with_repair.error_px < 30.0, "and the repair should undo most of it"
    assert with_repair.error_px < without.error_px / 2


def test_a_clean_calibration_is_left_alone():
    session = new_session()
    drive(session, SyntheticUser(seed=3))
    assert session.repaired == []
    assert session.messages == []
    assert session.verdict == "PASS"


def test_a_small_wobble_does_not_trigger_a_re_collection():
    """3x the median is not enough on its own — it must clear 60 px too."""
    session = glanced(*GLANCE_SMALL)
    assert session.repaired == []
    assert repair_notices(session) == []


def test_two_bad_points_are_both_re_collected():
    session = glanced(*GLANCE_TWO)
    assert len(session.repaired) == 2
    assert len(repair_notices(session)) == 2


def test_three_outliers_are_treated_as_systemic_and_left_alone():
    """More than two is not a glance away; re-collecting would not fix it."""
    session = glanced(*GLANCE_SYSTEMIC)
    assert session.repaired == []
    assert repair_notices(session) == []
    assert session.is_finished, "it still has to produce a verdict"


def test_the_repair_never_loops():
    """A point that is bad every time costs exactly one re-collection."""
    key, offset = GLANCE_ONE
    session = glanced(key, offset, forever=True)

    assert session.repaired == [key], "one repair pass, ever"
    assert len(repair_notices(session)) == 1
    assert session.is_finished
    assert session.cal_features[key] is not None, "a repair must never lose a point"


def test_the_worse_of_the_two_collections_is_discarded():
    """Both are judged by the same model, so the comparison is fair."""
    session = new_session()
    drive(session, SyntheticUser(seed=4))
    assert session.is_finished

    idx = 0
    good = session.cal_features[idx]
    good_residual = session._residual_of(good, session.cal_targets[idx])
    bad = np.array([good[0] + 0.25, good[1] + 0.25, good[2], good[3]])

    session._repair_before = {idx: (good, good_residual, (45, 45))}
    session.cal_features[idx] = bad                  # the re-collection
    session._resolve_repair()

    assert np.allclose(session.cal_features[idx], good), "the worse one is dropped"
    assert session.repaired == [idx]
    assert "keeping the first" in session.messages[-1]


def test_a_better_re_collection_is_kept():
    session = new_session()
    drive(session, SyntheticUser(seed=4))
    idx = 0
    good = session.cal_features[idx]
    bad = np.array([good[0] + 0.25, good[1] + 0.25, good[2], good[3]])
    bad_residual = session._residual_of(bad, session.cal_targets[idx])

    session._repair_before = {idx: (bad, bad_residual, (45, 45))}
    session.cal_features[idx] = good                 # the re-collection
    session._resolve_repair()

    assert np.allclose(session.cal_features[idx], good)
    assert "re-collected:" in session.messages[-1] and "was" in session.messages[-1]


def test_a_repair_that_collects_nothing_keeps_the_original():
    session = new_session()
    drive(session, SyntheticUser(seed=4))
    idx = 0
    good = session.cal_features[idx]
    session._repair_before = {
        idx: (good, session._residual_of(good, session.cal_targets[idx]), None)}
    session.cal_features[idx] = None                 # the re-collection saw nothing
    session._resolve_repair()

    assert np.allclose(session.cal_features[idx], good), "never lose a point"


def test_a_repaired_point_is_marked_in_the_diagnostics():
    key, offset = GLANCE_ONE
    session = glanced(key, offset)
    diagnostics = session.diagnostics()

    assert [p.index for p in diagnostics.points if p.repaired] == [key + 1]
    report = diagnostics.console_report()
    assert "re-collected" in report and "fit poorly" in report
    report.encode("cp1255")            # console-safe on a Windows code page


def test_messages_are_reported_to_the_console_once():
    session = glanced(*GLANCE_ONE)
    first = session.take_messages()
    assert any("fit poorly" in m for m in first)
    assert session.take_messages() == [], "already reported"


def test_restarting_clears_the_repair_state():
    session = glanced(*GLANCE_ONE)
    assert session.repaired

    session.restart()
    assert session.repaired == []
    assert session.messages == []
    assert session.take_messages() == []
