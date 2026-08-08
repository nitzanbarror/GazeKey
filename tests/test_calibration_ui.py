"""Calibration UI tests — rendered offscreen (QT_QPA_PLATFORM=offscreen).

No camera and no visible window: every screen is painted into a QPixmap and
checked for having drawn something, and the key handling is driven directly.
"""
import time

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

from gaze.calibration_session import CalibrationSession, Phase, SweepFailure
from gaze.features import FrameFeatures
from ui.calibration_screen import CalibrationScreen
from ui.gaze_demo import GazeDotDemo
from vision.pipeline import STATE_OK, GazeSample
from tests.test_calibration_session import (
    FRAME_DT,
    SCREEN,
    SyntheticUser,
    drive,
    new_session,
)

SIZE = (960, 540)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def render(widget) -> np.ndarray:
    """Paint the widget offscreen and return it as an RGB array."""
    widget.resize(*SIZE)
    pixmap = QPixmap(*SIZE)
    widget.render(pixmap)
    image = pixmap.toImage().convertToFormat(4)   # QImage.Format_RGB32
    buffer = image.bits().asstring(image.byteCount())
    return np.frombuffer(buffer, dtype=np.uint8).reshape(image.height(), -1, 4)


def ink(frame: np.ndarray) -> int:
    """How many distinct colours were painted — a cheap 'did it draw' check."""
    return len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))


def has_colour(frame: np.ndarray, colour, tolerance: int = 20) -> bool:
    """Was anything painted in (close to) this QColor? Frames are BGRA."""
    target = np.array([colour.blue(), colour.green(), colour.red()], dtype=np.int16)
    delta = np.abs(frame[..., :3].astype(np.int16) - target).max(axis=-1)
    return bool((delta <= tolerance).any())


def hue_pixels(frame: np.ndarray, kind: str, margin: int = 60) -> int:
    """Count amber ("warm") or blue ("cool") pixels.

    Strokes are drawn with alpha and antialiasing, so their exact RGB shifts;
    the red-vs-blue balance is what reliably identifies which ring was drawn.
    """
    blue = frame[..., 0].astype(np.int16)
    red = frame[..., 2].astype(np.int16)
    difference = (red - blue) if kind == "warm" else (blue - red)
    return int((difference > margin).sum())


SWEEP_PHASES = (Phase.SWEEP, Phase.SWEEP_FAILED)


def session_at(phase: Phase, seed: int = 0,
               fixed_head: bool | None = None) -> CalibrationSession:
    """Drive a synthetic session until it reaches ``phase``.

    Stage A only exists in free-head mode, so those phases force it on.
    """
    if fixed_head is None:
        fixed_head = phase not in SWEEP_PHASES
    session = new_session(fixed_head=fixed_head)
    user = SyntheticUser(seed=seed, still_head=phase is Phase.SWEEP_FAILED)
    timestamp = 1_000.0
    for _ in range(20_000):
        if session.phase is phase:
            return session
        if session.phase is Phase.RESULTS:
            break
        timestamp += FRAME_DT
        session.update(user.frame(session, timestamp))
    assert session.phase is phase, f"never reached {phase}, stuck at {session.phase}"
    return session


class FakePipeline:
    """Feeds pre-baked samples to a widget's timer-driven drain()."""

    fps = 30.0

    def __init__(self, samples=()):
        self.pending = list(samples)

    def drain(self):
        out, self.pending = self.pending, []
        return out


class SyntheticPipeline:
    """Generates one live frame per drain, aimed at whatever the session asks for."""

    fps = 30.0

    def __init__(self, session, seed: int = 0):
        self.session = session
        self.user = SyntheticUser(seed=seed)
        self.t = 1_000.0

    def drain(self):
        self.t += FRAME_DT
        features = self.user.frame(self.session, self.t)
        state = STATE_OK if features.valid else "no_face"
        return [GazeSample(features, state, self.t)]


# ------------------------------------------------------------------ rendering
# NOTE: the Qt "offscreen" platform ships no font database, so drawText() is a
# no-op here. Geometry is therefore checked against the painted pixels and the
# copy against text_lines(), which is the same source render_into() draws from.
@pytest.mark.parametrize(
    "phase",
    [Phase.SWEEP, Phase.SWEEP_FAILED, Phase.POINTS, Phase.VALIDATION, Phase.RESULTS],
)
def test_every_phase_renders_without_error(qapp, phase):
    screen = CalibrationScreen(session_at(phase), pipeline=None)
    try:
        render(screen)                              # must not raise
        assert screen.text_lines(), f"{phase} has no on-screen copy"
        assert all(line.text.strip() for line in screen.text_lines())
    finally:
        screen.close()


@pytest.mark.parametrize("phase", [Phase.SWEEP, Phase.POINTS, Phase.VALIDATION])
def test_interactive_phases_paint_their_target(qapp, phase):
    screen = CalibrationScreen(session_at(phase), pipeline=None)
    try:
        assert ink(render(screen)) > 3, f"{phase} drew no shapes"
    finally:
        screen.close()


def test_results_screen_always_shows_the_measured_error(qapp):
    session = session_at(Phase.RESULTS)
    assert session.verdict == "PASS"
    screen = CalibrationScreen(session, pipeline=None)
    try:
        copy = " ".join(line.text for line in screen.text_lines())
        assert f"{session.error_px:.0f} px" in copy
        assert "Good" in copy

        session.verdict, session.error_px = "MARGINAL", 104.0
        copy = " ".join(line.text for line in screen.text_lines())
        assert "104 px" in copy and "Usable" in copy

        session.verdict, session.error_px = "FAIL", 310.0
        copy = " ".join(line.text for line in screen.text_lines())
        assert "310 px" in copy and "Too low" in copy
        assert "calibrate again" in copy
    finally:
        screen.close()


def test_results_screen_states_the_pass_thresholds(qapp):
    screen = CalibrationScreen(session_at(Phase.RESULTS), pipeline=None)
    try:
        copy = " ".join(line.text for line in screen.text_lines())
        assert "80 px" in copy and "130 px" in copy
    finally:
        screen.close()


# --------------------------------------------------------- results diagnostics
def test_results_screen_carries_the_diagnostics_summary(qapp):
    session = session_at(Phase.RESULTS)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        copy = " ".join(line.text for line in screen.text_lines())
        assert "hx" in copy and "span" in copy
        assert "head mode" in copy
        assert screen.diagnostics().likely_cause in copy
    finally:
        screen.close()


def test_results_screen_labels_every_point(qapp):
    session = session_at(Phase.RESULTS)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        render(screen)                      # sizes the widget for the map
        labels = screen.diagnostic_labels()
        assert len(labels) == 12, "9 calibration + 3 validation labels"

        rect = screen.diagnostics_rect()
        for text, position in labels:
            assert "px" in text
            assert 0 <= position.x() <= SIZE[0]
            assert rect.top() <= position.y() <= SIZE[1]

        report = screen.diagnostics()
        for point, (text, _) in zip(report.points, labels):
            assert f"{point.kept}/{point.collected}" in text
    finally:
        screen.close()


def test_results_map_draws_points_and_validation_arrows(qapp):
    from ui.calibration_screen import GOOD

    session = session_at(Phase.RESULTS)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        frame = render(screen)
        rect = screen.diagnostics_rect()
        assert rect.width() > 0 and rect.bottom() <= SIZE[1]

        band = frame[int(rect.top()):int(rect.bottom()),
                     int(rect.left()):int(rect.right())]
        assert ink(band) > 3, "the diagnostics map drew nothing"
        assert has_colour(band, GOOD), "well-fitting points should read green"

        # an arrow leaves each validation target for its prediction
        report = screen.diagnostics()
        assert all(p.prediction is not None for p in report.validation)
    finally:
        screen.close()


def test_results_map_keeps_the_screen_aspect_ratio(qapp):
    session = session_at(Phase.RESULTS)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        render(screen)
        rect = screen.diagnostics_rect()
        assert rect.width() / rect.height() == pytest.approx(
            SCREEN[0] / SCREEN[1], rel=0.02
        )
    finally:
        screen.close()


def test_failed_results_still_report_diagnostics(qapp):
    session = session_at(Phase.RESULTS)
    session.verdict, session.error_px = "FAIL", 310.0
    screen = CalibrationScreen(session, pipeline=None)
    try:
        copy = " ".join(line.text for line in screen.text_lines())
        assert "310 px" in copy
        assert "head mode" in copy
        assert screen.diagnostics().likely_cause in copy
        render(screen)
    finally:
        screen.close()


def test_target_stage_draws_at_the_target_position(qapp):
    session = session_at(Phase.POINTS)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        target = session.current_target()
        assert target is not None
        frame = render(screen)
        cx = int(target.position[0] * SIZE[0] / SCREEN[0])
        cy = int(target.position[1] * SIZE[1] / SCREEN[1])
        patch = frame[max(0, cy - 40):cy + 40, max(0, cx - 40):cx + 40]
        assert ink(patch) > 2, "no target drawn where the session says it is"

        elsewhere = frame[max(0, cy - 40):cy + 40, :40] if cx > 120 else \
            frame[max(0, cy - 40):cy + 40, -40:]
        assert ink(elsewhere) == 1, "target should not be painted screen-wide"
    finally:
        screen.close()


def test_target_stage_counts_the_points(qapp):
    screen = CalibrationScreen(session_at(Phase.POINTS), pipeline=None)
    try:
        copy = " ".join(line.text for line in screen.text_lines())
        assert "/ 9" in copy
    finally:
        screen.close()
    screen = CalibrationScreen(session_at(Phase.VALIDATION), pipeline=None)
    try:
        copy = " ".join(line.text for line in screen.text_lines())
        assert "Checking accuracy" in copy and "/ 3" in copy
    finally:
        screen.close()


# ------------------------------------------------- calm targets (ring-only cue)
def target_patch(screen: CalibrationScreen, frame: np.ndarray,
                 pad: int = 60) -> np.ndarray:
    """Crop the area around the current target."""
    target = screen.session.current_target()
    assert target is not None
    cx = int(target.position[0] * SIZE[0] / SCREEN[0])
    cy = int(target.position[1] * SIZE[1] / SCREEN[1])
    return frame[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]


def advance(session, user, timestamp, frames):
    for _ in range(frames):
        timestamp += FRAME_DT
        session.update(user.frame(session, timestamp))
    return timestamp


def test_nothing_near_the_target_animates_on_its_own(qapp):
    """Identical session state must paint identically however time passes."""
    session = session_at(Phase.POINTS)
    user = SyntheticUser(seed=40)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        timestamp = advance(session, user, 5_000.0, 4)
        first = render(screen)
        time.sleep(0.25)                       # only the wall clock moves
        second = render(screen)
        assert np.array_equal(first, second), "something is animating on a timer"

        timestamp = advance(session, user, timestamp, 30)   # into measurement
        assert not session.current_target().settling
        third = render(screen)
        time.sleep(0.25)
        fourth = render(screen)
        assert np.array_equal(third, fourth), "measurement phase animates"
    finally:
        screen.close()


def test_target_uses_one_neutral_colour_throughout(qapp):
    session = session_at(Phase.POINTS)
    user = SyntheticUser(seed=41)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        timestamp = advance(session, user, 6_000.0, 4)
        settling = target_patch(screen, render(screen))
        assert hue_pixels(settling, "warm") == 0, "target area is not neutral"
        assert hue_pixels(settling, "cool") == 0, "target area is not neutral"

        advance(session, user, timestamp, 30)
        measuring = target_patch(screen, render(screen))
        assert hue_pixels(measuring, "warm") == 0
        assert hue_pixels(measuring, "cool") == 0
    finally:
        screen.close()


def test_ring_is_empty_while_settling_and_fills_while_measuring(qapp):
    from ui.calibration_screen import TARGET_NEUTRAL

    def ring_pixels(patch):
        return int(np.sum(np.all(
            np.abs(patch[..., :3].astype(np.int16)
                   - np.array([TARGET_NEUTRAL.blue(), TARGET_NEUTRAL.green(),
                               TARGET_NEUTRAL.red()], dtype=np.int16)) <= 12,
            axis=-1)))

    session = session_at(Phase.POINTS)
    user = SyntheticUser(seed=42)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        timestamp = advance(session, user, 7_000.0, 4)
        assert session.current_target().settling
        settling = ring_pixels(target_patch(screen, render(screen)))

        timestamp = advance(session, user, timestamp, 26)
        assert not session.current_target().settling
        early = ring_pixels(target_patch(screen, render(screen)))

        advance(session, user, timestamp, 20)
        later = ring_pixels(target_patch(screen, render(screen)))

        assert settling > 0, "the dot itself should always be drawn"
        assert early > settling, "the ring must start filling when measuring starts"
        assert later > early, "the ring must keep filling with progress"
    finally:
        screen.close()


def test_no_settle_or_measure_text_competes_with_the_ring(qapp):
    session = session_at(Phase.POINTS)
    user = SyntheticUser(seed=43)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        settling_copy = " ".join(l.text for l in screen.text_lines())
        advance(session, user, 8_000.0, 30)
        measuring_copy = " ".join(l.text for l in screen.text_lines())
        assert settling_copy == measuring_copy, "text changes between the phases"
        for phrase in ("get ready", "hold still"):
            assert phrase not in settling_copy
    finally:
        screen.close()


def test_fixed_head_target_label_does_not_mention_stage_two(qapp):
    fixed = CalibrationScreen(session_at(Phase.POINTS), pipeline=None)
    free = CalibrationScreen(
        session_at(Phase.POINTS, fixed_head=False), pipeline=None)
    try:
        assert "Step 2 of 2" not in " ".join(l.text for l in fixed.text_lines())
        assert "Look at the dot" in " ".join(l.text for l in fixed.text_lines())
        assert "Step 2 of 2" in " ".join(l.text for l in free.text_lines())
    finally:
        fixed.close()
        free.close()


def test_sweep_screen_reports_failure_guidance(qapp):
    session = session_at(Phase.SWEEP_FAILED)
    assert session.sweep_failure is SweepFailure.NO_MOTION
    screen = CalibrationScreen(session, pipeline=None)
    try:
        copy = " ".join(line.text for line in screen.text_lines())
        assert "head barely moved" in copy
        assert "turn your head" in copy
        assert "Space to repeat" in copy
    finally:
        screen.close()


def test_sweep_screen_reports_pose_coverage(qapp):
    screen = CalibrationScreen(session_at(Phase.SWEEP), pipeline=None)
    try:
        copy = " ".join(line.text for line in screen.text_lines())
        assert "left–right" in copy and "up–down" in copy
    finally:
        screen.close()


# ------------------------------------------------------------- key handling
def test_escape_cancels(qapp):
    screen = CalibrationScreen(session_at(Phase.SWEEP), pipeline=None)
    received = []
    screen.finished.connect(received.append)
    try:
        screen.keyPressEvent(_key(Qt.Key_Escape))
    finally:
        screen.close()
    assert received == [None]


def test_space_repeats_a_rejected_sweep(qapp):
    session = session_at(Phase.SWEEP_FAILED)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        screen.keyPressEvent(_key(Qt.Key_Space))
    finally:
        screen.close()
    assert session.phase is Phase.SWEEP
    assert session.sweep_failure is None


def test_space_on_a_passing_result_finishes(qapp):
    session = session_at(Phase.RESULTS)
    screen = CalibrationScreen(session, pipeline=None)
    received = []
    screen.finished.connect(received.append)
    try:
        screen.keyPressEvent(_key(Qt.Key_Space))
    finally:
        screen.close()
    assert received == [session]


def test_space_on_a_failed_result_restarts_instead_of_finishing(qapp):
    session = session_at(Phase.RESULTS)
    session.verdict = "FAIL"
    screen = CalibrationScreen(session, pipeline=None)
    received = []
    screen.finished.connect(received.append)
    try:
        screen.keyPressEvent(_key(Qt.Key_Space))
    finally:
        screen.close()
    assert received == [], "a failed calibration must not be accepted"
    assert session.phase is Phase.POINTS, "head-rest mode restarts at the targets"


def test_r_restarts_the_session(qapp):
    session = session_at(Phase.POINTS)
    screen = CalibrationScreen(session, pipeline=None)
    try:
        screen.keyPressEvent(_key(Qt.Key_R))
    finally:
        screen.close()
    assert session.phase is Phase.POINTS
    assert all(f is None for f in session.cal_features)


def _key(code):
    from PyQt5.QtGui import QKeyEvent

    return QKeyEvent(QKeyEvent.KeyPress, code, Qt.NoModifier)


# --------------------------------------------------------------- pipeline feed
def test_samples_from_the_pipeline_reach_the_session(qapp):
    session = new_session(fixed_head=False)
    samples = [
        GazeSample(FrameFeatures(True, 0.5, 0.5, 3.0, 1.0, 0.0, 1_000.0 + i * FRAME_DT),
                   STATE_OK, 1_000.0 + i * FRAME_DT)
        for i in range(20)
    ]
    screen = CalibrationScreen(session, pipeline=FakePipeline(samples))
    try:
        screen._tick()
    finally:
        screen.close()
    assert session.sweep_progress > 0.0, "samples never reached the session"


def test_badge_warns_before_any_frame_arrives(qapp):
    screen = CalibrationScreen(new_session(), pipeline=FakePipeline())
    try:
        assert screen.badge_text() == "waiting for the camera…"
        screen.pipeline.pending = [
            GazeSample(FrameFeatures(True, 0.5, 0.5, 0.0, 0.0, 0.0, 1_000.0),
                       STATE_OK, 1_000.0)
        ]
        screen._tick()
        assert screen.badge_text() is None
    finally:
        screen.close()


def test_badge_reports_lost_tracking(qapp):
    import ui.calibration_screen as module

    screen = CalibrationScreen(new_session(), pipeline=FakePipeline())
    original = module.TRACKING_LOST_S
    try:
        screen._saw_any_frame = True
        module.TRACKING_LOST_S = -1.0        # anything counts as stale
        assert screen.badge_text() == "tracking lost — face the camera"
    finally:
        module.TRACKING_LOST_S = original
        screen.close()


def test_screen_without_a_pipeline_shows_no_badge(qapp):
    screen = CalibrationScreen(new_session(), pipeline=None)
    try:
        assert screen.badge_text() is None
    finally:
        screen.close()


# ------------------------------------------------------------------ gaze demo
def test_gaze_demo_renders_and_follows_the_dot(qapp):
    sample = GazeSample(
        FrameFeatures(True, 0.5, 0.5, 0.0, 0.0, 0.0, 1_000.0),
        STATE_OK, 1_000.0, x=SCREEN[0] * 0.25, y=SCREEN[1] * 0.75,
        stream_valid=True,
    )
    demo = GazeDotDemo(FakePipeline([sample]), SCREEN, validation_error_px=54.0)
    try:
        empty = render(demo)
        demo._tick()
        with_dot = render(demo)
    finally:
        demo.close()
    assert ink(empty) > 2, "reference grid should always be drawn"
    assert not np.array_equal(empty, with_dot), "the gaze dot never appeared"
    assert "54 px" in " ".join(demo.status_lines())


def test_full_session_through_the_widget_reaches_a_passing_result(qapp):
    """End-to-end wiring: pipeline -> widget timer -> session -> results -> demo."""
    session = new_session()
    screen = CalibrationScreen(session, pipeline=SyntheticPipeline(session, seed=21))
    accepted = []
    screen.finished.connect(accepted.append)
    try:
        for _ in range(3_000):
            screen._tick()
            if session.is_finished:
                break
        assert session.is_finished, f"stuck in {session.phase}"
        assert session.verdict == "PASS", f"{session.error_px:.1f} px"

        render(screen)                                   # results screen paints
        assert f"{session.error_px:.0f} px" in " ".join(
            line.text for line in screen.text_lines()
        )
        screen.keyPressEvent(_key(Qt.Key_Space))
        assert accepted == [session]
    finally:
        screen.close()

    # the fitted model then drives the demo dot
    target = SCREEN[0] * 0.7, SCREEN[1] * 0.3
    from tests.test_pipeline import features_for

    features = features_for(*target, 2_000.0)
    x, y = session.model.predict(features.vector())
    demo = GazeDotDemo(
        FakePipeline([GazeSample(features, STATE_OK, 2_000.0, x=x, y=y,
                                 stream_valid=True)]),
        SCREEN, validation_error_px=session.error_px,
    )
    try:
        before = render(demo)
        demo._tick()
        after = render(demo)
        assert not np.array_equal(before, after)
        assert np.hypot(x - target[0], y - target[1]) < 120
    finally:
        demo.close()


def test_gaze_demo_exits_on_q(qapp):
    demo = GazeDotDemo(FakePipeline(), SCREEN)
    closed = []
    demo.closed.connect(lambda: closed.append(True))
    demo.keyPressEvent(_key(Qt.Key_Q))
    assert closed == [True]


def test_gaze_demo_handles_a_pipeline_with_no_gaze(qapp):
    demo = GazeDotDemo(FakePipeline(), SCREEN)
    try:
        demo._tick()
        assert ink(render(demo)) > 2
        assert "not measured" in " ".join(demo.status_lines())
    finally:
        demo.close()


def demo_sample(**overrides) -> GazeSample:
    fields = dict(x=SCREEN[0] / 2, y=SCREEN[1] / 2, stream_valid=True)
    fields.update(overrides)
    return GazeSample(FrameFeatures(True, 0.5, 0.5, 0.0, 0.0, 0.0, 1_000.0),
                      STATE_OK, 1_000.0, **fields)


# ------------------------------------------------------- M3 fixation feedback
def test_dot_turns_green_while_fixating_and_blue_while_moving(qapp):
    from ui.gaze_demo import FIXATING, MOVING

    demo = GazeDotDemo(FakePipeline(), SCREEN)
    try:
        demo.pipeline.pending = [demo_sample(is_fixating=False)]
        demo._tick()
        assert demo.dot_colour() == MOVING
        moving_frame = render(demo)
        assert hue_pixels(moving_frame, "cool") > 0

        demo.pipeline.pending = [demo_sample(is_fixating=True)]
        demo._tick()
        assert demo.dot_colour() == FIXATING
        fixating_frame = render(demo)
        assert not np.array_equal(moving_frame, fixating_frame)
        assert has_colour(fixating_frame, FIXATING), "fixation colour missing"
    finally:
        demo.close()


def test_dot_turns_amber_while_holding_through_a_blink(qapp):
    from ui.gaze_demo import HELD

    demo = GazeDotDemo(FakePipeline(), SCREEN)
    try:
        demo.pipeline.pending = [demo_sample(is_fixating=True, held=True)]
        demo._tick()
        assert demo.dot_colour() == HELD
        frame = render(demo)
        assert hue_pixels(frame, "warm") > 0
        assert "holding through blink" in " ".join(demo.status_lines())
    finally:
        demo.close()


def test_dot_disappears_when_the_stream_goes_invalid(qapp):
    demo = GazeDotDemo(FakePipeline(), SCREEN)
    try:
        demo.pipeline.pending = [demo_sample(is_fixating=True)]
        demo._tick()
        with_dot = render(demo)

        demo.pipeline.pending = [
            GazeSample(FrameFeatures(valid=False, timestamp=1_001.0),
                       "no_face", 1_001.0, stream_valid=False)
        ]
        demo._tick()
        lost = render(demo)

        assert not np.array_equal(with_dot, lost)
        assert "tracking lost" in " ".join(demo.status_lines())
    finally:
        demo.close()


def test_demo_reports_the_fixation_settings(qapp):
    class Configured(FakePipeline):
        fixation_dispersion_px = 110.0
        fixation_window_ms = 150.0

    demo = GazeDotDemo(Configured(), SCREEN, validation_error_px=81.0)
    try:
        copy = " ".join(demo.status_lines())
        assert "dispersion <= 110 px" in copy
        assert "over 150 ms" in copy
        assert "green = fixating" in copy and "blue = moving" in copy
        demo.pipeline.pending = [demo_sample(is_fixating=True)]
        demo._tick()
        assert "FIXATING" in " ".join(demo.status_lines())
    finally:
        demo.close()
