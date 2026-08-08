"""M5 application flow — touch-up and in-place recalibration inside main.py.

Driven through :class:`main.GazeKeyApp` with stub screens, because what is
being checked is the *wiring*: what gets hidden, what comes back, and what
survives. The screens themselves are covered by their own tests.
"""
import pytest
from PyQt5.QtWidgets import QApplication

import main as app_main
from gaze.calibration import CalibrationModel
from gaze.drift import DriftMonitor, TouchUpResult
from tests.test_main_flow import StubApp, parse

SCREEN = (1366, 768)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class StubWidget:
    """Anything main.py hides, shows or closes."""

    def __init__(self):
        self.visible = True
        self.closed = False
        self.notices = []
        self.consuming_gaze = True

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def suspend(self):
        self.visible = False
        self.consuming_gaze = False

    def resume(self):
        self.visible = True
        self.consuming_gaze = True

    def close(self):
        self.closed = True

    def notice(self, text, seconds=4.0):
        self.notices.append(text)

    def session_state(self):
        return {"typed_text": "hel", "word": "hel", "language": "en",
                "page": 0, "paused": False}

    def restore_session(self, state):
        self.restored = state


class StubPipeline:
    fixation_dispersion_px = 110.0
    fixation_window_ms = 150.0

    def __init__(self):
        self.models = []

    def set_model(self, model):
        self.models.append(model)

    def drain(self):
        return []


@pytest.fixture
def gazekey(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(app_main, "calibration_path",
                        lambda: str(tmp_path / "calibration.json"))
    app = app_main.GazeKeyApp(StubApp(), parse())
    app.pipeline = StubPipeline()
    app.model = CalibrationModel(screen_size=SCREEN)
    app.model.comp.ok = True
    app.error_px = 95.0
    app.drift = DriftMonitor(validation_error_px=95.0)
    app.overlay = app.screen = StubWidget()
    app.exit_button = StubWidget()
    return app


# --------------------------------------------------- suspending and resuming
def test_suspending_hides_the_keyboard_and_its_x(gazekey):
    gazekey._suspend_keyboard()
    assert not gazekey.overlay.visible
    assert not gazekey.exit_button.visible
    assert not gazekey.overlay.consuming_gaze, \
        "a hidden overlay must stop draining the pipeline, not just stop showing"


def test_resuming_brings_back_every_exit_route(gazekey):
    """The X is one of four ways out — leaving it hidden quietly costs one."""
    gazekey._suspend_keyboard()
    gazekey._resume_keyboard()
    assert gazekey.overlay.visible
    assert gazekey.exit_button.visible, "the clickable X must come back"
    assert gazekey.screen is gazekey.overlay


def test_declining_to_quit_restores_the_x_button(gazekey, monkeypatch):
    """Regression: answering NO used to leave the X hidden for the session."""
    answered = {}

    class FakeChoiceScreen:
        def __init__(self, *args, **kwargs):
            self.closed = False

        class _Signal:
            def __init__(self, store):
                self.store = store

            def connect(self, slot):
                self.store["slot"] = slot

        @property
        def chosen(self):
            return self._Signal(answered)

        def showFullScreen(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(app_main, "ChoiceScreen", FakeChoiceScreen)
    gazekey.confirm_quit()
    assert not gazekey.exit_button.visible, "hidden while the question is up"

    answered["slot"]("no")
    assert gazekey.overlay.visible
    assert gazekey.exit_button.visible
    assert gazekey.app.quit_calls == 0


def test_confirming_quit_still_quits(gazekey, monkeypatch):
    answered = {}

    class FakeChoiceScreen:
        def __init__(self, *args, **kwargs):
            pass

        class _Signal:
            def connect(self, slot):
                answered["slot"] = slot

        @property
        def chosen(self):
            return self._Signal()

        def showFullScreen(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(app_main, "ChoiceScreen", FakeChoiceScreen)
    gazekey.confirm_quit()
    answered["slot"]("yes")
    assert gazekey.app.quit_calls == 1


# ---------------------------------------------------------------- touch-up
class FakeTouchUpScreen:
    """Captures the finished callback so a test can supply the result."""

    instances = []

    def __init__(self, session, pipeline=None, sound=None):
        self.session = session
        self.slot = None
        self.closed = False
        FakeTouchUpScreen.instances.append(self)

    class _Signal:
        def __init__(self, screen):
            self.screen = screen

        def connect(self, slot):
            self.screen.slot = slot

    @property
    def finished(self):
        return self._Signal(self)

    def showFullScreen(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def touchup(gazekey, monkeypatch):
    FakeTouchUpScreen.instances = []
    monkeypatch.setattr(app_main, "TouchUpScreen", FakeTouchUpScreen)
    return gazekey


def test_the_touch_up_suspends_the_keyboard_and_shows_one_target(touchup):
    touchup.on_touchup()
    screen = FakeTouchUpScreen.instances[-1]
    assert screen.session.budget_s < 10.0, "back to typing in under 10 s"
    assert not touchup.overlay.visible

    # the dot sits at the centre of the calibrated region, not of the screen —
    # a correction measured outside the fitted area is not a correction
    assert screen.session.target == touchup.region.centre
    assert touchup.region.contains(*screen.session.target)
    assert screen.session.target[1] > SCREEN[1] / 2, "region is the lower board"


def test_an_accepted_touch_up_moves_the_model_and_returns_to_typing(touchup,
                                                                    tmp_path):
    before = float(touchup.model.wx[0])
    touchup.on_touchup()
    screen = FakeTouchUpScreen.instances[-1]
    screen.session.result = TouchUpResult(dx=30.0, dy=-12.0, samples=40,
                                          accepted=True)
    screen.slot(screen.session.result)

    assert touchup.model.wx[0] == pytest.approx(before + 30.0)
    assert touchup.pipeline.models[-1] is touchup.model, "smoothing reset too"
    assert touchup.overlay.visible and touchup.exit_button.visible
    assert "corrected" in touchup.overlay.notices[-1]
    assert screen.closed


def test_an_accepted_touch_up_is_persisted(touchup, tmp_path):
    touchup.on_touchup()
    screen = FakeTouchUpScreen.instances[-1]
    screen.session.result = TouchUpResult(dx=30.0, dy=-12.0, samples=40,
                                          accepted=True)
    screen.slot(screen.session.result)
    saved = tmp_path / "calibration.json"
    assert saved.exists(), "M5 still writes the calibration it corrected"


def test_an_accepted_touch_up_clears_the_drift_flag(touchup):
    touchup.drift._corrections = [0.0, 0.0, 0.0]
    assert touchup.drift.drifting

    touchup.on_touchup()
    screen = FakeTouchUpScreen.instances[-1]
    screen.session.result = TouchUpResult(dx=10.0, dy=5.0, samples=40,
                                          accepted=True)
    screen.slot(screen.session.result)
    assert not touchup.drift.drifting, "the evidence was about the old model"


def test_a_refused_touch_up_leaves_the_model_alone(touchup):
    before = float(touchup.model.wx[0])
    touchup.on_touchup()
    screen = FakeTouchUpScreen.instances[-1]
    screen.slot(TouchUpResult(accepted=False, reason="too far to be drift"))

    assert touchup.model.wx[0] == before
    assert touchup.overlay.visible, "still goes straight back to typing"
    assert "too far" in touchup.overlay.notices[-1]


def test_cancelling_the_touch_up_returns_to_typing(touchup):
    before = float(touchup.model.wx[0])
    touchup.on_touchup()
    FakeTouchUpScreen.instances[-1].slot(None)
    assert touchup.model.wx[0] == before
    assert touchup.overlay.visible and touchup.exit_button.visible


def test_a_touch_up_without_a_model_does_nothing(touchup):
    touchup.model = None
    touchup.on_touchup()
    assert FakeTouchUpScreen.instances == []
    assert touchup.overlay.visible


# ------------------------------------------------------- in-place recalibration
def test_recalibrating_keeps_the_typed_text_and_skips_practice(gazekey,
                                                               monkeypatch):
    calls = {}

    def fake_run_calibration(on_done=None, cancel_quits=True):
        calls["on_done"] = on_done
        calls["cancel_quits"] = cancel_quits

    rebuilt = {}

    def fake_run_keyboard(state=None):
        rebuilt["state"] = state
        gazekey.overlay = StubWidget()

    monkeypatch.setattr(gazekey, "run_calibration", fake_run_calibration)
    monkeypatch.setattr(gazekey, "run_keyboard", fake_run_keyboard)
    monkeypatch.setattr(gazekey, "after_calibration",
                        lambda: calls.setdefault("practice", True))

    gazekey.on_recalibrate()
    assert not gazekey.overlay.visible, "the board steps aside to calibrate"
    assert calls["cancel_quits"] is False, "Esc must not kill the session"

    gazekey.error_px = 52.0
    gazekey.model = CalibrationModel(screen_size=SCREEN)   # the new fit
    calls["on_done"]()                       # the calibration screen finishes

    assert rebuilt["state"]["typed_text"] == "hel", "the sentence survives"
    assert "practice" not in calls, "straight back to typing, no drill"
    assert gazekey.overlay.visible and gazekey.exit_button.visible
    assert "52 px" in gazekey.overlay.notices[-1]
    assert gazekey.pipeline.models[-1] is gazekey.model, \
        "the new fit has to reach the pipeline, or nothing actually changed"


def test_recalibrating_does_not_go_back_to_a_startup_screen(gazekey):
    """There is no startup screen any more — the only way back is the keyboard."""
    assert not hasattr(gazekey, "ask_use_saved")
    source = app_main.GazeKeyApp.on_recalibrate.__doc__ or ""
    assert "in place" in source


def test_a_cancelled_in_place_recalibration_returns_to_typing(gazekey,
                                                              monkeypatch):
    captured = {}

    def fake_run_calibration(on_done=None, cancel_quits=True):
        captured["on_done"] = on_done

    monkeypatch.setattr(gazekey, "run_calibration", fake_run_calibration)
    monkeypatch.setattr(gazekey, "run_keyboard",
                        lambda state=None: setattr(gazekey, "overlay",
                                                   StubWidget()))
    gazekey.on_recalibrate()
    captured["on_done"]()                    # cancel_quits=False path
    assert gazekey.overlay.visible
    assert gazekey.app.quit_calls == 0, "cancelling mid-session must not quit"


# -------------------------------------------------------------- the region
def test_the_default_region_is_the_keyboard_area(gazekey):
    from interaction.layouts import keyboard_region

    assert gazekey.region == keyboard_region(SCREEN, 2 / 3)
    assert gazekey.region.share_of(SCREEN) < 1.0


def test_cal_region_full_restores_whole_screen_calibration(qapp):
    from gaze.region import full_screen_region

    app = app_main.GazeKeyApp(StubApp(), parse("--cal-region", "full"))
    assert app.region == full_screen_region(SCREEN)


def test_an_error_sized_layout_gets_a_region_that_bounds_it(qapp):
    """The paged board's height depends on the error, so bound it instead."""
    from interaction.layouts import (
        MAX_HEIGHT_RATIO,
        REGION_MARGIN_RATIO,
        build_keyboard,
    )

    app = app_main.GazeKeyApp(StubApp(), parse("--layout", "auto"))
    assert app.region.h > app_main.GazeKeyApp(StubApp(), parse()).region.h
    for error in (60.0, 81.0, 85.0):
        board = build_keyboard(SCREEN, error, layout="auto")
        assert all(app.region.holds_rect(k.rect)
                   for k in board.keys if k.selectable), f"at {error} px"
    assert app.region.h == pytest.approx(
        SCREEN[1] * (MAX_HEIGHT_RATIO + REGION_MARGIN_RATIO), abs=1.0)


def test_a_touch_up_reuses_the_calibrated_region(touchup):
    touchup.on_touchup()
    session = FakeTouchUpScreen.instances[-1].session
    assert session.region == touchup.region


def test_a_touch_up_rewrites_the_region_alongside_the_model(touchup, tmp_path):
    from gaze.region import read_region

    touchup.on_touchup()
    screen = FakeTouchUpScreen.instances[-1]
    screen.session.result = TouchUpResult(dx=8.0, dy=3.0, samples=40,
                                          accepted=True)
    screen.slot(screen.session.result)
    # model.save() rewrites the whole file, so the region has to be re-attached
    assert read_region(str(tmp_path / "calibration.json")) == touchup.region


def test_an_in_place_recalibration_keeps_the_same_region(gazekey, monkeypatch):
    captured = {}

    def fake_session(screen_size, **kwargs):
        captured["region"] = kwargs.get("region")
        raise RuntimeError("stop here")

    monkeypatch.setattr(app_main, "CalibrationSession", fake_session)
    monkeypatch.setattr(app_main, "CalibrationScreen", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        gazekey.on_recalibrate()
    assert captured["region"] == gazekey.region


def test_the_saved_region_wins_over_the_computed_one(gazekey, monkeypatch,
                                                     tmp_path):
    """--use-saved must measure against the area that error was measured in."""
    from gaze.region import Region, attach_region

    path = tmp_path / "calibration.json"
    path.write_text('{"wx": [1]}')
    stored = Region((0.0, 300.0, 1366.0, 468.0), name="keyboard")
    attach_region(str(path), stored)

    monkeypatch.setattr(gazekey, "after_calibration", lambda: None)
    monkeypatch.setattr(CalibrationModel, "load",
                        classmethod(lambda cls, *a: CalibrationModel(
                            validation_error_px=61.0)))
    gazekey.args = parse("--use-saved")
    gazekey._start_calibration_or_saved()
    assert gazekey.region == stored


# ------------------------------------------------------------------- startup
def test_every_launch_calibrates(gazekey, monkeypatch):
    """No stored calibration is trusted by default: the head has been re-seated."""
    ran = []
    monkeypatch.setattr(gazekey, "run_calibration",
                        lambda *a, **k: ran.append("calibrate"))
    monkeypatch.setattr(gazekey, "after_calibration",
                        lambda: ran.append("practice"))
    monkeypatch.setattr(CalibrationModel, "load",
                        classmethod(lambda cls, *a: CalibrationModel()))

    gazekey.args = parse()
    gazekey._start_calibration_or_saved()
    assert ran == ["calibrate"], "a saved model must not short-circuit the flow"


def test_the_dev_flag_can_still_load_a_saved_calibration(gazekey, monkeypatch):
    ran = []
    saved = CalibrationModel(validation_error_px=61.0)
    monkeypatch.setattr(gazekey, "run_calibration",
                        lambda *a, **k: ran.append("calibrate"))
    monkeypatch.setattr(gazekey, "after_calibration",
                        lambda: ran.append("practice"))
    monkeypatch.setattr(CalibrationModel, "load",
                        classmethod(lambda cls, *a: saved))

    gazekey.args = parse("--use-saved")
    gazekey._start_calibration_or_saved()
    assert ran == ["practice"]
    assert gazekey.model is saved and gazekey.error_px == 61.0


def test_the_drill_is_opt_in(gazekey, monkeypatch):
    """calibrate -> keyboard, with no ten-target tax on every launch."""
    went = []
    monkeypatch.setattr(gazekey, "run_keyboard",
                        lambda state=None: went.append("keyboard"))
    monkeypatch.setattr(app_main, "PracticeScreen",
                        lambda *a, **k: went.append("practice"))

    gazekey.args = parse()
    gazekey.after_calibration()
    assert went == ["keyboard"]


def test_asking_for_the_drill_still_runs_it(gazekey, monkeypatch):
    built = {}

    class FakePracticeScreen:
        def __init__(self, session, *a, **k):
            built["session"] = session

        class _Signal:
            def connect(self, slot):
                pass

        @property
        def finished(self):
            return self._Signal()

        def showFullScreen(self):
            pass

    monkeypatch.setattr(app_main, "PracticeScreen", FakePracticeScreen)
    monkeypatch.setattr(gazekey, "_show", lambda widget: None)

    gazekey.args = parse("--practice")
    gazekey.after_calibration()
    session = built["session"]
    assert session.targets == 10
    assert session.region == gazekey.region, "the drill measures in-region"


def test_the_dev_flag_falls_back_to_calibrating(gazekey, monkeypatch):
    ran = []
    monkeypatch.setattr(gazekey, "run_calibration",
                        lambda *a, **k: ran.append("calibrate"))
    monkeypatch.setattr(CalibrationModel, "load",
                        classmethod(lambda cls, *a: None))

    gazekey.args = parse("--use-saved")
    gazekey._start_calibration_or_saved()
    assert ran == ["calibrate"], "nothing stored for this screen and camera"
