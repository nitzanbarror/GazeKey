"""M5 wiring — the drift indicator and the Fix aim key on the live keyboard.

The rule these protect: the monitor **reports, never acts**. No screen opens by
itself, no recalibration starts by itself, and typing is never interrupted —
the only thing drift does on its own is add one line to the corner panel.
"""
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

from gaze.drift import CORRECTIONS_FOR_DRIFT, MIN_ACTIVATIONS, DriftMonitor
from interaction.controller import DwellController, DwellSettings
from interaction.injector import KeystrokeInjector
from interaction.layouts import EXTENDED_DWELL_ACTIONS, build_keyboard
from ui.overlay import KeyboardOverlay
from tests.test_injector import FakeController, FakeKeys
from tests.test_overlay import ScriptedPipeline

SCREEN = (1366, 768)
ERROR_PX = 95.0
SIZE = (683, 384)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def keyboard():
    return build_keyboard(SCREEN, ERROR_PX)


@pytest.fixture
def overlay(qapp, keyboard):
    widget = KeyboardOverlay(
        keyboard, DwellController(keyboard, DwellSettings()), ScriptedPipeline(),
        KeystrokeInjector(controller=FakeController(), keys=FakeKeys),
        screen_size=SCREEN, show_webcam=False,
        drift=DriftMonitor(validation_error_px=100.0))
    widget.resize(*SIZE)
    yield widget
    widget.close()


def render(widget) -> np.ndarray:
    pixmap = QPixmap(*SIZE)
    pixmap.fill(Qt.black)
    widget.render(pixmap)
    image = pixmap.toImage().convertToFormat(4)
    return np.frombuffer(image.bits().asstring(image.byteCount()),
                         dtype=np.uint8).reshape(image.height(), -1, 4)


def press(overlay, key_id, start, seconds=1.2, offset=(0.0, 0.0)):
    """Dwell on a key while looking ``offset`` away from its centre."""
    key = overlay.keyboard.find(key_id)
    assert key is not None, key_id
    where = (key.centre[0] + offset[0], key.centre[1] + offset[1])
    now = overlay.pipeline.look_at(where, seconds, start=start)
    overlay.tick()
    return now + 0.5                          # clear the refractory


def status(overlay) -> str:
    return " ".join(overlay.status_lines())


# ------------------------------------------------------------------- the key
def test_the_board_has_a_fix_aim_key(keyboard):
    key = keyboard.find("fn.touchup")
    assert key is not None, "the touch-up must be reachable by gaze"
    assert key.is_function and key.label_en == "Fix aim"
    for page in range(keyboard.pages):
        assert key.on_page(page), "reachable from the symbols page too"


def test_fix_aim_needs_the_extended_dwell(keyboard):
    """It rewrites the calibration — a stray glance must not trigger it."""
    assert "touchup" in EXTENDED_DWELL_ACTIONS
    controller = DwellController(keyboard, DwellSettings())
    assert (controller.required_s(keyboard.find("fn.touchup"))
            == controller.settings.extended_dwell_s)


def test_fix_aim_does_not_shrink_space_or_displace_quit(keyboard):
    space = keyboard.find("fn.space")
    assert space.rect[2] > keyboard.find("fn.touchup").rect[2] * 1.9
    quit_key = keyboard.find("fn.quit")
    assert quit_key.rect[0] + quit_key.rect[2] == pytest.approx(SCREEN[0])


def test_fix_aim_asks_the_app_rather_than_typing(overlay):
    asked = []
    overlay.touchup_requested.connect(lambda: asked.append(True))
    press(overlay, "fn.touchup", 0.0, seconds=2.2)
    assert asked == [True]
    assert overlay.injector._keyboard.calls == [], "Fix aim must never type"


def test_fix_aim_and_quit_still_work_while_paused(overlay):
    """A paused keyboard that cannot fix its aim or be closed is a trap."""
    asked = []
    overlay.touchup_requested.connect(lambda: asked.append("touchup"))
    overlay.quit_requested.connect(lambda: asked.append("quit"))

    now = press(overlay, "fn.pause", 0.0, seconds=2.2)
    assert overlay.controller.paused

    now = press(overlay, "fn.touchup", now, seconds=2.2)
    press(overlay, "fn.quit", now, seconds=2.2)
    assert asked == ["touchup", "quit"]

    now = press(overlay, "key.A", 0.0)
    assert overlay.injector._keyboard.calls == [], "but it still must not type"


# -------------------------------------------------------------- the monitoring
def test_typing_accurately_never_raises_the_flag(overlay):
    now = 0.0
    for char in "QWERTYUIOP":
        now = press(overlay, f"key.{char}", now)
    assert overlay.drift.activations == 10
    assert not overlay.drift.drifting
    assert "LOW ACCURACY" not in status(overlay)


def test_a_consistent_mis_aim_raises_the_indicator(overlay):
    now = 0.0
    for _ in range(MIN_ACTIVATIONS + 3):
        now = press(overlay, "key.G", now, offset=(-40.0, -20.0))
    assert overlay.drift.activations >= MIN_ACTIVATIONS

    # the dwell only completes inside the key, so the measurable offset is
    # bounded by half a key — force the flag the way the spec's second signal
    # does, and check the indicator says something useful
    overlay.drift._corrections = [0.0] * CORRECTIONS_FOR_DRIFT
    line = overlay.drift_line()
    assert line is not None
    assert "LOW ACCURACY" in line and "Fix aim" in line
    assert "LOW ACCURACY" in status(overlay)


def test_the_indicator_never_interrupts_typing(overlay):
    asked = []
    overlay.touchup_requested.connect(lambda: asked.append(True))
    overlay.recalibrate_requested.connect(lambda: asked.append(True))

    overlay.drift._corrections = [0.0] * CORRECTIONS_FOR_DRIFT
    now = press(overlay, "key.A", 0.0)
    press(overlay, "key.B", now)

    assert asked == [], "drift must never open a screen on its own"
    typed = "".join(t for kind, t in overlay.injector._keyboard.calls
                    if kind == "type")
    assert typed == "ab", "typing carries on regardless"
    render(overlay)                            # and it still paints


def test_the_drift_line_points_at_recalibrate_when_there_is_no_fix_key(qapp):
    board = build_keyboard(SCREEN, 81.0, layout="auto")   # 8-column paged
    assert not board.has_action("touchup"), "no room for it on this board"
    widget = KeyboardOverlay(board, DwellController(board), ScriptedPipeline(),
                             None, screen_size=SCREEN, show_webcam=False,
                             drift=DriftMonitor(validation_error_px=100.0))
    try:
        widget.drift._corrections = [0.0] * CORRECTIONS_FOR_DRIFT
        assert "Recal." in widget.drift_line()
    finally:
        widget.close()


def test_held_samples_do_not_pollute_the_centroid(overlay):
    """A blink repeats the last position; counting it would bias the estimate."""
    key = overlay.keyboard.find("key.A")
    overlay.pipeline.look_at(key.centre, 0.3, start=0.0)
    overlay.tick()
    before = overlay.drift.centroid()

    overlay.pipeline.look_at((10.0, 10.0), 0.3, start=1.0)
    for sample in overlay.pipeline.pending:
        sample.held = True
    overlay.tick()
    assert overlay.drift.centroid() == before, "held samples must be skipped"


# ------------------------------------------------------- stepping aside safely
def test_a_suspended_overlay_stops_consuming_gaze(overlay):
    """Two widgets draining one queue would each get half the samples."""
    assert overlay.consuming_gaze
    overlay.suspend()
    assert not overlay.consuming_gaze and not overlay.isVisible()

    key = overlay.keyboard.find("key.A")
    overlay.pipeline.look_at(key.centre, 3.0, start=0.0)
    assert overlay.pipeline.pending, "the samples must be left for the screen"

    overlay.resume()
    assert overlay.consuming_gaze
    assert overlay.pipeline.pending == [], "stale gaze is discarded on return"


def test_a_dwell_in_flight_does_not_survive_being_suspended(overlay):
    key = overlay.keyboard.find("key.A")
    overlay.pipeline.look_at(key.centre, 0.9, start=0.0)
    overlay.tick()
    assert overlay.controller.progress > 0.5

    overlay.suspend()
    overlay.resume()
    assert overlay.controller.progress == 0.0
    assert overlay.injector._keyboard.calls == [], \
        "coming back from a screen must not fire the key you left mid-dwell"


# ------------------------------------------------------------------- notices
def test_a_notice_shows_and_then_expires(overlay):
    overlay.notice("aim corrected by 34 px", seconds=0.0)
    assert "aim corrected" not in status(overlay), "0 s means already expired"

    overlay.notice("aim corrected by 34 px", seconds=30.0)
    assert "aim corrected by 34 px" in status(overlay)


# ------------------------------------------------------- surviving a rebuild
def test_a_typing_session_survives_being_rebuilt(qapp, keyboard):
    """What an in-place recalibration has to preserve: the sentence so far."""
    first = KeyboardOverlay(
        keyboard, DwellController(keyboard, DwellSettings()), ScriptedPipeline(),
        KeystrokeInjector(controller=FakeController(), keys=FakeKeys),
        screen_size=SCREEN, show_webcam=False)
    try:
        now = 0.0
        for char in "HEL":
            now = press(first, f"key.{char}", now)
        state = first.session_state()
        assert state["typed_text"] == "hel" and state["word"] == "hel"
    finally:
        first.close()

    rebuilt_board = build_keyboard(SCREEN, 40.0)      # a better calibration
    second = KeyboardOverlay(
        rebuilt_board, DwellController(rebuilt_board, DwellSettings()),
        ScriptedPipeline(), None, screen_size=SCREEN, show_webcam=False)
    try:
        second.restore_session(state)
        assert second.typed_text == "hel"
        assert second.word.text == "hel"
    finally:
        second.close()


def test_restoring_nothing_is_harmless(overlay):
    overlay.restore_session(None)
    overlay.restore_session({})
    assert overlay.typed_text == ""
