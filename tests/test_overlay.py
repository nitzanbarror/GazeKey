"""Keyboard overlay tests — rendered offscreen, driven by scripted samples.

Covers the two things that silently break gaze typing: the window flags (an
overlay that takes focus swallows every keystroke it injects) and the wiring
from sample to activation to injection.
"""
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

from gaze.features import FrameFeatures
from interaction.controller import DwellController, DwellSettings
from interaction.layouts import build_keyboard
from ui.overlay import KeyboardOverlay
from vision.pipeline import STATE_OK, GazeSample
from tests.test_injector import FakeController, FakeKeys
from interaction.injector import KeystrokeInjector

SCREEN = (1366, 768)
ERROR_PX = 81.0
SIZE = (683, 384)          # rendered at half scale
DT = 1.0 / 30.0


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def keyboard():
    return build_keyboard(SCREEN, ERROR_PX)      # the real default: qwerty-tall


class ScriptedPipeline:
    """Hands the overlay a queue of samples, one drain at a time."""

    fps = 30.0

    def __init__(self):
        self.pending = []

    def drain(self):
        out, self.pending = self.pending, []
        return out

    def look_at(self, position, seconds, start, is_fixating=True,
                stream_valid=True, dt=DT):
        now = start
        for _ in range(max(1, int(round(seconds / dt)))):
            now += dt
            self.pending.append(GazeSample(
                FrameFeatures(True, 0.5, 0.5, 0.0, 0.0, 0.0, now),
                STATE_OK, now, x=position[0], y=position[1],
                is_fixating=is_fixating, stream_valid=stream_valid,
            ))
        return now


@pytest.fixture
def overlay(qapp, keyboard):
    controller = DwellController(keyboard, DwellSettings())
    injector = KeystrokeInjector(controller=FakeController(), keys=FakeKeys)
    widget = KeyboardOverlay(keyboard, controller, ScriptedPipeline(), injector,
                             screen_size=SCREEN, show_webcam=False)
    widget.resize(*SIZE)
    yield widget
    widget.close()


def render(widget) -> np.ndarray:
    pixmap = QPixmap(*SIZE)
    pixmap.fill(Qt.black)
    widget.render(pixmap)
    image = pixmap.toImage().convertToFormat(4)
    buffer = image.bits().asstring(image.byteCount())
    return np.frombuffer(buffer, dtype=np.uint8).reshape(image.height(), -1, 4)


def calls(overlay):
    return overlay.injector._keyboard.calls


# ------------------------------------------------------------- window contract
def test_overlay_never_takes_keyboard_focus(overlay):
    """If it did, every injected keystroke would land back in GazeKey."""
    flags = overlay.windowFlags()
    assert flags & Qt.WindowDoesNotAcceptFocus
    assert flags & Qt.WindowStaysOnTopHint
    assert flags & Qt.FramelessHint if hasattr(Qt, "FramelessHint") else True
    assert flags & Qt.FramelessWindowHint
    assert flags & Qt.Tool
    assert overlay.focusPolicy() == Qt.NoFocus
    assert overlay.testAttribute(Qt.WA_ShowWithoutActivating)


def test_overlay_is_translucent_and_click_through(overlay):
    assert overlay.testAttribute(Qt.WA_TranslucentBackground)
    assert overlay.testAttribute(Qt.WA_TransparentForMouseEvents), \
        "the overlay must not swallow clicks meant for the app behind it"


# -------------------------------------------------------------------- painting
def test_overlay_draws_the_keyboard(overlay):
    frame = render(overlay)
    board_top = int(overlay.keyboard.rect[1] * SIZE[1] / SCREEN[1])
    assert len(np.unique(frame[board_top:].reshape(-1, 4), axis=0)) > 3

    # The app being typed into has to stay visible: everything above the board
    # is left alone, apart from the capped status panel in the top-left corner.
    from ui.overlay import STATUS_MAX_WIDTH_RATIO

    edge = int(SIZE[0] * (STATUS_MAX_WIDTH_RATIO + 0.08))
    clear = frame[:board_top - 2, edge:]
    assert len(np.unique(clear.reshape(-1, 4), axis=0)) == 1, \
        "the overlay is painting over the application behind it"


def test_focused_key_and_progress_ring_are_drawn(overlay):
    key = overlay.keyboard.find("key.A")
    idle = render(overlay)
    overlay.pipeline.look_at(key.centre, 0.5, start=0.0)
    overlay.tick()
    focused = render(overlay)
    assert not np.array_equal(idle, focused), "no focus feedback drawn"
    assert overlay.controller.progress > 0.3


def test_gaze_cursor_can_be_switched_off(qapp, keyboard):
    controller = DwellController(keyboard, DwellSettings())
    on = KeyboardOverlay(keyboard, controller, ScriptedPipeline(), None,
                         screen_size=SCREEN, show_cursor=True, show_webcam=False)
    off = KeyboardOverlay(keyboard, DwellController(keyboard, DwellSettings()),
                          ScriptedPipeline(), None, screen_size=SCREEN,
                          show_cursor=False, show_webcam=False)
    try:
        for widget in (on, off):
            widget.resize(*SIZE)
            widget.pipeline.look_at((300.0, 100.0), 0.1, start=0.0)
            widget.tick()
        assert not np.array_equal(render(on), render(off))
    finally:
        on.close()
        off.close()


# --------------------------------------------------------------------- typing
def test_dwelling_on_a_key_injects_it(overlay):
    key = overlay.keyboard.find("key.A")
    overlay.pipeline.look_at(key.centre, 1.2, start=0.0)
    overlay.tick()
    assert calls(overlay) == [("type", "a")]
    assert overlay.history[-1] == "A"


def test_typing_hello_through_the_overlay(overlay):
    """The M4 checkpoint driven by scripted gaze instead of eyes."""
    now = 0.0
    for char in "HELLO":
        key = overlay.keyboard.find(f"key.{char}")
        assert key is not None and key.on_page(overlay.controller.page)
        now = overlay.pipeline.look_at(key.centre, 1.2, start=now)
        overlay.tick()
        now += 0.5                                  # let the refractory pass
    typed = "".join(text for kind, text in calls(overlay) if kind == "type")
    assert typed == "hello"


def test_space_and_backspace_use_key_codes(overlay):
    now = overlay.pipeline.look_at(
        overlay.keyboard.find("fn.space").centre, 1.2, start=0.0)
    overlay.tick()
    overlay.pipeline.look_at(
        overlay.keyboard.find("fn.backspace").centre, 1.2, start=now + 0.5)
    overlay.tick()
    assert calls(overlay) == [
        ("press", "space"), ("release", "space"),
        ("press", "backspace"), ("release", "backspace"),
    ]


def test_shift_capitalises_the_next_letter_only(overlay):
    now = overlay.pipeline.look_at(
        overlay.keyboard.find("fn.shift").centre, 1.2, start=0.0)
    overlay.tick()
    assert overlay.injector.shift_latched

    now = overlay.pipeline.look_at(
        overlay.keyboard.find("key.A").centre, 1.2, start=now + 0.5)
    overlay.tick()
    overlay.pipeline.look_at(
        overlay.keyboard.find("key.B").centre, 1.2, start=now + 0.5)
    overlay.tick()
    assert [text for _, text in calls(overlay)] == ["A", "b"]


# ---------------------------------------------------------------- app commands
def test_the_symbols_key_switches_pages_without_typing(overlay):
    overlay.pipeline.look_at(overlay.keyboard.find("fn.symbols").centre,
                             1.2, start=0.0)
    overlay.tick()
    assert overlay.controller.page == 1
    assert calls(overlay) == []
    assert "page 2 of 2" in " ".join(overlay.status_lines())
    assert overlay.keyboard.find("sym.0.0") is not None, "digits on the 123 page"


def test_pause_stops_typing_but_keeps_the_feedback(overlay):
    now = overlay.pipeline.look_at(
        overlay.keyboard.find("fn.pause").centre, 2.2, start=0.0)
    overlay.tick()
    assert overlay.controller.paused
    assert "PAUSED" in " ".join(overlay.status_lines())

    key = overlay.keyboard.find("key.A")
    overlay.pipeline.look_at(key.centre, 1.2, start=now + 0.5)
    overlay.tick()
    assert calls(overlay) == [], "a paused keyboard must not type"
    assert "ignored" in overlay.history[-1]
    render(overlay)                                  # still paints its feedback


def test_pause_can_be_lifted_again(overlay):
    pause = overlay.keyboard.find("fn.pause")
    now = overlay.pipeline.look_at(pause.centre, 2.2, start=0.0)
    overlay.tick()
    overlay.pipeline.look_at(pause.centre, 2.2, start=now + 0.5)
    overlay.tick()
    assert not overlay.controller.paused


def test_recalibrate_asks_the_app_rather_than_typing(overlay):
    seen = []
    overlay.recalibrate_requested.connect(lambda: seen.append(True))
    overlay.pipeline.look_at(
        overlay.keyboard.find("fn.recalibrate").centre, 2.2, start=0.0)
    overlay.tick()
    assert seen == [True]
    assert calls(overlay) == []


def test_the_lang_key_swaps_the_labels(overlay):
    assert overlay.language == "en"
    overlay.pipeline.look_at(overlay.keyboard.find("fn.lang").centre, 2.2, start=0.0)
    overlay.tick()
    assert overlay.language == "he"
    assert overlay.renderer.language == "he"
    assert calls(overlay) == []


# --------------------------------------------------------------------- status
def test_status_reports_tracking_loss(overlay):
    assert "waiting for the camera" in " ".join(overlay.status_lines())
    overlay.pipeline.look_at((300.0, 400.0), 0.1, start=0.0, stream_valid=False)
    overlay.tick()
    assert "tracking lost" in " ".join(overlay.status_lines())


def test_status_warns_when_injection_is_unavailable(qapp, keyboard):
    widget = KeyboardOverlay(keyboard, DwellController(keyboard), ScriptedPipeline(),
                             injector=None, screen_size=SCREEN, show_webcam=False)
    try:
        assert "no keystroke injection" in " ".join(widget.status_lines())
        widget.pipeline.look_at(keyboard.find("key.A").centre, 1.2, start=0.0)
        widget.tick()                     # must not raise without an injector
        assert "no injector" in widget.history[-1]
    finally:
        widget.close()


def test_status_warns_when_the_keys_are_below_nfr2(qapp):
    board = build_keyboard(SCREEN, 260.0, layout="auto")
    assert not board.meets_nfr2
    widget = KeyboardOverlay(board, DwellController(board), ScriptedPipeline(),
                             None, screen_size=SCREEN, show_webcam=False)
    try:
        status = " ".join(widget.status_lines())
        assert "keys below spec" in status
        # both axes, always: a warning that only quotes the width sends the
        # user chasing the wrong fix
        assert "width" in status and "height" in status
        assert f"{board.min_key_px:.0f} px" in status
        # and it must not be confused with the drift indicator, which is what
        # spec Section 9 calls "low accuracy" and has a different remedy
        assert "LOW ACCURACY" not in status
    finally:
        widget.close()


def test_a_lost_stream_does_not_advance_the_dwell(overlay):
    key = overlay.keyboard.find("key.A")
    overlay.pipeline.look_at(key.centre, 3.0, start=0.0, stream_valid=False)
    overlay.tick()
    assert calls(overlay) == [], "typing must not happen while tracking is lost"
