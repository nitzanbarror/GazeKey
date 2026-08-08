"""Setup-check screen tests — rendered offscreen, driven by scripted features.

The offscreen Qt platform has no font database, so the copy is checked through
``text_lines()`` (the same source the painter draws from) and the geometry
against painted pixels.
"""
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent, QPixmap
from PyQt5.QtWidgets import QApplication

from gaze.features import FrameFeatures
from gaze.setup_check import SetupCheckSession, SetupPhase
from ui.setup_check_screen import SetupCheckScreen
from vision.pipeline import STATE_OK, GazeSample

SCREEN = (1366, 768)
SIZE = (683, 384)
DT = 1.0 / 30.0
GOOD_SPAN = 0.051
BAD_SPAN = 0.024


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class ScriptedEye:
    """A pipeline whose frames answer whichever target is showing."""

    fps = 30.0

    def __init__(self, session, span=GOOD_SPAN, valid=True):
        self.session = session
        self.span = span
        self.valid = valid
        self.now = 0.0

    def drain(self):
        out = []
        for _ in range(30):                     # one second of frames per tick
            self.now += DT
            target = self.session.current_target()
            index = 1 if target is None else target.index
            hy = 0.30 + (0.0 if index == 1 else self.span)
            features = FrameFeatures(self.valid, 0.5, hy, 0.0, 0.0, 0.0, self.now)
            out.append(GazeSample(features, STATE_OK, self.now,
                                  x=0.0, y=0.0, is_fixating=True,
                                  stream_valid=True))
        return out


def screen_for(qapp, span=GOOD_SPAN, valid=True):
    from interaction.layouts import keyboard_region

    session = SetupCheckSession(SCREEN, region=keyboard_region(SCREEN, 2 / 3))
    widget = SetupCheckScreen(session, ScriptedEye(session, span, valid),
                              log=lambda _: None)
    widget.resize(*SIZE)
    return widget


def press(widget, key):
    widget.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, key, Qt.NoModifier))


def render(widget) -> np.ndarray:
    pixmap = QPixmap(*SIZE)
    pixmap.fill(Qt.black)
    widget.render(pixmap)
    image = pixmap.toImage().convertToFormat(4)
    return np.frombuffer(image.bits().asstring(image.byteCount()),
                         dtype=np.uint8).reshape(image.height(), -1, 4)


def run(widget, ticks=8):
    for _ in range(ticks):
        widget._tick()


# ------------------------------------------------------------------- the flow
def test_a_good_sitting_finishes_without_a_screen_to_dismiss(qapp):
    widget = screen_for(qapp, GOOD_SPAN)
    try:
        results = []
        widget.finished.connect(results.append)
        run(widget)
        assert len(results) == 1 and results[0].passed
        assert not results[0].overridden
    finally:
        widget.close()


def test_a_bad_sitting_stops_and_says_what_to_move(qapp):
    widget = screen_for(qapp, BAD_SPAN)
    try:
        results = []
        widget.finished.connect(results.append)
        run(widget)

        assert results == [], "a failed check must not start calibrating"
        assert widget.session.phase is SetupPhase.FAILED
        copy = " ".join(line for line, *_ in widget.text_lines()).lower()
        assert "camera looks too low" in copy
        assert "eye height" in copy
        assert "0.024" in copy, "the measurement is shown, never just a verdict"
        assert "space to check again" in copy and "esc to quit" in copy
    finally:
        widget.close()


def test_space_runs_the_check_again(qapp):
    widget = screen_for(qapp, BAD_SPAN)
    try:
        run(widget)
        assert widget.session.phase is SetupPhase.FAILED

        widget.pipeline.span = GOOD_SPAN          # the camera has been raised
        press(widget, Qt.Key_Space)
        assert widget.session.phase is SetupPhase.MEASURING

        results = []
        widget.finished.connect(results.append)
        run(widget)
        assert len(results) == 1 and results[0].passed
    finally:
        widget.close()


def test_enter_calibrates_anyway_and_says_so(qapp):
    widget = screen_for(qapp, BAD_SPAN)
    try:
        results = []
        widget.finished.connect(results.append)
        run(widget)
        press(widget, Qt.Key_Return)

        assert len(results) == 1
        assert results[0].overridden and not results[0].passed
    finally:
        widget.close()


def test_escape_cancels(qapp):
    widget = screen_for(qapp, BAD_SPAN)
    try:
        results = []
        widget.finished.connect(results.append)
        press(widget, Qt.Key_Escape)
        assert results == [None]
    finally:
        widget.close()


def test_it_finishes_only_once(qapp):
    widget = screen_for(qapp, GOOD_SPAN)
    try:
        results = []
        widget.finished.connect(results.append)
        run(widget, ticks=20)
        press(widget, Qt.Key_Escape)
        assert len(results) == 1
    finally:
        widget.close()


# ---------------------------------------------------------------- the picture
def test_the_dot_moves_from_the_top_target_to_the_bottom_one(qapp):
    widget = screen_for(qapp, GOOD_SPAN)
    try:
        widget._tick()
        assert widget.session.current_target().index == 1
        first = _brightest_row(render(widget))

        while widget.session.current_target().index == 1:
            widget._tick()
        second = _brightest_row(render(widget))

        assert second > first, "the second target should sit lower"
    finally:
        widget.close()


def test_the_copy_says_what_to_do_and_how_long_it_takes(qapp):
    widget = screen_for(qapp, GOOD_SPAN)
    try:
        copy = " ".join(line for line, *_ in widget.text_lines()).lower()
        assert "look at the dot" in copy
        assert "1 / 2" in copy
        assert "seconds" in copy
        assert "esc to cancel" in copy
    finally:
        widget.close()


def _brightest_row(frame) -> float:
    lit = np.argwhere(frame[..., :3].sum(axis=2) > 300)
    assert len(lit), "nothing was drawn"
    return float(np.median(lit[:, 0]))
