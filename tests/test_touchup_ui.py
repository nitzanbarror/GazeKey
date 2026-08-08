"""Touch-up screen tests — rendered offscreen, driven by scripted gaze."""
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent, QPixmap
from PyQt5.QtWidgets import QApplication

from gaze.drift import TouchUpPhase, TouchUpSession
from ui.touchup_screen import TouchUpScreen
from tests.test_overlay import ScriptedPipeline

SCREEN = (1366, 768)
SIZE = (683, 384)
CENTRE = (683.0, 384.0)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def screen(qapp):
    session = TouchUpSession(SCREEN)
    widget = TouchUpScreen(session, ScriptedPipeline())
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


def test_the_dot_is_drawn_in_the_middle(screen):
    frame = render(screen)
    lit = np.argwhere(frame[..., :3].sum(axis=2) > 200)
    assert len(lit), "nothing was drawn"
    rows, cols = lit[:, 0], lit[:, 1]
    assert abs(cols.mean() - SIZE[0] / 2) < SIZE[0] * 0.12
    assert abs(rows.mean() - SIZE[1] / 2) < SIZE[1] * 0.30


def test_holding_a_gaze_produces_a_correction(screen):
    results = []
    screen.finished.connect(results.append)
    screen.pipeline.look_at((CENTRE[0] + 40.0, CENTRE[1]), 3.0, start=0.0)
    screen._tick()

    assert len(results) == 1
    assert results[0].accepted
    assert results[0].dx == pytest.approx(-40.0, abs=1.0)


def test_the_ring_fills_while_it_measures(screen):
    before = render(screen)
    screen.pipeline.look_at(CENTRE, 1.0, start=0.0)
    screen._tick()
    assert screen.session.phase is TouchUpPhase.COLLECTING
    assert not np.array_equal(before, render(screen)), "no progress feedback"


def test_it_finishes_only_once(screen):
    results = []
    screen.finished.connect(results.append)
    screen.pipeline.look_at(CENTRE, 4.0, start=0.0)
    screen._tick()
    screen.pipeline.look_at(CENTRE, 2.0, start=5.0)
    screen._tick()
    assert len(results) == 1


def test_escape_cancels_and_reports_nothing(screen):
    results = []
    screen.finished.connect(results.append)
    screen.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape,
                                   Qt.NoModifier))
    assert results == [None]
    assert screen.session.phase is TouchUpPhase.DONE


def test_the_copy_says_what_to_do(screen):
    copy = " ".join(screen.text_lines()).lower()
    assert "look at the dot" in copy
    assert "esc" in copy


def test_a_click_sound_only_plays_on_a_real_result(qapp):
    beeps = []
    session = TouchUpSession(SCREEN)
    widget = TouchUpScreen(session, ScriptedPipeline(),
                           sound=lambda: beeps.append(True))
    try:
        widget.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape,
                                       Qt.NoModifier))
        assert beeps == [], "cancelling is not an outcome to celebrate"
    finally:
        widget.close()
