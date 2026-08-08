"""Gaze-choice screen tests — every question answerable by looking.

Since the startup "use the saved calibration?" question was removed (every
launch calibrates), the one question left is the Quit confirmation, so that is
what these drive.
"""
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent, QPixmap
from PyQt5.QtWidgets import QApplication

from interaction.controller import DwellSettings
from ui.choice_screen import ChoiceScreen
from tests.test_overlay import ScriptedPipeline

SCREEN = (1366, 768)
SIZE = (683, 384)
OPTIONS = [("yes", "YES, QUIT"), ("no", "NO, KEEP TYPING")]


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def screen(qapp):
    widget = ChoiceScreen("Quit GazeKey?", OPTIONS, SCREEN,
                          ScriptedPipeline(), DwellSettings(),
                          subtitle="Look at an answer and hold.")
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


def test_the_options_are_huge_targets(screen):
    for key in screen.board.keys:
        _, _, w, h = key.rect
        assert w > SCREEN[0] * 0.3 and h > SCREEN[1] * 0.2


def test_dwelling_on_an_option_answers_it(screen):
    answers = []
    screen.chosen.connect(answers.append)
    option = screen.board.find("choice.yes")
    screen.pipeline.look_at(option.centre, 1.2, start=0.0)
    screen._tick()
    assert answers == ["yes"]


def test_the_other_option_answers_differently(screen):
    answers = []
    screen.chosen.connect(answers.append)
    option = screen.board.find("choice.no")
    screen.pipeline.look_at(option.centre, 1.2, start=0.0)
    screen._tick()
    assert answers == ["no"]


def test_an_answer_is_emitted_only_once(screen):
    answers = []
    screen.chosen.connect(answers.append)
    option = screen.board.find("choice.yes")
    screen.pipeline.look_at(option.centre, 3.0, start=0.0)
    screen._tick()
    screen._tick()
    assert len(answers) == 1


def test_a_moving_gaze_answers_nothing(screen):
    answers = []
    screen.chosen.connect(answers.append)
    option = screen.board.find("choice.yes")
    screen.pipeline.look_at(option.centre, 3.0, start=0.0, is_fixating=False)
    screen._tick()
    assert answers == []


def test_escape_cancels(screen):
    answers = []
    screen.chosen.connect(answers.append)
    screen.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape,
                                   Qt.NoModifier))
    assert answers == [None]


def test_the_prompt_and_subtitle_are_shown(screen):
    copy = " ".join(screen.text_lines())
    assert "Quit GazeKey?" in copy
    assert "Look at an answer" in copy
    assert "hold" in copy.lower()


def test_the_screen_renders(screen):
    frame = render(screen)
    assert len(np.unique(frame.reshape(-1, 4), axis=0)) > 2


def test_the_live_gaze_dot_is_drawn(screen):
    before = render(screen)
    screen.pipeline.look_at((400.0, 200.0), 0.1, start=0.0, is_fixating=False)
    screen._tick()
    assert not np.array_equal(before, render(screen))


# ------------------------------------------------- answerable at the real accuracy
def region_screen(qapp):
    from interaction.layouts import keyboard_region

    region = keyboard_region(SCREEN)
    widget = ChoiceScreen("Quit GazeKey?", OPTIONS, SCREEN, ScriptedPipeline(),
                          DwellSettings(), subtitle="Look and hold.",
                          region=region)
    widget.resize(*SIZE)
    return widget, region


def test_the_options_move_inside_the_calibrated_region(qapp):
    """Answering at the centre of the screen would need accuracy nobody measured."""
    widget, region = region_screen(qapp)
    try:
        for key in widget.board.keys:
            assert region.holds_rect(key.rect), f"{key.id} outside the region"
            _, _, w, h = key.rect
            assert w > 400 and h > 150, "and still an enormous target"
    finally:
        widget.close()


def test_the_options_are_still_answerable_by_gaze_there(qapp):
    widget, region = region_screen(qapp)
    try:
        answers = []
        widget.chosen.connect(answers.append)
        option = widget.board.find("choice.yes")
        widget.pipeline.look_at(option.centre, 1.2, start=0.0)
        widget._tick()
        assert answers == ["yes"]
    finally:
        widget.close()


def test_the_copy_moves_with_the_options(qapp):
    """Text is read, not aimed at — but it must not land on top of the keys."""
    widget, region = region_screen(qapp)
    try:
        prompt, subtitle, hint = widget.text_tops()
        board_top = widget.board.rect[1] / SCREEN[1]
        board_bottom = (widget.board.rect[1] + widget.board.rect[3]) / SCREEN[1]
        assert prompt < subtitle < board_top, "prompt and subtitle sit above"
        assert hint > board_bottom, "the hint sits below"
        assert 0.0 <= prompt and hint <= 1.0, "and all of it stays on screen"
        render(widget)
    finally:
        widget.close()
