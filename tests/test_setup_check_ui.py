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


def keyboard_region():
    from interaction.layouts import keyboard_region as build

    return build(SCREEN, 2 / 3)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class ScriptedEye:
    """A pipeline whose frames answer whichever target is showing.

    ``backlog`` is the hazard this screen has to defend against: frames the
    camera produced before the window existed, handed over in one lump on the
    first drain.
    """

    fps = 30.0

    def __init__(self, session, span=GOOD_SPAN, valid=True, backlog=0):
        self.session = session
        self.span = span
        self.valid = valid
        self.now = 0.0
        self.backlog = backlog

    def _sample(self, hy):
        self.now += DT
        return GazeSample(
            FrameFeatures(self.valid, 0.5, hy, 0.0, 0.0, 0.0, self.now),
            STATE_OK, self.now, x=0.0, y=0.0, is_fixating=True,
            stream_valid=True)

    def drain(self):
        if self.backlog:
            # captured before the screen appeared: the user was looking at the
            # middle of the display, not at a dot that did not exist yet
            queued, self.backlog = self.backlog, 0
            return [self._sample(0.30 + self.span / 2) for _ in range(queued)]
        out = []
        for _ in range(30):                     # one second of frames per tick
            target = self.session.current_target()
            index = 1 if target is None else target.index
            out.append(self._sample(0.30 + (0.0 if index == 1 else self.span)))
        return out


def screen_for(qapp, span=GOOD_SPAN, valid=True, backlog=0):
    session = SetupCheckSession(SCREEN, region=keyboard_region())
    widget = SetupCheckScreen(session,
                              ScriptedEye(session, span, valid, backlog),
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
        assert widget.session.phase is SetupPhase.LEAD_IN, \
            "a retry starts from the instructions, not from a dot"

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
def test_the_dots_are_painted_on_the_calibration_grid_rows(qapp):
    """The check must aim at exactly the rows the nine dots use — in SCREEN
    pixels, on a full-size window, not in region-normalised space.

    Nothing caught the P0 under-read for a while because the geometry was only
    ever checked against the session's own numbers. This renders the widget at
    the real resolution and reads the dot back out of the painted pixels.
    """
    session = SetupCheckSession(SCREEN, region=keyboard_region())
    widget = SetupCheckScreen(session, None, log=lambda _: None)
    widget.resize(*SCREEN)
    try:
        grid_rows = sorted({y for _, y in keyboard_region().grid()})
        session.phase = SetupPhase.MEASURING          # skip the lead-in

        session._index = 0
        assert _dot_row(widget, SCREEN) == pytest.approx(grid_rows[0], abs=2), \
            "the top dot is not on the calibration grid's top row"

        session._index = 1
        assert _dot_row(widget, SCREEN) == pytest.approx(grid_rows[-1], abs=2), \
            "the bottom dot is not on the calibration grid's bottom row"

        separation = grid_rows[-1] - grid_rows[0]
        assert separation > SCREEN[1] * 0.5, \
            f"only {separation} px apart: too little to measure a span with"
    finally:
        widget.close()


def test_the_lead_in_shows_the_instructions_and_no_dot(qapp):
    """Where the reading happens — nothing is measured and nothing is aimed at."""
    widget = screen_for(qapp, GOOD_SPAN)
    try:
        assert widget.session.phase is SetupPhase.LEAD_IN
        copy = " ".join(line for line, *_ in widget.text_lines()).lower()
        assert "two dots are about to appear" in copy
        assert "seconds" in copy and "esc to cancel" in copy

        frame = render(widget)
        assert not _has_dot(frame), "a dot during the lead-in is a target the " \
                                    "user is being measured on before they read"
    finally:
        widget.close()


def test_once_measuring_the_copy_is_only_the_dot(qapp):
    widget = screen_for(qapp, GOOD_SPAN)
    try:
        while widget.session.phase is SetupPhase.LEAD_IN:
            widget._tick()
        copy = " ".join(line for line, *_ in widget.text_lines()).lower()
        assert "look at the dot" in copy and "1 / 2" in copy
        assert "about" not in copy, \
            "nothing new to read while the measurement is running"
    finally:
        widget.close()


def test_a_stale_backlog_is_discarded_before_the_first_target(qapp):
    """The P0's other half: 3 s of queued gaze must not reach the session."""
    widget = screen_for(qapp, GOOD_SPAN, backlog=90)
    try:
        widget._tick()                      # the drain that hands over the lump
        assert widget.session.phase is SetupPhase.LEAD_IN, \
            "a backlog spent the lead-in and possibly a whole target"

        results = []
        widget.finished.connect(results.append)
        run(widget, ticks=12)
        assert len(results) == 1 and results[0].passed
        assert results[0].hy_span == pytest.approx(GOOD_SPAN, abs=0.004)
    finally:
        widget.close()


def test_a_retry_also_discards_what_piled_up(qapp):
    widget = screen_for(qapp, BAD_SPAN)
    try:
        run(widget)
        assert widget.session.phase is SetupPhase.FAILED
        widget.pipeline.backlog = 90        # gaze from reading the failure screen
        press(widget, Qt.Key_Space)
        widget._tick()
        assert widget.session.phase is SetupPhase.LEAD_IN
    finally:
        widget.close()


def _dot_row(widget, size) -> float:
    """The painted row of the target dot, at full screen resolution."""
    pixmap = QPixmap(*size)
    pixmap.fill(Qt.black)
    widget.render(pixmap)
    image = pixmap.toImage().convertToFormat(4)
    frame = np.frombuffer(image.bits().asstring(image.byteCount()),
                          dtype=np.uint8).reshape(image.height(), -1, 4)
    lit = np.argwhere(frame[..., :3].sum(axis=2) > 400)
    assert len(lit), "no dot was drawn at all"
    return float(np.median(lit[:, 0]))


def _has_dot(frame) -> bool:
    """Is anything bright enough to be the target dot or its ring on screen?"""
    return bool((frame[..., :3].sum(axis=2) > 400).any())
