"""Practice screen tests — rendered offscreen, driven by scripted samples."""
import random

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent, QPixmap
from PyQt5.QtWidgets import QApplication

from gaze.features import FrameFeatures
from interaction.practice import PracticePhase, PracticeSession
from ui.practice_screen import PracticeScreen
from vision.pipeline import STATE_OK, GazeSample

SCREEN = (1366, 768)
SIZE = (683, 384)
DT = 1.0 / 30.0


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class ScriptedPipeline:
    fps = 30.0

    def __init__(self):
        self.pending = []

    def drain(self):
        out, self.pending = self.pending, []
        return out

    def look_at(self, position, seconds, start, is_fixating=True,
                stream_valid=True):
        now = start
        for _ in range(max(1, int(round(seconds / DT)))):
            now += DT
            self.pending.append(GazeSample(
                FrameFeatures(True, 0.5, 0.5, 0.0, 0.0, 0.0, now),
                STATE_OK, now, x=position[0], y=position[1],
                is_fixating=is_fixating, stream_valid=stream_valid,
            ))
        return now


@pytest.fixture
def screen(qapp):
    session = PracticeSession(SCREEN, hit_radius_px=95.0, targets=3,
                              rng=random.Random(3))
    widget = PracticeScreen(session, ScriptedPipeline())
    widget.resize(*SIZE)
    yield widget
    widget.close()


def render(widget) -> np.ndarray:
    pixmap = QPixmap(*SIZE)
    widget.render(pixmap)
    image = pixmap.toImage().convertToFormat(4)
    buffer = image.bits().asstring(image.byteCount())
    return np.frombuffer(buffer, dtype=np.uint8).reshape(image.height(), -1, 4)


def ink(frame) -> int:
    return len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))


def hue_pixels(frame, kind, margin=60) -> int:
    blue = frame[..., 0].astype(np.int16)
    red = frame[..., 2].astype(np.int16)
    difference = (red - blue) if kind == "warm" else (blue - red)
    return int((difference > margin).sum())


def key(code):
    return QKeyEvent(QKeyEvent.KeyPress, code, Qt.NoModifier)


# ------------------------------------------------------------------ rendering
def test_target_and_hit_ring_are_drawn(screen):
    frame = render(screen)
    assert ink(frame) > 2

    target = screen.session.current_target
    cx = int(target[0] * SIZE[0] / SCREEN[0])
    cy = int(target[1] * SIZE[1] / SCREEN[1])
    radius = int(screen.session.hit_radius_px * SIZE[0] / SCREEN[0]) + 6
    patch = frame[max(0, cy - radius):cy + radius, max(0, cx - radius):cx + radius]
    assert ink(patch) > 2, "no target drawn where the session says it is"


def test_the_hold_ring_fills_as_the_gaze_is_held(screen):
    target = screen.session.current_target
    empty = render(screen)
    assert hue_pixels(empty, "cool") == 0, "progress ring showing before any hold"

    screen.pipeline.look_at(target, 0.4, start=0.0)
    screen._tick()
    holding = render(screen)
    assert hue_pixels(holding, "cool") > 0, "the ring never started filling"

    partial = hue_pixels(holding, "cool")
    screen.pipeline.look_at(target, 0.2, start=0.4)
    screen._tick()
    assert hue_pixels(render(screen), "cool") > partial, "the ring stopped filling"


def test_the_live_gaze_dot_is_visible(screen):
    target = screen.session.current_target
    away = (target[0] + 400.0, target[1])
    before = render(screen)
    screen.pipeline.look_at(away, 0.1, start=0.0, is_fixating=False)
    screen._tick()
    assert not np.array_equal(before, render(screen)), "no live gaze dot drawn"


def test_a_hit_pops_and_moves_to_the_next_target(screen):
    first = screen.session.current_target
    screen.pipeline.look_at(first, 1.1, start=0.0)
    screen._tick()
    assert screen.session.index == 1
    assert screen.session.current_target != first
    assert screen._pop_at is not None, "no pop animation was triggered"
    render(screen)                              # the pop must paint cleanly


def test_the_click_sound_is_optional(qapp):
    beeps = []
    session = PracticeSession(SCREEN, 95.0, targets=2, rng=random.Random(5))
    widget = PracticeScreen(session, ScriptedPipeline(),
                            sound=lambda: beeps.append(True))
    try:
        widget.resize(*SIZE)
        widget.pipeline.look_at(session.current_target, 1.1, start=0.0)
        widget._tick()
        assert len(beeps) == 1, "the hit should have clicked"
    finally:
        widget.close()


def test_no_sound_hook_is_fine(screen):
    assert screen.sound is None
    screen.pipeline.look_at(screen.session.current_target, 1.1, start=0.0)
    screen._tick()                              # must not raise
    assert screen.session.index == 1


# ----------------------------------------------------------------------- copy
def test_running_copy_states_the_rules(screen):
    copy = " ".join(screen.text_lines())
    assert "1 / 3" in copy
    assert "hit radius 95 px" in copy
    assert "hold 0.8 s" in copy
    assert "Esc to skip" in copy


def test_summary_reports_hits_error_and_time(screen):
    now = 0.0
    for _ in range(3):
        target = screen.session.current_target
        now = screen.pipeline.look_at(target, 1.1, start=now)
        screen._tick()
        now += 0.1
    assert screen.session.phase is PracticePhase.DONE

    copy = " ".join(screen.text_lines())
    assert "Practice complete" in copy
    assert "Hits: 3 of 3" in copy
    assert "Average error while holding" in copy
    assert "Average time to hit" in copy
    assert ink(render(screen)) > 2, "the summary should draw its hit-rate bar"


def test_summary_is_emitted_once(screen):
    seen = []
    screen.finished.connect(seen.append)
    now = 0.0
    for _ in range(3):
        now = screen.pipeline.look_at(screen.session.current_target, 1.1, start=now)
        screen._tick()
        now += 0.1
    screen._tick()
    screen._tick()
    assert len(seen) == 1, "finished fired more than once"
    assert seen[0].hits == 3


# ------------------------------------------------------------------ skipping
def test_escape_skips_the_drill(screen):
    seen = []
    screen.finished.connect(seen.append)
    screen.keyPressEvent(key(Qt.Key_Escape))
    assert seen == [None]
    assert screen.session.phase is PracticePhase.DONE


def test_escape_after_the_summary_does_not_double_emit(screen):
    seen = []
    screen.finished.connect(seen.append)
    now = 0.0
    for _ in range(3):
        now = screen.pipeline.look_at(screen.session.current_target, 1.1, start=now)
        screen._tick()
        now += 0.1
    screen.keyPressEvent(key(Qt.Key_Escape))
    assert len(seen) == 1


def test_space_on_the_summary_continues(screen):
    seen = []
    screen.session.skip()
    screen.finished.connect(seen.append)
    screen.keyPressEvent(key(Qt.Key_Space))
    assert len(seen) == 1 and seen[0] is not None


def test_a_lost_stream_does_not_advance_the_hold(screen):
    target = screen.session.current_target
    screen.pipeline.look_at(target, 2.0, start=0.0, stream_valid=False)
    screen._tick()
    assert screen.session.index == 0
    assert screen.session.progress == 0.0
    render(screen)
