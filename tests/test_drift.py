"""Drift-monitor tests (spec 5.5) — no camera, no Qt, no clock.

Everything is driven from explicit timestamps, so the 3 s correction gap and
the 60 s window are tested at their real boundaries rather than by sleeping.
"""
import math

import pytest

from gaze.drift import (
    CLEAR_RATIO,
    CORRECTIONS_FOR_DRIFT,
    MIN_ACTIVATIONS,
    MIN_OFFSET_PX,
    DriftMonitor,
    offset_threshold_px,
)
from interaction.layouts import build_keyboard

SCREEN = (1366, 768)
ERROR_PX = 95.0


@pytest.fixture
def board():
    return build_keyboard(SCREEN, ERROR_PX)


@pytest.fixture
def monitor():
    return DriftMonitor(validation_error_px=100.0)      # threshold 100 px


def look_at(monitor, position, now, frames=10, dt=1 / 30):
    """Feed a short burst of gaze samples ending at ``now``."""
    start = now - frames * dt
    for i in range(frames):
        monitor.record_sample(position[0], position[1], start + i * dt)
    return now


def type_key(monitor, key, gaze, now):
    """Look somewhere, then fire a key — the weak ground truth of spec 5.5."""
    look_at(monitor, gaze, now)
    monitor.record_activation(key, now)


# ------------------------------------------------------------------ threshold
def test_the_threshold_follows_the_measured_accuracy():
    assert offset_threshold_px(120.0) == 120.0
    assert offset_threshold_px(20.0) == MIN_OFFSET_PX, "never below the floor"
    for unknown in (float("nan"), float("inf"), 0.0, None):
        assert offset_threshold_px(unknown) == MIN_OFFSET_PX


def test_config_can_override_the_threshold():
    assert offset_threshold_px(120.0, configured=80.0) == 80.0


# --------------------------------------------------------------- the centroid
def test_the_centroid_only_uses_the_recent_window(monitor):
    monitor.record_sample(100.0, 100.0, 0.0)
    monitor.record_sample(500.0, 500.0, 10.0)
    assert monitor.centroid(10.0) == (500.0, 500.0), "the old sample expired"


def test_without_samples_there_is_no_centroid_and_no_evidence(monitor, board):
    key = board.find("key.A")
    monitor.record_activation(key, 1.0)
    assert monitor.centroid(1.0) is None
    assert monitor.activations == 0, "an activation with no gaze proves nothing"


# ----------------------------------------------------------------- the offset
def test_a_consistent_offset_is_measured(monitor, board):
    """Gaze lands 60 px left and 20 px above every key centre."""
    now = 0.0
    for char in "ASDFGHJK":
        key = board.find(f"key.{char}")
        now += 1.0
        type_key(monitor, key, (key.centre[0] - 60.0, key.centre[1] - 20.0), now)

    assert monitor.activations == 8
    assert monitor.offset[0] == pytest.approx(60.0, abs=1.0)
    assert monitor.offset[1] == pytest.approx(20.0, abs=1.0)
    assert monitor.offset_px == pytest.approx(math.hypot(60.0, 20.0), abs=1.5)


def test_the_mean_is_exponentially_weighted(monitor, board):
    """A change in aim is followed, not averaged away over the whole session."""
    now = 0.0
    for _ in range(10):
        key = board.find("key.A")
        now += 1.0
        type_key(monitor, key, key.centre, now)
    assert monitor.offset_px < 1.0

    for _ in range(10):                       # the head shifts; aim goes with it
        key = board.find("key.A")
        now += 1.0
        type_key(monitor, key, (key.centre[0] - 80.0, key.centre[1]), now)
    assert monitor.offset[0] > 60.0, "the estimate has to track the new offset"


def test_only_character_keys_are_ground_truth(monitor, board):
    """Space is 250 px wide — its centre says nothing about where you looked."""
    space = board.find("fn.space")
    type_key(monitor, space, (space.centre[0] - 100.0, space.centre[1]), 1.0)
    assert monitor.activations == 0
    assert monitor.offset_px == 0.0


# ------------------------------------------------------------------- scoring
def test_a_few_activations_are_not_enough_to_flag(monitor, board):
    now = 0.0
    for _ in range(MIN_ACTIVATIONS - 1):
        key = board.find("key.A")
        now += 1.0
        type_key(monitor, key, (key.centre[0] - 400.0, key.centre[1]), now)
    assert monitor.offset_px > 300.0
    assert monitor.score == 0.0, "too little evidence to act on"
    assert not monitor.drifting


def test_an_offset_the_size_of_the_threshold_flags_drift(monitor, board):
    now = 0.0
    for _ in range(MIN_ACTIVATIONS + 3):
        key = board.find("key.A")
        now += 1.0
        type_key(monitor, key, (key.centre[0] - 110.0, key.centre[1]), now)
    assert monitor.offset_px > monitor.threshold_px
    assert monitor.drifting


def test_accurate_typing_never_flags(monitor, board):
    now = 0.0
    for char in "QWERTYUIOPASDFG":
        key = board.find(f"key.{char}")
        now += 1.0
        type_key(monitor, key, key.centre, now)
    assert monitor.score < 1.0
    assert not monitor.drifting


# ---------------------------------------------------- type-then-backspace rule
def correction(monitor, board, now, gap=1.0, off=0.0):
    """A character followed by a backspace ``gap`` seconds later."""
    key = board.find("key.A")
    type_key(monitor, key, (key.centre[0] - off, key.centre[1]), now)
    monitor.record_activation(board.find("fn.backspace"), now + gap)


def test_three_quick_corrections_in_a_minute_flag_drift(monitor, board):
    for i in range(CORRECTIONS_FOR_DRIFT):
        correction(monitor, board, now=1.0 + i * 5.0)
    assert monitor.corrections == CORRECTIONS_FOR_DRIFT
    assert monitor.score >= 1.0
    assert monitor.drifting, "spec 5.5: three in 60 s raises the score"


def test_two_corrections_alone_are_not_enough(monitor, board):
    for i in range(CORRECTIONS_FOR_DRIFT - 1):
        correction(monitor, board, now=1.0 + i * 5.0)
    assert not monitor.drifting


def test_a_slow_backspace_is_not_a_correction(monitor, board):
    for i in range(CORRECTIONS_FOR_DRIFT + 2):
        correction(monitor, board, now=1.0 + i * 10.0, gap=4.0)   # > 3 s
    assert monitor.corrections == 0, "deleting later is editing, not mis-aiming"
    assert not monitor.drifting


def test_corrections_expire_after_a_minute(monitor, board):
    for i in range(CORRECTIONS_FOR_DRIFT):
        correction(monitor, board, now=1.0 + i * 2.0)
    assert monitor.drifting

    look_at(monitor, (300.0, 300.0), now=200.0)      # much later, typing fine
    assert monitor.corrections == 0
    assert not monitor.drifting


def test_a_backspace_without_a_preceding_character_counts_for_nothing(monitor,
                                                                      board):
    monitor.record_activation(board.find("fn.backspace"), 1.0)
    assert monitor.corrections == 0


def test_weak_evidence_from_both_signals_adds_up(monitor, board):
    """Half an offset plus two corrections is drift; neither alone would be."""
    now = 0.0
    for _ in range(MIN_ACTIVATIONS + 2):
        key = board.find("key.A")
        now += 1.0
        type_key(monitor, key, (key.centre[0] - 60.0, key.centre[1]), now)
    assert not monitor.drifting, "60 px of 100 px on its own is not enough"

    for i in range(2):                    # still mis-aimed by the same 60 px
        correction(monitor, board, now=now + 1.0 + i * 2.0, off=60.0)
    assert monitor.corrections == 2, "two alone would not be enough either"
    assert monitor.drifting


# ------------------------------------------------------------------ stickiness
def test_the_flag_does_not_flicker(monitor, board):
    """Once flagged it stays flagged through the 0.7–1.0 band, then clears."""
    def type_off(pixels, count, start):
        now = start
        for _ in range(count):
            key = board.find("key.A")
            now += 1.0
            type_key(monitor, key, (key.centre[0] - pixels, key.centre[1]), now)
        return now

    now = type_off(130.0, MIN_ACTIVATIONS + 3, 0.0)
    assert monitor.drifting

    now = type_off(60.0, 3, now)          # aim improves, but not all the way
    assert CLEAR_RATIO <= monitor.score < 1.0
    assert monitor.drifting, "the indicator must not blink between keystrokes"

    type_off(0.0, 10, now)                # properly back on target
    assert monitor.score < CLEAR_RATIO
    assert not monitor.drifting


def test_a_reset_forgets_everything(monitor, board):
    for i in range(CORRECTIONS_FOR_DRIFT):
        correction(monitor, board, now=1.0 + i * 2.0)
    assert monitor.drifting

    monitor.reset()
    assert not monitor.drifting
    assert monitor.corrections == 0 and monitor.activations == 0
    assert monitor.offset_px == 0.0


def test_the_state_object_reports_what_it_saw(monitor, board):
    now = 0.0
    for _ in range(MIN_ACTIVATIONS + 3):
        key = board.find("key.A")
        now += 1.0
        type_key(monitor, key, (key.centre[0] - 110.0, key.centre[1]), now)
    state = monitor.state()
    assert state.activations == MIN_ACTIVATIONS + 3
    assert state.drifting and state.offset_px > 100.0
    assert "px" in state.summary()
