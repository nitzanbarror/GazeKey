"""Target-practice tests — scripted gaze streams, no camera, no Qt."""
import math
import random

import pytest

from interaction.practice import (
    MIN_HIT_RADIUS_PX,
    MIN_SEPARATION_RATIO,
    PracticePhase,
    PracticeSession,
    PracticeStats,
    hit_radius_for,
)

SCREEN = (1366, 768)
DT = 1.0 / 30.0


def new_session(**kwargs) -> PracticeSession:
    kwargs.setdefault("hit_radius_px", 95.0)
    kwargs.setdefault("rng", random.Random(7))
    return PracticeSession(SCREEN, **kwargs)


def feed(session, position, seconds, start=0.0, dt=DT,
         is_fixating=True, stream_valid=True):
    """Hold the gaze at ``position``; returns (events, end time)."""
    events = []
    now = start
    for _ in range(max(1, int(round(seconds / dt)))):
        now += dt
        event = session.update(position[0], position[1], is_fixating,
                               stream_valid, now)
        if event is not None:
            events.append(event)
    return events, now


def hit_current(session, start, offset=(0.0, 0.0), extra=0.25):
    """Look straight at the current target long enough to pop it."""
    target = session.current_target
    aim = (target[0] + offset[0], target[1] + offset[1])
    return feed(session, aim, session.hold_s + extra, start=start)


# ------------------------------------------------------------------ hit radius
def test_hit_radius_is_the_error_with_a_floor():
    assert hit_radius_for(95.1) == pytest.approx(95.1)
    assert hit_radius_for(120.0) == pytest.approx(120.0)
    assert hit_radius_for(40.0) == MIN_HIT_RADIUS_PX
    assert hit_radius_for(MIN_HIT_RADIUS_PX) == MIN_HIT_RADIUS_PX


@pytest.mark.parametrize("unknown", [None, float("nan"), float("inf")])
def test_unknown_accuracy_falls_back_to_the_floor(unknown):
    assert hit_radius_for(unknown) == MIN_HIT_RADIUS_PX


# --------------------------------------------------------------------- hitting
def test_holding_on_target_pops_it():
    session = new_session()
    first = session.current_target
    events, _ = hit_current(session, start=0.0)
    assert events == ["hit"]
    assert session.index == 1
    assert session.current_target != first, "a new target should be up"


def test_a_hit_needs_the_full_hold_time():
    session = new_session(hold_s=0.8)
    events, now = feed(session, session.current_target, 0.7, start=0.0)
    assert events == [], "popped early"
    assert 0.75 < session.progress < 1.0

    events, _ = feed(session, session.current_target, 0.2, start=now)
    assert events == ["hit"]


def test_a_moving_gaze_never_pops_a_target():
    session = new_session()
    events, _ = feed(session, session.current_target, 3.0, is_fixating=False)
    assert events == []
    assert session.progress == 0.0


def test_gaze_outside_the_radius_does_not_count():
    session = new_session(hit_radius_px=95.0)
    target = session.current_target
    just_outside = (target[0] + 96.0, target[1])
    events, _ = feed(session, just_outside, 2.0)
    assert events == []
    assert not session.inside


def test_gaze_just_inside_the_radius_counts():
    session = new_session(hit_radius_px=95.0)
    target = session.current_target
    events, _ = hit_current(session, start=0.0, offset=(94.0, 0.0))
    assert events == ["hit"]


def test_leaving_briefly_decays_instead_of_resetting():
    session = new_session(hold_s=0.8, grace_ms=200)
    target = session.current_target
    feed(session, target, 0.6, start=0.0)
    before = session.progress

    feed(session, (target[0] + 500.0, target[1]), 0.1, start=0.6)
    assert 0.0 < session.progress < before, "hold should decay, not vanish"

    events, _ = feed(session, target, 0.6, start=0.7)
    assert events == ["hit"], "returning quickly should finish the hold"


def test_staying_away_past_the_grace_clears_the_hold():
    session = new_session(hold_s=0.8, grace_ms=200)
    target = session.current_target
    feed(session, target, 0.6, start=0.0)
    feed(session, (target[0] + 500.0, target[1]), 0.4, start=0.6)
    assert session.progress == 0.0


def test_an_invalid_stream_freezes_the_hold():
    """Same rule as the keyboard: a blink costs time, not progress."""
    session = new_session(hold_s=0.8)
    target = session.current_target
    feed(session, target, 0.6, start=0.0)
    frozen = session.progress

    _, now = feed(session, target, 1.0, start=0.6, stream_valid=False)
    assert session.progress == pytest.approx(frozen)

    events, _ = feed(session, target, 0.3, start=now)
    assert events == ["hit"]


def test_a_nan_gaze_is_treated_as_no_data():
    session = new_session()
    target = session.current_target
    feed(session, target, 0.5, start=0.0)
    held = session.progress
    session.update(float("nan"), float("nan"), True, True, 0.6)
    assert session.progress == pytest.approx(held)


# --------------------------------------------------------------------- targets
def test_targets_stay_inside_the_screen_with_a_margin():
    session = new_session(targets=30)
    seen = []
    now = 0.0
    while session.phase is PracticePhase.RUNNING:
        seen.append(session.current_target)
        _, now = hit_current(session, start=now)
    assert len(seen) == 30
    for x, y in seen:
        assert 0.1 * SCREEN[0] < x < 0.9 * SCREEN[0]
        assert 0.1 * SCREEN[1] < y < 0.9 * SCREEN[1]


def test_consecutive_targets_are_far_enough_apart_to_need_a_saccade():
    session = new_session(targets=25)
    minimum = MIN_SEPARATION_RATIO * math.hypot(*SCREEN)
    previous = session.current_target
    now = 0.0
    while session.phase is PracticePhase.RUNNING:
        _, now = hit_current(session, start=now)
        current = session.current_target
        if current is None:
            break
        assert math.hypot(current[0] - previous[0],
                          current[1] - previous[1]) >= minimum
        previous = current


def test_target_positions_are_random_between_sessions():
    firsts = {PracticeSession(SCREEN, 95.0, rng=random.Random(seed)).current_target
              for seed in range(6)}
    assert len(firsts) > 1


# ----------------------------------------------------------------- the drill
def test_ten_targets_then_done():
    session = new_session(targets=10)
    now = 0.0
    for expected in range(1, 11):
        _, now = hit_current(session, start=now)
        assert session.index == expected
    assert session.phase is PracticePhase.DONE
    assert session.current_target is None
    assert session.stats.total == 10
    assert session.stats.hits == 10


def test_a_target_times_out_and_counts_as_a_miss():
    session = new_session(targets=2, timeout_s=1.0)
    target = session.current_target
    events, _ = feed(session, (target[0] + 600.0, target[1]), 1.5, start=0.0)
    assert events == ["miss"]
    assert session.index == 1
    assert session.stats.attempts[0].hit is False


def test_updates_after_the_drill_are_ignored():
    session = new_session(targets=1)
    hit_current(session, start=0.0)
    assert session.phase is PracticePhase.DONE
    assert session.update(100.0, 100.0, True, True, 99.0) is None


def test_skip_ends_the_drill_immediately():
    session = new_session(targets=10)
    feed(session, session.current_target, 0.3, start=0.0)
    session.skip()
    assert session.phase is PracticePhase.DONE
    assert session.current_target is None


# ------------------------------------------------------------------ statistics
def test_stats_report_hits_error_and_time():
    session = new_session(targets=5, hold_s=0.8)
    now = 0.0
    for _ in range(5):
        _, now = hit_current(session, start=now, offset=(30.0, 0.0), extra=0.1)
    stats = session.stats
    assert stats.total == 5 and stats.hits == 5
    assert stats.hit_rate == 1.0
    assert stats.mean_error_px == pytest.approx(30.0, abs=1.0)
    assert 0.8 <= stats.mean_time_s <= 1.2


def test_stats_average_only_over_the_hits():
    session = new_session(targets=2, timeout_s=1.5, hold_s=0.8)
    target = session.current_target
    _, now = feed(session, (target[0] + 600.0, target[1]), 2.0, start=0.0)  # miss
    hit_current(session, start=now, offset=(20.0, 0.0))                     # hit

    stats = session.stats
    assert stats.hits == 1 and stats.total == 2
    assert stats.hit_rate == pytest.approx(0.5)
    assert stats.mean_error_px == pytest.approx(20.0, abs=1.0)
    assert stats.mean_time_s < 1.5, "the timed-out target must not skew the mean"


def test_the_timeout_clock_restarts_for_each_target():
    session = new_session(targets=3, timeout_s=1.5, hold_s=0.8)
    now = 0.0
    for _ in range(3):
        _, now = hit_current(session, start=now)
    assert session.stats.hits == 3, "later targets inherited an expired clock"
    assert all(attempt.time_s < 1.5 for attempt in session.stats.attempts)


def test_summary_lines_are_readable():
    session = new_session(targets=3)
    now = 0.0
    for _ in range(3):
        _, now = hit_current(session, start=now)
    lines = session.stats.summary_lines()
    assert any("Hits: 3 of 3" in line for line in lines)
    assert any("Average aim error" in line and "px" in line for line in lines)
    assert any("while holding" in line and "px" in line for line in lines)
    assert any("Average time" in line and "s" in line for line in lines)


# ------------------------------------------------------- the comparable metric
def test_aim_error_counts_every_sample_not_just_the_ones_inside():
    """The number to compare two calibrations with — the approach included."""
    session = new_session(targets=1, hit_radius_px=100.0)
    target = session.current_target
    # a second of gaze 200 px off, then close in and hold to a hit
    _, now = feed(session, (target[0] + 200.0, target[1]), 1.0, start=0.0)
    feed(session, (target[0] + 20.0, target[1]), 1.0, start=now)

    attempt = session.stats.attempts[0]
    assert attempt.hit
    assert attempt.mean_error_px == pytest.approx(20.0, abs=1.0), \
        "the holding number only sees gaze already inside the radius"
    assert 100.0 < attempt.aim_error_px < 200.0, \
        "the aim number has to include the part that was badly aimed"
    assert attempt.aim_samples > attempt.samples


def test_aim_error_is_weighted_by_how_long_each_target_took():
    session = new_session(targets=2, hit_radius_px=100.0)
    now = 0.0
    for _ in range(2):
        _, now = hit_current(session, start=now)
    stats = session.stats
    assert stats.mean_aim_error_px == stats.mean_aim_error_px, "not NaN"
    per_attempt = [a.aim_error_px for a in stats.attempts]
    assert min(per_attempt) <= stats.mean_aim_error_px <= max(per_attempt)


def test_a_missed_target_still_reports_its_aim_error():
    session = new_session(targets=1, timeout_s=1.0)
    target = session.current_target
    feed(session, (target[0] + 600.0, target[1]), 1.5, start=0.0)
    attempt = session.stats.attempts[0]
    assert not attempt.hit
    assert attempt.mean_error_px != attempt.mean_error_px, "no holding samples"
    assert attempt.aim_error_px == pytest.approx(600.0, abs=1.0)
    assert session.stats.mean_aim_error_px == pytest.approx(600.0, abs=1.0)


def test_summary_survives_an_empty_drill():
    stats = PracticeStats()
    assert stats.total == 0 and stats.hits == 0
    assert stats.hit_rate == 0.0
    assert stats.mean_error_px != stats.mean_error_px      # NaN
    assert stats.summary_lines() == ["No targets attempted."]


def test_summary_handles_a_drill_with_no_hits():
    session = new_session(targets=1, timeout_s=0.5)
    target = session.current_target
    feed(session, (target[0] + 600.0, target[1]), 1.0, start=0.0)
    lines = session.stats.summary_lines()
    assert any("Hits: 0 of 1" in line for line in lines)
    assert any("-" in line for line in lines), "no data should read as a dash"
