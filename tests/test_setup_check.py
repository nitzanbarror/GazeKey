"""The 5-second setup check before the nine dots (spec 5.1c).

Driven from synthetic features, no camera and no window: the gate is only
worth having if it fires on the sitting that motivated it (hy span 0.024) and
stays quiet on the one that worked (0.051), so both are pinned here as real
numbers rather than as "small" and "large".
"""
import pytest

from gaze.features import FrameFeatures
from gaze.region import Region, full_screen_region
from gaze.setup_check import (
    MIN_HY_SPAN,
    MIN_SAMPLES,
    SetupCheckSession,
    SetupFailure,
    SetupPhase,
)

SCREEN = (1366, 768)
DT = 1.0 / 30.0

#: the two sittings this threshold was chosen between (both measured)
GOOD_SPAN = 0.051
BAD_SPAN = 0.024


def region() -> Region:
    """The default keyboard region on the development screen."""
    from interaction.layouts import keyboard_region

    return keyboard_region(SCREEN, 2.0 / 3.0)


def run(session, span, *, valid=True, seconds=12.0, start=0.0, hy_top=0.30):
    """Feed the session an eye whose hy moves ``span`` between the targets."""
    now = start
    for _ in range(int(seconds / DT)):
        now += DT
        target = session.current_target()
        if target is None:
            break
        hy = hy_top if target.index == 1 else hy_top + span
        session.update(FrameFeatures(valid, 0.5, hy, 0.0, 0.0, 0.0, now))
    return session.result


# ------------------------------------------------------------------- geometry
def test_the_two_targets_sit_at_the_calibration_rows():
    """Same 10%/90% heights as the 3x3 grid, so the spans are comparable."""
    area = region()
    session = SetupCheckSession(SCREEN, region=area)
    grid_ys = sorted({y for _, y in area.grid()})

    assert len(session.targets) == 2
    top, bottom = session.targets
    assert top[1] == pytest.approx(grid_ys[0], abs=1)
    assert bottom[1] == pytest.approx(grid_ys[-1], abs=1)
    assert top[0] == bottom[0] == pytest.approx(area.centre[0], abs=1)


def test_both_targets_are_inside_the_calibrated_region():
    area = region()
    session = SetupCheckSession(SCREEN, region=area)
    assert all(area.contains(*target) for target in session.targets)


def test_it_fits_in_about_five_seconds():
    session = SetupCheckSession(SCREEN, region=region())
    assert 4.0 <= session.budget_s <= 6.0, "the promise the copy makes"


def test_without_a_region_it_uses_the_whole_screen():
    session = SetupCheckSession(SCREEN)
    assert session.region == full_screen_region(SCREEN)


# -------------------------------------------------------------------- verdict
def test_the_good_sitting_passes():
    session = SetupCheckSession(SCREEN, region=region())
    result = run(session, GOOD_SPAN)

    assert result is not None and result.passed
    assert session.phase is SetupPhase.PASSED
    assert result.hy_span == pytest.approx(GOOD_SPAN, abs=1e-6)
    assert result.failure is None


def test_the_bad_sitting_fails_and_names_the_camera_height():
    session = SetupCheckSession(SCREEN, region=region())
    result = run(session, BAD_SPAN)

    assert result is not None and not result.passed
    assert session.phase is SetupPhase.FAILED
    assert result.failure is SetupFailure.LOW_SPAN
    headline, guidance = result.text
    assert "too low" in headline.lower()
    assert "eye height" in guidance
    assert "0.024" in result.measurement_line()


def test_the_default_threshold_separates_the_two_measured_sittings():
    assert BAD_SPAN < MIN_HY_SPAN < GOOD_SPAN
    # nearer the bad one: a check that fires on a usable sitting is worse than
    # no check, because the user learns to press through it
    assert MIN_HY_SPAN - BAD_SPAN < GOOD_SPAN - MIN_HY_SPAN


def test_the_direction_of_the_movement_does_not_matter():
    """hy grows downward here, but the sign is a convention, not evidence."""
    session = SetupCheckSession(SCREEN, region=region())
    result = run(session, -GOOD_SPAN, hy_top=0.6)
    assert result.passed and result.hy_span == pytest.approx(GOOD_SPAN, abs=1e-6)


def test_the_threshold_is_configurable():
    strict = SetupCheckSession(SCREEN, region=region(), min_hy_span=0.08)
    assert not run(strict, GOOD_SPAN).passed

    lenient = SetupCheckSession(SCREEN, region=region(), min_hy_span=0.01)
    assert run(lenient, BAD_SPAN).passed


def test_a_face_that_is_never_seen_fails_differently():
    session = SetupCheckSession(SCREEN, region=region())
    result = run(session, GOOD_SPAN, valid=False)

    assert not result.passed
    assert result.failure is SetupFailure.NO_FACE, \
        "no eyes is a lighting problem, not a camera-height one"
    assert "could not see" in result.text[0]
    assert result.samples == (0, 0)


def test_a_few_frames_are_not_enough_to_judge():
    session = SetupCheckSession(SCREEN, region=region(), collect_max_s=0.2)
    now = 0.0
    for i in range(int(2.0 / DT)):
        now += DT
        target = session.current_target()
        if target is None:
            break
        # one usable frame in every five: enough to finish, not enough to trust
        valid = (i % 5 == 0)
        hy = 0.30 if target.index == 1 else 0.30 + GOOD_SPAN
        session.update(FrameFeatures(valid, 0.5, hy, 0.0, 0.0, 0.0, now))

    assert session.result is not None
    assert min(session.result.samples) < MIN_SAMPLES
    assert session.result.failure is SetupFailure.NO_FACE


def test_the_median_ignores_a_stray_frame():
    """One glance away during a target must not decide the verdict."""
    session = SetupCheckSession(SCREEN, region=region())
    now = 0.0
    for i in range(int(12.0 / DT)):
        now += DT
        target = session.current_target()
        if target is None:
            break
        hy = 0.30 if target.index == 1 else 0.30 + GOOD_SPAN
        if i % 11 == 0:
            hy += 0.4                      # looked at something else entirely
        session.update(FrameFeatures(True, 0.5, hy, 0.0, 0.0, 0.0, now))

    assert session.result.passed
    assert session.result.hy_span == pytest.approx(GOOD_SPAN, abs=1e-6)


# --------------------------------------------------------------------- pacing
def test_the_settling_frames_are_thrown_away():
    """The saccade to the dot must not be measured (same rule as calibration)."""
    session = SetupCheckSession(SCREEN, region=region(), settle_ms=500.0)
    now = 0.0
    for _ in range(10):                    # 333 ms, still inside the settle
        now += DT
        session.update(FrameFeatures(True, 0.5, 0.3, 0.0, 0.0, 0.0, now))

    target = session.current_target()
    assert target.settling and target.samples == 0 and target.progress == 0.0


def test_a_target_ends_on_the_sample_count_not_the_clock():
    """A blink should lengthen a target, never corrupt it."""
    session = SetupCheckSession(SCREEN, region=region(), collect_samples=30,
                                collect_max_s=10.0)
    now, valid_frames = 0.0, 0
    while session.current_target() is not None and session.current_target().index == 1:
        now += DT
        valid = valid_frames % 2 == 0      # half the frames are blinks
        session.update(FrameFeatures(valid, 0.5, 0.3, 0.0, 0.0, 0.0, now))
        valid_frames += 1

    assert now > 2.0, "it should have waited out the blinks"
    assert now < 10.0, "and not run to the wall-clock cap"


def test_the_clock_still_caps_a_target_that_sees_nothing():
    session = SetupCheckSession(SCREEN, region=region(), collect_max_s=1.0,
                                settle_ms=0.0)
    now = 0.0
    for _ in range(int(1.5 / DT)):
        now += DT
        session.update(FrameFeatures(False, timestamp=now))
    assert session.current_target().index == 2, "target 1 should have given up"


# -------------------------------------------------------------- what happens next
def test_a_retry_measures_again_from_scratch():
    session = SetupCheckSession(SCREEN, region=region())
    assert not run(session, BAD_SPAN).passed

    session.restart()
    assert session.phase is SetupPhase.MEASURING
    assert session.result is None
    assert session.current_target().index == 1

    result = run(session, GOOD_SPAN, start=100.0)      # camera has been raised
    assert result.passed, "the retry must not remember the failed attempt"


def test_continuing_anyway_is_recorded_rather_than_hidden():
    session = SetupCheckSession(SCREEN, region=region())
    run(session, BAD_SPAN)
    result = session.override()

    assert result.overridden and not result.passed
    assert "continuing anyway" in result.console_line()


def test_a_finished_session_ignores_further_frames():
    session = SetupCheckSession(SCREEN, region=region())
    run(session, GOOD_SPAN)
    span = session.result.hy_span
    assert session.update(FrameFeatures(True, 0.5, 0.9, 0.0, 0.0, 0.0, 99.0)) is None
    assert session.result.hy_span == span
    assert session.is_finished


def test_the_console_line_reports_the_measurement_either_way():
    for span in (GOOD_SPAN, BAD_SPAN):
        session = SetupCheckSession(SCREEN, region=region())
        line = run(session, span).console_line()
        assert f"{span:.3f}" in line
        assert ("PASS" if span == GOOD_SPAN else "FAIL") in line
        line.encode("cp1255")              # a Windows console must be able to print it
