"""Calibration-region tests — the geometry, and everything that must sit inside it.

The point of a region is that a webcam's fixed angular error buys more accuracy
where it matters when it is not spread over parts of the screen nobody looks
at. These pin the two halves of that: the targets go in the region, and so does
every gaze-selectable thing.
"""
import json
import math

import pytest

from gaze.calibration import grid_points, validation_points
from gaze.region import (
    Region,
    attach_region,
    full_screen_region,
    read_region,
)
from interaction.layouts import (
    DEFAULT_HEIGHT_RATIO,
    MAX_HEIGHT_RATIO,
    REGION_MARGIN_RATIO,
    build_choice_board,
    build_keyboard,
    keyboard_region,
)

SCREEN = (1366, 768)


@pytest.fixture
def region():
    return keyboard_region(SCREEN)


# --------------------------------------------------------------- the geometry
def test_the_default_region_is_the_board_plus_a_margin_above_it(region):
    board_h = SCREEN[1] * DEFAULT_HEIGHT_RATIO
    margin = SCREEN[1] * REGION_MARGIN_RATIO
    assert region.y == pytest.approx(SCREEN[1] - board_h - margin)
    assert region.h == pytest.approx(board_h + margin)
    assert (region.x, region.w) == (0.0, float(SCREEN[0])), "full width"
    assert region.y + region.h == pytest.approx(SCREEN[1]), "docked at the bottom"


def test_the_margin_keeps_the_top_row_off_the_boundary():
    """The suggestion bar must not sit on the edge of the fitted area."""
    region = keyboard_region(SCREEN)
    board = build_keyboard(SCREEN, 95.0)
    top_row = min(key.rect[1] for key in board.keys if key.selectable)
    assert region.y < top_row, "the region has to reach above the topmost key"
    assert top_row - region.y == pytest.approx(SCREEN[1] * REGION_MARGIN_RATIO,
                                               abs=1.0)


def hull_top(region) -> int:
    """y of the topmost row of calibration dots."""
    return min(y for _, y in region.grid())


def test_the_default_margin_puts_the_top_key_row_inside_the_dot_hull():
    """The margin has to clear the *hull*, not just the region edge.

    Dots sit at 10/90% *of the region*, so a tenth of it lies outside their
    convex hull on every side. 5% left the top key row outside that hull and
    aiming there was extrapolated; 7.5% covers it. This is what
    REGION_MARGIN_RATIO is for, so it is pinned rather than left to drift.
    """
    board = build_keyboard(SCREEN, 95.0)
    top_key = min(key.rect[1] for key in board.keys if key.selectable)

    assert REGION_MARGIN_RATIO == 0.075
    assert hull_top(keyboard_region(SCREEN)) <= top_key, \
        "the topmost keys must be inside the dot hull, not extrapolated"
    assert hull_top(keyboard_region(SCREEN, DEFAULT_HEIGHT_RATIO, 0.05)) > top_key, \
        "5% was not enough — the regression this margin exists to prevent"


def test_the_bottom_overhang_is_unavoidable_but_smaller_than_full_screen():
    """No margin can cover a board that is flush with the screen edge."""
    region = keyboard_region(SCREEN)
    overhang = SCREEN[1] - max(y for _, y in region.grid())
    whole = SCREEN[1] - max(y for _, y in full_screen_region(SCREEN).grid())
    assert 0 < overhang < whole, "still better than calibrating the whole screen"


def test_the_region_covers_every_selectable_key_of_the_default_board():
    region = keyboard_region(SCREEN)
    board = build_keyboard(SCREEN, 95.0)
    outside = [key.id for key in board.keys
               if key.selectable and not region.holds_rect(key.rect)]
    assert outside == [], f"keys outside the calibrated area: {outside}"


def test_the_bound_for_error_sized_layouts_covers_the_paged_board():
    """The paged board's height depends on the error, so the region bounds it."""
    region = keyboard_region(SCREEN, MAX_HEIGHT_RATIO)
    for error in (60.0, 81.0, 85.0):
        board = build_keyboard(SCREEN, error, layout="auto")
        outside = [key.id for key in board.keys
                   if key.selectable and not region.holds_rect(key.rect)]
        assert outside == [], f"at {error} px: {outside}"


def test_a_region_is_smaller_than_the_screen_it_came_from(region):
    assert region.share_of(SCREEN) < 1.0
    assert 0.6 < region.share_of(SCREEN) < 0.8, "the point is a real saving"
    assert "keyboard" in region.describe(SCREEN)
    assert "%" in region.describe(SCREEN)


def test_containment_and_centre(region):
    assert region.contains(*region.centre)
    assert not region.contains(SCREEN[0] / 2, 10.0), "the top of the screen"
    assert region.centre[1] > SCREEN[1] / 2, "the board is the lower half"


# ----------------------------------------------------------------- the targets
def test_the_nine_dots_span_the_region_not_the_screen(region):
    points = region.grid()
    assert len(points) == 9
    for x, y in points:
        assert region.contains(x, y)
    xs = sorted({x for x, _ in points})
    ys = sorted({y for _, y in points})
    assert len(xs) == 3 and len(ys) == 3, "still a 3x3 grid"
    assert ys[0] == pytest.approx(region.y + 0.1 * region.h, abs=1.5)
    assert ys[2] == pytest.approx(region.y + 0.9 * region.h, abs=1.5)
    assert min(y for _, y in points) > SCREEN[1] * 0.1 + 1, \
        "a screen-spanning grid would have put a row near the top"


def test_the_validation_targets_are_inside_the_region_too(region):
    targets = region.validation()
    assert len(targets) == 3
    assert all(region.contains(x, y) for x, y in targets)
    assert not set(targets) & set(region.grid()), "must be fresh positions"


def test_a_full_screen_region_reproduces_the_old_layout_exactly():
    """--cal-region full has to be a true A/B baseline, not an approximation."""
    region = full_screen_region(SCREEN)
    assert region.grid() == grid_points(*SCREEN)
    assert region.validation() == validation_points(*SCREEN)
    assert region.share_of(SCREEN) == pytest.approx(1.0)


def test_points_scale_with_the_region_size():
    small = Region((100.0, 200.0, 400.0, 300.0), name="test")
    assert small.point(0.5, 0.5) == (300.0, 350.0)
    assert small.centre == (300.0, 350.0)
    for x, y in small.grid():
        assert 100 <= x <= 500 and 200 <= y <= 500


# ----------------------------------------------------------------- persistence
def test_a_region_rides_along_with_the_saved_calibration(tmp_path, region):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"wx": [1, 2], "validation_error_px": 61.0}))

    assert attach_region(str(path), region)
    stored = json.loads(path.read_text())
    assert stored["wx"] == [1, 2], "the model's own keys are untouched"
    assert stored["validation_error_px"] == 61.0

    back = read_region(str(path))
    assert back == region


def test_a_calibration_without_a_region_reports_none(tmp_path):
    """Files written before regions existed must not be handed a guess."""
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"wx": [1, 2]}))
    assert read_region(str(path)) is None


def test_unreadable_or_missing_files_are_survivable(tmp_path, region):
    missing = str(tmp_path / "nope.json")
    assert read_region(missing) is None
    assert attach_region(missing, region) is False

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert read_region(str(broken)) is None
    assert attach_region(str(broken), region) is False


def test_a_nonsense_region_is_rejected_rather_than_used():
    assert Region.from_dict({"rect": [0, 0, 0, 0]}) is None
    assert Region.from_dict({"rect": [0, 0, 10]}) is None
    assert Region.from_dict({"nope": 1}) is None
    assert Region.from_dict({"rect": [0, 0, 10, 10]}) == Region((0, 0, 10, 10))


def test_the_session_records_the_region_it_used(tmp_path):
    from gaze.calibration_session import CalibrationSession

    region = keyboard_region(SCREEN)
    session = CalibrationSession(SCREEN, region=region)
    assert session.cal_targets and set(session.cal_targets) == set(region.grid())

    path = str(tmp_path / "calibration.json")
    session.save(path)
    assert read_region(path) == region
    assert region.describe(SCREEN) in session.diagnostics().region_line()


def test_a_session_without_a_region_still_calibrates_the_whole_screen():
    from gaze.calibration_session import CalibrationSession

    session = CalibrationSession(SCREEN)
    assert session.region == full_screen_region(SCREEN)
    assert set(session.cal_targets) == set(grid_points(*SCREEN))


# ------------------------------------------- gaze-selectable UI stays in-region
def test_the_quit_confirmation_sits_inside_the_region(region):
    board = build_choice_board(SCREEN, [("yes", "YES, QUIT"),
                                        ("no", "NO, KEEP TYPING")],
                               region=region)
    for key in board.keys:
        assert region.holds_rect(key.rect), f"{key.id} is outside the region"
        _, _, w, h = key.rect
        assert w > 400 and h > 150, "and still a very large target"


def test_the_quit_confirmation_defaults_to_the_whole_screen():
    """No region given: unchanged behaviour, centred on the display."""
    board = build_choice_board(SCREEN, [("yes", "Y"), ("no", "N")])
    screen = full_screen_region(SCREEN)
    assert all(screen.holds_rect(key.rect) for key in board.keys)
    centre_y = board.rect[1] + board.rect[3] / 2
    assert centre_y == pytest.approx(SCREEN[1] / 2, abs=1.0)


def test_the_touch_up_dot_sits_at_the_centre_of_the_region(region):
    from gaze.drift import TouchUpSession

    session = TouchUpSession(SCREEN, region=region)
    assert session.target == region.centre
    assert region.contains(*session.target)
    assert session.max_offset_px == pytest.approx(0.25 * region.diagonal)


def test_practice_targets_stay_inside_the_region(region):
    import random

    from interaction.practice import PracticeSession

    session = PracticeSession(SCREEN, 95.0, targets=40, region=region,
                              rng=random.Random(3))
    seen = [session.current_target]
    for _ in range(39):
        seen.append(session._pick_target(seen[-1]))
    assert all(region.contains(x, y) for x, y in seen)
    assert max(y for _, y in seen) - min(y for _, y in seen) > region.h * 0.4, \
        "and still spread across it"


def test_practice_without_a_region_still_uses_the_whole_screen():
    from interaction.practice import PracticeSession

    session = PracticeSession(SCREEN, 95.0)
    assert session.region == full_screen_region(SCREEN)
    assert session.min_separation == pytest.approx(0.25 * math.hypot(*SCREEN))
