"""Keyboard layout tests — NFR-2 sizing and geometry, no camera, no Qt."""
import pytest

from interaction.layouts import (
    EXTENDED_DWELL_ACTIONS,
    MIN_COLUMNS,
    MIN_KEY_FLOOR_PX,
    NFR2_KEY_ERROR_RATIO,
    build_keyboard,
    required_key_px,
)

THIS_SETUP = (1366, 768)
THIS_ERROR = 81.0


def overlaps(a, b) -> bool:
    ax, ay, aw, ah = a.rect
    bx, by, bw, bh = b.rect
    return not (ax + aw <= bx + 1e-6 or bx + bw <= ax + 1e-6
                or ay + ah <= by + 1e-6 or by + bh <= ay + 1e-6)


# --------------------------------------------------------------- NFR-2 sizing
def test_required_key_size_is_twice_the_error():
    assert required_key_px(81.0) == pytest.approx(NFR2_KEY_ERROR_RATIO * 81.0)
    assert required_key_px(120.0) == pytest.approx(240.0)


def test_required_key_size_has_a_floor():
    assert required_key_px(5.0) == MIN_KEY_FLOOR_PX
    for unknown in (0.0, None, float("nan"), float("inf")):
        assert required_key_px(unknown) == MIN_KEY_FLOOR_PX


# ------------------------------------------------- how NFR-2 is actually judged
def test_nfr2_is_judged_on_the_minimum_dimension_of_the_real_keys():
    """Not the width, not the nominal cell — the smallest key on both axes.

    A key that is wide but short fails just as surely as a narrow one: an error
    of one radius in the short direction lands outside it either way.
    """
    board = build_keyboard(THIS_SETUP, THIS_ERROR)      # the tall default
    width, height = board.smallest_key()

    measured_w = min(k.rect[2] for k in board.keys if k.selectable)
    measured_h = min(k.rect[3] for k in board.keys if k.selectable)
    assert (width, height) == (measured_w, measured_h)

    assert board.usable_error_px() == pytest.approx(min(width, height) / 2.0)
    assert board.meets_nfr2 == (min(width, height) >= board.min_key_px - 1e-6)


def test_a_short_key_fails_nfr2_even_when_it_is_wide():
    """The 2/3-height board is 137 px wide and 93 px tall: height decides."""
    board = build_keyboard(THIS_SETUP, 60.0)            # needs 120 px
    width, height = board.smallest_key()
    assert width > 120.0 > height, "the axes have to disagree for this to test"
    assert not board.meets_nfr2, "a width-only check would have passed this"
    assert board.nfr2_axes() == (True, False)


def test_both_axes_are_always_reported():
    board = build_keyboard(THIS_SETUP, 60.0)
    line = board.nfr2_line()
    assert "width OK" in line and "height BELOW" in line
    assert line in board.summary()


def test_a_taller_board_can_satisfy_nfr2_at_the_same_accuracy():
    tight = build_keyboard(THIS_SETUP, 60.0, height_ratio=2 / 3)
    roomy = build_keyboard(THIS_SETUP, 60.0, height_ratio=0.88)
    assert not tight.meets_nfr2 and roomy.meets_nfr2, roomy.note
    assert "keyboard_height_ratio" in tight.note


@pytest.mark.parametrize("screen,error", [
    ((1366, 768), 81.0),
    ((1920, 1080), 40.0),
    ((1920, 1080), 81.0),
    ((2560, 1440), 60.0),
])
def test_every_selectable_key_meets_nfr2(screen, error):
    board = build_keyboard(screen, error, layout="auto")
    assert board.meets_nfr2, board.note
    minimum = required_key_px(error)
    for key in board.keys:
        if not key.selectable:
            continue
        _, _, w, h = key.rect
        assert w >= minimum - 1e-6, f"{key.id} is {w:.0f} px wide"
        assert h >= minimum - 1e-6, f"{key.id} is {h:.0f} px tall"


def test_keyboard_stays_on_screen_and_docks_at_the_bottom():
    width, height = THIS_SETUP
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    x, y, w, h = board.rect
    assert x == 0 and w == width
    assert y + h == pytest.approx(height), "keyboard must sit on the bottom edge"
    assert 0 < h <= height * 0.8

    for key in board.keys:
        kx, ky, kw, kh = key.rect
        assert kx >= -1e-6 and kx + kw <= width + 1e-6
        assert ky >= y - 1e-6 and ky + kh <= height + 1e-6


def test_keys_do_not_overlap_within_a_page():
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    for page in range(board.pages):
        keys = board.keys_on(page)
        for i, first in enumerate(keys):
            for second in keys[i + 1:]:
                assert not overlaps(first, second), f"{first.id} overlaps {second.id}"


# ------------------------------------------------------------ layout selection
def test_a_good_calibration_on_a_big_screen_gets_full_qwerty():
    board = build_keyboard((1920, 1080), 40.0, layout="auto")
    assert board.name == "qwerty"
    assert board.pages == 1
    assert board.columns == 10
    for char in "QWERTYUIOPASDFGHJKLZXCVBNM":
        assert board.find(f"key.{char}") is not None, char


def test_a_large_error_on_a_small_screen_falls_back_to_paging():
    """10 QWERTY columns x 162 px needs 1620 px; this screen has 1366."""
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    assert board.name == "paged"
    assert board.pages == 2
    assert board.columns == 8
    assert board.meets_nfr2


def test_this_setup_gets_usable_keys():
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    key_w, key_h = board.key_size
    assert key_w >= 162 and key_h >= 162
    assert board.rect[3] <= THIS_SETUP[1] * 0.8
    assert "meets NFR-2" in board.summary()


def test_every_letter_is_reachable_across_the_pages():
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    reachable = set()
    for page in range(board.pages):
        for key in board.selectable_on(page):
            if key.action == "char":
                reachable.add(key.payload)
    assert set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") <= reachable


def test_controls_are_on_every_page():
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    for action in ("space", "backspace", "enter", "shift", "lang",
                   "pause", "recalibrate", "page"):
        key = board.find(f"fn.{action}")
        assert key is not None, action
        for page in range(board.pages):
            assert key.on_page(page), f"{action} missing from page {page}"


def test_a_hopeless_accuracy_reports_instead_of_pretending():
    board = build_keyboard(THIS_SETUP, 260.0, layout="auto")
    assert not board.meets_nfr2
    assert "BELOW NFR-2" in board.summary()
    assert board.rect[3] <= THIS_SETUP[1] * 0.8, "still has to fit on screen"
    for key in board.selectable_on(0):
        kx, ky, kw, kh = key.rect
        assert kx + kw <= THIS_SETUP[0] + 1e-6


def test_a_narrow_screen_blames_width_and_names_the_accuracy_needed():
    """1366 px / 8 columns / 2 caps usable error at about 85 px."""
    board = build_keyboard((1366, 768), 110.0, layout="auto")
    assert not board.meets_nfr2
    assert "wider screen" in board.note
    assert "85 px" in board.note, board.note


def test_a_short_screen_blames_height_and_can_be_recovered():
    tight = build_keyboard((2560, 900), 120.0, layout="auto", max_height_ratio=0.3)
    assert not tight.meets_nfr2
    assert "max-height-ratio" in tight.note

    roomy = build_keyboard((2560, 900), 120.0, layout="auto", max_height_ratio=0.95)
    assert roomy.meets_nfr2, roomy.note


def test_the_usable_accuracy_ceiling_holds_at_the_boundary():
    ceiling = 1366 / MIN_COLUMNS / 2.0            # ~85 px
    assert build_keyboard((1366, 768), ceiling - 2, layout="auto").meets_nfr2
    assert not build_keyboard((1366, 768), ceiling + 5,
                              layout="auto").meets_nfr2


# ------------------------------------------------------------------- key data
def test_function_keys_are_marked_and_have_labels():
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    for key in board.keys:
        if key.action in ("char",):
            assert not key.is_function
            assert key.payload and key.label_en
        elif key.action != "suggestion":
            assert key.is_function
            assert key.label_en.strip()


def test_modal_keys_are_the_ones_that_get_the_extended_dwell():
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    extended = {key.action for key in board.keys
                if key.action in EXTENDED_DWELL_ACTIONS}
    assert extended == {"pause", "recalibrate", "lang"}   # no Quit on paged
    for action in ("space", "backspace", "enter", "shift"):
        assert action not in EXTENDED_DWELL_ACTIONS


# ------------------------------------------------------------- the touch-up key
def test_the_paged_board_takes_a_fix_aim_key_when_it_has_room():
    """A 9th control column only if it still clears the required key size."""
    roomy = build_keyboard((1920, 1080), 40.0, layout="auto")
    assert roomy.has_action("touchup")
    assert roomy.meets_nfr2, roomy.note

    cramped = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    assert not cramped.has_action("touchup"), \
        "not worth pushing every control below NFR-2 for"
    assert cramped.meets_nfr2, cramped.note


def test_the_tall_board_always_has_one():
    board = build_keyboard(THIS_SETUP, THIS_ERROR)
    assert board.has_action("touchup")


def test_the_denser_control_row_still_does_not_overlap_anything():
    board = build_keyboard(THIS_SETUP, THIS_ERROR)      # 10 letters, 11 controls
    for page in range(board.pages):
        keys = board.keys_on(page)
        for i, first in enumerate(keys):
            for second in keys[i + 1:]:
                assert not overlaps(first, second), \
                    f"{first.id} overlaps {second.id}"


def test_suggestion_slots_exist_but_are_not_selectable():
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    slots = [key for key in board.keys if key.action == "suggestion"]
    assert len(slots) == 3, "spec Section 8 wants a 3-slot suggestion bar"
    assert all(not slot.selectable for slot in slots), "not dwell targets until M6"


def test_hebrew_labels_are_present_on_the_qwerty_board():
    board = build_keyboard((1920, 1080), 40.0, language="he", layout="auto")
    hebrew = [key for key in board.keys
              if key.action == "char" and key.label_he != key.label_en]
    assert len(hebrew) >= 20, "Hebrew labels should ride on the same grid"
    assert board.find("key.Q").label("he") != "Q"
    assert board.find("key.Q").label("en") == "Q"


def test_key_hit_testing_and_margins():
    board = build_keyboard(THIS_SETUP, THIS_ERROR, layout="auto")
    key = board.find("key.A")
    x, y, w, h = key.rect
    assert key.contains(*key.centre)
    assert not key.contains(x + w + 5, y + h / 2)
    assert key.contains(x + w + w * 0.2, y + h / 2, margin=0.25)
    assert key.centre == (x + w / 2, y + h / 2)
