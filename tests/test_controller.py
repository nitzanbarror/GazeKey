"""Dwell controller tests (spec Section 12) — scripted gaze streams, no camera.

Every timing rule from spec Section 7 is driven here as an explicit stream of
samples so the behaviour is pinned rather than inferred.
"""
import pytest

from interaction.controller import (
    ALWAYS_LIVE_ACTIONS,
    Activation,
    DwellController,
    DwellSettings,
    DwellState,
)
from interaction.layouts import build_keyboard

SCREEN = (1366, 768)
ERROR_PX = 81.0
DT = 1.0 / 30.0


@pytest.fixture
def keyboard():
    # the paged board: gapless grid, non-selectable suggestion strip -
    # the geometry these generic dwell rules were written against
    return build_keyboard(SCREEN, ERROR_PX, layout="auto")


@pytest.fixture
def controller(keyboard):
    return DwellController(keyboard, DwellSettings())


def key_of(keyboard, key_id: str):
    key = keyboard.find(key_id)
    assert key is not None, f"{key_id} missing from the layout"
    return key


def stare(controller, position, seconds, start=0.0, dt=DT,
          is_fixating=True, stream_valid=True):
    """Hold the gaze at ``position``; returns (activations, end time)."""
    activations = []
    now = start
    steps = max(1, int(round(seconds / dt)))
    for _ in range(steps):
        now += dt
        fired = controller.update(position[0], position[1], is_fixating,
                                  stream_valid, now)
        if fired is not None:
            activations.append(fired)
    return activations, now


# ------------------------------------------------------------ dwell accumulation
def test_dwell_activates_after_the_configured_time(controller, keyboard):
    key = key_of(keyboard, "key.A")
    fired, _ = stare(controller, key.centre, 0.9)
    assert fired == [], "activated before the dwell time"
    assert controller.progress > 0.8

    fired, _ = stare(controller, key.centre, 0.3, start=0.9)
    assert len(fired) == 1
    assert fired[0].key.id == "key.A"
    assert not fired[0].suppressed


def test_progress_reports_how_far_the_dwell_has_come(controller, keyboard):
    key = key_of(keyboard, "key.A")
    assert controller.progress == 0.0
    assert controller.state is DwellState.IDLE

    stare(controller, key.centre, 0.5)
    assert 0.35 < controller.progress < 0.65
    assert controller.state is DwellState.DWELLING
    assert controller.focused_key.id == "key.A"


def test_dwell_only_advances_while_fixating(controller, keyboard):
    key = key_of(keyboard, "key.A")
    fired, now = stare(controller, key.centre, 3.0, is_fixating=False)
    assert fired == [], "a moving gaze must never activate a key"
    assert controller.progress == 0.0
    assert controller.focused_key.id == "key.A", "focus still lands, just no dwell"

    fired, _ = stare(controller, key.centre, 1.1, start=now)
    assert len(fired) == 1


def test_an_invalid_stream_freezes_the_dwell(controller, keyboard):
    """Spec Section 6: a blink freezes the timer, it does not reset it."""
    key = key_of(keyboard, "key.A")
    stare(controller, key.centre, 0.8)
    frozen = controller.progress

    _, now = stare(controller, key.centre, 1.0, start=0.8, stream_valid=False)
    assert controller.progress == pytest.approx(frozen), "dwell was not frozen"

    fired, _ = stare(controller, key.centre, 0.3, start=now)
    assert len(fired) == 1, "dwell should finish from where it froze"


def test_dwell_time_is_configurable(keyboard):
    slow = DwellController(keyboard, DwellSettings(dwell_s=1.8))
    key = key_of(keyboard, "key.A")
    fired, now = stare(slow, key.centre, 1.5)
    assert fired == []
    fired, _ = stare(slow, key.centre, 0.5, start=now)
    assert len(fired) == 1


# ------------------------------------------------------------------ hysteresis
def test_focus_survives_a_wobble_into_the_margin(controller, keyboard):
    """A focused key keeps focus inside its 25% margin (spec Section 7).

    The margin protects the board's outer edge and the non-selectable
    suggestion strip above the top row — and, since NFR-7, the strip of the
    *neighbouring key* it overlaps as well: hit regions are gapless, so asking
    the neighbours first is what made the margin inert between keys.
    """
    key = key_of(keyboard, "key.A")
    x, y, w, h = key.rect
    stare(controller, key.centre, 0.5)
    progress = controller.progress

    just_above = (x + w / 2, y - h * 0.1)           # into the suggestion strip
    stare(controller, just_above, 0.2, start=0.5)
    assert controller.focused_key.id == "key.A", "focus lost inside the margin"
    assert controller.progress > progress, "dwell should keep advancing"


def test_focus_survives_a_wobble_into_a_neighbouring_key(controller, keyboard):
    """The NFR-7 fix: the grown region wins against a neighbour too."""
    key = key_of(keyboard, "key.A")
    x, y, w, h = key.rect
    stare(controller, key.centre, 0.5)
    progress = controller.progress

    just_inside_next = (x + w * 1.10, y + h / 2)    # 10% into the key to its right
    stare(controller, just_inside_next, 0.2, start=0.5)
    assert controller.focused_key.id == "key.A", "the neighbour stole a wobble"
    assert controller.progress > progress, "dwell should keep advancing"


def test_a_neighbour_steals_focus_from_outside_the_grown_region(controller,
                                                                keyboard):
    left = key_of(keyboard, "key.A")
    right = key_of(keyboard, "key.B")
    stare(controller, left.centre, 0.5)
    assert controller.focused_key.id == left.id

    stare(controller, right.centre, 0.1, start=0.5)
    assert controller.focused_key.id == right.id, "neighbour never took focus"
    assert controller.progress < 0.2, "dwell must restart on the new key"


def test_a_stolen_dwell_decays_instead_of_vanishing(controller, keyboard):
    """Spec Section 7: a steal costs grace, not the whole dwell."""
    left = key_of(keyboard, "key.A")
    right = key_of(keyboard, "key.B")
    stare(controller, left.centre, 0.6)
    before = controller.progress

    stare(controller, right.centre, 0.1, start=0.6)       # half of the grace
    fired, _ = stare(controller, left.centre, 0.8, start=0.7)
    assert len(fired) == 1 and fired[0].key.id == left.id, \
        "the interrupted dwell should have resumed, not restarted"
    assert before > 0.5


def test_hysteresis_margin_is_configurable(keyboard):
    key = key_of(keyboard, "key.A")
    x, y, w, h = key.rect
    probe = (x + w / 2, y - h * 0.4)               # above the top letter row

    wide = DwellController(keyboard, DwellSettings(hysteresis_margin=0.5))
    stare(wide, key.centre, 0.4)
    stare(wide, probe, 0.1, start=0.4)
    assert wide.focused_key is not None and wide.focused_key.id == key.id

    narrow = DwellController(keyboard, DwellSettings(hysteresis_margin=0.05))
    stare(narrow, key.centre, 0.4)
    stare(narrow, probe, 0.5, start=0.4)
    assert narrow.focused_key is None or narrow.focused_key.id != key.id


# ----------------------------------------------------------------- grace period
def test_leaving_briefly_decays_instead_of_resetting(controller, keyboard):
    key = key_of(keyboard, "key.A")
    stare(controller, key.centre, 0.6)
    before = controller.progress

    away = (5.0, 5.0)                               # far from any key
    stare(controller, away, 0.1, start=0.6)         # half of the 200 ms grace
    assert 0.0 < controller.progress < before, "dwell should decay, not vanish"

    fired, _ = stare(controller, key.centre, 0.8, start=0.7)
    assert len(fired) == 1, "returning quickly should finish the dwell"
    # a reset (rather than a decay) would have needed the full second again


def test_staying_away_past_the_grace_drops_the_focus(controller, keyboard):
    key = key_of(keyboard, "key.A")
    stare(controller, key.centre, 0.6)
    stare(controller, (5.0, 5.0), 0.4, start=0.6)
    assert controller.progress == 0.0
    assert controller.focused_key is None
    assert controller.state is DwellState.IDLE


def test_grace_period_is_configurable(keyboard):
    key = key_of(keyboard, "key.A")
    quick = DwellController(keyboard, DwellSettings(grace_ms=50))
    stare(quick, key.centre, 0.6)
    stare(quick, (5.0, 5.0), 0.1, start=0.6)
    assert quick.focused_key is None, "a 50 ms grace should already be spent"


# ------------------------------------------------------------------ refractory
def test_a_key_cannot_repeat_inside_the_refractory_window(controller, keyboard):
    key = key_of(keyboard, "key.A")
    fired, now = stare(controller, key.centre, 1.1)
    assert len(fired) == 1

    fired, now = stare(controller, key.centre, 0.35, start=now)
    assert fired == [], "key repeated inside the 400 ms refractory"
    assert controller.progress == 0.0

    fired, _ = stare(controller, key.centre, 1.15, start=now)
    assert len(fired) == 1, "key should be available again after the refractory"


def test_refractory_does_not_block_a_different_key(controller, keyboard):
    first = key_of(keyboard, "key.A")
    second = key_of(keyboard, "key.B")
    fired, now = stare(controller, first.centre, 1.1)
    assert len(fired) == 1

    fired, _ = stare(controller, second.centre, 1.1, start=now)
    assert len(fired) == 1 and fired[0].key.id == second.id


def test_holding_a_key_repeats_it_at_the_expected_rate(controller, keyboard):
    """A long stare should type, pause for the refractory, then type again."""
    key = key_of(keyboard, "key.A")
    fired, _ = stare(controller, key.centre, 5.0)
    assert 2 <= len(fired) <= 4, f"{len(fired)} activations in 5 s"
    gaps = [b.timestamp - a.timestamp for a, b in zip(fired, fired[1:])]
    # 1.0 s dwell + 0.4 s refractory, minus one frame of quantisation
    assert all(gap >= 1.3 for gap in gaps), gaps


# --------------------------------------------------------------- extended dwell
@pytest.mark.parametrize("action", sorted({"pause", "recalibrate", "lang"}))
def test_modal_function_keys_need_the_extended_dwell(controller, keyboard, action):
    key = key_of(keyboard, f"fn.{action}")
    assert controller.required_s(key) == controller.settings.extended_dwell_s

    fired, now = stare(controller, key.centre, 1.5)
    assert fired == [], f"{action} fired at the normal dwell time"
    fired, _ = stare(controller, key.centre, 0.7, start=now)
    assert len(fired) == 1


def test_ordinary_keys_keep_the_normal_dwell(controller, keyboard):
    for key_id in ("key.A", "fn.space", "fn.backspace", "fn.enter"):
        key = key_of(keyboard, key_id)
        assert controller.required_s(key) == controller.settings.dwell_s


# ----------------------------------------------------------------------- pause
def test_pausing_suppresses_typing_but_keeps_the_feedback(controller, keyboard):
    controller.toggle_pause()
    assert controller.paused

    key = key_of(keyboard, "key.A")
    fired, _ = stare(controller, key.centre, 1.1)
    assert len(fired) == 1, "dwell and feedback must keep working while paused"
    assert fired[0].suppressed, "a paused keyboard must not inject"


@pytest.mark.parametrize("action", sorted(ALWAYS_LIVE_ACTIONS))
def test_the_live_keys_still_work_while_paused(action):
    """Unpause, fix the aim, recalibrate, leave — none of them type."""
    board = build_keyboard(SCREEN, ERROR_PX)          # the tall default
    controller = DwellController(board, DwellSettings())
    controller.toggle_pause()
    key = key_of(board, f"fn.{action}")
    fired, _ = stare(controller, key.centre, 2.2)
    assert len(fired) == 1
    assert not fired[0].suppressed, f"{action} must work while paused"


def test_pause_toggles_back(controller):
    assert controller.toggle_pause() is True
    assert controller.toggle_pause() is False
    assert not controller.paused


# ------------------------------------------------------------------- paging
def test_paging_changes_which_keys_are_reachable(controller, keyboard):
    assert keyboard.pages == 2, "this screen and accuracy should page"
    first_page_key = key_of(keyboard, "key.A")
    fired, _ = stare(controller, first_page_key.centre, 1.1)
    assert len(fired) == 1

    controller.next_page()
    assert controller.page == 1
    assert controller.focused_key is None, "paging must drop the old focus"

    fired, _ = stare(controller, first_page_key.centre, 1.5, start=2.0)
    assert len(fired) == 1
    assert fired[0].key.id != first_page_key.id, "still typing the old page"
    assert fired[0].key.pages == (1,), "that position should hold a page-2 key"


def test_paging_wraps_around(controller, keyboard):
    for _ in range(keyboard.pages):
        controller.next_page()
    assert controller.page == 0


# -------------------------------------------------------------------- misc
def test_gaze_outside_the_keyboard_selects_nothing(controller):
    fired, _ = stare(controller, (600.0, 20.0), 2.0)
    assert fired == []
    assert controller.focused_key is None


def test_suggestion_slots_are_not_dwell_targets_yet(controller, keyboard):
    slot = key_of(keyboard, "sug.0")
    fired, _ = stare(controller, slot.centre, 2.0)
    assert fired == [], "suggestion slots become selectable in M6"


def test_reset_clears_focus_and_refractory(controller, keyboard):
    key = key_of(keyboard, "key.A")
    fired, now = stare(controller, key.centre, 1.1)
    assert len(fired) == 1

    controller.reset()
    assert controller.focused_key is None
    fired, _ = stare(controller, key.centre, 1.1, start=now)
    assert len(fired) == 1, "reset should clear the refractory too"


def test_settings_come_from_config():
    from config import DEFAULTS

    settings = DwellSettings.from_config(DEFAULTS)
    assert settings.dwell_s == DEFAULTS["dwell_time_s"]
    assert settings.extended_dwell_s == DEFAULTS["extended_dwell_s"]
    assert settings.hysteresis_margin == DEFAULTS["hysteresis_margin"]
    assert settings.grace_ms == DEFAULTS["grace_period_ms"]
    assert settings.refractory_ms == DEFAULTS["refractory_ms"]
