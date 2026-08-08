"""Typing-stability diagnostics and the tuning knobs (--debug-typing).

Two things are pinned here: that the override plumbing has the right
precedence (flag > config > default), and that the dwell-loss accounting
attributes each loss to the cause that actually caused it — the numbers are
only worth logging if they are right.
"""
import argparse

import pytest

from config import DEFAULTS, resolve_tuning
from interaction.controller import DwellController, DwellSettings, DwellStats
from interaction.diagnostics import TypingDiagnostics, Window, _std
from interaction.layouts import build_keyboard
from vision.pipeline import GazePipeline

SCREEN = (1366, 768)
ERROR_PX = 44.3
DT = 1.0 / 30.0


@pytest.fixture
def board():
    return build_keyboard(SCREEN, ERROR_PX)


def args(**kwargs) -> argparse.Namespace:
    base = {"hysteresis": None, "min_cutoff": None, "grace_ms": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


# --------------------------------------------------------- override plumbing
def test_with_no_flags_and_no_config_the_defaults_apply():
    tuning = resolve_tuning(dict(DEFAULTS), args())
    assert tuning["hysteresis_margin"] == 0.25
    assert tuning["grace_ms"] == 200.0
    assert tuning["min_cutoff"] == 1.0


def test_config_is_used_when_the_flag_is_absent():
    config = dict(DEFAULTS)
    config.update({"hysteresis_margin": 0.45, "grace_period_ms": 350,
                   "one_euro_min_cutoff": 0.4})
    tuning = resolve_tuning(config, args())
    assert tuning["hysteresis_margin"] == 0.45
    assert tuning["grace_ms"] == 350.0
    assert tuning["min_cutoff"] == 0.4


def test_a_flag_beats_config():
    config = dict(DEFAULTS)
    config.update({"hysteresis_margin": 0.45, "grace_period_ms": 350,
                   "one_euro_min_cutoff": 0.4})
    tuning = resolve_tuning(config, args(hysteresis=0.6, grace_ms=120.0,
                                         min_cutoff=0.2))
    assert tuning["hysteresis_margin"] == 0.6
    assert tuning["grace_ms"] == 120.0
    assert tuning["min_cutoff"] == 0.2


def test_an_explicit_zero_is_a_real_request_not_a_fallback():
    config = dict(DEFAULTS)
    config["hysteresis_margin"] = 0.45
    assert resolve_tuning(config, args(hysteresis=0.0))["hysteresis_margin"] == 0.0


def test_resolve_tuning_survives_a_partial_config_and_no_args():
    tuning = resolve_tuning({}, None)
    assert tuning["hysteresis_margin"] == DEFAULTS["hysteresis_margin"]
    assert tuning["min_cutoff"] == DEFAULTS["one_euro_min_cutoff"]


def test_the_cli_defaults_leave_every_knob_absent():
    import main as app_main

    parsed = app_main.build_parser().parse_args([])
    assert parsed.hysteresis is None
    assert parsed.min_cutoff is None
    assert parsed.grace_ms is None
    assert parsed.debug_typing is False
    assert parsed.debug_interval == 5.0

    steered = app_main.build_parser().parse_args(
        ["--hysteresis", "0.45", "--min-cutoff", "0.3", "--grace-ms", "400",
         "--debug-typing", "--debug-interval", "2"])
    assert steered.hysteresis == 0.45 and steered.min_cutoff == 0.3
    assert steered.grace_ms == 400.0 and steered.debug_typing is True
    assert steered.debug_interval == 2.0


def test_min_cutoff_reaches_the_actual_filters():
    """A knob that does not reach the filter is a knob that does nothing."""
    pipeline = GazePipeline(min_cutoff=0.3, beta=0.01)
    assert all(f.min_cutoff == 0.3 for f in pipeline._filters)
    assert all(f.beta == 0.01 for f in pipeline._filters)

    pipeline.set_model(None)                 # rebuilds them
    assert all(f.min_cutoff == 0.3 for f in pipeline._filters)


def test_the_default_pipeline_keeps_the_spec_tuning():
    pipeline = GazePipeline()
    assert all(f.min_cutoff == 1.0 and f.beta == 0.007
               for f in pipeline._filters)


def test_lower_min_cutoff_actually_smooths_harder():
    from gaze.smoothing import OneEuroFilter

    def wobble(min_cutoff):
        filt = OneEuroFilter(min_cutoff=min_cutoff, beta=0.007)
        out = [filt.apply(500.0 + (40.0 if i % 2 else -40.0), i * DT)
               for i in range(60)]
        return _std(out[20:])

    assert wobble(0.3) < wobble(1.0) < wobble(4.0)


# --------------------------------------------------- dwell-loss accounting
def stare(controller, position, seconds, start=0.0, is_fixating=True,
          stream_valid=True):
    now = start
    for _ in range(max(1, int(round(seconds / DT)))):
        now += DT
        controller.update(position[0], position[1], is_fixating,
                          stream_valid, now)
    return now


def test_a_clean_dwell_loses_nothing(board):
    controller = DwellController(board, DwellSettings())
    stare(controller, board.find("key.H").centre, 1.2)
    stats = controller.stats
    assert stats.activations == 1
    assert stats.dwell_losses == 0
    assert stats.focus_changes == 1, "landing on the key is one change"


def test_a_focus_steal_is_counted_and_named(board):
    """One jittered frame across a row boundary — the reported failure."""
    controller = DwellController(board, DwellSettings())
    key, below = board.find("key.H"), board.find("key.N")
    now = stare(controller, key.centre, 0.7)
    assert controller.progress > 0.5

    controller.update(key.centre[0], below.centre[1], True, True, now + DT)
    stats = controller.stats
    assert stats.focus_stolen == 1
    assert stats.fixation_dropped == 0 and stats.grace_expired == 0
    assert controller.progress == 0.0, "the dwell is discarded outright"


def test_landing_on_a_key_with_no_progress_is_not_a_steal(board):
    controller = DwellController(board, DwellSettings())
    now = stare(controller, board.find("key.H").centre, 0.03)
    stare(controller, board.find("key.J").centre, 0.03, start=now)
    assert controller.stats.focus_changes == 2
    assert controller.stats.focus_stolen == 0, "nothing was accumulated to lose"


def test_a_fixation_drop_is_counted_once_per_wobble(board):
    controller = DwellController(board, DwellSettings())
    key = board.find("key.H")
    now = stare(controller, key.centre, 0.5)
    now = stare(controller, key.centre, 0.4, start=now, is_fixating=False)

    assert controller.stats.fixation_dropped == 1, "an edge, not a frame count"
    assert controller.stats.focus_stolen == 0

    now = stare(controller, key.centre, 0.2, start=now)
    stare(controller, key.centre, 0.2, start=now, is_fixating=False)
    assert controller.stats.fixation_dropped == 2


def test_grace_expiry_is_counted_when_the_gaze_leaves_everything(board):
    controller = DwellController(board, DwellSettings(grace_ms=100))
    now = stare(controller, board.find("key.H").centre, 0.6)
    stare(controller, (700.0, 20.0), 0.5, start=now)      # above the board

    assert controller.stats.grace_expired == 1
    assert controller.stats.focus_stolen == 0
    assert controller.focused_key is None


def test_leaving_without_any_progress_is_not_a_grace_expiry(board):
    controller = DwellController(board, DwellSettings(grace_ms=100))
    now = stare(controller, board.find("key.H").centre, 0.03)
    stare(controller, (700.0, 20.0), 0.5, start=now)
    assert controller.stats.grace_expired == 0


def test_a_lost_stream_is_counted_but_costs_no_progress(board):
    controller = DwellController(board, DwellSettings())
    key = board.find("key.H")
    now = stare(controller, key.centre, 0.5)
    progress = controller.progress
    stare(controller, key.centre, 0.3, start=now, stream_valid=False)

    assert controller.stats.stream_lost == 9
    assert controller.progress == pytest.approx(progress), "frozen, not reset"
    assert controller.stats.dwell_losses == 0


def test_stats_deltas_are_per_window(board):
    controller = DwellController(board, DwellSettings())
    stare(controller, board.find("key.H").centre, 1.2)
    mark = controller.stats.snapshot()
    stare(controller, board.find("key.J").centre, 1.2, start=5.0)

    window = controller.stats.since(mark)
    assert window.activations == 1, "only what happened after the mark"
    assert controller.stats.activations == 2


# ------------------------------------------------------------- the reporting
class Sample:
    def __init__(self, x, y, is_fixating=True, stream_valid=True, timestamp=0.0):
        self.x, self.y = x, y
        self.is_fixating = is_fixating
        self.stream_valid = stream_valid
        self.timestamp = timestamp
        self.has_gaze = True


def feed(diagnostics, points, start=0.0, is_fixating=True):
    now = start
    for x, y in points:
        now += DT
        diagnostics.record(Sample(x, y, is_fixating=is_fixating, timestamp=now))
    return now


def test_a_window_is_reported_on_the_interval(board):
    lines = []
    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, key_size=board.smallest_key(),
                                    interval_s=5.0, log=lines.append)
    diagnostics.tick(0.0)
    feed(diagnostics, [(700.0, 500.0)] * 30)
    assert diagnostics.tick(3.0) is None, "not yet"
    assert lines == []

    window = diagnostics.tick(5.5)
    assert window is not None and len(lines) == 1
    assert "[typing]" in lines[0] and "focus" in lines[0]
    assert "jitter x" in lines[0] and "spread x" in lines[0]


def test_the_window_measures_each_axis_separately(board):
    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, interval_s=1.0, log=lambda _: None)
    diagnostics.tick(0.0)
    # steady in x, wobbling +-40 px in y — the reported failure mode
    feed(diagnostics, [(700.0, 500.0 + (40 if i % 2 else -40))
                       for i in range(60)])
    window = diagnostics.tick(2.0)

    assert _std(window.xs) == pytest.approx(0.0, abs=0.01)
    assert _std(window.ys) == pytest.approx(40.0, abs=1.0)
    assert window.duty_cycle == 1.0


def test_the_duty_cycle_counts_only_fixating_samples(board):
    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, interval_s=1.0, log=lambda _: None)
    diagnostics.tick(0.0)
    now = feed(diagnostics, [(700.0, 500.0)] * 30)
    feed(diagnostics, [(700.0, 500.0)] * 10, start=now, is_fixating=False)
    window = diagnostics.tick(3.0)

    assert window.samples == 40 and window.fixating == 30
    assert window.duty_cycle == pytest.approx(0.75)
    assert len(window.hold_xs) == 30, "hold sd only measures held gaze"


def test_invalid_samples_are_not_measured(board):
    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, log=lambda _: None)
    sample = Sample(700.0, 500.0)
    sample.stream_valid = False
    diagnostics.record(sample)
    assert diagnostics.session.samples == 0


def test_the_summary_reports_the_session_and_names_a_cause(board):
    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, key_size=board.smallest_key(),
                                    log=lambda _: None)
    diagnostics.tick(0.0)
    key, below = board.find("key.H"), board.find("key.N")
    now = 0.0
    for _ in range(6):                     # dwell, then get robbed by a wobble
        now = stare(controller, key.centre, 0.7, start=now)
        controller.update(key.centre[0], below.centre[1], True, True, now + DT)
        now += DT
    feed(diagnostics, [(key.centre[0], key.centre[1] + (50 if i % 2 else -50))
                       for i in range(60)])
    diagnostics.close(now=30.0)

    summary = "\n".join(diagnostics.summary_lines())
    assert "typing-stability summary" in summary
    assert "focus stolen 6" in summary
    assert "jitter" in summary and "spread" in summary
    assert "focus steals" in diagnostics.likely_cause()
    assert "Vertical is the worse axis" in diagnostics.likely_cause()
    assert "min-cutoff" in diagnostics.likely_cause()


def test_a_healthy_session_says_so(board):
    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, key_size=board.smallest_key(),
                                    log=lambda _: None)
    diagnostics.tick(0.0)
    now = 0.0
    # no gap between keys: each stare ends inside its own refractory, so the
    # gaze never leaves a key holding part of a dwell
    for char in "HELP":
        now = stare(controller, board.find(f"key.{char}").centre, 1.1, start=now)
    feed(diagnostics, [(700.0, 500.0)] * 100)
    diagnostics.close(now=10.0)

    assert controller.stats.activations == 4
    assert controller.stats.dwell_losses == 0
    assert "healthy" in diagnostics.likely_cause()


def test_a_steady_gaze_that_never_activates_blames_calibration(board):
    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, key_size=board.smallest_key(),
                                    log=lambda _: None)
    diagnostics.tick(0.0)
    feed(diagnostics, [(700.0, 500.0)] * 100)      # rock steady, no dwelling
    diagnostics.close(now=10.0)
    assert "calibration problem" in diagnostics.likely_cause()


def test_the_summary_survives_a_session_with_no_samples(board):
    diagnostics = TypingDiagnostics(DwellController(board), log=lambda _: None)
    diagnostics.close()
    assert diagnostics.summary_lines() == ["[typing] no gaze samples were recorded."]


def test_every_reported_line_is_console_safe(board):
    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, key_size=board.smallest_key(),
                                    interval_s=1.0, log=lambda _: None)
    diagnostics.tick(0.0)
    feed(diagnostics, [(700.0, 500.0)] * 40)
    window = diagnostics.tick(2.0)
    diagnostics.close(now=2.0)
    window.line().encode("cp1255")
    "\n".join(diagnostics.summary_lines()).encode("cp1255")


# -------------------------------------------------- the overlay actually feeds it
@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def wired(qapp, board):
    from ui.overlay import KeyboardOverlay
    from tests.test_overlay import ScriptedPipeline

    controller = DwellController(board, DwellSettings())
    diagnostics = TypingDiagnostics(controller, key_size=board.smallest_key(),
                                    interval_s=1.0, log=lambda _: None)
    widget = KeyboardOverlay(board, controller, ScriptedPipeline(), None,
                             screen_size=SCREEN, show_webcam=False,
                             diagnostics=diagnostics)
    yield widget, diagnostics
    widget.close()


def test_the_overlay_feeds_the_diagnostics(wired):
    overlay, diagnostics = wired
    overlay.pipeline.look_at(overlay.keyboard.find("key.H").centre, 1.2,
                             start=0.0)
    overlay.tick()
    assert diagnostics.session.samples > 20
    assert diagnostics.session.fixating > 20


def test_the_overlay_reports_on_the_interval(wired):
    overlay, diagnostics = wired
    lines = []
    diagnostics.log = lines.append
    overlay.pipeline.look_at(overlay.keyboard.find("key.H").centre, 3.0,
                             start=0.0)
    overlay.tick()
    assert lines, "no window was reported despite 3 s of samples"
    assert "[typing]" in lines[0]


def test_an_overlay_without_diagnostics_is_unaffected(qapp, board):
    from ui.overlay import KeyboardOverlay
    from tests.test_overlay import ScriptedPipeline

    widget = KeyboardOverlay(board, DwellController(board), ScriptedPipeline(),
                             None, screen_size=SCREEN, show_webcam=False)
    try:
        assert widget.diagnostics is None
        widget.pipeline.look_at(board.find("key.H").centre, 1.2, start=0.0)
        widget.tick()                    # must not raise
    finally:
        widget.close()


# ------------------------------------------- what the knobs can and cannot do
# Pinned because it is counter-intuitive and cost a debugging session: two of
# the three stability knobs cannot affect typing *between* keys at all. If a
# sticky-focus rule is ever adopted these tests should change with it.
def test_hysteresis_does_nothing_between_keys(board):
    """Hit regions are gapless, so the challenger always wins outright.

    ``_hit_test`` returns the first key whose *unexpanded* region contains the
    point, and inside the board there is always one. The focused key's grown
    region is only consulted when nothing owns the point — i.e. off the board.
    """
    key = board.find("key.H")
    owners = {}
    for margin in (0.0, 0.25, 0.45, 0.90):
        controller = DwellController(board, DwellSettings(hysteresis_margin=margin))
        controller._focused = key
        owners[margin] = [
            (controller._hit_test(key.centre[0], key.centre[1] + dy) or key).id
            for dy in (-60, -47, -30, 0, 30, 47, 60)
        ]
    assert len(set(map(tuple, owners.values()))) == 1, \
        f"hysteresis changed ownership, so this note is out of date: {owners}"
    assert "key.N" in owners[0.90], "a 47 px wobble still crosses the row"


def test_hysteresis_does_still_act_off_the_board(board):
    """Where it is not dead: above the top row, nothing else owns the point."""
    key = board.find("key.Q")
    above = key.rect[1] - key.rect[3] * 0.15

    tight = DwellController(board, DwellSettings(hysteresis_margin=0.0))
    tight._focused = key
    generous = DwellController(board, DwellSettings(hysteresis_margin=0.25))
    generous._focused = key

    assert tight._hit_test(key.centre[0], above) is None
    assert generous._hit_test(key.centre[0], above) is key


def test_a_focus_steal_bypasses_the_grace_period_entirely(board):
    """Grace only runs when the gaze leaves *every* key, never on a steal."""
    key, below = board.find("key.H"), board.find("key.N")
    for grace in (0.0, 200.0, 2000.0):
        controller = DwellController(board, DwellSettings(grace_ms=grace))
        now = stare(controller, key.centre, 0.7)
        controller.update(key.centre[0], below.centre[1], True, True, now + DT)
        assert controller.progress == 0.0, \
            f"grace_ms={grace} did not soften the steal"
        assert controller.stats.grace_expired == 0


def test_smoothing_is_the_only_knob_that_reaches_the_jitter(board):
    """The one knob that does bite: less residual wobble, fewer boundaries hit.

    Measured as the peak-to-peak the filter still passes, against the half-row
    a wobble has to clear to steal focus.
    """
    from gaze.smoothing import OneEuroFilter

    def amplitude(min_cutoff, raw=50.0):
        filt = OneEuroFilter(min_cutoff=min_cutoff, beta=0.007)
        out = [filt.apply(raw if i % 2 else -raw, i * DT) for i in range(90)]
        return max(out[30:]) - min(out[30:])

    passed = [amplitude(mc) for mc in (0.3, 1.0, 4.0)]
    assert passed == sorted(passed), f"smoothing is not monotonic: {passed}"
    assert passed[0] < passed[-1] / 2, "the knob has to have real range"

    # One-Euro is adaptive, so its grip weakens as the input grows; +-100 px of
    # raw wobble is where the setting still decides the outcome outright.
    half_row = board.smallest_key()[1] / 2.0
    assert amplitude(0.3, raw=100.0) < half_row < amplitude(4.0, raw=100.0), \
        "at this wobble size the setting decides whether focus is stolen"
