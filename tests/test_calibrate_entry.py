"""Entry-point helper tests for calibrate.py — no camera, no Qt application."""
import random

import numpy as np
import pytest

import calibrate
from gaze.calibration_session import CalibrationSession
from gaze.features import FrameFeatures
from vision.pipeline import STATE_NO_CAMERA, STATE_OK, GazeSample
from tests.test_calibration_session import SCREEN, SyntheticUser, drive


class StubPipeline:
    """Returns a fixed batch of samples from every drain()."""

    last_error = "camera 0 not available"

    def __init__(self, samples):
        self.samples = list(samples)

    def drain(self):
        return list(self.samples)


def sample(state: str) -> GazeSample:
    return GazeSample(FrameFeatures(valid=state == STATE_OK, timestamp=1.0),
                      state, 1.0)


# ----------------------------------------------------------------- practice
def test_practice_flags_are_exposed():
    parser = calibrate.build_parser()
    help_text = parser.format_help()
    for flag in ("--practice", "--no-practice", "--practice-targets"):
        assert flag in help_text

    defaults = parser.parse_args([])
    assert defaults.practice is False
    assert defaults.no_practice is False
    assert defaults.practice_targets == 10
    assert parser.parse_args(["--practice-targets", "5"]).practice_targets == 5


def test_hit_radius_comes_from_the_measured_accuracy():
    from interaction.practice import MIN_HIT_RADIUS_PX, hit_radius_for

    assert hit_radius_for(95.1) == pytest.approx(95.1)
    assert hit_radius_for(40.0) == MIN_HIT_RADIUS_PX


# ---------------------------------------------------------------- head mode
def head_args(**overrides):
    import argparse

    defaults = dict(fixed_head=False, with_head_sweep=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_head_rest_is_the_default():
    from config import DEFAULTS

    assert calibrate.resolve_fixed_head(DEFAULTS, head_args()) is True


def test_with_head_sweep_opts_out():
    from config import DEFAULTS

    assert calibrate.resolve_fixed_head(
        DEFAULTS, head_args(with_head_sweep=True)) is False


def test_explicit_fixed_head_beats_a_config_that_says_otherwise():
    assert calibrate.resolve_fixed_head(
        {"fixed_head": False}, head_args(fixed_head=True)) is True


def test_config_can_turn_the_head_rest_assumption_off():
    assert calibrate.resolve_fixed_head({"fixed_head": False}, head_args()) is False


def test_head_mode_flags_are_mutually_exclusive():
    parser = calibrate.build_parser()
    assert parser.parse_args(["--fixed-head"]).fixed_head is True
    assert parser.parse_args(["--with-head-sweep"]).with_head_sweep is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--fixed-head", "--with-head-sweep"])


# ------------------------------------------------------------------- pacing
def pacing_args(**overrides):
    import argparse

    defaults = dict(slow=False, settle_ms=None, collect_samples=None,
                    collect_max_s=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_pacing_defaults_come_from_config():
    from config import DEFAULTS

    pacing = calibrate.resolve_pacing(DEFAULTS, pacing_args())
    assert pacing == {"settle_ms": 700.0, "collect_samples": 45,
                      "collect_max_s": 4.0}


def test_slow_preset_is_gentler():
    from config import DEFAULTS

    pacing = calibrate.resolve_pacing(DEFAULTS, pacing_args(slow=True))
    assert pacing["settle_ms"] == 1000.0
    assert pacing["collect_samples"] == 60
    assert pacing["collect_max_s"] == 4.0     # untouched by the preset


def test_explicit_flags_beat_the_slow_preset():
    from config import DEFAULTS

    pacing = calibrate.resolve_pacing(
        DEFAULTS, pacing_args(slow=True, settle_ms=250, collect_max_s=2.5)
    )
    assert pacing["settle_ms"] == 250.0
    assert pacing["collect_samples"] == 60     # still from the preset
    assert pacing["collect_max_s"] == 2.5


def test_stored_config_overrides_the_built_in_defaults():
    config = dict(calibration_settle_ms=1500, calibration_collect_samples=90,
                  calibration_collect_max_s=6.0)
    pacing = calibrate.resolve_pacing(config, pacing_args())
    assert pacing == {"settle_ms": 1500.0, "collect_samples": 90,
                      "collect_max_s": 6.0}


def test_pacing_reaches_the_session():
    from gaze.calibration_session import CalibrationSession

    session = CalibrationSession(SCREEN, **calibrate.resolve_pacing(
        {"calibration_settle_ms": 700, "calibration_collect_samples": 45,
         "calibration_collect_max_s": 4.0}, pacing_args(slow=True)))
    assert session.settle_s == pytest.approx(1.0)
    assert session.collect_samples == 60
    assert session.collect_max_s == 4.0


def test_cli_exposes_every_pacing_flag():
    help_text = calibrate.build_parser().format_help()
    for flag in ("--settle-ms", "--collect-samples", "--collect-max-s", "--slow"):
        assert flag in help_text


def test_cli_parses_pacing_flags_into_a_session():
    args = calibrate.build_parser().parse_args(
        ["--slow", "--collect-max-s", "6"]
    )
    pacing = calibrate.resolve_pacing(
        {"calibration_settle_ms": 700, "calibration_collect_samples": 45,
         "calibration_collect_max_s": 4.0}, args)
    assert pacing == {"settle_ms": 1000.0, "collect_samples": 60,
                      "collect_max_s": 6.0}


# --------------------------------------------------------- fixation settings
class StubFixationPipeline:
    fixation_dispersion_px = 110.0
    fixation_window_ms = 150.0
    tracking_hold_s = 0.3


def test_fixation_settings_are_reported(capsys):
    calibrate.report_fixation_settings(StubFixationPipeline(), 81.0)
    out = capsys.readouterr().out
    assert "dispersion <= 110 px" in out
    assert "over 150 ms" in out
    assert "hold 300 ms" in out
    assert "suggests" not in out, "110 px already suits an 81 px calibration"


def test_a_much_worse_accuracy_suggests_a_wider_threshold(capsys):
    from config import suggested_fixation_dispersion_px

    calibrate.report_fixation_settings(StubFixationPipeline(), 150.0)
    out = capsys.readouterr().out
    expected = f"{suggested_fixation_dispersion_px(150.0):.0f}"
    assert f"suggests {expected} px" in out
    assert f"--fixation-dispersion {expected}" in out
    out.encode("ascii")            # console-safe on any Windows code page


def test_an_explicit_threshold_is_not_second_guessed(capsys):
    calibrate.report_fixation_settings(StubFixationPipeline(), 150.0, explicit=True)
    assert "suggests" not in capsys.readouterr().out


def test_cli_exposes_the_fixation_threshold():
    parser = calibrate.build_parser()
    assert "--fixation-dispersion" in parser.format_help()
    assert parser.parse_args([]).fixation_dispersion is None
    assert parser.parse_args(["--fixation-dispersion", "140"]
                             ).fixation_dispersion == 140.0


def test_wait_for_camera_returns_true_on_a_real_frame():
    pipeline = StubPipeline([sample(STATE_OK)])
    assert calibrate.wait_for_camera(pipeline, timeout=1.0)


def test_wait_for_camera_ignores_no_camera_samples():
    pipeline = StubPipeline([sample(STATE_NO_CAMERA)])
    assert not calibrate.wait_for_camera(pipeline, timeout=0.3)


def test_wait_for_camera_times_out_on_silence():
    assert not calibrate.wait_for_camera(StubPipeline([]), timeout=0.2)


def test_camera_failure_report_is_actionable(capsys):
    calibrate.report_camera_failure(StubPipeline([]), 0, 8.0)
    out = capsys.readouterr().out
    assert "camera 0 produced no frames within 8 s" in out
    assert "--camera 1" in out
    assert "camera 0 not available" in out


# ------------------------------------------------------------- persistence 5.4
def calibrated_session(camera_index: int) -> CalibrationSession:
    session = CalibrationSession(SCREEN, camera_index=camera_index,
                                 rng=random.Random(7))
    drive(session, SyntheticUser(seed=31))
    assert session.verdict == "PASS"
    return session


def test_saved_calibration_is_reloaded_for_the_same_screen_and_camera(
    tmp_path, monkeypatch, capsys
):
    path = str(tmp_path / "calibration.json")
    monkeypatch.setattr(calibrate, "calibration_path", lambda: path)

    session = calibrated_session(camera_index=1)
    session.save(path)

    model = calibrate.load_saved_model(SCREEN, 1)
    assert model is not None
    assert model.validation_error_px == pytest.approx(session.error_px)
    assert "found a saved calibration" in capsys.readouterr().out

    feature = session.cal_features[0]
    np.testing.assert_allclose(model.predict(feature), session.model.predict(feature))


def test_saved_calibration_is_refused_for_another_screen_or_camera(
    tmp_path, monkeypatch
):
    path = str(tmp_path / "calibration.json")
    monkeypatch.setattr(calibrate, "calibration_path", lambda: path)
    calibrated_session(camera_index=1).save(path)

    assert calibrate.load_saved_model((1280, 720), 1) is None
    assert calibrate.load_saved_model(SCREEN, 0) is None


def test_missing_calibration_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "calibration_path",
                        lambda: str(tmp_path / "nothing.json"))
    assert calibrate.load_saved_model(SCREEN, 0) is None
