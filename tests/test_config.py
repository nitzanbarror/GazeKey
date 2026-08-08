"""Config tests (spec Section 10) — defaults must survive missing/bad files."""
import json

import pytest

from config import (
    DEFAULTS,
    DISPERSION_MAX_PX,
    DISPERSION_MIN_PX,
    DISPERSION_RATIO,
    load_config,
    save_config,
    suggested_fixation_dispersion_px,
)


def test_missing_file_yields_defaults(tmp_path):
    assert load_config(str(tmp_path / "nope.json")) == DEFAULTS


def test_partial_file_is_filled_with_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"dwell_time_s": 1.4}), encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg["dwell_time_s"] == 1.4
    assert cfg["refractory_ms"] == DEFAULTS["refractory_ms"]


def test_corrupt_file_yields_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_config(str(path)) == DEFAULTS


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"launch_missiles": True}), encoding="utf-8")
    cfg = load_config(str(path))
    assert "launch_missiles" not in cfg


# ------------------------------------------------------ fixation thresholds
def test_default_dispersion_suits_the_measured_accuracy():
    """81 px measured on this setup -> the shipped default (spec Section 6)."""
    assert suggested_fixation_dispersion_px(81.0) == pytest.approx(
        DEFAULTS["fixation_dispersion_px"], abs=1.0
    )


def test_dispersion_scales_with_calibration_error():
    for error in (60.0, 81.0, 120.0):
        assert suggested_fixation_dispersion_px(error) == pytest.approx(
            round(DISPERSION_RATIO * error), abs=0.5
        )


def test_dispersion_stays_below_the_minimum_key_pitch():
    """Spec NFR-2 sizes keys at >= 2x the error; fixation must break between keys."""
    for error in (40.0, 81.0, 120.0, 150.0):
        assert suggested_fixation_dispersion_px(error) < 2.0 * error


def test_dispersion_is_clamped_to_a_sane_band():
    assert suggested_fixation_dispersion_px(5.0) == DISPERSION_MIN_PX
    assert suggested_fixation_dispersion_px(9_000.0) == DISPERSION_MAX_PX


@pytest.mark.parametrize("unknown", [0.0, None, float("nan"), float("inf")])
def test_unknown_accuracy_falls_back_to_the_default(unknown):
    assert suggested_fixation_dispersion_px(unknown) == float(
        DEFAULTS["fixation_dispersion_px"]
    )


def test_fixation_settings_are_part_of_the_config():
    for key in ("fixation_window_ms", "fixation_dispersion_px", "tracking_hold_ms"):
        assert key in DEFAULTS
    assert DEFAULTS["fixation_window_ms"] == 150      # spec Section 6
    assert DEFAULTS["tracking_hold_ms"] == 300        # spec Section 6


def test_save_load_round_trip(tmp_path):
    path = str(tmp_path / "sub" / "config.json")   # directory is created
    cfg = load_config(path)
    cfg["language"] = "he"
    cfg["show_gaze_cursor"] = False
    save_config(cfg, path)
    reloaded = load_config(path)
    assert reloaded["language"] == "he"
    assert reloaded["show_gaze_cursor"] is False
    assert set(reloaded) == set(DEFAULTS)
