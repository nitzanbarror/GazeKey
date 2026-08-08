"""GazeKey — user configuration (spec Section 10).

Config lives in ``~/.gazekey/config.json``. Missing keys fall back to
:data:`DEFAULTS`, so an old or partial file never breaks startup.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

APP_DIR_NAME = ".gazekey"

DEFAULTS: Dict[str, Any] = {
    "dwell_time_s": 1.0,
    "extended_dwell_s": 2.0,
    "hysteresis_margin": 0.25,
    "grace_period_ms": 200,
    "refractory_ms": 400,
    "language": "en",
    "sound_feedback": True,
    "show_gaze_cursor": True,
    "camera_index": 0,
    "keyboard_height_ratio": 2.0 / 3.0,
    #: small live self-view in the keyboard, so the user can see they are tracked
    "show_webcam_preview": True,
    # True when the user's head rests on a physical support (the common case
    # for the target users): pose is constant, so the head-sweep stage is
    # skipped and pose-compensation sensitivities stay at zero.
    "fixed_head": True,
    # the 5-second setup check before the nine dots (spec 5.1c): how far the
    # vertical iris ratio must travel between the top and bottom of the
    # calibration region before calibrating is worth the user's time. Measured
    # sittings: 0.051 good, 0.024 with the camera below eye height.
    "setup_check_min_hy_span": 0.035,
    # calibration pacing — how long each target waits before collecting, how
    # many valid samples it wants, and the wall-clock cap per target
    "calibration_settle_ms": 700,
    "calibration_collect_samples": 45,
    "calibration_collect_max_s": 4.0,
    # fixation detection (spec Section 6) — see DISPERSION_RATIO below
    "fixation_window_ms": 150,
    "fixation_dispersion_px": 110,
    # One-Euro smoothing: lower min_cutoff = smoother but laggier. The knob for
    # gaze jitter that keeps crossing key boundaries mid-dwell.
    "one_euro_min_cutoff": 1.0,
    # how long a blink or a lost face keeps the last gaze position alive
    "tracking_hold_ms": 300,
    # drift monitor (spec 5.5): how large an offset between a key's centre and
    # the gaze that activated it counts as one full unit of drift evidence.
    # 0 = derive it from the measured calibration error, which is what an
    # offset has to reach before it starts costing keystrokes.
    "drift_offset_px": 0,
}

#: Fixation dispersion as a multiple of the measured calibration error.
#:
#: The verified I-DT detector thresholds ``Δx + Δy`` — the *sum* of the two
#: per-axis ranges inside the window — so the budget is shared by both axes.
#: Residual wander while genuinely staring scales with calibration accuracy, so
#: the threshold is derived from the measured validation error rather than
#: fixed. The ratio has to sit between two walls:
#:
#: * too low  → a real stare keeps breaking, dwell never completes;
#: * too high → a saccade across the screen reads as a fixation and keys fire
#:   in passing. Spec NFR-2 sizes keys at ≥ 2x the expected error, so staying
#:   under 2x guarantees a deliberate move to a neighbouring key always breaks
#:   fixation.
#:
#: 1.35 sits comfortably between them. At the 81 px measured on this setup that
#: gives ~110 px, against a minimum key pitch of ~162 px.
DISPERSION_RATIO = 1.35
DISPERSION_MIN_PX = 60.0
DISPERSION_MAX_PX = 220.0


def suggested_fixation_dispersion_px(validation_error_px: float) -> float:
    """Dispersion threshold that suits a given calibration accuracy.

    Returns the clamped :data:`DISPERSION_RATIO` multiple of the error, or the
    configured default when the error is unknown.
    """
    if not validation_error_px or validation_error_px != validation_error_px:
        return float(DEFAULTS["fixation_dispersion_px"])   # NaN / zero / None
    if validation_error_px == float("inf"):
        return float(DEFAULTS["fixation_dispersion_px"])
    scaled = DISPERSION_RATIO * float(validation_error_px)
    return float(min(DISPERSION_MAX_PX, max(DISPERSION_MIN_PX, round(scaled))))

#: The knobs that govern typing stability, and where each is read from.
#: ``(config key, argparse attribute)`` — see :func:`resolve_tuning`.
TUNING_KNOBS = {
    "hysteresis_margin": ("hysteresis_margin", "hysteresis"),
    "grace_ms": ("grace_period_ms", "grace_ms"),
    "min_cutoff": ("one_euro_min_cutoff", "min_cutoff"),
}


def resolve_tuning(config: Dict[str, Any], args: Any = None) -> Dict[str, float]:
    """Merge the stability knobs: an explicit flag beats config beats default.

    A flag is "absent" when it is ``None``, so ``--hysteresis 0`` is a real
    request for zero rather than a fallback to config.
    """
    resolved: Dict[str, float] = {}
    for name, (config_key, flag) in TUNING_KNOBS.items():
        value = config.get(config_key, DEFAULTS[config_key])
        override = getattr(args, flag, None) if args is not None else None
        resolved[name] = float(override if override is not None else value)
    return resolved


#: gentler pacing for first-time users (``calibrate.py --slow``)
SLOW_PRESET = {
    "calibration_settle_ms": 1000,
    "calibration_collect_samples": 60,
}


def app_dir() -> str:
    """Directory holding config, calibration and downloaded models."""
    return os.path.join(os.path.expanduser("~"), APP_DIR_NAME)


def config_path() -> str:
    return os.path.join(app_dir(), "config.json")


def calibration_path() -> str:
    return os.path.join(app_dir(), "calibration.json")


def models_dir() -> str:
    return os.path.join(app_dir(), "models")


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load config, filling in defaults for anything missing or unreadable."""
    path = path or config_path()
    cfg = dict(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            stored = json.load(fh)
        if isinstance(stored, dict):
            cfg.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except (OSError, json.JSONDecodeError):
        pass  # first run, or a corrupt file — defaults are fine
    return cfg


def save_config(cfg: Dict[str, Any], path: str | None = None) -> None:
    """Persist config (only known keys) to disk."""
    path = path or config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in cfg.items() if k in DEFAULTS})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
