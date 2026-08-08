"""Feature-extraction tests (spec Section 12) — synthetic landmarks, no camera.

The synthetic face is laid out in MediaPipe's normalised coordinate space with
the real index semantics:

* the user's **right** eye (33 outer / 133 inner) sits on the **image-left**,
* the user's **left** eye (362 inner / 263 outer) sits on the **image-right**.

``gaze`` in the helpers below is a signed shift of both irises along the
**image** x-axis (``+1`` = fully toward image-right), i.e. a conjugate gaze
sweep such as a real user makes when looking across the screen.
"""
import numpy as np
import pytest

from gaze.features import (
    EAR_BLINK_THRESHOLD,
    L_INNER,
    L_IRIS,
    L_LO,
    L_OUTER,
    L_UP,
    R_INNER,
    R_IRIS,
    R_LO,
    R_OUTER,
    R_UP,
    extract_features,
    eye_aspect_ratio,
)
from vision.face_tracker import eye_diagnostics

# eye geometry in normalised units
EYE_HALF_W = 0.05
EYE_HALF_H = 0.02
EYE_Y = 0.40
R_EYE_CX = 0.40   # user's right eye, image-left
L_EYE_CX = 0.60   # user's left eye,  image-right


def _set_iris(lm: np.ndarray, idx, cx: float, cy: float, r: float = 0.006) -> None:
    """Place a 5-point iris whose mean is exactly ``(cx, cy)``."""
    lm[idx[0], :2] = (cx, cy)
    lm[idx[1], :2] = (cx + r, cy)
    lm[idx[2], :2] = (cx - r, cy)
    lm[idx[3], :2] = (cx, cy + r)
    lm[idx[4], :2] = (cx, cy - r)


def make_face(gaze: float = 0.0, gaze_v: float = 0.0, openness: float = 1.0) -> np.ndarray:
    """Build a synthetic ``(478, 3)`` normalised landmark array.

    Args:
        gaze: iris shift along the image x-axis, ``-1..1``.
        gaze_v: iris shift along the image y-axis, ``-1..1`` (positive = down).
        openness: eyelid opening, ``1`` = wide open, ``0.15`` = blinking.
    """
    lm = np.zeros((478, 3), dtype=np.float64)
    lid = EYE_HALF_H * openness

    # eye corners (x ordering follows the real MediaPipe layout)
    lm[R_OUTER, :2] = (R_EYE_CX - EYE_HALF_W, EYE_Y)   # 33, image-left
    lm[R_INNER, :2] = (R_EYE_CX + EYE_HALF_W, EYE_Y)   # 133, toward the nose
    lm[L_INNER, :2] = (L_EYE_CX - EYE_HALF_W, EYE_Y)   # 362, toward the nose
    lm[L_OUTER, :2] = (L_EYE_CX + EYE_HALF_W, EYE_Y)   # 263, image-right

    # eyelids
    lm[R_UP, :2] = (R_EYE_CX, EYE_Y - lid)
    lm[R_LO, :2] = (R_EYE_CX, EYE_Y + lid)
    lm[L_UP, :2] = (L_EYE_CX, EYE_Y - lid)
    lm[L_LO, :2] = (L_EYE_CX, EYE_Y + lid)

    dx = gaze * 0.8 * EYE_HALF_W
    dy = gaze_v * 0.8 * EYE_HALF_H
    _set_iris(lm, R_IRIS, R_EYE_CX + dx, EYE_Y + dy)
    _set_iris(lm, L_IRIS, L_EYE_CX + dx, EYE_Y + dy)

    # a few non-eye landmarks so head-pose code has something to chew on
    lm[1, :2] = (0.50, 0.55)      # nose tip
    lm[152, :2] = (0.50, 0.80)    # chin
    lm[61, :2] = (0.43, 0.70)     # right mouth corner
    lm[291, :2] = (0.57, 0.70)    # left mouth corner
    return lm


# ------------------------------------------------------------------- basic math
def test_centered_iris_gives_half_ratios():
    f = extract_features(make_face(), yaw=0.0, pitch=0.0, roll=0.0, timestamp=1.5)
    assert f.valid
    assert f.hx == pytest.approx(0.5, abs=1e-9)
    assert f.hy == pytest.approx(0.5, abs=1e-9)
    assert f.timestamp == 1.5


def test_head_pose_is_passed_through_untouched():
    f = extract_features(make_face(), yaw=-7.5, pitch=3.25, roll=1.0)
    assert (f.yaw, f.pitch, f.roll) == (-7.5, 3.25, 1.0)
    assert f.vector().tolist() == pytest.approx([f.hx, f.hy, -7.5, 3.25])


def test_vertical_gaze_moves_hy_monotonically():
    hy = [extract_features(make_face(gaze_v=g), 0.0, 0.0, 0.0).hy
          for g in np.linspace(-1.0, 1.0, 9)]
    assert all(b > a for a, b in zip(hy, hy[1:])), hy
    assert min(hy) > 0.0 and max(hy) < 1.0


# ------------------------------------------------------------------ blink gating
def test_eye_aspect_ratio_open_vs_closed():
    open_lm, shut_lm = make_face(), make_face(openness=0.1)
    ear_open = eye_aspect_ratio(open_lm, R_UP, R_LO, R_INNER, R_OUTER)
    ear_shut = eye_aspect_ratio(shut_lm, R_UP, R_LO, R_INNER, R_OUTER)
    assert ear_open > EAR_BLINK_THRESHOLD > ear_shut


def test_blink_marks_frame_invalid():
    f = extract_features(make_face(openness=0.1), 0.0, 0.0, 0.0, timestamp=2.0)
    assert not f.valid
    assert np.isnan(f.hx) and np.isnan(f.hy)
    assert f.timestamp == 2.0


def test_half_open_eye_still_valid():
    assert extract_features(make_face(openness=0.6), 0.0, 0.0, 0.0).valid


# --------------------------------------------------------------- missing inputs
def test_no_face_is_invalid_with_nans():
    f = extract_features(None, np.nan, np.nan, np.nan, timestamp=3.0)
    assert not f.valid
    assert all(np.isnan(v) for v in (f.hx, f.hy, f.yaw, f.pitch, f.roll))
    assert f.timestamp == 3.0


def test_landmark_array_without_iris_points_is_invalid():
    """A 468-point mesh (refine_landmarks off) carries no iris — reject it."""
    assert not extract_features(np.zeros((468, 3)), 0.0, 0.0, 0.0).valid


def test_degenerate_eye_geometry_is_invalid():
    lm = make_face()
    lm[R_OUTER, :2] = lm[R_INNER, :2]   # zero-width eye
    assert not extract_features(lm, 0.0, 0.0, 0.0).valid


# ------------------------------------------------------- horizontal gaze sweep
def test_horizontal_gaze_sweep_moves_hx_monotonically():
    """The M1 checkpoint: hx must sweep as the user looks left → right.

    Regression guard. Both eyes are measured along the **image** x-axis, so a
    conjugate sweep moves them together. Measuring each eye along its own
    nasal→temporal axis instead makes one ratio rise while the other falls,
    and averaging them cancels the horizontal signal almost exactly — the
    defect this test was written for.
    """
    hx = [extract_features(make_face(gaze=g), 0.0, 0.0, 0.0).hx
          for g in np.linspace(-1.0, 1.0, 9)]
    assert all(b > a for a, b in zip(hx, hx[1:])), hx
    assert min(hx) > 0.0 and max(hx) < 1.0
    assert max(hx) - min(hx) > 0.5, f"horizontal signal too weak: {hx}"


def test_per_eye_horizontal_ratios_move_together():
    """Both eyes must respond to a conjugate sweep in the same direction."""
    left_gaze = eye_diagnostics(make_face(gaze=-0.8))
    right_gaze = eye_diagnostics(make_face(gaze=+0.8))
    assert right_gaze.hx_right > left_gaze.hx_right
    assert right_gaze.hx_left > left_gaze.hx_left


def test_hx_agrees_with_the_image_axis_ratio():
    """The independent image-axis diagnostic must track hx."""
    for g in np.linspace(-1.0, 1.0, 9):
        face = make_face(gaze=g)
        assert extract_features(face, 0.0, 0.0, 0.0).hx == pytest.approx(
            eye_diagnostics(face).hx_image, abs=1e-9
        )


def test_image_axis_ratio_sweeps_monotonically():
    hx = [eye_diagnostics(make_face(gaze=g)).hx_image
          for g in np.linspace(-1.0, 1.0, 9)]
    assert all(b > a for a, b in zip(hx, hx[1:])), hx
    assert min(hx) > 0.0 and max(hx) < 1.0


def test_diagnostics_report_ear_per_eye():
    d = eye_diagnostics(make_face(openness=0.1))
    assert d.ear_right < EAR_BLINK_THRESHOLD
    assert d.ear_left < EAR_BLINK_THRESHOLD
