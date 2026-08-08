"""Tests proving the calibration pipeline works end-to-end on a realistic
simulated eye. Run: pytest -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaze.calibration import (CalibrationModel, aggregate_point, grid_points,
                              validation_points, PASS_PX)
from gaze.smoothing import OneEuroFilter, FixationDetector

W, H = 1920, 1080
rng = np.random.default_rng(42)


# ---------------- realistic simulated eye ----------------
def natural_pose_for_target(px, py):
    """Real users rotate the head slightly toward what they look at
    (~40% head / 60% eyes). This puts pose variation INTO the calibration
    data, which is what lets the model learn pose compensation."""
    nx, ny = px / W - 0.5, py / H - 0.5
    return 12.0 * nx, 8.0 * ny   # yaw, pitch in degrees


def true_features_for_screen_point(px, py, yaw, pitch):
    """Inverse ground-truth model: where the iris ratios sit when a real user
    looks at (px,py) with head pose (yaw,pitch). Non-linear in gaze and
    coupled with pose, like a real eye."""
    nx, ny = px / W - 0.5, py / H - 0.5           # -0.5..0.5
    # eyes cover whatever the head doesn't
    hx = 0.5 + 0.32 * nx + 0.08 * nx * ny + 0.06 * nx * nx * nx - 0.011 * yaw
    hy = 0.5 + 0.22 * ny + 0.06 * nx * nx + 0.04 * ny * ny - 0.009 * pitch
    return np.array([hx, hy, yaw, pitch])


def noisy_samples(px, py, yaw, pitch, n=30, noise=0.004, outlier_frac=0.15):
    """n frames while user stares at (px,py): gaussian sensor noise + a
    fraction of gross outliers (blinks half-open, glances away)."""
    base = true_features_for_screen_point(px, py, yaw, pitch)
    s = np.tile(base, (n, 1))
    s[:, :2] += rng.normal(0, noise, size=(n, 2))
    s[:, 2:] += rng.normal(0, 0.3, size=(n, 2))
    n_out = int(n * outlier_frac)
    idx = rng.choice(n, n_out, replace=False)
    s[idx, 0] += rng.choice([-1, 1], n_out) * rng.uniform(0.05, 0.15, n_out)
    s[idx, 1] += rng.choice([-1, 1], n_out) * rng.uniform(0.05, 0.15, n_out)
    return s


def simulate_head_sweep(n=90):
    """User stares at screen center while slowly rotating head +-12 deg."""
    cx, cy = W / 2, H / 2
    t = np.linspace(0, 1, n)
    yaws = 12 * np.sin(2 * np.pi * 1.5 * t)
    pitches = 8 * np.sin(2 * np.pi * 1.0 * t + 1.3)
    rows = []
    for yaw, pitch in zip(yaws, pitches):
        f = true_features_for_screen_point(cx, cy, yaw, pitch)
        f = f + np.array([rng.normal(0, 0.004), rng.normal(0, 0.004),
                          rng.normal(0, 0.3), rng.normal(0, 0.3)])
        rows.append(f)
    return np.array(rows)


def run_calibration():
    feats, targs = [], []
    for (px, py) in grid_points(W, H):
        yaw, pitch = natural_pose_for_target(px, py)
        agg, n = aggregate_point(noisy_samples(px, py, yaw, pitch))
        assert agg is not None, "point should survive outlier rejection"
        feats.append(agg); targs.append([px, py])
    m = CalibrationModel(screen_size=(W, H))
    ok = m.fit_pose_compensation(simulate_head_sweep())
    assert ok, "head-sweep step must be accepted"
    m.fit(np.array(feats), np.array(targs, dtype=float))
    return m


# ---------------- tests ----------------
def test_outlier_rejection_keeps_enough_samples():
    s = noisy_samples(960, 540, 0, 0)
    agg, n = aggregate_point(s)
    assert agg is not None and n >= 15


def test_calibration_passes_validation_gate():
    m = run_calibration()
    vf, vt = [], []
    for (px, py) in validation_points(W, H):
        yaw, pitch = natural_pose_for_target(px, py)
        agg, _ = aggregate_point(noisy_samples(px, py, yaw, pitch))
        vf.append(agg); vt.append([px, py])
    err, verdict = m.validate(np.array(vf), np.array(vt, dtype=float))
    print(f"\n  validation error = {err:.1f} px  ({verdict})")
    assert verdict == "PASS", f"error {err:.1f}px > {PASS_PX}px"


def test_accuracy_across_whole_screen():
    m = run_calibration()
    errs = []
    for px in np.linspace(0.15 * W, 0.85 * W, 8):
        for py in np.linspace(0.15 * H, 0.85 * H, 5):
            yaw, pitch = natural_pose_for_target(px, py)
            f = true_features_for_screen_point(px, py, yaw, pitch)
            pred = m.predict(f)
            errs.append(np.hypot(pred[0] - px, pred[1] - py))
    mean_err = float(np.mean(errs)); max_err = float(np.max(errs))
    print(f"\n  whole-screen mean={mean_err:.1f}px  max={max_err:.1f}px")
    assert mean_err < 40 and max_err < 90


def test_head_movement_tolerance():
    """After calibration, the user shifts posture: an EXTRA rotation of up to
    ±8° on top of the natural target-following pose. The learned pose terms
    must absorb most of it (spec NFR-4)."""
    m = run_calibration()
    for dyaw, dpitch in [(6, 0), (-6, 0), (0, 5), (0, -5), (8, 4), (-8, -4)]:
        errs = []
        for (px, py) in validation_points(W, H):
            yaw, pitch = natural_pose_for_target(px, py)
            f = true_features_for_screen_point(px, py, yaw + dyaw, pitch + dpitch)
            pred = m.predict(f)
            errs.append(np.hypot(pred[0] - px, pred[1] - py))
        e = float(np.mean(errs))
        print(f"\n  extra pose dyaw={dyaw:+.0f} dpitch={dpitch:+.0f} -> err {e:.1f}px")
        assert e < PASS_PX, f"pose delta ({dyaw},{dpitch}) error {e:.1f}px"


def test_linear_model_would_fail_but_quadratic_passes():
    """Sanity: proves the quadratic terms matter on this non-linear eye."""
    feats, targs = [], []
    for (px, py) in grid_points(W, H):
        yaw, pitch = natural_pose_for_target(px, py)
        agg, _ = aggregate_point(noisy_samples(px, py, yaw, pitch))
        feats.append(agg); targs.append([px, py])
    F, T = np.array(feats), np.array(targs, dtype=float)
    # linear fit (bias + hx + hy only)
    A = np.hstack([np.ones((9, 1)), F[:, :2]])
    wx, *_ = np.linalg.lstsq(A, T[:, 0], rcond=None)
    wy, *_ = np.linalg.lstsq(A, T[:, 1], rcond=None)
    quad = run_calibration()
    lin_errs, quad_errs = [], []
    for px in np.linspace(0.15 * W, 0.85 * W, 6):
        for py in np.linspace(0.15 * H, 0.85 * H, 4):
            yaw, pitch = natural_pose_for_target(px, py)
            f = true_features_for_screen_point(px, py, yaw, pitch)
            row = np.array([1, f[0], f[1]])
            lin_errs.append(np.hypot(row @ wx - px, row @ wy - py))
            qp = quad.predict(f)
            quad_errs.append(np.hypot(qp[0] - px, qp[1] - py))
    print(f"\n  linear(old)={np.mean(lin_errs):.1f}px  quadratic(new)={np.mean(quad_errs):.1f}px")
    assert np.mean(quad_errs) < np.mean(lin_errs) * 0.7  # clearly better


def test_drift_touch_up():
    m = run_calibration()
    m.apply_offset(-60.0, 35.0)  # simulate drift correction of a +60,-35 shift
    yaw, pitch = natural_pose_for_target(960, 540)
    f = true_features_for_screen_point(960, 540, yaw, pitch)
    x, y = m.predict(f)
    assert abs((x + 60) - 960) < 100  # offset applied to predictions


def test_persistence_roundtrip(tmp_path):
    m = run_calibration()
    m.validation_error_px = 42.0
    p = str(tmp_path / "cal.json")
    m.save(p)
    m2 = CalibrationModel.load(p, (W, H), 0)
    assert m2 is not None
    yaw, pitch = natural_pose_for_target(500, 500)
    f = true_features_for_screen_point(500, 500, yaw, pitch)
    assert np.allclose(m.predict(f), m2.predict(f))
    # incompatible screen -> refuse to load
    assert CalibrationModel.load(p, (1280, 720), 0) is None


def test_one_euro_smooths_jitter_keeps_saccades():
    fx, fy = OneEuroFilter(), OneEuroFilter()
    t = 0.0
    # jitter around a point
    outs = []
    for i in range(60):
        t += 1 / 30
        outs.append(fx.apply(500 + rng.normal(0, 15), t))
    jitter_std = np.std(outs[20:])
    assert jitter_std < 8, f"smoothed jitter std {jitter_std:.1f}"
    # saccade: jump to 1500 — must arrive fast
    for i in range(6):
        t += 1 / 30
        last = fx.apply(1500, t)
    assert abs(last - 1500) < 120, f"saccade lag too big: {1500 - last:.0f}px"


def test_fixation_detector():
    d = FixationDetector()
    t = 0.0
    fix = False
    for i in range(15):
        t += 1 / 30
        fix = d.update(500 + rng.normal(0, 8), 500 + rng.normal(0, 8), t)
    assert fix, "stable gaze must be classified as fixation"
    t += 1 / 30
    assert not d.update(1200, 300, t), "saccade must break fixation"


def test_head_sweep_rejected_if_user_does_not_move():
    """If the user keeps the head still during the sweep, the step must be
    detected as invalid so the UI can ask them to repeat it."""
    from gaze.calibration import CalibrationModel
    cx, cy = W / 2, H / 2
    rows = [true_features_for_screen_point(cx, cy, 0.3, -0.2) +
            rng.normal(0, 0.003, 4) for _ in range(90)]
    m = CalibrationModel(screen_size=(W, H))
    assert m.fit_pose_compensation(np.array(rows)) is False
