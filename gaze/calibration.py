"""GazeKey — calibration model. VERIFIED REFERENCE IMPLEMENTATION — use as-is.

Two-stage calibration (both stages are mandatory):

STAGE A — Head-sweep pose compensation (~5 s):
  The user fixates a single CENTER dot while slowly rotating the head
  left/right and up/down. Because gaze is FIXED while pose VARIES, this
  isolates the pose sensitivity of the iris ratios:
      s = d(iris ratio)/d(degree of yaw or pitch)
  Without this step, pose and gaze are confounded (users turn their head
  toward calibration targets), and NO regression can learn compensation —
  this is the root cause of drift with head movement.

STAGE B — 9-point gaze mapping:
  Pose-COMPENSATED iris ratios (hx_c, hy_c) are mapped to screen pixels by a
  ridge-regularized 2nd-order polynomial per axis:
      X = a0 + a1*hx_c + a2*hy_c + a3*hx_c^2 + a4*hy_c^2 + a5*hx_c*hy_c
"""
from __future__ import annotations
import json, os, time
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

RIDGE_LAMBDA = 1e-3
STD_FLOOR_RATIO = 0.02      # floor for hx/hy std in standardization
Z_CLAMP = 3.0               # clamp standardized features (no wild extrapolation)
PASS_PX = 80.0
MARGINAL_PX = 130.0
MIN_SAMPLES_PER_POINT = 15
MIN_SWEEP_SAMPLES = 40
MIN_SWEEP_POSE_RANGE_DEG = 6.0  # sweep must cover at least this much rotation


def _design_row(hx: float, hy: float) -> np.ndarray:
    return np.array([1.0, hx, hy, hx * hx, hy * hy, hx * hy])


def reject_outliers(samples: np.ndarray) -> np.ndarray:
    """samples: (n, 4) = [hx, hy, yaw, pitch] for ONE calibration point.
    Keep rows within 1.5*IQR of the median on hx and hy."""
    keep = np.ones(len(samples), dtype=bool)
    for col in (0, 1):
        x = samples[:, col]
        q1, q3 = np.percentile(x, [25, 75])
        iqr = max(q3 - q1, 1e-9)
        keep &= np.abs(x - np.median(x)) <= 1.5 * iqr + 1e-12
    return samples[keep]


def aggregate_point(samples: np.ndarray) -> Tuple[np.ndarray | None, int]:
    clean = reject_outliers(samples)
    if len(clean) < MIN_SAMPLES_PER_POINT:
        return None, len(clean)
    return np.median(clean, axis=0), len(clean)


@dataclass
class PoseCompensator:
    """Learned from the head-sweep. Removes head-pose effect from iris ratios."""
    # sensitivities: d(hx)/d(yaw), d(hx)/d(pitch), d(hy)/d(yaw), d(hy)/d(pitch)
    s: np.ndarray = field(default_factory=lambda: np.zeros(4))
    ref_yaw: float = 0.0
    ref_pitch: float = 0.0
    ok: bool = False

    def fit(self, sweep: np.ndarray) -> bool:
        """sweep: (n,4) valid frames of the head-sweep (gaze fixed at center).
        Least-squares of ratio deviation against pose deviation."""
        if len(sweep) < MIN_SWEEP_SAMPLES:
            return False
        yaw, pitch = sweep[:, 2], sweep[:, 3]
        if (yaw.max() - yaw.min()) < MIN_SWEEP_POSE_RANGE_DEG and \
           (pitch.max() - pitch.min()) < MIN_SWEEP_POSE_RANGE_DEG:
            return False  # user didn't actually move the head
        self.ref_yaw, self.ref_pitch = float(np.median(yaw)), float(np.median(pitch))
        P = np.column_stack([yaw - self.ref_yaw, pitch - self.ref_pitch])
        reg = 1e-6 * np.eye(2)
        PtP = P.T @ P + reg
        for i, col in enumerate((0, 1)):        # hx, hy
            d = sweep[:, col] - np.median(sweep[:, col])
            coef = np.linalg.solve(PtP, P.T @ d)
            self.s[2 * i:2 * i + 2] = coef      # [d/dyaw, d/dpitch]
        self.ok = True
        return True

    def compensate(self, f: np.ndarray) -> Tuple[float, float]:
        """f = [hx, hy, yaw, pitch] -> pose-corrected (hx_c, hy_c)."""
        hx, hy, yaw, pitch = f
        dy, dp = yaw - self.ref_yaw, pitch - self.ref_pitch
        hx_c = hx - (self.s[0] * dy + self.s[1] * dp)
        hy_c = hy - (self.s[2] * dy + self.s[3] * dp)
        return hx_c, hy_c


@dataclass
class CalibrationModel:
    comp: PoseCompensator = field(default_factory=PoseCompensator)
    wx: np.ndarray = field(default_factory=lambda: np.zeros(6))
    wy: np.ndarray = field(default_factory=lambda: np.zeros(6))
    mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    std: np.ndarray = field(default_factory=lambda: np.ones(2))
    validation_error_px: float = float("inf")
    screen_size: Tuple[int, int] = (0, 0)
    camera_index: int = 0
    created_at: float = 0.0

    # ---------- stage A ----------
    def fit_pose_compensation(self, sweep_samples: np.ndarray) -> bool:
        """sweep_samples: (n,4) valid frames from the head-sweep step.
        Returns False if the sweep was insufficient (repeat the step)."""
        return self.comp.fit(sweep_samples)

    # ---------- stage B ----------
    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """features: (n,4) aggregated per-point [hx,hy,yaw,pitch];
        targets: (n,2) screen px. Pose compensation applied internally."""
        C = np.array([self.comp.compensate(f) for f in features])
        self.mean = C.mean(axis=0)
        self.std = np.maximum(C.std(axis=0), STD_FLOOR_RATIO)
        Z = np.clip((C - self.mean) / self.std, -Z_CLAMP, Z_CLAMP)
        A = np.vstack([_design_row(hx, hy) for hx, hy in Z])
        reg = RIDGE_LAMBDA * np.eye(A.shape[1]); reg[0, 0] = 0.0
        AtA = A.T @ A + reg
        self.wx = np.linalg.solve(AtA, A.T @ targets[:, 0])
        self.wy = np.linalg.solve(AtA, A.T @ targets[:, 1])

    def predict(self, feature: np.ndarray) -> Tuple[float, float]:
        hx_c, hy_c = self.comp.compensate(feature)
        z = np.clip((np.array([hx_c, hy_c]) - self.mean) / self.std,
                    -Z_CLAMP, Z_CLAMP)
        row = _design_row(z[0], z[1])
        return float(row @ self.wx), float(row @ self.wy)

    def predict_many(self, features: np.ndarray) -> np.ndarray:
        return np.array([self.predict(f) for f in features])

    # ---------- validation ----------
    def validate(self, features: np.ndarray, targets: np.ndarray) -> Tuple[float, str]:
        pred = self.predict_many(features)
        err = float(np.mean(np.linalg.norm(pred - targets, axis=1)))
        self.validation_error_px = err
        verdict = "PASS" if err <= PASS_PX else ("MARGINAL" if err <= MARGINAL_PX else "FAIL")
        return err, verdict

    def worst_fit_points(self, features: np.ndarray, targets: np.ndarray, k: int = 2):
        pred = self.predict_many(features)
        res = np.linalg.norm(pred - targets, axis=1)
        return list(np.argsort(res)[-k:])

    # ---------- drift touch-up ----------
    def apply_offset(self, dx: float, dy: float) -> None:
        self.wx[0] += dx
        self.wy[0] += dy

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "wx": self.wx.tolist(), "wy": self.wy.tolist(),
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "comp_s": self.comp.s.tolist(),
                "comp_ref": [self.comp.ref_yaw, self.comp.ref_pitch],
                "comp_ok": self.comp.ok,
                "validation_error_px": self.validation_error_px,
                "screen_size": list(self.screen_size),
                "camera_index": self.camera_index,
                "created_at": self.created_at or time.time(),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str, screen_size: Tuple[int, int], camera_index: int):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        if tuple(d["screen_size"]) != tuple(screen_size) or d["camera_index"] != camera_index:
            return None
        comp = PoseCompensator(np.array(d["comp_s"]), d["comp_ref"][0],
                               d["comp_ref"][1], d["comp_ok"])
        return cls(comp, np.array(d["wx"]), np.array(d["wy"]),
                   np.array(d["mean"]), np.array(d["std"]),
                   d["validation_error_px"], tuple(d["screen_size"]),
                   d["camera_index"], d["created_at"])


def grid_points(w: int, h: int) -> List[Tuple[int, int]]:
    xs, ys = [0.1, 0.5, 0.9], [0.1, 0.5, 0.9]
    return [(int(x * w), int(y * h)) for y in ys for x in xs]


def validation_points(w: int, h: int) -> List[Tuple[int, int]]:
    return [(int(0.3 * w), int(0.3 * h)),
            (int(0.7 * w), int(0.7 * h)),
            (int(0.7 * w), int(0.3 * h))]
