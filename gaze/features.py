"""GazeKey — feature extraction from MediaPipe Face Mesh landmarks.

VERIFIED REFERENCE IMPLEMENTATION — use as-is, do not rewrite.
Landmarks arrive as a (478, 3) array of normalized [x, y, z] coords
from MediaPipe FaceMesh with refine_landmarks=True.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# MediaPipe landmark indices (refine_landmarks=True)
R_IRIS = [468, 469, 470, 471, 472]   # right eye iris (user's right = image left)
L_IRIS = [473, 474, 475, 476, 477]
R_INNER, R_OUTER = 133, 33           # right-eye corners
L_INNER, L_OUTER = 362, 263
R_UP, R_LO = 159, 145                # right-eye lids
L_UP, L_LO = 386, 374

EAR_BLINK_THRESHOLD = 0.18


@dataclass
class FrameFeatures:
    valid: bool
    hx: float = float("nan")
    hy: float = float("nan")
    yaw: float = float("nan")
    pitch: float = float("nan")
    roll: float = float("nan")
    timestamp: float = 0.0

    def vector(self) -> np.ndarray:
        """Feature vector fed to the calibration model."""
        return np.array([self.hx, self.hy, self.yaw, self.pitch], dtype=np.float64)


def eye_aspect_ratio(lm: np.ndarray, up: int, lo: int, inner: int, outer: int) -> float:
    v = np.linalg.norm(lm[up, :2] - lm[lo, :2])
    h = np.linalg.norm(lm[inner, :2] - lm[outer, :2])
    return float(v / h) if h > 1e-9 else 0.0


def _eye_ratios(lm: np.ndarray, iris_idx, inner, outer, up, lo):
    iris = lm[iris_idx, :2].mean(axis=0)
    a, b = lm[inner, :2], lm[outer, :2]
    hx_den = b[0] - a[0]
    hy_den = lm[lo, 1] - lm[up, 1]
    if abs(hx_den) < 1e-9 or abs(hy_den) < 1e-9:
        return None
    hx = (iris[0] - a[0]) / hx_den
    hy = (iris[1] - lm[up, 1]) / hy_den
    return hx, hy


def extract_features(landmarks: np.ndarray, yaw: float, pitch: float, roll: float,
                     timestamp: float = 0.0) -> FrameFeatures:
    """landmarks: (478,3) normalized; yaw/pitch/roll in degrees from solvePnP."""
    if landmarks is None or landmarks.shape[0] < 478:
        return FrameFeatures(valid=False, timestamp=timestamp)

    ear_r = eye_aspect_ratio(landmarks, R_UP, R_LO, R_INNER, R_OUTER)
    ear_l = eye_aspect_ratio(landmarks, L_UP, L_LO, L_INNER, L_OUTER)
    if min(ear_r, ear_l) < EAR_BLINK_THRESHOLD:
        return FrameFeatures(valid=False, timestamp=timestamp)

    r = _eye_ratios(landmarks, R_IRIS, R_OUTER, R_INNER, R_UP, R_LO)
    l = _eye_ratios(landmarks, L_IRIS, L_INNER, L_OUTER, L_UP, L_LO)
    if r is None or l is None:
        return FrameFeatures(valid=False, timestamp=timestamp)

    hx = (r[0] + l[0]) / 2.0
    hy = (r[1] + l[1]) / 2.0
    return FrameFeatures(True, hx, hy, yaw, pitch, roll, timestamp)
