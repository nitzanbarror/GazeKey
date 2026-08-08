"""GazeKey — head pose from a rigid 6-point face subset (spec Section 4).

``cv2.solvePnP`` is run on six landmarks that barely move with expression
(nose tip, chin, both outer eye corners, both mouth corners) against a
canonical 3D face model. The resulting rotation is decomposed into
yaw / pitch / roll in degrees.

Coordinate frames
-----------------
The 3D model is expressed in the same axis convention as the image:
``+X`` right, ``+Y`` down, ``+Z`` away from the camera. A person looking
straight into the camera therefore has a rotation matrix close to identity.

Angle conventions (documented so the calibration data stays consistent):

* ``yaw``   > 0 — the user turns their head to **their own right**
  (in an un-mirrored camera image the face points toward image-left).
* ``pitch`` > 0 — the user tips their head **up** (chin toward the camera).
* ``roll``  > 0 — the user tilts their head toward their **left** shoulder.

The absolute signs do not matter to the calibration fit (it learns a linear
sensitivity either way) but they must never change between calibration and
runtime, so they are pinned by ``tests/test_head_pose.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

#: MediaPipe indices of the rigid subset, in the same order as MODEL_POINTS.
POSE_LANDMARK_IDS: Tuple[int, ...] = (1, 152, 33, 263, 61, 291)

#: Canonical face model in millimetres (X right, Y down, Z away from camera).
MODEL_POINTS: np.ndarray = np.array(
    [
        (0.0, 0.0, 0.0),        # 1   nose tip
        (0.0, 63.6, 12.5),      # 152 chin
        (-43.3, -32.7, 26.0),   # 33  right eye, outer corner (image-left)
        (43.3, -32.7, 26.0),    # 263 left eye, outer corner  (image-right)
        (-28.9, 28.9, 24.1),    # 61  right mouth corner      (image-left)
        (28.9, 28.9, 24.1),     # 291 left mouth corner       (image-right)
    ],
    dtype=np.float64,
)

_SOLVER = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)


@dataclass
class HeadPose:
    """Head rotation in degrees plus the raw solvePnP extrinsics."""

    yaw: float
    pitch: float
    roll: float
    rvec: np.ndarray
    tvec: np.ndarray


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics approximated from the frame size (f ≈ image width)."""
    f = float(width)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def euler_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of :func:`rotation_matrix_to_euler` (angles in degrees)."""
    x, y, z = math.radians(-pitch), math.radians(yaw), math.radians(roll)
    rx = np.array([[1, 0, 0],
                   [0, math.cos(x), -math.sin(x)],
                   [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0],
                   [math.sin(z), math.cos(z), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def rotation_matrix_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Decompose ``R = Rz(roll)·Ry(yaw)·Rx(-pitch)`` into degrees.

    Returns:
        ``(yaw, pitch, roll)`` following the conventions in the module docstring.
    """
    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    if sy > 1e-6:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:  # gimbal lock — roll and yaw are degenerate
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0.0
    return math.degrees(y), math.degrees(-x), math.degrees(z)


class HeadPoseEstimator:
    """Stateless solvePnP head-pose estimator.

    Stateless on purpose: SQPnP is a globally optimal solver, so seeding it
    from the previous frame would only risk locking onto a stale pose.
    """

    def __init__(self) -> None:
        self._cam: np.ndarray | None = None
        self._cam_size: Tuple[int, int] | None = None
        self._dist = np.zeros((4, 1), dtype=np.float64)

    def _intrinsics(self, width: int, height: int) -> np.ndarray:
        if self._cam_size != (width, height):
            self._cam = camera_matrix(width, height)
            self._cam_size = (width, height)
        assert self._cam is not None
        return self._cam

    def estimate(
        self,
        landmarks_px: np.ndarray,
        image_size: Tuple[int, int],
        ids: Sequence[int] = POSE_LANDMARK_IDS,
    ) -> HeadPose | None:
        """Estimate head pose from pixel-space landmarks.

        Args:
            landmarks_px: ``(n, 2+)`` array of landmark pixel coordinates.
            image_size: ``(width, height)`` of the frame the landmarks came from.
            ids: landmark indices matching :data:`MODEL_POINTS` row order.

        Returns:
            A :class:`HeadPose`, or ``None`` if the solver failed.
        """
        if landmarks_px is None or landmarks_px.shape[0] <= max(ids):
            return None
        image_points = np.ascontiguousarray(
            landmarks_px[list(ids), :2].astype(np.float64)
        )
        if not np.isfinite(image_points).all():
            return None

        cam = self._intrinsics(*image_size)
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, cam, self._dist, flags=_SOLVER
        )
        if not ok:
            return None
        try:  # Levenberg-Marquardt polish; harmless if it cannot improve
            rvec, tvec = cv2.solvePnPRefineLM(
                MODEL_POINTS, image_points, cam, self._dist, rvec, tvec
            )
        except cv2.error:
            pass

        rmat, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = rotation_matrix_to_euler(rmat)
        if not all(math.isfinite(v) for v in (yaw, pitch, roll)):
            return None
        return HeadPose(yaw, pitch, roll, rvec, tvec)

    def project_axes(
        self,
        pose: HeadPose,
        image_size: Tuple[int, int],
        length: float = 60.0,
    ) -> np.ndarray:
        """Project a 3D axis gizmo at the nose tip, for the debug overlay.

        Returns:
            ``(4, 2)`` pixel points: origin, +X (red), +Y (green), +Z (blue).
        """
        pts = np.array(
            [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]],
            dtype=np.float64,
        )
        cam = self._intrinsics(*image_size)
        projected, _ = cv2.projectPoints(pts, pose.rvec, pose.tvec, cam, self._dist)
        return projected.reshape(-1, 2)
