"""Head-pose tests — synthetic projections of the canonical model, no camera.

These pin the yaw/pitch/roll sign conventions documented in
:mod:`vision.head_pose`. If they ever change, every saved calibration becomes
invalid, so the conventions are asserted explicitly rather than implied.
"""
import cv2
import numpy as np
import pytest

from vision.head_pose import (
    MODEL_POINTS,
    POSE_LANDMARK_IDS,
    HeadPoseEstimator,
    camera_matrix,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
)

WIDTH, HEIGHT = 640, 480
DISTANCE_MM = 600.0


def project_face(yaw: float, pitch: float, roll: float,
                 distance: float = DISTANCE_MM) -> np.ndarray:
    """Render the 6 model points as a full-size landmark array in pixels."""
    rmat = euler_to_rotation_matrix(yaw, pitch, roll)
    rvec, _ = cv2.Rodrigues(rmat)
    tvec = np.array([[0.0], [0.0], [distance]])
    projected, _ = cv2.projectPoints(
        MODEL_POINTS, rvec, tvec, camera_matrix(WIDTH, HEIGHT), np.zeros((4, 1))
    )
    landmarks = np.full((478, 2), np.nan)
    landmarks[list(POSE_LANDMARK_IDS)] = projected.reshape(-1, 2)
    return landmarks


# ------------------------------------------------------------ euler round-trip
@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [(0, 0, 0), (15, 0, 0), (-15, 0, 0), (0, 12, 0), (0, -12, 0),
     (0, 0, 20), (10, -8, 5), (-22, 14, -11)],
)
def test_euler_matrix_round_trip(yaw, pitch, roll):
    rmat = euler_to_rotation_matrix(yaw, pitch, roll)
    assert rotation_matrix_to_euler(rmat) == pytest.approx((yaw, pitch, roll), abs=1e-6)


def test_identity_rotation_is_zero_angles():
    assert rotation_matrix_to_euler(np.eye(3)) == pytest.approx((0.0, 0.0, 0.0))


# ------------------------------------------------------- documented conventions
def test_positive_yaw_points_the_face_toward_image_left():
    """yaw > 0 = user turns to their own right = image-left in a raw frame."""
    normal = euler_to_rotation_matrix(20.0, 0.0, 0.0) @ np.array([0.0, 0.0, -1.0])
    assert normal[0] < 0


def test_positive_pitch_points_the_face_up():
    """+Y is down in this frame, so 'up' means a negative y component."""
    normal = euler_to_rotation_matrix(0.0, 20.0, 0.0) @ np.array([0.0, 0.0, -1.0])
    assert normal[1] < 0


def test_positive_roll_tilts_toward_the_users_left_shoulder():
    """The user's right eye (model x < 0) rises in the image."""
    right_eye = MODEL_POINTS[2]                       # landmark 33
    rotated = euler_to_rotation_matrix(0.0, 0.0, 15.0) @ right_eye
    assert rotated[1] < right_eye[1]                  # smaller y = higher up


# -------------------------------------------------------------- solvePnP round-trip
@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [(0, 0, 0), (12, 0, 0), (-12, 0, 0), (0, 10, 0), (0, -10, 0),
     (0, 0, 8), (15, -9, 6), (-18, 7, -5)],
)
def test_solvepnp_recovers_known_angles(yaw, pitch, roll):
    pose = HeadPoseEstimator().estimate(project_face(yaw, pitch, roll), (WIDTH, HEIGHT))
    assert pose is not None
    assert (pose.yaw, pose.pitch, pose.roll) == pytest.approx((yaw, pitch, roll), abs=0.5)


def test_yaw_is_monotonic_across_a_head_sweep():
    """What the Stage-A head sweep relies on: pose varies smoothly and signed."""
    estimator = HeadPoseEstimator()
    yaws = []
    for true_yaw in np.linspace(-15, 15, 11):
        pose = estimator.estimate(project_face(true_yaw, 0, 0), (WIDTH, HEIGHT))
        assert pose is not None
        yaws.append(pose.yaw)
    assert all(b > a for a, b in zip(yaws, yaws[1:])), yaws


def test_pose_is_stable_under_small_landmark_noise():
    rng = np.random.default_rng(7)
    estimator = HeadPoseEstimator()
    truth = project_face(8.0, -4.0, 2.0)
    angles = []
    for _ in range(30):
        noisy = truth.copy()
        noisy[list(POSE_LANDMARK_IDS)] += rng.normal(0, 0.5, size=(6, 2))
        pose = estimator.estimate(noisy, (WIDTH, HEIGHT))
        assert pose is not None
        angles.append((pose.yaw, pose.pitch, pose.roll))
    spread = np.std(np.array(angles), axis=0)
    assert spread.max() < 2.0, spread


# ----------------------------------------------------------------- bad input
def test_missing_landmarks_return_none():
    assert HeadPoseEstimator().estimate(np.zeros((10, 2)), (WIDTH, HEIGHT)) is None
    assert HeadPoseEstimator().estimate(None, (WIDTH, HEIGHT)) is None


def test_nan_landmarks_return_none():
    landmarks = project_face(0, 0, 0)
    landmarks[POSE_LANDMARK_IDS[0]] = (np.nan, np.nan)
    assert HeadPoseEstimator().estimate(landmarks, (WIDTH, HEIGHT)) is None


def test_project_axes_returns_four_points():
    estimator = HeadPoseEstimator()
    pose = estimator.estimate(project_face(5, 5, 0), (WIDTH, HEIGHT))
    assert pose is not None
    axes = estimator.project_axes(pose, (WIDTH, HEIGHT))
    assert axes.shape == (4, 2)
    assert np.isfinite(axes).all()
