"""Debug-view rendering tests — draw one frame offscreen, no camera, no window.

Only the drawing helpers are exercised (``cv2.imshow`` needs a display); this
catches the kind of overlay bug that would otherwise crash on first run.
"""
import time

import numpy as np
import pytest

import debug_vision
from tests.test_features import make_face
from vision.face_tracker import TrackingResult, eye_diagnostics
from vision.head_pose import HeadPoseEstimator
from tests.test_head_pose import project_face

WIDTH, HEIGHT = 640, 480


def make_result(face: bool = True) -> TrackingResult:
    from gaze.features import extract_features

    if not face:
        return TrackingResult(
            features=extract_features(None, np.nan, np.nan, np.nan, 0.0),
            face_found=False, blink=False, timestamp=0.0,
        )
    landmarks = make_face(gaze=0.3)
    landmarks_px = project_face(6.0, -3.0, 2.0)
    landmarks_px = np.nan_to_num(landmarks_px, nan=100.0)
    estimator = HeadPoseEstimator()
    pose = estimator.estimate(project_face(6.0, -3.0, 2.0), (WIDTH, HEIGHT))
    return TrackingResult(
        features=extract_features(landmarks, 6.0, -3.0, 2.0, 0.0),
        face_found=True,
        blink=False,
        timestamp=0.0,
        landmarks=landmarks,
        landmarks_px=landmarks_px,
        pose=pose,
        diagnostics=eye_diagnostics(landmarks),
    )


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("show_cloud", [False, True])
def test_overlay_draws_without_error(mirrored, show_cloud):
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    result = make_result()
    debug_vision._draw_overlay(canvas, result, mirrored, show_cloud)
    assert canvas.any(), "overlay drew nothing"


def test_overlay_tolerates_a_missing_face():
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    debug_vision._draw_overlay(canvas, make_result(face=False), True, True)
    assert not canvas.any()


def test_hud_and_gauges_render():
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    debug_vision._text(canvas, [("hello", (255, 255, 255)), ("world", (0, 255, 0))])
    debug_vision._bar(canvas, 14, 400, 220, 12, 0.42, "hx", (255, 255, 255))
    debug_vision._bar(canvas, 14, 424, 220, 12, float("nan"), "hy", (255, 255, 255))
    assert canvas.any()


def test_key_handling():
    assert debug_vision._handle_keys(ord("q")) == "quit"
    assert debug_vision._handle_keys(27) == "quit"
    assert debug_vision._handle_keys(ord("m")) == "mirror"
    assert debug_vision._handle_keys(ord("l")) == "cloud"
    assert debug_vision._handle_keys(ord("r")) == "reset"
    assert debug_vision._handle_keys(-1) is None


def test_wait_for_first_frame_returns_the_frame():
    from tests.test_camera import FakeCapture
    from vision.camera import CameraSource

    cam = CameraSource(index=0, capture_factory=lambda i: FakeCapture()).start()
    try:
        frame, reason = debug_vision._wait_for_first_frame(cam, timeout=3.0)
        assert reason == "ok"
        assert frame is not None
    finally:
        cam.stop()


def test_wait_for_first_frame_times_out_on_a_dead_camera():
    from tests.test_camera import FakeCapture
    from vision.camera import CameraSource

    cam = CameraSource(index=0, reconnect_interval_s=0.05,
                       capture_factory=lambda i: FakeCapture(opens=False)).start()
    try:
        started = time.monotonic()
        frame, reason = debug_vision._wait_for_first_frame(cam, timeout=0.3)
        elapsed = time.monotonic() - started
    finally:
        cam.stop()
    assert (frame, reason) == (None, "timeout")
    assert 0.25 < elapsed < 2.0, f"timeout not honoured ({elapsed:.2f}s)"


def test_wait_for_first_frame_can_be_aborted_by_the_user():
    from tests.test_camera import FakeCapture
    from vision.camera import CameraSource

    cam = CameraSource(index=0, reconnect_interval_s=0.05,
                       capture_factory=lambda i: FakeCapture(opens=False)).start()
    try:
        frame, reason = debug_vision._wait_for_first_frame(
            cam, timeout=30.0, pump=lambda: "quit"
        )
    finally:
        cam.stop()
    assert (frame, reason) == (None, "quit")


def test_startup_failure_message_is_actionable(capsys):
    from tests.test_camera import FakeCapture
    from vision.camera import CameraSource

    cam = CameraSource(index=3, reconnect_interval_s=0.05,
                       capture_factory=lambda i: FakeCapture(opens=False)).start()
    try:
        debug_vision._wait_for_first_frame(cam, timeout=0.2)
        debug_vision._print_startup_failure(3, cam, 5.0)
    finally:
        cam.stop()
    out = capsys.readouterr().out
    assert "camera 3 produced no frames within 5 s" in out
    assert "--camera 1" in out
    assert "--startup-timeout" in out
    assert "not available" in out          # the underlying camera error


def test_stability_tracker_ignores_nans():
    stats = debug_vision._Stability()
    for value in [0.5, float("nan"), 0.51, 0.49, 0.5, 0.5]:
        stats.push(value)
    assert stats.std() < 0.02
    stats.clear()
    assert np.isnan(stats.std())
