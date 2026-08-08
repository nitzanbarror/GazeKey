"""GazePipeline tests — fake camera and fake tracker, no webcam, no Qt."""
import random
import time

import numpy as np
import pytest

from gaze.calibration_session import Phase
from gaze.features import FrameFeatures
from vision.camera import CameraSource
from vision.face_tracker import TrackingResult
from vision.pipeline import (
    STATE_BLINK,
    STATE_NO_CAMERA,
    STATE_NO_FACE,
    STATE_OK,
    GazePipeline,
)
from tests.test_calibration_pipeline import (
    H,
    W,
    natural_pose_for_target,
    true_features_for_screen_point,
)
from tests.test_calibration_session import SyntheticUser, drive, new_session
from tests.test_camera import FakeCapture, wait_until


class FakeTracker:
    """Stands in for :class:`~vision.face_tracker.FaceTracker`."""

    backend_name = "fake"

    def __init__(self, result_fn):
        self._result_fn = result_fn
        self.calls = 0
        self.closed = False

    def process(self, image, timestamp):
        self.calls += 1
        return self._result_fn(self.calls, timestamp)

    def close(self):
        self.closed = True


def valid_result(hx=0.5, hy=0.5, yaw=0.0, pitch=0.0):
    def make(_call, timestamp):
        features = FrameFeatures(True, hx, hy, yaw, pitch, 0.0, timestamp)
        return TrackingResult(features, face_found=True, blink=False,
                              timestamp=timestamp)
    return make


def make_pipeline(result_fn, capture_kwargs=None, **kwargs):
    tracker = FakeTracker(result_fn)
    captures = []

    def camera_factory():
        def capture_factory(index):
            cap = FakeCapture(**(capture_kwargs or {}))
            captures.append(cap)
            return cap
        return CameraSource(index=0, reconnect_interval_s=0.05,
                            capture_factory=capture_factory)

    pipeline = GazePipeline(
        camera_factory=camera_factory,
        tracker_factory=lambda: tracker,
        **kwargs,
    )
    return pipeline, tracker, captures


def collect(pipeline, count=10, timeout=3.0):
    """Drain until ``count`` samples have been seen."""
    samples = []
    deadline = time.time() + timeout
    while len(samples) < count and time.time() < deadline:
        samples.extend(pipeline.drain())
        time.sleep(0.005)
    return samples


# ------------------------------------------------------------------- basics
def test_pipeline_emits_samples_without_a_model():
    pipeline, tracker, _ = make_pipeline(valid_result())
    with pipeline:
        samples = collect(pipeline, 5)
    assert len(samples) >= 5
    assert all(s.state == STATE_OK and s.valid for s in samples)
    assert all(not s.has_gaze for s in samples), "no model -> no screen position"
    assert pipeline.backend_name == "fake"
    assert tracker.closed


def test_pipeline_reports_tracking_states():
    def result_fn(call, timestamp):
        if call % 3 == 0:
            features = FrameFeatures(valid=False, timestamp=timestamp)
            return TrackingResult(features, face_found=True, blink=True,
                                  timestamp=timestamp)
        if call % 3 == 1:
            features = FrameFeatures(valid=False, timestamp=timestamp)
            return TrackingResult(features, face_found=False, blink=False,
                                  timestamp=timestamp)
        return valid_result()(call, timestamp)

    pipeline, _, _ = make_pipeline(result_fn)
    with pipeline:
        samples = collect(pipeline, 30)
    states = {s.state for s in samples}
    assert {STATE_OK, STATE_BLINK, STATE_NO_FACE} <= states


def test_pipeline_reports_a_missing_camera_without_crashing():
    pipeline, _, _ = make_pipeline(valid_result(), capture_kwargs={"opens": False})
    with pipeline:
        samples = collect(pipeline, 3)
    assert samples, "pipeline must keep producing samples with no camera"
    assert all(s.state == STATE_NO_CAMERA for s in samples)
    assert all(not s.valid for s in samples)
    assert not pipeline.camera_connected


def test_startup_failure_is_recorded_not_raised():
    def exploding_tracker():
        raise RuntimeError("no mediapipe model")

    pipeline = GazePipeline(tracker_factory=exploding_tracker)
    pipeline.start()
    try:
        assert pipeline.last_error and "no mediapipe model" in pipeline.last_error
        assert pipeline.drain() == []
    finally:
        pipeline.stop()


def test_measures_its_own_frame_rate():
    pipeline, _, _ = make_pipeline(valid_result())
    with pipeline:
        collect(pipeline, 20)
        assert wait_until(lambda: pipeline.fps > 0)


# ----------------------------------------------------------------- prediction
def fitted_model():
    """A model calibrated on the synthetic eye from the verified test suite."""
    session = new_session()
    drive(session, SyntheticUser(seed=42))
    assert session.phase is Phase.RESULTS and session.verdict == "PASS"
    return session.model


def features_for(px, py, timestamp, jitter=0.0, rng=None):
    yaw, pitch = natural_pose_for_target(px, py)
    vector = true_features_for_screen_point(px, py, yaw, pitch)
    if jitter and rng is not None:
        vector = vector + np.concatenate([rng.normal(0, jitter, 2), np.zeros(2)])
    return FrameFeatures(True, *vector[:2], vector[2], vector[3], 0.0, timestamp)


def test_predicts_screen_coordinates_once_a_model_is_installed():
    model = fitted_model()
    target = (int(0.5 * W), int(0.5 * H))

    def result_fn(_call, timestamp):
        return TrackingResult(features_for(*target, timestamp),
                              face_found=True, blink=False, timestamp=timestamp)

    pipeline, _, _ = make_pipeline(result_fn, model=model)
    with pipeline:
        samples = collect(pipeline, 20)
    gazing = [s for s in samples if s.has_gaze]
    assert gazing, "model installed but no gaze produced"
    last = gazing[-1]
    assert np.hypot(last.raw_x - target[0], last.raw_y - target[1]) < 80


def test_model_can_be_installed_while_running():
    target = (int(0.3 * W), int(0.7 * H))

    def result_fn(_call, timestamp):
        return TrackingResult(features_for(*target, timestamp),
                              face_found=True, blink=False, timestamp=timestamp)

    pipeline, _, _ = make_pipeline(result_fn)
    with pipeline:
        before = collect(pipeline, 5)
        pipeline.set_model(fitted_model())
        after = collect(pipeline, 25)
    assert all(not s.has_gaze for s in before)
    assert any(s.has_gaze for s in after)
    assert pipeline.model is not None


def test_one_euro_smoothing_reduces_jitter():
    model = fitted_model()
    rng = np.random.default_rng(3)
    target = (int(0.5 * W), int(0.5 * H))

    def result_fn(_call, timestamp):
        return TrackingResult(
            features_for(*target, timestamp, jitter=0.006, rng=rng),
            face_found=True, blink=False, timestamp=timestamp,
        )

    pipeline, _, _ = make_pipeline(result_fn, model=model,
                                   capture_kwargs={"delay": 0.02})
    with pipeline:
        samples = [s for s in collect(pipeline, 40, timeout=6.0) if s.has_gaze]
    assert len(samples) >= 25
    raw = np.array([[s.raw_x, s.raw_y] for s in samples[10:]])
    smoothed = np.array([[s.x, s.y] for s in samples[10:]])
    assert smoothed.std(axis=0).mean() < raw.std(axis=0).mean(), "no smoothing effect"


def test_smoothing_can_be_disabled():
    model = fitted_model()
    target = (int(0.5 * W), int(0.5 * H))

    def result_fn(_call, timestamp):
        return TrackingResult(features_for(*target, timestamp),
                              face_found=True, blink=False, timestamp=timestamp)

    pipeline, _, _ = make_pipeline(result_fn, model=model, smoothing=False)
    with pipeline:
        samples = [s for s in collect(pipeline, 10) if s.has_gaze]
    assert samples
    assert all(s.x == s.raw_x and s.y == s.raw_y for s in samples)


def test_invalid_frames_carry_no_gaze():
    model = fitted_model()

    def result_fn(_call, timestamp):
        features = FrameFeatures(valid=False, timestamp=timestamp)
        return TrackingResult(features, face_found=True, blink=True,
                              timestamp=timestamp)

    pipeline, _, _ = make_pipeline(result_fn, model=model)
    with pipeline:
        samples = collect(pipeline, 10)
    assert samples and all(not s.has_gaze for s in samples)


# ------------------------------------------------- M3: fixation + hold policy
def tracking_pipeline(**kwargs) -> GazePipeline:
    """A pipeline used only through track(); no camera or thread involved."""
    kwargs.setdefault("model", fitted_model())
    return GazePipeline(**kwargs)


def stare(pipeline, target, start=0.0, frames=20, dt=1 / 30, jitter=0.0, seed=0):
    """Feed a steady gaze and return every sample produced."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(frames):
        features = features_for(*target, start + i * dt, jitter=jitter, rng=rng)
        samples.append(pipeline.track(features))
    return samples


def test_steady_gaze_is_reported_as_a_fixation():
    pipeline = tracking_pipeline()
    samples = stare(pipeline, (int(0.5 * W), int(0.5 * H)), frames=20)
    assert not samples[0].is_fixating, "needs history before it can decide"
    assert samples[-1].is_fixating, "a steady stare must register as a fixation"
    assert all(s.stream_valid and not s.held for s in samples)


def test_a_saccade_breaks_the_fixation():
    pipeline = tracking_pipeline()
    stare(pipeline, (int(0.2 * W), int(0.2 * H)), start=0.0, frames=20)
    jumped = pipeline.track(features_for(int(0.8 * W), int(0.8 * H), 20 / 30))
    assert not jumped.is_fixating, "a jump across the screen is not a fixation"


def test_fixation_returns_after_the_gaze_settles_again():
    pipeline = tracking_pipeline()
    stare(pipeline, (int(0.2 * W), int(0.2 * H)), frames=20)
    settled = stare(pipeline, (int(0.8 * W), int(0.8 * H)),
                    start=20 / 30, frames=25)
    assert settled[-1].is_fixating


def test_dispersion_threshold_is_configurable():
    target = (int(0.5 * W), int(0.5 * H))
    strict = tracking_pipeline(fixation_dispersion_px=1.0)
    generous = tracking_pipeline(fixation_dispersion_px=400.0)
    assert not stare(strict, target, frames=20, jitter=0.01, seed=5)[-1].is_fixating
    assert generous.fixation_dispersion_px == 400.0
    assert stare(generous, target, frames=20, jitter=0.01, seed=5)[-1].is_fixating


def test_fixation_window_is_configurable():
    pipeline = tracking_pipeline(fixation_window_ms=2_000.0)
    samples = stare(pipeline, (int(0.5 * W), int(0.5 * H)), frames=20)
    assert not samples[-1].is_fixating, "a 2 s window needs far more history"


def blink(pipeline, timestamp):
    return pipeline.track(FrameFeatures(valid=False, timestamp=timestamp),
                          STATE_BLINK)


def test_blink_holds_the_last_position_for_the_hold_window():
    pipeline = tracking_pipeline(tracking_hold_ms=300.0)
    target = (int(0.5 * W), int(0.5 * H))
    last = stare(pipeline, target, frames=20)[-1]
    assert last.is_fixating

    held = blink(pipeline, last.timestamp + 0.1)   # 100 ms into the blink
    assert held.stream_valid and held.held
    assert (held.x, held.y) == (last.x, last.y), "position must be held, not moved"
    assert held.is_fixating, "dwell must freeze, not reset (spec Section 6)"
    assert not held.valid, "the frame itself is still invalid"


def test_hold_expires_after_300ms_and_the_stream_goes_invalid():
    pipeline = tracking_pipeline(tracking_hold_ms=300.0)
    base = stare(pipeline, (int(0.5 * W), int(0.5 * H)), frames=20)[-1].timestamp

    assert blink(pipeline, base + 0.29).stream_valid, "still inside the hold"
    lost = blink(pipeline, base + 0.31)
    assert not lost.stream_valid
    assert not lost.has_gaze and not lost.is_fixating


def test_hold_duration_is_configurable():
    pipeline = tracking_pipeline(tracking_hold_ms=1_000.0)
    base = stare(pipeline, (int(0.5 * W), int(0.5 * H)), frames=20)[-1].timestamp
    assert blink(pipeline, base + 0.9).stream_valid
    assert not blink(pipeline, base + 1.1).stream_valid


def test_gaze_resumes_cleanly_after_a_long_blink():
    pipeline = tracking_pipeline()
    target = (int(0.5 * W), int(0.5 * H))
    base = stare(pipeline, target, frames=20)[-1].timestamp
    blink(pipeline, base + 0.5)                    # past the hold

    resumed = stare(pipeline, target, start=base + 0.6, frames=20)
    assert not resumed[0].is_fixating, "must rebuild history before deciding"
    assert resumed[-1].is_fixating
    assert all(s.stream_valid for s in resumed)


def test_without_a_model_there_is_no_gaze_and_no_fixation():
    pipeline = GazePipeline()
    sample = pipeline.track(features_for(int(0.5 * W), int(0.5 * H), 1.0))
    assert not sample.has_gaze and not sample.stream_valid
    assert not sample.is_fixating and not sample.held
    assert sample.valid, "the frame itself was fine — there is just no model"


def test_installing_a_model_clears_stale_hold_and_fixation_state():
    pipeline = tracking_pipeline(tracking_hold_ms=300.0)
    stare(pipeline, (int(0.5 * W), int(0.5 * H)), frames=20)

    pipeline.set_model(fitted_model())
    after = blink(pipeline, 20 / 30 + 0.05)
    assert not after.stream_valid, "hold must not survive a recalibration"
    assert not after.is_fixating


def test_missing_camera_frames_go_through_the_same_hold_policy():
    pipeline, _, _ = make_pipeline(valid_result(), capture_kwargs={"opens": False})
    with pipeline:
        samples = collect(pipeline, 3)
    assert all(s.state == STATE_NO_CAMERA for s in samples)
    assert all(not s.stream_valid and not s.is_fixating for s in samples)


def test_fixation_travels_through_the_worker_thread():
    """End-to-end: the queue really carries is_fixating, not just track()."""
    model = fitted_model()
    target = (int(0.5 * W), int(0.5 * H))

    def result_fn(_call, timestamp):
        return TrackingResult(features_for(*target, timestamp),
                              face_found=True, blink=False, timestamp=timestamp)

    pipeline, _, _ = make_pipeline(result_fn, model=model,
                                   capture_kwargs={"delay": 0.01})
    with pipeline:
        samples = collect(pipeline, 40, timeout=6.0)
    assert any(s.is_fixating for s in samples), "no fixation reached the queue"
    assert all(s.stream_valid for s in samples if s.has_gaze)


def test_stop_shuts_down_camera_and_tracker():
    pipeline, tracker, captures = make_pipeline(valid_result())
    pipeline.start()
    collect(pipeline, 3)
    pipeline.stop()
    assert tracker.closed
    assert captures and all(c.released for c in captures)
