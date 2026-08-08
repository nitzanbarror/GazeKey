"""CameraSource tests — a fake capture device, no webcam required.

Covers the non-negotiable from spec Section 13: the pipeline must not crash
when the camera is unplugged, and must pick it up again when it returns.
"""
import threading
import time

import numpy as np
import pytest

from vision.camera import CameraSource


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture``."""

    def __init__(self, opens: bool = True, fail_after: int | None = None,
                 delay: float = 0.002):
        self.opens = opens
        self.fail_after = fail_after
        self.delay = delay
        self.reads = 0
        self.released = False
        self.props: dict = {}

    def isOpened(self) -> bool:          # noqa: N802 - mirrors the cv2 API
        return self.opens and not self.released

    def set(self, prop, value) -> bool:
        self.props[prop] = value
        return True

    def read(self):
        if self.released:
            return False, None
        time.sleep(self.delay)
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.full((4, 4, 3), self.reads % 256, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def wait_until(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def recorder():
    """Factory that records every capture it hands out (thread-safe)."""
    made: list[FakeCapture] = []
    lock = threading.Lock()

    def make(specs):
        def factory(index):
            with lock:
                spec = specs[min(len(made), len(specs) - 1)]
                cap = FakeCapture(**spec)
                made.append(cap)
            return cap
        return factory

    return made, make


def test_delivers_frames_and_reports_connected(recorder):
    made, make = recorder
    cam = CameraSource(index=0, capture_factory=make([{}])).start()
    try:
        frame = cam.read(timeout=2.0)
        assert frame is not None
        assert frame.image.shape == (4, 4, 3)
        assert cam.connected
        assert wait_until(lambda: cam.fps > 0)
    finally:
        cam.stop()
    assert made[0].released


def test_capture_properties_are_applied(recorder):
    made, make = recorder
    cam = CameraSource(index=0, width=1280, height=720, target_fps=60,
                       capture_factory=make([{}])).start()
    try:
        assert cam.read(timeout=2.0) is not None
        assert 1280 in made[0].props.values()
        assert 720 in made[0].props.values()
    finally:
        cam.stop()


def test_read_returns_the_newest_frame(recorder):
    made, make = recorder
    cam = CameraSource(index=0, capture_factory=make([{"delay": 0.001}])).start()
    try:
        first = cam.read(timeout=2.0)
        assert first is not None
        assert wait_until(lambda: cam.frames_captured > first.index + 5)
        latest = cam.latest()
        assert latest is not None and latest.index > first.index
    finally:
        cam.stop()


def test_unplug_does_not_crash_and_reconnects(recorder):
    """Device dies after 3 frames; the source must reopen it by itself."""
    made, make = recorder
    factory = make([{"fail_after": 3}, {}])
    cam = CameraSource(index=0, reconnect_interval_s=0.05,
                       capture_factory=factory).start()
    try:
        assert cam.read(timeout=2.0) is not None
        assert wait_until(lambda: made[0].released), "dead device was never released"
        assert wait_until(lambda: len(made) >= 2), "no reconnect attempt"
        assert wait_until(lambda: cam.connected)
        before = cam.frames_captured
        assert wait_until(lambda: cam.frames_captured > before), "no frames after reconnect"
    finally:
        cam.stop()


def test_camera_that_never_opens_is_reported_not_connected(recorder):
    made, make = recorder
    cam = CameraSource(index=9, reconnect_interval_s=0.05,
                       capture_factory=make([{"opens": False}])).start()
    try:
        assert cam.read(timeout=0.2) is None
        assert not cam.connected
        assert cam.last_error and "not available" in cam.last_error
        assert wait_until(lambda: len(made) >= 2), "should keep retrying"
    finally:
        cam.stop()


def test_factory_exceptions_are_swallowed():
    def exploding_factory(index):
        raise OSError("camera on fire")

    cam = CameraSource(index=0, reconnect_interval_s=0.05,
                       capture_factory=exploding_factory).start()
    try:
        assert cam.read(timeout=0.2) is None
        assert not cam.connected
        assert cam.last_error and "open failed" in cam.last_error
    finally:
        cam.stop()


def test_stop_is_idempotent_and_start_is_reentrant(recorder):
    made, make = recorder
    cam = CameraSource(index=0, capture_factory=make([{}]))
    cam.start()
    cam.start()
    assert cam.read(timeout=2.0) is not None
    cam.stop()
    cam.stop()
    assert all(cap.released for cap in made)


def test_context_manager_releases_device(recorder):
    made, make = recorder
    with CameraSource(index=0, capture_factory=make([{}])) as cam:
        assert cam.read(timeout=2.0) is not None
    assert made[0].released
