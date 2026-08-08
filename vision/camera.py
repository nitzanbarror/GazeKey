"""GazeKey — threaded webcam capture with automatic reconnect (spec Section 3).

The grabber thread always keeps only the **latest** frame, so a slow consumer
(MediaPipe) never works on stale images and never stalls capture. Unplugging
the camera does not raise: the source flips to ``connected = False`` and keeps
retrying in the background until the device comes back.

Frames only ever live in memory — nothing is written to disk (spec Section 13).
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

#: consecutive failed reads before the device is considered gone
_MAX_READ_FAILURES = 5


@dataclass
class Frame:
    """A single captured frame."""

    image: np.ndarray
    timestamp: float
    index: int


def _default_capture_factory(index: int) -> "cv2.VideoCapture":
    """Open a capture device, preferring DirectShow on Windows (fast open)."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return cv2.VideoCapture(index)


class CameraSource:
    """Background webcam reader exposing the most recent frame.

    Example:
        >>> cam = CameraSource(index=0).start()          # doctest: +SKIP
        >>> frame = cam.read(timeout=1.0)                # doctest: +SKIP
        >>> cam.stop()                                   # doctest: +SKIP
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        reconnect_interval_s: float = 1.0,
        capture_factory: Callable[[int], object] = _default_capture_factory,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.reconnect_interval_s = reconnect_interval_s
        self._capture_factory = capture_factory

        self._cap: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._new_frame = threading.Event()
        self._lock = threading.Lock()

        self._frame: Optional[Frame] = None
        self._frame_index = 0
        self._connected = False
        self._last_error: Optional[str] = None
        self._fps = 0.0
        self._fps_window: list[float] = []

    # ------------------------------------------------------------------ state
    @property
    def connected(self) -> bool:
        """True while the device is open and delivering frames."""
        return self._connected

    @property
    def fps(self) -> float:
        """Measured capture rate over the last ~30 frames."""
        return self._fps

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def frames_captured(self) -> int:
        return self._frame_index

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "CameraSource":
        """Start the grabber thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="gazekey-camera", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the grabber thread and release the device."""
        self._stop.set()
        self._new_frame.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self._release()

    def __enter__(self) -> "CameraSource":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------------ read
    def read(self, timeout: float | None = 1.0) -> Optional[Frame]:
        """Return the newest unread frame, waiting up to ``timeout`` seconds.

        Returns ``None`` on timeout (including while the camera is unplugged).
        """
        if timeout is not None and timeout > 0:
            if not self._new_frame.wait(timeout):
                return None
        self._new_frame.clear()
        with self._lock:
            return self._frame

    def latest(self) -> Optional[Frame]:
        """Non-blocking peek at the last frame, however old it is."""
        with self._lock:
            return self._frame

    # --------------------------------------------------------------- internals
    def _open(self) -> bool:
        self._release()
        try:
            cap = self._capture_factory(self.index)
        except Exception as exc:  # pragma: no cover - backend specific
            self._last_error = f"open failed: {exc}"
            return False
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self._last_error = f"camera {self.index} not available"
            return False

        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, self.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
            (cv2.CAP_PROP_FPS, self.target_fps),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
        ):
            try:
                cap.set(prop, value)
            except Exception:  # pragma: no cover - some backends reject these
                pass

        self._cap = cap
        self._connected = True
        self._last_error = None
        return True

    def _release(self) -> None:
        cap, self._cap = self._cap, None
        self._connected = False
        if cap is not None:
            try:
                cap.release()
            except Exception:  # pragma: no cover
                pass

    def _publish(self, image: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            self._frame_index += 1
            self._frame = Frame(image, now, self._frame_index)
        self._new_frame.set()

        self._fps_window.append(now)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)
        span = self._fps_window[-1] - self._fps_window[0]
        if len(self._fps_window) > 1 and span > 0:
            self._fps = (len(self._fps_window) - 1) / span

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            if self._cap is None:
                if not self._open():
                    self._stop.wait(self.reconnect_interval_s)
                    continue
                failures = 0

            try:
                ok, image = self._cap.read()  # type: ignore[union-attr]
            except Exception as exc:  # pragma: no cover - backend specific
                ok, image = False, None
                self._last_error = f"read failed: {exc}"

            if not ok or image is None:
                failures += 1
                if failures >= _MAX_READ_FAILURES:
                    self._last_error = self._last_error or "camera disconnected"
                    self._release()
                    self._fps = 0.0
                    self._fps_window.clear()
                    self._stop.wait(self.reconnect_interval_s)
                continue

            failures = 0
            self._publish(image)

        self._release()
