"""GazeKey — the vision worker thread (spec Section 3).

Camera → MediaPipe → features → gaze prediction → smoothing, all on a
background thread. Results are pushed into a thread-safe queue that the Qt
thread drains from a timer, so the UI never blocks on vision work and the
pipeline has no Qt dependency at all (which also makes it testable headless).

The calibration model can be swapped in at any time — during a calibration
session there is no model yet and samples carry features only.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

from gaze.calibration import CalibrationModel
from gaze.features import FrameFeatures
from gaze.smoothing import FixationDetector, OneEuroFilter
from vision.camera import CameraSource
from vision.face_tracker import FaceTracker

#: how a frame failed, for the UI's tracking indicator
STATE_OK = "ok"
STATE_BLINK = "blink"
STATE_NO_FACE = "no_face"
STATE_NO_CAMERA = "no_camera"


@dataclass
class GazeSample:
    """One frame's worth of output from the pipeline.

    Two different notions of "valid" live here and they are not the same thing:

    ``valid``
        this *frame* produced usable features (a face, not blinking);
    ``stream_valid``
        a usable gaze position is available *right now* — either fresh, or
        held over a blink for up to ``tracking_hold_ms`` (spec Section 6).
        Once the hold expires the stream goes invalid: the UI shows "tracking
        lost" and a dwell timer freezes rather than resetting.
    """

    features: FrameFeatures
    state: str
    timestamp: float
    x: float = float("nan")          # smoothed screen px, NaN without a model
    y: float = float("nan")
    raw_x: float = float("nan")      # unsmoothed prediction
    raw_y: float = float("nan")
    is_fixating: bool = False
    stream_valid: bool = False
    held: bool = False               # x/y carried over from before a blink
    #: (478, 2) landmarks in **pixels**, and only when a caller has asked for
    #: them (``GazePipeline.landmarks_enabled``). Off by default: nothing in
    #: the app needs them, and an array per frame on every sample is waste.
    #: In memory only, like every other frame-derived thing here (NFR-4).
    #:
    #: Pixels rather than the normalised coordinates, because normalised x and
    #: y are divided by *different* numbers (640 and 480 here): any measurement
    #: that mixes the two axes — a distance, an angle, a perpendicular — is
    #: wrong in that space. The verified ``hx``/``hy`` are unaffected, being
    #: ratios within one axis.
    landmarks_px: Optional[np.ndarray] = None

    @property
    def valid(self) -> bool:
        """True when the frame produced usable features."""
        return self.features.valid

    @property
    def has_gaze(self) -> bool:
        """True when a calibrated screen position is available."""
        return bool(np.isfinite(self.x) and np.isfinite(self.y))


class GazePipeline:
    """Background camera → gaze pipeline feeding a queue of :class:`GazeSample`."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        backend: str = "auto",
        model: Optional[CalibrationModel] = None,
        smoothing: bool = True,
        fixation_window_ms: float = 150.0,
        fixation_dispersion_px: float = 110.0,
        tracking_hold_ms: float = 300.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        camera_factory: Optional[Callable[[], CameraSource]] = None,
        tracker_factory: Optional[Callable[[], FaceTracker]] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.camera_index = camera_index
        self._camera_factory = camera_factory or (
            lambda: CameraSource(index=camera_index, width=width, height=height)
        )
        self._tracker_factory = tracker_factory or (
            lambda: FaceTracker(backend=backend, log=log)
        )
        self._smoothing = smoothing
        self._log = log

        self.fixation_window_ms = fixation_window_ms
        self.fixation_dispersion_px = fixation_dispersion_px
        self.tracking_hold_s = tracking_hold_ms / 1000.0
        #: One-Euro tuning (spec Section 6). Lowering ``min_cutoff`` smooths
        #: harder at the cost of lag — the knob for gaze jitter that keeps
        #: crossing key boundaries and stealing focus mid-dwell.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)

        self._queue: "queue.Queue[GazeSample]" = queue.Queue()
        self._model_lock = threading.Lock()
        self._model = model
        self._filters = self._new_filters()
        self._fixation = FixationDetector(fixation_window_ms, fixation_dispersion_px)
        self._last_gaze: Optional[tuple[float, float, float]] = None
        self._last_fixating = False

        self._preview_lock = threading.Lock()
        self._preview: Optional[np.ndarray] = None
        self.preview_enabled = False
        #: attach per-frame landmarks to each sample — see GazeSample. Only the
        #: --feature-lab diagnostic turns this on.
        self.landmarks_enabled = False
        self._camera: Optional[CameraSource] = None
        self._tracker: Optional[FaceTracker] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._fps = 0.0
        self._backend_name = ""
        self._error: Optional[str] = None

    # ------------------------------------------------------------------ state
    @property
    def fps(self) -> float:
        """End-to-end pipeline rate (frames fully processed per second)."""
        return self._fps

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def camera_connected(self) -> bool:
        return bool(self._camera and self._camera.connected)

    @property
    def last_error(self) -> Optional[str]:
        return self._error or (self._camera.last_error if self._camera else None)

    # -------------------------------------------------------------- lifecycle
    def start(self, ready_timeout: float = 20.0) -> "GazePipeline":
        """Start the worker thread and wait for the tracker to be constructed."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run, name="gazekey-pipeline", daemon=True
        )
        self._thread.start()
        self._ready.wait(ready_timeout)
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=3.0)

    def __enter__(self) -> "GazePipeline":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------------- data
    def drain(self) -> List[GazeSample]:
        """Take every sample produced since the last call (never blocks)."""
        out: List[GazeSample] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    # ----------------------------------------------------------- self-view
    def _store_preview(self, image: np.ndarray, width: int = 192) -> None:
        """Keep a small mirrored copy of the latest frame for the self-view.

        In memory only, and only while a view is actually showing — nothing is
        ever written to disk (spec Section 13).
        """
        height = max(1, int(image.shape[0] * width / max(image.shape[1], 1)))
        try:
            small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        except Exception:                                # pragma: no cover
            return
        with self._preview_lock:
            self._preview = cv2.cvtColor(cv2.flip(small, 1), cv2.COLOR_BGR2RGB)

    def preview_frame(self) -> Optional[np.ndarray]:
        """The latest mirrored RGB thumbnail, or ``None``."""
        with self._preview_lock:
            return None if self._preview is None else self._preview.copy()

    def _new_filters(self) -> tuple:
        """A fresh One-Euro pair at the configured tuning, one per axis."""
        return (OneEuroFilter(min_cutoff=self.min_cutoff, beta=self.beta),
                OneEuroFilter(min_cutoff=self.min_cutoff, beta=self.beta))

    def set_model(self, model: Optional[CalibrationModel]) -> None:
        """Install (or clear) the model and reset smoothing, fixation and hold."""
        with self._model_lock:
            self._model = model
            self._filters = self._new_filters()
            self._fixation = FixationDetector(
                self.fixation_window_ms, self.fixation_dispersion_px
            )
            self._last_gaze = None
            self._last_fixating = False

    @property
    def model(self) -> Optional[CalibrationModel]:
        with self._model_lock:
            return self._model

    # ---------------------------------------------------------------- tracking
    def track(self, features: FrameFeatures, state: str = STATE_OK) -> GazeSample:
        """Turn one frame's features into a :class:`GazeSample`.

        Prediction → One-Euro smoothing → I-DT fixation → blink-hold policy.
        Public so the whole gaze policy can be exercised without a camera.
        """
        nan = float("nan")
        timestamp = features.timestamp
        with self._model_lock:
            model = self._model
            filters = self._filters
            fixation = self._fixation

            if model is None:
                self._last_gaze = None
                self._last_fixating = False
                return GazeSample(features, state, timestamp)

            if features.valid:
                raw_x, raw_y = model.predict(features.vector())
                if self._smoothing:
                    x = filters[0].apply(raw_x, timestamp)
                    y = filters[1].apply(raw_y, timestamp)
                else:
                    x, y = raw_x, raw_y
                is_fixating = fixation.update(x, y, timestamp)
                self._last_gaze = (x, y, timestamp)
                self._last_fixating = is_fixating
                return GazeSample(features, state, timestamp, x, y, raw_x, raw_y,
                                  is_fixating=is_fixating, stream_valid=True)

            # Invalid frame (blink or lost face). Hold the last position for up
            # to tracking_hold_ms; the fixation verdict freezes rather than
            # resetting, so a blink mid-dwell does not throw the dwell away.
            if (self._last_gaze is not None
                    and timestamp - self._last_gaze[2] <= self.tracking_hold_s):
                held_x, held_y, _ = self._last_gaze
                return GazeSample(features, state, timestamp, held_x, held_y,
                                  nan, nan, is_fixating=self._last_fixating,
                                  stream_valid=True, held=True)

            # Held too long — the stream is now invalid.
            self._last_fixating = False
            fixation.reset()
            return GazeSample(features, state, timestamp)

    def _run(self) -> None:
        # MediaPipe graphs are thread-affine: build the tracker on this thread.
        try:
            self._tracker = self._tracker_factory()
            self._backend_name = self._tracker.backend_name
            self._camera = self._camera_factory().start()
        except Exception as exc:
            self._error = f"pipeline startup failed: {exc}"
            self._log(f"[GazeKey] {self._error}")
            self._ready.set()
            return
        self._ready.set()

        last = time.time()
        try:
            while not self._stop.is_set():
                frame = self._camera.read(timeout=0.2)
                if frame is None:
                    lost = FrameFeatures(valid=False, timestamp=time.time())
                    self._queue.put(self.track(lost, STATE_NO_CAMERA))
                    self._fps = 0.0
                    continue

                if self.preview_enabled:
                    self._store_preview(frame.image)
                result = self._tracker.process(frame.image, frame.timestamp)
                features = result.features
                if features.valid:
                    state = STATE_OK
                elif not result.face_found:
                    state = STATE_NO_FACE
                elif result.blink:
                    state = STATE_BLINK
                else:
                    state = STATE_NO_FACE

                sample = self.track(features, state)
                if self.landmarks_enabled:
                    sample.landmarks_px = result.landmarks_px
                self._queue.put(sample)

                now = time.time()
                dt = now - last
                last = now
                if dt > 0:
                    self._fps = 0.9 * self._fps + 0.1 / dt if self._fps else 1.0 / dt
        finally:
            if self._camera is not None:
                self._camera.stop()
            if self._tracker is not None:
                self._tracker.close()
