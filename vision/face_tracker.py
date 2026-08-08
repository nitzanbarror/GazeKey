"""GazeKey — MediaPipe face tracking → :class:`gaze.features.FrameFeatures`.

Two MediaPipe front-ends are supported; both produce the same ``(478, 3)``
array of normalised landmarks (the attention/iris mesh), which is then handed
to the **verified** :func:`gaze.features.extract_features` unchanged:

``legacy``
    ``mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)`` — the API named
    in the spec. Available in MediaPipe ≤ 0.10.2x (Python ≤ 3.12).
``tasks``
    ``mediapipe.tasks.python.vision.FaceLandmarker`` — the only API in recent
    wheels (and in every Python 3.13 build), where ``mp.solutions`` no longer
    exists. Needs the ``face_landmarker.task`` bundle, which embeds the same
    iris mesh ``refine_landmarks=True`` used to switch on.

``backend="auto"`` picks ``legacy`` when present, otherwise ``tasks``.

Nothing here touches the disk apart from reading the model bundle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from gaze.features import (
    EAR_BLINK_THRESHOLD,
    L_INNER,
    L_IRIS,
    L_LO,
    L_OUTER,
    L_UP,
    R_INNER,
    R_IRIS,
    R_LO,
    R_OUTER,
    R_UP,
    FrameFeatures,
    _eye_ratios,
    extract_features,
    eye_aspect_ratio,
)
from vision.head_pose import HeadPose, HeadPoseEstimator

NUM_LANDMARKS = 478


@dataclass
class EyeDiagnostics:
    """Per-eye numbers shown in the debug view (never fed to calibration).

    ``hx_right`` / ``hx_left`` are the two halves of the averaged ``hx`` that
    :mod:`gaze.features` produces, so a conjugate sweep moves both together.

    ``hx_image`` recomputes the same thing straight from the image axis
    (``0`` = image-left corner, ``1`` = image-right corner), independently of
    which landmark is labelled "inner" or "outer" — a cross-check that the
    corner arguments are the right way round.
    """

    hx_right: float
    hx_left: float
    hx_image: float
    ear_right: float
    ear_left: float


@dataclass
class TrackingResult:
    """Everything one frame produced."""

    features: FrameFeatures
    face_found: bool
    blink: bool
    timestamp: float
    landmarks: Optional[np.ndarray] = None      # (478, 3) normalised
    landmarks_px: Optional[np.ndarray] = None   # (478, 2) pixels
    pose: Optional[HeadPose] = None
    diagnostics: Optional[EyeDiagnostics] = None


# --------------------------------------------------------------------- backends
class _LegacyFaceMeshBackend:
    """``mp.solutions.face_mesh`` (MediaPipe ≤ 0.10.2x)."""

    name = "legacy"

    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float):
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def landmarks(self, rgb: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        result = self._mesh.process(rgb)
        faces = getattr(result, "multi_face_landmarks", None)
        if not faces:
            return None
        return np.array(
            [(p.x, p.y, p.z) for p in faces[0].landmark], dtype=np.float64
        )

    def close(self) -> None:
        self._mesh.close()


class _TasksBackend:
    """``mediapipe.tasks.python.vision.FaceLandmarker`` in VIDEO mode."""

    name = "tasks"

    def __init__(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: Optional[str],
        log: Callable[[str], None],
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        from vision.model_assets import ensure_face_landmarker

        self._mp = mp
        self._path = model_path or ensure_face_landmarker(log=log)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def landmarks(self, rgb: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        # VIDEO mode demands strictly increasing integer millisecond stamps.
        ms = max(int(timestamp * 1000.0), self._last_ms + 1)
        self._last_ms = ms
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._landmarker.detect_for_video(image, ms)
        if not result.face_landmarks:
            return None
        return np.array(
            [(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float64
        )

    def close(self) -> None:
        self._landmarker.close()


def _legacy_available() -> bool:
    try:
        import mediapipe as mp
    except ImportError:  # pragma: no cover - mediapipe is a hard dependency
        return False
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


# ----------------------------------------------------------------- face tracker
class FaceTracker:
    """Turns BGR frames into :class:`~gaze.features.FrameFeatures`."""

    def __init__(
        self,
        backend: str = "auto",
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        log: Callable[[str], None] = print,
    ) -> None:
        if backend == "auto":
            backend = "legacy" if _legacy_available() else "tasks"
        if backend == "legacy":
            self._backend: object = _LegacyFaceMeshBackend(
                min_detection_confidence, min_tracking_confidence
            )
        elif backend == "tasks":
            self._backend = _TasksBackend(
                min_detection_confidence, min_tracking_confidence, model_path, log
            )
        else:
            raise ValueError(f"unknown backend {backend!r} (auto|legacy|tasks)")
        self._pose = HeadPoseEstimator()

    @property
    def backend_name(self) -> str:
        """``"legacy"`` or ``"tasks"``."""
        return self._backend.name  # type: ignore[attr-defined]

    @property
    def pose_estimator(self) -> HeadPoseEstimator:
        """The solvePnP estimator (exposed so the debug view can draw axes)."""
        return self._pose

    def process(
        self, frame_bgr: np.ndarray, timestamp: Optional[float] = None
    ) -> TrackingResult:
        """Run the mesh on one BGR frame and extract gaze features."""
        t = time.time() if timestamp is None else timestamp
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        landmarks = self._backend.landmarks(rgb, t)  # type: ignore[attr-defined]
        if landmarks is None or landmarks.shape[0] < NUM_LANDMARKS:
            return TrackingResult(
                features=extract_features(None, np.nan, np.nan, np.nan, t),
                face_found=False,
                blink=False,
                timestamp=t,
            )

        landmarks_px = landmarks[:, :2] * np.array([width, height], dtype=np.float64)
        diagnostics = eye_diagnostics(landmarks)
        blink = min(diagnostics.ear_right, diagnostics.ear_left) < EAR_BLINK_THRESHOLD

        pose = self._pose.estimate(landmarks_px, (width, height))
        if pose is None:
            features = FrameFeatures(valid=False, timestamp=t)
        else:
            features = extract_features(landmarks, pose.yaw, pose.pitch, pose.roll, t)

        return TrackingResult(
            features=features,
            face_found=True,
            blink=blink,
            timestamp=t,
            landmarks=landmarks,
            landmarks_px=landmarks_px,
            pose=pose,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        self._backend.close()  # type: ignore[attr-defined]

    def __enter__(self) -> "FaceTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ------------------------------------------------------------------ diagnostics
def _image_axis_ratio(
    landmarks: np.ndarray, iris_idx, corner_a: int, corner_b: int
) -> float:
    """Iris position between two eye corners along the **image** x-axis."""
    iris_x = float(landmarks[iris_idx, 0].mean())
    lo = float(min(landmarks[corner_a, 0], landmarks[corner_b, 0]))
    hi = float(max(landmarks[corner_a, 0], landmarks[corner_b, 0]))
    if hi - lo < 1e-9:
        return float("nan")
    return (iris_x - lo) / (hi - lo)


def eye_diagnostics(landmarks: np.ndarray) -> EyeDiagnostics:
    """Per-eye ratios and EARs for the debug view."""
    right = _eye_ratios(landmarks, R_IRIS, R_OUTER, R_INNER, R_UP, R_LO)
    left = _eye_ratios(landmarks, L_IRIS, L_INNER, L_OUTER, L_UP, L_LO)
    hx_r_img = _image_axis_ratio(landmarks, R_IRIS, R_INNER, R_OUTER)
    hx_l_img = _image_axis_ratio(landmarks, L_IRIS, L_INNER, L_OUTER)
    return EyeDiagnostics(
        hx_right=float("nan") if right is None else right[0],
        hx_left=float("nan") if left is None else left[0],
        hx_image=float(np.nanmean([hx_r_img, hx_l_img])),
        ear_right=eye_aspect_ratio(landmarks, R_UP, R_LO, R_INNER, R_OUTER),
        ear_left=eye_aspect_ratio(landmarks, L_UP, L_LO, L_INNER, L_OUTER),
    )
