"""GazeKey — MediaPipe model bundle management.

Only needed for the **Tasks** backend (see :mod:`vision.face_tracker`).
Recent MediaPipe wheels (0.10.3x, and every build for Python 3.13) ship the
Tasks API only; the legacy ``mp.solutions.face_mesh`` graph — which embedded
its own model — is gone. The Tasks ``FaceLandmarker`` therefore needs an
external ``face_landmarker.task`` bundle, which contains the same attention
(iris) mesh that ``refine_landmarks=True`` used to enable: 478 landmarks.

The bundle is cached once in ``~/.gazekey/models/`` and checksum-verified.
No video, frames or landmarks are ever written to disk — only this model file.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from typing import Callable

from config import models_dir

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def face_landmarker_path() -> str:
    """Where the model bundle is cached (it need not exist yet)."""
    return os.path.join(models_dir(), MODEL_FILENAME)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_face_landmarker(
    auto_download: bool = True,
    log: Callable[[str], None] = print,
) -> str:
    """Return the path to a verified ``face_landmarker.task``.

    Downloads it once (~3.7 MB) if missing and ``auto_download`` is True.

    Raises:
        FileNotFoundError: model missing and downloading is disabled.
        RuntimeError: the download failed or the checksum did not match.
    """
    path = face_landmarker_path()
    if os.path.exists(path) and _sha256(path) == MODEL_SHA256:
        return path

    if os.path.exists(path):  # corrupt / partial download
        os.remove(path)
    if not auto_download:
        raise FileNotFoundError(
            f"MediaPipe model bundle missing: {path}\n"
            f"Download it manually from {MODEL_URL}"
        )

    log(f"[GazeKey] downloading MediaPipe face landmarker model (~3.7 MB)\n"
        f"          from {MODEL_URL}\n"
        f"          to   {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except Exception as exc:  # network down, proxy, ...
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Could not download the model bundle: {exc}") from exc

    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError(
            f"Model checksum mismatch (got {digest}, expected {MODEL_SHA256})"
        )
    os.replace(tmp, path)
    log("[GazeKey] model ready.")
    return path
