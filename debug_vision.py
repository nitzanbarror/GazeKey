"""GazeKey — Milestone 1 debug view.

Live OpenCV window showing the vision pipeline: landmarks, iris ratios,
head pose, blink flag and frame rate.

Run:
    python debug_vision.py                 # camera 0
    python debug_vision.py --camera 1
    python debug_vision.py --no-mirror     # show the raw camera image
    python debug_vision.py --startup-timeout 15   # for a slow-waking camera

The window is created before the camera is polled, and startup aborts with an
explanatory message if no frame arrives within --startup-timeout seconds, so a
busy or missing device never looks like a silent hang.

Keys:
    q / Esc  quit          m  toggle mirrored view
    l        toggle the full 478-landmark cloud
    r        reset the stability statistics

Features are always computed on the **un-mirrored** frame so the head-pose
sign conventions stay fixed; mirroring only affects what is drawn.
"""
from __future__ import annotations

import argparse
import collections
import time
from typing import Deque, Optional

import cv2
import numpy as np

from config import load_config
from gaze.features import L_INNER, L_IRIS, L_LO, L_OUTER, L_UP
from gaze.features import R_INNER, R_IRIS, R_LO, R_OUTER, R_UP
from vision.camera import CameraSource, Frame
from vision.face_tracker import FaceTracker, TrackingResult
from vision.head_pose import POSE_LANDMARK_IDS

WINDOW = "GazeKey — M1 vision debug"

#: give up if the camera has not delivered a single frame within this long
STARTUP_TIMEOUT_S = 5.0

WHITE = (255, 255, 255)
GREY = (150, 150, 150)
GREEN = (80, 230, 120)
RED = (60, 60, 240)
AMBER = (40, 190, 250)
CYAN = (240, 200, 60)
MAGENTA = (220, 80, 220)

EYE_POINTS = (R_INNER, R_OUTER, R_UP, R_LO, L_INNER, L_OUTER, L_UP, L_LO)


class _Stability:
    """Rolling spread of a signal — 'are the values steady while staring?'"""

    def __init__(self, maxlen: int = 45) -> None:
        self.buf: Deque[float] = collections.deque(maxlen=maxlen)

    def push(self, value: float) -> None:
        if np.isfinite(value):
            self.buf.append(float(value))

    def std(self) -> float:
        return float(np.std(self.buf)) if len(self.buf) > 4 else float("nan")

    def clear(self) -> None:
        self.buf.clear()


def _text(img, lines, origin=(12, 12), scale=0.5, line_h=20) -> None:
    """Draw a translucent HUD panel with coloured text lines."""
    x, y = origin
    width = 8 + max(
        int(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0])
        for t, _ in lines
    )
    panel = img[y : y + line_h * len(lines) + 10, x : x + width + 8]
    if panel.size:
        cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0, panel)
    for i, (t, colour) in enumerate(lines):
        cv2.putText(
            img, t, (x + 6, y + line_h * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA,
        )


def _bar(img, x, y, w, h, value, label, colour) -> None:
    """Horizontal 0..1 gauge with a marker at ``value``."""
    cv2.rectangle(img, (x, y), (x + w, y + h), GREY, 1)
    cv2.line(img, (x + w // 2, y), (x + w // 2, y + h), (90, 90, 90), 1)
    if np.isfinite(value):
        pos = int(x + np.clip(value, 0.0, 1.0) * w)
        cv2.rectangle(img, (pos - 2, y - 2), (pos + 2, y + h + 2), colour, -1)
    txt = f"{label} {value:.3f}" if np.isfinite(value) else f"{label}  --"
    cv2.putText(img, txt, (x + w + 8, y + h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def _draw_overlay(canvas, res: TrackingResult, mirrored: bool, show_cloud: bool):
    """Draw landmarks and the pose gizmo onto the (possibly mirrored) canvas."""
    if res.landmarks_px is None:
        return
    width = canvas.shape[1]

    def to_px(pt) -> tuple[int, int]:
        x, y = float(pt[0]), float(pt[1])
        if mirrored:
            x = width - 1 - x
        return int(round(x)), int(round(y))

    pts = res.landmarks_px
    if show_cloud:
        for p in pts:
            cv2.circle(canvas, to_px(p), 1, (70, 90, 70), -1)

    for idx in EYE_POINTS:
        cv2.circle(canvas, to_px(pts[idx]), 2, CYAN, -1)
    for idx in POSE_LANDMARK_IDS:
        cv2.circle(canvas, to_px(pts[idx]), 3, MAGENTA, -1)
    for iris in (R_IRIS, L_IRIS):
        centre = pts[iris].mean(axis=0)
        cv2.circle(canvas, to_px(centre), 3, GREEN, -1)
        for idx in iris[1:]:
            cv2.circle(canvas, to_px(pts[idx]), 1, GREEN, -1)


def _draw_axes(canvas, tracker: FaceTracker, res: TrackingResult, mirrored: bool):
    """Project the head-pose axis gizmo at the nose tip."""
    if res.pose is None:
        return
    height, width = canvas.shape[:2]
    axes = tracker.pose_estimator.project_axes(res.pose, (width, height))

    def to_px(pt) -> tuple[int, int]:
        x = width - 1 - float(pt[0]) if mirrored else float(pt[0])
        return int(round(x)), int(round(float(pt[1])))

    origin = to_px(axes[0])
    for end, colour in zip(axes[1:], ((80, 80, 255), (80, 255, 80), (255, 160, 80))):
        cv2.line(canvas, origin, to_px(end), colour, 2, cv2.LINE_AA)


def _show_splash(blank: np.ndarray, message: str) -> None:
    """Create the window immediately and paint a status message in it."""
    canvas = blank.copy()
    cv2.putText(canvas, message, (20, blank.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW, canvas)
    cv2.waitKey(1)   # forces the window manager to actually map the window


def _wait_for_first_frame(camera: CameraSource, timeout: float,
                          pump=None) -> tuple[Optional[Frame], str]:
    """Block until the camera delivers its first frame.

    Args:
        camera: a started :class:`~vision.camera.CameraSource`.
        timeout: seconds to wait before giving up.
        pump: called between polls to keep the UI alive; returning ``"quit"``
            aborts the wait.

    Returns:
        ``(frame, reason)`` where reason is ``"ok"``, ``"timeout"`` or ``"quit"``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = camera.read(timeout=0.1)
        if frame is not None:
            return frame, "ok"
        if pump is not None and pump() == "quit":
            return None, "quit"
    return None, "timeout"


def _print_startup_failure(index: int, camera: CameraSource, timeout: float) -> None:
    """Explain a startup timeout instead of leaving the user staring at a window."""
    print(
        f"\n[GazeKey] ERROR: camera {index} produced no frames within {timeout:.0f} s."
        f"\n          last camera error: {camera.last_error or 'none reported'}"
        f"\n"
        f"\n          Things to check:"
        f"\n            * another app (Zoom, Teams, the Camera app) may be holding"
        f"\n              the webcam - close it and try again"
        f"\n            * try another index:  python debug_vision.py --camera 1"
        f"\n            * Windows: Settings > Privacy & security > Camera >"
        f"\n              'Let desktop apps access your camera' must be on"
        f"\n            * a built-in privacy shutter or a disabled device driver"
        f"\n"
        f"\n          Raise the limit with --startup-timeout SECONDS if your"
        f"\n          camera is simply slow to wake up.\n"
    )


def run(args: argparse.Namespace) -> int:
    cfg = load_config()
    index = args.camera if args.camera is not None else int(cfg["camera_index"])

    print(f"[GazeKey] starting vision debug on camera {index} ...")
    tracker = FaceTracker(backend=args.backend)
    print(f"[GazeKey] mediapipe backend: {tracker.backend_name}")

    camera = CameraSource(index=index, width=args.width, height=args.height).start()
    mirrored, show_cloud = not args.no_mirror, False
    stab_hx, stab_hy = _Stability(), _Stability()
    stab_yaw, stab_pitch = _Stability(), _Stability()
    fps, last_t = 0.0, time.time()
    last_frame_index = -1
    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    try:
        # Open the window before waiting for the camera, so a slow or missing
        # device can never look like a silent hang.
        _show_splash(blank, f"opening camera {index} ...")
        print("[GazeKey] debug window open.")

        frame, reason = _wait_for_first_frame(
            camera, args.startup_timeout, pump=lambda: _handle_keys(cv2.waitKey(30))
        )
        if reason == "quit":
            print("[GazeKey] quit before the first frame arrived.")
            return 1
        if frame is None:
            _print_startup_failure(index, camera, args.startup_timeout)
            return 2
        height, width = frame.image.shape[:2]
        print(f"[GazeKey] first frame received ({width}x{height}) — streaming.")

        while True:
            frame = camera.read(timeout=0.5)
            if frame is None or frame.index == last_frame_index:
                canvas = blank.copy()
                msg = "camera disconnected - reconnecting..." if not camera.connected \
                    else "waiting for frames..."
                cv2.putText(canvas, msg, (20, args.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, canvas)
                if _handle_keys(cv2.waitKey(30)) == "quit":
                    break
                continue
            last_frame_index = frame.index

            res = tracker.process(frame.image, frame.timestamp)

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            f = res.features
            if f.valid:
                stab_hx.push(f.hx)
                stab_hy.push(f.hy)
                stab_yaw.push(f.yaw)
                stab_pitch.push(f.pitch)

            canvas = frame.image.copy()
            if mirrored:
                canvas = cv2.flip(canvas, 1)
            _draw_overlay(canvas, res, mirrored, show_cloud)
            _draw_axes(canvas, tracker, res, mirrored)

            d = res.diagnostics
            status = ("VALID", GREEN) if f.valid else (
                ("BLINK", AMBER) if res.blink else
                ("NO FACE", RED) if not res.face_found else ("INVALID", RED))
            lines = [
                (f"backend {tracker.backend_name}   pipeline {fps:4.1f} fps"
                 f"   capture {camera.fps:4.1f} fps", WHITE),
                (f"status  {status[0]}", status[1]),
                (f"hx {f.hx:6.3f}  hy {f.hy:6.3f}"
                 f"   (std {stab_hx.std():.4f} / {stab_hy.std():.4f})", WHITE),
                (f"yaw {f.yaw:6.1f}  pitch {f.pitch:6.1f}  roll {f.roll:6.1f}"
                 f"   (std {stab_yaw.std():.2f} / {stab_pitch.std():.2f})", WHITE),
                (f"EAR  R {d.ear_right:.3f}  L {d.ear_left:.3f}"
                 f"   blink={res.blink}" if d else "EAR   --", WHITE),
                (f"per-eye hx  R {d.hx_right:6.3f}  L {d.hx_left:6.3f}"
                 f"   image-axis {d.hx_image:6.3f}" if d else "per-eye hx  --", CYAN),
                ("q quit   m mirror   l landmark cloud   r reset stats", GREY),
            ]
            _text(canvas, lines)

            base_y = canvas.shape[0] - 74
            _bar(canvas, 14, base_y, 220, 12, f.hx, "hx", WHITE)
            _bar(canvas, 14, base_y + 24, 220, 12, f.hy, "hy", WHITE)
            _bar(canvas, 14, base_y + 48, 220, 12,
                 d.hx_image if d else float("nan"), "hx(image axis)", CYAN)

            cv2.imshow(WINDOW, canvas)
            action = _handle_keys(cv2.waitKey(1))
            if action == "quit":
                break
            if action == "mirror":
                mirrored = not mirrored
            elif action == "cloud":
                show_cloud = not show_cloud
            elif action == "reset":
                for s in (stab_hx, stab_hy, stab_yaw, stab_pitch):
                    s.clear()
    finally:
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


def _handle_keys(key: int) -> Optional[str]:
    if key in (27, ord("q")):
        return "quit"
    if key == ord("m"):
        return "mirror"
    if key == ord("l"):
        return "cloud"
    if key == ord("r"):
        return "reset"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="GazeKey M1 vision debug view")
    p.add_argument("--camera", type=int, default=None, help="camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--backend", choices=("auto", "legacy", "tasks"), default="auto")
    p.add_argument("--no-mirror", action="store_true",
                   help="show the raw (un-mirrored) camera image")
    p.add_argument("--startup-timeout", type=float, default=STARTUP_TIMEOUT_S,
                   help="seconds to wait for the first frame before giving up")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
