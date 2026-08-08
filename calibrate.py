"""GazeKey — Milestone 2 entry point: calibration session + gaze-dot demo.

Run:
    python calibrate.py                 # calibrate, then show the gaze dot
    python calibrate.py --use-saved     # skip calibration if a saved one fits
    python calibrate.py --camera 1

Flow:
    head sweep (stage A)  ->  9 points (stage B)  ->  3 validation targets
    ->  results screen (measured error + verdict)  ->  gaze-dot demo

The model is saved to ``~/.gazekey/calibration.json`` and is only reloaded for
the same screen resolution and camera index (spec 5.4).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from config import (
    SLOW_PRESET,
    calibration_path,
    load_config,
    suggested_fixation_dispersion_px,
)
from gaze.calibration import CalibrationModel
from gaze.calibration_session import CalibrationSession
from gaze.region import full_screen_region
from interaction.practice import PracticeSession, hit_radius_for
from ui.calibration_screen import CalibrationScreen
from ui.gaze_demo import GazeDotDemo
from ui.practice_screen import PracticeScreen
from vision.pipeline import STATE_NO_CAMERA, GazePipeline

STARTUP_TIMEOUT_S = 8.0


def wait_for_camera(pipeline: GazePipeline, timeout: float) -> bool:
    """Block until the pipeline produces a frame that actually came from the camera."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for sample in pipeline.drain():
            if sample.state != STATE_NO_CAMERA:
                return True
        time.sleep(0.02)
    return False


def report_camera_failure(pipeline: GazePipeline, index: int, timeout: float,
                          command: str = "calibrate.py") -> None:
    print(
        f"\n[GazeKey] ERROR: camera {index} produced no frames within {timeout:.0f} s."
        f"\n          last error: {pipeline.last_error or 'none reported'}"
        f"\n"
        f"\n          Things to check:"
        f"\n            * another app (Zoom, Teams, the Camera app) may be holding"
        f"\n              the webcam - close it and try again"
        f"\n            * try another index:  python {command} --camera 1"
        f"\n            * Windows: Settings > Privacy & security > Camera >"
        f"\n              'Let desktop apps access your camera' must be on\n"
    )


def resolve_fixed_head(config: dict, args: argparse.Namespace) -> bool:
    """Head-rest mode unless the user explicitly asks for the head sweep."""
    if getattr(args, "with_head_sweep", False):
        return False
    if getattr(args, "fixed_head", False):
        return True
    return bool(config.get("fixed_head", True))


def resolve_region(screen_size, args: argparse.Namespace):
    """Which area to calibrate over.

    Defaults to ``full`` here, unlike ``main.py``: this tool ends at a
    free-form gaze demo over the whole display, so a keyboard-shaped region
    would leave most of that demo showing extrapolation.
    """
    from interaction.layouts import DEFAULT_HEIGHT_RATIO, keyboard_region

    if getattr(args, "cal_region", "full") == "keyboard":
        return keyboard_region(screen_size, DEFAULT_HEIGHT_RATIO)
    return full_screen_region(screen_size)


def resolve_pacing(config: dict, args: argparse.Namespace) -> dict:
    """Merge pacing settings: config defaults < --slow preset < explicit flags."""
    pacing = {
        "settle_ms": float(config["calibration_settle_ms"]),
        "collect_samples": int(config["calibration_collect_samples"]),
        "collect_max_s": float(config["calibration_collect_max_s"]),
    }
    if getattr(args, "slow", False):
        pacing["settle_ms"] = float(SLOW_PRESET["calibration_settle_ms"])
        pacing["collect_samples"] = int(SLOW_PRESET["calibration_collect_samples"])
    if getattr(args, "settle_ms", None) is not None:
        pacing["settle_ms"] = float(args.settle_ms)
    if getattr(args, "collect_samples", None) is not None:
        pacing["collect_samples"] = int(args.collect_samples)
    if getattr(args, "collect_max_s", None) is not None:
        pacing["collect_max_s"] = float(args.collect_max_s)
    return pacing


def report_fixation_settings(pipeline: GazePipeline, error_px: float,
                             explicit: bool = False) -> None:
    """Print the active I-DT settings and, if it differs, what the measured
    accuracy suggests instead (spec Section 6 / config.DISPERSION_RATIO)."""
    active = pipeline.fixation_dispersion_px
    print(f"[GazeKey] fixation: dispersion <= {active:.0f} px over "
          f"{pipeline.fixation_window_ms:.0f} ms, "
          f"hold {pipeline.tracking_hold_s * 1000:.0f} ms")
    if explicit:
        return
    suggested = suggested_fixation_dispersion_px(error_px)
    if abs(suggested - active) >= 15.0:
        print(f"          your {error_px:.0f} px accuracy suggests "
              f"{suggested:.0f} px - run with "
              f"--fixation-dispersion {suggested:.0f} to try it.")


def load_saved_model(screen_size, camera_index: int) -> Optional[CalibrationModel]:
    """Load a stored calibration if it matches this screen and camera (spec 5.4)."""
    path = calibration_path()
    model = CalibrationModel.load(path, screen_size, camera_index)
    if model is not None:
        print(f"[GazeKey] found a saved calibration ({model.validation_error_px:.0f} px, "
              f"{screen_size[0]}x{screen_size[1]}, camera {camera_index}).")
    return model


def run(args: argparse.Namespace) -> int:
    config = load_config()
    index = args.camera if args.camera is not None else int(config["camera_index"])

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    # The calibration window closes before the demo window opens; without this
    # Qt would treat that gap as "last window closed" and end the event loop.
    app.setQuitOnLastWindowClosed(False)
    geometry = app.primaryScreen().geometry()
    screen_size = (geometry.width(), geometry.height())
    print(f"[GazeKey] screen {screen_size[0]}x{screen_size[1]}, camera {index}")

    saved = load_saved_model(screen_size, index)
    if args.use_saved and saved is None:
        print("[GazeKey] no calibration matching this screen and camera — "
              "recalibrating.")

    print("[GazeKey] starting the vision pipeline ...")
    pipeline = GazePipeline(
        camera_index=index,
        backend=args.backend,
        fixation_window_ms=float(config["fixation_window_ms"]),
        fixation_dispersion_px=(
            args.fixation_dispersion
            if args.fixation_dispersion is not None
            else float(config["fixation_dispersion_px"])
        ),
        tracking_hold_ms=float(config["tracking_hold_ms"]),
    )
    pipeline.start()
    if pipeline.last_error:
        print(f"[GazeKey] ERROR: {pipeline.last_error}")
        pipeline.stop()
        return 2
    print(f"[GazeKey] mediapipe backend: {pipeline.backend_name}")

    if not wait_for_camera(pipeline, args.startup_timeout):
        report_camera_failure(pipeline, index, args.startup_timeout)
        pipeline.stop()
        return 2
    print("[GazeKey] camera streaming.")

    state: dict = {"model": None, "error_px": float("nan"), "exit": 0}

    def show_demo() -> None:
        demo = GazeDotDemo(pipeline, screen_size, state["error_px"])
        demo.closed.connect(app.quit)
        demo.showFullScreen()
        state["demo"] = demo   # keep a reference alive
        print("[GazeKey] gaze demo running - press q or Esc to exit.")

    def show_practice() -> None:
        """Aiming drill between the results screen and the free-form demo."""
        radius = hit_radius_for(state["error_px"])
        session = PracticeSession(screen_size, radius, targets=args.practice_targets)
        screen = PracticeScreen(session, pipeline,
                                sound=app.beep if bool(config["sound_feedback"])
                                else None)

        def on_practice_finished(stats) -> None:
            if stats is None:
                print("[GazeKey] practice skipped.")
            else:
                for line in stats.summary_lines():
                    print(f"[GazeKey] practice: {line}")
            screen.close()
            show_demo()

        screen.finished.connect(on_practice_finished)
        screen.showFullScreen()
        state["practice"] = screen
        print(f"[GazeKey] practice: {args.practice_targets} targets, "
              f"hit radius {radius:.0f} px, hold {session.hold_s:.1f} s "
              f"(Esc to skip).")

    def after_calibration(practice: bool) -> None:
        pipeline.set_model(state["model"])
        report_fixation_settings(pipeline, state["error_px"],
                                 explicit=args.fixation_dispersion is not None)
        show_practice() if practice else show_demo()

    if args.use_saved and saved is not None:
        state["model"] = saved
        state["error_px"] = saved.validation_error_px
        after_calibration(practice=args.practice)
    else:
        pacing = resolve_pacing(config, args)
        fixed_head = resolve_fixed_head(config, args)
        print(f"[GazeKey] head mode: "
              + ("fixed (head rest) - skipping the head sweep"
                 if fixed_head else "free - head sweep first"))
        print(f"[GazeKey] pacing: settle {pacing['settle_ms']:.0f} ms, "
              f"{pacing['collect_samples']} samples/point, "
              f"max {pacing['collect_max_s']:.1f} s/point"
              + ("   (--slow preset)" if args.slow else ""))
        region = resolve_region(screen_size, args)
        print(f"[GazeKey] {region.describe(screen_size)}")
        session = CalibrationSession(screen_size, camera_index=index,
                                     fixed_head=fixed_head, region=region,
                                     **pacing)
        screen = CalibrationScreen(session, pipeline)

        def on_finished(result: Optional[CalibrationSession]) -> None:
            if result is None:
                if session.is_finished:          # cancelled from the results screen
                    print(session.diagnostics().console_report())
                print("[GazeKey] calibration cancelled.")
                state["exit"] = 1
                screen.close()
                app.quit()
                return
            print(result.diagnostics().console_report())
            path = calibration_path()
            result.save(path)
            state["model"] = result.model
            state["error_px"] = result.error_px
            print(f"[GazeKey] calibration {result.verdict}: "
                  f"{result.error_px:.1f} px  (saved to {path})")
            screen.close()
            after_calibration(practice=not args.no_practice)

        screen.finished.connect(on_finished)
        screen.showFullScreen()
        print("[GazeKey] calibration screen open — follow the on-screen prompts.")

    try:
        app.exec_()
    finally:
        pipeline.stop()
    return int(state["exit"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GazeKey calibration (M2)")
    parser.add_argument("--camera", type=int, default=None, help="camera index")
    parser.add_argument("--backend", choices=("auto", "legacy", "tasks"),
                        default="auto")
    parser.add_argument("--use-saved", action="store_true",
                        help="reuse a stored calibration and go straight to the demo")
    parser.add_argument("--startup-timeout", type=float, default=STARTUP_TIMEOUT_S,
                        help="seconds to wait for the first camera frame")
    parser.add_argument("--cal-region", choices=("full", "keyboard"),
                        default="full",
                        help="area the 9 dots span; 'full' here because this "
                             "tool ends at a whole-screen gaze demo")

    pacing = parser.add_argument_group(
        "pacing", "how long each target waits and how much it measures"
    )
    pacing.add_argument("--settle-ms", type=float, default=None,
                        help="pause before a target starts measuring "
                             "(default 700)")
    pacing.add_argument("--collect-samples", type=int, default=None,
                        help="valid samples wanted per target (default 45)")
    pacing.add_argument("--collect-max-s", type=float, default=None,
                        help="wall-clock cap per target (default 4)")
    pacing.add_argument("--slow", action="store_true",
                        help="gentler preset for first-time use: "
                             "1000 ms settle, 60 samples per target")
    head = parser.add_mutually_exclusive_group()
    head.add_argument("--fixed-head", action="store_true",
                      help="head rests on a support (default): skip the head "
                           "sweep and leave pose compensation off")
    head.add_argument("--with-head-sweep", action="store_true",
                      help="head moves freely: run the head-sweep stage first")
    practice = parser.add_argument_group(
        "practice", "aiming drill: hold a fixated gaze on a target to pop it"
    )
    practice.add_argument("--practice", action="store_true",
                          help="run the drill with --use-saved (it already runs "
                               "after a fresh calibration)")
    practice.add_argument("--no-practice", action="store_true",
                          help="skip the drill after calibrating")
    practice.add_argument("--practice-targets", type=int, default=10,
                          metavar="N", help="how many targets (default 10)")
    parser.add_argument("--fixation-dispersion", type=float, default=None,
                        metavar="PX",
                        help="I-DT dispersion threshold in px "
                             "(default 110, ~1.35x the calibration error)")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
