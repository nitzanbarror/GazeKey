"""GazeKey — the application. One command, one flow, no terminal needed.

    python main.py

Launch → 5-second setup check → calibration → validation gate → keyboard.
(The aiming drill sits behind ``--practice``; the setup check is skippable with
``--skip-setup-check`` and never blocks — a failed one offers to calibrate
anyway.)

**Every launch calibrates.** There is no "use the saved one?" question, because
for this user population the honest answer is almost always no: the head is
re-seated on its support between sessions, and a calibration measured against
last night's head position is stale in a way the user cannot see — it just
feels like the keyboard got worse. Calibration is still saved to disk (the M5
touch-up and recalibration write to it, and it is what ``--use-saved`` reads),
but the saved file is a development convenience, not the default path.

Every step is answerable by looking: the calibration, the practice drill, the
keyboard and the quit confirmation are all gaze-driven, and there are four
independent ways out (see :func:`exit_routes`) so a gaze user is never trapped
in an app they cannot close.

``calibrate.py`` still exists as a development tool for working on calibration
alone; users do not need it.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from calibrate import report_camera_failure, resolve_pacing, wait_for_camera
from config import (
    calibration_path,
    load_config,
    resolve_tuning,
    suggested_fixation_dispersion_px,
)
from gaze.calibration import CalibrationModel
from gaze.calibration_session import CalibrationSession
from gaze.drift import DriftMonitor, TouchUpSession
from gaze.region import attach_region, full_screen_region, read_region
from gaze.setup_check import MIN_HY_SPAN, SetupCheckSession
from interaction.controller import DwellController, DwellSettings
from interaction.diagnostics import TypingDiagnostics
from interaction.hotkey import HOTKEY_LABEL, GlobalQuitHotkey
from interaction.injector import make_injector
from interaction.layouts import MAX_HEIGHT_RATIO, build_keyboard, keyboard_region
from interaction.practice import PracticeSession, hit_radius_for
from interaction.prediction import WordPredictor
from ui.calibration_screen import CalibrationScreen
from ui.choice_screen import ChoiceScreen
from ui.exit_button import ExitButton
from ui.overlay import KeyboardOverlay
from ui.practice_screen import PracticeScreen
from ui.setup_check_screen import SetupCheckScreen
from ui.touchup_screen import TouchUpScreen
from vision.pipeline import GazePipeline

STARTUP_TIMEOUT_S = 8.0


def user_words_path() -> str:
    import os

    from config import app_dir

    return os.path.join(app_dir(), "user_words.json")


def exit_routes() -> list[str]:
    """Every way out of the app, for the startup banner."""
    return [
        f"{HOTKEY_LABEL}          quit from anywhere, even mid-typing",
        "Quit key             on the keyboard (hold 2 s, then confirm by looking)",
        "X button             top-right corner of the keyboard (clickable)",
        "Ctrl+C               in this terminal",
    ]


def print_banner(routes: list[str]) -> None:
    print("\n" + "=" * 68)
    print("  GazeKey - how to get out")
    for route in routes:
        print(f"    {route}")
    print("=" * 68 + "\n")


class GazeKeyApp:
    """Owns the whole flow and the objects that outlive each screen."""

    def __init__(self, app: QApplication, args: argparse.Namespace) -> None:
        self.app = app
        self.args = args
        self.config = load_config()
        self.camera_index = (args.camera if args.camera is not None
                             else int(self.config["camera_index"]))
        geometry = app.primaryScreen().geometry()
        self.screen_size = (geometry.width(), geometry.height())
        self.geometry = geometry
        self.region = self.calibration_region()

        self.tuning = resolve_tuning(self.config, args)
        self.pipeline: Optional[GazePipeline] = None
        self.typing: Optional[TypingDiagnostics] = None
        self.injector = None
        self.predictor: Optional[WordPredictor] = None
        self.hotkey: Optional[GlobalQuitHotkey] = None
        self.exit_button: Optional[ExitButton] = None
        self.screen = None                 # whichever screen is showing
        self.overlay: Optional[KeyboardOverlay] = None
        self.drift: Optional[DriftMonitor] = None
        self.model: Optional[CalibrationModel] = None
        self.error_px = float("nan")
        self.exit_code = 0
        self._quitting = False

    # ----------------------------------------------------------------- region
    def calibration_region(self):
        """The screen area to calibrate over (spec 5.6).

        The keyboard is everything the user actually selects, so that is what
        the nine dots span — spreading them over the whole display would spend
        most of a webcam's limited accuracy on places nobody looks.
        ``--cal-region full`` restores whole-screen calibration for comparison.
        """
        if self.args.cal_region == "full":
            return full_screen_region(self.screen_size)
        # qwerty-tall's geometry depends only on the height ratio, so the
        # region matches the board exactly. The error-sized layouts cannot be
        # measured before the calibration exists, so bound them instead.
        ratio = (float(self.args.height_ratio)
                 if self.args.layout == "qwerty-tall" else MAX_HEIGHT_RATIO)
        return keyboard_region(self.screen_size, ratio)

    # ------------------------------------------------------------------ setup
    def start(self) -> int:
        print(f"[GazeKey] screen {self.screen_size[0]}x{self.screen_size[1]}, "
              f"camera {self.camera_index}")
        print(f"[GazeKey] calibrating over the "
              f"{self.region.describe(self.screen_size)}")
        self.pipeline = GazePipeline(
            camera_index=self.camera_index,
            backend=self.args.backend,
            fixation_window_ms=float(self.config["fixation_window_ms"]),
            fixation_dispersion_px=float(self.config["fixation_dispersion_px"]),
            tracking_hold_ms=float(self.config["tracking_hold_ms"]),
            min_cutoff=self.tuning["min_cutoff"],
        )
        self.pipeline.start()
        if self.pipeline.last_error:
            print(f"[GazeKey] ERROR: {self.pipeline.last_error}")
            return 2
        print(f"[GazeKey] mediapipe backend: {self.pipeline.backend_name}")
        if not wait_for_camera(self.pipeline, self.args.startup_timeout):
            report_camera_failure(self.pipeline, self.camera_index,
                                  self.args.startup_timeout, command="main.py")
            return 2

        self.hotkey = GlobalQuitHotkey(self.request_quit)
        routes = exit_routes()
        if not self.hotkey.start():
            routes[0] = f"{HOTKEY_LABEL}          UNAVAILABLE on this machine"
        print_banner(routes)

        self.predictor = WordPredictor(path=user_words_path())
        self.injector = make_injector()

        self._start_calibration_or_saved()
        return None                        # the event loop takes over

    def _start_calibration_or_saved(self) -> None:
        """Calibrate — always, unless the dev-only ``--use-saved`` says not to."""
        if self.args.use_saved:
            saved = CalibrationModel.load(calibration_path(), self.screen_size,
                                          self.camera_index)
            if saved is not None:
                self.model = saved
                self.error_px = saved.validation_error_px
                stored = read_region(calibration_path())
                if stored is not None:
                    self.region = stored     # the area that error applies to
                print(f"[GazeKey] --use-saved: reusing the stored calibration "
                      f"({self.error_px:.1f} px over the "
                      f"{self.region.describe(self.screen_size)}). "
                      f"Not the normal path.")
                self.after_calibration()
                return
            print("[GazeKey] --use-saved: nothing stored for this screen and "
                  "camera - calibrating.")
        self.run_setup_check()

    # ------------------------------------------------------- the setup check
    def run_setup_check(self) -> None:
        """Two dots, five seconds: is this sitting worth calibrating? (5.1c)

        A camera below eye height is visible *before* the nine dots and
        invisible during them — the user only finds out forty seconds later,
        as a verdict they cannot interpret. This measures the one number that
        predicts it and refuses to spend their patience on a sitting that
        cannot work. It never blocks: the failure screen offers to calibrate
        anyway.
        """
        if self.args.skip_setup_check:
            print("[GazeKey] --skip-setup-check: going straight to the dots.")
            self.run_calibration()
            return

        session = SetupCheckSession(
            self.screen_size, region=self.region,
            min_hy_span=float(self.config.get(
                "setup_check_min_hy_span", MIN_HY_SPAN)))
        print(f"[GazeKey] setup check: 2 targets, ~{session.budget_s:.0f} s, "
              f"need an hy span of {session.min_hy_span:.3f} "
              f"(--skip-setup-check to skip).")
        screen = SetupCheckScreen(session, self.pipeline)

        def finished(result) -> None:
            if result is None:
                print("[GazeKey] setup check cancelled.")
                self.request_quit(code=1)
                return
            if not result.passed:
                print(f"[GazeKey] WARNING: {result.text[0]} - "
                      f"{result.measurement_line()}. Calibrating anyway.")
            self.run_calibration()

        screen.finished.connect(finished)
        self._show(screen)

    # ------------------------------------------------------------- the flow
    def _show(self, widget) -> None:
        previous, self.screen = self.screen, widget
        if previous is not None:
            previous.close()
        widget.showFullScreen()

    def run_calibration(self, on_done=None, cancel_quits: bool = True) -> None:
        """Run a full 9-point session.

        Args:
            on_done: what to do once a model exists. Defaults to the practice
                drill; an in-place recalibration passes a callback that goes
                straight back to the keyboard instead.
            cancel_quits: at startup there is nowhere to go back to, so Esc
                ends the app; from the keyboard it just returns to typing.
        """
        pacing = resolve_pacing(self.config, self.args)
        session = CalibrationSession(
            self.screen_size, camera_index=self.camera_index,
            fixed_head=bool(self.config.get("fixed_head", True)),
            region=self.region, **pacing)
        print("[GazeKey] calibration: head-rest mode, "
              f"settle {pacing['settle_ms']:.0f} ms, "
              f"{pacing['collect_samples']} samples/point")
        screen = CalibrationScreen(session, self.pipeline)
        finish = on_done or self.after_calibration

        def finished(result) -> None:
            if result is None:
                print("[GazeKey] calibration cancelled.")
                if cancel_quits:
                    self.request_quit(code=1)
                else:
                    finish()               # back to typing on the old model
                return
            print(result.diagnostics().console_report())
            result.save(calibration_path())        # model + its region
            self.model = result.model
            self.error_px = result.error_px
            print(f"[GazeKey] calibration {result.verdict}: "
                  f"{result.error_px:.1f} px in-region (saved)")
            finish()

        screen.finished.connect(finished)
        self._show(screen)

    def after_calibration(self) -> None:
        """Install the model and go to the keyboard — or the drill, if asked.

        The aiming drill is **opt-in** (``--practice``). It measures rather
        than teaches, and putting a ten-target test between someone and the
        thing they opened the app to do is a tax on every launch.
        """
        self.pipeline.set_model(self.model)
        self._report_fixation()
        if not self.args.practice:
            self.run_keyboard()
            return
        radius = hit_radius_for(self.error_px)
        session = PracticeSession(self.screen_size, radius,
                                  targets=self.args.practice_targets,
                                  region=self.region)
        screen = PracticeScreen(
            session, self.pipeline,
            sound=self.app.beep if bool(self.config["sound_feedback"]) else None)
        print(f"[GazeKey] practice: {self.args.practice_targets} targets in the "
              f"{session.region.name} region, hit radius {radius:.0f} px "
              f"(Esc skips).")

        def finished(stats) -> None:
            if stats is None:
                print("[GazeKey] practice skipped.")
            else:
                for line in stats.summary_lines():
                    print(f"[GazeKey] practice: {line}")
            self.run_keyboard()

        screen.finished.connect(finished)
        self._show(screen)

    def run_keyboard(self, state: Optional[dict] = None) -> None:
        """Build and show the keyboard, optionally restoring a typing session."""
        keyboard = build_keyboard(
            self.screen_size, self.error_px,
            language=str(self.config["language"]),
            layout=self.args.layout,
            height_ratio=float(self.args.height_ratio),
        )
        print(f"[GazeKey] keyboard: {keyboard.summary()}")
        if keyboard.note:
            print(f"[GazeKey] WARNING: {keyboard.note}")
        self._check_region_covers(keyboard)

        settings = DwellSettings.from_config(self.config)
        if self.args.dwell is not None:
            settings.dwell_s = float(self.args.dwell)
        settings.hysteresis_margin = self.tuning["hysteresis_margin"]
        settings.grace_ms = self.tuning["grace_ms"]
        controller = DwellController(keyboard, settings)

        self.typing = None
        if self.args.debug_typing:
            self.typing = TypingDiagnostics(
                controller, key_size=keyboard.smallest_key(),
                interval_s=float(self.args.debug_interval))
            print(f"[GazeKey] --debug-typing: reporting every "
                  f"{self.args.debug_interval:.0f} s. Tuning: hysteresis "
                  f"{settings.hysteresis_margin:.2f}, grace "
                  f"{settings.grace_ms:.0f} ms, min-cutoff "
                  f"{self.tuning['min_cutoff']:.2f}")
        self.drift = DriftMonitor(
            validation_error_px=self.error_px,
            configured_offset_px=float(self.config.get("drift_offset_px", 0)),
        )
        overlay = KeyboardOverlay(
            keyboard, controller, self.pipeline, self.injector,
            screen_size=self.screen_size,
            show_cursor=bool(self.config["show_gaze_cursor"])
            and not self.args.no_cursor,
            language=str(self.config["language"]),
            predictor=self.predictor,
            show_webcam=bool(self.config.get("show_webcam_preview", True))
            and not self.args.no_webcam,
            drift=self.drift,
            diagnostics=self.typing,
        )
        overlay.setGeometry(self.geometry)
        overlay.recalibrate_requested.connect(self.on_recalibrate)
        overlay.touchup_requested.connect(self.on_touchup)
        overlay.quit_requested.connect(self.confirm_quit)
        overlay.restore_session(state)

        if self.screen is not None and self.screen is not overlay:
            self.screen.close()
        self.screen = self.overlay = overlay
        overlay.show()
        self._place_exit_button(keyboard)

        print(f"[GazeKey] drift monitor: flags at "
              f"{self.drift.threshold_px:.0f} px of offset, or 3 quick "
              f"backspaces in a minute")
        print(f"[GazeKey] dwell {settings.dwell_s:.2f} s. Click into the app you "
              f"want to type into, then look at a key and hold.")

    def _check_region_covers(self, keyboard) -> None:
        """Say so if any key ended up outside the area that was calibrated.

        The region is computed before the board exists, so this is the check
        that the estimate was right — silently selecting keys the model was
        never fitted for is exactly the failure regions exist to prevent.
        """
        outside = [key.id for key in keyboard.keys
                   if key.selectable and not self.region.holds_rect(key.rect)]
        if outside:
            print(f"[GazeKey] WARNING: {len(outside)} key(s) sit outside the "
                  f"calibrated region ({', '.join(outside[:6])}"
                  f"{', ...' if len(outside) > 6 else ''}); aim there is "
                  f"extrapolated. Lower --height-ratio or use --cal-region full.")

    def _place_exit_button(self, keyboard) -> None:
        """One X button for the whole session, re-parked on the current board."""
        if self.exit_button is None:
            self.exit_button = ExitButton()
            self.exit_button.clicked_quit.connect(self.request_quit)
        self.exit_button.place_at((keyboard.rect[0] + keyboard.rect[2],
                                   keyboard.rect[1]))
        self.exit_button.show()

    # ------------------------------------------------- suspending the keyboard
    def _suspend_keyboard(self) -> None:
        """Put the keyboard *and its X* aside while a screen takes over.

        ``suspend()`` stops the overlay consuming gaze, not just showing it —
        two widgets draining the same pipeline queue would each get half the
        samples, and the hidden one would keep typing behind the screen.
        """
        if self.overlay is not None:
            self.overlay.suspend()
        if self.exit_button is not None:
            self.exit_button.hide()

    def _resume_keyboard(self, notice: str = "") -> None:
        """Bring both back. Every route out of a screen goes through here, so
        the clickable X can never be left hidden — losing it would quietly cost
        the user one of their four ways out."""
        if self.overlay is None:
            return
        self.screen = self.overlay
        self.overlay.resume()
        if self.exit_button is not None:
            self.exit_button.show()
        if notice:
            self.overlay.notice(notice)

    # -------------------------------------------------------- M5: drift repair
    def on_touchup(self) -> None:
        """One centre target, one translation correction, straight back to typing."""
        if self.model is None:
            return
        session = TouchUpSession(self.screen_size, region=self.region)
        screen = TouchUpScreen(
            session, self.pipeline,
            sound=self.app.beep if bool(self.config["sound_feedback"]) else None)
        print(f"[GazeKey] touch-up: one target, "
              f"{session.budget_s:.1f} s (Esc cancels).")

        def finished(result) -> None:
            screen.close()
            if result is None:
                print("[GazeKey] touch-up cancelled.")
                self._resume_keyboard("touch-up cancelled")
                return
            if result.accepted and session.apply_to(self.model):
                self.model.save(calibration_path())
                attach_region(calibration_path(), self.region)
                self.pipeline.set_model(self.model)      # also resets smoothing
                if self.drift is not None:
                    self.drift.reset()
                print(f"[GazeKey] touch-up applied: "
                      f"{result.dx:+.0f}, {result.dy:+.0f} px "
                      f"from {result.samples} samples (saved).")
            else:
                print(f"[GazeKey] touch-up rejected: {result.message()}")
            self._resume_keyboard(result.message())

        screen.finished.connect(finished)
        self._suspend_keyboard()
        self.screen = screen
        screen.showFullScreen()

    def on_recalibrate(self) -> None:
        """Full 9-point recalibration in place — then back to the same sentence."""
        print("[GazeKey] recalibrating in place; typed text is kept.")
        state = self.overlay.session_state() if self.overlay is not None else None
        self._suspend_keyboard()

        def back_to_typing() -> None:
            # Install the fit that was just measured — without this the session
            # would come back looking identical while still predicting from the
            # old model, which is the one failure the user could not diagnose.
            self.pipeline.set_model(self.model)
            self._report_fixation()
            self.overlay = None            # rebuilt at the new key size
            self.run_keyboard(state)
            self._resume_keyboard(f"recalibrated: {self.error_px:.0f} px")

        self.run_calibration(on_done=back_to_typing, cancel_quits=False)

    # ------------------------------------------------------------------- exits
    def confirm_quit(self) -> None:
        """The Quit key asks first — a mis-dwell should not end the session."""
        screen = ChoiceScreen(
            "Quit GazeKey?", [("yes", "YES, QUIT"), ("no", "NO, KEEP TYPING")],
            self.screen_size, self.pipeline,
            DwellSettings.from_config(self.config),
            subtitle="Look at an answer and hold.",
            region=self.region,          # answerable at the calibrated accuracy
        )

        def answered(choice) -> None:
            screen.close()
            if choice == "yes":
                self.request_quit()
            else:
                self._resume_keyboard()

        screen.chosen.connect(answered)
        self._suspend_keyboard()
        self.screen = screen
        screen.showFullScreen()

    def request_quit(self, code: int = 0) -> None:
        if self._quitting:
            return
        self._quitting = True
        self.exit_code = code
        print("[GazeKey] shutting down.")
        self.app.quit()

    def shutdown(self) -> None:
        self.report_typing()
        if self.hotkey is not None:
            self.hotkey.stop()
        if self.exit_button is not None:
            self.exit_button.close()
        if self.overlay is not None and self.overlay is not self.screen:
            self.overlay.close()          # hidden behind a fullscreen screen
        if self.screen is not None:
            self.screen.close()
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.injector is not None:
            self.injector.close()
        if self.predictor is not None:
            self.predictor.save()

    def report_typing(self) -> None:
        """Print the --debug-typing summary once, on the way out."""
        if self.typing is None:
            return
        typing, self.typing = self.typing, None
        typing.close(now=time.time())
        for line in typing.summary_lines():
            print(line)

    def _report_fixation(self) -> None:
        active = self.pipeline.fixation_dispersion_px
        print(f"[GazeKey] fixation: dispersion <= {active:.0f} px over "
              f"{self.pipeline.fixation_window_ms:.0f} ms")
        suggested = suggested_fixation_dispersion_px(self.error_px)
        if abs(suggested - active) >= 15.0:
            print(f"          your {self.error_px:.0f} px accuracy suggests "
                  f"{suggested:.0f} px (fixation_dispersion_px in config)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GazeKey — gaze keyboard")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "legacy", "tasks"),
                        default="auto")
    parser.add_argument("--use-saved", action="store_true",
                        help="DEV ONLY: skip calibration and load the saved "
                             "one, for working on the UI. Every real launch "
                             "calibrates, because a stored calibration goes "
                             "stale as soon as the head is re-seated.")
    parser.add_argument("--practice", action="store_true",
                        help="run the aiming drill between calibration and the "
                             "keyboard (opt-in; the default goes straight to "
                             "typing)")
    parser.add_argument("--practice-targets", type=int, default=10)
    parser.add_argument("--skip-setup-check", action="store_true",
                        help="skip the 5-second camera check before the nine "
                             "dots (it measures whether the camera can see "
                             "your eyes move up and down at all)")
    parser.add_argument("--cal-region", choices=("keyboard", "full"),
                        default="keyboard",
                        help="what the 9 calibration dots span: the keyboard "
                             "area plus a small margin (default), or the whole "
                             "screen")
    parser.add_argument("--dwell", type=float, default=None,
                        help="dwell time in seconds (spec allows 0.5-2.0)")
    parser.add_argument("--layout", choices=("qwerty-tall", "paged", "auto"),
                        default="qwerty-tall")
    parser.add_argument("--height-ratio", type=float, default=2.0 / 3.0,
                        help="how much screen height the keyboard uses")
    parser.add_argument("--no-cursor", action="store_true")
    parser.add_argument("--no-webcam", action="store_true")

    tuning = parser.add_argument_group(
        "typing stability",
        "knobs for unstable selection; each falls back to config, then to the "
        "default, so an absent flag changes nothing"
    )
    tuning.add_argument("--hysteresis", type=float, default=None, metavar="F",
                        help="the focused key keeps ownership over a region "
                             "grown by this fraction per side (default 0.25); "
                             "a neighbour has to win the point from outside it")
    tuning.add_argument("--min-cutoff", type=float, default=None, metavar="HZ",
                        help="One-Euro min_cutoff (default 1.0); lower is "
                             "smoother and laggier")
    tuning.add_argument("--grace-ms", type=float, default=None, metavar="MS",
                        help="how long a dwell survives losing its key "
                             "(default 200), whether the gaze left every key "
                             "or a neighbour stole focus; it decays over this "
                             "and resumes if the gaze comes back")
    tuning.add_argument("--debug-typing", action="store_true",
                        help="log focus churn, dwell-loss causes, gaze "
                             "std-dev per axis and fixation duty cycle while "
                             "typing, then a summary on quit")
    tuning.add_argument("--debug-interval", type=float, default=5.0,
                        metavar="S", help="seconds between debug lines")
    parser.add_argument("--settle-ms", type=float, default=None)
    parser.add_argument("--collect-samples", type=int, default=None)
    parser.add_argument("--collect-max-s", type=float, default=None)
    parser.add_argument("--slow", action="store_true",
                        help="gentler calibration pacing")
    parser.add_argument("--startup-timeout", type=float, default=STARTUP_TIMEOUT_S)
    return parser


def run(args: argparse.Namespace) -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    gazekey = GazeKeyApp(app, args)
    early = gazekey.start()
    if early is not None:
        gazekey.shutdown()
        return early
    try:
        app.exec_()
    except KeyboardInterrupt:
        print("\n[GazeKey] interrupted.")
    finally:
        gazekey.shutdown()
    return gazekey.exit_code


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
