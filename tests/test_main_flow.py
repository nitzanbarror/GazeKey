"""Integrated-flow tests for main.py — one entry point, no terminal needed."""
import argparse

import pytest

import main as app_main


def parse(*argv):
    return app_main.build_parser().parse_args(list(argv))


# -------------------------------------------------------------------- the CLI
def test_the_defaults_need_no_arguments():
    args = parse()
    assert args.camera is None
    assert args.use_saved is False
    assert args.practice is False, "the drill is opt-in, not a launch tax"
    assert args.cal_region == "keyboard"
    assert args.layout == "qwerty-tall"
    assert args.height_ratio == pytest.approx(2 / 3)
    assert args.practice_targets == 10


def test_there_is_no_startup_question_left_to_ask():
    """A stored calibration is stale once the head is re-seated: no dialog."""
    with pytest.raises(SystemExit):
        parse("--recalibrate")             # removed: calibrating IS the default


def test_practice_is_opt_in_now():
    with pytest.raises(SystemExit):
        parse("--no-practice")             # removed: it is off by default
    assert parse("--practice").practice is True


def test_the_flow_can_be_steered_from_the_cli():
    assert parse("--use-saved").use_saved is True
    assert parse("--practice").practice is True
    assert parse("--cal-region", "full").cal_region == "full"
    assert parse("--layout", "paged").layout == "paged"
    assert parse("--no-webcam").no_webcam is True
    assert parse("--dwell", "1.4").dwell == pytest.approx(1.4)


def test_calibration_pacing_flags_are_available_here_too():
    """The user never has to run calibrate.py, so its options live here."""
    args = parse("--slow", "--settle-ms", "900")
    from calibrate import resolve_pacing
    from config import DEFAULTS

    pacing = resolve_pacing(DEFAULTS, args)
    assert pacing["settle_ms"] == 900.0
    assert pacing["collect_samples"] == 60          # from --slow


# ------------------------------------------------------------------- the flow
class StubApp:
    """Enough QApplication for GazeKeyApp's constructor."""

    def __init__(self, size=(1366, 768)):
        self._size = size
        self.quit_calls = 0

    def primaryScreen(self):                        # noqa: N802 - Qt API
        class Screen:
            def geometry(inner):
                class Rect:
                    def width(self):
                        return 1366

                    def height(self):
                        return 768
                return Rect()
        return Screen()

    def quit(self):
        self.quit_calls += 1

    def beep(self):
        pass


@pytest.fixture
def gazekey():
    return app_main.GazeKeyApp(StubApp(), parse())


def test_the_screen_size_comes_from_the_display(gazekey):
    assert gazekey.screen_size == (1366, 768)


def test_quit_is_idempotent(gazekey):
    gazekey.request_quit()
    gazekey.request_quit()
    gazekey.request_quit()
    assert gazekey.app.quit_calls == 1, "quitting twice must not double-fire"
    assert gazekey.exit_code == 0


def test_a_cancelled_calibration_exits_non_zero(gazekey):
    gazekey.request_quit(code=1)
    assert gazekey.exit_code == 1


def test_shutdown_is_safe_before_anything_started(gazekey):
    gazekey.shutdown()                              # must not raise


def test_user_words_live_beside_the_calibration():
    from config import app_dir

    assert app_main.user_words_path().startswith(app_dir())
    assert app_main.user_words_path().endswith("user_words.json")


# ------------------------------------------------------------ the exit banner
def test_every_exit_route_is_announced():
    routes = app_main.exit_routes()
    assert len(routes) == 4
    joined = " ".join(routes).lower()
    for expected in ("ctrl+alt+q", "quit key", "x button", "ctrl+c"):
        assert expected in joined


def test_the_camera_error_names_the_command_the_user_actually_ran(capsys):
    from calibrate import report_camera_failure

    class Stub:
        last_error = "camera 0 not available"

    report_camera_failure(Stub(), 0, 8.0, command="main.py")
    out = capsys.readouterr().out
    assert "python main.py --camera 1" in out
    assert "calibrate.py" not in out
