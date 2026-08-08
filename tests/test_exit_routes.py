"""Exit-route tests — a gaze user must never be trapped in the app.

Four independent ways out, so no single failure (no pynput permission, no
mouse, a mis-aimed dwell) can leave someone stuck staring at a keyboard they
cannot close.
"""
import numpy as np
import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QApplication

import main as app_main
from interaction.hotkey import DEFAULT_HOTKEY, HOTKEY_LABEL, GlobalQuitHotkey
from ui.exit_button import ExitButton


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


# ------------------------------------------------------- 1. the global hotkey
def test_the_hotkey_is_ctrl_alt_q():
    assert DEFAULT_HOTKEY == "<ctrl>+<alt>+q"
    assert HOTKEY_LABEL == "Ctrl+Alt+Q"


def test_pressing_the_hotkey_quits(monkeypatch):
    registered = {}

    class FakeHotKeys:
        def __init__(self, mapping):
            registered.update(mapping)
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

    monkeypatch.setattr("pynput.keyboard.GlobalHotKeys", FakeHotKeys)
    quits = []
    hotkey = GlobalQuitHotkey(lambda: quits.append(True), log=lambda _m: None)
    assert hotkey.start()
    assert hotkey.active

    registered[DEFAULT_HOTKEY]()
    assert quits == [True]
    hotkey.stop()
    assert not hotkey.active


def test_the_hotkey_only_fires_once(monkeypatch):
    registered = {}

    class FakeHotKeys:
        def __init__(self, mapping):
            registered.update(mapping)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr("pynput.keyboard.GlobalHotKeys", FakeHotKeys)
    quits = []
    GlobalQuitHotkey(lambda: quits.append(True), log=lambda _m: None).start()
    for _ in range(3):
        registered[DEFAULT_HOTKEY]()
    assert quits == [True], "a held hotkey must not quit repeatedly"


def test_a_hotkey_that_cannot_register_degrades_gracefully(monkeypatch):
    def explode(_mapping):
        raise OSError("no permission")

    monkeypatch.setattr("pynput.keyboard.GlobalHotKeys", explode)
    messages = []
    hotkey = GlobalQuitHotkey(lambda: None, log=messages.append)
    assert hotkey.start() is False, "must report failure, not raise"
    assert not hotkey.active
    assert hotkey.error and "no permission" in hotkey.error
    assert messages and "unavailable" in messages[0]


# --------------------------------------------------------- 2. the exit button
def test_the_exit_button_is_clickable_and_does_not_take_focus(qapp):
    button = ExitButton()
    try:
        flags = button.windowFlags()
        assert flags & Qt.WindowStaysOnTopHint
        assert flags & Qt.WindowDoesNotAcceptFocus
        assert flags & Qt.FramelessWindowHint
        assert button.focusPolicy() == Qt.NoFocus
        assert button.testAttribute(Qt.WA_ShowWithoutActivating)
        assert not button.testAttribute(Qt.WA_TransparentForMouseEvents), \
            "the X is the one thing that must receive clicks"
    finally:
        button.close()


def test_clicking_the_exit_button_asks_to_quit(qapp):
    button = ExitButton()
    quits = []
    button.clicked_quit.connect(lambda: quits.append(True))
    try:
        event = QMouseEvent(QMouseEvent.MouseButtonPress, QPoint(10, 10),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        button.mousePressEvent(event)
        assert quits == [True]
    finally:
        button.close()


def test_the_exit_button_parks_in_the_keyboard_corner(qapp):
    button = ExitButton(size=40)
    try:
        button.place_at((1366.0, 256.0))
        assert button.x() == 1366 - 40 - 10
        assert button.y() == 256 + 10
    finally:
        button.close()


def test_the_exit_button_draws_an_x(qapp):
    button = ExitButton(size=40)
    try:
        pixmap = QPixmap(40, 40)
        pixmap.fill(Qt.black)
        button.render(pixmap)
        image = pixmap.toImage().convertToFormat(4)
        frame = np.frombuffer(image.bits().asstring(image.byteCount()),
                              dtype=np.uint8).reshape(40, -1, 4)
        assert len(np.unique(frame.reshape(-1, 4), axis=0)) > 2
    finally:
        button.close()


# ------------------------------------------------- 3. the Quit key on the board
def test_the_quit_key_exists_and_needs_the_extended_dwell():
    from interaction.controller import DwellController, DwellSettings
    from interaction.layouts import EXTENDED_DWELL_ACTIONS, build_keyboard

    board = build_keyboard((1366, 768), 95.0)
    quit_key = board.find("fn.quit")
    assert quit_key is not None, "there must be a Quit key on the keyboard"
    assert quit_key.action in EXTENDED_DWELL_ACTIONS

    controller = DwellController(board, DwellSettings())
    assert controller.required_s(quit_key) == controller.settings.extended_dwell_s


def test_the_quit_key_asks_before_quitting(qapp):
    """It must not type, and it must not quit on its own — it asks."""
    from interaction.controller import DwellController, DwellSettings
    from interaction.injector import KeystrokeInjector
    from interaction.layouts import build_keyboard
    from ui.overlay import KeyboardOverlay
    from tests.test_injector import FakeController, FakeKeys
    from tests.test_overlay import ScriptedPipeline

    board = build_keyboard((1366, 768), 95.0)
    injector = KeystrokeInjector(controller=FakeController(), keys=FakeKeys)
    overlay = KeyboardOverlay(board, DwellController(board, DwellSettings()),
                              ScriptedPipeline(), injector,
                              screen_size=(1366, 768), show_webcam=False)
    asked = []
    overlay.quit_requested.connect(lambda: asked.append(True))
    try:
        overlay.pipeline.look_at(board.find("fn.quit").centre, 2.2, start=0.0)
        overlay.tick()
        assert asked == [True], "Quit should ask the app, not act alone"
        assert injector._keyboard.calls == [], "Quit must never type"
    finally:
        overlay.close()


def test_the_confirmation_offers_yes_and_no(qapp):
    from interaction.layouts import build_choice_board

    board = build_choice_board((1366, 768),
                               [("yes", "YES, QUIT"), ("no", "NO, KEEP TYPING")])
    assert [key.payload for key in board.keys] == ["yes", "no"]
    for key in board.keys:
        _, _, w, h = key.rect
        assert w > 400 and h > 150, "confirmation targets must be very large"


# -------------------------------------------------------------- 4. the banner
def test_the_banner_lists_every_exit_route(capsys):
    routes = app_main.exit_routes()
    app_main.print_banner(routes)
    printed = capsys.readouterr().out

    assert HOTKEY_LABEL in printed
    assert "Quit key" in printed
    assert "X button" in printed
    assert "Ctrl+C" in printed
    assert len(routes) == 4
    printed.encode("ascii")             # console-safe on any code page


def test_the_banner_says_when_the_hotkey_is_unavailable(capsys):
    routes = app_main.exit_routes()
    routes[0] = f"{HOTKEY_LABEL}          UNAVAILABLE on this machine"
    app_main.print_banner(routes)
    assert "UNAVAILABLE" in capsys.readouterr().out
