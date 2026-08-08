"""GazeKey — global quit hotkey.

A gaze user cannot alt-tab to a terminal and press Ctrl+C, so the app must be
killable from anywhere. This registers an OS-wide hotkey (default
``Ctrl+Alt+Q``) through pynput and calls back on the caller's behalf.

It is one of several independent exit routes on purpose: if the listener cannot
start — no permission, no display, pynput unavailable — the failure is reported
and the app keeps running with the on-screen routes intact, rather than
refusing to start.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

DEFAULT_HOTKEY = "<ctrl>+<alt>+q"
HOTKEY_LABEL = "Ctrl+Alt+Q"


class GlobalQuitHotkey:
    """Listens for a system-wide key combination and fires a callback once."""

    def __init__(
        self,
        on_quit: Callable[[], None],
        combination: str = DEFAULT_HOTKEY,
        log: Callable[[str], None] = print,
    ) -> None:
        self.combination = combination
        self._on_quit = on_quit
        self._log = log
        self._listener = None
        self._fired = threading.Event()
        self.error: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._listener is not None

    def start(self) -> bool:
        """Register the hotkey. Returns False (and logs) if it cannot be."""
        try:
            from pynput import keyboard as pynput_keyboard

            self._listener = pynput_keyboard.GlobalHotKeys(
                {self.combination: self._fire}
            )
            self._listener.start()
        except Exception as exc:                    # pragma: no cover - OS bound
            self.error = str(exc)
            self._listener = None
            self._log(f"[GazeKey] global quit hotkey unavailable: {exc}")
            return False
        return True

    def _fire(self) -> None:
        if self._fired.is_set():
            return
        self._fired.set()
        self._log(f"[GazeKey] {HOTKEY_LABEL} pressed — quitting.")
        self._on_quit()

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:                       # pragma: no cover
                pass
