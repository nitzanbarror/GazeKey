"""GazeKey — OS keystroke injection via pynput (spec Section 8).

Characters are injected as **text**, not as physical key positions, so Hebrew
(and anything else non-ASCII) does not depend on the OS keyboard layout being
switched. Enter, Backspace and Space go through real key codes so applications
treat them as the control keys they are.

The pynput objects are injected rather than imported at module scope, which
keeps the module importable — and testable — on machines with no display
server and no input permissions.
"""
from __future__ import annotations

from typing import Callable, List, Optional

#: actions that map to a real key code rather than typed text
SPECIAL_KEYS = ("enter", "backspace", "space", "tab", "esc")


class InjectionUnavailable(RuntimeError):
    """pynput could not take control of the keyboard."""


class KeystrokeInjector:
    """Types characters and presses control keys in the focused application."""

    def __init__(self, controller=None, keys=None,
                 log: Callable[[str], None] = print) -> None:
        if controller is None or keys is None:
            try:
                from pynput import keyboard as pynput_keyboard
            except Exception as exc:                      # pragma: no cover
                raise InjectionUnavailable(
                    f"pynput is unavailable: {exc}"
                ) from exc
            controller = controller or pynput_keyboard.Controller()
            keys = keys or pynput_keyboard.Key
        self._keyboard = controller
        self._keys = keys
        self._log = log
        self._shift_latched = False
        self.typed: List[str] = []       # what was sent, for the debug HUD

    # ------------------------------------------------------------------ shift
    @property
    def shift_latched(self) -> bool:
        """True when the next character will be capitalised."""
        return self._shift_latched

    def latch_shift(self) -> bool:
        """Toggle the one-shot shift; returns its new state."""
        self._shift_latched = not self._shift_latched
        return self._shift_latched

    # ------------------------------------------------------------------- send
    def type_text(self, text: str) -> None:
        """Type literal text, applying (and clearing) a latched shift."""
        if not text:
            return
        if self._shift_latched:
            text = text.upper()
            self._shift_latched = False
        else:
            text = text.lower() if len(text) == 1 and text.isalpha() else text
        self._keyboard.type(text)
        self.typed.append(text)

    def press_special(self, action: str) -> None:
        """Press one of :data:`SPECIAL_KEYS` by key code."""
        key = getattr(self._keys, action, None)
        if key is None:
            raise ValueError(f"{action!r} is not a pynput key")
        self._keyboard.press(key)
        self._keyboard.release(key)
        self.typed.append(f"<{action}>")

    def send(self, action: str, payload: str = "") -> bool:
        """Perform a key action. Returns True if anything was injected."""
        if action == "char":
            self.type_text(payload)
            return True
        if action in SPECIAL_KEYS:
            self.press_special(action)
            return True
        if action == "shift":
            self.latch_shift()
            return False          # latching is local state, nothing is injected
        return False              # page / lang / pause / recalibrate are app-level

    def close(self) -> None:
        """Release a latched shift so nothing is left held down."""
        self._shift_latched = False


def make_injector(log: Callable[[str], None] = print
                  ) -> Optional[KeystrokeInjector]:
    """Build an injector, or return ``None`` with an explanation logged.

    Keystroke injection is the one part of GazeKey that can be refused by the
    OS, so the app degrades to "everything works except typing" rather than
    failing to start.
    """
    try:
        return KeystrokeInjector(log=log)
    except InjectionUnavailable as exc:
        log(f"[GazeKey] keystroke injection disabled: {exc}")
        return None
