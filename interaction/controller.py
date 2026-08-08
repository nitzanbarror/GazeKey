"""GazeKey — dwell interaction state machine (spec Section 7).

Per focused key: ``IDLE -> FOCUS -> DWELLING -> ACTIVATED``.

The rules, all of which exist to stop a wandering gaze typing by accident:

* dwell advances **only** while the gaze is inside the key *and*
  :class:`~gaze.smoothing.FixationDetector` says the user is actually fixating;
* **hysteresis** — a focused key's hit region grows by 25% on each side, and a
  neighbour only steals focus once the gaze is inside its *unexpanded* region.
  Hit regions are gapless, so between two letters the neighbour wins outright;
  what the margin really buys is the board's outer edge and the non-selectable
  suggestion strip, where a slightly-off prediction would otherwise drop the
  dwell with nothing to hand it to;
* **grace** — leaving the region decays the accumulated dwell over 200 ms
  instead of throwing it away, so a momentary wobble is not punished;
* **refractory** — after a key fires it cannot fire again for 400 ms;
* **extended dwell** — PAUSE, RECALIBRATE and LANG need 2 s, not 1 s;
* an **invalid stream** (blink held too long, face lost) freezes the dwell
  where it is: it neither advances nor decays (spec Section 6).

While paused the machine keeps running and keeps giving feedback, but every
activation except PAUSE and RECALIBRATE comes back marked ``suppressed`` so the
app knows not to inject it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Tuple

from interaction.layouts import EXTENDED_DWELL_ACTIONS, Key, Keyboard

#: Keys that must keep working while the keyboard is paused. None of them types
#: anything, and each is something a paused user still has to be able to reach:
#: unpausing, fixing the aim that made them pause, and leaving. A paused
#: keyboard whose Quit key does nothing is a trap.
ALWAYS_LIVE_ACTIONS = frozenset({"pause", "recalibrate", "touchup", "quit"})


class DwellState(str, Enum):
    IDLE = "idle"
    FOCUS = "focus"
    DWELLING = "dwelling"
    ACTIVATED = "activated"


@dataclass
class DwellSettings:
    """Timings from config (spec Section 10)."""

    dwell_s: float = 1.0
    extended_dwell_s: float = 2.0
    hysteresis_margin: float = 0.25
    grace_ms: float = 200.0
    refractory_ms: float = 400.0

    @classmethod
    def from_config(cls, config: dict) -> "DwellSettings":
        return cls(
            dwell_s=float(config.get("dwell_time_s", 1.0)),
            extended_dwell_s=float(config.get("extended_dwell_s", 2.0)),
            hysteresis_margin=float(config.get("hysteresis_margin", 0.25)),
            grace_ms=float(config.get("grace_period_ms", 200)),
            refractory_ms=float(config.get("refractory_ms", 400)),
        )


@dataclass
class Activation:
    """A key that just completed its dwell."""

    key: Key
    timestamp: float
    suppressed: bool = False     # paused: feedback happened, injection must not


@dataclass
class DwellStats:
    """Why dwells complete or don't — the raw counts behind ``--debug-typing``.

    Purely observational: nothing here changes what the machine does. The three
    loss causes are mutually exclusive per event, so they add up to "times the
    user was trying to select something and did not".
    """

    frames: int = 0
    stream_lost: int = 0
    #: the focused key changed (including the first landing on a key)
    focus_changes: int = 0
    #: ...and it changed while a dwell was part-way through, discarding it
    focus_stolen: int = 0
    #: the fixation detector let go mid-dwell, so progress stalled
    fixation_dropped: int = 0
    #: the gaze left every key and the accumulated dwell decayed to nothing
    grace_expired: int = 0
    activations: int = 0

    @property
    def dwell_losses(self) -> int:
        return self.focus_stolen + self.fixation_dropped + self.grace_expired

    def snapshot(self) -> "DwellStats":
        return replace(self)

    def since(self, earlier: "DwellStats") -> "DwellStats":
        """This minus an earlier snapshot, for per-window reporting."""
        return DwellStats(**{
            name: getattr(self, name) - getattr(earlier, name)
            for name in (
                "frames", "stream_lost", "focus_changes", "focus_stolen",
                "fixation_dropped", "grace_expired", "activations",
            )
        })


class DwellController:
    """Drives key focus and dwell from a stream of gaze samples."""

    def __init__(
        self,
        keyboard: Keyboard,
        settings: Optional[DwellSettings] = None,
        page: int = 0,
    ) -> None:
        self.keyboard = keyboard
        self.settings = settings or DwellSettings()
        self.page = page
        self.paused = False

        self._focused: Optional[Key] = None
        self._accumulated = 0.0
        self._leaving = False
        self._leave_accumulated = 0.0
        self._last_t: Optional[float] = None
        self._refractory_until: Dict[str, float] = {}
        self._was_fixating = False
        #: observational only — see :class:`DwellStats`
        self.stats = DwellStats()

    # ------------------------------------------------------------------ state
    @property
    def focused_key(self) -> Optional[Key]:
        return self._focused

    @property
    def progress(self) -> float:
        """0..1 of the way to activating the focused key."""
        if self._focused is None:
            return 0.0
        return min(1.0, self._accumulated / self.required_s(self._focused))

    @property
    def state(self) -> DwellState:
        if self._focused is None:
            return DwellState.IDLE
        return DwellState.DWELLING if self._accumulated > 0 else DwellState.FOCUS

    def required_s(self, key: Key) -> float:
        """Dwell time for this key — longer for the modal function keys."""
        if key.action in EXTENDED_DWELL_ACTIONS:
            return self.settings.extended_dwell_s
        return self.settings.dwell_s

    def is_live(self, key: Key) -> bool:
        """Would activating this key actually do something right now?"""
        return not self.paused or key.action in ALWAYS_LIVE_ACTIONS

    # --------------------------------------------------------------- controls
    def set_page(self, page: int) -> None:
        if page != self.page:
            self.page = page % max(self.keyboard.pages, 1)
            self._clear_focus()

    def next_page(self) -> None:
        self.set_page(self.page + 1)

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def reset(self) -> None:
        self._clear_focus()
        self._refractory_until.clear()
        self._last_t = None

    def _clear_focus(self) -> None:
        self._focused = None
        self._accumulated = 0.0
        self._leaving = False

    # ------------------------------------------------------------------ input
    def update(
        self,
        x: float,
        y: float,
        is_fixating: bool,
        stream_valid: bool,
        now: float,
    ) -> Optional[Activation]:
        """Advance the machine by one sample; returns a key if one fired.

        A thin wrapper around :meth:`_advance` so the fixation edge can be
        tracked in one place instead of at every return.
        """
        self.stats.frames += 1
        activation = self._advance(x, y, is_fixating, stream_valid, now)
        self._was_fixating = bool(is_fixating and stream_valid)
        return activation

    def _advance(
        self,
        x: float,
        y: float,
        is_fixating: bool,
        stream_valid: bool,
        now: float,
    ) -> Optional[Activation]:
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now

        if not stream_valid:
            # Tracking lost: freeze rather than reset, so a blink mid-dwell
            # costs time but not progress (spec Section 6).
            self.stats.stream_lost += 1
            return None

        key = self._hit_test(x, y)
        if key is None:
            self._decay(dt)
            return None

        if self._focused is None or key.id != self._focused.id:
            self.stats.focus_changes += 1
            if self._accumulated > 0.0:
                self.stats.focus_stolen += 1
            self._focused = key
            self._accumulated = 0.0
            self._leaving = False
            return None                     # focus lands this frame, no dwell yet

        self._leaving = False
        if now < self._refractory_until.get(key.id, 0.0):
            self._accumulated = 0.0
            return None

        if not is_fixating:
            # Count the edge, not every frame: one wobble is one lost dwell.
            if self._accumulated > 0.0 and self._was_fixating:
                self.stats.fixation_dropped += 1
            return None                     # inside the key but the eye is moving

        self._accumulated += dt
        if self._accumulated >= self.required_s(key):
            self._accumulated = 0.0
            self._refractory_until[key.id] = now + self.settings.refractory_ms / 1000.0
            self.stats.activations += 1
            return Activation(key, now, suppressed=not self.is_live(key))
        return None

    # -------------------------------------------------------------- internals
    def _hit_test(self, x: float, y: float) -> Optional[Key]:
        """Which key owns this point, honouring hysteresis around the focus."""
        candidates = self.keyboard.selectable_on(self.page)
        for key in candidates:                       # unexpanded regions win
            if key.contains(x, y):
                return key
        if self._focused is not None and self._focused.on_page(self.page):
            if self._focused.contains(x, y, self.settings.hysteresis_margin):
                return self._focused
        return None

    def _decay(self, dt: float) -> None:
        """Bleed the accumulated dwell away over the grace period."""
        if self._focused is None:
            return
        if not self._leaving:
            self._leaving = True
            self._leave_accumulated = max(self._accumulated, 1e-9)
        grace_s = max(self.settings.grace_ms / 1000.0, 1e-6)
        self._accumulated -= self._leave_accumulated * dt / grace_s
        if self._accumulated <= 0.0:
            if self._leave_accumulated > 1e-6:      # there was real progress
                self.stats.grace_expired += 1
            self._clear_focus()
