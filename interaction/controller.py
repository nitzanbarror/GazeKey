"""GazeKey — dwell interaction state machine (spec Section 7).

Per focused key: ``IDLE -> FOCUS -> DWELLING -> ACTIVATED``.

The rules, all of which exist to stop a wandering gaze typing by accident:

* dwell advances **only** while the gaze is inside the key *and*
  :class:`~gaze.smoothing.FixationDetector` says the user is actually fixating;
* **hysteresis** — the focused key keeps ownership everywhere inside its
  **grown** region (25% per side), so a challenger has to win the point from
  *outside* that region before it takes over. Hit regions are gapless, so
  consulting the challengers first (as this did until NFR-7 was measured) let a
  neighbour win outright the moment the gaze crossed the boundary and made the
  margin inert between keys;
* **grace** — losing the region decays the accumulated dwell over 200 ms
  instead of throwing it away, so a momentary wobble is not punished. This
  applies whether the gaze left every key **or** a neighbour took focus: a
  single jittered frame across a row boundary costs one frame of decay, and
  coming back resumes the dwell where it was;
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

#: How much of a key is its **core** — the part no amount of hysteresis can
#: take away. Taken off each side, so 0.25 leaves the middle half. Deliberately
#: a constant rather than the tunable margin: the margin says how sticky the
#: focused key is, the core says what every other key still owns whatever the
#: margin is set to, and raising ``--hysteresis`` must never make a key
#: unreachable by looking straight at it.
CORE_MARGIN = 0.25


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

    A steal is only a *loss* once its grace has run out. While the dwell is
    still decaying it can come back, and one that does is counted as
    :attr:`focus_recovered` instead — which is the whole point of the rule, and
    would be invisible if the steal were counted the moment it happened.
    """

    frames: int = 0
    stream_lost: int = 0
    #: the focused key changed (including the first landing on a key)
    focus_changes: int = 0
    #: ...and the dwell it interrupted decayed away before the gaze came back
    focus_stolen: int = 0
    #: ...or the gaze came back inside the grace and the dwell carried on
    focus_recovered: int = 0
    #: the fixation detector let go mid-dwell, so progress stalled
    fixation_dropped: int = 0
    #: the gaze left every key and the accumulated dwell decayed to nothing
    grace_expired: int = 0
    activations: int = 0

    _FIELDS = (
        "frames", "stream_lost", "focus_changes", "focus_stolen",
        "focus_recovered", "fixation_dropped", "grace_expired", "activations",
    )

    @property
    def dwell_losses(self) -> int:
        return self.focus_stolen + self.fixation_dropped + self.grace_expired

    def snapshot(self) -> "DwellStats":
        return replace(self)

    def since(self, earlier: "DwellStats") -> "DwellStats":
        """This minus an earlier snapshot, for per-window reporting."""
        return DwellStats(**{
            name: getattr(self, name) - getattr(earlier, name)
            for name in DwellStats._FIELDS
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
        #: dwell taken from a key by a steal, still decaying: ``(key id,
        #: remaining seconds, seconds it had when it was taken)``. One slot —
        #: a second steal replaces it, so a wander across three keys keeps only
        #: the most recent, which is the one the gaze is likely to return to.
        self._carry: Optional[Tuple[str, float, float]] = None
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
            self._carry = None          # that key may not exist on this page

    def next_page(self) -> None:
        self.set_page(self.page + 1)

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def reset(self) -> None:
        self._clear_focus()
        self._carry = None
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
            # costs time but not progress (spec Section 6). A carried dwell is
            # frozen with it — the stream being down is not the user wandering.
            self.stats.stream_lost += 1
            return None

        self._decay_carry(dt)
        key = self._hit_test(x, y)
        if key is None:
            self._decay(dt)
            return None

        if self._focused is None or key.id != self._focused.id:
            self.stats.focus_changes += 1
            self._steal_focus(key)
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
        """Which key owns this point, honouring hysteresis around the focus.

        **The focused key is asked first.** Hit regions are gapless, so asking
        the challengers first — which is what this did until NFR-7 measured it
        — meant one of them always owned the point inside the board and the
        margin only ever acted off the board. Sticky focus is what the
        hysteresis rule was always meant to say: a challenger has to win the
        point from outside the focused key's grown region.

        **With one bound: a key always owns its own core**, the middle of it
        left when the same margin is taken off each side. The margin is a
        fraction of the *focused* key's size, and Space is 250 px against a
        124 px Backspace beside it — unbounded, a focused Space would reach
        past Backspace's centre and the user could never select it. The bound
        is the guarantee worth stating plainly: **hysteresis can never cost you
        a key you are looking straight at.**
        """
        focused = self._focused
        margin = self.settings.hysteresis_margin
        candidates = self.keyboard.selectable_on(self.page)
        if (focused is not None and focused.on_page(self.page)
                and focused.contains(x, y, margin)
                and not self._in_another_core(x, y, focused, candidates)):
            return focused
        for key in candidates:
            if key.contains(x, y):
                return key
        return None

    @staticmethod
    def _in_another_core(x: float, y: float, focused: Key,
                         candidates: List[Key]) -> bool:
        """Is the point in the heart of some *other* key? (negative margin)"""
        return any(key.id != focused.id and key.contains(x, y, -CORE_MARGIN)
                   for key in candidates)

    # ------------------------------------------------------- the grace budget
    def _grace_s(self) -> float:
        return max(self.settings.grace_ms / 1000.0, 1e-6)

    def _steal_focus(self, key: Key) -> None:
        """Move focus to ``key``, carrying the interrupted dwell rather than
        discarding it — and picking up this key's own carried dwell if it has
        one waiting.

        Order matters: the carry is claimed *before* the old key's dwell
        replaces it, so bouncing H → N → H restores H's progress and leaves N's
        (a frame or two) decaying in its place.
        """
        restored = self._take_carry(key.id)
        if self._accumulated > 0.0 and self._focused is not None:
            self._carry = (self._focused.id, self._accumulated, self._accumulated)
        self._focused = key
        self._accumulated = restored
        self._leaving = False

    def _take_carry(self, key_id: str) -> float:
        """Whatever dwell this key had left when it was stolen from."""
        if self._carry is None or self._carry[0] != key_id:
            return 0.0
        remaining = self._carry[1]
        self._carry = None
        if remaining > 0.0:
            self.stats.focus_recovered += 1
        return max(0.0, remaining)

    def _decay_carry(self, dt: float) -> None:
        """Bleed a stolen-from key's dwell away over the same grace period.

        It expires as a ``focus_stolen`` rather than a ``grace_expired``,
        because that is what actually happened to it: a neighbour took the key,
        and the gaze never came back in time.
        """
        if self._carry is None:
            return
        key_id, remaining, taken = self._carry
        remaining -= taken * dt / self._grace_s()
        if remaining <= 0.0:
            self._carry = None
            self.stats.focus_stolen += 1
            return
        self._carry = (key_id, remaining, taken)

    def _decay(self, dt: float) -> None:
        """Bleed the accumulated dwell away over the grace period."""
        if self._focused is None:
            return
        if not self._leaving:
            self._leaving = True
            self._leave_accumulated = max(self._accumulated, 1e-9)
        self._accumulated -= self._leave_accumulated * dt / self._grace_s()
        if self._accumulated <= 0.0:
            if self._leave_accumulated > 1e-6:      # there was real progress
                self.stats.grace_expired += 1
            self._clear_focus()
