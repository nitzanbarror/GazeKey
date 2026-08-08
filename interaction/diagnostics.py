"""GazeKey — typing-stability diagnostics (``--debug-typing``).

When typing is unstable the symptom is always the same — "it won't select" —
but the causes are not, and they need different fixes:

* **gaze jitter** crossing a key boundary steals focus and discards the dwell.
  Shows up as a high per-axis ``jitter`` against the key size, and as ``steal``
  dominating the loss counts. The fix is smoothing (``--min-cutoff``) or
  stickier focus;
* **the fixation detector letting go** mid-dwell stalls progress without moving
  focus. Shows up as ``fixdrop`` and a low duty cycle. The fix is the
  dispersion threshold;
* **a mapping that is simply off** puts the gaze on the wrong key entirely.
  Shows up as *low* jitter with few losses and no activations, and no
  interaction knob will fix it — that is a calibration problem.

So this module measures all three separately rather than reporting one number.
It is observational: nothing here changes what the keyboard does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from interaction.controller import DwellController, DwellStats

#: how often a live line is printed while typing
DEFAULT_INTERVAL_S = 5.0

#: jitter above this fraction of the key size crosses a boundary often enough
#: to matter — the threshold the summary judges against
JITTER_FRACTION = 0.25

#: samples per bucket when measuring jitter — ~1/3 s at 30 fps, long enough to
#: see a wobble and short enough that the user is still on one key
JITTER_BUCKET = 10


def _std(values: Sequence[float]) -> float:
    """Population standard deviation, NaN with nothing to measure."""
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _jitter(values: Sequence[float], bucket: int = JITTER_BUCKET) -> float:
    """Local instability: the median deviation *within* short buckets.

    A window-wide standard deviation is useless here — it is dominated by the
    user deliberately moving between keys, which is not instability at all. A
    session moving across the board reads ~350 px on x while wobbling 6 px,
    so the whole-window figure would name the wrong axis. Bucketing to a third
    of a second isolates the wobble; the median across buckets discards the one
    or two that straddle a saccade.
    """
    buckets = [values[i:i + bucket] for i in range(0, len(values), bucket)]
    stds = sorted(_std(b) for b in buckets if len(b) >= 2)
    if not stds:
        return float("nan")
    middle = len(stds) // 2
    if len(stds) % 2:
        return stds[middle]
    return 0.5 * (stds[middle - 1] + stds[middle])


def _px(value: float) -> str:
    return "-" if value != value else f"{value:.0f}"


@dataclass
class Window:
    """One reporting window's worth of measurement."""

    seconds: float = 0.0
    samples: int = 0
    fixating: int = 0
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    hold_xs: List[float] = field(default_factory=list)
    hold_ys: List[float] = field(default_factory=list)
    dwell: DwellStats = field(default_factory=DwellStats)

    @property
    def duty_cycle(self) -> float:
        return self.fixating / self.samples if self.samples else float("nan")

    @property
    def focus_rate(self) -> float:
        return self.dwell.focus_changes / self.seconds if self.seconds else 0.0

    @property
    def jitter(self) -> Tuple[float, float]:
        """Per-axis local instability in px — the number that predicts steals."""
        return _jitter(self.xs), _jitter(self.ys)

    def line(self) -> str:
        """The one-line live report."""
        duty = self.duty_cycle
        jitter_x, jitter_y = self.jitter
        return (
            f"[typing] {self.seconds:4.1f}s  "
            f"focus {self.focus_rate:4.1f}/s  "
            f"resets: steal {self.dwell.focus_stolen:<3d} "
            f"fixdrop {self.dwell.fixation_dropped:<3d} "
            f"grace {self.dwell.grace_expired:<3d}  "
            f"jitter x {_px(jitter_x):>3} y {_px(jitter_y):>3} px  "
            f"spread x {_px(_std(self.xs)):>4} y {_px(_std(self.ys)):>4}  "
            f"fixating {'-' if duty != duty else f'{duty * 100:.0f}%'}  "
            f"keys {self.dwell.activations}"
        )


class TypingDiagnostics:
    """Accumulates sample- and controller-level evidence while typing.

    Feed it every sample the overlay drains, and call :meth:`tick` once per
    frame; it prints a line every ``interval_s`` and a summary on request.
    """

    def __init__(
        self,
        controller: DwellController,
        key_size: Tuple[float, float] = (0.0, 0.0),
        interval_s: float = DEFAULT_INTERVAL_S,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.controller = controller
        self.key_size = key_size
        self.interval_s = float(interval_s)
        self.log = log if log is not None else print

        self.windows: List[Window] = []
        self.session = Window()
        self._current = Window()
        self._mark = controller.stats.snapshot()
        self._started: Optional[float] = None
        self._window_start: Optional[float] = None

    # ------------------------------------------------------------------ input
    def record(self, sample) -> None:
        """One drained :class:`~vision.pipeline.GazeSample`."""
        if not (getattr(sample, "stream_valid", False)
                and getattr(sample, "has_gaze", False)):
            return
        # The window clock starts with the first sample, not the first tick, so
        # a burst of samples drained in one go still reports the span it covers.
        if self._started is None:
            self._started = self._window_start = float(sample.timestamp)
        x, y = float(sample.x), float(sample.y)
        for window in (self._current, self.session):
            window.samples += 1
            window.xs.append(x)
            window.ys.append(y)
            if sample.is_fixating:
                window.fixating += 1
                window.hold_xs.append(x)
                window.hold_ys.append(y)

    def tick(self, now: float) -> Optional[Window]:
        """Close and report a window if the interval has elapsed."""
        if self._started is None:
            self._started = self._window_start = now
            return None
        if now - self._window_start < self.interval_s:
            return None

        window = self._current
        window.seconds = now - self._window_start
        current = self.controller.stats.snapshot()
        window.dwell = current.since(self._mark)

        self.log(window.line())
        self.windows.append(window)
        self._current = Window()
        self._mark = current
        self._window_start = now
        return window

    # ---------------------------------------------------------------- summary
    def close(self, now: Optional[float] = None) -> None:
        """Fold whatever is left of the open window into the session."""
        if self._started is not None and now is not None:
            self.session.seconds = now - self._started
        self.session.dwell = self.controller.stats.snapshot()

    def summary_lines(self) -> List[str]:
        """The end-of-session report, for the console."""
        session = self.session
        if not session.samples:
            return ["[typing] no gaze samples were recorded."]

        dwell = session.dwell
        completed = dwell.activations
        losses = dwell.dwell_losses
        attempted = completed + losses
        lines = [
            "[typing] typing-stability summary",
            f"  duration        : {session.seconds:.0f} s over "
            f"{len(self.windows)} window(s), {session.samples} samples",
            f"  keys activated  : {completed}"
            + (f"  ({completed / attempted * 100:.0f}% of "
               f"{attempted} dwell attempts)" if attempted else ""),
            f"  dwell losses    : {losses} total - "
            f"focus stolen {dwell.focus_stolen}, "
            f"fixation dropped {dwell.fixation_dropped}, "
            f"grace expired {dwell.grace_expired}",
            f"  focus changes   : {dwell.focus_changes} "
            f"({session.focus_rate:.1f}/s)",
            f"  jitter          : x {_px(session.jitter[0])} px, "
            f"y {_px(session.jitter[1])} px (local wobble - what steals focus)",
            f"  spread          : x {_px(_std(session.xs))} px, "
            f"y {_px(_std(session.ys))} px (whole window, incl. moving between keys)",
            f"  fixation duty   : "
            f"{'-' if session.duty_cycle != session.duty_cycle else f'{session.duty_cycle * 100:.0f}%'}",
            f"  tracking lost   : {dwell.stream_lost} frames",
        ]
        if self.key_size[0] and self.key_size[1]:
            lines.append(f"  key size        : "
                         f"{self.key_size[0]:.0f}x{self.key_size[1]:.0f} px")
        lines.append(f"  most likely     : {self.likely_cause()}")
        return lines

    def likely_cause(self) -> str:
        """One line naming what is actually limiting typing."""
        session = self.session
        dwell = session.dwell
        if not session.samples:
            return "no gaze reached the keyboard at all."

        key_w, key_h = self.key_size
        hold_x, hold_y = session.jitter
        losses = dwell.dwell_losses

        if dwell.stream_lost > session.samples:
            return ("tracking was lost more often than it was held - lighting "
                    "or camera placement, not the interaction settings.")

        if losses and dwell.focus_stolen >= 0.6 * losses:
            share = dwell.focus_stolen / losses * 100
            axis = ""
            if key_w and key_h and hold_y == hold_y and hold_x == hold_x:
                if hold_y / key_h > hold_x / key_w:
                    axis = (f" Vertical is the worse axis: jitter y "
                            f"{hold_y:.0f} px against a {key_h:.0f} px row, "
                            f"versus x {hold_x:.0f} px against {key_w:.0f} px.")
                else:
                    axis = (f" Horizontal is the worse axis: jitter x "
                            f"{hold_x:.0f} px against a {key_w:.0f} px column, "
                            f"versus y {hold_y:.0f} px against {key_h:.0f} px.")
            return (f"{share:.0f}% of lost dwells were focus steals - gaze is "
                    f"crossing key boundaries mid-dwell.{axis} Try a lower "
                    f"--min-cutoff (more smoothing).")

        if losses and dwell.fixation_dropped >= 0.6 * losses:
            return ("most dwells died because the fixation detector let go, "
                    "not because focus moved - raise fixation_dispersion_px.")

        if (key_h and hold_y == hold_y
                and hold_y < JITTER_FRACTION * key_h and not dwell.activations):
            return ("the gaze is steady but nothing activates - it is landing "
                    "on the wrong key, which is a calibration problem, not an "
                    "interaction one.")

        if not losses and dwell.activations:
            return "typing looks healthy - no dwell was lost."

        return ("no single dominant cause; compare the per-window lines above "
                "for where it degrades.")
