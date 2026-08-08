"""GazeKey — the calibration region: the part of the screen gaze is fitted over.

A webcam gaze estimator has a fixed angular error, so spreading a calibration
over the whole screen spends most of its accuracy on places the user never
looks. Everything they actually select lives in the keyboard at the bottom of
the display — so that is where the nine calibration dots go, and that is where
the error is measured.

The consequences are deliberate and worth stating:

* **the reported validation error is an in-region error.** It is the number
  NFR-2 is evaluated against, because it is the accuracy that applies where
  keys are;
* **every gaze-selectable target must be inside the region.** A YES/NO
  confirmation or a touch-up dot placed at the centre of the *screen* would sit
  outside the fitted area and be aimed at with accuracy nobody measured;
* **outside the region the model extrapolates.** The verified core clamps
  standardized features to ±3σ so it degrades rather than exploding, but the
  gaze cursor drawn up there is expected to be loose. That is fine: nothing is
  selected up there.

A full-screen :class:`Region` reproduces the old behaviour exactly, which is
what ``--cal-region full`` uses for an A/B comparison.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from gaze.calibration import grid_points, validation_points

#: key under which the region rides along in ``~/.gazekey/calibration.json``
REGION_KEY = "region"


@dataclass(frozen=True)
class Region:
    """A rectangle of screen, in the same pixels as the calibration targets."""

    rect: Tuple[float, float, float, float]
    name: str = "screen"

    # ------------------------------------------------------------------ shape
    @property
    def x(self) -> float:
        return self.rect[0]

    @property
    def y(self) -> float:
        return self.rect[1]

    @property
    def w(self) -> float:
        return self.rect[2]

    @property
    def h(self) -> float:
        return self.rect[3]

    @property
    def centre(self) -> Tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    @property
    def diagonal(self) -> float:
        return math.hypot(self.w, self.h)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Is this point inside, optionally with a fractional slack?"""
        grow_x, grow_y = self.w * margin, self.h * margin
        return (self.x - grow_x <= x <= self.x + self.w + grow_x
                and self.y - grow_y <= y <= self.y + self.h + grow_y)

    def holds_rect(self, rect: Sequence[float], tol: float = 1e-6) -> bool:
        """Does this region contain a whole ``(x, y, w, h)`` box?

        ``tol`` absorbs float dust: a keyboard row divided into fractional
        units lands its bottom edge a 1e-13 of a pixel past the screen edge,
        which is arithmetic, not a key hanging off the calibrated area.
        """
        rx, ry, rw, rh = rect
        return (rx >= self.x - tol and ry >= self.y - tol
                and rx + rw <= self.x + self.w + tol
                and ry + rh <= self.y + self.h + tol)

    def point(self, fx: float, fy: float) -> Tuple[float, float]:
        """A point at fractional position ``(fx, fy)`` of the region."""
        return self.x + fx * self.w, self.y + fy * self.h

    # ---------------------------------------------------------------- targets
    def grid(self) -> List[Tuple[int, int]]:
        """The nine calibration targets, at 10/50/90% **of this region**.

        Laid out by the verified core's :func:`grid_points` on the region's own
        size and then offset, so the spacing rule stays in one place.
        """
        return self._offset(grid_points(int(round(self.w)), int(round(self.h))))

    def validation(self) -> List[Tuple[int, int]]:
        """The three fresh validation targets, at 30/70% of this region."""
        return self._offset(
            validation_points(int(round(self.w)), int(round(self.h))))

    def _offset(self, points) -> List[Tuple[int, int]]:
        return [(int(round(self.x)) + px, int(round(self.y)) + py)
                for px, py in points]

    # -------------------------------------------------------------- reporting
    def share_of(self, screen_size: Tuple[int, int]) -> float:
        """Fraction of the screen's area this region covers."""
        width, height = screen_size
        area = float(width) * float(height)
        return (self.w * self.h / area) if area > 0 else 0.0

    def describe(self, screen_size: Optional[Tuple[int, int]] = None) -> str:
        text = (f"{self.name} region {self.w:.0f}x{self.h:.0f} px "
                f"at ({self.x:.0f},{self.y:.0f})")
        if screen_size is not None:
            text += f" - {self.share_of(screen_size) * 100:.0f}% of the screen"
        return text

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"rect": [float(v) for v in self.rect], "name": self.name}

    @classmethod
    def from_dict(cls, data) -> Optional["Region"]:
        try:
            rect = tuple(float(v) for v in data["rect"])
        except (TypeError, KeyError, ValueError):
            return None
        if len(rect) != 4 or rect[2] <= 0 or rect[3] <= 0:
            return None
        return cls(rect, str(data.get("name", "screen")))


def full_screen_region(screen_size: Tuple[int, int]) -> Region:
    """The whole display — what calibration used before regions existed."""
    width, height = screen_size
    return Region((0.0, 0.0, float(width), float(height)), name="screen")


def attach_region(path: str, region: Optional[Region]) -> bool:
    """Record ``region`` in an already-written calibration file.

    Written as an extra key rather than through
    :meth:`~gaze.calibration.CalibrationModel.save`, because that method lives
    in the verified core and must not change. ``load`` there reads named keys
    and ignores everything else, so the extra key is invisible to it.
    """
    if region is None or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return False
        data[REGION_KEY] = region.to_dict()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def read_region(path: str) -> Optional[Region]:
    """The region a saved calibration was fitted over, if it recorded one.

    ``None`` for a file written before regions existed — the caller decides
    what to fall back to, rather than being handed a guess.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or REGION_KEY not in data:
        return None
    return Region.from_dict(data[REGION_KEY])
