"""GazeKey — fullscreen gaze-dot demo (Milestone 2 checkpoint).

A black canvas with a dot that follows the smoothed gaze prediction, so the
accuracy number from the validation gate can be judged by eye. Reference
crosses mark the calibration grid, and a corner read-out reports the frame
rate and the recorded validation error.

This is scaffolding for judging M2; the real keyboard overlay arrives in M4.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import QWidget

from gaze.calibration import grid_points

BACKGROUND = QColor(12, 12, 16)
GRID = QColor(56, 56, 66)
TEXT = QColor(170, 170, 182)

#: gaze dot colours — the M3 checkpoint is watching these switch
MOVING = QColor(90, 190, 255)     # blue: gaze is travelling (saccade)
FIXATING = QColor(90, 210, 140)   # green: I-DT says you are holding still
HELD = QColor(245, 190, 70)       # amber: blinked, last position held


class GazeDotDemo(QWidget):
    """Draws a dot wherever the pipeline says the user is looking."""

    closed = pyqtSignal()

    def __init__(
        self,
        pipeline,
        screen_size: Tuple[int, int],
        validation_error_px: float = float("nan"),
        parent: Optional[QWidget] = None,
        tick_ms: int = 16,
    ) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.screen_size = screen_size
        self.validation_error_px = validation_error_px

        self._x = float("nan")
        self._y = float("nan")
        self._state = "starting"
        self._is_fixating = False
        self._stream_valid = False
        self._held = False

        self.setWindowTitle("GazeKey — gaze demo")
        self.setStyleSheet("background: #0c0c10;")
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(tick_ms)

    # ------------------------------------------------------------------ input
    def _tick(self) -> None:
        if self.pipeline is not None:
            for sample in self.pipeline.drain():
                self._state = sample.state
                self._is_fixating = sample.is_fixating
                self._stream_valid = sample.stream_valid
                self._held = sample.held
                self._x, self._y = sample.x, sample.y
        self.update()

    def dot_colour(self) -> QColor:
        """Amber while a blink is being held, green when fixating, else blue."""
        if self._held:
            return HELD
        return FIXATING if self._is_fixating else MOVING

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.closed.emit()
            self.close()

    # ---------------------------------------------------------------- drawing
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        try:
            self.render_into(painter)
        finally:
            painter.end()

    def render_into(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND)
        self._draw_reference_grid(painter)
        self._draw_dot(painter)
        self._draw_status(painter)

    def _scale(self) -> Tuple[float, float]:
        width, height = self.screen_size
        return self.width() / max(width, 1), self.height() / max(height, 1)

    def _draw_reference_grid(self, painter: QPainter) -> None:
        """The 9 calibration positions, as a yardstick for the dot."""
        sx, sy = self._scale()
        painter.setPen(QPen(GRID, max(1.0, self.height() * 0.0015)))
        arm = self.height() * 0.012
        for x, y in grid_points(*self.screen_size):
            cx, cy = x * sx, y * sy
            painter.drawLine(QPointF(cx - arm, cy), QPointF(cx + arm, cy))
            painter.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy + arm))

    def _draw_dot(self, painter: QPainter) -> None:
        if not (self._stream_valid
                and math.isfinite(self._x) and math.isfinite(self._y)):
            return  # tracking lost — draw nothing rather than a stale lie
        sx, sy = self._scale()
        centre = QPointF(self._x * sx, self._y * sy)
        colour = self.dot_colour()
        radius = self.height() * 0.026

        halo = QColor(colour)
        halo.setAlphaF(0.22)
        painter.setPen(Qt.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(centre, radius, radius)

        painter.setBrush(colour)
        painter.drawEllipse(centre, radius * 0.34, radius * 0.34)

        if self._is_fixating and not self._held:
            # a ring at the dispersion threshold: everything inside it counts
            # as the same fixation
            ring = QColor(colour)
            ring.setAlphaF(0.5)
            painter.setPen(QPen(ring, max(1.5, self.height() * 0.002)))
            painter.setBrush(Qt.NoBrush)
            spread = self._dispersion_px() * sx / 2.0
            painter.drawEllipse(centre, spread, spread)

    def _dispersion_px(self) -> float:
        return float(getattr(self.pipeline, "fixation_dispersion_px", 110.0))

    def status_lines(self) -> list[str]:
        """The corner read-out (exposed so headless tests can check the copy)."""
        fps = getattr(self.pipeline, "fps", 0.0) if self.pipeline else 0.0
        window = float(getattr(self.pipeline, "fixation_window_ms", 150.0))
        error = self.validation_error_px
        accuracy = "not measured" if not math.isfinite(error) else f"{error:.0f} px"
        if not self._stream_valid:
            gaze = "tracking lost"
        elif self._held:
            gaze = "holding through blink"
        elif self._is_fixating:
            gaze = "FIXATING"
        else:
            gaze = "moving"
        return [
            f"calibration accuracy {accuracy}",
            f"pipeline {fps:4.1f} fps     tracking: {self._state}",
            f"gaze: {gaze}",
            f"fixation: dispersion <= {self._dispersion_px():.0f} px "
            f"over {window:.0f} ms",
            "green = fixating   blue = moving   amber = blink held",
            "q or Esc to exit",
        ]

    def _draw_status(self, painter: QPainter) -> None:
        font = QFont("Segoe UI", max(8, int(16 * self.height() / 1080)))
        painter.setFont(font)
        painter.setPen(TEXT)
        painter.drawText(
            QRectF(self.width() * 0.02, self.height() * 0.02,
                   self.width() * 0.5, self.height() * 0.2),
            Qt.AlignLeft | Qt.AlignTop,
            "\n".join(self.status_lines()),
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.stop()
        super().closeEvent(event)
